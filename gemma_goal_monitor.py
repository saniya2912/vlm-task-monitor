#!/usr/bin/env python3
"""
gemma_goal_monitor.py

Natural-language goal monitor using Google Gemma 3 VLM + Intel RealSense.

Requires:
    pip install "transformers>=4.50.0" accelerate
    (This is a newer transformers than the DINO scripts use — run in a fresh venv
     or upgrade after confirming DINO is no longer needed in the same env.)

Usage:
    source vlm_venv/bin/activate
    python gemma_goal_monitor.py

At the prompt, describe any goal in plain English, e.g.
  "Pick up the red apple and place it on the right side of the table"
  "Arrange the three cups so the blue cup is in the middle"
  "Stack the blocks with the largest block on the bottom"

The script will:
  1. Capture an initial frame from the camera.
  2. Ask Gemma to break your goal into observable sub-steps.
  3. Continuously check each step against the live feed.
  4. Show real-time progress in the terminal and as an OSD overlay.
"""

import re
import sys
import cv2
import torch
import numpy as np
import pyrealsense2 as rs
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID      = "google/gemma-3-12b-it"   # swap for "google/gemma-3-12b-it" if VRAM allows
STREAM_WIDTH  = 640
STREAM_HEIGHT = 480
INFER_EVERY_N = 10        # run VLM every N frames (lower = fresher but slower)
MAX_STEPS     = 6         # max sub-steps extracted from the goal
MAX_TOKENS_BREAKDOWN = 300
MAX_TOKENS_CHECK     = 80


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    try:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    except ImportError:
        sys.exit(
            "[ERROR] Gemma3ForConditionalGeneration not found.\n"
            "Run: pip install 'transformers>=4.50.0' accelerate"
        )

    print(f"[INFO] Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("[INFO] Model ready.\n")
    return processor, model


# ── VLM query helper ──────────────────────────────────────────────────────────

def _query(processor, model, pil_image, text: str, max_new_tokens: int) -> str:
    """Send image + text to Gemma 3 and return the decoded response."""
    content = []
    if pil_image is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})

    messages = [{"role": "user", "content": content}]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        images=[pil_image] if pil_image is not None else None,
        text=[text_prompt],
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    n_input  = inputs["input_ids"].shape[1]
    response = processor.decode(output_ids[0][n_input:], skip_special_tokens=True)
    return response.strip()


# ── Step 1: Goal breakdown ────────────────────────────────────────────────────

_BREAKDOWN_PROMPT = """\
You are a robot task planner with a camera view of the current scene.
Break the following goal into {max_steps} or fewer simple sub-steps.

Rules:
- Each step must describe a single visible, verifiable change or condition.
- Use plain, concrete language a camera can confirm (positions, colours, counts).
- Output ONLY a numbered list — no explanations, no extra text.

Goal: {goal}
"""


def break_down_goal(goal: str, processor, model, initial_frame=None) -> list:
    prompt = _BREAKDOWN_PROMPT.format(goal=goal, max_steps=MAX_STEPS)
    pil_image = (
        Image.fromarray(cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGB))
        if initial_frame is not None
        else None
    )
    response = _query(processor, model, pil_image, prompt, MAX_TOKENS_BREAKDOWN)
    return _parse_numbered_list(response)


def _parse_numbered_list(text: str) -> list:
    steps = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r'^[\d]+[\.\)]\s*|^[-•]\s*', '', line).strip()
        if cleaned:
            steps.append(cleaned)
    return steps[:MAX_STEPS]


# ── Step 2: Per-step completion check ────────────────────────────────────────

_CHECK_PROMPT = """\
Look at this image carefully. Answer with YES or NO (then a brief one-sentence reason) \
whether the following condition is currently met in the scene.

Condition: {condition}

Answer:"""


def check_step(condition: str, frame_bgr, processor, model) -> tuple:
    pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    prompt    = _CHECK_PROMPT.format(condition=condition)
    response  = _query(processor, model, pil_image, prompt, MAX_TOKENS_CHECK)
    is_done   = response.upper().startswith("YES")
    return is_done, response


# ── OSD overlay ───────────────────────────────────────────────────────────────

def _wrap(text: str, width: int) -> list:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur += (" " if cur else "") + w
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_overlay(frame, steps: list, step_status: list, active_idx: int) -> None:
    overlay    = frame.copy()
    sidebar_w  = 350
    cv2.rectangle(overlay, (0, 0), (sidebar_w, STREAM_HEIGHT), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "GOAL PROGRESS", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 255), 1)

    y = 48
    for i, (step, done) in enumerate(zip(steps, step_status)):
        if done:
            color, icon = (50, 220, 50),  "[X]"
        elif i == active_idx:
            color, icon = (60,  60, 255), ">>>"
        else:
            color, icon = (150, 150, 150), "[ ]"

        label = f"{icon} {i + 1}. {step}"
        for chunk in _wrap(label, 40):
            cv2.putText(frame, chunk, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            y += 16
        y += 5

    # Status bar
    done_count = sum(step_status)
    total      = len(steps)
    if done_count == total:
        status, bar_color = "ALL STEPS COMPLETE — TASK DONE!", (0, 220, 0)
    else:
        status, bar_color = (
            f"Active: step {active_idx + 1}/{total}",
            (0, 190, 230),
        )
    cv2.rectangle(frame, (0, STREAM_HEIGHT - 32), (STREAM_WIDTH, STREAM_HEIGHT),
                  (20, 20, 20), -1)
    cv2.putText(frame, status, (10, STREAM_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, bar_color, 2)


# ── Step 3: Monitoring loop ───────────────────────────────────────────────────

def monitor(steps: list, processor, model) -> None:
    pipeline  = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT,
                             rs.format.bgr8, 30)
    pipeline.start(rs_config)

    step_status  = [False] * len(steps)
    step_reasons = [""] * len(steps)
    frame_count  = 0

    def active_idx():
        for i, done in enumerate(step_status):
            if not done:
                return i
        return len(steps) - 1   # all done — keep last highlighted

    try:
        while True:
            frames      = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            frame_count += 1

            if frame_count % INFER_EVERY_N == 0:
                for i, step in enumerate(steps):
                    done, reason       = check_step(step, frame, processor, model)
                    step_status[i]     = done
                    step_reasons[i]    = reason

                # Terminal output
                print(f"\n--- Frame {frame_count} ---")
                for i, (step, done, reason) in enumerate(
                        zip(steps, step_status, step_reasons)):
                    marker = "[DONE]" if done else "[ .. ]"
                    print(f"  {marker} {i + 1}. {step}")
                    if reason:
                        print(f"           {reason[:90]}")
                if all(step_status):
                    print("\n>>> ALL STEPS COMPLETE — TASK DONE! <<<\n")

            idx = active_idx()
            draw_overlay(frame, steps, step_status, idx)
            cv2.imshow("Gemma Goal Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Gemma VLM Goal Monitor")
    print("=" * 60)

    goal_text = input("\nDescribe your goal in natural language:\n> ").strip()
    if not goal_text:
        print("No goal entered. Exiting.")
        return

    processor, model = load_model()

    # Capture initial frame for breakdown context
    print("[INFO] Capturing initial scene ...")
    pipeline  = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT,
                             rs.format.bgr8, 30)
    pipeline.start(rs_config)
    initial_frame = None
    for _ in range(15):          # discard first frames (auto-exposure settling)
        frames = pipeline.wait_for_frames()
        cf     = frames.get_color_frame()
        if cf:
            initial_frame = np.asanyarray(cf.get_data())
    pipeline.stop()

    print(f'[INFO] Breaking down: "{goal_text}"\n')
    steps = break_down_goal(goal_text, processor, model, initial_frame)

    if not steps:
        print("[ERROR] The VLM returned no steps. Try rephrasing your goal.")
        return

    print("Sub-steps extracted:")
    for i, s in enumerate(steps):
        print(f"  {i + 1}. {s}")

    confirm = input("\nStart monitoring? [Y/n]: ").strip().lower()
    if confirm == "n":
        return

    print("\n[INFO] Monitoring live feed ... (press 'q' to quit)\n")
    monitor(steps, processor, model)


if __name__ == "__main__":
    main()
