#!/usr/bin/env python3

import re
import sys
import cv2
import torch
import numpy as np
import pyrealsense2 as rs
from PIL import Image


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "google/gemma-3-12b-it"
STREAM_WIDTH = 640
STREAM_HEIGHT = 480
INFER_EVERY_N = 10
MAX_STEPS = 6
MAX_TOKENS_BREAKDOWN = 200
MAX_TOKENS_CHECK = 60


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
        device_map="auto"
    )

    model.eval()

    print("[INFO] Model ready.\n")

    return processor, model


# ── VLM query helper ──────────────────────────────────────────────────────────

def _query(processor, model, pil_image, text, max_new_tokens):

    content = []

    if pil_image is not None:
        content.append({"type": "image"})

    content.append({"type": "text", "text": text})

    messages = [{"role": "user", "content": content}]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        images=[pil_image] if pil_image is not None else None,
        text=[text_prompt],
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    n_input = inputs["input_ids"].shape[1]

    response = processor.decode(
        output_ids[0][n_input:],
        skip_special_tokens=True
    )

    return response.strip()


# ── Goal breakdown prompt ─────────────────────────────────────────────────────

_BREAKDOWN_PROMPT = """
You are a robot monitoring planner.

You receive:
1) A goal
2) A camera image

Convert the goal into conditions that can be verified from a single image.

Rules:
- Only include objects relevant to the goal.
- Ignore unrelated objects in the scene.
- Use object names from the goal when possible.
- Each step must describe a CURRENT STATE of the scene.
- Do NOT describe actions.

Return ONLY a numbered list.

Goal: {goal}
"""


# ── Step 1: Goal breakdown ────────────────────────────────────────────────────

def break_down_goal(goal, frame_bgr, processor, model):

    prompt = _BREAKDOWN_PROMPT.format(goal=goal)

    pil_image = Image.fromarray(
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    )

    response = _query(
        processor,
        model,
        pil_image,
        prompt,
        MAX_TOKENS_BREAKDOWN
    )

    return _parse_numbered_list(response)


# ── Step parser ───────────────────────────────────────────────────────────────

def _parse_numbered_list(text):

    steps = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        cleaned = re.sub(r'^[\d]+[\.\)]\s*', '', line).strip()

        # remove commentary
        if cleaned.lower().startswith(
            ("okay", "sure", "here", "this", "below")
        ):
            continue

        # remove leaked prompt text
        if cleaned.lower().startswith("goal"):
            continue

        steps.append(cleaned)

    return steps[:MAX_STEPS]


# ── Step verification ─────────────────────────────────────────────────────────

_CHECK_PROMPT = """
Look at this image.

Answer ONLY YES or NO.

Condition: {condition}

Answer:
"""


def check_step(condition, frame_bgr, processor, model):

    pil_image = Image.fromarray(
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    )

    prompt = _CHECK_PROMPT.format(condition=condition)

    response = _query(
        processor,
        model,
        pil_image,
        prompt,
        MAX_TOKENS_CHECK
    )

    answer = response.strip().upper()

    is_done = answer.startswith("YES")

    return is_done, response


# ── Overlay ───────────────────────────────────────────────────────────────────

def draw_overlay(frame, steps, step_status, active_idx):

    overlay = frame.copy()
    sidebar_w = 350

    cv2.rectangle(
        overlay,
        (0, 0),
        (sidebar_w, STREAM_HEIGHT),
        (25, 25, 25),
        -1
    )

    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 40

    for i, (step, done) in enumerate(zip(steps, step_status)):

        if done:
            color, icon = (50, 220, 50), "[X]"
        elif i == active_idx:
            color, icon = (60, 60, 255), ">>>"
        else:
            color, icon = (150, 150, 150), "[ ]"

        label = f"{icon} {i+1}. {step}"

        cv2.putText(
            frame,
            label,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1
        )

        y += 22


# ── Monitoring loop ───────────────────────────────────────────────────────────

def monitor(steps, processor, model):

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        STREAM_WIDTH,
        STREAM_HEIGHT,
        rs.format.bgr8,
        30
    )

    pipeline.start(config)

    step_status = [False] * len(steps)

    frame_count = 0

    def active_idx():
        for i, done in enumerate(step_status):
            if not done:
                return i
        return None

    try:

        while True:

            frames = pipeline.wait_for_frames()
            cf = frames.get_color_frame()

            if not cf:
                continue

            frame = np.asanyarray(cf.get_data())

            frame_count += 1

            idx = active_idx()

            if idx is None:

                cv2.putText(
                    frame,
                    "TASK COMPLETE",
                    (10, STREAM_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("Gemma Goal Monitor", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                continue

            if frame_count % INFER_EVERY_N == 0:

                step = steps[idx]

                done, reason = check_step(
                    step,
                    frame,
                    processor,
                    model
                )

                if done:
                    step_status[idx] = True

                print(f"\n--- Frame {frame_count} ---")
                print(f"[CHECKING] Step {idx+1}: {step}")

                marker = "[DONE]" if done else "[ .. ]"

                print(marker, step)

                if reason:
                    print(reason)

            draw_overlay(frame, steps, step_status, idx)

            cv2.imshow("Gemma Goal Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        pipeline.stop()
        cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():

    print("\nGemma Goal Monitor\n")

    goal_text = input("Describe your goal:\n> ").strip()

    if not goal_text:
        return

    processor, model = load_model()

    # capture frame for planning
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        STREAM_WIDTH,
        STREAM_HEIGHT,
        rs.format.bgr8,
        30
    )

    pipeline.start(config)

    frame = None

    for _ in range(20):
        frames = pipeline.wait_for_frames()
        cf = frames.get_color_frame()
        if cf:
            frame = np.asanyarray(cf.get_data())

    pipeline.stop()

    steps = break_down_goal(
        goal_text,
        frame,
        processor,
        model
    )

    if not steps:
        print("No steps generated")
        return

    print("\nSubsteps:")

    for i, s in enumerate(steps):
        print(f"{i+1}. {s}")

    start = input("\nStart monitoring? [Y/n]: ").lower()

    if start == "n":
        return

    monitor(steps, processor, model)


if __name__ == "__main__":
    main()