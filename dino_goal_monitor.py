import re
import cv2
import torch
import numpy as np
import pyrealsense2 as rs
from groundingdino.util.inference import load_model as gd_load_model, predict
import groundingdino.datasets.transforms as T
from PIL import Image

# ── Model / stream constants (mirrored from live_monitor.py) ──────────────────
CONFIG_PATH    = "GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH   = "groundingdino_swint_ogc.pth"
BOX_THRESHOLD  = 0.4
TEXT_THRESHOLD = 0.25
STREAM_WIDTH   = 640
STREAM_HEIGHT  = 480

KNOWN_FRUITS = {"apple", "pear", "orange", "banana", "grape", "mango",
                "strawberry", "blueberry", "peach", "plum", "kiwi", "lemon"}


# ── Step 1: Goal Parsing ──────────────────────────────────────────────────────

def parse_goal(goal_text: str) -> dict:
    """
    Rule-based parser for natural-language ordering goals.

    Supported patterns:
      "Arrange the fruits left to right as apple, orange, pear"
      "Order: orange, apple, pear"
      "Place apple then orange then pear from left to right"

    Returns:
      {
        "task_type": "ordering",
        "objects":   ["apple", "orange", "pear"],
        "order":     ["apple", "orange", "pear"],
        "axis":      "left_to_right"
      }

    Raises ValueError if no recognisable fruit sequence is found.
    """
    text = goal_text.lower().strip()

    # Determine axis (default left_to_right)
    if "right to left" in text:
        axis = "right_to_left"
    else:
        axis = "left_to_right"

    # --- Strategy 1: look for a comma/then-separated list after a keyword -----
    # Matches "as X, Y, Z" / "order: X, Y, Z" / "place X then Y then Z"
    list_pattern = re.search(
        r'(?:as|order\s*:|place|arrange|sort|put)\s+'
        r'((?:[a-z]+(?:\s*,\s*|\s+then\s+))*[a-z]+)',
        text
    )
    if list_pattern:
        raw = list_pattern.group(1)
        tokens = re.split(r'\s*,\s*|\s+then\s+', raw)
        fruits = [t.strip() for t in tokens if t.strip() in KNOWN_FRUITS]
        if fruits:
            return _build_result(fruits, axis)

    # --- Strategy 2: scan entire text for any known fruit mentions in order ---
    words = re.findall(r'[a-z]+', text)
    fruits = [w for w in words if w in KNOWN_FRUITS]
    # Deduplicate while preserving order
    seen = set()
    unique_fruits = []
    for f in fruits:
        if f not in seen:
            seen.add(f)
            unique_fruits.append(f)

    if unique_fruits:
        return _build_result(unique_fruits, axis)

    raise ValueError(
        f"Could not extract a fruit ordering from goal: '{goal_text}'\n"
        "Example: \"Arrange the fruits left to right as apple, orange, pear\""
    )


def _build_result(fruits: list, axis: str) -> dict:
    return {
        "task_type": "ordering",
        "objects":   fruits,
        "order":     fruits,
        "axis":      axis,
    }


# ── Step 2: Monitoring Configuration ─────────────────────────────────────────

def build_monitor_config(parsed_goal: dict) -> dict:
    """Convert a parsed goal dict into a monitoring configuration."""
    if parsed_goal["task_type"] != "ordering":
        raise NotImplementedError(
            f"Unsupported task type: {parsed_goal['task_type']}"
        )
    return {
        "monitor_type": "ordering",
        "objects":      parsed_goal["objects"],
        "target_order": parsed_goal["order"],
    }


# ── Step 3: Detection helpers (reimplemented; live_monitor.py untouched) ─────

def load_model():
    model = gd_load_model(CONFIG_PATH, WEIGHTS_PATH)
    model.eval()
    return model


def _preprocess_frame(frame_bgr):
    frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image  = Image.fromarray(frame_rgb)
    transform  = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(pil_image, None)
    return image_tensor


def extract_order(model, frame_bgr, objects: list, text_prompt: str):
    """
    Run Grounding DINO and return (order, boxes) for the requested objects.

    order : list[str]  — labels sorted left-to-right by centre-x
    boxes : list[tuple] — (x1, y1, x2, y2, label)
    """
    h, w = frame_bgr.shape[:2]
    image_tensor = _preprocess_frame(frame_bgr)

    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

    if boxes is None or len(boxes) == 0:
        return [], []

    allowed = set(objects)
    detections = []
    for box, phrase in zip(boxes, phrases):
        cx_norm, cy_norm, bw_norm, bh_norm = box.tolist()
        cx = cx_norm * w
        cy = cy_norm * h
        bw = bw_norm * w
        bh = bh_norm * h
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)
        label = phrase.strip().lower()
        if label in allowed:
            detections.append((cx, label, x1, y1, x2, y2))

    detections.sort(key=lambda d: d[0])
    order     = [d[1] for d in detections]
    boxes_out = [(d[2], d[3], d[4], d[5], d[1]) for d in detections]
    return order, boxes_out


def draw_boxes(frame, boxes):
    for (x1, y1, x2, y2, label) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def evaluate_order(current_order: list, target_order: list) -> str:
    if len(current_order) < len(target_order):
        return "IN PROGRESS"
    if current_order == target_order:
        return "TASK COMPLETED"
    for i, (cur, tgt) in enumerate(zip(current_order, target_order)):
        if cur != tgt:
            return f"Expected {tgt} at position {i + 1}"
    return "IN PROGRESS"


# ── Main: goal-to-monitor pipeline ───────────────────────────────────────────

def main():
    # 1. Get goal from user
    goal_text = input("Enter goal: ").strip()
    if not goal_text:
        print("No goal entered. Exiting.")
        return

    # 2. Parse
    try:
        parsed = parse_goal(goal_text)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return

    print(f"Parsed goal  : {parsed['task_type']} task")
    print(f"Target order : {parsed['order']}")

    # 3. Build monitoring config
    config = build_monitor_config(parsed)
    target_order = config["target_order"]
    text_prompt  = " . ".join(config["objects"])

    print("Starting monitoring...  (press 'q' in the video window to quit)\n")

    # 4. Load model
    model = load_model()

    # 5. Start RealSense
    pipeline   = rs.pipeline()
    rs_config  = rs.config()
    rs_config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT,
                             rs.format.bgr8, 30)
    pipeline.start(rs_config)

    try:
        while True:
            frames      = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # 6. Detect & evaluate
            current_order, boxes = extract_order(model, frame,
                                                  config["objects"],
                                                  text_prompt)
            status = evaluate_order(current_order, target_order)

            # 7. Terminal feedback
            print(f"Current order : {current_order}")
            print(f"Status        : {status}")
            print("-" * 40)

            # 8. OSD overlay
            frame = draw_boxes(frame, boxes)
            cv2.putText(frame, f"Target : {target_order}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame, f"Current: {current_order}", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Status : {status}", (10, 79),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Goal Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
