import cv2
import torch
import numpy as np
import pyrealsense2 as rs
from groundingdino.util.inference import load_model as gd_load_model, predict
import groundingdino.datasets.transforms as T
from PIL import Image

CONFIG_PATH = "GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "groundingdino_swint_ogc.pth"
TEXT_PROMPT = "apple . pear . orange"
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.25
TARGET_ORDER = ["apple", "orange", "pear"]
STREAM_WIDTH = 640
STREAM_HEIGHT = 480


def load_model():
    model = gd_load_model(CONFIG_PATH, WEIGHTS_PATH)
    model.eval()
    return model


def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(pil_image, None)
    return image_tensor


def extract_order(model, frame_bgr):
    h, w = frame_bgr.shape[:2]
    image_tensor = preprocess_frame(frame_bgr)

    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

    if boxes is None or len(boxes) == 0:
        return [], []

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
        if label in ["apple", "pear", "orange"]:
            detections.append((cx, label, x1, y1, x2, y2))

    detections.sort(key=lambda d: d[0])
    order = [d[1] for d in detections]
    boxes_out = [(d[2], d[3], d[4], d[5], d[1]) for d in detections]
    return order, boxes_out


def draw_boxes(frame, boxes):
    for (x1, y1, x2, y2, label) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def evaluate_order(current_order):
    if len(current_order) < 3:
        return "IN PROGRESS"
    if current_order == TARGET_ORDER:
        return "TASK COMPLETED"
    for i, (cur, tgt) in enumerate(zip(current_order, TARGET_ORDER)):
        if cur != tgt:
            return f"Expected {tgt} at position {i + 1}"
    return "IN PROGRESS"


def monitor():
    model = load_model()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            current_order, boxes = extract_order(model, frame)
            status = evaluate_order(current_order)

            frame = draw_boxes(frame, boxes)

            print(f"Current order : {current_order}")
            print(f"Status        : {status}")
            print("-" * 40)

            cv2.putText(frame, f"Order: {current_order}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {status}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cv2.imshow("Fruit Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    monitor()
