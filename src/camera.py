"""
Real-time traffic sign detection via webcam.

Usage:
    python -m src.camera --weights experiments/checkpoints/best.pt [--cam 0]

Keys:
    q  — quit
    s  — save current frame to experiments/screenshots/
"""
import argparse
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from src.config import IMG_SIZE, MEAN, STD, NUM_CLASSES, CKPT_DIR
from src.model import build_model

# Canonical class names. 0..42 = German GTSRB (order = ClassId), then
# 43..45 = RU signs from RTSD (tools/rtsd_to_crops.py). Single source of
# truth — keep length == config.NUM_CLASSES (asserted below).
CLASS_NAMES = [
    "Speed limit (20)",    "Speed limit (30)",    "Speed limit (50)",
    "Speed limit (60)",    "Speed limit (70)",    "Speed limit (80)",
    "End speed limit (80)","Speed limit (100)",   "Speed limit (120)",
    "No passing",          "No passing >3.5t",    "Right-of-way",
    "Priority road",       "Yield",               "Stop",
    "No vehicles",         "No vehicles >3.5t",   "No entry",
    "General caution",     "Dangerous curve L",   "Dangerous curve R",
    "Double curve",        "Bumpy road",          "Slippery road",
    "Road narrows R",      "Road work",           "Traffic signals",
    "Pedestrians",         "Children crossing",   "Bicycles crossing",
    "Beware ice/snow",     "Wild animals",        "End restrictions",
    "Turn right ahead",    "Turn left ahead",     "Ahead only",
    "Go straight or right","Go straight or left", "Keep right",
    "Keep left",           "Roundabout",          "End no passing",
    "End no passing >3.5t",
    # ── RU (RTSD) ──
    "Pedestrian crossing", "Speed bump",          "Parking",
]

assert len(CLASS_NAMES) == NUM_CLASSES


_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_model(weights: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(freeze_backbone=False)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()
    return model


@torch.no_grad()
def classify_frame(model, frame_bgr: np.ndarray, device: torch.device):
    """Return (class_id, confidence) for the central crop of the frame."""
    h, w = frame_bgr.shape[:2]
    size = min(h, w)
    crop = frame_bgr[
        (h - size) // 2 : (h + size) // 2,
        (w - size) // 2 : (w + size) // 2,
    ]
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = _transform(pil).unsqueeze(0).to(device)
    probs  = torch.softmax(model(tensor), dim=1)[0]
    cls    = probs.argmax().item()
    return cls, probs[cls].item()


def draw_overlay(frame, cls_id: int, conf: float, fps: float):
    h, w = frame.shape[:2]
    size = min(h, w)
    x0 = (w - size) // 2
    y0 = (h - size) // 2

    # green square around the classification region
    cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (0, 220, 0), 2)

    label = f"{CLASS_NAMES[cls_id]}  {conf*100:.1f}%"
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)


def run(weights: Path, cam_id: int | str = 0, device_str: str = "auto"):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    model = load_model(weights, device)
    print("Model loaded.")

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_id}")

    save_dir = Path("experiments/screenshots")
    save_dir.mkdir(parents=True, exist_ok=True)

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cls_id, conf = classify_frame(model, frame, device)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        draw_overlay(frame, cls_id, conf, fps)
        cv2.imshow("Traffic Sign Classifier", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = save_dir / f"frame_{int(time.time())}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=CKPT_DIR / "best.pt")
    parser.add_argument("--cam", default="0",
                        help="camera index (0,1,...) or stream URL")
    parser.add_argument("--device",  type=str,  default="auto")
    args = parser.parse_args()
    cam = int(args.cam) if args.cam.isdigit() else args.cam
    run(args.weights, cam, args.device)
