"""
Capture an OUT-OF-DOMAIN hold-out set: real webcam crops of one sign class.

Why: accuracy on the GTSRB test split is misleading — it is the same clean
domain as training.  The only honest measure of "does this work in real life"
is the model's accuracy on signs seen through *this* webcam.  This tool builds
that set, one class at a time.

Usage:
    python -m src.holdout_capture --class-id 0 [--cam 0] [--out data/holdout]

Hold the sign in front of the camera.  The colour detector draws candidate
boxes; press:
    s  — save the largest detected ROI crop to data/holdout/<class-id>/
    c  — save a centre square crop instead (fallback if detector misses)
    q  — quit

Saved crops are raw RGB regions (NO CLAHE) — they go through the exact eval
transform of the PyTorch stack (src.dataset.get_transforms(train=False)).
"""
import argparse
import time
from pathlib import Path

import cv2

from src.detector import detect
from src.camera import CLASS_NAMES   # English, length-43, asserted there

DETECT_WIDTH = 640


def _largest_roi(frame):
    scale = DETECT_WIDTH / frame.shape[1]
    small = cv2.resize(frame, (DETECT_WIDTH, int(frame.shape[0] * scale)))
    rois = detect(small)
    if not rois:
        return None
    x, y, w, h = max(rois, key=lambda r: r[2] * r[3])
    inv = 1.0 / scale
    return int(x * inv), int(y * inv), int(w * inv), int(h * inv)


def _centre_square(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    return (w - s) // 2, (h - s) // 2, s, s


def run(class_id: int, cam_id, out_root: Path):
    if not (0 <= class_id <= 42):
        raise SystemExit("--class-id must be 0..42")
    out_dir = out_root / str(class_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = len(list(out_dir.glob("*.jpg")))
    name = CLASS_NAMES[class_id]
    print(f"Class {class_id}: {name}  →  {out_dir}  ({saved} already there)")
    print("[s] save detector ROI  [c] save centre crop  [q] quit")

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {cam_id}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        roi = _largest_roi(frame)
        view = frame.copy()
        if roi:
            x, y, w, h = roi
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.putText(view, f"[{class_id}] {name}   saved={saved}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        cv2.imshow("holdout capture", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        box = None
        if key == ord("s"):
            box = roi if roi else None
            if box is None:
                print("  no detection — use [c] for centre crop")
        elif key == ord("c"):
            box = _centre_square(frame)
        if box:
            x, y, w, h = box
            crop = frame[max(0, y):y + h, max(0, x):x + w]
            if crop.size:
                p = out_dir / f"{int(time.time()*1000)}.jpg"
                cv2.imwrite(str(p), crop)
                saved += 1
                print(f"  saved {p.name}  (total {saved})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {saved} crops in {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-id", type=int, required=True, help="GTSRB ClassId 0..42")
    ap.add_argument("--cam", default="0", help="camera index or video path")
    ap.add_argument("--out", type=Path, default=Path("data/holdout"))
    a = ap.parse_args()
    cam = int(a.cam) if str(a.cam).isdigit() else a.cam
    run(a.class_id, cam, a.out)
