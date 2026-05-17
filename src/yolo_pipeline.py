"""
B3 inference: YOLO locates signs  ->  EfficientNet classifies each crop.

This replaces the old heuristic detector + threshold/entropy/NMS/tracking
machinery entirely.  Two models, each doing one clean job:

  Stage 1  YOLOv8n  (1 class "sign")          -> bounding boxes
  Stage 2  EfficientNet-B0 (GTSRB, 43 classes) -> the sign's class

No hand-tuned HSV ranges, no margin/entropy gates: YOLO's own confidence is
the only detection knob; the classifier just labels what YOLO found.

Usage (live camera):
    python -m src.yolo_pipeline --cam 0

Usage (test on a driving video, write annotated copy, no window):
    python -m src.yolo_pipeline --cam road.mp4 --save out.mp4 --no-show

Keys (when shown):  q quit   s save frame

--yolo-imgsz larger than the 416 used in training recovers small/distant
signs (YOLO can infer at higher res than it trained at).
--yolo-conf default 0.35 (raised from 0.25): in-domain precision was 0.805,
higher conf trims false boxes.
--persist N keeps a detection alive N frames (greedy IoU) so brief misses
don't flicker; 0 disables.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.config import CKPT_DIR
from src.model import build_model
from src.dataset import get_transforms
from src.camera import CLASS_NAMES   # English, length-43, asserted there

_YOLO_DEFAULT = Path("runs/detect/experiments/yolo/gtsdb/weights/best.pt")


def load_models(yolo_path: Path, clf_path: Path, device):
    from ultralytics import YOLO
    yolo = YOLO(str(yolo_path))
    clf = build_model(freeze_backbone=False)
    clf.load_state_dict(torch.load(clf_path, map_location=device))
    clf.to(device).eval()
    return yolo, clf


@torch.no_grad()
def classify_crops(clf, tf, device, crops_bgr):
    """Batch-classify a list of BGR crops -> [(class_id, conf), ...]."""
    if not crops_bgr:
        return []
    batch = torch.stack([
        tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
        for c in crops_bgr
    ]).to(device)
    probs = torch.softmax(clf(batch), dim=1)
    conf, cls = probs.max(dim=1)
    return list(zip(cls.cpu().tolist(), conf.cpu().tolist()))


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def run(cam, yolo_path, clf_path, yolo_imgsz, yolo_conf,
        save=None, show=True, persist=5, clf_conf=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    yolo, clf = load_models(yolo_path, clf_path, device)
    tf = get_transforms(train=False)   # same eval transform as Stage-2 training
    print("Models loaded." + ("  [q] quit  [s] save" if show else ""))

    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {cam}")
    save_dir = Path("experiments/screenshots")
    save_dir.mkdir(parents=True, exist_ok=True)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = None                       # lazily created (need frame size)

    tracks: list[dict] = []
    prev = time.time()
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1

        r = yolo.predict(frame, imgsz=yolo_imgsz, conf=yolo_conf,
                         verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy().astype(int) if r.boxes else []

        crops, kept = [], []
        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            c = frame[y1:y2, x1:x2]
            if c.size:
                crops.append(c)
                kept.append((x1, y1, x2, y2))
        preds = classify_crops(clf, tf, device, crops)
        # Classifier-confidence gate: a far/blurry or out-of-dataset sign
        # usually yields low top-1 softmax — drop it until it is read
        # confidently (this is what makes labels appear only once close).
        dets = [(b, cid, cf) for b, (cid, cf) in zip(kept, preds)
                if cf >= clf_conf]

        # ── Temporal persistence: keep a detection alive for `persist`
        # frames so brief misses don't make boxes flicker. Greedy IoU match
        # of this frame's detections to existing tracks. ───────────────────
        if persist > 0:
            for tr in tracks:
                tr["hit"] = False
            for box, cid, cf in dets:
                best, bt = 0.3, None
                for tr in tracks:
                    if tr["hit"]:
                        continue
                    v = _iou(box, tr["box"])
                    if v >= best:
                        best, bt = v, tr
                if bt is None:
                    tracks.append({})
                    bt = tracks[-1]
                bt.update(box=box, cid=cid, conf=cf, ttl=persist, hit=True)
            for tr in tracks:
                if not tr.get("hit"):
                    tr["ttl"] -= 1
            tracks = [t for t in tracks if t["ttl"] > 0]
            shown = [(t["box"], t["cid"], t["conf"]) for t in tracks]
        else:
            shown = dets

        for (x1, y1, x2, y2), cid, cf in shown:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            label = f"{CLASS_NAMES[cid]} {cf*100:.0f}%"
            cv2.putText(frame, label, (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS {fps:.1f}  signs {len(shown)}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if save is not None:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(save), cv2.VideoWriter_fourcc(*"mp4v"),
                    src_fps, (w, h))
            writer.write(frame)
            if n_total and fi % 100 == 0:
                print(f"  {fi}/{n_total} frames")

        if show:
            cv2.imshow("B3: YOLO -> EfficientNet", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                p = save_dir / f"b3_{int(time.time())}.jpg"
                cv2.imwrite(str(p), frame)
                print(f"saved {p}")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"saved annotated video -> {save}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="0", help="camera index or video path")
    ap.add_argument("--yolo", type=Path, default=_YOLO_DEFAULT)
    ap.add_argument("--clf", type=Path, default=CKPT_DIR / "best.pt")
    ap.add_argument("--yolo-imgsz", type=int, default=960,
                    help="YOLO inference resolution (higher recovers small signs)")
    ap.add_argument("--yolo-conf", type=float, default=0.35,
                    help="YOLO detection confidence (0.25->0.35: fewer "
                         "false boxes; in-domain precision was 0.805)")
    ap.add_argument("--persist", type=int, default=5,
                    help="keep a detection for N frames to stop flicker "
                         "(0 disables temporal smoothing)")
    ap.add_argument("--clf-conf", type=float, default=0.5,
                    help="hide a sign unless classifier top-1 >= this "
                         "(filters far/blurry & many out-of-dataset signs)")
    ap.add_argument("--save", type=Path, default=None,
                    help="write annotated .mp4 here (good for video tests)")
    ap.add_argument("--no-show", action="store_true",
                    help="run without a display window (use with --save)")
    a = ap.parse_args()
    cam = int(a.cam) if str(a.cam).isdigit() else a.cam
    if not a.yolo.exists():
        raise SystemExit(f"YOLO weights not found: {a.yolo}")
    if not a.clf.exists():
        raise SystemExit(f"classifier weights not found: {a.clf}")
    run(cam, a.yolo, a.clf, a.yolo_imgsz, a.yolo_conf,
        save=a.save, show=not a.no_show, persist=a.persist,
        clf_conf=a.clf_conf)
