"""
Honest IN-DOMAIN evaluation of the full B3 system on GTSDB val.

Why this exists: the phone-screen hold-out measured the wrong domain. B3 is a
*road-scene* sign recogniser (GTSDB detector + GTSRB classifier). The 90 GTSDB
val frames are real German street scenes YOLO never trained on, and gt.txt has
their true class IDs (0..42, same scheme as GTSRB). So this is the real metric.

Three separated numbers (so we know WHICH stage is the bottleneck):

  1. Detector  — recall / precision of YOLO on real scenes (IoU>=0.5).
  2. Classifier — EfficientNet accuracy on the *true* sign boxes (every GT
     box cropped from the real image), independent of the detector. This is
     the honest replacement for the phone hold-out.
  3. End-to-end — fraction of GT signs both detected AND correctly classified
     from the predicted box (what the real system actually delivers).

Usage:
    python -m src.evaluate_b3
        [--yolo runs/detect/experiments/yolo/gtsdb/weights/best.pt]
        [--clf experiments/checkpoints/best.pt]
        [--yolo-imgsz 960] [--yolo-conf 0.25] [--dump 20]
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.config import CKPT_DIR
from src.model import build_model
from src.dataset import get_transforms
from src.camera import CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent
VAL_DIR = ROOT.parent / "datasets" / "gtsdb_yolo" / "images" / "val"
GT_TXT = (ROOT.parent / "datasets" / "gtsdb"
          / "TrainIJCNN2013" / "TrainIJCNN2013" / "gt.txt")
_YOLO_DEFAULT = Path("runs/detect/experiments/yolo/gtsdb/weights/best.pt")


def parse_gt_with_class(p: Path) -> dict[str, list[tuple]]:
    """stem -> [(x1,y1,x2,y2,classid), ...]; robust to trailing spaces/CRLF."""
    out: dict[str, list[tuple]] = defaultdict(list)
    for raw in p.read_text(errors="replace").splitlines():
        parts = [s.strip() for s in raw.strip().split(";")]
        if len(parts) < 6:
            continue
        try:
            x1, y1, x2, y2, cid = (int(parts[i]) for i in range(1, 6))
        except ValueError:
            continue
        if x2 > x1 and y2 > y1 and 0 <= cid <= 42:
            out[Path(parts[0]).stem].append((x1, y1, x2, y2, cid))
    return out


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua


@torch.no_grad()
def _classify(clf, tf, device, img_rgb, boxes):
    if not boxes:
        return []
    crops = []
    for (x1, y1, x2, y2) in boxes:
        c = img_rgb[max(0, y1):y2, max(0, x1):x2]
        if c.size == 0:
            c = np.zeros((8, 8, 3), np.uint8)
        crops.append(tf(Image.fromarray(c)))
    out = clf(torch.stack(crops).to(device))
    return out.argmax(1).cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", type=Path, default=_YOLO_DEFAULT)
    ap.add_argument("--clf", type=Path, default=CKPT_DIR / "best.pt")
    ap.add_argument("--yolo-imgsz", type=int, default=960)
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--dump", type=int, default=20,
                    help="save this many annotated val frames for eyeballing")
    a = ap.parse_args()

    if not VAL_DIR.is_dir() or not GT_TXT.exists():
        raise SystemExit(f"need {VAL_DIR} and {GT_TXT}")
    for p in (a.yolo, a.clf):
        if not p.exists():
            raise SystemExit(f"weights not found: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    from ultralytics import YOLO
    yolo = YOLO(str(a.yolo))
    clf = build_model(freeze_backbone=False)
    clf.load_state_dict(torch.load(a.clf, map_location=device))
    clf.to(device).eval()
    tf = get_transforms(train=False)

    gt = parse_gt_with_class(GT_TXT)
    imgs = sorted(VAL_DIR.glob("*.jpg"))
    dump_dir = ROOT / "experiments" / "b3_eval"
    dump_dir.mkdir(parents=True, exist_ok=True)

    n_gt = n_pred = n_matched = 0
    clf_correct = clf_total = 0          # classifier on TRUE boxes
    e2e_correct = 0                      # detected AND correctly classified
    per_cls = defaultdict(lambda: [0, 0])  # cid -> [e2e_correct, gt_count]

    for i, ip in enumerate(imgs):
        bgr = cv2.imread(str(ip))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gts = gt.get(ip.stem, [])
        n_gt += len(gts)

        r = yolo.predict(bgr, imgsz=a.yolo_imgsz, conf=a.yolo_conf,
                         verbose=False)[0]
        preds = (r.boxes.xyxy.cpu().numpy().astype(int).tolist()
                 if r.boxes is not None and len(r.boxes) else [])
        n_pred += len(preds)

        # 2. classifier on the true sign boxes (detector-independent)
        true_boxes = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in gts]
        clf_pred = _classify(clf, tf, device, rgb, true_boxes)
        for (_, _, _, _, cid), pc in zip(gts, clf_pred):
            clf_total += 1
            clf_correct += int(pc == cid)
            per_cls[cid][1] += 1

        # 1+3. match preds to GT, end-to-end on predicted boxes
        pred_cls = _classify(clf, tf, device, rgb, preds)
        used = set()
        for gi, (gx1, gy1, gx2, gy2, cid) in enumerate(gts):
            best, bj = a.iou, -1
            for pj, pb in enumerate(preds):
                if pj in used:
                    continue
                v = iou((gx1, gy1, gx2, gy2), pb)
                if v >= best:
                    best, bj = v, pj
            if bj >= 0:
                used.add(bj)
                n_matched += 1
                if pred_cls[bj] == cid:
                    e2e_correct += 1
                    per_cls[cid][0] += 1

        if i < a.dump:
            for (x1, y1, x2, y2), pc in zip(preds, pred_cls):
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(bgr, CLASS_NAMES[pc], (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)
            cv2.imwrite(str(dump_dir / ip.name), bgr)

    det_rec = n_matched / n_gt if n_gt else 0.0
    det_prec = n_matched / n_pred if n_pred else 0.0
    clf_acc = clf_correct / clf_total if clf_total else 0.0
    e2e = e2e_correct / n_gt if n_gt else 0.0

    print(f"\n{'='*60}")
    print(f"  B3 IN-DOMAIN (GTSDB val: {len(imgs)} frames, {n_gt} signs)")
    print(f"{'='*60}")
    print(f"  1. Detector   recall={det_rec:.3f}  precision={det_prec:.3f}")
    print(f"  2. Classifier on TRUE boxes   acc={clf_acc:.3f}"
          f"  ({clf_correct}/{clf_total})   <- honest Stage-2 number")
    print(f"  3. End-to-end (detected & correct) = {e2e:.3f}"
          f"  ({e2e_correct}/{n_gt})")
    print(f"{'='*60}")
    print("  per-class end-to-end (signs with >=3 GT):")
    for cid in sorted(per_cls):
        ok, tot = per_cls[cid]
        if tot >= 3:
            flag = "" if ok / tot >= 0.6 else "  <- weak"
            print(f"   [{cid:2d}] {CLASS_NAMES[cid]:<22} "
                  f"{ok}/{tot}={ok/tot:.2f}{flag}")
    print(f"\n  annotated frames -> {dump_dir}")


if __name__ == "__main__":
    main()
