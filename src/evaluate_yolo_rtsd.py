"""
Honest detector-only evaluation of the GTSDB-trained YOLO on RU road scenes.

Why this exists: evaluate_b3 measures the detector on GTSDB val (German scenes,
no RU signs). It says nothing about whether the 1-class "sign" YOLO — never
trained on RTSD — actually localises the 5 RU signs we added to Stage-2.
This script answers exactly that, on real RTSD frames.

Numbers (IoU>=0.5, YOLO is class-agnostic so this is pure box localisation):

  - per RU target class: recall = GT boxes detected / GT boxes total
  - OVERALL (all RTSD sign categories in these frames): generalisation recall
    of a German-trained detector to Russian street scenes

Precision is intentionally NOT reported: RTSD frames contain many sign types
outside our 5 targets, so a YOLO box on a non-target sign is correct-but-
unscored, not a false positive — a precision number here would be misleading.

Usage:
    python -m src.evaluate_yolo_rtsd
        [--yolo runs/detect/experiments/yolo/gtsdb/weights/best.pt]
        [--yolo-imgsz 960] [--yolo-conf 0.25] [--iou 0.5] [--limit 0]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from src.evaluate_b3 import iou

ROOT = Path(__file__).resolve().parent.parent
RTSD = ROOT.parent / "datasets" / "rstd"
FRAMES = RTSD / "rtsd-frames"
VAL_ANNO = RTSD / "val_anno.json"
_YOLO_DEFAULT = Path("runs/detect/experiments/yolo/gtsdb/weights/best.pt")

# RTSD code -> (canonical Stage-2 id, readable) — same map as rtsd_to_crops.py
TARGETS = {
    "5_19_1": (43, "Pedestrian crossing"),
    "5_20":   (44, "Speed bump"),
    "6_4":    (45, "Parking"),
    "3_27":   (46, "No stopping"),
    "3_28":   (47, "No parking"),
}


def _frame_index(root: Path) -> dict[str, Path]:
    """basename -> path (nesting-proof, per CLAUDE.md lesson 7)."""
    idx: dict[str, Path] = {}
    for p in root.rglob("*.jpg"):
        idx.setdefault(p.name, p)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", type=Path, default=_YOLO_DEFAULT)
    ap.add_argument("--yolo-imgsz", type=int, default=960)
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap evaluated frames (0 = all); rare classes kept")
    a = ap.parse_args()

    if not VAL_ANNO.exists() or not FRAMES.is_dir():
        raise SystemExit(f"need {VAL_ANNO} and {FRAMES}")
    if not a.yolo.exists():
        raise SystemExit(f"weights not found: {a.yolo}")

    d = json.loads(VAL_ANNO.read_text(encoding="utf-8"))
    catid2name = {c["id"]: c["name"] for c in d["categories"]}
    id2file = {im["id"]: im["file_name"] for im in d["images"]}

    # frame -> list of (x1,y1,x2,y2, rtsd_name)  for ALL signs in that frame
    per_frame: dict[int, list] = defaultdict(list)
    for an in d["annotations"]:
        x, y, w, h = an["bbox"]
        if w < 1 or h < 1:
            continue
        per_frame[an["image_id"]].append(
            (int(x), int(y), int(x + w), int(y + h),
             catid2name.get(an["category_id"], "?")))

    # only frames that (a) hold >=1 target sign and (b) exist on disk
    fidx = _frame_index(FRAMES)
    frames = []
    for img_id, boxes in per_frame.items():
        if not any(n in TARGETS for *_, n in boxes):
            continue
        fp = fidx.get(Path(id2file[img_id]).name)
        if fp is not None:
            frames.append((img_id, fp))
    frames.sort(key=lambda t: t[1].name)

    if a.limit and len(frames) > a.limit:
        # keep every frame holding a rare class (46/47), sample the rest
        rare = {"3_27", "3_28"}
        keep = [f for f in frames
                if any(n in rare for *_, n in per_frame[f[0]])]
        rest = [f for f in frames if f not in keep]
        step = max(1, len(rest) // max(1, a.limit - len(keep)))
        frames = keep + rest[::step]
        frames.sort(key=lambda t: t[1].name)

    print(f"Evaluating {len(frames)} RTSD val frames "
          f"(imgsz={a.yolo_imgsz}, conf={a.yolo_conf}, IoU>={a.iou})")
    from ultralytics import YOLO
    yolo = YOLO(str(a.yolo))

    tgt = {nm: [0, 0] for nm in TARGETS}        # name -> [hit, total]
    ov_hit = ov_tot = 0                          # all RTSD signs
    for k, (img_id, fp) in enumerate(frames):
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        r = yolo.predict(bgr, imgsz=a.yolo_imgsz, conf=a.yolo_conf,
                         verbose=False)[0]
        preds = (r.boxes.xyxy.cpu().numpy().astype(int).tolist()
                 if r.boxes is not None and len(r.boxes) else [])
        gts = per_frame[img_id]
        used = set()
        for (gx1, gy1, gx2, gy2, name) in gts:
            ov_tot += 1
            is_t = name in TARGETS
            if is_t:
                tgt[name][1] += 1
            best, bj = a.iou, -1
            for pj, pb in enumerate(preds):
                if pj in used:
                    continue
                v = iou((gx1, gy1, gx2, gy2), pb)
                if v >= best:
                    best, bj = v, pj
            if bj >= 0:
                used.add(bj)
                ov_hit += 1
                if is_t:
                    tgt[name][0] += 1
        if (k + 1) % 200 == 0:
            print(f"  ...{k + 1}/{len(frames)}")

    print(f"\n{'='*60}")
    print(f"  YOLO on RTSD val ({len(frames)} RU frames) — detector only")
    print(f"{'='*60}")
    print("  RU target-class recall (does YOLO localise these at all):")
    for nm, (cid, label) in TARGETS.items():
        hit, tot = tgt[nm]
        rec = hit / tot if tot else 0.0
        flag = "" if rec >= 0.6 else ("  <- BLIND" if rec < 0.2 else "  <- weak")
        print(f"   [{cid:2d}] {label:<20} {nm:<7} "
              f"{hit:>4}/{tot:<4} = {rec:.3f}{flag}")
    print(f"{'-'*60}")
    print(f"  OVERALL recall (ALL RTSD sign types in these frames): "
          f"{ov_hit}/{ov_tot} = {ov_hit / ov_tot if ov_tot else 0:.3f}")
    print(f"  (German-trained YOLO generalisation to RU street scenes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
