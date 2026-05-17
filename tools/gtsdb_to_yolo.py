"""
GTSDB (FullIJCNN2013, .ppm + gt.txt)  ->  Ultralytics YOLO dataset.

B3 architecture: YOLO only learns to LOCATE signs, so every GTSDB class
(0..42) collapses to a single class `sign`.  The 43 cropped-sign subfolders
(00..42) inside the Train dir are NOT scene images — they are excluded; only
the 600 full-scene frames are used.  Frames with no annotation are kept as
negatives (empty label files) — they cut detector false positives.

gt.txt rows:  filename;x1;y1;x2;y2;ClassID   (';'-sep, pixel xyxy)
Data is dirty (trailing spaces, CRLF) — parsing strips robustly.

Output (Ultralytics layout, sibling of the raw dataset):
    datasets/gtsdb_yolo/
        images/{train,val}/*.jpg      (.ppm transcoded to JPEG)
        labels/{train,val}/*.txt      (normalised cx cy w h, class 0)
        gtsdb.yaml

Usage (CPU only, no GPU needed):
    python tools/gtsdb_to_yolo.py
"""
import argparse
import random
from pathlib import Path

from PIL import Image

# project/ -> repo root -> datasets/
ROOT = Path(__file__).resolve().parent.parent
DEF_SRC = ROOT.parent / "datasets" / "gtsdb"
DEF_OUT = ROOT.parent / "datasets" / "gtsdb_yolo"


def parse_gt(gt_path: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    """filename -> list of (x1, y1, x2, y2). Class is dropped (1-class)."""
    boxes: dict[str, list[tuple[int, int, int, int]]] = {}
    for raw in gt_path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 5:
            continue
        fn = parts[0]
        try:
            x1, y1, x2, y2 = (int(parts[i]) for i in range(1, 5))
        except ValueError:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.setdefault(fn, []).append((x1, y1, x2, y2))
    return boxes


def to_yolo(box, w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    return cx, cy, bw, bh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEF_SRC,
                    help="raw GTSDB root (contains TrainIJCNN2013/, gt.txt)")
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    train_dir = a.src / "TrainIJCNN2013" / "TrainIJCNN2013"
    gt_path = train_dir / "gt.txt"
    if not gt_path.exists():
        gt_path = a.src / "gt.txt"
    if not train_dir.is_dir() or not gt_path.exists():
        raise SystemExit(f"GTSDB not found: {train_dir} / {gt_path}")

    boxes = parse_gt(gt_path)
    # ONLY the 600 full-scene frames (NNNNN.ppm at train_dir top level),
    # never the 00..42 cropped-sign subfolders.
    scenes = sorted(p for p in train_dir.glob("*.ppm"))
    if not scenes:
        raise SystemExit(f"no scene .ppm in {train_dir}")

    random.seed(a.seed)
    shuffled = scenes[:]
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * a.val_frac)
    val_set = {p.name for p in shuffled[:n_val]}

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)

    n = {"train": 0, "val": 0}
    n_boxes = {"train": 0, "val": 0}
    n_neg = {"train": 0, "val": 0}
    for ppm in scenes:
        split = "val" if ppm.name in val_set else "train"
        stem = ppm.stem
        with Image.open(ppm) as im:
            im = im.convert("RGB")
            w, h = im.size
            im.save(a.out / f"images/{split}/{stem}.jpg", quality=95)

        lines = []
        for b in boxes.get(ppm.name, []):
            cx, cy, bw, bh = to_yolo(b, w, h)
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (a.out / f"labels/{split}/{stem}.txt").write_text("\n".join(lines))
        n[split] += 1
        n_boxes[split] += len(lines)
        if not lines:
            n_neg[split] += 1

    yaml = (
        f"# GTSDB, single class for B3 (YOLO locates, EfficientNet classifies)\n"
        f"path: {a.out.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['sign']\n"
    )
    (a.out / "gtsdb.yaml").write_text(yaml)

    print(f"images: train={n['train']} val={n['val']}")
    print(f"boxes:  train={n_boxes['train']} val={n_boxes['val']}")
    print(f"negatives (no sign): train={n_neg['train']} val={n_neg['val']}")
    print(f"wrote {a.out/'gtsdb.yaml'}")


if __name__ == "__main__":
    main()
