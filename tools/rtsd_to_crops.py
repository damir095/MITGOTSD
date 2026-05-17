"""
RTSD detection annotations + extracted frames  ->  Stage-2 sign crops.

Crops the bounding box of each target RU sign out of its frame, so the
EfficientNet classifier can be retrained on GTSRB(43) + these new classes.
Train/val split follows RTSD's own train_anno / val_anno (no leakage).

Canonical new class ids (German GTSRB keeps 0..42, RU appended):
    43  pedestrian_crossing   (RTSD 5_19_1)
    44  speed_bump            (RTSD 5_20)
    45  parking               (RTSD 6_4)

Output (ImageFolder-style, consumed by the extended Stage-2 loader):
    datasets/rtsd_crops/{train,val}/<class_id>/<n>.jpg

Usage (after tools/rtsd_extract.py finished):
    python tools/rtsd_to_crops.py
"""
import argparse
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEF_RTSD = ROOT.parent / "datasets" / "rstd"
DEF_OUT = ROOT.parent / "datasets" / "rtsd_crops"

# RTSD code -> (canonical class id, readable name)
TARGETS = {
    "5_19_1": (43, "Pedestrian crossing"),
    "5_20":   (44, "Speed bump"),
    "6_4":    (45, "Parking"),
    "3_27":   (46, "No stopping"),
    "3_28":   (47, "No parking"),
}


def process(anno_path: Path, frames_root: Path, split: str,
            out_root: Path) -> dict:
    d = json.loads(anno_path.read_text(encoding="utf-8"))
    catid2name = {c["id"]: c["name"] for c in d["categories"]}
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    counts = {cid: 0 for cid, _ in TARGETS.values()}
    missing_frames = set()

    for an in d["annotations"]:
        name = catid2name.get(an["category_id"])
        if name not in TARGETS:
            continue
        cls_id, _ = TARGETS[name]
        # frames were normalised to a single rtsd-frames/<file> level
        frame = frames_root / Path(id2file[an["image_id"]]).name
        if not frame.exists():
            missing_frames.add(frame.name)
            continue
        x, y, w, h = an["bbox"]
        x, y = max(0, int(x)), max(0, int(y))
        w, h = int(w), int(h)
        if w < 4 or h < 4:
            continue
        out_dir = out_root / split / str(cls_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(frame) as im:
                crop = im.convert("RGB").crop((x, y, x + w, y + h))
                crop.save(out_dir / f"{an['id']}.jpg", quality=95)
        except OSError:
            continue
        counts[cls_id] += 1

    return {"counts": counts, "missing": len(missing_frames)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtsd", type=Path, default=DEF_RTSD)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    a = ap.parse_args()

    frames_root = a.rtsd / "rtsd-frames"
    if not frames_root.is_dir():
        raise SystemExit(f"run tools/rtsd_extract.py first: {frames_root} missing")

    total = {cid: 0 for cid, _ in TARGETS.values()}
    for split, af in (("train", "train_anno.json"), ("val", "val_anno.json")):
        ap_ = a.rtsd / af
        if not ap_.exists():
            raise SystemExit(f"missing {ap_}")
        r = process(ap_, frames_root, split, a.out)
        print(f"{split}: {r['counts']}"
              + (f"  (skipped {r['missing']} not-extracted frames)"
                 if r["missing"] else ""))
        for k, v in r["counts"].items():
            total[k] += v

    print("\nTOTAL crops per class:")
    name = {cid: nm for _, (cid, nm) in
            zip(TARGETS, TARGETS.values())}
    for cid in sorted(total):
        print(f"  [{cid}] {name[cid]:<22} {total[cid]}")
    print(f"\nout -> {a.out}")


if __name__ == "__main__":
    main()
