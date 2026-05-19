"""
RTSD detection annotations + extracted frames  ->  1-class YOLO dataset.

Why: the GTSDB-trained detector is blind to RU blue-square signs (recall
0.03-0.09 on RTSD — see src/evaluate_yolo_rtsd.py). To fix that without
losing German performance we train a *combined* 1-class detector on
GTSDB + RTSD scenes. This builds the RTSD half.

Every annotation in a kept frame (any sign type) collapses to class 0
"sign" — exactly like tools/gtsdb_to_yolo.py. Train/val split follows
RTSD's own train_anno / val_anno (no leakage).

The on-disk rtsd-frames are the selectively-extracted ~20k frames (those
containing our 5 target classes). "Pedestrian crossing only" (5_19_1) is
overwhelmingly dominant (~12.4k frames) and is capped so the rarer
blue-square scenes (5_20/6_4) and the upload size stay sane.

Output (Ultralytics layout):
    datasets/rtsd_yolo/{images,labels}/{train,val}/...

Usage:
    python tools/rtsd_to_yolo.py            # cap "5_19_1-only" train frames
    python tools/rtsd_to_yolo.py --cap-common 0   # keep everything
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEF_RTSD = ROOT.parent / "datasets" / "rstd"
DEF_OUT = ROOT.parent / "datasets" / "rtsd_yolo"

TARGETS = {"5_19_1", "5_20", "6_4", "3_27", "3_28"}
# frames whose target set is a subset of this get capped (the dominant,
# easy-to-find class — we keep plenty but not all 12k of them)
COMMON = {"5_19_1"}


def frame_index(frames_root: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in frames_root.rglob("*.jpg"):
        idx.setdefault(p.name, p)
    return idx


def build(anno_path: Path, fidx: dict[str, Path], split: str,
          out: Path, cap_common: int) -> dict:
    d = json.loads(anno_path.read_text(encoding="utf-8"))
    cat = {c["id"]: c["name"] for c in d["categories"]}
    img = {im["id"]: im for im in d["images"]}

    boxes = defaultdict(list)          # img_id -> [(x,y,w,h), ...] all signs
    targets = defaultdict(set)         # img_id -> {target names present}
    for an in d["annotations"]:
        boxes[an["image_id"]].append(an["bbox"])
        nm = cat.get(an["category_id"])
        if nm in TARGETS:
            targets[an["image_id"]].add(nm)

    # candidate frames: have >=1 target sign AND exist on disk
    cand = []
    for img_id, tset in targets.items():
        if not tset:
            continue
        fp = fidx.get(Path(img[img_id]["file_name"]).name)
        if fp is not None:
            cand.append((img_id, fp, tset))
    cand.sort(key=lambda t: t[1].name)   # deterministic

    common, kept = 0, []
    for img_id, fp, tset in cand:
        if split == "train" and cap_common and tset <= COMMON:
            common += 1
            if common > cap_common:
                continue
        kept.append((img_id, fp))

    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_box = 0
    for img_id, fp in kept:
        meta = img[img_id]
        W, H = meta["width"], meta["height"]
        lines = []
        for (x, y, w, h) in boxes[img_id]:
            if w < 1 or h < 1:
                continue
            cx = min(max((x + w / 2) / W, 0.0), 1.0)
            cy = min(max((y + h / 2) / H, 0.0), 1.0)
            nw = min(max(w / W, 0.0), 1.0)
            nh = min(max(h / H, 0.0), 1.0)
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_box += 1
        if not lines:
            continue
        shutil.copy2(fp, img_dir / fp.name)
        (lbl_dir / (fp.stem + ".txt")).write_text("\n".join(lines) + "\n")

    return {"frames": len(kept), "boxes": n_box,
            "capped_skipped": max(0, common - cap_common) if cap_common else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtsd", type=Path, default=DEF_RTSD)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--cap-common", type=int, default=2500,
                    help="max train frames whose only target is 5_19_1 "
                         "(0 = keep all)")
    a = ap.parse_args()

    fr = a.rtsd / "rtsd-frames"
    if not fr.is_dir():
        raise SystemExit(f"run tools/rtsd_extract.py first: {fr} missing")
    if a.out.exists():
        shutil.rmtree(a.out)
    fidx = frame_index(fr)

    for split, af in (("train", "train_anno.json"), ("val", "val_anno.json")):
        ap_ = a.rtsd / af
        if not ap_.exists():
            raise SystemExit(f"missing {ap_}")
        r = build(ap_, fidx, split, a.out, a.cap_common)
        print(f"{split}: {r['frames']} frames, {r['boxes']} sign boxes"
              + (f"  (capped -{r['capped_skipped']} '5_19_1-only')"
                 if r["capped_skipped"] else ""))

    yaml = (a.out / "rtsd.yaml")
    yaml.write_text(
        f"path: {a.out.as_posix()}\n"
        "train: images/train\nval: images/val\n"
        "nc: 1\nnames: ['sign']\n")
    print(f"\nout -> {a.out}\nyaml -> {yaml}")


if __name__ == "__main__":
    main()
