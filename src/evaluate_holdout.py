"""
Evaluate the trained PyTorch model on the OUT-OF-DOMAIN hold-out set
(real webcam crops captured with src.holdout_capture).

This is the metric that actually matters.  GTSRB-test accuracy says how well
the model memorised the clean domain; hold-out accuracy says whether it works
on this camera.  The whole point of Этап 2 (heavy augmentation) is to move
*this* number, not the GTSRB one.

Layout read:  <holdout>/<ClassId>/*.jpg|*.png   (ClassId = 0..42)

Standalone:
    python -m src.evaluate_holdout [--weights ...best.pt] [--holdout data/holdout]
Also called automatically at the end of train.py if the folder is populated.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import recall_score, f1_score

from src.config import CKPT_DIR, NUM_CLASSES
from src.dataset import get_transforms
from src.model import build_model
from src.camera import CLASS_NAMES

_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def list_holdout(holdout_dir: Path) -> list[tuple[Path, int]]:
    """Return (image_path, class_id) for every crop under <dir>/<ClassId>/."""
    holdout_dir = Path(holdout_dir)
    items: list[tuple[Path, int]] = []
    if not holdout_dir.exists():
        return items
    for cls_dir in sorted(holdout_dir.iterdir()):
        if not cls_dir.is_dir() or not cls_dir.name.isdigit():
            continue
        cid = int(cls_dir.name)
        if not (0 <= cid < NUM_CLASSES):
            continue
        for img in cls_dir.iterdir():
            if img.suffix.lower() in _EXT:
                items.append((img, cid))
    return items


@torch.no_grad()
def run_holdout(model, device, holdout_dir: Path = Path("data/holdout")) -> dict | None:
    items = list_holdout(holdout_dir)
    if not items:
        print(f"\n[holdout] none found at {holdout_dir} — skipping "
              f"(capture with: python -m src.holdout_capture --class-id N)")
        return None

    tf = get_transforms(train=False)
    model.eval()
    y_true, y_pred = [], []
    for path, cid in items:
        try:
            x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        except OSError:
            print(f"  unreadable, skipped: {path}")
            continue
        y_pred.append(int(model(x).argmax(1).item()))
        y_true.append(cid)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = (y_true == y_pred).mean()
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*56}")
    print(f"  OUT-OF-DOMAIN HOLD-OUT  ({len(y_true)} crops, "
          f"{len(set(y_true))} classes present)")
    print(f"  Accuracy:         {acc:.4f}")
    print(f"  Recall   (macro): {rec:.4f}")
    print(f"  F1       (macro): {f1:.4f}")
    print(f"{'='*56}")
    # Per-class so it is obvious WHICH signs fail in the real world.
    for cid in sorted(set(y_true)):
        m = y_true == cid
        a = (y_pred[m] == cid).mean()
        flag = "" if a >= 0.8 else "  <-- weak"
        print(f"  [{cid:2d}] {CLASS_NAMES[cid]:<22} acc={a:.2f} (n={m.sum()}){flag}")
    return {"accuracy": acc, "recall": rec, "f1": f1, "n": len(y_true)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=CKPT_DIR / "best.pt")
    ap.add_argument("--holdout", type=Path, default=Path("data/holdout"))
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not a.weights.exists():
        raise SystemExit(f"weights not found: {a.weights} (train first)")
    model = build_model(freeze_backbone=False)
    model.load_state_dict(torch.load(a.weights, map_location=device))
    model.to(device)
    run_holdout(model, device, a.holdout)


if __name__ == "__main__":
    main()
