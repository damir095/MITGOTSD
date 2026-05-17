"""
Main training entry point.

Usage:
    python train.py
"""
import sys
from pathlib import Path

# allow `from src.xxx import ...` from project root
sys.path.insert(0, str(Path(__file__).parent))

import torch

from src.config import KAGGLE_ROOT, CKPT_DIR
from src.dataset import get_loaders
from src.model import build_model
from src.train import run_training
from src.evaluate import predict, print_report


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not (KAGGLE_ROOT / "Train.csv").exists():
        print(
            f"\nKaggle GTSRB copy not found at {KAGGLE_ROOT}\n"
            "Expected Train.csv / Test.csv (comma-CSV) + Train/ Test/ .png.\n"
            "Set GTSRB_KAGGLE_ROOT to the dataset root if it lives elsewhere.\n"
        )
        return

    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(KAGGLE_ROOT)
    print(f"  train={len(train_loader.dataset)}  "
          f"val={len(val_loader.dataset)}  "
          f"test={len(test_loader.dataset)}")

    model = build_model(freeze_backbone=True).to(device)
    run_training(model, train_loader, val_loader, device)

    # ── Evaluate best checkpoint on test set ───────────────────────────────
    print("\nEvaluating best checkpoint on test set...")
    model.load_state_dict(torch.load(CKPT_DIR / "best.pt", map_location=device))
    y_true, y_pred = predict(model, test_loader, device)
    print_report(y_true, y_pred)

    # ── Out-of-domain hold-out (the metric that actually matters) ───────────
    # GTSRB-test accuracy only proves clean-domain memorisation. If you have
    # captured real webcam crops (src.holdout_capture), report on them too.
    from src.evaluate_holdout import run_holdout
    run_holdout(model, device, Path("data/holdout"))


if __name__ == "__main__":
    main()
