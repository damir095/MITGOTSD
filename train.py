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

from src.config import DATA_RAW, CKPT_DIR
from src.dataset import get_loaders
from src.model import build_model
from src.train import run_training
from src.evaluate import predict, print_report


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not DATA_RAW.exists() or not any(DATA_RAW.iterdir()):
        print(
            f"\nDataset not found at {DATA_RAW}\n"
            "Download GTSRB from https://benchmark.ini.rub.de/gtsrb_dataset.html\n"
            "and extract so the structure is:\n"
            "  data/raw/Train/<ClassId>/  (PPM images + GT-*.csv)\n"
            "  data/raw/Test/             (PPM images)\n"
            "  data/raw/GT-final_test.csv\n"
        )
        return

    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(DATA_RAW)
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


if __name__ == "__main__":
    main()
