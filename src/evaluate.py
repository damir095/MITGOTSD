import torch
import numpy as np
from torch.amp import autocast
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import USE_AMP, NUM_CLASSES

CLASS_NAMES = [str(i).zfill(5) for i in range(NUM_CLASSES)]  # 00000 … 00042


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        with autocast("cuda", enabled=USE_AMP):
            logits = model(imgs)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())

    return np.array(all_labels), np.array(all_preds)


def print_report(y_true, y_pred):
    recall = recall_score(y_true, y_pred, average="macro")
    f1     = f1_score(y_true, y_pred, average="macro")
    acc    = (y_true == y_pred).mean()

    print(f"\n{'='*50}")
    print(f"  Accuracy (macro): {acc:.4f}")
    print(f"  Recall   (macro): {recall:.4f}  {'✓' if recall >= 0.95 else '✗'} (target ≥ 0.95)")
    print(f"  F1       (macro): {f1:.4f}  {'✓' if f1 >= 0.95 else '✗'} (target ≥ 0.95)")
    print(f"{'='*50}\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    return {"accuracy": acc, "recall": recall, "f1": f1}


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — GTSRB Test Set")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def find_worst_classes(y_true, y_pred, n=10):
    """Return n classes with lowest per-class recall."""
    recalls = recall_score(y_true, y_pred, average=None)
    worst = np.argsort(recalls)[:n]
    print(f"\nWorst {n} classes by recall:")
    for cls in worst:
        print(f"  Class {cls:02d}: recall={recalls[cls]:.4f}")
    return worst
