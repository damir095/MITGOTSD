import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.config import (
    LR_HEAD, LR_FULL, EPOCHS_HEAD, EPOCHS_FULL,
    WEIGHT_DECAY, LABEL_SMOOTH, USE_AMP, CKPT_DIR
)


def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, leave=False, desc="train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast("cuda", enabled=USE_AMP):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, leave=False, desc="eval"):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast("cuda", enabled=USE_AMP):
            logits = model(imgs)
        # CE in fp32: fp16 logits on an under-trained net overflow -> nan
        # val loss (cosmetic for acc-based selection, but hides the signal).
        loss = criterion(logits.float(), labels)

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


def run_training(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler    = GradScaler("cuda", enabled=USE_AMP)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    # ── Phase 1: head only ──────────────────────────────────────────────────
    print("\n=== Phase 1: training classifier head ===")
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_HEAD)

    for epoch in range(1, EPOCHS_HEAD + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"[P1 {epoch:02d}/{EPOCHS_HEAD}] "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f}")

    # ── Phase 2: full fine-tune ─────────────────────────────────────────────
    print("\n=== Phase 2: full fine-tune ===")
    from src.model import unfreeze_all
    unfreeze_all(model)

    optimizer = AdamW(model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FULL)

    for epoch in range(1, EPOCHS_FULL + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"[P2 {epoch:02d}/{EPOCHS_FULL}] "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), CKPT_DIR / "best.pt")
            print(f"  ✓ saved best model (val acc={vl_acc:.4f})")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    return model
