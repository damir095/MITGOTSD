from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_RAW   = ROOT / "data" / "raw"
DATA_PROC  = ROOT / "data" / "processed"
CKPT_DIR   = ROOT / "experiments" / "checkpoints"
LOG_DIR    = ROOT / "experiments" / "logs"

NUM_CLASSES = 43
IMG_SIZE    = 48        # resize all images to 48x48
BATCH_SIZE  = 64
NUM_WORKERS = 2

# Training
LR_HEAD     = 1e-3      # phase 1: classifier head only
LR_FULL     = 1e-4      # phase 2: full fine-tune
EPOCHS_HEAD = 5
EPOCHS_FULL = 30
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1

# ImageNet stats (EfficientNet pretrained on ImageNet)
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# Mixed precision
USE_AMP = True
