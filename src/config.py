import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_RAW   = ROOT / "data" / "raw"
DATA_PROC  = ROOT / "data" / "processed"
CKPT_DIR   = ROOT / "experiments" / "checkpoints"
LOG_DIR    = ROOT / "experiments" / "logs"

# Kaggle GTSRB copy (comma-CSV + .png) that ships with this repo, one level
# above project/.  This is what src/dataset.py reads directly — no conversion.
# Override with the GTSRB_KAGGLE_ROOT env var if the copy lives elsewhere.
KAGGLE_ROOT = Path(
    os.environ.get(
        "GTSRB_KAGGLE_ROOT",
        ROOT.parent / "datasets" / "meowmeowmeowmeowmeow"
        / "gtsrb-german-traffic-sign" / "versions" / "1",
    )
)

def _int_env(name: str, default: int) -> int:
    """Override a knob from the environment (smoke runs without code edits),
    e.g.  set GTSRB_EPOCHS_FULL=1  before  python train.py"""
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# 43 German GTSRB + 5 RU (43 ped-crossing, 44 speed bump, 45 parking,
# 46 no-stopping, 47 no-parking).
# Must stay in lockstep with camera.CLASS_NAMES (asserted there).
NUM_CLASSES = 48

# RU sign crops produced by tools/rtsd_to_crops.py:
#   <RTSD_CROPS>/{train,val}/{43,44,45}/*.jpg
RTSD_CROPS = Path(
    os.environ.get("RTSD_CROPS_ROOT", ROOT.parent / "datasets" / "rtsd_crops")
)

IMG_SIZE    = 48        # resize all images to 48x48
# 64 OOMs EfficientNet-B0 on the MX330 (2 GB VRAM). 16 fits with AMP; drop to
# GTSRB_BATCH_SIZE=8 if you still see CUDA out-of-memory.
BATCH_SIZE  = _int_env("GTSRB_BATCH_SIZE", 16)
NUM_WORKERS = _int_env("GTSRB_NUM_WORKERS", 2)

# Training
LR_HEAD     = 1e-3      # phase 1: classifier head only
LR_FULL     = 1e-4      # phase 2: full fine-tune
EPOCHS_HEAD = _int_env("GTSRB_EPOCHS_HEAD", 5)
EPOCHS_FULL = _int_env("GTSRB_EPOCHS_FULL", 30)
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1

# ImageNet stats (EfficientNet pretrained on ImageNet)
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# Mixed precision
USE_AMP = True
