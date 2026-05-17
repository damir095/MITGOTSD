"""
GTSRB loader — reads the **Kaggle-format** copy directly (comma-CSV + .png).

The Kaggle Train.csv / Test.csv already carry everything we need:

    Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
    27,26,5,5,22,20,20,Train/20/00020_00000_00000.png

so there is no on-disk conversion step.  ``ClassId`` is taken straight from
the CSV as the integer label (0..42) — this sidesteps the ImageFolder
string-sort class-ordering trap ('0','1','10','11','2',...).

ROI cropping: like the Keras v2 notebook, every image is cropped to its
``Roi.*`` box so the sign fills the frame.  Inference (the colour detector)
feeds the classifier an approximately tight box, so training on tight crops
keeps train/inference framing consistent.
"""
import csv
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import (
    KAGGLE_ROOT, RTSD_CROPS, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, MEAN, STD
)


class GTSRBKaggleDataset(Dataset):
    """One Kaggle CSV split (Train.csv or Test.csv), ROI-cropped."""

    def __init__(self, root: Path, csv_name: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        # (img_path, class_id, (x1, y1, x2, y2))
        self.samples: list[tuple[Path, int, tuple[int, int, int, int]]] = []
        skipped = 0

        csv_path = self.root / csv_name
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)             # comma-delimited
            for row in reader:
                x1, y1 = int(row["Roi.X1"]), int(row["Roi.Y1"])
                x2, y2 = int(row["Roi.X2"]), int(row["Roi.Y2"])
                # Degenerate ROI → unusable crop; drop it (same guard as the
                # notebook's error counter).
                if x2 <= x1 or y2 <= y1:
                    skipped += 1
                    continue
                img_path = self.root / row["Path"]
                self.samples.append(
                    (img_path, int(row["ClassId"]), (x1, y1, x2, y2))
                )

        if not self.samples:
            raise RuntimeError(
                f"No usable rows in {csv_path}. Check GTSRB_KAGGLE_ROOT "
                f"(currently {self.root})."
            )
        if skipped:
            print(f"  {csv_name}: skipped {skipped} rows with degenerate ROI")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, (x1, y1, x2, y2) = self.samples[idx]
        img = Image.open(path).convert("RGB").crop((x1, y1, x2, y2))
        if self.transform:
            img = self.transform(img)
        return img, label


class _AddGaussianNoise:
    """Sensor/compression noise. Module-level class so it pickles for
    DataLoader workers on Windows (a lambda would not)."""

    def __init__(self, std: float = 0.04):
        self.std = std

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t + torch.randn_like(t) * self.std

    def __repr__(self):
        return f"{type(self).__name__}(std={self.std})"


def get_transforms(train: bool):
    """
    Train transforms are deliberately heavy **domain randomization**.

    GTSRB is clean daylight photos; the live input is a sign seen through a
    webcam (warm white balance, glare, softness) framed by a colour detector
    (loose, off-centre, slightly skewed).  The previous mild augmentation let
    the model memorise the clean domain (val acc 0.99 by epoch 3) and it then
    failed completely on real input.  Each transform below targets one of the
    observed gaps.  Eval transforms stay clean — never augment evaluation.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # framing: detector boxes are off-centre / loose / rotated / skewed
            transforms.RandomAffine(
                degrees=15, translate=(0.12, 0.12),
                scale=(0.80, 1.20), shear=8,
            ),
            transforms.RandomPerspective(distortion_scale=0.30, p=0.5),
            # lighting / white-balance: warm room light, screen colour cast
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.10,
            ),
            # webcam softness / occasional motion blur
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                p=0.3,
            ),
            transforms.RandomHorizontalFlip(p=0.0),   # signs are NOT symmetric
            transforms.ToTensor(),
            _AddGaussianNoise(std=0.04),               # sensor/JPEG noise
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class RtsdCropsDataset(Dataset):
    """RU sign crops: <root>/<split>/<class_id>/*.jpg.

    Label is the **folder name parsed as int** (43/44/45) — deliberately NOT
    ImageFolder, whose string-sorted class indices are the well-known
    class-ordering trap. Empty/missing dir -> length 0 (extension optional).
    """

    def __init__(self, root: Path, split: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        base = Path(root) / split
        if not base.is_dir():
            return
        for cls_dir in sorted(base.iterdir()):
            if not cls_dir.is_dir() or not cls_dir.name.isdigit():
                continue
            cid = int(cls_dir.name)
            for img in cls_dir.iterdir():
                if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((img, cid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_loaders(root: Path = KAGGLE_ROOT, rtsd_root: Path = RTSD_CROPS):
    root = Path(root)
    train_ds = GTSRBKaggleDataset(root, "Train.csv",
                                  transform=get_transforms(train=True))
    test_ds  = GTSRBKaggleDataset(root, "Test.csv",
                                  transform=get_transforms(train=False))

    # 90/10 val split from training data
    val_size  = int(0.1 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_split, val_split = torch.utils.data.random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # val Subset must use test-time transforms: point it at a second dataset
    # instance built without augmentation (same seed-42 index split is shared).
    val_split.dataset = GTSRBKaggleDataset(
        root, "Train.csv", transform=get_transforms(train=False)
    )

    # RU extension: concat RTSD crops (own train/val split, no leakage).
    # Absent rtsd_crops -> empty datasets -> behaviour identical to pre-RU.
    rtsd_tr = RtsdCropsDataset(rtsd_root, "train", get_transforms(train=True))
    rtsd_va = RtsdCropsDataset(rtsd_root, "val",   get_transforms(train=False))
    Concat = torch.utils.data.ConcatDataset
    train_set = Concat([train_split, rtsd_tr]) if len(rtsd_tr) else train_split
    val_set   = Concat([val_split, rtsd_va])   if len(rtsd_va) else val_split
    print(f"  train: GTSRB={train_size} + RTSD={len(rtsd_tr)}  |  "
          f"val: GTSRB={val_size} + RTSD={len(rtsd_va)}  |  "
          f"test(GTSRB only)={len(test_ds)}")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    return train_loader, val_loader, test_loader
