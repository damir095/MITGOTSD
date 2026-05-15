import csv
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import (
    DATA_RAW, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, MEAN, STD
)


class GTSRBTrainDataset(Dataset):
    """GTSRB training split — reads annotation CSVs from each class folder."""

    def __init__(self, root: Path, transform=None):
        self.root = root / "Train"
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for label in sorted(self.root.iterdir()):
            if not label.is_dir():
                continue
            class_id = int(label.name)
            csv_path = label / f"GT-{label.name}.csv"
            with csv_path.open() as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    img_path = label / row["Filename"]
                    self.samples.append((img_path, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class GTSRBTestDataset(Dataset):
    """GTSRB test split — reads GT-final_test.csv from Test folder."""

    def __init__(self, root: Path, transform=None):
        self.root = root / "Test"
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        csv_path = root / "GT-final_test.csv"
        with csv_path.open() as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                img_path = self.root / row["Filename"]
                self.samples.append((img_path, int(row["ClassId"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
            transforms.RandomHorizontalFlip(p=0.0),   # signs are NOT symmetric
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_loaders(root: Path = DATA_RAW):
    train_ds = GTSRBTrainDataset(root, transform=get_transforms(train=True))
    test_ds  = GTSRBTestDataset(root,  transform=get_transforms(train=False))

    # 90/10 val split from training data
    val_size  = int(0.1 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_split, val_split = torch.utils.data.random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # val uses test-time transforms
    val_split.dataset = GTSRBTrainDataset(root, transform=get_transforms(train=False))

    train_loader = DataLoader(
        train_split, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_split, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    return train_loader, val_loader, test_loader
