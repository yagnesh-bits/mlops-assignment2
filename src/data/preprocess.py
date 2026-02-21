"""
src/data/preprocess.py
Data loading, augmentation, and preprocessing for Cats vs Dogs dataset.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]   # ImageNet stats
STD  = [0.229, 0.224, 0.225]
CLASSES = ["cat", "dog"]


# ── Transforms ───────────────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    """Augmented transform for training set."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_val_transform() -> transforms.Compose:
    """Deterministic transform for validation / test / inference."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ── Dataset ──────────────────────────────────────────────────────────────────

class CatsDogsDataset(Dataset):
    """
    Expects a directory layout:
        root/
          cat.0.jpg
          cat.1.jpg
          dog.0.jpg
          ...
    OR the standard ImageFolder layout:
        root/
          cat/  *.jpg
          dog/  *.jpg
    """

    def __init__(self, root: str, transform=None, imagefolder: bool = False):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        if imagefolder:
            for cls_idx, cls_name in enumerate(CLASSES):
                cls_dir = self.root / cls_name
                if cls_dir.exists():
                    for img_path in cls_dir.glob("*.jpg"):
                        self.samples.append((img_path, cls_idx))
        else:
            for img_path in self.root.glob("*.jpg"):
                stem = img_path.stem.lower()
                if stem.startswith("cat"):
                    self.samples.append((img_path, 0))
                elif stem.startswith("dog"):
                    self.samples.append((img_path, 1))

        logger.info(f"Loaded {len(self.samples)} samples from {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Preprocessing Pipeline ───────────────────────────────────────────────────

def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load a single image from disk and apply val transform.
    Returns a (1, 3, 224, 224) tensor ready for inference.
    """
    transform = get_val_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)
    return tensor.unsqueeze(0)


def preprocess_pil_image(pil_image: Image.Image) -> torch.Tensor:
    """Transform a PIL image to a model-ready tensor (batch size 1)."""
    transform = get_val_transform()
    tensor = transform(pil_image)
    return tensor.unsqueeze(0)


def split_dataset(
    raw_dir: str,
    processed_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[int, int, int]:
    """
    Split flat raw dataset into train/val/test sub-directories.

    Args:
        raw_dir:       Directory containing cat.*.jpg / dog.*.jpg files.
        processed_dir: Output directory for split data.
        train_ratio:   Fraction for training.
        val_ratio:     Fraction for validation.
        seed:          Random seed.

    Returns:
        Tuple of (n_train, n_val, n_test) counts.
    """
    np.random.seed(seed)
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    cat_files = sorted(raw_dir.glob("cat.*.jpg"))
    dog_files = sorted(raw_dir.glob("dog.*.jpg"))

    def _split(files):
        n = len(files)
        idx = np.random.permutation(n)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        return (
            [files[i] for i in idx[:n_train]],
            [files[i] for i in idx[n_train:n_train + n_val]],
            [files[i] for i in idx[n_train + n_val:]],
        )

    cat_train, cat_val, cat_test = _split(cat_files)
    dog_train, dog_val, dog_test = _split(dog_files)

    splits = {
        "train": cat_train + dog_train,
        "val":   cat_val   + dog_val,
        "test":  cat_test  + dog_test,
    }

    for split_name, files in splits.items():
        for cls in CLASSES:
            (processed_dir / split_name / cls).mkdir(parents=True, exist_ok=True)
        for src in files:
            cls = "cat" if src.stem.startswith("cat") else "dog"
            dst = processed_dir / split_name / cls / src.name
            shutil.copy2(src, dst)

    n_train = len(splits["train"])
    n_val   = len(splits["val"])
    n_test  = len(splits["test"])
    logger.info(f"Split complete – train:{n_train} val:{n_val} test:{n_test}")
    return n_train, n_val, n_test


# ── DataLoader Factories ─────────────────────────────────────────────────────

def get_dataloaders(
    processed_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train / val / test DataLoaders."""
    processed_dir = Path(processed_dir)

    train_ds = CatsDogsDataset(
        processed_dir / "train", get_train_transform(), imagefolder=True
    )
    val_ds = CatsDogsDataset(
        processed_dir / "val", get_val_transform(), imagefolder=True
    )
    test_ds = CatsDogsDataset(
        processed_dir / "test", get_val_transform(), imagefolder=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
