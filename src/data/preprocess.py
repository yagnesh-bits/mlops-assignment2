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
CLASSES = ["Cat", "Dog"]


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


def _detect_layout(raw_dir: Path) -> str:
    """
    Detect whether raw_dir uses flat (cat.0.jpg / dog.0.jpg) or
    ImageFolder (cat/ dog/ sub-directories) layout.

    Returns: "flat" | "imagefolder"
    Raises:  ValueError if neither layout is detected.
    """
    # ImageFolder: at least one of the class sub-dirs exists and contains images
    imagefolder_hits = sum(
        1 for cls in CLASSES
        if (raw_dir / cls).is_dir() and any((raw_dir / cls).glob("*.jpg"))
    )
    if imagefolder_hits > 0:
        return "imagefolder"

    # Flat: files named cat.*.jpg / dog.*.jpg in the root
    flat_hits = sum(
        1 for cls in CLASSES
        if any(raw_dir.glob(f"{cls}.*.jpg"))
    )
    if flat_hits > 0:
        return "flat"

    raise ValueError(
        f"No recognisable image layout found in '{raw_dir}'.\n"
        f"Expected either:\n"
        f"  Flat layout   : {raw_dir}/cat.0.jpg, {raw_dir}/dog.0.jpg, …\n"
        f"  ImageFolder   : {raw_dir}/cat/*.jpg,  {raw_dir}/dog/*.jpg"
    )


def _collect_files_by_class(raw_dir: Path, layout: str) -> dict:
    """
    Return {{class_name: [Path, ...]}} for every class in CLASSES.

    Supports both 'flat' and 'imagefolder' layouts.
    """
    files_by_class: dict = {}

    if layout == "imagefolder":
        for cls in CLASSES:
            cls_dir = raw_dir / cls
            found = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.jpeg")) + sorted(cls_dir.glob("*.png"))
            files_by_class[cls] = found
            logger.info(f"ImageFolder – {cls}: {len(found)} images")

    else:  # flat
        for cls in CLASSES:
            found = sorted(raw_dir.glob(f"{cls}.*.jpg")) + sorted(raw_dir.glob(f"{cls}.*.jpeg")) + sorted(raw_dir.glob(f"{cls}.*.png"))
            files_by_class[cls] = found
            logger.info(f"Flat – {cls}: {len(found)} images")

    return files_by_class


def split_dataset(
    raw_dir: str,
    processed_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[int, int, int]:
    """
    Split a raw dataset into train/val/test ImageFolder sub-directories.

    Automatically detects the input layout:

    **Flat layout** (Kaggle default)::

        raw_dir/
            cat.0.jpg
            cat.1.jpg
            dog.0.jpg
            …

    **ImageFolder layout**::

        raw_dir/
            cat/
                image001.jpg
                …
            dog/
                image001.jpg
                …

    In both cases the output is always written as ImageFolder::

        processed_dir/
            train/cat/*.jpg   train/dog/*.jpg
            val/cat/*.jpg     val/dog/*.jpg
            test/cat/*.jpg    test/dog/*.jpg

    Args:
        raw_dir:       Source directory (flat or ImageFolder layout).
        processed_dir: Destination directory for the split data.
        train_ratio:   Fraction of data for training   (default 0.8).
        val_ratio:     Fraction of data for validation (default 0.1).
                       Test fraction = 1 - train_ratio - val_ratio.
        seed:          Random seed for reproducibility.

    Returns:
        Tuple of (n_train, n_val, n_test) total image counts.

    Raises:
        ValueError: If no images are found or ratios are invalid.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1.0, "
            f"got {train_ratio} + {val_ratio} = {train_ratio + val_ratio}"
        )

    np.random.seed(seed)
    raw_dir       = Path(raw_dir)
    processed_dir = Path(processed_dir)

    # ── Auto-detect input layout ──────────────────────────────────────────────
    layout = _detect_layout(raw_dir)
    logger.info(f"Detected layout: '{layout}' in '{raw_dir}'")

    # ── Collect per-class file lists ──────────────────────────────────────────
    files_by_class = _collect_files_by_class(raw_dir, layout)

    empty_classes = [cls for cls, files in files_by_class.items() if len(files) == 0]
    if empty_classes:
        raise ValueError(f"No images found for class(es): {empty_classes} in '{raw_dir}'")

    # ── Split each class independently (preserves class balance) ─────────────
    def _split(files):
        n       = len(files)
        idx     = np.random.permutation(n)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        return (
            [files[i] for i in idx[:n_train]],
            [files[i] for i in idx[n_train : n_train + n_val]],
            [files[i] for i in idx[n_train + n_val :]],
        )

    split_files: dict = {"train": [], "val": [], "test": []}
    for cls, files in files_by_class.items():
        tr, va, te = _split(files)
        split_files["train"].append((cls, tr))
        split_files["val"].append((cls, va))
        split_files["test"].append((cls, te))

    # ── Copy files into ImageFolder output structure ──────────────────────────
    for split_name, cls_file_pairs in split_files.items():
        for cls, files in cls_file_pairs:
            out_cls_dir = processed_dir / split_name / cls
            out_cls_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                dst = out_cls_dir / src.name
                shutil.copy2(src, dst)

    n_train = sum(len(f) for _, f in split_files["train"])
    n_val   = sum(len(f) for _, f in split_files["val"])
    n_test  = sum(len(f) for _, f in split_files["test"])

    logger.info(
        f"Split complete – train:{n_train}  val:{n_val}  test:{n_test}  "
        f"(total:{n_train + n_val + n_test})"
    )
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
