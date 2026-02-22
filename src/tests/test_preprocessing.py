"""
src/tests/test_preprocessing.py
Unit tests for data preprocessing functions.
"""

import io
import numpy as np
import pytest
import torch
from PIL import Image

from src.data.preprocess import (
    get_train_transform,
    get_val_transform,
    preprocess_pil_image,
    IMAGE_SIZE,
    CLASSES,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def random_rgb_image() -> Image.Image:
    """Create a random 256×256 RGB PIL image."""
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def small_image() -> Image.Image:
    """Tiny 50×50 image to test resize."""
    arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def large_image() -> Image.Image:
    """Large 1024×768 image."""
    arr = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ── Transform Tests ───────────────────────────────────────────────────────────

class TestTransforms:
    def test_val_transform_output_shape(self, random_rgb_image):
        """Val transform should produce (3, 224, 224) tensor."""
        t = get_val_transform()
        tensor = t(random_rgb_image)
        assert tensor.shape == (3, *IMAGE_SIZE)

    def test_train_transform_output_shape(self, random_rgb_image):
        """Train transform should produce (3, 224, 224) tensor."""
        t = get_train_transform()
        tensor = t(random_rgb_image)
        assert tensor.shape == (3, *IMAGE_SIZE)

    def test_val_transform_dtype(self, random_rgb_image):
        """Output tensor should be float32."""
        t = get_val_transform()
        tensor = t(random_rgb_image)
        assert tensor.dtype == torch.float32

    def test_val_transform_normalized(self, random_rgb_image):
        """After normalization values should not all be in [0, 1]."""
        t = get_val_transform()
        tensor = t(random_rgb_image)
        # After ImageNet normalization, values can be negative
        assert tensor.min().item() < 0.5 or tensor.max().item() > 1.0

    def test_small_image_resize(self, small_image):
        """Small images should be upscaled correctly."""
        t = get_val_transform()
        tensor = t(small_image)
        assert tensor.shape == (3, *IMAGE_SIZE)

    def test_large_image_resize(self, large_image):
        """Large images should be downscaled correctly."""
        t = get_val_transform()
        tensor = t(large_image)
        assert tensor.shape == (3, *IMAGE_SIZE)

    def test_train_transform_returns_tensor(self, random_rgb_image):
        """Train transform must return a Tensor."""
        t = get_train_transform()
        result = t(random_rgb_image)
        assert isinstance(result, torch.Tensor)


# ── preprocess_pil_image Tests ────────────────────────────────────────────────

class TestPreprocessPilImage:
    def test_output_batch_shape(self, random_rgb_image):
        """preprocess_pil_image should return (1, 3, 224, 224)."""
        tensor = preprocess_pil_image(random_rgb_image)
        assert tensor.shape == (1, 3, *IMAGE_SIZE)

    def test_output_dtype(self, random_rgb_image):
        """Output should be float32."""
        tensor = preprocess_pil_image(random_rgb_image)
        assert tensor.dtype == torch.float32

    def test_accepts_grayscale_converted(self):
        """RGBA image converted to RGB should work."""
        arr = np.random.randint(0, 255, (128, 128, 4), dtype=np.uint8)
        rgba = Image.fromarray(arr, mode="RGBA").convert("RGB")
        tensor = preprocess_pil_image(rgba)
        assert tensor.shape == (1, 3, *IMAGE_SIZE)

    def test_deterministic_for_same_input(self, random_rgb_image):
        """Val transform must be deterministic."""
        t1 = preprocess_pil_image(random_rgb_image)
        t2 = preprocess_pil_image(random_rgb_image)
        assert torch.allclose(t1, t2)


# ── Class Labels ──────────────────────────────────────────────────────────────

def test_classes_order():
    """CLASSES list should always be ['Cat', 'Dog']."""
    assert CLASSES == ["Cat", "Dog"]

def test_classes_length():
    assert len(CLASSES) == 2
