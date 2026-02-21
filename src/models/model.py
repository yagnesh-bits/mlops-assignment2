"""
src/models/model.py
CNN model definitions for Cats vs Dogs binary classification.
Includes a simple baseline CNN and a transfer-learning variant (MobileNetV2).
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


# ── Baseline CNN ─────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    Lightweight baseline CNN for binary image classification.
    Input: (B, 3, 224, 224)
    Output: (B, 2)  – raw logits
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # → 112×112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # → 56×56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # → 28×28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # → 4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Transfer Learning Model (MobileNetV2) ────────────────────────────────────

class TransferModel(nn.Module):
    """
    Fine-tuned MobileNetV2 for binary classification.
    Faster convergence and higher accuracy than baseline CNN.
    """

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = False):
        super().__init__()
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        if freeze_backbone:
            for param in backbone.features.parameters():
                param.requires_grad = False

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ── Factory ──────────────────────────────────────────────────────────────────

def build_model(
    model_type: str = "cnn",
    num_classes: int = 2,
    **kwargs,
) -> nn.Module:
    """
    Factory function.

    Args:
        model_type: "cnn" | "mobilenet"
        num_classes: number of output classes (2 for binary)

    Returns:
        Instantiated nn.Module
    """
    if model_type == "cnn":
        return SimpleCNN(num_classes=num_classes, **kwargs)
    elif model_type == "mobilenet":
        return TransferModel(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Choose 'cnn' or 'mobilenet'.")


# ── Inference Utilities ───────────────────────────────────────────────────────

def load_model(checkpoint_path: str, model_type: str = "cnn", device: str = "cpu") -> nn.Module:
    """Load a saved model checkpoint."""
    model = build_model(model_type)
    state = torch.load(checkpoint_path, map_location=device)
    # Support both raw state-dict and wrapped checkpoint
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict(
    model: nn.Module,
    tensor: torch.Tensor,
    device: str = "cpu",
    class_names: Tuple[str, ...] = ("cat", "dog"),
) -> dict:
    """
    Run inference on a pre-processed tensor.

    Args:
        model:       Trained model in eval mode.
        tensor:      Shape (1, 3, 224, 224).
        device:      "cpu" or "cuda".
        class_names: Ordered class labels.

    Returns:
        dict with keys: label, confidence, probabilities
    """
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx = int(probs.argmax())
    return {
        "label": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {name: float(p) for name, p in zip(class_names, probs)},
    }
