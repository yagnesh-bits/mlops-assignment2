"""
src/tests/test_model.py
Unit tests for model construction and inference utilities.
"""

import numpy as np
import pytest
import torch

from src.models.model import SimpleCNN, build_model, predict


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cnn_model():
    model = SimpleCNN(num_classes=2)
    model.eval()
    return model


@pytest.fixture
def dummy_tensor():
    """Single image batch of shape (1, 3, 224, 224)."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def batch_tensor():
    """Batch of 4 images."""
    return torch.randn(4, 3, 224, 224)


# ── Model Construction ────────────────────────────────────────────────────────

class TestBuildModel:
    def test_build_cnn(self):
        model = build_model("cnn")
        assert isinstance(model, SimpleCNN)

    def test_build_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_model("resnet999")

    def test_model_has_parameters(self):
        model = build_model("cnn")
        params = list(model.parameters())
        assert len(params) > 0

    def test_output_shape_single(self, cnn_model, dummy_tensor):
        with torch.no_grad():
            output = cnn_model(dummy_tensor)
        assert output.shape == (1, 2)

    def test_output_shape_batch(self, cnn_model, batch_tensor):
        with torch.no_grad():
            output = cnn_model(batch_tensor)
        assert output.shape == (4, 2)

    def test_output_dtype(self, cnn_model, dummy_tensor):
        with torch.no_grad():
            output = cnn_model(dummy_tensor)
        assert output.dtype == torch.float32


# ── Inference / predict() ─────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_dict(self, cnn_model, dummy_tensor):
        result = predict(cnn_model, dummy_tensor)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, cnn_model, dummy_tensor):
        result = predict(cnn_model, dummy_tensor)
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_predict_label_valid(self, cnn_model, dummy_tensor):
        result = predict(cnn_model, dummy_tensor)
        assert result["label"] in ("cat", "dog")

    def test_predict_confidence_in_range(self, cnn_model, dummy_tensor):
        result = predict(cnn_model, dummy_tensor)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(self, cnn_model, dummy_tensor):
        result = predict(cnn_model, dummy_tensor)
        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-5

    def test_predict_confidence_matches_max_prob(self, cnn_model, dummy_tensor):
        result = predict(cnn_model, dummy_tensor)
        max_prob = max(result["probabilities"].values())
        assert abs(result["confidence"] - max_prob) < 1e-6

    def test_predict_custom_class_names(self, cnn_model, dummy_tensor):
        result = predict(cnn_model, dummy_tensor, class_names=("kitty", "puppy"))
        assert result["label"] in ("kitty", "puppy")

    def test_predict_deterministic(self, cnn_model, dummy_tensor):
        r1 = predict(cnn_model, dummy_tensor)
        r2 = predict(cnn_model, dummy_tensor)
        assert r1["label"] == r2["label"]
        assert abs(r1["confidence"] - r2["confidence"]) < 1e-6


# ── SimpleCNN Architecture ────────────────────────────────────────────────────

class TestSimpleCNN:
    def test_gradient_flows(self):
        model = SimpleCNN()
        model.train()
        tensor = torch.randn(2, 3, 224, 224)
        output = model(tensor)
        loss = output.sum()
        loss.backward()
        # Check at least one parameter has a gradient
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_dropout_in_eval(self, cnn_model, dummy_tensor):
        """In eval mode, two identical forward passes should give same result."""
        with torch.no_grad():
            out1 = cnn_model(dummy_tensor)
            out2 = cnn_model(dummy_tensor)
        assert torch.allclose(out1, out2)
