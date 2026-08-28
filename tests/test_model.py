import sys
from pathlib import Path

import pytest
import torch

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from model import CIFAR10CNN, get_model


def test_cifar_model_output_shape():
    """Test that the CIFAR-10 CNN returns 10 class predictions."""
    model = CIFAR10CNN(num_classes=10)

    # CIFAR-10 input: batch_size x channels x height x width
    inputs = torch.randn(4, 3, 32, 32)

    outputs = model(inputs)

    assert outputs.shape == (4, 10)


def test_model_output_batch_size():
    """Test that the model preserves different batch sizes."""
    model = CIFAR10CNN(num_classes=10)

    for batch_size in [1, 4, 8]:
        inputs = torch.randn(batch_size, 3, 32, 32)
        outputs = model(inputs)

        assert outputs.shape == (batch_size, 10)


def test_factory_creates_cifar_model():
    """Test that the model factory creates the correct architecture."""
    model = get_model("cifar_cnn", 10)

    assert isinstance(model, CIFAR10CNN)


def test_factory_rejects_unknown_architecture():
    """Test that an unsupported architecture raises ValueError."""
    with pytest.raises(ValueError):
        get_model("unknown", 10)


def test_model_forward_pass():
    """Test that the model can successfully perform inference."""
    model = CIFAR10CNN(num_classes=10)

    inputs = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs is not None
    assert outputs.shape == (1, 10)


def test_model_has_trainable_parameters():
    """Test that the model contains trainable parameters."""
    model = CIFAR10CNN(num_classes=10)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(trainable_parameters) > 0