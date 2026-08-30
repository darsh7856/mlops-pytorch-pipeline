import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from model import CIFAR10CNN, get_model


def test_cifar_model_output_shape():
    model = CIFAR10CNN(num_classes=10)

    inputs = torch.randn(4, 3, 32, 32)
    outputs = model(inputs)

    assert outputs.shape == (4, 10)


def test_model_output_batch_size():
    model = CIFAR10CNN(num_classes=10)

    for batch_size in [1, 4, 8]:
        inputs = torch.randn(batch_size, 3, 32, 32)
        outputs = model(inputs)

        assert outputs.shape == (batch_size, 10)


def test_factory_creates_cifar_model():
    model = get_model("cifar_cnn", 10)

    assert isinstance(model, CIFAR10CNN)


def test_factory_rejects_unknown_architecture():
    with pytest.raises(ValueError):
        get_model("unknown", 10)


def test_model_forward_pass():
    model = CIFAR10CNN(num_classes=10)
    model.eval()

    inputs = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs is not None
    assert outputs.shape == (1, 10)


def test_model_has_trainable_parameters():
    model = CIFAR10CNN(num_classes=10)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(trainable_parameters) > 0


def test_model_parameters_are_finite():
    model = CIFAR10CNN(num_classes=10)

    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()


def test_model_supports_custom_num_classes():
    num_classes = 5
    model = CIFAR10CNN(num_classes=num_classes)

    inputs = torch.randn(2, 3, 32, 32)
    outputs = model(inputs)

    assert outputs.shape == (2, num_classes)


def test_model_train_and_eval_modes():
    model = CIFAR10CNN(num_classes=10)

    model.train()
    assert model.training is True

    model.eval()
    assert model.training is False