import numpy as np
import pytest

torch = pytest.importorskip("torch")

from msbgen.models.bitdepth_recovery import (  # noqa: E402
    BitplaneDataset,
    BitplaneResidualCNN,
    next_bit_target,
    quantize,
    recover_bitplanes,
)


def test_quantize_and_next_plane_match_integer_bits():
    image = torch.tensor([[[[0, 1, 127, 128, 255]]]], dtype=torch.uint8)
    quantized = quantize(image, 4, 8)
    assert quantized.mul(255).round().squeeze().tolist() == [0, 0, 112, 128, 240]
    assert next_bit_target(image, 4, 8).squeeze().tolist() == [0, 0, 1, 0, 1]


def test_model_and_dataset_shapes():
    image = np.arange(64 * 64 * 3, dtype=np.uint16).reshape(64, 64, 3) % 256
    dataset = BitplaneDataset([image], input_bits=4, patch_size=48)
    inputs, targets = dataset[0]
    model = BitplaneResidualCNN(channels=3, depth=1, filters=4)
    logits = model(inputs[None])
    assert inputs.shape == targets.shape == (3, 48, 48)
    assert logits.shape == (1, 3, 48, 48)
    assert set(targets.unique().tolist()) <= {0.0, 1.0}


def test_sequential_recovery_adds_one_plane_per_model():
    class ConstantPlane(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, x):
            return torch.full_like(x, self.value)

    image = torch.zeros(1, 1, 2, 2)
    result = recover_bitplanes(image, [ConstantPlane(0.0), ConstantPlane(20.0)], 6, 8)
    expected = (2 * torch.sigmoid(torch.tensor(0.0)) + torch.sigmoid(torch.tensor(20.0))) / 255
    assert torch.allclose(result, torch.full_like(result, expected))
