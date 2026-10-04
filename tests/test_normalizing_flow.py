import numpy as np
import pytest

torch = pytest.importorskip("torch")

from msbgen.core import keep_top_bits, msb_consistency, summary_method_kind  # noqa: E402
from msbgen.models import (  # noqa: E402
    MSBNormalizingFlow,
    NormalizingFlowConfig,
    load_reconstructor,
    save_checkpoint,
)


def _model() -> MSBNormalizingFlow:
    torch.manual_seed(0)
    return MSBNormalizingFlow(NormalizingFlowConfig(
        crop_size=16, scales=2, blocks_per_scale=1, condition_width=8, hidden_width=8,
    )).eval()


def test_glow_scales_are_exactly_invertible():
    model = _model()
    for scale, flow in enumerate(model.flows):
        channels = 2 ** (scale + 2)
        size = model.config.crop_size // (2 ** (scale + 1))
        value = torch.randn(2, channels, size, size)
        condition = torch.randn(2, model.config.condition_width, size, size)
        latent, logdet = flow(value, c=[condition])
        recovered, inverse_logdet = flow(latent, c=[condition], rev=True)
        torch.testing.assert_close(recovered, value, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(logdet + inverse_logdet, torch.zeros_like(logdet), atol=1e-5, rtol=1e-5)


def test_likelihood_and_seeded_samples_preserve_transmitted_bits():
    model = _model()
    images = torch.randint(0, 256, (2, 1, 32, 32), dtype=torch.uint8)
    top = model.keep_top(images)

    nll = model.loss(images, generator=torch.Generator().manual_seed(4), reduction="none")
    assert nll.shape == (2,) and torch.isfinite(nll).all()

    sample_a = model.sample(top, generator=torch.Generator().manual_seed(5))
    sample_b = model.sample(top, generator=torch.Generator().manual_seed(5))
    sample_c = model.sample(top, generator=torch.Generator().manual_seed(6))
    assert sample_a.shape == (2, 1, 32, 32)
    assert torch.equal(sample_a, sample_b)
    assert not torch.equal(sample_a, sample_c)
    rounded = torch.floor(sample_a * 255 + 0.5)
    assert torch.equal(model.keep_top(rounded), top)


def test_checkpoint_reconstructor_round_trip(tmp_path):
    model = _model()
    path = tmp_path / "normalizing-flow" / "best.pt"
    save_checkpoint(path, model, step=1)
    reconstructor = load_reconstructor(path, device="cpu")

    image = np.random.default_rng(2).integers(0, 256, (32, 32), dtype=np.uint8)
    top = keep_top_bits(image, 1)
    first = reconstructor.reconstruct(top, 1, 8)
    second = reconstructor.sample(top, 1, 8, seed=0)
    assert np.array_equal(first, second)
    assert msb_consistency(image, first, 1, 8) == 1.0
    assert reconstructor.sampler == {"patch_size": 16}
    assert summary_method_kind("normalizing-flow", {"model_kind": "normalizing_flow"}) == "generative"


def test_train_cli_builds_normalizing_flow_config(tmp_path, monkeypatch):
    from msbgen import training
    from msbgen.cli.train import main as train_main

    captured = {}

    def fake_train(config, resume=None):
        captured["config"] = config
        return tmp_path / "best.pt"

    monkeypatch.setattr(training, "train", fake_train)
    assert train_main([
        "--preset", "normalizing-flow", "--name", "nf", "--steps", "7",
        "--out-dir", str(tmp_path), "--device", "cpu", "--no-thermal-guard",
    ]) == 0
    config = captured["config"]
    assert config.model.kind == "normalizing_flow"
    assert config.model.k == 1 and config.model.crop_size == 128
    assert config.steps == 7 and config.batch_size == 16

    assert train_main([
        "--kind", "normalizing_flow", "--name", "nf-kind", "--steps", "3",
        "--out-dir", str(tmp_path), "--device", "cpu", "--no-thermal-guard",
    ]) == 0
    assert captured["config"].model.kind == "normalizing_flow"
