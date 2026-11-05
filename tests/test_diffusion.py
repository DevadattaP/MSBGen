import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from msbgen.core import FILL_RECONSTRUCTORS, keep_top_bits, msb_consistency  # noqa: E402
from msbgen.models import (  # noqa: E402
    DiffusionConfig,
    DiffusionReconstructor,
    MSBDiffusion,
    ModelReconstructor,
    load_model,
    load_reconstructor,
    save_checkpoint,
)
from msbgen.models.diffusion import cosine_schedule  # noqa: E402
from tests.test_models import TINY_CNN, _dataset  # noqa: E402

TINY = {"width": 8, "levels": 2, "blocks": 1}


def _model(seed=0, **kw) -> MSBDiffusion:
    torch.manual_seed(seed)
    return MSBDiffusion(DiffusionConfig(arch_args=TINY, **kw)).eval()


def _images(shape=(2, 1, 24, 40), seed=0) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(seed).integers(0, 256, shape).astype(np.uint8))


# ---------- model ----------


def test_cosine_schedule_endpoints():
    alpha, sigma = cosine_schedule(torch.linspace(0, 1, 11))
    assert alpha[0] == pytest.approx(1.0) and alpha[-1] == pytest.approx(0.0, abs=1e-6)
    torch.testing.assert_close(alpha**2 + sigma**2, torch.ones(11))
    assert torch.all(alpha[1:] < alpha[:-1])  # noise increases with t


def test_residual_round_trip_and_any_size():
    model = _model()
    x = _images()
    top = model.keep_top(x)
    r = model.to_residual(x, top)
    assert r.min() >= -1 and r.max() <= 1
    torch.testing.assert_close(model.from_residual(r, top) * 255, x.float())
    v = model.predict(r, torch.rand(2), model.condition(top))  # 24×40 is not a multiple of 4
    assert v.shape == r.shape


def test_loss_is_repeatable_with_a_generator():
    model = _model()
    x = _images()
    a = model.loss(x, generator=torch.Generator().manual_seed(3), reduction="none")
    b = model.loss(x, generator=torch.Generator().manual_seed(3), reduction="none")
    assert a.shape == (2,) and torch.equal(a, b)


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_sampler_with_oracle_recovers_the_image(eta):
    # If the network predicted the true v, DDIM must land exactly on the original residual.
    model = _model()
    x = _images()
    top = model.keep_top(x)
    r = model.to_residual(x, top)

    def oracle(x_t, t, cond):
        alpha, sigma = (v[:, None, None, None] for v in cosine_schedule(t))
        eps = (x_t - alpha * r) / sigma
        return alpha * eps - sigma * r

    out = model.sample(top, steps=10, eta=eta, generator=torch.Generator().manual_seed(0), predict=oracle)
    torch.testing.assert_close(out * 255, x.float(), atol=1e-3, rtol=0)


def test_samples_keep_transmitted_bits_and_depend_on_seed():
    model = _model()
    x = _images()
    top = model.keep_top(x)
    s0 = model.sample(top, steps=4, generator=torch.Generator().manual_seed(0))
    s0b = model.sample(top, steps=4, generator=torch.Generator().manual_seed(0))
    s1 = model.sample(top, steps=4, eta=1.0, generator=torch.Generator().manual_seed(1))
    assert torch.equal(s0, s0b) and not torch.equal(s0, s1)
    for s in (s0, s1):
        assert msb_consistency(x.numpy(), s.numpy() * 255, 1, 8) == 1.0


def test_sampling_and_validation_call_the_thermal_hook():
    # Diffusion sampling is sustained GPU load: the thermal guard must run inside it.
    from torch.utils.data import DataLoader

    from msbgen.training import validate

    model = _model()
    calls = []
    model.sample(model.keep_top(_images()), steps=5, on_step=lambda: calls.append(1))
    assert len(calls) == 5
    calls.clear()
    loader = DataLoader(list(_images((4, 1, 32, 32))), batch_size=2)  # 2 batches
    validate(model, loader, torch.device("cpu"), sample_steps=3, on_step=lambda: calls.append(1))
    assert len(calls) == 2 * (1 + 3)  # per batch + per denoising step


def test_config_validation():
    with pytest.raises(ValueError):
        DiffusionConfig(k=8)
    with pytest.raises(ValueError):
        DiffusionConfig(eta=2.0)


# ---------- checkpoints / reconstructor ----------


def test_checkpoint_and_reconstructor(tmp_path):
    model = _model()
    save_checkpoint(tmp_path / "diff" / "best.pt", model, step=3)
    loaded, payload = load_model(tmp_path / "diff" / "best.pt")
    assert payload["model_kind"] == "diffusion" and isinstance(loaded, MSBDiffusion)

    calls = []
    rec = load_reconstructor(tmp_path / "diff" / "best.pt", device="cpu", steps=3, on_step=lambda: calls.append(1))
    assert isinstance(rec, DiffusionReconstructor) and rec.stochastic and rec.steps == 3
    assert rec.name == "diff" and rec.supports(1, 8, 1) and not rec.supports(2, 8, 1)
    img = np.random.default_rng(4).integers(0, 256, (40, 56), dtype=np.uint8)
    top = keep_top_bits(img, 1)
    a, b = rec.reconstruct(top, 1, 8), rec.sample(top, 1, 8, seed=0)
    assert a.shape == img.shape and np.array_equal(a, b)  # reconstruct = seed 0
    assert len(calls) == 2 * 3  # the hook ran at every step of both samples
    assert not np.array_equal(a, rec.sample(top, 1, 8, seed=1))
    assert msb_consistency(img, a, 1, 8) == 1.0
    tiled = DiffusionReconstructor(tmp_path / "diff" / "best.pt", device="cpu", steps=3, tile=16, margin=4)
    assert tiled.reconstruct(top, 1, 8).shape == img.shape


def test_load_reconstructor_regression(tmp_path):
    from msbgen.models import ModelConfig, MSBNet

    save_checkpoint(tmp_path / "cnn" / "best.pt", MSBNet(ModelConfig(arch="cnn", arch_args=TINY_CNN)), step=1)
    rec = load_reconstructor(tmp_path / "cnn" / "best.pt", device="cpu", steps=5, eta=0.5)  # ignored
    assert isinstance(rec, ModelReconstructor) and not rec.stochastic


def test_random_fill_samples():
    rf = FILL_RECONSTRUCTORS["random"]
    top = keep_top_bits(np.random.default_rng(0).integers(0, 256, (16, 16), dtype=np.uint8), 1)
    assert rf.stochastic and not FILL_RECONSTRUCTORS["mid"].stochastic
    assert np.array_equal(rf.reconstruct(top, 1, 8), rf.sample(top, 1, 8, seed=0))
    assert not np.array_equal(rf.sample(top, 1, 8, seed=0), rf.sample(top, 1, 8, seed=1))
    mid = FILL_RECONSTRUCTORS["mid"]
    assert np.array_equal(mid.sample(top, 1, 8, seed=5), mid.reconstruct(top, 1, 8))


# ---------- training, evaluation, API ----------


def test_train_diffusion_cli_resume_and_evaluate(tmp_path, capsys):
    from msbgen.cli.evaluate import main as evaluate_main
    from msbgen.cli.train import main as train_main

    data = _dataset(tmp_path / "data")
    common = ["--preset", "diffusion", "--name", "diff", "--train", str(data), "--val", str(data),
              "--out-dir", str(tmp_path / "runs"), "--width", "8", "--levels", "2", "--blocks", "1",
              "--batch-size", "2", "--crop", "32", "--val-every", "2", "--log-every", "2",
              "--val-sample-steps", "2", "--workers", "0", "--device", "cpu", "--no-thermal-guard"]
    assert train_main([*common, "--steps", "4"]) == 0
    run = tmp_path / "runs" / "diff"
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["model_kind"] == "diffusion" and cfg["ema"] == 0.999 and cfg["model"]["arch_args"] == TINY
    log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    vals = [r for r in log if "val_psnr" in r]
    assert [r["step"] for r in vals] == [0, 2, 4]
    for r in vals:
        assert r["val_msb_consistency"] == 1.0 and "best_val_loss" in r and "val_psnr_random" in r
    assert all(r["train_psnr"] is None for r in log if "loss" in r)
    ckpt = torch.load(run / "last.pt", weights_only=True)
    assert "train_state_dict" in ckpt  # raw weights for resume; state_dict holds the EMA weights

    assert train_main([*common, "--steps", "6", "--resume", str(run / "last.pt")]) == 0
    log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    assert log[-1]["step"] == 6

    out = tmp_path / "eval"
    capsys.readouterr()
    assert evaluate_main([str(data), "--models", str(run / "best.pt"), "--samples", "2", "--sample-steps", "2",
                          "--out", str(out), "--save-images", "1", "--device", "cpu", "--no-thermal-guard"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    methods = summary["methods"]
    assert list(methods) == ["zero", "mid", "random", "diff"]
    assert methods["diff"]["sample_steps"] == 2 and methods["diff"]["msb_consistency"]["mean"] == 1.0
    for name in ("diff", "random"):
        assert methods[name]["psnr_avg"] is not None and methods[name]["diversity"]["mean"] > 0
    assert methods["mid"]["psnr_avg"] is None and methods["mid"]["time_s"] is not None
    assert (out / "images" / "img0_diff_samples.png").exists() and (out / "images" / "img0_random_samples.png").exists()
    assert "PSNR avg" in capsys.readouterr().out


def test_api_lists_and_runs_diffusion(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from PIL import Image
    import io

    from msbgen.api import model_registry
    from msbgen.main import app
    from tests.test_api import _use_models_dir

    checks = []
    monkeypatch.setattr(model_registry, "thermal_check", lambda force=False: checks.append(force))  # no real sensors

    model = MSBDiffusion(DiffusionConfig(arch_args=TINY, sample_steps=2))
    save_checkpoint(tmp_path / "diff" / "best.pt", model, step=5, metrics={"val_psnr": 14.0, "val_gain_db": -3.0})
    _use_models_dir(monkeypatch, tmp_path)
    client = TestClient(app)
    models = client.get("/api/models").json()["models"]
    assert [(m["name"], m["kind"], m["arch"], m["output"]) for m in models] == [("diff", "diffusion", "tunet", "residual")]

    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (20, 24), dtype=np.uint8)).save(buf, "PNG")
    res = client.post("/api/reconstruct", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert res.status_code == 200, res.text
    diff = res.json()["results"][-1]
    assert diff["name"] == "diff" and diff["kind"] == "model" and diff["msb_consistency"] == 1.0
    assert checks.count(False) == 2  # after each of the 2 sampling steps
    assert checks.count(True) == 1  # forced check before the model starts
