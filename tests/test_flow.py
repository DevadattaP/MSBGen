import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from msbgen.core import keep_top_bits, msb_consistency, summary_method_kind  # noqa: E402
from msbgen.models import (  # noqa: E402
    FlowConfig,
    FlowReconstructor,
    MSBFlow,
    load_model,
    load_reconstructor,
    save_checkpoint,
)
from tests.test_diffusion import TINY, _images  # noqa: E402
from tests.test_models import _dataset  # noqa: E402


def _model(seed=0, **kw) -> MSBFlow:
    torch.manual_seed(seed)
    return MSBFlow(FlowConfig(arch_args=TINY, **kw)).eval()


# ---------- model ----------


def test_loss_is_repeatable_and_matches_the_velocity_target():
    model = _model()
    x = _images()
    a = model.loss(x, generator=torch.Generator().manual_seed(3), reduction="none")
    b = model.loss(x, generator=torch.Generator().manual_seed(3), reduction="none")
    assert a.shape == (2,) and torch.equal(a, b)

    # A network that outputs the exact target ε − r has zero loss.
    top = model.keep_top(x)
    r = model.to_residual(x, top)
    gen = torch.Generator().manual_seed(5)
    torch.rand(2, generator=gen)  # loss() draws t first, then ε
    eps = torch.randn(r.shape, generator=gen)
    model.predict = lambda x_t, t, cond: eps - r
    assert model.loss(x, generator=torch.Generator().manual_seed(5)).item() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_sampler_with_oracle_recovers_the_image(solver):
    # The path is a straight line, so with the true velocity any step count lands exactly on r.
    model = _model()
    x = _images()
    top = model.keep_top(x)
    r = model.to_residual(x, top)

    def oracle(x_t, t, cond):
        return (x_t - r) / t[:, None, None, None]  # = ε − r on the path through r

    for steps in (1, 3, 10):
        out = model.sample(top, steps=steps, solver=solver, generator=torch.Generator().manual_seed(0), predict=oracle)
        torch.testing.assert_close(out * 255, x.float(), atol=1e-3, rtol=0)


def test_samples_keep_transmitted_bits_and_depend_on_seed():
    model = _model()
    x = _images()
    top = model.keep_top(x)
    s0 = model.sample(top, steps=4, generator=torch.Generator().manual_seed(0))
    s0b = model.sample(top, steps=4, generator=torch.Generator().manual_seed(0))
    s1 = model.sample(top, steps=4, generator=torch.Generator().manual_seed(1))
    assert torch.equal(s0, s0b) and not torch.equal(s0, s1)
    for s in (s0, s1):
        assert msb_consistency(x.numpy(), s.numpy() * 255, 1, 8) == 1.0


def test_sampling_and_validation_call_the_thermal_hook():
    from torch.utils.data import DataLoader

    from msbgen.training import validate

    model = _model()
    calls = []
    top = model.keep_top(_images())
    model.sample(top, steps=5, on_step=lambda: calls.append(1))
    assert len(calls) == 5  # Euler: one network evaluation per step
    calls.clear()
    model.sample(top, steps=5, solver="heun", on_step=lambda: calls.append(1))
    assert len(calls) == 2 * 5 - 1  # Heun: two per step, plain Euler on the last
    calls.clear()
    loader = DataLoader(list(_images((4, 1, 32, 32))), batch_size=2)  # 2 batches
    metrics = validate(model, loader, torch.device("cpu"), sample_steps=3, on_step=lambda: calls.append(1))
    assert len(calls) == 2 * (1 + 3) and metrics["val_msb_consistency"] == 1.0


def test_config_validation():
    with pytest.raises(ValueError):
        FlowConfig(k=8)
    with pytest.raises(ValueError):
        FlowConfig(solver="rk4")
    with pytest.raises(ValueError):
        _model().sample(torch.zeros(1, 1, 8, 8), steps=1, solver="rk4")


# ---------- checkpoints / reconstructor ----------


def test_checkpoint_and_reconstructor(tmp_path):
    save_checkpoint(tmp_path / "fm" / "best.pt", _model(solver="heun"), step=3)
    loaded, payload = load_model(tmp_path / "fm" / "best.pt")
    assert payload["model_kind"] == "flow" and isinstance(loaded, MSBFlow) and loaded.config.solver == "heun"

    calls = []
    rec = load_reconstructor(tmp_path / "fm" / "best.pt", device="cpu", steps=3, eta=0.5,  # eta: diffusion only
                             on_step=lambda: calls.append(1))
    assert isinstance(rec, FlowReconstructor) and rec.stochastic
    assert rec.sampler == {"sample_steps": 3, "solver": "heun"} and rec.label == "Flow matching (residual)"
    img = np.random.default_rng(4).integers(0, 256, (40, 56), dtype=np.uint8)
    top = keep_top_bits(img, 1)
    a, b = rec.reconstruct(top, 1, 8), rec.sample(top, 1, 8, seed=0)
    assert a.shape == img.shape and np.array_equal(a, b)  # reconstruct = seed 0
    assert len(calls) == 2 * (2 * 3 - 1)
    assert not np.array_equal(a, rec.sample(top, 1, 8, seed=1))
    assert msb_consistency(img, a, 1, 8) == 1.0
    euler = load_reconstructor(tmp_path / "fm" / "best.pt", device="cpu", steps=3, solver="euler", tile=16, margin=4)
    assert euler.solver == "euler" and euler.reconstruct(top, 1, 8).shape == img.shape


# ---------- training, evaluation, API ----------


def test_train_flow_cli_and_evaluate(tmp_path, capsys):
    from msbgen.cli.evaluate import main as evaluate_main
    from msbgen.cli.train import main as train_main

    data = _dataset(tmp_path / "data")
    common = ["--preset", "flow", "--name", "fm", "--train", str(data), "--val", str(data),
              "--out-dir", str(tmp_path / "runs"), "--width", "8", "--levels", "2", "--blocks", "1",
              "--batch-size", "2", "--crop", "32", "--val-every", "2", "--log-every", "2",
              "--val-sample-steps", "2", "--workers", "0", "--device", "cpu", "--no-thermal-guard"]
    assert train_main([*common, "--steps", "4", "--solver", "heun"]) == 0
    run = tmp_path / "runs" / "fm"
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["model_kind"] == "flow" and cfg["ema"] == 0.999 and cfg["model"]["solver"] == "heun"
    log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    vals = [r for r in log if "val_psnr" in r]
    assert [r["step"] for r in vals] == [0, 2, 4] and all("best_val_loss" in r for r in vals)
    assert all(r["train_psnr"] is None for r in log if "loss" in r)

    with pytest.raises(SystemExit):  # the solver is a flow-matching option
        train_main(["--preset", "diffusion", "--solver", "heun"])

    out = tmp_path / "eval"
    assert evaluate_main([str(data), "--models", str(run / "best.pt"), "--samples", "2", "--sample-steps", "2",
                          "--solver", "euler", "--fills", "mid", "--out", str(out), "--save-images", "0",
                          "--device", "cpu", "--no-thermal-guard"]) == 0
    methods = json.loads((out / "summary.json").read_text())["methods"]
    fm = methods["fm"]
    assert fm["sample_steps"] == 2 and fm["solver"] == "euler" and "eta" not in fm
    assert fm["label"] == "Flow matching (residual)" and fm["diversity"]["mean"] > 0
    assert summary_method_kind("fm", fm) == "generative"


def test_api_lists_and_runs_flow(tmp_path, monkeypatch):
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from msbgen.api import model_registry
    from msbgen.main import app
    from tests.test_api import _use_models_dir

    checks = []
    monkeypatch.setattr(model_registry, "_thermal_check", lambda: checks.append(1))
    save_checkpoint(tmp_path / "fm" / "best.pt", MSBFlow(FlowConfig(arch_args=TINY, sample_steps=2)), step=5)
    _use_models_dir(monkeypatch, tmp_path)
    client = TestClient(app)
    models = client.get("/api/models").json()["models"]
    assert [(m["name"], m["kind"], m["arch"]) for m in models] == [("fm", "flow", "tunet")]

    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (20, 24), dtype=np.uint8)).save(buf, "PNG")
    res = client.post("/api/reconstruct", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert res.status_code == 200, res.text
    fm = res.json()["results"][-1]
    assert fm["name"] == "fm" and fm["label"] == "Flow matching (residual)" and fm["msb_consistency"] == 1.0
    assert len(checks) == 2
