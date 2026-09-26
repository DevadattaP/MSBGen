import dataclasses
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from msbgen.core import keep_top_bits, msb_consistency  # noqa: E402
from msbgen.models import ModelConfig, ModelReconstructor, MSBNet, load_model, save_checkpoint  # noqa: E402
from msbgen.training import TrainConfig, train  # noqa: E402

TINY_UNET = {"width": 4, "levels": 2}
TINY_CNN = {"width": 8, "depth": 3}


def _dataset(folder, count=3, shape=(48, 64), seed=0):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    for i in range(count):
        img = ((x * (i + 2) + y * 3) % 256 + rng.integers(0, 8, shape)).clip(0, 255).astype(np.uint8)
        np.save(folder / f"img{i}.npy", img)
    return folder


# ---------- MSBNet ----------


def test_encode_input_planes_msb_first():
    model = MSBNet(ModelConfig(arch="cnn", k=2, arch_args=TINY_CNN))
    top = torch.tensor([[[[0b11000000, 0b10000000, 0b01000000, 0]]]], dtype=torch.float32)
    planes = model.encode_input(top) + 0.5  # back to {0, 1}
    assert planes.shape == (1, 2, 1, 4)
    assert planes[0, 0, 0].tolist() == [1, 1, 0, 0]  # B7
    assert planes[0, 1, 0].tolist() == [1, 0, 1, 0]  # B6


@pytest.mark.parametrize("arch,args", [("cnn", TINY_CNN), ("unet", TINY_UNET)])
def test_any_input_size(arch, args):
    model = MSBNet(ModelConfig(arch=arch, output="direct", arch_args=args))
    out = model(torch.zeros(2, 1, 37, 53))  # not a multiple of 2^levels
    assert out.shape == (2, 1, 37, 53)
    assert 0 <= out.min() and out.max() <= 1


def test_residual_mode_keeps_transmitted_bits():
    torch.manual_seed(0)
    model = MSBNet(ModelConfig(arch="unet", output="residual", k=2, arch_args=TINY_UNET)).eval()
    img = np.random.default_rng(1).integers(0, 256, (1, 1, 32, 32)).astype(np.uint8)
    top = keep_top_bits(img, 2)
    with torch.no_grad():
        est = model(torch.from_numpy(top).float()).numpy() * 255
    assert msb_consistency(img, est, 2, 8) == 1.0


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(arch="transformer")
    with pytest.raises(ValueError):
        ModelConfig(output="both")
    with pytest.raises(ValueError):
        ModelConfig(k=8)


# ---------- checkpoints / reconstructor ----------


def test_checkpoint_roundtrip(tmp_path):
    model = MSBNet(ModelConfig(arch="unet", arch_args=TINY_UNET)).eval()
    save_checkpoint(tmp_path / "run" / "best.pt", model, step=7, metrics={"val_psnr": 20.0})
    loaded, payload = load_model(tmp_path / "run" / "best.pt")
    assert payload["step"] == 7 and loaded.config == model.config
    x = torch.randint(0, 2, (1, 1, 16, 16)).float() * 128
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))


def test_tiled_inference_matches_full(tmp_path):
    # A 3-layer CNN sees ±3 px, so a margin of 4 makes tiling exact.
    model = MSBNet(ModelConfig(arch="cnn", output="direct", arch_args=TINY_CNN))
    save_checkpoint(tmp_path / "cnn" / "best.pt", model, step=1)
    full = ModelReconstructor(tmp_path / "cnn" / "best.pt", device="cpu", tile=1000)
    tiled = ModelReconstructor(tmp_path / "cnn" / "best.pt", device="cpu", tile=16, margin=4)
    top = keep_top_bits(np.random.default_rng(2).integers(0, 256, (40, 50), dtype=np.uint8), 1)
    np.testing.assert_allclose(tiled.reconstruct(top, 1, 8), full.reconstruct(top, 1, 8), atol=1e-4)
    assert full.name == "cnn" and full.supports(1, 8, 1) and not full.supports(2, 8, 1)


# ---------- training ----------


def test_train_and_resume(tmp_path):
    data = _dataset(tmp_path / "data")
    cfg = TrainConfig(
        name="tiny", model=ModelConfig(arch="cnn", output="residual", arch_args=TINY_CNN),
        train_dirs=[str(data)], val_dirs=[str(data)], out_dir=str(tmp_path / "runs"),
        steps=6, batch_size=2, crop=32, warmup=2, val_every=3, val_crop=32, log_every=3, workers=0, device="cpu",
    )
    best = train(cfg)
    run = tmp_path / "runs" / "tiny"
    assert best.exists() and (run / "last.pt").exists() and (run / "config.json").exists()
    log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    vals = [r for r in log if "val_psnr" in r]
    trains = [r for r in log if "loss" in r]
    assert [r["step"] for r in vals] == [0, 3, 6]  # step 0 = untrained model
    assert [r["step"] for r in trains] == [3, 6]
    for r in vals:
        assert {"epoch", "val_loss", "val_ssim", "val_ssim_mid", "val_gain_db"} <= set(r)
        assert r["val_msb_consistency"] == 1.0  # residual output
    assert trains[0]["train_psnr"] > 0 and 0 < trains[0]["epoch"] < trains[1]["epoch"]

    # Simulate a crash after the last checkpoint: history past step 6 must be dropped on resume.
    with (run / "log.jsonl").open("a") as f:
        f.write(json.dumps({"step": 7, "loss": 1.0}) + "\n")
    train(dataclasses.replace(cfg, steps=9), resume=run / "last.pt")
    log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    steps = [(r["step"], "val_psnr" in r) for r in log]
    assert len(steps) == len(set(steps)) and log[-1]["step"] == 9
    assert not any(r["step"] == 7 and r.get("loss") == 1.0 for r in log)

    from msbgen.cli.plot_runs import main as plot_main

    out = tmp_path / "plots"
    assert plot_main([str(run), "--out", str(out), "--x", "epoch"]) == 0
    for name in ["tiny_history.csv", "tiny_overview.png", "loss.png", "val_psnr.png", "val_gain.png", "val_ssim.png", "lr.png"]:
        assert (out / name).exists(), name
    rows = (out / "tiny_history.csv").read_text().splitlines()
    assert rows[0].startswith("step,epoch,loss") and len(rows) == 1 + 4  # one row per step: 0, 3, 6, 9


# ---------- CLIs ----------


def test_evaluate_cli(tmp_path, capsys):
    from msbgen.cli.evaluate import main

    data = _dataset(tmp_path / "test", count=2)
    model = MSBNet(ModelConfig(arch="unet", arch_args=TINY_UNET))
    save_checkpoint(tmp_path / "runs" / "u" / "best.pt", model, step=1)
    out = tmp_path / "out"
    assert main([str(data), "--models", str(tmp_path / "runs" / "u" / "best.pt"), "--out", str(out),
                 "--save-images", "1", "--device", "cpu"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert list(summary["methods"]) == ["zero", "mid", "u"]
    assert summary["methods"]["u"]["msb_consistency"]["mean"] == 1.0
    assert summary["methods"]["mid"]["lpips"] is None
    assert (out / "images" / "img0_compare.png").exists() and (out / "images" / "img0_u.png").exists()
    assert "mid" in capsys.readouterr().out


def test_prepare_archive_source(tmp_path):
    from msbgen.data.prepare import prepare
    from msbgen.data.sources import Download, Source

    src = tmp_path / "src" / "Pics"
    src.mkdir(parents=True)
    Image.new("RGB", (20, 10), (255, 0, 0)).save(src / "a.png")
    Image.new("L", (8, 8), 7).save(src / "b.png")
    archive = tmp_path / "pics.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for f in src.iterdir():
            zf.write(f, f"Pics/{f.name}")
    source = Source(name="pics", description="test", archive=True, extracted_dir="Pics",
                    downloads=(Download(archive.as_uri(), "pics.zip", archive.stat().st_size),))
    manifest = prepare(source, tmp_path / "data")
    assert manifest["count"] == 2
    red = np.load(tmp_path / "data" / "processed" / "pics" / "a.npy")
    assert red.shape == (10, 20) and red.dtype == np.uint8 and red[0, 0] == 76  # BT.601 luma of pure red
    assert prepare(source, tmp_path / "data")["created"] == manifest["created"]  # second run is a no-op


def test_sources_resolve():
    from msbgen.data.sources import resolve

    assert [s.name for s in resolve(["div2k", "kodak", "div2k-train"])] == ["div2k-train", "div2k-valid", "kodak"]
    with pytest.raises(KeyError):
        resolve(["imagenet"])
