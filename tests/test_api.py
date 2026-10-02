import base64
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from msbgen.main import app

client = TestClient(app)


def _encode(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _post(data: bytes, name: str = "x.png"):
    return client.post("/api/bitplanes", files={"file": (name, data, "application/octet-stream")})


def _bits_of_constant_image(body, channel_index: int) -> list[int]:
    """For a constant-valued channel, each plane is all-0 or all-1: read the bits MSB→LSB."""
    return [int(p["ones_ratio"]) for p in body["planes"] if p["channel_index"] == channel_index]


def test_index_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "MSBGen" in res.text


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_grayscale_returns_8_planes():
    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    res = _post(_encode(Image.fromarray(arr, "L")))
    assert res.status_code == 200
    body = res.json()
    assert (body["color_mode"], body["bit_depth"]) == ("L", 8)
    assert [c["name"] for c in body["channels"]] == ["L"]
    assert [p["bit"] for p in body["planes"]] == list(range(7, -1, -1))
    # A 0..255 ramp has exactly half the pixels set in every plane.
    assert all(p["ones_ratio"] == pytest.approx(0.5) for p in body["planes"])


def test_rgb_returns_24_planes():
    arr = np.random.default_rng(1).integers(0, 256, (10, 12, 3), dtype=np.uint8)
    body = _post(_encode(Image.fromarray(arr, "RGB"))).json()
    assert [c["name"] for c in body["channels"]] == ["R", "G", "B"]
    assert len(body["planes"]) == 24
    assert (body["width"], body["height"]) == (12, 10)
    assert body["notes"] == []


def test_rgba_drops_alpha_keeps_rgb_values():
    img = Image.new("RGBA", (4, 4), (0b10110001, 20, 30, 128))
    body = _post(_encode(img)).json()
    assert body["color_mode"] == "RGB"
    assert [c["label"] for c in body["channels"]] == ["Red", "Green", "Blue"]
    assert len(body["planes"]) == 24
    assert _bits_of_constant_image(body, 0) == [1, 0, 1, 1, 0, 0, 0, 1]
    assert any("alpha" in n.lower() for n in body["notes"])


def test_gray_alpha_becomes_grayscale():
    img = Image.new("LA", (4, 4), (0b01100110, 255))
    body = _post(_encode(img)).json()
    assert body["color_mode"] == "L"
    assert len(body["planes"]) == 8
    assert _bits_of_constant_image(body, 0) == [0, 1, 1, 0, 0, 1, 1, 0]
    assert any("alpha" in n.lower() for n in body["notes"])


@pytest.mark.parametrize(
    ("mode", "value", "fmt"),
    [("CMYK", (0, 255, 255, 0), "JPEG"), ("CMYK", (0, 255, 255, 0), "TIFF"), ("LAB", (100, 128, 200), "TIFF")],
)
def test_other_colour_spaces_converted_to_rgb(mode, value, fmt):
    img = Image.new(mode, (8, 8), value)
    body = _post(_encode(img, fmt), f"x.{fmt.lower()}").json()
    assert body["color_mode"] == "RGB"
    assert len(body["planes"]) == 24
    assert any(f"Converted {mode} to RGB" in n for n in body["notes"])


def test_cmyk_pure_red_converts_exactly():
    img = Image.new("CMYK", (4, 4), (0, 255, 255, 0))  # 0% cyan, 100% magenta + yellow
    body = _post(_encode(img, "TIFF"), "x.tif").json()
    assert _bits_of_constant_image(body, 0) == [1] * 8  # R = 255
    assert _bits_of_constant_image(body, 1) == [0] * 8  # G = 0
    assert _bits_of_constant_image(body, 2) == [0] * 8  # B = 0


def test_16bit_grayscale_keeps_16_planes():
    arr = np.full((4, 4), 0xAB12, dtype=np.uint16)
    body = _post(_encode(Image.fromarray(arr))).json()
    assert (body["color_mode"], body["bit_depth"]) == ("L", 16)
    assert len(body["planes"]) == 16
    assert body["planes"][0]["weight"] == 32768
    assert _bits_of_constant_image(body, 0) == [int(b) for b in f"{0xAB12:016b}"]


def test_16bit_rgb_png_keeps_48_planes():
    rgb = np.empty((4, 5, 3), dtype=np.uint16)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 0xF00F, 0x0101, 0x8000
    ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    body = _post(png.tobytes()).json()
    assert (body["color_mode"], body["bit_depth"]) == ("RGB", 16)
    assert len(body["planes"]) == 48
    assert _bits_of_constant_image(body, 0) == [int(b) for b in f"{0xF00F:016b}"]
    assert _bits_of_constant_image(body, 2) == [int(b) for b in f"{0x8000:016b}"]


def test_16bit_rgba_png_drops_alpha_keeps_16_bits():
    bgra = np.zeros((3, 3, 4), dtype=np.uint16)
    bgra[..., 2], bgra[..., 3] = 0xC003, 0x1234  # R channel, alpha
    ok, png = cv2.imencode(".png", bgra)
    assert ok
    body = _post(png.tobytes()).json()
    assert (body["color_mode"], body["bit_depth"]) == ("RGB", 16)
    assert len(body["planes"]) == 48
    assert _bits_of_constant_image(body, 0) == [int(b) for b in f"{0xC003:016b}"]
    assert any("alpha" in n.lower() for n in body["notes"])


def _decode_data_url(url: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))


def test_rgb_combined_planes_colour_each_bit():
    # Pixel 0: R=0b10000000 G=0b10000001 B=0          → B7 = (1,1,0) yellow, B0 = (0,1,0) green
    # Pixel 1: R=0b00000001 G=0          B=0b10000001 → B7 = (0,0,1) blue,   B0 = (1,0,1) magenta
    arr = np.array([[[0x80, 0x81, 0x00], [0x01, 0x00, 0x81]]], dtype=np.uint8)
    body = _post(_encode(Image.fromarray(arr, "RGB"))).json()
    combined = body["combined"]
    assert [c["bit"] for c in combined] == list(range(7, -1, -1))
    b7 = _decode_data_url(combined[0]["image"])
    b0 = _decode_data_url(combined[-1]["image"])
    assert b7[0].tolist() == [[255, 255, 0], [0, 0, 255]]
    assert b0[0].tolist() == [[0, 255, 0], [255, 0, 255]]
    assert _decode_data_url(combined[3]["image"]).max() == 0  # bit 4 unset everywhere


def test_16bit_rgb_has_16_combined_planes():
    ok, png = cv2.imencode(".png", np.full((2, 2, 3), 0x8000, dtype=np.uint16))
    body = _post(png.tobytes()).json()
    assert [c["bit"] for c in body["combined"]] == list(range(15, -1, -1))
    assert _decode_data_url(body["combined"][0]["image"]).min() == 255  # B15 set in all channels


def test_grayscale_has_no_combined_planes():
    body = _post(_encode(Image.new("L", (4, 4), 7))).json()
    assert body["combined"] == []


def test_1bit_image_has_single_plane():
    img = Image.fromarray(np.eye(6, dtype=bool))
    body = _post(_encode(img)).json()
    assert (body["color_mode"], body["bit_depth"]) == ("L", 1)
    assert len(body["planes"]) == 1
    assert body["planes"][0]["ones_ratio"] == pytest.approx(1 / 6)


def test_palette_expanded_with_note():
    img = Image.new("RGB", (4, 4), (200, 10, 10)).convert("P")
    body = _post(_encode(img)).json()
    assert body["color_mode"] == "RGB"
    assert any("palette" in n.lower() for n in body["notes"])


def test_invalid_image_rejected():
    res = _post(b"not an image", "bad.png")
    assert res.status_code == 400


def _analyse(data: bytes, name: str = "x.png"):
    return client.post("/api/analysis", files={"file": (name, data, "application/octet-stream")})


def test_analysis_rgb():
    y, x = np.mgrid[0:32, 0:48]
    arr = np.stack([(x * 5) % 256, (y * 7) % 256, (x + y) % 256], axis=-1).astype(np.uint8)
    res = _analyse(_encode(Image.fromarray(arr, "RGB")))
    assert res.status_code == 200
    body = res.json()
    assert (body["color_mode"], body["bit_depth"]) == ("RGB", 8)
    assert len(body["planes"]) == 24
    assert body["correlation"]["labels"][:2] == ["R·B7", "R·B6"]
    assert len(body["correlation"]["matrix"]) == 24
    recs = body["reconstructions"]
    assert [r["k"] for r in recs] == list(range(1, 9))
    assert recs[0]["bits"] == [7] and recs[2]["bits"] == [7, 6, 5]
    assert recs[-1]["psnr"] == {"zero": None, "mid": None, "random": None}  # lossless → infinite → null
    assert recs[0]["psnr"]["mid"] > recs[0]["psnr"]["random"] > recs[0]["psnr"]["zero"]
    assert _decode_data_url(recs[-1]["image"]).tolist() == arr.tolist()


def test_analysis_msb_reconstruction_image_is_midpoint():
    arr = np.array([[0, 127, 128, 255]], dtype=np.uint8).repeat(4, axis=0)
    body = _analyse(_encode(Image.fromarray(arr, "L"))).json()
    assert body["correlation"]["labels"] == [f"B{k}" for k in range(7, -1, -1)]
    b7 = _decode_data_url(body["reconstructions"][0]["image"])[..., 0]
    assert b7[0].tolist() == [64, 64, 192, 192]  # 63.5 and 191.5 rounded half up
    assert body["reconstructions"][0]["ssim"]["mid"] is None  # 4×4 is below the SSIM window


def test_analysis_constant_image_has_null_correlations():
    body = _analyse(_encode(Image.new("L", (16, 16), 100))).json()
    assert all(p["spatial_correlation"] is None for p in body["planes"])
    assert body["correlation"]["matrix"][0][0] is None


def test_analysis_rejects_invalid_image():
    assert _analyse(b"nope").status_code == 400


def _use_models_dir(monkeypatch, path, **overrides):
    import dataclasses

    from msbgen.api import model_registry, routes
    from msbgen.config import settings

    # No LPIPS / DISTS unless a test asks: the real ones download network weights.
    overrides = {"perceptual": False, "eval_dir": path / "eval", **overrides}
    patched = dataclasses.replace(settings, models_dir=path, device="cpu", **overrides)
    monkeypatch.setattr(model_registry, "settings", patched)
    monkeypatch.setattr(routes, "settings", patched)


def test_reconstruct_perceptual_metrics(tmp_path, monkeypatch):
    from msbgen.api import model_registry

    _use_models_dir(monkeypatch, tmp_path, perceptual=True, perceptual_max_pixels=16 * 16)
    fake = {"lpips": lambda ref, est, rng: float(np.abs(ref - est).mean() / rng), "dists": lambda *_: 0.25}
    monkeypatch.setattr(model_registry, "perceptual_metrics", lambda: fake)
    body = _post_reconstruct(_encode(Image.new("L", (16, 16), 100)))
    assert body["perceptual"] == ["lpips", "dists"]
    mid = next(r for r in body["results"] if r["name"] == "mid")
    assert mid["dists"] == 0.25 and 0 < mid["lpips"] < 1
    body = _post_reconstruct(_encode(Image.new("L", (17, 16), 100)))  # over perceptual_max_pixels
    assert body["perceptual"] == [] and body["results"][0]["lpips"] is None
    assert any("LPIPS and DISTS skipped" in n for n in body["notes"])


def test_evaluations(tmp_path, monkeypatch):
    import json
    import os

    _use_models_dir(monkeypatch, tmp_path)
    assert client.get("/api/evaluations").json()["evaluations"] == []
    stat = {"mean": 18.5, "std": 1.2}
    for name, kid_patch, mtime in (("old", None, 1_000), ("new", 128, 2_000)):
        folder = tmp_path / "eval" / name
        folder.mkdir(parents=True)
        meta = {"data": "data/kodak", "images": 24, "k": 1, "bit_depth": 8, "samples": 8}
        if kid_patch:
            meta["kid_patch"] = kid_patch
        methods = {
            "mid": {"label": "Midpoint fill", "psnr": stat, "lpips": None},
            "diff": {"label": "Diffusion", "psnr": stat, "kid": {"mean": 0.07, "std": 0.002, "patches": 7488},
                     "checkpoint": "runs/diff/best.pt", "config": {"k": 1}, "sample_steps": 50},
            "unet": {"label": "U-Net", "psnr": stat, "checkpoint": "runs/unet/best.pt", "config": {"k": 1}},
        }
        (folder / "summary.json").write_text(json.dumps({"meta": meta, "methods": methods}))
        os.utime(folder / "summary.json", (mtime, mtime))
    (tmp_path / "eval" / "broken").mkdir()
    (tmp_path / "eval" / "broken" / "summary.json").write_text("{not json")

    evals = client.get("/api/evaluations").json()["evaluations"]
    assert [(e["name"], e["kid_patch"]) for e in evals] == [("new", 128), ("old", None)]  # newest first, broken skipped
    methods = {m["name"]: m for m in evals[0]["methods"]}
    assert [methods[n]["kind"] for n in ("mid", "diff", "unet")] == ["fill", "generative", "regression"]
    assert methods["diff"]["metrics"]["kid"] == {"mean": 0.07, "std": 0.002}
    assert methods["mid"]["metrics"]["psnr"] == stat and methods["mid"]["metrics"]["lpips"] is None


def test_models_and_reconstruct(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from msbgen.models import ModelConfig, MSBNet, save_checkpoint

    _use_models_dir(monkeypatch, tmp_path)
    assert client.get("/api/models").json()["models"] == []

    torch.manual_seed(0)
    model = MSBNet(ModelConfig(arch="unet", output="residual", arch_args={"width": 4, "levels": 2}))
    save_checkpoint(tmp_path / "tiny-unet" / "best.pt", model, step=5, metrics={"val_psnr": 21.0, "val_gain_db": 1.5})
    models = client.get("/api/models").json()["models"]
    assert [(m["name"], m["k"], m["step"], m["val_gain_db"]) for m in models] == [("tiny-unet", 1, 5, 1.5)]

    arr = np.random.default_rng(3).integers(0, 256, (24, 20, 3), dtype=np.uint8)
    body = _post_reconstruct(_encode(Image.fromarray(arr, "RGB")))
    assert any("grayscale" in n for n in body["notes"])
    assert [r["name"] for r in body["results"]] == ["zero", "mid", "random", "tiny-unet"]
    zero, mid, random, model = body["results"]
    assert mid["psnr_gain"] == 0 and mid["msb_consistency"] == 1.0
    assert random["kind"] == "fill" and random["psnr_gain"] < 0 and random["msb_consistency"] == 1.0
    assert model["kind"] == "model" and model["msb_consistency"] == 1.0
    assert zero["psnr_gain"] < 0
    assert _decode_data_url(body["original"]).shape == (24, 20, 3)


def _post_reconstruct(data: bytes, **form):
    res = client.post("/api/reconstruct", files={"file": ("x.png", data, "image/png")}, data=form)
    assert res.status_code == 200, res.text
    return res.json()


def test_reconstruct_errors(tmp_path, monkeypatch):
    _use_models_dir(monkeypatch, tmp_path)
    png16 = _encode(Image.fromarray(np.full((4, 4), 0xAB12, dtype=np.uint16)))
    assert client.post("/api/reconstruct", files={"file": ("x.png", png16)}).status_code == 400
    png8 = _encode(Image.new("L", (16, 16), 100))
    res = client.post("/api/reconstruct", files={"file": ("x.png", png8)}, data={"models": "nope"})
    assert res.status_code == 404
    assert [r["name"] for r in _post_reconstruct(png8)["results"]] == ["zero", "mid", "random"]
