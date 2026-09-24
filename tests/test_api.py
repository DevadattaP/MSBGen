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
    import base64

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
