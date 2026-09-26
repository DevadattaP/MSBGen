"""Decoding uploaded images into integer numpy arrays, and PNG encoding helpers.

Every image is normalised to one of two colour modes, keeping its bit depth:

* grayscale ("L"): single-channel sources (1-bit, 8-bit, 16-bit gray); gray+alpha
  drops its alpha.
* RGB ("RGB"): everything else — RGBA (alpha dropped), palette, CMYK, YCbCr, LAB,
  HSV (converted), and 16-bit colour.

A separate 8-bit preview is produced for display.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

EXIF_ORIENTATION = 0x0112
TIFF_BITS_PER_SAMPLE = 258
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

CHANNEL_NAMES: dict[str, list[str]] = {"L": ["L"], "RGB": ["R", "G", "B"]}
CHANNEL_LABELS: dict[str, list[str]] = {"L": ["Gray"], "RGB": ["Red", "Green", "Blue"]}
PILLOW_16BIT_GRAY_MODES = {"I;16", "I;16L", "I;16B", "I;16N"}
# Palette index i = 4R + 2G + B (each bit 0/1) → colour (255R, 255G, 255B).
RGB_BITS_PALETTE = [255 * ((i >> s) & 1) for i in range(8) for s in (2, 1, 0)]


class ImageDecodeError(ValueError):
    """Raised when uploaded bytes cannot be turned into an integer image."""


@dataclass
class LoadedImage:
    array: np.ndarray  # (H, W, 1) or (H, W, 3); uint8 for 1/8-bit, uint16 for 16-bit
    color_mode: str  # "L" or "RGB"
    bit_depth: int  # 1, 8 or 16 — number of bit-planes per channel
    preview: np.ndarray  # uint8 (H, W) or (H, W, 3) for on-screen display
    original_mode: str
    original_format: str | None
    notes: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]

    @property
    def channel_names(self) -> list[str]:
        return CHANNEL_NAMES[self.color_mode]

    @property
    def channel_labels(self) -> list[str]:
        return CHANNEL_LABELS[self.color_mode]


def load_image(data: bytes, max_pixels: int | None = None) -> LoadedImage:
    """Decode image bytes into grayscale or RGB, preserving the bit depth."""
    try:
        img = Image.open(io.BytesIO(data))
    except UnidentifiedImageError as exc:
        raise ImageDecodeError("could not decode image: not a recognised image format") from exc
    except OSError as exc:
        raise ImageDecodeError(f"could not decode image: {exc}") from exc

    if max_pixels is not None and img.width * img.height > max_pixels:
        raise ImageDecodeError(
            f"image is {img.width}x{img.height} ({img.width * img.height:,} px); "
            f"limit is {max_pixels:,} px"
        )

    original_mode, original_format = img.mode, img.format

    # Pillow has no 16-bit colour modes and would silently truncate to 8 bits.
    if expected_mode := _detect_16bit_colour(data, img):
        array, decoded_mode = _decode_16bit_colour(data, expected_mode)
        notes = []
        if decoded_mode in {"LA", "RGBA"}:
            array = array[..., :-1]
            notes.append("Dropped alpha channel.")
        color_mode = "L" if decoded_mode == "LA" else "RGB"
        return LoadedImage(
            array, color_mode, 16, to_preview(array, 16), original_mode, original_format, notes
        )

    try:
        img.load()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError(f"could not decode image: {exc}") from exc

    notes: list[str] = []
    if img.getexif().get(EXIF_ORIENTATION, 1) != 1:
        img = ImageOps.exif_transpose(img)
        notes.append("Applied EXIF orientation.")

    mode = img.mode
    if mode in PILLOW_16BIT_GRAY_MODES:
        array = np.asarray(img).astype(np.uint16)[..., np.newaxis]
        return LoadedImage(array, "L", 16, to_preview(array, 16), original_mode, original_format, notes)

    if mode == "I":
        wide = np.asarray(img)
        if wide.size and (wide.min() < 0 or wide.max() > 0xFFFF):
            raise ImageDecodeError("32-bit integer images outside the 16-bit range are not supported")
        array = wide.astype(np.uint16)[..., np.newaxis]
        notes.append("32-bit integer image with values in the 16-bit range; treated as 16-bit.")
        return LoadedImage(array, "L", 16, to_preview(array, 16), original_mode, original_format, notes)

    if mode == "F":
        raise ImageDecodeError("floating-point images (mode F) are not supported")

    if mode == "1":
        array = np.asarray(img).astype(np.uint8)[..., np.newaxis]
        return LoadedImage(array, "L", 1, to_preview(array, 1), original_mode, original_format, notes)

    if mode in {"L", "LA"}:
        if mode == "LA":
            notes.append("Dropped alpha channel.")
        target = "L"
    else:
        # Palette values are colour-table indices, not gray levels, so palettes go to RGB too.
        if "A" in mode or "transparency" in img.info:
            notes.append("Dropped alpha channel.")
        if mode in {"P", "PA"}:
            notes.append("Expanded palette image to RGB.")
        elif mode not in {"RGB", "RGBA"}:
            notes.append(f"Converted {mode} to RGB.")
        target = "RGB"

    try:
        converted = img.convert(target)
    except ValueError as exc:
        raise ImageDecodeError(f"unsupported image mode {mode}") from exc
    display = np.asarray(converted, dtype=np.uint8)
    array = display[..., np.newaxis] if target == "L" else display
    return LoadedImage(array, target, 8, display, original_mode, original_format, notes)


def _detect_16bit_colour(data: bytes, img: Image.Image) -> str | None:
    """Return the colour mode ("RGB", "RGBA", "LA") of a 16-bit multi-channel file, else None."""
    if data.startswith(PNG_SIGNATURE) and len(data) >= 26:
        bit_depth, colour_type = data[24], data[25]
        if bit_depth == 16:
            return {2: "RGB", 4: "LA", 6: "RGBA"}.get(colour_type)
        return None
    if img.format == "TIFF":
        bits = img.tag_v2.get(TIFF_BITS_PER_SAMPLE, (8,))
        bits = bits if isinstance(bits, tuple) else (bits,)
        if len(bits) > 1 and max(bits) == 16:
            return {2: "LA", 3: "RGB", 4: "RGBA"}.get(len(bits))
    return None


def _decode_16bit_colour(data: bytes, expected_mode: str) -> tuple[np.ndarray, str]:
    try:
        import cv2
    except ImportError as exc:
        raise ImageDecodeError("16-bit colour images require opencv-python to be installed") from exc

    array = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if array is None or array.dtype != np.uint16:
        raise ImageDecodeError("could not decode 16-bit colour image")
    if array.ndim == 2:
        array = array[..., np.newaxis]
    channels = array.shape[2]

    if expected_mode == "LA":
        # OpenCV may expand gray+alpha to BGRA (gray replicated) or keep 2 channels.
        if channels == 4:
            return array[..., [0, 3]], "LA"
        if channels == 2:
            return array, "LA"
    elif expected_mode == "RGB" and channels == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2RGB), "RGB"
    elif expected_mode == "RGBA" and channels == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA), "RGBA"
    raise ImageDecodeError(f"unexpected {channels}-channel decode for 16-bit {expected_mode} image")


def to_grayscale(array: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB → (H, W, 1) luma (ITU-R BT.601, as Pillow's "L"), same dtype and bit depth.

    Grayscale input (H, W, 1) is returned unchanged.
    """
    if array.shape[-1] == 1:
        return array
    luma = array.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    return np.rint(luma).astype(array.dtype)[..., np.newaxis]


def to_preview(array: np.ndarray, bit_depth: int) -> np.ndarray:
    """8-bit version of an (H, W, 1) or (H, W, 3) array for display only."""
    if bit_depth == 16:
        array = (array >> 8).astype(np.uint8)
    elif bit_depth == 1:
        array = array * np.uint8(255)
    return array[..., 0] if array.shape[-1] == 1 else array


def encode_rgb_bits_png(bits: np.ndarray) -> bytes:
    """Encode an (H, W, 3) {0,1} array as an 8-colour palette PNG (each channel 0 or 255)."""
    index = ((bits[..., 0] << 2) | (bits[..., 1] << 1) | bits[..., 2]).astype(np.uint8)
    img = Image.frombytes("P", (index.shape[1], index.shape[0]), np.ascontiguousarray(index).tobytes())
    img.putpalette(RGB_BITS_PALETTE)
    buf = io.BytesIO()
    img.save(buf, format="PNG", bits=3)
    return buf.getvalue()


def encode_png(array: np.ndarray, binary: bool = False) -> bytes:
    """Encode a uint8 (H, W) / (H, W, 2|3|4) array as PNG bytes.

    With `binary=True` the array is a {0,1} bit-plane and is stored as a 1-bit PNG,
    which keeps many planes of a large image small.
    """
    if binary:
        img = Image.fromarray(array.astype(bool))
    else:
        img = Image.fromarray(array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
