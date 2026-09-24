"""Framework-independent image / bit-plane logic (no web dependencies)."""

from msbgen.core.bitplanes import (
    binary_entropy,
    combine_channel_planes,
    infer_bit_depth,
    reconstruct_from_planes,
    slice_bitplanes,
)
from msbgen.core.image_io import (
    ImageDecodeError,
    LoadedImage,
    encode_png,
    encode_rgb_bits_png,
    load_image,
)

__all__ = [
    "ImageDecodeError",
    "LoadedImage",
    "binary_entropy",
    "combine_channel_planes",
    "encode_png",
    "encode_rgb_bits_png",
    "infer_bit_depth",
    "load_image",
    "reconstruct_from_planes",
    "slice_bitplanes",
]
