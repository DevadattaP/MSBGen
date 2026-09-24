import numpy as np
import pytest

from msbgen.core import (
    binary_entropy,
    combine_channel_planes,
    reconstruct_from_planes,
    slice_bitplanes,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_grayscale_shape_and_values(rng):
    img = rng.integers(0, 256, size=(17, 23), dtype=np.uint8)
    planes = slice_bitplanes(img)
    assert planes.shape == (1, 8, 17, 23)
    assert set(np.unique(planes)) <= {0, 1}


def test_rgb_gives_24_planes(rng):
    img = rng.integers(0, 256, size=(9, 11, 3), dtype=np.uint8)
    planes = slice_bitplanes(img)
    assert planes.shape == (3, 8, 9, 11)


def test_known_pixel_values():
    img = np.array([[0, 1, 128, 255, 0b10100101]], dtype=np.uint8)
    planes = slice_bitplanes(img)[0]  # (8, 1, 5)
    assert planes[:, 0, 0].tolist() == [0] * 8
    assert planes[:, 0, 1].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert planes[7, 0, 2] == 1 and planes[:7, 0, 2].sum() == 0
    assert planes[:, 0, 3].tolist() == [1] * 8
    assert planes[:, 0, 4].tolist() == [1, 0, 1, 0, 0, 1, 0, 1]


def test_lossless_reconstruction(rng):
    img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    assert np.array_equal(reconstruct_from_planes(slice_bitplanes(img)), img)


def test_msb_only_reconstruction_is_128_b7(rng):
    img = rng.integers(0, 256, size=(8, 8), dtype=np.uint8)
    msb = reconstruct_from_planes(slice_bitplanes(img), bits=[7])[..., 0]
    assert np.array_equal(msb, (img >= 128).astype(np.uint8) * 128)


def test_rgba_and_cmyk_give_32_planes(rng):
    img = rng.integers(0, 256, size=(5, 6, 4), dtype=np.uint8)
    planes = slice_bitplanes(img)
    assert planes.shape == (4, 8, 5, 6)
    assert np.array_equal(reconstruct_from_planes(planes), img)


def test_16bit_lossless(rng):
    img = rng.integers(0, 65536, size=(7, 9, 3), dtype=np.uint16)
    planes = slice_bitplanes(img)
    assert planes.shape == (3, 16, 7, 9)
    out = reconstruct_from_planes(planes)
    assert out.dtype == np.uint16
    assert np.array_equal(out, img)


def test_1bit_from_bool():
    img = np.array([[True, False], [False, True]])
    planes = slice_bitplanes(img)
    assert planes.shape == (1, 1, 2, 2)
    assert planes[0, 0].tolist() == [[1, 0], [0, 1]]


def test_combine_channel_planes(rng):
    img = rng.integers(0, 256, size=(5, 6, 3), dtype=np.uint8)
    planes = slice_bitplanes(img)
    for k in range(8):
        combined = combine_channel_planes(planes, k)
        assert combined.shape == (5, 6, 3)
        assert np.array_equal(combined, (img >> k) & 1)


def test_rejects_unsupported_dtype():
    with pytest.raises(ValueError):
        slice_bitplanes(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        slice_bitplanes(np.zeros((4, 4), dtype=np.uint8), bit_depth=16)


def test_binary_entropy():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert binary_entropy(0.5) == pytest.approx(1.0)
