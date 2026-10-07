import numpy as np
import pytest

from msbgen.core import decode_top_planes, encode_top_planes


@pytest.mark.parametrize("shape", [(17, 23), (16, 20, 3)])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_top_plane_stream_roundtrip(shape, k):
    image = np.random.default_rng(4).integers(0, 256, shape, dtype=np.uint8)
    stream = encode_top_planes(image, k)
    decoded = decode_top_planes(stream, image.shape, k)
    expected = ((image.astype(np.uint32) >> (8 - k)) << (8 - k)).astype(np.uint8)
    assert np.array_equal(decoded, expected)


def test_stream_is_lossless_for_constant_plane():
    image = np.full((8, 9), 255, dtype=np.uint8)
    stream = encode_top_planes(image, 1)
    assert np.array_equal(decode_top_planes(stream, image.shape, 1), image & 128)


def test_invalid_k_is_rejected():
    image = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        encode_top_planes(image, 8)


def test_rejects_truncated_stream():
    image = np.random.default_rng(5).integers(0, 256, (8, 9), dtype=np.uint8)
    with pytest.raises(ValueError):
        decode_top_planes(encode_top_planes(image, 2)[:-1], image.shape, 2)