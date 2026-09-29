import numpy as np
import pytest

from msbgen.core import (
    FILL_RECONSTRUCTORS,
    Reconstructor,
    evaluate,
    keep_top_bits,
    msb_consistency,
    msb_reconstruction,
    reconstruct_and_evaluate,
    round_estimate,
)


@pytest.fixture
def img():
    return np.random.default_rng(0).integers(0, 256, (32, 40), dtype=np.uint8)


def test_keep_top_bits():
    x = np.array([[0b10110110, 0b01111111]], np.uint8)
    assert keep_top_bits(x, 1).tolist() == [[128, 0]]
    assert keep_top_bits(x, 3).tolist() == [[0b10100000, 0b01100000]]
    assert keep_top_bits(x, 8).tolist() == x.tolist()
    with pytest.raises(ValueError):
        keep_top_bits(x, 0)


def test_fills_match_msb_reconstruction(img):
    for k in (1, 3, 7):
        for name, rec in FILL_RECONSTRUCTORS.items():
            assert np.array_equal(rec.reconstruct(keep_top_bits(img, k), k, 8), msb_reconstruction(img, k, name))


def test_reconstructor_only_sees_top_bits(img):
    seen = {}

    class Spy(Reconstructor):
        name, label = "spy", "Spy"

        def reconstruct(self, top, k, bit_depth):
            seen["top"] = top.copy()
            return top.astype(np.float64)

    reconstruct_and_evaluate(Spy(), img, 2)
    assert np.all(seen["top"] % 64 == 0)  # the 6 lower bits are always zero


def test_random_fill_keeps_top_bits_and_is_reproducible(img):
    for k in (1, 3, 7, 8):
        est = msb_reconstruction(img, k, "random")
        assert np.array_equal(est, np.floor(est))  # integer levels, like a real image
        assert est.min() >= 0 and est.max() <= 255
        assert msb_consistency(img, est, k, 8) == 1.0
        assert np.array_equal(est, msb_reconstruction(img, k, "random"))  # fixed seed
    assert np.array_equal(msb_reconstruction(img, 8, "random"), img)  # nothing missing → lossless
    low = msb_reconstruction(img, 1, "random") - keep_top_bits(img, 1)
    assert len(np.unique(low)) > 100  # the missing bits really vary


def test_round_estimate_half_up_and_clipped():
    assert round_estimate(np.array([63.5, 191.5, -3.0, 300.0]), 8).tolist() == [64, 192, 0, 255]


def test_msb_consistency(img):
    assert msb_consistency(img, msb_reconstruction(img, 1, "mid"), 1, 8) == 1.0
    flipped = img.astype(np.float64)
    flipped[0, 0] = 255 - flipped[0, 0]  # changes the MSB of one pixel
    assert msb_consistency(img, flipped, 1, 8) == pytest.approx(1 - 1 / img.size)


def test_evaluate_fields(img):
    scores = evaluate(img, img.astype(float), 1, 8)
    assert scores["psnr"] == float("inf") and scores["mae"] == 0 and scores["msb_consistency"] == 1.0
    assert scores["ssim"] == pytest.approx(1.0)
    assert evaluate(img, img.astype(float), 1, 8, with_ssim=False)["ssim"] is None
