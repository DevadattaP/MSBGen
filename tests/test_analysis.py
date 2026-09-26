import numpy as np
import pytest

from msbgen.core import (
    analyze_image,
    context_entropy,
    msb_reconstruction,
    plane_correlation,
    psnr,
    slice_bitplanes,
    spatial_correlation,
    ssim,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _smooth_image(h=64, w=80):
    y, x = np.mgrid[0:h, 0:w]
    return ((x * 3 + y * 2) % 256).astype(np.uint8)


# ---------- metrics ----------


def test_psnr_known_value():
    a = np.zeros((4, 4), np.uint8)
    b = np.full((4, 4), 255 / 10, np.float64)  # MSE = 25.5², PSNR = 20 dB
    assert psnr(a, b, 255) == pytest.approx(20.0)
    assert psnr(a, a, 255) == float("inf")


def test_ssim_matches_skimage(rng):
    skm = pytest.importorskip("skimage.metrics")
    img = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    noisy = np.clip(img + rng.normal(0, 20, img.shape), 0, 255)
    expected = skm.structural_similarity(
        img.astype(np.float64), noisy, data_range=255, channel_axis=-1,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )
    assert ssim(img, noisy, 255) == pytest.approx(expected, abs=1e-9)
    assert ssim(img, img, 255) == pytest.approx(1.0)


def test_ssim_none_for_tiny_image():
    assert ssim(np.zeros((5, 5)), np.zeros((5, 5)), 255) is None


# ---------- entropy / correlation ----------


def test_context_entropy_constant_and_random(rng):
    assert context_entropy(np.zeros((32, 32), np.uint8)) == 0.0
    noise = rng.integers(0, 2, (256, 256), dtype=np.uint8)
    assert context_entropy(noise) == pytest.approx(1.0, abs=0.01)


def test_context_entropy_detects_structure():
    stripes = np.tile(np.array([[0, 0, 1, 1]], np.uint8), (32, 8))  # 50% ones, i.i.d. H = 1
    assert context_entropy(stripes) < 0.1


def test_spatial_correlation():
    stripes = np.tile(np.array([[0, 1]], np.uint8), (16, 8))  # alternates horizontally
    # Horizontal neighbours always differ (-1), vertical always agree (+1).
    assert spatial_correlation(stripes) == pytest.approx(0.0)
    blocks = np.kron(np.eye(2, dtype=np.uint8), np.ones((16, 16), np.uint8))
    assert spatial_correlation(blocks) > 0.9
    assert spatial_correlation(np.ones((8, 8), np.uint8)) is None


def test_plane_correlation(rng):
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    img[..., 1] = img[..., 0]  # G = R → identical planes
    img[..., 2] = 255 - img[..., 0]  # B = ~R → inverted planes
    corr = plane_correlation(slice_bitplanes(img))
    assert corr.shape == (24, 24)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(np.diag(corr[:8, 8:16]), 1.0)
    assert np.allclose(np.diag(corr[:8, 16:]), -1.0)
    assert np.abs(corr[0, 1:8]).max() < 0.1  # random image: planes independent


def test_plane_correlation_constant_plane_is_nan():
    img = np.array([[0, 1], [2, 3]], np.uint8)  # only B0, B1 vary
    corr = plane_correlation(slice_bitplanes(img))
    assert np.isnan(corr[0, 0])  # B7 row (MSB first)
    assert corr[7, 7] == pytest.approx(1.0)


# ---------- k-MSB reconstruction ----------


def test_msb_reconstruction_fills():
    img = np.array([[0b10110110]], np.uint8)
    assert msb_reconstruction(img, 1, "zero")[0, 0] == 128
    assert msb_reconstruction(img, 1, "mid")[0, 0] == 128 + 63.5
    assert msb_reconstruction(img, 3, "zero")[0, 0] == 0b10100000
    assert msb_reconstruction(img, 8, "mid")[0, 0] == img[0, 0]
    with pytest.raises(ValueError):
        msb_reconstruction(img, 0)
    with pytest.raises(ValueError):
        msb_reconstruction(img, 1, "median")


def test_midpoint_psnr_matches_uniform_theory(rng):
    # For uniform pixels, the dropped bits are uniform: MSE = (4^(n-k) - 1) / 12.
    img = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    for k in (1, 4, 7):
        expected = 10 * np.log10(255**2 / ((4 ** (8 - k) - 1) / 12))
        assert psnr(img, msb_reconstruction(img, k, "mid"), 255) == pytest.approx(expected, abs=0.1)


def test_analyze_image_structure():
    img = np.stack([_smooth_image()] * 3, axis=-1)
    a = analyze_image(img)
    assert (a.bit_depth, a.channels) == (8, 3)
    assert [p.bit for p in a.planes[:8]] == list(range(7, -1, -1))
    assert [r.k for r in a.reconstructions] == list(range(1, 9))
    assert [r.bits_per_pixel for r in a.reconstructions] == [3 * k for k in range(1, 9)]
    # PSNR grows with k; the full set of planes is lossless.
    psnrs = [r.psnr["mid"] for r in a.reconstructions]
    assert all(x < y for x, y in zip(psnrs, psnrs[1:]))
    assert a.reconstructions[-1].psnr["zero"] == float("inf")
    assert a.reconstructions[-1].ssim["mid"] == pytest.approx(1.0)
    # A smooth image's MSB is far cheaper with context than i.i.d.
    msb = a.planes[0]
    assert msb.context_entropy < 0.2 < msb.entropy
    coded = [r.coded_bpp for r in a.reconstructions]
    assert all(x <= y for x, y in zip(coded, coded[1:]))


def test_analyze_16bit_grayscale(rng):
    img = rng.integers(0, 65536, (24, 24), dtype=np.uint16)
    a = analyze_image(img, with_ssim=False)
    assert (a.bit_depth, a.channels) == (16, 1)
    assert len(a.planes) == 16 and len(a.reconstructions) == 16
    assert a.correlation.shape == (16, 16)
    assert a.reconstructions[0].ssim["mid"] is None
