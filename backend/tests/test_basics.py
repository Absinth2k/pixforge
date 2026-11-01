"""Fast tests that need neither weights nor a GPU."""

from __future__ import annotations

import numpy as np
import pytest

from pixforge.denoise.estimate import estimate_noise
from pixforge.models.hat_mask import shifted_window_mask


def test_mask_shape_and_values() -> None:
    m = shifted_window_mask(64, 96)
    assert m.shape == ((64 // 16) * (96 // 16), 256, 256)
    assert set(np.unique(m)) <= {0.0, -100.0}
    # Interior windows are never split, so their masks are all zero.
    assert not m[0].any()


def test_mask_matches_torch_reference() -> None:
    torch = pytest.importorskip("torch")
    from pixforge.models.hat_arch import window_partition

    def reference(h: int, w: int, ws: int = 16) -> np.ndarray:  # upstream math
        ss = ws // 2
        img = torch.zeros((1, h, w, 1))
        cnt = 0
        for hs in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            for wsl in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                img[:, hs, wsl, :] = cnt
                cnt += 1
        mw = window_partition(img, ws).view(-1, ws * ws)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0).numpy()

    for h, w in ((64, 64), (128, 192)):
        np.testing.assert_array_equal(shifted_window_mask(h, w), reference(h, w))


def test_mask_rejects_unaligned() -> None:
    with pytest.raises(ValueError):
        shifted_window_mask(70, 64)


@pytest.mark.parametrize("sigma", [2.0, 5.0, 10.0])
def test_noise_estimate_tracks_gaussian_noise(sigma: float) -> None:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float32) / 512
    clean = np.stack([0.3 + 0.4 * xx, 0.5 * np.ones_like(xx), 0.6 - 0.3 * yy], -1)
    noisy = clean + rng.normal(0, sigma / 255, clean.shape).astype(np.float32)
    est = estimate_noise(noisy)
    # Luma noise of independent RGB noise is sigma * 0.668 (BT.601 weights).
    assert est.luma_sigma == pytest.approx(sigma * 0.668, rel=0.15)


def test_noise_estimate_clean_image_is_near_zero() -> None:
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32) / 256
    clean = np.stack([xx, yy, 0.5 * np.ones_like(xx)], -1)
    assert estimate_noise(clean).luma_sigma < 0.3
