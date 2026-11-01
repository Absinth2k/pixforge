"""Blind noise estimation for photos.

Method: one level of a Haar wavelet transform; the diagonal detail band (HH) of a
natural image is dominated by noise, and for Gaussian noise of std sigma its robust
spread median(|HH|) / 0.6745 equals sigma. Texture and edges also leak into HH, so the
estimate is taken only over the flattest blocks (lowest gradient energy). Luma and
chroma are measured separately because camera/phone noise differs a lot between them,
and because images that were already denoised (e.g. by DxO PureRAW or a phone ISP)
typically show almost no chroma noise.

All sigmas are reported in 8-bit units (0..255) for readability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MAD_TO_SIGMA = 1.0 / 0.6745


@dataclass(frozen=True)
class NoiseEstimate:
    luma_sigma: float  # 8-bit units
    chroma_sigma: float  # 8-bit units, mean of Cb and Cr
    flat_fraction: float  # share of blocks used

    @property
    def level(self) -> float:
        """Single 0..100 score for the UI (rough; ~sigma 12 luma maps to 100)."""
        combined = self.luma_sigma + 0.5 * self.chroma_sigma
        return float(np.clip(combined / 12.0 * 100.0, 0.0, 100.0))


def _rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    m = np.array(
        [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]],
        dtype=np.float32,
    )
    return rgb @ m.T


def _haar_hh(ch: np.ndarray) -> np.ndarray:
    h, w = ch.shape
    c = ch[: h - h % 2, : w - w % 2]
    a, b = c[0::2, 0::2], c[0::2, 1::2]
    cc, d = c[1::2, 0::2], c[1::2, 1::2]
    return (a - b - cc + d) * 0.5


def _block_sigmas(hh: np.ndarray, block: int) -> np.ndarray:
    h, w = hh.shape
    nh, nw = h // block, w // block
    blocks = np.abs(hh[: nh * block, : nw * block]).reshape(nh, block, nw, block)
    return np.median(blocks.transpose(0, 2, 1, 3).reshape(nh * nw, -1), axis=1) * _MAD_TO_SIGMA


def _block_activity(y: np.ndarray, block: int) -> np.ndarray:
    """Gradient energy per block on the half-resolution grid used by HH."""
    h, w = y.shape
    lo = y[: h - h % 2, : w - w % 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    gy, gx = np.gradient(lo)
    g = np.hypot(gx, gy)
    hh, ww = g.shape
    nh, nw = hh // block, ww // block
    return g[: nh * block, : nw * block].reshape(nh, block, nw, block).mean(axis=(1, 3)).ravel()


def estimate_noise(
    rgb: np.ndarray, block: int = 16, flat_quantile: float = 0.3, max_side: int = 2048
) -> NoiseEstimate:
    """Estimate noise of an RGB float image in 0..1, shape (h, w, 3).

    Large images are measured on a centered crop (not a resize, which would hide noise).
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("expected (h, w, 3) RGB")
    h, w = rgb.shape[:2]
    if max(h, w) > max_side:
        y0, x0 = max(0, (h - max_side) // 2), max(0, (w - max_side) // 2)
        rgb = rgb[y0 : y0 + max_side, x0 : x0 + max_side]
    ycc = _rgb_to_ycbcr(rgb[..., :3].astype(np.float32)) * 255.0

    act = _block_activity(ycc[..., 0], block)
    if act.size == 0:
        return NoiseEstimate(0.0, 0.0, 0.0)
    # Exclude clipped (pure black / white) blocks: they carry no noise.
    luma_lo = _block_mean(ycc[..., 0], block)
    valid = (luma_lo > 8) & (luma_lo < 247)
    if valid.sum() < 8:
        valid = np.ones_like(valid)
    thresh = np.quantile(act[valid], flat_quantile)
    sel = valid & (act <= thresh)

    sig = [_block_sigmas(_haar_hh(ycc[..., c]), block)[: act.size] for c in range(3)]
    luma = float(np.median(sig[0][sel]))
    chroma = float(np.median(np.concatenate([sig[1][sel], sig[2][sel]])))
    return NoiseEstimate(luma, chroma, float(sel.mean()))


def _block_mean(y: np.ndarray, block: int) -> np.ndarray:
    h, w = y.shape
    lo = y[: h - h % 2, : w - w % 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    hh, ww = lo.shape
    nh, nw = hh // block, ww // block
    return lo[: nh * block, : nw * block].reshape(nh, block, nw, block).mean(axis=(1, 3)).ravel()
