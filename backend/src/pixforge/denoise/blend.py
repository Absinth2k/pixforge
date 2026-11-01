"""Blend a denoised image with the original.

The strength slider is a mix between original and network output (not a network input), so
fine structures can be preserved. On top of that:
- colour noise is reduced more strongly than luminance noise (it is never wanted, while a
  little luminance grain looks natural);
- an edge/texture mask lowers the mix where the image has real structure, and keeps it in
  smooth areas such as sky and skin. This protection only makes sense for light noise: with
  strong noise it would keep noisy pixels exactly where the upscaler is most sensitive, so it
  fades out between luma sigma 1.5 and 3.
"""

from __future__ import annotations

import cv2
import numpy as np

_M = np.array(
    [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]], np.float32
)
_MI = np.linalg.inv(_M).astype(np.float32)


def structure_mask(denoised: np.ndarray, noise_sigma: float) -> np.ndarray:
    """0 in flat areas, towards 1 on edges and texture. Measured on the denoised image,
    so the noise itself does not count as structure."""
    y = denoised @ _M[0]
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    # Normalise by the noise level: an edge must stand out clearly from the noise floor.
    ref = max(noise_sigma / 255.0, 1.0 / 255.0) * 12.0
    m = np.clip(g / ref, 0.0, 1.0)
    m = cv2.GaussianBlur(m, (0, 0), 2.0)
    return m * m * (3.0 - 2.0 * m)  # smoothstep


def blend(
    original: np.ndarray,
    denoised: np.ndarray,
    strength: float,
    noise_sigma: float,
    detail_protect: float = 0.6,
    chroma_boost: float = 1.5,
) -> np.ndarray:
    """strength 0..100; detail_protect 0..1 (how much edges keep the original)."""
    if strength <= 0:
        return original
    a = float(np.clip(strength / 100.0, 0.0, 1.0))
    detail_protect *= float(np.clip((3.0 - noise_sigma) / 1.5, 0.0, 1.0))
    mask = structure_mask(denoised, noise_sigma) if detail_protect > 0 else 0.0
    a_luma = a * (1.0 - detail_protect * mask)
    a_chroma = np.minimum(1.0, a * chroma_boost)  # colour noise: no edge protection needed
    yo, yd = original @ _M.T, denoised @ _M.T
    out = yo.copy()
    out[..., 0] += (yd[..., 0] - yo[..., 0]) * a_luma
    out[..., 1:] += (yd[..., 1:] - yo[..., 1:]) * a_chroma
    return out @ _MI.T
