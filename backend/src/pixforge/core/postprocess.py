"""Classic image operations around the AI steps (all float32 RGB 0..1)."""

from __future__ import annotations

import cv2
import numpy as np

_Y = np.array([0.299, 0.587, 0.114], np.float32)


def resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """High-quality resize: Lanczos for enlarging / mild shrinking, area for strong shrinking."""
    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return img
    shrink = width < w * 0.5 or height < h * 0.5
    interp = cv2.INTER_AREA if shrink else cv2.INTER_LANCZOS4
    out = cv2.resize(img, (width, height), interpolation=interp)
    return out if out.ndim == img.ndim else out[..., None]


def luma(img: np.ndarray) -> np.ndarray:
    return img @ _Y


def sharpen(img: np.ndarray, amount: float, radius: float = 1.2) -> np.ndarray:
    """Unsharp mask on luma only (no colour fringes). amount: 0..100."""
    if amount <= 0:
        return img
    y = luma(img)
    blur = cv2.GaussianBlur(y, (0, 0), radius)
    detail = (y - blur) * (amount / 100.0 * 1.5)
    return img + detail[..., None]


def detail_mix(ai: np.ndarray, classic: np.ndarray, amount: float) -> np.ndarray:
    """Blend the AI result with a plain Lanczos upscale (counters the 'plastic' look)."""
    if amount <= 0:
        return ai
    a = float(np.clip(amount / 100.0, 0.0, 1.0))
    return ai * (1.0 - a) + classic * a


def add_grain(
    img: np.ndarray, amount: float, size: float = 1.0, seed: int = 0, rows: int = 1024
) -> np.ndarray:
    """Monochrome film-like grain, stronger in midtones. amount: 0..100."""
    if amount <= 0:
        return img
    rng = np.random.default_rng(seed)
    sigma = amount / 100.0 * 0.04
    out = img.copy()
    for y0 in range(0, img.shape[0], rows):
        band = out[y0 : y0 + rows]
        noise = rng.standard_normal(band.shape[:2]).astype(np.float32)
        if size > 0.5:
            noise = cv2.GaussianBlur(noise, (0, 0), size * 0.6)
            noise /= max(float(noise.std()), 1e-6)
        y = luma(band)
        weight = 4.0 * y * (1.0 - y)  # 0 in black/white, 1 in midtones
        band += (noise * weight * sigma)[..., None]
    return out


def clip01(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 1.0, out=img)
