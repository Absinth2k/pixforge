# Shifted-window attention mask for HAT / Swin-style models.
# Math identical to HAT.calculate_mask (XPixelGroup/HAT, Apache-2.0), ported to NumPy
# so the runtime does not need PyTorch.
"""Attention mask helper (NumPy only)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def _mask_cached(h: int, w: int, window_size: int) -> np.ndarray:
    ws, ss = window_size, window_size // 2
    if h % ws or w % ws:
        raise ValueError(f"tile {h}x{w} is not a multiple of the window size {ws}")
    img = np.zeros((h, w), dtype=np.float32)
    slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
    cnt = 0
    for hs in slices:
        for wsl in slices:
            img[hs, wsl] = cnt
            cnt += 1
    # window_partition -> (num_windows, ws*ws)
    mw = img.reshape(h // ws, ws, w // ws, ws).transpose(0, 2, 1, 3).reshape(-1, ws * ws)
    diff = mw[:, None, :] - mw[:, :, None]
    mask = np.where(diff != 0, np.float32(-100.0), np.float32(0.0))
    mask.setflags(write=False)
    return mask


def shifted_window_mask(h: int, w: int, window_size: int = 16) -> np.ndarray:
    """Return the (num_windows, ws*ws, ws*ws) float32 mask for an h x w feature map."""
    return _mask_cached(int(h), int(w), int(window_size))
