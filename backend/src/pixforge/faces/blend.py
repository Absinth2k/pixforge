"""Soft face mask and inverse-warp paste for restored faces."""

from __future__ import annotations

import cv2
import numpy as np


def face_mask(size: int = 512) -> np.ndarray:
    """Feathered ellipse covering forehead to chin in FFHQ-aligned space (float32 0..1)."""
    s = size / 512.0
    mask = np.zeros((size, size), np.float32)
    cv2.ellipse(
        mask,
        (round(256 * s), round(282 * s)),
        (round(172 * s), round(214 * s)),
        0,
        0,
        360,
        1.0,
        -1,
        cv2.LINE_AA,
    )
    return cv2.GaussianBlur(mask, (0, 0), 20.0 * s)


def paste(
    img: np.ndarray, face: np.ndarray, mask: np.ndarray, m: np.ndarray, strength: float
) -> np.ndarray:
    """Blend ``face`` (aligned crop) back into ``img`` using the inverse of affine ``m``.

    Only the bounding box of the face is warped, so huge images stay cheap.
    """
    h, w = img.shape[:2]
    inv = cv2.invertAffineTransform(m)
    n = face.shape[0]
    corners = np.array([[0, 0], [n, 0], [0, n], [n, n]], np.float32)
    pts = corners @ inv[:, :2].T + inv[:, 2]
    x0, y0 = np.floor(pts.min(0)).astype(int) - 2
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 2
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return img
    inv_local = inv.copy()
    inv_local[:, 2] -= (x0, y0)
    size = (x1 - x0, y1 - y0)
    warped = cv2.warpAffine(
        face, inv_local, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    alpha = cv2.warpAffine(
        mask,
        inv_local,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    alpha = (alpha * strength)[..., None]
    out = img if img.flags.writeable else img.copy()
    region = out[y0:y1, x0:x1]
    region += (warped - region) * alpha
    return out
