"""Colour management.

Models are trained on sRGB images. Policy:
- 8-bit sources with a non-sRGB ICC profile are converted to sRGB on load (exact, LittleCMS).
- 16-bit sources keep their colour space (no 8-bit round trip); the profile is re-embedded on
  export. On request they are converted to sRGB through a 33^3 3D-LUT built with LittleCMS
  and trilinear interpolation, which keeps 16-bit smoothness.
- Files without a profile are assumed to be sRGB.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageCms

if TYPE_CHECKING:
    from pixforge.core.io import ImageData

LUT_SIZE = 33


@lru_cache(maxsize=1)
def srgb_profile_bytes() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _profile(icc: bytes) -> ImageCms.ImageCmsProfile:
    return ImageCms.ImageCmsProfile(io.BytesIO(icc))


def is_srgb(icc: bytes | None) -> bool:
    if not icc:
        return True
    try:
        desc = ImageCms.getProfileDescription(_profile(icc)) or ""
    except Exception:
        return True  # unreadable profile: treat as sRGB rather than fail
    return "srgb" in desc.lower().replace(" ", "") or "iec61966" in desc.lower()


def profile_name(icc: bytes | None) -> str:
    if not icc:
        return "sRGB"
    try:
        return (ImageCms.getProfileDescription(_profile(icc)) or "ICC").strip()
    except Exception:
        return "ICC"


def _transform_u8(arr_u8: np.ndarray, src: bytes, dst: bytes) -> np.ndarray:
    im = Image.fromarray(arr_u8, "RGB")
    out = ImageCms.profileToProfile(
        im,
        _profile(src),
        _profile(dst),
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        outputMode="RGB",
    )
    return np.asarray(out)


@lru_cache(maxsize=4)
def _lut(src: bytes, dst: bytes) -> np.ndarray:
    """(N, N, N, 3) float32 LUT indexed [r, g, b]."""
    n = LUT_SIZE
    grid = np.linspace(0, 255, n).round().astype(np.uint8)
    r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")
    cube = np.stack([r, g, b], -1).reshape(n * n, n, 3)
    out = _transform_u8(cube, src, dst).astype(np.float32) / 255.0
    return out.reshape(n, n, n, 3)


def _apply_lut(rgb: np.ndarray, lut: np.ndarray, rows: int = 256) -> np.ndarray:
    n = lut.shape[0]
    out = np.empty_like(rgb)
    for y0 in range(0, rgb.shape[0], rows):
        x = np.clip(rgb[y0 : y0 + rows], 0.0, 1.0) * (n - 1)
        i0 = np.minimum(x.astype(np.int32), n - 2)
        f = x - i0
        r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
        fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
        acc = np.zeros(x.shape, np.float32)
        for dr, wr in ((0, 1 - fr), (1, fr)):
            for dg, wg in ((0, 1 - fg), (1, fg)):
                for db, wb in ((0, 1 - fb), (1, fb)):
                    acc += lut[r0 + dr, g0 + dg, b0 + db] * (wr * wg * wb)
        out[y0 : y0 + rows] = acc
    return out


def to_working_space(data: ImageData) -> None:
    """Called by the loader: decide the working colour space of ``data`` in place."""
    icc = data.source_icc
    if is_srgb(icc):
        data.icc = None
        return
    assert icc is not None
    if data.bit_depth <= 8:
        u8 = (np.clip(data.rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
        data.rgb = _transform_u8(u8, icc, srgb_profile_bytes()).astype(np.float32) / 255.0
        data.icc = None
        data.meta["converted_from"] = profile_name(icc)
    else:
        data.icc = icc
        data.meta["working_space"] = profile_name(icc)


def for_export(data: ImageData, mode: str) -> tuple[np.ndarray, bytes]:
    """Pixels and the ICC profile to embed for the chosen export mode ("embed" | "srgb")."""
    if data.icc is None:
        return data.rgb, srgb_profile_bytes()
    if mode == "srgb":
        return _apply_lut(data.rgb, _lut(data.icc, srgb_profile_bytes())), srgb_profile_bytes()
    return data.rgb, data.icc
