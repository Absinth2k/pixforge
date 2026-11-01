"""Seamless tiled inference.

Tiles of a constant size (a new shape costs a DirectML shader compile) overlap by ``overlap``
input pixels. Each tile gets a separable weight w(y, x) = wy(y) * wx(x): almost zero within a
small margin of an inner tile border (networks produce their worst pixels there), then a
linear ramp up to 1. Because the tile positions form a full grid, the per-pixel sum of weights
is the product of the 1-D sums, so results are normalised without a full-size weight map.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

TileFn = Callable[[np.ndarray], np.ndarray]
ProgressFn = Callable[[int, int], None]

EPS_WEIGHT = 1e-4


class Cancelled(Exception):
    """Raised when the cancel callback returns True."""


@dataclass(frozen=True)
class TilePlan:
    tile_h: int
    tile_w: int
    ys: tuple[int, ...]
    xs: tuple[int, ...]
    padded_h: int
    padded_w: int

    @property
    def count(self) -> int:
        return len(self.ys) * len(self.xs)


def _ceil_to(v: int, a: int) -> int:
    return int(math.ceil(v / a) * a)


def axis_positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = tile - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than the tile")
    pos = list(range(0, length - tile, step))
    pos.append(length - tile)
    return sorted(set(pos))


def plan(h: int, w: int, tile: int, overlap: int, align: int = 1) -> TilePlan:
    """Tile layout for an h x w image. Small images get a smaller (aligned) tile."""
    if tile % align:
        raise ValueError(f"tile {tile} must be a multiple of {align}")
    th = min(tile, _ceil_to(h, align))
    tw = min(tile, _ceil_to(w, align))
    ph, pw = max(h, th), max(w, tw)
    ov_h = min(overlap, th // 2)
    ov_w = min(overlap, tw // 2)
    return TilePlan(
        th, tw, tuple(axis_positions(ph, th, ov_h)), tuple(axis_positions(pw, tw, ov_w)), ph, pw
    )


def axis_weights(starts: tuple[int, ...], tile: int, length: int, overlap: int) -> list[np.ndarray]:
    """1-D weights for each tile along one axis (all in output pixel units)."""
    margin = max(1, overlap // 8)
    ramp = max(1, overlap // 2)
    idx = np.arange(tile, dtype=np.float32)
    out = []
    for s in starts:
        d_left = idx if s > 0 else np.full(tile, np.inf, np.float32)
        d_right = (tile - 1 - idx) if s + tile < length else np.full(tile, np.inf, np.float32)
        d = np.minimum(d_left, d_right)
        out.append(np.clip((d - margin + 1) / ramp, EPS_WEIGHT, 1.0).astype(np.float32))
    return out


def run_tiled(
    img: np.ndarray,
    fn: TileFn,
    scale: int,
    tile: int,
    overlap: int = 32,
    align: int = 1,
    progress: ProgressFn | None = None,
    cancel: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Apply ``fn`` (tile -> tile upscaled by ``scale``) over ``img`` (h, w, c) seamlessly."""
    h, w = img.shape[:2]
    p = plan(h, w, tile, overlap, align)
    if (p.padded_h, p.padded_w) != (h, w):
        img = np.pad(
            img,
            ((0, p.padded_h - h), (0, p.padded_w - w), (0, 0)),
            mode="reflect" if min(h, w) > 1 else "edge",
        )
    H, W = p.padded_h * scale, p.padded_w * scale
    th, tw = p.tile_h * scale, p.tile_w * scale
    ov_h = min(overlap, p.tile_h // 2) * scale
    ov_w = min(overlap, p.tile_w // 2) * scale
    wys = axis_weights(tuple(y * scale for y in p.ys), th, H, ov_h)
    wxs = axis_weights(tuple(x * scale for x in p.xs), tw, W, ov_w)
    norm_y = np.zeros(H, np.float32)
    norm_x = np.zeros(W, np.float32)
    for y, wy in zip(p.ys, wys, strict=True):
        norm_y[y * scale : y * scale + th] += wy
    for x, wx in zip(p.xs, wxs, strict=True):
        norm_x[x * scale : x * scale + tw] += wx

    out: np.ndarray | None = None
    done = 0
    for y, wy in zip(p.ys, wys, strict=True):
        for x, wx in zip(p.xs, wxs, strict=True):
            if cancel and cancel():
                raise Cancelled
            res = fn(np.ascontiguousarray(img[y : y + p.tile_h, x : x + p.tile_w]))
            if res.shape[:2] != (th, tw):
                raise ValueError(f"tile function returned {res.shape}, expected {(th, tw)}")
            if out is None:
                out = np.zeros((H, W, res.shape[2]), np.float32)
            Y, X = y * scale, x * scale
            out[Y : Y + th, X : X + tw] += res * (wy[:, None, None] * wx[None, :, None])
            done += 1
            if progress:
                progress(done, p.count)
    assert out is not None
    out /= norm_y[:, None, None]
    out /= norm_x[None, :, None]
    return out[: h * scale, : w * scale]
