"""Seamless tiling: tiled results must match a single full-image pass."""

from __future__ import annotations

import numpy as np
import pytest
from fakes import NearestBackend

from pixforge.core.tiling import Cancelled, axis_positions, plan, run_tiled


def _img(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((h, w, 3), dtype=np.float32)


@pytest.mark.parametrize(("h", "w"), [(300, 500), (64, 64), (130, 1000), (17, 23), (512, 384)])
def test_no_seams_despite_corrupted_tile_borders(h: int, w: int) -> None:
    img = _img(h, w)
    b = NearestBackend(scale=2, alignment=16, edge=3)
    out = run_tiled(img, b.run_tile, 2, tile=128, overlap=32, align=16)
    ref = img.repeat(2, 0).repeat(2, 1)
    assert out.shape == ref.shape
    # The fake corrupts every tile border. At the outer image border no neighbour can
    # replace those pixels (real networks see padding there), so only inner seams count.
    e = b.edge * 2
    inner = (slice(e, -e), slice(e, -e))
    # Inner border garbage gets weight 1e-4: the error stays far below one 8-bit step.
    assert np.abs(out[inner] - ref[inner]).max() < 1e-3


def test_constant_tile_shape() -> None:
    b = NearestBackend(scale=1, alignment=16, edge=0)
    run_tiled(_img(700, 450), b.run_tile, 1, tile=192, overlap=32, align=16)
    assert set(b.calls) == {(192, 192)}


def test_small_image_uses_small_aligned_tile() -> None:
    p = plan(40, 70, tile=256, overlap=32, align=16)
    assert (p.tile_h, p.tile_w) == (64, 128)  # smallest buckets that fit, one tile
    assert p.count == 1
    # a preview region just above 384 px fits one 448x320 tile instead of two 384x384 tiles
    p = plan(280, 402, tile=384, overlap=32, align=16)
    assert (p.tile_h, p.tile_w, p.count) == (320, 448, 1)
    # tile shapes stay within a small set and never exceed the VRAM budget of tile x tile
    for h in range(20, 900, 23):
        for w in range(20, 900, 29):
            q = plan(h, w, 384, 32, 16)
            assert q.tile_h in (64, 128, 192, 256, 320, 384, 448, 512)
            assert q.tile_w in (64, 128, 192, 256, 320, 384, 448, 512)
            assert q.tile_h * q.tile_w <= 384 * 384


def test_positions_cover_axis() -> None:
    pos = axis_positions(1000, 256, 32)
    assert pos[0] == 0 and pos[-1] == 1000 - 256
    assert all(b - a <= 256 - 32 for a, b in zip(pos, pos[1:], strict=False))


def test_progress_and_cancel() -> None:
    b = NearestBackend(scale=1, edge=0)
    seen: list[tuple[int, int]] = []
    run_tiled(_img(300, 300), b.run_tile, 1, 128, 32, 16, progress=lambda d, n: seen.append((d, n)))
    assert seen[-1][0] == seen[-1][1] and len(seen) == seen[-1][1]
    with pytest.raises(Cancelled):
        run_tiled(_img(300, 300), b.run_tile, 1, 128, 32, 16, cancel=lambda: True)
