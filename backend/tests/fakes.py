"""Test doubles that behave like InferenceBackends without any model or GPU."""

from __future__ import annotations

import numpy as np

from pixforge.backends.base import DeviceInfo, InferenceBackend, OutOfMemory


class NearestBackend(InferenceBackend):
    """Nearest-neighbour upscaler that deliberately corrupts the outer ``edge`` pixels of every
    tile (like real networks do at tile borders). Seamless tiling must hide that."""

    def __init__(
        self, scale: int = 2, alignment: int = 16, edge: int = 3, oom_above: int | None = None
    ) -> None:
        self.scale = scale
        self.alignment = alignment
        self.edge = edge
        self.oom_above = oom_above
        self.calls: list[tuple[int, int]] = []

    def load(self, model_id: object) -> None:  # pragma: no cover - nothing to load
        pass

    def run_tile(self, tile: np.ndarray) -> np.ndarray:
        h, w = tile.shape[:2]
        self.calls.append((h, w))
        if self.oom_above and max(h, w) > self.oom_above:
            raise OutOfMemory("fake out of memory")
        out = tile.repeat(self.scale, 0).repeat(self.scale, 1).copy()
        e = self.edge * self.scale
        if e:
            out[:e] = 1.0
            out[-e:] = 1.0
            out[:, :e] = 0.0
            out[:, -e:] = 0.0
        return out

    def device_info(self) -> DeviceInfo:
        return DeviceInfo("Fake GPU", "gpu", "fake", vram_mb=8192, discrete=True, device_index=0)


class IdentityFaceBackend(InferenceBackend):
    """Returns a constant colour for every 512 px face crop."""

    scale = 1
    alignment = 1

    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.calls = 0

    def load(self, model_id: object) -> None:  # pragma: no cover
        pass

    def run_tile(self, tile: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.full_like(tile, self.value)

    def device_info(self) -> DeviceInfo:
        return DeviceInfo("Fake GPU", "gpu", "fake", vram_mb=8192)
