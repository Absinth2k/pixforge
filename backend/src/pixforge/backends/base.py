"""Inference backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DeviceInfo:
    """What the backend runs on, for the status bar and for tile-size decisions."""

    name: str  # e.g. "AMD Radeon RX 9070 XT"
    kind: str  # "gpu" | "cpu"
    provider: str  # e.g. "DmlExecutionProvider", "CPUExecutionProvider", "torch-rocm"
    vram_mb: int | None = None  # dedicated video memory, None for CPU
    discrete: bool | None = None
    device_index: int | None = None  # DXGI adapter index for DirectML


class BackendError(RuntimeError):
    """Raised with a user-presentable message."""


class InferenceBackend(ABC):
    """Runs one image-to-image model on tiles.

    Tiles are float32 RGB arrays in 0..1 with shape (h, w, 3); alpha is handled by the
    pipeline. ``run_tile`` returns (h * scale, w * scale, 3).
    """

    #: integer upscale factor of the loaded model (1 for denoisers / restorers)
    scale: int = 1
    #: tile sides must be a multiple of this (HAT window size = 16)
    alignment: int = 1

    @abstractmethod
    def load(self, model_id: str | Path) -> None:
        """Load a model by registry id or by file path."""

    @abstractmethod
    def run_tile(self, tile: np.ndarray) -> np.ndarray:
        """Process one (h, w, 3) float32 tile and return the result as (H, W, 3) float32."""

    @abstractmethod
    def device_info(self) -> DeviceInfo:
        """Describe the device this backend runs on."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release device resources."""
