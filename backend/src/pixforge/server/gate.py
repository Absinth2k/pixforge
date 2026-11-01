"""One GPU, two kinds of work: interactive previews must not wait for a long export.

All inference goes through ``GatedBackend.run_tile``. A global lock serialises device access
(one DirectML device, limited VRAM), and while a preview is waiting, export work pauses before
its next tile. Previews therefore interleave with an export at tile granularity.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import numpy as np

from pixforge.backends.base import DeviceInfo, InferenceBackend

_local = threading.local()


class Gate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count_lock = threading.Lock()
        self._previews_waiting = 0

    @contextmanager
    def preview(self):  # type: ignore[no-untyped-def]
        """Mark the current thread as doing preview work (high priority)."""
        old = getattr(_local, "priority", False)
        _local.priority = True
        try:
            yield
        finally:
            _local.priority = old

    def run(self, fn, *args):  # type: ignore[no-untyped-def]
        if getattr(_local, "priority", False):
            with self._count_lock:
                self._previews_waiting += 1
            try:
                with self._lock:
                    return fn(*args)
            finally:
                with self._count_lock:
                    self._previews_waiting -= 1
        while self._previews_waiting > 0:
            time.sleep(0.003)
        with self._lock:
            return fn(*args)


class GatedBackend(InferenceBackend):
    """Wraps a real backend; every tile goes through the gate."""

    def __init__(self, inner: InferenceBackend, gate: Gate) -> None:
        self.inner = inner
        self.gate = gate

    # forward attributes the pipeline reads (scale, alignment, precision, jpeg_qf, ...)
    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self.inner, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in ("inner", "gate"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    @property
    def scale(self) -> int:  # type: ignore[override]
        return self.inner.scale

    @property
    def alignment(self) -> int:  # type: ignore[override]
        return self.inner.alignment

    def load(self, model_id: object) -> None:  # pragma: no cover - loaded by the factory
        self.inner.load(model_id)  # type: ignore[arg-type]

    def run_tile(self, tile: np.ndarray) -> np.ndarray:
        return self.gate.run(self.inner.run_tile, tile)

    def device_info(self) -> DeviceInfo:
        return self.inner.device_info()

    def close(self) -> None:
        self.inner.close()
