"""One GPU, three kinds of work, in this order of priority:

1. interactive previews (the user waits for them),
2. export jobs,
3. background tiles of the real preview result (only when nothing else wants the GPU).

All inference goes through ``GatedBackend.run_tile``. A global lock serialises device access
(one DirectML device, limited VRAM); lower priorities wait before their next tile while a
higher one is waiting. Work therefore interleaves at tile granularity.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import contextmanager

import numpy as np

from pixforge.backends.base import DeviceInfo, InferenceBackend

_local = threading.local()

PREVIEW, JOB, BACKGROUND = 2, 1, 0


class Gate:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._count_lock = threading.Lock()
        self._waiting = [0, 0, 0]  # per priority: threads waiting for or holding the lock
        #: set by the server: True while an export job is running (background tiles yield)
        self.jobs_active: Callable[[], bool] = lambda: False

    @contextmanager
    def priority(self, level: int):  # type: ignore[no-untyped-def]
        old = getattr(_local, "priority", JOB)
        _local.priority = level
        try:
            yield
        finally:
            _local.priority = old

    def preview(self):  # type: ignore[no-untyped-def]
        """Mark the current thread as doing preview work (high priority)."""
        return self.priority(PREVIEW)

    def background(self):  # type: ignore[no-untyped-def]
        return self.priority(BACKGROUND)

    def _blocked(self, p: int) -> bool:
        if p == PREVIEW:
            return False
        if self._waiting[PREVIEW] > 0:
            return True
        return p == BACKGROUND and (self._waiting[JOB] > 0 or self.jobs_active())

    def run(self, fn, *args):  # type: ignore[no-untyped-def]
        p = getattr(_local, "priority", JOB)
        while self._blocked(p):
            time.sleep(0.003 if p == JOB else 0.02)
        with self._count_lock:
            self._waiting[p] += 1
        try:
            with self.lock:
                return fn(*args)
        finally:
            with self._count_lock:
                self._waiting[p] -= 1


class _Closed(Exception):
    pass


class GatedBackend(InferenceBackend):
    """Wraps a real backend; every tile goes through the gate.

    The pipeline may unload a model while another thread still holds this wrapper (an export
    with model A, a preview switching to model B). The close happens under the gate lock, so
    it never interrupts a running tile; a later ``run_tile`` asks ``reacquire`` for a fresh
    backend of the same model instead of crashing.
    """

    def __init__(
        self,
        inner: InferenceBackend,
        gate: Gate,
        model_id: str = "",
        reacquire: Callable[[str], InferenceBackend] | None = None,
    ) -> None:
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "reacquire", reacquire)
        object.__setattr__(self, "closed", False)

    # forward attributes the pipeline reads (scale, alignment, precision, jpeg_qf, ...)
    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self.inner, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in ("inner", "gate", "model_id", "reacquire", "closed"):
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

    def _run(self, tile: np.ndarray) -> np.ndarray:
        if self.closed:
            raise _Closed
        return self.inner.run_tile(tile)

    def run_tile(self, tile: np.ndarray) -> np.ndarray:
        try:
            return self.gate.run(self._run, tile)
        except _Closed:
            if self.reacquire is None:
                raise RuntimeError(f"model {self.model_id} was unloaded") from None
            fresh = self.reacquire(self.model_id)
            for attr in ("jpeg_qf",):  # per-image state set by the pipeline
                if hasattr(self.inner, attr):
                    setattr(fresh, attr, getattr(self.inner, attr))
            return fresh.run_tile(tile)

    def device_info(self) -> DeviceInfo:
        return self.inner.device_info()

    def close(self) -> None:
        with self.gate.lock:
            if not self.closed:
                self.closed = True
                self.inner.close()
