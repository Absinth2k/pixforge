"""Model downloads in the background, one after another, progress pushed as events."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from pixforge.models import download as dl
from pixforge.models import registry

log = logging.getLogger(__name__)


@dataclass
class DownloadState:
    id: str
    precision: str
    total: int
    done: int = 0
    state: str = "queued"  # queued | running | done | error | cancelled
    error: str | None = None


class Downloads:
    def __init__(
        self,
        publish: Callable[[dict[str, Any]], None],
        device_kind: Callable[[], str],
        on_installed: Callable[[str], None] | None = None,
    ) -> None:
        self.publish = publish
        self.device_kind = device_kind
        self.on_installed = on_installed
        self.items: dict[str, DownloadState] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._cancel: set[str] = set()
        self._thread = threading.Thread(target=self._run, name="pixforge-downloads", daemon=True)
        self._thread.start()

    def request(self, model_ids: list[str]) -> list[dict[str, Any]]:
        out = []
        with self._lock:
            for mid in model_ids:
                need = dl.needed(mid, self.device_kind())
                cur = self.items.get(mid)
                if need is None:
                    continue
                if cur is None or cur.state in ("error", "cancelled", "done"):
                    cur = DownloadState(mid, need[0], need[1])
                    self.items[mid] = cur
                    self._cancel.discard(mid)
                    self._emit(cur)
                out.append(asdict(cur))
        self._wake.set()
        return out

    def cancel(self, model_id: str) -> None:
        with self._lock:
            cur = self.items.get(model_id)
            if cur and cur.state in ("queued", "running"):
                self._cancel.add(model_id)
                if cur.state == "queued":
                    cur.state = "cancelled"
                    self._emit(cur)

    def state(self, model_id: str) -> dict[str, Any] | None:
        cur = self.items.get(model_id)
        return asdict(cur) if cur else None

    def _emit(self, s: DownloadState) -> None:
        self.publish({"type": "download", **asdict(s)})

    def _next(self) -> DownloadState | None:
        with self._lock:
            return next((s for s in self.items.values() if s.state == "queued"), None)

    def _run(self) -> None:
        while True:
            s = self._next()
            if s is None:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            s.state = "running"
            self._emit(s)
            last = [0.0]

            def progress(
                done: int, total: int, s: DownloadState = s, last: list[float] = last
            ) -> None:
                s.done, s.total = done, total
                now = time.monotonic()
                if now - last[0] > 0.25 or done >= total:
                    last[0] = now
                    self._emit(s)

            try:
                dl.download(s.id, s.precision, progress, cancel=lambda s=s: s.id in self._cancel)
                s.state, s.done = "done", s.total
                if self.on_installed:
                    self.on_installed(s.id)
            except dl.DownloadCancelled:
                s.state = "cancelled"
            except dl.DownloadError as exc:
                s.state, s.error = "error", str(exc)
            except Exception as exc:  # keep the worker alive
                log.exception("download of %s failed", s.id)
                s.state, s.error = "error", f"{exc.__class__.__name__}: {exc}"
            self._emit(s)


def missing_defaults(device_kind: str) -> list[str]:
    return [m.id for m in registry.all_models() if m.default and dl.needed(m.id, device_kind)]
