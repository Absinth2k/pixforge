"""Export queue: one worker thread processes jobs in order; pause, cancel, reorder."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pixforge.core import naming
from pixforge.core.io import FORMAT_EXT, ExportOptions, save_image
from pixforge.core.pipeline import Pipeline, Settings
from pixforge.core.tiling import Cancelled

log = logging.getLogger(__name__)


@dataclass
class ExportTarget:
    options: ExportOptions = field(default_factory=ExportOptions)
    folder: str | None = None  # None = next to the source
    pattern: str = naming.DEFAULT_PATTERN
    on_exists: str = "rename"


@dataclass
class Job:
    id: str
    image_id: str
    name: str
    settings: Settings
    target: ExportTarget
    status: str = "queued"  # queued | running | done | error | cancelled | skipped
    stage: str = ""
    progress: float = 0.0
    eta_s: float | None = None
    output: str | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    started: float | None = None
    finished: float | None = None
    _cancel: bool = False

    def public(self) -> dict[str, object]:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "name": self.name,
            "crop": self.settings.crop is not None,
            "task": self.settings.task,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "eta_s": self.eta_s,
            "output": self.output,
            "error": self.error,
            "warnings": self.warnings,
            "seconds": round((self.finished or time.time()) - self.started, 1)
            if self.started
            else None,
        }


class JobQueue:
    def __init__(
        self,
        pipeline: Pipeline,
        loader: Callable[[str], object],
        notify: Callable[[dict[str, object]], None],
    ) -> None:
        self.pipeline = pipeline
        self.loader = loader  # image_id -> (ImageData, Analysis)
        self.notify = notify
        self.jobs: list[Job] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._paused = False
        self._thread = threading.Thread(target=self._run, name="pixforge-jobs", daemon=True)
        self._thread.start()

    # -- control -----------------------------------------------------------------------
    def submit(self, image_id: str, name: str, settings: Settings, target: ExportTarget) -> Job:
        job = Job(uuid.uuid4().hex[:12], image_id, name, settings, target)
        with self._lock:
            self.jobs.append(job)
        self._emit(job)
        self._wake.set()
        return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            for j in self.jobs:
                if j.id == job_id and j.status in ("queued", "running"):
                    j._cancel = True
                    if j.status == "queued":
                        j.status = "cancelled"
                        self._emit(j)
                    return True
        return False

    def pause(self, paused: bool) -> None:
        self._paused = paused
        self.notify({"type": "queue", "paused": paused})
        if not paused:
            self._wake.set()

    @property
    def paused(self) -> bool:
        return self._paused

    def reorder(self, job_ids: list[str]) -> None:
        """New order of the queued jobs (others keep their place at the front)."""
        with self._lock:
            pos = {jid: i for i, jid in enumerate(job_ids)}
            fixed = [j for j in self.jobs if j.status != "queued" or j.id not in pos]
            queued = sorted(
                (j for j in self.jobs if j.status == "queued" and j.id in pos),
                key=lambda j: pos[j.id],
            )
            self.jobs = fixed + queued
        self.notify({"type": "queue", "order": [j.id for j in self.jobs]})

    def clear_finished(self) -> None:
        with self._lock:
            self.jobs = [j for j in self.jobs if j.status in ("queued", "running")]

    @property
    def busy(self) -> bool:
        """An export is being processed right now."""
        return any(j.status == "running" for j in self.jobs)

    def list(self) -> list[dict[str, object]]:
        with self._lock:
            return [j.public() for j in self.jobs]

    # -- worker ------------------------------------------------------------------------
    def _next(self) -> Job | None:
        with self._lock:
            return next((j for j in self.jobs if j.status == "queued"), None)

    def _emit(self, job: Job) -> None:
        self.notify({"type": "job", **job.public()})

    def _run(self) -> None:
        while True:
            job = None if self._paused else self._next()
            if job is None:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            self._process(job)

    def _process(self, job: Job) -> None:
        job.status, job.started = "running", time.time()
        self._emit(job)
        last = [0.0]

        def progress(stage: str, frac: float, _msg: str) -> None:
            job.stage, job.progress = stage, frac
            el = time.time() - (job.started or time.time())
            job.eta_s = round(el / frac - el, 1) if frac > 0.03 else None
            now = time.time()
            if now - last[0] > 0.25 or frac >= 1.0:
                last[0] = now
                self._emit(job)

        try:
            img, analysis = self.loader(job.image_id)  # type: ignore[misc]
            res = self.pipeline.process(
                img, job.settings, progress=progress, cancel=lambda: job._cancel, analysis=analysis
            )
            job.warnings = res.warnings
            t = job.target
            W, H = res.image.size
            src = Path(img.path) if img.path else Path(job.name)
            folder = Path(t.folder) if t.folder else None
            ext = FORMAT_EXT[t.options.format]
            out = naming.output_path(
                src, folder, t.pattern, ext, res.factor, res.model, W, H, t.on_exists
            )
            if out is None:
                job.status = "skipped"
                job.warnings.append("Datei existiert bereits, übersprungen")
            else:
                opts = t.options
                if opts.bit_depth == 0:  # 0 = like the source
                    opts = ExportOptions(**{**opts.__dict__, "bit_depth": img.bit_depth})
                save_image(res.image, out, opts)
                job.output, job.status = str(out), "done"
            job.progress = 1.0
        except Cancelled:
            job.status = "cancelled"
        except Exception as exc:  # report, keep the queue alive
            log.exception("job %s failed", job.id)
            job.status, job.error = "error", _friendly(exc)
        job.finished = time.time()
        self._emit(job)


def _friendly(exc: Exception) -> str:
    from pixforge.backends.base import BackendError
    from pixforge.models.registry import ModelNotInstalled

    if isinstance(exc, (BackendError, ModelNotInstalled, ValueError)):
        return str(exc)
    if isinstance(exc, MemoryError):
        return "Zu wenig Arbeitsspeicher für dieses Bild. Kleinere Ausgabegröße wählen."
    if isinstance(exc, PermissionError):
        return "Keine Schreibrechte im Zielordner."
    return f"Unerwarteter Fehler: {exc.__class__.__name__}: {exc}"
