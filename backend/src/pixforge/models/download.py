"""Model download from the GitHub release ``models-v1``: resumable, size and SHA256 checked.

Files are written to ``<name>.part`` first; an interrupted download continues with an HTTP
range request. Only a file whose size and SHA256 match the registry is renamed into place,
so a half-downloaded or corrupted model can never be loaded.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from pixforge.models import registry
from pixforge.paths import models_dir

log = logging.getLogger(__name__)

CHUNK = 1 << 20
TIMEOUT_S = 30
Progress = Callable[[int, int], None]


class DownloadError(RuntimeError):
    """User-presentable download failure."""


class DownloadCancelled(Exception):
    pass


def precision_for(device_kind: str, info: registry.ModelInfo) -> str:
    """fp16 for graphics cards (1.7x faster), fp32 for the processor."""
    want = "fp16" if device_kind == "gpu" else "fp32"
    return want if want in info.files else next(iter(info.files))


def needed(model_id: str, device_kind: str) -> tuple[str, int] | None:
    """(precision, bytes) still to download for this device, or None if usable already."""
    info = registry.get(model_id)
    p = precision_for(device_kind, info)
    if (models_dir() / info.file_name(p)).is_file():
        return None
    return p, info.files[p].size


def _hash_file(path: Path) -> hashlib._Hash:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h


def download(
    model_id: str,
    precision: str,
    progress: Progress | None = None,
    cancel: Callable[[], bool] | None = None,
    base_url: str | None = None,
) -> Path:
    info = registry.get(model_id)
    mf = info.files.get(precision)
    if mf is None:
        raise DownloadError(f"Für „{info.display_name()}“ gibt es keine {precision}-Datei.")
    dest = models_dir() / mf.file
    if dest.is_file() and dest.stat().st_size == mf.size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have > mf.size:
        part.unlink()
        have = 0
    h = _hash_file(part) if have else hashlib.sha256()
    url = (base_url or registry.release_url()) + mf.file
    req = urllib.request.Request(url, headers={"User-Agent": "Pixforge"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    log.info(
        "downloading %s (%s) from %s, %d of %d bytes present",
        model_id,
        precision,
        url,
        have,
        mf.size,
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            if have and r.status != 206:  # server ignored the range: start over
                have, h = 0, hashlib.sha256()
                part.unlink(missing_ok=True)
            with part.open("ab") as f:
                done = have
                if progress:
                    progress(done, mf.size)
                while chunk := r.read(CHUNK):
                    if cancel and cancel():
                        raise DownloadCancelled
                    f.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, mf.size)
    except DownloadCancelled:
        raise
    except urllib.error.HTTPError as exc:
        raise DownloadError(
            f"Download von „{info.display_name()}“ fehlgeschlagen (Server: {exc.code})."
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DownloadError(
            f"Download von „{info.display_name()}“ unterbrochen. Bitte Internetverbindung "
            "prüfen; beim nächsten Versuch geht es an der gleichen Stelle weiter."
        ) from exc
    size = part.stat().st_size
    if size != mf.size or h.hexdigest() != mf.sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"Die Datei von „{info.display_name()}“ ist beschädigt angekommen und wurde "
            "verworfen. Bitte noch einmal versuchen."
        )
    os.replace(part, dest)
    log.info("model %s (%s) installed", model_id, precision)
    return dest
