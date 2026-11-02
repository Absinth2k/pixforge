"""The real result for the preview, computed tile by tile in the background.

The quick preview (``preview.render``) is only exact at 100 % zoom and closer. To show the
real result at every zoom level, a worker thread processes the image in a fixed grid of
source tiles, exactly like the export (real context around each tile, faces included), and
keeps every finished tile as a small JPEG pyramid (full output resolution, 1/2, 1/4, ...).
The viewer draws the level that matches the screen, so a fully zoomed-out view shows the real
result downsampled, not a model run on a tiny copy of the image.

- One session per (image, stage, settings). The crop is not part of the key: it only limits
  which tiles are computed, their content is the same.
- Only the most recently opened session is worked on; tiles in the visible region first,
  then outwards. A settings change starts a new session and cancels the running tile.
- Tiles have the lowest GPU priority: they pause while a preview or an export wants the GPU.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from pixforge.core.io import ImageData
from pixforge.core.pipeline import Analysis, Pipeline, Settings, target_size
from pixforge.core.tiling import Cancelled
from pixforge.faces.restore import align_matrix, crop_size_in_image, size_factor
from pixforge.server.gate import Gate
from pixforge.server.preview import render_exact, stage_settings

log = logging.getLogger(__name__)

#: wait this long after a session was opened before using the GPU for it (quick clicks
#: through models or sliders would otherwise start, and cancel, a tile each time)
START_DELAY_S = 0.6
#: total memory for finished tiles of all sessions (JPEG bytes)
MAX_BYTES = 1024 * 2**20
#: smallest pyramid level (longest side, px)
MIN_LEVEL_PX = 24
JPEG_QUALITY = (92, 90)  # full resolution, reduced levels

Rect = tuple[int, int, int, int]  # x, y, w, h in source px
Loader = Callable[[str], tuple[ImageData, Analysis]]
Prepare = Callable[[str, ImageData, Settings], Settings]


def _jpeg(rgb: np.ndarray, quality: int) -> bytes:
    u8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def pyramid(rgb: np.ndarray, levels: int) -> list[bytes]:
    """JPEGs of ``rgb`` at 1, 1/2, 1/4, ... (``levels`` entries)."""
    out = [_jpeg(rgb, JPEG_QUALITY[0])]
    cur = rgb
    for _ in range(1, levels):
        h, w = cur.shape[:2]
        cur = cv2.resize(
            cur, (max(1, round(w / 2)), max(1, round(h / 2))), interpolation=cv2.INTER_AREA
        )
        out.append(_jpeg(cur, JPEG_QUALITY[1]))
    return out


def level_count(tile_out_px: float) -> int:
    return 1 + max(0, int(math.floor(math.log2(max(1.0, tile_out_px / MIN_LEVEL_PX)))))


def _intersects(a: Rect, b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def face_squares(
    pipe: Pipeline, analysis: Analysis, s: Settings, factor: float
) -> list[tuple[float, float, float, float]]:
    """Source-px boxes of the aligned 512 px squares of all faces that will be restored.

    A tile touching such a square is processed together with the whole square, so a face
    split by the tile grid is restored identically on both sides.
    """
    if s.faces.enabled is False or s.faces.strength <= 0:
        return []
    out = []
    for f in pipe.select_faces(analysis.faces, s.faces):
        try:
            m = align_matrix(f.landmarks)
        except ValueError:
            continue
        if size_factor(crop_size_in_image(m) * factor) <= 0.0:
            continue  # too large to be restored at this output size
        inv = cv2.invertAffineTransform(m)
        n = 512.0
        corners = np.array([[0, 0], [n, 0], [0, n], [n, n]], np.float32)
        pts = corners @ inv[:, :2].T + inv[:, 2]
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        out.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
    return out


@dataclass
class Session:
    id: str
    key: str
    image_id: str
    stage: str
    settings: Settings
    width: int
    height: int
    factor: float
    tile: int
    area: Rect
    focus: tuple[float, float, float, float] | None = None
    tiles: dict[tuple[int, int], list[bytes]] = field(default_factory=dict)
    failed: set[tuple[int, int]] = field(default_factory=set)
    opened: float = field(default_factory=time.time)
    used: float = field(default_factory=time.time)
    state: str = "waiting"  # waiting | running | paused | done | error
    error: str | None = None
    squares: list[tuple[float, float, float, float]] | None = None
    prepared: bool = False

    @property
    def cols(self) -> int:
        return math.ceil(self.width / self.tile)

    @property
    def rows(self) -> int:
        return math.ceil(self.height / self.tile)

    @property
    def levels(self) -> int:
        return level_count(self.tile * self.factor)

    def rect(self, i: int, j: int) -> Rect:
        x, y = i * self.tile, j * self.tile
        return x, y, min(self.tile, self.width - x), min(self.tile, self.height - y)

    def in_area(self) -> list[tuple[int, int]]:
        ax, ay, aw, ah = self.area
        t = self.tile
        return [
            (i, j)
            for j in range(ay // t, min(self.rows, math.ceil((ay + ah) / t)))
            for i in range(ax // t, min(self.cols, math.ceil((ax + aw) / t)))
        ]

    def pending(self) -> list[tuple[int, int]]:
        return [k for k in self.in_area() if k not in self.tiles and k not in self.failed]

    def next_tile(self) -> tuple[int, int] | None:
        todo = self.pending()
        if not todo:
            return None
        f = self.focus or (0.0, 0.0, float(self.width), float(self.height))
        fx, fy = f[0] + f[2] / 2, f[1] + f[3] / 2

        def order(k: tuple[int, int]) -> tuple[bool, float]:
            x, y, w, h = self.rect(*k)
            return (not _intersects((x, y, w, h), f), math.hypot(x + w / 2 - fx, y + h / 2 - fy))

        return min(todo, key=order)

    @property
    def nbytes(self) -> int:
        return sum(len(b) for lv in self.tiles.values() for b in lv)

    def counts(self) -> tuple[int, int]:
        area = self.in_area()
        return sum(1 for k in area if k in self.tiles), len(area)

    def public(self) -> dict[str, Any]:
        done, total = self.counts()
        return {
            "id": self.id,
            "image_id": self.image_id,
            "tile": self.tile,
            "factor": self.factor,
            "cols": self.cols,
            "rows": self.rows,
            "width": self.width,
            "height": self.height,
            "levels": self.levels,
            "done": [[i, j] for (i, j) in self.tiles],
            "done_count": done,
            "total": total,
            "state": self.state,
            "error": self.error,
        }


def settings_key(image_id: str, stage: str, s: Settings) -> str:
    d = dataclasses.asdict(replace(s, crop=None, tile=None, jpeg_qf=None))
    return json.dumps([image_id, stage, d], sort_keys=True, default=str)


class TileRenderer:
    def __init__(
        self,
        pipeline: Pipeline,
        loader: Loader,
        gate: Gate,
        publish: Callable[[dict[str, Any]], None],
        tile_size: Callable[[], int],
        prepare: Prepare | None = None,
        max_bytes: int = MAX_BYTES,
        start_delay: float = START_DELAY_S,
    ) -> None:
        self.pipeline = pipeline
        self.loader = loader
        self.gate = gate
        self.publish = publish
        self.tile_size = tile_size
        self.prepare = prepare or (lambda _id, _img, s: s)
        self.max_bytes = max_bytes
        self.start_delay = start_delay
        self.sessions: OrderedDict[str, Session] = OrderedDict()  # by key
        self.by_id: dict[str, Session] = {}
        self.active: Session | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="pixforge-tiles", daemon=True)
        self._thread.start()

    # -- API ---------------------------------------------------------------------------
    def open(
        self,
        image_id: str,
        width: int,
        height: int,
        settings: Settings,
        stage: str,
        focus: tuple[float, float, float, float] | None,
        area: Rect | None,
    ) -> Session:
        s = stage_settings(settings, stage)
        _, _, factor = target_size(width, height, s)
        key = settings_key(image_id, stage, s)
        with self._lock:
            sess = self.sessions.get(key)
            if sess is None:
                sess = Session(
                    id=uuid.uuid4().hex[:12],
                    key=key,
                    image_id=image_id,
                    stage=stage,
                    settings=s,
                    width=width,
                    height=height,
                    factor=factor,
                    tile=max(64, self.tile_size()),
                    area=(0, 0, width, height),
                )
                self.sessions[key] = sess
                self.by_id[sess.id] = sess
            if self.active is not sess:
                sess.opened = time.time()
            sess.used = time.time()
            sess.focus = focus
            sess.area = area or (0, 0, width, height)
            if sess.state == "done" and sess.pending():
                sess.state = "waiting"  # a larger crop / no crop needs more tiles
            self.sessions.move_to_end(key)
            self.active = sess
        self._wake.set()
        return sess

    def get(self, sid: str) -> Session | None:
        return self.by_id.get(sid)

    def tile(self, sid: str, i: int, j: int, level: int) -> bytes | None:
        sess = self.by_id.get(sid)
        if sess is None:
            return None
        lv = sess.tiles.get((i, j))
        if lv is None:
            return None
        return lv[max(0, min(level, len(lv) - 1))]

    def drop_image(self, image_id: str) -> None:
        with self._lock:
            for key in [k for k, s in self.sessions.items() if s.image_id == image_id]:
                sess = self.sessions.pop(key)
                self.by_id.pop(sess.id, None)
                if self.active is sess:
                    self.active = None

    def stop(self) -> None:
        self._stop = True
        self._wake.set()

    # -- worker ------------------------------------------------------------------------
    def _set_state(self, sess: Session, state: str, error: str | None = None) -> None:
        if sess.state == state and sess.error == error:
            return
        sess.state, sess.error = state, error
        done, total = sess.counts()
        self.publish(
            {
                "type": "tiles",
                "session": sess.id,
                "state": state,
                "done": done,
                "total": total,
                "error": error,
            }
        )

    def _evict(self, keep: Session) -> None:
        with self._lock:
            total = sum(s.nbytes for s in self.sessions.values())
            for key in list(self.sessions):
                if total <= self.max_bytes:
                    break
                s = self.sessions[key]
                if s is keep:
                    continue
                total -= s.nbytes
                del self.sessions[key]
                self.by_id.pop(s.id, None)

    def _run(self) -> None:
        while not self._stop:
            sess = self.active
            if sess is None or sess.state in ("done", "error"):
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            wait = sess.opened + self.start_delay - time.time()
            if wait > 0:
                self._wake.wait(wait)
                self._wake.clear()
                continue
            if self.gate.jobs_active():
                self._set_state(sess, "paused")
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            k = sess.next_tile()
            if k is None:
                self._set_state(sess, "done")
                continue
            self._set_state(sess, "running")
            try:
                self._compute(sess, k)
            except Cancelled:
                continue
            except Exception as exc:  # keep the worker alive, show the problem in the UI
                log.exception("tile %s of session %s failed", k, sess.id)
                sess.failed.add(k)
                self._set_state(sess, "error", f"{exc.__class__.__name__}: {exc}")

    def _compute(self, sess: Session, k: tuple[int, int]) -> None:
        def cancel() -> bool:
            return self._stop or self.active is not sess

        t0 = time.perf_counter()
        img, an = self.loader(sess.image_id)
        with self.gate.background():
            if not sess.prepared:
                sess.settings = self.prepare(sess.image_id, img, sess.settings)
                sess.squares = face_squares(self.pipeline, an, sess.settings, sess.factor)
                sess.prepared = True
            x, y, w, h = sess.rect(*k)
            # grow the processed region over every face square the tile touches
            x0, y0, x1, y1 = float(x), float(y), float(x + w), float(y + h)
            for q in sess.squares or []:
                if _intersects((x, y, w, h), (q[0] - 2, q[1] - 2, q[2] + 4, q[3] + 4)):
                    x0, y0 = min(x0, q[0]), min(y0, q[1])
                    x1, y1 = max(x1, q[0] + q[2]), max(y1, q[1] + q[3])
            ex0, ey0 = max(0, math.floor(x0)), max(0, math.floor(y0))
            ex1, ey1 = min(sess.width, math.ceil(x1)), min(sess.height, math.ceil(y1))
            f = sess.factor
            out = render_exact(
                self.pipeline,
                img,
                an,
                sess.settings,
                (ex0, ey0, ex1 - ex0, ey1 - ey0),
                f,
                cancel=cancel,
            )
        ox, oy = round(x * f) - round(ex0 * f), round(y * f) - round(ey0 * f)
        tw, th = round((x + w) * f) - round(x * f), round((y + h) * f) - round(y * f)
        out = np.ascontiguousarray(out[oy : oy + th, ox : ox + tw])
        sess.tiles[k] = pyramid(out, sess.levels)
        done, total = sess.counts()
        log.debug(
            "tile %s/%s of %s: %d ms", done, total, sess.id, (time.perf_counter() - t0) * 1000
        )
        self.publish(
            {"type": "tile", "session": sess.id, "i": k[0], "j": k[1], "done": done, "total": total}
        )
        self._evict(sess)
        if done >= total:
            self._set_state(sess, "done")
