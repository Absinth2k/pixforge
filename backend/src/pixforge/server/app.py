"""Pixforge HTTP API for the desktop app (127.0.0.1 only, token protected).

Images are loaded and analysed once; the UI then asks for small JPEG renderings of the
visible region (original or processed), never for full-size images. Progress of export jobs is
pushed over a WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from pixforge import __version__
from pixforge.backends.base import InferenceBackend
from pixforge.core import io as pio
from pixforge.core.io import ExportOptions, ImageData, ImageLoadError, load_image
from pixforge.core.pipeline import (
    Analysis,
    DenoiseSettings,
    FaceSettings,
    Pipeline,
    Settings,
    available_ram_mb,
    estimate_seconds,
    target_size,
)
from pixforge.core.tiling import Cancelled
from pixforge.denoise.models import suggest
from pixforge.models import registry
from pixforge.server import preview as pv
from pixforge.server.gate import Gate, GatedBackend
from pixforge.server.jobs import ExportTarget, JobQueue

log = logging.getLogger(__name__)
MAX_LOADED = 4  # full-resolution images kept in RAM (analysis is always kept)


# ------------------------------------------------------------------------- schemas
class DenoiseIn(BaseModel):
    mode: Literal["auto", "on", "off"] = "auto"
    strength: float | None = Field(None, ge=0, le=100)
    model: str = "denoise"
    detail_protect: float = Field(60, ge=0, le=100)


class FacesIn(BaseModel):
    enabled: bool | None = None
    model: str = "face"
    strength: float = Field(80, ge=0, le=100)
    min_size: float = 16
    only_largest: bool = False
    exclude: list[int] = []


class SettingsIn(BaseModel):
    task: Literal["upscale", "denoise"] = "upscale"
    model: str = "photo"
    scale: float = Field(4.0, gt=0, le=16)
    target_width: int | None = None
    target_height: int | None = None
    denoise: DenoiseIn = DenoiseIn()
    jpeg_strength: float = Field(0, ge=0, le=100)
    sharpen: float = Field(0, ge=0, le=100)
    detail_mix: float = Field(0, ge=0, le=100)
    grain: float = Field(0, ge=0, le=100)
    faces: FacesIn = FacesIn()
    crop: tuple[int, int, int, int] | None = None
    tile: int | None = None

    def to_settings(self) -> Settings:
        d = self.model_dump()
        return Settings(
            task=d["task"],
            model=d["model"],
            scale=d["scale"],
            target_width=d["target_width"],
            target_height=d["target_height"],
            denoise=DenoiseSettings(**d["denoise"]),
            jpeg_strength=d["jpeg_strength"],
            sharpen=d["sharpen"],
            detail_mix=d["detail_mix"],
            grain=d["grain"],
            faces=FaceSettings(**{**d["faces"], "exclude": tuple(d["faces"]["exclude"])}),
            crop=tuple(d["crop"]) if d["crop"] else None,
            tile=d["tile"],
        )


class RegionIn(BaseModel):
    x: float
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class PreviewIn(BaseModel):
    image_id: str
    settings: SettingsIn = SettingsIn()
    region: RegionIn
    out_w: int = Field(gt=0, le=8192)
    out_h: int = Field(gt=0, le=8192)
    stage: Literal["original", "denoised", "result"] = "result"


class AddImagesIn(BaseModel):
    paths: list[str]


class ExportIn(BaseModel):
    format: Literal["jpeg", "png", "tiff", "webp"] = "jpeg"
    quality: int = Field(92, ge=1, le=100)
    bit_depth: int = 0  # 0 = like the source (PNG/TIFF)
    compression: Literal["lzw", "zip", "none"] = "lzw"
    webp_lossless: bool = False
    color: Literal["embed", "srgb"] = "embed"
    keep_exif: bool = True
    folder: str | None = None
    pattern: str = "{name}_pixforge"
    on_exists: Literal["rename", "overwrite", "skip"] = "rename"


class JobIn(BaseModel):
    image_id: str
    settings: SettingsIn = SettingsIn()
    export: ExportIn = ExportIn()


class EstimateIn(BaseModel):
    image_id: str
    settings: SettingsIn = SettingsIn()


# ------------------------------------------------------------------------- image store
@dataclass
class Entry:
    id: str
    path: Path
    analysis: Analysis
    info: dict[str, Any]
    thumb: bytes
    img: ImageData | None = None
    used: float = 0.0


class ImageStore:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.entries: OrderedDict[str, Entry] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, path: str) -> Entry:
        p = Path(path)
        img = load_image(p)
        an = self.pipeline.analyze(img, Settings(model="clean"))
        eid = uuid.uuid4().hex[:12]
        h, w = img.rgb.shape[:2]
        sug = {k: _suggestion(an, img, k) for k in ("photo", "clean")}
        info = {
            "id": eid,
            "path": str(p),
            "name": p.name,
            "width": w,
            "height": h,
            "bit_depth": img.bit_depth,
            "format": img.format,
            "has_alpha": img.alpha is not None,
            "meta": {k: v for k, v in img.meta.items() if isinstance(v, (str, int, float, bool))},
            "noise": {
                "luma": round(an.noise.luma_sigma, 2),
                "chroma": round(an.noise.chroma_sigma, 2),
                "grayscale": an.noise.is_grayscale,
            },
            "denoise": sug,
            "faces": [f.to_dict() for f in an.faces],
            "color_space": "sRGB" if img.icc is None else "embedded",
        }
        entry = Entry(eid, p, an, info, pio.thumbnail_bytes(img.rgb), img, time.time())
        with self._lock:
            self.entries[eid] = entry
            self._evict()
        return entry

    def get(self, eid: str) -> Entry:
        e = self.entries.get(eid)
        if e is None:
            raise HTTPException(404, "Bild nicht gefunden")
        return e

    def image(self, eid: str) -> tuple[ImageData, Analysis]:
        e = self.get(eid)
        with self._lock:
            if e.img is None:
                e.img = load_image(e.path)
            e.used = time.time()
            self.entries.move_to_end(eid)
            self._evict()
            return e.img, e.analysis

    def remove(self, eid: str) -> None:
        with self._lock:
            self.entries.pop(eid, None)

    def _evict(self) -> None:
        loaded = [e for e in self.entries.values() if e.img is not None]
        for e in sorted(loaded, key=lambda e: e.used)[: max(0, len(loaded) - MAX_LOADED)]:
            e.img = None


def _suggestion(an: Analysis, img: ImageData, family: str) -> dict[str, Any]:
    s = suggest(an.noise, img.meta, family)
    return {
        "strength": round(s.strength),
        "reason": s.reason,
        "message": s.message("de"),
        "message_en": s.message("en"),
    }


# ------------------------------------------------------------------------- events
class Events:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queues: set[asyncio.Queue[dict[str, Any]]] = set()

    def publish(self, event: dict[str, Any]) -> None:
        loop = self.loop
        if loop is None:
            return
        for q in list(self.queues):
            loop.call_soon_threadsafe(q.put_nowait, event)


def _jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    u8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise HTTPException(500, "JPEG-Kodierung fehlgeschlagen")
    return buf.tobytes()


# ------------------------------------------------------------------------- app
def create_app(
    token: str, device: str = "auto", backend_factory: Any = None, detector: Any = None
) -> FastAPI:
    gate = Gate()

    def gated_factory(model_id: str) -> InferenceBackend:
        if backend_factory is not None:
            inner = backend_factory(model_id)
        else:
            from pixforge.backends.onnx_backend import OnnxBackend

            inner = OnnxBackend(device)
            inner.load(model_id)
        return GatedBackend(inner, gate)

    pipeline = Pipeline(device=device, backend_factory=gated_factory, detector=detector)
    store = ImageStore(pipeline)
    events = Events()
    queue = JobQueue(pipeline, store.image, events.publish)
    preview_gen: dict[str, int] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        events.loop = asyncio.get_running_loop()
        yield

    app = FastAPI(title="Pixforge", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Preview-Ms"],
    )
    app.state.pipeline, app.state.store, app.state.queue, app.state.events = (
        pipeline,
        store,
        queue,
        events,
    )

    def auth(request: Request) -> None:
        got = request.headers.get("x-pixforge-token") or request.query_params.get("t")
        if got != token:
            raise HTTPException(401, "invalid token")

    dep = [Depends(auth)]

    @app.get("/api/health", dependencies=dep)
    def health() -> dict[str, Any]:
        from pixforge.backends.onnx_backend import list_devices

        devs = list_devices()
        chosen = (
            devs[0]
            if device == "auto"
            else next(
                (
                    d
                    for d in devs
                    if (d.kind == "cpu" and device == "cpu")
                    or (d.kind == "gpu" and device == f"gpu:{d.device_index}")
                ),
                devs[0],
            )
        )
        missing = [
            m.id
            for m in registry.all_models()
            if m.default and not registry.installed_precisions(m.id)
        ]
        return {
            "version": __version__,
            "device": chosen.__dict__,
            "devices": [d.__dict__ for d in devs],
            "missing_defaults": missing,
        }

    @app.get("/api/models", dependencies=dep)
    def models() -> list[dict[str, Any]]:
        out = []
        for m in registry.all_models():
            out.append(
                {
                    "id": m.id,
                    "kind": m.kind,
                    "name": m.name,
                    "description": m.description,
                    "scale": m.scale,
                    "license": m.license,
                    "noncommercial": m.noncommercial,
                    "default": m.default,
                    "installed": registry.installed_precisions(m.id),
                    "size": {p: f.size for p, f in m.files.items()},
                }
            )
        return out

    @app.post("/api/images", dependencies=dep)
    async def add_images(body: AddImagesIn) -> dict[str, Any]:
        added, errors = [], []
        for path in body.paths:
            try:
                e = await run_in_threadpool(store.add, path)
                added.append(e.info)
            except ImageLoadError as exc:
                errors.append({"path": path, "error": str(exc)})
            except Exception as exc:  # pragma: no cover - unexpected decoder failures
                log.exception("loading %s failed", path)
                errors.append({"path": path, "error": f"{exc.__class__.__name__}: {exc}"})
        return {"images": added, "errors": errors}

    @app.get("/api/images", dependencies=dep)
    def list_images() -> list[dict[str, Any]]:
        return [e.info for e in store.entries.values()]

    @app.delete("/api/images/{eid}", dependencies=dep)
    def remove_image(eid: str) -> dict[str, bool]:
        store.remove(eid)
        return {"ok": True}

    @app.get("/api/images/{eid}/thumb", dependencies=dep)
    def thumb(eid: str) -> Response:
        return Response(store.get(eid).thumb, media_type="image/jpeg")

    @app.get("/api/images/{eid}/view", dependencies=dep)
    async def view(eid: str, x: float, y: float, w: float, h: float, ow: int, oh: int) -> Response:
        img, _ = await run_in_threadpool(store.image, eid)
        rgb = await run_in_threadpool(pv.view_original, img, pv.Region(x, y, w, h), ow, oh)
        return Response(_jpeg(rgb, 92), media_type="image/jpeg")

    @app.post("/api/preview", dependencies=dep)
    async def preview(body: PreviewIn) -> Response:
        gen = preview_gen.get(body.image_id, 0) + 1
        preview_gen[body.image_id] = gen  # newer request for this image cancels older ones
        img, an = await run_in_threadpool(store.image, body.image_id)
        settings = body.settings.to_settings()
        region = pv.Region(**body.region.model_dump())

        def work() -> np.ndarray:
            with gate.preview():
                return pv.render(
                    pipeline,
                    img,
                    an,
                    settings,
                    region,
                    body.out_w,
                    body.out_h,
                    body.stage,
                    cancel=lambda: preview_gen.get(body.image_id) != gen,
                )

        t = time.perf_counter()
        try:
            rgb = await run_in_threadpool(work)
        except Cancelled:
            return Response(status_code=204)
        except registry.ModelNotInstalled as exc:
            raise HTTPException(409, str(exc)) from exc
        ms = round((time.perf_counter() - t) * 1000)
        return Response(_jpeg(rgb, 90), media_type="image/jpeg", headers={"X-Preview-Ms": str(ms)})

    @app.post("/api/estimate", dependencies=dep)
    def estimate(body: EstimateIn) -> dict[str, Any]:
        e = store.get(body.image_id)
        s = body.settings.to_settings()
        w, h = e.info["width"], e.info["height"]
        rw, rh = (s.crop[2], s.crop[3]) if s.crop else (w, h)
        try:
            W, H, f = target_size(rw, rh, s)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        kind = health()["device"]["kind"]
        warnings = []
        ram = available_ram_mb()
        need_mb = W * H * 3 * 4 * 3 / 2**20
        if ram is not None and need_mb > ram * 0.7:
            warnings.append(
                f"Braucht etwa {need_mb / 1024:.1f} GB Arbeitsspeicher, frei sind "
                f"{ram / 1024:.1f} GB."
            )
        if f > 4 and s.task == "upscale":
            warnings.append("Ab 6× werden Details teilweise neu erzeugt.")
        return {
            "in_w": rw,
            "in_h": rh,
            "out_w": W,
            "out_h": H,
            "factor": f,
            "megapixels": round(W * H / 1e6, 1),
            "seconds": round(estimate_seconds(w, h, s, kind), 1),
            "warnings": warnings,
        }

    @app.post("/api/jobs", dependencies=dep)
    def submit(body: JobIn) -> dict[str, Any]:
        e = store.get(body.image_id)
        ex = body.export
        opts = ExportOptions(
            format=ex.format,
            quality=ex.quality,
            bit_depth=ex.bit_depth,
            compression=ex.compression,
            webp_lossless=ex.webp_lossless,
            color=ex.color,
            keep_exif=ex.keep_exif,
        )
        settings = body.settings.to_settings()
        pattern = ex.pattern
        if settings.crop and pattern == "{name}_pixforge":
            pattern = "{name}_pixforge_ausschnitt"
        target = ExportTarget(opts, ex.folder, pattern, ex.on_exists)
        return queue.submit(e.id, e.info["name"], settings, target).public()

    @app.get("/api/jobs", dependencies=dep)
    def jobs() -> dict[str, Any]:
        return {"jobs": queue.list(), "paused": queue.paused}

    @app.post("/api/jobs/{jid}/cancel", dependencies=dep)
    def cancel(jid: str) -> dict[str, bool]:
        return {"ok": queue.cancel(jid)}

    @app.post("/api/queue/pause", dependencies=dep)
    def pause(paused: bool = True) -> dict[str, bool]:
        queue.pause(paused)
        return {"paused": paused}

    @app.post("/api/queue/order", dependencies=dep)
    def order(ids: list[str]) -> dict[str, bool]:
        queue.reorder(ids)
        return {"ok": True}

    @app.websocket("/api/events")
    async def ws(websocket: WebSocket) -> None:
        if websocket.query_params.get("t") != token:
            await websocket.close(code=4401)
            return
        await websocket.accept()
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        events.queues.add(q)
        try:
            await websocket.send_json({"type": "hello", "version": __version__})
            while True:
                await websocket.send_json(await q.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            events.queues.discard(q)

    return app
