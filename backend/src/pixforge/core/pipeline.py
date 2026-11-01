"""The Pixforge processing pipeline.

    analyse  ->  JPEG repair  ->  denoise  ->  upscale (1-2 passes)  ->  detail mix
             ->  faces  ->  sharpen  ->  grain  ->  resize to target

All steps are optional except the upscale. Every heavy step runs tiled through an
InferenceBackend, so the same code serves the CLI, the server and the tests (with a fake
backend).
"""

from __future__ import annotations

import ctypes
import gc
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from pixforge.backends.base import InferenceBackend, OutOfMemory
from pixforge.core import postprocess as pp
from pixforge.core.io import ImageData
from pixforge.core.tiling import Cancelled, run_tiled
from pixforge.denoise import blend as dn_blend
from pixforge.denoise.estimate import NoiseEstimate, estimate_noise
from pixforge.denoise.models import DenoiseSuggestion, suggest
from pixforge.faces.detect import Face, FaceDetector
from pixforge.faces.restore import FaceRestorer

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float, str], None]  # stage, overall 0..1, message
BackendFactory = Callable[[str], InferenceBackend]

TILE_SIZES = (384, 256, 192, 128)
MAX_FACTOR = 16.0
#: real image context (source px) processed around a crop and cut away afterwards
CROP_MARGIN = 32
#: SCUNet seconds per MP (GPU measured, CPU estimated)
DENOISE_S_PER_MP = {"gpu": 0.4, "cpu": 10.0}
# Measured on an RX 9070 XT / Core Ultra 9 285K (docs/backend.md), seconds per input MP.
SPEED_S_PER_MP = {
    ("gpu", "HAT"): 10.0,
    ("gpu", "HAT-L"): 19.5,
    ("cpu", "HAT"): 300.0,
    ("cpu", "HAT-L"): 600.0,
}


# ----------------------------------------------------------------------------- settings
@dataclass
class DenoiseSettings:
    mode: str = "auto"  # auto | on | off
    strength: float | None = None  # 0..100; None = automatic suggestion
    model: str = "denoise"
    detail_protect: float = 60.0  # 0..100


@dataclass
class FaceSettings:
    enabled: bool | None = None  # None = on if faces are found
    model: str = "face"
    strength: float = 80.0  # 0..100
    min_size: float = 16.0  # px in the source image
    only_largest: bool = False
    exclude: tuple[int, ...] = ()


@dataclass
class Settings:
    #: "upscale" (default) or "denoise" (same size, noise reduction only, no upscaling model)
    task: str = "upscale"
    model: str = "photo"
    scale: float = 4.0
    target_width: int | None = None
    target_height: int | None = None
    denoise: DenoiseSettings = field(default_factory=DenoiseSettings)
    jpeg_strength: float = 0.0  # FBCNN, 0 = off
    sharpen: float = 0.0
    detail_mix: float = 0.0
    grain: float = 0.0
    faces: FaceSettings = field(default_factory=FaceSettings)
    tile: int | None = None  # None = from device memory
    overlap: int = 32
    keep_intermediates: bool = False
    #: (x, y, width, height) in source pixels: process only this region as a new image
    crop: tuple[int, int, int, int] | None = None


@dataclass
class Analysis:
    noise: NoiseEstimate
    denoise: DenoiseSuggestion
    faces: list[Face]
    width: int
    height: int


@dataclass
class Result:
    image: ImageData
    analysis: Analysis
    model: str
    factor: float
    device: str
    seconds: float
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    faces_restored: int = 0
    denoise_strength: float = 0.0
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)


# ----------------------------------------------------------------------------- helpers
def _model_kind(model_id: str) -> str:
    from pixforge.models import registry

    try:
        return registry.get(model_id).kind
    except KeyError:
        return model_id  # unknown ids (tests, file paths) are their own kind


def target_size(w: int, h: int, s: Settings) -> tuple[int, int, float]:
    """Output size and effective factor from scale or target width/height.

    ``w``/``h`` are the size of the processed region (the crop, if one is set).
    """
    if s.task == "denoise":
        return w, h, 1.0
    if s.target_width and s.target_height:
        f = min(s.target_width / w, s.target_height / h)
    elif s.target_width:
        f = s.target_width / w
    elif s.target_height:
        f = s.target_height / h
    else:
        f = float(s.scale)
    if f <= 0:
        raise ValueError("Skalierung muss größer als 0 sein")
    if f > MAX_FACTOR:
        raise ValueError(f"Höchstens {MAX_FACTOR:g}× möglich")
    return max(1, round(w * f)), max(1, round(h * f)), f


def resolve_model(model_id: str, factor: float) -> str:
    """For families with several native scales (clean-x2/3/4, max-x2/4) pick the smallest
    native scale that reaches the factor in one pass (else the largest)."""
    from pixforge.models import registry

    m = re.fullmatch(r"(.+)-x(\d)", model_id)
    if not m:
        return model_id
    family = m.group(1)
    members = sorted(
        (info.scale, info.id)
        for info in registry.all_models()
        if re.fullmatch(rf"{re.escape(family)}-x\d", info.id)
    )
    for scale, mid in members:
        if scale >= factor - 1e-6:
            return mid
    return members[-1][1]


def available_ram_mb() -> int | None:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
            return int(st.ullAvailPhys // 2**20)
    except (AttributeError, OSError):
        pass
    return None


def auto_tile(backend: InferenceBackend) -> int:
    """Largest tile whose estimated VRAM need stays within half of the card's memory."""
    info = backend.device_info()
    if info.kind != "gpu":
        return 192
    fp16 = getattr(backend, "precision", "fp32") != "fp32"
    base_mb, kb_per_px = (500.0, 38.0) if fp16 else (900.0, 56.0)
    budget = (info.vram_mb or 4096) * 0.5
    for t in TILE_SIZES:
        if base_mb + kb_per_px * t * t / 1024.0 <= budget:
            return t
    return TILE_SIZES[-1]


def estimate_seconds(w: int, h: int, settings: Settings, device_kind: str) -> float:
    """Rough processing time for the UI (upscale dominates)."""
    if settings.crop:
        w, h = settings.crop[2], settings.crop[3]
    mp = w * h / 1e6
    if settings.task == "denoise":
        return mp * DENOISE_S_PER_MP[device_kind]
    _, _, f = target_size(w, h, settings)
    arch = "HAT-L" if settings.model.startswith("max") else "HAT"
    per_mp = SPEED_S_PER_MP[(device_kind, arch)]
    passes = 1.0 if f <= 4 else 1.0 + (f / 4) ** 2
    t = mp * per_mp * passes
    if settings.denoise.mode != "off":
        t += mp * (0.4 if device_kind == "gpu" else 12.0)
    return t


# ----------------------------------------------------------------------------- pipeline
class Pipeline:
    def __init__(
        self,
        device: str = "auto",
        backend_factory: BackendFactory | None = None,
        detector: FaceDetector | None = None,
    ) -> None:
        self.device = device
        self._factory = backend_factory or self._onnx_backend
        self._backends: dict[str, InferenceBackend] = {}
        self._detector = detector
        #: called with (model_id, loading) around model loads, e.g. to tell the UI
        self.on_model_load: Callable[[str, bool], None] | None = None

    def _onnx_backend(self, model_id: str) -> InferenceBackend:
        from pixforge.backends.onnx_backend import OnnxBackend

        b = OnnxBackend(self.device)
        b.load(model_id)
        return b

    def backend(self, model_id: str) -> InferenceBackend:
        if model_id not in self._backends:
            # One model per kind (upscale, denoise, face, ...) in VRAM: switching from "photo"
            # to "max-x4" frees "photo" first instead of stacking models until the card spills.
            kind = _model_kind(model_id)
            for other in [m for m in self._backends if _model_kind(m) == kind]:
                log.info("unloading model %s", other)
                self._backends.pop(other).close()
            gc.collect()
            log.info("loading model %s", model_id)
            if self.on_model_load:
                self.on_model_load(model_id, True)
            try:
                self._backends[model_id] = self._factory(model_id)
            finally:
                if self.on_model_load:
                    self.on_model_load(model_id, False)
        return self._backends[model_id]

    def release(self, keep: set[str] | None = None) -> None:
        for mid in list(self._backends):
            if keep is None or mid not in keep:
                self._backends.pop(mid).close()

    @property
    def detector(self) -> FaceDetector:
        if self._detector is None:
            self._detector = FaceDetector()
        return self._detector

    # -- analysis (cheap, done when an image is added) --------------------------------
    def analyze(
        self, img: ImageData, settings: Settings | None = None, detect_faces: bool = True
    ) -> Analysis:
        settings = settings or Settings()
        noise = estimate_noise(img.rgb)
        # In denoise-only mode no photo model runs that could remove noise by itself.
        model = "clean" if settings.task == "denoise" else settings.model
        sug = suggest(noise, img.meta, model)
        faces: list[Face] = []
        if detect_faces:
            try:
                faces = self.detector.detect(img.rgb, min_size=0)
            except Exception as exc:  # detection must never break the pipeline
                log.warning("face detection failed: %s", exc)
        h, w = img.rgb.shape[:2]
        return Analysis(noise, sug, faces, w, h)

    # -- processing --------------------------------------------------------------------
    def process(
        self,
        img: ImageData,
        settings: Settings | None = None,
        progress: ProgressFn | None = None,
        cancel: Callable[[], bool] | None = None,
        analysis: Analysis | None = None,
    ) -> Result:
        s = settings or Settings()
        if s.crop is not None:
            return self._process_crop(img, s, progress, cancel, analysis)
        t_start = time.perf_counter()
        h, w = img.rgb.shape[:2]
        W, H, factor = target_size(w, h, s)
        denoise_only = s.task == "denoise"
        model_id = "denoise" if denoise_only else resolve_model(s.model, factor)
        warnings: list[str] = []
        timings: dict[str, float] = {}
        inter: dict[str, np.ndarray] = {}

        out_mb = W * H * 3 * 4 * 3 / 2**20
        ram = available_ram_mb()
        if ram is not None and out_mb > ram * 0.7:
            warnings.append(
                f"Sehr große Ausgabe ({W}×{H}): braucht etwa {out_mb / 1024:.1f} GB "
                f"Arbeitsspeicher, frei sind {ram / 1024:.1f} GB."
            )

        an = analysis or self.analyze(img, s, detect_faces=s.faces.enabled is not False)
        # The suggestion depends on the model (photo models remove noise themselves); an
        # analysis cached by the server was made once, so re-derive it for these settings.
        an = replace(an, denoise=suggest(an.noise, img.meta, "clean" if denoise_only else s.model))
        faces = self._select_faces(an.faces, s.faces)
        want_faces = bool(faces) and s.faces.enabled is not False
        den_strength = self._denoise_strength(s, an)
        if denoise_only and den_strength == 0 and s.denoise.mode != "off":
            warnings.append(
                f"{an.denoise.message()}: an der Rauschminderung gibt es nichts zu tun."
            )

        stages = [
            ("jpeg", 0.08 if s.jpeg_strength > 0 else 0.0),
            ("denoise", 0.08 if den_strength > 0 else 0.0),
            ("upscale", 0.0 if denoise_only else 0.76 if factor <= 4 else 0.86),
            ("faces", 0.06 if want_faces else 0.0),
            ("finish", 0.02),
        ]
        total = sum(v for _, v in stages)
        offsets: dict[str, tuple[float, float]] = {}
        acc = 0.0
        for name, v in stages:
            offsets[name] = (acc / total, v / total)
            acc += v

        def report(stage: str, frac: float, msg: str = "") -> None:
            if progress:
                o, span = offsets[stage]
                progress(stage, min(1.0, o + span * frac), msg)

        def check_cancel() -> None:
            if cancel and cancel():
                raise Cancelled

        rgb = img.rgb

        # 1. JPEG artefacts
        if s.jpeg_strength > 0:
            t = time.perf_counter()
            rgb = self._jpeg(rgb, s, lambda d, n: report("jpeg", d / n), cancel, warnings)
            timings["jpeg"] = time.perf_counter() - t

        # 2. Denoise
        if den_strength > 0:
            t = time.perf_counter()
            b = self.backend(s.denoise.model)
            den = self._tiled(b, rgb, s, lambda d, n: report("denoise", d / n), cancel, warnings)
            rgb = dn_blend.blend(
                rgb, den, den_strength, an.noise.luma_sigma, s.denoise.detail_protect / 100.0
            )
            if s.keep_intermediates:
                inter["denoised"] = rgb
            del den
            timings["denoise"] = time.perf_counter() - t
        check_cancel()

        # 3. Upscale
        t = time.perf_counter()
        device_name = ""
        if denoise_only:
            up = np.array(rgb, dtype=np.float32, copy=True)
            if den_strength > 0:
                device_name = self.backend(s.denoise.model).device_info().name
        else:
            up = self._upscale(rgb, s, model_id, factor, W, H, report, cancel, warnings)
            device_name = self.backend(model_id).device_info().name
        timings["upscale"] = time.perf_counter() - t
        check_cancel()

        # 4. Detail mix with a plain Lanczos upscale
        if s.detail_mix > 0 and not denoise_only:
            up = pp.detail_mix(up, pp.resize(rgb, W, H), s.detail_mix)

        # 5. Faces
        restored = 0
        if want_faces:
            t = time.perf_counter()
            fr = FaceRestorer(self.backend(s.faces.model))
            fx, fy = W / w, H / h
            scaled = [
                Face(
                    (f.box[0] * fx, f.box[1] * fy, f.box[2] * fx, f.box[3] * fy),
                    f.landmarks * np.array([fx, fy], np.float32),
                    f.score,
                )
                for f in faces
            ]
            up = np.ascontiguousarray(up)
            up = fr.restore(
                up, scaled, s.faces.strength, progress=lambda d, n: report("faces", d / n)
            )
            restored = len(scaled)
            device_name = device_name or self.backend(s.faces.model).device_info().name
            timings["faces"] = time.perf_counter() - t
        check_cancel()

        # 6. Sharpen, grain, alpha
        report("finish", 0.0, "")
        up = pp.sharpen(up, s.sharpen)
        up = pp.add_grain(up, s.grain, size=max(1.0, factor / 2), seed=w * 31 + h)
        up = pp.clip01(up.astype(np.float32, copy=False))
        alpha = None
        if img.alpha is not None:
            alpha = np.clip(pp.resize(img.alpha, W, H), 0.0, 1.0)
        report("finish", 1.0, "")

        meta = dict(img.meta)
        meta["pixforge"] = {
            "model": model_id,
            "factor": round(factor, 4),
            "denoise": round(den_strength, 1),
            "faces": restored,
        }
        out = replace(img, rgb=up, alpha=alpha, meta=meta)
        return Result(
            out,
            an,
            model_id,
            factor,
            device_name,
            time.perf_counter() - t_start,
            timings,
            warnings,
            restored,
            den_strength,
            inter,
        )

    # -- steps -------------------------------------------------------------------------
    def _upscale(
        self,
        rgb: np.ndarray,
        s: Settings,
        model_id: str,
        factor: float,
        W: int,
        H: int,
        report: Callable[..., None],
        cancel: Callable[[], bool] | None,
        warnings: list[str],
    ) -> np.ndarray:
        b = self.backend(model_id)
        native = b.scale
        if factor <= native + 1e-6:
            up = self._tiled(b, rgb, s, lambda d, n: report("upscale", d / n), cancel, warnings)
        else:
            mid_w, mid_h = round(W / native), round(H / native)
            first = self._tiled(
                b, rgb, s, lambda d, n: report("upscale", 0.2 * d / n), cancel, warnings
            )
            first = pp.resize(first, mid_w, mid_h)
            up = self._tiled(
                b, first, s, lambda d, n: report("upscale", 0.2 + 0.8 * d / n), cancel, warnings
            )
            del first
        return pp.resize(up, W, H)

    def _process_crop(
        self,
        img: ImageData,
        s: Settings,
        progress: ProgressFn | None,
        cancel: Callable[[], bool] | None,
        analysis: Analysis | None,
    ) -> Result:
        """Process only ``s.crop`` plus CROP_MARGIN of real context, cut away afterwards."""
        H0, W0 = img.rgb.shape[:2]
        x, y, cw, ch = (int(round(v)) for v in s.crop)  # type: ignore[union-attr]
        x, y = max(0, min(x, W0 - 1)), max(0, min(y, H0 - 1))
        cw, ch = max(1, min(cw, W0 - x)), max(1, min(ch, H0 - y))
        ml, mt = min(CROP_MARGIN, x), min(CROP_MARGIN, y)
        mr, mb = min(CROP_MARGIN, W0 - x - cw), min(CROP_MARGIN, H0 - y - ch)
        X0, Y0 = x - ml, y - mt
        region = (slice(Y0, y + ch + mb), slice(X0, x + cw + mr))
        sub_alpha = None if img.alpha is None else np.ascontiguousarray(img.alpha[region])
        sub_img = replace(img, rgb=np.ascontiguousarray(img.rgb[region]), alpha=sub_alpha)
        W, H, factor = target_size(cw, ch, replace(s, crop=None))
        an_full = analysis or self.analyze(img, s, detect_faces=s.faces.enabled is not False)
        off = np.array([X0, Y0], np.float32)
        faces = []
        for f in an_full.faces:  # faces centred inside the crop, in region coordinates
            fx, fy = f.box[0] + f.box[2] / 2, f.box[1] + f.box[3] / 2
            if x <= fx < x + cw and y <= fy < y + ch:
                box = (f.box[0] - X0, f.box[1] - Y0, f.box[2], f.box[3])
                faces.append(Face(box, f.landmarks - off, f.score))
        sh, sw = sub_img.rgb.shape[:2]
        an = Analysis(an_full.noise, an_full.denoise, faces, sw, sh)
        inner = replace(s, crop=None, scale=factor, target_width=None, target_height=None)
        res = self.process(sub_img, inner, progress, cancel, an)
        L, T = round(ml * factor), round(mt * factor)

        def cut(a: np.ndarray) -> np.ndarray:
            a = np.ascontiguousarray(a[T : T + H, L : L + W])
            return pp.resize(a, W, H) if a.shape[:2] != (H, W) else a

        meta = dict(res.image.meta)
        meta["pixforge"] = dict(meta.get("pixforge", {}), crop=[x, y, cw, ch])
        alpha = None if res.image.alpha is None else cut(res.image.alpha)
        res.image = replace(res.image, rgb=cut(res.image.rgb), alpha=alpha, meta=meta)
        res.analysis = an_full
        return res

    def _denoise_strength(self, s: Settings, an: Analysis) -> float:
        mode = s.denoise.mode
        if mode == "off":
            return 0.0
        if s.denoise.strength is not None:
            return float(np.clip(s.denoise.strength, 0, 100))
        if s.task == "denoise" and an.denoise.reason == "model_handles":
            # the analysis was made for a photo model, but none runs in denoise-only mode
            return float(suggest(an.noise, None, "clean").strength)
        if mode == "on" and an.denoise.strength == 0:
            if an.denoise.reason == "model_handles":  # user overrides the photo-model default
                return float(suggest(an.noise, None, "clean").strength)
            return 50.0
        return an.denoise.strength

    @staticmethod
    def _select_faces(faces: list[Face], fs: FaceSettings) -> list[Face]:
        if fs.enabled is False:
            return []
        sel = [f for i, f in enumerate(faces) if f.size >= fs.min_size and i not in fs.exclude]
        return sel[:1] if fs.only_largest else sel

    def _tiled(
        self,
        b: InferenceBackend,
        rgb: np.ndarray,
        s: Settings,
        progress: Callable[[int, int], None],
        cancel: Callable[[], bool] | None,
        warnings: list[str],
    ) -> np.ndarray:
        tile = s.tile or auto_tile(b)
        tile = max(b.alignment, tile - tile % b.alignment) if b.alignment > 1 else tile
        while True:
            try:
                return run_tiled(
                    rgb, b.run_tile, b.scale, tile, s.overlap, b.alignment, progress, cancel
                )
            except OutOfMemory:
                smaller = [t for t in TILE_SIZES if t < tile and t % max(b.alignment, 1) == 0]
                if not smaller:
                    raise
                tile = smaller[0]
                msg = f"Grafikkarte hat zu wenig Speicher, ich nehme kleinere Kacheln ({tile} px)."
                log.warning(msg)
                if msg not in warnings:
                    warnings.append(msg)

    def _jpeg(
        self,
        rgb: np.ndarray,
        s: Settings,
        progress: Callable[[int, int], None],
        cancel: Callable[[], bool] | None,
        warnings: list[str],
    ) -> np.ndarray:
        b = self.backend("jpeg")
        # One quality estimate for the whole image (per-tile estimates would cause seams).
        h, w = rgb.shape[:2]
        t = min(256, (min(h, w) // 8) * 8)
        if t >= 64 and hasattr(b, "jpeg_qf"):
            qfs = []
            for cy, cx in ((0.3, 0.3), (0.3, 0.7), (0.7, 0.3), (0.7, 0.7)):
                y0 = int(np.clip(cy * h - t / 2, 0, h - t))
                x0 = int(np.clip(cx * w - t / 2, 0, w - t))
                b.jpeg_qf = -1.0
                b.run_tile(np.ascontiguousarray(rgb[y0 : y0 + t, x0 : x0 + t]))
                qfs.append(float(np.asarray(b.last_extra_outputs[0]).ravel()[0]))
            b.jpeg_qf = float(np.mean(qfs))
        fixed = self._tiled(b, rgb, s, progress, cancel, warnings)
        a = float(np.clip(s.jpeg_strength / 100.0, 0, 1))
        return rgb * (1 - a) + fixed * a
