"""Preview rendering: only the visible region, only at the resolution the screen shows.

At "fit" zoom the visible region of a 12 MP photo would cost ~2 minutes at full quality. The
preview therefore downsamples the source region first so that the *output* matches the screen
resolution (models then run on a few hundred pixels). At 100 % / 200 % zoom the visible region
is small and the real source pixels are processed, so what you see is what you export.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from pixforge.core import postprocess as pp
from pixforge.core.io import ImageData
from pixforge.core.pipeline import Analysis, Pipeline, Settings, target_size
from pixforge.faces.detect import Face

STAGES = ("original", "denoised", "result")
#: real context around the visible region (export uses 32 px; 16 is enough on screen and
#: keeps a typical 100 % view inside a single tile)
PREVIEW_MARGIN = 16


@dataclass(frozen=True)
class Region:
    x: float
    y: float
    w: float
    h: float

    def clipped(self, width: int, height: int) -> tuple[int, int, int, int]:
        x0 = int(max(0, min(width - 1, np.floor(self.x))))
        y0 = int(max(0, min(height - 1, np.floor(self.y))))
        x1 = int(max(x0 + 1, min(width, np.ceil(self.x + self.w))))
        y1 = int(max(y0 + 1, min(height, np.ceil(self.y + self.h))))
        return x0, y0, x1 - x0, y1 - y0


def _native_scale(model_id: str, factor: float) -> int:
    from pixforge.core.pipeline import resolve_model
    from pixforge.models import registry

    try:
        return registry.get(resolve_model(model_id, factor)).scale
    except KeyError:
        return max(1, round(factor))


def view_original(img: ImageData, region: Region, out_w: int, out_h: int) -> np.ndarray:
    h, w = img.rgb.shape[:2]
    x, y, rw, rh = region.clipped(w, h)
    return pp.resize(np.ascontiguousarray(img.rgb[y : y + rh, x : x + rw]), out_w, out_h)


def render(
    pipe: Pipeline,
    img: ImageData,
    analysis: Analysis,
    settings: Settings,
    region: Region,
    out_w: int,
    out_h: int,
    stage: str = "result",
    cancel: Callable[[], bool] | None = None,
) -> np.ndarray:
    """The ``stage`` image of ``region`` (source px) rendered at out_w x out_h."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    if stage == "original":
        return view_original(img, region, out_w, out_h)
    H0, W0 = img.rgb.shape[:2]
    x, y, rw, rh = region.clipped(W0, H0)
    s = replace(settings, crop=None)
    if stage == "denoised":
        s = replace(
            s,
            task="denoise",
            sharpen=0.0,
            grain=0.0,
            jpeg_strength=s.jpeg_strength,
            faces=replace(s.faces, enabled=False),
        )
    _, _, factor = target_size(W0, H0, s)
    # Source scale so that the output of the region matches the screen (never above 1).
    s_src = min(1.0, out_w / (rw * factor), out_h / (rh * factor))
    if s_src < 1.0 and s.task == "upscale":
        # Zoomed out: run the model once at its native scale on a source downsampled so that
        # its output matches the screen. Running 4x and shrinking to 2x (or two passes for 6x)
        # would cost up to 4x more for pixels nobody can see at this zoom.
        native = _native_scale(s.model, factor)
        s_src = min(1.0, out_w / (rw * native), out_h / (rh * native))
        factor = float(native)
    m = PREVIEW_MARGIN
    ml, mt = min(m, x), min(m, y)
    mr, mb = min(m, W0 - x - rw), min(m, H0 - y - rh)
    X0, Y0 = x - ml, y - mt
    sub = np.ascontiguousarray(img.rgb[Y0 : y + rh + mb, X0 : x + rw + mr])
    sh, sw = sub.shape[:2]
    if s_src < 1.0:
        sw2, sh2 = max(16, round(sw * s_src)), max(16, round(sh * s_src))
        sub = pp.resize(sub, sw2, sh2)
        sx, sy = sw2 / sw, sh2 / sh
    else:
        sx = sy = 1.0
    faces = []
    for f in analysis.faces:
        cx, cy = f.box[0] + f.box[2] / 2, f.box[1] + f.box[3] / 2
        if X0 <= cx < X0 + sw and Y0 <= cy < Y0 + sh:
            box = ((f.box[0] - X0) * sx, (f.box[1] - Y0) * sy, f.box[2] * sx, f.box[3] * sy)
            lm = (f.landmarks - np.array([X0, Y0], np.float32)) * np.array([sx, sy], np.float32)
            faces.append(Face(box, lm, f.score))
    sub_an = Analysis(analysis.noise, analysis.denoise, faces, sub.shape[1], sub.shape[0])
    inner = replace(s, scale=factor, target_width=None, target_height=None)
    res = pipe.process(replace(img, rgb=sub, alpha=None), inner, cancel=cancel, analysis=sub_an)
    out = res.image.rgb
    # cut the context margin away (in output pixels) and fit to the requested size
    L, T = round(ml * sx * factor), round(mt * sy * factor)
    R, B = round(mr * sx * factor), round(mb * sy * factor)
    oh, ow = out.shape[:2]
    out = np.ascontiguousarray(out[T : oh - B if B else oh, L : ow - R if R else ow])
    return pp.resize(out, out_w, out_h)
