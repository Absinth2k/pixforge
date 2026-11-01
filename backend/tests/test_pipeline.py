"""Pipeline logic with fake backends (no models, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest
from fakes import IdentityFaceBackend, NearestBackend

from pixforge.core.io import ImageData
from pixforge.core.pipeline import (
    DenoiseSettings,
    FaceSettings,
    Pipeline,
    Settings,
    auto_tile,
    resolve_model,
    target_size,
)
from pixforge.denoise.estimate import NoiseEstimate
from pixforge.denoise.models import suggest
from pixforge.faces.detect import Face
from pixforge.faces.restore import FFHQ_512


class NoFaces:
    def detect(self, rgb: np.ndarray, min_size: float = 0) -> list[Face]:
        return []


def _pipe(scale: int = 4, oom_above: int | None = None, faces: object = None) -> Pipeline:
    backends = {}

    def factory(mid: str) -> object:
        if mid.startswith("face"):
            b = IdentityFaceBackend()
        else:
            b = NearestBackend(
                scale=1 if mid in ("denoise", "jpeg") else scale,
                alignment=16,
                edge=2,
                oom_above=oom_above,
            )
        backends[mid] = b
        return b

    p = Pipeline(backend_factory=factory, detector=faces or NoFaces())  # type: ignore[arg-type]
    p.fakes = backends  # type: ignore[attr-defined]
    return p


def _image(h: int = 96, w: int = 128, alpha: bool = False) -> ImageData:
    rgb = np.random.default_rng(1).random((h, w, 3), dtype=np.float32) * 0.5 + 0.25
    a = np.linspace(0, 1, h * w, dtype=np.float32).reshape(h, w) if alpha else None
    return ImageData(rgb=rgb, alpha=a)


def test_target_size_variants() -> None:
    assert target_size(100, 50, Settings(scale=4)) == (400, 200, 4.0)
    assert target_size(100, 50, Settings(target_width=250))[:2] == (250, 125)
    assert target_size(100, 50, Settings(target_height=300))[:2] == (600, 300)
    w, h, f = target_size(100, 50, Settings(target_width=1000, target_height=300))
    assert (w, h) == (600, 300) and f == 6.0
    with pytest.raises(ValueError):
        target_size(100, 50, Settings(scale=20))


def test_family_resolution() -> None:
    assert resolve_model("clean-x4", 2) == "clean-x2"
    assert resolve_model("clean-x4", 2.5) == "clean-x3"
    assert resolve_model("clean-x2", 6) == "clean-x4"
    assert resolve_model("max-x4", 1.5) == "max-x2"
    assert resolve_model("photo", 2) == "photo"


@pytest.mark.parametrize("scale", [1.0, 2.0, 4.0, 6.0])
def test_output_size_and_passes(scale: float) -> None:
    p = _pipe()
    img = _image()
    res = p.process(
        img,
        Settings(
            scale=scale, denoise=DenoiseSettings(mode="off"), faces=FaceSettings(enabled=False)
        ),
    )
    assert res.image.size == (round(128 * scale), round(96 * scale))
    calls = p.fakes["photo"].calls  # type: ignore[attr-defined]
    two_pass = scale > 4
    first_area = 96 * 128
    assert (sum(h * w for h, w in calls) > first_area * 1.5) == two_pass


def test_alpha_is_scaled_and_kept() -> None:
    res = _pipe().process(
        _image(alpha=True),
        Settings(scale=2, denoise=DenoiseSettings(mode="off"), faces=FaceSettings(enabled=False)),
    )
    assert res.image.alpha is not None and res.image.alpha.shape == (192, 256)


def test_out_of_memory_falls_back_to_smaller_tiles() -> None:
    p = _pipe(oom_above=200)
    res = p.process(
        _image(400, 400),
        Settings(
            scale=2,
            tile=384,
            denoise=DenoiseSettings(mode="off"),
            faces=FaceSettings(enabled=False),
        ),
    )
    assert res.image.size == (800, 800)
    assert any("kleinere Kacheln" in w for w in res.warnings)


def test_progress_is_monotonic_and_complete() -> None:
    seen: list[float] = []
    _pipe().process(
        _image(),
        Settings(
            scale=2,
            denoise=DenoiseSettings(mode="on", strength=50),
            faces=FaceSettings(enabled=False),
        ),
        progress=lambda s, f, m: seen.append(f),
    )
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)


def test_auto_tile_respects_vram() -> None:
    from pixforge.backends.base import DeviceInfo

    class Card(NearestBackend):
        def __init__(self, vram: int, precision: str) -> None:
            super().__init__()
            self.vram, self.precision = vram, precision

        def device_info(self) -> DeviceInfo:
            return DeviceInfo("x", "gpu", "fake", vram_mb=self.vram)

    assert auto_tile(Card(16384, "mixed-fp16")) == 384
    assert auto_tile(Card(8192, "mixed-fp16")) == 256
    assert auto_tile(Card(4096, "mixed-fp16")) == 192
    assert auto_tile(Card(8192, "fp32")) == 192


def test_faces_are_pasted_into_the_face_region_only() -> None:
    lm = FFHQ_512 / 4 + np.array([10, 10], np.float32)  # a 128 px face at (10, 10)
    face = Face((10, 10, 128, 128), lm, 0.99)

    class OneFace:
        def detect(self, rgb: np.ndarray, min_size: float = 0) -> list[Face]:
            return [face]

    img = _image(200, 200)
    img.rgb[:] = 0.0
    res = _pipe(faces=OneFace()).process(
        img,
        Settings(
            scale=2,
            denoise=DenoiseSettings(mode="off"),
            faces=FaceSettings(enabled=True, strength=100),
        ),
    )
    out = res.image.rgb
    assert res.faces_restored == 1
    cy, cx = int((10 + 70) * 2), int((10 + 64) * 2)  # inside the face
    assert out[cy, cx].mean() > 0.9
    assert out[380, 380].mean() < 0.05  # far away: untouched


def test_denoise_suggestions() -> None:
    dxo = suggest(NoiseEstimate(5.7, 0.16, 0.3))
    assert dxo.reason == "already_denoised" and dxo.strength == 0
    clean = suggest(NoiseEstimate(1.0, 0.7, 0.3))
    assert clean.reason == "clean" and clean.strength == 0
    noisy = suggest(NoiseEstimate(5.0, 2.8, 0.3), upscale_model="clean-x4")
    assert noisy.reason == "noisy" and noisy.strength == pytest.approx(84.0)
    light = suggest(NoiseEstimate(2.0, 1.2, 0.3), upscale_model="clean-x4")
    assert light.strength == pytest.approx(60.0)
    gan = suggest(NoiseEstimate(5.0, 2.8, 0.3), upscale_model="photo-sharp")
    assert gan.reason == "model_handles" and gan.strength == 0
    # Black-and-white scan: no colour noise by nature, must not count as "already denoised".
    bw = suggest(NoiseEstimate(4.0, 0.05, 0.3, colorfulness=0.3), upscale_model="clean-x4")
    assert bw.reason == "noisy"
