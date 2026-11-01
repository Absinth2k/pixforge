"""Tests with the real ONNX models. Skipped when the models (or, for the parity tests, the
original PyTorch weights) are not present on this machine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixforge.core.io import ImageData
from pixforge.models import registry

WEIGHTS = Path(__file__).resolve().parents[2] / "weights"


def _installed(model_id: str) -> bool:
    return bool(registry.installed_precisions(model_id))


def _photo(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack(
        [0.5 + 0.3 * np.sin(xx / 7), (xx + yy) / (h + w), ((xx // 9 + yy // 9) % 2) * 0.5 + 0.2], -1
    )
    return np.clip(img + np.random.default_rng(0).normal(0, 0.02, img.shape), 0, 1).astype(
        np.float32
    )


def test_registry_entries_are_consistent() -> None:
    ids = [m.id for m in registry.all_models()]
    assert len(ids) == len(set(ids))
    for kind in ("upscale", "denoise", "face", "detect"):
        assert registry.default_model(kind).kind == kind
    for m in registry.all_models():
        for p, f in m.files.items():
            assert f.file == m.file_name(p) and len(f.sha256) == 64 and f.size > 0


@pytest.mark.parametrize("model_id", ["photo", "clean-x2", "denoise", "jpeg"])
def test_model_runs_one_tile(model_id: str) -> None:
    if not _installed(model_id):
        pytest.skip(f"{model_id} not installed")
    from pixforge.backends.onnx_backend import OnnxBackend

    b = OnnxBackend()
    b.load(model_id)
    t = 64 if b.alignment <= 64 else b.alignment
    out = b.run_tile(_photo(t, t))
    assert out.shape == (t * b.scale, t * b.scale, 3)
    assert np.isfinite(out).all()


def test_pipeline_end_to_end_with_real_models() -> None:
    if not (_installed("photo") and _installed("denoise")):
        pytest.skip("models not installed")
    from pixforge.core.pipeline import DenoiseSettings, FaceSettings, Pipeline, Settings

    img = ImageData(rgb=_photo(80, 120))
    res = Pipeline().process(
        img,
        Settings(
            model="photo",
            scale=3,
            denoise=DenoiseSettings(mode="on", strength=50),
            faces=FaceSettings(enabled=False),
        ),
    )
    assert res.image.size == (360, 240)
    assert np.isfinite(res.image.rgb).all()
    assert res.image.rgb.min() >= 0.0 and res.image.rgb.max() <= 1.0


@pytest.mark.parametrize(
    ("model_id", "arch", "weights", "size"),
    [
        ("photo", "hat", "hat/Real_HAT_GAN_SRx4.pth", 64),
        ("denoise", "scunet", "denoise/scunet_color_real_psnr.pth", 128),
    ],
)
def test_onnx_matches_pytorch(model_id: str, arch: str, weights: str, size: int) -> None:
    torch = pytest.importorskip("torch")
    if not (WEIGHTS / weights).is_file() or not _installed(model_id):
        pytest.skip("weights or model missing")
    import onnxruntime as ort

    from pixforge.models.export_onnx import build_spec
    from pixforge.models.hat_mask import shifted_window_mask

    spec = build_spec(arch, WEIGHTS / weights, "HAT", 4)
    x = _photo(size, size).transpose(2, 0, 1)[None].copy()
    with torch.no_grad():
        ref = spec.model(torch.from_numpy(x)).numpy()
    feeds = {"input": x}
    if arch == "hat":
        feeds["attn_mask"] = shifted_window_mask(size, size)
    for precision, min_psnr in (("fp32", 90.0), ("fp16", 50.0)):
        sess = ort.InferenceSession(
            str(registry.model_path(model_id, precision)), providers=["CPUExecutionProvider"]
        )
        out = sess.run(None, feeds)[0]
        mse = float(np.mean((np.clip(ref, 0, 1) - np.clip(out, 0, 1)) ** 2))
        psnr = 10 * np.log10(1 / max(mse, 1e-20))
        assert psnr > min_psnr, f"{model_id} {precision}: {psnr:.1f} dB"


def test_face_detection_on_synthetic_has_no_false_positive() -> None:
    if not _installed("face-detect"):
        pytest.skip("face detector not installed")
    from pixforge.faces.detect import FaceDetector

    assert FaceDetector().detect(_photo(480, 640)) == []
