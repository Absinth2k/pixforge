"""Verify an exported HAT ONNX model against the PyTorch reference.

Run from backend/:  .venv/Scripts/python ../bench/verify_hat_onnx.py <weights.pth> <model.onnx>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from pixforge.models.hat_arch import HAT, hat_config, load_hat_state_dict
from pixforge.models.hat_mask import shifted_window_mask


def synthetic_photo(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Smooth gradients + edges + texture + mild noise, float32 RGB 0..1, NCHW."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 0.5 + 0.4 * np.sin(xx / 9.0) * np.cos(yy / 13.0)
    g = (xx + yy) / (h + w)
    b = ((xx // 12 + yy // 12) % 2).astype(np.float32) * 0.6 + 0.2
    img = np.stack([r, g, b])
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)[None].astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.clip(a, 0, 1)
    b = np.clip(b, 0, 1)
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--variant", default="HAT")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--sizes", default="64x64,128x192,256x256")
    args = ap.parse_args()

    model = HAT(hat_config(args.variant, args.scale)).eval()
    load_hat_state_dict(model, torch.load(args.weights, map_location="cpu", weights_only=True))

    sessions = {
        "cpu": ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"]),
        "dml": ort.InferenceSession(
            str(args.onnx), providers=["DmlExecutionProvider", "CPUExecutionProvider"]
        ),
    }
    ok = True
    for spec in args.sizes.split(","):
        h, w = (int(v) for v in spec.split("x"))
        x = synthetic_photo(h, w)
        mask = shifted_window_mask(h, w)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        for name, sess in sessions.items():
            t = time.perf_counter()
            out = sess.run(None, {"input": x, "attn_mask": mask})[0]
            dt = time.perf_counter() - t
            p = psnr(ref, out)
            ok &= p > 50
            print(f"{spec:>9} {name}: PSNR {p:6.2f} dB  max|d| {np.abs(ref - out).max():.2e}  "
                  f"{dt * 1000:7.0f} ms (first run)")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
