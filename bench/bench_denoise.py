"""Denoiser comparison: SCUNet vs NAFNet vs OpenCV fastNlMeans (and none).

Part A (real photos, no ground truth): noise estimate before/after, detail retention in
textured areas, timing, plus side-by-side crops for visual judgement.
Part B (synthetic): photos downscaled 4x (nearly noise-free) + signal-dependent noise at
three levels, scored with PSNR / SSIM against the clean version.

Run from backend/:
  .venv/Scripts/python ../bench/bench_denoise.py --photos C:/path/to/raws --models-dir ../bench/out
Inputs stay local; outputs go to bench/out/denoise (gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

from pixforge.backends.onnx_backend import OnnxBackend
from pixforge.denoise.estimate import estimate_noise

RAW_EXT = {".cr3", ".cr2", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
CROP = 768  # multiple of 64 (SCUNet)


# ---------------------------------------------------------------- loading
def load_photo(path: Path, cache: Path) -> np.ndarray:
    """RGB float32 0..1. RAWs are developed once with LibRaw defaults and cached."""
    cached = cache / (path.stem + ".png")
    if cached.exists():
        img = cv2.imread(str(cached), cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
    if path.suffix.lower() in RAW_EXT:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            rgb16 = raw.postprocess(use_camera_wb=True, output_bps=16)
        cache.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cached), cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR))
        return rgb16.astype(np.float32) / 65535.0
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / scale


def pick_crops(img: np.ndarray) -> dict[str, tuple[int, int]]:
    """Top-left corners of a flat, a textured and a central crop."""
    h, w = img.shape[:2]
    y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(y, (w // 16, h // 16), interpolation=cv2.INTER_AREA)
    act = cv2.blur(np.abs(cv2.Laplacian(small, cv2.CV_32F)), (CROP // 16, CROP // 16))
    bright = cv2.blur(small, (CROP // 16, CROP // 16))
    m = CROP // 32
    inner = np.zeros_like(act, bool)
    inner[m:-m, m:-m] = True
    ok = inner & (bright > 0.08) & (bright < 0.92)
    flat = np.unravel_index(np.where(ok, act, np.inf).argmin(), act.shape)
    tex = np.unravel_index(np.where(ok, act, -np.inf).argmax(), act.shape)

    def corner(cy: int, cx: int) -> tuple[int, int]:
        y0 = int(np.clip(cy * 16 - CROP // 2, 0, h - CROP))
        x0 = int(np.clip(cx * 16 - CROP // 2, 0, w - CROP))
        return y0, x0

    return {"flat": corner(*flat), "texture": corner(*tex), "center": corner(h // 32, w // 32)}


# ---------------------------------------------------------------- methods
def nlmeans(img: np.ndarray) -> np.ndarray:
    est = estimate_noise(img)
    # OpenCV's h is in 8-bit RGB units; luma sigma * 1.5 ~ RGB sigma, mild strength.
    h = max(2.0, est.luma_sigma * 1.5)
    hc = max(2.0, est.chroma_sigma * 1.5)
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    out = cv2.fastNlMeansDenoisingColored(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), None, h, hc, 7, 21)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


class ModelMethod:
    def __init__(self, path: Path) -> None:
        self.backend = OnnxBackend()
        self.backend.load(path)
        self.name = path.stem

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.clip(self.backend.run_tile(img), 0, 1)


# ---------------------------------------------------------------- metrics
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim_y(a: np.ndarray, b: np.ndarray) -> float:
    ya = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    yb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = 0.01**2, 0.03**2

    def g(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    ma, mb = g(ya), g(yb)
    va, vb, cov = g(ya * ya) - ma**2, g(yb * yb) - mb**2, g(ya * yb) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma**2 + mb**2 + c1) * (va + vb + c2))
    return float(s.mean())


def detail_energy(img: np.ndarray) -> float:
    """Mean |Laplacian| of luma on a 2x-downscaled copy (noise mostly averaged out)."""
    y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    y = cv2.resize(y, (y.shape[1] // 2, y.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return float(np.abs(cv2.Laplacian(y, cv2.CV_32F)).mean())


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


NOISE_LEVELS = {  # (shot, read) variance terms in linear light
    "leicht": (0.0015, 0.00001),
    "mittel": (0.005, 0.00005),
    "stark": (0.015, 0.0002),
}


def add_noise(clean: np.ndarray, shot: float, read: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lin = srgb_to_linear(clean)
    noisy = lin + rng.normal(size=lin.shape) * np.sqrt(shot * lin + read)
    return linear_to_srgb(noisy).astype(np.float32)


# ---------------------------------------------------------------- sheets
def sheet(rows: list[tuple[str, np.ndarray]], out: Path, zoom: int = 384) -> None:
    """Center 'zoom' px of each result side by side with labels."""
    tiles = []
    for label, img in rows:
        h, w = img.shape[:2]
        c = img[(h - zoom) // 2 : (h + zoom) // 2, (w - zoom) // 2 : (w + zoom) // 2]
        t = cv2.cvtColor((np.clip(c, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        t = cv2.copyMakeBorder(t, 28, 0, 0, 4, cv2.BORDER_CONSTANT, value=(24, 24, 24))
        cv2.putText(
            t, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA
        )
        tiles.append(t)
    cv2.imwrite(str(out), np.hstack(tiles))


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, default=Path("../bench/out"))
    ap.add_argument(
        "--models",
        nargs="+",
        default=["scunet_real_psnr_fp16", "scunet_real_psnr", "nafnet_sidd_fp16", "nafnet_sidd"],
    )
    ap.add_argument("--out", type=Path, default=Path("../bench/out/denoise"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    methods: dict[str, object] = {"nlmeans": nlmeans}
    for m in args.models:
        methods[m] = ModelMethod(args.models_dir / f"{m}.onnx")
    print(
        "device:",
        next(v for v in methods.values() if isinstance(v, ModelMethod)).backend.device_info().name,
    )

    files = sorted(p for p in args.photos.iterdir() if p.suffix.lower() in RAW_EXT | IMG_EXT)
    results: dict[str, list[dict[str, object]]] = {"real": [], "synthetic": []}
    timing: dict[str, list[float]] = {k: [] for k in methods}

    for f in files:
        img = load_photo(f, args.out / "cache")
        est0 = estimate_noise(img)
        print(
            f"\n{f.name}: {img.shape[1]}x{img.shape[0]}  noise luma {est0.luma_sigma:.2f} "
            f"chroma {est0.chroma_sigma:.2f}"
        )
        for region, (y0, x0) in pick_crops(img).items():
            crop = np.ascontiguousarray(img[y0 : y0 + CROP, x0 : x0 + CROP])
            e = estimate_noise(crop)
            d0 = detail_energy(crop)
            rows = [("Original", crop)]
            entry = {
                "file": f.name,
                "region": region,
                "luma": round(e.luma_sigma, 2),
                "chroma": round(e.chroma_sigma, 2),
                "methods": {},
            }
            for name, fn in methods.items():
                t = time.perf_counter()
                res = fn(crop)  # type: ignore[operator]
                dt = time.perf_counter() - t
                timing[name].append(dt / (CROP * CROP / 1e6))
                er = estimate_noise(res)
                entry["methods"][name] = {
                    "luma_after": round(er.luma_sigma, 2),
                    "chroma_after": round(er.chroma_sigma, 2),
                    "detail_kept": round(detail_energy(res) / max(d0, 1e-6), 3),
                }
                rows.append((name, res))
            sheet(rows, args.out / f"{f.stem}_{region}.png")
            results["real"].append(entry)
            print(
                f"  {region:8} luma {e.luma_sigma:5.2f} chroma {e.chroma_sigma:5.2f} | "
                + "  ".join(
                    f"{k}: {v['luma_after']:.2f}/{v['chroma_after']:.2f} det {v['detail_kept']:.2f}"
                    for k, v in entry["methods"].items()
                )
            )

        # Part B: synthetic noise on a 4x-downscaled (clean) copy
        small = cv2.resize(
            img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA
        )
        sh, sw = small.shape[:2]
        clean = np.ascontiguousarray(
            small[(sh - CROP) // 2 : (sh + CROP) // 2, (sw - CROP) // 2 : (sw + CROP) // 2]
        )
        if clean.shape[:2] != (CROP, CROP):
            continue
        for lvl, (shot, read) in NOISE_LEVELS.items():
            noisy = add_noise(clean, shot, read, seed=zlib.crc32(f"{f.name}{lvl}".encode()))
            entry = {
                "file": f.name,
                "level": lvl,
                "noisy": {
                    "psnr": round(psnr(noisy, clean), 2),
                    "ssim": round(ssim_y(noisy, clean), 4),
                },
                "methods": {},
            }
            rows = [("sauber", clean), (f"Rauschen {lvl}", noisy)]
            for name, fn in methods.items():
                res = fn(noisy)  # type: ignore[operator]
                entry["methods"][name] = {
                    "psnr": round(psnr(res, clean), 2),
                    "ssim": round(ssim_y(res, clean), 4),
                }
                rows.append((name, res))
            if lvl == "mittel":
                sheet(rows, args.out / f"{f.stem}_synthetic_{lvl}.png")
            results["synthetic"].append(entry)
            print(
                f"  synth {lvl:6} noisy {entry['noisy']['psnr']:.2f} dB | "
                + "  ".join(
                    f"{k}: {v['psnr']:.2f}/{v['ssim']:.3f}" for k, v in entry["methods"].items()
                )
            )

    results["timing_s_per_mp"] = {
        k: round(float(np.median(v[1:] or v)), 3) for k, v in timing.items() if v
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\ntiming s/MP:", results["timing_s_per_mp"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
