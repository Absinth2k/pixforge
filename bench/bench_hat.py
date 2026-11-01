"""Throughput / VRAM benchmark for Pixforge ONNX models.

Each configuration runs in a fresh subprocess so the VRAM numbers are clean.
Run from backend/:  .venv/Scripts/python ../bench/bench_hat.py --models a.onnx b.onnx
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

OVERLAP = 32  # px of input overlap between neighbouring tiles


def child(model: str, tile: int, device: str, runs: int) -> dict[str, object]:
    import numpy as np

    sys.path.insert(0, str(Path(__file__).parent))
    from gpumem import GpuMemSampler

    from pixforge.backends.onnx_backend import OnnxBackend

    mem = GpuMemSampler().start() if device != "cpu" else None
    b = OnnxBackend(device)
    t0 = time.perf_counter()
    b.load(model)
    load_s = time.perf_counter() - t0
    x = np.random.default_rng(0).random((tile, tile, 3), dtype=np.float32)
    t0 = time.perf_counter()
    b.run_tile(x)
    first_s = time.perf_counter() - t0
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        b.run_tile(x)
        times.append(time.perf_counter() - t0)
    dt = float(np.median(times))
    peak = mem.stop() if mem else 0
    useful = (tile - OVERLAP) ** 2
    return {
        "model": Path(model).name,
        "device": b.device_info().name,
        "tile": tile,
        "ms_per_tile": round(dt * 1000, 1),
        "s_per_input_mp": round(dt / (tile * tile / 1e6), 2),
        "s_per_input_mp_with_overlap": round(dt / (useful / 1e6), 2),
        "s_per_output_mp": round(dt / (tile * tile * b.scale**2 / 1e6), 3),
        "vram_peak_mb": round(peak / 2**20) if peak else None,
        "load_s": round(load_s, 2),
        "first_run_s": round(first_s, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tiles", default="128,192,256,384,512")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("../bench/out/bench_hat.jsonl"))
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--tile", type=int)
    args = ap.parse_args()

    if args.child:
        res = child(args.models[0], args.tile, args.device, args.runs)
        print("RESULT " + json.dumps(res))
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        for model in args.models:
            for tile in (int(t) for t in args.tiles.split(",")):
                cmd = [sys.executable, __file__, "--child", "--models", model, "--tile", str(tile),
                       "--device", args.device, "--runs", str(args.runs)]
                proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
                line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
                if line is None:
                    err = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["?"]
                    print(f"{Path(model).name} tile {tile}: FAILED {err[0][:200]}")
                    continue
                res = json.loads(line[7:])
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                print(f"{res['model']:32} {res['device'][:22]:22} tile {tile:4}: "
                      f"{res['ms_per_tile']:8.0f} ms  {res['s_per_input_mp_with_overlap']:7.1f} s/MP(in)"
                      f"  VRAM {res['vram_peak_mb']} MB  first {res['first_run_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
