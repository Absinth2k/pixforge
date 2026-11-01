"""Build all ONNX model files listed in models.json (developer tool, needs the `export` extra).

    python -m pixforge.models.build --weights ../weights [--out DIR] [--only photo ...] [--force]

Writes ``<id>.<precision>.onnx`` into the output directory (default: the user's model folder)
and records size and SHA-256 of every file back into models.json, so the release assets and
the registry always match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from pixforge.models.registry import registry_path
from pixforge.paths import models_dir


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_one(model: dict, precision: str, weights_root: Path, out: Path) -> None:
    from pixforge.models.export_onnx import build_spec, export

    b = model["build"]
    src = weights_root / b["weights"]
    if b["arch"] == "copy":
        shutil.copyfile(src, out)
        return
    spec = build_spec(b["arch"], src, b.get("variant", "HAT"), int(model["scale"]))
    meta = {
        "pixforge.id": model["id"],
        "pixforge.arch": b.get("variant", b["arch"]) if b["arch"] == "hat" else b["arch"],
        "pixforge.scale": str(spec.scale),
        "pixforge.align": str(spec.alignment),
        "pixforge.window": "16",
        "pixforge.range": spec.value_range,
        "pixforge.source": Path(b["weights"]).name,
        "pixforge.license": spec.license,
        "pixforge.precision": "mixed-fp16" if precision == "fp16" else "fp32",
    }
    export(
        spec,
        out,
        128 if spec.alignment <= 64 else spec.alignment,
        fp16=precision == "fp16",
        metadata=meta,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = args.out or models_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    reg_file = registry_path()
    data = json.loads(reg_file.read_text(encoding="utf-8"))
    for model in data["models"]:
        if args.only and model["id"] not in args.only:
            continue
        for precision in model["build"].get("precisions", ["fp16", "fp32"]):
            name = f"{model['id']}.{precision}.onnx"
            path = out_dir / name
            if args.force or not path.exists():
                t = time.perf_counter()
                print(f"building {name} ...", flush=True)
                build_one(model, precision, args.weights, path)
                print(f"  done in {time.perf_counter() - t:.0f} s", flush=True)
            model.setdefault("files", {})[precision] = {
                "file": name,
                "sha256": sha256(path),
                "size": path.stat().st_size,
            }
    reg_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"registry updated: {reg_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
