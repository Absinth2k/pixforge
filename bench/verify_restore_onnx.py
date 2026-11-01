"""Export SCUNet / NAFNet / FBCNN to ONNX (fp32 + mixed fp16) and verify against PyTorch.

Run from backend/:  .venv/Scripts/python ../bench/verify_restore_onnx.py ../weights/denoise ../bench/out
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from pixforge.models.export_onnx import build_spec, export

sys.path.insert(0, str(Path(__file__).parent))
from verify_hat_onnx import psnr, synthetic_photo  # noqa: E402

JOBS = [
    ("scunet", "scunet_color_real_psnr.pth", "scunet_real_psnr"),
    ("nafnet", "NAFNet-SIDD-width64.pth", "nafnet_sidd"),
    ("fbcnn", "fbcnn_color.pth", "fbcnn"),
]


def main() -> int:
    wdir, odir = Path(sys.argv[1]), Path(sys.argv[2])
    ok = True
    for arch, wname, stem in JOBS:
        spec = build_spec(arch, wdir / wname)
        for fp16 in (False, True):
            out = odir / f"{stem}{'_fp16' if fp16 else ''}.onnx"
            meta = {"pixforge.arch": arch, "pixforge.scale": "1",
                    "pixforge.align": str(spec.alignment), "pixforge.license": spec.license,
                    "pixforge.source": wname,
                    "pixforge.precision": "mixed-fp16" if fp16 else "fp32"}
            export(spec, out, 128, fp16=fp16, metadata=meta)
            x = synthetic_photo(256, 320)
            extra = {k: np.array(f(256)) for k, f in spec.extra_inputs.items()}
            with torch.no_grad():
                targs = [torch.from_numpy(x), *(torch.from_numpy(v) for v in extra.values())]
                ref = spec.model(*targs)
                ref = (ref[0] if isinstance(ref, tuple) else ref).numpy()
            for prov in ("CPUExecutionProvider", "DmlExecutionProvider"):
                so = ort.SessionOptions()
                so.log_severity_level = 3
                if prov.startswith("Dml"):
                    so.enable_mem_pattern = False
                sess = ort.InferenceSession(str(out), so, providers=[prov])
                y = sess.run(None, {"input": x, **extra})[0]
                p = psnr(ref, y)
                ok &= p > 50
                print(f"{out.name:28} {out.stat().st_size / 2**20:6.1f} MB  {prov[:3]}  "
                      f"PSNR {p:6.2f} dB")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
