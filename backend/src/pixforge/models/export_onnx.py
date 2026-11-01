"""Convert upstream PyTorch checkpoints to ONNX (developer tool, needs the `export` extra).

Usage:
    python -m pixforge.models.export_onnx hat weights/hat/Real_HAT_GAN_SRx4.pth out.onnx \
        --variant HAT --scale 4 [--fp16] [--calib-dir photos/]
    python -m pixforge.models.export_onnx scunet weights/denoise/scunet_color_real_psnr.pth out.onnx

``--fp16`` produces a mixed-precision model: weights and most math in float16, but every
node whose float32 activations come within 1/8 of the float16 range on the calibration
inputs stays in float32 (HAT's residual stream reaches ~1e6 in the last group).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from pixforge.models.fbcnn_arch import FBCNN
from pixforge.models.hat_arch import HAT, hat_config, load_hat_state_dict
from pixforge.models.hat_mask import shifted_window_mask
from pixforge.models.nafnet_arch import NAFNet
from pixforge.models.scunet_arch import SCUNet

OPSET = 18
FP16_MAX = 65504.0
FP16_HEADROOM = 8.0
CALIB_SIZE = 64


@dataclass
class ExportSpec:
    """How to trace one architecture."""

    model: torch.nn.Module
    scale: int
    alignment: int
    license: str
    extra_inputs: dict[str, Callable[[int], np.ndarray]] = field(default_factory=dict)
    extra_dynamic: dict[str, dict[int, str]] = field(default_factory=dict)
    output_names: tuple[str, ...] = ("output",)
    calib_size: int = CALIB_SIZE
    #: fixed square input size (face restorers); None = dynamic height/width
    fixed_size: int | None = None
    #: value range of input/output images: "0..1" or "-1..1"
    value_range: str = "0..1"


class _GFPGANInference(torch.nn.Module):
    """Deterministic GFPGAN forward (fixed noise buffers), image output only."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x, return_rgb=False, randomize_noise=False)[0]


def _load(weights: Path) -> dict[str, torch.Tensor]:
    state = torch.load(weights, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params"):
        if key in state:
            return state[key]
    return state


def build_spec(arch: str, weights: Path, variant: str = "HAT", scale: int = 4) -> ExportSpec:
    if arch == "hat":
        hat = HAT(hat_config(variant, scale)).eval()
        load_hat_state_dict(hat, torch.load(weights, map_location="cpu", weights_only=True))
        return ExportSpec(
            hat,
            scale,
            16,
            "Apache-2.0 (XPixelGroup/HAT)",
            extra_inputs={"attn_mask": lambda t: shifted_window_mask(t, t, 16)},
            extra_dynamic={"attn_mask": {0: "num_windows"}},
        )
    if arch == "scunet":
        net = SCUNet().eval()
        net.load_state_dict(_load(weights), strict=True)
        return ExportSpec(net, 1, 64, "Apache-2.0 (cszn/SCUNet)", calib_size=128)
    if arch == "nafnet":
        naf = NAFNet().eval()
        naf.load_state_dict(_load(weights), strict=True)
        return ExportSpec(naf, 1, 16, "MIT (megvii-research/NAFNet)")
    if arch == "fbcnn":
        fb = FBCNN().eval()
        fb.load_state_dict(_load(weights), strict=True)
        return ExportSpec(
            fb,
            1,
            8,
            "Apache-2.0 (jiaxi-jiang/FBCNN)",
            extra_inputs={"qf": lambda t: np.full((1, 1), -1.0, np.float32)},
            output_names=("output", "qf_pred"),
        )
    if arch == "restoreformer":
        from pixforge.models.restoreformer_arch import RestoreFormerPP, load_restoreformer_pp

        rf = RestoreFormerPP().eval()
        load_restoreformer_pp(rf, torch.load(weights, map_location="cpu", weights_only=True))
        return ExportSpec(
            rf,
            1,
            512,
            "Apache-2.0 (wzhouxiff/RestoreFormerPlusPlus)",
            fixed_size=512,
            value_range="-1..1",
        )
    if arch == "gfpgan":
        from pixforge.models.gfpgan_arch import GFPGANv1Clean

        gf = GFPGANv1Clean(
            out_size=512,
            num_style_feat=512,
            channel_multiplier=2,
            decoder_load_path=None,
            fix_decoder=False,
            num_mlp=8,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True,
        )
        gf.load_state_dict(_load(weights), strict=True)
        return ExportSpec(
            _GFPGANInference(gf.eval()),
            1,
            512,
            "Apache-2.0 + NVIDIA Source Code License (non-commercial) (TencentARC/GFPGAN)",
            fixed_size=512,
            value_range="-1..1",
        )
    raise ValueError(f"unknown arch {arch!r}")


def export(
    spec: ExportSpec,
    out: Path,
    tile: int,
    fp16: bool = False,
    metadata: dict[str, str] | None = None,
    calib_dir: Path | None = None,
) -> Path:
    """Export with input ``input`` (1, 3, h, w) RGB 0..1 plus the spec's extra inputs.

    h and w are dynamic (multiples of ``spec.alignment``); ``tile`` only sets the trace shape.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    if spec.fixed_size:
        tile = spec.fixed_size
    names = ["input", *spec.extra_inputs]
    args = (
        torch.rand(1, 3, tile, tile),
        *(torch.from_numpy(np.array(f(tile))) for f in spec.extra_inputs.values()),
    )
    dynamic = (
        {}
        if spec.fixed_size
        else {
            "input": {2: "height", 3: "width"},
            "output": {2: "out_height", 3: "out_width"},
            **spec.extra_dynamic,
        }
    )
    with torch.no_grad():
        torch.onnx.export(
            spec.model,
            args,
            str(out),
            input_names=names,
            output_names=list(spec.output_names),
            dynamic_axes=dynamic,
            opset_version=OPSET,
            dynamo=False,
            # Folding would bake per-block position-bias gathers into large constants;
            # ONNX Runtime folds them at load time instead.
            do_constant_folding=False,
        )
    proto = onnx.load(str(out))
    if fp16:
        cs = spec.calib_size
        feeds = [
            {"input": x, **{k: np.array(f(cs)) for k, f in spec.extra_inputs.items()}}
            for x in calibration_inputs(calib_dir, size=cs)
        ]
        peaks = activation_peaks(proto, feeds)
        limit = FP16_MAX / FP16_HEADROOM
        # A node must stay float32 if it reads OR writes a large tensor; otherwise the
        # cast to float16 in front of it already overflows (e.g. HAT's final LayerNorm).
        block = [
            n.name
            for n in proto.graph.node
            if max((peaks.get(t, 0.0) for t in (*n.input, *n.output)), default=0.0) > limit
        ]
        print(f"fp16: keeping {len(block)} of {len(proto.graph.node)} nodes in float32")
        proto = _to_fp16(proto, block)
    for k, v in (metadata or {}).items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(proto, str(out))
    onnx.checker.check_model(str(out))
    return out


def calibration_inputs(
    calib_dir: Path | None = None, n_synthetic: int = 10, size: int = CALIB_SIZE
) -> list[np.ndarray]:
    """NCHW float32 crops: synthetic stress patterns plus optional real photos."""
    s = size
    rng = np.random.default_rng(1234)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) / s
    out: list[np.ndarray] = [
        np.zeros((3, s, s), np.float32),
        np.ones((3, s, s), np.float32),
        rng.random((3, s, s), dtype=np.float32),
        np.repeat((rng.random((1, s, s)) > 0.5).astype(np.float32), 3, 0),
        np.stack([xx, yy, 1 - xx]),
        np.stack([np.ones_like(xx), np.zeros_like(xx), np.zeros_like(xx)]),  # saturated red
        np.repeat(((xx * 8).astype(int) % 2).astype(np.float32)[None], 3, 0),  # stripes
    ]
    while len(out) < n_synthetic:
        base = rng.random(3)[:, None, None] * 0.8
        freq = rng.uniform(2, 20)
        tex = 0.3 * np.sin(xx * freq * 6.28 + rng.uniform(0, 6)) * np.cos(yy * freq * 3.1)
        img = base + tex[None] + rng.normal(0, rng.uniform(0, 0.1), (3, s, s))
        out.append(np.clip(img, 0, 1).astype(np.float32))
    if calib_dir:
        from PIL import Image

        for p in sorted(calib_dir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
                continue
            im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            for _ in range(4):
                y = rng.integers(0, im.shape[0] - s)
                x = rng.integers(0, im.shape[1] - s)
                out.append(np.ascontiguousarray(im[y : y + s, x : x + s].transpose(2, 0, 1)))
    return [x[None] for x in out]


def activation_peaks(
    proto: onnx.ModelProto, feeds: list[dict[str, np.ndarray]]
) -> dict[str, float]:
    """Max |value| per tensor name over the feeds, by running the float32 graph on the CPU."""
    probe = onnx.ModelProto()
    probe.CopyFrom(proto)
    producer = {o: n.name for n in probe.graph.node for o in n.output}
    existing = {o.name for o in probe.graph.output}
    probe.graph.output.extend(
        onnx.helper.make_empty_tensor_value_info(o) for o in producer if o not in existing
    )
    so = ort.SessionOptions()
    so.log_severity_level = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(probe.SerializeToString(), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    peaks: dict[str, float] = {}
    for feed in feeds:
        for name, val in zip(names, sess.run(None, feed), strict=True):
            if isinstance(val, np.ndarray) and val.dtype == np.float32 and val.size:
                peaks[name] = max(peaks.get(name, 0.0), float(np.abs(val).max()))
    return peaks


def _to_fp16(proto: onnx.ModelProto, node_block_list: list[str]) -> onnx.ModelProto:
    """Store weights and compute in float16, keep float32 graph inputs/outputs."""
    from onnxruntime.transformers.float16 import convert_float_to_float16

    proto = convert_float_to_float16(
        proto,
        keep_io_types=True,
        disable_shape_infer=False,
        node_block_list=node_block_list,
    )
    _remove_cast_roundtrips(proto.graph)
    _toposort(proto.graph)
    return proto


def _remove_cast_roundtrips(graph: onnx.GraphProto) -> None:
    """Rewire float32 -> Cast(fp16) -> Cast(fp32) chains to use the float32 tensor directly.

    The converter inserts such round trips between two blocked (float32) nodes, which
    defeats the purpose of blocking them: the value overflows in the middle.
    """
    f32, f16 = onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16
    by_output = {o: n for n in graph.node for o in n.output}

    def cast_to(node: onnx.NodeProto) -> int | None:
        if node.op_type != "Cast":
            return None
        return next((a.i for a in node.attribute if a.name == "to"), None)

    replace: dict[str, str] = {}
    for node in graph.node:
        if cast_to(node) != f32:
            continue
        prev = by_output.get(node.input[0])
        if prev is None or cast_to(prev) != f16:
            continue
        src = by_output.get(prev.input[0])
        # Only if the source really is float32 (a blocked node or a graph input).
        src_is_f32 = src is None or src.op_type != "Cast" or cast_to(src) == f32
        if src_is_f32:
            replace[node.output[0]] = prev.input[0]
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in replace:
                node.input[i] = replace[name]
    for out in graph.output:
        if out.name in replace:
            raise RuntimeError("cast round trip on a graph output")
    # Drop casts whose outputs are no longer used.
    while True:
        used = {i for n in graph.node for i in n.input} | {o.name for o in graph.output}
        dead = [n for n in graph.node if n.op_type == "Cast" and n.output[0] not in used]
        if not dead:
            break
        for n in dead:
            graph.node.remove(n)


def _toposort(graph: onnx.GraphProto) -> None:
    """The fp16 converter appends input casts at the end; restore topological order."""
    available = {i.name for i in graph.input} | {t.name for t in graph.initializer} | {""}
    pending = list(graph.node)
    ordered: list[onnx.NodeProto] = []
    while pending:
        rest = []
        for node in pending:
            if all(i in available for i in node.input):
                ordered.append(node)
                available.update(node.output)
            else:
                rest.append(node)
        if len(rest) == len(pending):
            raise RuntimeError("graph has a cycle or dangling inputs")
        pending = rest
    del graph.node[:]
    graph.node.extend(ordered)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arch", choices=["hat", "scunet", "nafnet", "fbcnn"])
    ap.add_argument("weights", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--variant", default="HAT", choices=["HAT", "HAT-S", "HAT-L"])
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--tile", type=int, default=128, help="trace size (graph stays dynamic)")
    ap.add_argument("--fp16", action="store_true", help="mixed precision, see module doc")
    ap.add_argument("--calib-dir", type=Path, help="extra photos for fp16 calibration")
    args = ap.parse_args()

    spec = build_spec(args.arch, args.weights, args.variant, args.scale)
    if args.tile % spec.alignment:
        raise SystemExit(f"tile must be a multiple of {spec.alignment}")
    meta = {
        "pixforge.arch": args.variant if args.arch == "hat" else args.arch,
        "pixforge.scale": str(spec.scale),
        "pixforge.align": str(spec.alignment),
        "pixforge.window": "16",
        "pixforge.source": args.weights.name,
        "pixforge.license": spec.license,
        "pixforge.precision": "mixed-fp16" if args.fp16 else "fp32",
    }
    export(spec, args.out, args.tile, fp16=args.fp16, metadata=meta, calib_dir=args.calib_dir)
    size = args.out.stat().st_size / 2**20
    print(f"wrote {args.out} ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
