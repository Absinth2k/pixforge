"""ONNX Runtime backend: DirectML on any DX12 GPU (NVIDIA, AMD, Intel), CPU as fallback."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from pixforge.backends.base import BackendError, DeviceInfo, InferenceBackend
from pixforge.models.hat_mask import shifted_window_mask

log = logging.getLogger(__name__)

DML = "DmlExecutionProvider"
CPU = "CPUExecutionProvider"


@dataclass(frozen=True)
class _Candidate:
    info: DeviceInfo
    providers: list[str | tuple[str, dict[str, str]]]


def list_devices() -> list[DeviceInfo]:
    """All devices ONNX Runtime can use here, best first (discrete GPU > iGPU > CPU)."""
    return [c.info for c in _candidates()]


def _cpu_name() -> str:
    try:
        for d in ort.get_ep_devices():
            if d.ep_name == CPU:
                return d.device.metadata.get("Description", "CPU")
    except AttributeError:
        pass
    return os.environ.get("PROCESSOR_IDENTIFIER", "CPU")


def _candidates() -> list[_Candidate]:
    available = set(ort.get_available_providers())
    gpus: list[_Candidate] = []
    if DML in available:
        try:
            ep_devices = [d for d in ort.get_ep_devices() if d.ep_name == DML]
        except AttributeError:  # older onnxruntime without the device API
            ep_devices = []
        for d in ep_devices:
            md = d.device.metadata
            idx = int(md.get("DxgiAdapterNumber", 0))
            vram = md.get("DxgiVideoMemory", "").replace("MB", "").strip()
            info = DeviceInfo(
                name=md.get("Description", "GPU"),
                kind="gpu",
                provider=DML,
                vram_mb=int(vram) if vram.isdigit() else None,
                discrete=md.get("Discrete") == "1",
                device_index=idx,
            )
            gpus.append(_Candidate(info, [(DML, {"device_id": str(idx)}), CPU]))
        if not ep_devices:
            info = DeviceInfo("DirectML GPU", "gpu", DML, device_index=0)
            gpus.append(_Candidate(info, [(DML, {"device_id": "0"}), CPU]))
    gpus.sort(key=lambda c: (not c.info.discrete, -(c.info.vram_mb or 0)))
    cpu = _Candidate(DeviceInfo(_cpu_name(), "cpu", CPU), [CPU])
    return [*gpus, cpu]


class OnnxBackend(InferenceBackend):
    """Runs Pixforge ONNX models (HAT, denoisers, face restorers).

    ``device``: "auto" (best GPU, else CPU), "cpu", or "gpu:<dxgi index>".
    The graphs have dynamic height/width. Pinning them via free-dimension overrides
    was measured and gives no speedup on DirectML, so tiles may vary in size; keep
    the size constant anyway, because every new shape costs a one-time shader compile.
    """

    def __init__(self, device: str = "auto") -> None:
        self._device_req = device
        self._session: ort.InferenceSession | None = None
        self._info: DeviceInfo | None = None
        self._needs_mask = False
        self._window = 16
        self._input_dtype = np.float32

    # -- InferenceBackend -------------------------------------------------
    def load(self, model_id: str | Path) -> None:
        path = Path(model_id)
        if not path.is_file():
            from pixforge.models.registry import model_path  # lazy: registry is optional here

            path = model_path(str(model_id))
        cand = self._pick_candidate()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        if cand.info.provider == DML:
            # Required by the DirectML EP.
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            self._session = ort.InferenceSession(str(path), so, providers=cand.providers)
        except Exception as exc:  # pragma: no cover - depends on driver
            if cand.info.provider == DML and self._device_req == "auto":
                log.warning("DirectML session failed (%s), falling back to CPU", exc)
                cand = _candidates()[-1]
                self._session = ort.InferenceSession(str(path), so, providers=cand.providers)
            else:
                raise BackendError(f"Modell konnte nicht geladen werden: {exc}") from exc
        self._info = cand.info

        meta = self._session.get_modelmeta().custom_metadata_map
        self.scale = int(meta.get("pixforge.scale", "1"))
        self._window = int(meta.get("pixforge.window", "16"))
        names = [i.name for i in self._session.get_inputs()]
        self._needs_mask = "attn_mask" in names
        self.alignment = self._window if self._needs_mask else 1
        in_type = self._session.get_inputs()[0].type
        self._input_dtype = np.float16 if "float16" in in_type else np.float32

    def run_tile(self, tile: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise BackendError("Kein Modell geladen")
        if tile.ndim != 3 or tile.shape[2] != 3:
            raise ValueError(f"expected (h, w, 3) tile, got {tile.shape}")
        h, w = tile.shape[:2]
        if self.alignment > 1 and (h % self.alignment or w % self.alignment):
            raise ValueError(f"tile {h}x{w} not aligned to {self.alignment}")
        x = np.ascontiguousarray(tile.transpose(2, 0, 1)[None], dtype=self._input_dtype)
        feeds: dict[str, np.ndarray] = {"input": x}
        if self._needs_mask:
            feeds["attn_mask"] = shifted_window_mask(h, w, self._window).astype(
                self._input_dtype, copy=False
            )
        try:
            y = self._session.run(None, feeds)[0]
        except Exception as exc:
            msg = str(exc)
            if "887A0005" in msg or "887A0006" in msg or "OUTOFMEMORY" in msg.upper():
                raise BackendError("Grafikkarte hat zu wenig Speicher") from exc
            raise
        return np.ascontiguousarray(y[0].transpose(1, 2, 0), dtype=np.float32)

    def device_info(self) -> DeviceInfo:
        if self._info is None:
            return self._pick_candidate().info
        return self._info

    def close(self) -> None:
        self._session = None

    # -- helpers ----------------------------------------------------------
    def _pick_candidate(self) -> _Candidate:
        cands = _candidates()
        req = self._device_req
        if req == "auto":
            return cands[0]
        if req == "cpu":
            return cands[-1]
        if req.startswith("gpu:"):
            idx = int(req.split(":", 1)[1])
            for c in cands:
                if c.info.kind == "gpu" and c.info.device_index == idx:
                    return c
            raise BackendError(f"GPU {idx} nicht gefunden")
        raise ValueError(f"unknown device spec {req!r}")
