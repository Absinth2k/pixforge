"""Model registry: what Pixforge knows about, and where the files live."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path

from pixforge.paths import models_dir

PRECISIONS = ("fp16", "fp32")


@dataclass(frozen=True)
class ModelFile:
    file: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ModelInfo:
    id: str
    kind: str  # upscale | denoise | jpeg | face | detect
    name: dict[str, str]
    description: dict[str, str]
    scale: int
    license: str
    origin: str
    build: dict[str, object]
    files: dict[str, ModelFile] = field(default_factory=dict)
    default: bool = False
    noncommercial: bool = False

    def display_name(self, lang: str = "de") -> str:
        return self.name.get(lang) or self.name.get("en") or self.id

    def precisions(self) -> list[str]:
        return list(self.build.get("precisions", PRECISIONS))  # type: ignore[arg-type]

    def file_name(self, precision: str) -> str:
        if precision in self.files:
            return self.files[precision].file
        return f"{self.id}.{precision}.onnx"


class ModelNotInstalled(FileNotFoundError):
    def __init__(self, info: ModelInfo) -> None:
        super().__init__(f"Modell „{info.display_name()}“ ist nicht installiert")
        self.info = info


def registry_path() -> Path:
    return Path(str(resources.files("pixforge.models").joinpath("models.json")))


@lru_cache(maxsize=1)
def _load() -> tuple[str, dict[str, ModelInfo]]:
    data = json.loads(registry_path().read_text(encoding="utf-8"))
    out: dict[str, ModelInfo] = {}
    for m in data["models"]:
        files = {p: ModelFile(**f) for p, f in m.get("files", {}).items()}
        out[m["id"]] = ModelInfo(
            id=m["id"],
            kind=m["kind"],
            name=m["name"],
            description=m["description"],
            scale=int(m["scale"]),
            license=m["license"],
            origin=m["origin"],
            build=m.get("build", {}),
            files=files,
            default=bool(m.get("default", False)),
            noncommercial=bool(m.get("noncommercial", False)),
        )
    return data.get("release_url", ""), out


def all_models() -> list[ModelInfo]:
    return list(_load()[1].values())


def release_url() -> str:
    return _load()[0]


def get(model_id: str) -> ModelInfo:
    try:
        return _load()[1][model_id]
    except KeyError:
        raise KeyError(f"unbekanntes Modell {model_id!r}") from None


def default_model(kind: str) -> ModelInfo:
    for m in all_models():
        if m.kind == kind and m.default:
            return m
    raise KeyError(f"kein Standardmodell für {kind!r}")


def installed_precisions(model_id: str) -> list[str]:
    info = get(model_id)
    return [p for p in info.precisions() if (models_dir() / info.file_name(p)).is_file()]


def model_path(model_id: str, precision: str | None = None) -> Path:
    """Path of an installed model file.

    ``precision`` None picks whatever is installed, preferring fp16. A requested precision
    falls back to the other one if only that is installed.
    """
    info = get(model_id)
    order = [precision] if precision else []
    order += [p for p in ("fp16", "fp32") if p not in order]
    for p in order:
        if p not in info.precisions():
            continue
        path = models_dir() / info.file_name(p)
        if path.is_file():
            return path
    raise ModelNotInstalled(info)
