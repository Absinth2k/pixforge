"""Model download: resume, checksum, friendly errors.

``urllib.request.urlopen`` is replaced by a fake server (no sockets: on the development
machine an antivirus resets loopback connections now and then, which made a real local HTTP
server flaky).
"""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from pixforge.models import download as dl
from pixforge.models import registry

DATA = bytes(range(256)) * 4096  # 1 MiB


class FakeServer:
    def __init__(self) -> None:
        self.payload = DATA
        self.ranges: list[str | None] = []
        self.honour_range = True

    def urlopen(self, req: urllib.request.Request, timeout: float = 0) -> object:
        rng = req.get_header("Range")
        self.ranges.append(rng)
        if req.full_url.endswith("missing.onnx"):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)  # type: ignore[arg-type]
        start = int(rng.split("=")[1].rstrip("-")) if rng and self.honour_range else 0

        class Resp(io.BytesIO):
            status = 206 if rng and self.honour_range else 200

        return Resp(self.payload[start:])


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    monkeypatch.setenv("PIXFORGE_MODELS", str(tmp_path / "models"))
    fake = FakeServer()
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)

    def fake_get(model_id: str) -> registry.ModelInfo:
        name = "missing.onnx" if model_id == "missing" else f"{model_id}.fp16.onnx"
        f = registry.ModelFile(name, hashlib.sha256(DATA).hexdigest(), len(DATA))
        return registry.ModelInfo(
            model_id, "upscale", {"de": model_id}, {}, 4, "MIT", "", {}, {"fp16": f}
        )

    monkeypatch.setattr(registry, "get", fake_get)
    return fake


URL = "https://example.invalid/models-v1/"


def test_download_checks_and_installs(server: FakeServer) -> None:
    seen: list[int] = []
    path = dl.download("m", "fp16", lambda d, n: seen.append(d), base_url=URL)
    assert path.read_bytes() == DATA
    assert seen[-1] == len(DATA)
    assert not path.with_name(path.name + ".part").exists()
    assert dl.needed("m", "gpu") is None
    assert server.ranges == [None]


def test_interrupted_download_resumes(server: FakeServer, tmp_path: Path) -> None:
    part = tmp_path / "models" / "m.fp16.onnx.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(DATA[:300_000])
    path = dl.download("m", "fp16", base_url=URL)
    assert path.read_bytes() == DATA
    assert server.ranges == ["bytes=300000-"]


def test_server_without_range_support_starts_over(server: FakeServer, tmp_path: Path) -> None:
    server.honour_range = False
    part = tmp_path / "models" / "m.fp16.onnx.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"x" * 1000)  # would be wrong if appended to
    assert dl.download("m", "fp16", base_url=URL).read_bytes() == DATA


def test_corrupt_download_is_discarded(server: FakeServer, tmp_path: Path) -> None:
    server.payload = DATA[:-1] + b"\x00"
    with pytest.raises(dl.DownloadError, match="beschädigt"):
        dl.download("m", "fp16", base_url=URL)
    assert not list((tmp_path / "models").iterdir())  # neither the model nor a .part


def test_server_error_is_friendly(server: FakeServer) -> None:
    with pytest.raises(dl.DownloadError, match="404"):
        dl.download("missing", "fp16", base_url=URL)


def test_cancel_keeps_the_part_for_later(server: FakeServer, tmp_path: Path) -> None:
    with pytest.raises(dl.DownloadCancelled):
        dl.download("m", "fp16", cancel=lambda: True, base_url=URL)
    assert not (tmp_path / "models" / "m.fp16.onnx").exists()
