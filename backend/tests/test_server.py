"""HTTP API with fake backends (no models, no GPU)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from fakes import IdentityFaceBackend, NearestBackend
from fastapi.testclient import TestClient
from PIL import Image

from pixforge.faces.detect import Face
from pixforge.server.app import create_app

TOKEN = "test-token"
H = {"X-Pixforge-Token": TOKEN}


class NoFaces:
    def detect(self, rgb: np.ndarray, min_size: float = 0) -> list[Face]:
        return []


def factory(model_id: str) -> object:
    if model_id.startswith("face"):
        return IdentityFaceBackend()
    scale = 1 if model_id in ("denoise", "jpeg") else 4
    return NearestBackend(scale=scale, alignment=16, edge=0)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app(TOKEN, backend_factory=factory, detector=NoFaces())
    with TestClient(app) as c:
        c.tmp = tmp_path  # type: ignore[attr-defined]
        yield c


def _photo(tmp: Path, w: int = 160, h: int = 120) -> Path:
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack([xx * 255 // w, yy * 255 // h, np.full_like(xx, 128)], -1).astype(np.uint8)
    p = tmp / "foto.png"
    Image.fromarray(arr).save(p)
    return p


def _add(client: TestClient) -> dict:
    r = client.post("/api/images", json={"paths": [str(_photo(client.tmp))]}, headers=H)
    assert r.status_code == 200, r.text
    return r.json()["images"][0]


def test_token_required(client: TestClient) -> None:
    assert client.get("/api/health").status_code == 401
    assert client.get("/api/health", headers={"X-Pixforge-Token": "wrong"}).status_code == 401
    assert client.get("/api/models", params={"t": TOKEN}).status_code == 200


def test_add_image_info_and_thumb(client: TestClient) -> None:
    info = _add(client)
    assert (info["width"], info["height"]) == (160, 120)
    assert set(info["denoise"]) == {"photo", "clean"}
    r = client.get(f"/api/images/{info['id']}/thumb", headers=H)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    bad = client.post("/api/images", json={"paths": ["C:/does/not/exist.jpg"]}, headers=H)
    assert bad.json()["errors"] and not bad.json()["images"]


@pytest.mark.parametrize("stage", ["original", "denoised", "result"])
def test_preview_returns_requested_size(client: TestClient, stage: str) -> None:
    info = _add(client)
    body = {
        "image_id": info["id"],
        "region": {"x": 20, "y": 10, "w": 80, "h": 60},
        "out_w": 320,
        "out_h": 240,
        "stage": stage,
        "settings": {"denoise": {"mode": "on", "strength": 50}},
    }
    r = client.post("/api/preview", json=body, headers=H)
    assert r.status_code == 200, r.text
    with Image.open(__import__("io").BytesIO(r.content)) as im:
        assert im.size == (320, 240)


def test_zoomed_out_preview_downsamples_the_source(client: TestClient) -> None:
    info = _add(client)
    body = {
        "image_id": info["id"],
        "region": {"x": 0, "y": 0, "w": 160, "h": 120},
        "out_w": 320,
        "out_h": 240,
        "settings": {"scale": 4, "denoise": {"mode": "off"}},
    }
    r = client.post("/api/preview", json=body, headers=H)
    assert r.status_code == 200
    assert r.headers["X-Preview-Kind"] == "approx"
    photo = client.app.state.pipeline.backend("photo")  # type: ignore[attr-defined]
    # output 320 px wide at 4x needs only 80 px of source, not 160
    assert max(w for _, w in photo.inner.calls) <= 128  # 80 px in the 128 bucket, not 192


def test_strongly_zoomed_out_preview_shows_the_original(client: TestClient) -> None:
    """A model on a tiny copy invents structures; the background tiles bring the result."""
    info = _add(client)
    body = {
        "image_id": info["id"],
        "region": {"x": 0, "y": 0, "w": 160, "h": 120},
        "out_w": 160,
        "out_h": 120,
        "settings": {"scale": 4, "denoise": {"mode": "off"}},
    }
    r = client.post("/api/preview", json=body, headers=H)
    assert r.status_code == 200 and r.headers["X-Preview-Kind"] == "original"
    assert "photo" not in client.app.state.pipeline._backends  # type: ignore[attr-defined]


def _wait_tiles(client: TestClient, body: dict, timeout: float = 30) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.post("/api/tiles", json=body, headers=H)
        assert r.status_code == 200, r.text
        d = r.json()
        if d["state"] in ("done", "error"):
            return d
        time.sleep(0.1)
    raise AssertionError(f"tiles not done: {d}")


def _textured(tmp: Path, w: int = 160, h: int = 120) -> Path:
    yy, xx = np.mgrid[0:h, 0:w]
    check = ((xx // 6 + yy // 6) % 2) * 70
    arr = np.stack([xx * 150 // w + check, yy * 150 // h + check, 60 + check], -1)
    p = tmp / "muster.png"
    Image.fromarray(arr.astype(np.uint8)).save(p)
    return p


def _assemble(client: TestClient, d: dict, query: str = "") -> np.ndarray:
    import io

    full = np.zeros((240, 320, 3), np.float32)
    for i, j in d["done"]:
        r = client.get(f"/api/tiles/{d['id']}/{i}/{j}/0{query}", headers=H)
        assert r.status_code == 200
        with Image.open(io.BytesIO(r.content)) as im:
            a = np.asarray(im.convert("RGB"), np.float32) / 255
        y, x = j * 128, i * 128
        full[y : y + a.shape[0], x : x + a.shape[1]] = a
        # reduced pyramid levels exist and halve the size
        r1 = client.get(f"/api/tiles/{d['id']}/{i}/{j}/1{query}", headers=H)
        with Image.open(io.BytesIO(r1.content)) as im1:
            assert abs(im1.size[0] - a.shape[1] / 2) <= 1
    return full


def test_background_tiles_assemble_to_the_export_result(client: TestClient) -> None:
    r = client.post("/api/images", json={"paths": [str(_textured(client.tmp))]}, headers=H)
    info = r.json()["images"][0]
    client.app.state.tiles.tile_size = lambda: 64  # type: ignore[attr-defined]
    settings = {"scale": 2, "denoise": {"mode": "off"}, "sharpen": 60, "detail_mix": 40}
    body = {
        "image_id": info["id"],
        "settings": settings,
        "focus": {"x": 100, "y": 0, "w": 60, "h": 60},
    }
    d = _wait_tiles(client, body)
    assert d["state"] == "done", d
    assert (d["cols"], d["rows"], d["total"], d["done_count"]) == (3, 2, 6, 6)
    # the finishing is applied per request, so it can be switched without a model run
    finished = _assemble(client, d, "?sh=60&dm=40")
    plain = _assemble(client, d)
    pipe = client.app.state.pipeline  # type: ignore[attr-defined]
    img, an = client.app.state.store.image(info["id"])  # type: ignore[attr-defined]
    from pixforge.server.app import SettingsIn

    ref = pipe.process(img, SettingsIn(**settings).to_settings(), analysis=an).image.rgb
    err = np.abs(finished - ref).mean()
    assert err < 0.006, err  # same pixels as the export, up to JPEG (no seams at the tiles)
    assert np.abs(plain - ref).mean() > 2 * err  # and the finishing really made a difference
    # other finishing values: the same finished session, no new model run
    again = client.post(
        "/api/tiles", json={**body, "settings": {**settings, "sharpen": 0, "grain": 30}}, headers=H
    ).json()
    assert again["id"] == d["id"] and again["state"] == "done"


def test_tiles_only_inside_the_crop(client: TestClient) -> None:
    info = _add(client)
    client.app.state.tiles.tile_size = lambda: 64  # type: ignore[attr-defined]
    body = {
        "image_id": info["id"],
        "settings": {"scale": 2, "denoise": {"mode": "off"}, "crop": [40, 10, 40, 40]},
    }
    d = _wait_tiles(client, body)
    assert d["total"] == 2 and sorted(map(tuple, d["done"])) == [(0, 0), (1, 0)]
    assert client.get(f"/api/tiles/{d['id']}/2/1/0", headers=H).status_code == 404


def test_estimate(client: TestClient) -> None:
    info = _add(client)
    r = client.post(
        "/api/estimate", json={"image_id": info["id"], "settings": {"scale": 6}}, headers=H
    )
    d = r.json()
    assert (d["out_w"], d["out_h"]) == (960, 720)
    assert any("neu erzeugt" in w for w in d["warnings"])
    r = client.post(
        "/api/estimate",
        json={"image_id": info["id"], "settings": {"task": "denoise", "crop": [0, 0, 50, 40]}},
        headers=H,
    )
    assert (r.json()["out_w"], r.json()["out_h"]) == (50, 40)


def test_export_job_writes_file_and_reports_progress(client: TestClient) -> None:
    info = _add(client)
    with client.websocket_connect(f"/api/events?t={TOKEN}") as ws:
        assert ws.receive_json()["type"] == "hello"
        r = client.post(
            "/api/jobs",
            json={
                "image_id": info["id"],
                "settings": {"scale": 2, "denoise": {"mode": "off"}},
                "export": {"format": "png", "folder": str(client.tmp), "pattern": "{name}_{scale}"},
            },
            headers=H,
        )
        job_id = r.json()["id"]
        statuses = []
        deadline = time.time() + 20
        while time.time() < deadline:
            ev = ws.receive_json()
            if ev.get("type") == "job" and ev["id"] == job_id:
                statuses.append(ev["status"])
                if ev["status"] in ("done", "error"):
                    break
    assert statuses[-1] == "done", statuses
    out = Path(ev["output"])
    assert out.name == "foto_2x.png" and out.exists()
    with Image.open(out) as im:
        assert im.size == (320, 240)


def test_websocket_rejects_wrong_token(client: TestClient) -> None:
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/api/events?t=nope") as ws:
        ws.receive_json()
