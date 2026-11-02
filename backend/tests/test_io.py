"""Loading and saving: formats, bit depth, orientation, ICC and EXIF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from pixforge.core import naming
from pixforge.core.color import is_srgb, srgb_profile_bytes
from pixforge.core.io import EXIF_ORIENTATION, EXIF_SOFTWARE, ExportOptions, load_image, save_image


def _gradient(h: int = 40, w: int = 60) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([xx / w, yy / h, np.full_like(xx, 0.5)], -1)


def test_jpeg_orientation_is_applied(tmp_path: Path) -> None:
    arr = (_gradient() * 255).astype(np.uint8)
    exif = Image.Exif()
    exif[EXIF_ORIENTATION] = 6  # rotate 90 CW for display
    p = tmp_path / "rot.jpg"
    Image.fromarray(arr).save(p, quality=95, exif=exif.tobytes())
    img = load_image(p)
    assert img.size == (40, 60)  # width and height swapped
    out = tmp_path / "out.jpg"
    save_image(img, out, ExportOptions(format="jpeg"))
    with Image.open(out) as im:
        ex = im.getexif()
        assert ex[EXIF_ORIENTATION] == 1
        assert str(ex[EXIF_SOFTWARE]).startswith("Pixforge")
        assert im.info.get("icc_profile")


@pytest.mark.parametrize(
    ("fmt", "bits"), [("png", 16), ("png", 8), ("tiff", 16), ("tiff", 8), ("webp", 8)]
)
def test_roundtrip_precision(tmp_path: Path, fmt: str, bits: int) -> None:
    src = tmp_path / "src.png"
    Image.fromarray((_gradient() * 255).astype(np.uint8)).save(src)
    img = load_image(src)
    img.rgb = _gradient()  # finer than 8 bit
    out = tmp_path / f"o.{fmt}"
    save_image(img, out, ExportOptions(format=fmt, bit_depth=bits, webp_lossless=True))
    back = load_image(out)
    tol = 1 / 65535 * 2 if bits == 16 else 1 / 255
    assert back.bit_depth == (16 if bits == 16 else 8)
    assert np.abs(back.rgb - img.rgb).max() <= tol + 1e-6


def test_png16_keeps_icc_and_alpha(tmp_path: Path) -> None:
    src = tmp_path / "a.png"
    Image.fromarray((_gradient() * 255).astype(np.uint8)).save(src)
    img = load_image(src)
    img.alpha = np.linspace(0, 1, 40 * 60, dtype=np.float32).reshape(40, 60)
    out = tmp_path / "a16.png"
    save_image(img, out, ExportOptions(format="png", bit_depth=16))
    with Image.open(out) as im:
        assert im.info.get("icc_profile") == srgb_profile_bytes()
    back = load_image(out)
    assert back.alpha is not None and np.abs(back.alpha - img.alpha).max() < 1e-4


def test_non_srgb_8bit_is_converted(tmp_path: Path) -> None:
    lab_like = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    assert not is_srgb(lab_like)
    assert is_srgb(srgb_profile_bytes()) and is_srgb(None)


def test_naming(tmp_path: Path) -> None:
    src = tmp_path / "Foto 1.jpg"
    name = naming.expand("{name}_{scale}_{model}_{w}x{h}", src, 4.0, "photo", 800, 600)
    assert name == "Foto 1_4x_photo_800x600"
    p1 = naming.output_path(src, tmp_path, "{name}_pixforge", ".jpg", 4, "photo", 1, 1)
    assert p1 is not None
    p1.write_bytes(b"x")
    p2 = naming.output_path(src, tmp_path, "{name}_pixforge", ".jpg", 4, "photo", 1, 1)
    assert p2 is not None and p2.name == "Foto 1_pixforge_2.jpg"
    assert (
        naming.output_path(src, tmp_path, "{name}_pixforge", ".jpg", 4, "photo", 1, 1, "skip")
        is None
    )
    assert naming.expand('a<b>:"c', src, 2, "m", 1, 1) == "a_b___c"


def _gray_icc(gamma: float = 2.2) -> bytes:
    """A minimal ICC v2 grayscale display profile (desc, wtpt, kTRC), built here so that no
    third-party profile has to live in the repository."""
    import struct

    def tag_desc(text: str) -> bytes:
        a = text.encode("ascii") + b"\0"
        return b"desc\0\0\0\0" + struct.pack(">I", len(a)) + a + b"\0" * 4 + b"\0" * 4 + b"\0" * 71

    wtpt = b"XYZ \0\0\0\0" + struct.pack(">iii", 63190, 65536, 54061)  # D50 in s15.16
    ktrc = b"curv\0\0\0\0" + struct.pack(">IH", 1, round(gamma * 256)) + b"\0\0"
    tags = [(b"desc", tag_desc("Test Gray")), (b"wtpt", wtpt), (b"kTRC", ktrc)]
    offset = 128 + 4 + 12 * len(tags)
    table, data = b"", b""
    for sig, body in tags:
        body += b"\0" * (-len(body) % 4)
        table += sig + struct.pack(">II", offset + len(data), len(body))
        data += body
    size = offset + len(data)
    header = (
        struct.pack(">I", size)
        + b"lcms"
        + struct.pack(">I", 0x02100000)
        + b"mntr"
        + b"GRAY"
        + b"XYZ "
        + b"\0" * 12
        + b"acsp"
        + b"\0" * 28  # platform, flags, manufacturer, model, attributes, intent
        + struct.pack(">iii", 63190, 65536, 54061)
        + b"\0" * 48
    )
    assert len(header) == 128
    return header + struct.pack(">I", len(tags)) + table + data


def test_grayscale_jpeg_with_gray_profile_opens(tmp_path: Path) -> None:
    """Black-and-white JPEGs often carry a gray profile; it used to fail the RGB transform."""
    ramp = np.tile(np.linspace(0, 255, 64).astype(np.uint8), (32, 1))
    p = tmp_path / "sw.jpg"
    Image.fromarray(ramp, "L").save(p, quality=95, icc_profile=_gray_icc())
    img = load_image(p)
    assert img.rgb.shape == (32, 64, 3) and img.icc is None
    assert img.meta.get("converted_from") == "Test Gray"
    assert np.abs(img.rgb[..., 0] - img.rgb[..., 2]).max() < 0.02  # stays neutral gray
    assert np.all(np.diff(img.rgb[16, :, 1]) >= -0.01)  # tone order kept


def test_unusable_profile_is_ignored(tmp_path: Path) -> None:
    lab = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    p = tmp_path / "lab.jpg"
    Image.fromarray(np.full((16, 16, 3), 128, np.uint8)).save(p, icc_profile=lab)
    img = load_image(p)
    assert img.rgb.shape == (16, 16, 3)
    assert "ignoriert" in str(img.meta.get("color_warning"))
