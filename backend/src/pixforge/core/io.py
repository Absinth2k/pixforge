"""Image loading and saving with metadata (EXIF, ICC) handling.

Pixel data inside Pixforge is always float32 RGB in 0..1, shape (h, w, 3), plus an optional
float32 alpha plane (h, w). Orientation from EXIF is applied on load, so downstream code never
has to think about it; the saved EXIF then says "normal orientation".
"""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pixforge import __version__
from pixforge.core import color

RAW_EXT = {".cr2", ".cr3", ".nef", ".nrw", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}
HEIF_EXT = {".heic", ".heif", ".avif", ".hif"}
STD_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
SUPPORTED_EXT = RAW_EXT | HEIF_EXT | STD_EXT

EXIF_ORIENTATION = 0x0112
EXIF_SOFTWARE = 0x0131
EXIF_MAKE = 0x010F
EXIF_MODEL = 0x0110
EXIF_IFD = 0x8769
EXIF_ISO = 0x8827
EXIF_FOCAL = 0x920A

PHONE_MAKES = (
    "apple",
    "samsung",
    "google",
    "xiaomi",
    "huawei",
    "oneplus",
    "oppo",
    "vivo",
    "motorola",
    "nokia",
    "sony mobile",
    "honor",
    "realme",
    "fairphone",
)


class ImageLoadError(RuntimeError):
    """User-presentable load error."""


@dataclass
class ImageData:
    rgb: np.ndarray  # float32 (h, w, 3), 0..1, working color space
    alpha: np.ndarray | None = None  # float32 (h, w)
    bit_depth: int = 8  # source bit depth per channel (8 or 16)
    icc: bytes | None = None  # profile of ``rgb`` if it is NOT sRGB, else None
    source_icc: bytes | None = None  # profile embedded in the source file
    exif: bytes | None = None  # EXIF with orientation already reset to 1
    path: Path | None = None
    format: str = ""
    meta: dict[str, object] = field(default_factory=dict)  # iso, make, model, is_phone

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]


# ------------------------------------------------------------------------------ loading
def load_image(path: str | Path) -> ImageData:
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"Datei nicht gefunden: {path}")
    ext = path.suffix.lower()
    try:
        if ext in RAW_EXT:
            return _load_raw(path)
        if ext in HEIF_EXT:
            import pi_heif

            pi_heif.register_heif_opener()
        return _load_pil(path)
    except ImageLoadError:
        raise
    except Exception as exc:  # pragma: no cover - depends on file
        raise ImageLoadError(f"Bild konnte nicht gelesen werden ({path.name}): {exc}") from exc


def _load_pil(path: Path) -> ImageData:
    with Image.open(path) as im:
        fmt = (im.format or path.suffix.lstrip(".")).upper()
        icc = im.info.get("icc_profile")
        exif = im.getexif()
        orientation = int(exif.get(EXIF_ORIENTATION, 1) or 1)
        meta = _exif_meta(exif)
        sixteen = im.mode in ("I;16", "I;16B", "I;16L", "I", "RGB;16", "RGBA;16") or (
            fmt in ("PNG", "TIFF") and _is_16bit_file(path)
        )
        if sixteen and fmt in ("PNG", "TIFF"):
            rgb, alpha = _decode_16bit_cv2(path)
            bit_depth = 16
        else:
            rgb, alpha = _pil_to_float(im)
            bit_depth = 8
    if orientation != 1:
        rgb = _apply_orientation(rgb, orientation)
        alpha = _apply_orientation(alpha, orientation) if alpha is not None else None
    exif_bytes = _reset_orientation(exif)

    data = ImageData(
        rgb=rgb,
        alpha=alpha,
        bit_depth=bit_depth,
        exif=exif_bytes,
        path=path,
        format=fmt,
        source_icc=icc,
        meta=meta,
    )
    color.to_working_space(data)
    return data


def _load_raw(path: Path) -> ImageData:
    try:
        import rawpy
    except ImportError as exc:  # pragma: no cover
        raise ImageLoadError("RAW-Unterstützung ist nicht installiert") from exc
    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=True, output_bps=16, output_color=rawpy.ColorSpace.sRGB
        )
    rgb = rgb16.astype(np.float32) / 65535.0
    meta: dict[str, object] = {"raw": True}
    try:  # LibRaw does not expose EXIF; Pillow can read it from DNG/TIFF-based RAWs
        with Image.open(path) as im:
            meta |= _exif_meta(im.getexif())
    except Exception:
        pass
    return ImageData(rgb=rgb, bit_depth=16, path=path, format="RAW", meta=meta)


def _is_16bit_file(path: Path) -> bool:
    arr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    return arr is not None and arr.dtype == np.uint16


def _decode_16bit_cv2(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    arr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ImageLoadError(f"16-Bit-Bild konnte nicht gelesen werden: {path.name}")
    arr = arr.astype(np.float32) / (65535.0 if arr.dtype == np.uint16 else 255.0)
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2), None
    if arr.shape[2] == 4:
        return np.ascontiguousarray(arr[..., 2::-1]), np.ascontiguousarray(arr[..., 3])
    return np.ascontiguousarray(arr[..., ::-1]), None


def _pil_to_float(im: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    if im.mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
    if im.mode == "CMYK":
        im = im.convert("RGB")
    if im.mode in ("L", "I", "F"):
        im = im.convert("RGB")
    if im.mode == "LA":
        im = im.convert("RGBA")
    alpha = None
    if im.mode == "RGBA":
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return np.ascontiguousarray(arr[..., :3]), np.ascontiguousarray(arr[..., 3])
    if im.mode != "RGB":
        im = im.convert("RGB")
    return np.asarray(im, dtype=np.float32) / 255.0, alpha


def _apply_orientation(arr: np.ndarray, orientation: int) -> np.ndarray:
    ops = {
        2: lambda a: a[:, ::-1],
        3: lambda a: a[::-1, ::-1],
        4: lambda a: a[::-1],
        5: lambda a: np.swapaxes(a, 0, 1),
        6: lambda a: np.swapaxes(a, 0, 1)[:, ::-1],
        7: lambda a: np.swapaxes(a, 0, 1)[::-1, ::-1],
        8: lambda a: np.swapaxes(a, 0, 1)[::-1],
    }
    op = ops.get(orientation)
    return np.ascontiguousarray(op(arr)) if op else arr


def _reset_orientation(exif: Image.Exif) -> bytes | None:
    if not len(exif):
        return None
    exif[EXIF_ORIENTATION] = 1
    try:
        return exif.tobytes()
    except Exception:  # broken maker notes etc.
        return None


def _exif_meta(exif: Image.Exif) -> dict[str, object]:
    meta: dict[str, object] = {}
    make = str(exif.get(EXIF_MAKE, "") or "").strip("\x00 ").strip()
    model = str(exif.get(EXIF_MODEL, "") or "").strip("\x00 ").strip()
    if make:
        meta["make"] = make
    if model:
        meta["model"] = model
    try:
        sub = exif.get_ifd(EXIF_IFD)
        iso = sub.get(EXIF_ISO)
        if isinstance(iso, tuple):
            iso = iso[0]
        if iso:
            meta["iso"] = int(iso)
        focal = sub.get(EXIF_FOCAL)
        if focal:
            meta["focal_mm"] = float(focal)
    except Exception:
        pass
    low = make.lower()
    meta["is_phone"] = any(low.startswith(p) for p in PHONE_MAKES) or (
        isinstance(meta.get("focal_mm"), float) and float(meta["focal_mm"]) < 10.0  # type: ignore[arg-type]
    )
    return meta


# ------------------------------------------------------------------------------ saving
@dataclass
class ExportOptions:
    format: str = "jpeg"  # jpeg | png | tiff | webp
    quality: int = 92  # jpeg / webp
    bit_depth: int = 8  # png / tiff: 8 or 16
    compression: str = "lzw"  # tiff: lzw | zip | none
    webp_lossless: bool = False
    color: str = "embed"  # embed: keep working profile + embed ICC; srgb: convert to sRGB
    keep_exif: bool = True


FORMAT_EXT = {"jpeg": ".jpg", "png": ".png", "tiff": ".tif", "webp": ".webp"}


def save_image(data: ImageData, path: str | Path, opts: ExportOptions | None = None) -> Path:
    opts = opts or ExportOptions()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb, icc = color.for_export(data, opts.color)
    exif = _export_exif(data.exif) if opts.keep_exif else None
    fmt = opts.format.lower()
    bit_depth = opts.bit_depth if fmt in ("png", "tiff") else 8
    alpha = data.alpha if fmt in ("png", "tiff", "webp") else None

    if fmt in ("jpeg", "jpg"):
        if data.alpha is not None:  # flatten on white like most viewers do
            a = data.alpha[..., None]
            rgb = rgb * a + (1.0 - a)
        im = Image.fromarray(_to_u8(rgb), "RGB")
        kwargs: dict[str, object] = {"quality": int(opts.quality), "optimize": True}
        kwargs["subsampling"] = 0 if opts.quality >= 90 else 2
        _save_pil(im, path, "JPEG", icc, exif, **kwargs)
    elif fmt == "webp":
        im = _pil_rgb_a(rgb, alpha)
        _save_pil(
            im,
            path,
            "WEBP",
            icc,
            exif,
            quality=int(opts.quality),
            lossless=opts.webp_lossless,
            method=5,
        )
    elif fmt == "png":
        if bit_depth == 16:
            _save_png16(rgb, alpha, path, icc, exif)
        else:
            _save_pil(_pil_rgb_a(rgb, alpha), path, "PNG", icc, exif, compress_level=6)
    elif fmt in ("tiff", "tif"):
        _save_tiff(rgb, alpha, path, bit_depth, opts.compression, icc, data)
    else:
        raise ValueError(f"unbekanntes Format {opts.format!r}")
    return path


def _to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _to_u16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def _pil_rgb_a(rgb: np.ndarray, alpha: np.ndarray | None) -> Image.Image:
    if alpha is None:
        return Image.fromarray(_to_u8(rgb), "RGB")
    return Image.fromarray(np.dstack([_to_u8(rgb), _to_u8(alpha)]), "RGBA")


def _save_pil(
    im: Image.Image, path: Path, fmt: str, icc: bytes | None, exif: bytes | None, **kwargs: object
) -> None:
    if icc:
        kwargs["icc_profile"] = icc
    if exif:
        kwargs["exif"] = exif
    im.save(path, fmt, **kwargs)


def _export_exif(src: bytes | None) -> bytes:
    exif = Image.Exif()
    if src:
        try:
            exif.load(src)
        except Exception:
            exif = Image.Exif()
    exif[EXIF_ORIENTATION] = 1
    exif[EXIF_SOFTWARE] = f"Pixforge {__version__}"
    for tag in (0xA002, 0xA003):  # PixelX/YDimension describe the old size; drop them
        try:
            ifd = exif.get_ifd(EXIF_IFD)
            ifd.pop(tag, None)
        except Exception:
            pass
    return exif.tobytes()


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _save_png16(
    rgb: np.ndarray, alpha: np.ndarray | None, path: Path, icc: bytes | None, exif: bytes | None
) -> None:
    arr = _to_u16(rgb)[..., ::-1]
    if alpha is not None:
        arr = np.dstack([arr, _to_u16(alpha)])
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(arr), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise OSError("PNG-Kodierung fehlgeschlagen")
    png = buf.tobytes()
    extra = b""
    if icc:
        extra += _png_chunk(b"iCCP", b"ICC Profile\x00\x00" + zlib.compress(icc))
    if exif:
        raw = exif[6:] if exif.startswith(b"Exif\x00\x00") else exif
        extra += _png_chunk(b"eXIf", raw)
    ihdr_end = 8 + 4 + 4 + 13 + 4  # signature + IHDR chunk
    path.write_bytes(png[:ihdr_end] + extra + png[ihdr_end:])


def _save_tiff(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    path: Path,
    bit_depth: int,
    compression: str,
    icc: bytes | None,
    data: ImageData,
) -> None:
    import tifffile

    conv = _to_u16 if bit_depth == 16 else _to_u8
    arr = conv(rgb)
    extrasamples = None
    if alpha is not None:
        arr = np.dstack([arr, conv(alpha)])
        extrasamples = [2]  # unassociated alpha
    comp = {"lzw": "lzw", "zip": "zlib", "none": None}.get(compression, "lzw")
    extratags = []
    if icc:
        extratags.append((34675, "B", len(icc), icc, True))
    meta_make = str(data.meta.get("make", ""))
    meta_model = str(data.meta.get("model", ""))
    if meta_make:
        extratags.append((271, "s", 0, meta_make, True))
    if meta_model:
        extratags.append((272, "s", 0, meta_model, True))
    tifffile.imwrite(
        path,
        arr,
        photometric="rgb",
        compression=comp,
        software=f"Pixforge {__version__}",
        extrasamples=extrasamples,
        extratags=extratags,
        metadata=None,
        predictor=comp is not None,
    )


def thumbnail_bytes(rgb: np.ndarray, max_side: int = 256, quality: int = 85) -> bytes:
    """Small JPEG for UI thumbnails."""
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    out = io.BytesIO()
    Image.fromarray(_to_u8(small)).save(out, "JPEG", quality=quality)
    return out.getvalue()
