"""Pixforge command line.

pixforge photo.jpg -o out --model photo --scale 4 --faces
pixforge C:\\Fotos --width 6000 --format tiff --bits 16
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from pixforge import __version__


def _parse_denoise(v: str) -> tuple[str, float | None]:
    v = v.strip().lower()
    if v in ("auto", "off", "on"):
        return v, None
    try:
        return "on", float(v)
    except ValueError:
        raise argparse.ArgumentTypeError("auto, off, on oder 0-100") from None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pixforge",
        description="KI-Hochskalierung für Fotos (HAT, lokal).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("inputs", nargs="*", type=Path, help="Bilder oder Ordner")
    ap.add_argument("-o", "--output", type=Path, help="Zielordner (Standard: wie Quelle)")
    ap.add_argument(
        "--model",
        default="photo",
        help="photo, photo-sharp, clean, max (oder genaue ID, siehe --list-models)",
    )
    size = ap.add_argument_group("Größe")
    size.add_argument("--scale", type=float, default=4.0, help="Faktor, z. B. 2, 4, 6 (Standard 4)")
    size.add_argument("--width", type=int, help="Zielbreite in Pixeln")
    size.add_argument("--height", type=int, help="Zielhöhe in Pixeln")
    adj = ap.add_argument_group("Einstellungen (0-100)")
    adj.add_argument(
        "--denoise",
        type=_parse_denoise,
        default=("auto", None),
        help="auto (Standard), off, on oder Stärke 0-100",
    )
    adj.add_argument("--jpeg", type=float, default=0.0, help="JPEG-Artefakte entfernen")
    adj.add_argument("--sharpen", type=float, default=0.0)
    adj.add_argument("--detail", type=float, default=0.0, help="Detail-Mischung mit Lanczos")
    adj.add_argument("--grain", type=float, default=0.0)
    fc = ap.add_argument_group("Gesichter")
    fc.add_argument("--faces", action="store_true", help="Gesichter verbessern")
    fc.add_argument("--face-model", default="face", help="face (Standard) oder face-gfpgan")
    fc.add_argument("--face-strength", type=float, default=80.0)
    fc.add_argument("--largest-face", action="store_true", help="nur das größte Gesicht")
    ex = ap.add_argument_group("Export")
    ex.add_argument("--format", default="jpeg", choices=["jpeg", "png", "tiff", "webp"])
    ex.add_argument("--quality", type=int, default=92)
    ex.add_argument(
        "--bits",
        type=int,
        choices=[8, 16],
        default=None,
        help="8 oder 16 (PNG/TIFF; Standard: wie Quelle)",
    )
    ex.add_argument("--compression", default="lzw", choices=["lzw", "zip", "none"])
    ex.add_argument("--srgb", action="store_true", help="nach sRGB umrechnen statt Profil behalten")
    ex.add_argument(
        "--pattern", default="{name}_pixforge", help="Dateiname: {name} {scale} {model} {w} {h}"
    )
    ex.add_argument("--exists", default="rename", choices=["rename", "overwrite", "skip"])
    ex.add_argument("--no-exif", action="store_true")
    hw = ap.add_argument_group("Hardware")
    hw.add_argument("--device", default="auto", help="auto, cpu oder gpu:<n>")
    hw.add_argument("--tile", type=int, help="Kachelgröße (Standard: automatisch)")
    ap.add_argument("-r", "--recursive", action="store_true", help="Ordner rekursiv durchsuchen")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=f"Pixforge {__version__}")
    return ap


MODEL_ALIASES = {"clean": "clean-x4", "max": "max-x4"}


def collect_inputs(paths: list[Path], recursive: bool) -> list[Path]:
    from pixforge.core.io import SUPPORTED_EXT

    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            files += sorted(f for f in it if f.is_file() and f.suffix.lower() in SUPPORTED_EXT)
        elif p.is_file():
            files.append(p)
        else:
            print(f"Nicht gefunden: {p}", file=sys.stderr)
    return files


def _fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}" if sec >= 60 else f"{sec} s"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    from pixforge.models import registry

    if args.list_models:
        for m in registry.all_models():
            inst = ", ".join(registry.installed_precisions(m.id)) or "nicht installiert"
            nc = "  [nicht-kommerziell]" if m.noncommercial else ""
            print(f"{m.id:13} {m.kind:8} {m.display_name():32} {inst}{nc}")
        return 0
    if args.list_devices:
        from pixforge.backends.onnx_backend import list_devices

        for d in list_devices():
            extra = f", {d.vram_mb} MB" if d.vram_mb else ""
            idx = f"gpu:{d.device_index}" if d.kind == "gpu" else "cpu"
            print(f"{idx:6} {d.name}{extra}")
        return 0
    files = collect_inputs(args.inputs, args.recursive)
    if not files:
        print("Keine Bilder angegeben. Hilfe: pixforge --help", file=sys.stderr)
        return 2

    from pixforge.core import naming
    from pixforge.core.io import FORMAT_EXT, ExportOptions, ImageLoadError, load_image, save_image
    from pixforge.core.pipeline import DenoiseSettings, FaceSettings, Pipeline, Settings
    from pixforge.core.tiling import Cancelled

    mode, strength = args.denoise
    settings = Settings(
        model=MODEL_ALIASES.get(args.model, args.model),
        scale=args.scale,
        target_width=args.width,
        target_height=args.height,
        denoise=DenoiseSettings(mode=mode, strength=strength),
        jpeg_strength=args.jpeg,
        sharpen=args.sharpen,
        detail_mix=args.detail,
        grain=args.grain,
        faces=FaceSettings(
            enabled=args.faces,
            model=args.face_model,
            strength=args.face_strength,
            only_largest=args.largest_face,
        ),
        tile=args.tile,
    )
    pipe = Pipeline(device=args.device)
    failed = 0
    for i, src in enumerate(files, 1):
        prefix = f"[{i}/{len(files)}] {src.name}"
        try:
            img = load_image(src)
            t0 = time.perf_counter()

            def progress(
                stage: str, frac: float, msg: str, _t0: float = t0, _prefix: str = prefix
            ) -> None:
                el = time.perf_counter() - _t0
                eta = f"  noch ~{_fmt_time(el / frac - el)}" if frac > 0.03 else ""
                print(f"\r{_prefix}  {stage:8} {frac * 100:5.1f} %{eta}      ", end="", flush=True)

            res = pipe.process(img, settings, progress=progress)
            fmt = args.format
            bits = args.bits or (16 if img.bit_depth > 8 and fmt in ("png", "tiff") else 8)
            opts = ExportOptions(
                format=fmt,
                quality=args.quality,
                bit_depth=bits,
                compression=args.compression,
                color="srgb" if args.srgb else "embed",
                keep_exif=not args.no_exif,
            )
            W, H = res.image.size
            out = naming.output_path(
                src,
                args.output,
                args.pattern,
                FORMAT_EXT[fmt],
                res.factor,
                res.model,
                W,
                H,
                args.exists,
            )
            print()
            if out is None:
                print(f"  übersprungen (Datei existiert): {src.name}")
                continue
            save_image(res.image, out, opts)
            info = [f"{W}x{H}", res.model, f"{_fmt_time(res.seconds)}", res.device]
            if res.denoise_strength:
                info.append(f"entrauscht {res.denoise_strength:.0f}")
            elif res.analysis.denoise.reason != "noisy":
                info.append(res.analysis.denoise.message())
            if res.faces_restored:
                info.append(f"{res.faces_restored} Gesicht(er)")
            print(f"  -> {out}  ({', '.join(info)})")
            for w in res.warnings:
                print(f"  Hinweis: {w}")
        except Cancelled:
            print("\nAbgebrochen.")
            return 130
        except (ImageLoadError, registry.ModelNotInstalled, ValueError) as exc:
            failed += 1
            print(f"\n{prefix}: {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nAbgebrochen.")
            return 130
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
