"""Output file names: pattern expansion and collision handling."""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_PATTERN = "{name}_pixforge"
_ALLOWED = {"name", "scale", "model", "w", "h"}
_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def format_scale(f: float) -> str:
    return f"{f:g}x"


def expand(pattern: str, src: Path, scale: float, model: str, w: int, h: int) -> str:
    """Expand {name}, {scale}, {model}, {w}, {h}. Unknown placeholders are kept literally."""
    values = {
        "name": src.stem,
        "scale": format_scale(scale),
        "model": model,
        "w": str(w),
        "h": str(h),
    }

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        return values[key] if key in _ALLOWED else m.group(0)

    name = re.sub(r"\{(\w+)\}", sub, pattern or DEFAULT_PATTERN)
    name = _INVALID.sub("_", name).strip(" .")
    return name or src.stem


def output_path(
    src: Path,
    out_dir: Path | None,
    pattern: str,
    ext: str,
    scale: float,
    model: str,
    w: int,
    h: int,
    on_exists: str = "rename",
) -> Path | None:
    """Target path, or None if the file exists and ``on_exists`` is "skip".

    on_exists: "rename" (append _2, _3 ...), "overwrite" or "skip".
    """
    folder = out_dir if out_dir is not None else src.parent
    base = expand(pattern, src, scale, model, w, h)
    path = folder / f"{base}{ext}"
    if not path.exists() or on_exists == "overwrite":
        return path
    if on_exists == "skip":
        return None
    n = 2
    while (folder / f"{base}_{n}{ext}").exists():
        n += 1
    return folder / f"{base}_{n}{ext}"
