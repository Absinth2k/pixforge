"""Per-user application directories."""

from __future__ import annotations

import os
from pathlib import Path


def app_home() -> Path:
    """%LOCALAPPDATA%\\Pixforge, overridable with PIXFORGE_HOME (tests, portable mode)."""
    override = os.environ.get("PIXFORGE_HOME")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "Pixforge"


def models_dir() -> Path:
    override = os.environ.get("PIXFORGE_MODELS")
    return Path(override) if override else app_home() / "models"


def logs_dir() -> Path:
    return app_home() / "logs"
