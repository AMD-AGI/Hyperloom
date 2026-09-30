# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate a literal Python startup overlay directory."""

from __future__ import annotations

import json
from pathlib import Path


def overlay_is_loadable(overlay: str) -> bool:
    """Whether a startup overlay can install an authored kernel."""
    if not overlay:
        return False
    try:
        if not (Path(overlay) / "sitecustomize.py").is_file():
            return False
        manifest = Path(overlay) / "_overlay_manifest.json"
        if not manifest.is_file():
            return True
        spec = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(spec, dict) and bool(spec.get("modules") or spec.get("rebinds") or spec.get("captures"))


def validate_overlay_pythonpath(value: str) -> str:
    """Return one loadable directory, rejecting lists and ambiguous paths."""
    if not value:
        return ""
    path = Path(value)
    if ":" in value or ".." in path.parts or any(char in value for char in "\n\r\0"):
        raise ValueError("overlay_pythonpath must name one literal directory")
    if not overlay_is_loadable(value):
        raise ValueError(f"overlay_pythonpath is not loadable: {value!r}")
    return value
