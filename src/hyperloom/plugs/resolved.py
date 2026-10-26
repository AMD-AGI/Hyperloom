# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``plugs.resolved.json``: the session's record of which implementation fills each socket."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from hyperloom.plugs.errors import ManifestError
from hyperloom.plugs.manifest import SCHEMA, ResolvedManifest, validate_document

RESOLVED_FILENAME = "plugs.resolved.json"


def resolved_path(session_dir: Path) -> Path:
    """Where ``session_dir`` keeps its resolved manifest."""
    return Path(session_dir) / RESOLVED_FILENAME


def read_resolved(session_dir: Path) -> dict[str, Any] | None:
    """The session's previous selection, or None when it has none. A file that cannot be trusted is refused.

    Falling back to the defaults on a damaged file would silently resume the session on other implementations
    than it started with.
    """
    path = resolved_path(session_dir)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot use {str(path)!r}: {exc}; fix or remove the file to resume") from exc
    return validate_document(raw, str(path))


def write_resolved(session_dir: Path, manifest: ResolvedManifest, fingerprint: str) -> Path:
    """Write ``manifest`` atomically, with the layer that chose each socket and the fingerprint; return the path.

    Configuration is written as laid down, with its ``${VAR}`` references unexpanded, so no secret that the
    environment supplies is copied to disk.
    """
    path = resolved_path(session_dir)
    payload = {
        "schema": SCHEMA,
        "profile": manifest.profile,
        "fingerprint": fingerprint,
        "sockets": {name: {"impl": s.impl, "config": dict(s.config)} for name, s in sorted(manifest.sockets.items())},
        "layers": {name: s.layer for name, s in sorted(manifest.sockets.items())},
    }
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path
