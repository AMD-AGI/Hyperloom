"""Session manifest writer (DESIGN v0.6.1 §23).

The manifest is the **first** file written to a session directory after
``make_session_dir()`` runs and is the canonical session-resume tag.
Atomic write via tmp + ``os.replace``.

Why this lives in its own module:

* Separation of concerns — ``paths.py`` owns where things live;
  ``manifest.py`` owns what's in the resume tag.
* Avoids a hard dependency from ``paths`` (called by everything) on
  ``argparse`` and Python version helpers.

Schema v1::

    {
      "schema_version": 1,
      "session_id":     "<UTC_YYYYMMDDTHHMMSSZ>_<uuid8>",
      "created_at_utc": "...",
      "model_path":     "...",
      "model_name":     "...",
      "framework":      "sglang|vllm",
      "gpu_type":       "mi300x|mi325x|mi355x|''",
      "tp":             N or null,
      "workload":       {"isl":..., "osl":..., "max_model_len":...,
                         "precision":..., "conc":...},
      "objective":      {"kind":"gain_pct|tput|baseline|time_only",
                         "value":...},
      "max_minutes":    N,
      "code_revision":  "<git sha or empty>",
      "pid":            N,
      "host":           "..."
    }
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .session_paths import manifest_path

SCHEMA_VERSION = 1


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_revision() -> str:
    """Best-effort short git SHA of the repo containing this package; empty on failure."""
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        return ""


def _objective_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Mirror cli._run_optimize's objective derivation, without importing it."""
    if getattr(args, "target_gain", None):
        return {"kind": "gain_pct", "value": float(args.target_gain)}
    if getattr(args, "target_tput", None):
        return {"kind": "tput", "value": float(args.target_tput)}
    if getattr(args, "target_baseline_dir", None):
        return {"kind": "baseline", "value": str(args.target_baseline_dir)}
    return {"kind": "time_only", "value": None}


def build_session_id(model_name: str = "") -> str:
    """Derive an internal session_id label.

    The label is **not** used for any path computation (paths are fixed
    at ``/workspace/hyperloom``); it only goes into manifest.json,
    SharedState.session_id, and log/report metadata so multiple
    archived sessions are distinguishable.
    """
    stem = (model_name or "session").strip().replace("/", "_") or "session"
    return f"{stem}_{_utc_now_compact()}_{uuid.uuid4().hex[:8]}"


def build_manifest(
    session_dir: Path,
    *,
    args: argparse.Namespace | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    model_path = ""
    model_name = ""
    framework = os.environ.get("FRAMEWORK", "")
    gpu_type = os.environ.get("GPU_TYPE", "")
    workload: dict[str, Any] = {
        "isl": int(os.environ["ISL"]) if os.environ.get("ISL", "").strip().isdigit() else None,
        "osl": int(os.environ["OSL"]) if os.environ.get("OSL", "").strip().isdigit() else None,
        "max_model_len": int(os.environ["MAX_MODEL_LEN"])
            if os.environ.get("MAX_MODEL_LEN", "").strip().isdigit() else None,
        "precision": os.environ.get("PRECISION", "") or None,
        "conc": int(os.environ["CONC"]) if os.environ.get("CONC", "").strip().isdigit() else None,
    }
    tp = int(os.environ["TP"]) if os.environ.get("TP", "").strip().isdigit() else None
    if args is not None:
        if getattr(args, "model", None):
            model_path = str(args.model)
            model_name = Path(model_path).name
        if getattr(args, "framework", None):
            framework = str(args.framework)
        if getattr(args, "gpu_type", None):
            gpu_type = str(args.gpu_type)
        if getattr(args, "isl", None) is not None:
            workload["isl"] = int(args.isl)
        if getattr(args, "osl", None) is not None:
            workload["osl"] = int(args.osl)
        if getattr(args, "precision", None):
            workload["precision"] = str(args.precision)
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id":     session_id or build_session_id(model_name),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_dir":    str(session_dir),
        "model_path":     model_path,
        "model_name":     model_name,
        "framework":      framework or "sglang",
        "gpu_type":       gpu_type,
        "tp":             tp,
        "workload":       workload,
        "objective":      _objective_summary(args) if args is not None else {"kind": "time_only", "value": None},
        "max_minutes":    int((getattr(args, "max_hours", 0) or 0) * 60) if args is not None else 0,
        "code_revision":  _git_revision(),
        "pid":            os.getpid(),
        "host":           platform.node() or socket.gethostname() or "",
    }


def write_manifest(
    session_dir: Path,
    *,
    args: argparse.Namespace | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Atomically write ``manifest.json`` under session_dir.

    Returns the manifest dict (so the caller can echo it / reuse the
    derived session_id label without re-reading the file).
    """
    sd = Path(session_dir)
    manifest = build_manifest(sd, args=args, session_id=session_id)
    target = manifest_path(sd)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    tmp_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_path, target)
    return manifest


def load_manifest(session_dir: Path) -> dict[str, Any]:
    """Read ``manifest.json`` for an existing session.

    Raises ``FileNotFoundError`` if the file is missing — that signal
    is what ``--resume`` uses to refuse a fresh sandbox.
    """
    p = manifest_path(Path(session_dir))
    if not p.exists():
        raise FileNotFoundError(
            f"manifest.json not found under {session_dir} — "
            f"the session was never initialised; cannot --resume"
        )
    with p.open(encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "SCHEMA_VERSION",
    "build_manifest",
    "build_session_id",
    "load_manifest",
    "write_manifest",
]
