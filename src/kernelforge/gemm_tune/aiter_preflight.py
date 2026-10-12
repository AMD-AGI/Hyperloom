# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Preflight: is the aiter that TUNES the same as the aiter that SERVES?"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


def serve_aiter_path() -> str | None:
    """Realpath of the aiter package ``import aiter`` would resolve to, or None."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("aiter")
    except Exception:  # noqa: BLE001 - unresolvable / broken package means "no serving aiter"
        return None
    if spec is None or not spec.origin:
        return None
    return os.path.realpath(os.path.dirname(spec.origin))


def is_aligned(serve: str, root: str) -> bool:
    """True if the serving aiter and the tuner root come from one installation."""
    s = serve.replace("\\", "/").rstrip("/")
    r = root.replace("\\", "/").rstrip("/")
    if s == r or s.startswith(r + "/"):
        return True
    parent, _, name = r.rpartition("/")
    return bool(parent) and name == "aiter_meta" and s == f"{parent}/aiter"


def classify(serve: str | None, root: str | None, commit: str | None) -> tuple[list[str], list[str]]:
    """Pure decision logic -> (hard_problems, soft_warnings). No I/O."""
    hard: list[str] = []
    soft: list[str] = []
    if serve is None:
        hard.append("serving aiter is not importable (`import aiter` failed)")
    if not root:
        soft.append("AITER_ROOT_DIR unset -> tuner aiter is not pinned to the serving aiter")
    if serve and root and not is_aligned(serve, root):
        hard.append(
            f"MISALIGNED: serving aiter ({serve}) is not the tuner root ({root}); "
            "the tuned CSV may not be dispatchable / may be stale at serve time"
        )
    if not commit:
        soft.append(
            "AITER_COMMIT unset -> tuned-CSV provenance falls back to the installed "
            "aiter distribution version (coarser than a commit)"
        )
    return hard, soft


def _installed_aiter_version() -> str | None:
    """``<dist>==<version>`` for the installed aiter, or None."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:  # noqa: BLE001 - stdlib shape differs on exotic runtimes
        return None
    for dist in ("amd-aiter", "aiter"):
        try:
            found = version(dist)
        except PackageNotFoundError:
            continue
        except Exception:  # noqa: BLE001 - a broken dist-info must not break preflight
            return None
        if found:
            return f"{dist}=={found}"
    return None


def _resolve_root(env: Mapping[str, str]) -> str | None:
    root = env.get("AITER_ROOT_DIR")
    if not root:
        return None
    rp = os.path.realpath(root)
    return rp if Path(rp).is_dir() else None


def collect(env: Mapping[str, str] | None = None) -> dict:
    """Structured alignment status for programmatic use (e.g. the tuner CLI)."""
    e = os.environ if env is None else env
    serve = serve_aiter_path()
    root = _resolve_root(e)
    # Classify on the env var alone -- an operator who wants exact provenance still gets told to set AITER_COMMIT --
    # but record the package-version fallback, so the audit artifact carries a real pin instead of null.
    commit_env = e.get("AITER_COMMIT")
    hard, soft = classify(serve, root, commit_env)
    commit = commit_env or _installed_aiter_version()
    return {
        "serve_aiter": serve,
        "tuner_root": root,
        "aiter_commit": commit,
        "aligned": bool(serve and root and is_aligned(serve, root)),
        "hard": hard,
        "soft": soft,
    }
