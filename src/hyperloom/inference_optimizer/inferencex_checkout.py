# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve and fetch the InferenceX checkout supported by synthetic and MLPerf runs."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_REPO = "https://github.com/SemiAnalysisAI/InferenceX.git"
# MUST stay in lockstep with INFERENCEX_REF in assets/install.sh.
DEFAULT_REF = "3d5581562f643f9bdeb8410cd924e2c70906c966"


def head_sha(path: Path | str) -> str:
    """Full SHA at ``path``'s HEAD, or "" when it cannot be read."""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (out.stdout or "").strip() if out.returncode == 0 else ""


def _ref_matches(path: Path | str, ref: str) -> bool:
    """Whether the checkout at ``path`` is at ``ref``."""
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", ref or ""):
        return True
    head = head_sha(path)
    if not head:
        return True
    return head.startswith(ref.lower()) or ref.lower().startswith(head)


def checkout_ok(path: Path | str, *, ref: str | None = None) -> bool:
    """True when ``path`` is a usable InferenceX checkout at the expected ref."""
    if not (Path(path) / "benchmarks" / "benchmark_lib.sh").is_file():
        return False
    if ref is None:
        ref = os.environ.get("INFERENCEX_REF") or DEFAULT_REF
    if not ref:
        return True
    return _ref_matches(path, ref)


def destination_name(ref: str) -> str:
    """Per-revision checkout dir name, matching install.sh's ``InferenceX@<sha>``."""
    slug = ref if re.fullmatch(r"[0-9a-fA-F]{7,40}", ref or "") else re.sub(r"[^A-Za-z0-9._-]", "-", ref or "head")
    return f"InferenceX@{slug}"


def clone(dest: Path) -> str | None:
    """Clone InferenceX into ``dest`` (writable), pinned to INFERENCEX_REF."""
    repo = os.environ.get("INFERENCEX_REPO") or DEFAULT_REPO
    ref = os.environ.get("INFERENCEX_REF") or DEFAULT_REF
    dest_str = str(dest)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if re.fullmatch(r"[0-9a-fA-F]{7,40}", ref):
            subprocess.run(["git", "init", "-q", dest_str], check=True, timeout=60)
            subprocess.run(
                ["git", "-C", dest_str, "fetch", "-q", "--depth", "1", repo, ref],
                check=True,
                timeout=600,
            )
            subprocess.run(
                ["git", "-C", dest_str, "checkout", "-q", "FETCH_HEAD"],
                check=True,
                timeout=120,
            )
        else:
            subprocess.run(
                ["git", "clone", "-q", "--depth", "1", "--branch", ref, repo, dest_str],
                check=True,
                timeout=600,
            )
        # ref="" : the tree was just checked out at `ref` by construction, so re-deriving the pin here would only
        # re-read what we wrote.
        if not checkout_ok(dest, ref=""):
            raise OSError(f"clone reported success but {dest_str} is missing benchmarks/benchmark_lib.sh")
        log.info("InferenceX cloned into %s at %s", dest_str, head_sha(dest) or ref)
        return dest_str
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        log.warning("InferenceX clone into %s failed: %s", dest_str, exc)
        shutil.rmtree(dest, ignore_errors=True)
        return None


def detect() -> str:
    """A writable InferenceX checkout at the pinned revision, or "" when there is none.

    Detection order: the Magpie submodule (canonical post-install.sh), then the
    installer's per-revision cache checkout (resolved via ``resolve_dep_dir`` so
    a process that did not inherit INFERENCEX_PATH still finds it), then the
    bare dir. A complete checkout at the wrong revision is refused rather than
    accepted silently.
    """
    from .session.paths import magpie_dir as _magpie_default, resolve_dep_dir as _resolve_dep_dir

    magpie_env = os.environ.get("MAGPIE_PATH")
    magpie_root = Path(magpie_env) if magpie_env else _magpie_default()
    want_ref = os.environ.get("INFERENCEX_REF") or DEFAULT_REF
    for candidate in (
        magpie_root / "InferenceX",
        _resolve_dep_dir(destination_name(want_ref)),
        _resolve_dep_dir("InferenceX"),
    ):
        if checkout_ok(candidate):
            if os.access(candidate, os.W_OK):
                return str(candidate)
            print(
                "Preflight: skipping non-writable auto-detected "
                f"InferenceX checkout at {candidate}; cloning a "
                "writable checkout instead."
            )
        elif (Path(candidate) / "benchmarks" / "benchmark_lib.sh").is_file():
            print(
                f"Preflight: ignoring InferenceX at {candidate}: it is at "
                f"{head_sha(candidate)[:12] or 'an unreadable ref'}, "
                f"not the pinned {want_ref[:12]}."
            )
    return ""
