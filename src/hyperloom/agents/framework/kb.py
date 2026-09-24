# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knowledge-base selector + contributor for framework-agent."""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path


_log = logging.getLogger(__name__)


# Per-framework KB partition root under ``<KB_ROOT>/framework_optimization/``.
_FRAMEWORK_OPTIMIZATION_ROOT: str = "framework_optimization"

# ---------------------------------------------------------------------------
# PR ledger vocabulary — single source of truth shared with kb_writeback.py
# (orchestrator cannot import from this package, so the constants must live here).
# ---------------------------------------------------------------------------

#: Append-log filename inside the framework_optimization/ directory.
#: Must remain stable; kb_writeback.LESSONS_FILE and fa-CLI readers match it.
LESSONS_FILE: str = "lessons.jsonl"

#: Allowed ``outcome`` values for a PR ledger row.
OUTCOME_INTEGRATED: str = "integrated"
OUTCOME_ALREADY_PRESENT: str = "already_present"
OUTCOME_REVERTED_SMOKE_FAIL: str = "reverted_smoke_fail"
OUTCOME_REJECTED_APPLY_FAIL: str = "rejected_apply_fail"
OUTCOME_REVERTED_SWITCH_OFF_PARITY: str = "reverted_switch_off_parity"
OUTCOME_REVERTED_PARITY_INCONCLUSIVE: str = "reverted_parity_inconclusive"
ALLOWED_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_INTEGRATED,
        OUTCOME_REVERTED_SMOKE_FAIL,
        OUTCOME_REJECTED_APPLY_FAIL,
        OUTCOME_ALREADY_PRESENT,
        OUTCOME_REVERTED_SWITCH_OFF_PARITY,
        OUTCOME_REVERTED_PARITY_INCONCLUSIVE,
    }
)

#: The only supported override for the mutable KB root; both this module and
#: ``kb_writeback`` honour it. It reaches the process through the
#: ``INFERENCE_OPTIMIZER_`` prefix rule in the ``common/env_safety`` dotenv
#: allowlist, which is a prefix rule rather than an entry for this name.
KB_ROOT_ENV: str = "INFERENCE_OPTIMIZER_FA_KB_PATH"

#: Workspace subdirectory holding this KB. Deliberately not ``kb``: that is the
#: legacy recipe root (``inference_optimizer.cli.kb._legacy_recipe_root``, still
#: read by the one-time recipe migration), and ``list_domains`` reports every
#: directory under this root as a framework domain, so sharing it would surface
#: recipe trees as framework domains. The current recipe root is
#: ``<workspace>/knowledge`` and never collided.
_MUTABLE_KB_DIRNAME: str = "framework-kb"

#: Where the writer put this partition before it was given its own directory.
#: Same value as ``inference_optimizer.cli.kb._legacy_recipe_root``'s leaf, which
#: this package cannot import; the guard test asserts they still agree.
_LEGACY_WORKSPACE_KB_DIRNAME: str = "kb"

#: Workspace root when ``USER_DATA_PATH`` is unset. Mirrors
#: ``session.paths.DEFAULT_SESSION_DIR``, which this package cannot import:
#: the ``fa`` CLI runs standalone and must not depend on inference_optimizer.
_DEFAULT_WORKSPACE_ROOT: str = "/workspace/hyperloom"
_POD_LOCAL_WORKSPACE: str = "/workspace"


def _default_workspace_root() -> str:
    """Container images ship a writable ``/workspace``; bare metal off root has neither it nor permission to create it, so fall back to the caller's dir."""
    probe = _POD_LOCAL_WORKSPACE
    while not os.path.exists(probe) and probe != os.path.dirname(probe):
        probe = os.path.dirname(probe)
    if os.access(probe, os.W_OK):
        return _DEFAULT_WORKSPACE_ROOT
    return os.path.join(os.getcwd(), "session")


#: Withdrawn override. Only the reader honoured it, so setting it split the KB
#: in two. ``FRAMEWORK_AGENT_ROOT`` is deliberately absent: it means "where
#: this skill is installed", is used for other purposes, and never reached the
#: reader anyway because no installer exports it.
_REMOVED_KB_ROOT_ENV: str = "FRAMEWORK_AGENT_KB_DIR"


def prepare_kb_environment() -> None:
    """Start-up sequence for the framework KB: report the environment, then migrate."""
    try:
        check_kb_configuration()
        migrate_legacy_partition_once()
    except Exception:  # noqa: BLE001 — start-up for an advisory KB may not fail a run
        _log.warning("FRAMEWORK KB: start-up preparation failed; continuing without it", exc_info=True)


def check_kb_configuration() -> None:
    """Report an environment naming a KB variable this build no longer reads."""
    if not os.environ.get(_REMOVED_KB_ROOT_ENV, "").strip():
        return
    _log.warning(
        "FRAMEWORK KB: %s is set but no longer read. It only ever redirected the reader, which is "
        "how reads and writes came to point at different places; it is now ignored and this KB "
        "resolves to %s. Use %s instead — that one moves both halves together.",
        _REMOVED_KB_ROOT_ENV,
        mutable_kb_root(),
        KB_ROOT_ENV,
    )


def mutable_kb_root() -> Path:
    """Root of the KB partition this session reads and writes."""
    override = os.environ.get(KB_ROOT_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    workspace = os.environ.get("USER_DATA_PATH", "").strip() or _default_workspace_root()
    return Path(workspace).expanduser() / _MUTABLE_KB_DIRNAME


def framework_optimization_root() -> Path:
    """The partition holding the lessons ledger, for reader and writer alike."""
    return mutable_kb_root() / _FRAMEWORK_OPTIMIZATION_ROOT


def _resolve_kb_root() -> Path:
    """Resolve the active KB root each call (so tests can monkeypatch env)."""
    return mutable_kb_root()


def migrate_legacy_partition_once() -> Path | None:
    """Carry the framework partition over from the legacy ``<workspace>/kb`` root."""
    if os.environ.get(KB_ROOT_ENV, "").strip():
        return None

    workspace = Path(os.environ.get("USER_DATA_PATH", "").strip() or _default_workspace_root()).expanduser()
    source = workspace / _LEGACY_WORKSPACE_KB_DIRNAME / _FRAMEWORK_OPTIMIZATION_ROOT
    destination = framework_optimization_root()

    try:
        if not source.is_dir() or not any(source.iterdir()):
            return None
        if destination.exists() and any(destination.iterdir()):
            return None
        _copy_partition_atomically(source, destination)
    except Exception:  # noqa: BLE001 — a convenience copy may not stop the run
        _log.warning(
            "FRAMEWORK KB: could not carry the legacy partition over from %s; continuing with "
            "whatever is at %s. The FRAMEWORK phase treats a missing ledger as a cold start, so "
            "it may re-propose PRs it has already tried.",
            source,
            destination,
            exc_info=True,
        )
        return None

    _log.warning(
        "FRAMEWORK KB: migrated the legacy partition %s -> %s. The source is left in place; "
        "remove it once the new location looks right.",
        source,
        destination,
    )
    return destination


def _copy_partition_atomically(source: Path, destination: Path) -> None:
    """Copy ``source`` onto a not-yet-existing ``destination`` in one visible step."""
    root = destination.parent
    staging = root.with_name(f"{root.name}.migrating-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        # symlinks=True: copy links as links.
        shutil.copytree(source, staging, symlinks=True)
        root.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise



def read_pr_ledger(kb_root: Path | None = None) -> list[dict]:
    """Read the framework PR outcome ledger from ``lessons.jsonl``."""
    root = kb_root or _resolve_kb_root()
    path = root / _FRAMEWORK_OPTIMIZATION_ROOT / LESSONS_FILE
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


__all__ = [
    "LESSONS_FILE",
    "ALLOWED_OUTCOMES",
    "OUTCOME_INTEGRATED",
    "OUTCOME_ALREADY_PRESENT",
    "OUTCOME_REVERTED_SMOKE_FAIL",
    "OUTCOME_REJECTED_APPLY_FAIL",
    "OUTCOME_REVERTED_SWITCH_OFF_PARITY",
    "OUTCOME_REVERTED_PARITY_INCONCLUSIVE",
    "framework_optimization_root",
    "prepare_kb_environment",
    "read_pr_ledger",
]
