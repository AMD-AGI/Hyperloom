"""Patch lifecycle primitives for framework_integrate (P3 PR-H).

Shared between kernel and framework patch handlers. PR-H lands the
framework side; an opt-in followup will refactor
:mod:`kernel_request_handlers` to share these helpers without
behavioural drift.

Public surface:

* :func:`generate_patch_id`           -- ``fw-yyyymmdd-tokenhex8``
  identifier used as the rollback key.
* :func:`backup_files`                -- atomic snapshot of every file
  the patch touches; returns a :class:`BackupRef` consumed by
  :func:`rollback_backup`.
* :func:`apply_patch`                 -- ``git apply``-equivalent wrapper
  (uses ``patch -p1`` when patch-ng is unavailable).
* :func:`rollback_backup`             -- restore the file tree from a
  :class:`BackupRef`; LIFO is the caller's responsibility (most
  recent backup pops first).
* :func:`decide_verdict`              -- 3-gate KEEP / REVERT /
  NEEDS_REVIEW decision; reuse-target across kernel + framework.

All helpers are best-effort tolerant: filesystem errors during backup
are escalated (apply must not proceed without a working snapshot), but
rollback errors are logged + swallowed so a failed rollback never
masks the underlying verdict.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal


log = logging.getLogger(__name__)


Verdict = Literal["KEEP", "REVERT", "NEEDS_REVIEW"]


@dataclass(frozen=True)
class BackupRef:
    """Reference to a per-patch on-disk snapshot.

    ``backup_root`` is the directory containing one renamed-original
    file per ``files`` entry. ``files`` lists the absolute paths that
    were backed up (the ones the patch is about to mutate). ``patch_id``
    is the rollback-key handed back to the caller.
    """

    patch_id: str
    backup_root: Path
    files: tuple[Path, ...] = ()
    created_at_ms: int = 0


def generate_patch_id(prefix: str = "fw") -> str:
    """Return ``<prefix>-YYYYMMDD-<token_hex8>``.

    Default ``fw`` prefix per design §4.9; kernel handlers can pass
    ``kn`` when this module is extracted.
    """
    today = time.strftime("%Y%m%d", time.gmtime())
    return f"{prefix}-{today}-{secrets.token_hex(4)}"


# ---------------------------------------------------------------------------
# Backup / rollback
# ---------------------------------------------------------------------------
def backup_files(
    patch_id: str,
    files: Iterable[Path],
    *,
    session_dir: Path,
) -> BackupRef:
    """Snapshot every ``files`` entry under
    ``session_dir/runs/framework/<patch_id>/backup/``.

    Raises :class:`FileNotFoundError` when an entry does not exist;
    raises :class:`OSError` on permission / disk problems. Callers
    convert these into IntegrateFailure(backup_failed).
    """
    backup_root = (
        Path(session_dir) / "runs" / "framework" / patch_id / "backup"
    )
    backup_root.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for src in files:
        src = Path(src)
        if not src.is_file():
            raise FileNotFoundError(
                f"backup_files: source not a file: {src}"
            )
        # Mirror the absolute path under backup_root with leading slash
        # stripped so two patches that touch /a/b.py + /a/c.py can
        # coexist under their own backup dirs.
        rel = src.as_posix().lstrip("/")
        dst = backup_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        saved.append(src.resolve())
    return BackupRef(
        patch_id=patch_id,
        backup_root=backup_root,
        files=tuple(saved),
        created_at_ms=int(time.time() * 1000),
    )


def rollback_backup(ref: BackupRef) -> dict[str, str]:
    """Restore every file in ``ref`` from its snapshot. Best-effort.

    Returns a dict ``{abs_path: 'restored' | 'missing_backup' | 'error: ...'}``
    so callers can audit-log partial failures without bubbling them up.
    """
    status: dict[str, str] = {}
    for src in ref.files:
        rel = src.as_posix().lstrip("/")
        snapshot = ref.backup_root / rel
        if not snapshot.is_file():
            status[str(src)] = "missing_backup"
            log.warning("rollback_backup: snapshot missing for %s", src)
            continue
        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(snapshot), str(src))
            status[str(src)] = "restored"
        except OSError as exc:
            status[str(src)] = f"error: {exc}"
            log.warning("rollback_backup: %s failed: %s", src, exc)
    return status


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
class PatchApplyError(RuntimeError):
    """Raised when ``git apply`` / ``patch -p1`` exits non-zero."""


def apply_patch(
    patch_path: Path,
    *,
    cwd: Path | None = None,
    strip: int = 1,
    timeout_sec: int = 60,
) -> subprocess.CompletedProcess:
    """Apply a unified diff with ``git apply`` (fall back to ``patch -p1``).

    Caller is responsible for backing up files first (see
    :func:`backup_files`). Raises :class:`PatchApplyError` on
    non-zero exit; the caller converts to IntegrateFailure
    (reason='patch_apply_failed').
    """
    if not patch_path.is_file():
        raise PatchApplyError(f"patch file not found: {patch_path}")
    # Prefer git apply when a git binary is on PATH -- better diagnostics +
    # works on bare unified diffs.
    git_bin = shutil.which("git")
    if git_bin is not None:
        cmd = [git_bin, "apply", f"--unsafe-paths", str(patch_path)]
        proc = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None,
            check=False, capture_output=True, text=True,
            timeout=timeout_sec,
        )
        if proc.returncode == 0:
            return proc
        log.debug(
            "apply_patch: git apply rc=%d stderr=%r; trying patch -p%d",
            proc.returncode, proc.stderr.strip()[:200], strip,
        )
    patch_bin = shutil.which("patch")
    if patch_bin is None:
        raise PatchApplyError("neither git nor patch is available on PATH")
    cmd = [patch_bin, f"-p{strip}", "-i", str(patch_path)]
    proc = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        check=False, capture_output=True, text=True,
        timeout=timeout_sec,
    )
    if proc.returncode != 0:
        raise PatchApplyError(
            f"patch -p{strip} rc={proc.returncode}: "
            f"stderr={proc.stderr.strip()[:400]!r}"
        )
    return proc


# ---------------------------------------------------------------------------
# Verdict (KEEP / REVERT / NEEDS_REVIEW)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VerdictInputs:
    """Inputs to :func:`decide_verdict` (immutable, audit-loggable)."""

    baseline_tput: float
    baseline_accuracy: float
    tput_after: float
    accuracy_after: float | None = None
    min_throughput_gain_pct: float = 3.0
    max_accuracy_drop_pct: float = 1.0
    bench_ok: bool = True
    bench_reason: str = ""


@dataclass(frozen=True)
class VerdictResult:
    """Output of :func:`decide_verdict`."""

    verdict: Verdict
    gain_pct: float
    accuracy_drop_pct: float
    reason: str


def decide_verdict(inp: VerdictInputs) -> VerdictResult:
    """Apply the 3-gate decision:

    1. Bench succeeded? bench_ok=False -> REVERT (regardless of numbers).
    2. Throughput gain >= min_throughput_gain_pct? Below -> REVERT.
    3. Accuracy drop <= max_accuracy_drop_pct? Above -> REVERT.

    NEEDS_REVIEW is reserved for the ambiguous case: bench_ok=True with
    a usable tput_after but missing / nan accuracy_after (caller treats
    NEEDS_REVIEW as "don't rollback, surface to operator").
    """
    if not inp.bench_ok:
        return VerdictResult(
            verdict="REVERT",
            gain_pct=0.0,
            accuracy_drop_pct=0.0,
            reason=inp.bench_reason or "bench_failed",
        )
    baseline = max(inp.baseline_tput, 1e-9)
    gain_pct = (inp.tput_after - inp.baseline_tput) / baseline * 100.0
    if gain_pct < inp.min_throughput_gain_pct:
        return VerdictResult(
            verdict="REVERT",
            gain_pct=gain_pct,
            accuracy_drop_pct=0.0,
            reason=(
                f"gain {gain_pct:.2f}% below threshold "
                f"{inp.min_throughput_gain_pct:.2f}%"
            ),
        )
    if inp.accuracy_after is None:
        return VerdictResult(
            verdict="NEEDS_REVIEW",
            gain_pct=gain_pct,
            accuracy_drop_pct=0.0,
            reason="accuracy_after missing; operator decision required",
        )
    drop_pct = (
        (inp.baseline_accuracy - inp.accuracy_after)
        / max(inp.baseline_accuracy, 1e-9) * 100.0
    )
    if drop_pct > inp.max_accuracy_drop_pct:
        return VerdictResult(
            verdict="REVERT",
            gain_pct=gain_pct,
            accuracy_drop_pct=drop_pct,
            reason=(
                f"accuracy drop {drop_pct:.2f}% exceeds limit "
                f"{inp.max_accuracy_drop_pct:.2f}%"
            ),
        )
    return VerdictResult(
        verdict="KEEP",
        gain_pct=gain_pct,
        accuracy_drop_pct=drop_pct,
        reason="all gates passed",
    )


__all__ = [
    "BackupRef",
    "PatchApplyError",
    "Verdict",
    "VerdictInputs",
    "VerdictResult",
    "apply_patch",
    "backup_files",
    "decide_verdict",
    "generate_patch_id",
    "rollback_backup",
]
