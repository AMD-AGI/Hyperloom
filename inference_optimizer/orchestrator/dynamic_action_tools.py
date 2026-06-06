# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tool whitelist for the dynamic_action sub-agent.

* ``read_source``             — files inside ``framework_source_roots``
                                 (capped by ``MAX_READ_SOURCE_CHARS``).
* ``read_session_artifact``   — paths under prefix whitelist with deny
                                 segments + cross-``dyn_id`` isolation.
* ``apply_patch_in_worktree`` — ``git apply`` inside the per-dispatch
                                 worktree only.
* ``emit_proposal``           — terminal signal; validated by
                                 :mod:`dynamic_action_proposal`.
* ``run_bench``               — gated by :data:`BENCH_TOOL_ENABLED_V1`;
                                 excluded from :data:`ALL_DYNAMIC_TOOLS`
                                 and returns ``bench_tool_disabled_v1``
                                 while the registry is empty.

Each tool returns a JSON-serialisable dict.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .framework_paths import resolve_source_file_allowlist


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool ids and hard limits
# ---------------------------------------------------------------------------
TOOL_READ_SOURCE: str = "read_source"
TOOL_READ_SESSION_ARTIFACT: str = "read_session_artifact"
TOOL_RUN_BENCH: str = "run_bench"
TOOL_APPLY_PATCH_IN_WORKTREE: str = "apply_patch_in_worktree"
TOOL_EMIT_PROPOSAL: str = "emit_proposal"

# When False, ``run_bench`` is excluded from the tool surface and
# ``BENCH_REGISTRY`` is empty; flip together with real probe bodies.
BENCH_TOOL_ENABLED_V1: bool = False

ALL_DYNAMIC_TOOLS: frozenset[str] = frozenset(
    {
        TOOL_READ_SOURCE,
        TOOL_READ_SESSION_ARTIFACT,
        TOOL_APPLY_PATCH_IN_WORKTREE,
        TOOL_EMIT_PROPOSAL,
    } | ({TOOL_RUN_BENCH} if BENCH_TOOL_ENABLED_V1 else set()),
)
DYNAMIC_RESOURCE_TOOLS: frozenset[str] = ALL_DYNAMIC_TOOLS - {TOOL_EMIT_PROPOSAL}

# Per-call response size cap for ``read_source``.
MAX_READ_SOURCE_CHARS: int = 16_000

# Session-relative path prefixes the sub-agent may read.
SESSION_ARTIFACT_ALLOWED_PREFIXES: tuple[str, ...] = (
    "runs/grid/",
    "agents/orchestration/dynamic_actions/",
    "runs/dynamic/",
)

# Segments denied even when the prefix matches; the cross-``dyn_id``
# check below is layered on top of this list.
SESSION_ARTIFACT_DENY_SEGMENTS: tuple[str, ...] = (
    "inbox.jsonl",
    "outbox.jsonl",
    "agents/critic/",
    "runs/specialist/",
)


# ---------------------------------------------------------------------------
# Bench registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchSpec:
    """One entry in the bench whitelist; ``script_path`` is resolved
    against the package's ``benches/`` directory at call time."""

    bench_id: str
    description: str
    wall_clock_sec: float
    script_path: str


# Populated alongside real probe implementations; the gate above
# guards the empty case.
BENCH_REGISTRY: dict[str, BenchSpec] = {}

# Hard ceiling on a single ``run_bench`` invocation.
MAX_BENCH_WALL_CLOCK_SEC: float = 60.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _error(reason: str, **extra: Any) -> dict[str, Any]:
    """Standard tool error envelope: ``{ok: False, reason, ...}``."""
    return {"ok": False, "reason": reason, **extra}


def _ok(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": True}
    if payload:
        out.update(payload)
    return out


def _effective_read_limit(max_bytes: int | None) -> int:
    """Resolve a caller-supplied ``max_bytes`` against the hard cap.

    ``None`` / non-positive falls back to ``MAX_READ_SOURCE_CHARS``;
    any positive value is clamped down to it so a sub-agent can ask
    for a smaller targeted read but never exceed the ceiling.
    """
    try:
        n = int(max_bytes) if max_bytes is not None else 0
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return MAX_READ_SOURCE_CHARS
    return min(n, MAX_READ_SOURCE_CHARS)


def _path_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


# ---------------------------------------------------------------------------
# read_source
# ---------------------------------------------------------------------------
def read_source(path: str, max_bytes: int | None = None) -> dict[str, Any]:
    """Read a file under ``framework_source_roots``.

    ``max_bytes`` lets the sub-agent request a smaller targeted read;
    it is clamped to ``MAX_READ_SOURCE_CHARS`` and defaults to it.
    Failures (path outside roots, missing file, etc.) are returned as
    ``_error`` envelopes so the sub-agent can self-correct without
    aborting the turn loop.
    """
    raw = str(path or "").strip()
    if not raw:
        return _error("path_required")
    if any(ch in raw for ch in ("*", "?", "[")):
        return _error("globbing_not_allowed", path=raw)
    target = Path(raw)
    if not target.is_absolute():
        return _error("path_must_be_absolute", path=raw)
    if ".." in target.parts:
        return _error("path_traversal_denied", path=raw)
    # Containment is checked on the resolved path so symlinks + ``..``
    # cannot escape the framework source roots.
    roots = resolve_source_file_allowlist()
    if not any(_path_under(target, Path(root)) for root in roots):
        return _error("path_outside_framework_source_roots", path=raw)
    if not target.exists():
        return _error("not_found", path=raw)
    if target.is_dir():
        return _error("directory_listing_not_allowed", path=raw)
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return _error("read_failed", path=raw, detail=repr(exc))
    limit = _effective_read_limit(max_bytes)
    truncated = len(text) > limit
    if truncated:
        text = text[:limit]
    return _ok({
        "path": raw,
        "content": text,
        "truncated": truncated,
        "bytes_returned": len(text),
    })


# ---------------------------------------------------------------------------
# read_session_artifact
# ---------------------------------------------------------------------------
def read_session_artifact(
    session_dir: Path, relative_path: str, *, dyn_id: str,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Read a whitelisted artefact under ``session_dir``.

    ``dyn_id`` is the current dispatch's id; reads addressed at any
    *other* ``dynamic_actions/<other_id>/`` directory are denied.
    ``max_bytes`` is clamped to ``MAX_READ_SOURCE_CHARS``.
    """
    raw = str(relative_path or "").strip()
    if not raw:
        return _error("path_required")
    if raw.startswith("/"):
        return _error("path_must_be_session_relative", path=raw)
    if ".." in Path(raw).parts:
        return _error("path_traversal_denied", path=raw)
    if not any(raw.startswith(p) for p in SESSION_ARTIFACT_ALLOWED_PREFIXES):
        return _error("path_not_in_allowed_prefixes", path=raw)
    for seg in SESSION_ARTIFACT_DENY_SEGMENTS:
        if seg in raw:
            return _error("path_in_deny_list", path=raw, segment=seg)
    # Cross-dyn_id isolation covers both the per-dispatch artefact dir
    # and the per-dispatch worktree under runs/dynamic/.
    for prefix in (
        "agents/orchestration/dynamic_actions/",
        "runs/dynamic/",
    ):
        if raw.startswith(prefix):
            head = raw[len(prefix):].split("/", 1)[0]
            if head and head != dyn_id:
                return _error(
                    "cross_dyn_id_isolation",
                    path=raw,
                    requested_dyn_id=head,
                    current_dyn_id=dyn_id,
                )
            break
    target = (Path(session_dir) / raw).resolve()
    if not _path_under(target, Path(session_dir)):
        return _error("path_escapes_session_dir", path=raw)
    if not target.exists():
        return _error("not_found", path=raw)
    if target.is_dir():
        return _error("directory_listing_not_allowed", path=raw)
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return _error("read_failed", path=raw, detail=repr(exc))
    limit = _effective_read_limit(max_bytes)
    truncated = len(text) > limit
    if truncated:
        text = text[:limit]
    return _ok({
        "path": raw,
        "content": text,
        "truncated": truncated,
        "bytes_returned": len(text),
    })


# ---------------------------------------------------------------------------
# run_bench
# ---------------------------------------------------------------------------
async def run_bench(
    bench_id: str,
    *,
    worktree: Path,
    call_id: str,
    params: dict[str, Any] | None = None,
    bench_dir_root: Path | None = None,
) -> dict[str, Any]:
    """Execute a registered micro-bench inside the worktree.

    Output lands under ``worktree/scratch/bench/<bench_id>/<call_id>/``
    and is destroyed with the worktree (never surfaces in artefacts).
    ``bench_dir_root`` overrides the script discovery root for tests.
    """
    if not BENCH_TOOL_ENABLED_V1:
        return _error(
            "bench_tool_disabled_v1",
            bench_id=bench_id,
            note=(
                "run_bench is disabled; use read_source + "
                "read_session_artifact for exploration."
            ),
        )
    bench = BENCH_REGISTRY.get(bench_id)
    if bench is None:
        return _error(
            "unknown_bench_id",
            bench_id=bench_id,
            allowed=sorted(BENCH_REGISTRY),
        )
    timeout = min(bench.wall_clock_sec, MAX_BENCH_WALL_CLOCK_SEC)
    if bench_dir_root is None:
        bench_dir_root = Path(__file__).parent.parent / "benches"
    script = bench_dir_root / Path(bench.script_path).name
    if not script.exists():
        return _error(
            "bench_script_missing",
            bench_id=bench_id,
            script_path=str(script),
        )
    scratch = (
        Path(worktree) / "scratch" / "bench" / bench_id / str(call_id)
    )
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("DYNAMIC_BENCH_OUTPUT_DIR", str(scratch))
    env.setdefault("DYNAMIC_BENCH_WORKTREE", str(worktree))
    env.setdefault("DYNAMIC_BENCH_PARAMS_JSON", json.dumps(params or {}))
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", str(script),
            cwd=str(worktree),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return _error("spawn_failed", bench_id=bench_id, detail=repr(exc))
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await proc.wait()
        except ProcessLookupError:
            pass
        return _error(
            "timed_out", bench_id=bench_id, wall_clock_sec=timeout,
        )
    stdout = stdout_bytes.decode("utf-8", errors="replace")[-2000:]
    stderr = stderr_bytes.decode("utf-8", errors="replace")[-2000:]
    return _ok({
        "bench_id": bench_id,
        "exit_code": proc.returncode,
        "stdout_tail": stdout,
        "stderr_tail": stderr,
        "output_dir": str(scratch),
    })


# ---------------------------------------------------------------------------
# apply_patch_in_worktree
# ---------------------------------------------------------------------------
_PATCH_PATH_RE = re.compile(r"^(?:---|\+\+\+) (?:a|b)/(?P<path>.+)$", re.M)


def apply_patch_in_worktree(
    worktree: Path, patch_text: str,
) -> dict[str, Any]:
    """Try ``git apply`` inside the worktree (self-check). The patch is
    not committed; the runner resets the worktree on terminate."""
    if not patch_text or not patch_text.strip():
        return _error("empty_patch")
    worktree = Path(worktree)
    if not worktree.is_dir():
        return _error("worktree_missing", path=str(worktree))
    for hit in _PATCH_PATH_RE.finditer(patch_text):
        cand = hit.group("path").strip()
        if cand.startswith("/") or ".." in Path(cand).parts:
            return _error("patch_path_escapes_worktree", offending=cand)
    try:
        proc = subprocess.run(
            ["git", "apply", "--check", "-"],
            cwd=str(worktree),
            input=patch_text,
            text=True,
            capture_output=True,
            timeout=20.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return _error("git_apply_failed_to_spawn", detail=repr(exc))
    if proc.returncode != 0:
        return _error(
            "git_apply_rejected",
            stderr_tail=(proc.stderr or "").strip()[-2000:],
        )
    try:
        proc2 = subprocess.run(
            ["git", "apply", "-"],
            cwd=str(worktree),
            input=patch_text,
            text=True,
            capture_output=True,
            timeout=20.0,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _error("git_apply_timed_out", detail=repr(exc))
    if proc2.returncode != 0:
        return _error(
            "git_apply_unexpected_failure_after_check",
            stderr_tail=(proc2.stderr or "").strip()[-2000:],
        )
    return _ok({"applied": True})


def capture_worktree_cumulative_diff(worktree: Path) -> str | None:
    """Return ``git diff HEAD`` output for ``worktree``.

    * ``""``     — clean worktree.
    * ``<diff>`` — uncommitted-change diff.
    * ``None``   — git failure / not a repo; callers skip the
                   cumulative-diff check rather than aborting.
    """
    worktree = Path(worktree)
    if not worktree.is_dir():
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(worktree), "diff", "HEAD"],
            capture_output=True, text=True, timeout=20.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout or ""


def reset_worktree(worktree: Path) -> None:
    """Discard uncommitted changes + untracked files in ``worktree``."""
    worktree = Path(worktree)
    if not worktree.is_dir():
        return
    try:
        subprocess.run(
            ["git", "reset", "--hard"],
            cwd=str(worktree), capture_output=True,
            timeout=20.0, check=False,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=str(worktree), capture_output=True,
            timeout=20.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


__all__ = [
    "ALL_DYNAMIC_TOOLS",
    "BENCH_REGISTRY",
    "BENCH_TOOL_ENABLED_V1",
    "BenchSpec",
    "DYNAMIC_RESOURCE_TOOLS",
    "MAX_BENCH_WALL_CLOCK_SEC",
    "MAX_READ_SOURCE_CHARS",
    "SESSION_ARTIFACT_ALLOWED_PREFIXES",
    "SESSION_ARTIFACT_DENY_SEGMENTS",
    "TOOL_APPLY_PATCH_IN_WORKTREE",
    "TOOL_EMIT_PROPOSAL",
    "TOOL_READ_SESSION_ARTIFACT",
    "TOOL_READ_SOURCE",
    "TOOL_RUN_BENCH",
    "apply_patch_in_worktree",
    "capture_worktree_cumulative_diff",
    "read_session_artifact",
    "read_source",
    "reset_worktree",
    "run_bench",
]
