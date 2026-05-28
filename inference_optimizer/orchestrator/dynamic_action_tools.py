"""dynamic_action.MD P3 §4 — tool whitelist for the dynamic sub-agent.

Three live resource tools + one terminal signal + ``run_bench`` gated
off in v1 (see :data:`BENCH_TOOL_ENABLED_V1`).

* ``read_source``                  — path inside framework_source_roots
                                      + ``MAX_READ_SOURCE_CHARS``
* ``read_session_artifact``        — whitelist roots + blacklist guards
* ``apply_patch_in_worktree``      — ``git apply`` inside the worktree
                                      only; out-of-tree paths denied
* ``emit_proposal``                — terminal signal; validation lives
                                      in :mod:`dynamic_action_proposal`.
* ``run_bench`` (DISABLED IN v1)   — until real probes land
  (``dynamic_action_gaps.md`` G1), the tool is excluded from
  :data:`ALL_DYNAMIC_TOOLS`, :data:`BENCH_REGISTRY` is empty, and
  any call returns ``bench_tool_disabled_v1``.

Each tool returns a JSON-serialisable dict so the runner can both
journal the result and feed it back to the LLM verbatim.
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

# v1 disable flag for the bench tool (dynamic_action_gaps.md G1). When
# False, ``run_bench`` is removed from the sub-agent's tool surface,
# :data:`BENCH_REGISTRY` is empty, and the runner treats any
# ``run_bench`` call as an unknown tool. Flip to True together with
# real probe implementations.
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

# P3 §4.1.a — single read_source response is hard-capped (≈4K tokens at
# 4 chars per token).
MAX_READ_SOURCE_CHARS: int = 16_000

# P3 §4.1.b — relative paths under SESSION_DIR that the sub-agent may
# read. Anything not matching one of these prefixes is denied.
SESSION_ARTIFACT_ALLOWED_PREFIXES: tuple[str, ...] = (
    "runs/grid/",
    "agents/orchestration/dynamic_actions/",
    "runs/dynamic/",
)

# P3 §4.1.b — explicit deny list. ``read_session_artifact`` rejects any
# path containing these segments even when the prefix would otherwise
# allow it (matters when ``agents/orchestration/dynamic_actions/<dyn_id>/``
# accidentally addresses a *different* dyn_id).
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
    """One entry in the bench whitelist.

    The registry is a code-level constant; sub-agents cannot extend it
    at runtime. ``script_path`` is intentionally relative — the runner
    resolves it against the package's ``benches/`` dir at call time so
    tests can monkey-patch with a stub script.
    """

    bench_id: str
    description: str
    wall_clock_sec: float
    script_path: str


# v1 registry — empty until real probes land (see G1 + BENCH_TOOL_ENABLED_V1).
# v2 candidates (kept here as docstring + commented descriptor so the
# next implementation pass has the schema target):
#   * kernel_attention_timing      — Single attention layer forward timing
#   * kernel_gemm_timing           — GEMM op timing + occupancy
#   * kernel_kvcache_layout        — KV cache layout read/write throughput
#   * inference_short_prompt       — Short prompt end-to-end latency
BENCH_REGISTRY: dict[str, BenchSpec] = {}

# Absolute hard ceiling regardless of per-bench config (Q2 decision).
MAX_BENCH_WALL_CLOCK_SEC: float = 60.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _error(reason: str, **extra: Any) -> dict[str, Any]:
    """Standardise the tool error envelope. Sub-agent always sees
    {ok: False, reason, ...} so the loop logic is symmetric."""
    return {"ok": False, "reason": reason, **extra}


def _ok(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": True}
    if payload:
        out.update(payload)
    return out


def _path_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


# ---------------------------------------------------------------------------
# read_source
# ---------------------------------------------------------------------------
def read_source(path: str) -> dict[str, Any]:
    """Read a framework_source_roots file. P3 §4.1.a contract.

    Failures (path outside roots, missing file, oversize, etc.) come
    back as ``_error`` envelopes so the sub-agent can self-correct
    without aborting the turn loop.
    """
    raw = str(path or "").strip()
    if not raw:
        return _error("path_required")
    if any(ch in raw for ch in ("*", "?", "[")):
        return _error("globbing_not_allowed", path=raw)
    target = Path(raw)
    if not target.is_absolute():
        return _error("path_must_be_absolute", path=raw)
    roots = resolve_source_file_allowlist()
    target_str = str(target)
    if not any(target_str.startswith(root) for root in roots):
        return _error("path_outside_framework_source_roots", path=raw)
    if not target.exists():
        return _error("not_found", path=raw)
    if target.is_dir():
        return _error("directory_listing_not_allowed", path=raw)
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return _error("read_failed", path=raw, detail=repr(exc))
    truncated = len(text) > MAX_READ_SOURCE_CHARS
    if truncated:
        text = text[:MAX_READ_SOURCE_CHARS]
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
) -> dict[str, Any]:
    """Read a whitelisted artefact under ``session_dir``.

    ``dyn_id`` is the current dispatch's id; reads addressed at *other*
    ``dynamic_actions/<other_id>/`` directories are denied even when
    the prefix matches (P3 §4.1.b black-list).
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
    # Cross-dyn_id isolation: any read addressed at
    # ``agents/orchestration/dynamic_actions/<other>/`` rejected.
    if raw.startswith("agents/orchestration/dynamic_actions/"):
        suffix = raw[len("agents/orchestration/dynamic_actions/"):]
        head = suffix.split("/", 1)[0]
        if head and head != dyn_id:
            return _error(
                "cross_dyn_id_isolation",
                path=raw,
                requested_dyn_id=head,
                current_dyn_id=dyn_id,
            )
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
    truncated = len(text) > MAX_READ_SOURCE_CHARS
    if truncated:
        text = text[:MAX_READ_SOURCE_CHARS]
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

    ``bench_dir_root`` lets tests stub the script discovery root; in
    production it defaults to ``<package>/benches``.     Output lands under
    ``worktree/scratch/bench/<bench_id>/<call_id>/`` and is *not*
    recovered (P3 §6 recovery whitelist).
    """
    if not BENCH_TOOL_ENABLED_V1:
        return _error(
            "bench_tool_disabled_v1",
            bench_id=bench_id,
            note=(
                "run_bench is gated off in v1 until real probes land "
                "(dynamic_action_gaps.md G1). Proceed using read_source "
                "+ read_session_artifact only."
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
    # Reject patches that try to touch out-of-tree paths.
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
    # The --check pass succeeded; do the real apply so the sub-agent can
    # iterate. The runner will git reset --hard on termination.
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
    """Return ``git diff HEAD`` output for ``worktree`` (gap G6).

    Used by the runner at ``emit_proposal`` time to validate that the
    proposal's ``patch_text`` matches the current worktree state when
    the sub-agent has applied one or more patches during iteration.

    Returns:
    * ``""``     — clean worktree (no uncommitted changes).
    * ``<diff>`` — uncommitted-change diff.
    * ``None``   — git failure (worktree not a repo / timeout); the
                   caller skips the cumulative-diff check rather than
                   bricking the dispatch.
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
    """Roll the worktree back to a clean state (used on runner exit so
    the next dispatch sees a fresh tree)."""
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
