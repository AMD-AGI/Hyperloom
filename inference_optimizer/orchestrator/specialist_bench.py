# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""In-loop micro-benchmark surface for bench-enabled specialists.

Migrated out of the retired ``dynamic_action_tools`` module. Exposes the
``run_bench`` tool plus the worktree git helpers it relies on. The tool is
gated behind :data:`BENCH_TOOL_ENABLED` (OFF until the real probe bodies +
GPU-lane throttle land); while OFF, ``run_bench`` returns a disabled envelope
and :data:`BENCH_REGISTRY` is empty.

Only ``mode == 'patch'`` specialists with ``bench == True`` are granted this
tool (see specialist_profile / SpecialistRunner).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOOL_RUN_BENCH: str = "run_bench"

# When False, ``run_bench`` is excluded from the tool surface and
# ``BENCH_REGISTRY`` is empty. Enabled now that the bench probe scripts +
# GPU-lane throttle are wired; the probe bodies under ``benches/`` are still
# lightweight placeholders that emit a result.json without starting a server.
BENCH_TOOL_ENABLED: bool = True


@dataclass(frozen=True)
class BenchSpec:
    """One entry in the bench whitelist; ``script_path`` is resolved
    against the package's ``benches/`` directory at call time."""

    bench_id: str
    description: str
    wall_clock_sec: float
    script_path: str


# Bench whitelist. Each entry maps a stable ``bench_id`` to a script under the
# package ``benches/`` directory. Scripts are worktree-scoped probes that read
# the ``SPECIALIST_BENCH_*`` env contract and must never start a serving
# process (see benches/README.md).
BENCH_REGISTRY: dict[str, BenchSpec] = {
    spec.bench_id: spec
    for spec in (
        BenchSpec(
            bench_id="kernel_attention_timing",
            description=(
                "Micro-time the attention kernel path in the worktree "
                "(prefill + decode shapes)."
            ),
            wall_clock_sec=45.0,
            script_path="kernel_attention_timing.sh",
        ),
        BenchSpec(
            bench_id="kernel_gemm_timing",
            description="Micro-time representative GEMM shapes in the worktree.",
            wall_clock_sec=45.0,
            script_path="kernel_gemm_timing.sh",
        ),
        BenchSpec(
            bench_id="kernel_kvcache_layout",
            description=(
                "Probe KV-cache layout / paging cost for the worktree build."
            ),
            wall_clock_sec=45.0,
            script_path="kernel_kvcache_layout.sh",
        ),
        BenchSpec(
            bench_id="inference_short_prompt",
            description=(
                "Short-prompt single-process inference micro-bench (no served "
                "endpoint)."
            ),
            wall_clock_sec=60.0,
            script_path="inference_short_prompt.sh",
        ),
    )
}

# Hard ceiling on a single ``run_bench`` invocation.
MAX_BENCH_WALL_CLOCK_SEC: float = 60.0


def _error(reason: str, **extra: Any) -> dict[str, Any]:
    """Build a failure result envelope.

    Args:
        reason: Human-readable failure reason.
        **extra: Additional fields to merge into the envelope.

    Returns:
        Dict with ``ok=False`` plus the reason and any extra fields.
    """
    return {"ok": False, "reason": reason, **extra}


def _ok(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a success result envelope.

    Args:
        payload: Optional fields to merge into the envelope.

    Returns:
        Dict with ``ok=True`` plus any payload fields.
    """
    out: dict[str, Any] = {"ok": True}
    if payload:
        out.update(payload)
    return out


async def run_bench(
    bench_id: str,
    *,
    worktree: Path,
    call_id: str,
    params: dict[str, Any] | None = None,
    bench_dir_root: Path | None = None,
) -> dict[str, Any]:
    """Execute a registered micro-bench inside the worktree.

    Output lands under ``worktree/scratch/bench/<bench_id>/<call_id>/`` and is
    destroyed with the worktree. ``bench_dir_root`` overrides script discovery
    for tests.

    Args:
        bench_id: Registered micro-bench identifier to run.
        worktree: Worktree the bench runs inside.
        call_id: Unique call id used to scope the output directory.
        params: Optional bench parameters passed via env JSON.
        bench_dir_root: Override for bench-script discovery (tests).

    Returns:
        A result dict with ``ok`` plus bench output, or an error dict when the
        bench is disabled/unknown/missing or the run fails or times out.
    """
    if not BENCH_TOOL_ENABLED:
        return _error(
            "bench_tool_disabled",
            bench_id=bench_id,
            note=(
                "run_bench is disabled; use read-only investigation tools "
                "for exploration."
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
    env.setdefault("SPECIALIST_BENCH_OUTPUT_DIR", str(scratch))
    env.setdefault("SPECIALIST_BENCH_WORKTREE", str(worktree))
    env.setdefault("SPECIALIST_BENCH_PARAMS_JSON", json.dumps(params or {}))
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


_PATCH_PATH_RE = re.compile(r"^(?:---|\+\+\+) (?:a|b)/(?P<path>.+)$", re.M)


def apply_patch_in_worktree(
    worktree: Path, patch_text: str,
) -> dict[str, Any]:
    """Try ``git apply`` inside the worktree (self-check); not committed.

    Args:
        worktree: Worktree the patch is applied inside.
        patch_text: The unified-diff text to apply.

    Returns:
        A result dict with ``applied`` on success, or an error dict for an
        empty patch, missing worktree, path escape, or git-apply failure.
    """
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

    Args:
        worktree: Worktree to capture the cumulative diff from.

    Returns:
        The ``git diff HEAD`` output, ``""`` for a clean worktree, or ``None``
        on git failure / not a repo.
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
    """Discard uncommitted changes + untracked files in ``worktree``.

    Args:
        worktree: Worktree to hard-reset and clean.
    """
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
    "BENCH_REGISTRY",
    "BENCH_TOOL_ENABLED",
    "BenchSpec",
    "MAX_BENCH_WALL_CLOCK_SEC",
    "TOOL_RUN_BENCH",
    "apply_patch_in_worktree",
    "capture_worktree_cumulative_diff",
    "reset_worktree",
    "run_bench",
]
