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
# ``BENCH_REGISTRY`` is empty; flipped together with real probe bodies in the
# bench-enable phase.
BENCH_TOOL_ENABLED: bool = False


@dataclass(frozen=True)
class BenchSpec:
    """One entry in the bench whitelist; ``script_path`` is resolved
    against the package's ``benches/`` directory at call time."""

    bench_id: str
    description: str
    wall_clock_sec: float
    script_path: str


# Populated alongside real probe implementations; the gate above guards the
# empty case.
BENCH_REGISTRY: dict[str, BenchSpec] = {}

# Hard ceiling on a single ``run_bench`` invocation.
MAX_BENCH_WALL_CLOCK_SEC: float = 60.0


def _error(reason: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "reason": reason, **extra}


def _ok(payload: dict[str, Any] | None = None) -> dict[str, Any]:
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
    """Try ``git apply`` inside the worktree (self-check); not committed."""
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
