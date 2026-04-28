"""Process-management helpers — DESIGN §4.7.

Wrap every server lifecycle / shell quirk that has bitten us in production.
All callers go through here so IR-4 / IR-5 / IR-6 stay enforced.

STATUS (v0.7):
    Pure-Python implementations. Subprocess-touching helpers split into a
    thin shell wrapper (``_run_pgrep`` / ``_run_kill``) so unit tests can
    patch them without spawning real processes. The functions never raise
    on a missing ``pgrep`` / ``kill``; on Windows or stripped containers
    they degrade to ``0`` / no-op so the rest of the orchestrator can keep
    running with mock backends.

References:
    - DESIGN §4.5 IR-4 / IR-5 / IR-6
    - DESIGN §4.7 Process management traps (sprint+marathon merged set)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, Literal

from .kernel_opt_constants import SERVER_KILL_WAIT_S


__all__ = [
    "prepend_venv_path",
    "safe_kill_server",
    "wait_kill_settle",
    "unset_profile_envs",
    "pick_filtered_trace",
    "assert_user_tp_respected",
    "vllm_flag_translator",
    "enforce_run_baseline_sh",
    "ProcessManagementError",
    "FRAMEWORK_PATTERNS",
]


class ProcessManagementError(RuntimeError):
    """Raised when an Iron Rule invariant is detected at runtime."""


# Patterns we ``pgrep -f`` for — anchored enough that the conductor's own
# command line cannot match them (IR-5).
FRAMEWORK_PATTERNS: dict[str, str] = {
    "sglang": "sglang.launch_server",
    "vllm": "vllm.entrypoints.openai.api_server",
}


# --------------------------------------------------------------------------
# §4.7 PATH / venv hygiene
# --------------------------------------------------------------------------
def prepend_venv_path(
    env: dict[str, str] | None = None,
    *,
    venv_bin: str = "/opt/venv/bin",
) -> dict[str, str]:
    """Return a copy of ``env`` with ``venv_bin`` prepended to ``PATH``.

    Idempotent: if ``venv_bin`` is already the first PATH entry the env is
    returned unchanged (still as a copy).
    """
    src = dict(env) if env is not None else dict(os.environ)
    current = src.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if not parts or parts[0] != venv_bin:
        parts = [venv_bin] + [p for p in parts if p != venv_bin]
    src["PATH"] = os.pathsep.join(parts)
    return src


# --------------------------------------------------------------------------
# §4.7 / IR-5 safe server kill (subprocess seam)
# --------------------------------------------------------------------------
def _run_pgrep(pattern: str) -> list[int]:
    """List PIDs whose full command line matches ``pattern``.

    Tests patch this. Returns an empty list when ``pgrep`` is unavailable.
    """
    if shutil.which("pgrep") is None:
        return []
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", pattern],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except subprocess.CalledProcessError:
        return []  # exit code 1 == no matches
    pids: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def _run_kill(pid: int, *, signal_num: int = 15) -> bool:
    """Best-effort signal delivery. Returns True on success."""
    try:
        os.kill(pid, signal_num)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False


def safe_kill_server(framework: Literal["sglang", "vllm"]) -> int:
    """Kill the *specific* launch process for the given framework (IR-5).

    Refuses anything that would invoke ``pkill -f sglang`` (which would
    take the conductor itself down, see DESIGN §4.5 IR-5).

    Returns the number of pids signalled. ``0`` if nothing matched.
    """
    if framework not in FRAMEWORK_PATTERNS:
        raise ProcessManagementError(
            f"unknown framework {framework!r}; "
            f"expected one of {sorted(FRAMEWORK_PATTERNS)}"
        )
    pattern = FRAMEWORK_PATTERNS[framework]
    pids = _run_pgrep(pattern)
    sent = 0
    for pid in pids:
        if _run_kill(pid):
            sent += 1
    return sent


def wait_kill_settle(
    seconds: int = SERVER_KILL_WAIT_S,
    *,
    framework: Literal["sglang", "vllm"] | None = None,
) -> bool:
    """Sleep then verify there are no surviving framework processes.

    Returns True when ``pgrep`` comes back empty after the sleep.
    """
    time.sleep(max(0, int(seconds)))
    if framework is None:
        return True
    survivors = _run_pgrep(FRAMEWORK_PATTERNS[framework])
    return not survivors


# --------------------------------------------------------------------------
# §4.7 trace / profile env hygiene
# --------------------------------------------------------------------------
def unset_profile_envs(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return ``env`` with PROFILE / SGLANG_TORCH_PROFILER_DIR removed.

    The original mapping is not mutated — callers receive a copy.
    """
    src = dict(env) if env is not None else dict(os.environ)
    for k in ("PROFILE", "SGLANG_TORCH_PROFILER_DIR"):
        src.pop(k, None)
    return src


def pick_filtered_trace(trace_dir: Path) -> Path | None:
    """Find ``filtered-TP-0.trace.json.gz`` under ``trace_dir`` (recursive).

    Returns the most recently-modified hit, or ``None`` if there is none.
    """
    trace_dir = Path(trace_dir)
    if not trace_dir.is_dir():
        return None
    hits = list(trace_dir.rglob("filtered-TP-0.trace.json.gz"))
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


# --------------------------------------------------------------------------
# §4.7 user-input invariants
# --------------------------------------------------------------------------
def assert_user_tp_respected(prompt_tp: int, detected_gpus: int) -> None:
    """Crash hard if scheduler tried to override user-specified TP value.

    ``prompt_tp`` is what the user (or the runbook) asked for; it must be
    ≤ ``detected_gpus`` and the runtime must not silently shrink it.
    """
    if int(prompt_tp) <= 0:
        raise ProcessManagementError(
            f"prompt_tp must be positive, got {prompt_tp!r}"
        )
    if int(prompt_tp) > int(detected_gpus):
        raise ProcessManagementError(
            f"requested TP={prompt_tp} exceeds detected GPUs={detected_gpus}; "
            "refusing to silently downscale"
        )


# --------------------------------------------------------------------------
# §4.7 vllm flag translation
# --------------------------------------------------------------------------
_VLLM_FLAG_MAP: dict[str, str] = {
    "--disable-log-requests": "--disable-log-stats",
}


def vllm_flag_translator(args: Iterable[str]) -> list[str]:
    """Map sglang-style flags to vllm equivalents in-place (returning copy).

    Currently handles ``--disable-log-requests`` → ``--disable-log-stats``.
    Unknown flags pass through untouched.
    """
    out: list[str] = []
    for a in args:
        out.append(_VLLM_FLAG_MAP.get(a, a))
    return out


# --------------------------------------------------------------------------
# §4.7 IR-3 enforcement — must use scripts/run_baseline.sh
# --------------------------------------------------------------------------
_RUN_BASELINE_REQUIRED_ACTIONS: frozenset[str] = frozenset(
    {"baseline", "bench_runner", "integrate"}
)


def enforce_run_baseline_sh(
    action_name: str,
    *,
    script_path: Path | None = None,
) -> None:
    """Guard: certain actions MUST execute via ``scripts/run_baseline.sh``.

    Verifies the script exists; raises :class:`ProcessManagementError` if
    the action is in the required set but the script is missing.
    """
    if action_name not in _RUN_BASELINE_REQUIRED_ACTIONS:
        return
    if script_path is None:
        candidate = Path("scripts/run_baseline.sh")
    else:
        candidate = Path(script_path)
    if not candidate.is_file():
        raise ProcessManagementError(
            f"action {action_name!r} requires {candidate} (IR-3) but it is missing"
        )
