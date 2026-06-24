"""Shared ``runtime.cli`` subprocess bridge for the sibling-agent backends.

Owns the command assembly and timeout / spawn / non-zero-rc → BackendError
mapping shared by critic_agent and robustness_agent; callers keep only their own
phase pre-validation and per-agent constants.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, Sequence

from .base import BackendError

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids an import cycle)
    from .critic_agent import RuntimeCall


def invoke_runtime_cli(
    call: "RuntimeCall",
    *,
    module: str,
    agent_label: str,
    timeout_sec: float,
    extra_args: Sequence[str] = (),
    stderr_truncate: int = 500,
) -> None:
    """Run ``python -m <module> <call.phase> --request ... --out ... <extra_args>``.

    Raises BackendError if the subprocess times out, cannot start, or exits non-zero.
    """
    cmd = [
        sys.executable,
        "-m",
        module,
        call.phase,
        "--request",
        str(call.request_path),
        "--out",
        str(call.out_path),
        *extra_args,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(call.cwd),
            env=call.env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise BackendError(
            f"{agent_label} runtime.cli {call.phase} timed out after "
            f"{timeout_sec}s (cwd={call.cwd})"
        ) from exc
    except FileNotFoundError as exc:
        raise BackendError(
            f"{agent_label} runtime.cli {call.phase} could not start "
            f"(python={sys.executable!r}, cwd={call.cwd}): {exc}"
        ) from exc

    if proc.returncode != 0:
        raise BackendError(
            f"{agent_label} runtime.cli {call.phase} exited rc={proc.returncode}: "
            f"stderr={proc.stderr.strip()[:stderr_truncate]!r}"
        )


__all__ = ["invoke_runtime_cli"]
