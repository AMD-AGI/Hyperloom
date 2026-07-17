# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for install.sh ``ensure_ray_started`` serving_slot validation.

A Ray head from an older install (pre-serving_slot) or a manual ``ray start``
lacks the ``serving_slot`` custom resource the Ray execution backend needs.
``ensure_ray_started`` must not silently reuse such a head: it validates the
resource on a live head and restarts the head to declare it when missing
(install time is side-effect free, since nothing depends on the head yet).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

KERNEL_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = KERNEL_ROOT / "scripts" / "install.sh"


def _extract_shell_function(name: str) -> str:
    """Extract a top-level ``name() { ... }`` shell function body from install.sh."""
    lines = INSTALL_SH.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line == f"{name}() {{")
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start : end + 1])


def _run_ensure_ray_started(tmp_path: Path, *, status_rc: int, status_out: str):
    """Run the extracted ``ensure_ray_started`` against a fake ``ray`` CLI.

    Args:
        status_rc: Exit code the fake ``ray status`` returns (0 = head up).
        status_out: Text the fake ``ray status`` prints (checked for serving_slot).

    Returns:
        ``(stdout, ray_trace)`` where ``ray_trace`` is every ``ray ...``
        invocation (captured to a file so caller-side ``>/dev/null`` redirects on
        ``ray start`` / ``ray stop`` do not hide them).
    """
    trace = tmp_path / "ray_trace.txt"
    fn = _extract_shell_function("ensure_ray_started")
    harness = f"""
set -uo pipefail
export RAY_TRACE_FILE={trace}
export RAY_STATUS_RC={status_rc}
export RAY_STATUS_OUT={status_out!r}
CHECK_ONLY=0
DRY_RUN=0
SKIP_RAY_START=0
RAY_NUM_GPUS=2
log() {{ echo "LOG $*"; }}
warn() {{ echo "WARN $*"; }}
ensure_fd_limit_for_ray() {{ :; }}
python3() {{ echo 0; }}
ray() {{
  printf 'ray %s\\n' "$*" >> "$RAY_TRACE_FILE"
  if [ "${{1:-}}" = "status" ]; then
    printf '%s\\n' "$RAY_STATUS_OUT"
    return "$RAY_STATUS_RC"
  fi
  return 0
}}
{fn}
ensure_ray_started
"""
    proc = subprocess.run(["bash", "-c", harness], text=True, capture_output=True, check=False)
    ray_trace = trace.read_text(encoding="utf-8") if trace.exists() else ""
    return proc.stdout, ray_trace


def _start_lines(trace: str) -> list[str]:
    """Return the ``ray start ...`` invocations recorded in the trace."""
    return [ln for ln in trace.splitlines() if ln.startswith("ray start")]


def test_ensure_ray_started_reuses_head_with_serving_slot(tmp_path):
    """A live head that already declares serving_slot is reused (no restart)."""
    stdout, trace = _run_ensure_ray_started(
        tmp_path, status_rc=0, status_out="0.0/1.0 serving_slot"
    )
    assert "ray head already running with serving_slot" in stdout
    assert not _start_lines(trace), f"must not restart a healthy head; trace={trace!r}"
    assert "ray stop" not in trace


def test_ensure_ray_started_restarts_head_without_serving_slot(tmp_path):
    """A live head lacking serving_slot is torn down and restarted WITH it."""
    stdout, trace = _run_ensure_ray_started(
        tmp_path, status_rc=0, status_out="0.0/8.0 GPU"
    )
    assert "restarting head to declare it" in stdout
    assert "ray stop --force" in trace
    starts = _start_lines(trace)
    assert starts, f"a fresh head must be started; trace={trace!r}"
    assert "serving_slot" in starts[0]
    assert "--resources" in starts[0]


def test_ensure_ray_started_starts_fresh_head_when_absent(tmp_path):
    """No live head: a fresh head is started declaring serving_slot."""
    stdout, trace = _run_ensure_ray_started(tmp_path, status_rc=1, status_out="")
    assert "no live ray head detected" in stdout
    starts = _start_lines(trace)
    assert starts, f"a fresh head must be started; trace={trace!r}"
    assert "serving_slot" in starts[0]


def test_ensure_ray_started_skipped_when_flag_set(tmp_path):
    """SKIP_RAY_START=1 is honored: no ray status/start/stop at all."""
    trace = tmp_path / "ray_trace.txt"
    fn = _extract_shell_function("ensure_ray_started")
    harness = f"""
set -uo pipefail
export RAY_TRACE_FILE={trace}
CHECK_ONLY=0
DRY_RUN=0
SKIP_RAY_START=1
log() {{ echo "LOG $*"; }}
warn() {{ echo "WARN $*"; }}
ray() {{ printf 'ray %s\\n' "$*" >> "$RAY_TRACE_FILE"; }}
{fn}
ensure_ray_started
"""
    proc = subprocess.run(["bash", "-c", harness], text=True, capture_output=True, check=False)
    assert "skipping ray head startup" in proc.stdout
    assert not trace.exists() or trace.read_text(encoding="utf-8") == ""
