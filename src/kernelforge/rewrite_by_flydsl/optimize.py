# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""OPTIMIZE phase — hand the correct FlyDSL kernel to the existing forge-loop.

forge-loop is explicitly designed to be shelled out as an isolated, hard-killable
subprocess (see its CLI docstring), so the rewrite layer reuses it verbatim: no
refactor, and every forge-loop capability (baseline anchor, full-suite validation,
profiler + analyst, AVO supervisor, KB, candidate archive) applies to the FlyDSL
kernel unchanged. We only parse its sentinel-wrapped JSON result.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from kernelforge.llm.git import git
from kernelforge.config import Config
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec

log = logging.getLogger(__name__)

_RESULT_RE = re.compile(r"__FORGE_RESULT__(.*?)__FORGE_RESULT__", re.DOTALL)

# forge-loop announces its experiment id on stdout at loop start ("Experiment: <id>",
# see loop.runner), so it is present in the captured output even when the loop is
# later hard-killed. Used to decide whether a --result-json file belongs to THIS run.
_EXPERIMENT_RE = re.compile(r"^\s*Experiment:\s*(\S+)\s*$", re.MULTILINE)


def _announced_experiment_id(stdout_text: str) -> str | None:
    """The experiment_id forge-loop announced on stdout this run, or None."""
    m = _EXPERIMENT_RE.search(stdout_text)
    return m.group(1) if m else None


def _forge_loop_argv() -> list[str]:
    """Invoke forge-loop with the SAME interpreter + package as THIS process.

    Prefer ``sys.executable -m kernelforge.cli`` — the exact entry the
    ``kernelforge`` console script maps to (``kernelforge.cli:main``) — so an
    editable install or a multi-venv PATH cannot launch a DIFFERENT installed
    version than the code running right now (cf. ``python -m pip`` over ``pip``).
    Fall back to the PATH console script only if there is no usable interpreter.
    """
    if sys.executable:
        return [sys.executable, "-m", "kernelforge.cli"]
    exe = shutil.which("kernelforge")
    return [exe] if exe else ["kernelforge"]


def _poll_process(proc) -> int | None:
    """Return a subprocess status while remaining compatible with test doubles."""
    poll = getattr(proc, "poll", None)
    if callable(poll):
        return poll()
    return getattr(proc, "returncode", 0)


def _wait_process(proc, timeout: float | None = None) -> int | None:
    """Wait for a subprocess, tolerating minimal test doubles."""
    wait = getattr(proc, "wait", None)
    if not callable(wait):
        return _poll_process(proc)
    try:
        return wait(timeout=timeout)
    except TypeError:
        return wait()


def _terminate_process_group(proc, grace_sec: float = 10.0) -> None:
    """Terminate the complete forge-loop process group, then force-kill it."""
    if _poll_process(proc) is not None:
        return
    pid = getattr(proc, "pid", None)
    try:
        if pid:
            os.killpg(pid, signal.SIGTERM)
        else:
            proc.terminate()
    except (AttributeError, OSError):
        # The process may have exited between poll and signal delivery.
        pass
    try:
        _wait_process(proc, timeout=grace_sec)
        return
    except subprocess.TimeoutExpired:
        # Escalate below when the graceful termination window expires.
        pass
    try:
        if pid:
            os.killpg(pid, signal.SIGKILL)
        else:
            proc.kill()
    except (AttributeError, OSError):
        # A concurrent process exit makes the force-kill unnecessary.
        pass
    try:
        _wait_process(proc, timeout=5.0)
    except subprocess.TimeoutExpired:
        # Best-effort final reap; the caller will still restore the verified best.
        pass


def _restore_best_kernel(
    spec: RewriteSpec,
    *,
    best_commit: str,
    fallback_content: bytes | None,
    fallback_mode: int | None,
) -> bool:
    """Restore the last verified FlyDSL kernel after a clean exit or hard stop."""
    kernel = Path(spec.flydsl_kernel)
    workspace = Path(spec.workspace).resolve()
    try:
        relative = kernel.resolve().relative_to(workspace)
    except ValueError:
        relative = None

    if best_commit and relative is not None:
        exists = git(
            "-C",
            str(workspace),
            "cat-file",
            "-e",
            f"{best_commit}^{{commit}}",
            check=False,
        )
        if exists.returncode == 0:
            restored = git(
                "-C",
                str(workspace),
                "restore",
                "--source",
                best_commit,
                "--staged",
                "--worktree",
                "--",
                relative.as_posix(),
                check=False,
            )
            if restored.returncode == 0:
                return True

    if fallback_content is None:
        return False
    kernel.parent.mkdir(parents=True, exist_ok=True)
    kernel.write_bytes(fallback_content)
    if fallback_mode is not None:
        kernel.chmod(fallback_mode)
    return True


def run_optimize(
    spec: RewriteSpec,
    driver_path: str,
    config: Config,
    *,
    experiments_dir: str,
    max_hours: float = 1.0,
    git_branch: str = "forge-rewrite-optimize",
    permission_mode: str | None = None,
    supervisor_backend: str = "codex",
    profile_timeout_sec: int = 1800,
    result_json: str | None = None,
    deadline_unix: float | None = None,
    stop_at_unix: float | None = None,
) -> dict:
    """Run forge-loop over the FlyDSL kernel; return its parsed result dict.

    Returns {} when forge-loop cannot be launched or its result cannot be parsed
    (the caller then reports flydsl_best_ms as unknown).
    """
    if result_json is None:
        result_json = str(Path(experiments_dir) / "forge_loop_result.json")

    cmd = _forge_loop_argv() + [
        "forge-loop",
        "--kernel",
        spec.flydsl_kernel,
        "--driver",
        driver_path,
        "--workspace",
        spec.workspace,
        "--experiments-dir",
        str(experiments_dir),
        "--result-json",
        result_json,
        "--snr-threshold",
        str(spec.snr_threshold),
        "--max-hours",
        str(max(1.0, max_hours)),
        "--git-branch",
        git_branch,
        "--gpu-target",
        config.gpu_target,
        "--kernel-backend",
        "flydsl",
        "--task-type",
        "flydsl2flydsl",
        "--source-files",
        spec.flydsl_kernel,
        "--target-functions",
        spec.builder_symbol,
        # The outer rewrite pipeline exclusively owns rewrite KB read/write.
        # Prevent the nested optimizer from touching the generic forge-loop KB.
        "--no-experience-kb",
        # The rewrite driver has already passed its independent dual-path
        # preparation and preflight. The single-path forge-loop preparer has a
        # different contract and must never rewrite it.
        "--no-prepare-task",
        "--supervisor-backend",
        supervisor_backend,
        "--profile-timeout-sec",
        str(profile_timeout_sec),
    ]
    if config.gpu_type:
        cmd += ["--gpu-type", config.gpu_type]
    if deadline_unix and deadline_unix > 0:
        cmd += ["--deadline-unix", str(deadline_unix)]
    # Propagate the selected model only when one is configured; an empty
    # agent_model lets forge-loop resolve its own default from the environment
    # (KERNEL_AGENTS_MODEL). ``Config`` exposes the model as ``agent_model`` —
    # there is no ``config.model``.
    if config.agent_model:
        cmd += ["--model", config.agent_model]
    if permission_mode:
        cmd += ["--permission-mode", permission_mode]

    log.info("optimize: launching forge-loop over %s", spec.flydsl_kernel_name)
    print(f"  [forge-rewrite] optimize: {' '.join(cmd)}", flush=True)

    # Stream forge-loop output through stdout for the caller while collecting it
    # to parse the sentinel-wrapped result.
    collected: list[str] = []
    kernel_path = Path(spec.flydsl_kernel)
    fallback_content = kernel_path.read_bytes() if kernel_path.is_file() else None
    fallback_mode = kernel_path.stat().st_mode & 0o777 if kernel_path.is_file() else None
    terminated_for_deadline = False
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=spec.workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert proc.stdout is not None

        def _stream_output() -> None:
            for line in proc.stdout:
                collected.append(line)
                # The outer rewrite publishes the same __FORGE_RESULT__ contract
                # as forge-loop. Keep the nested sentinel for local parsing, but
                # do not leak it to callers that must see only the final
                # framework-level patch result.
                if "__FORGE_RESULT__" in line:
                    continue
                sys.stdout.write(line)
                sys.stdout.flush()

        stream_thread = threading.Thread(
            target=_stream_output,
            name="forge-rewrite-optimize-output",
            daemon=True,
        )
        stream_thread.start()
        while _poll_process(proc) is None:
            if stop_at_unix and time.time() >= stop_at_unix:
                terminated_for_deadline = True
                print(
                    "  [forge-rewrite] optimize cutoff reached; terminating forge-loop "
                    "and restoring the latest verified best",
                    flush=True,
                )
                _terminate_process_group(proc)
                break
            time.sleep(0.1)
        _wait_process(proc)
        stream_thread.join(timeout=5.0)
    except Exception as e:  # noqa: BLE001 - a launch/stream failure must not crash the whole rewrite pipeline
        # Honor this function's contract ("Returns {} when forge-loop cannot be
        # launched"): a missing kernelforge on PATH, a bad interpreter, or a
        # malformed command must NOT propagate a traceback out of run_rewrite (which
        # would skip the final result + sentinel). The caller then keeps the
        # port-only baseline as the final result.
        log.warning("optimize: forge-loop could not be launched/run (%s: %s)", type(e).__name__, e)
        print(
            f"  [forge-rewrite] OPTIMIZE launch failed ({type(e).__name__}: {e}); keeping the port-only result",
            flush=True,
        )
        _restore_best_kernel(
            spec,
            best_commit="",
            fallback_content=fallback_content,
            fallback_mode=fallback_mode,
        )
        return {"terminated_for_deadline": True} if terminated_for_deadline else {}
    stdout_text = "".join(collected)

    # Trust --result-json only if it belongs to THIS run, keyed on experiment_id.
    # forge-loop writes the file on every new best (not only at the end) and stamps
    # its experiment_id into it, and announces that same id on stdout. So even when
    # the loop is hard-killed (e.g. an outer time-budget SIGTERM) AFTER it produced a
    # better kernel, the file it left is still this run's result and we report it. A
    # mismatched/absent id means the file is stale (a prior run reusing this
    # experiments-dir) and is ignored.
    expected_id = _announced_experiment_id(stdout_text)
    try:
        parsed = json.loads(Path(result_json).read_text())
    except (OSError, ValueError):
        parsed = None
    result: dict = {}
    if parsed is not None and expected_id and parsed.get("experiment_id") == expected_id:
        result = parsed

    # Otherwise fall back to the stdout sentinel — inherently this run's output
    # (captured live), and only emitted on a clean exit.
    m = _RESULT_RE.search(stdout_text) if not result else None
    if m is not None:
        try:
            result = json.loads(m.group(1))
        except ValueError:
            log.warning("optimize: could not parse forge-loop sentinel JSON")
    if not result:
        log.warning(
            "optimize: no trusted forge-loop result (exit %s, expected experiment_id %s)",
            proc.returncode,
            expected_id,
        )

    restored = _restore_best_kernel(
        spec,
        best_commit=str(result.get("best_commit") or ""),
        fallback_content=fallback_content,
        fallback_mode=fallback_mode,
    )
    if not result and not terminated_for_deadline:
        return {}
    return {
        **result,
        "terminated_for_deadline": terminated_for_deadline,
        "best_kernel_restored": restored,
    }
