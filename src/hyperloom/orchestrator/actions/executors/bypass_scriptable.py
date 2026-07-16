# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Bypass scriptable (server-less) benchmark path.

Some frameworks (xDiT diffusion) are server-less: they run a single CLI
benchmark script that writes an InferenceX-shaped ``inferencex_result.json``
directly (framework/workload_kind/throughput_unit/quality_gate), with no
OpenAI server and no HTTP client. There is no meaningful Python-orchestration
equivalent, so bypass runs the self-contained scriptable benchmark script.

Script resolution (bypass owns its choice; it does NOT depend on Magpie being
importable):
  1. ``$HYPERLOOM_BYPASS_SCRIPTS_DIR`` (operator override / vendored dir),
  2. Magpie's ``scripts/benchmark`` via ``$MAGPIE_PATH`` (reuse when present),
  3. ``<inferencex>/benchmarks`` (staged copies).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def _scriptable_script_name(framework: str, runner_type: str) -> str:
    """Return the scriptable entrypoint name (e.g. xdit_mi300x.sh)."""
    return f"{framework}_{runner_type}.sh"


def resolve_scriptable_script(framework: str, runner_type: str, inferencex_root: str) -> Path | None:
    """Resolve the scriptable benchmark script path.

    Args:
        framework: Scriptable framework name (e.g. xdit).
        runner_type: GPU runner (e.g. mi300x).
        inferencex_root: InferenceX checkout root (fallback location).

    Returns:
        The resolved script path, or None when not found.
    """
    name = _scriptable_script_name(framework, runner_type)
    candidates: list[Path] = []
    override = os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "").strip()
    if override:
        candidates.append(Path(override) / name)
    magpie_path = os.environ.get("MAGPIE_PATH", "").strip()
    if magpie_path:
        candidates.append(Path(magpie_path, "Magpie", "scripts", "benchmark", name))
    candidates.append(Path(inferencex_root, "benchmarks", name))
    for c in candidates:
        if c.is_file():
            return c
    return None


def build_scriptable_env(
    bench: dict[str, Any],
    runner_type: str,
    workspace: Path,
    *,
    profile: bool = False,
    profile_dir: str | None = None,
) -> dict[str, str]:
    """Build the env for a scriptable benchmark script.

    Args:
        bench: The ``benchmark`` section of the config.
        runner_type: Resolved runner type.
        workspace: Per-run workspace directory.
        profile: Whether the torch profiler is enabled for this run.
        profile_dir: Directory the profiler traces should be written to.

    Returns:
        The environment mapping for the scriptable subprocess.
    """
    env = os.environ.copy()
    env["MODEL"] = str(bench.get("model") or env.get("MODEL", ""))
    if bench.get("precision"):
        env["PRECISION"] = str(bench["precision"])
    for key, value in (bench.get("envs") or {}).items():
        env[str(key).upper()] = str(value)
    env["RUNNER_TYPE"] = runner_type
    env["RESULT_FILENAME"] = "inferencex_result"
    env["RESULT_DIR"] = str(workspace)
    # Profiler: scriptable scripts (e.g. xDiT) gate tracing on PROFILE=1 and
    # read the trace dir from VLLM/SGLANG_TORCH_PROFILER_DIR (mirrors the
    # serving path's _server_env). Only set when enabled so default runs are
    # untouched.
    if profile:
        env["PROFILE"] = "1"
        if profile_dir:
            env["VLLM_TORCH_PROFILER_DIR"] = profile_dir
            env["SGLANG_TORCH_PROFILER_DIR"] = profile_dir
    return env


def run_scriptable(
    *,
    framework: str,
    runner_type: str,
    inferencex_root: str,
    bench: dict[str, Any],
    workspace: Path,
    timeout_s: float,
    profile: bool = False,
    profile_dir: str | None = None,
) -> tuple[int, str | None]:
    """Run the scriptable benchmark script.

    Args:
        framework: Scriptable framework name.
        runner_type: GPU runner type.
        inferencex_root: InferenceX checkout root.
        bench: The ``benchmark`` section of the config.
        workspace: Per-run workspace directory.
        timeout_s: Subprocess timeout.
        profile: Whether the torch profiler is enabled for this run.
        profile_dir: Directory the profiler traces should be written to.

    Returns:
        ``(returncode, error)`` — error is a string when a pre-run problem
        occurred (script missing), else None.
    """
    script = resolve_scriptable_script(framework, runner_type, inferencex_root)
    if script is None:
        return 2, f"scriptable benchmark script not found for {framework}_{runner_type}.sh"
    env = build_scriptable_env(
        bench, runner_type, workspace, profile=profile, profile_dir=profile_dir
    )
    cmd = ["bash", str(script)]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _write_logs(workspace, "", f"scriptable benchmark timed out after {timeout_s}s")
        return 124, None
    _write_logs(workspace, proc.stdout or "", proc.stderr or "")
    return proc.returncode, None


def _write_logs(workspace: Path, stdout: str, stderr: str) -> None:
    """Persist scriptable subprocess logs (best-effort)."""
    try:
        if stdout:
            (workspace / "scriptable_stdout.log").write_text(stdout, encoding="utf-8")
        if stderr:
            (workspace / "scriptable_stderr.log").write_text(stderr, encoding="utf-8")
    except OSError:
        pass
