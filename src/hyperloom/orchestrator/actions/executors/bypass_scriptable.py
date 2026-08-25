# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bypass scriptable (server-less) benchmark path.

Some frameworks (xDiT diffusion) are server-less: they run a single CLI
benchmark script that writes an InferenceX-shaped ``inferencex_result.json``
directly (framework/workload_kind/throughput_unit/quality_gate), with no
OpenAI server and no HTTP client. There is no meaningful Python-orchestration
equivalent, so bypass runs the self-contained scriptable benchmark script.

Script resolution (bypass owns its choice; it does NOT depend on Magpie being
importable):
  1. ``$HYPERLOOM_BYPASS_SCRIPTS_DIR`` (operator override / vendored dir),
  2. the package's bundled ``assets/benchmark_scripts`` (frameworks whose
     entrypoint is supplied by the operator),
  3. Magpie's ``scripts/benchmark`` via ``$MAGPIE_PATH`` (reuse when present),
  4. ``<inferencex>/benchmarks`` (staged copies).
"""

from __future__ import annotations

import os
import subprocess
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import build_benchmark_env
from hyperloom.inference_optimizer.session.paths import asset_root


def _scriptable_script_name(framework: str, runner_type: str) -> str:
    """Return the scriptable entrypoint name (e.g. xdit_mi300x.sh)."""
    return f"{framework}_{runner_type}.sh"


def scriptable_script_candidates(
    framework: str,
    runner_type: str,
    inferencex_root: str,
    bench: dict[str, Any] | None = None,
) -> list[Path]:
    """Return the search list ``resolve_scriptable_script`` walks, in order.

    The list is the forensic trail for a miss: launch-time fail-fast and the
    pre-spawn rc=2 path both print it so the operator can see which drawers
    were opened. Existence is not checked here.

    Args:
        framework: Scriptable framework name (e.g. xdit).
        runner_type: GPU runner (e.g. mi300x).
        inferencex_root: InferenceX checkout root (fallback location).
        bench: Materialized benchmark section; absolute ``benchmark_script``
            is listed first when present.

    Returns:
        Candidate paths in resolution order, duplicates dropped.
    """
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    explicit = str((bench or {}).get("benchmark_script") or "").strip()
    if explicit:
        _add(Path(explicit))
    name = _scriptable_script_name(framework, runner_type)
    override = os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "").strip()
    if override:
        _add(Path(override) / name)
    # Bundled entrypoints are version-matched to this checkout, so they must be
    # reachable by name too: any rebuild path that re-pins the bare
    # {framework}_{runner}.sh would otherwise resolve to nothing.
    _add(asset_root() / "assets" / "benchmark_scripts" / name)
    magpie_path = os.environ.get("MAGPIE_PATH", "").strip()
    if magpie_path:
        _add(Path(magpie_path, "Magpie", "scripts", "benchmark", name))
    _add(Path(inferencex_root, "benchmarks", name))
    return candidates


def resolve_scriptable_script(
    framework: str,
    runner_type: str,
    inferencex_root: str,
    bench: dict[str, Any] | None = None,
) -> Path | None:
    """Resolve the scriptable benchmark script path.

    Args:
        framework: Scriptable framework name (e.g. xdit).
        runner_type: GPU runner (e.g. mi300x).
        inferencex_root: InferenceX checkout root (fallback location).
        bench: Materialized benchmark section; absolute ``benchmark_script``
            wins when present.

    Returns:
        The resolved script path, or None when not found.
    """
    for candidate in scriptable_script_candidates(framework, runner_type, inferencex_root, bench):
        if candidate.is_file():
            return candidate
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
    # Defaults are overridable by the YAML envs; run-scoped values are not.
    defaults: dict[str, str] = {"MODEL": str(bench.get("model") or os.environ.get("MODEL", ""))}
    if bench.get("precision"):
        defaults["PRECISION"] = str(bench["precision"])
    run_scoped: dict[str, str] = {
        "RUNNER_TYPE": runner_type,
        "RESULT_FILENAME": "inferencex_result",
        "RESULT_DIR": str(workspace),
    }
    # Scriptable scripts (e.g. xDiT) gate tracing on PROFILE=1 and read the
    # trace dir from VLLM/SGLANG_TORCH_PROFILER_DIR.
    if profile:
        run_scoped["PROFILE"] = "1"
        if profile_dir:
            run_scoped["VLLM_TORCH_PROFILER_DIR"] = profile_dir
            run_scoped["SGLANG_TORCH_PROFILER_DIR"] = profile_dir
    return build_benchmark_env(defaults, bench.get("envs"), run_scoped)


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
    script = resolve_scriptable_script(framework, runner_type, inferencex_root, bench)
    if script is None:
        name = _scriptable_script_name(framework, runner_type)
        candidates = scriptable_script_candidates(framework, runner_type, inferencex_root, bench)
        tried = "\n".join(f"  - {path}" for path in candidates) or "  (none)"
        error = f"scriptable benchmark script not found for {name}"
        # Pre-spawn miss never opens Popen, so there is no child stderr. Write
        # the search list onto the scriptable log so grid_runner's on-disk
        # fallback (and the Magpie-compatible alias write_report builds) can
        # carry the diagnostic instead of a blank abort_reason.json.
        _write_logs(workspace, "", f"{error}\ntried:\n{tried}\n")
        return 2, error
    env = build_scriptable_env(bench, runner_type, workspace, profile=profile, profile_dir=profile_dir)
    cmd = ["bash", str(script)]
    # Streamed straight to disk instead of captured in memory: a runner killed
    # from outside (lease reap / OOM) must still leave a forensic trail.
    with ExitStack() as stack:
        stdout_sink = stack.enter_context(_open_log_sink(workspace, "scriptable_stdout.log"))
        stderr_sink = stack.enter_context(_open_log_sink(workspace, "scriptable_stderr.log"))
        proc = subprocess.Popen(  # noqa: S603 — cmd is this module's own bash entrypoint
            cmd,
            env=env,
            stdout=stdout_sink,
            stderr=stderr_sink,
        )
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            _write_logs(
                workspace,
                "",
                f"scriptable benchmark timed out after {timeout_s}s",
                append=True,
            )
            return 124, None
    return proc.returncode, None


def _open_log_sink(workspace: Path, name: str):
    """Open a streaming log sink under ``workspace``, falling back to DEVNULL.

    An unwritable workspace must not stop the benchmark, so the sink degrades
    instead of raising.
    """
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        return (workspace / name).open("wb")
    except OSError:
        return nullcontext(subprocess.DEVNULL)


def _write_logs(workspace: Path, stdout: str, stderr: str, *, append: bool = False) -> None:
    """Persist scriptable subprocess logs (best-effort).

    ``append`` keeps already-streamed output intact when a late marker (e.g. a
    timeout note) is added.
    """
    mode = "a" if append else "w"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        if stdout:
            with (workspace / "scriptable_stdout.log").open(mode, encoding="utf-8") as fh:
                fh.write(stdout)
        if stderr:
            with (workspace / "scriptable_stderr.log").open(mode, encoding="utf-8") as fh:
                fh.write(stderr)
    except OSError:
        pass
