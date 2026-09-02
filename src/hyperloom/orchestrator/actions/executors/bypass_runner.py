# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bypass benchmark runner (CLI).

Drop-in alternative to ``python -m Magpie -v benchmark ... --run-mode local``.
It accepts the same CLI flags and the same environment contract, and writes a
Magpie-compatible workspace + ``benchmark_report.json`` so Hyperloom's
executors and collectors consume bypass runs unchanged.

Execution is orchestrated in Python (no shell scripts): start the framework
server, wait for HTTP readiness, run the InferenceX benchmark client, then
optionally run lm-eval. This depends on the InferenceX checkout (benchmark
client + lm-eval), but NOT on the Magpie repository.

Scope: ``--run-mode local`` for sglang/vllm/atom, plus the optional ``RUN_EVAL``
accuracy pass. Also covers the server_lifecycle reuse protocol (persist server
on the first round, reuse on the next), the scriptable (server-less) path for
xDiT diffusion, the multi-node remote-client path (``BENCHMARK_BASE_URL``), and
an additive ``bypass_analysis`` block. Docker/Ray remain deferred.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.env_safety import build_benchmark_env

from . import bypass_analysis
from . import bypass_engine
from . import bypass_report
from . import bypass_scriptable

_FALSE_VALUES = frozenset({"false", "0", "no", "off", ""})


def _as_int(value: Any, default: int) -> int:
    """Coerce to int, tolerating None/str; return default on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_opt_int(value: Any) -> int | None:
    """Coerce to int, or None when unset/blank/invalid."""
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any, default: float) -> float:
    """Coerce to float, tolerating None/str; return default on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _run_eval_enabled(bench_envs: dict[str, Any]) -> bool:
    """Whether RUN_EVAL requests an accuracy pass.

    The materialized YAML wins; the ambient env only fills an absent key, so a
    stale exported ``RUN_EVAL`` cannot resurrect an eval the session turned off.
    """
    raw = bench_envs.get("RUN_EVAL")
    if raw is None:
        raw = os.environ.get("RUN_EVAL", "false")
    return str(raw).strip().lower() not in _FALSE_VALUES


def _tokenize_extra_args(bench_envs: dict[str, Any], framework: str) -> list[str]:
    """Return the framework's extra server args as a token list."""
    key = {"sglang": "EXTRA_SGLANG_ARGS", "vllm": "EXTRA_VLLM_ARGS", "atom": "EXTRA_ATOM_ARGS"}.get(framework, "")
    raw = str(os.environ.get(key) or bench_envs.get(key) or "").strip()
    if not raw:
        return []
    import shlex

    try:
        return shlex.split(raw)
    except ValueError:
        return raw.split()


# Reuse verdicts for a persistent lifecycle server (see _server_reusable).
_REUSE = "reuse"  # healthy port + our pid/meta present -> attach a client round
_BOOT = "boot"  # port not up -> this round boots the server
_FOREIGN = "foreign"  # healthy port but no pid/meta -> not ours; refuse


def _server_reusable(base_url: str, pid_dir: str | None, framework: str, port: int) -> str:
    """Classify whether a persistent lifecycle server can be reused.

    Reuse requires BOTH a healthy ``/health`` and this run's pid/meta files, so
    a port held by a server bypass did not launch (foreign/zombie) is never
    silently reused or booted over - it is reported so the caller fails loudly.

    Returns one of ``_REUSE`` / ``_BOOT`` / ``_FOREIGN``.
    """
    if not bypass_engine.server_health_ok(base_url):
        return _BOOT
    if pid_dir and bypass_engine.lifecycle_files_present(pid_dir, framework, port):
        return _REUSE
    return _FOREIGN


def run_benchmark(
    config_path: Path,
    output_dir: Path,
    *,
    phase: str = "all",
    pid_dir: str | None = None,
    cleanup: bool = True,
) -> int:
    """Run a benchmark, optionally as a lifecycle phase.

    phase="all" (default): start server -> client -> teardown (unchanged).
    phase="server": start a persistent server, write pid/meta, exit WITHOUT
        tearing it down (a later client phase reuses it). Requires pid_dir.
    phase="client": reuse the already-running server; run client (+optional
        eval); tear the server down only when cleanup is True.

    Args:
        config_path: Materialized benchmark config YAML.
        output_dir: Output root for the workspace.
        phase: Lifecycle phase (all|server|client).
        pid_dir: Shared dir for pid/meta files (required for server/client).
        cleanup: When phase=client, whether to teardown the server after.

    Returns:
        Process exit code (0 on success).
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    bench = cfg.get("benchmark") or {}
    framework = str(bench.get("framework") or "sglang").lower()
    model = str(bench.get("model") or os.environ.get("MODEL", ""))
    bench_envs = dict(bench.get("envs") or {})
    timeout_s = _as_float(bench.get("timeout_seconds"), 3600.0)

    # Scriptable (server-less) frameworks (e.g. xDiT diffusion): no server,
    # no HTTP client. Run the self-contained scriptable benchmark script,
    # which writes inferencex_result.json with a quality_gate.
    from hyperloom.inference_optimizer import framework_registry

    if framework_registry.is_scriptable(framework):
        return _run_scriptable_benchmark(
            framework=framework,
            model=model,
            bench=bench,
            bench_envs=bench_envs,
            timeout_s=timeout_s,
            output_dir=output_dir,
        )

    if framework not in bypass_engine.SERVER_FRAMEWORKS:
        _emit_failure(output_dir, framework, model, f"unsupported framework: {framework!r}")
        return 2

    inferencex_root = bypass_engine.resolve_inferencex_root(bench)
    if not inferencex_root or not Path(inferencex_root).is_dir():
        _emit_failure(
            output_dir,
            framework,
            model,
            f"InferenceX path not resolvable/usable: {inferencex_root!r}",
        )
        return 2

    workspace = bypass_report.create_workspace(output_dir, framework)
    _snapshot_config(workspace, cfg)

    tp = _as_int(os.environ.get("TP") or bench_envs.get("TP"), 1)
    conc = _as_int(os.environ.get("CONC") or bench_envs.get("CONC"), 32)
    isl = _as_int(os.environ.get("ISL") or bench_envs.get("ISL"), 1024)
    osl = _as_int(os.environ.get("OSL") or bench_envs.get("OSL"), 512)
    rrr = _as_float(os.environ.get("RANDOM_RANGE_RATIO") or bench_envs.get("RANDOM_RANGE_RATIO"), 0.5)
    max_model_len = os.environ.get("MAX_MODEL_LEN") or bench_envs.get("MAX_MODEL_LEN")
    max_model_len_i = _as_int(max_model_len, 0) or None
    port = _as_int(os.environ.get("PORT") or bench_envs.get("PORT"), bypass_engine.DEFAULT_PORT)

    profiler = (bench.get("profiler") or {}).get("torch_profiler") or {}
    profile = bool(profiler.get("enabled"))
    profile_dir = str(workspace / "torch_trace") if profile else None
    if profile_dir:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)

    server_log = workspace / "server.log"
    base_url = f"http://127.0.0.1:{port}"

    # Multi-node remote client: Hyperloom injects BENCHMARK_BASE_URL (+
    # MAGPIE_RUN_PHASE=client) so the benchmark targets a head-pod server
    # instead of launching one locally. bypass mirrors that: no local server,
    # client (+eval) against the remote base_url, no teardown (remote server is
    # not ours). See _multi_node_env.magpie_remote_env.
    remote_base_url = os.environ.get("BENCHMARK_BASE_URL", "").strip()
    if remote_base_url:
        start = time.time()
        rc = _run_client_and_eval(
            inferencex_root=inferencex_root,
            model=model,
            base_url=remote_base_url,
            isl=isl,
            osl=osl,
            conc=conc,
            rrr=rrr,
            profile=profile,
            bench_envs=bench_envs,
            workspace=workspace,
            timeout_s=timeout_s,
        )
        return _finalize_report(
            workspace=workspace,
            framework=framework,
            model=model,
            server_log=server_log,
            bench_envs=bench_envs,
            start=start,
            rc=rc,
            profile=profile,
        )

    # server_lifecycle.server_ready_timeout_s (injected by inject_lifecycle,
    # default SERVER_READY_TIMEOUT_SEC / INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC) is the
    # server-boot budget for lifecycle rounds. It only bounds waiting for the
    # server to come up; the client benchmark still uses timeout_seconds. When
    # absent (non-lifecycle run) fall back to timeout_s so behavior is unchanged.
    sl = bench.get("server_lifecycle") or {}
    server_ready_timeout = _as_float(sl.get("server_ready_timeout_s"), timeout_s)

    if phase == "server":
        if not pid_dir:
            _emit_failure(output_dir, framework, model, "phase=server requires pid_dir", workspace=workspace)
            return 2
        return _run_server_phase(
            framework=framework,
            model=model,
            tp=tp,
            port=port,
            max_model_len=max_model_len_i,
            profile=profile,
            profile_dir=profile_dir,
            bench_envs=bench_envs,
            server_log=server_log,
            base_url=base_url,
            server_ready_timeout_s=server_ready_timeout,
            pid_dir=pid_dir,
            workspace=workspace,
            output_dir=output_dir,
        )

    if phase == "client":
        return _run_client_phase(
            framework=framework,
            model=model,
            port=port,
            conc=conc,
            isl=isl,
            osl=osl,
            rrr=rrr,
            profile=profile,
            bench_envs=bench_envs,
            inferencex_root=inferencex_root,
            base_url=base_url,
            server_log=server_log,
            timeout_s=timeout_s,
            workspace=workspace,
            pid_dir=pid_dir,
            cleanup=cleanup,
            start=time.time(),
        )

    # YAML-driven lifecycle: run_grid injects benchmark.server_lifecycle
    # (cleanup/pid_dir/port) and drives warmup(cleanup=false)+measure(cleanup=
    # true) as two identical calls, delegating phase choice to us. Honor it so
    # bypass reuse works through run_grid with no scheduler changes.
    if phase == "all" and bool(sl.get("enabled")):
        sl_cleanup = bool(sl.get("cleanup", True))
        sl_pid_dir = str(sl.get("pid_dir") or workspace)
        verdict = _server_reusable(base_url, sl_pid_dir, framework, port)
        if verdict == _REUSE:
            # A persistent server from a prior round is up AND ours: reuse it.
            return _run_client_phase(
                framework=framework,
                model=model,
                port=port,
                conc=conc,
                isl=isl,
                osl=osl,
                rrr=rrr,
                profile=profile,
                bench_envs=bench_envs,
                inferencex_root=inferencex_root,
                base_url=base_url,
                server_log=server_log,
                timeout_s=timeout_s,
                workspace=workspace,
                pid_dir=sl_pid_dir,
                cleanup=sl_cleanup,
                start=time.time(),
            )
        if verdict == _FOREIGN:
            # Healthy port but no pid/meta: a server we did not launch holds it.
            # Refuse rather than reuse (reuse-key mismatch) or boot over it.
            _write_report(
                workspace,
                framework,
                model,
                False,
                time.time(),
                [f"port {port} in use by a non-bypass server (no lifecycle pid/meta)"],
                profiling_enabled=profile,
            )
            return 1
        # verdict == _BOOT: no server yet. Start + persist, run this round's
        # client, then honor cleanup.
        return _run_lifecycle_all(
            framework=framework,
            model=model,
            tp=tp,
            port=port,
            max_model_len=max_model_len_i,
            profile=profile,
            profile_dir=profile_dir,
            bench_envs=bench_envs,
            server_log=server_log,
            base_url=base_url,
            timeout_s=timeout_s,
            server_ready_timeout_s=server_ready_timeout,
            pid_dir=sl_pid_dir,
            cleanup=sl_cleanup,
            inferencex_root=inferencex_root,
            conc=conc,
            isl=isl,
            osl=osl,
            rrr=rrr,
            workspace=workspace,
            output_dir=output_dir,
        )

    # phase == "all": start server, run client, always teardown.
    server_env = _server_env(profile, profile_dir, bench_envs)
    extra_args = _tokenize_extra_args(bench_envs, framework)
    try:
        server_cmd = bypass_engine.build_server_command(
            framework=framework,
            model=model,
            tp=tp,
            port=port,
            max_model_len=max_model_len_i,
            extra_args=extra_args,
            profile_dir=profile_dir,
            python_exe=sys.executable,
            framework_python=str(bench_envs.get("HYPERLOOM_FRAMEWORK_PYTHON") or ""),
        )
    except ValueError as exc:
        _emit_failure(output_dir, framework, model, str(exc), workspace=workspace)
        return 2

    start = time.time()
    server_proc = _launch_server(server_cmd, server_env, server_log)
    try:
        if not bypass_engine.wait_for_server_ready(base_url, timeout_s=server_ready_timeout):
            _write_report(
                workspace,
                framework,
                model,
                False,
                start,
                ["server did not become ready"],
                profiling_enabled=profile,
            )
            return 1
        rc = _run_client_and_eval(
            inferencex_root=inferencex_root,
            model=model,
            base_url=base_url,
            isl=isl,
            osl=osl,
            conc=conc,
            rrr=rrr,
            profile=profile,
            bench_envs=bench_envs,
            workspace=workspace,
            timeout_s=timeout_s,
        )
    finally:
        _terminate_server(server_proc)

    return _finalize_report(
        workspace=workspace,
        framework=framework,
        model=model,
        server_log=server_log,
        bench_envs=bench_envs,
        start=start,
        rc=rc,
        profile=profile,
    )


def _run_server_phase(
    *,
    framework,
    model,
    tp,
    port,
    max_model_len,
    profile,
    profile_dir,
    bench_envs,
    server_log,
    base_url,
    server_ready_timeout_s,
    pid_dir,
    workspace,
    output_dir,
) -> int:
    """Start a persistent server, write pid/meta, and exit without teardown."""
    server_env = _server_env(profile, profile_dir, bench_envs)
    extra_args = _tokenize_extra_args(bench_envs, framework)
    try:
        server_cmd = bypass_engine.build_server_command(
            framework=framework,
            model=model,
            tp=tp,
            port=port,
            max_model_len=max_model_len,
            extra_args=extra_args,
            profile_dir=profile_dir,
            python_exe=sys.executable,
            framework_python=str(bench_envs.get("HYPERLOOM_FRAMEWORK_PYTHON") or ""),
        )
    except ValueError as exc:
        _emit_failure(output_dir, framework, model, str(exc), workspace=workspace)
        return 2
    proc = _launch_server(server_cmd, server_env, server_log)
    if not bypass_engine.wait_for_server_ready(base_url, timeout_s=server_ready_timeout_s):
        _terminate_server(proc)
        _write_report(
            workspace,
            framework,
            model,
            False,
            time.time(),
            ["server did not become ready"],
            profiling_enabled=profile,
        )
        return 1
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = proc.pid
    bypass_engine.write_lifecycle_files(
        pid_dir=pid_dir,
        framework=framework,
        port=port,
        pid=proc.pid,
        pgid=pgid,
        model=model,
    )
    # Do NOT terminate: the server stays up for the reuse client phase.
    return 0


def _run_client_phase(
    *,
    framework,
    model,
    port,
    conc,
    isl,
    osl,
    rrr,
    profile,
    bench_envs,
    inferencex_root,
    base_url,
    server_log,
    timeout_s,
    workspace,
    pid_dir,
    cleanup,
    start,
) -> int:
    """Reuse a running server; run client (+eval); teardown when cleanup."""
    if not pid_dir:
        _write_report(
            workspace,
            framework,
            model,
            False,
            start,
            ["phase=client requires pid_dir"],
            profiling_enabled=profile,
        )
        return 1
    verdict = _server_reusable(base_url, pid_dir, framework, port)
    if verdict != _REUSE:
        reason = (
            "no healthy server to reuse"
            if verdict == _BOOT
            else f"port {port} in use by a non-bypass server (no lifecycle pid/meta)"
        )
        _write_report(workspace, framework, model, False, start, [reason], profiling_enabled=profile)
        return 1
    try:
        rc = _run_client_and_eval(
            inferencex_root=inferencex_root,
            model=model,
            base_url=base_url,
            isl=isl,
            osl=osl,
            conc=conc,
            rrr=rrr,
            profile=profile,
            bench_envs=bench_envs,
            workspace=workspace,
            timeout_s=timeout_s,
        )
    finally:
        if cleanup and pid_dir:
            from ._server_lifecycle import teardown_lifecycle_server

            teardown_lifecycle_server(pid_dir=pid_dir, framework=framework, port=port)
    return _finalize_report(
        workspace=workspace,
        framework=framework,
        model=model,
        server_log=server_log,
        bench_envs=bench_envs,
        start=start,
        rc=rc,
        profile=profile,
    )


def _run_lifecycle_all(
    *,
    framework,
    model,
    tp,
    port,
    max_model_len,
    profile,
    profile_dir,
    bench_envs,
    server_log,
    base_url,
    timeout_s,
    server_ready_timeout_s,
    pid_dir,
    cleanup,
    inferencex_root,
    conc,
    isl,
    osl,
    rrr,
    workspace,
    output_dir,
) -> int:
    """Start + persist a server, run this round's client, teardown iff cleanup.

    Used for the first round of a YAML-driven lifecycle sequence: the server is
    left running (pid/meta written) so a later reuse round can attach; the
    server is only torn down when this round requests cleanup.
    """
    server_env = _server_env(profile, profile_dir, bench_envs)
    extra_args = _tokenize_extra_args(bench_envs, framework)
    try:
        server_cmd = bypass_engine.build_server_command(
            framework=framework,
            model=model,
            tp=tp,
            port=port,
            max_model_len=max_model_len,
            extra_args=extra_args,
            profile_dir=profile_dir,
            python_exe=sys.executable,
            framework_python=str(bench_envs.get("HYPERLOOM_FRAMEWORK_PYTHON") or ""),
        )
    except ValueError as exc:
        _emit_failure(output_dir, framework, model, str(exc), workspace=workspace)
        return 2
    start = time.time()
    proc = _launch_server(server_cmd, server_env, server_log)
    if not bypass_engine.wait_for_server_ready(base_url, timeout_s=server_ready_timeout_s):
        _terminate_server(proc)
        _write_report(
            workspace,
            framework,
            model,
            False,
            start,
            ["server did not become ready"],
            profiling_enabled=profile,
        )
        return 1
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = proc.pid
    bypass_engine.write_lifecycle_files(
        pid_dir=pid_dir,
        framework=framework,
        port=port,
        pid=proc.pid,
        pgid=pgid,
        model=model,
    )
    rc = _run_client_and_eval(
        inferencex_root=inferencex_root,
        model=model,
        base_url=base_url,
        isl=isl,
        osl=osl,
        conc=conc,
        rrr=rrr,
        profile=profile,
        bench_envs=bench_envs,
        workspace=workspace,
        timeout_s=timeout_s,
    )
    if cleanup:
        _terminate_server(proc)
        from ._server_lifecycle import teardown_lifecycle_server

        teardown_lifecycle_server(pid_dir=pid_dir, framework=framework, port=port)
    return _finalize_report(
        workspace=workspace,
        framework=framework,
        model=model,
        server_log=server_log,
        bench_envs=bench_envs,
        start=start,
        rc=rc,
        profile=profile,
    )


def _run_scriptable_benchmark(
    *,
    framework,
    model,
    bench,
    bench_envs,
    timeout_s,
    output_dir,
) -> int:
    """Run a server-less scriptable benchmark (e.g. xDiT) and write the report."""
    inferencex_root = bypass_engine.resolve_inferencex_root(bench)
    workspace = bypass_report.create_workspace(output_dir, framework)
    _snapshot_config(workspace, {"benchmark": bench})
    runner_type = str(bench.get("runner_type") or os.environ.get("RUNNER_TYPE") or "mi300x").lower()
    # Profiler parity with the serving path: honor torch_profiler.enabled so
    # scriptable scripts (xDiT) trace into the workspace torch_trace dir.
    profiler = (bench.get("profiler") or {}).get("torch_profiler") or {}
    profile = bool(profiler.get("enabled"))
    profile_dir = str(workspace / "torch_trace") if profile else None
    if profile_dir:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
    start = time.time()
    rc, error = bypass_scriptable.run_scriptable(
        framework=framework,
        runner_type=runner_type,
        inferencex_root=str(inferencex_root or ""),
        bench=bench,
        workspace=workspace,
        timeout_s=timeout_s,
        profile=profile,
        profile_dir=profile_dir,
    )
    if error is not None:
        _write_report(workspace, framework, model, False, start, [error], profiling_enabled=profile)
        return 2
    raw = _load_raw_result(workspace)
    success = rc == 0 and raw is not None
    errors: list[str] = []
    if rc != 0:
        errors.append(f"scriptable benchmark exited {rc}")
    if raw is None:
        errors.append("inferencex_result.json not produced")
    _write_report(
        workspace,
        framework,
        model,
        success,
        start,
        errors,
        raw=raw,
        profiling_enabled=profile,
    )
    return 0 if success else (rc or 1)


def _ensure_eval_deps(python_exe: str) -> None:
    """Ensure ``lm_eval`` is importable by ``python_exe`` before an accuracy pass.

    The Magpie path relies on InferenceX's ``benchmark_lib.sh`` runtime shim to
    auto-install ``lm_eval`` when RUN_EVAL is on; bypass does not shell through
    that shim, so on a bypass-only box (Magpie install skipped) ``lm_eval`` may
    never have been installed and the eval subprocess dies immediately. Mirror
    the shim here: probe-then-install with the SAME interpreter that runs eval.

    Best-effort: a failed install is not fatal here — the eval subprocess will
    then fail and be surfaced through its exit code (``_finalize_report`` already
    fails the run on a non-zero eval rc), so we never crash the whole benchmark
    on a transient pip error.

    Args:
        python_exe (str): The interpreter that will run ``python -m lm_eval``.
    """
    probe = subprocess.run([python_exe, "-c", "import lm_eval"], capture_output=True)
    if probe.returncode == 0:
        return
    subprocess.run(
        [python_exe, "-m", "pip", "install", "--quiet", "--no-cache-dir", "lm_eval"],
        check=False,
    )


def _run_client_and_eval(
    *,
    inferencex_root,
    model,
    base_url,
    isl,
    osl,
    conc,
    rrr,
    profile,
    bench_envs,
    workspace,
    timeout_s,
) -> int:
    """Run the InferenceX client, then optional eval; return client rc."""
    # Honor materializer-computed request sizing (env then YAML envs) so the
    # benchmark scale matches Magpie; fall back to build_client_command
    # defaults (conc*10 / 2*conc) when unset.
    num_prompts = _as_opt_int(os.environ.get("NUM_PROMPTS") or bench_envs.get("NUM_PROMPTS"))
    num_warmups = _as_opt_int(os.environ.get("NUM_WARMUPS") or bench_envs.get("NUM_WARMUPS"))
    client_cmd = bypass_engine.build_client_command(
        inferencex_root=inferencex_root,
        python_exe=sys.executable,
        model=model,
        base_url=base_url,
        isl=isl,
        osl=osl,
        conc=conc,
        random_range_ratio=rrr,
        result_dir=str(workspace),
        result_filename="inferencex_result",
        num_prompts=num_prompts,
        num_warmups=num_warmups,
        profile=profile,
        trust_remote_code=True,
    )
    rc = _run_subprocess(client_cmd, timeout_s, workspace, "client")
    if rc == 0 and _run_eval_enabled(bench_envs):
        _ensure_eval_deps(sys.executable)
        eval_cmd = bypass_engine.build_eval_command(
            python_exe=sys.executable,
            model=model,
            base_url=base_url,
            conc=conc,
            out_dir=str(workspace / "lm_eval"),
            tasks=str(bench_envs.get("MAGPIE_EVAL_TASKS") or os.environ.get("MAGPIE_EVAL_TASKS", "")).strip()
            or "gsm8k",
            limit=(str(bench_envs.get("MAGPIE_EVAL_LIMIT") or os.environ.get("MAGPIE_EVAL_LIMIT", "")).strip() or None),
        )
        eval_rc = _run_subprocess(eval_cmd, timeout_s, workspace, "eval")
        # Magpie's ``run_eval ... || exit $?`` aborts the benchmark when the
        # accuracy pass fails, so a healthy client run with a failed eval is a
        # failed run - not a silently-passing one. Propagate the eval exit code
        # so _finalize_report fails the run and emits the same marker baseline's
        # eval-rooted RUN_EVAL=false fallback keys on (_EVAL_FAILURE_MARKERS).
        if eval_rc != 0:
            _write_eval_returncode(workspace, eval_rc)
            return eval_rc
    return rc


def _finalize_report(*, workspace, framework, model, server_log, bench_envs, start, rc, profile=False) -> int:
    """Parse raw result, build analysis, write report; return exit code."""
    raw = _load_raw_result(workspace)
    eval_rc = _read_eval_returncode(workspace)
    success = rc == 0 and eval_rc == 0 and raw is not None
    errors: list[str] = []
    if eval_rc != 0:
        # Mirror InferenceX's benchmark_lib.sh message so baseline's
        # _is_eval_rooted_failure recognizes a bypass eval failure too.
        errors.append(f"run_eval failed with exit code {eval_rc}")
    elif rc != 0:
        errors.append(f"benchmark client exited {rc}")
    if raw is None:
        errors.append("inferencex_result.json not produced")
    client_stderr = _read_log(workspace / "client_stderr.log")
    analysis = bypass_analysis.build_analysis(
        workspace=workspace,
        server_log=server_log,
        success=success,
        stderr_text=client_stderr,
        run_eval=_run_eval_enabled(bench_envs),
    )
    _write_report(
        workspace,
        framework,
        model,
        success,
        start,
        errors,
        raw=raw,
        analysis=analysis,
        profiling_enabled=profile,
    )
    if success:
        return 0
    return rc or eval_rc or 1


def _server_env(
    profile: bool,
    profile_dir: str | None,
    bench_envs: dict | None = None,
) -> dict[str, str]:
    """Build the server subprocess env from the materialized benchmark envs.

    The whole mapping is exported, so an env-only candidate is a real experiment
    rather than a rerun of the baseline. That also carries
    ``AITER_LOG_TUNED_CONFIG`` through to the server, which bypass runs need:
    without it their logs have no tuned-config hit lines, and both the GEMM
    demand list and the apply verdict silently lose their input.
    """
    profiler_dirs = (
        dict.fromkeys(("VLLM_TORCH_PROFILER_DIR", "SGLANG_TORCH_PROFILER_DIR", "ATOM_TORCH_PROFILER_DIR"), profile_dir)
        if profile and profile_dir
        else None
    )
    return build_benchmark_env(bench_envs, profiler_dirs)


def _launch_server(cmd: list[str], env: dict[str, str], server_log: Path) -> subprocess.Popen:
    """Launch the server in its own session, redirecting logs to server.log."""
    log_fh = open(server_log, "w", encoding="utf-8")  # noqa: SIM115 - closed on terminate
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    # Stash the log handle on the proc so _terminate_server can close it; the
    # child holds its own dup'd fd, so closing ours does not truncate the log.
    proc._bypass_log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def _terminate_server(proc: subprocess.Popen | None) -> None:
    """Best-effort teardown of the server process group."""
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=30)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    # Close the server.log handle opened by _launch_server (the child kept its
    # own dup'd fd) so repeated lifecycle rounds don't leak file descriptors.
    log_fh = getattr(proc, "_bypass_log_fh", None)
    if log_fh is not None:
        try:
            log_fh.close()
        except OSError:
            pass


def _run_subprocess(cmd: list[str], timeout_s: float, workspace: Path, tag: str) -> int:
    """Run a client/eval subprocess, appending logs; return its exit code."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=build_benchmark_env(),
        )
    except subprocess.TimeoutExpired:
        _append_log(workspace, tag, "", f"{tag} timed out after {timeout_s}s")
        return 124
    _append_log(workspace, tag, proc.stdout or "", proc.stderr or "")
    return proc.returncode


def _append_log(workspace: Path, tag: str, stdout: str, stderr: str) -> None:
    """Persist a subprocess's stdout/stderr for debugging (best-effort)."""
    try:
        if stdout:
            (workspace / f"{tag}_stdout.log").write_text(stdout, encoding="utf-8")
        if stderr:
            (workspace / f"{tag}_stderr.log").write_text(stderr, encoding="utf-8")
    except OSError:
        pass


def _snapshot_config(workspace: Path, cfg: dict[str, Any]) -> None:
    """Persist the effective config into the workspace (best-effort)."""
    try:
        (workspace / "config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def _load_raw_result(workspace: Path) -> dict[str, Any] | None:
    """Load ``inferencex_result.json`` from the workspace, if present."""
    path = workspace / "inferencex_result.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# Sentinel file carrying a failed eval's exit code from _run_client_and_eval to
# _finalize_report (which only receives the client rc). Keeps the client/eval
# split out of the phase call signatures while still failing the run on eval
# failure. Absent/unreadable means "eval did not fail".
_EVAL_RC_FILE = "eval_returncode"


def _write_eval_returncode(workspace: Path, rc: int) -> None:
    """Persist a failed eval's exit code for _finalize_report (best-effort)."""
    try:
        (workspace / _EVAL_RC_FILE).write_text(str(int(rc)), encoding="utf-8")
    except OSError:
        pass


def _read_eval_returncode(workspace: Path) -> int:
    """Read the eval exit code sentinel; 0 when absent/unreadable."""
    try:
        return int((workspace / _EVAL_RC_FILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def _write_report(
    workspace: Path,
    framework: str,
    model: str,
    success: bool,
    start: float,
    errors: list[str],
    *,
    raw: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
    profiling_enabled: bool = False,
) -> None:
    """Build and write the Magpie-compatible report."""
    report = bypass_report.build_report(
        raw,
        framework=framework,
        model=model,
        success=success,
        workspace_dir=str(workspace),
        execution_time=time.time() - start,
        errors=errors,
        analysis=analysis,
        profiling_enabled=profiling_enabled,
    )
    bypass_report.write_report(workspace, report)


def _read_log(path: Path) -> str:
    """Read a log file for analysis, tolerating absence (best-effort)."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _emit_failure(
    output_dir: Path,
    framework: str,
    model: str,
    error: str,
    *,
    workspace: Path | None = None,
) -> None:
    """Emit a failing report + workspace for a pre-launch error."""
    ws = workspace or bypass_report.create_workspace(output_dir, framework)
    _write_report(ws, framework, model, False, time.time(), [error])


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the bypass CLI parser (Magpie-compatible flags)."""
    parser = argparse.ArgumentParser(prog="hyperloom-bypass-benchmark")
    sub = parser.add_subparsers(dest="mode", required=True)
    bench = sub.add_parser("benchmark", help="Run a framework benchmark")
    bench.add_argument("--benchmark-config", required=True)
    bench.add_argument("--output-dir", required=True)
    bench.add_argument("--run-mode", default="local")
    bench.add_argument("--phase", default="all", choices=["all", "server", "client"])
    bench.add_argument("--server-lifecycle-pid-dir", default=None)
    bench.add_argument("--server-lifecycle-cleanup", default="true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = _build_arg_parser().parse_args(argv)
    if args.mode != "benchmark":
        print(f"unsupported mode: {args.mode}", file=sys.stderr)
        return 2
    if args.run_mode != "local":
        print(
            f"bypass runner supports --run-mode local only, got {args.run_mode}",
            file=sys.stderr,
        )
        return 2
    cleanup = str(getattr(args, "server_lifecycle_cleanup", "true")).strip().lower() not in _FALSE_VALUES
    return run_benchmark(
        Path(args.benchmark_config),
        Path(args.output_dir),
        phase=args.phase,
        pid_dir=args.server_lifecycle_pid_dir,
        cleanup=cleanup,
    )


if __name__ == "__main__":
    raise SystemExit(main())
