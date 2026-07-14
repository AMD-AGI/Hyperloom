# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Bypass benchmark runner (CLI).

Drop-in alternative to ``python -m Magpie -v benchmark ... --run-mode local``.
It accepts the same CLI flags and the same environment contract, and writes a
Magpie-compatible workspace + ``benchmark_report.json`` so Hyperloom's
executors and collectors consume bypass runs unchanged.

Execution is orchestrated in Python (no shell scripts): start the framework
server, wait for HTTP readiness, run the InferenceX benchmark client, then
optionally run lm-eval. This depends on the InferenceX checkout (benchmark
client + lm-eval), but NOT on the Magpie repository.

Scope: single-node ``--run-mode local`` for sglang/vllm/atom, plus the
optional ``RUN_EVAL`` accuracy pass. Docker/Ray/scriptable/server-lifecycle and
richer analysis are deferred.
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

from . import bypass_analysis
from . import bypass_engine
from . import bypass_report

_FALSE_VALUES = frozenset({"false", "0", "no", "off", ""})


def _as_int(value: Any, default: int) -> int:
    """Coerce to int, tolerating None/str; return default on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    """Coerce to float, tolerating None/str; return default on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _run_eval_enabled(bench_envs: dict[str, Any]) -> bool:
    """Whether RUN_EVAL requests an accuracy pass (env then YAML envs)."""
    raw = os.environ.get("RUN_EVAL")
    if raw is None:
        raw = str(bench_envs.get("RUN_EVAL", "false"))
    return str(raw).strip().lower() not in _FALSE_VALUES


def _tokenize_extra_args(bench_envs: dict[str, Any], framework: str) -> list[str]:
    """Return the framework's extra server args as a token list."""
    key = {"sglang": "EXTRA_SGLANG_ARGS", "vllm": "EXTRA_VLLM_ARGS", "atom": "EXTRA_ATOM_ARGS"}.get(
        framework, ""
    )
    raw = str(os.environ.get(key) or bench_envs.get(key) or "").strip()
    if not raw:
        return []
    import shlex

    try:
        return shlex.split(raw)
    except ValueError:
        return raw.split()


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

    if framework not in bypass_engine.SERVER_FRAMEWORKS:
        _emit_failure(output_dir, framework, model, f"unsupported framework: {framework!r}")
        return 2

    inferencex_root = bypass_engine.resolve_inferencex_root(bench)
    if not inferencex_root or not Path(inferencex_root).is_dir():
        _emit_failure(
            output_dir, framework, model,
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

    if phase == "server":
        if not pid_dir:
            _emit_failure(output_dir, framework, model, "phase=server requires pid_dir", workspace=workspace)
            return 2
        return _run_server_phase(
            framework=framework, model=model, tp=tp, port=port,
            max_model_len=max_model_len_i, profile=profile, profile_dir=profile_dir,
            bench_envs=bench_envs, server_log=server_log, base_url=base_url,
            timeout_s=timeout_s, pid_dir=pid_dir, workspace=workspace, output_dir=output_dir,
        )

    if phase == "client":
        return _run_client_phase(
            framework=framework, model=model, port=port, conc=conc, isl=isl, osl=osl,
            rrr=rrr, profile=profile, bench_envs=bench_envs, inferencex_root=inferencex_root,
            base_url=base_url, server_log=server_log, timeout_s=timeout_s,
            workspace=workspace, pid_dir=pid_dir, cleanup=cleanup, start=time.time(),
        )

    # phase == "all": start server, run client, always teardown.
    server_env = _server_env(profile, profile_dir)
    extra_args = _tokenize_extra_args(bench_envs, framework)
    try:
        server_cmd = bypass_engine.build_server_command(
            framework=framework, model=model, tp=tp, port=port,
            max_model_len=max_model_len_i, extra_args=extra_args, profile_dir=profile_dir,
        )
    except ValueError as exc:
        _emit_failure(output_dir, framework, model, str(exc), workspace=workspace)
        return 2

    start = time.time()
    server_proc = _launch_server(server_cmd, server_env, server_log)
    try:
        if not bypass_engine.wait_for_server_ready(base_url, timeout_s=timeout_s):
            _write_report(workspace, framework, model, False, start, ["server did not become ready"])
            return 1
        rc = _run_client_and_eval(
            inferencex_root=inferencex_root, model=model, base_url=base_url,
            isl=isl, osl=osl, conc=conc, rrr=rrr, profile=profile,
            bench_envs=bench_envs, workspace=workspace, timeout_s=timeout_s,
        )
    finally:
        _terminate_server(server_proc)

    return _finalize_report(
        workspace=workspace, framework=framework, model=model, server_log=server_log,
        bench_envs=bench_envs, start=start, rc=rc,
    )


def _run_server_phase(
    *, framework, model, tp, port, max_model_len, profile, profile_dir,
    bench_envs, server_log, base_url, timeout_s, pid_dir, workspace, output_dir,
) -> int:
    """Start a persistent server, write pid/meta, and exit without teardown."""
    server_env = _server_env(profile, profile_dir)
    extra_args = _tokenize_extra_args(bench_envs, framework)
    try:
        server_cmd = bypass_engine.build_server_command(
            framework=framework, model=model, tp=tp, port=port,
            max_model_len=max_model_len, extra_args=extra_args, profile_dir=profile_dir,
        )
    except ValueError as exc:
        _emit_failure(output_dir, framework, model, str(exc), workspace=workspace)
        return 2
    proc = _launch_server(server_cmd, server_env, server_log)
    if not bypass_engine.wait_for_server_ready(base_url, timeout_s=timeout_s):
        _terminate_server(proc)
        _write_report(workspace, framework, model, False, time.time(), ["server did not become ready"])
        return 1
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = proc.pid
    bypass_engine.write_lifecycle_files(
        pid_dir=pid_dir, framework=framework, port=port, pid=proc.pid, pgid=pgid, model=model,
    )
    # Do NOT terminate: the server stays up for the reuse client phase.
    return 0


def _run_client_phase(
    *, framework, model, port, conc, isl, osl, rrr, profile, bench_envs,
    inferencex_root, base_url, server_log, timeout_s, workspace, pid_dir, cleanup, start,
) -> int:
    """Reuse a running server; run client (+eval); teardown when cleanup."""
    if not bypass_engine.server_health_ok(base_url):
        _write_report(workspace, framework, model, False, start, ["no healthy server to reuse"])
        return 1
    try:
        rc = _run_client_and_eval(
            inferencex_root=inferencex_root, model=model, base_url=base_url,
            isl=isl, osl=osl, conc=conc, rrr=rrr, profile=profile,
            bench_envs=bench_envs, workspace=workspace, timeout_s=timeout_s,
        )
    finally:
        if cleanup and pid_dir:
            from ._server_lifecycle import teardown_lifecycle_server

            teardown_lifecycle_server(pid_dir=pid_dir, framework=framework, port=port)
    return _finalize_report(
        workspace=workspace, framework=framework, model=model, server_log=server_log,
        bench_envs=bench_envs, start=start, rc=rc,
    )


def _run_client_and_eval(
    *, inferencex_root, model, base_url, isl, osl, conc, rrr, profile,
    bench_envs, workspace, timeout_s,
) -> int:
    """Run the InferenceX client, then optional eval; return client rc."""
    client_cmd = bypass_engine.build_client_command(
        inferencex_root=inferencex_root, python_exe=sys.executable, model=model,
        base_url=base_url, isl=isl, osl=osl, conc=conc, random_range_ratio=rrr,
        result_dir=str(workspace), result_filename="inferencex_result",
        profile=profile, trust_remote_code=True,
    )
    rc = _run_subprocess(client_cmd, timeout_s, workspace, "client")
    if rc == 0 and _run_eval_enabled(bench_envs):
        eval_cmd = bypass_engine.build_eval_command(
            python_exe=sys.executable, model=model, base_url=base_url, conc=conc,
            out_dir=str(workspace / "lm_eval"),
            tasks=os.environ.get("MAGPIE_EVAL_TASKS", "gsm8k").strip() or "gsm8k",
            limit=(os.environ.get("MAGPIE_EVAL_LIMIT", "").strip() or None),
        )
        _run_subprocess(eval_cmd, timeout_s, workspace, "eval")
    return rc


def _finalize_report(*, workspace, framework, model, server_log, bench_envs, start, rc) -> int:
    """Parse raw result, build analysis, write report; return exit code."""
    raw = _load_raw_result(workspace)
    success = rc == 0 and raw is not None
    errors: list[str] = []
    if rc != 0:
        errors.append(f"benchmark client exited {rc}")
    if raw is None:
        errors.append("inferencex_result.json not produced")
    client_stderr = _read_log(workspace / "client_stderr.log")
    analysis = bypass_analysis.build_analysis(
        workspace=workspace, server_log=server_log, success=success,
        stderr_text=client_stderr, run_eval=_run_eval_enabled(bench_envs),
    )
    _write_report(workspace, framework, model, success, start, errors, raw=raw, analysis=analysis)
    return 0 if success else (rc or 1)



def _server_env(profile: bool, profile_dir: str | None) -> dict[str, str]:
    """Build the server subprocess env (inherits parent + profiler dirs)."""
    env = os.environ.copy()
    if profile and profile_dir:
        env["VLLM_TORCH_PROFILER_DIR"] = profile_dir
        env["SGLANG_TORCH_PROFILER_DIR"] = profile_dir
        env["ATOM_TORCH_PROFILER_DIR"] = profile_dir
    return env


def _launch_server(cmd: list[str], env: dict[str, str], server_log: Path) -> subprocess.Popen:
    """Launch the server in its own session, redirecting logs to server.log."""
    log_fh = open(server_log, "w", encoding="utf-8")  # noqa: SIM115 - closed on terminate
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


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


def _run_subprocess(cmd: list[str], timeout_s: float, workspace: Path, tag: str) -> int:
    """Run a client/eval subprocess, appending logs; return its exit code."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
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
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8",
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