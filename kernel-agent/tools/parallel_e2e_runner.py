#!/usr/bin/env python3
"""End-to-end Kernel Agent runner for real model/profile/backend testing.

Flow:
1. Run SGLang baseline/profile for a model to produce a real trace.
2. Analyze the trace and pick a hot kernel.
3. Launch backend optimization attempts in parallel: backend x replicas.
4. Summarize attempts and, when safe evidence exists, emit patch/retest status.

The runner does not fabricate patch effectiveness. If no patchable source or
valid benchmark harness exists, it records that as the experiment outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
TRACE_TOOL = ROOT / "tools" / "tracelens_analysis.py"
OPT_TOOL = ROOT / "tools" / "kernel_optimization.py"
BASELINE_SCRIPT = REPO_ROOT / "marathon/skills/scripts/run_baseline.sh"

# Local sibling import for the collective-name fallback. tools/ is on
# sys.path when this file is run via `python -m` / direct subprocess from
# the kernel_optimization handler; use a relative-style import via the
# tools dir.
sys.path.insert(0, str(ROOT / "tools"))
from _collective_names import kernel_name_implies_multigpu  # noqa: E402
sys.path.pop(0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    if "ANTHROPIC_AUTH_TOKEN" in env:
        env.setdefault("ANTHROPIC_API_KEY", env["ANTHROPIC_AUTH_TOKEN"])
        env.setdefault("OPENAI_API_KEY", env["ANTHROPIC_AUTH_TOKEN"])
        env.setdefault("OOB_API_KEY", env["ANTHROPIC_AUTH_TOKEN"])
    if "AMD_API_KEY" in env:
        env.setdefault("AMD_LLM_API_KEY", env["AMD_API_KEY"])
        env.setdefault("LLM_API_KEY", env["AMD_API_KEY"])
        env.setdefault("GEAK_API_KEY", env["AMD_API_KEY"])
    if "ANTHROPIC_BASE_URL" in env:
        env.setdefault("OPENAI_BASE_URL", env["ANTHROPIC_BASE_URL"])
        env.setdefault("OOB_BASE_URL", env["ANTHROPIC_BASE_URL"])
        env.setdefault("LLM_API_BASE", env["ANTHROPIC_BASE_URL"])
    return env


def _extract_trailing_json(text: str) -> dict[str, Any]:
    """Parse the last top-level JSON object in *text*.

    kernel_optimization.py prints non-JSON lines (e.g. ray.init banner) before
    its result JSON; we tolerate that by scanning from the end.
    """
    if not text:
        raise ValueError("empty stdout")
    end = text.rfind("}")
    if end == -1:
        return json.loads(text)
    depth = 0
    in_str = False
    esc = False
    for i in range(end, -1, -1):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                return json.loads(text[i:end + 1])
    return json.loads(text)


def run_json(cmd: list[str], *, env: dict[str, str], timeout_s: int, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=timeout_s,
        )
        log.write(proc.stdout or "")
        log.write(f"\n[exit_code] {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}; see {log_path}")
    return _extract_trailing_json(proc.stdout or "")


def parse_run_context(path: Path) -> dict[str, str]:
    context: dict[str, str] = {}
    if not path.exists():
        return context
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        context[key] = value.strip().strip("'").strip('"')
    return context


def terminate_owned_server(context: dict[str, str], log_path: Path) -> None:
    pid_s = context.get("SERVER_PID", "")
    if not pid_s.isdigit():
        return
    pid = int(pid_s)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"Terminating owned SGLang server pid={pid}\n")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_baseline_profile(args: argparse.Namespace, run_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    result_dir = run_dir / "baseline"
    trace_dir = run_dir / "traces"
    log_path = run_dir / "logs" / "baseline_profile.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_env = {
        **env,
        "MODEL": args.model_path,
        "TP": str(args.tp),
        "CONC": str(args.conc),
        "ISL": str(args.isl),
        "OSL": str(args.osl),
        "PORT": str(args.port),
        "FRAMEWORK": "sglang",
        "RESULT_DIR": str(result_dir),
        "TRACE_DIR": str(trace_dir),
        "INFERENCEX_PATH": args.inferencex_path,
        "NUM_PROMPTS_MULTIPLIER": str(args.num_prompts_multiplier),
        "GPU_CHECK_RETRIES": "1",
        "SGLANG_EXTRA_ARGS": args.sglang_extra_args,
    }
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Starting baseline/profile\n")
        proc = subprocess.run(
            ["bash", str(BASELINE_SCRIPT)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=cmd_env,
            timeout=int(args.baseline_timeout_min * 60),
        )
        log.write(f"\n[exit_code] {proc.returncode}\n")
    context = parse_run_context(result_dir / "run_context.env")
    terminate_owned_server(context, log_path)
    if proc.returncode != 0:
        raise RuntimeError(f"baseline/profile failed; see {log_path}")
    trace_path = context.get("TRACE_FOR_ANALYSIS", "")
    baseline_json = result_dir / f"baseline_sglang_tp{args.tp}_conc{args.conc}_isl{args.isl}_osl{args.osl}.json"
    return {
        "result_dir": str(result_dir),
        "trace_dir": str(trace_dir),
        "trace_path": trace_path,
        "baseline_json": str(baseline_json) if baseline_json.exists() else "",
        "log_path": str(log_path),
        "context": context,
    }


def _ensure_ray_via_helper(num_gpus: int, log_path: Path) -> bool:
    """Use the kernel-agent self-contained ray_runtime helper."""
    sys.path.insert(0, str(ROOT / "tools" / "backends"))
    from ray_runtime import ensure_ray_cluster  # type: ignore
    return ensure_ray_cluster(num_gpus=num_gpus, log_path=log_path)


def _stop_ray_via_helper(started: bool, log_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "tools" / "backends"))
    from ray_runtime import stop_ray_if_owned  # type: ignore
    stop_ray_if_owned(started, log_path=log_path)


def choose_candidate(candidates: list[dict[str, Any]],
                     kernel_name: str = "",
                     kernel_id: str = "") -> dict[str, Any]:
    if kernel_id:
        for c in candidates:
            if c.get("kernel_id") == kernel_id:
                return c
        raise RuntimeError(f"kernel_id not found in candidates: {kernel_id}")
    if kernel_name:
        for c in candidates:
            if c.get("name") == kernel_name:
                return c
        raise RuntimeError(f"kernel name not found in candidates: {kernel_name}")
    patchable = [c for c in candidates if c.get("source_file") and Path(str(c["source_file"])).exists()]
    if patchable:
        return patchable[0]
    if not candidates:
        raise RuntimeError("no hot kernels found")
    return candidates[0]


def run_one_attempt(
    *,
    backend: str,
    replica: int,
    gpu_id: int,
    args: argparse.Namespace,
    run_dir: Path,
    env: dict[str, str],
    kernel_id: str,
    source_file: str,
    harness_path: str,
    num_gpus: int = 1,
) -> dict[str, Any]:
    # Do NOT set HIP/ROCR/CUDA_VISIBLE_DEVICES here. Ray assigns them inside
    # workers; forcing them on the driver clashes with set_visible_accelerator_ids.
    local_env = {
        **env,
        "WORKSPACE_PATH": str(args.workspace_path),
        "KERNEL_AGENT_NUM_GPUS": str(num_gpus),
    }
    log_path = run_dir / "logs" / "parallel" / f"{backend}_replica{replica}.log"
    cmd = [
        sys.executable, str(OPT_TOOL),
        "--kernel-id", kernel_id,
        "--session-id", args.session_id,
        "--backends", backend,
        "--budget-minutes", str(args.backend_budget_min),
        "--geak-budget-min", str(args.geak_budget_min),
        "--oob-max-turns", str(args.oob_max_turns),
        "--num-gpus", str(num_gpus),
    ]
    if args.geak_cost_limit is not None:
        cmd.extend(["--geak-cost-limit", str(args.geak_cost_limit)])
    if source_file:
        cmd.extend(["--source-file", source_file])
    if harness_path:
        cmd.extend(["--test-harness-path", harness_path])
    started = time.time()
    try:
        result = run_json(cmd, env=local_env, timeout_s=int(args.backend_budget_min * 60) + 120, log_path=log_path)
        status = "ok"
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
        status = "failed"
    return {
        "backend": backend,
        "replica": replica,
        "gpu_id": gpu_id,
        "num_gpus": num_gpus,
        "status": status,
        "elapsed_s": round(time.time() - started, 2),
        "log_path": str(log_path),
        "result": result,
    }


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    write_json(run_dir / "parallel_e2e_summary.json", summary)
    lines = [
        "# Kernel Agent Parallel E2E Summary",
        "",
        f"- Session: `{summary['session_id']}`",
        f"- Model: `{summary['model_path']}`",
        f"- Trace: `{summary.get('trace_path', '')}`",
        f"- Selected kernel: `{summary.get('selected_kernel', {}).get('name', '')}`",
        "",
        "## Backend Attempts",
        "",
    ]
    for item in summary.get("parallel_results", []):
        result = item.get("result", {})
        attempts = result.get("attempts") or []
        attempt_status = attempts[0].get("status") if attempts else item.get("status")
        decision = result.get("proposal", {}).get("decision", "n/a")
        lines.append(
            f"- {item['backend']} replica {item['replica']} GPU {item['gpu_id']}: "
            f"{attempt_status}, decision={decision}, elapsed={item['elapsed_s']}s"
        )
    lines.extend([
        "",
        "## Patch/Retest",
        "",
        summary.get("patch_retest_status", "not attempted"),
        "",
    ])
    (run_dir / "parallel_e2e_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kernel Agent real parallel E2E")
    parser.add_argument("--model-path", default="/wekafs/models/Qwen3-30B-A3B")
    parser.add_argument("--workspace-path", default=os.environ.get("WORKSPACE_PATH", "/workspace"))
    parser.add_argument("--session-id", default=f"qwen3-30b-{int(time.time())}")
    parser.add_argument("--env-file", default="/wekafs/xiaofei/AgentKernelArena/.env")
    parser.add_argument("--inferencex-path", default="/wekafs/fully-local/inference_optimization/InferenceX")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--conc", type=int, default=4)
    parser.add_argument("--isl", type=int, default=256)
    parser.add_argument("--osl", type=int, default=128)
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--num-prompts-multiplier", type=int, default=1)
    parser.add_argument("--sglang-extra-args", default="")
    parser.add_argument("--baseline-timeout-min", type=float, default=45)
    parser.add_argument("--backend-budget-min", type=float, default=60,
                        help="Wall-clock budget per backend attempt in minutes "
                             "(default 60). Applies to claude/codex OOB "
                             "backends. Agents are told to early-exit as "
                             "soon as they hit >=1.50x with passing correctness; "
                             "otherwise they iterate up to ~85%% of this budget "
                             "and SIGTERM at 100%%.")
    parser.add_argument("--geak-budget-min", type=float, default=90,
                        help="Per-attempt wall-clock budget for GEAK only "
                             "(default 90 min). GEAK runs N sub-agent tasks "
                             "serially + a select_patch round; 60 min "
                             "consistently SIGTERMs the select_patch round "
                             "(observed r38/r39).")
    parser.add_argument("--replicas-per-backend", type=int, default=2)
    parser.add_argument("--backends", default=None,
                        help="Comma list of agentic backends. Defaults to "
                             "'geak,claude,codex,cursor' when CURSOR_API_KEY is set, "
                             "otherwise 'geak,claude,codex' (cursor auto-skipped). "
                             "Pass an explicit value to force-include any backend "
                             "(missing keys will surface as 401 attempts). Note: "
                             "'llm' single-shot backend was removed (max_tokens=2048 "
                             "truncated >4KB kernels).")
    parser.add_argument("--oob-max-turns", type=int, default=100)
    parser.add_argument("--geak-cost-limit", type=float, default=None)
    parser.add_argument("--num-gpus-override", type=int, default=0,
                        help="If >0, override candidate.num_gpus_recommended for "
                             "every backend task. Use 2 to test the multi-GPU "
                             "communication-kernel path explicitly.")
    parser.add_argument("--total-gpus", type=int, default=8,
                        help="Total GPUs available on this host; used to cap "
                             "concurrency (default 8 for MI355X box).")
    parser.add_argument("--trace-path", default="")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--kernel-name", default="",
                        help="Pick this exact kernel name from the trace "
                             "(default: first patchable hot kernel).")
    parser.add_argument("--kernel-id", default="",
                        help="Pick by kernel_id (k001/k002/...); takes "
                             "precedence over --kernel-name.")
    parser.add_argument("--reuse-candidates-from", default="",
                        help="Reuse a previous run's kernel_candidates.json "
                             "instead of re-running the trace analysis.")
    args = parser.parse_args()

    # Auto-derive --backends when caller did not pass one. Cursor needs
    # CURSOR_API_KEY (separate Cursor gateway, not the AMD LiteLLM gateway);
    # skip it from the default set when no key is provisioned to avoid
    # spending replica slots on guaranteed 401s. Explicit --backends always
    # wins (failure surfaces in the attempt log).
    if args.backends is None:
        if os.environ.get("CURSOR_API_KEY", "").strip():
            args.backends = "geak,claude,codex,cursor"
        else:
            args.backends = "geak,claude,codex"

    workspace = Path(args.workspace_path)
    run_dir = workspace / "kernel-agent" / "runs" / args.session_id
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **load_env_file(Path(args.env_file)), "WORKSPACE_PATH": str(workspace)}

    summary: dict[str, Any] = {
        "session_id": args.session_id,
        "model_path": args.model_path,
        "created_at": utc_now(),
    }
    try:
        if args.skip_baseline:
            trace_path = args.trace_path
            baseline = {"trace_path": trace_path}
        else:
            baseline = run_baseline_profile(args, run_dir, env)
            trace_path = baseline.get("trace_path", "")
        if not trace_path or not Path(trace_path).exists():
            raise RuntimeError(f"trace path missing after baseline/profile: {trace_path}")
        summary["baseline"] = baseline
        summary["trace_path"] = trace_path

        if args.reuse_candidates_from:
            src = Path(args.reuse_candidates_from)
            if not src.exists():
                raise RuntimeError(f"--reuse-candidates-from path missing: {src}")
            data = json.loads(src.read_text())
            candidates = data if isinstance(data, list) else (
                data.get("hot_kernels") or data.get("kernel_candidates") or [])
            # Mirror to this session's expected location so kernel_optimization
            # can find it via the default candidates_path.
            (run_dir / "kernel_candidates.json").write_text(json.dumps(candidates, indent=2))
            analysis = {"trace_report_path": str(src), "reused": True}
        else:
            analysis = run_json([
                sys.executable, str(TRACE_TOOL),
                "--trace-input", trace_path,
                "--session-id", args.session_id,
                "--model-name", Path(args.model_path).name,
                "--framework", "sglang",
                "--budget-minutes", "60",
            ], env=env, timeout_s=3600, log_path=run_dir / "logs" / "tracelens_analysis_driver.log")
            candidates = analysis.get("hot_kernels", [])
        selected = choose_candidate(candidates, kernel_name=args.kernel_name, kernel_id=args.kernel_id)
        summary["analysis"] = {
            "trace_report_path": analysis.get("trace_report_path"),
            "num_hot_kernels": len(candidates),
        }
        summary["selected_kernel"] = selected

        source_file = str(selected.get("source_file") or "")
        # tracelens_analysis writes `benchmark_files` (plural list); the older
        # `test_harness_path` / `benchmark_file` (singular) keys were never
        # populated, which silently disabled patch_retest. Pick a multi-GPU
        # safe single test: prefer `bench`-style scripts, then `test_*` files.
        bench_files = list(selected.get("benchmark_files") or [])
        # is_multigpu has two paths into True: TraceLens flagged it OR the
        # leaf kernel name matches a known collective (all_reduce /
        # all_gather / reduce_scatter / all_to_all / broadcast / nccl_* /
        # rccl_*). The latter is a fallback for the r24 custom_allreduce
        # case where TraceLens reported is_multigpu=False / num_gpus=1.
        selected_name = str(selected.get("name") or "")
        name_says_collective = kernel_name_implies_multigpu(selected_name)
        is_multigpu = bool(selected.get("is_multigpu")) or name_says_collective
        if name_says_collective and not bool(selected.get("is_multigpu")):
            summary["multigpu_inferred_from_name"] = (
                f"is_multigpu inferred from kernel name {selected_name!r} "
                "(TraceLens did not flag is_multigpu=True)"
            )
        harness_path = ""
        if bench_files:
            preferred = [b for b in bench_files if "bench" in Path(b).name.lower()]
            if not preferred and not is_multigpu:
                preferred = [b for b in bench_files if Path(b).name.startswith("test_")]
            harness_path = (preferred or bench_files)[0]
        if not source_file:
            summary["source_resolution"] = "no source_file in trace/TraceLens output; optimization will run prompt-only"
        if not harness_path:
            summary["benchmark_resolution"] = "no benchmark/test harness resolved; GEAK may be slower or fail"

        # GPU budgeting: communication kernels need >=2 GPUs; computation
        # kernels run on 1. Total concurrency is capped by total_gpus.
        if args.num_gpus_override > 0:
            per_task_gpus = args.num_gpus_override
        else:
            per_task_gpus = int(selected.get("num_gpus_recommended") or 1)
            if is_multigpu and per_task_gpus < 2:
                # The kernel is a collective (per is_multigpu) but
                # TraceLens reported num_gpus_recommended < 2. Force-bump
                # to args.tp (or 2 floor) so the GEAK drop below fires
                # and so torchrun --nproc>=2 is required by the harness.
                per_task_gpus = max(2, int(getattr(args, "tp", 0) or 2))
                summary.setdefault(
                    "per_task_gpus_inferred",
                    f"raised to {per_task_gpus} from collective name "
                    "pattern; TraceLens reported num_gpus_recommended<2",
                )
        backends = [b.strip() for b in args.backends.split(",") if b.strip()]
        # GEAK is fundamentally a SINGLE-GPU patch evaluator (its sub-agent
        # framework spawns nested ray.remote(num_gpus=1) tasks for patch
        # validation). On multi-GPU collective kernels the nested ray task
        # only sees 1 GPU even when the parent has 2+, so any test that
        # calls torch.cuda.set_device(rank>=1) inside torchrun fails with
        # "HIP error: invalid device ordinal" (observed in r20/r22). Drop
        # GEAK from the backend list when per_task_gpus >= 2; let claude /
        # codex / cursor handle multi-GPU collectives via standalone HIP /
        # torchrun.
        # Set ALLOW_GEAK_MULTIGPU=1 to bypass this guard (e.g. when verifying
        # whether an upstream GEAK fix has lifted the limitation).
        backends_dropped: list[str] = []
        if (per_task_gpus >= 2
                and "geak" in backends
                and os.environ.get("ALLOW_GEAK_MULTIGPU") != "1"):
            backends = [b for b in backends if b != "geak"]
            backends_dropped.append("geak (multi-GPU collective unsupported by GEAK sub-agent ray nesting; set ALLOW_GEAK_MULTIGPU=1 to bypass)")
        max_concurrent = max(1, args.total_gpus // max(1, per_task_gpus))
        total_jobs = len(backends) * args.replicas_per_backend
        summary["gpu_plan"] = {
            "per_task_gpus": per_task_gpus,
            "total_gpus": args.total_gpus,
            "max_concurrent_tasks": max_concurrent,
            "total_jobs": total_jobs,
            "backends_dropped": backends_dropped,
        }
        if not backends:
            raise RuntimeError(
                "All backends were dropped (likely all incompatible with this "
                f"kernel's per_task_gpus={per_task_gpus}). Selected backends: "
                f"{args.backends}. Dropped: {backends_dropped}"
            )
        jobs = []
        ray_log = run_dir / "logs" / "ray.log"
        ray_started_by_runner = _ensure_ray_via_helper(args.tp, ray_log)
        summary["ray_started_by_runner"] = ray_started_by_runner
        # ThreadPool fan-out is just for issuing Ray submissions; Ray itself
        # serialises GPU contention via num_gpus reservations.
        with ThreadPoolExecutor(max_workers=min(total_jobs, max_concurrent)) as pool:
            for backend in backends:
                for replica in range(args.replicas_per_backend):
                    jobs.append(pool.submit(
                        run_one_attempt,
                        backend=backend,
                        replica=replica,
                        gpu_id=-1,
                        args=args,
                        run_dir=run_dir,
                        env=env,
                        kernel_id=selected["kernel_id"],
                        source_file=source_file,
                        harness_path=harness_path,
                        num_gpus=per_task_gpus,
                    ))
            parallel_results = [job.result() for job in as_completed(jobs)]
        _stop_ray_via_helper(ray_started_by_runner, ray_log)
        parallel_results.sort(key=lambda x: (x["backend"], x["replica"]))
        summary["parallel_results"] = parallel_results
        if not source_file:
            summary["patch_retest_status"] = "not attempted: no patchable source resolved from real trace"
        elif not harness_path:
            summary["patch_retest_status"] = (
                "not attempted: source resolved but no benchmark/test harness was resolved "
                "for safe patch validation"
            )
        else:
            summary["patch_retest_status"] = (
                "not attempted automatically: backend outputs require review before applying to runtime source"
            )
        summary["completed_at"] = utc_now()
        write_summary(run_dir, summary)
        print(json.dumps({
            "status": "succeeded",
            "run_dir": str(run_dir),
            "summary_json": str(run_dir / "parallel_e2e_summary.json"),
            "summary_md": str(run_dir / "parallel_e2e_summary.md"),
            "selected_kernel": selected,
            "patch_retest_status": summary["patch_retest_status"],
        }, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        try:
            _stop_ray_via_helper(bool(summary.get("ray_started_by_runner")), run_dir / "logs" / "ray.log")
        except Exception:
            pass
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["completed_at"] = utc_now()
        write_summary(run_dir, summary)
        print(json.dumps({
            "status": "failed",
            "run_dir": str(run_dir),
            "summary_json": str(run_dir / "parallel_e2e_summary.json"),
            "error": summary["error"],
        }, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
