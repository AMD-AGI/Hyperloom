"""Real ``baseline`` ActionRunner — runs Magpie SGLang benchmark.

DESIGN v0.6 §15.2 + §16.1 baseline action.

Wire-up:

    sub.register_executor("baseline", baseline_executor)

Orchestration emits ``delegate{action_name="baseline", params={...}}``;
SubAgentRunner pulls this runner, acquires the action's lanes
(`benchmark_lane` + `server_lifecycle`), runs the Magpie CLI as a
subprocess, parses ``benchmark_report.json``, and returns the result on
the bus as a ``delegated_result`` event so Orchestration can read the
real ``baseline_tput`` next tick.

The runner honours the following RunnerContext.task.params keys
(all optional — defaults below come from BASELINE_DEFAULT_CONFIG):

    config_path:  absolute path to a Magpie YAML config to use
    output_dir:   workspace root for Magpie outputs
    timeout_sec:  hard timeout (overrides YAML's timeout_seconds)

Implementation notes:

* We don't import Magpie programmatically (its CLI takes care of
  InferenceX setup, GPU monitor, workspace creation). subprocess.run
  is the cleanest seam.
* Parses ``benchmark_report.json`` rather than ``inferencex_result.json``
  because the former has the cleaner top-level schema.
* Returns ``error_class`` on failure so the coordinator can route to
  Robustness RCA later (P1-7).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ...paths import asset_root
from ..sub_agent_runner import RunnerContext
from ._grid_runner import server_args_env_name


log = logging.getLogger(__name__)


# Legacy module-level constant kept pointing at the sglang yaml so existing
# tests that import it as a fixture path continue to work. Runtime selection
# of sglang vs vllm yaml goes through `_default_baseline_config()` below.
BASELINE_DEFAULT_CONFIG = (
    asset_root() / "scripts" / "configs" / "baseline_sglang.yaml"
)
BASELINE_DEFAULT_TIMEOUT_SEC = 1200


def _default_baseline_config() -> Path:
    """Resolve default Magpie YAML based on $FRAMEWORK env (sglang/vllm)."""
    fw = os.environ.get("FRAMEWORK", "sglang").strip().lower()
    name = "baseline_vllm.yaml" if fw == "vllm" else "baseline_sglang.yaml"
    return asset_root() / "scripts" / "configs" / name


def _materialize_config_with_envs(
    config_path: Path,
    output_dir: Path,
    *,
    extra_sglang_args: str = "",
    extra_server_args: str = "",
    extra_envs: dict[str, Any] | None = None,
    model_path: str | None = None,
    gpu_type: str | None = None,
) -> Path:
    """Render a per-run Magpie YAML with caller-provided overrides.

    ``model_path`` (when non-empty) overrides the YAML's ``benchmark.model``
    field. This is the single most important override: every shipped config
    under ``scripts/configs/`` has a hardcoded model path (legacy default:
    Qwen-Qwen3-8B) that would otherwise silently win over the user's
    ``--model`` / ``MODEL_PATH`` selection. Always pass ``model_path`` from
    the CLI / SharedState.

    ``gpu_type`` (e.g. ``mi300x`` / ``mi355x``) injects ``benchmark.runner_type``
    so Magpie picks the matching ``{framework}_{gpu_type}.sh`` benchmark
    script. We also ``pop`` any explicit ``benchmark.benchmark_script``
    field, otherwise Magpie's priority-1 user-specified path would win
    over runner_type and lock the run to the wrong GPU's script.
    """
    server_args = (extra_server_args or extra_sglang_args).strip()
    # Always materialize: ISL/OSL/MAX_MODEL_LEN/PRECISION from env should
    # override the yaml defaults even when no other explicit overrides are
    # passed. The short-circuit "return config_path" path led to the yaml's
    # hardcoded ISL=256/OSL=256 winning over the user's --isl/--osl.
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    bench = cfg.setdefault("benchmark", {})
    if model_path:
        bench["model"] = str(model_path)
    # Precision override from CLI/env.
    precision = os.environ.get("PRECISION", "").strip()
    if precision:
        bench["precision"] = precision
    if gpu_type:
        bench["runner_type"] = str(gpu_type)
        bench.pop("benchmark_script", None)
    envs = bench.setdefault("envs", {})
    # Inject runtime-resolvable envs from $ENV. Without these the yaml
    # defaults (ISL=256, OSL=256, TP=1, CONC=8, ROCR_VISIBLE_DEVICES="1")
    # silently win over the user's --tp / TP=N / etc., which is fatal for
    # large models that need TP=8 (e.g. DeepSeek-R1-0528).
    for env_key in ("ISL", "OSL", "MAX_MODEL_LEN", "TP", "CONC"):
        val = os.environ.get(env_key, "").strip()
        if val:
            envs[env_key] = int(val)
    # ROCR_VISIBLE_DEVICES: explicit env override wins. Otherwise, when TP
    # was overridden upward (yaml default TP=1, user set TP=8), expand the
    # GPU list to match TP (0,1,...,TP-1). This avoids vLLM/SGLang seeing
    # only 1 device and OOM-ing on multi-GPU models.
    explicit_rocr = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    if explicit_rocr:
        envs["ROCR_VISIBLE_DEVICES"] = explicit_rocr
    else:
        tp_in_yaml = int(envs.get("TP", 1))
        existing_rocr = str(envs.get("ROCR_VISIBLE_DEVICES", "")).strip()
        existing_count = (
            len([x for x in existing_rocr.split(",") if x.strip()])
            if existing_rocr else 0
        )
        if tp_in_yaml > 1 and existing_count < tp_in_yaml:
            envs["ROCR_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in range(tp_in_yaml)
            )
    # Always run accuracy eval (GSM8K) as part of the benchmark.
    # Magpie's benchmark scripts check RUN_EVAL=true and call run_eval
    # while the server is still alive, avoiding an extra server restart.
    envs.setdefault("RUN_EVAL", "true")

    # Resolve ISL/OSL/CONC early — needed by both profiler config and
    # adaptive NUM_PROMPTS below.
    # Priority: env (CLI-exported) > yaml envs (may be yaml defaults like CONC=8).
    isl_val = int(os.environ.get("ISL") or envs.get("ISL") or 256)
    osl_val = int(os.environ.get("OSL") or envs.get("OSL") or 256)
    conc_val = int(os.environ.get("CONC") or envs.get("CONC") or 8)

    # TraceLens #126: compute steady-state window for profiling configs.
    # Only inject into profile yamls (detected by PROFILE env or
    # torch_profiler.enabled in the yaml). The formulas match issue #126 §3.1.6.
    is_profile = (
        str(envs.get("PROFILE", "")).strip() == "1"
        or (bench.get("profiler", {}).get("torch_profiler", {}).get("enabled") is True)
    )
    if is_profile:
        delay_iters = 5 * conc_val
        max_iters = max(4, min(64, 16 * osl_val // max(conc_val, 1)))
        fw = str(bench.get("framework") or "").lower()
        if "vllm" in fw:
            existing_vllm_args = str(envs.get("EXTRA_VLLM_ARGS", ""))
            # Magpie's vllm_*.sh expands $EXTRA_VLLM_ARGS unquoted, so
            # ${WORKSPACE_DIR} is substituted at server-launch time.
            # capture_torch_profiler_dir enables graph-capture tracing
            # (#126 §3.1.4 item 1) — TraceLens needs this to resolve the
            # actual kernels executed under cudagraph mode.
            profiler_args = (
                f"--profiler-config.delay_iterations {delay_iters} "
                f"--profiler-config.max_iterations {max_iters} "
                f"--profiler-config.capture_torch_profiler_dir "
                f"${{WORKSPACE_DIR}}/torch_trace/capture_traces"
            )
            if "delay_iterations" not in existing_vllm_args:
                envs["EXTRA_VLLM_ARGS"] = f"{existing_vllm_args} {profiler_args}".strip()
        else:
            import json as _json
            try:
                extra_body = _json.loads(str(envs.get("PROFILE_EXTRA_BODY", "{}")))
            except (ValueError, TypeError):
                extra_body = {}
            # Always override start_step/num_steps with computed values —
            # the yaml template has placeholder defaults for CONC=8.
            extra_body["start_step"] = delay_iters
            extra_body["num_steps"] = max_iters
            extra_body.setdefault("shape_discovery", True)
            extra_body.setdefault("roofline_annotations", True)
            envs["PROFILE_EXTRA_BODY"] = _json.dumps(extra_body)
    seq_cost = isl_val + osl_val
    if seq_cost <= 1024:
        factor = 10
    elif seq_cost <= 4096:
        factor = 5
    elif seq_cost <= 16384:
        factor = 3
    else:
        factor = 2
    if "NUM_PROMPTS" not in envs:
        envs["NUM_PROMPTS"] = max(conc_val * factor, conc_val)
    if "NUM_WARMUPS" not in envs:
        envs["NUM_WARMUPS"] = min(conc_val, 8)
    if server_args:
        envs[server_args_env_name(bench.get("framework"))] = server_args
    for key, value in (extra_envs or {}).items():
        envs[str(key)] = str(value)
    materialized = output_dir / "baseline_config.with_envs.yaml"
    with materialized.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return materialized


class BaselineExecutor:
    """Class form for tests / DI; ``baseline_executor`` is the bare callable."""

    def __init__(
        self,
        *,
        magpie_python: str | None = None,
        default_config_path: Path | str | None = None,
        default_output_root: Path | str | None = None,
        default_timeout_sec: int = BASELINE_DEFAULT_TIMEOUT_SEC,
        cwd: Path | str = "/tmp",
    ):
        from ._grid_runner import _resolve_magpie_python, _resolve_output_root
        self.magpie_python = magpie_python or _resolve_magpie_python()
        # None = resolve from $FRAMEWORK at call time. Tests may pass an
        # explicit fixture path which then wins over the env-based resolver.
        self.default_config_path = (
            Path(default_config_path) if default_config_path else None
        )
        self.default_output_root = Path(
            default_output_root or _resolve_output_root()
        )
        self.default_timeout_sec = default_timeout_sec
        self.cwd = Path(cwd)

    def _resolve_default_config(self) -> Path:
        """Hook for subclasses (ProfileExecutor) to swap the resolver."""
        return _default_baseline_config()

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        params = ctx.task.params or {}
        config_path = Path(
            params.get("config_path")
            or self.default_config_path
            or self._resolve_default_config()
        )
        if not config_path.exists():
            raise FileNotFoundError(f"baseline config not found: {config_path}")

        output_dir = Path(
            params.get("output_dir")
            or (self.default_output_root / f"baseline-{ctx.task.task_id[:8]}")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        timeout_sec = int(params.get("timeout_sec") or self.default_timeout_sec)
        # Resolve model path: task.params['model_path'] (Coordinator-supplied) >
        # $MODEL_PATH (CLI re-exported). If neither, leave the YAML's hardcoded
        # `model:` alone so unit tests with explicit fixture paths still work.
        resolved_model = (
            str(params.get("model_path") or "").strip()
            or os.environ.get("MODEL_PATH", "").strip()
        )
        # Same pattern for gpu_type: cli.py canonicalizes (mi325x->mi300x) and
        # re-exports $GPU_TYPE; tests / Coordinator can also override per-task.
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower()
            or os.environ.get("GPU_TYPE", "").strip().lower()
        )
        config_path = _materialize_config_with_envs(
            config_path,
            output_dir,
            extra_sglang_args=str(params.get("extra_sglang_args") or ""),
            extra_server_args=str(
                params.get("extra_server_args")
                or params.get("extra_vllm_args")
                or ""
            ),
            extra_envs=dict(params.get("extra_envs") or {}),
            model_path=resolved_model,
            gpu_type=resolved_gpu,
        )

        cmd = [
            self.magpie_python, "-m", "Magpie", "-v", "benchmark",
            "--benchmark-config", str(config_path),
            "--output-dir", str(output_dir),
            "--run-mode", "local",
        ]
        env = os.environ.copy()
        # Make sure the venv is first in PATH so the benchmark script's
        # `python3` resolves to one with torch+rocm. Magpie YAML also sets
        # this but defending in depth costs nothing.
        env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"

        log.info("baseline_executor: launching Magpie cmd=%s output_dir=%s",
                 cmd, output_dir)

        # subprocess.run is sync — wrap in asyncio.to_thread so we don't
        # block the Coordinator reactor loop.
        try:
            proc = await asyncio.to_thread(
                subprocess.run, cmd,
                capture_output=True, text=True, timeout=timeout_sec,
                env=env, cwd=str(self.cwd),
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "failed",
                "error_class": "timeout",
                "error": f"baseline benchmark exceeded {timeout_sec}s: {exc}",
                "output_dir": str(output_dir),
            }

        if proc.returncode != 0:
            tail_stderr = (proc.stderr or "")[-2000:]
            return {
                "status": "failed",
                "error_class": "subprocess_nonzero",
                "returncode": proc.returncode,
                "error": tail_stderr,
                "output_dir": str(output_dir),
            }

        # Locate the workspace Magpie created (benchmark_<framework>_<ts>/).
        candidates = sorted(output_dir.glob("benchmark_*"))
        if not candidates:
            return {
                "status": "failed",
                "error_class": "no_workspace",
                "error": "Magpie completed but produced no benchmark_* workspace",
                "output_dir": str(output_dir),
            }
        workspace = candidates[-1]
        report_path = workspace / "benchmark_report.json"
        if not report_path.exists():
            return {
                "status": "failed",
                "error_class": "no_report",
                "error": f"benchmark_report.json missing under {workspace}",
                "output_dir": str(output_dir),
                "workspace": str(workspace),
            }

        with report_path.open(encoding="utf-8") as f:
            report = json.load(f)

        tput = report.get("throughput", {}) or {}
        latency = report.get("latency", {}) or {}
        ttft = latency.get("ttft", {}) or {}
        e2el = latency.get("e2el", {}) or {}

        result = {
            "status": "succeeded" if report.get("success") else "failed",
            "framework": report.get("framework"),
            "model": report.get("model"),
            "request_throughput": tput.get("request_throughput"),
            "output_throughput": tput.get("output_throughput"),
            "total_token_throughput": tput.get("total_token_throughput"),
            "completed_requests": tput.get("completed_requests"),
            "duration_seconds": tput.get("duration_seconds"),
            "ttft_mean_ms": ttft.get("mean_ms"),
            "ttft_p99_ms": ttft.get("p99_ms"),
            "e2el_mean_ms": e2el.get("mean_ms"),
            "e2el_p99_ms": e2el.get("p99_ms"),
            "report_path": str(report_path),
            "workspace": str(workspace),
        }

        # Parse accuracy eval results (GSM8K). RUN_EVAL=true was injected
        # into the yaml so Magpie ran lm-eval while the server was still up.
        from ._accuracy_gate import parse_eval_results
        eval_data = parse_eval_results(workspace)
        if eval_data.get("accuracy") is not None:
            result["accuracy"] = eval_data["accuracy"]
            result["accuracy_task"] = eval_data.get("task", "gsm8k")
            result["accuracy_metric"] = eval_data.get("metric", "")
            result["accuracy_source"] = eval_data.get("source_file", "")
            log.info("baseline_executor: accuracy=%.4f (%s)",
                     result["accuracy"], result["accuracy_task"])
        else:
            log.warning("baseline_executor: accuracy eval not found: %s",
                        eval_data.get("error", "unknown"))

        log.info(
            "baseline_executor: success tput=%.1f tok/s/gpu (output) e2el=%.1fms",
            result["output_throughput"] or 0.0,
            result["e2el_mean_ms"] or 0.0,
        )
        return result


# Module-level callable so callers can do ``register_executor("baseline",
# baseline_executor)`` without instantiating.
baseline_executor = BaselineExecutor()


__all__ = [
    "BASELINE_DEFAULT_CONFIG",
    "BASELINE_DEFAULT_TIMEOUT_SEC",
    "BaselineExecutor",
    "baseline_executor",
]
