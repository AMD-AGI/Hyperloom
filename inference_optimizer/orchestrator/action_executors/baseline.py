# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Real ``baseline`` ActionRunner — runs Magpie SGLang benchmark.

 + §16.1 baseline action.

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
import importlib.util
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from ...compat.payload_aliases import read_extra_server_args
from ...paths import asset_root
from ...session_paths import runs_dir
from ..sub_agent_runner import RunnerContext
from ._grid_runner import sanitize_result_dir, sanitize_script_name
from ._subprocess_kill import (
    _process_group_alive,
    _signal_group,
    run_with_session_kill,
)
from ._workload_envs import (
    default_baseline_config,
    materialize_config_with_envs,
)
from .benchmark_result import (
    extract_benchmark_measurement,
    harvest_leaked_artifacts,
)


log = logging.getLogger(__name__)


# Magpie's built-in InferenceX benchmark scripts that honour
# ``MAGPIE_RUN_PHASE=server``/``client`` and therefore support the
# server_lifecycle reuse protocol. Static mirror of Magpie's
# ``benchmarker.MAGPIE_BUILTIN_SCRIPTS`` — duplicated (not imported) so
# Hyperloom keeps no import-time dependency on Magpie internals (Magpie is
# not importable in the unit-test sandbox). Keep in sync with Magpie. The
# cold-start double-run guard only engages when the resolved benchmark
# script is one of these; any other script (model-specific InferenceX
# scripts, exotic GPU types without a generic Magpie script) falls back to
# the legacy single round. ``atom_*`` per AMD-AGI/Magpie#34 (atom-as-a-
# framework requires that PR, which also ships the phase-aware atom
# scripts, so listing them here can never outrun Magpie support).
MAGPIE_BUILTIN_SCRIPTS = frozenset(
    {
        "vllm_mi300x.sh",
        "vllm_mi355x.sh",
        "sglang_mi300x.sh",
        "sglang_mi355x.sh",
        "atom_mi300x.sh",
        "atom_mi355x.sh",
    }
)

# Default HTTP port Magpie's server_lifecycle binds the persistent server
# on when ``benchmark.envs.PORT`` is unset (matches Magpie's
# ``benchmarker._reuse_benchmark_port`` fallback). The double-run guard
# pins it explicitly into the per-round YAML so Magpie's reuse keying and
# our defensive teardown agree on the same port.
BASELINE_REUSE_PORT_DEFAULT = 8888

# Server-boot budget for the persistent server phase (server_lifecycle).
# In lifecycle mode Magpie applies ``timeout_seconds`` to the client phase
# only and gates server startup by this value. Matches Magpie's own
# default; override via ``INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC``.
BASELINE_SERVER_READY_TIMEOUT_SEC = 2700


# Legacy module-level constant kept pointing at the sglang yaml so existing
# tests that import it as a fixture path continue to work. Runtime selection
# of sglang vs vllm yaml goes through `default_baseline_config()` (re-exported
# below as the legacy `_default_baseline_config` alias).
BASELINE_DEFAULT_CONFIG = (
    asset_root() / "scripts" / "configs" / "baseline_sglang.yaml"
)
BASELINE_DEFAULT_TIMEOUT_SEC = 7800           # WARM-start cap, 130 min (raised for Qwen3-32B TP=1 CONC=64 ISL/OSL=1024 NUM_PROMPTS=320 ~82 min workload)
BASELINE_COLD_START_TIMEOUT_SEC = 9000        # COLD-start cap, 150 min (includes ~20 min cuda graph capture)
COLD_START_KERNEL_THRESHOLD = 20              # < N .so files under aiter jit/build/ ⇒ COLD

# Legacy fallback probe order for aiter's JIT cache dir. Used only when
# `importlib.util.find_spec("aiter")` cannot resolve aiter dynamically
# (e.g. probe invoked from a venv where aiter isn't importable). First
# path that exists wins. Override via env
# `INFERENCE_OPTIMIZER_AITER_JIT_DIR=/abs/path` (tried before this list).
AITER_JIT_PROBE_PATHS: tuple[str, ...] = (
    "/sgl-workspace/aiter/aiter/jit",
    "/sgl-workspace/aiter/aiter/jit/build",
    "/usr/local/lib/python3.10/dist-packages/aiter/jit",
    "/usr/local/lib/python3.12/dist-packages/aiter/jit",
    "/usr/local/lib/python3.10/site-packages/aiter/jit",
    "/usr/local/lib/python3.12/site-packages/aiter/jit",
    "/opt/venv/lib/python3.10/site-packages/aiter/jit",
    "/opt/venv/lib/python3.12/site-packages/aiter/jit",
)


# Underscore-prefixed aliases re-exported for callers/tests; the canonical
# names live in `_workload_envs`.
_default_baseline_config = default_baseline_config
_materialize_config_with_envs = materialize_config_with_envs


def _resolve_aiter_jit_dir_dynamic() -> list[str]:
    """Locate aiter's ``jit/`` dir via Python's import machinery.

    Wheel-packaged aiter ships ~80 pre-built ``.so`` directly under
    ``<aiter>/jit/``; only runtime-JIT staging (a handful of patched
    kernels) lives under ``<aiter>/jit/build/<module>/build/``. Counting
    at ``<aiter>/jit/`` therefore correctly reflects a warm install,
    while the legacy fixed list (precise to ``jit/build``) mis-reports
    every wheel install as COLD.

    Returns an ordered candidate list (``jit`` preferred over
    ``jit/build``). Empty if aiter cannot be located.

    Returns:
        list[str]: Ordered candidate jit-dir paths (``jit`` before
            ``jit/build``), or an empty list when aiter is not importable.
    """
    try:
        spec = importlib.util.find_spec("aiter")
    except (ImportError, ValueError):  # noqa: BLE001 — aiter not importable
        return []
    if spec is None or not spec.origin:
        return []
    aiter_root = Path(spec.origin).parent
    return [
        str(aiter_root / "jit"),
        str(aiter_root / "jit" / "build"),
    ]


def _probe_aiter_jit_cache() -> dict[str, Any]:
    """Inspect aiter's ``jit/`` dir to decide cold vs warm start.

    Pure read-only filesystem probe — no subprocess, no GPU touch.
    Resolution order:

      1. ``$INFERENCE_OPTIMIZER_AITER_JIT_DIR`` env override
      2. Dynamic ``<aiter>/jit`` then ``<aiter>/jit/build`` resolved via
         ``importlib.util.find_spec("aiter")``
      3. Legacy ``AITER_JIT_PROBE_PATHS`` fallback

    First existing dir wins; we count ``.so`` files recursively and sum
    their byte sizes. Any IO error degrades to ``probe_status="error"``
    so callers (and unit tests on hosts with no aiter install) fall back
    to the default WARM timeout instead of crashing.

    Returns a dict with keys:
        path           Path that was probed, or None if nothing found.
        kernel_count   Number of `.so` files under `path` (recursive).
        size_mb        Total size of those `.so` files, in MiB (int).
        is_cold        True iff kernel_count < COLD_START_KERNEL_THRESHOLD;
                       None when probe failed.
        probe_status   "found" | "not_found" | "error".

    Returns:
        dict[str, Any]: Probe info with keys ``path``, ``kernel_count``,
            ``size_mb``, ``is_cold`` and ``probe_status``.
    """
    info: dict[str, Any] = {
        "path": None,
        "kernel_count": 0,
        "size_mb": 0,
        "is_cold": None,
        "probe_status": "not_found",
    }
    candidates: list[str] = []
    override = os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "").strip()
    if override:
        candidates.append(override)
    candidates.extend(_resolve_aiter_jit_dir_dynamic())
    candidates.extend(AITER_JIT_PROBE_PATHS)

    try:
        chosen: Path | None = None
        for raw in candidates:
            p = Path(raw)
            if p.exists() and p.is_dir():
                chosen = p
                break
        if chosen is None:
            return info
        info["path"] = str(chosen)

        total_bytes = 0
        kernel_count = 0
        for so_path in chosen.rglob("*.so"):
            try:
                total_bytes += so_path.stat().st_size
                kernel_count += 1
            except OSError:
                continue
        info["kernel_count"] = kernel_count
        info["size_mb"] = total_bytes // (1024 * 1024)
        info["is_cold"] = kernel_count < COLD_START_KERNEL_THRESHOLD
        info["probe_status"] = "found"
        return info
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "baseline_executor: aiter jit cache probe failed: %s", exc,
        )
        info["probe_status"] = "error"
        info["is_cold"] = None
        return info


class BaselineExecutor:
    """Class form for tests / DI; ``baseline_executor`` is the bare callable.

    ``session_dir`` is the canonical session root the executor will derive
    its per-task workspace under (``<sd>/runs/baseline/<task_id>/``).
    When the SubAgentRunner pre-creates that workspace and injects it via
    ``ctx.extra["workspace"]``, the executor uses the injected path
    verbatim — ``session_dir`` is then only the fallback for direct
    instantiation in tests.
    """

    def __init__(
        self,
        *,
        magpie_python: str | None = None,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        default_timeout_sec: int = BASELINE_DEFAULT_TIMEOUT_SEC,
        cwd: Path | str = "/tmp",
    ):
        """Initialize the baseline executor with launch defaults.

        Args:
            magpie_python (str | None): Python interpreter used to invoke
                Magpie; resolved automatically when ``None``.
            default_config_path (Path | str | None): Default Magpie YAML config
                path; resolved from ``$FRAMEWORK`` at call time when ``None``.
            session_dir (Path | str | None): Canonical session root for
                per-task workspaces; resolved automatically when ``None``.
            default_timeout_sec (int): Default (warm-start) subprocess timeout.
            cwd (Path | str): Working directory for the Magpie subprocess.
        """
        from ._grid_runner import _resolve_magpie_python, _resolve_session_dir
        self.magpie_python = magpie_python or _resolve_magpie_python()
        # None = resolve from $FRAMEWORK at call time. Tests may pass an
        # explicit fixture path which then wins over the env-based resolver.
        self.default_config_path = (
            Path(default_config_path) if default_config_path else None
        )
        self.session_dir = Path(session_dir) if session_dir else _resolve_session_dir()
        self.default_timeout_sec = default_timeout_sec
        self.cwd = Path(cwd)

    def _resolve_default_config(self) -> Path:
        """Hook for subclasses (ProfileExecutor) to swap the resolver.

        Returns:
            Path: The default baseline Magpie YAML config path.
        """
        return _default_baseline_config()

    def _resolve_workspace(self, ctx: RunnerContext, action: str) -> Path:
        """Pick the per-task workspace dir for this executor invocation.

        Resolution order (highest priority first):

        1. ``task.params['output_dir']`` — explicit caller wins.
        2. ``ctx.extra['workspace']``    — SubAgentRunner pre-mkdir'd path.
        3. ``runs_dir(self.session_dir, action, ctx.task.task_id)``
           — direct-instantiation fallback (tests / examples that don't
           wire the Coordinator).

        Args:
            ctx (RunnerContext): The runner context (``task.params`` /
                ``extra``) used to resolve the workspace.
            action (str): The action name used in the fallback runs-dir path.

        Returns:
            Path: The resolved per-task workspace directory.
        """
        params = ctx.task.params or {}
        if params.get("output_dir"):
            return Path(params["output_dir"])
        extra = getattr(ctx, "extra", None) or {}
        if extra.get("workspace"):
            return Path(extra["workspace"])
        return runs_dir(self.session_dir, action, ctx.task.task_id)

    def _resolve_timeout(self, params: dict[str, Any]) -> int:
        """Pick the subprocess timeout for this baseline launch.

        Decision order:
        1. ``task.params['timeout_sec']`` — explicit caller wins, no probe.
        2. Probe ``aiter/jit/build/`` — if found AND
           ``kernel_count < COLD_START_KERNEL_THRESHOLD``, return the
           cold-start cap (env-overridable via
           ``INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC``).
        3. Otherwise fall back to ``self.default_timeout_sec`` (warm cap).

        Every path emits exactly one log line so the chosen timeout +
        rationale is greppable in ``optimizer_runs/run_*.log``.

        Args:
            params (dict[str, Any]): The task params, optionally carrying an
                explicit ``timeout_sec``.

        Returns:
            int: The resolved subprocess timeout in seconds.
        """
        explicit = params.get("timeout_sec")
        if explicit:
            timeout_sec = int(explicit)
            log.info(
                "baseline_executor: timeout=%ds (explicit task param)",
                timeout_sec,
            )
            return timeout_sec

        cache = _probe_aiter_jit_cache()
        cold_cap = int(os.environ.get(
            "INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC",
            BASELINE_COLD_START_TIMEOUT_SEC,
        ))
        if cache["probe_status"] == "found" and cache["is_cold"]:
            log.warning(
                "baseline_executor: COLD_START detected — aiter jit/build/ "
                "at %s has %d .so (< %d threshold), %d MB. Bumping timeout "
                "%ds -> %ds. First-time JIT compile on a new "
                "(model, dtype, TP, max_model_len) signature can take 30+ "
                "minutes for large FP8 / MoE models.",
                cache["path"], cache["kernel_count"],
                COLD_START_KERNEL_THRESHOLD, cache["size_mb"],
                self.default_timeout_sec, cold_cap,
            )
            return cold_cap
        if cache["probe_status"] == "found":
            log.info(
                "baseline_executor: WARM start — aiter jit/build/ at %s "
                "has %d .so, %d MB. Using default timeout=%ds.",
                cache["path"], cache["kernel_count"], cache["size_mb"],
                self.default_timeout_sec,
            )
            return self.default_timeout_sec
        log.warning(
            "baseline_executor: aiter jit cache not located "
            "(probe_status=%s). Using default timeout=%ds. Cold-start "
            "auto-bump disabled for this run.",
            cache["probe_status"], self.default_timeout_sec,
        )
        return self.default_timeout_sec

    def _after_materialize_config(
        self, config_path: Path, output_dir: Path,
    ) -> dict[str, Any] | None:
        """Hook for subclasses after YAML materialization, before launch.

        ProfileExecutor uses this to patch/validate the exact InferenceX
        checkout named by the rendered YAML. Baseline/params/backends keep the
        no-op default so their launch path is unchanged.

        Args:
            config_path (Path): The materialized YAML config path.
            output_dir (Path): The per-task workspace directory.

        Returns:
            dict[str, Any] | None: An early-return result dict to short-circuit
                the launch, or ``None`` to proceed with the normal launch path.
        """
        return None

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the Magpie baseline benchmark and parse its result.

        Materializes the workload config, resolves the timeout (with cold-start
        detection), restarts the multi-node server when required, launches
        Magpie via ``run_with_session_kill``, harvests leaked artifacts, parses
        ``benchmark_report.json`` and the accuracy eval, and returns a result
        dict the Coordinator promotes into SharedState.

        Args:
            ctx (RunnerContext): The runner context carrying ``task.params``
                (config / model / timeout knobs) and ``extra`` (workspace).

        Returns:
            dict[str, Any]: On success, a ``status="succeeded"`` dict with
                throughput / latency / accuracy measurements and artifact
                paths; on failure, a ``status="failed"`` dict with an
                ``error_class`` (``timeout``, ``subprocess_nonzero``,
                ``no_workspace``, ``no_report``, ``invalid_measurement`` ...).

        Raises:
            FileNotFoundError: If the resolved baseline config does not exist.
        """
        params = ctx.task.params or {}
        config_path = Path(
            params.get("config_path")
            or self.default_config_path
            or self._resolve_default_config()
        )
        if not config_path.exists():
            raise FileNotFoundError(f"baseline config not found: {config_path}")

        output_dir = self._resolve_workspace(ctx, "baseline")
        output_dir.mkdir(parents=True, exist_ok=True)

        timeout_sec = self._resolve_timeout(params)
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
        # Orchestration-supplied script + result_dir overrides. Both are
        # surfaced as ``task.params`` so the LLM can route around scripts
        # that hardcode ``--result-dir /workspace/`` (see SKILL.md
        # "Magpie leak-path salvage"). Sanitization at the executor
        # boundary turns any malformed override into ``error_class=
        # bad_param``; the Coordinator promotes that to a
        # ``policy_denied`` observation rather than an unsafe subprocess.
        try:
            override_script = sanitize_script_name(params.get("benchmark_script"))
            override_result_dir = sanitize_result_dir(params.get("result_dir"))
        except ValueError as exc:
            return {
                "status": "failed",
                "error_class": "bad_param",
                "error": str(exc),
                "output_dir": str(output_dir),
            }
        config_path = materialize_config_with_envs(
            config_path,
            output_dir,
            extra_server_args=read_extra_server_args(params),
            extra_envs=dict(params.get("extra_envs") or {}),
            model_path=resolved_model,
            gpu_type=resolved_gpu,
            benchmark_script=override_script,
        )
        # Stash for the result so Coordinator can plumb it forward to
        # downstream params/backends/sweep tasks (workload-contract reuse).
        materialized_config_path = config_path
        hook_result = self._after_materialize_config(config_path, output_dir)
        if hook_result is not None:
            hook_result.setdefault("materialized_config", str(config_path))
            hook_result.setdefault("output_dir", str(output_dir))
            return hook_result

        # Cold-start "warmup artifact" guard. The baseline action is the
        # FIRST step of the optimization flow, so the server is always
        # freshly booted for the model under test. The first benchmark
        # window therefore pays one-time cold-start costs (kernel JIT /
        # torch.compile first-compile, CUDA/HIP-graph first-capture,
        # KV-cache cold allocation, GPU not yet at boost clocks); the
        # client-side ``--num-warmups`` (hardcoded ``2 * CONC`` in
        # InferenceX's bench scripts) is far too small to absorb those.
        # Taking that contaminated number as the baseline inflates every
        # later gain into a fictitious 1600%+ "improvement" (see
        # hyperloom_models_jun1.csv rows tagged ``warmup``).
        #
        # Fix: run the benchmark TWICE against the SAME persistent server
        # via Magpie's ``server_lifecycle`` reuse protocol. Round 1 boots
        # the server and runs a full client load (paying every one-time
        # cold cost); round 2 re-attaches to the now-hot server (client
        # only, no restart) and its throughput is the clean baseline. This
        # eliminates ALL cold-start contamination — not just on-disk JIT
        # caches but also CUDA-graph capture, allocator and clock warmup —
        # because round 2 never restarts the server.
        #
        # Eligibility (else fall back to the legacy single round):
        #   * env ``INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN`` not disabled,
        #   * single-node (server_lifecycle is local-only; multi-node uses
        #     a long-lived RayJob server with no per-benchmark cold start),
        #   * the resolved benchmark script is a Magpie built-in that
        #     honours ``MAGPIE_RUN_PHASE`` (server_lifecycle requirement),
        #   * the profiler is off (server_lifecycle + persistent server is
        #     incompatible with torch_profiler unless cleanup=true).
        lifecycle = self._resolve_lifecycle_params(materialized_config_path)
        double_run = self._double_run_enabled() and lifecycle["eligible"]

        common = {
            "timeout_sec": timeout_sec,
            "override_result_dir": override_result_dir,
            "resolved_model": resolved_model,
            "materialized_config_path": materialized_config_path,
            "params": params,
            "ctx": ctx,
        }

        if not double_run:
            if self._double_run_enabled() and not lifecycle["eligible"]:
                log.info(
                    "baseline_executor: cold-start double-run not eligible "
                    "(%s); running single round.", lifecycle["reason"],
                )
            return await self._run_single_benchmark(
                config_path=config_path, output_dir=output_dir, **common,
            )

        framework = lifecycle["framework"]
        port = lifecycle["port"]
        # pid_dir is SHARED across both rounds (Magpie keys the persistent
        # server by ``<pid_dir>/<framework>_<port>.{pid,json}``) so round 2
        # discovers the server round 1 left running. Use the task root so
        # it is isolated per baseline task and torn down with it.
        pid_dir = output_dir
        try:
            # Round 1 (warmup): boot + run server, leave it running
            # (cleanup=false) so round 2 can re-attach. Throughput is
            # discarded — it carries the cold-start contamination.
            warmup_dir = output_dir / "warmup_round"
            warmup_cfg = self._write_lifecycle_config(
                materialized_config_path, warmup_dir,
                cleanup=False, pid_dir=pid_dir, port=port,
            )
            log.info(
                "baseline_executor: cold-start guard — warmup round "
                "(discarded, boots persistent server) in %s", warmup_dir,
            )
            warmup_result = await self._run_single_benchmark(
                config_path=warmup_cfg, output_dir=warmup_dir, **common,
            )
            if warmup_result.get("status") != "succeeded":
                # Hard failure in the warmup round (server never came up,
                # timeout, etc.) almost certainly recurs, so skip the
                # measured round. The finally block tears down any server
                # the warmup round may have left half-booted.
                warmup_result.setdefault("nonfatal_warnings", [])
                warmup_result["nonfatal_warnings"].append(
                    "baseline_warmup_round_failed",
                )
                log.warning(
                    "baseline_executor: warmup round failed (error_class=%s)"
                    "; skipping measured round",
                    warmup_result.get("error_class"),
                )
                return warmup_result
            warmup_tput = warmup_result.get("output_throughput")
            warmup_runtime = warmup_result.get("subprocess_runtime_sec")

            # Round 2 (measured): re-attach to the hot server (client
            # only). cleanup=true tears the server down after this round
            # on the happy path; the finally block is the safety net.
            measure_dir = output_dir / "measure_round"
            measure_cfg = self._write_lifecycle_config(
                materialized_config_path, measure_dir,
                cleanup=True, pid_dir=pid_dir, port=port,
            )
            log.info(
                "baseline_executor: cold-start guard — measured baseline "
                "round in %s (warmup tput=%.1f tok/s discarded, reusing "
                "hot server)", measure_dir, warmup_tput or 0.0,
            )
            result = await self._run_single_benchmark(
                config_path=measure_cfg, output_dir=measure_dir, **common,
            )
            if result.get("status") == "succeeded":
                result.setdefault("nonfatal_warnings", [])
                result["nonfatal_warnings"].append(
                    "baseline_double_run_discarded_first",
                )
                result["warmup_round_tput"] = warmup_tput
                # Overtime-kill anchor fix: the Coordinator promotes
                # ``subprocess_runtime_sec`` into
                # ``SharedState.baseline_runtime_sec``, which the
                # ExploreExecutor turns into the per-variant soft-kill
                # deadline (``baseline_runtime_sec * kill_ratio``). Explore
                # variants each RESTART the server (full server-boot +
                # client), but round 2 here reused the hot server
                # (client-only), so its wall-clock is far too small an
                # anchor and would soft-kill normal variants as
                # KILLED_OVERTIME. Report round 1's FULL server-boot +
                # client wall-clock as the anchor instead (it matches the
                # explore variants' run profile); keep round 2's client-only
                # time under a separate key for transparency / debugging.
                if isinstance(warmup_runtime, (int, float)) and warmup_runtime > 0:
                    result["measure_round_runtime_sec"] = result.get(
                        "subprocess_runtime_sec",
                    )
                    result["subprocess_runtime_sec"] = round(
                        float(warmup_runtime), 2,
                    )
                _hot = result.get("output_throughput") or 0.0
                _cold = warmup_tput or 0.0
                log.info(
                    "baseline_executor: cold-start guard — measured "
                    "baseline=%.1f tok/s (warmup=%.1f tok/s discarded; "
                    "artifact would have been +%.0f%%)",
                    _hot, _cold,
                    ((_hot / _cold - 1.0) * 100.0) if _cold > 0 else 0.0,
                )
            return result
        finally:
            # Defensive teardown: guarantees no persistent server leaks
            # regardless of which round failed. Idempotent — on the happy
            # path round 2's cleanup=true already removed the pid/meta
            # files, so this is a no-op.
            self._teardown_lifecycle_server(
                pid_dir=pid_dir, framework=framework, port=port,
            )

    @staticmethod
    def _double_run_enabled() -> bool:
        return os.environ.get(
            "INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", "1",
        ).strip().lower() not in ("0", "false", "no", "")

    def _resolve_lifecycle_params(
        self, materialized_config_path: Path,
    ) -> dict[str, Any]:
        """Inspect the materialized YAML to decide server_lifecycle
        eligibility for the cold-start double-run.

        Returns a dict with ``eligible`` (bool), ``framework`` (str),
        ``port`` (int) and ``reason`` (str, populated when ineligible).
        """
        info: dict[str, Any] = {
            "eligible": False,
            "framework": "",
            "port": BASELINE_REUSE_PORT_DEFAULT,
            "reason": "",
        }
        try:
            with Path(materialized_config_path).open(encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as exc:
            info["reason"] = f"could not read materialized config: {exc}"
            return info
        bench = cfg.get("benchmark") or {}
        info["framework"] = str(bench.get("framework") or "").lower()
        envs = bench.get("envs") or {}
        try:
            info["port"] = int(envs.get("PORT", BASELINE_REUSE_PORT_DEFAULT))
        except (TypeError, ValueError):
            info["port"] = BASELINE_REUSE_PORT_DEFAULT

        from ._multi_node_env import is_multi_node
        if is_multi_node():
            info["reason"] = "multi-node (server_lifecycle is local-only)"
            return info

        script_name = Path(str(bench.get("benchmark_script") or "")).name
        if script_name not in MAGPIE_BUILTIN_SCRIPTS:
            info["reason"] = (
                f"benchmark_script={script_name!r} is not a Magpie built-in "
                f"({sorted(MAGPIE_BUILTIN_SCRIPTS)})"
            )
            return info

        profiler_on = bool(
            (bench.get("profiler") or {})
            .get("torch_profiler", {})
            .get("enabled")
        )
        if profiler_on:
            info["reason"] = "torch_profiler enabled (incompatible with reuse)"
            return info

        info["eligible"] = True
        return info

    def _write_lifecycle_config(
        self,
        base_config_path: Path,
        dest_dir: Path,
        *,
        cleanup: bool,
        pid_dir: Path,
        port: int,
    ) -> Path:
        """Render a per-round YAML that injects ``benchmark.server_lifecycle``
        on top of the materialized baseline config.

        Both rounds share ``pid_dir`` + ``port`` so round 2 re-attaches to
        the server round 1 left running. Only ``cleanup`` differs (round 1
        persists the server, round 2 tears it down).
        """
        with Path(base_config_path).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        bench = cfg.setdefault("benchmark", {})
        ready_timeout = int(os.environ.get(
            "INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC",
            BASELINE_SERVER_READY_TIMEOUT_SEC,
        ))
        bench["server_lifecycle"] = {
            "enabled": True,
            "cleanup": bool(cleanup),
            "force_reuse": False,
            "pid_dir": str(pid_dir),
            "server_ready_timeout_s": ready_timeout,
        }
        # Pin PORT so Magpie's reuse keying and our teardown agree.
        envs = bench.setdefault("envs", {})
        envs["PORT"] = int(port)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = Path(dest_dir) / "baseline_lifecycle.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return out

    def _teardown_lifecycle_server(
        self, *, pid_dir: Path, framework: str, port: int,
    ) -> None:
        """Best-effort teardown of a persistent server left by the
        double-run rounds. Idempotent and never raises (safe in finally).

        On the happy path round 2's ``cleanup=true`` already removed the
        pid/meta files and killed the server, so this is a no-op. It only
        does real work on the abnormal paths (warmup-round failure, an
        exception between rounds, or a round 2 timeout that skipped
        Magpie's own cleanup).
        """
        base = Path(pid_dir)
        tag = f"{framework}_{port}"
        pid_file = base / f"{tag}.pid"
        meta_file = base / f"{tag}.json"
        server_pid: int | None = None
        server_pgid: int | None = None
        try:
            if pid_file.exists():
                parts = pid_file.read_text(encoding="utf-8").split()
                if parts:
                    server_pid = int(parts[0])
                if len(parts) > 1:
                    server_pgid = int(parts[1])
        except (OSError, ValueError):
            # Best-effort: proceed with whatever (if anything) was parsed
            # before the error. Teardown is defensive and must never raise.
            pass

        if server_pid is not None and os.name == "posix":
            # The persistent server is setsid'd into its own session, so
            # its pgid equals its pid unless the pid file recorded one.
            pgid = server_pgid if server_pgid is not None else server_pid
            _signal_group(pgid, signal.SIGTERM)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not _process_group_alive(pgid):
                    break
                time.sleep(0.1)
            if _process_group_alive(pgid):
                _signal_group(pgid, signal.SIGKILL)
            log.info(
                "baseline_executor: lifecycle teardown — reaped persistent "
                "server pgid=%d (%s:%d)", pgid, framework, port,
            )
        for p in (pid_file, meta_file):
            try:
                p.unlink()
            except OSError:
                # File already gone (round 2 cleanup=true removed it) or
                # unremovable; either way teardown must not raise.
                pass

    async def _run_single_benchmark(
        self,
        *,
        config_path: Path,
        output_dir: Path,
        timeout_sec: int,
        override_result_dir: str | None,
        resolved_model: str,
        materialized_config_path: Path,
        params: dict[str, Any],
        ctx: RunnerContext,
    ) -> dict[str, Any]:
        """Run one Magpie benchmark subprocess and parse its result.

        This is the single-round execution core extracted from
        ``__call__`` so the cold-start guard can invoke it twice (warmup
        round + measured round). ``output_dir`` is the per-round slot the
        Magpie workspace and harvested artifacts land under.
        """
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
        # #210 (Deval, comment 8): pin Magpie's InferenceX-resolution
        # to ``$INFERENCEX_PATH`` so Magpie loads the SAME InferenceX
        # checkout that Hyperloom's ``_inferencex_patcher`` has
        # patched. ``MAGPIE_INFERENCEX_PATH`` is the highest-precedence
        # resolution rung in Magpie's
        # ``_resolve_default_inferencex_dir`` (``Magpie/modes/
        # benchmark/inferencex.py:43``); without setting it, Magpie
        # falls through to ``./InferenceX`` next to its repo or the
        # ``$XDG_CACHE_HOME/magpie/InferenceX`` cache, either of which
        # may be a separate, unpatched checkout (the symptom reported
        # in #210 comments 4 + 6).
        inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
        if inferencex_path:
            env["MAGPIE_INFERENCEX_PATH"] = inferencex_path
        # Always-on ``$RESULT_DIR`` default: covers Magpie scripts that
        # respect the env var (and would otherwise fall back to a
        # hardcoded path under ``/workspace/``). Scripts that ignore
        # ``$RESULT_DIR`` still leak — the
        # ``extract_benchmark_measurement`` salvage pass picks those
        # up. Operators / Orchestration can override the destination
        # via ``task.params['result_dir']``.
        env["RESULT_DIR"] = override_result_dir or str(output_dir)
        # Pin SERVER_LOG / GPU_METRICS_CSV per-task so Magpie's
        # ``single_node/*.sh`` wrappers write the server log and per-second
        # GPU telemetry into the task workspace alongside
        # ``benchmark_report.json`` instead of leaking to
        # ``/workspace/server.log`` / ``/workspace/gpu_metrics.csv``.
        # ``harvest_leaked_artifacts`` still runs below as defense-in-depth
        # for any wrapper that hardcodes the destination ignoring the env.
        env["SERVER_LOG"] = str(output_dir / "server.log")
        env["GPU_METRICS_CSV"] = str(output_dir / "gpu_metrics.csv")

        # Multi-node mode (--nodes >= 2): inject MAGPIE_RUN_PHASE=client
        # + BENCHMARK_BASE_URL=<head pod ClusterIP> so Magpie skips its
        # own server launch and benchmark_serving targets the RayJob
        # head. ``magpie_remote_env()`` returns {} in single-node, so
        # ``env`` is unchanged on the default path.
        from ._multi_node_env import magpie_remote_env
        env.update(magpie_remote_env())

        # Multi-node only: restart sglang/vllm with this round's flags
        # so every benchmark runs against a fresh server (parity with
        # single-node Magpie's PHASE=all server lifecycle). No-op in
        # single-node. Profile rounds set ctx.extra["mn_round_restarted"]
        # before super().__call__() to claim the restart; honour that
        # flag so each Magpie spawn corresponds to exactly one server
        # boot.
        from ._multi_node_server_lifecycle import (
            ServerRestartFailed,
            restart_server_for_round,
        )
        ctx_extra = getattr(ctx, "extra", None) or {}
        if not ctx_extra.get("mn_round_restarted"):
            try:
                # PD knobs (pd_mode / pd_prefill_nodes / pd_decode_nodes
                # / pd_prefill_tp / pd_decode_tp / pd_transfer_backend /
                # pd_ib_device) are resolved by the helper from $PD_* env
                # (set by cli.py) and fall back to state.json — keeping
                # this call site identical between colocated and
                # disaggregated runs (the agent only changes CLI flags).
                await restart_server_for_round(
                    extra_server_args=read_extra_server_args(params),
                    framework=os.environ.get("FRAMEWORK") or None,
                    model_path=resolved_model or None,
                    tp=int(os.environ.get("TP") or 0) or None,
                    ep=int(os.environ.get("EP") or 0) or None,
                )
            except ServerRestartFailed as exc:
                return {
                    "status": "failed",
                    "error_class": "mn_server_restart_failed",
                    "error": str(exc),
                    "output_dir": str(output_dir),
                }

        from ._multi_node_env import log_mn_banner
        log_mn_banner("baseline_executor", log, output_dir=str(output_dir))
        log.info("baseline_executor: launching Magpie cmd=%s output_dir=%s",
                 cmd, output_dir)

        # Magpie is launched via ``run_with_session_kill`` (a
        # ``subprocess.run``-compatible wrapper that ALSO tears down
        # the entire descendant tree on every exit path — success,
        # nonzero, timeout, exception). Plain ``subprocess.run`` leaks
        # vLLM / SGLang server processes for any wrapper that
        # ``nohup`` / ``setsid`` / daemonizes the server (bugs.md §B);
        # those leaks were the root cause of the bash-source race in
        # bugs.md §C #1, where a leaked bash re-sources a benchmark
        # script while the next Magpie subprocess is mid-
        # ``shutil.copy2``. See ``_subprocess_kill.py``.
        subprocess_started_unix = time.time()
        try:
            proc = await asyncio.to_thread(
                run_with_session_kill, cmd,
                env=env, cwd=str(self.cwd), timeout=timeout_sec,
            )
            subprocess_runtime_sec = max(
                0.0, time.time() - subprocess_started_unix,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_candidates = sorted(output_dir.glob("benchmark_*"))
            timeout_destination = (
                timeout_candidates[-1] if timeout_candidates else output_dir
            )
            timeout_harvested = harvest_leaked_artifacts(
                timeout_destination,
                subprocess_started_unix=subprocess_started_unix,
            )
            return {
                "status": "failed",
                "error_class": "timeout",
                "error": f"baseline benchmark exceeded {timeout_sec}s: {exc}",
                "output_dir": str(output_dir),
                "harvested_artifacts": [str(dst) for _, dst in timeout_harvested],
                "nonfatal_warnings": [
                    f"harvested_leaked_artifact:{src}"
                    for src, _ in timeout_harvested
                ],
            }
        proc_returncode = proc.returncode
        proc_stdout = proc.stdout
        proc_stderr = proc.stderr

        # Locate the workspace Magpie created (benchmark_<framework>_<ts>/).
        candidates = sorted(output_dir.glob("benchmark_*"))
        # Always-on artifact harvest. Magpie's shell wrappers hardcode
        # ``/workspace/server.log`` + ``/workspace/gpu_metrics.csv`` +
        # ``/workspace/profile_*.trace.json.gz`` + ``/workspace/
        # inferencex_result*.json`` (see ``harvest_leaked_artifacts``
        # for the full list). Without this pass the NFS clone of
        # ``<session>/runs/baseline/<task_id>/`` is missing the
        # wrapper-side artifacts even on a fully successful run.
        # Runs unconditionally (success / failure / no_workspace) so
        # diagnostics for the failure paths survive too. Mtime gating
        # rejects stale leaks from prior runs. Destination prefers the
        # benchmark workspace dir; falls back to the task output_dir
        # when Magpie never created one.
        harvest_destination = candidates[-1] if candidates else output_dir
        harvested = harvest_leaked_artifacts(
            harvest_destination,
            subprocess_started_unix=subprocess_started_unix,
        )
        if harvested:
            log.info(
                "baseline_executor: harvested %d leaked artifact(s) "
                "into workspace: %s",
                len(harvested),
                ", ".join(str(src.name) for src, _ in harvested),
            )
        if not candidates:
            failure_extras = {
                "output_dir": str(output_dir),
                "harvested_artifacts": [str(dst) for _, dst in harvested],
            }
            # Magpie never created a benchmark_* workspace, so the wrapper
            # never wrote server.log. Persist the captured stderr/stdout to
            # a file so the failure survives the NFS clone and S3 archive
            # (without this, no_workspace failures leave zero on-disk logs).
            captured = (proc_stderr or "") + (proc_stdout or "")
            stderr_log_path: str | None = None
            if captured.strip():
                try:
                    log_file = output_dir / "baseline_stderr.log"
                    log_file.write_text(captured, encoding="utf-8")
                    stderr_log_path = str(log_file)
                except OSError as exc:
                    log.warning(
                        "baseline_executor: failed to persist stderr log: %s",
                        exc,
                    )
            if stderr_log_path:
                failure_extras["stderr_log_path"] = stderr_log_path
            if proc_returncode != 0:
                tail = (proc_stderr or proc_stdout or "")[-2000:]
                return {
                    "status": "failed",
                    "error_class": "subprocess_nonzero",
                    "returncode": proc_returncode,
                    "error": tail,
                    **failure_extras,
                }
            return {
                "status": "failed",
                "error_class": "no_workspace",
                "error": "Magpie completed but produced no benchmark_* workspace",
                **failure_extras,
            }
        workspace = candidates[-1]
        report_path = workspace / "benchmark_report.json"
        report: dict[str, Any] | None = None
        if report_path.exists():
            try:
                with report_path.open(encoding="utf-8") as f:
                    loaded = json.load(f)
                report = loaded if isinstance(loaded, dict) else None
            except (OSError, json.JSONDecodeError):
                report = None

        measurement = extract_benchmark_measurement(
            report,
            workspace=workspace,
            subprocess_started_unix=subprocess_started_unix,
        )
        warnings = list(measurement.pop("nonfatal_warnings", []) or [])
        if proc_returncode != 0:
            warnings.append("magpie_nonzero_after_valid_measurement")
        for leak_src, _ in harvested:
            warnings.append(f"harvested_leaked_artifact:{leak_src}")

        if not measurement.get("valid_measurement"):
            if proc_returncode != 0:
                tail = (proc_stderr or proc_stdout or "")[-2000:]
                error_class = "subprocess_nonzero"
                error = tail
            elif not report_path.exists():
                error_class = "no_report"
                error = f"benchmark_report.json missing under {workspace}"
            else:
                error_class = "invalid_measurement"
                error = "benchmark report did not contain positive throughput and completed requests"
            return {
                "status": "failed",
                "error_class": error_class,
                "returncode": proc_returncode,
                "error": error,
                "output_dir": str(output_dir),
                "workspace": str(workspace),
                "report_path": str(report_path) if report_path.exists() else None,
                "reported_success": measurement.get("reported_success"),
                "nonfatal_warnings": warnings,
            }

        result = {
            "status": "succeeded",
            **measurement,
            "nonfatal_warnings": warnings,
            "returncode": proc_returncode,
            "report_path": str(report_path) if report_path.exists() else None,
            "workspace": str(workspace),
            # Path to the materialized YAML used for THIS baseline. Coordinator
            # promotes this into SharedState.baseline_config_path so subsequent
            # params/backends/sweep tasks reuse it as their `config_path` —
            # without this, `_grid_runner._build_variant_yaml` would render
            # variants from the shipped YAML's smoke defaults (CONC=8/ISL=256/
            # OSL=256/TP=1) and produce ~10x lower throughput than baseline.
            # See `_workload_envs.py` for the bug history.
            "materialized_config": str(materialized_config_path),
            # Wall-clock of the Magpie subprocess (success path only).
            # Coordinator promotes this into
            # ``SharedState.baseline_runtime_sec`` so the explore
            # overtime-kill gate (``--explore-overtime-kill-ratio``)
            # can derive a per-variant deadline of
            # ``baseline_runtime_sec * ratio``. Measured around the
            # ``run_with_session_kill`` call only; not exposed on the
            # failure paths (timeout / nonzero / no_workspace /
            # no_report / invalid_measurement) so a botched baseline
            # cannot accidentally seed a tiny / huge deadline.
            "subprocess_runtime_sec": round(subprocess_runtime_sec, 2),
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
            "baseline_executor: %s tput=%.1f tok/s/gpu (output) e2el=%.1fms",
            "success_with_warning" if warnings else "success",
            result["output_throughput"] or 0.0,
            result["e2el_mean_ms"] or 0.0,
        )
        return result


# Module-level callable so callers can do ``register_executor("baseline",
# baseline_executor)`` without instantiating.
baseline_executor = BaselineExecutor()


__all__ = [
    "AITER_JIT_PROBE_PATHS",
    "BASELINE_COLD_START_TIMEOUT_SEC",
    "BASELINE_DEFAULT_CONFIG",
    "BASELINE_DEFAULT_TIMEOUT_SEC",
    "BaselineExecutor",
    "COLD_START_KERNEL_THRESHOLD",
    "baseline_executor",
]
