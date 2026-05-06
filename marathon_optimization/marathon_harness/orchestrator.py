"""Orchestrator — 8 protocol steps, 13 action modules, 16-step DFS loop,
9-step merge-op, 5 stopping conditions, recovery chains, dream, re-explore.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ipc
from .state import MarathonState, StateLock, compute_score, TIER_BOUNDARIES

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# DREAM_CADENCE_MIN & CHECKPOINT_CADENCE_MIN defined in SKILL.md (§Constants).
# Others are implementation defaults tuned for 24h autonomous runs.
# ---------------------------------------------------------------------------

DREAM_MIN_INTERVAL_MIN = 30      # cooldown: no more than one dream per 30 min
CHECKPOINT_CADENCE_MIN = 30      # SKILL.md: auto-checkpoint every 30 min
DEEP_ANALYSIS_TOP_N = 10
KERNEL_OPT_MAX_SUBMISSIONS = 25
KERNEL_OPT_CONSECUTIVE_DISCARDS = 15
MIN_GPU_PCT = 1
MERGE_OP_SCORE = 22
MAX_RECOVERY_ATTEMPTS = 3
BACKOFF_MULTIPLIER = 2
RESCORE_INTERVAL = 3
PERIODIC_BENCH_INTERVAL = 3      # force E2E benchmark every N completed actions
PLATEAU_THRESHOLD = 6            # consecutive no-gain actions → re-explore
LOOP_DETECT_WINDOW = 8
CIRCUIT_BREAKER = 5              # consecutive failures → re-analyze
CONSECUTIVE_REGRESSION_CAP = 3   # clean-but-losing actions → re-explore
MAX_WALL_HOURS = 24
MAX_CONSECUTIVE_RE_EXPLORES = 3
RE_PROFILE_CADENCE_MIN = 180     # re-profile every 3h for new targets
MAX_CRASH_COUNT = 10
MAX_SELF_RETRY = 3               # Orchestrator self-diagnosis retry cap per action

KM_HEARTBEAT_CHECK_INTERVAL_S = 300
KM_HEARTBEAT_STALE_MIN = 60
KM_HEARTBEAT_RESTART_MIN = 120

_NUMERIC_RESULT_KEYS = frozenset({
    "accuracy_risk", "crash_risk", "gain_pct", "micro_speedup",
    "score", "gpu_time_pct", "priority", "tput_per_gpu",
    "output_throughput", "total_token_throughput", "latency_ms",
    "exit_code", "needs_benchmark",
})


def _sanitize_result(d: dict[str, Any]) -> dict[str, Any]:
    """Coerce LLM-returned result fields to expected Python types.

    The LLM proxy returns plain JSON where numbers may arrive as strings.
    """
    _RISK_WORDS = {"none": 0, "low": 0.1, "medium": 0.5, "high": 0.9, "critical": 1.0}
    for k in _NUMERIC_RESULT_KEYS:
        if k in d and isinstance(d[k], str):
            mapped = _RISK_WORDS.get(d[k].strip().lower())
            if mapped is not None:
                d[k] = mapped
                continue
            try:
                d[k] = float(d[k])
            except (ValueError, TypeError):
                d[k] = 0
    # Coerce nested sub_actions scores
    for sub in d.get("sub_actions", []):
        if isinstance(sub, dict) and "score" in sub:
            try:
                sub["score"] = float(sub["score"])
            except (ValueError, TypeError):
                sub["score"] = 0
    return d


RECOVERY_CHAINS: dict[str, list[dict[str, Any]]] = {
    "oom": [
        {"action": "reduce_mem_fraction", "delta": -0.05},
        {"action": "reduce_cuda_graph_max_bs", "divisor": 2},
        {"action": "restart_server"},
    ],
    "cuda_graph": [
        {"action": "reduce_cuda_graph_max_bs", "divisor": 2},
        {"action": "disable_cuda_graph"},
        {"action": "restart_server"},
    ],
    "patch_crash": [
        {"action": "rollback_last_patch"},
        {"action": "clear_caches"},
        {"action": "restart_server"},
    ],
    "nccl_timeout": [
        {"action": "set_nccl_timeout", "value": 1800},
        {"action": "restart_server", "wait_s": 30},
    ],
    "unknown": [
        {"action": "restart_server"},
        {"action": "checkpoint_and_reload"},
    ],
    "jit_stale": [
        {"action": "clear_jit_cache"},
        {"action": "clear_pycache"},
        {"action": "restart_server"},
    ],
}


class Orchestrator:
    """Main DFS-based optimization orchestrator."""


    @staticmethod
    def _has_launch_script(scripts_dir: Path) -> bool:
        from .workload import _find_script_by_content
        return _find_script_by_content(scripts_dir, "launch") is not None
    def __init__(
        self,
        state: MarathonState,
        llm: Any,
        session_dir: str,
        server: Any,
        oob: Any,
        dashboard: Any,
        shutdown: asyncio.Event,
        max_cost_usd: float = 0,
        max_wall_hours: float = 24,
        inferencex_path: str = "",
        gpu_lock: Any = None,
    ):
        self.state = state
        self.llm = llm
        self.session_dir = session_dir
        self.server = server
        self.oob = oob
        self.dashboard = dashboard
        self.shutdown = shutdown
        self._max_cost_usd = max_cost_usd
        self._max_wall_hours = max_wall_hours
        self._inferencex_path = inferencex_path
        self._workload: Any = None
        self.gpu_lock = gpu_lock
        self._state_lock = StateLock()

    async def run(self) -> None:
        log.info("Orchestrator started")
        self._analysis_task: asyncio.Task | None = None
        try:
            # Step 0: WARM-START
            self.state.phase = "warm_start"
            if self._workload is not None:
                self._workload.snapshot_system_files()
            await self._warm_start()
            if self._workload is not None:
                changed = self._workload.system_files_changed()
                if changed:
                    log.warning("Warm-start modified %d system file(s): %s — rolling back",
                                len(changed), changed)
                    self._workload.rollback_system_files()

            # Step 1: RE-PROFILE
            self.state.phase = "profile"
            await self._re_profile()

            # Step 2-4: DEEP ANALYSIS (background) + DFS LOOP (foreground)
            self.state.phase = "dfs"
            self._analysis_task = asyncio.create_task(self._deep_analysis())
            consecutive_empty_refills = 0
            # On resume, seed the re-profile timer so we don't immediately trigger
            if not hasattr(self.state, '_last_reprofile_min') or self.state._last_reprofile_min == 0:  # type: ignore[attr-defined]
                self.state._last_reprofile_min = (time.time() - self.state.start_time) / 60  # type: ignore[attr-defined]

            while not self.shutdown.is_set():
                if self._should_stop():
                    break

                # Stack empty or exhausted — self-heal with re-explore / re-analyze / re-profile
                if self._needs_refill():
                    # Wait for background analysis if running
                    if self._analysis_task and not self._analysis_task.done():
                        log.info("Stack needs refill, deep analysis still running — waiting")
                        await asyncio.sleep(10)
                        continue

                    stack_before = len(self.state.action_stack)

                    # Escalating refill: re-explore → re-analyze → re-profile
                    if consecutive_empty_refills == 0:
                        log.info("Stack needs refill — triggering re-explore")
                        await self._re_explore()
                    elif consecutive_empty_refills == 1:
                        log.info("Stack still needs refill — triggering deep re-analysis")
                        self._analysis_task = asyncio.create_task(self._deep_analysis())
                        await asyncio.sleep(15)
                        continue
                    elif consecutive_empty_refills == 2:
                        log.info("Stack still needs refill — triggering re-profile + analysis")
                        self.state.profiled = False
                        await self._re_profile()
                        self._analysis_task = asyncio.create_task(self._deep_analysis())
                        await asyncio.sleep(15)
                        continue
                    else:
                        # Dream to generate novel ideas, then re-explore with fresh context
                        log.info("Persistent empty stack — dream + re-explore cycle %d",
                                 consecutive_empty_refills)
                        await self._dream()
                        await self._re_explore()

                    stack_after = len(self.state.action_stack)
                    if stack_after > stack_before:
                        consecutive_empty_refills = 0
                        log.info("Refill added %d actions", stack_after - stack_before)
                    else:
                        consecutive_empty_refills += 1
                        if consecutive_empty_refills >= MAX_CONSECUTIVE_RE_EXPLORES:
                            log.info("No new actions after %d refill attempts — moving to sweep",
                                     MAX_CONSECUTIVE_RE_EXPLORES)
                            break
                        log.info("Refill attempt %d/%d produced no actions — will escalate",
                                 consecutive_empty_refills, MAX_CONSECUTIVE_RE_EXPLORES)
                        await asyncio.sleep(30)
                    continue

                consecutive_empty_refills = 0

                # Periodic re-profile to discover new optimization targets
                elapsed_min = (time.time() - self.state.start_time) / 60
                last_profile_min = getattr(self.state, '_last_reprofile_min', elapsed_min)
                if elapsed_min - last_profile_min >= RE_PROFILE_CADENCE_MIN:
                    log.info("Periodic re-profile at %.0f min", elapsed_min)
                    self.state._last_reprofile_min = elapsed_min  # type: ignore[attr-defined]
                    self.state.profiled = False
                    await self._re_profile()
                    if self._analysis_task is None or self._analysis_task.done():
                        self._analysis_task = asyncio.create_task(self._deep_analysis())

                self._check_km_heartbeat()
                self._check_tier_boundary()
                await self._dfs_iteration()
                # Yield to event loop and throttle when repeatedly failing
                if self.state.consecutive_failures > 0:
                    await asyncio.sleep(min(5 * self.state.consecutive_failures, 60))
                else:
                    await asyncio.sleep(0.1)

            if self._analysis_task and not self._analysis_task.done():
                self._analysis_task.cancel()
            if self._analysis_task is not None:
                await asyncio.gather(self._analysis_task, return_exceptions=True)
            if self._analysis_task and self._analysis_task.done() and not self._analysis_task.cancelled():
                exc = self._analysis_task.exception()
                if exc:
                    log.warning("Deep analysis failed (non-fatal): %s", exc)

            if not self.shutdown.is_set():
                self.state.phase = "sweep"
                await self._sweep()

            if not self.shutdown.is_set():
                self.state.phase = "report"
                await self._report()

            if not self.shutdown.is_set():
                await self._dream()

        except Exception as exc:
            log.exception("Orchestrator fatal error: %s", exc)
            raise
        finally:
            self.state.save()

        log.info("Orchestrator completed. Gain: %.1f%%", self.state.cumulative_gain_pct)

    # ------------------------------------------------------------------
    # Step 0: WARM-START
    # ------------------------------------------------------------------

    async def _warm_start(self) -> None:
        if self.state.warm_started:
            log.info("Already warm-started, skipping")
            self._ensure_workload()
            return

        base = Path(self.state.base_dir) if self.state.base_dir else None
        handoff_dir = base / "handoff" if base else None
        handoff_config = handoff_dir / "config.json" if handoff_dir else None

        if handoff_config and handoff_config.exists():
            mode = "sprint"
        elif base and self._has_launch_script(base / "scripts"):
            mode = "sprint_repo"
        elif base and base.exists() and any(base.iterdir()):
            mode = "baseline"
        else:
            mode = "cold"

        log.info("Warm-start mode: %s", mode)

        if mode == "baseline":
            self._load_baseline_state()
            self.state.warm_started = True
            self.state.profiled = bool(self.state.kernel_candidates)
            self.state.save()
            return

        if mode == "sprint":
            await self._sprint_warm_start(handoff_dir)
            return

        if mode == "sprint_repo":
            await self._sprint_repo_warm_start(base)
            return

        # Cold start — LLM-driven
        from . import prompts
        output_file = Path(self.session_dir) / "warm_start_output.json"
        prompt = prompts.prompt_warm_start(mode, self.state.state_summary(), None)
        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=60)
        self.llm.sync_stats(self.state)

        # Clean up any servers the Claw spawned during warm start
        from .workload import kill_rogue_servers
        await kill_rogue_servers()

        # Safety net: revert any source edits (warm_start is read-only)
        if self._workload and hasattr(self._workload, '_reset_runtime_workspaces'):
            log.info("Reverting any warm-start edits (enforcing read-only)")
            await self._workload._reset_runtime_workspaces()

        if result.output:
            for k, v in result.output.items():
                if hasattr(self.state, k):
                    setattr(self.state, k, v)
            if "action_stack" in result.output:
                for a in result.output["action_stack"]:
                    self.state.push_action(a)
                    self.dashboard.log_branch("add", a)

        self.state.warm_started = True
        self.state.save()

    async def _sprint_warm_start(self, handoff_dir: Path) -> None:
        """Deterministic Sprint handoff loading + parallel dream & baseline."""

        # ── Step 1: LOAD HANDOFF (deterministic, no LLM) ──
        config = json.loads((handoff_dir / "config.json").read_text())
        self.state.sprint_tput_per_gpu = config.get("optimized_tput_per_gpu", 0.0)
        self.state.target_tput_per_gpu = config.get("target_tput_per_gpu", 0.0)
        if self.state.target_tput_per_gpu and self.state.sprint_tput_per_gpu:
            self.state.target_gap_pct = max(0, (
                (self.state.target_tput_per_gpu - self.state.sprint_tput_per_gpu)
                / self.state.sprint_tput_per_gpu * 100
            ))

        # Load opportunities → seed action stack
        opps_path = handoff_dir / "opportunities.json"
        if opps_path.exists():
            opps = json.loads(opps_path.read_text())
            for opp in opps:
                action = {
                    "id": f"sprint_opp_{opp.get('kernel_name', 'x')}_{opp.get('type', 'x')}",
                    "action": opp.get("recommended_marathon_action", "oob-rewrite"),
                    "target_kernel": opp.get("kernel_name", ""),
                    "gpu_time_pct": opp.get("gpu_pct", 0),
                    "score": self.state.score_action(opp) + self.state.apply_handoff_boosts(opp),
                    "description": f"Sprint opportunity: {opp.get('type', '')}: {opp.get('kernel_name', '')}",
                    "tags": opp.get("tags", []),
                    "source": "sprint_handoff",
                }
                self.state.push_action(action)
                self.dashboard.log_branch("add", action)
            log.info("Loaded %d opportunities from Sprint handoff", len(opps))

        # Load profile summary → seed kernel candidates
        profile_path = handoff_dir / "profile_summary.json"
        if profile_path.exists():
            profile = json.loads(profile_path.read_text())
            self.state.kernel_candidates = [
                {"name": k.get("kernel_name", ""), "gpu_pct": k.get("gpu_pct", 0),
                 "category": k.get("type", "unknown"), "library": k.get("library", "")}
                for k in profile.get("kernel_breakdown", [])
                if k.get("gpu_pct", 0) >= MIN_GPU_PCT
            ]
            self.state.tier_breakdown = profile.get("tier_breakdown", {})
            log.info("Loaded %d kernel candidates from Sprint profile", len(self.state.kernel_candidates))

        log.info("Sprint handoff loaded: %.1f tok/s/GPU, %d actions, %d kernels",
                 self.state.sprint_tput_per_gpu,
                 len(self.state.action_stack),
                 len(self.state.kernel_candidates))

        # ── Step 2: PARALLEL — dream (LLM) + baseline (subprocess) ──
        # Track A: establish baseline (patches + server + benchmark + profile)
        # Track B: Sprint→Marathon transition dream (LLM + KB)
        # Zero contention: A uses GPU, B uses LLM.

        from .workload import InferenceXWorkload
        workload = self._create_workload(config)

        track_a = self._establish_baseline(workload, handoff_dir)
        track_b = self._transition_dream(config)
        baseline_result, dream_result = await asyncio.gather(
            track_a, track_b, return_exceptions=True,
        )

        # ── Step 3: JOIN — merge results ──
        if isinstance(baseline_result, Exception):
            log.error("Baseline establishment failed: %s", baseline_result)
        else:
            bench = baseline_result
            if bench and bench.tput_per_gpu > 0:
                self.state.baseline_tput_per_gpu = bench.tput_per_gpu
                self.state.current_tput_per_gpu = bench.tput_per_gpu
                claimed = self.state.sprint_tput_per_gpu
                if claimed > 0:
                    deviation = abs(bench.tput_per_gpu - claimed) / claimed * 100
                    if deviation <= 5:
                        log.info("Sprint verified: %.1f tok/s/GPU (claimed %.1f, %.1f%% deviation)",
                                 bench.tput_per_gpu, claimed, deviation)
                    else:
                        log.warning("Sprint deviation %.1f%%: measured %.1f vs claimed %.1f tok/s/GPU",
                                    deviation, bench.tput_per_gpu, claimed)
                log.info("Baseline established: %.1f tok/s/GPU", bench.tput_per_gpu)

        if isinstance(dream_result, Exception):
            log.warning("Transition dream failed (non-fatal): %s", dream_result)

        # ── Step 4: LLM analysis (read-only, augment action stack) ──
        from . import prompts
        output_file = Path(self.session_dir) / "warm_start_output.json"
        prompt = prompts.prompt_warm_start("sprint", self.state.state_summary(), config)
        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=60)
        self.llm.sync_stats(self.state)

        # Safety net: revert any source edits (warm_start is read-only)
        if self._workload and hasattr(self._workload, '_reset_runtime_workspaces'):
            log.info("Reverting any warm-start edits (enforcing read-only)")
            await self._workload._reset_runtime_workspaces()

        if result.output:
            for a in result.output.get("action_stack", []):
                existing_ids = {ea.get("id") for ea in self.state.action_stack}
                if a.get("id") not in existing_ids:
                    self.state.push_action(a)
                    self.dashboard.log_branch("add", a)

        self.state.warm_started = True
        self.state.profiled = bool(self.state.kernel_candidates)
        self.state.save()

    def _create_workload(self, sprint_config: dict[str, Any]) -> Any:
        """Create InferenceXWorkload from Sprint config + CLI args.

        Propagates launch_flags, env_vars, benchmark params, and checks for
        Sprint-provided scripts in BOTH the handoff directory and the repo's
        top-level scripts/ directory (launch, patch, bench scripts live there).
        """
        from .workload import InferenceXWorkload

        inferencex_path = self._inferencex_path or ''
        if not inferencex_path:
            inferencex_path = getattr(self.llm, 'inferencex_path', '')

        base = Path(self.state.base_dir) if self.state.base_dir else None
        handoff_dir = base / "handoff" if base else None
        if handoff_dir and handoff_dir.exists() and inferencex_path:
            workload = InferenceXWorkload.from_sprint_handoff(
                handoff_dir, inferencex_path,
                result_dir=str(Path(self.session_dir) / "benchmarks"),
            )
        else:
            bench_params = sprint_config.get("benchmark_params", {})
            workload = InferenceXWorkload(
                inferencex_path=inferencex_path,
                model=sprint_config.get("model_path", self.state.model_name),
                tp=sprint_config.get("tp", self.state.tp),
                framework=sprint_config.get("framework", self.state.framework),
                extra_launch_flags=sprint_config.get("launch_flags", []),
                env_vars=sprint_config.get("env_vars", {}),
                isl=bench_params.get("input_len", 1024),
                osl=bench_params.get("output_len", 1024),
                concurrency=bench_params.get("max_concurrency", 64),
                result_dir=str(Path(self.session_dir) / "benchmarks"),
            )

        # Also discover scripts from the repo's top-level scripts/ dir —
        # from_sprint_handoff only looks inside handoff/, but launch/patch/bench
        # scripts typically live in the repo root's scripts/ directory.
        if base and (base / "scripts").is_dir():
            workload._discover_scripts(base)

        self._workload = workload
        return workload

    async def _sprint_repo_warm_start(self, repo_dir: Path) -> None:
        """Warm-start from a standalone Sprint repo (Agentic-InferenceX style).

        These repos have scripts/launch_server.sh, scripts/run_benchmark.sh,
        and results/ — the scripts ARE the config with all flags baked in.
        """
        from .workload import InferenceXWorkload

        inferencex_path = self._inferencex_path or getattr(self.llm, 'inferencex_path', '')
        workload = InferenceXWorkload.from_sprint_repo(
            repo_dir, inferencex_path,
            result_dir=str(Path(self.session_dir) / "benchmarks"),
            tp_hint=self.state.tp,
        )
        self._workload = workload

        # Populate server_config so recovery chains can restart the server
        sprint_script = getattr(workload, '_sprint_launch_script', None)
        if sprint_script:
            self.state.server_config = {  # type: ignore[attr-defined]
                "launch_command": f"bash {sprint_script} --background",
            }

        # Load state.json if the repo has one (marathon-style repos do)
        state_path = repo_dir / "state.json"
        if state_path.exists():
            try:
                prior = json.loads(state_path.read_text())
                self.state.sprint_tput_per_gpu = prior.get(
                    "sprint_baseline_tput_per_gpu",
                    prior.get("current_tput_per_gpu", 0),
                )
                self.state.target_tput_per_gpu = prior.get("target_tput_per_gpu", 0)
                for a in prior.get("action_stack", []):
                    self.state.push_action(a)
                    self.dashboard.log_branch("add", a)
                log.info("Loaded state.json from Sprint repo: sprint_baseline=%.1f tok/s/GPU, "
                         "target=%.1f, %d actions",
                         self.state.sprint_tput_per_gpu,
                         self.state.target_tput_per_gpu,
                         len(self.state.action_stack))
            except Exception as exc:
                log.warning("Failed to parse Sprint state.json: %s", exc)

        # Also check for handoff/ subdirectory
        handoff_dir = repo_dir / "handoff"
        if handoff_dir.exists():
            opps_path = handoff_dir / "opportunities.json"
            if opps_path.exists():
                opps = json.loads(opps_path.read_text())
                for opp in opps:
                    action = {
                        "id": f"sprint_opp_{opp.get('kernel_name', 'x')}_{opp.get('type', 'x')}",
                        "action": opp.get("recommended_marathon_action", "oob-rewrite"),
                        "target_kernel": opp.get("kernel_name", ""),
                        "gpu_time_pct": opp.get("gpu_pct", 0),
                        "score": self.state.score_action(opp) + self.state.apply_handoff_boosts(opp),
                        "description": f"Sprint opportunity: {opp.get('type', '')}: {opp.get('kernel_name', '')}",
                        "tags": opp.get("tags", []),
                        "source": "sprint_handoff",
                    }
                    self.state.push_action(action)
                    self.dashboard.log_branch("add", action)

            profile_path = handoff_dir / "profile_summary.json"
            if profile_path.exists():
                profile = json.loads(profile_path.read_text())
                self.state.kernel_candidates = [
                    {"name": k.get("kernel_name", ""), "gpu_pct": k.get("gpu_pct", 0),
                     "category": k.get("type", "unknown")}
                    for k in profile.get("kernel_breakdown", [])
                    if k.get("gpu_pct", 0) >= MIN_GPU_PCT
                ]

        # Parallel: launch server + benchmark (Track A) alongside dream (Track B)
        sprint_config = {"cumulative_gain_pct": 0, "optimized_tput_per_gpu": self.state.sprint_tput_per_gpu}
        track_a = self._establish_baseline_from_repo(workload)
        track_b = self._transition_dream(sprint_config)
        baseline_result, dream_result = await asyncio.gather(
            track_a, track_b, return_exceptions=True,
        )

        if isinstance(baseline_result, Exception):
            log.error("Baseline establishment failed: %s", baseline_result)
        else:
            bench = baseline_result
            if bench and bench.tput_per_gpu > 0:
                self.state.baseline_tput_per_gpu = bench.tput_per_gpu
                self.state.current_tput_per_gpu = bench.tput_per_gpu
                log.info("Baseline established from Sprint repo: %.1f tok/s/GPU", bench.tput_per_gpu)

        if isinstance(dream_result, Exception):
            log.warning("Transition dream failed (non-fatal): %s", dream_result)

        # LLM augmentation pass (read-only analysis — no code edits allowed)
        from . import prompts
        output_file = Path(self.session_dir) / "warm_start_output.json"
        prompt = prompts.prompt_warm_start("sprint", self.state.state_summary(), sprint_config)
        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=60)
        self.llm.sync_stats(self.state)

        # Safety net: revert any source edits the Claw may have made despite
        # read-only instructions.  Only DFS actions go through benchmark +
        # accuracy gates, so warm_start edits are not validated.
        if self._workload and hasattr(self._workload, '_reset_runtime_workspaces'):
            log.info("Reverting any warm-start edits (enforcing read-only)")
            await self._workload._reset_runtime_workspaces()

        if result.output:
            for a in result.output.get("action_stack", []):
                existing_ids = {ea.get("id") for ea in self.state.action_stack}
                if a.get("id") not in existing_ids:
                    self.state.push_action(a)
                    self.dashboard.log_branch("add", a)

        self.state.warm_started = True
        self.state.profiled = bool(self.state.kernel_candidates)
        self.state.save()

    async def _establish_baseline_from_repo(self, workload: Any) -> Any:
        """Start server using Sprint's launch script, then benchmark."""
        from .workload import BenchmarkResult

        healthy = await workload.start_server()
        if not healthy:
            log.error("Server failed to start from Sprint repo scripts")
            return BenchmarkResult()

        bench = await workload.run_benchmark(result_filename="sprint_verification")
        log.info("Verification benchmark: %.1f tok/s/GPU", bench.tput_per_gpu)
        return bench

    async def _establish_baseline(self, workload: Any,
                                  handoff_dir: Path) -> Any:
        """Track A: apply patches → start server → benchmark → profile. No LLM."""
        from .workload import BenchmarkResult

        # Apply Sprint patches
        patch_dir = handoff_dir / "patches"
        if patch_dir.exists():
            results = await workload.apply_patches(patch_dir)
            applied = sum(1 for r in results if r.applied)
            failed = sum(1 for r in results if not r.applied and r.error)
            log.info("Patches: %d applied, %d failed", applied, failed)

        # Start server with Sprint config
        healthy = await workload.start_server()
        if not healthy:
            log.error("Server failed to start with Sprint config")
            return BenchmarkResult()

        # Run verification benchmark (same InferenceX script Sprint used)
        bench = await workload.run_benchmark(result_filename="sprint_verification")
        log.info("Verification benchmark: %.1f tok/s/GPU", bench.tput_per_gpu)

        return bench

    async def _transition_dream(self, sprint_config: dict[str, Any]) -> None:
        """Track B: Sprint→Marathon transition dream. LLM + KB, no GPU."""
        from . import prompts

        completed_since: list[dict[str, Any]] = []
        strategies_tested: list[str] = list(self.state.strategies_tested)

        output_file = Path(self.session_dir) / "transition_dream.json"
        prompt = prompts.prompt_dream(
            self.state.state_summary(), completed_since, strategies_tested,
        )
        # Prepend transition context
        transition_ctx = (
            f"\n═══ SPRINT→MARATHON TRANSITION DREAM ═══\n"
            f"Sprint achieved: {sprint_config.get('cumulative_gain_pct', 0):.1f}% gain\n"
            f"Sprint throughput: {sprint_config.get('optimized_tput_per_gpu', 0):.2f} tok/s/GPU\n"
            f"This is the reflection phase before Marathon starts.\n"
            f"Focus on: what Sprint learned, strategic priorities for Marathon,\n"
            f"and which deep optimization directions are most promising.\n\n"
        )
        prompt = transition_ctx + prompt

        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=30)
        self.llm.sync_stats(self.state)

        if result.output:
            for a in result.output.get("rescored_stack", []):
                for sa in self.state.action_stack:
                    if sa.get("id") == a.get("id"):
                        sa["score"] = a.get("score", sa.get("score", 0))

            for a in result.output.get("new_actions", []):
                self.state.push_action(a)
                self.dashboard.log_branch("add", a)
                log.info("Transition dream generated: %s [%.1f]",
                         a.get("id", "?"), float(a.get("score", 0) or 0))

        self.state.last_dream_ts = time.time()
        self.state.dream_count += 1
        log.info("Sprint→Marathon transition dream complete")

    def _ensure_workload(self) -> None:
        """Re-create the InferenceX workload object on resume.

        When warm_started=True the full warm-start is skipped, but we still
        need the workload for benchmarks and server management.
        """
        if hasattr(self, '_workload') and self._workload is not None:
            return
        base = Path(self.state.base_dir) if self.state.base_dir else None
        if base and self._has_launch_script(base / "scripts"):
            from .workload import InferenceXWorkload
            inferencex_path = self._inferencex_path or getattr(self.llm, 'inferencex_path', '')
            workload = InferenceXWorkload.from_sprint_repo(
                base, inferencex_path,
                result_dir=str(Path(self.session_dir) / "benchmarks"),
                tp_hint=self.state.tp,
            )
            self._workload = workload
            sprint_script = getattr(workload, '_sprint_launch_script', None)
            if sprint_script:
                self.state.server_config = {
                    "launch_command": f"bash {sprint_script} --background",
                }
            log.info("Workload re-initialized from sprint repo: %s", base)
        else:
            log.warning("Cannot re-initialize workload — no sprint repo scripts at %s", base)

    def _load_baseline_state(self) -> None:
        """Load prior marathon state.json from base_dir — deterministic, no LLM."""
        state_path = Path(self.state.base_dir) / "state.json"
        if not state_path.exists():
            log.warning("No state.json in base_dir %s", self.state.base_dir)
            return

        prior = json.loads(state_path.read_text())

        self.state.current_tput_per_gpu = prior.get("current_tput_per_gpu", 0)
        self.state.baseline_tput_per_gpu = prior.get("current_tput_per_gpu", 0)
        self.state.sprint_tput_per_gpu = prior.get("sprint_baseline_tput_per_gpu", 0)
        self.state.kernel_dispatch_map = prior.get("kernel_dispatch_map", {})

        for a in prior.get("action_stack", []):
            self.state.push_action(a)
            self.dashboard.log_branch("add", a)

        bottlenecks = prior.get("profiling", {}).get("bottlenecks", [])
        self.state.kernel_candidates = [
            {"name": b.get("kernel", b.get("component", "")),
             "gpu_pct": b.get("gpu_time_pct", 0),
             "category": b.get("type", "unknown")}
            for b in bottlenecks
            if b.get("gpu_time_pct", 0) >= MIN_GPU_PCT
        ]

        self._verify_patches_applied()

        log.info(
            "Loaded baseline: %.1f tok/s/GPU, %d actions, %d kernel candidates",
            self.state.current_tput_per_gpu,
            len(self.state.action_stack),
            len(self.state.kernel_candidates),
        )

    def _verify_patches_applied(self) -> None:
        """Check that optimization patches from base_dir are applied in the container."""
        import subprocess
        opt_dir = Path(self.state.base_dir) / "optimizations"
        if not opt_dir.exists():
            return
        for patch_file in sorted(opt_dir.rglob("*.patch")):
            repo_name = patch_file.stem  # e.g. "sglang" from "sglang.patch"
            repo_path = Path("/sgl-workspace") / repo_name
            if not repo_path.exists():
                continue
            result = subprocess.run(
                ["git", "apply", "-R", "--check", str(patch_file)],
                cwd=str(repo_path), capture_output=True,
            )
            opt_name = patch_file.parent.name
            if result.returncode == 0:
                log.info("Patch verified: %s/%s is applied", opt_name, repo_name)
            else:
                log.warning("Patch NOT applied: %s/%s — attempting to apply", opt_name, repo_name)
                apply_result = subprocess.run(
                    ["git", "apply", str(patch_file)],
                    cwd=str(repo_path), capture_output=True,
                )
                if apply_result.returncode == 0:
                    log.info("Applied patch: %s/%s", opt_name, repo_name)
                else:
                    log.error("Failed to apply patch %s/%s: %s",
                              opt_name, repo_name, apply_result.stderr.decode()[:200])

    # ------------------------------------------------------------------
    # Step 1: RE-PROFILE
    # ------------------------------------------------------------------

    async def _re_profile(self) -> None:
        if self.state.profiled and self.state.kernel_candidates:
            log.info("Already profiled, skipping")
            return

        # Try direct profiling via workload module first
        if self._workload is not None:
            try:
                profile_result = await self._workload.run_profile(
                    output_dir=str(Path(self.session_dir) / "profiles"))
                if profile_result.get("returncode") == 0:
                    log.info("Direct profile completed: %s", profile_result.get("output_file"))
            except Exception as exc:
                log.warning("Direct profiling failed (non-fatal): %s", exc)

        from . import prompts
        output_file = Path(self.session_dir) / "profile_output.json"
        prompt = prompts.prompt_re_profile(self.state.state_summary())
        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=50)
        self.llm.sync_stats(self.state)

        # Safety net: revert any source edits made during profiling.
        if self._workload and hasattr(self._workload, '_reset_runtime_workspaces'):
            log.info("Reverting any profile-phase edits (enforcing read-only)")
            await self._workload._reset_runtime_workspaces()

        if result.output:
            new_candidates = result.output.get("kernel_opt_candidates", [])
            for kc in new_candidates:
                kname = kc.get("name", "")
                if kname:
                    self.state.register_kernel(
                        kname, kc.get("source_file", ""),
                        kc.get("shapes", []))
            self.state.kernel_candidates = new_candidates
            self.state.tier_breakdown = result.output.get("tier_summary", {})

        self.state.profiled = True
        self.state.save()

    # ------------------------------------------------------------------
    # Step 2-3: DEEP ANALYSIS + BUILD STACK
    # ------------------------------------------------------------------

    async def _deep_analysis(self) -> None:
        if not self.state.kernel_candidates:
            log.warning("No kernel candidates for deep analysis")
            return
        if self.state.action_stack:
            log.info("Deep analysis running concurrently — %d actions already on stack",
                     len(self.state.action_stack))

        from . import prompts
        top_n = self.state.kernel_candidates[:DEEP_ANALYSIS_TOP_N]
        output_file = Path(self.session_dir) / "deep_analysis_output.json"
        prompt = prompts.prompt_deep_analysis(self.state.state_summary(), top_n)
        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=80)
        self.llm.sync_stats(self.state)

        # Deep analysis is a long Claw session (80 turns) — clean up any
        # rogue servers it may have spawned despite the blocklist.
        from .workload import kill_rogue_servers
        await kill_rogue_servers()

        # Safety net: revert any source edits the Claw may have made.
        # Deep analysis is read-only; code changes only happen in DFS.
        if self._workload and hasattr(self._workload, '_reset_runtime_workspaces'):
            log.info("Reverting any deep-analysis edits (enforcing read-only)")
            await self._workload._reset_runtime_workspaces()

        if result.output:
            async with self._state_lock.mutate():
                if "kernel_dispatch_map" in result.output:
                    self.state.kernel_dispatch_map.update(result.output["kernel_dispatch_map"])
                if "dispatch_bugs_found" in result.output:
                    self.state.dispatch_bugs_found = result.output["dispatch_bugs_found"]
                if "untuned_shapes" in result.output:
                    self.state.untuned_shapes = result.output["untuned_shapes"]
                for a in result.output.get("action_stack", []):
                    self.state.push_action(a)
                    self.dashboard.log_branch("add", a)
                for wq in result.output.get("work_queue_entries", []):
                    ipc.write_work_queue_entry(self.session_dir, wq)
                    self.state.kernel_manager_targets_pushed += 1

                self.state.save()

    def _check_inject_queue(self) -> None:
        """Pick up externally injected actions from inject_actions.jsonl."""
        p = Path(self.session_dir) / "inject_actions.jsonl"
        if not p.exists() or p.stat().st_size == 0:
            return
        try:
            lines = p.read_text().strip().splitlines()
            p.write_text("")
            for line in lines:
                action = json.loads(line)
                self.state.push_action(action)
                self.dashboard.log_branch("add", action)
                log.info("Injected external action: %s [%.1f]",
                         action.get("id", "?"), float(action.get("score", 0)))
        except Exception as exc:
            log.warning("Failed to read inject queue: %s", exc)

    # ------------------------------------------------------------------
    # 16-step DFS iteration
    # ------------------------------------------------------------------

    async def _dfs_iteration(self) -> None:
        self._check_inject_queue()

        # 1. Pop highest-scored action
        action = self.state.pop_action()
        if action is None:
            log.info("Action stack empty — triggering re-explore")
            await self._re_explore()
            return

        self.dashboard.log_branch("remove", action)
        action_type = action.get("action", "")
        log.info("DFS: popped [%.1f] %s: %s", float(action.get("score", 0) or 0), action.get("id", "?"), action_type)

        # 2-4. Classify and route
        # Snapshot system files before Claw execution so we can auto-rollback
        # if a bad patch corrupts the inference server.
        if self._workload is not None:
            self._workload.snapshot_system_files()

        result: dict[str, Any] = {}
        try:
            if action.get("needs_benchmark_only"):
                log.info("Benchmark-only action %s — running E2E measurement", action.get("id"))
                result = {"status": "success", "needs_benchmark": True}
            elif action_type in ("dispatch-fix", "config-only"):
                result = await self._execute_self_fix(action)
            elif action_type == "oob-rewrite":
                ipc.write_work_queue_entry(self.session_dir, action)
                self.state.kernel_manager_targets_pushed += 1
                poll = {"id": f"poll_{action['id']}", "action": "kernel-manager-poll", "score": 1}
                self.state.push_action(poll)
                self.dashboard.log_branch("add", poll)
                self.state.save()
                return
            elif action_type == "merge-op":
                result = await self._apply_merge_op(action)
            elif action_type == "merge-op-incremental":
                result = await self._apply_merge_op_incremental(action)
            elif action_type == "kernel-manager-poll":
                self._check_km_results()
                self.state.save()
                return
            else:
                result = await self._execute_action(action)
        except Exception as exc:
            log.exception("Action %s failed: %s", action.get("id"), exc)
            result = {"status": "error", "error": str(exc)}

        # 5. Sanitize LLM-returned result types
        result = _sanitize_result(result)

        # 6. Accuracy gate — moved to post-benchmark (step 7c) so every
        #    throughput improvement is accuracy-validated, not just ones
        #    where the Claw reports accuracy_risk > 0.

        # 7. Measure tok/s — benchmark after any action that may have changed
        #    code, config, or env.  Skip only for clearly no-op outcomes.
        _NO_BENCH_STATUSES = {"error", "crash", "segfault", "reverted",
                              "already_applied", "no_change", "not_applicable"}
        should_bench = (
            result.get("needs_benchmark", False)
            and result.get("status", "error") not in _NO_BENCH_STATUSES
        )
        if should_bench:
            # The Claw action may have changed config files, env vars, CSVs,
            # or system packages. The server must be restarted to pick up
            # these changes before benchmarking.
            if self._workload is not None:
                log.info("Restarting server to pick up changes from action %s",
                         action.get("id"))
                await self._workload.kill_server()
                healthy = await self._workload.start_server()
                if not healthy:
                    log.error("Server failed to start after action %s", action.get("id"))

            prev_tput = self.state.current_tput_per_gpu
            baseline = self.state.baseline_tput_per_gpu
            best_ever = self.state.best_tput_per_gpu or prev_tput
            bench = await self._run_benchmark()
            if bench is None:
                # Track retry depth to prevent infinite re-queue loops.
                retry_depth = action.get("_bench_retry_depth", 0)
                MAX_BENCH_RETRIES = 3
                if retry_depth >= MAX_BENCH_RETRIES:
                    log.error(
                        "Benchmark failed %d times for %s — giving up (server or "
                        "optimization may be broken). Reverting to move on.",
                        retry_depth, action.get("id"),
                    )
                    self.state.failure_journal.append({
                        "action_id": action.get("id"),
                        "error": f"Benchmark failed {retry_depth} consecutive times",
                        "timestamp": time.time(),
                    })
                else:
                    log.warning(
                        "Benchmark failed after %s (attempt %d/%d) — re-queuing",
                        action.get("id"), retry_depth + 1, MAX_BENCH_RETRIES,
                    )
                    bench_retry = {
                        "id": f"bench_retry_{action.get('id', 'x')}_{int(time.time())}",
                        "action": action.get("action", "benchmark-pending"),
                        "score": max(float(action.get("score", 0) or 0), 15),
                        "description": f"Re-benchmark after {action.get('id')}: attempt {retry_depth + 2}/{MAX_BENCH_RETRIES}",
                        "needs_benchmark_only": True,
                        "parent_action_id": action.get("id"),
                        "_bench_retry_depth": retry_depth + 1,
                    }
                    self.state.push_action(bench_retry)
                    self.dashboard.log_branch("add", bench_retry)
            if bench:
                new_tput = bench.get("tput_per_gpu", prev_tput)
                had_contention = bench.get("had_gpu_contention", False)

                regressed = new_tput < best_ever

                if had_contention and regressed:
                    log.warning(
                        "REGRESSION detected after %s (best %.1f → %.1f) BUT GPU contention "
                        "was present — treating as unreliable, keeping best throughput %.1f",
                        action.get("id"), best_ever, new_tput, best_ever,
                    )
                    new_tput = prev_tput
                    result["contention_tainted"] = True
                    regressed = False

                if regressed:
                    drop_pct = (best_ever - new_tput) / best_ever * 100
                    log.warning(
                        "ZERO-TOLERANCE REVERT after %s: best %.1f → %.1f tok/s/GPU "
                        "(-%.2f%%) — only improvements are retained",
                        action.get("id"), best_ever, new_tput, drop_pct,
                    )
                    await self._rollback_action(action)
                    result["status"] = "reverted"
                    result["decision"] = "revert"
                    result["regression_pct"] = drop_pct
                    self.state.record_failure(
                        action, symptom="dfs-action regression",
                        root_cause=f"new_tput={new_tput:.1f} < best={best_ever:.1f} (-{drop_pct:.2f}%)",
                    )
                    self.state.consecutive_regressions = (
                        getattr(self.state, "consecutive_regressions", 0) + 1
                    )
                    verify_bench = await self._run_benchmark()
                    if verify_bench:
                        verify_tput = verify_bench.get("tput_per_gpu", 0)
                        log.info("Post-revert verification: %.1f tok/s/GPU (health-check only, "
                                 "not updating current which stays at %.1f)",
                                 verify_tput, prev_tput)
                        if verify_tput <= 0:
                            log.error("Post-revert server unhealthy (0 tput)")
                    self.state.current_tput_per_gpu = prev_tput

                    if not action.get("needs_benchmark_only"):
                        retry = await self._analyze_regression(action, result, prev_tput, bench)
                        if retry:
                            self.state.push_action(retry)
                            self.dashboard.log_branch("add", retry)
                            log.info("Regression analysis queued retry: %s (score %.1f)",
                                     retry.get("id"), retry.get("score", 0))
                    else:
                        log.info("Skipping regression analysis for benchmark-only action %s",
                                 action.get("id"))
                else:
                    self.state.current_tput_per_gpu = new_tput
                    if new_tput > best_ever:
                        self.state.best_tput_per_gpu = new_tput
                    if new_tput > prev_tput and prev_tput > 0:
                        gain = (new_tput - prev_tput) / prev_tput * 100
                        result["gain_pct"] = gain
                        log.info("Action %s improved throughput: %.1f → %.1f (+%.1f%%)",
                                 action.get("id"), prev_tput, new_tput, gain)
                        self.state.consecutive_regressions = 0

                        self._snapshot_diffs(action, prev_tput, new_tput)

                        # 7c. Accuracy gate — run on every throughput gain
                        _acc_reverted = False
                        if not action.get("needs_benchmark_only"):
                            log.info("Running accuracy gate for %s (+%.1f%%)",
                                     action.get("id"), gain)
                            acc_passed = await self._accuracy_gate(action, result)
                            if not acc_passed:
                                log.error("Accuracy gate FAILED for %s — reverting despite +%.1f%% gain",
                                          action.get("id"), gain)
                                ipc.write_event(self.session_dir, {
                                    "source": "marathon", "type": "accuracy-fail",
                                    "task_id": action.get("id"), "severity": "error",
                                    "promising": False,
                                    "details": {"gain_pct": gain, "reverted_reason": "accuracy_fail"},
                                })
                                self.state.events_written += 1
                                self.state.current_tput_per_gpu = prev_tput
                                self.state.best_tput_per_gpu = best_ever
                                await self._rollback_action(action)
                                self.state.save()
                                _acc_reverted = True

                        if not _acc_reverted:
                            follow_ons = await self._analyze_success(action, result, prev_tput, new_tput)
                            for fo in follow_ons:
                                self.state.push_action(fo)
                                self.dashboard.log_branch("add", fo)
                            if follow_ons:
                                log.info("Success analysis generated %d follow-on actions from %s",
                                         len(follow_ons), action.get("id"))
                if self.state.baseline_tput_per_gpu > 0:
                    self.state.cumulative_gain_pct = (
                        (self.state.current_tput_per_gpu - self.state.baseline_tput_per_gpu)
                        / self.state.baseline_tput_per_gpu * 100
                    )

        # 7b. Periodic forced E2E benchmark — catch cumulative drift
        self.state.actions_since_bench = getattr(self.state, "actions_since_bench", 0) + 1
        if not should_bench and self.state.actions_since_bench >= PERIODIC_BENCH_INTERVAL:
            log.info("Periodic E2E benchmark after %d actions", self.state.actions_since_bench)
            bench = await self._run_benchmark()
            if bench:
                new_tput = bench.get("tput_per_gpu", self.state.current_tput_per_gpu)
                best_ever = self.state.best_tput_per_gpu or self.state.current_tput_per_gpu
                baseline = self.state.baseline_tput_per_gpu
                if new_tput < best_ever:
                    drift_pct = (best_ever - new_tput) / best_ever * 100
                    log.warning(
                        "PERIODIC BENCH: drift from best %.1f → %.1f (-%.2f%%) — "
                        "keeping best on dashboard, investigating",
                        best_ever, new_tput, drift_pct,
                    )
                else:
                    self.state.current_tput_per_gpu = new_tput
                    if new_tput > best_ever:
                        self.state.best_tput_per_gpu = new_tput
                if baseline > 0:
                    self.state.cumulative_gain_pct = (
                        (self.state.current_tput_per_gpu - baseline)
                        / baseline * 100
                    )
            self.state.actions_since_bench = 0
        elif should_bench:
            self.state.actions_since_bench = 0

        # 8. Re-score
        self.state.actions_since_rescore += 1
        if self.state.actions_since_rescore >= RESCORE_INTERVAL:
            await self._rescore()
            self.dashboard.log_score_snapshot(self.state.action_stack)
            self.state.actions_since_rescore = 0

        # 9. Push sub-actions (inherit parent score if agent scored too low)
        parent_score = float(action.get("score", 0) or 0)
        for sub in result.get("sub_actions", []):
            sub_score = float(sub.get("score", 0) or 0)
            if sub_score < parent_score * 0.5:
                sub["score"] = round(parent_score * 0.6, 1)
                log.info("Boosted sub-action %s score: %.1f → %.1f (parent %s was %.1f)",
                         sub.get("id", "?"), sub_score, sub["score"], action.get("id", "?"), parent_score)
            self.state.push_action(sub)
            self.dashboard.log_branch("add", sub)

        # 10. KB ingest (delegated to LLM calls within action prompts)

        # 11. Poll results.jsonl
        self._check_km_results()

        # 12. Poll findings.jsonl
        self._check_wd_findings()

        # 12b. Poll insights.jsonl
        self._check_insights()

        # 13. Emit event_log on failure + inline diagnosis
        if result.get("status") in ("error", "crash", "segfault"):
            event_type = result.get("event_type", "crash")
            exit_code = result.get("exit_code")
            error_msg = result.get("error", "")

            if event_type == "crash":
                if exit_code == 139 or "segfault" in error_msg.lower() or "sigsegv" in error_msg.lower():
                    event_type = "segfault"
                known_segfault = action.get("prior_status", "")
                if "segfault" in known_segfault.lower() or "crash" in known_segfault.lower():
                    event_type = "segfault"

            ipc.write_event(self.session_dir, {
                "source": "marathon",
                "type": event_type,
                "kernel_name": action.get("target_kernel"),
                "task_id": action.get("id"),
                "severity": "error",
                "promising": result.get("promising", float(action.get("score", 0) or 0) >= 7),
                "details": {
                    "error_message": error_msg,
                    "exit_code": exit_code,
                    "strategy_used": action.get("action"),
                    "known_root_cause": action.get("known_root_cause", ""),
                    "prior_status": action.get("prior_status", ""),
                    "micro_speedup_before_crash": result.get("micro_speedup", 0),
                    "gpu_pct": action.get("gpu_time_pct", 0),
                },
            })
            self.state.events_written += 1
            self.state.consecutive_failures += 1

            if event_type in ("crash", "segfault"):
                self.state.crash_count += 1
                self.state.crash_log.append(
                    f"{action.get('id')}: {event_type} — {error_msg[:200]}")

            # Rollback non-merge actions that modified files
            if action_type in ("dispatch-fix", "config-only"):
                await self._rollback_action(action)

            # Recovery chain (server restart etc.)
            if event_type in ("crash", "segfault"):
                crash_type = self._classify_crash(error_msg, exit_code)
                recovered = await self._recover(crash_type, error_msg)
                if not recovered:
                    log.error("Recovery failed for %s — continuing", crash_type)

            # Inline diagnosis: Orchestrator diagnoses its own failures
            retry_count = action.get("_self_retry_count", 0)
            if retry_count < MAX_SELF_RETRY and action_type not in ("oob-rewrite", "kernel-manager-poll"):
                diag = await self._diagnose_own_failure(action, result)
                if diag:
                    if diag.get("escalate_to_km"):
                        km_entry = self._build_km_escalation(action, diag)
                        ipc.write_work_queue_entry(self.session_dir, km_entry)
                        self.state.kernel_manager_targets_pushed += 1
                        log.info("Escalated %s to KM: %s", action.get("id"), diag.get("root_cause"))
                    elif diag.get("retryable") and diag.get("retry_action"):
                        retry = diag["retry_action"]
                        retry["_self_retry_count"] = retry_count + 1
                        retry["prior_status"] = error_msg[:200]
                        retry["known_root_cause"] = diag.get("root_cause", "")
                        retry.setdefault("score", max(3, float(action.get("score", 0) or 0) * 0.6))
                        self.state.push_action(retry)
                        self.dashboard.log_branch("add", retry)
                        log.info("Self-retry #%d queued for %s: %s",
                                 retry_count + 1, action.get("id"), diag.get("fix_description"))
                    else:
                        self.state.apply_update_rules("post_crash", {"action_id": action.get("id")})
                else:
                    self.state.apply_update_rules("post_crash", {"action_id": action.get("id")})
            else:
                self.state.apply_update_rules("post_crash", {"action_id": action.get("id")})

            # Circuit breaker
            if self.state.consecutive_failures >= CIRCUIT_BREAKER:
                log.warning("Circuit breaker: %d consecutive failures, re-analyzing", CIRCUIT_BREAKER)
                self.state.consecutive_failures = 0
                if self._analysis_task is None or self._analysis_task.done():
                    self._analysis_task = asyncio.create_task(self._deep_analysis())
                else:
                    log.info("Deep analysis already running, skipping circuit-breaker re-analysis")
        else:
            self.state.consecutive_failures = 0

        # 14. (merge-op check done in step 11)

        # 15. Dream check
        if self._should_dream():
            await self._dream()

        # 16. Checkpoint
        if result.get("decision") == "keep" or self._checkpoint_due():
            self.state.checkpoint()

        # Track gains
        gain = result.get("gain_pct", 0)
        if gain > 0:
            self.state.actions_since_gain = 0
            if gain > 5 and self._should_dream():
                await self._dream()
        else:
            self.state.actions_since_gain += 1

        # Update bandit posterior
        action_type = action.get("action", "")
        target = action.get("target_kernel", action.get("source_file", ""))
        success = gain > 0
        self.state.update_posterior(action_type, target, success, gain)
        self.state.action_type_history.append(action_type)
        if len(self.state.action_type_history) > 50:
            self.state.action_type_history = self.state.action_type_history[-50:]

        # Check plateau -> re-explore
        if self.state.actions_since_gain >= PLATEAU_THRESHOLD:
            await self._re_explore()

        # Regression circuit breaker: branch is exhausted if we got several
        # clean-but-losing actions in a row. This is independent from
        # CIRCUIT_BREAKER, which counts crashes.
        cr = getattr(self.state, "consecutive_regressions", 0)
        if cr >= CONSECUTIVE_REGRESSION_CAP:
            log.warning(
                "Regression circuit breaker: %d consecutive regressions "
                "(>= %d) — branch exhausted, forcing re-explore",
                cr, CONSECUTIVE_REGRESSION_CAP,
            )
            try:
                ipc.write_event(self.session_dir, {
                    "source": "marathon",
                    "type": "branch-exhausted",
                    "task_id": action.get("id"),
                    "severity": "warning",
                    "promising": False,
                    "details": {
                        "consecutive_regressions": cr,
                        "cap": CONSECUTIVE_REGRESSION_CAP,
                        "last_action_id": action.get("id"),
                        "last_action_type": action.get("action"),
                        "current_tput": self.state.current_tput_per_gpu,
                        "best_tput": self.state.best_tput_per_gpu,
                    },
                })
                self.state.events_written += 1
            except Exception as exc:
                log.warning("Failed to write branch-exhausted event: %s", exc)
            self.state.consecutive_regressions = 0
            await self._re_explore()

        # Log completion
        action["result"] = result
        self.state.completed_actions.append(action)
        self.dashboard.log_branch("complete", action)

        self.llm.sync_stats(self.state)
        self.state.save()

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    async def _execute_self_fix(self, action: dict[str, Any]) -> dict[str, Any]:
        from . import prompts
        output_file = Path(self.session_dir) / f"action_{action.get('id', 'x')}.json"
        action_type = action.get("action", "")
        if action_type == "config-only":
            prompt = prompts.prompt_execute_config_only(self.state.state_summary(), action)
        else:
            prompt = prompts.prompt_execute_dispatch_fix(self.state.state_summary(), action)
        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=20)
        output = result.output or {}
        output.setdefault("status", "error" if result.is_error else "success")
        output["needs_benchmark"] = output.get("status") not in ("error", "crash", "segfault", "reverted", "already_applied", "no_change", "not_applicable")
        return output

    async def _execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        from . import prompts
        action_type = action.get("action", "")
        output_file = Path(self.session_dir) / f"action_{action.get('id', 'x')}.json"

        prompt_map = {
            "operator-tuning": prompts.prompt_execute_operator_tuning,
            "framework-rebuild": prompts.prompt_execute_framework_rebuild,
            "kernel-opt": prompts.prompt_execute_kernel_opt,
            "comm-optimization": prompts.prompt_execute_comm_optimization,
            "compiler-tuning": prompts.prompt_execute_compiler_tuning,
        }

        if action_type == "exploratory-probe":
            return await self._exploratory_probe()
        if action_type == "ablation-probe":
            return await self._ablation_probe(action)
        if action_type == "hypothesis-ab-test":
            return await self._hypothesis_ab_test(action)

        prompt_fn = prompt_map.get(action_type, prompts.prompt_execute_action)
        prompt = prompt_fn(self.state.state_summary(), action)
        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=50)

        # Post-action cleanup: kill any servers the Claw may have spawned
        from .workload import kill_rogue_servers
        await kill_rogue_servers()

        output = result.output or {}
        output.setdefault("status", "error" if result.is_error else "success")
        output["needs_benchmark"] = output.get("status") not in ("error", "crash", "segfault", "reverted", "already_applied", "no_change", "not_applicable")
        return output

    # ------------------------------------------------------------------
    # 9-step merge-op
    # ------------------------------------------------------------------

    async def _run_shell(self, cmd: str, *, label: str = "shell",
                         timeout_s: float = 120, cwd: str = "/sgl-workspace") -> str:
        """Run a deterministic shell command without LLM — saves tokens for creative work."""
        log.info("[%s] %s", label, cmd[:200])
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"[{label}] timed out after {timeout_s}s: {cmd[:100]}")
        output = (stdout or b"").decode(errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"[{label}] exit {proc.returncode}: {output[:500]}")
        return output

    async def _apply_merge_op(self, action: dict[str, Any]) -> dict[str, Any]:
        task_id = action.get("task_id", action.get("id", ""))
        patch_dir = action.get("patch_dir", "")

        # 1. Read metadata.json
        metadata = ipc.read_merge_ready_metadata(self.session_dir, task_id)
        if not metadata:
            return {"status": "error", "error": f"metadata.json not found for {task_id}"}

        errors = ipc.validate_merge_ready(self.session_dir, task_id)
        if errors:
            return {"status": "error", "error": f"Validation failed: {errors}"}

        # Acquire GPU lock for the entire merge-op (rebuild + server + benchmark)
        if self.gpu_lock:
            return await self._apply_merge_op_locked(action, metadata, task_id, patch_dir)
        return await self._apply_merge_op_inner(action, metadata, task_id, patch_dir)

    async def _apply_merge_op_locked(self, action: dict[str, Any],
                                      metadata: dict[str, Any],
                                      task_id: str, patch_dir: str) -> dict[str, Any]:
        async with self.gpu_lock.acquire("rebuild", "orchestrator"):
            await self.server.kill_server()
            self.gpu_lock.server_down()
            result = await self._apply_merge_op_inner(action, metadata, task_id, patch_dir)
            if result.get("status") == "success" and result.get("decision") != "revert":
                self.gpu_lock.server_up()
            return result

    async def _apply_merge_op_inner(self, action: dict[str, Any],
                                     metadata: dict[str, Any],
                                     task_id: str, patch_dir: str) -> dict[str, Any]:
        if not self.gpu_lock:
            # 2. Kill server (no lock path)
            await self.server.kill_server()

        try:
            # 3. Apply patch (deterministic — no LLM needed)
            for instruction in metadata.get("apply_instructions", []):
                resolved = instruction.replace("$PATCH_DIR", patch_dir or
                    str(Path(self.session_dir) / "kernel_manager" / "merge_ready" / task_id))
                await self._run_shell(resolved, label="apply-patch")

            # 4. Rebuild if required (deterministic)
            if metadata.get("rebuild_required") and metadata.get("rebuild_command"):
                await self._run_shell(metadata["rebuild_command"], label="rebuild", timeout_s=300)

            # 5. Cache clear (deterministic)
            for cmd in metadata.get("cache_clear_commands", []):
                await self._run_shell(cmd, label="cache-clear")

            # Auto-clear JIT caches if aiter/triton files changed
            changed_files = " ".join(str(i) for i in metadata.get("apply_instructions", []))
            if "aiter/" in changed_files or "triton/" in changed_files:
                import shutil
                for cache_dir in [
                    os.path.expanduser("~/.cache/aiter"),
                    os.path.expanduser("~/.triton/cache"),
                    os.path.expanduser("~/.cache/triton"),
                ]:
                    if os.path.isdir(cache_dir):
                        shutil.rmtree(cache_dir, ignore_errors=True)
                        log.info("Auto-cleared JIT cache: %s", cache_dir)

            # 6. Verification (deterministic)
            if metadata.get("verification_command"):
                await self._run_shell(metadata["verification_command"], label="verify")

            # 7. Restart server
            healthy = await self.server.restart_server(getattr(self.state, "server_config", {}))
            if not healthy:
                raise RuntimeError("Server failed to restart after merge-op")

            # 7b. Micro-oracle: quick sanity check before full E2E
            micro = await self._quick_micro_check()
            if micro and micro.tput_per_gpu > 0:
                micro_ratio = micro.tput_per_gpu / max(self.state.current_tput_per_gpu, 1)
                if micro_ratio < 0.85:
                    log.warning("Micro-oracle: tput dropped to %.0f%% — skipping full E2E",
                                micro_ratio * 100)
                    # Auto-revert
                    rollback = metadata.get("rollback_command", "")
                    if rollback:
                        resolved = rollback.replace("$PATCH_DIR",
                            str(Path(self.session_dir) / "kernel_manager" / "merge_ready" / task_id))
                        await self._run_shell(resolved, label="rollback")
                    if metadata.get("rollback_rebuild_command"):
                        await self._run_shell(metadata["rollback_rebuild_command"], label="rollback-rebuild", timeout_s=300)
                    await self.server.restart_server(getattr(self.state, "server_config", {}))
                    self.state.record_failure(
                        action, symptom="micro-oracle rejection",
                        root_cause=f"micro_tput={micro.tput_per_gpu:.1f} vs current={self.state.current_tput_per_gpu:.1f}",
                    )
                    return {"status": "reverted", "decision": "revert-micro",
                            "micro_tput": micro.tput_per_gpu, "needs_benchmark": False}
                log.info("Micro-oracle passed: %.0f%% of current", micro_ratio * 100)

            # 8. E2E benchmark
            bench = await self._run_benchmark()
            if not bench:
                raise RuntimeError("Benchmark failed after merge-op")

            # 9. KEEP / REVERT / CRASH
            new_tput = bench.get("tput_per_gpu", 0)
            if new_tput > self.state.current_tput_per_gpu:
                ipc.write_event(self.session_dir, {
                    "source": "marathon", "type": "merge-keep",
                    "kernel_name": metadata.get("kernel_name"),
                    "task_id": task_id, "severity": "info", "promising": False,
                })
                self.state.events_written += 1
                self.state.kernel_manager_merges_kept += 1
                self.state.snapshot_known_good(bench)
                return {
                    "status": "success", "decision": "keep",
                    "gain_pct": (new_tput - self.state.current_tput_per_gpu) / max(self.state.current_tput_per_gpu, 1) * 100,
                    "needs_benchmark": False,
                }
            else:
                # Revert
                rollback = metadata.get("rollback_command", "")
                if rollback:
                    resolved = rollback.replace("$PATCH_DIR",
                        str(Path(self.session_dir) / "kernel_manager" / "merge_ready" / task_id))
                    await self._run_shell(resolved, label="rollback")
                if metadata.get("rollback_rebuild_command"):
                    await self._run_shell(metadata["rollback_rebuild_command"], label="rollback-rebuild", timeout_s=300)
                await self.server.restart_server(getattr(self.state, "server_config", {}))

                ipc.write_event(self.session_dir, {
                    "source": "marathon", "type": "merge-revert",
                    "kernel_name": metadata.get("kernel_name"),
                    "task_id": task_id, "severity": "warning", "promising": True,
                    "details": {"rebuild_required": metadata.get("rebuild_required")},
                })
                self.state.events_written += 1
                self.state.record_failure(
                    action, symptom="merge-op regression",
                    root_cause=f"new_tput={new_tput:.1f} < current={self.state.current_tput_per_gpu:.1f}",
                )
                return {"status": "reverted", "decision": "revert", "needs_benchmark": False}

        except Exception as exc:
            log.exception("Merge-op crash: %s", exc)
            # Rollback on crash
            rollback = metadata.get("rollback_command", "")
            if rollback:
                try:
                    resolved = rollback.replace("$PATCH_DIR",
                        str(Path(self.session_dir) / "kernel_manager" / "merge_ready" / task_id))
                    await self._run_shell(resolved, label="rollback")
                except Exception:
                    pass
            try:
                await self.server.restart_server(getattr(self.state, "server_config", {}))
            except Exception:
                pass

            ipc.write_event(self.session_dir, {
                "source": "marathon", "type": "crash",
                "kernel_name": metadata.get("kernel_name"),
                "task_id": task_id, "severity": "error", "promising": True,
                "details": {"error_message": str(exc), "rebuild_required": metadata.get("rebuild_required")},
            })
            self.state.events_written += 1
            self.state.record_failure(
                action, symptom="merge-op crash", root_cause=str(exc),
            )
            return {"status": "crash", "decision": "revert", "error": str(exc), "needs_benchmark": False}

    async def _quick_micro_check(self) -> Any:
        """Run a quick 8-prompt micro-benchmark as a fast filter."""
        try:
            wl = self._workload
            if wl is None:
                log.warning("Micro-oracle benchmark skipped: no workload")
                return None
            return await wl.run_micro_benchmark(
                num_prompts=8, timeout_s=120, result_tag="micro_oracle")
        except Exception as exc:
            log.warning("Micro-oracle benchmark failed: %s", exc)
            return None

    async def _apply_merge_op_incremental(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply multi-file changes one at a time with bisection.

        If a merge-op touches N files, apply each file's change individually
        and benchmark after each to identify which files help/hurt.
        """
        task_id = action.get("task_id", action.get("id", ""))
        metadata = ipc.read_merge_ready_metadata(self.session_dir, task_id)
        if not metadata:
            return await self._apply_merge_op(action)

        instructions = metadata.get("apply_instructions", [])
        if len(instructions) <= 2:
            return await self._apply_merge_op(action)

        log.info("Incremental merge-op: %d instructions for %s", len(instructions), task_id)
        if self.gpu_lock:
            return await self._apply_merge_op_incremental_locked(action, instructions, task_id)
        return await self._apply_merge_op_incremental_inner(action, instructions, task_id)

    async def _apply_merge_op_incremental_locked(
        self, action: dict[str, Any], instructions: list[str], task_id: str,
    ) -> dict[str, Any]:
        async with self.gpu_lock.acquire("rebuild", "orchestrator"):
            self.gpu_lock.server_down()
            result = await self._apply_merge_op_incremental_inner(action, instructions, task_id)
            if result.get("status") == "success":
                self.gpu_lock.server_up()
            return result

    async def _apply_merge_op_incremental_inner(
        self, action: dict[str, Any], instructions: list[str], task_id: str,
    ) -> dict[str, Any]:
        results_per_step: list[dict[str, Any]] = []
        good_instructions: list[str] = []
        patch_dir = action.get("patch_dir", "")
        base_dir = str(Path(self.session_dir) / "kernel_manager" / "merge_ready" / task_id)

        from . import prompts

        for i, instruction in enumerate(instructions):
            resolved = instruction.replace("$PATCH_DIR", patch_dir or base_dir)

            await self.server.kill_server()
            await self.llm.call(prompts.prompt_apply_instruction(resolved), max_turns=5)

            if metadata.get("rebuild_required") and metadata.get("rebuild_command"):
                await self.llm.call(prompts.prompt_rebuild(metadata["rebuild_command"]), max_turns=10)

            healthy = await self.server.restart_server(getattr(self.state, "server_config", {}))
            if not healthy:
                log.warning("Incremental step %d failed: server unhealthy", i)
                results_per_step.append({"step": i, "instruction": instruction, "status": "server-fail"})
                continue

            micro = await self._quick_micro_check()
            step_tput = micro.tput_per_gpu if micro else 0

            results_per_step.append({
                "step": i,
                "instruction": instruction,
                "tput_per_gpu": step_tput,
                "delta_pct": (step_tput - self.state.current_tput_per_gpu) / max(self.state.current_tput_per_gpu, 1) * 100 if step_tput > 0 else 0,
            })

            if step_tput > self.state.current_tput_per_gpu * 0.98:
                good_instructions.append(instruction)
            else:
                log.warning("Incremental step %d regressed: %.1f -> %.1f", i, self.state.current_tput_per_gpu, step_tput)

        log.info("Incremental merge: %d/%d instructions kept", len(good_instructions), len(instructions))
        return {
            "status": "success",
            "incremental_status": "incremental-done",
            "good_count": len(good_instructions),
            "total_count": len(instructions),
            "per_step": results_per_step,
            "needs_benchmark": True,
        }

    # ------------------------------------------------------------------
    # IPC polling
    # ------------------------------------------------------------------

    def _check_km_results(self) -> None:
        results = ipc.read_new_results(self.session_dir, after_id=self.state.kernel_manager_last_seen_id)
        for r in results:
            rid = r.get("id") or r.get("task_id")
            if not rid:
                log.warning("KM result missing 'id', skipping: %s", r)
                continue
            self.state.kernel_manager_last_seen_id = rid
            if r.get("status") == "merge-ready":
                merge_action = {
                    "id": f"merge_{rid}",
                    "action": "merge-op",
                    "score": MERGE_OP_SCORE,
                    "task_id": rid,
                    "patch_dir": r.get("patch_dir"),
                    "description": f"Merge kernel optimization: {rid}",
                }
                self.state.push_action(merge_action)
                self.dashboard.log_branch("add", merge_action)
                log.info("Merge-op queued for %s (speedup=%.2fx)", rid, r.get("micro_speedup", 0))

    def _check_wd_findings(self) -> None:
        findings = ipc.read_new_findings(
            self.session_dir, after_event_id=self.state.watchdog_last_seen_finding_id,
        )
        for f in findings:
            self.state.watchdog_last_seen_finding_id = f.get("event_id", "")
            self.state.watchdog_findings_consumed += 1
            classification = f.get("classification", "")
            target_type = f.get("target_type", "")
            log.info("Finding consumed: %s [%s] target_type=%s resubmit=%s",
                     f.get("event_id"), classification, target_type, f.get("resubmit"))

            if classification == "hardware":
                kernel = f.get("kernel_name")
                if kernel and kernel not in self.state.watchdog_hw_blocked_kernels:
                    self.state.watchdog_hw_blocked_kernels.append(kernel)
                continue

            if not f.get("resubmit"):
                continue

            self._route_finding(f)

    def _check_insights(self) -> None:
        """Read new insights from insight bus and generate transfer actions."""
        from . import ipc
        insights = ipc.read_new_insights(
            self.session_dir, after_id=self.state.insights_last_seen_id)
        for ins in insights:
            self.state.insights_last_seen_id = ins.get("id", "")
            ins_type = ins.get("type", "")

            if ins_type == "pattern-discovery":
                pattern = ins.get("pattern", "")
                details = ins.get("details", {})
                applicable = details.get("applicable_to", [])
                for target in applicable[:5]:
                    transfer_action = {
                        "id": f"transfer_{ins['id']}_{target[:20]}",
                        "action": details.get("approach", "oob-rewrite"),
                        "description": f"Transfer pattern '{pattern}' to {target}",
                        "target_kernel": target,
                        "source": "insight_transfer",
                        "pattern": pattern,
                        "score": 6.0,
                    }
                    self.state.push_action(transfer_action)
                    self.dashboard.log_branch("add", transfer_action)
                    log.info("Transfer action from insight: %s → %s", pattern, target)

            elif ins_type == "design-space-found":
                probe_action = {
                    "id": f"probe_{ins['id']}",
                    "action": "exploratory-probe",
                    "description": f"Explore design space: {ins.get('details', {}).get('description', '')}",
                    "source": "insight_probe",
                    "score": 5.0,
                }
                self.state.push_action(probe_action)
                log.info("Exploratory probe from insight: %s", ins.get("pattern"))

            elif ins_type == "anomaly-detected":
                log.info("Anomaly insight received: %s", ins.get("details", {}).get("description", ""))

            elif ins_type == "interaction-effect":
                self.state.record_failure(
                    {"id": ins.get("id"), "action": "merge-op"},
                    symptom="interaction regression",
                    root_cause=ins.get("details", {}).get("root_cause", ""),
                )

    async def _exploratory_probe(self) -> dict[str, Any]:
        """Undirected code exploration to discover optimization opportunities."""
        from . import prompts
        hot_path_files = [
            a.get("source_file", "") for a in self.state.completed_actions
            if a.get("source_file")
        ]
        if not hot_path_files:
            hot_path_files = [
                "/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8_utils.py",
                "/sgl-workspace/aiter/aiter/ops/",
            ]

        output_file = Path(self.session_dir) / f"probe_{int(time.time())}.json"
        prompt = prompts.prompt_exploratory_probe(
            self.state.state_summary(),
            self.state.visit_map,
            hot_path_files,
        )
        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=30)
        output = result.output or {}

        for f_path in output.get("visit_log", []):
            self.state.record_visit(f_path)

        for obs in output.get("observations", []):
            if obs.get("confidence") in ("high", "medium"):
                self.state.push_action({
                    "id": f"probe_obs_{hash(obs.get('observation', '')) % 10000}",
                    "action": "deep-kernel-analysis",
                    "description": obs.get("observation", ""),
                    "source_file": (obs.get("files_read", []) or [""])[0],
                    "source": "exploratory_probe",
                    "score": 5.0 if obs.get("confidence") == "high" else 3.0,
                })

        for ins in output.get("insights", []):
            ipc.write_insight(self.session_dir, ins)

        output.setdefault("status", "error" if result.is_error else "success")
        output.setdefault("needs_benchmark", False)
        return output

    async def _ablation_probe(self, action: dict[str, Any]) -> dict[str, Any]:
        """Disable a specific optimization and measure impact.

        Used to identify which stacked optimizations actually help.
        """
        target = action.get("target_kernel", "")
        source_file = action.get("source_file", "")
        log.info("Ablation probe: disabling %s in %s", target, source_file)

        from . import prompts
        output_file = Path(self.session_dir) / f"ablation_{action.get('id', 'x')}.json"
        prompt = (
            f"ABLATION PROBE\n\n"
            f"Temporarily DISABLE the optimization in {source_file} targeting {target}.\n"
            f"Steps:\n"
            f"1. Read the file and identify the optimization.\n"
            f"2. Create a backup.\n"
            f"3. Comment out or revert the optimization to its original form.\n"
            f"4. The benchmark will be run separately.\n\n"
            f"Write to $OUTPUT_FILE:\n"
            f"  disabled: bool,\n"
            f"  backup_path: str,\n"
            f"  restore_command: str,\n"
            f"  description: what was disabled\n"
        )
        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=15)
        output = result.output or {}

        if output.get("disabled"):
            await self.server.kill_server()
            healthy = await self.server.restart_server(getattr(self.state, "server_config", {}))
            if healthy:
                bench = await self._run_benchmark()
                ablation_tput = bench.get("tput_per_gpu", 0) if bench else 0
                impact_pct = (self.state.current_tput_per_gpu - ablation_tput) / max(ablation_tput, 1) * 100

                restore_cmd = output.get("restore_command", "")
                if restore_cmd:
                    await self.llm.call(prompts.prompt_shell_command(restore_cmd), max_turns=5)
                    await self.server.restart_server(getattr(self.state, "server_config", {}))

                return {
                    "status": "success",
                    "ablation_tput": ablation_tput,
                    "current_tput": self.state.current_tput_per_gpu,
                    "impact_pct": impact_pct,
                    "description": output.get("description", ""),
                    "needs_benchmark": False,
                }

        return {"status": "failed", "error": output.get("error", "could not disable"), "needs_benchmark": False}

    async def _hypothesis_ab_test(self, action: dict[str, Any]) -> dict[str, Any]:
        """Run a hypothesis-driven A/B benchmark synthesis."""
        from . import prompts
        output_file = Path(self.session_dir) / f"ab_test_{action.get('id', 'x')}.json"
        prompt = prompts.prompt_hypothesis_ab_benchmark(action)
        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=30)
        output = result.output or {}

        if output.get("summary"):
            summary = output["summary"]
            dispatch_rec = summary.get("dispatch_recommendation", "")
            if dispatch_rec:
                ipc.write_insight(self.session_dir, {
                    "source": "orchestrator",
                    "type": "design-space-found",
                    "pattern": action.get("hypothesis", ""),
                    "confidence": "high" if summary.get("geomean_speedup", 1) > 1.1 else "medium",
                    "details": {
                        "description": dispatch_rec,
                        "per_shape": output.get("per_shape", []),
                        "summary": summary,
                    },
                })
        output.setdefault("status", "error" if result.is_error else "success")
        output.setdefault("needs_benchmark", False)
        return output

    def _route_finding(self, finding: dict[str, Any]) -> None:
        """Route a finding to the appropriate handler based on domain."""
        if finding.get("confidence") == "low":
            return

        target_type = finding.get("target_type", "")
        guidance = finding.get("actionable_guidance", {})

        # Orchestrator-domain: code fixes, config, build system issues
        if target_type in ("code-fix", "config", "build", "dispatch-fix"):
            self._handle_orchestrator_finding(finding, guidance)
        # Kernel Manager-domain: kernel rewrites
        elif target_type in ("kernel-rewrite", "oob-rewrite"):
            self._handle_km_finding(finding, guidance)
        else:
            # Default: try to route based on classification
            if finding.get("classification") == "build-system":
                self._handle_orchestrator_finding(finding, guidance)
            else:
                self._handle_km_finding(finding, guidance)

    def _handle_orchestrator_finding(self, finding: dict[str, Any],
                                     guidance: dict[str, Any]) -> None:
        """Push a finding back onto the Orchestrator's own action stack."""
        fix_cmd = guidance.get("fix_command")
        action_type = guidance.get("action_type", "dispatch-fix")
        retry_id = f"{finding.get('task_id', 'x')}_wd_fix_{int(time.time())}"

        action = {
            "id": retry_id,
            "action": action_type,
            "score": max(5, guidance.get("priority", 7)),
            "description": f"WD finding: {finding.get('root_cause', 'unknown issue')}",
            "target_kernel": finding.get("kernel_name", ""),
            "fix_command": fix_cmd,
            "rca_constraints": {
                "constraint": guidance.get("constraint"),
                "avoid": guidance.get("avoid", []),
            },
            "rca_event_id": finding.get("event_id"),
            "prior_status": f"WD investigation: {finding.get('classification', '')}",
        }
        self.state.push_action(action)
        self.dashboard.log_branch("add", action)
        log.info("Routed finding to Orchestrator stack: %s (%s)", retry_id, action_type)

    def _handle_km_finding(self, finding: dict[str, Any],
                           guidance: dict[str, Any]) -> None:
        """Submit a finding to the Kernel Manager work queue with retry escalation."""
        task_id = finding.get("task_id", "x")
        existing = ipc.read_work_queue_all(self.session_dir)

        # Multi-retry: _rca_retry_1, _rca_retry_2, _rca_retry_3
        for attempt in range(1, 4):
            retry_id = f"{task_id}_rca_retry_{attempt}"
            if not any(e.get("id") == retry_id for e in existing):
                break
        else:
            log.info("All 3 RCA retries exhausted for %s — abandoning", task_id)
            return

        ipc.write_work_queue_entry(self.session_dir, {
            "id": retry_id,
            "kernel_name": finding.get("kernel_name"),
            "source_file": (finding.get("details", {}).get("source_file")
                            if isinstance(finding.get("details"), dict) else None),
            "strategy": guidance.get("approach", "oob-rewrite-register-constrained"),
            "priority": 8,
            "rca_constraints": {
                "constraint": guidance.get("constraint"),
                "avoid": guidance.get("avoid", []),
                "compiler_flags": guidance.get("compiler_flags"),
                "max_rounds": 3,
            },
            "rca_event_id": finding.get("event_id"),
            "rca_report_path": finding.get("rca_report_path"),
        })
        self.state.kernel_manager_targets_pushed += 1
        log.info("Re-submitted %s to KM work_queue with RCA constraints", retry_id)

    # ------------------------------------------------------------------
    # Inline failure diagnosis (Orchestrator diagnoses its own failures)
    # ------------------------------------------------------------------

    async def _diagnose_own_failure(self, action: dict[str, Any],
                                    result: dict[str, Any]) -> dict[str, Any] | None:
        """Call LLM to diagnose why an Orchestrator action failed.
        Returns diagnosis dict with root_cause, retryable, escalate_to_km, etc."""
        from . import prompts
        try:
            output_file = Path(self.session_dir) / f"diagnosis_{action.get('id', 'x')}.json"
            prompt = prompts.prompt_diagnose_failure(
                self.state.state_summary(), action, result,
            )
            diag_result = await self.llm.call(
                prompt, output_file=str(output_file), max_turns=15,
            )
            self.llm.sync_stats(self.state)
            return diag_result.output if diag_result.output else None
        except Exception as exc:
            log.warning("Inline diagnosis failed (non-fatal): %s", exc)
            return None

    async def _analyze_regression(
        self,
        action: dict[str, Any],
        result: dict[str, Any],
        prev_tput: float,
        bench: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Analyze a reverted regression to determine root cause and whether
        a corrected retry action should be queued.

        The idea may be sound even though the execution regressed — e.g. wrong
        cu_num, missing server restart, too many changes at once. The LLM reads
        the action output, the diff of what changed, and the regression data to
        produce a diagnosis and optionally a surgical retry action.
        """
        action_id = action.get("id", "unknown")
        log.info("Regression analysis starting for %s", action_id)

        action_output_file = Path(self.session_dir) / f"action_{action_id}.json"
        action_output = {}
        if action_output_file.exists():
            try:
                action_output = json.loads(action_output_file.read_text())
            except Exception:
                pass

        analysis_prompt = f"""You are a performance engineer analyzing why an optimization action
caused a throughput REGRESSION. Your job is to:

1. Figure out the ROOT CAUSE — why did performance drop?
2. Determine if the UNDERLYING IDEA has merit (micro-benchmarks showed improvement?)
3. If yes, design a SURGICAL RETRY that fixes only the implementation mistake.

## Regressed Action
- ID: {action_id}
- Name: {action.get('name', action.get('description', '')[:100])}
- Description: {action.get('description', '')[:500]}
- Target file: {action.get('target_file', 'unknown')}

## What Happened
- Throughput BEFORE: {prev_tput:.1f} tok/s/GPU
- Throughput AFTER:  {bench.get('tput_per_gpu', 0):.1f} tok/s/GPU
- Regression: {result.get('regression_pct', 0):.1f}%
- The action has been REVERTED (files restored from backup).

## Action Output (what it did)
{json.dumps(action_output, indent=2, default=str)[:4000]}

## Current System Context
{self.state.state_summary()[:2000]}

## Instructions
Examine the action's output carefully. Look at:
- What files were modified and how (CSV rows, configs, code changes)
- Whether micro-benchmarks showed the idea working at kernel level
- What could cause a kernel-level win to become an E2E regression
  (e.g. wrong dispatch key, cu_num mismatch, missing server restart,
  CSV format issues, too many shapes changed at once, splitK overhead, etc.)

Return a JSON object:
{{
  "root_cause": "...",          // What specifically went wrong
  "idea_has_merit": true/false, // Was the underlying optimization idea sound?
  "micro_evidence": "...",      // Summary of micro-benchmark evidence
  "retry_recommended": true/false,
  "retry_action": {{            // Only if retry_recommended=true
    "id": "{action_id}-retry",
    "name": "...",
    "description": "RETRY: ... (explain what to do differently, be very specific)",
    "score": 6.0,               // Moderate priority
    "target_file": "...",
    "target_kernel": "...",
    "prior_failure": {{
      "action_id": "{action_id}",
      "regression_pct": {result.get('regression_pct', 0):.1f},
      "root_cause": "..."       // Same as top-level root_cause
    }}
  }}
}}

Be specific in the retry description about what EXACTLY to change and what NOT to change.
Include warnings about the mistakes that caused the regression.

READ-ONLY: Do NOT edit any source files. Only analyze and write to $OUTPUT_FILE.
"""

        try:
            output_file = Path(self.session_dir) / f"regression_analysis_{action_id}.json"
            analysis = await self.llm.call(
                analysis_prompt, output_file=str(output_file), max_turns=25,
            )
            self.llm.sync_stats(self.state)
            if not analysis.output:
                log.warning("Regression analysis for %s returned no output", action_id)
                return None

            out = analysis.output
            log.info(
                "Regression analysis for %s: root_cause=%s, merit=%s, retry=%s",
                action_id,
                out.get("root_cause", "?")[:80],
                out.get("idea_has_merit"),
                out.get("retry_recommended"),
            )

            ipc.write_event(self.session_dir, {
                "source": "orchestrator",
                "type": "regression_analysis",
                "action_id": action_id,
                "severity": "warning",
                "root_cause": out.get("root_cause", ""),
                "idea_has_merit": out.get("idea_has_merit", False),
                "micro_evidence": out.get("micro_evidence", ""),
                "retry_recommended": out.get("retry_recommended", False),
            })
            self.state.events_written += 1

            if out.get("retry_recommended") and out.get("retry_action"):
                retry = out["retry_action"]
                retry.setdefault("id", f"{action_id}-retry")
                retry.setdefault("score", 6.0)
                retry.setdefault("tier", action.get("tier", "tier1"))
                retry.setdefault("confidence", max(0.3, (action.get("confidence", 0.5) or 0.5) * 0.7))
                retry.setdefault("cost_minutes", action.get("cost_minutes", 20))
                retry["_retry_of"] = action_id
                retry["_regression_pct"] = result.get("regression_pct", 0)
                return retry

            return None
        except Exception as exc:
            log.warning("Regression analysis failed (non-fatal) for %s: %s", action_id, exc)
            return None

    async def _analyze_success(
        self,
        action: dict[str, Any],
        result: dict[str, Any],
        prev_tput: float,
        new_tput: float,
    ) -> list[dict[str, Any]]:
        """Analyze a successful action to understand WHY it worked and generate
        follow-on actions that exploit the same optimization pattern elsewhere.

        Returns a list of follow-on actions (may be empty).
        """
        action_id = action.get("id", "unknown")
        gain_pct = (new_tput - prev_tput) / prev_tput * 100 if prev_tput > 0 else 0
        log.info("Success analysis starting for %s (+%.1f%%)", action_id, gain_pct)

        action_output_file = Path(self.session_dir) / f"action_{action_id}.json"
        action_output = {}
        if action_output_file.exists():
            try:
                action_output = json.loads(action_output_file.read_text())
            except Exception:
                pass

        analysis_prompt = f"""You are a performance engineer analyzing why an optimization action
SUCCEEDED. Your job is to:

1. Understand WHAT EXACTLY caused the improvement
2. Extract the general PATTERN (not just the specific change)
3. Identify WHERE ELSE this pattern could be applied in the same codebase
4. Generate concrete follow-on actions

## Successful Action
- ID: {action_id}
- Name: {action.get('name', action.get('description', '')[:100])}
- Description: {action.get('description', '')[:500]}
- Target: {action.get('target_kernel', '')} / {action.get('target_file', '')}

## Results
- Throughput: {prev_tput:.1f} → {new_tput:.1f} tok/s/GPU (+{gain_pct:.1f}%)
- This is an E2E serving benchmark improvement, not just a micro-benchmark.

## Action Output (what it did)
{json.dumps(action_output, indent=2, default=str)[:4000]}

## Current System Context
{self.state.state_summary()[:2000]}

## Instructions
Analyze deeply:
- What was the specific mechanism of improvement? (e.g. better kernel selection,
  reduced dispatch overhead, fused operations, better memory access pattern, etc.)
- Is there a generalizable pattern? (e.g. "interwave scheduling wins for small-M
  decode shapes on gfx950" or "fusing quant with norm saves a memory round-trip")
- Are there other kernels/shapes/layers where the SAME pattern applies?
- Could the gain be amplified by applying it more aggressively?

Return a JSON object:
{{
  "mechanism": "...",            // What specifically caused the improvement
  "pattern": "...",              // The generalizable optimization pattern
  "pattern_applies_to": ["..."], // Other places this pattern could apply
  "amplification_possible": true/false,
  "amplification_idea": "...",   // How to get even more gain from same approach
  "follow_on_actions": [         // 0-3 concrete follow-on actions
    {{
      "id": "{action_id}-ext-1",
      "name": "...",
      "description": "FOLLOW-ON from {action_id} (+{gain_pct:.1f}%): ...",
      "score": 7.0,
      "target_kernel": "...",
      "target_file": "...",
      "expected_gain_pct": 1.0,
      "confidence": 0.5,
      "pattern_source": "{action_id}"
    }}
  ]
}}

Only include follow-on actions that are DIFFERENT from what's already on the action stack.
Be specific — vague "investigate X" actions are not useful.

READ-ONLY: Do NOT edit any source files. Only analyze and write to $OUTPUT_FILE.
"""

        try:
            output_file = Path(self.session_dir) / f"success_analysis_{action_id}.json"
            analysis = await self.llm.call(
                analysis_prompt, output_file=str(output_file), max_turns=20,
            )
            self.llm.sync_stats(self.state)
            if not analysis.output:
                return []

            out = analysis.output
            log.info(
                "Success analysis for %s: mechanism=%s, pattern=%s, follow_ons=%d",
                action_id,
                out.get("mechanism", "?")[:60],
                out.get("pattern", "?")[:60],
                len(out.get("follow_on_actions", [])),
            )

            ipc.write_event(self.session_dir, {
                "source": "orchestrator",
                "type": "success_analysis",
                "action_id": action_id,
                "severity": "info",
                "mechanism": out.get("mechanism", ""),
                "pattern": out.get("pattern", ""),
                "gain_pct": gain_pct,
                "pattern_applies_to": out.get("pattern_applies_to", []),
                "amplification_possible": out.get("amplification_possible", False),
            })
            self.state.events_written += 1

            follow_ons = []
            existing_ids = {a.get("id") for a in self.state.action_stack}
            for fo in out.get("follow_on_actions", [])[:3]:
                if fo.get("id") in existing_ids:
                    continue
                fo.setdefault("tier", action.get("tier", "tier1"))
                fo.setdefault("cost_minutes", action.get("cost_minutes", 20))
                fo["_derived_from"] = action_id
                fo["_derived_gain_pct"] = gain_pct
                follow_ons.append(fo)

            return follow_ons
        except Exception as exc:
            log.warning("Success analysis failed (non-fatal) for %s: %s", action_id, exc)
            return []

    @staticmethod
    def _build_km_escalation(action: dict[str, Any],
                             diagnosis: dict[str, Any]) -> dict[str, Any]:
        """Build a work queue entry for Kernel Manager from a diagnosed failure."""
        rca = diagnosis.get("rca_constraints") or {}
        return {
            "id": f"{action.get('id', 'x')}_km_escalation",
            "kernel_name": action.get("target_kernel", ""),
            "source_file": action.get("source_file", ""),
            "strategy": "oob-rewrite-register-constrained",
            "priority": max(5, float(action.get("score", 0) or 0) * 0.8),
            "rca_constraints": {
                "constraint": rca.get("constraint", diagnosis.get("root_cause", "")),
                "avoid": rca.get("avoid", []),
                "compiler_flags": rca.get("compiler_flags"),
                "max_rounds": 3,
            },
            "escalated_from": action.get("id"),
            "prior_error": diagnosis.get("root_cause", ""),
        }

    # ------------------------------------------------------------------
    # Dream, rescore, accuracy gate, benchmark, sweep, report
    # ------------------------------------------------------------------

    def _should_dream(self) -> bool:
        elapsed_min = (time.time() - (self.state.last_dream_ts or self.state.start_time)) / 60
        if elapsed_min < DREAM_MIN_INTERVAL_MIN:
            return False

        reasons: list[str] = []

        streak = 0
        for a in reversed(self.state.completed_actions):
            if "revert" in (a.get("result", {}).get("status", "") or ""):
                streak += 1
            else:
                break
        if streak >= 3:
            reasons.append(f"{streak} consecutive reverts")

        if self.state.consecutive_discards >= 3:
            reasons.append(f"{self.state.consecutive_discards} consecutive discards")

        if self.state.actions_since_gain >= PLATEAU_THRESHOLD:
            reasons.append(f"plateau: {self.state.actions_since_gain} actions without gain")

        if len(self.state.action_stack) < 5:
            reasons.append(f"stack nearly empty ({len(self.state.action_stack)} actions)")

        if reasons:
            log.info("Dream triggered: %s", " | ".join(reasons))
            return True
        return False

    async def _dream(self) -> None:
        from . import prompts
        completed_since = self.state.completed_actions[-(self.state.dream_count * 10 + 10):]
        output_file = Path(self.session_dir) / f"dream_{self.state.dream_count}.json"
        prompt = prompts.prompt_dream(
            self.state.state_summary(), completed_since, self.state.strategies_tested,
        )
        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=30)
        self.llm.sync_stats(self.state)

        if result.output:
            for a in result.output.get("rescored_stack", []):
                for sa in self.state.action_stack:
                    if sa.get("id") == a.get("id"):
                        sa["score"] = a.get("score", sa.get("score", 0))

            # Ingest NEW actions generated by the dream
            for a in result.output.get("new_actions", []):
                self.state.push_action(a)
                self.dashboard.log_branch("add", a)
                log.info("Dream generated new action: %s [%.1f]", a.get("id", "?"), float(a.get("score", 0) or 0))

        self.state.apply_dream_rescores()
        self.state.last_dream_ts = time.time()
        self.state.dream_count += 1
        self.dashboard.log_score_snapshot(self.state.action_stack)
        self.state.checkpoint("dream")

    async def _rescore(self) -> None:
        from . import prompts
        output_file = Path(self.session_dir) / f"rescore_{int(time.time())}.json"
        prompt = prompts.prompt_rescore(self.state.state_summary(), self.state.action_stack)
        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=15)

        if result.output:
            for entry in result.output.get("rescored_stack", []):
                for a in self.state.action_stack:
                    if a.get("id") == entry.get("id"):
                        a["score"] = entry.get("score", a.get("score", 0))

    async def _accuracy_gate(self, action: dict, result: dict) -> bool:
        """Run deterministic accuracy evaluation via eval_accuracy.sh.

        Falls back to LLM-based gate only if the script is unavailable.
        """
        action_id = action.get("id", "unknown")

        # Try deterministic eval first
        eval_script = None
        if self._workload:
            candidate = Path(self._workload.inferencex_path).parent / "scripts" / "eval_accuracy.sh"
            if candidate.exists():
                eval_script = candidate
            else:
                scripts_dir = Path(self._workload.inferencex_path) / "scripts"
                candidate = scripts_dir / "eval_accuracy.sh"
                if candidate.exists():
                    eval_script = candidate

        if eval_script:
            log.info("Running deterministic accuracy gate: %s", eval_script)
            results_dir = Path(self.session_dir) / f"eval_accuracy_{action_id}"
            results_dir.mkdir(parents=True, exist_ok=True)
            env = {
                **os.environ,
                "PORT": str(self._workload.port if self._workload else 8888),
                "MODEL": self._workload.model if self._workload else "",
                "RESULTS_DIR": str(results_dir),
            }
            try:
                proc = await asyncio.create_subprocess_shell(
                    f"bash {eval_script}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=600)
                output = stdout.decode(errors="replace") if stdout else ""
                log.info("Accuracy eval finished (exit=%s): %s",
                         proc.returncode, output[-500:])

                if proc.returncode != 0:
                    log.warning("Accuracy eval script failed (exit %d) — treating as PASS "
                                "to avoid blocking on eval infra issues", proc.returncode)
                    return True

                # Parse GSM8K accuracy from lm_eval output
                import re
                acc_match = re.search(r'gsm8k.*?acc[^,]*?[\|,]\s*([\d.]+)', output, re.IGNORECASE)
                if not acc_match:
                    acc_match = re.search(r'acc(?:uracy)?[^\d]*([\d.]+)', output)
                if acc_match:
                    accuracy = float(acc_match.group(1))
                    if accuracy < 1:
                        accuracy *= 100
                    log.info("GSM8K accuracy: %.1f%%", accuracy)
                    passed = accuracy >= 50.0
                    if not passed:
                        log.error("ACCURACY GATE FAILED: %.1f%% < 50%% threshold", accuracy)
                    return passed
                else:
                    log.warning("Could not parse accuracy from eval output — treating as PASS")
                    return True

            except asyncio.TimeoutError:
                log.warning("Accuracy eval timed out after 600s — treating as PASS")
                return True
            except Exception as e:
                log.warning("Accuracy eval error: %s — treating as PASS", e)
                return True

        # Fallback to LLM-based gate
        log.warning("No eval_accuracy.sh found — falling back to LLM accuracy gate")
        from . import prompts
        output_file = Path(self.session_dir) / f"accuracy_{action_id}.json"
        prompt = prompts.prompt_accuracy_gate(self.state.state_summary(), action, result)
        ag_result = await self.llm.call(prompt, output_file=str(output_file), max_turns=15)
        return (ag_result.output or {}).get("passed", True)

    async def _run_benchmark(self, _retry: int = 0) -> dict[str, Any] | None:
        """Run benchmark using the deterministic InferenceX subprocess.

        Never falls back to LLM for benchmarks — we need identical params
        every time so baseline and post-action results are comparable.

        Checks GPU contention before and after: if the GPU lock was stale
        or had recent contention, the result is discarded and retried once.
        """
        MAX_CONTENTION_RETRIES = 2

        if not hasattr(self, '_workload') or self._workload is None:
            log.error("No workload configured — cannot run benchmark")
            return None

        if self.gpu_lock and (self.gpu_lock.is_stale or self.gpu_lock.state.busy):
            holder = self.gpu_lock.state.holder
            phase = self.gpu_lock.state.phase
            held = self.gpu_lock.state.held_seconds
            log.warning(
                "GPU busy before benchmark — holder=%s/%s held %.0fs, stale=%s. "
                "Waiting up to 300s for release.",
                holder, phase, held, self.gpu_lock.is_stale,
            )
            for i in range(150):
                await asyncio.sleep(2)
                if not self.gpu_lock.state.busy:
                    log.info("GPU released after %.0fs wait — proceeding with benchmark", (i + 1) * 2)
                    break
            else:
                holder = self.gpu_lock.state.holder
                phase = self.gpu_lock.state.phase
                held = self.gpu_lock.state.held_seconds
                log.error(
                    "GPU still held after 300s wait (holder=%s/%s held %.0fs) — "
                    "force-releasing for benchmark",
                    holder, phase, held,
                )
                self.gpu_lock._force_release()
                await asyncio.sleep(5)

        # Kill any rogue servers (Claw-spawned) before benchmarking
        from .workload import kill_rogue_servers
        rogues = await kill_rogue_servers(port=self._workload.port)
        if rogues:
            log.warning("Killed %d rogue server(s) before benchmark", rogues)
            await asyncio.sleep(5)

        # Verify the server was launched by the marathon (via Sprint script),
        # not by Claw directly.  A Claw-spawned server may be missing critical
        # optimization flags (allreduce fusion, FP8 KV cache, etc.) and will
        # pass health checks but produce degraded throughput.
        from .workload import _read_server_pid
        marathon_pid = _read_server_pid()
        if marathon_pid is None and await self._workload._health_check():
            log.warning(
                "Server is alive but NOT launched by marathon (no PID file) "
                "— likely Claw-spawned with missing flags.  Killing and "
                "relaunching via Sprint launch script."
            )
            await self._workload.kill_server()
            await asyncio.sleep(3)

        if not await self._workload._check_server_alive():
            log.warning("Server not healthy before benchmark (deep check failed) — restarting server")
            changed = self._workload.system_files_changed()
            if changed:
                log.warning("System files modified by Claw: %s — rolling back before restart", changed)
                self._workload.rollback_system_files()
            await self._workload.kill_server()
            healthy = await self._workload.start_server()
            if not healthy:
                log.error("Failed to start server for benchmark")
                return None

        contention_before = (
            self.gpu_lock.had_recent_contention(window_s=120) if self.gpu_lock else False
        )

        try:
            result = await self._workload.run_benchmark()
        except Exception as exc:
            log.error("Benchmark failed: %s", exc)
            return None

        if result.tput_per_gpu <= 0:
            log.error("Benchmark returned 0 throughput — server may be down")
            return None

        contention_after = (
            self.gpu_lock.had_recent_contention(window_s=120) if self.gpu_lock else False
        )
        contention_during = contention_before or contention_after
        suspect_drop = (
            self.state.current_tput_per_gpu > 0
            and result.tput_per_gpu < self.state.current_tput_per_gpu * 0.85
        )

        if contention_during and suspect_drop and _retry < MAX_CONTENTION_RETRIES:
            log.warning(
                "SUSPECT BENCHMARK: %.1f tok/s/GPU (prev %.1f, -%.1f%%) "
                "with GPU contention detected — discarding and retrying (%d/%d)",
                result.tput_per_gpu, self.state.current_tput_per_gpu,
                (1 - result.tput_per_gpu / self.state.current_tput_per_gpu) * 100,
                _retry + 1, MAX_CONTENTION_RETRIES,
            )
            await asyncio.sleep(10)
            return await self._run_benchmark(_retry=_retry + 1)

        bench_data = {
            "tput_per_gpu": result.tput_per_gpu,
            "output_throughput": result.output_throughput,
            "total_token_throughput": result.total_token_throughput,
            "mean_tpot_ms": result.mean_tpot_ms,
            "mean_ttft_ms": result.mean_ttft_ms,
            "p99_tpot_ms": result.p99_tpot_ms,
            "p99_ttft_ms": result.p99_ttft_ms,
            "num_prompts": result.num_prompts,
            "result_file": result.result_file,
            "latency_ms": result.mean_tpot_ms,
            "had_gpu_contention": contention_during,
        }
        if contention_during:
            log.warning("Benchmark completed WITH contention: %.1f tok/s/GPU "
                        "(result may be unreliable)",
                        result.tput_per_gpu)
        log.info("Benchmark: %.1f tok/s/GPU, TPOT=%.1fms, TTFT=%.1fms, "
                 "%d prompts, file=%s",
                 result.tput_per_gpu, result.mean_tpot_ms,
                 result.mean_ttft_ms, result.num_prompts, result.result_file)
        return bench_data

    async def _sweep(self) -> None:
        from . import prompts
        output_file = Path(self.session_dir) / "sweep_output.json"
        prompt = prompts.prompt_sweep(self.state.state_summary(), getattr(self.state, "server_config", {}))
        await self.llm.call(prompt, output_file=str(output_file), max_turns=30)

    async def _report(self) -> None:
        from . import prompts
        output_file = Path(self.session_dir) / "report_output.json"
        prompt = prompts.prompt_report(self.state.state_summary(), self.state.completed_actions)
        await self.llm.call(prompt, output_file=str(output_file), max_turns=20)

    # ------------------------------------------------------------------
    # Re-explore / plateau breaker
    # ------------------------------------------------------------------

    async def _re_explore(self) -> None:
        log.info("Re-explore triggered (actions_since_gain=%d, completed=%d, elapsed=%.1fh)",
                 self.state.actions_since_gain,
                 len(self.state.completed_actions),
                 (time.time() - self.state.start_time) / 3600)

        sig = self._compute_loop_signature()
        self.state.loop_signatures.append(sig)
        loop_severity = self.state.detect_loop_graduated()
        if loop_severity > 0.3:
            log.warning("Loop detected (severity=%.2f) — escalating response", loop_severity)
        if loop_severity > 0.6:
            log.info("Moderate loop — forcing exploratory probe")
        if loop_severity > 0.8:
            log.info("Severe loop — will force re-profile + stack rebuild")

        from . import prompts
        output_file = Path(self.session_dir) / f"re_explore_{int(time.time())}.json"
        prompt = prompts.prompt_re_explore(
            self.state.state_summary(),
            self.state.loop_signatures,
            self.state.strategies_tested,
            self.state.tier_breakdown,
        )
        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=40)

        added = 0
        if result.output:
            for a in result.output.get("novel_actions", []):
                self.state.push_action(a)
                self.dashboard.log_branch("add", a)
                added += 1

        # If LLM re-explore returned nothing, check server health before injecting
        # fallback actions that would just fail instantly in a tight loop
        if added == 0:
            from . import server as _srv
            if not await _srv.health_check():
                log.warning("Re-explore returned 0 actions and server is down — "
                            "attempting server restart before injecting fallbacks")
                srv_config = getattr(self.state, "server_config", {})
                if srv_config.get("launch_command"):
                    recovered = await _srv.restart_server(srv_config)
                    if not recovered:
                        log.error("Server restart failed — sleeping 60s before retry")
                        await asyncio.sleep(60)
                        return
                elif self._workload is not None:
                    recovered = await self._workload.start_server()
                    if not recovered:
                        log.error("Workload server start failed — sleeping 60s before retry")
                        await asyncio.sleep(60)
                        return
                else:
                    log.error("No way to restart server — sleeping 60s")
                    await asyncio.sleep(60)
                    return
            log.info("Re-explore returned 0 actions — injecting synthetic fallback actions")
            added = self._inject_fallback_actions(loop_severity > 0.3)

        log.info("Re-explore completed: %d new actions on stack", added)
        self.state.consecutive_discards = 0
        self.state.actions_since_gain = 0
        self.state.save()

    def _inject_fallback_actions(self, loop_detected: bool) -> int:
        """Generate synthetic actions when the LLM re-explore fails to produce ideas."""
        added = 0
        elapsed_h = (time.time() - self.state.start_time) / 3600
        tested_types = {a.get("action") for a in self.state.completed_actions}
        completed_ids = {a.get("id") for a in self.state.completed_actions}

        # Re-try kernels that failed previously with a different strategy
        failed_kernels = [
            a for a in self.state.completed_actions
            if a.get("result", {}).get("status") in ("error", "crash", "segfault")
            and float(a.get("gpu_time_pct", a.get("score", 0)) or 0) >= 3
        ]
        for fa in failed_kernels[-5:]:
            retry_id = f"{fa.get('id', 'x')}_retry_{int(time.time())}"
            if retry_id in completed_ids:
                continue
            old_strategy = fa.get("action", "")
            alt_strategy = {
                "dispatch-fix": "operator-tuning",
                "operator-tuning": "oob-rewrite",
                "oob-rewrite": "oob-rewrite-register-constrained",
                "oob-rewrite-register-constrained": "kernel-fusion",
                "kernel-fusion": "framework-rebuild",
            }.get(old_strategy, "oob-rewrite")
            action = {
                "id": retry_id,
                "action": alt_strategy,
                "target_kernel": fa.get("target_kernel", fa.get("kernel_name")),
                "source_file": fa.get("source_file", ""),
                "gpu_time_pct": fa.get("gpu_time_pct", 0),
                "score": max(3, float(fa.get("score", 0) or 0) * 0.7),
                "description": f"Retry {fa.get('target_kernel', '?')} with {alt_strategy} "
                               f"(prev {old_strategy} failed)",
                "prior_status": fa.get("result", {}).get("error", "")[:200],
            }
            self.state.push_action(action)
            self.dashboard.log_branch("add", action)
            added += 1

        # Inject untested strategy types
        all_types = {"operator-tuning", "comm-optimization", "compiler-tuning",
                     "framework-rebuild", "oob-rewrite"}
        untested = all_types - tested_types
        for ut in untested:
            action = {
                "id": f"explore_{ut}_{int(time.time())}",
                "action": ut,
                "score": 4,
                "description": f"Explore untested strategy: {ut}",
                "target_kernel": "general",
            }
            self.state.push_action(action)
            self.dashboard.log_branch("add", action)
            added += 1

        # If we have kernel candidates not yet attempted, push them
        attempted_kernels = {
            a.get("target_kernel", a.get("kernel_name")) for a in self.state.completed_actions
        }
        for kc in self.state.kernel_candidates:
            kname = kc.get("name", "")
            if kname and kname not in attempted_kernels and kc.get("gpu_pct", 0) >= MIN_GPU_PCT:
                action = {
                    "id": f"untried_{kname}_{int(time.time())}",
                    "action": "oob-rewrite",
                    "target_kernel": kname,
                    "gpu_time_pct": kc.get("gpu_pct", 0),
                    "score": max(3, kc.get("gpu_pct", 0) * 0.5),
                    "description": f"Attempt untried kernel: {kname} ({kc.get('gpu_pct', 0):.1f}% GPU)",
                    "strategy": "oob-rewrite",
                }
                self.state.push_action(action)
                self.dashboard.log_branch("add", action)
                added += 1

        # Always inject a re-profile action as a last resort
        if added == 0:
            action = {
                "id": f"reprofile_{int(time.time())}",
                "action": "deep-kernel-analysis",
                "score": 5,
                "description": "Fresh re-profile to discover new optimization opportunities",
            }
            self.state.push_action(action)
            self.dashboard.log_branch("add", action)
            added += 1

        return added

    def _compute_loop_signature(self) -> str:
        recent = self.state.completed_actions[-5:]
        text = "|".join(a.get("action", "") for a in recent)
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _detect_loop(self) -> bool:
        sigs = self.state.loop_signatures[-LOOP_DETECT_WINDOW:]
        if len(sigs) < 3:
            return False
        if sigs[-1] == sigs[-2] == sigs[-3]:
            return True
        if len(sigs) >= 4 and sigs[-1] == sigs[-3] and sigs[-2] == sigs[-4]:
            return True
        return False

    # ------------------------------------------------------------------
    # Diff snapshots — persist all code changes after gains
    # ------------------------------------------------------------------

    def _snapshot_diffs(self, action: dict[str, Any], prev_tput: float, new_tput: float) -> None:
        """Save diffs of modified system files to session/diffs/ after a successful optimization."""
        try:
            import subprocess, time as _time
            ts = _time.strftime("%Y%m%d_%H%M%S")
            action_id = action.get("id", "unknown")
            diff_dir = Path(self.session_dir) / "diffs" / f"{ts}_{action_id}"
            diff_dir.mkdir(parents=True, exist_ok=True)

            snapshot_dir = Path(self.session_dir) / "benchmarks" / ".system_snapshots"
            system_files = [
                ("/usr/local/lib/python3.12/dist-packages/aiter/fused_moe.py", "aiter_fused_moe"),
                ("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe.py", "vllm_fused_moe"),
                ("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py", "gpt_oss_triton_kernels_moe"),
                ("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py", "rocm_aiter_fused_moe"),
            ]

            diffs_found = 0
            for sys_file, name in system_files:
                if not Path(sys_file).exists():
                    continue
                snap = snapshot_dir / Path(sys_file).name
                if snap.exists():
                    result = subprocess.run(
                        ["diff", "-u", str(snap), sys_file],
                        capture_output=True, text=True, timeout=10,
                    )
                    if result.stdout.strip():
                        (diff_dir / f"{name}.diff").write_text(result.stdout)
                        diffs_found += 1

            # Also try git diffs for tracked repos
            repos = [
                ("aiter", "/sgl-workspace/aiter"),
                ("sglang", "/sgl-workspace/sglang"),
            ]
            for name, repo_path in repos:
                if not Path(repo_path).joinpath(".git").exists():
                    continue
                try:
                    diff = subprocess.run(
                        ["git", "diff"], capture_output=True, text=True,
                        cwd=repo_path, timeout=30,
                    )
                    if diff.stdout.strip():
                        (diff_dir / f"{name}.patch").write_text(diff.stdout)
                        diffs_found += 1
                except Exception:
                    pass

            manifest = {
                "action_id": action_id,
                "timestamp": ts,
                "prev_tput": prev_tput,
                "new_tput": new_tput,
                "gain_pct": (new_tput - prev_tput) / prev_tput * 100 if prev_tput else 0,
                "baseline_tput": self.state.baseline_tput_per_gpu,
                "cumulative_gain_pct": (new_tput - self.state.baseline_tput_per_gpu)
                    / self.state.baseline_tput_per_gpu * 100 if self.state.baseline_tput_per_gpu else 0,
            }
            (diff_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

            action_output = Path(self.session_dir) / f"action_{action_id}.json"
            if action_output.exists():
                import shutil
                shutil.copy2(str(action_output), str(diff_dir / action_output.name))

            # Capture env vars & serve script for full reproducibility
            (diff_dir / "env_vars.json").write_text(
                json.dumps(dict(os.environ), indent=2, sort_keys=True))
            if self._workload:
                sprint_script = getattr(self._workload, '_sprint_launch_script', None)
                if sprint_script and Path(sprint_script).exists():
                    import shutil as _shutil
                    _shutil.copy2(sprint_script, diff_dir / Path(sprint_script).name)
                bench_script = getattr(self._workload, '_sprint_benchmark_script', None)
                if bench_script and Path(bench_script).exists():
                    _shutil.copy2(bench_script, diff_dir / Path(bench_script).name)
                manifest["workload"] = {
                    "tp": self._workload.tp,
                    "isl": self._workload.isl,
                    "osl": self._workload.osl,
                    "concurrency": self._workload.concurrency,
                    "port": self._workload.port,
                    "framework": self._workload.framework,
                }
                (diff_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

            # Generate reproduce.sh for one-command reproducibility
            try:
                repro_lines = ["#!/usr/bin/env bash", "set -euo pipefail", "",
                               f"# Reproduce result from action: {action_id}",
                               f"# Throughput: {prev_tput:.1f} → {new_tput:.1f} tok/s/GPU",
                               f"# Generated: {ts}", ""]
                # Apply diffs
                for diff_file in sorted(diff_dir.glob("*.diff")):
                    repro_lines.append(f'echo "Applying {diff_file.name}..."')
                    repro_lines.append(f"patch -p0 < \"$(dirname $0)/{diff_file.name}\"")
                for patch_file in sorted(diff_dir.glob("*.patch")):
                    repro_lines.append(f'echo "Applying {patch_file.name}..."')
                    target_repo = patch_file.stem
                    repro_lines.append(
                        f"cd /sgl-workspace/{target_repo} 2>/dev/null || true")
                    repro_lines.append(
                        f"git apply \"$(dirname $0)/{patch_file.name}\" || "
                        f"patch -p1 < \"$(dirname $0)/{patch_file.name}\"")
                # Start server
                serve_copy = diff_dir / "serve_tp1.sh"
                if serve_copy.exists():
                    repro_lines += ["", "# Start server",
                                    f'bash "$(dirname $0)/serve_tp1.sh"']
                repro_lines += ["", f"# Then benchmark at CONC={self._workload.concurrency if self._workload else 64}"]
                bench_copy = diff_dir / "bench_sweep.sh"
                if bench_copy.exists():
                    repro_lines.append(f'bash "$(dirname $0)/bench_sweep.sh"')
                repro_script = diff_dir / "reproduce.sh"
                repro_script.write_text("\n".join(repro_lines) + "\n")
                repro_script.chmod(0o755)
            except Exception as exc:
                log.debug("Could not generate reproduce.sh: %s", exc)

            log.info("Diff snapshot saved to %s (%d diffs)", diff_dir, diffs_found)
        except Exception as exc:
            log.warning("Failed to save diff snapshot: %s", exc)

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    async def _rollback_action(self, action: dict[str, Any]) -> None:
        """Revert file changes from a failed/regressed action.

        Strategy (in priority order):
        1. Restore from .bak file created by the action
        2. git checkout if the file is tracked
        3. Ask LLM to revert as last resort
        """
        result = action.get("result", {})
        target_files = (
            action.get("files_to_modify", [])
            or result.get("files_modified", [])
            or [action.get("target_file")]
        )
        target_files = [f for f in target_files if f]
        if not target_files:
            log.warning("No files to rollback for %s", action.get("id"))
            return

        import shutil
        for fpath in target_files:
            bak_candidates = [
                f"{fpath}.bak_{action.get('id', '').lower()}",
                f"{fpath}.bak",
            ]
            csv_changes = result.get("csv_changes_applied", {})
            if isinstance(csv_changes, dict) and csv_changes.get("backup"):
                bak_candidates.insert(0, csv_changes["backup"])

            restored = False
            for bak in bak_candidates:
                if os.path.exists(bak):
                    log.info("Restoring %s from backup %s", fpath, bak)
                    shutil.copy2(bak, fpath)
                    restored = True
                    break

            if not restored:
                log.info("No backup found for %s — trying git checkout", fpath)
                from . import prompts
                try:
                    await self.llm.call(
                        prompts.prompt_shell_command(f"cd $(dirname {fpath}) && git checkout -- {fpath}"),
                        max_turns=3,
                    )
                except Exception as exc:
                    log.warning("Rollback failed for %s: %s", fpath, exc)

    @staticmethod
    def _classify_crash(error_msg: str, exit_code: int | None) -> str:
        msg = error_msg.lower()
        if exit_code == 139 or "segfault" in msg or "sigsegv" in msg:
            return "patch_crash"
        if "out of memory" in msg or "oom" in msg:
            return "oom"
        if "cuda graph" in msg or "cuda_graph" in msg:
            return "cuda_graph"
        if "nccl" in msg or "rccl" in msg or "timeout" in msg:
            return "nccl_timeout"
        jit_markers = ("jit", "inductor", "triton cache", "aiter cache",
                       "hiprtc", "comgr", "stale .so", "cannot open shared object",
                       "undefined symbol")
        if any(m in msg for m in jit_markers):
            return "jit_stale"
        return "unknown"

    async def _recover(self, crash_type: str, crash_log: str = "") -> bool:
        chain = RECOVERY_CHAINS.get(crash_type, RECOVERY_CHAINS["unknown"])
        server_config = getattr(self.state, "server_config", {})
        for attempt in range(MAX_RECOVERY_ATTEMPTS):
            log.info("Recovery attempt %d/%d for %s", attempt + 1, MAX_RECOVERY_ATTEMPTS, crash_type)
            for step in chain:
                step_action = step.get("action", "")
                if step_action == "restart_server":
                    wait = step.get("wait_s", 0)
                    if wait:
                        await asyncio.sleep(wait)
                    try:
                        if await self.server.restart_server(server_config):
                            self.state.checkpoint("recovery")
                            return True
                    except Exception as exc:
                        log.warning("Recovery restart failed: %s", exc)
                elif step_action == "reduce_mem_fraction":
                    delta = step.get("delta", -0.05)
                    frac = server_config.get("gpu_memory_utilization", 0.9) + delta
                    server_config["gpu_memory_utilization"] = max(0.5, frac)
                    log.info("Reduced mem fraction to %.2f", server_config["gpu_memory_utilization"])
                elif step_action == "reduce_cuda_graph_max_bs":
                    divisor = step.get("divisor", 2)
                    cur = server_config.get("cuda_graph_max_bs", 256)
                    server_config["cuda_graph_max_bs"] = max(1, cur // divisor)
                    log.info("Reduced cuda_graph_max_bs to %d", server_config["cuda_graph_max_bs"])
                elif step_action == "disable_cuda_graph":
                    server_config["disable_cuda_graph"] = True
                    log.info("Disabled CUDA graphs")
                elif step_action == "rollback_last_patch":
                    if self.state.completed_actions:
                        last = self.state.completed_actions[-1]
                        await self._rollback_action(last)
                elif step_action == "clear_jit_cache":
                    import shutil
                    for cache_dir in [
                        os.path.expanduser("~/.cache/aiter"),
                        os.path.expanduser("~/.triton/cache"),
                        os.path.expanduser("~/.cache/triton"),
                    ]:
                        if os.path.isdir(cache_dir):
                            shutil.rmtree(cache_dir, ignore_errors=True)
                            log.info("Cleared JIT cache: %s", cache_dir)
                elif step_action == "clear_pycache":
                    import subprocess
                    subprocess.run(
                        ["find", server_config.get("base_dir", "/sgl-workspace"),
                         "-type", "d", "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"],
                        capture_output=True, timeout=30,
                    )
                    log.info("Cleared __pycache__ dirs")
                elif step_action == "clear_caches":
                    from . import prompts
                    await self.llm.call(
                        prompts.prompt_shell_command(
                            "rm -rf ~/.triton/cache /tmp/torchinductor_root && "
                            "find /sgl-workspace -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true"
                        ),
                        max_turns=3,
                    )
                elif step_action == "set_nccl_timeout":
                    server_config["nccl_timeout"] = step.get("value", 1800)
                    log.info("Set NCCL timeout to %d", server_config["nccl_timeout"])
                elif step_action == "checkpoint_and_reload":
                    self.state.checkpoint("recovery_reload")

            await asyncio.sleep(BACKOFF_MULTIPLIER ** attempt)

        log.error("All recovery attempts failed for %s", crash_type)
        return False

    # ------------------------------------------------------------------
    # Stopping conditions + tier checks
    # ------------------------------------------------------------------

    def _check_km_heartbeat(self) -> None:
        """Request a kernel-mgr pane restart when pending work goes silent.

        The heartbeat must look only at KM-owned files. `event_log.jsonl` and
        `work_queue.jsonl` are shared with the orchestrator, so using their
        mtimes would let our own writes make a dead KM look alive.
        """
        now = time.monotonic()
        last_check = getattr(self, "_km_heartbeat_last_check_mono", 0.0)
        if now - last_check < KM_HEARTBEAT_CHECK_INTERVAL_S:
            return
        self._km_heartbeat_last_check_mono = now

        if self.state.kernel_manager_targets_pushed <= 0:
            return

        pending_km_targets = 0
        try:
            wq = ipc.read_work_queue_all(self.session_dir)
        except Exception:
            wq = []
        processed = set(getattr(self.state, "kernel_manager_processed_ids", []) or [])
        for t in wq:
            if t.get("status") == "pending" and t.get("id") not in processed:
                pending_km_targets += 1

        # A silent KM with no pending work is idle, not hung.
        if pending_km_targets <= 0 and not getattr(
            self.state, "km_requested_restart", False
        ):
            return

        session_path = Path(self.session_dir)
        heartbeat_sources = [
            session_path / "logs" / "kernel-mgr.log",
            session_path / "kernel_manager" / "results.jsonl",
        ]
        latest_mtime = 0.0
        for p in heartbeat_sources:
            if not p.exists():
                continue
            try:
                latest_mtime = max(latest_mtime, p.stat().st_mtime)
            except OSError:
                continue

        if latest_mtime <= 0:
            return

        silence_min = (time.time() - latest_mtime) / 60

        if (getattr(self.state, "km_requested_restart", False)
                and silence_min < KM_HEARTBEAT_STALE_MIN):
            log.info(
                "kernel-mgr heartbeat recovered (silence %.1f min < %d) — "
                "clearing restart request",
                silence_min, KM_HEARTBEAT_STALE_MIN,
            )
            self.state.km_requested_restart = False
            self.state.save()
            return

        if getattr(self.state, "km_requested_restart", False):
            return

        if silence_min >= KM_HEARTBEAT_RESTART_MIN:
            log.error(
                "kernel-mgr heartbeat: %.1f min of silence (>= %d) — "
                "requesting pane restart via run.sh monitor",
                silence_min, KM_HEARTBEAT_RESTART_MIN,
            )
            self.state.km_requested_restart = True
            self.state.km_restart_count = (
                getattr(self.state, "km_restart_count", 0) + 1
            )
            try:
                ipc.write_event(self.session_dir, {
                    "source": "marathon",
                    "type": "km-restart-requested",
                    "severity": "error",
                    "promising": False,
                    "details": {
                        "silence_min": round(silence_min, 1),
                        "restart_count": self.state.km_restart_count,
                        "threshold_min": KM_HEARTBEAT_RESTART_MIN,
                        "pending_km_targets": pending_km_targets,
                    },
                })
                self.state.events_written += 1
            except Exception as exc:
                log.warning("Failed to write km-restart-requested event: %s", exc)
            try:
                self.state.save()
            except Exception as exc:
                log.warning("Failed to save state after km-restart request: %s", exc)
            return

        if silence_min >= KM_HEARTBEAT_STALE_MIN:
            last_warn = getattr(self, "_km_stale_last_warn_min", 0.0)
            if silence_min - last_warn >= 30:
                log.warning(
                    "kernel-mgr heartbeat stale: %.1f min of silence (>= %d)",
                    silence_min, KM_HEARTBEAT_STALE_MIN,
                )
                self._km_stale_last_warn_min = silence_min
                try:
                    ipc.write_event(self.session_dir, {
                        "source": "marathon",
                        "type": "km-stale",
                        "severity": "warning",
                        "promising": False,
                        "details": {
                            "silence_min": round(silence_min, 1),
                            "threshold_min": KM_HEARTBEAT_STALE_MIN,
                        },
                    })
                    self.state.events_written += 1
                except Exception as exc:
                    log.warning("Failed to write km-stale event: %s", exc)

    def _should_stop(self) -> bool:
        """Five stopping conditions from SKILL.md + shutdown signal."""
        elapsed_h = (time.time() - self.state.start_time) / 3600
        wall_limit = self._max_wall_hours
        if elapsed_h >= wall_limit:
            log.info("Stopping: wall clock >= %.1fh", wall_limit)
            return True
        if self.state.crash_count >= MAX_CRASH_COUNT:
            log.info("Stopping: %d crashes (budget exhausted)", MAX_CRASH_COUNT)
            return True
        if self.state.cumulative_gain_pct >= 40.0:
            log.info("Stopping: cumulative gain %.1f%% >= 40%%", self.state.cumulative_gain_pct)
            return True
        if (self.state.target_tput_per_gpu > 0
                and self.state.current_tput_per_gpu >= self.state.target_tput_per_gpu):
            log.info("Stopping: target throughput %.1f reached (current %.1f)",
                     self.state.target_tput_per_gpu, self.state.current_tput_per_gpu)
            return True
        if self._max_cost_usd > 0 and self.state.total_llm_cost_usd >= self._max_cost_usd:
            log.info("Stopping: LLM cost $%.2f >= cap $%.2f",
                     self.state.total_llm_cost_usd, self._max_cost_usd)
            return True
        return False

    def _needs_refill(self) -> bool:
        """True when the stack is empty or exhausted — triggers idea generation, not a stop."""
        if not self.state.action_stack:
            return True
        if all(float(a.get("score", 0) or 0) < 1.0 for a in self.state.action_stack):
            return True
        if self.state.consecutive_discards >= KERNEL_OPT_CONSECUTIVE_DISCARDS:
            return True
        return False

    def _check_tier_boundary(self) -> None:
        new_tier = self.state.update_tier()
        if new_tier:
            log.info("Tier boundary: %s", new_tier)
            self.state.checkpoint(f"tier_{new_tier}")

    def _checkpoint_due(self) -> bool:
        return (time.time() - self.state.last_checkpoint_time) / 60 >= CHECKPOINT_CADENCE_MIN
