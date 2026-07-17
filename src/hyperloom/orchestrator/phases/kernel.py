# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KERNEL_AGENT phase handler: bf16-dense-GEMM fallback, GEAK e2e run,
GEMM-tuning keep/promote, and watermark-roofline gating."""

from __future__ import annotations
import asyncio
import json
import logging as _logging
import os
import signal
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from . import machine_state as _phase_state
from ..state.optimization_journal import (
    KIND_GEMM_TUNING,
    OUTCOME_KEEP,
    JournalEntry,
)
from ..bus.message_bus import Message
from ..loop.coordinator_helpers import (
    _GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT,
    _resolve_roofline_watermark_ratio,
    _resolve_serving_fidelity,
    _split_env_and_flags,
    effective_closing_grace_sec,
)
from .base import PhaseHandler

log = _logging.getLogger(__name__)


class KernelPhase(PhaseHandler):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

    async def _maybe_reprofile_for_kernel(self) -> None:
        """Reprofile inline when projected tput diverges from the last measured trace, so GEAK targets the live bottleneck."""
        before = self._last_measured_roofline_tput()
        cur = self._current_tput_from_validated_gain()
        if cur <= 0:
            return
        # With a measured trace, reprofile only on a material change.
        if before > 0 and abs(cur - before) / before < self._REPROFILE_CHANGE_TOL:
            return
        stack_len = int(getattr(self.shared_state, "cumulative_gain_validated_stack_len", 0) or 0)
        try:
            await self.sub.run_task(
                await self._enqueue_internal_analysis_task(reason=f"kernel_entry_g{stack_len}")
            )
        except Exception:  # noqa: BLE001 — never block GEAK on a reprofile failure
            log.exception("kernel-entry reprofile failed; GEAK proceeds on existing snapshot")
            return
        # Advance the anchor only when a new snapshot actually landed.
        after = self._last_measured_roofline_tput()
        if after > 0 and after != before:
            self.shared_state.last_roofline_tput = after
            self.shared_state.save(self.session_dir)
        else:
            log.warning("kernel-entry reprofile produced no new snapshot; GEAK targets existing trace")

    def _geak_enabled(self) -> bool:
        """Whether the KERNEL_AGENT phase is delegated to the GEAK e2e optimizer.

        The source of truth is the kernel backend order
        (``KERNEL_OPT_BACKEND_ORDER`` / ``KERNEL_OPT_BACKENDS``): when ``geak``
        appears there it owns the whole phase. The ``kernel_optimizer`` state
        field is the persisted record used as a resume fallback.
        """
        from ..kernel.request_handlers import geak_selected

        if geak_selected():
            return True
        return (
            str(getattr(self.shared_state, "kernel_optimizer", "") or "")
            .strip()
            .lower()
            == "geak"
        )

    async def _on_enter_kernel(self, *, from_phase: str) -> None:
        """Run deterministic KERNEL-entry setup before LLM kernel work (FP8 GEMM tuning gate).

        Args:
            from_phase: The phase being left, used only for logging.
        """
        if not self._kernel_enabled():
            log.info(
                "KERNEL entry hook fired with kernel_enabled=False (from=%s)",
                from_phase or "<unknown>",
            )
            return
        if self._geak_enabled():
            # GEAK owns the whole KERNEL_AGENT phase: one in-process e2e run
            # seeded with the EXPLORE best config, then hand straight to SWEEP.
            await self._run_geak_kernel_phase(from_phase=from_phase)
            return
        if not self._gemm_tuning_required_before_kernel_opt():
            # No GEMM tuning here: refresh the snapshot before the LLM drives GEAK.
            await self._maybe_reprofile_for_kernel()
            return

        # Refresh the snapshot before GEMM tuning targets the bottleneck.
        await self._maybe_reprofile_for_kernel()
        log.info(
            "KERNEL entry: running GEMM tuning before source-level kernel_opt",
        )
        self._record_phase_entry_evidence(
            gemm_tuning={"status": "running", "source": "kernel_entry_auto"},
        )
        run_gemm_tuning_handler = None
        try:
            from ..kernel.request_handlers import run_gemm_tuning_handler

            if self._bf16_dense_gemm_fallback_pending():
                log.info(
                    "KERNEL entry: resuming pending bf16 dense GEMM fallback "
                    "after prior forge fp8 no-candidate result"
                )
                result = await self._run_bf16_dense_gemm_fallback(
                    run_gemm_tuning_handler
                )
            else:
                result = await run_gemm_tuning_handler(
                    {
                        "task_id": "kernel_entry_gemm_tuning",
                        "reason": "kernel_entry_auto",
                    },
                    session_dir=self.session_dir,
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("KERNEL entry GEMM tuning failed")
            result = {
                "status": "failed",
                "decision": "REVERT",
                "error_class": exc.__class__.__name__,
                "error": repr(exc),
            }
        await self._handle_gemm_tuning_result(result)

        if (
            run_gemm_tuning_handler is not None
            and self._should_run_bf16_dense_gemm_fallback(result)
            and str(result.get("decision") or "").strip().upper() != "KEEP"
        ):
            log.info(
                "KERNEL entry: forge fp8 GEMM tuning found no candidate; "
                "trying bf16 dense fallback"
            )
            result = await self._run_bf16_dense_gemm_fallback(
                run_gemm_tuning_handler
            )
            await self._handle_gemm_tuning_result(result)

        status = str(result.get("status") or "unknown")
        await self.bus.append_and_seq(
            Message.new(
                "kernel_agent",
                "orchestration",
                "response",
                {
                    "in_reply_to": "",
                    "kind": "run_gemm_tuning_done",
                    "status": status,
                    "result": result,
                    "source": "kernel_entry_auto",
                },
                priority=1,
            )
        )
        self._record_phase_entry_evidence(
            gemm_tuning={
                "status": "done" if status in {"ok", "complete", "succeeded"} else status,
                "source": "kernel_entry_auto",
                "best_speedup": result.get("best_speedup"),
                "tuned_file": result.get("tuned_file"),
            },
        )
        # Capture explore + GEMM-tuning gains before inline GEAK.
        await self._maybe_reprofile_for_kernel()
        # Autonomous kernel fusion (forge-fusion) between GEMM tuning and generic
        # kernel-opt; gated + non-blocking (a failure falls through to kernel-opt).
        if self._fusion_required_before_kernel_opt():
            await self._run_forge_fusion_after_gemm()
            await self._maybe_reprofile_for_kernel()
        if self._should_continue_kernel_after_gemm():
            await self._run_kernel_opt_after_gemm()

    async def _run_bf16_dense_gemm_fallback(
        self,
        run_gemm_tuning_handler: Callable[..., Any],
    ) -> dict[str, Any]:
        """Run the single bf16 dense fallback and stamp retry provenance."""
        payload = {
            "task_id": "kernel_entry_gemm_tuning_bf16_fallback",
            "reason": "fp8_no_improvement_bf16_fallback",
            "precision": "bf16",
            "tuner": "sglang_dense_bf16",
        }
        try:
            result = await run_gemm_tuning_handler(
                payload,
                session_dir=self.session_dir,
            )
            if not isinstance(result, dict):
                result = {
                    "status": "failed",
                    "decision": "REVERT",
                    "error": "non-dict bf16 fallback result",
                }
        except Exception as exc:  # noqa: BLE001
            log.exception("KERNEL entry GEMM bf16 fallback failed")
            result = {
                "status": "failed",
                "decision": "REVERT",
                "error_class": exc.__class__.__name__,
                "error": repr(exc),
            }
        result.setdefault("task_id", payload["task_id"])
        result.setdefault("reason", payload["reason"])
        result.setdefault("source", payload["reason"])
        result.setdefault("backend", "forge")
        result.setdefault("precision", "bf16")
        result.setdefault("framework", getattr(self.shared_state, "framework", ""))
        return result

    def _should_run_bf16_dense_gemm_fallback(self, result: dict[str, Any]) -> bool:
        """Return True when a forge fp8 run should try bf16 dense GEMM tuning.

        Makes the ``sglang_dense_bf16`` fallback deterministic when the fp8 tuner
        produced no E2E-validatable candidate.
        """
        if not isinstance(result, dict):
            return False
        if str(result.get("backend") or "").strip().lower() != "forge":
            return False
        if str(result.get("precision") or "").strip().lower() != "fp8":
            return False
        framework = str(
            result.get("framework") or getattr(self.shared_state, "framework", "") or ""
        ).strip().lower()
        if framework != "sglang":
            return False
        if str(result.get("micro_decision") or "").strip().lower() != "no_improvement":
            return False
        if result.get("recommended_env") or result.get("extra_envs"):
            return False
        for tuner in result.get("tuners_run") or []:
            if not isinstance(tuner, dict):
                continue
            if str(tuner.get("status") or "").strip().lower() != "ok":
                continue
            try:
                improved = int(tuner.get("improved_shapes") or 0)
            except (TypeError, ValueError):
                improved = 0
            if improved > 0 and str(tuner.get("env_var") or "").strip() and str(
                tuner.get("env_value") or ""
            ).strip():
                return False
        return True

    def _bf16_dense_gemm_fallback_pending(self) -> bool:
        """Return True when a recorded fp8 no-op still needs its bf16 retry."""
        last = getattr(self.shared_state, "last_gemm_tuning", {}) or {}
        return (
            self._should_run_bf16_dense_gemm_fallback(last)
            and not self._bf16_dense_gemm_fallback_attempted()
        )

    def _bf16_dense_gemm_fallback_attempted(self) -> bool:
        """Detect whether the bf16 dense fallback has already been attempted."""
        attempts: list[Any] = []
        last = getattr(self.shared_state, "last_gemm_tuning", {}) or {}
        if isinstance(last, dict):
            attempts.append(last)
        attempts.extend(getattr(self.shared_state, "gemm_tuning_attempts", None) or [])
        return any(
            self._is_bf16_dense_gemm_fallback_attempt(entry)
            for entry in attempts
            if isinstance(entry, dict)
        )

    @staticmethod
    def _is_bf16_dense_gemm_fallback_attempt(entry: dict[str, Any]) -> bool:
        """Identify the fallback attempt across old and newly stamped records."""
        markers = {
            "kernel_entry_gemm_tuning_bf16_fallback",
            "fp8_no_improvement_bf16_fallback",
        }
        for key in ("task_id", "reason", "source"):
            if str(entry.get(key) or "").strip() in markers:
                return True
        if "kernel_entry_gemm_tuning_bf16_fallback" in str(
            entry.get("workspace") or ""
        ):
            return True
        if str(entry.get("precision") or "").strip().lower() != "bf16":
            return False
        if str(entry.get("tuner") or "").strip() == "sglang_dense_bf16":
            return True
        for tuner in entry.get("tuners_run") or []:
            if not isinstance(tuner, dict):
                continue
            if str(tuner.get("tuner") or "").strip() == "sglang_dense_bf16":
                return True
        return False

    @staticmethod
    def _resolve_bench_protocol(recipe_path: str) -> dict[str, Any]:
        """Extract Hyperloom's bench measurement protocol for the GEAK handoff.

        Reads the materialized baseline recipe's ``benchmark.envs`` (falling back
        to the process env) and returns only the keys that resolve, so absent
        values leave GEAK on its standalone defaults. Never raises.
        """
        envs: dict[str, Any] = {}
        try:
            import yaml

            if recipe_path and Path(recipe_path).is_file():
                cfg = yaml.safe_load(Path(recipe_path).read_text(encoding="utf-8")) or {}
                envs = ((cfg.get("benchmark") or {}).get("envs")) or {}
        except Exception:  # noqa: BLE001
            log.warning("bench_protocol: could not read recipe %r", recipe_path,
                        exc_info=True)
            envs = {}

        def _pick(key: str, cast: Callable[[str], Any]) -> Any:
            raw = envs.get(key)
            if raw is None or str(raw).strip() == "":
                raw = os.environ.get(key, "")
            raw = str(raw).strip()
            if not raw:
                return None
            try:
                return cast(raw)
            except (TypeError, ValueError):
                return None

        protocol: dict[str, Any] = {}
        for proto_key, env_key, cast in (
            ("random_range_ratio", "RANDOM_RANGE_RATIO", float),
            ("num_prompts", "NUM_PROMPTS", int),
            ("num_warmups", "NUM_WARMUPS", int),
            ("seed", "SEED", int),
        ):
            val = _pick(env_key, cast)
            if val is not None:
                protocol[proto_key] = val
        return protocol

    def _geak_timeouts(self) -> tuple[int, int, bool]:
        """Resolve the GEAK e2e timeouts from the live run budget.

        The KERNEL_AGENT phase-entry hook runs GEAK synchronously, so the run is
        capped to always finish with at least the closing-grace window left, and
        the runner's own budget is shrunk by a safety margin on top of that.

        Returns:
            tuple[int, int, bool]: ``(runner_timeout_s, kill_timeout_s,
            budget_known)``. ``runner_timeout_s`` is passed to the runner as its
            own e2e budget; ``kill_timeout_s`` is the hard subprocess kill
            (always ≤ remaining − closing_grace so the closing report can run).
            ``budget_known`` is ``False`` only when no run deadline is set
            (e.g. a unit test invoking the hook directly), where the env default
            is used verbatim.
        """
        # Standalone fallback ONLY: the 12h (43200s) default applies when no run
        # deadline is set (budget_known=False). A Hyperloom-driven run sources the
        # budget from the live deadline / phase allocation instead.
        env_default_timeout = int(os.environ.get("GEAK_E2E_TIMEOUT_S", "43200"))
        deadline = self._run_deadline
        if deadline is None:
            return env_default_timeout, env_default_timeout + 600, False
        remaining = deadline - time.monotonic()
        grace = effective_closing_grace_sec(
            float(getattr(self.shared_state, "max_minutes", 0) or 0), None,
        )
        margin = float(os.environ.get("GEAK_BUDGET_MARGIN_S", "300"))
        # Reserve the closing window: kill the subprocess with at least ``grace`` left.
        kill_budget = remaining - grace
        # Also honour the KERNEL_AGENT phase's own wall-clock budget:
        # cap by min(session, kernel_phase).
        phase_rem = _phase_state.phase_budget_remaining_seconds(
            self.shared_state, budget_pct=self._phase_budget_pct,
        )
        if phase_rem is not None:
            kill_budget = min(kill_budget, float(phase_rem))
        # The runner self-stops ``margin`` before the hard subprocess kill, which
        # reserves the closing-grace window.
        kill_timeout = int(max(0.0, kill_budget))
        runner_timeout = int(max(0.0, kill_budget - margin))
        return runner_timeout, kill_timeout, True

    async def _run_geak_kernel_phase(self, *, from_phase: str) -> None:
        """Delegate the KERNEL_AGENT phase to GEAK (one whole-pipeline e2e run).

        Builds a handoff from the EXPLORE best config, runs the GEAK
        runner out-of-process (it owns all Claude-SDK / Workflow detail),
        records the optimized launch/bench scripts + throughput into state, then
        signals SWEEP via the ``skip_to_sweep`` escalate hint.
        """
        state = self.shared_state
        cb = state.current_best or {}
        accepted_flags = str(cb.get("extra_server_args") or "")
        extra_envs = cb.get("extra_envs") or {}
        accepted_env = " ".join(f"{k}={v}" for k, v in dict(extra_envs).items())
        workload = {
            "isl": int(getattr(state, "isl", 0) or int(os.environ.get("ISL", "1024"))),
            "osl": int(getattr(state, "osl", 0) or int(os.environ.get("OSL", "1024"))),
            "conc": int(getattr(state, "conc", 0) or int(os.environ.get("CONC", "64"))),
        }
        # Forward the SAME bench knobs Hyperloom benched with so GEAK's internal
        # e2e measures identically; source = the baseline recipe's benchmark.envs
        # (process-env fallback). Only resolved keys are sent.
        bench_protocol = self._resolve_bench_protocol(
            str(getattr(state, "baseline_config_path", "") or "")
        )
        # Serving-launch fidelity: forward the SAME max-model-len / gpu-mem-util
        # the baseline served with so GEAK launches the identical engine and its
        # baseline matches raw_baseline_tput. Resolver parses these from the raw
        # baseline server-args (dedicated state.max_model_len wins; env last).
        try:
            from ..kernel.roofline_ceiling import read_baseline_server_args

            _baseline_srv_args = read_baseline_server_args(state) or ""
        except Exception:  # noqa: BLE001 — accessor is best-effort
            _baseline_srv_args = ""
        _serving_fidelity = _resolve_serving_fidelity(
            baseline_server_args=_baseline_srv_args,
            state_max_model_len=int(getattr(state, "max_model_len", 0) or 0),
        )

        handoff = {
            # v2 adds ``baseline_env_spec`` (the full layered env of current_best);
            # v1-only consumers ignore it and degrade to the flags/env-only baseline.
            "schema_version": 2,
            "model_path": str(getattr(state, "model_path", "") or os.environ.get("MODEL_PATH", "")),
            "framework": str(os.environ.get("FRAMEWORK", "") or "sglang"),
            "gpu_type": str(getattr(state, "gpu_type", "") or os.environ.get("GPU_TYPE", "")),
            "tp": int(os.environ.get("TP", "1") or 1),
            "workload": workload,
            "accepted_flags": accepted_flags,
            "accepted_env": accepted_env,
            "launch_recipe": str(getattr(state, "baseline_config_path", "") or ""),
            "raw_baseline_tput": float(getattr(state, "baseline_tput", 0.0) or 0.0),
            # Orchestrator throughput of the SAME config GEAK seeds its baseline
            # with, so run_e2e can compute a pure measurement divergence. 0.0 =>
            # no accepted config yet (falls back to raw baseline downstream).
            "orchestrator_best_tput_same_config": float(
                (state.current_best or {}).get("tput") or 0.0
            ) if isinstance(getattr(state, "current_best", None), dict) else 0.0,
            # Serving-launch fidelity (both optional; unset => GEAK adapter default).
            "max_model_len": int(getattr(state, "max_model_len", 0) or int(os.environ.get("MAX_MODEL_LEN", "0") or 0)),
            "mem_fraction": float(getattr(state, "mem_fraction", 0.0) or float(os.environ.get("GPU_MEMORY_UTILIZATION", "0") or 0.0)),
            "exp_root": str(self.session_dir / "geak"),
            # Macro-cycle-scoped eval_dir so a same-cycle resume reuses the
            # in-progress on-disk artifacts while a new cycle gets a fresh dir.
            "eval_dir": str(
                self.session_dir
                / "geak"
                / f"e2e_cycle{int(getattr(state, 'macro_cycle', 0) or 0)}"
            ),
            # Align GEAK's bench CLIENT to Hyperloom's exact one so final/sweep
            # numbers are cross-harness comparable.
            "bench_client": "auto",
            "inferencex_path": str(os.environ.get("INFERENCEX_PATH", "")),
            # Pin the serving GPU set: explicit visibility mask, else 0..tp-1.
            "gpu_ids": (
                os.environ.get("HIP_VISIBLE_DEVICES")
                or os.environ.get("CUDA_VISIBLE_DEVICES")
                or ",".join(str(i) for i in range(int(os.environ.get("TP", "1") or 1)))
            ),
        }
        if bench_protocol:
            handoff["bench_protocol"] = bench_protocol
        # Only forward resolved fidelity knobs; absence => GEAK adapter default.
        handoff.update(_serving_fidelity)
        # Full layered environment of current_best so GEAK's baseline ref ==
        # orchestrator best (config + source-patch snapshots + overlay).
        try:
            handoff["baseline_env_spec"] = self.build_env_spec()
        except Exception:  # noqa: BLE001 — env_spec is additive; never block handoff
            log.exception("geak: build_env_spec failed; handoff stays v1-compatible")

        out_dir = self.session_dir / "geak"
        out_dir.mkdir(parents=True, exist_ok=True)
        handoff_path = out_dir / "handoff.json"
        handoff_path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")

        from ..kernel.request_handlers import _kernel_agent_tool_path

        def _read_geak_result(path: Path) -> dict[str, Any]:
            if not path.is_file():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}

        def _promote_recovered_result(
            result: dict[str, Any],
            *,
            recovered_from: str,
            runner_timeout_s: int | None = None,
        ) -> None:
            state.geak_result = result
            # Rebench-first: record the recovered win as an UNVALIDATED candidate;
            # the caller enqueues the main-flow rebench that writes the headline.
            self._record_geak_candidate(result)
            self._record_geak_kernel_journey(result)
            evidence = {
                "status": result.get("status"),
                "throughput_speedup": result.get("throughput_speedup"),
                "final_throughput_tok_s": result.get("final_throughput_tok_s"),
                "eval_dir": result.get("eval_dir"),
                "report_path": result.get("report_path"),
                "recovered_from": recovered_from,
            }
            if runner_timeout_s is not None:
                evidence["runner_timeout_s"] = runner_timeout_s
            self._record_phase_entry_evidence(geak=evidence)
            # Set the wind-down hint BEFORE the durable save (it is in-memory only).
            state.set_pending_escalate_hint(_phase_state.ESCALATE_HINT_SKIP_TO_SWEEP)
            state.save(self.session_dir)

        def _finish_skip(result: dict[str, Any]) -> None:
            """Record a (failed/skipped) GEAK outcome + wind down to SWEEP.

            Always records the normalized outcome into ``geak_result``,
            mirrors the failure reason onto the phase-entry evidence (so the
            session-breakdown surfaces WHY the e2e run did not land), then sets
            the ``skip_to_sweep`` hint so the coordinator never deadlocks.
            """
            state.geak_result = result
            self._record_phase_entry_evidence(geak={
                "status": result.get("status"),
                "error_class": result.get("error_class"),
                "error": (str(result.get("error") or "")[:500] or None),
            })
            # Persist the wind-down hint durably.
            state.set_pending_escalate_hint(_phase_state.ESCALATE_HINT_SKIP_TO_SWEEP)
            state.save(self.session_dir)

        # Crash-recovery: a validated result.json written before a coordinator
        # crash is promoted on resume, guarded by ``_geak_win_already_recorded``
        # so a prior cycle's result.json does not short-circuit a fresh entry.
        result_path = out_dir / "result.json"
        recovered = _read_geak_result(result_path)
        if (
            recovered.get("status") == "ok"
            and not self._geak_win_already_recorded()
        ):
            log.info(
                "GEAK result.json exists but state has no recorded win "
                "(crash before handback); promoting recovered result."
            )
            _promote_recovered_result(recovered, recovered_from="existing_result_json")
            if recovered.get("status") == "ok":
                try:
                    await self._enqueue_internal_stack_rebench(
                        reason="geak_e2e_win_recovered"
                    )
                except Exception:  # noqa: BLE001 - defensive
                    log.exception("geak: enqueue rebench for recovered result failed")
            return

        try:
            runner = _kernel_agent_tool_path("backends/geak_runner.py")
        except Exception as exc:  # noqa: BLE001
            log.exception("GEAK runner not resolvable; skipping KERNEL")
            _finish_skip({"status": "error", "error_class": "runner_not_found",
                          "error": repr(exc)})
            return

        # Budget-aware timeouts: shrink to the remaining run deadline and always
        # reserve the closing-grace window.
        runner_timeout, kill_timeout, budget_known = self._geak_timeouts()
        min_run = int(os.environ.get("GEAK_MIN_RUN_S", "600"))
        if budget_known and runner_timeout < min_run:
            log.warning(
                "GEAK: only %ds budget remains (< min %ds); skipping e2e "
                "and winding down to SWEEP so the closing report runs in time.",
                runner_timeout, min_run,
            )
            _finish_skip({
                "status": "skipped",
                "error_class": "insufficient_budget",
                "error": (f"only {runner_timeout}s of KERNEL budget remained "
                          f"(< min {min_run}s); skipped to protect the closing "
                          f"report window"),
                "runner_timeout_s": runner_timeout,
            })
            return

        cmd = ["python3", str(runner), str(handoff_path), str(out_dir),
               "--timeout-s", str(runner_timeout)]
        log.info("KERNEL entry: delegating to GEAK e2e (from=%s) "
                 "runner_timeout=%ds kill_timeout=%ds budget_known=%s cmd=%s",
                 from_phase or "<unknown>", runner_timeout, kill_timeout,
                 budget_known, " ".join(cmd))

        # Run in its own process group so a timeout can SIGTERM the whole
        # runner -> run_e2e -> vllm/node tree (grace to flush result.json), then
        # SIGKILL, instead of orphaning run_e2e + its servers.
        term_grace = int(os.environ.get("GEAK_TERM_GRACE_S", "180"))

        def _run() -> subprocess.CompletedProcess:
            p = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=dict(os.environ), start_new_session=True,
            )

            def _killpg(sig: int) -> None:
                try:
                    os.killpg(os.getpgid(p.pid), sig)
                except (ProcessLookupError, PermissionError):
                    # Process already exited; nothing to signal.
                    pass

            try:
                out, err = p.communicate(timeout=kill_timeout)
            except subprocess.TimeoutExpired:
                _killpg(signal.SIGTERM)
                try:
                    out, err = p.communicate(timeout=term_grace)
                except subprocess.TimeoutExpired:
                    _killpg(signal.SIGKILL)
                    out, err = p.communicate()
                raise subprocess.TimeoutExpired(
                    cmd, kill_timeout, output=out, stderr=err,
                )
            return subprocess.CompletedProcess(cmd, p.returncode, out, err)

        try:
            proc = await asyncio.to_thread(_run)
            stderr_tail = (proc.stderr or "")[-2000:]
            if proc.returncode != 0:
                log.warning("GEAK runner rc=%s: %s", proc.returncode, stderr_tail)
        except subprocess.TimeoutExpired:
            log.warning("GEAK runner exceeded kill_timeout=%ds; SIGTERM'd "
                        "to let it flush, then reclaimed the closing window",
                        kill_timeout)
            # The graceful SIGTERM gives run_e2e a window to flush result.json;
            # keep a real win instead of discarding the phase as a timeout.
            recovered = _read_geak_result(result_path)
            if recovered.get("status") == "ok":
                log.info("GEAK flushed an OK result.json under SIGTERM "
                         "grace; promoting the recovered win despite the cap.")
                _promote_recovered_result(
                    recovered,
                    recovered_from="sigterm_flushed_result_json",
                    runner_timeout_s=runner_timeout,
                )
                # Rebench-first: enqueue the main-flow rebench (candidate stays
                # pending if a budget cap prevents it from running).
                try:
                    await self._enqueue_internal_stack_rebench(
                        reason="geak_e2e_win_sigterm_recovered"
                    )
                except Exception:  # noqa: BLE001 - defensive
                    log.exception(
                        "geak: enqueue rebench for sigterm-recovered result failed"
                    )
                return
            _finish_skip({
                "status": "error",
                "error_class": "timeout",
                "error": (f"GEAK e2e killed after {kill_timeout}s "
                          f"(budget-capped); closing window preserved"),
                "runner_timeout_s": runner_timeout,
                "kill_timeout_s": kill_timeout,
            })
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("GEAK runner crashed")
            _finish_skip({"status": "error", "error_class": "runner_crashed",
                          "error": repr(exc)})
            return

        result: dict[str, Any] = _read_geak_result(result_path)
        if not result:
            _finish_skip({
                "status": "error",
                "error_class": "no_result_json",
                "error": (f"runner rc={proc.returncode} produced no parseable "
                          f"result.json at {result_path}"),
                "stderr_tail": stderr_tail,
            })
            return
        # Carry the actual exit code so the breakdown can audit a nonzero rc.
        result.setdefault("returncode", proc.returncode)
        state.geak_result = result

        # Invariant guard: a GEAK run whose baseline ref failed to reproduce
        # ``orchestrator_best_tput_same_config`` optimized against a phantom
        # baseline, so its gain is non-comparable — never promote it.
        if str(result.get("status") or "") == "baseline_reproduction_failed":
            log.warning(
                "GEAK baseline_reproduction_failed: ref did not match "
                "orchestrator best (%s); refusing to promote a phantom-baseline gain",
                result.get("error"),
            )
            _finish_skip({
                "status": "baseline_reproduction_failed",
                "error_class": "baseline_reproduction_failed",
                "error": (str(result.get("error") or "")[:500] or
                          "GEAK baseline ref != orchestrator best (env_spec mismatch)"),
                "ref_tput": result.get("ref_tput"),
                "orchestrator_best_tput_same_config": result.get(
                    "orchestrator_best_tput_same_config"
                ),
            })
            return

        # Rebench-first: record the win as an UNVALIDATED candidate only; the
        # headline is written later from the measured rebench.
        self._record_geak_candidate(result)
        self._record_geak_kernel_journey(result)
        # Enqueue the same-harness config-identity rebench — the ONLY path that
        # writes the headline. Until it lands the candidate stays pending.
        if str(result.get("status") or "") == "ok":
            try:
                await self._enqueue_internal_stack_rebench(reason="geak_e2e_win")
            except Exception:  # noqa: BLE001 - defensive
                log.exception("geak: enqueue same-harness revalidation failed")
        self._record_phase_entry_evidence(geak={
            "status": result.get("status"),
            "throughput_speedup": result.get("throughput_speedup"),
            "final_throughput_tok_s": result.get("final_throughput_tok_s"),
            "eval_dir": result.get("eval_dir"),
            "report_path": result.get("report_path"),
            "runner_timeout_s": runner_timeout,
        })
        state.save(self.session_dir)
        await self.bus.append_and_seq(Message.new(
            "kernel_agent", "orchestration", "response",
            {
                "in_reply_to": "",
                "kind": "geak_e2e_done",
                "status": str(result.get("status") or "unknown"),
                "speedup": result.get("throughput_speedup"),
                "result_path": str(result_path),
            },
            priority=1,
        ))
        # KERNEL is a one-shot under GEAK: wind down to SWEEP (persist the hint).
        state.set_pending_escalate_hint(_phase_state.ESCALATE_HINT_SKIP_TO_SWEEP)
        state.save(self.session_dir)

    def _geak_win_already_recorded(self) -> bool:
        """Whether a GEAK e2e win is already in this session's state.

        Gates crash-recovery from an existing ``result.json`` so a prior cycle's
        win is not re-promoted on a later KERNEL entry.
        """
        return any(
            isinstance(item, dict) and item.get("action") == "geak_e2e"
            for item in (self.shared_state.optimization_stack or [])
        )

    def _geak_legacy_promote(self) -> bool:
        """Whether the GEAK win was already written before revalidation.

        Older sessions promoted GEAK directly into ``current_best`` /
        ``optimization_stack`` before the same-harness replay. Rebench-first
        sessions instead carry a ``geak_pending`` candidate and should promote
        from the measured replay result.
        """
        if self.shared_state.geak_pending:
            return False
        current_best = self.shared_state.current_best if isinstance(self.shared_state.current_best, dict) else {}
        return self._geak_win_already_recorded() or str(current_best.get("action") or "") == "geak_e2e"

    @staticmethod
    def _parse_geak_accepted_config(
        result: dict[str, Any],
    ) -> tuple[str, dict[str, str]]:
        """Parse ``result.accepted_config`` into (flags, env dict).

        Turns the bench-style ``{"flags":.., "env":..}`` blob into a reproducible
        (server-args, real-env) pair: any ``KEY=VAL`` token in ``env`` becomes a
        real env var; any ``--flag`` token folds into flags.
        """
        accepted_cfg = result.get("accepted_config") or {}
        accepted_flags = str(accepted_cfg.get("flags") or "").strip()
        parsed_envs, extra_flags = _split_env_and_flags(str(accepted_cfg.get("env") or ""))
        if extra_flags:
            accepted_flags = (accepted_flags + " " + extra_flags).strip()
        return accepted_flags, parsed_envs

    def _record_geak_candidate(self, result: dict[str, Any]) -> None:
        """Record a GEAK e2e win as an UNVALIDATED candidate (no headline).

        Stores the accepted config + the optimizer's own (audit-only)
        throughput/speedup under ``geak_pending`` without touching
        ``current_best`` / ``optimization_stack`` / ``cumulative_gain*``. The
        headline is written later from a measured rebench by
        ``_promote_geak_from_candidate``; the config is captured verbatim as the
        source the rebench launches from.
        """
        if not isinstance(result, dict) or result.get("status") not in ("ok",):
            return
        new_tput = float(result.get("final_throughput_tok_s") or 0.0)
        if new_tput <= 0:
            return
        accepted_flags, parsed_envs = self._parse_geak_accepted_config(result)
        base = float(self.shared_state.baseline_tput or 0.0)
        self_gain = ((new_tput - base) / base * 100.0) if base > 0 else None
        am = result.get("alignment_metrics") or {}
        self.shared_state.geak_pending = {
            "status": "awaiting_rebench",
            # Audit-only self-reported numbers (not the headline until rebench).
            "self_reported_tput": new_tput,
            "self_reported_speedup": result.get("throughput_speedup"),
            "self_reported_gain_pct": self_gain,
            "self_reported_basis": result.get("final_throughput_basis"),
            # Reproducible config the rebench launches from.
            "accepted_flags": accepted_flags,
            "accepted_envs": dict(parsed_envs),
            "final_overlay": result.get("final_overlay") or "",
            "final_launch_script": result.get("final_launch_script"),
            "bench_script": result.get("bench_script"),
            "eval_dir": result.get("eval_dir"),
            # GEAK's own within-harness speedups, for the report's audit cross-check.
            "alignment": {
                "hot_geak_speedup": am.get("hot_geak_speedup"),
                "cold_geak_speedup": am.get("cold_geak_speedup"),
                "hot_speedup": am.get("hot_speedup"),
                "cold_speedup": am.get("cold_speedup"),
                "final_basis": am.get("final_basis") or result.get("final_throughput_basis"),
                "geak_throughput_speedup": result.get("throughput_speedup"),
            },
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        # Surface a large cross-harness measurement divergence as a warning only.
        bb = result.get("baseline_basis") or {}
        mdiv = bb.get("measurement_divergence_pct")
        try:
            mdiv_f = abs(float(mdiv)) if mdiv is not None else None
        except (TypeError, ValueError):
            mdiv_f = None
        if mdiv_f is not None and mdiv_f > _GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT:
            log.warning(
                "geak candidate: large cross-harness measurement divergence "
                "%.2f%% (|.|>%.1f%%) - candidate held out of headline until a "
                "main-flow rebench validates it",
                float(mdiv), _GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT,
            )

    def _promote_geak_from_candidate(
        self,
        result: dict[str, Any],
        *,
        measured_tput: float,
        provenance: str,
    ) -> None:
        """Write the GEAK headline from a MEASURED main-flow rebench.

        The single headline writer: lifts ``current_best`` (config/overlay/scripts
        + the measured tput), appends the ``geak_e2e`` optimization_stack entry +
        gain ledger, and stamps ``cumulative_gain`` / ``cumulative_gain_validated``
        as the same-harness total ``(measured - baseline)/baseline``. Clears
        ``geak_pending`` and the revalidation flag.
        """
        if not isinstance(result, dict):
            return
        try:
            measured = float(measured_tput)
        except (TypeError, ValueError):
            return
        if measured <= 0:
            return
        accepted_flags, parsed_envs = self._parse_geak_accepted_config(result)

        cb = dict(self.shared_state.current_best or {})
        cb_envs = dict(cb.get("extra_envs") or {}) if isinstance(cb.get("extra_envs"), Mapping) else {}
        cb_envs.update(parsed_envs)
        cb.update({
            "action": "geak_e2e",
            "tput": measured,
            "ttft_mean_ms": result.get("ttft_ms"),
            "tpot_mean_ms": result.get("tpot_ms"),
            "extra_server_args": accepted_flags,
            "extra_envs": cb_envs,
            "geak_launch_script": result.get("final_launch_script"),
            "geak_bench_script": result.get("bench_script"),
            "geak_eval_dir": result.get("eval_dir"),
            "final_overlay": result.get("final_overlay") or "",
            "workspace": result.get("eval_dir"),
        })
        # Audit cross-check: GEAK's own within-harness speedups (not the headline).
        am = result.get("alignment_metrics") or {}
        cb["geak_alignment"] = {
            "hot_geak_speedup": am.get("hot_geak_speedup"),
            "cold_geak_speedup": am.get("cold_geak_speedup"),
            "hot_speedup": am.get("hot_speedup"),
            "cold_speedup": am.get("cold_speedup"),
            "final_basis": am.get("final_basis") or result.get("final_throughput_basis"),
            "geak_throughput_speedup": result.get("throughput_speedup"),
        }
        self.shared_state.current_best = cb

        ts = datetime.now(timezone.utc).isoformat()
        if not self._geak_win_already_recorded():
            entry = {
                "action": "geak_e2e",
                "variant_name": "geak_e2e",
                "tput": measured,
                "candidate_extra_server_args": accepted_flags,
                "extra_envs": dict(parsed_envs),
                "final_overlay": result.get("final_overlay") or "",
                "workspace": result.get("eval_dir"),
                "accepted_kernels": result.get("accepted_kernels") or [],
                "accepted_heads": result.get("accepted_heads") or [],
                "report_path": result.get("report_path"),
                "source": "geak_e2e",
                "ts": ts,
            }
            self.shared_state.optimization_stack.append(entry)
            self.shared_state.append_stack_gain_entry(
                action="geak_e2e",
                variant_name="geak_e2e",
                new_tput=measured,
                extra_server_args=accepted_flags,
                ts=ts,
            )

        base = float(self.shared_state.baseline_tput or 0.0)
        if base > 0:
            gain = (measured - base) / base * 100.0
            self.shared_state.cumulative_gain = gain
            self.shared_state.cumulative_gain_validated = gain
            self.shared_state.cumulative_gain_validated_ts = ts
            self.shared_state.cumulative_gain_validated_stack_len = len(
                self.shared_state.optimization_stack
            )
        self.shared_state.cumulative_gain_provenance = provenance
        self.shared_state.resume_pending_revalidation = False
        self.shared_state.geak_pending = {}

    def _record_geak_kernel_journey(self, result: dict[str, Any]) -> None:
        """Replay GEAK-e2e's kernel_journey.json into the breakdown recorder.

        GEAK-e2e emits a ``kernel_journey.json`` whose per-kernel sub-objects are
        shaped exactly as the recorder's ``record_kernel_{dispatch,backend_result,
        e2e}`` inputs; replay them verbatim so the assembler folds the e2e
        optimizer's kernels into ``kernel_journey``. Best-effort: a missing/partial
        file never breaks the phase.
        """
        if not isinstance(result, dict):
            return
        kj_path = str(result.get("kernel_journey_path") or "")
        if not kj_path:
            eval_dir = str(result.get("eval_dir") or "")
            if eval_dir:
                kj_path = str(Path(eval_dir) / "kernel_journey.json")
        if not kj_path or not Path(kj_path).is_file():
            return
        try:
            journey = json.loads(Path(kj_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(journey, dict):
            return

        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        sdir = self.session_dir
        commit = str(getattr(self.shared_state, "code_revision", "") or "")
        # Replay GEAK-e2e's discovery substream so the assembler backfills each
        # kernel's discovery-sourced fields; GEAK profiles via rocprofv3 (route
        # ``bypass``), ``tool="geak"`` for version provenance.
        for run in (journey.get("discovery_runs") or []):
            if not isinstance(run, dict):
                continue
            try:
                instrument.record_kernel_discovery(
                    sdir,
                    source=str(run.get("source") or "bypass"),
                    status=str(run.get("status") or "success"),
                    hot_kernels=list(run.get("hot_kernels") or []),
                    scan=run.get("scan") if isinstance(run.get("scan"), dict) else None,
                    tool="geak",
                )
            except Exception:  # noqa: BLE001
                log.debug("geak kernel_journey discovery replay failed",
                          exc_info=True)
        for k in (journey.get("kernels") or []):
            if not isinstance(k, dict):
                continue
            kid = str(k.get("kernel_id") or "")
            if not kid:
                continue
            disp = k.get("dispatch") if isinstance(k.get("dispatch"), dict) else {}
            try:
                instrument.record_kernel_dispatch(
                    sdir,
                    kernel_id=kid,
                    dispatched=bool(disp.get("dispatched", True)),
                    backends=list(disp.get("backends") or []),
                    skip_reason=str(disp.get("skip_reason") or ""),
                    orchestration_commit=commit,
                    task_group=disp.get("task_group"),
                )
                br = k.get("backend_result")
                if isinstance(br, dict):
                    instrument.record_kernel_backend_result(sdir, br)
                e2e = k.get("e2e")
                if isinstance(e2e, dict):
                    instrument.record_kernel_e2e(
                        sdir,
                        kernel_id=kid,
                        integrated=bool(e2e.get("integrated", False)),
                        e2e_gain_pct=e2e.get("e2e_gain_pct"),
                        validated=e2e.get("validated"),
                        decision=str(e2e.get("decision") or ""),
                        patch_path=e2e.get("patch_path"),
                        target_file=e2e.get("target_file"),
                        extra_server_args=str(e2e.get("extra_server_args") or ""),
                    )
            except Exception:  # noqa: BLE001
                log.debug("geak kernel_journey replay failed for %s", kid,
                          exc_info=True)
        for tool, meta in (journey.get("versions") or {}).items():
            if not isinstance(meta, dict):
                continue
            try:
                instrument.record_tool_version(
                    sdir,
                    tool=str(tool),
                    root=str(meta.get("root_dir") or "") or None,
                    version=str(meta.get("version") or meta.get("commit") or "") or None,
                )
            except Exception:  # noqa: BLE001
                pass

    def _ck_blockscale_switch_eligible(self, result: dict[str, Any]) -> bool:
        """Whether the fp8 block-scale CK backend switch should be E2E-validated.

        The CK backend switch (``SGLANG_FP8_BLOCKSCALE_CK_MAX_M``) routes the fp8
        block-scale GEMM from the Triton default to the aiter CK
        ``gemm_a8w8_blockscale`` kernel on gfx942; it is independent of the a8w8
        table tuner result and must be flipped + E2E-validated as its own
        candidate. Gated strictly to the forge backend on a
        sglang + fp8 + gfx942 + block-scale workload (block-scale asserted
        positively via ``weight_block_size``).

        Args:
            result (dict[str, Any]): The GEMM tuning handler result.

        Returns:
            bool: ``True`` only when the CK switch is the relevant lever.
        """
        if not isinstance(result, dict):
            return False
        from ..kernel.request_handlers import _resolve_gemm_tuning_backend

        backend = str(
            result.get("backend") or _resolve_gemm_tuning_backend({})
        ).strip().lower()
        if backend != "forge":
            return False
        framework = str(getattr(self.shared_state, "framework", "") or "").strip().lower()
        if framework != "sglang":
            return False
        if not self._ck_switch_precision_is_fp8(result):
            return False

        from hyperloom.inference_optimizer.gpu_types import _resolve_amd_gpu_type
        from ..actions.executors._workload_envs import _GFX942_GPU_TYPES

        gpu = _resolve_amd_gpu_type(getattr(self.shared_state, "gpu_type", "") or "")
        if gpu not in _GFX942_GPU_TYPES:
            return False

        # Block-scale fp8 only, asserted positively via ``weight_block_size``.
        from hyperloom.inference_optimizer.model_config_utils import _fp8_is_block_scale

        model_path = str(
            getattr(self.shared_state, "model_path", "")
            or os.environ.get("MODEL_PATH", "")
        )
        return _fp8_is_block_scale(model_path)

    def _ck_switch_precision_is_fp8(self, result: dict[str, Any]) -> bool:
        """Whether the workload runs fp8, resolved from any available signal.

        Accepts fp8 from, in order: ``shared_state.precision``, the forge
        ``result`` envelope's resolved precision, or the runtime
        ``--quantization`` resolved from the actual server args.

        Args:
            result (dict[str, Any]): The GEMM tuning handler result.

        Returns:
            bool: ``True`` when any signal resolves to fp8.
        """
        if str(getattr(self.shared_state, "precision", "") or "").strip().lower() == "fp8":
            return True
        if isinstance(result, dict) and str(result.get("precision") or "").strip().lower() == "fp8":
            return True
        try:
            from ..kernel.request_handlers import _resolve_forge_precision_and_quant

            precision, _ = _resolve_forge_precision_and_quant(self.shared_state, {})
            if str(precision or "").strip().lower() == "fp8":
                return True
        except Exception:  # noqa: BLE001 - best-effort runtime resolution
            pass
        return False

    async def _handle_gemm_tuning_result(self, result: dict[str, Any]) -> None:
        """Record and post-process a run_gemm_tuning result from any entrypoint.

        Both the KERNEL-entry auto hook and orchestration-issued
        ``run_gemm_tuning`` requests converge here so forge results never bypass
        per-tuner E2E validation.
        """
        self.shared_state.record_gemm_tuning(result)
        # Forge results route to the per-tuner E2E validator when table tuning
        # asked for it OR when the CK block-scale backend switch is eligible.
        if result.get("backend") == "forge" and (
            result.get("requires_e2e_validation")
            or self._ck_blockscale_switch_eligible(result)
        ):
            await self._validate_forge_gemm_tuning_e2e(result)
        else:
            self._promote_gemm_tuning_keep(result)
        self.shared_state.save(self.session_dir)

    def _journal_gemm_tuning_keep(
        self,
        entry: dict[str, Any],
        *,
        task_id: str = "",
    ) -> None:
        """Mirror an adopted GEMM-tuning stack entry as an optimization_journal KEEP row.

        Emits a KEEP journal row carrying the end-to-end ``throughput_after`` plus
        the originating ``task_id`` so the GEMM tuning point shows up on the
        phase_timeline alongside every other attempt. Best-effort.

        Args:
            entry: The ``optimization_stack`` entry just appended for this
                GEMM-tuning adoption (carries variant_name / tput / gain_pct /
                backend / tuned_file / ts).
            task_id: Originating task id used to join per-step token spend.
        """
        try:
            journal = self._ensure_journal()
            variant_name = str(entry.get("variant_name") or "gemm_tuning")
            backend = str(entry.get("backend") or "").strip().lower()
            try:
                tput = float(entry["tput"]) if entry.get("tput") is not None else None
            except (TypeError, ValueError):
                tput = None
            try:
                gain_pct = float(entry["gain_pct"]) if entry.get("gain_pct") is not None else None
            except (TypeError, ValueError):
                gain_pct = None
            metrics: dict[str, Any] = {}
            if entry.get("tuned_file"):
                metrics["tuned_file"] = str(entry.get("tuned_file"))
            journal.append_entry(
                JournalEntry(
                    phase=self._journal_entry_phase(),
                    iter=int(self.shared_state.tick or 0),
                    kind=KIND_GEMM_TUNING,
                    change=variant_name,
                    outcome=OUTCOME_KEEP,
                    gain_pct=gain_pct,
                    throughput_after=tput,
                    task_id=str(task_id or ""),
                    variant_name=variant_name,
                    ts=str(entry.get("ts") or ""),
                    provenance=f"gemm_tuning:{backend}" if backend else "gemm_tuning",
                    tick=int(self.shared_state.tick or 0),
                    metrics=metrics,
                )
            )
        except Exception:  # noqa: BLE001 — journaling is best-effort
            log.exception("gemm_tuning journal append failed")

    def _promote_gemm_tuning_keep(self, result: dict[str, Any]) -> None:
        """Promote a successful GEMM tuning run into the main gain ledger.

        Only acts on a successful, ``KEEP``-decision result with a speedup
        greater than 1.0 and a known baseline. Appends an entry to the
        optimization stack (deduped on tuned file), updates ``current_best``,
        and stamps ``cumulative_gain`` / ``cumulative_gain_validated`` since
        the GEMM benchmark is itself an end-to-end serving measurement.

        For forge-gemm-tune results (``requires_e2e_validation=True``), the
        entry is promoted but ``cumulative_gain_validated`` is NOT stamped —
        downstream E2E validation (explore action) must confirm the gain.

        Args:
            result (dict[str, Any]): The GEMM tuning handler result; ignored if
                not a successful KEEP.
        """
        if not isinstance(result, dict):
            return
        status = str(result.get("status") or "").strip().lower()
        decision = str(result.get("decision") or "").strip().upper()
        if status not in {"ok", "complete", "completed", "succeeded", "success"}:
            return
        if decision != "KEEP":
            return
        try:
            speedup = float(result.get("best_speedup") or 0.0)
            baseline = float(self.shared_state.baseline_tput or 0.0)
        except (TypeError, ValueError):
            return
        if speedup <= 1.0 or baseline <= 0:
            return

        backend = str(result.get("backend") or "geak").strip().lower()
        ts = datetime.now(timezone.utc).isoformat()

        # Resolve extra_envs: forge provides them; GEAK infers from tuned_file.
        if backend == "forge":
            extra_envs = dict(result.get("extra_envs") or result.get("recommended_env") or {})
            tuned_file = ""
            artifacts = result.get("artifacts") or {}
            if isinstance(artifacts, dict) and artifacts:
                tuned_file = str(next(iter(artifacts.values()), ""))
            if not tuned_file:
                tuned_file = str(next(iter(extra_envs.values()), "")) if extra_envs else ""
            variant_name = "forge_gemm_tuned"
        else:
            tuned_file = str(result.get("tuned_file") or "")
            extra_envs = (
                {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": tuned_file} if tuned_file else {}
            )
            variant_name = "a8w8_blockscale_tuned_gemm"

        # fp8 block-scale CK backend switch safety net (an operator-set value
        # wins via setdefault); the primary forge path validates it standalone.
        if self._ck_blockscale_switch_eligible(result):
            extra_envs.setdefault("SGLANG_FP8_BLOCKSCALE_CK_MAX_M", "256")

        final_report = str(result.get("final_report_path") or "")

        # GEAK path: E2E already validated internally.
        tuned_tput = baseline * speedup
        existing = {
            str(item.get("tuned_file") or "")
            for item in (self.shared_state.optimization_stack or [])
            if isinstance(item, dict) and item.get("action") == "gemm_tuning"
        }

        entry = {
            "action": "gemm_tuning",
            "variant_name": variant_name,
            "tuned_file": tuned_file,
            "final_report_path": final_report,
            "gain_pct": (speedup - 1.0) * 100.0,
            "tput": tuned_tput,
            "workspace": result.get("workspace"),
            "extra_envs": extra_envs,
            "backend": backend,
            "source": "kernel_entry_auto",
            "ts": ts,
        }
        if tuned_file not in existing:
            self.shared_state.optimization_stack.append(entry)
            self.shared_state.append_stack_gain_entry(
                action="gemm_tuning",
                variant_name=variant_name,
                new_tput=tuned_tput,
                ts=ts,
            )
            self._journal_gemm_tuning_keep(
                entry, task_id=str(result.get("task_id") or ""),
            )
        self.shared_state.current_best = {
            "action": "gemm_tuning",
            "engine": backend,
            "tput": tuned_tput,
            "variant_name": variant_name,
            "tuned_file": tuned_file,
            "final_report_path": final_report,
            "workspace": result.get("workspace"),
            "extra_envs": extra_envs,
        }
        self.shared_state.cumulative_gain = (speedup - 1.0) * 100.0
        self.shared_state.cumulative_gain_validated = self.shared_state.cumulative_gain
        self.shared_state.cumulative_gain_validated_ts = ts
        self.shared_state.cumulative_gain_validated_stack_len = len(
            self.shared_state.optimization_stack or []
        )

    def _replace_latest_gemm_tuning_attempt(self, result: dict[str, Any]) -> None:
        """Sync the latest GEMM history row after forge E2E rewrites ``result``."""
        if not isinstance(result, dict):
            return
        entry = dict(result)
        attempts = list(getattr(self.shared_state, "gemm_tuning_attempts", []) or [])
        if attempts and isinstance(attempts[-1], dict):
            entry.setdefault("ts", attempts[-1].get("ts"))
            attempts[-1] = entry
        else:
            entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
            attempts.append(entry)
        self.shared_state.gemm_tuning_attempts = attempts
        self.shared_state.last_gemm_tuning = entry

    async def _validate_forge_gemm_tuning_e2e(self, result: dict[str, Any]) -> None:
        """Sequentially E2E-validate each forge tuner's env independently.

        Like kernel_opt's per-kernel integrate: try each tuner's env one by
        one. KEEPs accumulate (stacked envs); REVERTs are discarded. This
        prevents one bad tuner from dragging down the whole set.
        """
        from ..kernel.request_handlers import integrate_handler

        tuners_run = result.get("tuners_run") or []
        # The list is already priority-sorted by forge CLI (fmoe_ck first).
        candidates = []
        for t in tuners_run:
            if not isinstance(t, dict):
                continue
            if t.get("status") != "ok":
                continue
            if int(t.get("improved_shapes") or 0) <= 0:
                continue
            env_var = str(t.get("env_var") or "").strip()
            env_value = str(t.get("env_value") or "").strip()
            if env_var and env_value:
                candidates.append({
                    "tuner": t.get("tuner") or "unknown",
                    "env_var": env_var,
                    "env_value": env_value,
                    "micro_speedup": float(t.get("best_micro_speedup") or 1.0),
                })

        # Standalone fp8 block-scale CK backend switch: inject as its own
        # candidate so the loop E2E-validates baseline Triton vs CK.
        if self._ck_blockscale_switch_eligible(result):
            if not any(
                c.get("env_var") == "SGLANG_FP8_BLOCKSCALE_CK_MAX_M"
                for c in candidates
            ):
                candidates.append({
                    "tuner": "ck_blockscale_backend_switch",
                    "env_var": "SGLANG_FP8_BLOCKSCALE_CK_MAX_M",
                    "env_value": "256",
                    "micro_speedup": 1.0,
                })

        if not candidates:
            log.info("forge gemm tuning: no candidates to E2E validate")
            return

        baseline_tput = float(self.shared_state.baseline_tput or 0.0)
        running_tput = float(
            (self.shared_state.current_best or {}).get("tput") or baseline_tput
        )
        stacked_envs: dict[str, str] = {}
        kept: list[dict[str, Any]] = []
        reverted: list[dict[str, Any]] = []
        ts = datetime.now(timezone.utc).isoformat()
        try:
            from ..actions.executors.explore import _compute_explore_variant_timeout

            per_tuner_timeout_sec = _compute_explore_variant_timeout(
                baseline_runtime_sec=float(getattr(self.shared_state, "baseline_runtime_sec", 0.0) or 0.0),
                kill_ratio=float(getattr(self.shared_state, "explore_overtime_kill_ratio", 1.5) or 1.5),
            )
        except Exception:  # noqa: BLE001 - conservative fallback
            per_tuner_timeout_sec = 15 * 60
        per_tuner_budget_minutes = max(1, int((per_tuner_timeout_sec + 59) // 60))

        for cand in candidates:
            tuner_name = cand["tuner"]
            env = {cand["env_var"]: cand["env_value"]}
            extra_server_args = (
                "--moe-runner-backend aiter"
                if tuner_name == "fmoe_ck" and str(getattr(self.shared_state, "framework", "") or "").lower() == "sglang"
                else ""
            )
            # Merge with previously KEEP'd envs.
            test_envs = dict(stacked_envs)
            test_envs.update(env)

            log.info(
                "forge gemm E2E: validating tuner=%s env=%s (base_tput=%.1f)",
                tuner_name, cand["env_var"], running_tput,
            )

            try:
                integrate_result = await integrate_handler(
                    {
                        "task_id": f"gemm_tune_e2e_{tuner_name}",
                        "kernel_id": f"gemm_tune_{tuner_name}",
                        "source": "forge_gemm_tuning",
                        "base_tput": running_tput,
                        "extra_server_args": extra_server_args,
                        "extra_envs": test_envs,
                        "keep_threshold_pct": 3.0,
                        "budget_minutes": per_tuner_budget_minutes,
                    },
                    session_dir=self.session_dir,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "forge gemm E2E: integrate failed for %s: %s",
                    tuner_name, exc,
                )
                reverted.append({**cand, "reason": repr(exc)})
                continue

            decision = str(integrate_result.get("decision") or "").upper()
            new_tput = float(integrate_result.get("new_tput") or 0.0)
            gain_pct = float(integrate_result.get("gain_pct") or 0.0)

            log.info(
                "forge gemm E2E: tuner=%s decision=%s new_tput=%.1f gain=%.2f%%",
                tuner_name, decision, new_tput, gain_pct,
            )

            if decision == "KEEP" and new_tput > running_tput:
                stacked_envs.update(env)
                running_tput = new_tput
                kept.append({**cand, "tput": new_tput, "gain_pct": gain_pct})

                entry = {
                    "action": "gemm_tuning",
                    "variant_name": f"forge_{tuner_name}",
                    "tuned_file": cand["env_value"],
                    "gain_pct": gain_pct,
                    "tput": new_tput,
                    "workspace": result.get("workspace"),
                    "extra_server_args": extra_server_args,
                    "extra_envs": dict(stacked_envs),
                    "backend": "forge",
                    "source": "kernel_entry_auto",
                    "ts": ts,
                }
                self.shared_state.optimization_stack.append(entry)
                self.shared_state.append_stack_gain_entry(
                    action="gemm_tuning",
                    variant_name=f"forge_{tuner_name}",
                    new_tput=new_tput,
                    ts=ts,
                )
                self._journal_gemm_tuning_keep(
                    entry, task_id=f"gemm_tune_e2e_{tuner_name}",
                )
            else:
                reverted.append({**cand, "reason": f"decision={decision}, gain={gain_pct:.2f}%"})

        # Update current_best and cumulative_gain with final stacked result.
        if kept:
            self.shared_state.current_best = {
                "action": "gemm_tuning",
                "engine": "forge",
                "tput": running_tput,
                "variant_name": "forge_gemm_tuned",
                "extra_server_args": "--moe-runner-backend aiter" if "AITER_CONFIG_FMOE" in stacked_envs else "",
                "extra_envs": stacked_envs,
                "workspace": result.get("workspace"),
            }
            total_gain = (running_tput - baseline_tput) / baseline_tput * 100.0 if baseline_tput > 0 else 0.0
            self.shared_state.cumulative_gain = total_gain
            self.shared_state.cumulative_gain_validated = total_gain
            self.shared_state.cumulative_gain_validated_ts = ts
            self.shared_state.cumulative_gain_validated_stack_len = len(
                self.shared_state.optimization_stack or []
            )
            log.info(
                "forge gemm E2E: %d tuners KEEP (total gain=+%.2f%%), %d REVERT",
                len(kept), total_gain, len(reverted),
            )
        else:
            stacked_envs = {}
            total_gain = 0.0
            log.info(
                "forge gemm E2E: all %d tuners REVERT, no E2E gain",
                len(reverted),
            )

        # Rewrite the stored result to the E2E-validated outcome so the LLM never
        # sees the raw combined recommended_env and issues a bundled integrate.
        result["e2e_results"] = {"kept": kept, "reverted": reverted}
        result["recommended_env_raw"] = dict(result.get("recommended_env") or {})
        result["extra_envs_raw"] = dict(result.get("extra_envs") or {})
        result["recommended_env"] = dict(stacked_envs)
        result["extra_envs"] = dict(stacked_envs)
        result["e2e_gain_pct"] = round(float(total_gain), 4)
        result["e2e_validated"] = True
        result["requires_e2e_validation"] = False
        if kept:
            result["status"] = "complete"
            result["decision"] = "KEEP"
        else:
            result["status"] = "complete"
            result["decision"] = "REVERT"
            result["micro_decision"] = "candidate_no_e2e_gain"
        self._replace_latest_gemm_tuning_attempt(result)

    def _should_continue_kernel_after_gemm(self) -> bool:
        """Decide whether to run source-level kernel_opt right after GEMM tuning.

        Returns:
            bool: ``True`` when the ``continue_kernel_after_gemm`` flag is set
                and there are untried hot reusable kernels remaining.
        """
        if not bool(getattr(self.shared_state, "continue_kernel_after_gemm", True)):
            return False
        return bool(self.shared_state.untried_hot_reusable_kernels())

    async def _run_kernel_opt_after_gemm(self) -> None:
        """Run the source-level kernel optimization batch after GEMM tuning."""
        cached = self.shared_state.last_trace_analyze or {}
        candidates_path = str(cached.get("candidates_path") or "")
        if not candidates_path:
            log.info("KERNEL entry: skip kernel_opt after GEMM; no candidates_path")
            return
        log.info(
            "KERNEL entry: continuing to source-level kernel_opt after GEMM tuning",
        )
        try:
            from ..kernel.request_handlers import run_optimization_handler

            result = await run_optimization_handler(
                {
                    "candidates_path": candidates_path,
                    "session_id": self.session_dir.name,
                },
                session_dir=self.session_dir,
                record_partial=self._record_kernel_opt_partial,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("KERNEL entry run_optimization after GEMM failed")
            result = {
                "status": "failed",
                "error_class": exc.__class__.__name__,
                "error": repr(exc),
            }
        await self.bus.append_and_seq(
            Message.new(
                "kernel_agent",
                "orchestration",
                "response",
                {
                    "in_reply_to": "",
                    "kind": "run_optimization_done",
                    "status": result.get("status", "ok") if isinstance(result, dict) else "failed",
                    "result": result,
                    "source": "kernel_entry_auto_after_gemm",
                },
                priority=1,
            )
        )
        if isinstance(result, dict) and not result.get("batch_mode"):
            self.shared_state.record_kernel_opt(result)
        self.shared_state.save(self.session_dir)

    def _fusion_required_before_kernel_opt(self) -> bool:
        """Gate the forge-fusion step in KERNEL entry.

        Runs only when: not disabled by ``HYPERLOOM_SKIP_FUSION``, the framework is
        fusion-eligible (sglang/vllm), a decode trace exists to discover from, and no
        fusion already succeeded this session (idempotent re-entry).
        """
        import os

        if str(os.environ.get("HYPERLOOM_SKIP_FUSION", "")).strip().lower() in ("1", "true", "yes", "on"):
            return False
        framework = str(getattr(self.shared_state, "framework", "") or "sglang").strip().lower()
        if framework not in ("sglang", "vllm", "vllm-aiter"):
            return False
        trace = str(getattr(self.shared_state, "last_profile_trace", "") or "").strip()
        if not trace:
            log.info("KERNEL entry: skip forge-fusion (no decode trace yet)")
            return False
        last = getattr(self.shared_state, "last_fusion", None)
        if isinstance(last, dict) and str(last.get("status") or "").strip() in ("ok", "complete", "kept"):
            return False
        return True

    async def _run_forge_fusion_after_gemm(self) -> None:
        """Run autonomous kernel fusion (forge-fusion) after GEMM tuning."""
        log.info("KERNEL entry: running forge-fusion (autonomous kernel fusion) after GEMM")
        try:
            from ..kernel.request_handlers import run_fusion_handler

            result = await run_fusion_handler(
                {"task_id": "kernel_entry_fusion", "reason": "kernel_entry_auto"},
                session_dir=self.session_dir,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("KERNEL entry forge-fusion failed")
            result = {
                "status": "failed", "decision": "REVERT", "engine": "forge_fusion",
                "error_class": exc.__class__.__name__, "error": repr(exc),
            }
        await self._handle_fusion_result(result)

    async def _handle_fusion_result(self, result: dict) -> None:
        """Record the forge-fusion result + surface it on the bus.

        Hands a KEPT fusion (source patch + env flags) to ``integrate_handler``
        for the real e2e re-baseline / adopt decision.
        """
        status = str(result.get("status") or "unknown") if isinstance(result, dict) else "failed"
        try:
            self.shared_state.last_fusion = result if isinstance(result, dict) else {"status": status}
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 - state shape tolerant (best-effort idempotency record)
            pass
        try:
            await self.bus.append_and_seq(
                Message.new(
                    "kernel_agent", "orchestration", "response",
                    {
                        "in_reply_to": "", "kind": "run_fusion_done", "status": status,
                        "result": result, "source": "kernel_entry_auto",
                    },
                    priority=1,
                )
            )
        except Exception:  # noqa: BLE001
            log.exception("failed to post run_fusion_done bus message")
        # A KEPT fusion is handed to integrate for the e2e re-baseline decision.
        if isinstance(result, dict) and result.get("kept") and result.get("requires_e2e_validation"):
            await self._integrate_fusion(result)

    async def _integrate_fusion(self, result: dict) -> None:
        """Hand a KEPT forge-fusion (source patch + env flags) to integrate for e2e adopt.

        forge-fusion is NOT env-only (``source='forge_fusion'``), so integrate runs the
        patch-apply path: it applies the fused-kernel source patch, sets the fusion env
        flags on the re-baseline server, and KEEPs only when measured e2e throughput
        clears the threshold. ``base_tput`` is filled from state by integrate_handler.
        """
        import os

        from ..kernel.request_handlers import integrate_handler, materialize_unified_patch_snapshot

        patch = str(result.get("patch") or "").strip()
        target_file = str(result.get("source_file") or result.get("target_file") or "").strip()
        kernel_repo = str(result.get("kernel_repo") or "").strip()
        env_flags = result.get("env_flags") or {}
        current_envs = {}
        if isinstance(self.shared_state.current_best, dict):
            current_envs = dict(self.shared_state.current_best.get("extra_envs") or {})
        merged_envs = {**current_envs, **{str(k): str(v) for k, v in env_flags.items()}}
        if not patch or not target_file:
            log.info("KERNEL entry: fusion KEPT but missing patch/target_file; skip integrate")
            return
        integ = None
        snapshot_dir = str(result.get("snapshot_dir") or "").strip()
        if not snapshot_dir and patch.endswith(".patch") and kernel_repo:
            try:
                snapshot_dir = await asyncio.to_thread(
                    materialize_unified_patch_snapshot,
                    patch_path=patch,
                    repo_root=kernel_repo,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("KERNEL entry fusion snapshot materialization failed")
                integ = {
                    "status": "failed", "decision": "REVERT",
                    "error_class": exc.__class__.__name__, "error": repr(exc),
                    "patch_path": patch, "target_file": target_file,
                }
        if integ is None:
            try:
                integ = await integrate_handler(
                    {
                        "task_id": "fusion_e2e",
                        "kernel_id": "forge_fusion",
                        "source": "forge_fusion",
                        "patch_path": patch,
                        "target_file": target_file,
                        "kernel_repo": kernel_repo,
                        "snapshot_dir": snapshot_dir,
                        "extra_envs": merged_envs,
                        "keep_threshold_pct": float(os.environ.get("HYPERLOOM_FUSION_KEEP_PCT", "3.0")),
                    },
                    session_dir=self.session_dir,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("KERNEL entry fusion integrate failed")
                integ = {"status": "failed", "decision": "REVERT",
                         "error_class": exc.__class__.__name__, "error": repr(exc)}
        decision = str(integ.get("decision") or "").strip().upper() if isinstance(integ, dict) else "REVERT"
        gain = integ.get("gain_pct") if isinstance(integ, dict) else None
        log.info("KERNEL entry: fusion integrate decision=%s gain_pct=%s", decision, gain)
        self._promote_fusion_integrate_keep(result, integ, extra_envs=merged_envs)
        try:
            self.shared_state.last_fusion_integrate = integ
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            pass
        try:
            await self.bus.append_and_seq(
                Message.new(
                    "kernel_agent", "orchestration", "response",
                    {
                        "in_reply_to": "", "kind": "fusion_integrate_done",
                        "status": integ.get("status", "failed") if isinstance(integ, dict) else "failed",
                        "decision": decision, "gain_pct": gain, "result": integ,
                        "source": "kernel_entry_auto",
                    },
                    priority=1,
                )
            )
        except Exception:  # noqa: BLE001
            log.exception("failed to post fusion_integrate_done bus message")

    def _promote_fusion_integrate_keep(
        self,
        fusion_result: dict,
        integrate_result: dict,
        *,
        extra_envs: dict[str, str] | None = None,
    ) -> None:
        """Promote a forge-fusion e2e KEEP into the main optimization stack."""
        if not isinstance(fusion_result, dict) or not isinstance(integrate_result, dict):
            return
        if str(integrate_result.get("decision") or "").strip().upper() != "KEEP":
            return
        try:
            new_tput = float(integrate_result.get("new_tput") or 0.0)
            gain = float(integrate_result.get("gain_pct") or 0.0)
        except (TypeError, ValueError):
            return
        if new_tput <= 0:
            return

        patch = str(fusion_result.get("patch") or integrate_result.get("patch_path") or "")
        existing = {
            str(item.get("patch_path") or "")
            for item in (self.shared_state.optimization_stack or [])
            if isinstance(item, dict) and item.get("action") == "fusion"
        }
        ts = datetime.now(timezone.utc).isoformat()
        envs = dict(extra_envs or integrate_result.get("extra_envs") or fusion_result.get("env_flags") or {})
        extra_args = str(integrate_result.get("extra_server_args") or "")
        entry = {
            "action": "fusion",
            "variant_name": "forge_fusion",
            "backend": "forge",
            "engine": "forge_fusion",
            "provenance": "forge_fusion",
            "source": "kernel_entry_auto",
            "tput": new_tput,
            "gain_pct": gain,
            "workspace": integrate_result.get("workspace"),
            "patch_path": patch,
            "target_file": fusion_result.get("source_file") or integrate_result.get("target_file"),
            "extra_envs": envs,
            "extra_server_args": extra_args,
            "kernel_speedup": fusion_result.get("kernel_speedup"),
            "best_pattern": fusion_result.get("best_pattern"),
            "ts": ts,
        }
        if patch not in existing:
            self.shared_state.optimization_stack.append(entry)
            self.shared_state.append_stack_gain_entry(
                action="fusion",
                variant_name="forge_fusion",
                new_tput=new_tput,
                extra_server_args=extra_args,
                ts=ts,
            )
        self.shared_state.current_best = {
            "action": "fusion",
            "backend": "forge",
            "engine": "forge_fusion",
            "tput": new_tput,
            "variant_name": "forge_fusion",
            "workspace": integrate_result.get("workspace"),
            "patch_path": patch,
            "target_file": entry["target_file"],
            "extra_envs": envs,
            "extra_server_args": extra_args,
        }
        self.shared_state.cumulative_gain = gain
        self.shared_state.cumulative_gain_validated = gain
        self.shared_state.cumulative_gain_validated_ts = ts
        self.shared_state.cumulative_gain_validated_stack_len = len(
            self.shared_state.optimization_stack or []
        )

    def _current_tput_from_validated_gain(self) -> float:
        """Project current tput from ``baseline_tput * (1 + cumulative_gain_validated/100)``; 0.0 when baseline unknown (watermark not-yet-armed).

        Returns:
            The projected current throughput, or ``0.0`` when the baseline is
            unknown.
        """
        state = self.shared_state
        try:
            base = float(state.baseline_tput or 0.0)
        except (TypeError, ValueError):
            base = 0.0
        if base <= 0:
            return 0.0
        try:
            gain = float(state.cumulative_gain_validated or 0.0)
        except (TypeError, ValueError):
            gain = 0.0
        return base * (1.0 + gain / 100.0)

    def _last_measured_roofline_tput(self) -> float:
        """Measured tok/s of the most recent roofline snapshot; 0.0 when none."""
        snaps = getattr(self.shared_state, "roofline_snapshots", None) or []
        for snap in reversed(snaps):
            if not isinstance(snap, dict):
                continue
            try:
                tput = float(snap.get("achieved_tok_per_sec") or 0.0)
            except (TypeError, ValueError):
                tput = 0.0
            if tput > 0:
                return tput
        return 0.0

    def _needs_roofline_for_watermark(self) -> bool:
        """True iff projected tput crossed the watermark over ``last_roofline_tput`` (False until PRELUDE roofline ran, or while auto_roofline_pending_task_id is in-flight).

        Returns:
            ``True`` when a fresh roofline is warranted because projected tput
            crossed the watermark ratio; ``False`` otherwise (including the
            bootstrap and in-flight re-arm guards).
        """
        state = self.shared_state
        try:
            last_rl = float(state.last_roofline_tput or 0.0)
        except (TypeError, ValueError):
            last_rl = 0.0
        if (state.auto_roofline_pending_task_id or "").strip():
            return False
        if last_rl <= 0:
            try:
                failure_streak = int(getattr(state, "roofline_failure_streak", 0) or 0)
            except (TypeError, ValueError):
                failure_streak = 0
            if failure_streak <= 0:
                return False
            try:
                last_rl = float(state.baseline_tput or 0.0)
            except (TypeError, ValueError):
                last_rl = 0.0
            if last_rl <= 0:
                return False
        cur = self._current_tput_from_validated_gain()
        if cur <= 0:
            return False
        return cur / last_rl >= _resolve_roofline_watermark_ratio()

    async def _maybe_enqueue_watermark_roofline(
        self,
        *,
        reason: str,
    ) -> bool:
        """Enqueue a fresh roofline if the watermark crossed; idempotency-keyed via ``reason``, stamps auto_roofline_pending_task_id. Returns True when enqueued.

        Args:
            reason: Tag used in the task's idempotency key and logging.

        Returns:
            ``True`` if a roofline task was enqueued, else ``False``.
        """
        if not self._needs_roofline_for_watermark():
            return False
        try:
            task = await self._enqueue_internal_analysis_task(reason=reason)
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception(
                "watermark-roofline (%s): failed to enqueue: %r",
                reason,
                exc,
            )
            return False
        self.shared_state.auto_roofline_pending_task_id = task.task_id
        log.info(
            "watermark-roofline (%s): enqueued task=%s (cur=%.2f, last_roofline=%.2f, ratio>=%.2f)",
            reason,
            task.task_id,
            self._current_tput_from_validated_gain(),
            float(self.shared_state.last_roofline_tput or 0.0),
            self._ROOFLINE_WATERMARK_RATIO,
        )
        return True

    def _cached_kernel_request(self, kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Return a cached programmatic_handler result if applicable (cache key last_trace_analyze).

        Args:
            kind: The kernel request kind; only ``trace_analyze`` is cacheable.
            payload: The merged request payload; its ``trace_input`` /
                ``trace_dir`` must match the cached entry for a hit.

        Returns:
            A synthesized cached result dict on a cache hit, else ``None``.
        """
        if kind != "trace_analyze":
            return None
        cached = self.shared_state.last_trace_analyze or {}
        if not isinstance(cached, dict) or not cached:
            return None
        trace_input = payload.get("trace_input") or payload.get("trace_dir")
        if not trace_input or trace_input != cached.get("trace_input"):
            return None
        candidates_path = cached.get("candidates_path")
        if not candidates_path or not Path(candidates_path).exists():
            return None
        return {
            "status": "ok",
            "candidates_path": candidates_path,
            "hot_kernels_top15": cached.get("hot_kernels_top15", []),
            "reusable_native_kernel_ids": cached.get("reusable_native_kernel_ids", []),
            "cached_at": cached.get("ts"),
            "note": "served from shared_state.last_trace_analyze cache",
        }
