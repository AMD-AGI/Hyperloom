# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""KERNEL_AGENT phase handler: bf16-dense-GEMM fallback, PerfSkills e2e run,
GEMM-tuning keep/promote, and watermark-roofline gating."""

from __future__ import annotations
import asyncio
import json
import logging as _logging
import os
import signal
import subprocess
import time
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
from ..loop.coordinator_helpers import (  # noqa: F401 - re-exported for callers/tests
    _BASELINE_FINGERPRINT_KEYS,
    _baseline_params_fingerprint,
    _dedupe_extra_server_args,
    _infer_model_class_from_config,
    _merge_cumulative_extra_sglang_args,
    _parse_baseline_workload_extra,
    _parse_iso_unix,
    _resolve_roofline_watermark_ratio,
    effective_closing_grace_sec,
    format_exc_brief,
    serialize_verdict_advisory,
)
from ..loop.coordinator import (
    PERFSKILLS_GEAK_BACKEND,
    _relabel_perfskills_geak_journey,
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
        # With a measured trace, reprofile only on a material change; with none,
        # fall through so GEAK still gets a real trace.
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

    def _perfskills_enabled(self) -> bool:
        """Whether the KERNEL_AGENT phase is delegated to the PerfSkills e2e optimizer.

        The single source of truth is the kernel backend order
        (``KERNEL_OPT_BACKEND_ORDER`` / ``KERNEL_OPT_BACKENDS``): when
        ``perfskills`` appears there, it owns the whole phase.  The
        ``kernel_optimizer`` state field is the persisted record of that
        decision (derived from the order at startup); it is used as a resume
        fallback so this stays correct even when the env var is not re-exported
        in a fresh shell.
        """
        from ..kernel.request_handlers import perfskills_selected

        if perfskills_selected():
            return True
        return (
            str(getattr(self.shared_state, "kernel_optimizer", "") or "")
            .strip()
            .lower()
            == "perfskills"
        )

    async def _on_enter_kernel(self, *, from_phase: str) -> None:
        """Run deterministic KERNEL-entry setup before LLM kernel work (FP8 GEMM tuning gate).

        Args:
            from_phase: The phase being left, used only for logging.
        """
        if not self._kernel_enabled():
            # Should not happen — --no-kernel routes EXPLORE → SWEEP.
            log.info(
                "KERNEL entry hook fired with kernel_enabled=False (from=%s)",
                from_phase or "<unknown>",
            )
            return
        if self._perfskills_enabled():
            # PerfSkills owns the whole KERNEL_AGENT phase: one in-process e2e run
            # seeded with the EXPLORE best config, then hand straight to SWEEP
            # (which reuses PerfSkills' final_launch.sh + bench_e2e.sh).
            await self._run_perfskills_kernel_phase(from_phase=from_phase)
            return
        if not self._gemm_tuning_required_before_kernel_opt():
            # No GEMM tuning here: refresh the snapshot (explore gains) before the LLM drives GEAK.
            await self._maybe_reprofile_for_kernel()
            return

        # Refresh the snapshot (explore gains) before GEMM tuning targets the bottleneck.
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
        # Capture explore + GEMM-tuning gains before inline GEAK targets the bottleneck.
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

        Recent production runs showed the automatic KERNEL-entry GEMM step can
        stop after a single fp8 a8w8/a8w8_blockscale no-op. Historical wins came
        from a follow-up ``sglang_dense_bf16`` run, so make that fallback
        deterministic when the fp8 tuner produced no E2E-validatable candidate.
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
        """Extract Hyperloom's bench measurement protocol for the PerfSkills handoff.

        Reads the materialized baseline recipe's ``benchmark.envs`` (the exact
        knobs Magpie benched with) and falls back to the process env. Returns
        only the keys that resolve so absent values leave PerfSkills on its own
        standalone defaults. Never raises — measurement-protocol propagation must
        not block the KERNEL_AGENT phase.
        """
        envs: dict[str, Any] = {}
        try:
            import yaml  # local import: yaml is not a coordinator top-level dep

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

    def _perfskills_timeouts(self) -> tuple[int, int, bool]:
        """Resolve the PerfSkills e2e timeouts from the live run budget.

        The KERNEL_AGENT phase-entry hook runs PerfSkills synchronously, so a fixed
        subprocess default would (a) ignore ``--max-hours`` / the run deadline
        and (b) keep the tick loop from reaching the deadline → closing-phase
        check until it returns. To stay inside the budget we cap the run so it
        ALWAYS finishes with at least the closing-grace window left, and shrink
        the runner's own budget by a safety margin on top of that.

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
        # deadline is set (budget_known=False) — e.g. a unit test invoking the
        # hook directly, or PerfSkills run outside an orchestrated session. When
        # Hyperloom DRIVES the run (deadline known) the budget MUST come from
        # Hyperloom's live deadline / KERNEL_AGENT phase allocation, so this default
        # never caps a Hyperloom-driven run (a long --max-hours session can
        # legitimately allot KERNEL more than 12h).
        env_default_timeout = int(os.environ.get("PERFSKILLS_E2E_TIMEOUT_S", "43200"))
        deadline = self._run_deadline
        if deadline is None:
            return env_default_timeout, env_default_timeout + 600, False
        remaining = deadline - time.monotonic()
        grace = effective_closing_grace_sec(
            float(getattr(self.shared_state, "max_minutes", 0) or 0), None,
        )
        margin = float(os.environ.get("PERFSKILLS_BUDGET_MARGIN_S", "300"))
        # Reserve the closing window: the subprocess (incl. result.json flush)
        # must be killed with at least ``grace`` left so closing can still run.
        kill_budget = remaining - grace
        # Also honour the KERNEL_AGENT phase's own wall-clock budget: PerfSkills runs
        # synchronously inside the phase-entry hook, so a run longer than the
        # phase allocation would overrun the phase budget the same way it would
        # overrun the session deadline. Cap by min(session, kernel_phase).
        phase_rem = _phase_state.phase_budget_remaining_seconds(
            self.shared_state, budget_pct=self._phase_budget_pct,
        )
        if phase_rem is not None:
            kill_budget = min(kill_budget, float(phase_rem))
        # Hyperloom-authoritative budget: the runner self-stops ``margin`` before
        # the hard subprocess kill, and the kill reserves the closing-grace
        # window. Derived purely from the live budget — the 12h env default does
        # NOT cap it (requirement: PerfSkills time comes from Hyperloom here).
        kill_timeout = int(max(0.0, kill_budget))
        runner_timeout = int(max(0.0, kill_budget - margin))
        return runner_timeout, kill_timeout, True

    async def _run_perfskills_kernel_phase(self, *, from_phase: str) -> None:
        """Delegate the KERNEL_AGENT phase to PerfSkills (one whole-pipeline e2e run).

        Builds a handoff from the EXPLORE best config, runs the PerfSkills
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
        # Bench measurement protocol: forward the SAME knobs Hyperloom
        # actually benched with so PerfSkills' internal e2e measures identically.
        # Without this PerfSkills falls back to its own standalone defaults
        # (e.g. RANDOM_RANGE_RATIO=1 fixed-length vs Hyperloom's 0 variable-length)
        # and the cross-harness numbers diverge. Source of truth = the materialized
        # baseline recipe's benchmark.envs (the exact values Magpie ran), with a
        # process-env fallback. Only keys that resolve are sent; absent keys leave
        # PerfSkills on its own defaults so it still runs standalone.
        bench_protocol = self._resolve_bench_protocol(
            str(getattr(state, "baseline_config_path", "") or "")
        )
        handoff = {
            "schema_version": 1,
            "model_path": str(getattr(state, "model_path", "") or os.environ.get("MODEL_PATH", "")),
            "framework": str(os.environ.get("FRAMEWORK", "") or "sglang"),
            "gpu_type": str(getattr(state, "gpu_type", "") or os.environ.get("GPU_TYPE", "")),
            "tp": int(os.environ.get("TP", "1") or 1),
            "workload": workload,
            "accepted_flags": accepted_flags,
            "accepted_env": accepted_env,
            "launch_recipe": str(getattr(state, "baseline_config_path", "") or ""),
            "raw_baseline_tput": float(getattr(state, "baseline_tput", 0.0) or 0.0),
            "exp_root": str(self.session_dir / "perfskills"),
            # Align PerfSkills' bench CLIENT to Hyperloom's exact one (InferenceX
            # benchmark_serving.py) so final/sweep numbers are cross-harness comparable.
            "bench_client": "auto",
            "inferencex_path": str(os.environ.get("INFERENCEX_PATH", "")),
            # Pin the serving / optimization GPU set so PerfSkills never guesses:
            # honour an explicit visibility mask, else 0..tp-1 (matches run_e2e
            # map_args' own default). Removes ambiguity when Hyperloom drives.
            "gpu_ids": (
                os.environ.get("HIP_VISIBLE_DEVICES")
                or os.environ.get("CUDA_VISIBLE_DEVICES")
                or ",".join(str(i) for i in range(int(os.environ.get("TP", "1") or 1)))
            ),
        }
        if bench_protocol:
            handoff["bench_protocol"] = bench_protocol

        out_dir = self.session_dir / "perfskills"
        out_dir.mkdir(parents=True, exist_ok=True)
        handoff_path = out_dir / "handoff.json"
        handoff_path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")

        from ..kernel.request_handlers import _kernel_agent_tool_path

        def _read_perfskills_result(path: Path) -> dict[str, Any]:
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
            state.perfskills_result = result
            self._promote_perfskills_result(result)
            self._record_perfskills_kernel_journey(result)
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
            self._record_phase_entry_evidence(perfskills=evidence)
            state.save(self.session_dir)
            state.set_pending_escalate_hint(_phase_state.ESCALATE_HINT_SKIP_TO_SWEEP)

        def _finish_skip(result: dict[str, Any]) -> None:
            """Record a (failed/skipped) PerfSkills outcome + wind down to SWEEP.

            Always records the normalized outcome into ``perfskills_result``,
            mirrors the failure reason onto the phase-entry evidence (so the
            session-breakdown surfaces WHY the e2e run did not land), then sets
            the ``skip_to_sweep`` hint so the coordinator never deadlocks.
            """
            state.perfskills_result = result
            self._record_phase_entry_evidence(perfskills={
                "status": result.get("status"),
                "error_class": result.get("error_class"),
                "error": (str(result.get("error") or "")[:500] or None),
            })
            state.save(self.session_dir)
            state.set_pending_escalate_hint(_phase_state.ESCALATE_HINT_SKIP_TO_SWEEP)

        # Crash-recovery: a validated result.json written before the coordinator
        # crashed (handback never reached state.save) must be promoted on resume.
        # Guard with ``_perfskills_win_already_recorded`` so a prior cycle's
        # result.json (``perfskills/`` is a fixed path) does not short-circuit a
        # fresh KERNEL entry in a later macro-cycle.
        result_path = out_dir / "result.json"
        recovered = _read_perfskills_result(result_path)
        if (
            recovered.get("status") == "ok"
            and not self._perfskills_win_already_recorded()
        ):
            log.info(
                "PerfSkills result.json exists but state has no recorded win "
                "(crash before handback); promoting recovered result."
            )
            _promote_recovered_result(recovered, recovered_from="existing_result_json")
            return

        try:
            runner = _kernel_agent_tool_path("backends/perfskills_runner.py")
        except Exception as exc:  # noqa: BLE001
            log.exception("PerfSkills runner not resolvable; skipping KERNEL")
            _finish_skip({"status": "error", "error_class": "runner_not_found",
                          "error": repr(exc)})
            return

        # Budget-aware timeouts: shrink to the remaining run deadline and always
        # reserve the closing-grace window.
        runner_timeout, kill_timeout, budget_known = self._perfskills_timeouts()
        min_run = int(os.environ.get("PERFSKILLS_MIN_RUN_S", "600"))
        if budget_known and runner_timeout < min_run:
            log.warning(
                "PerfSkills: only %ds budget remains (< min %ds); skipping e2e "
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
        log.info("KERNEL entry: delegating to PerfSkills e2e (from=%s) "
                 "runner_timeout=%ds kill_timeout=%ds budget_known=%s cmd=%s",
                 from_phase or "<unknown>", runner_timeout, kill_timeout,
                 budget_known, " ".join(cmd))

        # Run in its own process group so a timeout can SIGTERM the whole
        # runner -> run_e2e -> vllm/node tree (grace to flush result.json), then
        # SIGKILL. A bare subprocess.run(timeout=) would SIGKILL only the direct
        # child and orphan run_e2e + its servers.
        term_grace = int(os.environ.get("PERFSKILLS_TERM_GRACE_S", "180"))

        def _run() -> subprocess.CompletedProcess:
            p = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=dict(os.environ), start_new_session=True,
            )

            def _killpg(sig: int) -> None:
                try:
                    os.killpg(os.getpgid(p.pid), sig)
                except (ProcessLookupError, PermissionError):
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
                log.warning("PerfSkills runner rc=%s: %s", proc.returncode, stderr_tail)
        except subprocess.TimeoutExpired:
            log.warning("PerfSkills runner exceeded kill_timeout=%ds; SIGTERM'd "
                        "to let it flush, then reclaimed the closing window",
                        kill_timeout)
            # The graceful SIGTERM gives run_e2e a window to flush result.json
            # (recover-from-disk). If it landed a real win, keep it instead of
            # discarding the whole KERNEL_AGENT phase as a timeout.
            recovered = _read_perfskills_result(result_path)
            if recovered.get("status") == "ok":
                log.info("PerfSkills flushed an OK result.json under SIGTERM "
                         "grace; promoting the recovered win despite the cap.")
                _promote_recovered_result(
                    recovered,
                    recovered_from="sigterm_flushed_result_json",
                    runner_timeout_s=runner_timeout,
                )
                return
            _finish_skip({
                "status": "error",
                "error_class": "timeout",
                "error": (f"PerfSkills e2e killed after {kill_timeout}s "
                          f"(budget-capped); closing window preserved"),
                "runner_timeout_s": runner_timeout,
                "kill_timeout_s": kill_timeout,
            })
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("PerfSkills runner crashed")
            _finish_skip({"status": "error", "error_class": "runner_crashed",
                          "error": repr(exc)})
            return

        result: dict[str, Any] = _read_perfskills_result(result_path)
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
        state.perfskills_result = result

        self._promote_perfskills_result(result)
        self._record_perfskills_kernel_journey(result)
        self._record_phase_entry_evidence(perfskills={
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
                "kind": "perfskills_e2e_done",
                "status": str(result.get("status") or "unknown"),
                "speedup": result.get("throughput_speedup"),
                "result_path": str(result_path),
            },
            priority=1,
        ))
        # KERNEL is a one-shot under PerfSkills: wind down to SWEEP.
        state.set_pending_escalate_hint(_phase_state.ESCALATE_HINT_SKIP_TO_SWEEP)

    def _perfskills_win_already_recorded(self) -> bool:
        """Whether a PerfSkills e2e win is already in this session's state.

        Used to gate crash-recovery from an existing ``result.json`` so a prior
        cycle's win (``perfskills/`` is a fixed path) is not re-promoted on a
        later KERNEL entry. Mirrors the ``optimization_stack`` dedup in
        ``_promote_perfskills_result``.
        """
        return any(
            isinstance(item, dict) and item.get("action") == "perfskills_e2e"
            for item in (self.shared_state.optimization_stack or [])
        )

    def _promote_perfskills_result(self, result: dict[str, Any]) -> None:
        """Fold a PerfSkills e2e win into current_best + the validated gain ledger.

        Also appends an ``optimization_stack`` entry and the matching
        ``gain_per_stack_entry`` so the session-breakdown attribution section
        credits the e2e gain to a concrete stack entry (carrying the per-kernel
        / head / config evidence from ``result.json``) instead of leaving the
        gain unattributed.
        """
        if not isinstance(result, dict) or result.get("status") not in ("ok",):
            return
        new_tput = float(result.get("final_throughput_tok_s") or 0.0)
        base = float(self.shared_state.baseline_tput or 0.0)
        if new_tput <= 0:
            return
        cb = dict(self.shared_state.current_best or {})
        cb.update({
            "action": "perfskills_e2e",
            "tput": new_tput,
            "ttft_mean_ms": result.get("ttft_ms"),
            "tpot_mean_ms": result.get("tpot_ms"),
            # Sweep-reuse handles: the optimized self-contained launch + bench scripts.
            "perfskills_launch_script": result.get("final_launch_script"),
            "perfskills_bench_script": result.get("bench_script"),
            "perfskills_eval_dir": result.get("eval_dir"),
            "workspace": result.get("eval_dir"),
        })
        self.shared_state.current_best = cb

        # Attribute the e2e gain to a concrete optimization_stack entry so the
        # breakdown's attribution / optimization_stack sections reflect it (the
        # native lanes do the same via append_stack_gain_entry).
        ts = datetime.now(timezone.utc).isoformat()
        accepted_cfg = result.get("accepted_config") or {}
        already = any(
            isinstance(item, dict) and item.get("action") == "perfskills_e2e"
            for item in (self.shared_state.optimization_stack or [])
        )
        if not already:
            entry = {
                "action": "perfskills_e2e",
                "variant_name": "perfskills_e2e",
                "tput": new_tput,
                "candidate_extra_server_args": str(accepted_cfg.get("flags") or ""),
                "extra_envs": (
                    {"PERFSKILLS_ACCEPTED_ENV": str(accepted_cfg.get("env"))}
                    if accepted_cfg.get("env") else {}
                ),
                "workspace": result.get("eval_dir"),
                # Per-kernel / head evidence for the attribution + lifecycle view.
                "accepted_kernels": result.get("accepted_kernels") or [],
                "accepted_heads": result.get("accepted_heads") or [],
                "report_path": result.get("report_path"),
                "source": "perfskills_e2e",
                "ts": ts,
            }
            self.shared_state.optimization_stack.append(entry)
            self.shared_state.append_stack_gain_entry(
                action="perfskills_e2e",
                variant_name="perfskills_e2e",
                new_tput=new_tput,
                extra_server_args=str(accepted_cfg.get("flags") or ""),
                ts=ts,
            )
        if base > 0:
            gain = (new_tput - base) / base * 100.0
            self.shared_state.cumulative_gain = gain
            self.shared_state.cumulative_gain_validated = gain
            self.shared_state.cumulative_gain_validated_ts = (
                datetime.now(timezone.utc).isoformat()
            )

    def _record_perfskills_kernel_journey(self, result: dict[str, Any]) -> None:
        """Replay GEAK-e2e's kernel_journey.json into the breakdown recorder.

        GEAK-e2e is a whole-pipeline e2e optimizer whose authored kernels do
        not go through the per-kernel SDK recorder path. It emits a
        self-contained ``kernel_journey.json`` whose per-kernel
        sub-objects are shaped EXACTLY as the recorder's
        ``record_kernel_{dispatch,backend_result,e2e}`` inputs (see GEAK-e2e
        ``interface/run_e2e.py`` ``build_kernel_journey``). We replay them
        verbatim so the assembler folds the e2e optimizer's kernels into
        ``kernel_journey`` next to tracelens discovery — no mapping logic here,
        the contract file owns it. Best-effort: a missing/partial file never
        breaks the phase.
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

        # The GEAK-e2e pipeline's GEAK is a distinct variant ("geak_v4") from
        # the kernel-agent's generic ``geak`` backend. Relabel it before replay
        # so SBD/trace (versions map, kernel_journey backend lanes) never
        # conflate the two provenances. See ``_relabel_perfskills_geak_journey``.
        _relabel_perfskills_geak_journey(journey)

        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        sdir = self.session_dir
        commit = str(getattr(self.shared_state, "code_revision", "") or "")
        # Replay GEAK-e2e's discovery substream so the
        # assembler backfills each kernel's discovery-sourced fields
        # (name/gpu_pct/bound_type/source_file). GEAK-e2e profiles via rocprofv3,
        # not tracelens, so the route is ``bypass``; ``tool=geak_v4`` keeps the
        # version provenance under the GEAK-e2e variant instead of minting an
        # empty bypass entry (and apart from the generic ``geak`` lane).
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
                    tool=PERFSKILLS_GEAK_BACKEND,
                )
            except Exception:  # noqa: BLE001
                log.debug("perfskills kernel_journey discovery replay failed",
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
                log.debug("perfskills kernel_journey replay failed for %s", kid,
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

        The CK backend switch (``SGLANG_FP8_BLOCKSCALE_CK_MAX_M``) routes the
        fp8 block-scale GEMM from the Triton default to the aiter CK
        ``gemm_a8w8_blockscale`` kernel on gfx942. This is the big lever
        (~2x at decode M, ~+109% e2e) and is INDEPENDENT of the a8w8 table
        tuning result: the table tuner routinely reports ``no_improvement``
        because the CK default is already optimal, yet the switch itself must
        still be flipped and E2E-validated as its own gemm_tuning candidate.

        Gated strictly so it only fires for the forge backend on a
        sglang + fp8 + gfx942 + block-scale workload. fp8 is accepted from any
        signal — session precision, the resolved forge result, or a runtime
        ``--quantization fp8`` server arg (session/yaml precision may still read
        ``bf16``). Block-scale is required positively (the checkpoint declares
        ``weight_block_size``), which naturally excludes per-tensor, static and
        per-channel/per-token fp8 — those take other GEMM paths and must never
        be switched here.

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

        from hyperloom.inference_optimizer.cli.model_gate import _resolve_amd_gpu_type
        from ..actions.executors._workload_envs import _GFX942_GPU_TYPES

        gpu = _resolve_amd_gpu_type(getattr(self.shared_state, "gpu_type", "") or "")
        if gpu not in _GFX942_GPU_TYPES:
            return False

        # Block-scale fp8 only, asserted positively: the CK patch only rewrites
        # the block-scale path (``aiter_w8a8_block_fp8_linear`` /
        # ``gemm_a8w8_blockscale``), so the checkpoint must declare
        # ``weight_block_size``. This excludes per-tensor, static and
        # per-channel/per-token fp8, which take other GEMM paths.
        from hyperloom.inference_optimizer.model_config_utils import _fp8_is_block_scale

        model_path = str(
            getattr(self.shared_state, "model_path", "")
            or os.environ.get("MODEL_PATH", "")
        )
        return _fp8_is_block_scale(model_path)

    def _ck_switch_precision_is_fp8(self, result: dict[str, Any]) -> bool:
        """Whether the workload runs fp8, resolved from any available signal.

        The session-level ``precision`` is not authoritative: precision is often
        resolved at runtime from server args (``--quantization fp8``) while the
        session/yaml precision still reads ``bf16``. Accept fp8 from, in order:

        1. ``shared_state.precision`` (session-level), OR
        2. the forge ``result`` envelope, which stamps the resolved precision
           (see ``_run_forge_gemm_tuning``), OR
        3. the runtime ``--quantization`` resolved by
           ``_resolve_forge_precision_and_quant`` from the actual server args.

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
        ``run_gemm_tuning`` requests must converge here; otherwise forge
        results can bypass per-tuner E2E validation.
        """
        self.shared_state.record_gemm_tuning(result)
        # Forge results route to the per-tuner E2E validator when table tuning
        # asked for it OR when the CK block-scale backend switch is eligible —
        # the latter is a standalone lever that must be validated even when the
        # a8w8 table tuner reported no_improvement (decision != KEEP).
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

        GEMM-tuning adoptions previously landed only in ``optimization_stack``
        / ``roofline_progress.trajectory`` and never as a ``phase_timeline``
        event. As a result the run's serving throughput was invisible to the
        timeline (and to downstream throughput-attempt series that read the
        flat phase_timeline). Emitting a KEEP journal row — carrying the
        end-to-end ``throughput_after`` plus the originating ``task_id`` for
        token attribution — closes that gap so the GEMM tuning point shows up
        alongside every other attempt. Best-effort: journaling failures never
        abort the run.

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

        # Resolve extra_envs: forge provides recommended_env/extra_envs;
        # GEAK infers from tuned_file.
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

        # fp8 block-scale CK backend switch (still attributed to gemm_tuning).
        # The primary forge path validates this as a standalone candidate in
        # _validate_forge_gemm_tuning_e2e (see _handle_gemm_tuning_result
        # routing); this inline-promote path only injects it as a safety net
        # for an eligible forge result that reaches inline promotion without
        # the validator. setdefault so an operator-set value always wins.
        if self._ck_blockscale_switch_eligible(result):
            extra_envs.setdefault("SGLANG_FP8_BLOCKSCALE_CK_MAX_M", "256")

        final_report = str(result.get("final_report_path") or "")

        # GEAK path: E2E already validated internally.
        # (Forge path is handled by _validate_forge_gemm_tuning_e2e before this.)
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
        # Sort by priority: fmoe_ck (MoE) first, dense tuners second.
        # The original list is already priority-sorted by forge CLI.
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

        # Standalone fp8 block-scale CK backend switch: independent of the a8w8
        # table tuner outcome (often no_improvement because the CK default is
        # already optimal). Inject it as its own candidate so the loop below
        # E2E-validates baseline Triton vs CK and, on KEEP, attributes the gain
        # to gemm_tuning. Shape matches a table candidate exactly.
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
            # Merge with previously KEEP'd envs for stacked validation.
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

                # Push to optimization_stack.
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

        # Rewrite the stored GEMM tuning result to the *E2E-validated*
        # outcome. This prevents the orchestration LLM from seeing the raw
        # combined recommended_env and issuing a bundled integrate later.
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
        """True iff projected tput crossed the 10% watermark over ``last_roofline_tput`` (bootstrap guard: False until PRELUDE roofline ran; re-arm guard: False while auto_roofline_pending_task_id is in-flight).

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
