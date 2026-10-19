# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The GEAK 2a fallback: replay the win through GEAK's own end-to-end harness.

Reached only when the orchestrator's own 2b rebench cannot rule on a GEAK win.
Replaying GEAK's launch script reproduces the accepted configuration by
construction, which is what makes a succeeded replay engagement proof.
"""

from __future__ import annotations

import logging as _logging
from typing import TYPE_CHECKING, Any, Mapping

from hyperloom.common.env import env_int

from ..actions.executors._accuracy_gate import EVAL_KIND_ACCURACY_UNAVAILABLE, accuracy_passed
from ..collaborator import CoordinatorCollaborator
from ..kernel.geak_config import (
    _geak_has_accepted_kernel,
    _geak_overlay_digest,
    _geak_sweep_measured_tput,
    _normalize_geak_overlay_dir,
    geak_overlay_is_loadable,
)
from ._geak_revalidation import _GeakRebench, _ReplayRuling

if TYPE_CHECKING:
    from ..state.task_registry import Task

log = _logging.getLogger(__name__)


class GeakHarnessReplay(CoordinatorCollaborator):
    """Validates a GEAK win by re-running it under GEAK's own benchmark harness."""

    async def validate_geak_via_geak_harness(self, *, reason: str) -> dict[str, Any]:
        """2a fallback - validate the geak win by REPLAYING it through
        GEAK's own ``bench_e2e.sh`` (the harness that produced the headline
        result), so the optimized config engages BY CONSTRUCTION regardless of
        winner kind (tuned-config / kernel / overlay / flag). Because the replay
        reproduces the optimized config from ``result.json`` directly, a
        ``succeeded`` status is itself the engagement proof. The validated gain
        is the MEASURED replay throughput over the orchestrator's raw baseline
        (``(measured - baseline) / baseline``); GEAK's own ``throughput_speedup``
        / ``hot_geak_speedup`` serves only as a ``> 1.0`` sanity gate, not as the
        reported number. Recorded under a distinct provenance
        (``geak_same_harness_geak``) because the measurement came from GEAK's
        harness rather than the orchestrator's. Used only when 2b (orchestrator
        harness) is inconclusive.

        Args:
            reason: Human-readable reason stamped in logs/return.

        Returns:
            A summary dict describing whether validation succeeded.
        """
        from hyperloom.common.perf_metric import is_agentx_mode
        from ..actions.executors._workload_envs import agentx_enabled, geak_metric_axis

        mode = str(self.shared_state.benchmark_mode or "").strip()
        agentx = is_agentx_mode(mode) if mode else agentx_enabled()
        if agentx:
            return {
                "validated": False,
                "status": "incomparable",
                "reason": "geak_harness_unsupported_canonical_workload",
            }
        ps = self.shared_state.geak_result if isinstance(self.shared_state.geak_result, dict) else {}
        if str(ps.get("status") or "") != "ok" and not _geak_has_accepted_kernel(ps):
            return {"validated": False, "skipped": True, "reason": "no_geak_result"}
        from hyperloom.common.jsonio import read_json

        handoff = read_json(self.session_dir / "geak" / "handoff.json", default={}, require_dict=True)
        env_spec = self.build_env_spec()
        # Overlay identity, captured BEFORE the replay so it can be compared
        # after. 2a replays GEAK's own launch script, which is why a
        # ``succeeded`` status proves the *config* engaged -- but the overlay is
        # a separate artifact on a path in ``result.json``, and it can be gone
        # or inert by the time the replay runs. Measured over
        # ``/shared_nfs/hyperloom-claw``: of 64 runs declaring a
        # ``final_overlay``, 25 name a directory that does not exist and 30 name
        # one holding no ``sitecustomize.py``. Only 9 can install a kernel. A
        # non-empty string is therefore not evidence the kernel ran, and using
        # it as evidence would stamp 55 flag-only measurements as kernel wins.
        # Same check 2b runs; see ``geak_overlay_is_loadable``.
        ps_overlay_2a = _normalize_geak_overlay_dir(str(ps.get("final_overlay") or "").strip())
        overlay_digest_before = _geak_overlay_digest(ps_overlay_2a) if ps_overlay_2a else ""
        am = ps.get("alignment_metrics") or {}
        # Read GEAK's OWN within-harness speedup on the SAME basis it promoted
        # (result.throughput_speedup == cold_geak_speedup when final_basis=="cold",
        # else the hot within-GEAK ratio; see run_e2e final-basis selection). It is
        # used solely to sanity-check that GEAK claimed a win on its promoted basis
        # before the replay measurement is promoted as the headline.
        # Falls back to the explicit within-GEAK ratios when throughput_speedup is
        # missing (older result.json), preferring the promoted basis.
        try:
            geak_sp = float(ps.get("throughput_speedup") or 0.0)
        except (TypeError, ValueError):
            geak_sp = 0.0
        if geak_sp <= 0:
            basis = str(am.get("final_basis") or ps.get("final_throughput_basis") or "hot")
            fallback_key = "cold_geak_speedup" if basis == "cold" else "hot_geak_speedup"
            try:
                geak_sp = float(am.get(fallback_key) or am.get("hot_geak_speedup") or 0.0)
            except (TypeError, ValueError):
                geak_sp = 0.0
        regimes = ps.get("validated_regimes") or []
        reg = regimes[0] if regimes and isinstance(regimes[0], dict) else {}
        try:
            conc = int(reg.get("conc") or 64)
            isl = int(reg.get("isl") or 1024)
            osl = int(reg.get("osl") or 1024)
        except (TypeError, ValueError):
            conc, isl, osl = 64, 1024, 1024
        from hyperloom.inference_optimizer.session.session_paths import unique_runs_dir
        from ..actions.executors._geak_sweep import sweep_via_geak

        timeout = env_int("SWEEP_VARIANT_TIMEOUT_SEC", default=2400)
        res = await sweep_via_geak(
            result=ps,
            handoff=handoff,
            env_spec=env_spec,
            conc_values=[conc],
            isl_osl_configs=[f"{isl}:{osl}"],
            # A kernel-lane re-benchmark, not a dispatched action: it borrows the
            # ``integrate`` workspace namespace under a task id that names it,
            # as the stack re-validation and the GEAK integrate rebench do.
            output_root=unique_runs_dir(self.session_dir, "integrate", "revalidate_geak"),
            variant_timeout_sec=timeout,
            repeats=3,
            # Single-point validated replay pins the headline protocol (num_prompts
            # etc.) so it is protocol-identical to the reported result.
            pin_num_prompts=True,
            # The axis the handoff sent GEAK, so the replay measures what GEAK accepted on.
            metric_axis=geak_metric_axis(benchmark_mode=mode, grading=self.shared_state.grading or None),
        )
        measurement = res.get("promotion_measurement")
        measurement = measurement if isinstance(measurement, Mapping) else {}
        baseline_accuracy = float(self.shared_state.baseline_accuracy or 0.0)
        replay_accuracy = measurement.get("accuracy")
        accuracy_failure = ""
        if baseline_accuracy > 0:
            if (
                isinstance(replay_accuracy, bool)
                or not isinstance(replay_accuracy, (int, float))
                or not 0.0 <= replay_accuracy <= 1.0
            ):
                accuracy_failure = EVAL_KIND_ACCURACY_UNAVAILABLE
            elif not accuracy_passed(baseline_accuracy, float(replay_accuracy)):
                accuracy_failure = "accuracy_drop"
        if accuracy_failure:
            self._coord.phase_kernel.reject_geak_promotion(
                {
                    **ps,
                    "fallback_result": res,
                    "failure_reason": accuracy_failure,
                    "baseline_accuracy": baseline_accuracy,
                },
                measured_tput=_geak_sweep_measured_tput(res) or 0.0,
                current_best_tput=float((self.shared_state.current_best or {}).get("tput") or 0.0),
                reason=accuracy_failure,
            )
            self.shared_state.save(self.session_dir)
            return {"validated": False, "status": "no_promote", "reason": accuracy_failure}
        if str(res.get("status") or "") == "succeeded" and geak_sp > 1.0:
            # Rebench-first: write the headline from the GEAK-harness MEASURED
            # throughput (engages by construction via the launch-script replay),
            # keeping the leaderboard number a same-harness total rather than a
            # self-reported speedup.
            measured = _geak_sweep_measured_tput(res)
            if measured is None:
                log.warning("geak 2a: succeeded sweep but no measurable throughput; candidate stays pending")
                return {"validated": False, "status": res.get("status"), "reason": reason}
            # The replay proves the config engaged. It does not prove the
            # overlay did: the overlay has to still be loadable, and still be
            # the same overlay, at the moment the replay ran.
            overlay_loaded_2a = bool(ps_overlay_2a) and geak_overlay_is_loadable(ps_overlay_2a)
            if overlay_loaded_2a and overlay_digest_before:
                overlay_loaded_2a = _geak_overlay_digest(ps_overlay_2a) == overlay_digest_before
            if ps_overlay_2a and not overlay_loaded_2a:
                log.warning(
                    "geak 2a: overlay %r is not loadable evidence "
                    "(loadable=%r digest before=%r after=%r) -> gain credited "
                    "to config, not to a kernel",
                    ps_overlay_2a,
                    geak_overlay_is_loadable(ps_overlay_2a),
                    overlay_digest_before,
                    _geak_overlay_digest(ps_overlay_2a),
                )
            accepted = self._coord.phase_kernel.promote_geak_from_candidate(
                ps,
                measured_tput=measured,
                provenance="geak_same_harness_geak",
                overlay_loaded=overlay_loaded_2a,
                measurement_provenance=measurement or res,
            )
            try:
                self.shared_state.save(self.session_dir)
            except Exception:
                log.exception("geak 2a: SharedState.save failed")
            if not accepted:
                return {
                    "validated": False,
                    "status": "no_promote",
                    "reason": str(self.shared_state.geak_result.get("revalidation_error") or "promotion_rejected"),
                }
            gain_out = float(self.shared_state.cumulative_gain_validated)
            return {"validated": True, "gain": gain_out, "reason": reason}
        if res.get("error"):
            reason = str(res["error"])
        log.warning(
            "geak 2a fallback did not validate (status=%r geak_speedup=%r reason=%s accuracy_failure=%s)",
            res.get("status"),
            geak_sp,
            reason,
            accuracy_failure,
        )
        return {
            "validated": False,
            "status": "failed" if accuracy_failure else res.get("status"),
            "reason": accuracy_failure or reason,
        }

    async def _replay_via_geak_harness(
        self,
        result: dict,
        *,
        task: "Task",
        rebench: _GeakRebench,
    ) -> _ReplayRuling:
        """Replay an inconclusive 2b revalidation through the GEAK harness (2a).

        Best-effort: a successful replay clears the pending flag and rules the
        round ``validated``; anything else leaves the round as the 2b rebench
        found it.
        """
        measured, got_hash = rebench.measured, rebench.got_hash
        promoted = False
        decision = "fallback"
        log.warning(
            "geak 2b revalidation inconclusive (measured=%r got_hash=%r expected=%r) -> GEAK-harness 2a fallback",
            measured,
            got_hash,
            (task.params or {}).get("expected_cfg_hash"),
        )
        fallback_result: dict[str, Any]
        try:
            # Routed via ``_coord`` so a test / caller that overrides
            # ``coordinator._validate_geak_via_geak_harness`` still wins
            # (bare-name delegation resolves it back onto this class).
            fallback_result = await self.validate_geak_via_geak_harness(reason="2b_inconclusive")
        except Exception as exc:
            log.exception("geak 2a GEAK-harness fallback failed")
            fallback_result = {
                "validated": False,
                "reason": repr(exc),
            }
        if not bool(fallback_result.get("validated")):
            from ..phases.geak_rebench import INCOMPARABLE_REVALIDATION

            geak_result = dict(self.shared_state.geak_result) if isinstance(self.shared_state.geak_result, dict) else {}
            # 2a reports a refusal it will repeat for this workload
            # as a typed status. Persist it beside the verdict: the
            # reason text is for the report, and a later KERNEL entry
            # needs to know the replay is settled, not merely broken.
            refusal = str(fallback_result.get("status") or "")
            geak_result["revalidation_status"] = "no_promote" if refusal == "no_promote" else "fallback_failed"
            geak_result["revalidation_error_class"] = refusal
            geak_result["revalidation_error"] = str(
                fallback_result.get("reason") or refusal or "GEAK harness fallback did not validate"
            )[:500]
            self.shared_state.geak_result = geak_result
            self._record_geak_rebench_conclusion(
                final_status=geak_result["revalidation_status"],
                final_error=str(geak_result.get("revalidation_error") or ""),
            )
            self.shared_state.geak_pending = {}
            result["status"] = refusal if refusal in {INCOMPARABLE_REVALIDATION, "no_promote"} else "failed"
        else:
            promoted = True
            decision = "validated"
        return _ReplayRuling(decision=decision, promoted=promoted)
