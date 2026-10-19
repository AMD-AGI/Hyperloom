# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The GEAK 2a fallback: replay the win through GEAK's own end-to-end harness.

Reached only when the orchestrator's own 2b rebench cannot rule on a GEAK win.
Replaying GEAK's launch script reproduces the accepted configuration by
construction, which is what makes a succeeded replay engagement proof.
"""

from __future__ import annotations

import logging as _logging
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple

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
from ._geak_revalidation import FALLBACK, VALIDATED, _GeakRebench, _ReplayRuling

if TYPE_CHECKING:
    from ..state.task_registry import Task

log = _logging.getLogger(__name__)


class _ReplayRegime(NamedTuple):
    """The single operating point a 2a replay re-measures at."""

    conc: int
    isl: int
    osl: int


class _ReplayOverlay(NamedTuple):
    """The kernel overlay a 2a replay expected, as it stood before the replay ran."""

    path: str
    digest_before: str


def _promotion_measurement(res: Mapping[str, Any]) -> Mapping[str, Any]:
    """The measurement a GEAK sweep promotes on, or an empty mapping when it carries none."""
    measurement = res.get("promotion_measurement")
    return measurement if isinstance(measurement, Mapping) else {}


def _geak_own_speedup(accepted: Mapping[str, Any]) -> float:
    """GEAK's own within-harness speedup, read on the basis it promoted on.

    It serves only to sanity-check that GEAK claimed a win before the replay
    measurement becomes the headline. Older ``result.json`` files carry no
    ``throughput_speedup``, so the explicit within-GEAK ratios stand in,
    preferring the promoted basis.
    """
    try:
        speedup = float(accepted.get("throughput_speedup") or 0.0)
    except (TypeError, ValueError):
        speedup = 0.0
    if speedup > 0:
        return speedup
    metrics = accepted.get("alignment_metrics") or {}
    basis = str(metrics.get("final_basis") or accepted.get("final_throughput_basis") or "hot")
    fallback_key = "cold_geak_speedup" if basis == "cold" else "hot_geak_speedup"
    try:
        return float(metrics.get(fallback_key) or metrics.get("hot_geak_speedup") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _replay_regime(accepted: Mapping[str, Any]) -> _ReplayRegime:
    """The operating point a 2a replay re-measures, falling back to the canonical one."""
    regimes = accepted.get("validated_regimes") or []
    first = regimes[0] if regimes and isinstance(regimes[0], dict) else {}
    try:
        return _ReplayRegime(
            conc=int(first.get("conc") or 64),
            isl=int(first.get("isl") or 1024),
            osl=int(first.get("osl") or 1024),
        )
    except (TypeError, ValueError):
        return _ReplayRegime(conc=64, isl=1024, osl=1024)


def _overlay_survived_replay(overlay: _ReplayOverlay) -> bool:
    """Whether the kernel overlay was still installable, and still the same one, at replay time.

    A succeeded replay proves the configuration engaged; it proves nothing
    about the overlay, which is a separate artifact on a path in
    ``result.json`` and can be gone or inert by then. Of 64 observed runs
    declaring a ``final_overlay``, 25 named a directory that did not exist and
    30 named one holding no ``sitecustomize.py``, so a non-empty path is not
    evidence that a kernel ran.
    """
    if not overlay.path or not geak_overlay_is_loadable(overlay.path):
        return False
    return not overlay.digest_before or _geak_overlay_digest(overlay.path) == overlay.digest_before


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
        from ..actions.executors._workload_envs import agentx_enabled

        mode = str(self.shared_state.benchmark_mode or "").strip()
        agentx = is_agentx_mode(mode) if mode else agentx_enabled()
        if agentx:
            return {
                "validated": False,
                "status": "incomparable",
                "reason": "geak_harness_unsupported_canonical_workload",
            }
        accepted = self.shared_state.geak_result if isinstance(self.shared_state.geak_result, dict) else {}
        if str(accepted.get("status") or "") != "ok" and not _geak_has_accepted_kernel(accepted):
            return {"validated": False, "skipped": True, "reason": "no_geak_result"}
        # The overlay's identity is captured before the replay so it can be
        # compared after; see :func:`_overlay_survived_replay`.
        overlay_path = _normalize_geak_overlay_dir(str(accepted.get("final_overlay") or "").strip())
        overlay = _ReplayOverlay(
            path=overlay_path,
            digest_before=_geak_overlay_digest(overlay_path) if overlay_path else "",
        )
        geak_speedup = _geak_own_speedup(accepted)
        res = await self._replay_through_geak(accepted, mode)
        accuracy_failure = self._replay_accuracy_failure(_promotion_measurement(res))
        if accuracy_failure:
            self._refuse_replay_on_accuracy(accepted, res, accuracy_failure)
            return {"validated": False, "status": "no_promote", "reason": accuracy_failure}
        if str(res.get("status") or "") == "succeeded" and geak_speedup > 1.0:
            return self._promote_replayed_geak_win(accepted, res, overlay, reason=reason)
        if res.get("error"):
            reason = str(res["error"])
        log.warning(
            "geak 2a fallback did not validate (status=%r geak_speedup=%r reason=%s)",
            res.get("status"),
            geak_speedup,
            reason,
        )
        return {"validated": False, "status": res.get("status"), "reason": reason}

    async def _replay_through_geak(self, accepted: Mapping[str, Any], mode: str) -> dict[str, Any]:
        """Re-run an accepted GEAK win through GEAK's own end-to-end harness."""
        from hyperloom.common.jsonio import read_json
        from hyperloom.inference_optimizer.session.session_paths import unique_runs_dir
        from ..actions.executors._geak_sweep import sweep_via_geak
        from ..actions.executors._workload_envs import geak_metric_axis

        regime = _replay_regime(accepted)
        return await sweep_via_geak(
            result=accepted,
            handoff=read_json(self.session_dir / "geak" / "handoff.json", default={}, require_dict=True),
            env_spec=self.build_env_spec(),
            conc_values=[regime.conc],
            isl_osl_configs=[f"{regime.isl}:{regime.osl}"],
            # A kernel-lane re-benchmark, not a dispatched action: it borrows the
            # ``integrate`` workspace namespace under a task id that names it,
            # as the stack re-validation and the GEAK integrate rebench do.
            output_root=unique_runs_dir(self.session_dir, "integrate", "revalidate_geak"),
            variant_timeout_sec=env_int("SWEEP_VARIANT_TIMEOUT_SEC", default=2400),
            repeats=3,
            # Single-point validated replay pins the headline protocol (num_prompts
            # etc.) so it is protocol-identical to the reported result.
            pin_num_prompts=True,
            # The axis the handoff sent GEAK, so the replay measures what GEAK accepted on.
            metric_axis=geak_metric_axis(benchmark_mode=mode, grading=self.shared_state.grading or None),
        )

    def _replay_accuracy_failure(self, measurement: Mapping[str, Any]) -> str:
        """Why the replay's accuracy disqualifies the win, or ``""`` when it does not.

        A session that never measured a baseline accuracy has nothing to hold
        the replay to.
        """
        baseline_accuracy = float(self.shared_state.baseline_accuracy or 0.0)
        if baseline_accuracy <= 0:
            return ""
        replay_accuracy = measurement.get("accuracy")
        if (
            isinstance(replay_accuracy, bool)
            or not isinstance(replay_accuracy, (int, float))
            or not 0.0 <= replay_accuracy <= 1.0
        ):
            return EVAL_KIND_ACCURACY_UNAVAILABLE
        if not accuracy_passed(baseline_accuracy, float(replay_accuracy)):
            return "accuracy_drop"
        return ""

    def _refuse_replay_on_accuracy(
        self,
        accepted: Mapping[str, Any],
        res: Mapping[str, Any],
        failure: str,
    ) -> None:
        """Close the candidate out on the accuracy its 2a replay measured."""
        self._coord.phase_kernel.reject_geak_promotion(
            {
                **accepted,
                "fallback_result": res,
                "failure_reason": failure,
                "baseline_accuracy": float(self.shared_state.baseline_accuracy or 0.0),
            },
            measured_tput=_geak_sweep_measured_tput(res) or 0.0,
            current_best_tput=float((self.shared_state.current_best or {}).get("tput") or 0.0),
            reason=failure,
        )
        self.shared_state.save(self.session_dir)

    def _promote_replayed_geak_win(
        self,
        accepted: Mapping[str, Any],
        res: Mapping[str, Any],
        overlay: _ReplayOverlay,
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Write the headline a succeeded 2a replay earns; report what it settled.

        The leaderboard number is the measured same-harness total rather than
        GEAK's self-reported speedup, because the launch-script replay engages
        the optimized config by construction.
        """
        measured = _geak_sweep_measured_tput(res)
        if measured is None:
            log.warning("geak 2a: succeeded sweep but no measurable throughput; candidate stays pending")
            return {"validated": False, "status": res.get("status"), "reason": reason}
        overlay_loaded = _overlay_survived_replay(overlay)
        if overlay.path and not overlay_loaded:
            log.warning(
                "geak 2a: overlay %r is not loadable evidence "
                "(loadable=%r digest before=%r after=%r) -> gain credited "
                "to config, not to a kernel",
                overlay.path,
                geak_overlay_is_loadable(overlay.path),
                overlay.digest_before,
                _geak_overlay_digest(overlay.path),
            )
        promoted = self._coord.phase_kernel.promote_geak_from_candidate(
            accepted,
            measured_tput=measured,
            provenance="geak_same_harness_geak",
            overlay_loaded=overlay_loaded,
            measurement_provenance=_promotion_measurement(res) or res,
        )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:
            log.exception("geak 2a: SharedState.save failed")
        if not promoted:
            return {
                "validated": False,
                "status": "no_promote",
                "reason": str(self.shared_state.geak_result.get("revalidation_error") or "promotion_rejected"),
            }
        return {"validated": True, "gain": float(self.shared_state.cumulative_gain_validated), "reason": reason}

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
        decision = FALLBACK
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
            decision = VALIDATED
        return _ReplayRuling(decision=decision, promoted=promoted)
