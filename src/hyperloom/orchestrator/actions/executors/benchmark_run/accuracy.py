# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Eval-failure routing and the accuracy stop of ``BenchmarkRunExecutor``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import RUN_AFTER_MOE_RUNNER_FAILURE, RUN_INITIAL
from .._subprocess_kill import server_log_death_excerpt
from .records import (
    _DOUBLE_RUN_ROUND_DIRS,
    _is_double_run_accuracy_handoff,
    _iter_log_tails,
    _requested_framework,
    _should_establish_quality_ref,
)

if TYPE_CHECKING:
    from ....loop.sub_agent_runner import RunnerContext

log = logging.getLogger(__name__)

# Markers identifying an InferenceX ``run_eval`` (lm-eval) failure as the root cause of a benchmark non-zero exit.
_EVAL_FAILURE_MARKERS = (
    "run_eval failed with exit code",
    "ERROR: run_eval failed",
    "Unknown parameter: --concurrent-requests",
)
# Markers showing the eval failed because the server was not reachable, rather than because the model scored badly or
# the framework lacks a capability. The eval harness reports both the same way -- a non-zero ``run_eval`` -- so without
# this distinction a torn-down server is indistinguishable from an accuracy gap, and gets routed to the enablement lane
# to hunt for a capability that was never missing.
_EVAL_SERVER_UNREACHABLE_MARKERS = (
    "ClientConnectorError",
    "Cannot connect to host",
    "Connect call failed",
)
# Bounded per-file read so log scanning never slurps a multi-GB server.log.
_LOG_SCAN_MAX_BYTES = 262_144


class _BaselineAccuracyMixin:
    """Eval-failure routing and the accuracy stop of a benchmark run."""

    @staticmethod
    def _failure_carries_markers(
        result: dict[str, Any],
        markers: tuple[str, ...],
        context_markers: tuple[str, ...] | None = None,
    ) -> bool:
        """Whether a failed result's error tail, warnings or logs hit a marker."""

        def _hit(text: str) -> bool:
            if not any(m in text for m in markers):
                return False
            return not context_markers or any(m in text for m in context_markers)

        if _hit(str(result.get("error") or "")):
            return True
        for w in result.get("nonfatal_warnings") or []:
            if _hit(str(w)):
                return True
        out_dir = result.get("output_dir")
        if not out_dir:
            return False
        root = Path(out_dir)
        # Double-run: the failure markers may live in the sibling warmup round, so climb to the shared task root to
        # scan both rounds.
        if root.name in _DOUBLE_RUN_ROUND_DIRS:
            root = root.parent
        if not root.exists():
            return False
        return any(_hit(tail) for _, tail in _iter_log_tails(root, max_bytes=_LOG_SCAN_MAX_BYTES))

    @staticmethod
    def _eval_failure_evidence(result: dict[str, Any]) -> tuple[bool, str]:
        """Detect an eval-rooted baseline failure and capture bounded evidence."""

        def _window(text: str) -> str | None:
            for m in _EVAL_FAILURE_MARKERS:
                i = text.find(m)
                if i != -1:
                    return text[max(0, i - 200) : i + len(m) + 400]
            return None

        w = _window(str(result.get("error") or ""))
        if w is not None:
            return True, w
        for warn in result.get("nonfatal_warnings") or []:
            w = _window(str(warn))
            if w is not None:
                return True, w
        out_dir = result.get("output_dir")
        if not out_dir:
            return False, ""
        root = Path(out_dir)
        # Double-run: the failure markers may live in the sibling warmup round, so climb to the shared task root to
        # scan both rounds.
        if root.name in _DOUBLE_RUN_ROUND_DIRS:
            root = root.parent
        if not root.exists():
            return False, ""
        for path, tail in _iter_log_tails(root, max_bytes=_LOG_SCAN_MAX_BYTES):
            window = _window(tail)
            if window is not None:
                return True, f"{path.name}: {window}"
        return False, ""

    @staticmethod
    def _is_server_unreachable_eval_failure(result: dict[str, Any]) -> bool:
        """Whether an eval-rooted failure happened because the server had gone away.

        Evidence of a refused connection says the eval never reached a verdict, so the run carries no information
        about accuracy or about a missing framework capability -- the two things the enablement lane exists to chase.

        Deliberately not routed through ``_failure_carries_markers``: that climbs out of a round directory to the
        shared task root, so a warmup round whose server crashed would decide the classification of a measure round
        that did reach a verdict -- demoting a real accuracy failure out of the enablement lane, the inverse of what
        this check is for. Scan this round's own directory instead. The markers are frequently absent from
        ``result['error']`` -- that is why the eval-rooted check reads logs at all -- and when the eval-rooted verdict
        came from a log, the refusal that caused it is in the same log.
        """
        texts = [str(result.get("error") or "")]
        texts.extend(str(warning) for warning in result.get("nonfatal_warnings") or [])
        if any(marker in text for text in texts for marker in _EVAL_SERVER_UNREACHABLE_MARKERS):
            return True
        out_dir = result.get("output_dir")
        if not out_dir:
            return False
        root = Path(out_dir)
        if not root.is_dir():
            return False
        return any(
            marker in tail
            for _, tail in _iter_log_tails(root, max_bytes=_LOG_SCAN_MAX_BYTES)
            for marker in _EVAL_SERVER_UNREACHABLE_MARKERS
        )

    async def _run_retrying(self, ctx: RunnerContext, *, recorder: Any = None) -> dict[str, Any]:
        """Run the Magpie baseline; an eval-rooted failure fails the round, a MoE-runner failure retries once."""
        result = await self._run_pass(ctx, recorder=recorder, attempt_reason=RUN_INITIAL)
        # An eval that never reached a verdict because the server was gone is a broken measurement, not a statement
        # about accuracy or about a missing framework capability. Leaving it in the eval-rooted branch stamps it as an
        # eval-failure contract and hands it to the enablement lane, which then spends rounds looking for a capability
        # gap that the evidence does not support. It stays a failed baseline either way -- nothing is salvaged and the
        # accuracy gate is untouched -- but it is counted as an ordinary baseline failure, so the existing
        # ``_BASELINE_MAX_TOTAL_FAILURES`` backstop ends the run with the cause it actually had.
        if result.get("status") != "succeeded" and self._is_eval_rooted_failure(result):
            if self._is_server_unreachable_eval_failure(result):
                log.warning(
                    "baseline_executor: the accuracy eval failed because the server was unreachable, not because "
                    "the eval reached a verdict; recording an ordinary baseline failure rather than an enablement "
                    "gap."
                )
                result.setdefault("nonfatal_warnings", []).append("eval_failed_server_unreachable")
            else:
                _, evidence = self._eval_failure_evidence(result)
                if self._eval_enablement_active(ctx):
                    from .._accuracy_gate import EVAL_KIND_RUNTIME_FAILURE

                    log.warning(
                        "baseline_executor: eval-rooted failure; routing to "
                        "enablement instead of salvaging (RUN_EVAL stays on)."
                    )
                    self._stamp_eval_failure_contract(
                        ctx, result, kind=EVAL_KIND_RUNTIME_FAILURE, observed_accuracy=None, evidence=evidence
                    )
                    return result
                result.setdefault("nonfatal_warnings", [])
                result["accuracy_source"] = "eval_unavailable"
                if _should_establish_quality_ref(getattr(ctx.task, "kind", ""), ctx.task.params or {}):
                    log.error(
                        "baseline_executor: failure is eval-rooted (InferenceX run_eval aborted the benchmark) on a "
                        "genuine baseline, whose whole purpose is to establish the accuracy reference. Stopping now "
                        "-- fix the accuracy eval (see the benchmark stdout/stderr for the run_eval error)."
                    )
                    result["nonfatal_warnings"].append("eval_failed_no_fallback_baseline_requires_accuracy")
                    self._request_eval_rooted_baseline_stop(ctx, result)
                else:
                    log.error(
                        "baseline_executor: failure is eval-rooted (InferenceX run_eval aborted the benchmark); "
                        "the round fails with no throughput-only retry."
                    )
                return result
        if result.get("status") != "succeeded" and self._is_moe_runner_rooted_failure(result):
            log.warning(
                "baseline_executor: the server died on a MoE runner backend "
                "that has no implementation for this checkpoint's quant "
                "scheme; retrying once without --moe-runner-backend so the "
                "framework picks the backend itself."
            )
            retry = await self._run_pass(
                ctx,
                recorder=recorder,
                attempt_reason=RUN_AFTER_MOE_RUNNER_FAILURE,
                force_drop_moe_runner_backend=True,
            )
            retry.setdefault("nonfatal_warnings", []).append("moe_runner_backend_fallback_dropped_flag")
            result = retry
        self._maybe_stop_on_missing_baseline_accuracy(ctx, result)
        return result

    @staticmethod
    def _accuracy_stop_death_evidence(result: dict[str, Any]) -> str:
        """The server's dying words, for a round that reached the accuracy stop without a reference.

        A round whose server dies *after* the throughput benchmark still reports
        ``succeeded`` -- the numbers were already measured. The accuracy eval then runs,
        per the round script, against a port nobody is listening on: every request is
        refused, no ``results*.json`` is written, and the session stops for a missing
        accuracy reference under a message that blames the baseline setup, which is the
        one thing that was not wrong.

        The marker this returns proves only that the server died somewhere in that log,
        never that it died before the eval or that it is why the reference is missing --
        so callers must report it as evidence to read, not as an established cause.
        """
        path = str(result.get("server_log_path") or "").strip()
        if not path:
            return ""
        # ``server_log_death_excerpt`` already answers ``None`` for a log it cannot read,
        # so an unreadable path is simply "no death on record" -- no guard of our own.
        return server_log_death_excerpt(path) or ""

    def _eval_enablement_active(self, ctx: RunnerContext) -> bool:
        """Whether an eval failure should route into enablement this run."""
        from .._accuracy_gate import eval_enablement_allowed
        from .._multi_node_env import is_multi_node

        extra = getattr(ctx, "extra", None) or {}
        if not eval_enablement_allowed(self._resolve_shared_state(extra.get("shared_state"))):
            return False
        if is_multi_node():
            return False
        return _should_establish_quality_ref(getattr(ctx.task, "kind", ""), ctx.task.params or {})

    def _stamp_eval_failure_contract(
        self,
        ctx: RunnerContext,
        result: dict[str, Any],
        *,
        kind: str,
        observed_accuracy: float | None,
        evidence: str,
    ) -> dict[str, Any]:
        """Mark ``result`` as an eval-rooted baseline failure for enablement."""
        from .._accuracy_gate import (
            BASELINE_EVAL_ACCURACY_FLOOR_KEY,
            BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY,
            BASELINE_EVAL_EVIDENCE_KEY,
            BASELINE_EVAL_FAILED_KEY,
            BASELINE_EVAL_FAILURE_KIND_KEY,
            BASELINE_EVAL_OBSERVED_ACCURACY_KEY,
            DEFAULT_ENABLEMENT_ACCURACY_FLOOR,
            eval_contract_fingerprint,
        )

        params = ctx.task.params or {}
        framework = _requested_framework(params)
        model = params.get("model") or params.get("resolved_model")
        floor = DEFAULT_ENABLEMENT_ACCURACY_FLOOR
        result[BASELINE_EVAL_FAILED_KEY] = True
        result[BASELINE_EVAL_FAILURE_KIND_KEY] = kind
        result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] = observed_accuracy
        result[BASELINE_EVAL_ACCURACY_FLOOR_KEY] = floor
        result[BASELINE_EVAL_EVIDENCE_KEY] = (evidence or "")[:4000]
        # Fingerprint derives from the materialized YAML contract fields only — task/metric are result outputs and may
        # be absent on eval crash, so they must not participate in the stable identity.
        result[BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY] = eval_contract_fingerprint(
            config_path=result.get("materialized_config"),
            framework=framework,
            model=model,
        )
        result["eval_origin"] = "eval"
        return result

    def _request_eval_rooted_baseline_stop(
        self,
        ctx: RunnerContext,
        result: dict[str, Any],
    ) -> None:
        """Halt the run for an eval-rooted baseline failure, fail-fast path."""
        from .._accuracy_gate import accuracy_meets_floor, request_baseline_accuracy_stop

        params = ctx.task.params or {}
        framework = _requested_framework(params)
        extra = getattr(ctx, "extra", None) or {}
        shared_state = extra.get("shared_state") or self.shared_state
        salvaged = self._salvage_sibling_baseline_accuracy(result, framework)
        if salvaged is not None:
            acc_val = self._apply_salvaged_accuracy(result, salvaged, shared_state)
            # ``accuracy_meets_floor`` already means "finite, strictly positive and >= floor", so floor 0.0 is the
            # legacy "any usable accuracy".
            if accuracy_meets_floor(acc_val, 0.0):
                log.warning(
                    "baseline_executor: eval-rooted baseline failure, but salvaged "
                    "a valid baseline accuracy=%.4f from a sibling attempt (%s); "
                    "not stopping the run",
                    acc_val,
                    salvaged.get("source_file", ""),
                )
                return
            # A measured zero is a broken baseline, not a usable reference: it must still reach the stop below, now
            # with the score on record.
            log.warning(
                "baseline_executor: eval-rooted baseline failure and the sibling "
                "attempt measured accuracy=%.4f (%s); stopping the run",
                acc_val,
                salvaged.get("source_file", ""),
            )
        request_baseline_accuracy_stop(
            shared_state,
            context=f"baseline:{framework or 'unknown'}:eval_aborted",
        )

    def _maybe_stop_on_missing_baseline_accuracy(
        self,
        ctx: RunnerContext,
        result: dict[str, Any],
    ) -> None:
        """Halt the run when a genuine baseline produced no accuracy result."""
        params = ctx.task.params
        if not _should_establish_quality_ref(getattr(ctx.task, "kind", ""), params):
            return
        # A failed status must NOT skip straight past the salvage below.
        if result.get("status") != "succeeded" and not self._is_eval_rooted_failure(result):
            return
        acc = result.get("accuracy")
        eval_enablement = self._eval_enablement_active(ctx)
        from .._accuracy_gate import (
            DEFAULT_ENABLEMENT_ACCURACY_FLOOR,
            accuracy_meets_floor,
            request_baseline_accuracy_stop,
        )

        floor = DEFAULT_ENABLEMENT_ACCURACY_FLOOR
        # Floor 0.0 is the non-enablement "any positive accuracy is usable" rule; ``accuracy_meets_floor`` already
        # means "finite, strictly positive and >= floor", so it rejects zero either way.
        usable_floor = floor if eval_enablement else 0.0
        if accuracy_meets_floor(acc, usable_floor):
            return
        framework = _requested_framework(params)
        shared_state = ctx.extra.get("shared_state") or self.shared_state
        # Session-level salvage (complements #942): the cold-start guard and the coordinator's retries each run in
        # their own ``runs/baseline/<attempt>`` dir. #942 keeps every attempt's eval output inside that attempt's
        # ``$RESULT_DIR``, but the accuracy-stop decision runs on the *deciding* attempt -- whose dir can be empty
        # when a prior sibling attempt already produced a valid ``results*.json``.
        salvaged = self._salvage_sibling_baseline_accuracy(result, framework)
        if salvaged is not None:
            acc_val = self._apply_salvaged_accuracy(
                result,
                salvaged,
                shared_state,
                expected_handoff=_is_double_run_accuracy_handoff(result, salvaged),
            )
            if accuracy_meets_floor(acc_val, usable_floor):
                return
            # Salvaged, but unusable (zero or still under the floor): that is a real quality signal, so fall through
            # with the observed value rather than reporting it as a missing measurement.
            acc = acc_val

        # An eval that never reached a verdict because the server was unreachable produced no accuracy signal, so
        # neither the stop below nor the enablement routing has anything to act on; stamping either would put an
        # accuracy-flavoured verdict on evidence that says only that the measurement broke. But when the salvage
        # above did find a measured accuracy, there IS a signal -- a real one, from a round that reached a verdict --
        # and returning here would throw it away, leaving a genuinely under-floor model anchoring nothing.
        if (
            salvaged is None
            and result.get("status") != "succeeded"
            and self._is_server_unreachable_eval_failure(result)
        ):
            return

        # Route into enablement instead of stopping: the throughput baseline stays for diagnostics but is blocked from
        # anchoring.
        if eval_enablement:
            self._route_baseline_accuracy_to_enablement(ctx, result, acc, floor)
            return
        # Only a *missing* reference can be explained by a dead server. A measured zero -- including one salvaged
        # from a sibling attempt -- means the eval did run and did write its results, so blaming the death here would
        # state the opposite of what the artifacts show, however dead the server later became.
        context, cause = f"baseline:{framework or 'unknown'}", ""
        if acc is None and (death := self._accuracy_stop_death_evidence(result)):
            log.error(
                "baseline_executor: no accuracy reference, and this round's server.log "
                "records a fatal engine death. The marker carries no ordering against "
                "the eval, so the two are not established to be the same failure -- but "
                "a broken setup is no longer the only suspect, and this is the excerpt "
                "to read before looking at any config:\n%s",
                death,
            )
            context, cause = (
                f"{context}:server_died",
                "no accuracy result, and a fatal engine death on record for this round",
            )
        request_baseline_accuracy_stop(shared_state, context=context, cause=cause)

    def _route_baseline_accuracy_to_enablement(
        self,
        ctx: RunnerContext,
        result: dict[str, Any],
        acc: Any,
        floor: float,
    ) -> None:
        """Stamp an under-floor baseline accuracy as an eval failure for enablement to act on."""
        from .._accuracy_gate import (
            EVAL_KIND_GENERATION_PATHOLOGY,
            classify_accuracy_failure,
            eval_probe_summary,
        )

        kind = classify_accuracy_failure(acc, floor)
        evidence = (
            f"baseline accuracy did not meet floor: accuracy={acc} floor={floor} "
            f"task={result.get('accuracy_task')} metric={result.get('accuracy_metric')} "
            f"source={result.get('accuracy_source')}"
        )
        # A tripped probe means the eval was cut short because the model never stopped generating, not that it
        # answered and got them wrong.
        if probe := result.get("eval_probe"):
            kind = EVAL_KIND_GENERATION_PATHOLOGY
            evidence = f"{evidence}; {eval_probe_summary(probe)}"
        self._stamp_eval_failure_contract(
            ctx,
            result,
            kind=kind or "",
            observed_accuracy=float(acc) if isinstance(acc, (int, float)) else None,
            evidence=evidence,
        )
        log.warning(
            "baseline_executor: accuracy %s below floor %.4f (kind=%s); routing to "
            "enablement instead of stopping the run.",
            acc,
            floor,
            kind,
        )

    def _apply_salvaged_accuracy(
        self,
        result: dict[str, Any],
        salvaged: dict[str, Any],
        shared_state: Any,
        *,
        expected_handoff: bool = False,
    ) -> float:
        """Record a salvaged sibling accuracy, publishing it as the gate reference only when it can serve as one."""
        from .._accuracy_gate import accuracy_meets_floor

        acc_val = float(salvaged["accuracy"])
        result["accuracy"] = acc_val
        result["accuracy_task"] = salvaged.get("task", "gsm8k")
        result["accuracy_metric"] = salvaged.get("metric", "")
        result["accuracy_source"] = salvaged.get("source_file", "")
        if expected_handoff:
            log.info(
                "baseline_executor: cold-start guard — reading accuracy=%.4f from "
                "the warmup round (%s), the only round that measures it",
                acc_val,
                salvaged.get("source_file", ""),
            )
        else:
            log.warning(
                "baseline_executor: this attempt's RESULT_DIR had no accuracy, "
                "but salvaged a measured baseline accuracy=%.4f from a sibling "
                "attempt (%s)",
                acc_val,
                salvaged.get("source_file", ""),
            )
            result.setdefault("nonfatal_warnings", []).append("baseline_accuracy_salvaged_from_sibling_attempt")
        if shared_state is not None and accuracy_meets_floor(acc_val, 0.0):
            shared_state.baseline_accuracy = acc_val
        return acc_val

    def _salvage_sibling_baseline_accuracy(
        self,
        result: dict[str, Any],
        framework: str | None,
    ) -> dict[str, Any] | None:
        """Return a measured accuracy from a sibling baseline attempt, if any."""
        out = result.get("output_dir")
        if not out:
            return None
        runs_root = Path(out).parent  # .../runs/baseline
        if not runs_root.exists():
            return None
        try:
            from .._accuracy_gate import _finite_score, parse_eval_results

            eval_data = parse_eval_results(runs_root, framework=framework)
        except (OSError, ValueError, AttributeError, TypeError):
            log.debug("baseline_executor: sibling-accuracy salvage scan failed", exc_info=True)
            return None
        if _finite_score(eval_data.get("accuracy")) is None:
            return None
        return eval_data
