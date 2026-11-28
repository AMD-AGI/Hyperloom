# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Eval-failure routing, the accuracy stop and the AgentX accuracy round of ``BenchmarkRunExecutor``.

:func:`run_accuracy_round` is the one entry to the D9 accuracy round of a slot, which the baseline runs here once its
throughput round succeeded, and a candidate gate through :func:`run_candidate_accuracy_round` once its throughput
cleared the KEEP threshold. Every gate, the baseline's included, reads the round's score through
``parse_eval_results``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.agentx_workload import is_mlperf_backend
from hyperloom.common.perf_metric import agentx_active
from hyperloom.inference_optimizer.trace.task_progress import heartbeat_while_output_flows
from inference_suite.contracts import FailedStep
from inference_suite.server_log import death_excerpt
from ...stop_attribution import stopped_by_the_run_class
from .._accuracy_gate import (
    ACCURACY_DEFINITION_DIGEST_KEY,
    EVAL_KIND_QUALITY_GATE_MISSING,
    parse_eval_results,
)
from .._failed_step import failed_step_error_class
from ..grid._grid_inputs import session_grid_bounds
from .._subprocess_kill import resolve_benchmark_timeouts
from .._suite.rounds import AccuracyRound, SuiteRoundRefused, deadline_unix, run_accuracy_suite_round, session_route
from .records import (
    ACCURACY_EVAL_KIND_KEY,
    ACCURACY_STAGE_KEY,
    _BENCHMARK_MODE_ATTR,
    _DOUBLE_RUN_ROUND_DIRS,
    _iter_log_tails,
    _requested_framework,
    _should_establish_quality_ref,
    record_eval_score,
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
#: The result key a slot's accuracy round is recorded under.
ACCURACY_ROUND_KEY = "agentx_accuracy_round"
_ACCURACY_ROUND_FAILED = "agentx_accuracy_round_failed"


@dataclass(frozen=True)
class AccuracyRoundOutcome:
    """What a slot's accuracy round left: where it ran, and the error class it ended on, ``""`` when it completed."""

    run_dir: Path
    error_class: str

    def to_record(self) -> dict[str, str]:
        return {"run_dir": str(self.run_dir), "error_class": self.error_class}


def _native_agentx(state: Any) -> bool:
    return agentx_active(benchmark_mode=getattr(state, _BENCHMARK_MODE_ATTR, "")) and not is_mlperf_backend()


def accuracy_round_due(state: Any) -> bool:
    """Whether the session ``state`` runs the D9 accuracy round: native AgentX outside MLPerf, and not opted out."""
    return _native_agentx(state) and state.agentx_accuracy


async def run_accuracy_round(request: AccuracyRound, *, session: Path) -> AccuracyRoundOutcome:
    """Run the D9 accuracy round ``request`` names into ``<slot>/accuracy``.

    The round serves the slot's launch on a server of its own with the base's golden layer left out, and runs only the
    eval.
    """
    slot = request.slot
    route = session_route(session)
    if route is None:
        log.error("accuracy_round: %s froze no base launch to serve the accuracy round on", session)
        return AccuracyRoundOutcome(slot, failed_step_error_class(FailedStep.CONFIG_INVALID))
    try:
        async with heartbeat_while_output_flows(unit="accuracy_round", label="eval") as activity:
            ran = await asyncio.to_thread(run_accuracy_suite_round, route, request, on_poll=activity.note)
    except SuiteRoundRefused as exc:
        log.error("accuracy_round: refused before it ran: %s", exc)
        return AccuracyRoundOutcome(slot, exc.error_class)
    step = ran.report.failed_step
    return AccuracyRoundOutcome(ran.layout.root, "" if step is None else failed_step_error_class(step))


async def run_candidate_accuracy_round(slot: Path, launch: Path, state: Any, *, session: Path) -> AccuracyRoundOutcome:
    """Run the accuracy round of a candidate whose throughput cleared the KEEP threshold, capped as a benchmark round.

    ``launch`` is the AgentX point the candidate ran in ``slot``.
    """
    timeout_sec = resolve_benchmark_timeouts()[1]
    deadline = deadline_unix(session_grid_bounds(state)[0])
    return await run_accuracy_round(AccuracyRound(launch, slot, timeout_sec, deadline), session=session)


def record_accuracy_round(result: dict[str, Any], outcome: AccuracyRoundOutcome) -> None:
    """Record ``outcome`` in ``result``: a round that failed leaves a warning, never a verdict of its own."""
    result[ACCURACY_ROUND_KEY] = outcome.to_record()
    if outcome.error_class:
        log.warning("accuracy_round: the AgentX accuracy round ended %s in %s", outcome.error_class, outcome.run_dir)
        result.setdefault("nonfatal_warnings", []).append(_ACCURACY_ROUND_FAILED)


class _BaselineAccuracyMixin:
    """Eval-failure routing and the accuracy stop of a benchmark run."""

    @staticmethod
    def _failure_carries_markers(result: dict[str, Any], markers: tuple[str, ...]) -> bool:
        """Whether a failed result's error tail, warnings or logs hit a marker."""

        def _hit(text: str) -> bool:
            return any(m in text for m in markers)

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

    async def _run_with_accuracy(self, ctx: RunnerContext, *, recorder: Any = None) -> dict[str, Any]:
        """Run the baseline and its accuracy round, and rule on the accuracy the attempt left."""
        result = await self._run_pass(ctx, recorder=recorder)
        if self._eval_failed(result) and self._ruled_on_failed_eval(ctx, result):
            return result
        await self._run_baseline_accuracy_round(ctx, result)
        self._maybe_stop_on_missing_baseline_accuracy(ctx, result)
        return result

    def _eval_failed(self, result: dict[str, Any]) -> bool:
        """Whether the attempt's own eval failed: it failed the round, or it failed after a bench the result keeps."""
        if result.get("status") != "succeeded":
            return self._is_eval_rooted_failure(result)
        return (result.get(ACCURACY_STAGE_KEY) or {}).get("error_class") == FailedStep.EVAL_FAILED.value

    def _ruled_on_failed_eval(self, ctx: RunnerContext, result: dict[str, Any]) -> bool:
        """Rule on an attempt whose eval failed; False for an eval that never reached the server.

        An eval that never reached a verdict because the server was gone is a broken measurement, not a statement
        about accuracy or about a missing framework capability. Ruling on it as an eval failure hands it to the
        enablement lane, which then spends rounds looking for a capability gap the evidence does not support. A
        failed round stays an ordinary baseline failure, so the ``_BASELINE_MAX_TOTAL_FAILURES`` backstop ends the
        run with the cause it had; a bench that stands is ruled like any result without a score.
        """
        if self._is_server_unreachable_eval_failure(result):
            log.warning(
                "baseline_executor: the accuracy eval failed because the server was unreachable, not because "
                "the eval reached a verdict; it is not ruled on as an eval failure."
            )
            result.setdefault("nonfatal_warnings", []).append("eval_failed_server_unreachable")
            return False
        if self._eval_enablement_active(ctx):
            from .._accuracy_gate import EVAL_KIND_RUNTIME_FAILURE

            log.warning("baseline_executor: eval-rooted failure; routing to enablement (RUN_EVAL stays on).")
            evidence = (result.get(ACCURACY_STAGE_KEY) or {}).get("error") or self._eval_failure_evidence(result)[1]
            self._stamp_eval_failure_contract(
                result, kind=EVAL_KIND_RUNTIME_FAILURE, observed_accuracy=None, evidence=evidence
            )
            return True
        result.setdefault("nonfatal_warnings", [])
        result["accuracy_source"] = "eval_unavailable"
        if _should_establish_quality_ref(getattr(ctx.task, "kind", ""), ctx.task.params or {}):
            log.error(
                "baseline_executor: the accuracy eval failed on a genuine baseline, whose whole purpose is to "
                "establish the accuracy reference. Stopping now -- fix the accuracy eval (see the round's eval "
                "output for the error)."
            )
            result["nonfatal_warnings"].append("eval_failed_no_fallback_baseline_requires_accuracy")
            self._request_eval_rooted_baseline_stop(ctx)
        else:
            log.error("baseline_executor: the accuracy eval failed; the attempt is not run again without it.")
        return True

    async def _run_baseline_accuracy_round(self, ctx: RunnerContext, result: dict[str, Any]) -> None:
        """Run the accuracy round of a genuine native AgentX baseline that succeeded, unless the session opted out.

        It runs after the attempt released its server, so the round's own server has the GPUs to itself.
        """
        params = ctx.task.params or {}
        if result.get("status") != "succeeded" or not _should_establish_quality_ref(
            getattr(ctx.task, "kind", ""), params
        ):
            return
        state = self._resolve_shared_state(ctx.extra.get("shared_state"))
        if not accuracy_round_due(state):
            return
        slot = Path(result["output_dir"])
        if slot.name in _DOUBLE_RUN_ROUND_DIRS:
            slot = slot.parent
        request = AccuracyRound(
            Path(result["materialized_config"]),
            slot,
            self._resolve_timeout(params),
            deadline_unix(session_grid_bounds(state)[0]),
        )
        record_accuracy_round(result, await run_accuracy_round(request, session=self.session_dir))
        eval_data = parse_eval_results(slot, _requested_framework(params), state.benchmark_mode)
        if not record_eval_score(result, eval_data):
            log.warning("baseline_executor: the accuracy round left no score: %s", eval_data.get("error"))

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
        # ``death_excerpt`` already answers ``None`` for a log it cannot read,
        # so an unreadable path is simply "no death on record" -- no guard of our own.
        return death_excerpt(path) or ""

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

    @staticmethod
    def _stamp_eval_failure_contract(
        result: dict[str, Any],
        *,
        kind: str,
        observed_accuracy: float | None,
        evidence: str,
    ) -> dict[str, Any]:
        """Mark ``result`` as an eval-rooted baseline failure for enablement."""
        from .._accuracy_gate import (
            BASELINE_EVAL_ACCURACY_FLOOR_KEY,
            BASELINE_EVAL_EVIDENCE_KEY,
            BASELINE_EVAL_FAILED_KEY,
            BASELINE_EVAL_FAILURE_KIND_KEY,
            BASELINE_EVAL_OBSERVED_ACCURACY_KEY,
            run_eval_definition_digest,
        )
        from hyperloom.common.gain_math import DEFAULT_ENABLEMENT_ACCURACY_FLOOR

        floor = DEFAULT_ENABLEMENT_ACCURACY_FLOOR
        result[BASELINE_EVAL_FAILED_KEY] = True
        result[BASELINE_EVAL_FAILURE_KIND_KEY] = kind
        result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] = observed_accuracy
        result[BASELINE_EVAL_ACCURACY_FLOOR_KEY] = floor
        result[BASELINE_EVAL_EVIDENCE_KEY] = (evidence or "")[:4000]
        accuracy_round = result.get(ACCURACY_ROUND_KEY)
        if accuracy_round and not result.get(ACCURACY_DEFINITION_DIGEST_KEY):
            # A round that left no score still planned the definition it failed under.
            result[ACCURACY_DEFINITION_DIGEST_KEY] = run_eval_definition_digest(Path(accuracy_round["run_dir"]))
        result["eval_origin"] = "eval"
        return result

    def _request_eval_rooted_baseline_stop(self, ctx: RunnerContext) -> None:
        """Halt the run for an eval-rooted baseline failure, fail-fast path."""
        from .._accuracy_gate import request_baseline_accuracy_stop

        framework = _requested_framework(ctx.task.params or {})
        extra = getattr(ctx, "extra", None) or {}
        request_baseline_accuracy_stop(
            extra.get("shared_state") or self.shared_state,
            context=f"baseline:{framework or 'unknown'}:eval_aborted",
        )

    def _maybe_stop_on_missing_baseline_accuracy(
        self,
        ctx: RunnerContext,
        result: dict[str, Any],
    ) -> None:
        """Halt the run when a genuine baseline produced no accuracy result.

        A baseline standing on its cold warm-up is not ruled on, nor one whose eval the run itself stopped: the run
        kept the eval from finishing, which says nothing about accuracy.
        """
        params = ctx.task.params
        if result.get("status") != "succeeded" or not _should_establish_quality_ref(
            getattr(ctx.task, "kind", ""), params
        ):
            return
        eval_stop_class = (result.get(ACCURACY_STAGE_KEY) or {}).get("error_class")
        if result.get("measure_round_dropped") or stopped_by_the_run_class(eval_stop_class) is not None:
            return
        shared_state = ctx.extra.get("shared_state") or self.shared_state
        # An AgentX session that opted out of the accuracy round has no reference by design; its gates skip accuracy.
        if _native_agentx(shared_state) and not shared_state.agentx_accuracy:
            return
        acc = result.get("accuracy")
        eval_enablement = self._eval_enablement_active(ctx)
        from hyperloom.common.gain_math import DEFAULT_ENABLEMENT_ACCURACY_FLOOR, accuracy_meets_floor

        from .._accuracy_gate import request_baseline_accuracy_stop

        floor = DEFAULT_ENABLEMENT_ACCURACY_FLOOR
        # Floor 0.0 is the non-enablement "any positive accuracy is usable" rule; ``accuracy_meets_floor`` already
        # means "finite, strictly positive and >= floor", so it rejects zero either way.
        usable_floor = floor if eval_enablement else 0.0
        if accuracy_meets_floor(acc, usable_floor):
            return
        framework = _requested_framework(params)
        context = f"baseline:{framework or 'unknown'}"
        # The operator's script broke the scriptable contract: no specialist can repair it.
        if result.get(ACCURACY_EVAL_KIND_KEY) == EVAL_KIND_QUALITY_GATE_MISSING:
            request_baseline_accuracy_stop(
                shared_state,
                context=f"{context}:{EVAL_KIND_QUALITY_GATE_MISSING}",
                cause=self._quality_gate_missing_cause(result),
            )
            return
        # Route into enablement instead of stopping: the throughput baseline stays for diagnostics but is blocked from
        # anchoring.
        if eval_enablement:
            self._route_baseline_accuracy_to_enablement(ctx, result, acc, floor)
            return
        # Only a *missing* reference can be explained by a dead server. A measured zero means the eval did run and
        # did write its results, so blaming the death here would state the opposite of what the artifacts show,
        # however dead the server later became.
        cause = ""
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

    @staticmethod
    def _quality_gate_missing_cause(result: dict[str, Any]) -> str:
        report = result.get("accuracy_source") or "no round report found"
        return f"the entrypoint's result JSON carries no quality_gate block (report: {report})"

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
