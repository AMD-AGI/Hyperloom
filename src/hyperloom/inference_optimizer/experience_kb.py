# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Best-effort Experience service reads for the Framework orchestration and specialist prompts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from hyperloom.common.perf_metric import agentx_active
from hyperloom.inference_optimizer.experience_kb_service import REQUEST_TIMEOUT_SECONDS, mapping_schema_ref
from hyperloom_kb import ConfigurationError, ReadQuery, RemoteClient, RemoteClientError, RemoteConfig
from hyperloom_kb.collect import MappingError

log = logging.getLogger(__name__)

_FRAMEWORK_DECISION = "Select the next framework optimization to benchmark."
_SPECIALIST_DECISION = "Propose framework optimizations for this specialist investigation."
# The injected records, rendered whole, stop before this many characters in every prompt that carries them.
RENDER_BUDGET_CHARS = 40_000
# A service or planner gateway this many reads in a row could not answer stays unread for the session, so a hung
# gateway costs a run a few read timeouts rather than one per orchestration turn and specialist dispatch.
READS_OFF_AFTER_FAILURES = 3


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _current_best_throughput(current_best: dict[str, Any]) -> float:
    for name in ("tput", "output_throughput"):
        throughput = current_best.get(name)
        if not isinstance(throughput, bool) and isinstance(throughput, (int, float)) and throughput > 0:
            return float(throughput)
    return 0.0


@dataclass(frozen=True)
class ExperienceKBEvidence:
    tick: int
    read_id: str
    status: str
    prompt_block: str
    rendered_refs: tuple[dict[str, str], ...]
    experiences: tuple[dict[str, Any], ...] = ()


class ExperienceKBIntegration:
    """One fail-open Experience service client and decision-context read cache."""

    def __init__(self, client: Any, schema_ref: str) -> None:
        self.client = client
        # The service may hold several schemas; a run reads the one it writes.
        self.schema_ref = schema_ref
        self._cache_tick: int | None = None
        self._by_context: dict[str, ExperienceKBEvidence] = {}
        self._failed_in_a_row = 0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ExperienceKBIntegration | None:
        values = os.environ if env is None else env
        try:
            config = RemoteConfig.from_env(values, timeout_seconds=REQUEST_TIMEOUT_SECONDS)
            if config is None:
                return None
            schema_ref = mapping_schema_ref()
        except (RemoteClientError, ConfigurationError, MappingError):
            log.exception("Experience service configuration is invalid; reads are disabled")
            return None
        return cls(RemoteClient(config), schema_ref)

    def _context(self, state: Any, observations: dict[str, str]) -> dict[str, Any]:
        identity_sources = {
            "model": getattr(state, "model_name", None),
            "gpu": getattr(state, "gpu_type", None),
            "framework": getattr(state, "framework", None),
            "model_type": getattr(state, "model_type", None),
            "framework_version": getattr(state, "framework_version", None),
            "precision": getattr(state, "precision", None),
        }
        identity = {name: _json_safe(value) for name, value in identity_sources.items() if value not in (None, "")}
        workload: dict[str, Any] = {}
        for name in ("tp", "ep", "conc", "isl", "osl", "max_model_len"):
            value = getattr(state, name, None)
            if value not in (None, ""):
                workload[name] = _json_safe(value)
        baseline_tput = float(getattr(state, "baseline_tput", 0.0) or 0.0)
        current_best = getattr(state, "current_best", None) or {}
        current_best_tput = _current_best_throughput(current_best)
        return {
            "identity": identity,
            "workload": workload,
            "objective": {
                "id": "e2e_throughput@v1",
                "direction": "higher_is_better",
            },
            "benchmark_baseline": {
                "throughput": baseline_tput,
                "source": "original_recipe_measurement",
            },
            "current_best": {
                "configuration": _json_safe(current_best),
                "throughput": current_best_tput,
            },
            "observations": observations,
        }

    def build_context(self, state: Any, untested_proposals: str = "") -> dict[str, Any]:
        observations: dict[str, str] = {}
        bottleneck = state.current_top_bottleneck().strip()
        if bottleneck:
            observations["bottleneck"] = bottleneck
        if untested_proposals:
            observations["untested_directions"] = untested_proposals
        return self._context(state, observations)

    def build_specialist_context(self, state: Any, params: dict[str, Any]) -> dict[str, Any]:
        observations: dict[str, str] = {}
        bottleneck = state.current_top_bottleneck().strip()
        if bottleneck:
            observations["bottleneck"] = bottleneck
        pr_lead = params.get("pr_lead")
        for name, value in (
            ("specialist_domain", params.get("domain")),
            ("investigation", params.get("gap_symptom")),
            ("task", params.get("task_description")),
            ("upstream_pr", pr_lead.get("title") if isinstance(pr_lead, dict) else None),
        ):
            text = str(value or "").strip()
            if text:
                observations[name] = text
        return self._context(state, observations)

    def _read(self, state: Any, decision: str, context: dict[str, Any]) -> ExperienceKBEvidence:
        tick = int(getattr(state, "tick", 0) or 0)
        context_hash = hashlib.sha256(
            json.dumps(
                {"decision": decision, "context": context},
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        if tick != self._cache_tick:
            self._by_context.clear()
            self._cache_tick = tick
        cached = self._by_context.get(context_hash)
        if cached is not None:
            return cached
        if self._failed_in_a_row >= READS_OFF_AFTER_FAILURES:
            return ExperienceKBEvidence(
                tick=tick,
                read_id="",
                status="unavailable",
                prompt_block="",
                rendered_refs=(),
            )
        result = self.client.read(
            ReadQuery(decision, context, schema_ref=self.schema_ref, render_budget_chars=RENDER_BUDGET_CHARS)
        )
        self._failed_in_a_row = 0 if result.status == "completed" else self._failed_in_a_row + 1
        if self._failed_in_a_row == READS_OFF_AFTER_FAILURES:
            log.warning(
                "Experience KB reads are off for the rest of this session after %d failed reads: %s",
                READS_OFF_AFTER_FAILURES,
                "; ".join(result.warnings) or result.status,
            )
        evidence = ExperienceKBEvidence(
            tick=tick,
            read_id=result.read_id,
            status=result.status,
            prompt_block=result.prompt_block,
            rendered_refs=tuple(item.to_dict() for item in result.rendered_refs),
            experiences=tuple(dict(item) for item in result.experiences),
        )
        self._by_context[context_hash] = evidence
        return evidence

    def read_for_framework(
        self,
        state: Any,
        *,
        untested_proposals: str = "",
    ) -> ExperienceKBEvidence:
        return self._read(state, _FRAMEWORK_DECISION, self.build_context(state, untested_proposals))

    def read_for_specialist(self, state: Any, params: dict[str, Any]) -> ExperienceKBEvidence:
        return self._read(state, _SPECIALIST_DECISION, self.build_specialist_context(state, params))


def _reads_off_reason(state: Any) -> str:
    """Why this run must not read Experiences, or ``""``: it reads only what the mapping would publish of it."""
    from hyperloom.common.perf_metric import GRADED_OUTPUT
    from hyperloom.inference_optimizer.breakdown.session_facts import grading_block

    # The schema cannot tell an agentic workload from a synthetic one: the mapping publishes no AgentX Experience,
    # so an AgentX run must not read the synthetic ones either.
    if agentx_active(benchmark_mode=getattr(state, "benchmark_mode", "")):
        return "its schema cannot represent an AgentX workload"
    objective = str(grading_block(state).get("objective") or "")
    if objective != GRADED_OUTPUT:
        return f"it is graded on {objective or 'no recorded objective'}, and Experiences record throughput only"
    return ""


def integration_for(state: Any) -> ExperienceKBIntegration | None:
    """Build the session's Experience service integration, or None when this run must not read Experiences."""
    if reason := _reads_off_reason(state):
        log.info("Experience KB reads are off for this run: %s", reason)
        return None
    return ExperienceKBIntegration.from_env()


__all__ = ["ExperienceKBEvidence", "ExperienceKBIntegration", "integration_for"]
