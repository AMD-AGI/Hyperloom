# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Best-effort Experience service reads for the Framework orchestration and specialist prompts."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hyperloom.common.perf_metric import agentx_active
from hyperloom.inference_optimizer.experience_kb_service import REQUEST_TIMEOUT_SECONDS, mapping_schema_ref
from hyperloom_kb import (
    MAX_SIGNAL_TEXT_CHARS,
    ConfigurationError,
    QuerySignal,
    ReadQuery,
    RemoteClient,
    RemoteClientError,
    RemoteConfig,
)
from hyperloom_kb.collect import MappingError

log = logging.getLogger(__name__)

OBJECTIVE = "e2e_throughput@v1"
# How much each signal of a read counts, by weight. An Experience scores the sum of the weights of the signals it
# matches: an exact signal, named by the field it looks up, adds its whole weight; a text signal, named by the
# observation it carries, adds its weight times how well the text matched, which is 1 only for a record holding
# every word of it.
SIGNAL_WEIGHTS: dict[float, tuple[str, ...]] = {
    1.0: ("identity.model",),
    0.8: ("identity.framework",),
    0.6: ("identity.gpu", "bottleneck", "investigation"),
    0.5: ("identity.precision", "identity.model_type", "task"),
    0.4: ("identity.tp", "objective", "untested_directions", "specialist_domain", "upstream_pr"),
    0.3: ("identity.conc", "identity.isl", "identity.osl", "identity.framework_version"),
    0.2: ("identity.ep", "identity.max_model_len"),
}
_WEIGHT_OF = {name: weight for weight, names in SIGNAL_WEIGHTS.items() for name in names}
# The session state attribute holding each identity field of the schema this run reads.
_IDENTITY_ATTRIBUTES = {
    "model": "model_name",
    "gpu": "gpu_type",
    "framework": "framework",
    "model_type": "model_type",
    "framework_version": "framework_version",
    "precision": "precision",
    "tp": "tp",
    "ep": "ep",
    "conc": "conc",
    "isl": "isl",
    "osl": "osl",
    "max_model_len": "max_model_len",
}
# The injected records, rendered whole, stop before this many characters in every prompt that carries them.
RENDER_BUDGET_CHARS = 40_000
# A service this many reads in a row could not answer stays unread for the session, so a hung or stopped service
# costs a run a few read timeouts rather than one per orchestration turn and specialist dispatch.
READS_OFF_AFTER_FAILURES = 3


def read_signals(
    identity: Mapping[str, Any], objective: str, observations: Mapping[str, Any]
) -> tuple[QuerySignal, ...]:
    """The signals of one read: an exact one per identity value and for the objective, a text one per observation.

    Only what ``SIGNAL_WEIGHTS`` weighs is sent, and no value that is unset: empty, or the 0 a session state leaves
    an unset size at. An observation is cut to the longest text a signal carries.
    """

    exact = {**{f"identity.{name}": value for name, value in identity.items()}, "objective": objective}
    signals = [
        QuerySignal(
            field=field,
            value=value if isinstance(value, (str, int, float, bool)) else str(value),
            weight=_WEIGHT_OF[field],
        )
        for field, value in exact.items()
        if field in _WEIGHT_OF and value not in (None, "", 0)
    ]
    for name, value in observations.items():
        text = str(value or "").strip()[:MAX_SIGNAL_TEXT_CHARS]
        if name in _WEIGHT_OF and text:
            signals.append(QuerySignal(text=text, weight=_WEIGHT_OF[name]))
    return tuple(signals)


@dataclass(frozen=True)
class ExperienceKBEvidence:
    tick: int
    read_id: str
    status: str
    prompt_block: str
    rendered_refs: tuple[dict[str, str], ...]
    experiences: tuple[dict[str, Any], ...] = ()


class ExperienceKBIntegration:
    """One fail-open Experience service client and a cache of the tick's reads by their signals."""

    def __init__(self, client: Any, schema_ref: str) -> None:
        self.client = client
        # The service may hold several schemas; a run reads the one it writes.
        self.schema_ref = schema_ref
        self._cache_tick: int | None = None
        self._by_signals: dict[tuple[QuerySignal, ...], ExperienceKBEvidence] = {}
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

    def _read(self, state: Any, observations: Mapping[str, Any]) -> ExperienceKBEvidence:
        tick = int(getattr(state, "tick", 0) or 0)
        identity = {name: getattr(state, attribute, None) for name, attribute in _IDENTITY_ATTRIBUTES.items()}
        signals = read_signals(identity, OBJECTIVE, {"bottleneck": state.current_top_bottleneck(), **observations})
        if tick != self._cache_tick:
            self._by_signals.clear()
            self._cache_tick = tick
        cached = self._by_signals.get(signals)
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
            ReadQuery(signals, schema_ref=self.schema_ref, render_budget_chars=RENDER_BUDGET_CHARS)
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
        self._by_signals[signals] = evidence
        return evidence

    def read_for_framework(
        self,
        state: Any,
        *,
        untested_proposals: str = "",
    ) -> ExperienceKBEvidence:
        return self._read(state, {"untested_directions": untested_proposals})

    def read_for_specialist(self, state: Any, params: dict[str, Any]) -> ExperienceKBEvidence:
        pr_lead = params.get("pr_lead")
        return self._read(
            state,
            {
                "specialist_domain": params.get("domain"),
                "investigation": params.get("gap_symptom"),
                "task": params.get("task_description"),
                "upstream_pr": pr_lead.get("title") if isinstance(pr_lead, dict) else None,
            },
        )


def _reads_off_reason(state: Any) -> str:
    """Why this run must not read Experiences, or ``""``: it reads only what the mapping would publish of it."""
    from hyperloom.common.perf_metric import GRADED_OUTPUT
    from hyperloom.inference_optimizer.breakdown.session_facts import grading_block

    # The schema cannot tell an agentic workload from a synthetic one: the mapping publishes no AgentX Experience,
    # so an AgentX run must not read the synthetic ones either.
    if agentx_active(benchmark_mode=getattr(state, "benchmark_mode", "")):
        return "its schema cannot represent an AgentX workload"
    grading = grading_block(state)
    if grading.get("prompt_dataset"):
        return "its schema cannot represent a prompt dataset"
    objective = str(grading.get("objective") or "")
    if objective != GRADED_OUTPUT:
        return f"it is graded on {objective or 'no recorded objective'}, and Experiences record throughput only"
    return ""


def integration_for(state: Any) -> ExperienceKBIntegration | None:
    """Build the session's Experience service integration, or None when this run must not read Experiences."""
    if reason := _reads_off_reason(state):
        log.info("Experience KB reads are off for this run: %s", reason)
        return None
    return ExperienceKBIntegration.from_env()


__all__ = [
    "OBJECTIVE",
    "SIGNAL_WEIGHTS",
    "ExperienceKBEvidence",
    "ExperienceKBIntegration",
    "integration_for",
    "read_signals",
]
