"""Unit tests for :class:`ActionLadder`."""

from __future__ import annotations

import pytest

from robustness_agent.decision.action_ladder import (
    ActionLadder,
    ActionLadderConfig,
)
from robustness_agent.role.envelope import IntentType
from robustness_agent.signals import Symptom, SymptomSeverity

pytestmark = pytest.mark.asyncio


def _sym(
    name: str,
    severity: SymptomSeverity,
    *,
    summary: str = "summary",
    subject: dict | None = None,
    evidence: dict | None = None,
    suggestion: str = "",
) -> Symptom:
    return Symptom(
        name=name,
        severity=severity,
        summary=summary,
        evidence=evidence or {},
        subject=subject or {"k": "v"},
        source="test",
        suggestion=suggestion,
    )


# ---------------------------------------------------------------------------
# Tier mapping
# ---------------------------------------------------------------------------

async def test_low_severity_yields_observation_send_message():
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("pod_no_metrics", SymptomSeverity.LOW)],
        tick_index=0,
        now_unix=1.0,
    )
    assert len(out.intents) == 1
    assert out.intents[0].type is IntentType.SEND_MESSAGE
    assert out.intents[0].payload["topic"] == "observation"
    assert out.findings and out.findings[0].symptom_name == "pod_no_metrics"


async def test_medium_severity_yields_alert_only():
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("crash_count_rising", SymptomSeverity.MEDIUM)],
        tick_index=0,
        now_unix=1.0,
    )
    assert len(out.intents) == 1
    assert out.intents[0].type is IntentType.ALERT
    assert out.intents[0].payload["severity"] == "medium"


async def test_high_crash_emits_alert_plus_escalate():
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("crash_count_high", SymptomSeverity.HIGH, suggestion="revert")],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ALERT in types
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    escalate = next(i for i in out.intents if i.type is IntentType.ESCALATE_STRATEGY_CHANGE)
    assert escalate.payload["next_action_hint"] == "revert"


async def test_high_agent_stall_emits_escalate_with_default_hint():
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("agent_stall", SymptomSeverity.HIGH, subject={"agent": "kernel"})],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ALERT in types
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types


async def test_repeated_failure_with_family_triggers_prune_branch():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "repeated_failure",
                SymptomSeverity.HIGH,
                evidence={"family": "kernel_opt", "count": 3},
                subject={"family": "kernel_opt"},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.PRUNE_BRANCH in types
    prune = next(i for i in out.intents if i.type is IntentType.PRUNE_BRANCH)
    assert prune.payload["family"] == "kernel_opt"


# ---------------------------------------------------------------------------
# Cooldown / heartbeat
# ---------------------------------------------------------------------------

async def test_no_symptoms_falls_back_to_heartbeat():
    ladder = ActionLadder()
    out = await ladder.decide([], tick_index=0, now_unix=1.0)
    assert len(out.intents) == 1
    assert out.intents[0].type is IntentType.SEND_MESSAGE
    assert out.intents[0].payload["topic"] == "heartbeat"
    assert out.findings == []


async def test_cooldown_suppresses_duplicate_within_window():
    ladder = ActionLadder(config=ActionLadderConfig(cooldown_ticks=3))
    sym = _sym("crash_count_rising", SymptomSeverity.MEDIUM, subject={"k": "1"})
    first = await ladder.decide([sym], tick_index=0, now_unix=1.0)
    suppressed = await ladder.decide([sym], tick_index=1, now_unix=2.0)
    assert any(i.type is IntentType.ALERT for i in first.intents)
    # Suppressed tick: no symptom-derived intent emitted, only heartbeat.
    assert all(
        not (i.type is IntentType.ALERT) for i in suppressed.intents
    )
    assert any(i.payload.get("topic") == "heartbeat" for i in suppressed.intents)
    assert suppressed.findings == []


async def test_cooldown_releases_after_window():
    ladder = ActionLadder(config=ActionLadderConfig(cooldown_ticks=3))
    sym = _sym("crash_count_rising", SymptomSeverity.MEDIUM, subject={"k": "1"})
    await ladder.decide([sym], tick_index=0, now_unix=1.0)
    out = await ladder.decide([sym], tick_index=3, now_unix=4.0)
    assert any(i.type is IntentType.ALERT for i in out.intents)


async def test_finding_carries_serialised_intents_and_evidence():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "crash_count_high",
                SymptomSeverity.HIGH,
                summary="crash_count=5",
                evidence={"crash_count": 5},
                suggestion="revert",
            )
        ],
        tick_index=2,
        now_unix=10.0,
    )
    assert out.findings
    finding = out.findings[0]
    assert finding.symptom_name == "crash_count_high"
    assert finding.tick_index == 2
    assert finding.timestamp_unix == 10.0
    assert any(i["intent_type"] == "escalate_strategy_change" for i in finding.intents)
    assert finding.evidence == {"crash_count": 5}


async def test_rca_provider_is_invoked_when_supplied():
    ladder = ActionLadder()

    class StubRca:
        def summarize(self, sym: Symptom) -> str:
            return f"rca({sym.name})"

    out = await ladder.decide(
        [_sym("crash_count_rising", SymptomSeverity.MEDIUM)],
        tick_index=0,
        now_unix=1.0,
        rca_provider=StubRca(),
    )
    assert out.findings[0].rca_text == "rca(crash_count_rising)"


async def test_rca_provider_failure_does_not_break_ladder():
    ladder = ActionLadder()

    class BadRca:
        def summarize(self, sym):
            raise RuntimeError("oops")

    out = await ladder.decide(
        [_sym("crash_count_rising", SymptomSeverity.MEDIUM)],
        tick_index=0,
        now_unix=1.0,
        rca_provider=BadRca(),
    )
    assert out.findings[0].rca_text == ""
    assert any(i.type is IntentType.ALERT for i in out.intents)
