"""v0.8 M5 — Specialist sub-agent framework tests.

Covers KB_design §3.5 + §3.13 M5:

* Intent layer: ``SPECIALIST_DONE`` parses through the standard envelope
  validator (intent_parser).
* PolicyGate R2 (``specialist_dispatch_source``): only Orchestration
  dispatches, domain ∈ SPECIALIST_DOMAIN_KEYS, gap_canonical_id
  required, max_turns bounded.
* PolicyGate R3 (``specialist_done_source``): only ``specialist:<task_id>``
  from_agent may emit specialist_done, payload schema enforced.
* Specialist prompt builder: 9 sections present, system / user split.
* SpecialistRunner: happy path (Mock backend emits specialist_done) +
  failure synthesis (no done → empty done synthesised).
* SharedState round bookkeeping (record_specialist_round /
  bump_specialist_domain_empty_streak / update_last_specialist).
* research_lane lane registration + LANE_CONFLICTS empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends.base import BackendTurnResult
from inference_optimizer.orchestrator.backends.mock_backend import (
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.intent_parser import (
    Intent,
    IntentType,
    IntentValidationError,
    validate_envelope,
)
from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.policy import (
    PolicyDenied,
    PolicyGate,
    SPECIALIST_ACTION_NAME,
    SPECIALIST_DISPATCH_SOURCE_ALLOWLIST,
    SPECIALIST_FROM_AGENT_PREFIX,
)
from inference_optimizer.orchestrator.resource_lock import (
    KNOWN_LANES, LANE_CONFLICTS,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.specialist_domains import (
    DEFAULT_SPECIALIST_MAX_TURNS,
    SPECIALIST_DOMAINS,
    SPECIALIST_DOMAINS_M5,
    SPECIALIST_DOMAIN_KEYS,
    SPECIALIST_MAX_TURNS_HARD_CAP,
    get_domain,
)
from inference_optimizer.orchestrator.specialist_runner import (
    DEFAULT_SPECIALIST_TOOLS,
    SPECIALIST_TOOL_DENYLIST,
    SpecialistRunner,
    build_empty_specialist_done,
)
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    build_specialist_prompts,
    build_specialist_prompts_for_domain,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def gate() -> PolicyGate:
    return PolicyGate(role_registry=default_role_registry())


@dataclass
class _StubTask:
    task_id: str
    kind: str = "specialist"
    params: dict[str, Any] | None = None


def _valid_done_payload(
    *,
    gap: str = "gap.attention.fp8_kv",
    domain: str = "framework_specialist",
    empty: bool = False,
    proposals: list[dict[str, Any]] | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "gap_canonical_id": gap,
        "domain": domain,
        "proposal_set": proposals if proposals is not None else (
            [] if empty else [{"name": "v1", "extra_args": "--flag"}]
        ),
        "empty": empty,
        "summary": "stub run summary",
    }
    if empty and "summary" not in (extras or {}):
        payload["summary"] = "no useful proposals this round"
    if extras:
        payload.update(extras)
    return payload


# ===========================================================================
# 1. specialist_domains catalogue
# ===========================================================================
def test_specialist_domains_catalogue_has_six_entries():
    assert len(SPECIALIST_DOMAINS) == 6
    assert SPECIALIST_DOMAIN_KEYS == frozenset(
        d.key for d in SPECIALIST_DOMAINS
    )


def test_framework_specialist_is_M5_active():
    assert "framework_specialist" in SPECIALIST_DOMAINS_M5
    other = SPECIALIST_DOMAIN_KEYS - SPECIALIST_DOMAINS_M5
    assert other == {
        "kernel_specialist", "comm_specialist", "compiler_specialist",
        "system_specialist", "pr_intel_specialist",
    }


def test_get_domain_returns_none_for_unknown():
    assert get_domain("nonsense_specialist") is None
    assert get_domain("framework_specialist").kb_anchor == "framework"


# ===========================================================================
# 2. intent_parser — SPECIALIST_DONE envelope round-trip
# ===========================================================================
def test_specialist_done_envelope_passes_validation():
    envelope = {
        "intents": [{
            "intent_type": "specialist_done",
            "payload": _valid_done_payload(),
        }]
    }
    intents = validate_envelope(envelope)
    assert len(intents) == 1
    assert intents[0].type == IntentType.SPECIALIST_DONE


def test_specialist_done_envelope_missing_required_field():
    bad_payload = _valid_done_payload()
    bad_payload.pop("summary")
    envelope = {"intents": [{
        "intent_type": "specialist_done",
        "payload": bad_payload,
    }]}
    with pytest.raises(IntentValidationError, match="summary"):
        validate_envelope(envelope)


# ===========================================================================
# 3. PolicyGate R2 — specialist_dispatch_source
# ===========================================================================
def test_R2_orchestration_can_dispatch_specialist(gate):
    gate.validate_intent("orchestration", Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "specialist",
            "params": {
                "domain": "framework_specialist",
                "gap_canonical_id": "gap.kv.fp8",
                "max_turns": 4,
            },
        },
    ))


def test_R2_robustness_cannot_dispatch_specialist(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("robustness", Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "specialist",
                "params": {
                    "domain": "framework_specialist",
                    "gap_canonical_id": "gap.kv.fp8",
                },
            },
        ))
    assert exc.value.rule == "specialist_dispatch_source"
    assert "Orchestration" in (exc.value.hint or "")


def test_R2_unknown_domain_denied(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "specialist",
                "params": {
                    "domain": "fake_specialist",
                    "gap_canonical_id": "gap.x",
                },
            },
        ))
    assert exc.value.rule == "specialist_dispatch_source"
    assert "domain" in (exc.value.hint or "")


def test_R2_missing_gap_denied(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "specialist",
                "params": {"domain": "framework_specialist"},
            },
        ))
    assert exc.value.rule == "specialist_dispatch_source"
    assert "gap" in str(exc.value)


def test_R2_max_turns_excess_denied(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "specialist",
                "params": {
                    "domain": "framework_specialist",
                    "gap_canonical_id": "gap.x",
                    "max_turns": SPECIALIST_MAX_TURNS_HARD_CAP + 1,
                },
            },
        ))
    assert exc.value.rule == "specialist_dispatch_source"
    assert "max_turns" in str(exc.value)


def test_R2_max_turns_zero_denied(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "specialist",
                "params": {
                    "domain": "framework_specialist",
                    "gap_canonical_id": "gap.x",
                    "max_turns": 0,
                },
            },
        ))
    assert exc.value.rule == "specialist_dispatch_source"


def test_R2_specialist_action_skips_unknown_action_registry_path(gate):
    """``specialist`` has no yaml meta; the synthetic action_name must
    bypass the standard ActionRegistry lookup that would otherwise
    deny it as ``unknown_action``."""
    # Even when an ActionRegistry is wired, specialist still passes
    # (the registry-lookup path is guarded by the specialist branch
    # before the unknown_action gate fires).
    from inference_optimizer.orchestrator.action_registry import ActionRegistry
    gate_with_registry = PolicyGate(
        role_registry=default_role_registry(),
        action_registry=ActionRegistry().load(),
    )
    gate_with_registry.validate_intent("orchestration", Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": SPECIALIST_ACTION_NAME,
            "params": {
                "domain": "framework_specialist",
                "gap_canonical_id": "gap.x",
            },
        },
    ))


# ===========================================================================
# 4. PolicyGate R3 — specialist_done_source
# ===========================================================================
def test_R3_specialist_done_from_specialist_agent_ok(gate):
    gate.validate_intent(
        f"{SPECIALIST_FROM_AGENT_PREFIX}task-abc",
        Intent(type=IntentType.SPECIALIST_DONE, payload=_valid_done_payload()),
    )


def test_R3_specialist_done_from_orchestration_denied(gate):
    """Non-``specialist:*`` agents cannot emit specialist_done — the
    orchestration role's allowed_intents set doesn't include
    SPECIALIST_DONE so the standard role-matrix gate fires."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.SPECIALIST_DONE,
            payload=_valid_done_payload(),
        ))
    assert exc.value.rule == "role"


def test_R3_specialist_done_missing_gap_denied(gate):
    payload = _valid_done_payload()
    payload.pop("gap_canonical_id")
    payload["gap_canonical_id"] = ""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            f"{SPECIALIST_FROM_AGENT_PREFIX}task-abc",
            Intent(type=IntentType.SPECIALIST_DONE, payload=payload),
        )
    assert exc.value.rule == "specialist_done_source"


def test_R3_specialist_done_unknown_domain_denied(gate):
    payload = _valid_done_payload(domain="nonsense_specialist")
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            f"{SPECIALIST_FROM_AGENT_PREFIX}task-abc",
            Intent(type=IntentType.SPECIALIST_DONE, payload=payload),
        )
    assert exc.value.rule == "specialist_done_source"


def test_R3_specialist_done_empty_true_requires_reason(gate):
    payload = _valid_done_payload(empty=True, proposals=[])
    payload["summary"] = ""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            f"{SPECIALIST_FROM_AGENT_PREFIX}task-abc",
            Intent(type=IntentType.SPECIALIST_DONE, payload=payload),
        )
    assert exc.value.rule == "specialist_done_source"


def test_R3_specialist_done_empty_with_proposals_denied(gate):
    payload = _valid_done_payload(
        empty=True, proposals=[{"name": "should_not_be_here"}],
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            f"{SPECIALIST_FROM_AGENT_PREFIX}task-abc",
            Intent(type=IntentType.SPECIALIST_DONE, payload=payload),
        )
    assert exc.value.rule == "specialist_done_source"


def test_R3_specialist_done_variant_missing_name_denied(gate):
    payload = _valid_done_payload(
        empty=False, proposals=[{"extra_args": "--no-name"}],
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            f"{SPECIALIST_FROM_AGENT_PREFIX}task-abc",
            Intent(type=IntentType.SPECIALIST_DONE, payload=payload),
        )
    assert exc.value.rule == "specialist_done_source"


def test_R3_specialist_can_emit_send_message_heartbeat(gate):
    gate.validate_intent(
        f"{SPECIALIST_FROM_AGENT_PREFIX}task-abc",
        Intent(type=IntentType.SEND_MESSAGE,
               payload={"topic": "heartbeat", "body_md": "still thinking"}),
    )


def test_R3_specialist_cannot_emit_propose_action(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            f"{SPECIALIST_FROM_AGENT_PREFIX}task-abc",
            Intent(type=IntentType.PROPOSE_ACTION,
                   payload={"action_name": "explore",
                            "predicted_gain_pct": 0.0}),
        )
    assert exc.value.rule == "specialist_done_source"


def test_R3_specialist_missing_task_id_suffix_denied(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            SPECIALIST_FROM_AGENT_PREFIX,  # no suffix
            Intent(type=IntentType.SPECIALIST_DONE,
                   payload=_valid_done_payload()),
        )
    assert exc.value.rule == "specialist_done_source"


def test_R3_specialist_done_confidence_out_of_range_denied(gate):
    payload = _valid_done_payload()
    payload["confidence"] = 1.5
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            f"{SPECIALIST_FROM_AGENT_PREFIX}task-abc",
            Intent(type=IntentType.SPECIALIST_DONE, payload=payload),
        )
    assert exc.value.rule == "specialist_done_source"


# ===========================================================================
# 5. research_lane lane registration (KB_design §3.7)
# ===========================================================================
def test_research_lane_in_known_lanes():
    assert "research_lane" in KNOWN_LANES


def test_research_lane_has_no_conflicts():
    assert LANE_CONFLICTS["research_lane"] == frozenset()
    # And no other lane lists research_lane as a conflict.
    for lane, conflicts in LANE_CONFLICTS.items():
        assert "research_lane" not in conflicts, (
            f"lane={lane!r} unexpectedly conflicts with research_lane "
            f"(KB_design §3.7 Inv-7.2: research_lane is conflict-free)"
        )


# ===========================================================================
# 6. specialist_prompt_builder — 9-section assembly
# ===========================================================================
def test_prompt_builder_emits_nine_sections():
    sys_p, usr_p = build_specialist_prompts_for_domain(
        task_id="task-001",
        domain_key="framework_specialist",
        gap_canonical_id="gap.scheduler.long_isl",
        gpu_type="MI300X",
        tp=8,
    )
    # System sections (1, 8, 9)
    assert "## 1. IDENTITY & AUTONOMY" in sys_p
    assert "## 8. OUTPUT PROTOCOL" in sys_p
    assert "## 9. IRON RULES" in sys_p
    # User sections (2-7)
    assert "## 2. HARDWARE CONTEXT" in usr_p
    assert "## 3. GAP STATEMENT" in usr_p
    assert "## 4. CORTEX KB SUB-GRAPH" in usr_p
    assert "## 5. WARM-START RECIPE SUMMARY" in usr_p
    assert "## 6. PR FEED" in usr_p
    assert "## 7. LOCAL SOURCE NAVIGATION HINT" in usr_p


def test_prompt_builder_uses_none_placeholder_for_empty_sections():
    sys_p, usr_p = build_specialist_prompts_for_domain(
        task_id="task-002",
        domain_key="framework_specialist",
    )
    # Several user-side sections will be empty → "(none)" placeholder.
    assert "(none)" in usr_p


def test_prompt_builder_pr_feed_unavailable_renders_explanatory_line():
    sys_p, usr_p = build_specialist_prompts_for_domain(
        task_id="task-003",
        domain_key="framework_specialist",
        pr_monitor_available=False,
    )
    assert "pr_monitor unavailable" in usr_p


def test_prompt_builder_unknown_domain_raises():
    with pytest.raises(ValueError, match="unknown specialist domain"):
        build_specialist_prompts_for_domain(
            task_id="t", domain_key="nope_specialist",
        )


# ===========================================================================
# 7. SpecialistRunner — happy path + failure synth
# ===========================================================================
@pytest.mark.asyncio
async def test_specialist_runner_happy_path(tmp_path):
    """MockBackend emits a valid specialist_done; runner persists files."""
    done_payload = _valid_done_payload(
        proposals=[
            {"name": "max_seqs_512", "extra_args": "--max-num-seqs 512"},
            {"name": "kv_fp8", "extra_args": "--kv-cache-dtype fp8"},
        ],
    )
    plan = ScriptedPlan(turns=[MockTurn(intents=[
        Intent(type=IntentType.SPECIALIST_DONE, payload=done_payload),
    ])])

    runner = SpecialistRunner(
        backend_factory=lambda domain: MockBackend(plan, name=domain.key),
        session_dir=tmp_path,
    )
    task = _StubTask(
        task_id="task-xyz",
        params={
            "domain": "framework_specialist",
            "gap_canonical_id": "gap.scheduler",
            "max_turns": 4,
        },
    )
    ctx = RunnerContext(task=task, lease=None, extra={})
    result = await runner.run(ctx)

    assert result.status == "succeeded"
    assert result.specialist_done["domain"] == "framework_specialist"
    assert result.specialist_done["gap_canonical_id"] == "gap.scheduler"
    assert len(result.specialist_done["proposal_set"]) == 2
    assert result.turns_used == 1

    workspace = tmp_path / "runs" / "specialist" / "task-xyz"
    assert (workspace / "prompt.md").exists()
    assert (workspace / "transcript.jsonl").exists()
    assert (workspace / "heartbeat.json").exists()
    assert (workspace / "specialist_done.json").exists()
    prompt_text = (workspace / "prompt.md").read_text(encoding="utf-8")
    assert "## 1. IDENTITY & AUTONOMY" in prompt_text
    transcript_text = (workspace / "transcript.jsonl").read_text(encoding="utf-8")
    assert "specialist_done" in transcript_text


@pytest.mark.asyncio
async def test_specialist_runner_synthesises_empty_done_on_max_turns(tmp_path):
    """When the backend never emits specialist_done, the runner caps at
    max_turns and synthesises an empty done so the EXPLORE round
    can proceed (Inv-5.3)."""
    # Plan keeps emitting heartbeats; never produces a done.
    heartbeat_intent = Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "still working"},
    )
    plan = ScriptedPlan(
        turns=[MockTurn(intents=[heartbeat_intent])],
        loop_last=True,
    )
    runner = SpecialistRunner(
        backend_factory=lambda domain: MockBackend(plan),
        session_dir=tmp_path,
    )
    task = _StubTask(
        task_id="task-stale",
        params={
            "domain": "framework_specialist",
            "gap_canonical_id": "gap.x",
            "max_turns": 2,
        },
    )
    ctx = RunnerContext(task=task, lease=None, extra={})
    result = await runner.run(ctx)

    assert result.status == "empty_synthesised"
    assert result.specialist_done["empty"] is True
    assert result.specialist_done["proposal_set"] == []
    assert result.specialist_done["domain"] == "framework_specialist"
    assert "max_turns_exhausted" in result.specialist_done.get("reason", "")
    assert result.turns_used == 2  # max_turns reached
    # The synthesised payload is still valid against PolicyGate R3.
    gate = PolicyGate(role_registry=default_role_registry())
    gate.validate_intent(
        f"{SPECIALIST_FROM_AGENT_PREFIX}task-stale",
        Intent(type=IntentType.SPECIALIST_DONE,
               payload=result.specialist_done),
    )


@pytest.mark.asyncio
async def test_specialist_runner_backend_error_synthesises_empty_done(tmp_path):
    from inference_optimizer.orchestrator.backends.base import BackendError

    plan = ScriptedPlan(turns=[
        MockTurn(raise_error=BackendError("rate limited")),
    ])
    runner = SpecialistRunner(
        backend_factory=lambda domain: MockBackend(plan),
        session_dir=tmp_path,
    )
    task = _StubTask(
        task_id="task-err",
        params={
            "domain": "framework_specialist",
            "gap_canonical_id": "gap.x",
            "max_turns": 2,
        },
    )
    ctx = RunnerContext(task=task, lease=None, extra={})
    result = await runner.run(ctx)

    assert result.status == "stale"
    assert result.specialist_done["empty"] is True
    assert "rate limited" in result.error


@pytest.mark.asyncio
async def test_specialist_runner_unknown_domain_synthesises_empty(tmp_path):
    plan = ScriptedPlan(turns=[])
    runner = SpecialistRunner(
        backend_factory=lambda domain: MockBackend(plan),
        session_dir=tmp_path,
    )
    task = _StubTask(
        task_id="task-unk",
        params={
            "domain": "made_up_specialist",
            "gap_canonical_id": "gap.x",
        },
    )
    ctx = RunnerContext(task=task, lease=None, extra={})
    result = await runner.run(ctx)
    assert result.status == "empty_synthesised"
    assert result.specialist_done["empty"] is True
    assert "unknown specialist domain" in result.specialist_done["reason"]


def test_specialist_tool_denylist_excludes_write_paths():
    """Sanity: the denylist captures every cortex_kb write surface
    listed in KB_design §3.11 R4."""
    for forbidden in (
        "Edit", "Write",
        "mcp__cortex_kb__hypothesize",
        "mcp__cortex_kb__ingest_attempt",
        "mcp__cortex_kb__verify",
        "mcp__cortex_kb__commit",
    ):
        assert forbidden in SPECIALIST_TOOL_DENYLIST
        assert forbidden not in DEFAULT_SPECIALIST_TOOLS


def test_build_empty_specialist_done_is_R3_valid():
    """Helper used by every failure path must always produce a payload
    PolicyGate R3 will accept."""
    done = build_empty_specialist_done(
        gap_canonical_id="gap.x",
        domain="framework_specialist",
        reason="example reason",
    )
    gate = PolicyGate(role_registry=default_role_registry())
    gate.validate_intent(
        f"{SPECIALIST_FROM_AGENT_PREFIX}task-test",
        Intent(type=IntentType.SPECIALIST_DONE, payload=done),
    )


# ===========================================================================
# 8. SharedState specialist round bookkeeping
# ===========================================================================
def test_shared_state_specialist_rounds_default_empty():
    s = SharedState()
    assert s.specialist_rounds == []
    assert s.specialist_domain_empty_streak == {}
    assert s.last_specialist == {}
    assert s.research_lane_capacity == 1


def test_record_specialist_round_dedup_by_round_id():
    s = SharedState()
    s.record_specialist_round({
        "round_id": "explore-001",
        "domains": ["framework_specialist"],
        "proposals_total": 2,
    })
    s.record_specialist_round({
        "round_id": "explore-001",
        "domains": ["framework_specialist"],
        "proposals_total": 5,    # updated count
    })
    s.record_specialist_round({
        "round_id": "explore-002",
        "domains": ["framework_specialist"],
        "proposals_total": 1,
    })
    assert len(s.specialist_rounds) == 2
    by_round = {r["round_id"]: r for r in s.specialist_rounds}
    assert by_round["explore-001"]["proposals_total"] == 5


def test_bump_specialist_domain_empty_streak():
    s = SharedState()
    assert s.bump_specialist_domain_empty_streak("framework_specialist",
                                                  empty=True) == 1
    assert s.bump_specialist_domain_empty_streak("framework_specialist",
                                                  empty=True) == 2
    # A non-empty proposal_set resets.
    assert s.bump_specialist_domain_empty_streak("framework_specialist",
                                                  empty=False) == 0
    # Other domains don't share state.
    assert s.bump_specialist_domain_empty_streak("kernel_specialist",
                                                  empty=True) == 1
    assert s.specialist_domain_empty_streak["framework_specialist"] == 0
    assert s.specialist_domain_empty_streak["kernel_specialist"] == 1


def test_update_last_specialist_snapshot():
    s = SharedState()
    s.update_last_specialist({
        "task_id": "task-001",
        "domain": "framework_specialist",
        "status": "succeeded",
    })
    assert s.last_specialist["task_id"] == "task-001"
    # Non-dict inputs are ignored.
    s.update_last_specialist("garbage")  # type: ignore[arg-type]
    assert s.last_specialist["task_id"] == "task-001"


def test_research_lane_capacity_is_core_state_field():
    """LLM cannot raise research_lane_capacity mid-flight."""
    from inference_optimizer.orchestrator.policy import CORE_STATE_FIELDS
    assert "research_lane_capacity" in CORE_STATE_FIELDS
    assert "specialist_rounds" in CORE_STATE_FIELDS
    assert "last_specialist" in CORE_STATE_FIELDS
