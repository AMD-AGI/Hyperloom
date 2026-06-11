# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR-A6 (Arbor-into-Hyperloom): per-domain specialist prompt templates.

Pins that every domain has a focus template, each rendered prompt mentions
its signature techniques, and the M5 active set covers all domains.
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.specialist_domains import (
    SPECIALIST_DOMAIN_KEYS,
    SPECIALIST_DOMAINS,
    SPECIALIST_DOMAINS_M5,
    get_domain,
)
from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
    _DOMAIN_FOCUS_TEMPLATES,
    SpecialistPromptInputs,
    build_specialist_prompts,
)


def _build(domain_key: str) -> str:
    domain = get_domain(domain_key)
    assert domain is not None, domain_key
    inp = SpecialistPromptInputs(
        task_id=f"task-{domain_key}",
        domain=domain,
        max_turns=4,
        gap_canonical_id=f"gap.{domain_key}.example",
        gap_symptom="example symptom",
        gap_layer=domain.layer,
        workspace_path=f"/tmp/test/{domain_key}",
    )
    system, user = build_specialist_prompts(inp)
    return system + "\n" + user


# 1. Coverage — every catalogue domain has a focus template
def test_every_domain_has_focus_template():
    for domain in SPECIALIST_DOMAINS:
        assert domain.key in _DOMAIN_FOCUS_TEMPLATES, (
            f"missing per-domain template for {domain.key!r}"
        )


def test_specialist_domains_m5_covers_all_active_domains():
    """The M5 active set covers the full catalogue (seven entries after P3_17 retired session_steward_specialist)."""
    assert SPECIALIST_DOMAINS_M5 == SPECIALIST_DOMAIN_KEYS
    assert len(SPECIALIST_DOMAINS_M5) == 7


# 2. Per-domain content checks — each template mentions its signature
def test_serving_specialist_mentions_scheduler_and_kv_cache():
    text = _build("serving_specialist")
    for marker in (
        "serving_specialist", "scheduler", "cuda_graph", "kv_cache",
        "max-num-seqs",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_serving_specialist_has_source_patch_playbook():
    """The serving focus must guide authoring source patches and carry the framework safety priors (ALWAYS_ON / NEVER_TOUCH)."""
    text = _build("serving_specialist")
    for marker in (
        "Source-patch playbook",   # the code-authoring section
        "block_manager",            # kv-cache module mapping
        "add_seq_group",            # upstream call-order contract to preserve
        "NEVER_TOUCH",              # safety classification from Arbor KB
        "VLLM_ROCM_USE_AITER",      # ALWAYS_ON umbrella flag
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_kernel_switch_specialist_mentions_aiter_and_attention_backends():
    text = _build("kernel_switch_specialist")
    for marker in (
        "kernel_switch_specialist", "aiter", "ROCM_AITER_MLA", "TRITON_MLA",
        "CDNA3",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_comm_specialist_mentions_quickreduce_and_topology():
    text = _build("comm_specialist")
    for marker in (
        "comm_specialist", "QuickReduce", "allreduce", "RCCL",
        "NCCL_MIN_NCHANNELS",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_compiler_specialist_mentions_torch_compile_and_triton():
    text = _build("compiler_specialist")
    for marker in (
        "compiler_specialist", "torch.compile", "inductor", "triton",
        "AMDGCN", "num_warps",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_system_specialist_mentions_kfd_and_rocm_smi():
    text = _build("system_specialist")
    for marker in (
        "system_specialist", "KFD", "rocm-smi", "HSA_ENABLE_SDMA",
        "numactl",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_pr_intel_specialist_mentions_cross_repo_research():
    text = _build("pr_intel_specialist")
    for marker in (
        "pr_intel_specialist", "cross-repo", "mcp__pr_monitor",
        "ROCm/aiter", "do NOT propose source patches",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


# 3. SpecialistRunner no longer marks any domain as "generic template"
@pytest.mark.asyncio
async def test_runner_does_not_log_generic_template_for_any_domain(tmp_path):
    """When the M5 active set covers a domain, the runner must NOT add a generic-template note."""
    from inference_optimizer.orchestrator.backends.mock_backend import (
        MockBackend, MockTurn, ScriptedPlan,
    )
    from inference_optimizer.protocol.intent import Intent, IntentType
    from inference_optimizer.orchestrator.specialist_runner import SpecialistRunner
    from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
    from inference_optimizer.orchestrator.task_registry import Task

    done = {
        "gap_canonical_id": "gap.x",
        "domain": "kernel_switch_specialist",
        "proposal_set": [],
        "empty": True,
        "summary": "test",
        "reason": "test",
        "confidence": 0.0,
        "new_findings": [],
        "residual_questions": [],
    }
    plan = ScriptedPlan(turns=[
        MockTurn(intents=[Intent(type=IntentType.SPECIALIST_DONE, payload=done)]),
    ])
    runner = SpecialistRunner(
        backend_factory=lambda d: MockBackend(plan, name="mock"),
        session_dir=tmp_path,
        default_max_turns=2,
    )
    task = Task(
        task_id="t-kernel",
        kind="specialist",
        state="queued",
        params={
            "domain": "kernel_switch_specialist",
            "gap_canonical_id": "gap.x",
            "max_turns": 2,
        },
        idempotency_key="t-kernel",
        requires_lanes=tuple(),
    )
    ctx = RunnerContext(task=task, lease=None, extra={})
    result = await runner.run(ctx)
    for note in result.notes or []:
        assert "generic prompt template" not in note, (
            f"PR-A6 should have widened SPECIALIST_DOMAINS_M5 to cover "
            f"kernel_switch_specialist; got note={note!r}"
        )


# Merged from test_v08_m5_specialist.py

"""v0.8 M5 — Specialist sub-agent framework tests (KB_design §3.5 + §3.13 M5)."""


from dataclasses import dataclass
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends.mock_backend import (
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from inference_optimizer.protocol.intent import (
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
    SPECIALIST_FROM_AGENT_PREFIX,
)
from inference_optimizer.orchestrator.resource_lock import (
    KNOWN_LANES, LANE_CONFLICTS,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.specialist_domains import (
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


# Test fixtures
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
    domain: str = "serving_specialist",
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


# 1. specialist_domains catalogue
def test_specialist_domains_catalogue_has_seven_entries():
    """P3_17 retired session_steward_specialist; the active catalogue has seven entries."""
    assert len(SPECIALIST_DOMAINS) == 7
    assert SPECIALIST_DOMAIN_KEYS == frozenset(
        d.key for d in SPECIALIST_DOMAINS
    )


def test_serving_specialist_is_M5_active():
    """PR-A6 widened ``SPECIALIST_DOMAINS_M5`` to the full catalogue (every domain now has a focus template)."""
    assert "serving_specialist" in SPECIALIST_DOMAINS_M5
    # PR-A6: M5 active set now equals the full catalogue.
    assert SPECIALIST_DOMAINS_M5 == SPECIALIST_DOMAIN_KEYS


def test_get_domain_returns_none_for_unknown():
    assert get_domain("nonsense_specialist") is None
    assert get_domain("serving_specialist").kb_anchor == "framework"


# 2. intent_parser — SPECIALIST_DONE envelope round-trip
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


# 3. PolicyGate R2 — specialist_dispatch_source
def test_R2_orchestration_can_dispatch_specialist(gate):
    gate.validate_intent("orchestration", Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "specialist",
            "params": {
                "domain": "serving_specialist",
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
                    "domain": "serving_specialist",
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
    assert exc.value.rule == "specialist_unknown_domain"
    assert "tag" in (exc.value.hint or "")


def test_R2_missing_gap_denied(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "specialist",
                "params": {"domain": "serving_specialist"},
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
                    "domain": "serving_specialist",
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
                    "domain": "serving_specialist",
                    "gap_canonical_id": "gap.x",
                    "max_turns": 0,
                },
            },
        ))
    assert exc.value.rule == "specialist_dispatch_source"


def test_R2_specialist_action_skips_unknown_action_registry_path(gate):
    """The synthetic ``specialist`` action_name bypasses the ActionRegistry lookup that would deny it as ``unknown_action``."""
    # Even with an ActionRegistry wired, the specialist branch is checked before the unknown_action gate.
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
                "domain": "serving_specialist",
                "gap_canonical_id": "gap.x",
            },
        },
    ))


# 4. PolicyGate R3 — specialist_done_source
def test_R3_specialist_done_from_specialist_agent_ok(gate):
    gate.validate_intent(
        f"{SPECIALIST_FROM_AGENT_PREFIX}task-abc",
        Intent(type=IntentType.SPECIALIST_DONE, payload=_valid_done_payload()),
    )


def test_R3_specialist_done_from_orchestration_denied(gate):
    """Non-``specialist:*`` agents cannot emit specialist_done (the role-matrix gate fires)."""
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


# 5. research_lane lane registration (KB_design §3.7)
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


# 6. specialist_prompt_builder — 9-section assembly
def test_prompt_builder_emits_nine_sections():
    sys_p, usr_p = build_specialist_prompts_for_domain(
        task_id="task-001",
        domain_key="serving_specialist",
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
    assert "## 4. KB CONTEXT (optional, advisory)" in usr_p
    assert "## 5. WARM-START RECIPE SUMMARY" in usr_p
    assert "## 6. PR FEED" in usr_p
    assert "## 7. LOCAL SOURCE NAVIGATION HINT" in usr_p


def test_prompt_builder_uses_none_placeholder_for_empty_sections():
    sys_p, usr_p = build_specialist_prompts_for_domain(
        task_id="task-002",
        domain_key="serving_specialist",
    )
    # Several user-side sections will be empty → "(none)" placeholder.
    assert "(none)" in usr_p


def test_prompt_builder_pr_feed_unavailable_renders_explanatory_line():
    sys_p, usr_p = build_specialist_prompts_for_domain(
        task_id="task-003",
        domain_key="serving_specialist",
        pr_monitor_available=False,
    )
    assert "pr_monitor unavailable" in usr_p


def test_prompt_builder_unknown_domain_raises():
    with pytest.raises(ValueError, match="unknown specialist domain"):
        build_specialist_prompts_for_domain(
            task_id="t", domain_key="nope_specialist",
        )


# 7. SpecialistRunner — happy path + failure synth
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
            "domain": "serving_specialist",
            "gap_canonical_id": "gap.scheduler",
            "max_turns": 4,
        },
    )
    ctx = RunnerContext(task=task, lease=None, extra={})
    result = await runner.run(ctx)

    assert result.status == "succeeded"
    assert result.specialist_done["domain"] == "serving_specialist"
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
    """When the backend never emits specialist_done, the runner caps at max_turns and synthesises an empty done (Inv-5.3)."""
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
            "domain": "serving_specialist",
            "gap_canonical_id": "gap.x",
            "max_turns": 2,
        },
    )
    ctx = RunnerContext(task=task, lease=None, extra={})
    result = await runner.run(ctx)

    assert result.status == "empty_synthesised"
    assert result.specialist_done["empty"] is True
    assert result.specialist_done["proposal_set"] == []
    assert result.specialist_done["domain"] == "serving_specialist"
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
            "domain": "serving_specialist",
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


def test_specialist_tool_denylist_excludes_kb_write_paths():
    """The denylist still captures every cortex_kb write surface (Coordinator is the sole KB writer); PR-A2 lifted Edit/Write/MultiEdit out so specialists can patch their isolated worktree."""
    for forbidden in (
        "mcp__cortex_kb__propose_point",
    ):
        assert forbidden in SPECIALIST_TOOL_DENYLIST
        assert forbidden not in DEFAULT_SPECIALIST_TOOLS
    # PR-A2: write tools are in the default whitelist, not the denylist.
    for write_tool in ("Edit", "Write", "MultiEdit"):
        assert write_tool not in SPECIALIST_TOOL_DENYLIST
        assert write_tool in DEFAULT_SPECIALIST_TOOLS


def test_build_empty_specialist_done_is_R3_valid():
    """The failure-path helper must always produce a payload PolicyGate R3 accepts."""
    done = build_empty_specialist_done(
        gap_canonical_id="gap.x",
        domain="serving_specialist",
        reason="example reason",
    )
    gate = PolicyGate(role_registry=default_role_registry())
    gate.validate_intent(
        f"{SPECIALIST_FROM_AGENT_PREFIX}task-test",
        Intent(type=IntentType.SPECIALIST_DONE, payload=done),
    )


# 8. SharedState specialist round bookkeeping
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
        "domains": ["serving_specialist"],
        "proposals_total": 2,
    })
    s.record_specialist_round({
        "round_id": "explore-001",
        "domains": ["serving_specialist"],
        "proposals_total": 5,    # updated count
    })
    s.record_specialist_round({
        "round_id": "explore-002",
        "domains": ["serving_specialist"],
        "proposals_total": 1,
    })
    assert len(s.specialist_rounds) == 2
    by_round = {r["round_id"]: r for r in s.specialist_rounds}
    assert by_round["explore-001"]["proposals_total"] == 5


def test_bump_specialist_domain_empty_streak():
    s = SharedState()
    assert s.bump_specialist_domain_empty_streak("serving_specialist",
                                                  empty=True) == 1
    assert s.bump_specialist_domain_empty_streak("serving_specialist",
                                                  empty=True) == 2
    # A non-empty proposal_set resets.
    assert s.bump_specialist_domain_empty_streak("serving_specialist",
                                                  empty=False) == 0
    # Other domains don't share state.
    assert s.bump_specialist_domain_empty_streak("kernel_switch_specialist",
                                                  empty=True) == 1
    assert s.specialist_domain_empty_streak["serving_specialist"] == 0
    assert s.specialist_domain_empty_streak["kernel_switch_specialist"] == 1


def test_update_last_specialist_snapshot():
    s = SharedState()
    s.update_last_specialist({
        "task_id": "task-001",
        "domain": "serving_specialist",
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
    assert "gpu_specialist_capacity" in CORE_STATE_FIELDS
    assert "specialist_rounds" in CORE_STATE_FIELDS
    assert "last_specialist" in CORE_STATE_FIELDS
