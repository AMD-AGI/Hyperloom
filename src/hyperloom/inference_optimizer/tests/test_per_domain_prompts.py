# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-domain specialist prompt templates.

Pins that every domain has a focus template, each rendered prompt mentions
its signature techniques, and the active set covers all domains.
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.specialists.domains import (
    SPECIALIST_DOMAIN_KEYS,
    SPECIALIST_DOMAINS,
    get_domain,
)
from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
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
        assert domain.key in _DOMAIN_FOCUS_TEMPLATES, f"missing per-domain template for {domain.key!r}"


def test_specialist_domain_keys_covers_all_active_domains():
    """The active key set is exactly the catalogue, with no duplicate keys.

    Asserted as a property rather than a count: a hard-coded total has to be
    edited every time a domain is added, which makes the edit routine and stops
    it from signalling anything.
    """
    assert SPECIALIST_DOMAIN_KEYS == frozenset(d.key for d in SPECIALIST_DOMAINS)
    assert len(SPECIALIST_DOMAIN_KEYS) == len(SPECIALIST_DOMAINS)


# 2. Per-domain content checks — each template mentions its signature
def test_serving_specialist_mentions_scheduler_and_kv_cache():
    text = _build("serving_specialist")
    for marker in (
        "serving_specialist",
        "scheduler",
        "cuda_graph",
        "kv_cache",
        "max-num-seqs",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_serving_specialist_has_source_patch_playbook():
    """The serving focus must guide authoring source patches and carry the framework safety priors (ALWAYS_ON / NEVER_TOUCH)."""
    text = _build("serving_specialist")
    for marker in (
        "Source-patch playbook",
        "block_manager",
        "add_seq_group",
        "NEVER_TOUCH",
        "VLLM_ROCM_USE_AITER",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_kernel_switch_specialist_mentions_aiter_and_attention_backends():
    text = _build("kernel_switch_specialist")
    for marker in (
        "kernel_switch_specialist",
        "aiter",
        "ROCM_AITER_MLA",
        "TRITON_MLA",
        "CDNA3",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_comm_specialist_mentions_quickreduce_and_topology():
    text = _build("comm_specialist")
    for marker in (
        "comm_specialist",
        "QuickReduce",
        "allreduce",
        "RCCL",
        "NCCL_MIN_NCHANNELS",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_compiler_specialist_mentions_torch_compile_and_triton():
    text = _build("compiler_specialist")
    for marker in (
        "compiler_specialist",
        "torch.compile",
        "inductor",
        "triton",
        "AMDGCN",
        "num_warps",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_system_specialist_mentions_kfd_and_rocm_smi():
    text = _build("system_specialist")
    for marker in (
        "system_specialist",
        "KFD",
        "rocm-smi",
        "HSA_ENABLE_SDMA",
        "numactl",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_candidate_discovery_specialist_mentions_find_rank_judge():
    text = _build("candidate_discovery_specialist")
    for marker in (
        "candidate_discovery_specialist",
        "mcp__pr_monitor",
        "already_present",
        "not_applicable",
        "worth_a_bench",
        "you do not apply or benchmark",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_static_recon_specialist_mentions_reconnaissance_and_bridge_candidates():
    """The static-recon focus must steer read-only source grep for disabled switches and a bridge_candidates output block."""
    text = _build("static_recon_specialist")
    for marker in (
        "static-recon",
        "read-only",
        "_supported()",
        "bridge_candidates",
        "predicate_file",
        "Never write a patch",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_static_recon_specialist_renders_seed_checklist_and_model_info():
    """When a seed checklist + model_info are supplied they render into the prompt."""
    domain = get_domain("static_recon_specialist")
    assert domain is not None
    inp = SpecialistPromptInputs(
        task_id="task-recon",
        domain=domain,
        max_turns=4,
        gpu_type="MI300X",
        precision="fp8",
        model_info={"attention_type": "GQA", "is_moe": False, "quantization": "fp8"},
        static_recon_checklist="- **rocm.fp8.cutlass_only_guard** (domain_hint=`freeform`)\n  - detect: grep cutlass_fp8_supported",
        workspace_path="/tmp/test/recon",
    )
    system, user = build_specialist_prompts(inp)
    text = (system + "\n" + user).lower()
    assert "rocm.fp8.cutlass_only_guard" in text
    assert "seed checklist" in text
    assert "attention=gqa" in text


def test_enablement_specialist_mentions_runnability_and_authoring():
    """The enablement focus (rendered from build_mandate) must steer patch authoring gated on runnability, not perf."""
    text = _build("enablement_specialist")
    for marker in (
        "enablement specialist",
        "authoring",
        "runnability",
        "git apply --check",
        "boot",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def _build_split(domain_key: str) -> tuple[str, str]:
    domain = get_domain(domain_key)
    assert domain is not None, domain_key
    inp = SpecialistPromptInputs(
        task_id=f"task-{domain_key}",
        domain=domain,
        max_turns=4,
        gap_canonical_id=f"gap.{domain_key}.example",
        gap_symptom="Model architecture 'DeepseekV4ForCausalLM' is not supported",
        gap_layer=domain.layer,
        gap_evidence={"model": "deepseek-ai/DeepSeek-V4"},
        # Perf context that MUST be stripped for the enablement domain.
        roofline_evidence={"roofline_snapshot_id": 7, "executive_summary": {"compute_pct": 50}},
        warm_start_recipe={"name": "r1"},
        warm_start_lessons=[{"attrs": {"statement": "prior keep lesson"}}],
        warm_start_pitfalls=[{"attrs": {"description": "prior revert pitfall"}}],
        kb_subgraph={"nodes": ["x"]},
        workspace_path=f"/tmp/test/{domain_key}",
    )
    return build_specialist_prompts(inp)


def test_enablement_user_prompt_injects_ladder_book_and_keeps_essentials():
    """Enablement dispatch carries the ladder book + gap + PR monitor + source hint in the USER prompt."""
    _system, user = _build_split("enablement_specialist")
    assert "## 1b. ENABLEMENT PLAYBOOK" in user
    assert "ENABLEMENT METHODOLOGY" in user
    assert "Rung 5" in user
    assert "## 3. GAP STATEMENT" in user
    assert "## 6. PR MONITOR" in user
    assert "## 7. LOCAL SOURCE NAVIGATION HINT" in user


def test_enablement_user_prompt_strips_perf_only_sections():
    """Pre-baseline perf context is omitted for the enablement domain."""
    _system, user = _build_split("enablement_specialist")
    for banned in (
        "## 4. KB CONTEXT",
        "## 4a. ROOFLINE EVIDENCE",
        "## 5. WARM-START RECIPE",
        "## 5b. RELATED LESSONS",
        "## 5c. KNOWN PITFALLS",
    ):
        assert banned not in user, f"perf section leaked into enablement prompt: {banned!r}"


def test_enablement_book_lives_in_user_not_system_prompt():
    """The per-task book/mandate stays out of the cached system prompt."""
    system, user = _build_split("enablement_specialist")
    assert "ENABLEMENT METHODOLOGY" not in system
    assert "ENABLEMENT METHODOLOGY" in user


def test_enablement_mandate_carries_the_dispatch_evidence():
    """Source context and ranked refs discovered by the Coordinator reach the mandate.

    The Coordinator computes both before dispatch; a mandate rendered without them
    tells the agent to find a bridge while withholding the candidates already found
    for it, and drops the checkpoint weight inventory a weight-init retry needs.
    """
    domain = get_domain("enablement_specialist")
    assert domain is not None
    weights = "CHECKPOINT WEIGHTS: model.layers.0.mlp.gate_up_proj.weight [8192, 4096]"
    inp = SpecialistPromptInputs(
        task_id="task-enablement-evidence",
        domain=domain,
        max_turns=4,
        gap_canonical_id="gap.enablement.weight_init",
        gap_symptom="KeyError: 'gate_up_proj' while loading weights",
        gap_layer=domain.layer,
        gap_evidence={"model": "Qwen/Qwen3-8B"},
        framework="vllm",
        enablement_source_context=weights,
        enablement_candidate_refs=("ROCm/vllm#123", "vllm-project/vllm#456"),
    )
    _system, user = build_specialist_prompts(inp)
    assert "SOURCE CONTEXT" in user
    assert weights in user
    assert "CANDIDATE BRIDGING" in user
    assert "ROCm/vllm#123" in user
    assert "vllm-project/vllm#456" in user


def test_enablement_mandate_omits_evidence_headers_when_not_supplied():
    """No dispatch evidence => no empty scaffolding, and the book still renders."""
    _system, user = _build_split("enablement_specialist")
    assert "## 1b. ENABLEMENT PLAYBOOK" in user
    assert "SOURCE CONTEXT" not in user
    assert "CANDIDATE BRIDGING" not in user


def test_perf_specialist_prompt_unchanged_keeps_perf_context():
    """A perf domain still carries roofline / recipe / KG sections and no ladder book."""
    _system, user = _build_split("serving_specialist")
    assert "## 4a. ROOFLINE EVIDENCE" in user
    assert "## 5. WARM-START RECIPE" in user
    assert "ENABLEMENT PLAYBOOK" not in user


def test_static_recon_shared_expert_model_features_line():
    """Model-features line includes shared_expert=True and n_shared= for shared-expert MoE."""
    domain = get_domain("static_recon_specialist")
    assert domain is not None
    inp = SpecialistPromptInputs(
        task_id="task-shared-moe",
        domain=domain,
        max_turns=4,
        gpu_type="MI355X",
        precision="mxfp8",
        model_info={
            "attention_type": "GQA",
            "is_moe": True,
            "quantization": "mxfp8",
            "has_shared_expert": True,
            "num_shared_experts": 1,
        },
        workspace_path="/tmp/test/recon",
    )
    system, user = build_specialist_prompts(inp)
    text = system + "\n" + user
    assert "shared_expert=True" in text
    assert "n_shared=1" in text
    assert "shared-expert fusion" in text.lower()


def test_static_recon_shared_expert_advisory_present():
    """Fusion advisory paragraph appears when model has shared expert."""
    domain = get_domain("static_recon_specialist")
    assert domain is not None
    inp = SpecialistPromptInputs(
        task_id="task-advisory",
        domain=domain,
        max_turns=4,
        gpu_type="MI355X",
        precision="mxfp8",
        model_info={
            "is_moe": True,
            "has_shared_expert": True,
            "num_shared_experts": 2,
        },
        workspace_path="/tmp/test/recon",
    )
    system, user = build_specialist_prompts(inp)
    text = system + "\n" + user
    assert "grouped-gemm" in text.lower() or "grouped gemm" in text.lower()
    assert "expert parallelism" in text.lower() or "EP" in text


def test_static_recon_no_shared_expert_no_advisory():
    """Neither shared_expert= nor fusion advisory must appear for plain MoE."""
    domain = get_domain("static_recon_specialist")
    assert domain is not None
    inp = SpecialistPromptInputs(
        task_id="task-plain-moe",
        domain=domain,
        max_turns=4,
        gpu_type="MI355X",
        precision="mxfp8",
        model_info={"attention_type": "GQA", "is_moe": True, "quantization": "mxfp8"},
        workspace_path="/tmp/test/recon",
    )
    system, user = build_specialist_prompts(inp)
    text = system + "\n" + user
    assert "shared_expert=" not in text
    assert "shared-expert fusion advisory" not in text.lower()


def test_static_recon_existing_markers_unaffected_by_shared_expert_change():
    """Existing static-recon prompt markers still render regardless of shared-expert state."""
    domain = get_domain("static_recon_specialist")
    assert domain is not None
    for model_info in (
        {"is_moe": False},
        {"is_moe": True, "has_shared_expert": True, "num_shared_experts": 1},
    ):
        inp = SpecialistPromptInputs(
            task_id="task-regression",
            domain=domain,
            max_turns=4,
            gpu_type="MI355X",
            precision="mxfp8",
            model_info=model_info,
            workspace_path="/tmp/test/recon",
        )
        system, user = build_specialist_prompts(inp)
        text = (system + "\n" + user).lower()
        for marker in ("static-recon", "bridge_candidates", "predicate_file", "never write a patch"):
            assert marker in text, f"regression: {marker!r} missing for model_info={model_info}"


# 3. SpecialistRunner no longer marks any domain as "generic template"
@pytest.mark.asyncio
async def test_runner_does_not_log_generic_template_for_any_domain(tmp_path):
    """When the active set covers a domain, the runner must NOT add a generic-template note."""
    from hyperloom.orchestrator.roles.mock_backend import (
        MockBackend,
        MockTurn,
        ScriptedPlan,
    )
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
    from hyperloom.orchestrator.specialists.runner import SpecialistRunner
    from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
    from hyperloom.orchestrator.state.task_registry import Task

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
    plan = ScriptedPlan(
        turns=[
            MockTurn(intents=[Intent(type=IntentType.SPECIALIST_DONE, payload=done)]),
        ]
    )
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
            f"PR-A6 should have widened SPECIALIST_DOMAIN_KEYS to cover kernel_switch_specialist; got note={note!r}"
        )


"""Specialist sub-agent framework tests."""


from dataclasses import dataclass
from typing import Any

import pytest

from hyperloom.orchestrator.roles.mock_backend import (
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from hyperloom.inference_optimizer.protocol.intent import (
    Intent,
    IntentType,
    IntentValidationError,
    validate_envelope,
)
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.policy.gate import (
    PolicyDenied,
    PolicyGate,
)
from hyperloom.orchestrator.bus.resource_lock import (
    KNOWN_LANES,
    LANE_CONFLICTS,
)
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.specialists.domains import (
    SPECIALIST_DOMAINS,
    SPECIALIST_DOMAIN_KEYS,
    SPECIALIST_MAX_TURNS_HARD_CAP,
    get_domain,
)
from hyperloom.orchestrator.specialists.runner import (
    SPECIALIST_TOOL_DENYLIST,
    SpecialistRunner,
    build_empty_specialist_done,
)
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    build_specialist_prompts,
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
        "proposal_set": proposals
        if proposals is not None
        else ([] if empty else [{"name": "v1", "extra_args": "--flag"}]),
        "empty": empty,
        "summary": "stub run summary",
    }
    if empty and "summary" not in (extras or {}):
        payload["summary"] = "no useful proposals this round"
    if extras:
        payload.update(extras)
    return payload


# 1. specialist_domains catalogue
def test_specialist_domains_catalogue_is_well_formed():
    """Every catalogue entry is complete and uniquely keyed.

    Guards what actually breaks a dispatch — a blank key, a missing KB anchor
    PolicyGate validates against, or a duplicate key that shadows an earlier
    entry — rather than the entry count.
    """
    assert SPECIALIST_DOMAINS
    assert SPECIALIST_DOMAIN_KEYS == frozenset(d.key for d in SPECIALIST_DOMAINS)
    assert len(SPECIALIST_DOMAIN_KEYS) == len(SPECIALIST_DOMAINS)
    for domain in SPECIALIST_DOMAINS:
        assert domain.key.strip(), "domain key must not be blank"
        assert domain.kb_anchor.strip(), f"{domain.key} has no KB anchor"
        assert domain.layer.strip(), f"{domain.key} has no layer label"
        assert domain.description.strip(), f"{domain.key} has no description"


def test_serving_specialist_is_M5_active():
    """The active set covers the full catalogue (every domain now has a focus template)."""
    assert "serving_specialist" in SPECIALIST_DOMAIN_KEYS


def test_get_domain_returns_none_for_unknown():
    assert get_domain("nonsense_specialist") is None
    assert get_domain("serving_specialist").kb_anchor == "framework"


# 2. intent_parser — SPECIALIST_DONE envelope round-trip
def test_specialist_done_envelope_passes_validation():
    envelope = {
        "intents": [
            {
                "intent_type": "specialist_done",
                "payload": _valid_done_payload(),
            }
        ]
    }
    intents = validate_envelope(envelope)
    assert len(intents) == 1
    assert intents[0].type == IntentType.SPECIALIST_DONE


def test_specialist_done_envelope_missing_required_field():
    bad_payload = _valid_done_payload()
    bad_payload.pop("summary")
    envelope = {
        "intents": [
            {
                "intent_type": "specialist_done",
                "payload": bad_payload,
            }
        ]
    }
    with pytest.raises(IntentValidationError, match="summary"):
        validate_envelope(envelope)


# 3. PolicyGate R2 — specialist_dispatch_source
def test_R2_orchestration_can_dispatch_specialist(gate):
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "specialist",
                "params": {
                    "domain": "serving_specialist",
                    "gap_canonical_id": "gap.kv.fp8",
                    "max_turns": 4,
                },
            },
        ),
    )


def test_R2_robustness_cannot_dispatch_specialist(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "robustness",
            Intent(
                type=IntentType.DELEGATE,
                payload={
                    "action_name": "specialist",
                    "params": {
                        "domain": "serving_specialist",
                        "gap_canonical_id": "gap.kv.fp8",
                    },
                },
            ),
        )
    assert exc.value.rule == "specialist_dispatch_source"
    assert "Orchestration" in (exc.value.hint or "")


def test_R2_unknown_domain_allowed(gate):
    """An unknown domain tag is observed, not denied; SpecialistRunner synthesizes an empty result."""
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "specialist",
                "params": {
                    "domain": "fake_specialist",
                    "gap_canonical_id": "gap.x",
                },
            },
        ),
    )


def test_R2_missing_gap_denied(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.DELEGATE,
                payload={
                    "action_name": "specialist",
                    "params": {"domain": "serving_specialist"},
                },
            ),
        )
    assert exc.value.rule == "specialist_dispatch_source"
    assert "gap" in str(exc.value)


def test_R2_max_turns_excess_denied(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.DELEGATE,
                payload={
                    "action_name": "specialist",
                    "params": {
                        "domain": "serving_specialist",
                        "gap_canonical_id": "gap.x",
                        "max_turns": SPECIALIST_MAX_TURNS_HARD_CAP + 1,
                    },
                },
            ),
        )
    assert exc.value.rule == "specialist_dispatch_source"
    assert "max_turns" in str(exc.value)


def test_R2_max_turns_zero_allowed_unbounded(gate):
    # max_turns=0 is accepted as "unbounded" (depth bounded by wall-clock budget).
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "specialist",
                "params": {
                    "domain": "serving_specialist",
                    "gap_canonical_id": "gap.x",
                    "max_turns": 0,
                },
            },
        ),
    )


def test_R2_max_turns_negative_denied(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.DELEGATE,
                payload={
                    "action_name": "specialist",
                    "params": {
                        "domain": "serving_specialist",
                        "gap_canonical_id": "gap.x",
                        "max_turns": -1,
                    },
                },
            ),
        )
    assert exc.value.rule == "specialist_dispatch_source"
    assert "max_turns" in str(exc.value)


# research_lane lane registration.
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
def _build_serving_prompt(**kwargs: Any) -> tuple[str, str]:
    domain = get_domain("serving_specialist")
    assert domain is not None
    return build_specialist_prompts(SpecialistPromptInputs(domain=domain, **kwargs))


def test_prompt_builder_emits_nine_sections():
    sys_p, usr_p = _build_serving_prompt(
        task_id="task-001",
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
    assert "## 6. PR MONITOR" in usr_p
    assert "## 7. LOCAL SOURCE NAVIGATION HINT" in usr_p


def test_prompt_builder_uses_none_placeholder_for_empty_sections():
    sys_p, usr_p = _build_serving_prompt(
        task_id="task-002",
    )
    # Several user-side sections will be empty → "(none)" placeholder.
    assert "(none)" in usr_p


def test_prompt_builder_pr_monitor_unavailable_renders_explanatory_line():
    sys_p, usr_p = _build_serving_prompt(
        task_id="task-003",
        pr_monitor_available=False,
    )
    assert "unavailable" in usr_p


def test_pr_monitor_section_lists_all_granted_tools():
    """Every tool in PR_MONITOR_TOOL_NAMES must appear in the rendered §6."""
    from hyperloom.orchestrator.policy.gate import PR_MONITOR_TOOL_NAMES

    _sys, usr_p = _build_serving_prompt(task_id="tool-drift", pr_monitor_available=True)
    prefix = "mcp__pr_monitor__"
    for tool_full in PR_MONITOR_TOOL_NAMES:
        short = tool_full[len(prefix) :]
        assert short in usr_p, (
            f"Granted PR Monitor tool '{short}' is not advertised in §6 PR MONITOR. "
            "Update _section_pr_feed in specialist_prompt_builder.py."
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
    plan = ScriptedPlan(
        turns=[
            MockTurn(
                intents=[
                    Intent(type=IntentType.SPECIALIST_DONE, payload=done_payload),
                ]
            )
        ]
    )

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
    """When the backend never emits specialist_done, the runner caps at max_turns and synthesises an empty done."""
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


@pytest.mark.asyncio
async def test_specialist_runner_backend_error_synthesises_empty_done(tmp_path):
    from hyperloom.orchestrator.roles.base import BackendError

    plan = ScriptedPlan(
        turns=[
            MockTurn(raise_error=BackendError("rate limited")),
        ]
    )
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


def test_specialist_tool_denylist_blocks_dangerous_tools():
    """KillShell and SlashCommand are denied to enforce the process-safety prompt rule."""
    assert "KillShell" in SPECIALIST_TOOL_DENYLIST
    assert "SlashCommand" in SPECIALIST_TOOL_DENYLIST
    for write_tool in ("Edit", "Write", "MultiEdit"):
        assert write_tool not in SPECIALIST_TOOL_DENYLIST


def test_build_empty_specialist_done_shape():
    """The failure-path helper must always produce a well-formed done payload."""
    done = build_empty_specialist_done(
        gap_canonical_id="gap.x",
        domain="serving_specialist",
        reason="example reason",
    )
    assert done["gap_canonical_id"] == "gap.x"
    assert done["domain"] == "serving_specialist"
    assert done["proposal_set"] == []
    assert done["empty"] is True
    assert done["summary"]


# 8. SharedState specialist round bookkeeping
def test_shared_state_specialist_rounds_default_empty():
    s = SharedState()
    assert s.specialist_rounds == []
    assert s.last_specialist == {}
    assert s.research_lane_capacity == 1
    assert s.rounds_since_last_specialist == {}
    assert s.rounds_since_last_keep == {}


# 8b. Per-anchor coverage counters (point 1)
def test_domain_round_counters_tick_all_anchors():
    from hyperloom.orchestrator.specialists.domains import (
        KNOWLEDGE_DOMAIN_TAGS,
    )

    s = SharedState()
    s.bump_domain_round_counters()
    s.bump_domain_round_counters()
    for anchor in KNOWLEDGE_DOMAIN_TAGS:
        assert s.rounds_since_last_specialist[anchor] == 2
        assert s.rounds_since_last_keep[anchor] == 2


def test_note_specialist_dispatched_resets_only_its_anchor():
    s = SharedState()
    s.bump_domain_round_counters()
    s.bump_domain_round_counters()
    # serving_specialist maps to the "framework" kb_anchor.
    s.note_specialist_dispatched("serving_specialist")
    assert s.rounds_since_last_specialist["framework"] == 0
    # A different anchor is untouched.
    assert s.rounds_since_last_specialist["kernel_agent"] == 2
    # keep counter is independent of the dispatch reset.
    assert s.rounds_since_last_keep["framework"] == 2


def test_note_domain_keep_resets_keep_counter_by_anchor():
    s = SharedState()
    s.bump_domain_round_counters()
    s.note_domain_keep("framework")
    assert s.rounds_since_last_keep["framework"] == 0
    assert s.rounds_since_last_specialist["framework"] == 1


def test_best_gap_for_anchor_picks_high_severity_least_attempted():
    s = SharedState()
    s.upsert_gap({"canonical_id": "gap.a", "domain_hint": "serving_specialist", "severity": "medium"})
    s.upsert_gap({"canonical_id": "gap.b", "domain_hint": "framework", "severity": "high"})
    # framework is serving_specialist's anchor -> both resolve to "framework".
    assert s.best_gap_for_anchor("framework") == "gap.b"
    assert s.best_gap_for_anchor("serving_specialist") == "gap.b"
    # An anchor with no matching gap returns "".
    assert s.best_gap_for_anchor("communication") == ""


def test_stalled_domains_reports_over_threshold_widest_first():
    s = SharedState()
    for _ in range(5):
        s.bump_domain_round_counters()
    # framework recently dispatched -> below specialist threshold.
    s.note_specialist_dispatched("framework")
    s.note_domain_keep("framework")
    stalled = s.stalled_domains(specialist_threshold=3, keep_threshold=3)
    assert "framework" not in stalled
    assert "kernel_agent" in stalled  # never dispatched/kept -> at 5
    # Ordering: widest gap first, ties broken by anchor name.
    assert stalled == sorted(
        stalled,
        key=lambda a: (
            -max(
                s.rounds_since_last_specialist.get(a, 0),
                s.rounds_since_last_keep.get(a, 0),
            ),
            a,
        ),
    )


def test_record_specialist_round_dedup_by_round_id():
    s = SharedState()
    s.record_specialist_round(
        {
            "round_id": "explore-001",
            "domains": ["serving_specialist"],
            "proposals_total": 2,
        }
    )
    s.record_specialist_round(
        {
            "round_id": "explore-001",
            "domains": ["serving_specialist"],
            "proposals_total": 5,
        }
    )
    s.record_specialist_round(
        {
            "round_id": "explore-002",
            "domains": ["serving_specialist"],
            "proposals_total": 1,
        }
    )
    assert len(s.specialist_rounds) == 2
    by_round = {r["round_id"]: r for r in s.specialist_rounds}
    assert by_round["explore-001"]["proposals_total"] == 5


def test_update_last_specialist_snapshot():
    s = SharedState()
    s.update_last_specialist(
        {
            "task_id": "task-001",
            "domain": "serving_specialist",
            "status": "succeeded",
        }
    )
    assert s.last_specialist["task_id"] == "task-001"
    # Non-dict inputs are ignored.
    s.update_last_specialist("garbage")  # type: ignore[arg-type]
    assert s.last_specialist["task_id"] == "task-001"


def test_research_lane_capacity_is_core_state_field():
    """LLM cannot raise research_lane_capacity mid-flight."""
    from hyperloom.orchestrator.policy.gate import CORE_STATE_FIELDS

    assert "research_lane_capacity" in CORE_STATE_FIELDS
    assert "gpu_specialist_capacity" in CORE_STATE_FIELDS
    assert "specialist_rounds" in CORE_STATE_FIELDS
    assert "last_specialist" in CORE_STATE_FIELDS


# --------------------------------------------------------------------------- #
# Read-only specialists never receive the patch-authoring contract
# --------------------------------------------------------------------------- #
# Derived from the property under test: a research-mode domain is exactly one
# the registry declares as such, so a new one is covered without an edit here.
READONLY_DOMAIN_KEYS = tuple(sorted(d.key for d in SPECIALIST_DOMAINS if d.default_mode == "research"))

# Every phrase that promises patch authoring, a worktree, or a GPU. A
# research-mode dispatch is leased none of them.
PATCH_CAPABILITY_PHRASES = (
    "author source patches",
    "optionally author patches",
    "patches_written",
    "artifacts_written",
    "your own worktree",
    "VISIBLE_DEVICES",
)


def test_readonly_domains_never_grant_patch_authoring():
    """Read-only domains must not be told they may author patches anywhere in
    the prompt — identity, iron rules, and output protocol alike."""
    for key in READONLY_DOMAIN_KEYS:
        system, user = build_specialist_prompts(
            SpecialistPromptInputs(
                task_id=f"task-{key}",
                domain=get_domain(key),
                max_turns=4,
                mode="research",
                gap_canonical_id=f"gap.{key}.test",
                workspace_path=f"/ws/{key}",
            )
        )
        whole = system + user
        for phrase in PATCH_CAPABILITY_PHRASES:
            assert phrase not in whole, f"{key} prompt leaks {phrase!r} to a read-only dispatch"


def test_readonly_dispatch_states_the_read_only_boundary():
    """The iron rule that replaces the staging grant must say so explicitly."""
    system, _ = build_specialist_prompts(
        SpecialistPromptInputs(
            task_id="task-ro",
            domain=get_domain("static_recon_specialist"),
            max_turns=4,
            mode="research",
            workspace_path="/ws/ro",
        )
    )
    assert "Read-only dispatch:" in system
    assert "MUST NOT author" in system


def test_cross_domain_research_dispatch_drops_patch_deliverable():
    """Mode outranks scope: a read-only `domains` dispatch must not be promised
    the coupled cross-domain patch."""
    _, user = build_specialist_prompts(
        SpecialistPromptInputs(
            task_id="task-domains-ro",
            domain=get_domain("candidate_discovery_specialist"),
            max_turns=4,
            scope="domains",
            mode="research",
            gap_canonical_id="gap.domains.test",
            workspace_path="/ws/domains-ro",
        )
    )
    assert "- deliverable: findings and up to 6 ranked config variants (read-only; no patch)" in user
    assert "coupled patch" not in user


def test_freeform_research_dispatch_drops_patch_deliverable():
    """A bare freeform dispatch resolves to research mode, so its mandate must
    not promise a patch deliverable."""
    system, user = build_specialist_prompts(
        SpecialistPromptInputs(
            task_id="task-ff",
            domain=get_domain("serving_specialist"),
            max_turns=4,
            scope="freeform",
            mode="research",
            task_description="look around",
            workspace_path="/ws/ff",
        )
    )
    whole = system + user
    for phrase in PATCH_CAPABILITY_PHRASES:
        assert phrase not in whole, f"freeform research prompt leaks {phrase!r}"


def test_patch_mode_keeps_full_authoring_contract():
    """The gating must not strip anything from a patch-capable dispatch."""
    system, user = build_specialist_prompts(
        SpecialistPromptInputs(
            task_id="task-patch",
            domain=get_domain("serving_specialist"),
            max_turns=4,
            mode="patch",
            allocated_gpu_ids=[0, 1],
            gap_canonical_id="gap.serving.test",
            workspace_path="/ws/patch",
        )
    )
    whole = system + user
    for phrase in (
        "author source patches",
        "patches_written",
        "artifacts_written",
        "your own worktree",
        "VISIBLE_DEVICES",
    ):
        assert phrase in whole, f"patch-mode prompt lost {phrase!r}"


def test_patch_mode_without_gpu_keeps_the_authoring_clause():
    """The no-GPU iron rule still offers patch authoring in patch mode; only
    research mode drops it."""
    kwargs = dict(
        task_id="task-patch-cpu",
        domain=get_domain("serving_specialist"),
        max_turns=4,
        allocated_gpu_ids=[],
        workspace_path="/ws/patch-cpu",
    )
    patch_system, _ = build_specialist_prompts(SpecialistPromptInputs(mode="patch", **kwargs))
    research_system, _ = build_specialist_prompts(SpecialistPromptInputs(mode="research", **kwargs))
    assert "optionally author patches" in patch_system
    assert "optionally author patches" not in research_system


# --------------------------------------------------------------------------- #
# Stage-2 guard: mandate section carries run-status when available
# --------------------------------------------------------------------------- #
def test_mandate_section_renders_run_status():
    """§0 MANDATE must contain baseline, validated gain, and KEEP threshold
    when those fields are non-zero."""
    domain = get_domain("serving_specialist")
    assert domain is not None
    inp = SpecialistPromptInputs(
        task_id="task-mandate",
        domain=domain,
        max_turns=4,
        gap_canonical_id="gap.serving.test",
        framework="vllm",
        baseline_tput=6232.0,
        current_tput=6848.5,
        cumulative_gain_validated=9.89,
        keep_threshold_pct=1.0,
        applied_stack=[{"variant_name": "--kv-cache-dtype fp8_e4m3", "gain_pct": 5.99}],
    )
    _, user = build_specialist_prompts(inp)
    assert "## 0. MANDATE" in user
    assert "6232" in user
    assert "9.89" in user or "+9.89" in user
    assert "1.00%" in user or "1.0%" in user
    assert "--kv-cache-dtype fp8_e4m3" in user


# --------------------------------------------------------------------------- #
# Stage-3 guard: enablement ladder book appears exactly once
# --------------------------------------------------------------------------- #
def test_enablement_ladder_rendered_exactly_once():
    """ENABLEMENT METHODOLOGY must appear only in §1b, not also in §10."""
    domain = get_domain("enablement_specialist")
    assert domain is not None
    inp = SpecialistPromptInputs(
        task_id="task-enablement-dedup",
        domain=domain,
        max_turns=4,
        gap_canonical_id="gap.enablement.unknown",
        gap_symptom="vllm cannot launch ModelFoo: unknown",
        gap_evidence={"model": "ModelFoo", "failure_kind": "unknown"},
        framework="vllm",
        # notes is empty (no stacked patches, no build failure)
        notes="",
    )
    _, user = build_specialist_prompts(inp)
    count = user.count("ENABLEMENT METHODOLOGY")
    assert count == 1, f"Expected 'ENABLEMENT METHODOLOGY' exactly once, found {count}"


# --------------------------------------------------------------------------- #
# Stage-4 guard: KB-write iron rule is gone
# --------------------------------------------------------------------------- #
def test_kb_write_iron_rule_absent():
    """Rule 3 (KB writes) is dead text — confirm it no longer appears."""
    domain = get_domain("serving_specialist")
    assert domain is not None
    inp = SpecialistPromptInputs(
        task_id="task-iron",
        domain=domain,
        max_turns=4,
        gap_canonical_id="gap.serving.iron_rule",
    )
    system, _ = build_specialist_prompts(inp)
    assert "RecipeKB.put_recipe" not in system
    assert "NEVER** write to the Recipe KB" not in system
