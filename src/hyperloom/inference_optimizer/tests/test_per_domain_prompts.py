# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-domain specialist prompt templates."""

from __future__ import annotations

import pytest

from hyperloom.common.failure_signature import classify_failure
from hyperloom.orchestrator.specialists.domains import (
    SPECIALIST_DOMAINS,
    get_domain,
)
from hyperloom.orchestrator.prompts.specialist_focus import _DOMAIN_FOCUS_TEMPLATES
from hyperloom.orchestrator.specialists.prompt_inputs import SpecialistPromptInputs
from hyperloom.orchestrator.prompts.specialist_prompt_builder import build_specialist_prompts

# Every enablement round is dispatched on a classified verdict.
_DISPATCHED = classify_failure(
    "ValueError: Model architectures ['DeepseekV4ForCausalLM'] are not supported for now."
).to_dict()


def _build(domain_key: str) -> str:
    domain = get_domain(domain_key)
    assert domain is not None, domain_key
    inp = SpecialistPromptInputs(
        domain=domain,
        max_turns=4,
        gap_canonical_id=f"gap.{domain_key}.example",
        gap_symptom="example symptom",
        gap_layer=domain.layer,
        framework="sglang",
        workspace_path=f"/tmp/test/{domain_key}",
        enablement_failure_signature=_DISPATCHED,
    )
    system, user = build_specialist_prompts(inp)
    return system + "\n" + user


# 1. Coverage — every catalogue domain has a focus template
def test_every_domain_has_focus_template():
    for domain in SPECIALIST_DOMAINS:
        assert domain.key in _DOMAIN_FOCUS_TEMPLATES, f"missing per-domain template for {domain.key!r}"


# 2. Per-domain output contracts — the keys each consumer parses back
def test_candidate_discovery_specialist_schema_is_an_upstream_candidate():
    """The FRAMEWORK funnel drops a discovery proposal that carries neither pr_url nor head_sha, and branches on its verdict."""
    text = _build("candidate_discovery_specialist")
    assert '"pr_url"' in text and '"head_sha"' in text
    assert '"extra_args"' not in text
    for verdict in ("already_present", "not_applicable", "worth_a_bench"):
        assert verdict in text, f"missing verdict {verdict!r}"


def test_static_recon_prompt_names_the_keys_the_recon_consumer_reads():
    """consume_static_recon seeds a gap only from a ``bridge_candidates`` entry carrying ``predicate_file`` and ``why_disabled_here``."""
    text = _build("static_recon_specialist")
    for key in ("bridge_candidates", "predicate_file", "why_disabled_here"):
        assert key in text, f"missing {key!r}"


def test_static_recon_specialist_renders_seed_checklist_and_model_info():
    """When a seed checklist + model_info are supplied they render into the prompt."""
    domain = get_domain("static_recon_specialist")
    assert domain is not None
    inp = SpecialistPromptInputs(
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


def _build_split(domain_key: str) -> tuple[str, str]:
    domain = get_domain(domain_key)
    assert domain is not None, domain_key
    inp = SpecialistPromptInputs(
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
        framework="sglang",
        workspace_path=f"/tmp/test/{domain_key}",
        enablement_failure_signature=_DISPATCHED,
    )
    return build_specialist_prompts(inp)


def test_enablement_user_prompt_keeps_the_gap_and_the_tools_to_find_a_fix():
    """Stripping the perf context leaves the dispatched gap, the PR query tools and the source trees."""
    from hyperloom.orchestrator.policy.gate import PR_MONITOR_TOOL_NAMES

    domain = get_domain("enablement_specialist")
    assert domain is not None
    _system, user = build_specialist_prompts(
        SpecialistPromptInputs(
            domain=domain,
            max_turns=4,
            gap_canonical_id="gap.enablement.example",
            gap_symptom="server exited during the model registry lookup",
            framework="sglang",
            framework_source_roots=("/src/sglang",),
            enablement_failure_signature=_DISPATCHED,
        )
    )
    assert "server exited during the model registry lookup" in user
    assert "/src/sglang" in user
    for tool in PR_MONITOR_TOOL_NAMES:
        assert tool.removeprefix("mcp__pr_monitor__") in user, tool


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
    """Source context and ranked refs discovered by the Coordinator reach the mandate."""
    domain = get_domain("enablement_specialist")
    assert domain is not None
    weights = "CHECKPOINT WEIGHTS: model.layers.0.mlp.gate_up_proj.weight [8192, 4096]"
    inp = SpecialistPromptInputs(
        domain=domain,
        max_turns=4,
        gap_canonical_id="gap.enablement.weight_init",
        gap_symptom="KeyError: 'gate_up_proj' while loading weights",
        gap_layer=domain.layer,
        gap_evidence={"model": "Qwen/Qwen3-8B"},
        framework="vllm",
        enablement_source_context=weights,
        enablement_candidate_refs=("ROCm/vllm#123", "vllm-project/vllm#456"),
        enablement_failure_signature=_DISPATCHED,
    )
    _system, user = build_specialist_prompts(inp)
    assert "SOURCE CONTEXT" in user
    assert weights in user
    assert "CANDIDATE BRIDGING" in user
    assert "ROCm/vllm#123" in user
    assert "vllm-project/vllm#456" in user


def test_enablement_mandate_renders_the_dispatched_signature():
    """The verdict the round was dispatched on reaches the prompt verbatim; the builder never re-classifies a log to recover it."""
    domain = get_domain("enablement_specialist")
    assert domain is not None
    signature = classify_failure(
        'Traceback (most recent call last):\n  File "/opt/vllm/vllm/model_executor/models/registry.py", line 7, in resolve\n'
        "ValueError: Model architectures ['GlmForCausalLM'] are not supported for now."
    )
    inp = SpecialistPromptInputs(
        domain=domain,
        max_turns=4,
        gap_canonical_id="gap.enablement.missing_arch",
        gap_symptom="boot failed",
        gap_layer=domain.layer,
        gap_evidence={"model": "zai-org/GLM-5"},
        framework="vllm",
        enablement_failure_signature=signature.to_dict(),
    )
    _system, user = build_specialist_prompts(inp)
    assert "FAILURE CLASS: missing_model_arch" in user
    assert "/opt/vllm/vllm/model_executor/models/registry.py" in user
    assert "GlmForCausalLM" in user


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


def test_static_recon_no_shared_expert_no_advisory():
    """Neither shared_expert= nor fusion advisory must appear for plain MoE."""
    domain = get_domain("static_recon_specialist")
    assert domain is not None
    inp = SpecialistPromptInputs(
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


# 3. SpecialistRunner no longer marks any domain as "generic template"
@pytest.mark.asyncio
async def test_runner_does_not_log_generic_template_for_any_domain(tmp_path, monkeypatch):
    """When the active set covers a domain, the runner must NOT add a generic-template note."""
    from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
    from hyperloom.orchestrator.state.task_registry import Task

    done = {
        "gap_canonical_id": "gap.x",
        "domain": "kernel_switch_specialist",
        "proposal_set": [],
        "summary": "test",
        "reason": "test",
        "confidence": 0.0,
        "new_findings": [],
        "residual_questions": [],
    }
    runner = _cli_runner(tmp_path, monkeypatch, behavior="done_only", payload=done)
    task = Task(
        task_id="t-kernel",
        kind="specialist",
        state="queued",
        params={
            "domain": "kernel_switch_specialist",
            "framework": "sglang",
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

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.policy.denial import PolicyDenied
from hyperloom.orchestrator.policy.gate import (
    PolicyGate,
)
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.specialists.domains import (
    SPECIALIST_DOMAINS,
    SPECIALIST_DOMAIN_KEYS,
    SPECIALIST_MAX_TURNS_HARD_CAP,
    get_domain,
)
from hyperloom.orchestrator.specialists.runner import (
    SpecialistRunner,
    SpecialistSubprocessConfig,
)
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.specialists.prompt_inputs import SpecialistPromptInputs
from hyperloom.orchestrator.prompts.specialist_prompt_builder import build_specialist_prompts

from .conftest import init_git_repo, make_fake_claude


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
        "summary": "stub run summary",
    }
    if empty and "summary" not in (extras or {}):
        payload["summary"] = "no useful proposals this round"
    if extras:
        payload.update(extras)
    return payload


# 1. specialist_domains catalogue
def test_specialist_domains_catalogue_is_well_formed():
    """Every catalogue entry is complete and uniquely keyed."""
    assert SPECIALIST_DOMAINS
    assert SPECIALIST_DOMAIN_KEYS == frozenset(d.key for d in SPECIALIST_DOMAINS)
    assert len(SPECIALIST_DOMAIN_KEYS) == len(SPECIALIST_DOMAINS)
    for domain in SPECIALIST_DOMAINS:
        assert domain.key.strip(), "domain key must not be blank"
        assert domain.kb_anchor.strip(), f"{domain.key} has no KB anchor"
        assert domain.layer.strip(), f"{domain.key} has no layer label"
        assert domain.description.strip(), f"{domain.key} has no description"


def test_get_domain_returns_none_for_unknown():
    assert get_domain("nonsense_specialist") is None
    assert get_domain("serving_specialist").kb_anchor == "framework"


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


# 6. specialist_prompt_builder — PR monitor section
def _build_serving_prompt(**kwargs: Any) -> tuple[str, str]:
    domain = get_domain("serving_specialist")
    assert domain is not None
    return build_specialist_prompts(SpecialistPromptInputs(domain=domain, **kwargs))


def test_pr_monitor_section_lists_all_granted_tools():
    """Every tool in PR_MONITOR_TOOL_NAMES must appear in the rendered §6."""
    from hyperloom.orchestrator.policy.gate import PR_MONITOR_TOOL_NAMES

    _sys, usr_p = _build_serving_prompt(pr_monitor_available=True)
    prefix = "mcp__pr_monitor__"
    for tool_full in PR_MONITOR_TOOL_NAMES:
        short = tool_full[len(prefix) :]
        assert short in usr_p, (
            f"Granted PR Monitor tool '{short}' is not advertised in §6 PR MONITOR. "
            "Update _section_pr_feed in specialist_prompt_builder.py."
        )


# 7. SpecialistRunner — happy path + failure synth
def _cli_runner(tmp_path, monkeypatch, *, behavior: str, payload: dict[str, Any] | None = None) -> SpecialistRunner:
    """A runner whose specialist is a fake ``claude`` CLI over a git framework checkout."""
    repo = tmp_path / "framework"
    init_git_repo(repo)
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(repo))
    return SpecialistRunner(
        subprocess_config=SpecialistSubprocessConfig(
            claude_executable=str(make_fake_claude(tmp_path / "bin", behavior=behavior, payload=payload)),
            framework_source_roots=(str(repo),),
            poll_interval_seconds=0.2,
        ),
        session_dir=tmp_path / "session",
    )


@pytest.mark.asyncio
async def test_specialist_runner_happy_path(tmp_path, monkeypatch):
    """The CLI writes a valid specialist_done; runner persists files."""
    done_payload = _valid_done_payload(
        proposals=[
            {"name": "max_seqs_512", "extra_args": "--max-num-seqs 512"},
            {"name": "kv_fp8", "extra_args": "--kv-cache-dtype fp8"},
        ],
    )
    runner = _cli_runner(tmp_path, monkeypatch, behavior="done_only", payload=done_payload)
    task = _StubTask(
        task_id="task-xyz",
        params={
            "domain": "serving_specialist",
            "framework": "sglang",
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

    workspace = tmp_path / "session" / "runs" / "specialist" / "task-xyz"
    assert (workspace / "prompt.md").exists()
    assert (workspace / "transcript.jsonl").exists()
    assert (workspace / "heartbeat.json").exists()
    assert (workspace / "specialist_done.json").exists()
    # The identity section reaches the CLI as its system prompt.
    system_text = (workspace / "system_prompt.md").read_text(encoding="utf-8")
    assert "## 1. IDENTITY & AUTONOMY" in system_text
    transcript_text = (workspace / "transcript.jsonl").read_text(encoding="utf-8")
    assert "subprocess_result" in transcript_text


@pytest.mark.asyncio
async def test_specialist_runner_synthesises_empty_done_when_the_cli_writes_none(tmp_path, monkeypatch):
    """A CLI that exits cleanly without a specialist_done gets an empty done synthesised for it."""
    runner = _cli_runner(tmp_path, monkeypatch, behavior="no_done")
    task = _StubTask(
        task_id="task-stale",
        params={
            "domain": "serving_specialist",
            "framework": "sglang",
            "gap_canonical_id": "gap.x",
            "max_turns": 2,
        },
    )
    ctx = RunnerContext(task=task, lease=None, extra={})
    result = await runner.run(ctx)

    assert result.status == "empty_synthesised"
    assert result.specialist_done["proposal_set"] == []
    assert result.specialist_done["domain"] == "serving_specialist"
    assert result.specialist_done["reason"] == "no_specialist_done_emitted"


@pytest.mark.asyncio
async def test_specialist_runner_unknown_domain_synthesises_empty(tmp_path):
    runner = SpecialistRunner(
        subprocess_config=SpecialistSubprocessConfig(),
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
    assert result.specialist_done["proposal_set"] == []
    assert "unknown specialist domain" in result.specialist_done["reason"]


# 8. SharedState specialist round bookkeeping
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


# --------------------------------------------------------------------------- # Read-only specialists never receive
# the patch-authoring contract --------------------------------------------------------------------------- # Derived
# from the property under test: a research-mode domain is exactly one the registry declares as such, so a new one is
# covered without an edit here.
READONLY_DOMAIN_KEYS = tuple(sorted(d.key for d in SPECIALIST_DOMAINS if d.default_mode == "research"))

# Every phrase that promises patch authoring, a worktree, or a GPU.
PATCH_CAPABILITY_PHRASES = (
    "author source patches",
    "optionally author patches",
    "patches_written",
    "artifacts_written",
    "your own worktree",
    "VISIBLE_DEVICES",
)


def test_readonly_domains_never_grant_patch_authoring():
    """Read-only domains must not be told they may author patches anywhere in the prompt — identity, iron rules, and output protocol alike."""
    for key in READONLY_DOMAIN_KEYS:
        system, user = build_specialist_prompts(
            SpecialistPromptInputs(
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
            domain=get_domain("static_recon_specialist"),
            max_turns=4,
            mode="research",
            workspace_path="/ws/ro",
        )
    )
    assert "Read-only dispatch:" in system
    assert "MUST NOT author" in system


def test_cross_domain_research_dispatch_drops_patch_deliverable():
    """Mode outranks scope: a read-only `domains` dispatch must not be promised the coupled cross-domain patch."""
    _, user = build_specialist_prompts(
        SpecialistPromptInputs(
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
    """A bare freeform dispatch resolves to research mode, so its mandate must not promise a patch deliverable."""
    system, user = build_specialist_prompts(
        SpecialistPromptInputs(
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
    """The no-GPU iron rule still offers patch authoring in patch mode; only research mode drops it."""
    kwargs = dict(
        domain=get_domain("serving_specialist"),
        max_turns=4,
        allocated_gpu_ids=[],
        workspace_path="/ws/patch-cpu",
    )
    patch_system, _ = build_specialist_prompts(SpecialistPromptInputs(mode="patch", **kwargs))
    research_system, _ = build_specialist_prompts(SpecialistPromptInputs(mode="research", **kwargs))
    assert "optionally author patches" in patch_system
    assert "optionally author patches" not in research_system


# --------------------------------------------------------------------------- # Stage-2 guard: mandate section carries
# run-status when available --------------------------------------------------------------------------- #
def test_mandate_section_renders_run_status():
    """§0 MANDATE must contain baseline, validated gain, and KEEP threshold when those fields are non-zero."""
    domain = get_domain("serving_specialist")
    assert domain is not None
    inp = SpecialistPromptInputs(
        domain=domain,
        max_turns=4,
        gap_canonical_id="gap.serving.test",
        framework="vllm",
        baseline_tput=6232.0,
        current_tput=6848.5,
        current_launch_args="--moe-backend aiter --kv-cache-dtype fp8_e4m3",
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
    assert "--moe-backend aiter --kv-cache-dtype fp8_e4m3" in user


# --------------------------------------------------------------------------- # Stage-3 guard: enablement ladder book
# appears exactly once --------------------------------------------------------------------------- #
def test_enablement_ladder_rendered_exactly_once():
    """ENABLEMENT METHODOLOGY must appear only in §1b, not also in §10."""
    domain = get_domain("enablement_specialist")
    assert domain is not None
    inp = SpecialistPromptInputs(
        domain=domain,
        max_turns=4,
        gap_canonical_id="gap.enablement.unknown",
        gap_symptom="vllm cannot launch ModelFoo: unknown",
        gap_evidence={"model": "ModelFoo", "failure_kind": "unknown"},
        framework="vllm",
        # notes is empty (no stacked patches, no build failure)
        notes="",
        enablement_failure_signature=classify_failure("vllm cannot launch ModelFoo: unknown").to_dict(),
    )
    _, user = build_specialist_prompts(inp)
    count = user.count("ENABLEMENT METHODOLOGY")
    assert count == 1, f"Expected 'ENABLEMENT METHODOLOGY' exactly once, found {count}"


@pytest.mark.parametrize("no_source_changes", [True, False])
def test_a_patch_mode_mandate_says_only_configuration_is_kept_exactly_under_no_source_changes(no_source_changes):
    """A benching or enablement specialist still runs in patch mode under the flag; it is told what will be kept."""
    domain = get_domain("serving_specialist")
    inp = SpecialistPromptInputs(
        domain=domain,
        max_turns=4,
        gap_canonical_id="gap.serving.example",
        gap_symptom="example symptom",
        framework="vllm",
        workspace_path="/tmp/test/serving",
        no_source_changes=no_source_changes,
    )
    system, user = build_specialist_prompts(inp)
    text = system + "\n" + user
    assert ("deliver configuration proposals" in text) is no_source_changes
    assert ("Altitude — where the durable wins are" in text) is not no_source_changes
