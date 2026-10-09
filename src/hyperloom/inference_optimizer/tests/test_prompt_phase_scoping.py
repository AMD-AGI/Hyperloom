# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase scoping of the orchestration system prompt and the Critic judge bundle."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hyperloom.inference_optimizer.session.paths import asset_prompt_references_dir, asset_system_prompts_dir
from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.orchestrator.phases import machine_state as _ps
from hyperloom.orchestrator.prompts.prompt_builder import (
    _filter_rules_fragment,
    build_orchestration_prompt,
    default_enabled_actions,
)


# Marker text identifying each phase-scoped prompt module.
KERNEL_REQUEST_REF = "## 6. KERNEL-OPT REQUEST REFERENCE"
IDEA_GENERATION = "### IDEA GENERATION"
BASELINE_FINGERPRINT = "eight params fields"  # now in the reference doc, not in the prompt
SPECIALIST_DIALS = "### One specialist, three dials"
SPECIALIST_WATCH = "### Watching a running specialist"
SPECIALIST_DOMAIN = "### Choosing specialist domain"
WEB_SEARCH = "### Web search"
KERNEL_REQUEST_KINDS = "### Kernel request kinds"
ROOFLINE_BLOCK = "### Roofline / profile analysis"

# One goal block per phase.
PHASE_GOAL_BLOCKS = {
    _ps.PHASE_PRELUDE: "### PRELUDE — phase goal",
    _ps.PHASE_ENABLEMENT: "### ENABLEMENT — phase goal",
    _ps.PHASE_FRAMEWORK_AGENT: "### OPTIMIZE — phase goal",
    _ps.PHASE_KERNEL_AGENT: "### KERNEL — phase goal",
    _ps.PHASE_SWEEP: "### SWEEP — phase goal",
    _ps.PHASE_CLOSE: "### CLOSE — phase goal",
}

# Analysis-driven targeting is unreachable once the levers are gone.
ROOFLINE_PHASES = {
    _ps.PHASE_PRELUDE,
    _ps.PHASE_ENABLEMENT,
    _ps.PHASE_FRAMEWORK_AGENT,
    _ps.PHASE_KERNEL_AGENT,
}

# Only EXPLORE lets the LLM emit `delegate{specialist}`; FRAMEWORK_AGENT specialists come from the Coordinator's
# authoring pump but stay steerable.
SPECIALIST_DISPATCH_OPS = (SPECIALIST_DIALS, SPECIALIST_DOMAIN, WEB_SEARCH)
ALL_SPECIALIST_OPS = (*SPECIALIST_DISPATCH_OPS, SPECIALIST_WATCH)

ALWAYS_ON = (
    "## 1. MISSION",
    "## 2. SESSION CONTEXT",
    "## 3. PIPELINE & TIME BUDGET",
    "## 3a. PHASE CONTRACT",
    "## 4. ACTIONS YOU MAY USE",
    "## 5. DECISION FRAMEWORK",
    "## 7. RULES & OUTPUT PROTOCOL",
    "### Phase awareness",
    "### Hard rules",
    "### Pulling what the projection does not carry",
    "### SESSION_DIR contract",
    "### Output protocol",
    "RULE F3",
    "RULE F4",
)


@pytest.fixture(scope="module")
def registry() -> dict:
    return ACTION_CATALOGUE


def _build(registry: dict, phase: str) -> str:
    return build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        framework="sglang",
        kernel_enabled=True,
        framework_agent_phase_enabled=True,
        objective_kind="gain_pct",
        objective_value=15.0,
        max_minutes=480,
        phase=phase,
        rules_fragment_path=asset_system_prompts_dir() / "orchestration.md",
    )


def test_cycle_strategy_is_rendered_in_place_of_the_default_arc(registry):
    strategy = {
        "focus": "moe_dispatch",
        "score": 2.5,
        "rationale": "not saturated in latest roofline snapshot",
        "prior_cycles": [{"cycle": 1, "focus": "attention", "gain_delta": 1.2, "saturated_at_start": []}],
    }
    prompt = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        macro_cycle=2,
        cycle_strategy=strategy,
        phase="FRAMEWORK_AGENT",
        rules_fragment_path=asset_system_prompts_dir() / "orchestration.md",
    )

    assert "focus=moe_dispatch" in prompt
    assert "rationale: not saturated in latest roofline snapshot" in prompt
    assert "previous cycles:" in prompt
    assert "cycle=1 focus=attention" in prompt
    assert "Default arc" not in prompt


# Phase-scoped modules render only where the behaviour exists
def test_kernel_request_reference_only_in_kernel_phase(registry):
    """Kernel REQUEST payload templates are legal only in KERNEL_AGENT."""
    for phase in _ps.PHASE_NAMES:
        text = _build(registry, phase)
        if phase == _ps.PHASE_KERNEL_AGENT:
            assert KERNEL_REQUEST_REF in text
        else:
            assert KERNEL_REQUEST_REF not in text, f"kernel request ref leaked into {phase}"


def test_idea_generation_only_in_explore_phase(registry):
    """The explore grid idea pipeline is unreachable outside EXPLORE."""
    for phase in _ps.PHASE_NAMES:
        text = _build(registry, phase)
        if phase == _ps.PHASE_FRAMEWORK_AGENT:
            assert IDEA_GENERATION in text
        else:
            assert IDEA_GENERATION not in text, f"idea generation leaked into {phase}"


def test_baseline_recovery_detail_only_in_prelude(registry):
    """PRELUDE and ENABLEMENT render F1/F2 (the phases where baselines keep failing)."""
    refs_dir = asset_prompt_references_dir()
    for phase in _ps.PHASE_NAMES:
        text = _build(registry, phase)
        # Detailed fingerprint text lives in the reference doc, not in any prompt.
        assert BASELINE_FINGERPRINT not in text, f"baseline fingerprint detail leaked into {phase} prompt"
        if phase in (_ps.PHASE_PRELUDE, _ps.PHASE_ENABLEMENT):
            assert "RULE F1" in text
            assert "RULE F2" in text
        else:
            assert "RULE F1" not in text
            assert "RULE F2" not in text
    # The reference doc itself must contain the fingerprint detail.
    failure_ref = refs_dir / "failure_recovery.md"
    if failure_ref.exists():
        assert BASELINE_FINGERPRINT in failure_ref.read_text(encoding="utf-8")


def test_specialist_dispatch_prose_only_in_explore(registry):
    """How to shape a dispatch matters only where the LLM can emit one."""
    for phase in _ps.PHASE_NAMES:
        text = _build(registry, phase)
        for marker in SPECIALIST_DISPATCH_OPS:
            if phase == _ps.PHASE_FRAMEWORK_AGENT:
                assert marker in text, f"{marker} missing from {phase}"
            else:
                assert marker not in text, f"{marker} leaked into {phase}"


def test_specialist_watching_prose_spans_both_dispatching_phases(registry):
    """A live specialist can exist in EXPLORE and FRAMEWORK_AGENT; the LLM steers both."""
    with_specialists = {_ps.PHASE_FRAMEWORK_AGENT, _ps.PHASE_FRAMEWORK_AGENT}
    for phase in _ps.PHASE_NAMES:
        text = _build(registry, phase)
        if phase in with_specialists:
            assert SPECIALIST_WATCH in text, f"{SPECIALIST_WATCH} missing from {phase}"
        else:
            assert SPECIALIST_WATCH not in text, f"{SPECIALIST_WATCH} leaked into {phase}"


def test_phase_goal_blocks_render_only_in_their_own_phase(registry):
    """A phase states its own goal; the other phases' goals are unreachable from it."""
    for phase in _ps.PHASE_NAMES:
        text = _build(registry, phase)
        for owner, marker in PHASE_GOAL_BLOCKS.items():
            if owner == phase:
                assert marker in text, f"{marker} missing from its own phase"
            else:
                assert marker not in text, f"{marker} leaked into {phase}"


def test_kernel_request_kinds_only_in_kernel_phase(registry):
    """The request-kind whitelist is scoped to KERNEL_AGENT where the Coordinator routes them."""
    for phase in _ps.PHASE_NAMES:
        text = _build(registry, phase)
        if phase == _ps.PHASE_KERNEL_AGENT:
            assert KERNEL_REQUEST_KINDS in text
        else:
            assert KERNEL_REQUEST_KINDS not in text, f"kernel request kinds leaked into {phase}"


def test_roofline_targeting_drops_out_of_sweep_and_close(registry):
    """SWEEP validates and CLOSE reports; neither can act on a bottleneck class."""
    for phase in _ps.PHASE_NAMES:
        text = _build(registry, phase)
        if phase in ROOFLINE_PHASES:
            assert ROOFLINE_BLOCK in text, f"{ROOFLINE_BLOCK} missing from {phase}"
        else:
            assert ROOFLINE_BLOCK not in text, f"{ROOFLINE_BLOCK} leaked into {phase}"


def test_payload_contracts_render_in_every_phase(registry):
    """Payload templates are no longer phase-scoped, so every phase carries them."""
    for phase in _ps.PHASE_NAMES:
        text = _build(registry, phase)
        assert "GRID INPUT (REQUIRED)" in text
        assert "reasoning, provenance" in text
        assert "Every variant requires action-time reasoning" in text
        assert "EMIT: delegate{action_name='specialist'" in text
        assert "- **explore** —" in text
        assert "- **specialist** —" in text


@pytest.mark.parametrize("phase", _ps.PHASE_NAMES)
def test_always_on_modules_survive_every_phase(registry, phase):
    """Scoping never removes the north star or the cross-phase planning facts."""
    text = _build(registry, phase)
    for marker in ALWAYS_ON:
        assert marker in text, f"{marker} missing from {phase}"


def test_generic_recovery_survives_outside_prelude(registry):
    """Any action can fail, so the generic recovery surfaces stay everywhere."""
    text = _build(registry, _ps.PHASE_CLOSE)
    assert "### FAILURE RECOVERY" in text
    assert "last_action_failures" in text


# Back-compat: an unscoped build is a superset
def test_unscoped_build_renders_every_module(registry):
    """A caller that does not track phases keeps the pre-scoping prompt."""
    text = _build(registry, "")
    for marker in (
        KERNEL_REQUEST_REF,
        IDEA_GENERATION,
        KERNEL_REQUEST_KINDS,
        ROOFLINE_BLOCK,
        "GRID INPUT (REQUIRED)",
        *PHASE_GOAL_BLOCKS.values(),
        *ALL_SPECIALIST_OPS,
        *ALWAYS_ON,
    ):
        assert marker in text, f"{marker} missing from the unscoped build"
    # Detailed fingerprint text now lives in the reference doc, not in the prompt.
    assert BASELINE_FINGERPRINT not in text


def test_scoped_builds_are_strictly_smaller(registry):
    """Every phase pays less than the unscoped superset."""
    unscoped = len(_build(registry, "").splitlines())
    for phase in _ps.PHASE_NAMES:
        assert len(_build(registry, phase).splitlines()) < unscoped, f"{phase} did not shrink"


def test_maintainer_header_never_reaches_the_model(registry):
    """The fragment's leading blockquote is maintainer documentation."""
    for phase in ("", *_ps.PHASE_NAMES):
        assert "rules fragment** consumed by" not in _build(registry, phase)


# Rules-fragment tag filtering
FRAGMENT = """\
> maintainer note, stripped

### Common block

always here

<!-- phase: EXPLORE, FRAMEWORK_AGENT -->
### Scoped block

only for explore

### Trailing common block

also always here
"""


def test_filter_keeps_untagged_blocks_in_every_phase():
    for phase in ("", "PRELUDE", "EXPLORE", "CLOSE"):
        out = _filter_rules_fragment(FRAGMENT, phase=phase)
        assert "### Common block" in out
        assert "### Trailing common block" in out


def test_filter_drops_tagged_block_outside_its_phases():
    out = _filter_rules_fragment(FRAGMENT, phase="CLOSE")
    assert "### Scoped block" not in out
    assert "only for explore" not in out


def test_filter_keeps_tagged_block_inside_its_phases():
    for phase in ("EXPLORE", "FRAMEWORK_AGENT", ""):
        out = _filter_rules_fragment(FRAGMENT, phase=phase)
        assert "### Scoped block" in out
        assert "only for explore" in out


def test_filter_strips_tag_comments_and_leading_blockquote():
    out = _filter_rules_fragment(FRAGMENT, phase="EXPLORE")
    assert "<!-- phase:" not in out
    assert "maintainer note" not in out


def test_phase_argument_is_case_insensitive(registry):
    assert _build(registry, "kernel_agent") == _build(registry, _ps.PHASE_KERNEL_AGENT)


# Coordinator re-scopes the override at the phase seam
def _machine_with_stub_coordinator(session_dir, *, user_supplied: bool = False):
    """Build a MachinePhase over a real Coordinator with a controlled rebuild function."""
    from hyperloom.orchestrator.collaborator import OrchestrationPrompt
    from hyperloom.orchestrator.phases.machine import MachinePhase

    from .conftest import make_coordinator

    rebuild_calls: list[dict] = []

    def _rebuild(**kwargs) -> str:
        rebuild_calls.append(kwargs)
        return f"PROMPT[phase={kwargs.get('phase')}]"

    coord = make_coordinator(session_dir)
    coord.shared_state.session_id = "t"
    coord.shared_state.macro_cycle = 3
    coord.shared_state.orchestration_memory = {"next_cycle_directive": "keep pushing MoE dispatch", "for_cycle": 2}
    coord.orch_prompt = OrchestrationPrompt(
        overrides={"orchestration": "ORIGINAL"},
        is_user_supplied=user_supplied,
        rebuild=_rebuild,
    )
    return MachinePhase(coord), coord, rebuild_calls


def test_phase_seam_rescopes_the_override_and_keeps_the_cycle_directive(tmp_path):
    handler, coord, calls = _machine_with_stub_coordinator(tmp_path)

    assert handler._reseed_orch_prompt_for_phase("kernel_agent") is True
    assert coord.orch_prompt.overrides["orchestration"] == "PROMPT[phase=KERNEL_AGENT]"
    assert len(calls) == 1
    assert calls[0]["macro_cycle"] == 3
    assert calls[0]["cycle_directive"] == "keep pushing MoE dispatch"
    assert calls[0]["phase"] == "KERNEL_AGENT"
    assert calls[0].get("cycle_strategy") is not None


def test_phase_seam_never_clobbers_a_user_supplied_prompt(tmp_path):
    handler, coord, calls = _machine_with_stub_coordinator(tmp_path, user_supplied=True)

    assert handler._reseed_orch_prompt_for_phase("EXPLORE") is False
    assert coord.orch_prompt.overrides["orchestration"] == "ORIGINAL"
    assert calls == []


def test_phase_seam_ignores_a_blank_phase(tmp_path):
    handler, coord, calls = _machine_with_stub_coordinator(tmp_path)

    assert handler._reseed_orch_prompt_for_phase("") is False
    assert coord.orch_prompt.overrides["orchestration"] == "ORIGINAL"
    assert calls == []


def test_phase_seam_snapshots_the_scope_it_installed(tmp_path):
    """Each scope the model actually ran under must leave its own artefact."""
    handler, _coord, _calls = _machine_with_stub_coordinator(tmp_path)

    assert handler._reseed_orch_prompt_for_phase("EXPLORE") is True

    snapshot = tmp_path / "agents" / "orchestration" / "system_prompt.c3.EXPLORE.snapshot.md"
    assert snapshot.read_text(encoding="utf-8") == "PROMPT[phase=EXPLORE]"


@pytest.mark.asyncio
async def test_the_reseeded_override_is_what_the_orchestration_turn_loads(tmp_path):
    handler, coord, _calls = _machine_with_stub_coordinator(tmp_path)

    assert handler._reseed_orch_prompt_for_phase("EXPLORE") is True

    assert await coord.conversation.load_system_prompt("orchestration") == "PROMPT[phase=EXPLORE]"


def test_cycle_strategy_rows_do_not_nest_the_prior_cycles(tmp_path):
    """The prior rows are rendered into the prompt, never stored inside the next row."""
    handler, coord, calls = _machine_with_stub_coordinator(tmp_path)
    state = coord.shared_state
    for cycle in range(1, 5):
        state.macro_cycle = cycle
        coord.phase_macro_cycle.record_cycle_strategy_for_current_cycle()
        handler._reseed_orch_prompt_for_phase("EXPLORE")

    assert [row["cycle"] for row in state.cycle_strategy_log] == [1, 2, 3, 4]
    assert all("prior_cycles" not in row for row in state.cycle_strategy_log)
    assert len(calls[-1]["cycle_strategy"]["prior_cycles"]) == 3


@pytest.mark.parametrize(
    ("macro_cycle", "target_reached_at"),
    [(0, ""), (_ps.DEFAULT_MAX_MACRO_CYCLES - 1, ""), (0, "2026-01-01T00:00:00+00:00"), (0, " ")],
    ids=["open", "last_cycle", "target_reached", "blank_target_reached"],
)
def test_reloop_line_and_transition_agree_at_the_cycle_limits(macro_cycle, target_reached_at):
    s = _render_state(_ps.PHASE_SWEEP)
    s.macro_cycle = macro_cycle
    s.target_reached_at = target_reached_at
    s.last_conc_sweep = {"status": "succeeded", "summary": {"successful_pairs": 1}}
    transition = _ps.compute_next_phase(s)
    loops_back = bool(transition) and transition[0] == _ps.PHASE_FRAMEWORK_AGENT

    reloop, _ = _ps.cycle_reloop_decision(s)

    assert reloop is loops_back
    assert f"cycle_reloop_feasible={'true' if reloop else 'false'}" in _ps.phase_status_summary(s)


def test_phase_seam_snapshot_never_overwrites_the_boot_file(tmp_path):
    """The unsuffixed file stays the boot scope so existing readers keep working."""
    boot = tmp_path / "agents" / "orchestration" / "system_prompt.c3.snapshot.md"
    boot.parent.mkdir(parents=True, exist_ok=True)
    boot.write_text("BOOT", encoding="utf-8")
    handler, _coord, _calls = _machine_with_stub_coordinator(tmp_path)

    handler._reseed_orch_prompt_for_phase("CLOSE")

    assert boot.read_text(encoding="utf-8") == "BOOT"


def test_phase_seam_survives_an_unwritable_session_dir(tmp_path):
    """A failed snapshot must not abort the phase transition."""
    blocker = tmp_path / "agents"
    blocker.write_text("not a directory", encoding="utf-8")
    handler, coord, _calls = _machine_with_stub_coordinator(tmp_path)

    assert handler._reseed_orch_prompt_for_phase("SWEEP") is True
    assert coord.orch_prompt.overrides["orchestration"] == "PROMPT[phase=SWEEP]"


# Snapshot paths: one artefact per scope the model ran under
def test_prompt_snapshot_path_is_phase_suffixed(tmp_path):
    from hyperloom.inference_optimizer.session.session_paths import agent_prompt_snapshot

    assert agent_prompt_snapshot(tmp_path, "orchestration", macro_cycle=0).name == "system_prompt.c0.snapshot.md"
    scoped = agent_prompt_snapshot(tmp_path, "orchestration", macro_cycle=0, phase="explore")
    assert scoped.name == "system_prompt.c0.EXPLORE.snapshot.md"
    # A blank phase must not produce a stray dot in the stem.
    assert (
        agent_prompt_snapshot(tmp_path, "orchestration", macro_cycle=0, phase="  ").name
        == "system_prompt.c0.snapshot.md"
    )


def test_boot_snapshot_records_the_phase_it_was_scoped_to(tmp_path):
    """The boot prompt is already phase-scoped, so it needs its own suffixed copy."""
    from hyperloom.inference_optimizer.cli.bootstrap import _snapshot_system_prompts

    _snapshot_system_prompts(
        tmp_path,
        prompts={"orchestration": "BOOT PROMPT", "critic": "CRITIC"},
        orchestration_phase="PRELUDE",
        macro_cycle=0,
    )

    agents = tmp_path / "agents"
    assert (agents / "orchestration" / "system_prompt.c0.snapshot.md").read_text(encoding="utf-8") == "BOOT PROMPT"
    assert (agents / "orchestration" / "system_prompt.c0.PRELUDE.snapshot.md").read_text(
        encoding="utf-8",
    ) == "BOOT PROMPT"
    # Critic is not phase-scoped at the system-prompt level.
    assert not (agents / "critic" / "system_prompt.c0.PRELUDE.snapshot.md").exists()


def test_boot_snapshot_without_a_phase_keeps_the_legacy_layout(tmp_path):
    from hyperloom.inference_optimizer.cli.bootstrap import _snapshot_system_prompts

    _snapshot_system_prompts(tmp_path, prompts={"orchestration": "BOOT"}, macro_cycle=0)

    orch = tmp_path / "agents" / "orchestration"
    assert (orch / "system_prompt.c0.snapshot.md").read_text(encoding="utf-8") == "BOOT"
    assert list(orch.glob("system_prompt.c0.*.snapshot.md")) == []


# Critic: phase is structurally deliverable and injected one phase at a time
def test_judge_bundle_to_dict_carries_phase():
    """The on-disk bundle records the phase, so audits are not misled."""
    from hyperloom.agents.critic.runtime.decision_reviewer import JudgeBundle

    bundle = JudgeBundle(kind="coordinator_inbox", session_id="s", decision_id=None, phase="EXPLORE")
    assert bundle.to_dict()["phase"] == "EXPLORE"


def test_inject_phase_constraints_delivers_only_the_active_phase():
    from hyperloom.orchestrator.roles.critic_agent import (
        _PHASE_ORIENTATION,
        _inject_phase_constraints,
    )

    bundle: dict = {"proposals": []}
    _inject_phase_constraints(bundle, "kernel_agent")

    assert bundle["phase"] == "KERNEL_AGENT"
    rc = bundle["review_constraints"]
    assert rc["phase"] == "KERNEL_AGENT"
    assert rc["phase_orientation"] == _PHASE_ORIENTATION["KERNEL_AGENT"]


def test_inject_phase_constraints_is_a_noop_without_a_phase():
    """Never assert a phase that was not delivered."""
    from hyperloom.orchestrator.roles.critic_agent import _inject_phase_constraints

    bundle: dict = {"proposals": []}
    _inject_phase_constraints(bundle, "")
    assert "phase" not in bundle
    assert "review_constraints" not in bundle


# Reloop feasibility must reach the phases where the defer/advance call is made.


def _render_state(phase: str, max_minutes: float = 120.0):
    """Build a minimal SharedState parked in ``phase`` with a live clock."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    s = SharedState()
    s.phase = phase
    s.max_minutes = max_minutes
    s.start_ts = datetime.now(timezone.utc).isoformat()
    return s


def _reloop_line(phase: str) -> str | None:
    out = _ps.phase_status_summary(_render_state(phase))
    return next((line for line in out.splitlines() if line.startswith("reloop")), None)


def test_reloop_line_reaches_every_mid_chain_phase():
    for phase in (_ps.PHASE_FRAMEWORK_AGENT, _ps.PHASE_FRAMEWORK_AGENT, _ps.PHASE_KERNEL_AGENT, _ps.PHASE_SWEEP):
        line = _reloop_line(phase)
        assert line is not None, f"reloop line missing for {phase}"
        # The field name is the prompt/doc contract; models grep for it verbatim.
        assert "cycle_reloop_feasible=" in line
        assert "threshold_sec=" in line
        assert "session_remaining_sec=" in line


def test_reloop_line_absent_in_wind_down_phases():
    for phase in (_ps.PHASE_PRELUDE, _ps.PHASE_CLOSE):
        assert _reloop_line(phase) is None, f"reloop line leaked into {phase}"


def test_reloop_is_a_projection_before_sweep():
    assert "(projected)" in (_reloop_line(_ps.PHASE_FRAMEWORK_AGENT) or "")
    assert "(projected)" not in (_reloop_line(_ps.PHASE_SWEEP) or "")


def test_reloop_feasibility_matches_the_transition_decision():
    s = _render_state(_ps.PHASE_SWEEP)
    reloop, _ = _ps._reloop_decision(_ps.workflow_predicate_inputs(s))
    expected = "true" if reloop else "false"
    assert f"cycle_reloop_feasible={expected}" in (_reloop_line(_ps.PHASE_SWEEP) or "")


def test_reloop_infeasible_when_the_target_phase_is_disabled():
    s = _render_state(_ps.PHASE_SWEEP)
    s.framework_agent_phase_enabled = False
    line = next(line for line in _ps.phase_status_summary(s).splitlines() if line.startswith("reloop"))
    assert "cycle_reloop_feasible=false" in line


def test_the_payload_contract_lists_only_required_keys():
    """Required keys and value constraints are different claims."""
    from hyperloom.inference_optimizer.protocol.intent import IntentType
    from hyperloom.orchestrator.roles.mcp_emit_intent import (
        _PAYLOAD_REQUIRED,
        payload_contract,
    )

    rendered = payload_contract(IntentType)

    assert "alert:{severity,summary}" in rendered
    assert "severity ∈" not in rendered
    assert "scope" not in rendered
    for intent_type, required in _PAYLOAD_REQUIRED.items():
        assert f"{intent_type.value}:{{{','.join(required)}}}" in rendered


def test_the_payload_constraints_are_rendered_separately():
    """State the closed value sets and the optional dials as what they are."""
    from hyperloom.inference_optimizer.protocol.intent import IntentType
    from hyperloom.orchestrator.roles.mcp_emit_intent import payload_constraints

    rendered = payload_constraints(IntentType)

    assert "alert.severity ∈ low|medium|high" in rendered
    assert "prune_branch.scope ∈ family|queued" in rendered
    assert "extend_lease.reason is optional" in rendered


def test_both_provider_descriptions_carry_the_constraints():
    """Claude's tool description and Codex's output block say the same thing."""
    from hyperloom.inference_optimizer.protocol.intent import IntentType
    from hyperloom.orchestrator.roles.codex import build_output_instructions
    from hyperloom.orchestrator.roles.mcp_emit_intent import (
        EMIT_INTENT_TOOL_INPUT_SCHEMA,
        payload_constraints,
    )

    constraints = payload_constraints(IntentType)
    claude_description = EMIT_INTENT_TOOL_INPUT_SCHEMA["properties"]["payload"]["description"]
    codex_block = build_output_instructions(frozenset(IntentType))

    assert constraints in claude_description
    assert constraints in codex_block


def test_an_unknown_transport_is_refused_not_rendered_empty(registry):
    """A transport nobody declares must not quietly delete the output protocol."""
    from hyperloom.orchestrator.prompts.transport import TRANSPORTS

    with pytest.raises(ValueError, match="carrier-pigeon"):
        build_orchestration_prompt(
            action_registry=registry,
            enabled_actions=default_enabled_actions(no_kernel=False),
            framework="sglang",
            kernel_enabled=True,
            framework_agent_phase_enabled=True,
            objective_kind="gain_pct",
            objective_value=15.0,
            max_minutes=480,
            transport="carrier-pigeon",
            rules_fragment_path=asset_system_prompts_dir() / "orchestration.md",
        )

    assert "carrier-pigeon" not in TRANSPORTS


@pytest.mark.parametrize("transport", ["", "tools", "structured_output"])
def test_every_declared_transport_still_renders(registry, transport):
    """The declared transports, and the unscoped default, keep working."""
    prompt = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        framework="sglang",
        kernel_enabled=True,
        framework_agent_phase_enabled=True,
        objective_kind="gain_pct",
        objective_value=15.0,
        max_minutes=480,
        transport=transport,
        rules_fragment_path=asset_system_prompts_dir() / "orchestration.md",
    )

    assert "Output protocol" in prompt


def test_a_role_with_no_constraints_gets_no_constraints_line():
    """Say nothing rather than an empty clause."""
    from hyperloom.inference_optimizer.protocol.intent import IntentType
    from hyperloom.orchestrator.roles.codex import build_output_instructions
    from hyperloom.orchestrator.roles.mcp_emit_intent import build_intent_envelope_schema

    only_send = frozenset({IntentType.SEND_MESSAGE})

    block = build_output_instructions(only_send)
    schema = build_intent_envelope_schema(only_send)
    payload_description = schema["properties"]["intents"]["items"]["properties"]["payload"]["description"]

    assert "Constraints" not in block
    assert "Constraints" not in payload_description
    assert "send_message:{topic}" in block
