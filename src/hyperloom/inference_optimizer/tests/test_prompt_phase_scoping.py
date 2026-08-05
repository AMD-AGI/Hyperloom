# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase scoping of the orchestration system prompt and the Critic judge bundle.

Locks both halves of the contract: a phase receives only the modules whose
behaviour it can reach, and the cross-phase planning facts a ``skip_to_*``
decision needs survive every phase.
"""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir
from hyperloom.orchestrator.actions.registry import ActionRegistry
from hyperloom.orchestrator.phases import machine_state as _ps
from hyperloom.orchestrator.prompts.prompt_builder import (
    _filter_rules_fragment,
    build_orchestration_prompt,
    default_enabled_actions,
)


# Marker text identifying each phase-scoped prompt module.
KERNEL_REQUEST_REF = "## 6. KERNEL-OPT REQUEST REFERENCE"
IDEA_GENERATION = "### IDEA GENERATION"
BASELINE_FINGERPRINT = "eight params fields"
SPECIALIST_DIALS = "### One specialist, four dials"
SPECIALIST_WATCH = "### Watching a running specialist"
SPECIALIST_DOMAIN = "### Choosing specialist domain"
WEB_SEARCH = "### Web search"

# Only EXPLORE lets the LLM emit `delegate{specialist}`; FRAMEWORK_AGENT
# specialists come from the Coordinator's authoring pump but stay steerable.
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
    "### Pulling context on a delta turn",
    "### SESSION_DIR contract",
    "### Output protocol",
    "RULE F3",
    "RULE F4",
)


@pytest.fixture(scope="module")
def registry() -> ActionRegistry:
    return ActionRegistry().load()


def _build(registry: ActionRegistry, phase: str) -> str:
    return build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        framework="sglang",
        kernel_enabled=True,
        explore_enabled=True,
        framework_agent_phase_enabled=True,
        objective_kind="gain_pct",
        objective_value=15.0,
        max_minutes=480,
        phase=phase,
        rules_fragment_path=asset_system_prompts_dir() / "orchestration.md",
    )


# ---------------------------------------------------------------------------
# Phase-scoped modules render only where the behaviour exists
# ---------------------------------------------------------------------------
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
        if phase == _ps.PHASE_EXPLORE:
            assert IDEA_GENERATION in text
        else:
            assert IDEA_GENERATION not in text, f"idea generation leaked into {phase}"


def test_baseline_recovery_detail_only_in_prelude(registry):
    """Only PRELUDE can re-propose baseline, so only it needs the fingerprint."""
    for phase in _ps.PHASE_NAMES:
        text = _build(registry, phase)
        if phase == _ps.PHASE_PRELUDE:
            assert BASELINE_FINGERPRINT in text
            assert "RULE F1" in text
            assert "RULE F2" in text
        else:
            assert BASELINE_FINGERPRINT not in text, f"baseline fingerprint leaked into {phase}"
            assert "RULE F1" not in text
            assert "RULE F2" not in text


def test_specialist_dispatch_prose_only_in_explore(registry):
    """How to shape a dispatch matters only where the LLM can emit one."""
    for phase in _ps.PHASE_NAMES:
        text = _build(registry, phase)
        for marker in SPECIALIST_DISPATCH_OPS:
            if phase == _ps.PHASE_EXPLORE:
                assert marker in text, f"{marker} missing from {phase}"
            else:
                assert marker not in text, f"{marker} leaked into {phase}"


def test_specialist_watching_prose_spans_both_dispatching_phases(registry):
    """A live specialist can exist in EXPLORE and FRAMEWORK_AGENT; the LLM steers both."""
    with_specialists = {_ps.PHASE_EXPLORE, _ps.PHASE_FRAMEWORK_AGENT}
    for phase in _ps.PHASE_NAMES:
        text = _build(registry, phase)
        if phase in with_specialists:
            assert SPECIALIST_WATCH in text, f"{SPECIALIST_WATCH} missing from {phase}"
        else:
            assert SPECIALIST_WATCH not in text, f"{SPECIALIST_WATCH} leaked into {phase}"


def test_payload_contracts_are_scoped_to_the_proposable_set(registry):
    """A phase gets payload templates only for the actions it can propose."""
    for phase in _ps.PHASE_NAMES:
        text = _build(registry, phase)
        proposable = set(_ps.llm_proposable_actions_for(phase))
        if "explore" in proposable:
            assert "GRID INPUT (REQUIRED)" in text
        else:
            assert "GRID INPUT (REQUIRED)" not in text, f"explore grid schema leaked into {phase}"
        if "specialist" in proposable:
            assert "EMIT: delegate{action_name='specialist'" in text
        else:
            assert "EMIT: delegate{action_name='specialist'" not in text, f"specialist payload leaked into {phase}"
        # Descriptions survive so a skip_to_* decision can still compare phases.
        assert "- **explore** —" in text
        assert "- **specialist** —" in text
        assert "- **sweep** —" in text


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


# ---------------------------------------------------------------------------
# Back-compat: an unscoped build is a superset
# ---------------------------------------------------------------------------
def test_unscoped_build_renders_every_module(registry):
    """A caller that does not track phases keeps the pre-scoping prompt."""
    text = _build(registry, "")
    for marker in (
        KERNEL_REQUEST_REF,
        IDEA_GENERATION,
        BASELINE_FINGERPRINT,
        "GRID INPUT (REQUIRED)",
        *ALL_SPECIALIST_OPS,
        *ALWAYS_ON,
    ):
        assert marker in text, f"{marker} missing from the unscoped build"


def test_scoped_builds_are_strictly_smaller(registry):
    """Every phase pays less than the unscoped superset."""
    unscoped = len(_build(registry, "").splitlines())
    for phase in _ps.PHASE_NAMES:
        assert len(_build(registry, phase).splitlines()) < unscoped, f"{phase} did not shrink"


def test_maintainer_header_never_reaches_the_model(registry):
    """The fragment's leading blockquote is maintainer documentation."""
    for phase in ("", *_ps.PHASE_NAMES):
        assert "rules fragment** consumed by" not in _build(registry, phase)


# ---------------------------------------------------------------------------
# Rules-fragment tag filtering
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Coordinator re-scopes the override at the phase seam
# ---------------------------------------------------------------------------
def _machine_with_stub_coordinator(*, user_supplied: bool = False):
    """Build a MachinePhase over a minimal coordinator stub.

    Returns ``(phase_handler, coord, rebuild_calls)`` where ``rebuild_calls``
    records the kwargs handed to the stubbed prompt rebuilder.
    """
    from types import SimpleNamespace

    from hyperloom.orchestrator.phases.machine import MachinePhase
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(session_id="t", macro_cycle=3)
    state.orchestration_memory = {"next_cycle_directive": "keep pushing MoE dispatch"}
    rebuild_calls: list[dict] = []

    def _rebuild(**kwargs) -> str:
        rebuild_calls.append(kwargs)
        return f"PROMPT[phase={kwargs.get('phase')}]"

    coord = SimpleNamespace(
        shared_state=state,
        system_prompt_overrides={"orchestration": "ORIGINAL"},
        _rebuild_orch_prompt=_rebuild,
        _orch_prompt_is_user_supplied=user_supplied,
    )
    return MachinePhase(coord), coord, rebuild_calls


def test_phase_seam_rescopes_the_override_and_keeps_the_cycle_directive():
    handler, coord, calls = _machine_with_stub_coordinator()

    assert handler._reseed_orch_prompt_for_phase("kernel_agent") is True
    assert coord.system_prompt_overrides["orchestration"] == "PROMPT[phase=KERNEL_AGENT]"
    assert calls == [
        {
            "macro_cycle": 3,
            "cycle_directive": "keep pushing MoE dispatch",
            "phase": "KERNEL_AGENT",
        }
    ]


def test_phase_seam_never_clobbers_a_user_supplied_prompt():
    handler, coord, calls = _machine_with_stub_coordinator(user_supplied=True)

    assert handler._reseed_orch_prompt_for_phase("EXPLORE") is False
    assert coord.system_prompt_overrides["orchestration"] == "ORIGINAL"
    assert calls == []


def test_phase_seam_ignores_a_blank_phase():
    handler, coord, calls = _machine_with_stub_coordinator()

    assert handler._reseed_orch_prompt_for_phase("") is False
    assert coord.system_prompt_overrides["orchestration"] == "ORIGINAL"
    assert calls == []


def test_reseed_for_phase_is_reachable_through_the_coordinator_delegation_map():
    """The collaborator method must be routed, or the seam hook is a no-op."""
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    assert Coordinator._DELEGATED.get("_reseed_orch_prompt_for_phase") == "phase_machine"
    assert "phase_machine" in Coordinator._COLLAB_MODULES


# ---------------------------------------------------------------------------
# Critic: phase is structurally deliverable and injected one phase at a time
# ---------------------------------------------------------------------------
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
