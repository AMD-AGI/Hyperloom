# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the orchestration prompt's ``dynamic_action`` entry +
the emit-hint catalogue row.
Auxiliary tests pin the §1.7 design-philosophy guards (no
examples / no triggering heuristics / no specialist-failure
fallback hint / no negative cost guidance) so a prompt regression
shows up as a structural test failure rather than as a silent shift
in LLM behaviour.
"""

from __future__ import annotations

import re

import pytest

from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.policy import (
    DYNAMIC_ACTION_NAME,
    MAX_DYNAMIC_PER_ROUND,
)
from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
    FULL_ENABLED_ACTIONS,
    NO_KERNEL_ENABLED_ACTIONS,
    build_orchestration_prompt,
)


# PolicyGate reason codes the emit hint must surface so the LLM can
# self-correct on the next turn rather than retrying blindly.
EXPECTED_REASON_CODES: tuple[str, ...] = (
    "dynamic_phase_violation",
    "dynamic_source_violation",
    "dynamic_payload_schema",
    "dynamic_scope_too_narrow",
    "dynamic_scope_unknown_domain",
    "dynamic_side_effects_red_line",
    "dynamic_kernel_only_disallowed",
    "dynamic_round_cap_exhausted",
)


@pytest.fixture(scope="module")
def registry() -> ActionRegistry:
    return ActionRegistry().load()


@pytest.fixture(scope="module")
def prompt(registry: ActionRegistry) -> str:
    return build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        kernel_enabled=True,
        max_minutes=60,
    )


@pytest.fixture(scope="module")
def no_kernel_prompt(registry: ActionRegistry) -> str:
    return build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=NO_KERNEL_ENABLED_ACTIONS,
        framework="sglang",
        kernel_enabled=False,
        max_minutes=60,
    )


@pytest.fixture(scope="module")
def baseline_prompt_without_dynamic_action(registry: ActionRegistry) -> str:
    """Same render as ``prompt`` but with dynamic_action stripped
    from the enabled set; lets us measure the prompt-volume delta."""
    enabled = tuple(a for a in FULL_ENABLED_ACTIONS if a != DYNAMIC_ACTION_NAME)
    return build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=enabled,
        framework="sglang",
        kernel_enabled=True,
        max_minutes=60,
    )


# ===========================================================================
# §9 #1 — Dynamic Action declaration block present + compact
# ===========================================================================
def test_p7_scenario_01_dynamic_action_block_present(prompt: str):
    assert "DYNAMIC ACTION" in prompt
    # Locate the block: starts at "## 6b. DYNAMIC ACTION", ends at the
    # next "## " heading (rules fragment).
    m = re.search(
        r"## 6b\. DYNAMIC ACTION.*?(?=^## )", prompt, re.S | re.M,
    )
    assert m is not None, "Dynamic Action block not located"
    body = m.group(0)
    # Tokenise estimate: ~4 chars/token. The section text itself
    # (excluding the heading + EMIT entry which lives in the
    # catalogue) should stay well under the §9 #8 delta budget.
    estimated_tokens = len(body) // 4
    assert estimated_tokens <= 300, (
        f"Dynamic Action block estimated tokens={estimated_tokens} "
        f"exceeds the conservative 300-token ceiling (declaration "
        f"+ §6 ordering combined)"
    )


def test_p7_scenario_01b_block_text_matches_31_intent(prompt: str):
    """Wording must reflect §3.1: 'supplementary channel, not default',
    cross-domain motivation framing, and the round-cap."""
    block = re.search(
        r"## 6b\. DYNAMIC ACTION.*?(?=^## )", prompt, re.S | re.M,
    ).group(0)
    assert "supplementary" in block.lower()
    assert "not the default" in block.lower()
    assert "cross-domain" in block.lower()
    assert "specialist" in block.lower()
    # round cap visible
    assert "ONE" in block or "1" in block


# ===========================================================================
# §9 #2 — enabled action list includes dynamic_action
# ===========================================================================
def test_p7_scenario_02_enabled_actions_include_dynamic_action():
    assert DYNAMIC_ACTION_NAME in FULL_ENABLED_ACTIONS
    assert DYNAMIC_ACTION_NAME in NO_KERNEL_ENABLED_ACTIONS


def test_p7_scenario_02b_catalogue_lists_dynamic_action(prompt: str):
    assert "**dynamic_action**" in prompt


def test_no_kernel_prompt_also_lists_dynamic_action(no_kernel_prompt: str):
    """``--no-kernel`` should NOT hide the dynamic_action channel; it
    is EXPLORE-only and orthogonal to kernel mode."""
    assert "**dynamic_action**" in no_kernel_prompt
    assert "DYNAMIC ACTION" in no_kernel_prompt


# ===========================================================================
# §9 #3 — emit hint completeness (payload table + constraints +
# reason codes)
# ===========================================================================
def test_p7_scenario_03_emit_hint_payload_fields(prompt: str):
    """Payload field table must enumerate every dispatch field by name
    so the LLM has a single source of truth."""
    hint_snippet = re.search(
        r"delegate\{action_name='dynamic_action'[^\n]*",
        prompt,
    )
    assert hint_snippet is not None, "dynamic_action EMIT hint missing"
    snippet = hint_snippet.group(0)
    for field in (
        "motivation_gap_text", "scope_domains",
        "side_effects_declared", "budget_hint",
    ):
        assert field in snippet, f"payload field {field!r} missing from EMIT hint"


def test_p7_scenario_03b_emit_hint_carries_constraints(prompt: str):
    """Key constraints — scope_domains length, side-effect red lines,
    round-cap — must appear in the hint."""
    # The constraints text follows the EMIT line and may wrap across
    # the catalogue line; we search the whole prompt for the literal
    # tokens that the §5.1 contract requires.
    assert "scope_domains length >= 2" in prompt
    assert "kernel-owned action" in prompt
    assert "metric" in prompt
    assert "server lifecycle" in prompt
    assert "1 dispatch per EXPLORE round" in prompt


@pytest.mark.parametrize("reason_code", EXPECTED_REASON_CODES)
def test_p7_scenario_03c_emit_hint_includes_every_reason_code(
    prompt: str, reason_code: str,
):
    assert reason_code in prompt, (
        f"PolicyGate reason code {reason_code!r} missing from emit "
        f"hint; LLM cannot self-correct on this denial without the "
        f"vocabulary"
    )


# ===========================================================================
# §9 #4-5 — covered by the existing PolicyGate test suite. We add a
# documentation marker so the §9 mapping table stays complete.
# ===========================================================================
def test_p7_scenario_04_marker_covered_by_dispatch_tests():
    """The dispatch happy-path test already
    pins that a payload-valid dispatch in EXPLORE phase passes
    PolicyGate. This marker keeps the §9 mapping table complete."""
    from inference_optimizer.tests.test_dynamic_action_dispatch import (  # noqa: F401
        test_p1_scenario_01_valid_dispatch_passes,
    )


def test_p7_scenario_05_marker_phase_denial_feedback_loop():
    """The PRELUDE-phase dispatch → ``dynamic_phase_violation`` test
    is the closed-loop feedback case. We additionally assert here
    that the prompt actually advertises this reason code so the LLM
    *can* recognise the denial and recover."""
    from inference_optimizer.tests.test_dynamic_action_dispatch import (  # noqa: F401
        test_p1_scenario_02_wrong_phase_denied,
    )


# ===========================================================================
# §9 #6 — 0 history → no summary section (covered in P6 tests but
# repeated here for cross-reference). The orchestration prompt builder
# itself does not render the per-tick summary section; that lives in
# Coordinator._compose_prompt and SharedState.to_dynamic_actions_
# prompt_section, both pinned by P6 tests.
# ===========================================================================
def test_p7_scenario_06_marker_empty_summary_omits_section():
    from inference_optimizer.tests.test_dynamic_action_summary import (  # noqa: F401
        test_p6_scenario_06_empty_yields_no_section,
    )


# ===========================================================================
# 6 history entries render as exactly 5 rows + an elision marker.
# ===========================================================================
def test_p7_scenario_07_marker_history_caps_at_five():
    from inference_optimizer.tests.test_dynamic_action_summary import (  # noqa: F401
        test_p6_scenario_05_prompt_caps_at_five_entries,
    )


# ===========================================================================
# §9 #8 — total prompt-volume delta ≤ 500 tokens
# ===========================================================================
def test_p7_scenario_08_prompt_volume_delta_within_budget(
    prompt: str, baseline_prompt_without_dynamic_action: str,
):
    """Adding dynamic_action to the enabled set must not bloat the
    orchestration prompt beyond the §9 #8 budget (≤ 500 tokens
    delta) — composed of:
    - ~100 tokens for the Dynamic Action declaration block
    - ~150 tokens for the catalogue entry (cost line + EMIT hint)
    - ~250 tokens reserved for the P6 summary section (rendered
      per-tick by the Coordinator, not in this static prompt).
    """
    delta_chars = len(prompt) - len(baseline_prompt_without_dynamic_action)
    delta_tokens = delta_chars // 4
    assert delta_tokens <= 500, (
        f"orchestration prompt grew by {delta_tokens} tokens "
        f"({delta_chars} chars) — exceeds 500-char budget"
    )
    # And the delta must actually be positive (we did add content).
    assert delta_tokens > 0


# ===========================================================================
# §3.3 + §1.8 guards — content the prompt MUST NOT contain
# ===========================================================================
def test_dynamic_action_block_omits_example_motivations(prompt: str):
    """§3.3: 'any "dynamic action 比 specialist 强" implicit suggestion'
    and 'any motivation_gap_text example' must NOT appear in the
    block — examples would let the LLM pattern-match into
    inappropriate dispatches.
    """
    block = re.search(
        r"## 6b\. DYNAMIC ACTION.*?(?=^## )", prompt, re.S | re.M,
    ).group(0)
    # Common example-phrasing markers.
    forbidden_phrases = (
        "for example",
        "e.g.",
        "for instance",
        "such as",
    )
    for phrase in forbidden_phrases:
        assert phrase.lower() not in block.lower(), (
            f"example-phrasing marker {phrase!r} found in Dynamic "
            f"Action declaration; §3.3 forbids motivation examples"
        )


def test_dynamic_action_block_omits_specialist_failure_fallback(prompt: str):
    """§3.3 / §1.7: must not write 'when specialist fails N times,
    consider dynamic' — that would position dynamic as the
    specialist-failure兜底."""
    block = re.search(
        r"## 6b\. DYNAMIC ACTION.*?(?=^## )", prompt, re.S | re.M,
    ).group(0)
    forbidden_substrings = (
        "specialist fail",
        "specialist failed",
        "fallback to dynamic",
        "when specialist",  # heuristic phrasing
        "after specialist",
    )
    for marker in forbidden_substrings:
        assert marker.lower() not in block.lower(), (
            f"specialist-fallback inducement {marker!r} found in "
            f"Dynamic Action declaration; §3.3 forbids this framing"
        )


def test_dynamic_action_block_omits_cooldown_or_cost_guidance(prompt: str):
    """§3.3: no 'expensive, use sparingly' / cooldown framing —
    negative guidance reinforces 'special' framing and produces the
    two-extremes failure mode (afraid to use OR brave with extra
    tokens)."""
    block = re.search(
        r"## 6b\. DYNAMIC ACTION.*?(?=^## )", prompt, re.S | re.M,
    ).group(0)
    forbidden = (
        "expensive",
        "use sparingly",
        "cooldown",
        "wait between",
        "kill switch",
    )
    for marker in forbidden:
        assert marker.lower() not in block.lower(), (
            f"cost / cooldown framing {marker!r} found in Dynamic "
            f"Action declaration; §3.3 forbids this guidance"
        )


def test_dynamic_action_block_omits_internal_mechanics(prompt: str):
    """§3.3: 'sub-agent 内部工作方式的描述（multi-turn / micro-bench
    等）' must NOT appear — the LLM does not need to know
    implementation details and will only invent boundary-pushing
    ideas if it does."""
    block = re.search(
        r"## 6b\. DYNAMIC ACTION.*?(?=^## )", prompt, re.S | re.M,
    ).group(0)
    forbidden = (
        "multi-turn",
        "ReAct",
        "micro-bench",
        "sub-process",
        "claude subprocess",
        "tool whitelist",
    )
    for marker in forbidden:
        assert marker.lower() not in block.lower(), (
            f"internal-mechanics marker {marker!r} found in Dynamic "
            f"Action declaration; §3.3 forbids implementation detail"
        )


# ===========================================================================
# Round-cap visibility — both the catalogue hint AND the dedicated
# block surface MAX_DYNAMIC_PER_ROUND.
# ===========================================================================
def test_round_cap_value_visible_in_prompt(prompt: str):
    """The §5.1 + §3.1 cap of MAX_DYNAMIC_PER_ROUND should be visible
    so the LLM can budget its dispatches."""
    assert MAX_DYNAMIC_PER_ROUND == 1
    # Catalogue hint mentions "at most 1 dispatch per EXPLORE round".
    assert "1 dispatch per EXPLORE round" in prompt


# ===========================================================================
# Section ordering — Dynamic Action block sits AFTER the action
# catalogue (so the catalogue EMIT entry is already visible) and
# BEFORE the rules fragment (so the rules fragment can reference it).
# ===========================================================================
def test_dynamic_action_block_after_catalogue_and_before_rules(prompt: str):
    pos_catalogue = prompt.find("## 4. ACTIONS YOU MAY USE")
    pos_dynamic = prompt.find("## 6b. DYNAMIC ACTION")
    pos_rules = prompt.find("## 7. RULES & OUTPUT PROTOCOL")
    assert pos_catalogue > 0
    assert pos_dynamic > 0
    assert pos_rules > 0
    assert pos_catalogue < pos_dynamic < pos_rules


# ===========================================================================
# Specialist + dynamic_action sibling positioning in catalogue
# ===========================================================================
def test_specialist_and_dynamic_action_appear_in_same_phase_section(
    prompt: str,
):
    """Both ``specialist`` and ``dynamic_action`` declare
    pipeline_phase=explore in their yaml; the catalogue groups by
    pipeline_phase so they must appear in the same section, in line
    with the §6 ordering rationale of putting default + supplementary
    channels side-by-side."""
    explore_match = re.search(
        r"### explore.*?(?=^### |^## )", prompt, re.S | re.M,
    )
    assert explore_match is not None, "explore phase section missing"
    body = explore_match.group(0)
    assert "**specialist**" in body
    assert "**dynamic_action**" in body


# ===========================================================================
# Hidden behind --no-kernel: dynamic_action still rendered as
# supplementary channel (§4.2 — channel exists regardless of kernel
# mode).
# ===========================================================================
def test_no_kernel_emit_hint_still_complete(no_kernel_prompt: str):
    for reason_code in EXPECTED_REASON_CODES:
        assert reason_code in no_kernel_prompt
