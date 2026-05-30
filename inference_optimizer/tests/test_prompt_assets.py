"""Verify the system_prompts/*.md assets and the cli builder helpers.

Phase 1 of the prompt-builder refactor collapsed the previous
``orchestration.md`` / ``orchestration.no_kernel.md`` pair into:

* ``orchestration.md`` — small "rules + output protocol" fragment that
  the prompt builder consumes as section 7.
* :func:`inference_optimizer.cli._build_orchestration_prompt` — composes
  the full system prompt from typed inputs (action registry +
  enabled_actions + objective + framework + max_minutes), and
  ``orchestration.no_kernel.md`` was removed (the split is now a
  parameter, not a file).

This test pins the new contract so accidental regressions surface here
instead of silently degrading the orchestration loop.
"""

from __future__ import annotations

import pytest

from inference_optimizer.cli import _build_orchestration_prompt
from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.objective import (
    TargetGainObjective,
    TimeOnlyObjective,
)
from inference_optimizer.paths import asset_system_prompts_dir


# ---------------------------------------------------------------------------
# .md asset files
# ---------------------------------------------------------------------------
def test_orchestration_md_is_rules_fragment():
    """``orchestration.md`` is now a *fragment*, not a full system prompt.

    We expect it to contain the SESSION_DIR contract and Output protocol
    blocks, but NOT the kernel-opt pipeline (that's builder-injected).
    """
    p = asset_system_prompts_dir() / "orchestration.md"
    assert p.is_file(), f"missing orchestration prompt fragment: {p}"
    text = p.read_text(encoding="utf-8")
    assert "SESSION_DIR" in text
    assert "Output protocol" in text
    # Builder injects these as section headers / payload templates — they
    # MUST NOT appear in the rules fragment.
    assert "## DECISION FRAMEWORK" not in text
    assert "## 6. KERNEL-OPT REQUEST REFERENCE" not in text
    assert "payload templates" not in text


def test_orchestration_no_kernel_md_was_removed():
    """The legacy ``orchestration.no_kernel.md`` file must be gone."""
    p = asset_system_prompts_dir() / "orchestration.no_kernel.md"
    assert not p.exists(), (
        f"orchestration.no_kernel.md should have been removed by "
        f"prompt_builder Phase 1 — found at {p}"
    )


def test_critic_md_is_rules_fragment():
    """``critic.md`` is a concise rules fragment, not a full system prompt.

    v0.8 §3.3 added a per-phase review-contract section; issue #170 added
    the optional Web verification block. The line cap tracks both bumps.
    """
    p = asset_system_prompts_dir() / "critic.md"
    assert p.is_file(), f"missing critic rules fragment: {p}"
    text = p.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # Cap bumped 30 → 45 by N35 (structured carve-out bullets), then
    # 45 → 65 by N38 (the "primary per-proposal rule" section that
    # introduces the action_verdict_policy lookup as the canonical
    # source of truth), then 65 → 75 to absorb F-phase v0.8 carve-out
    # bullets that overlap with N38's text, then 75 → 95 by issue #170
    # (Web verification — when/why/how to use web_search / web_fetch and
    # the mandatory source-citation rule). Future bumps require equivalent
    # justification (new mechanism, not just adding more action names).
    assert len(lines) <= 95, (
        f"critic.md should be a concise fragment, got {len(lines)} non-empty lines"
    )
    assert "judge_bundle" in text
    assert "Hard rules" in text
    assert "target_proposal_msg_id" not in text
    assert "baseline / profile / target_analysis" not in text


# ---------------------------------------------------------------------------
# CLI builder helper
# ---------------------------------------------------------------------------
@pytest.fixture
def registry() -> ActionRegistry:
    return ActionRegistry().load()


def test_build_full_prompt_contains_kernel_opt_pipeline(registry):
    text = _build_orchestration_prompt(
        no_kernel=False,
        framework="sglang",
        objective=TargetGainObjective(target_gain_pct=10.0),
        max_minutes=120,
        action_registry=registry,
    )
    # Builder-generated section headers
    assert "## 1. MISSION" in text
    assert "## 2. SESSION CONTEXT" in text
    assert "## 3. PIPELINE & TIME BUDGET" in text
    assert "## 4. ACTIONS YOU MAY USE" in text
    assert "## 5. DECISION FRAMEWORK" in text
    assert "## 6. KERNEL-OPT REQUEST REFERENCE" in text
    assert "## 7. RULES & OUTPUT PROTOCOL" in text
    # Section 6 payload-template markers (one per kernel-owned action).
    # These are the unique request shapes the builder emits in section 6;
    # the rules fragment uses a different surface form ("kind MUST be
    # EXACTLY one of trace_analyze / ...") so it won't false-positive.
    assert "kind: 'trace_analyze'" in text
    assert "kind: 'run_optimization'" in text
    assert "kind: 'integrate'" in text
    # Action catalogue includes the kernel-owned actions
    assert "kernel_opt" in text
    assert "integrate" in text
    # Mission stresses cumulative_gain
    assert "cumulative_gain" in text
    # Session context shows the objective + budget
    assert "gain_pct=10" in text
    assert "max_minutes      : 120" in text
    assert "framework        : sglang" in text


def test_build_no_kernel_prompt_drops_kernel_pipeline_and_actions(registry):
    text = _build_orchestration_prompt(
        no_kernel=True,
        framework="vllm",
        objective=TimeOnlyObjective(),
        max_minutes=60,
        action_registry=registry,
    )
    # Builder must still emit the spine
    assert "## 1. MISSION" in text
    assert "## 4. ACTIONS YOU MAY USE" in text
    assert "## 5. DECISION FRAMEWORK" in text
    assert "## 7. RULES & OUTPUT PROTOCOL" in text
    # Kernel-opt request-reference block must be absent (builder skipped
    # section 6 because no kernel-owned actions are enabled).
    assert "## 6. KERNEL-OPT REQUEST REFERENCE" not in text
    assert "kind: 'trace_analyze'" not in text
    assert "kind: 'run_optimization'" not in text
    # Kernel-owned action names must NOT appear as catalogue bullets
    # (the bare word may still appear inside the rules fragment, e.g.
    # in a hard rule that mentions kernel_opt by name — that's fine).
    for forbidden in (
        "kernel_opt", "integrate", "deep_kernel_analysis",
        "operator_tuning", "vendor_kernel_config",
    ):
        assert f"**{forbidden}**" not in text, (
            f"no-kernel prompt must not advertise {forbidden!r} as a catalogue entry"
        )
    # `profile` only feeds kernel-opt, so its catalogue entry is also gone.
    assert "**profile**" not in text
    # Session context reflects the run flags
    assert "kernel_enabled   : false" in text
    assert "framework        : vllm" in text
    assert "objective        : time_only" in text


def test_build_two_modes_differ_full_is_longer(registry):
    full = _build_orchestration_prompt(
        no_kernel=False,
        framework="sglang",
        objective=TimeOnlyObjective(),
        max_minutes=120,
        action_registry=registry,
    )
    bare = _build_orchestration_prompt(
        no_kernel=True,
        framework="sglang",
        objective=TimeOnlyObjective(),
        max_minutes=120,
        action_registry=registry,
    )
    assert full != bare
    assert len(full) > len(bare)


def test_build_is_deterministic(registry):
    """Two builds with identical inputs must be byte-identical."""
    args = dict(
        no_kernel=False,
        framework="sglang",
        objective=TargetGainObjective(target_gain_pct=10.0),
        max_minutes=120,
        action_registry=registry,
    )
    a = _build_orchestration_prompt(**args)
    b = _build_orchestration_prompt(**args)
    assert a == b


def test_build_includes_explore_action_in_both_modes(registry):
    """v0.8 M3 + KB_gaps/Gap-10: ``explore`` is the canonical
    grid-runner advertised in both kernel-on and no-kernel runs.
    The v0.6 standalone ``validate_stack`` action / phase header
    has been retired (the rebench is inlined into ``explore``)."""
    for no_kernel in (False, True):
        text = _build_orchestration_prompt(
            no_kernel=no_kernel,
            framework="sglang",
            objective=TimeOnlyObjective(),
            max_minutes=60,
            action_registry=registry,
        )
        assert "explore" in text, (
            f"explore missing in prompt (no_kernel={no_kernel})"
        )
        # The deprecated ``validate_stack`` action must not appear as
        # an enabled catalogue entry — only as historical / DEPRECATED
        # tag references inside the v0.8 transition explainer.
        assert "### validate" not in text, (
            f"unexpected validate phase header (no_kernel={no_kernel})"
        )
