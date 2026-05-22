"""N35 (May 2026) critic-prompt exploration-action carve-out tests.

Regression from the DSR1-0528 production run on 2026-05-21 (session
``20260521T150208Z``): after baseline + roofline succeeded, the LLM
correctly proposed ``params`` three times with strong evidence from
``analysis.md`` (99.9% GPU idle, host-bound, N22 keyword matches for
"host-bound" / "cuda graphs" / "torch.compile"). The critic refused
all three with ``advise`` / ``needs_review`` verdicts because the
proposal "did not provide comparable before/after benchmark or
accuracy-gate evidence."

This is a chicken-and-egg deadlock: ``params`` (and ``backends`` /
``sweep`` / ``kernel_opt``) are the actions that GENERATE the
before/after data the gate was supposed to protect. Refusing them
blocks the entire optimization loop after baseline -- robustness then
escalated ``strategy_change`` -> ``report``, and the session exited
clean (N34 Bug #4 fix) but with zero cumulative gain and zero stack
entries.

The N35 fix extends the N33 archival carve-out: the before/after
benchmark gate now ONLY applies to actions that PROMOTE the stack or
CLAIM a validated gain (``integrate`` / ``validate_stack``). All
other actions -- archival OR exploration -- must approve when they
are the natural next TODO per orchestration's sequencing rules.
"""
from __future__ import annotations

from pathlib import Path


CRITIC_MD = (
    Path(__file__).resolve().parent.parent
    / "orchestrator" / "system_prompts" / "critic.md"
)


def test_critic_md_carves_out_exploration_actions():
    """The exploration / measurement carve-out must list every action
    that runs benchmarks or variants to generate data. Missing any one
    of them reproduces the DSR1-0528 deadlock for that family.
    """
    text = CRITIC_MD.read_text(encoding="utf-8")
    assert "Exploration / measurement" in text, (
        "expected the exploration-action carve-out to live in critic.md"
    )
    expected_actions = [
        "baseline",
        "profile",
        "roofline",
        "params",
        "backends",
        "sweep",
        "kernel_opt",
        "pmc_roofline",
        "compiler_tuning",
        "comm_optimization",
        "operator_tuning",
        "vendor_kernel_config",
        "deep_kernel_analysis",
        "recover",
    ]
    for action in expected_actions:
        assert f"`{action}`" in text, (
            f"{action!r} missing from exploration carve-out -- proposals "
            f"of {action!r} will be blocked by the before/after gate"
        )


def test_critic_md_keeps_archival_carve_out():
    """The N33 archival carve-out must survive the N35 extension --
    refusing ``report`` would still bypass the N33 silent-tick safety
    net via wall-clock idle."""
    text = CRITIC_MD.read_text(encoding="utf-8")
    assert "Archival" in text
    for archival in ("`report`", "`session_breakdown`", "`target_analysis`"):
        assert archival in text


def test_critic_md_keeps_promote_actions_under_the_gate():
    """``integrate`` and ``validate_stack`` are the only actions that
    mutate ``current_best`` / ``optimization_stack`` /
    ``cumulative_gain_validated``. They MUST stay under the
    before/after gate; otherwise a single unvalidated KEEP can land in
    the report."""
    text = CRITIC_MD.read_text(encoding="utf-8")
    # The hard rule must explicitly name the two promote-stack actions
    # as the only ones the gate applies to.
    assert "`integrate`" in text and "`validate_stack`" in text, (
        "the before/after gate must explicitly scope to integrate + "
        "validate_stack so future readers don't accidentally widen it"
    )
    assert "PROMOTE" in text or "promote" in text, (
        "expected the rule body to call out PROMOTE-the-stack semantics "
        "as the reason these two actions are gated"
    )


def test_critic_md_explains_chicken_and_egg_rationale():
    """Future maintainers must understand WHY exploration actions are
    carved out (so they don't 're-tighten' the rule). Verify the
    rationale string is present."""
    text = CRITIC_MD.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "chicken-and-egg" in lowered or "deadlock" in lowered, (
        "expected the rationale to call out the chicken-and-egg / "
        "deadlock condition so the carve-out isn't reverted"
    )
