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
        # N37 (May 2026): validate_stack belongs in the exploration /
        # measurement bucket — it RE-BENCHES the optimization stack to
        # generate a fresh ``cumulative_gain_validated`` number. Per
        # the executor docstring it is "a measurement, not a decision
        # gate" -- it does NOT mutate ``current_best`` /
        # ``optimization_stack``, only the validated-gain scalar.
        # Pre-N37 N35 listed it under the gated/promote bucket which
        # produced a chicken-and-egg loop on the DSR1-0528 production
        # run (LLM proposes validate_stack per TODO 5/5 → critic
        # demands before/after data → validate_stack is the action
        # that produces that data → deadlock → robustness escalates
        # report).
        "validate_stack",
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


def test_critic_md_keeps_integrate_under_the_gate():
    """``integrate`` is the only action that MUTATES the optimization
    stack (appends a KEEP'd kernel patch with an E2E gain claim). It
    MUST stay under the before/after gate; otherwise a single
    unvalidated KEEP can land in the report. After N37 it is the only
    action in the gated bucket -- ``validate_stack`` was moved to the
    exploration carve-out because it is a measurement, not a
    promotion (see N37 docstring above).
    """
    text = CRITIC_MD.read_text(encoding="utf-8")
    assert "`integrate`" in text, (
        "integrate must be explicitly named under the before/after gate"
    )
    assert "PROMOTE" in text or "promote" in text, (
        "expected the rule body to call out PROMOTE-the-stack semantics "
        "as the reason integrate is gated"
    )


def _split_critic_sections(text: str) -> dict[str, str]:
    """Split critic.md into (archival_block, exploration_block,
    gated_block) chunks for section-precise asserts. Crude line-based
    splitter, but stable enough for the bullet structure landed in
    N35/N37."""
    sections: dict[str, list[str]] = {
        "archival": [],
        "exploration": [],
        "gated": [],
    }
    current: str | None = None
    for line in text.splitlines():
        low = line.lower()
        if "archival" in low and "*" in line:
            current = "archival"
        elif "exploration" in low and "measurement" in low:
            current = "exploration"
        elif "before/after benchmark gate only applies" in low:
            current = "gated"
        elif line.startswith("* ") and current is not None:
            # Top-level bullet → exits the sub-bullet section.
            current = None
        if current is not None:
            sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


def test_critic_md_validate_stack_lives_in_exploration_not_gated():
    """N37: ``validate_stack`` is a measurement action (per executor
    docstring "Validate-stack is a measurement, not a decision gate";
    it does NOT mutate ``current_best`` / ``optimization_stack``,
    only the validated-gain scalar). Refusing it on before/after
    grounds is a chicken-and-egg block — ``validate_stack`` is the
    action that PRODUCES the validated number. It MUST appear in the
    exploration carve-out and MUST NOT appear in the gated bucket.
    """
    text = CRITIC_MD.read_text(encoding="utf-8")
    sections = _split_critic_sections(text)
    assert "`validate_stack`" in sections["exploration"], (
        "validate_stack must live in the exploration / measurement "
        "carve-out bucket; section content was:\n"
        + sections["exploration"]
    )
    assert "`validate_stack`" not in sections["gated"], (
        "validate_stack must NOT appear in the gated bucket -- it is a "
        "measurement action, not a promotion. Pre-N37 listing it here "
        "caused the DSR1-0528 production deadlock; section content was:"
        "\n" + sections["gated"]
    )
    # And to anchor the inverse semantics: integrate MUST still be in
    # the gated bucket (it really does promote stack entries).
    assert "`integrate`" in sections["gated"], (
        "integrate must stay in the gated bucket — it actually "
        "mutates optimization_stack/current_best and needs evidence"
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
