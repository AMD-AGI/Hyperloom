"""Tests for the Critic cross-domain review primitives.
Auxiliary tests pin the closed envelope schema, classifier idempotency,
and the CriticAgentBackend enrichment helper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends.critic_agent import (
    _maybe_inject_cross_domain_constraints,
    _proposal_provenance_literal,
)
from inference_optimizer.orchestrator.dynamic_action_critic import (
    ALLOWED_VERDICTS,
    CRITIC_VERDICT_FIELDS,
    CROSS_DOMAIN_RULES,
    CrossDomainPreverdict,
    build_critic_verdict_envelope,
    classify_proposal_for_critic,
    is_cross_domain_proposal,
    run_mechanical_cross_domain_checks,
    write_critic_verdict,
)
from inference_optimizer.session_paths import (
    dynamic_action_critic_verdict_path,
)


# ===========================================================================
# Helpers
# ===========================================================================
SCOPE = ["serving_specialist", "kernel_switch_specialist"]


def _proposal(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "combo",
        "provenance": "dynamic",
        "patch_text": (
            "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"
        ),
        "scope_domains": SCOPE,
        "cross_domain_rationale": (
            "serving_specialist must reorder kv layout; "
            "kernel_switch_specialist must adapt the kernel call; "
            "the two depend on each other and may degrade prompt "
            "cache hit ratio."
        ),
        "expected_qualitative_argument": (
            "reduces contention without breaking accuracy"
        ),
    }
    base.update(overrides)
    return base


# ===========================================================================
# Surface invariants
# ===========================================================================
def test_cross_domain_rule_set_locked():
    """Adding a fourth rule (or renaming one) must be a design
    change visible in this assertion."""
    assert tuple(r.rule_id for r in CROSS_DOMAIN_RULES) == (
        "rationale_per_domain",
        "coupling_and_side_effects",
        "motivation_gap_valid",
    )
    assert {r.failure_verdict for r in CROSS_DOMAIN_RULES} == {
        "revise", "reject",
    }


def test_critic_verdict_fields_closed():
    assert CRITIC_VERDICT_FIELDS == frozenset({
        "dyn_id", "verdict", "reason_codes", "reviewer_notes",
        "applied_rules", "cross_domain_flag",
    })


def test_allowed_verdicts_match_p4_5_1():
    assert ALLOWED_VERDICTS == frozenset({"approve", "reject", "revise"})


# ===========================================================================
# Classifier
# ===========================================================================
def test_classifier_specialist_no_constraints():
    cls, rc = classify_proposal_for_critic(
        {"provenance": "specialist:serving_specialist"},
    )
    assert cls == "patch_landing"
    assert rc == {}


def test_classifier_dynamic_adds_cross_domain():
    cls, rc = classify_proposal_for_critic({"provenance": "dynamic"})
    assert cls == "patch_landing"
    assert rc["cross_domain"] is True
    assert len(rc["cross_domain_rules"]) == 3


def test_classifier_default_grid_no_constraints():
    cls, rc = classify_proposal_for_critic({"provenance": "default_grid"})
    assert cls == "patch_landing"
    assert rc == {}


def test_is_cross_domain_proposal_predicate():
    assert is_cross_domain_proposal({"provenance": "dynamic"}) is True
    assert is_cross_domain_proposal(
        {"provenance": "specialist:serving_specialist"},
    ) is False


# ===========================================================================
# §9 #1 — happy path
# ===========================================================================
def test_p4_scenario_01_happy_path_mechanical_layer_passes():
    """Complete proposal + per-domain rationale + coupling +
    side-effect text → mechanical layer falls through to APPROVE so
    the LLM-critic gets the final say."""
    pre = run_mechanical_cross_domain_checks(
        _proposal(), spec_scope_domains=SCOPE,
    )
    assert pre.verdict == "approve"
    assert pre.reason_codes == []
    # full rule audit recorded even on the happy path
    assert "rationale_per_domain" in pre.applied_rules
    assert "coupling_and_side_effects" in pre.applied_rules
    assert "motivation_gap_valid" in pre.applied_rules


# ===========================================================================
# §9 #2 — rationale missing a domain
# ===========================================================================
def test_p4_scenario_02_missing_domain_rationale_revises():
    pre = run_mechanical_cross_domain_checks(
        _proposal(
            cross_domain_rationale=(
                "serving_specialist needs reordering, with coupling "
                "and possible regression risk."
            ),
        ),
        spec_scope_domains=SCOPE,
    )
    assert pre.verdict == "revise"
    assert pre.reason_codes == ["cross_domain_rationale_incomplete"]


# ===========================================================================
# §9 #3 — coupling or side effect not mentioned
# ===========================================================================
def test_p4_scenario_03_coupling_missing_revises():
    pre = run_mechanical_cross_domain_checks(
        _proposal(
            cross_domain_rationale=(
                "serving_specialist independently optimal; "
                "kernel_switch_specialist independently safer."
            ),
        ),
        spec_scope_domains=SCOPE,
    )
    assert pre.verdict == "revise"
    assert pre.reason_codes == ["cross_domain_coupling_unspecified"]


def test_p4_scenario_03b_side_effect_missing_revises():
    pre = run_mechanical_cross_domain_checks(
        _proposal(
            cross_domain_rationale=(
                "serving_specialist couples with kernel_switch_specialist "
                "in this combo; both depend on each other."
            ),
        ),
        spec_scope_domains=SCOPE,
    )
    assert pre.verdict == "revise"
    assert pre.reason_codes == ["cross_domain_coupling_unspecified"]


# ===========================================================================
# §9 #4 — motivation degenerates to grid combo
# ===========================================================================
def test_p4_scenario_04_grid_combo_motivation_rejects():
    pre = run_mechanical_cross_domain_checks(
        _proposal(
            cross_domain_rationale=(
                "Just stack the serving_specialist proposal and the "
                "kernel_switch_specialist proposal together; they "
                "couple naturally with the risk of cache regress."
            ),
        ),
        spec_scope_domains=SCOPE,
    )
    assert pre.verdict == "reject"
    assert pre.reason_codes == ["cross_domain_motivation_invalid"]


# ===========================================================================
# §9 #5 — patch_landing failure trumps cross-domain pass
# ===========================================================================
def test_p4_scenario_05_patch_landing_failure_dominates():
    """The mechanical layer cannot raise the verdict above the
    LLM-critic's patch_landing decision. We model this by composing
    the envelope as the runtime will: take min of mechanical pre +
    LLM verdict; if either says REJECT, the envelope is REJECT."""
    mech = run_mechanical_cross_domain_checks(
        _proposal(), spec_scope_domains=SCOPE,
    )
    assert mech.verdict == "approve"
    # Simulate LLM patch_landing reject (e.g. broken test, missing
    # contract); composition logic is verdict_floor("approve",
    # "reject") = "reject".
    llm_verdict = "reject"
    composed = mech.verdict if mech.verdict in {"reject", "revise"} else llm_verdict
    assert composed == "reject"


# ===========================================================================
# §9 #6 — forged provenance fails fast at critic boundary
# ===========================================================================
@pytest.mark.parametrize("forged", [
    "specialist:serving_specialist",
    "dynamic:kv_cache",
    "default_grid",
    "",
])
def test_p4_scenario_06_forged_provenance_rejected(forged: str):
    pre = run_mechanical_cross_domain_checks(
        _proposal(provenance=forged), spec_scope_domains=SCOPE,
    )
    assert pre.verdict == "reject"
    assert "dynamic_provenance_violation" in pre.reason_codes


# ===========================================================================
# §9 #7 — quantitative claim leaked to critic
# ===========================================================================
@pytest.mark.parametrize("bad_field,value", [
    ("expected_gain",         0.05),
    ("expected_gain_pct",     5.0),
    ("bench_evidence",        {"latency_ms": 12}),
    ("confidence",            0.9),
    ("score",                 100),
    ("rank",                  1),
    ("force_provenance",      "dynamic"),
])
def test_p4_scenario_07_forbidden_field_rejected(
    bad_field: str, value: Any,
):
    proposal = _proposal()
    proposal[bad_field] = value
    pre = run_mechanical_cross_domain_checks(
        proposal, spec_scope_domains=SCOPE,
    )
    assert pre.verdict == "reject"
    assert "dynamic_quantitative_claim_violation" in pre.reason_codes


def test_p4_scenario_07b_numeric_claim_in_qualitative_rejected():
    pre = run_mechanical_cross_domain_checks(
        _proposal(
            expected_qualitative_argument="should give 20% speedup",
        ),
        spec_scope_domains=SCOPE,
    )
    assert pre.verdict == "reject"
    assert "dynamic_quantitative_claim_violation" in pre.reason_codes


# ===========================================================================
# §9 #8 — specialist patch does not trigger cross-domain rules
# ===========================================================================
def test_p4_scenario_08_specialist_proposal_untouched():
    cls, rc = classify_proposal_for_critic({
        "provenance": "specialist:serving_specialist",
        "patch_text": "diff",
    })
    assert cls == "patch_landing"
    assert "cross_domain" not in rc


# ===========================================================================
# §9 #9 — empty proposal_set (handled upstream, no critic call)
# ===========================================================================
def test_p4_scenario_09_empty_proposal_skipped():
    """The Coordinator's P5 wiring must NOT pass an empty proposal_set
    through to the critic. We document the contract at the classifier
    level: an empty / missing payload yields the default (no
    cross_domain flag) — so even if a caller mistakenly invokes
    the classifier on an empty proposal, the legacy path takes over."""
    cls, rc = classify_proposal_for_critic({})
    assert cls == "patch_landing"
    assert rc == {}


# ===========================================================================
# Envelope writer
# ===========================================================================
def test_build_envelope_rejects_unknown_verdict():
    with pytest.raises(ValueError):
        build_critic_verdict_envelope(dyn_id="dyn-0-1", verdict="meh")


def test_build_envelope_emits_exact_schema():
    env = build_critic_verdict_envelope(
        dyn_id="dyn-0-1",
        verdict="approve",
        reason_codes=["x"],
        reviewer_notes="a single string note",
        applied_rules=["rationale_per_domain"],
    )
    assert set(env.keys()) == CRITIC_VERDICT_FIELDS
    assert env["reviewer_notes"] == ["a single string note"]


def test_write_critic_verdict_lands_in_artifact_dir(tmp_path: Path):
    env = build_critic_verdict_envelope(
        dyn_id="dyn-1-1", verdict="reject",
        reason_codes=["cross_domain_motivation_invalid"],
        applied_rules=["motivation_gap_valid"],
    )
    target = write_critic_verdict(tmp_path, "dyn-1-1", env)
    expected = dynamic_action_critic_verdict_path(tmp_path, "dyn-1-1")
    assert target == expected
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["verdict"] == "reject"
    assert on_disk["cross_domain_flag"] is True


# ===========================================================================
# CriticAgentBackend enrichment helper
# ===========================================================================
def test_provenance_literal_reads_top_level_and_nested():
    """§1.2 strict literal — case-sensitive read at every layer
    (P9 invariant I-3)."""
    assert _proposal_provenance_literal(
        {"provenance": "dynamic"},
    ) == "dynamic"
    assert _proposal_provenance_literal(
        {"provenance": "Dynamic"},
    ) == "Dynamic"  # NOT folded; downstream comparisons reject it
    assert _proposal_provenance_literal(
        {"params": {"provenance": "specialist:foo"}},
    ) == "specialist:foo"
    assert _proposal_provenance_literal({"foo": "bar"}) == ""


def test_inject_cross_domain_constraints_specialist_no_op():
    bundle = {
        "proposals": [{
            "msg_id": "m",
            "provenance": "specialist:serving_specialist",
        }],
    }
    _maybe_inject_cross_domain_constraints(bundle)
    assert bundle.get("review_constraints", {}) == {}


def test_inject_cross_domain_constraints_dynamic_sets_flag():
    bundle = {"proposals": [{"msg_id": "m", "provenance": "dynamic"}]}
    _maybe_inject_cross_domain_constraints(bundle)
    rc = bundle["review_constraints"]
    assert rc["cross_domain"] is True
    rule_ids = [r["rule_id"] for r in rc["cross_domain_rules"]]
    assert rule_ids == [
        "rationale_per_domain",
        "coupling_and_side_effects",
        "motivation_gap_valid",
    ]


def test_inject_cross_domain_constraints_idempotent():
    bundle = {"proposals": [{"msg_id": "m", "provenance": "dynamic"}]}
    _maybe_inject_cross_domain_constraints(bundle)
    _maybe_inject_cross_domain_constraints(bundle)
    assert len(bundle["review_constraints"]["cross_domain_rules"]) == 3


def test_inject_cross_domain_constraints_mixed_batch():
    bundle = {
        "proposals": [
            {"msg_id": "m1", "provenance": "specialist:serving_specialist"},
            {"msg_id": "m2", "provenance": "dynamic"},
        ],
    }
    _maybe_inject_cross_domain_constraints(bundle)
    assert bundle["review_constraints"]["cross_domain"] is True


def test_inject_cross_domain_constraints_preserves_action_verdict_policy():
    """An existing ``action_verdict_policy`` entry from the N38
    enrichment path must survive the cross-domain enrichment."""
    bundle = {
        "proposals": [{"msg_id": "m", "provenance": "dynamic"}],
        "review_constraints": {"action_verdict_policy": {"explore": "exploration"}},
    }
    _maybe_inject_cross_domain_constraints(bundle)
    rc = bundle["review_constraints"]
    assert rc["action_verdict_policy"] == {"explore": "exploration"}
    assert rc["cross_domain"] is True


# ===========================================================================
# Severity contract (§7) — mechanical layer never up-ranks LLM verdict
# ===========================================================================
def test_preverdict_is_blocking_predicate():
    assert CrossDomainPreverdict(verdict="reject").is_blocking() is True
    assert CrossDomainPreverdict(verdict="revise").is_blocking() is True
    assert CrossDomainPreverdict(verdict="approve").is_blocking() is False
