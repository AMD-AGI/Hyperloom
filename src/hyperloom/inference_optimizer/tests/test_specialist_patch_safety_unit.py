# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the universal patch-safety contract (diff structural checks,
git grounding, missing-target detection, and quantitative-claim guards)."""

from __future__ import annotations


import pytest

from hyperloom.orchestrator.specialists import patch_safety as ps


_DIFF = "diff --git a/foo.py b/foo.py\nindex 111..222 100644\n--- a/foo.py\n+++ b/foo.py\n@@ -1,2 +1,2 @@\n-old\n+new\n"


# ---- path helpers ---------------------------------------------------------
def test_strip_path_prefix():
    assert ps._strip_path_prefix("a/b/c.py", 0) == "a/b/c.py"
    assert ps._strip_path_prefix("a/b/c.py", 1) == "b/c.py"
    assert ps._strip_path_prefix("a/b/c.py", 5) == "c.py"


def test_patch_file_targets():
    pairs = ps.patch_file_targets(_DIFF)
    assert pairs == [("a/foo.py", "b/foo.py")]
    assert ps.patch_file_targets("") == []
    # trailing timestamp on the header is stripped
    txt = "--- a/x.py\t2026-01-01\n+++ b/x.py\t2026-01-01\n"
    assert ps.patch_file_targets(txt) == [("a/x.py", "b/x.py")]


def test_patch_targets_missing(tmp_path):
    (tmp_path / "foo.py").write_text("x", encoding="utf-8")
    # foo.py exists at strip level 1 -> not missing
    assert ps.patch_targets_missing(_DIFF, tmp_path) == []
    miss_diff = _DIFF.replace("foo.py", "ghost.py")
    assert ps.patch_targets_missing(miss_diff, tmp_path) == ["a/ghost.py"]


def test_patch_targets_missing_dev_null_exempt(tmp_path):
    create = "--- /dev/null\n+++ b/newfile.py\n@@ -0,0 +1 @@\n+content\n"
    assert ps.patch_targets_missing(create, tmp_path) == []


# ---- regex helpers --------------------------------------------------------
def test_numeric_claims():
    text = "this gives 12% and 3.5x speedup of 4, 100ms latency"
    hits = ps.numeric_claims(text)
    assert any("%" in h for h in hits)
    assert any("x" in h.lower() for h in hits)
    assert ps.numeric_claims("") == []


def test_is_unified_diff():
    assert ps.is_unified_diff(_DIFF) is True
    assert ps.is_unified_diff("just text") is False


def test_patch_escapes_tree():
    assert ps.patch_escapes_tree(_DIFF) is None
    # target after the b/ prefix begins with "/" -> absolute escape
    abs_diff = "--- a/foo.py\n+++ b//etc/passwd\n"
    assert ps.patch_escapes_tree(abs_diff) == "/etc/passwd"
    dotdot = "--- a/../../escape.py\n+++ b/ok.py\n"
    assert ps.patch_escapes_tree(dotdot) == "../../escape.py"


# ---- cross-domain rule descriptors ----------------------------------------
def test_cross_domain_rule_descriptors():
    desc = ps.cross_domain_rule_descriptors()
    assert len(desc) == len(ps.CROSS_DOMAIN_RULES)
    assert {"rule_id", "description", "failure_verdict", "failure_reason_code"} <= set(desc[0])


# ---- PatchGroundingResult.is_garbage --------------------------------------
def test_is_garbage():
    assert ps.PatchGroundingResult(ps.GROUND_NOT_DIFF).is_garbage is True
    assert ps.PatchGroundingResult(ps.GROUND_PATH_ESCAPE).is_garbage is True
    assert ps.PatchGroundingResult(ps.GROUND_MISSING_TARGET).is_garbage is True
    assert ps.PatchGroundingResult(ps.GROUND_STALE).is_garbage is False
    assert ps.PatchGroundingResult(ps.GROUND_APPLIES).is_garbage is False


# ---- ground_patch_text ----------------------------------------------------
def test_ground_not_diff():
    res = ps.ground_patch_text("no hunks", base_checkout=None)
    assert res.verdict == ps.GROUND_NOT_DIFF


def test_ground_path_escape():
    escape_diff = "--- a/foo.py\n+++ b/../../escape.py\n@@ -1 +1 @@\n-old\n+new\n"
    res = ps.ground_patch_text(escape_diff, base_checkout=None)
    assert res.verdict == ps.GROUND_PATH_ESCAPE


def test_ground_unchecked_no_base():
    res = ps.ground_patch_text(_DIFF, base_checkout=None)
    assert res.verdict == ps.GROUND_UNCHECKED


def test_ground_missing_target(tmp_path):
    res = ps.ground_patch_text(_DIFF, base_checkout=tmp_path)
    assert res.verdict == ps.GROUND_MISSING_TARGET


def test_ground_applies(tmp_path, monkeypatch):
    (tmp_path / "foo.py").write_text("old\n", encoding="utf-8")

    class _Proc:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(ps.subprocess, "run", lambda *a, **k: _Proc())
    res = ps.ground_patch_text(_DIFF, base_checkout=tmp_path)
    assert res.verdict == ps.GROUND_APPLIES


def test_ground_stale(tmp_path, monkeypatch):
    (tmp_path / "foo.py").write_text("old\n", encoding="utf-8")

    class _Proc:
        returncode = 1
        stderr = "patch does not apply"

    monkeypatch.setattr(ps.subprocess, "run", lambda *a, **k: _Proc())
    res = ps.ground_patch_text(_DIFF, base_checkout=tmp_path)
    assert res.verdict == ps.GROUND_STALE
    assert "does not apply" in res.detail


def test_ground_git_unavailable(tmp_path, monkeypatch):
    (tmp_path / "foo.py").write_text("old\n", encoding="utf-8")

    def _raise(*a, **k):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(ps.subprocess, "run", _raise)
    res = ps.ground_patch_text(_DIFF, base_checkout=tmp_path)
    assert res.verdict == ps.GROUND_UNCHECKED


# ---- PatchSafetyReport.notes ----------------------------------------------
def test_patch_safety_report_notes():
    rep = ps.PatchSafetyReport(
        dropped=[
            {"path": "p1", "verdict": ps.GROUND_NOT_DIFF, "detail": "d"},
            {"path": "p2", "verdict": ps.GROUND_MISSING_TARGET, "detail": "miss"},
        ],
        grounding={"p3": ps.GROUND_STALE},
        numeric_warnings=["12%"],
        forbidden_fields=["confidence"],
    )
    notes = rep.notes()
    joined = "\n".join(notes)
    assert "patch_safety_dropped" in joined
    assert "patch_safety_missing_target" in joined
    assert "patch_safety_stale" in joined
    assert "patch_safety_numeric" in joined
    assert "patch_safety_forbidden_fields" in joined


def test_patch_safety_report_notes_empty():
    assert ps.PatchSafetyReport().notes() == []


# ---- scan_quantitative_claims ---------------------------------------------
def test_scan_quantitative_claims():
    payload = {
        "expected_gain": 12.0,
        "summary": "gives 20% boost",
        "proposal_set": [
            {"score": 1, "confidence": 0.9, "expected_qualitative_argument": "3x faster"},
            "not-a-dict",
        ],
    }
    forbidden, warnings = ps.scan_quantitative_claims(payload)
    assert "expected_gain" in forbidden
    assert "score" in forbidden
    assert "confidence" in forbidden
    assert any("%" in w for w in warnings)
    assert any("x" in w.lower() for w in warnings)


def test_scan_quantitative_claims_empty():
    assert ps.scan_quantitative_claims({}) == ([], [])


def test_round_level_confidence_is_not_a_per_proposal_gain_claim():
    """The output schema asks for it and the round audit records it, so flagging
    it at the top level only made our own template a violation."""
    forbidden, _ = ps.scan_quantitative_claims({"confidence": 0.6})

    assert forbidden == []


# ---- strip_forbidden_proposal_fields --------------------------------------
def test_forbidden_fields_are_stripped_so_the_critic_cannot_reject_on_format():
    payload = {
        "expected_gain": 9.0,
        "confidence": 0.6,
        "summary": "keep me",
        "proposal_set": [
            {"name": "v1", "confidence": 0.4, "score": 3, "reason": "keep me too"},
            "not-a-dict",
        ],
    }

    removed = ps.strip_forbidden_proposal_fields(payload)

    assert set(removed) == {"expected_gain", "confidence", "score"}
    assert "expected_gain" not in payload
    assert payload["confidence"] == 0.6  # round-level self-assessment survives
    assert payload["summary"] == "keep me"
    assert payload["proposal_set"][0] == {"name": "v1", "reason": "keep me too"}
    assert payload["proposal_set"][1] == "not-a-dict"


def test_a_gain_claim_under_the_coordinators_own_field_name_is_stripped_too():
    """``predicted_gain_pct`` is the Coordinator's estimate on a propose_action
    intent, which is exactly what made it a convenient place for a specialist
    to put a number the guard was meant to strip."""
    payload = {"proposal_set": [{"name": "v1", "predicted_gain_pct": 12.0, "reason": "keep me"}]}

    removed = ps.strip_forbidden_proposal_fields(payload)

    assert removed == ["predicted_gain_pct"]
    assert payload["proposal_set"][0] == {"name": "v1", "reason": "keep me"}


def test_stripping_a_clean_payload_changes_nothing():
    payload = {"proposal_set": [{"name": "v1", "reason": "why"}]}

    assert ps.strip_forbidden_proposal_fields(payload) == []
    assert payload == {"proposal_set": [{"name": "v1", "reason": "why"}]}


@pytest.mark.parametrize("payload", [None, [], "", 0])
def test_stripping_tolerates_a_payload_that_is_not_a_dict(payload):
    assert ps.strip_forbidden_proposal_fields(payload) == []


# ---- quantitative_claim_rule_descriptor ------------------------------------
def test_the_rule_the_critic_gets_lists_exactly_what_the_runner_strips():
    """A hand-copied field list in the prompt is how the Critic came to reject
    over a field the runner never enforced."""
    rule = ps.quantitative_claim_rule_descriptor()

    assert set(rule["forbidden_proposal_fields"]) == set(ps.FORBIDDEN_PROPOSAL_FIELDS)


def test_a_format_slip_is_advisory_not_a_reject():
    rule = ps.quantitative_claim_rule_descriptor()

    assert rule["failure_verdict"] == "advise"
    assert rule["failure_reason_code"] == ps.QUANTITATIVE_CLAIM_REASON_CODE


# ---- advisory_only_reason_codes --------------------------------------------
def test_every_rule_that_asked_for_advice_is_enforceable():
    codes = ps.advisory_only_reason_codes()

    assert ps.QUANTITATIVE_CLAIM_REASON_CODE in codes
    for rule in ps.cross_domain_rule_descriptors():
        if rule["failure_verdict"] == ps.ADVISE_VERDICT:
            assert rule["failure_reason_code"] in codes


def test_a_rule_asking_for_a_reject_is_left_alone(monkeypatch):
    """The set is derived from the descriptors, so a rule keeping ``reject`` stays out of it."""
    hard = ps.CrossDomainRule(
        rule_id="hard_guard",
        description="a violation here is grounds for refusal",
        failure_verdict="reject",
        failure_reason_code="cross_domain_hard_guard",
    )
    monkeypatch.setattr(ps, "CROSS_DOMAIN_RULES", ps.CROSS_DOMAIN_RULES + (hard,))

    codes = ps.advisory_only_reason_codes()

    assert "cross_domain_hard_guard" not in codes
    assert ps.QUANTITATIVE_CLAIM_REASON_CODE in codes


def test_the_codes_carry_no_blank_entry():
    assert "" not in ps.advisory_only_reason_codes()


# ---- vet_patches ----------------------------------------------------------
def test_vet_patches(tmp_path, monkeypatch):
    good = tmp_path / "good.patch"
    (tmp_path / "foo.py").write_text("old\n", encoding="utf-8")
    good.write_text(_DIFF, encoding="utf-8")
    bad = tmp_path / "bad.patch"
    bad.write_text("not a diff", encoding="utf-8")

    class _Proc:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(ps.subprocess, "run", lambda *a, **k: _Proc())
    kept, dropped, grounding = ps.vet_patches([str(good), str(bad)], base_checkout=tmp_path)
    assert str(good) in kept
    assert any(d["verdict"] == ps.GROUND_NOT_DIFF for d in dropped)


def test_vet_patches_unreadable(tmp_path):
    kept, dropped, grounding = ps.vet_patches([str(tmp_path / "missing.patch")], base_checkout=None)
    assert kept == []
    assert dropped[0]["verdict"] == "unreadable"
