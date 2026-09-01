# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the universal patch-safety contract (diff structural checks,
git grounding, missing-target detection, and quantitative-claim guards)."""

from __future__ import annotations


import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.specialists import patch_safety as ps
from hyperloom.orchestrator.specialists import runner


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


def test_parse_patch_targets_classifies_modify_create_and_delete():
    patch = (
        _DIFF + "diff --git a/new.py b/new.py\n"
        "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+new\n" + "diff --git a/old.py b/old.py\n"
        "--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n"
    )

    parsed = ps.parse_patch_targets(patch)

    assert parsed.existing == ("foo.py", "old.py")
    assert parsed.created == ("new.py",)
    assert parsed.all == ("foo.py", "old.py", "new.py")


def test_parse_patch_targets_falls_back_for_mode_only_and_rename():
    mode_only = "diff --git a/script.py b/script.py\nold mode 100644\nnew mode 100755\n"
    rename = "diff --git a/old.py b/new.py\nsimilarity index 100%\nrename from old.py\nrename to new.py\n"

    assert ps.parse_patch_targets(mode_only).existing == ("script.py",)
    parsed_rename = ps.parse_patch_targets(rename)
    assert parsed_rename.existing == ("old.py",)
    assert parsed_rename.created == ("new.py",)


def test_parse_patch_targets_rejects_root_escape():
    try:
        ps.parse_patch_targets("diff --git a/good.py b/../../escape.py\n--- a/good.py\n+++ b/../../escape.py\n")
    except ValueError as exc:
        assert "unsafe patch target path" in str(exc)
    else:
        raise AssertionError("root-escaping patch target was accepted")


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


# ---- scan_numeric_claims ---------------------------------------------------
def test_scan_numeric_claims():
    payload = {
        "summary": "gives 20% boost",
        "proposal_set": [
            {"expected_qualitative_argument": "3x faster"},
            "not-a-dict",
        ],
    }
    warnings = ps.scan_numeric_claims(payload)
    assert any("%" in w for w in warnings)
    assert any("x" in w.lower() for w in warnings)


def test_scan_numeric_claims_empty():
    assert ps.scan_numeric_claims({}) == []


def test_the_numeric_scan_answers_only_the_question_no_one_else_answers():
    """It used to return the same ``keys & FORBIDDEN_*`` intersection
    ``strip_forbidden_proposal_fields`` computes, for a caller that discarded
    it: one question with two implementations, free to drift apart. The numbers
    in the prose are what this scan alone finds."""
    payload = {
        "expected_gain": 12.0,
        "summary": "gives 20% boost",
        "proposal_set": [{"score": 1, "confidence": 0.9}],
    }

    assert ps.scan_numeric_claims(payload) == ["20%"]


# ---- strip_forbidden_proposal_fields --------------------------------------
def test_round_level_confidence_is_not_a_per_proposal_gain_claim():
    """The output schema asks for a round-level self-assessment and the round
    audit records it, so stripping it at the top level only made our own
    template a violation. Per proposal it is the ranking claim the guard is
    about, and one function now decides both."""
    payload = {"confidence": 0.6, "proposal_set": [{"confidence": 0.4}]}

    assert ps.strip_forbidden_proposal_fields(payload) == ["confidence"]
    assert payload["confidence"] == 0.6
    assert payload["proposal_set"][0] == {}


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


# ---- advisory_rules_govern -------------------------------------------------
@pytest.mark.parametrize("action_name", ["specialist", "explore"])
def test_the_rules_govern_the_proposal_kinds_they_are_written_about(action_name):
    """``proposal_set[*]`` reaches review as a specialist proposal or the explore
    grid it is materialised into; the framework candidate is the one the
    quantitative-claim rule names by exception."""
    assert ps.advisory_rules_govern(action_name) is True


@pytest.mark.parametrize("action_name", ["integrate_patch", "kernel_opt", "sweep", "baseline", "", None])
def test_integrate_patch_is_never_governed_by_an_advisory_rule(action_name):
    """None of these carries a specialist ``proposal_set``, and holding an
    ``integrate_patch`` reject to ``advise`` would land the refused patch."""
    assert ps.advisory_rules_govern(action_name) is False


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
    kept, dropped, grounding, spans_roots = ps.vet_patches([str(good), str(bad)], base_checkout=tmp_path)
    assert str(good) in kept
    assert any(d["verdict"] == ps.GROUND_NOT_DIFF for d in dropped)
    assert not spans_roots


def test_vet_patches_unreadable(tmp_path):
    kept, dropped, grounding, spans_roots = ps.vet_patches([str(tmp_path / "missing.patch")], base_checkout=None)
    assert kept == []
    assert dropped[0]["verdict"] == "unreadable"
    assert not spans_roots


# ---- nested root collapse + GROUND_AMBIGUOUS_ROOT -------------------------
def _make_git_repo(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "base"],
        check=True,
    )
    return root


def test_nested_root_collapse_picks_outer(tmp_path):
    outer = _make_git_repo(tmp_path / "sglang", {"python/sglang/srt/foo.py": "old\n"})
    inner = outer / "python"
    inner.mkdir(exist_ok=True)
    diff = "--- a/python/sglang/srt/foo.py\n+++ b/python/sglang/srt/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
    res = ps.resolve_patch_apply_root([diff], explicit_root=None, candidate_roots=[outer, inner])
    assert res.root is not None
    assert res.root.resolve() == outer.resolve()
    assert res.reason == ""


def test_disjoint_ambiguity_still_fails(tmp_path):
    tree_a = _make_git_repo(tmp_path / "a", {"srt/foo.py": "old\n"})
    tree_b = _make_git_repo(tmp_path / "b", {"srt/foo.py": "old\n"})
    diff = "--- a/srt/foo.py\n+++ b/srt/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
    res = ps.resolve_patch_apply_root([diff], explicit_root=None, candidate_roots=[tree_a, tree_b])
    assert res.root is None
    assert res.reason == "ambiguous_root"


def test_ground_patch_text_returns_ambiguous_root_verdict(tmp_path):
    tree_a = _make_git_repo(tmp_path / "a", {"foo.py": "old\n"})
    tree_b = _make_git_repo(tmp_path / "b", {"foo.py": "old\n"})
    diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
    res = ps.ground_patch_text(diff, base_checkout=tree_a, candidate_roots=(tree_b,))
    assert res.verdict == ps.GROUND_AMBIGUOUS_ROOT
    assert res.is_garbage


def test_vet_patches_ambiguous_root_not_labeled_missing_target(tmp_path):
    tree_a = _make_git_repo(tmp_path / "a", {"foo.py": "old\n"})
    tree_b = _make_git_repo(tmp_path / "b", {"foo.py": "old\n"})
    diff_file = tmp_path / "p.patch"
    diff_file.write_text("--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n", encoding="utf-8")
    _, dropped, grounding, _ = ps.vet_patches([str(diff_file)], base_checkout=tree_a, candidate_roots=(tree_b,))
    assert len(dropped) == 1
    assert dropped[0]["verdict"] == ps.GROUND_AMBIGUOUS_ROOT
    assert grounding[str(diff_file)] == ps.GROUND_AMBIGUOUS_ROOT


# ---- grounding root selection ----------------------------------------------
def test_grounding_root_uses_a_harvest_the_whole_set_agrees_on():
    root = runner._grounding_explicit_root(
        declared="",
        patches=["/wt/patches/_worktree_diff.patch"],
        patch_roots={"/wt/patches/_worktree_diff.patch": "/sgl-workspace/sglang"},
    )
    assert root == Path("/sgl-workspace/sglang")


def test_grounding_root_declines_when_a_hand_authored_patch_rides_along():
    """Its target tree is unknown; the harvest root would drop it as a mismatch."""
    root = runner._grounding_explicit_root(
        declared="",
        patches=["/wt/patches/_worktree_diff.patch", "/wt/patches/manual.patch"],
        patch_roots={"/wt/patches/_worktree_diff.patch": "/sgl-workspace/sglang"},
    )
    assert root is None


def test_grounding_root_declines_when_harvests_disagree():
    root = runner._grounding_explicit_root(
        declared="",
        patches=["/a.patch", "/b.patch"],
        patch_roots={"/a.patch": "/tree/one", "/b.patch": "/tree/two"},
    )
    assert root is None


def test_grounding_root_prefers_a_declared_source_root():
    root = runner._grounding_explicit_root(
        declared="/declared/tree",
        patches=["/a.patch"],
        patch_roots={"/a.patch": "/harvested/tree"},
    )
    assert root == Path("/declared/tree")


def test_grounding_root_declines_for_an_empty_set():
    assert runner._grounding_explicit_root(declared="", patches=[], patch_roots={}) is None


def test_a_deleted_line_that_looks_like_a_header_is_not_a_path():
    """A hunk body line is not a header, whatever it starts with.

    Deleting a source line that begins with ``--`` renders as ``--- ...`` in
    the diff. Reading lines independently cannot tell that from a header, and
    a comment naming an absolute path got the whole patch rejected as a
    traversal.
    """
    for body in ("-- /etc/hosts is read at startup", "-- ../legacy/foo is gone"):
        diff = f"--- a/x.sql\n+++ b/x.sql\n@@ -1,2 +1,1 @@\n-{body}\n keep\n"
        assert ps.patch_escapes_tree(diff) is None


def test_the_escape_check_reads_the_paths_the_applier_resolves():
    """Both sides of every header pair, normalised the way ``-p1`` strips them."""
    assert ps.patch_escapes_tree("--- a/ok.py\n+++ b//etc/passwd\n") == "/etc/passwd"
    assert ps.patch_escapes_tree("--- a/../../escape.py\n+++ b/ok.py\n") == "../../escape.py"
    assert ps.patch_escapes_tree("--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n") is None
