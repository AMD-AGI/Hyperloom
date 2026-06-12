# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the universal patch-safety contract (diff structural checks,
git grounding, missing-target detection, and quantitative-claim guards)."""
from __future__ import annotations

from pathlib import Path

from inference_optimizer.orchestrator import specialist_patch_safety as ps


_DIFF = (
    "diff --git a/foo.py b/foo.py\n"
    "index 111..222 100644\n"
    "--- a/foo.py\n"
    "+++ b/foo.py\n"
    "@@ -1,2 +1,2 @@\n"
    "-old\n"
    "+new\n"
)


# ---- path helpers ---------------------------------------------------------
def test_strip_path_prefix():
    assert ps._strip_path_prefix("a/b/c.py", 0) == "a/b/c.py"
    assert ps._strip_path_prefix("a/b/c.py", 1) == "b/c.py"
    assert ps._strip_path_prefix("a/b/c.py", 5) == "c.py"  # level >= parts


def test_patch_file_targets():
    pairs = ps.patch_file_targets(_DIFF)
    assert pairs == [("a/foo.py", "b/foo.py")]
    assert ps.patch_file_targets("") == []
    # header with trailing timestamp stripped
    txt = "--- a/x.py\t2026-01-01\n+++ b/x.py\t2026-01-01\n"
    assert ps.patch_file_targets(txt) == [("a/x.py", "b/x.py")]


def test_patch_targets_missing(tmp_path):
    (tmp_path / "foo.py").write_text("x", encoding="utf-8")
    # foo.py exists at strip level 1 -> not missing
    assert ps.patch_targets_missing(_DIFF, tmp_path) == []
    # non-existent target
    miss_diff = _DIFF.replace("foo.py", "ghost.py")
    assert ps.patch_targets_missing(miss_diff, tmp_path) == ["a/ghost.py"]


def test_patch_targets_missing_dev_null_exempt(tmp_path):
    create = (
        "--- /dev/null\n"
        "+++ b/newfile.py\n"
        "@@ -0,0 +1 @@\n"
        "+content\n"
    )
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


def test_normalise_diff_for_compare():
    out = ps.normalise_diff_for_compare(_DIFF)
    assert "index" not in out
    assert "diff --git" not in out
    assert "@@" in out


def test_patch_escapes_tree():
    assert ps.patch_escapes_tree(_DIFF) is None
    # cand after the b/ prefix begins with "/" -> absolute escape
    abs_diff = "--- a/foo.py\n+++ b//etc/passwd\n"
    assert ps.patch_escapes_tree(abs_diff) == "/etc/passwd"
    dotdot = "--- a/../../escape.py\n+++ b/ok.py\n"
    assert ps.patch_escapes_tree(dotdot) == "../../escape.py"


# ---- cross-domain rule descriptors ----------------------------------------
def test_cross_domain_rule_descriptors():
    desc = ps.cross_domain_rule_descriptors()
    assert len(desc) == len(ps.CROSS_DOMAIN_RULES)
    assert {"rule_id", "description", "failure_verdict",
            "failure_reason_code"} <= set(desc[0])


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
    escape_diff = (
        "--- a/foo.py\n"
        "+++ b/../../escape.py\n"
        "@@ -1 +1 @@\n"
        "-old\n+new\n"
    )
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
        "confidence": 0.9,
        "summary": "gives 20% boost",
        "proposal_set": [
            {"score": 1, "expected_qualitative_argument": "3x faster"},
            "not-a-dict",
        ],
    }
    forbidden, warnings = ps.scan_quantitative_claims(payload)
    assert "confidence" in forbidden
    assert "score" in forbidden
    assert any("%" in w for w in warnings)
    assert any("x" in w.lower() for w in warnings)


def test_scan_quantitative_claims_empty():
    assert ps.scan_quantitative_claims({}) == ([], [])


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
    kept, dropped, grounding = ps.vet_patches(
        [str(good), str(bad)], base_checkout=tmp_path)
    assert str(good) in kept
    assert any(d["verdict"] == ps.GROUND_NOT_DIFF for d in dropped)


def test_vet_patches_unreadable(tmp_path):
    kept, dropped, grounding = ps.vet_patches(
        [str(tmp_path / "missing.patch")], base_checkout=None)
    assert kept == []
    assert dropped[0]["verdict"] == "unreadable"
