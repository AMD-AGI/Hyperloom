# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Step 1 — FRAMEWORK_PR candidate artifacts + outcome classification."""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.orchestrator.framework_pr_artifacts import (
    candidate_slug,
    summarize_candidate_outcomes,
    write_decision_json,
)


# candidate_slug
def test_candidate_slug_sanitizes_url():
    slug = candidate_slug("https://github.com/ROCm/vllm/pull/1234")
    assert "/" not in slug and ":" not in slug
    assert slug.strip("-") == slug  # no leading/trailing hyphens
    assert slug


def test_candidate_slug_empty_defaults():
    assert candidate_slug("") == "candidate"
    assert candidate_slug("///") == "candidate"


def test_candidate_slug_caps_length():
    assert len(candidate_slug("x" * 500)) == 96


# write_decision_json
def test_write_decision_json_roundtrip(tmp_path: Path):
    dest = write_decision_json(
        tmp_path,
        candidate_id="ROCm/vllm#42",
        batch_id="batch-001",
        status="kept",
        kept=True,
        provenance="raw_diff",
        reason="",
        gain_pct=7.5,
        accuracy_pass=True,
        extra={"workspace": "/tmp/x"},
    )
    assert dest is not None
    p = Path(dest)
    assert p.name == "decision.json"
    assert p.parent.parent.name == "framework_pr"  # runs/framework_pr/<slug>/
    data = json.loads(p.read_text())
    assert data["candidate_id"] == "ROCm/vllm#42"
    assert data["batch_id"] == "batch-001"
    assert data["status"] == "kept"
    assert data["kept"] is True
    assert data["provenance"] == "raw_diff"
    assert data["gain_pct"] == 7.5
    assert data["accuracy_pass"] is True
    assert data["workspace"] == "/tmp/x"
    assert data["ts"]


def test_write_decision_json_normalizes_optional_numerics(tmp_path: Path):
    dest = write_decision_json(
        tmp_path,
        candidate_id="cand-x",
        status="critic_denied",
        gain_pct=None,
        accuracy_pass=None,
    )
    data = json.loads(Path(dest).read_text())
    assert data["gain_pct"] is None
    assert data["accuracy_pass"] is None
    assert data["kept"] is False


def test_write_decision_json_never_raises_on_bad_session_dir(tmp_path: Path):
    # A file (not a dir) as session_dir → mkdir fails internally → returns None.
    bad = tmp_path / "afile"
    bad.write_text("x", encoding="utf-8")
    out = write_decision_json(bad, candidate_id="c", status="failed")
    assert out is None


# summarize_candidate_outcomes
def test_summarize_empty_discovery():
    s = summarize_candidate_outcomes([])
    assert s["outcome_class"] == "empty_discovery"
    assert s["total"] == 0
    assert s["keeps"] == 0


def test_summarize_tested_no_keep():
    progress = [
        {"status": "reverted", "kept": False, "batch_id": "b1"},
        {"status": "apply_failed", "kept": False, "batch_id": "b1"},
        {"status": "critic_denied", "kept": False, "batch_id": "b1"},
    ]
    s = summarize_candidate_outcomes(progress)
    assert s["outcome_class"] == "tested_no_keep"
    assert s["keeps"] == 0
    assert s["total"] == 3
    # critic_denied is not a "tested" (applied) status; reverted + apply_failed are.
    assert s["tested"] == 2
    assert s["by_status"]["reverted"] == 1
    assert s["by_status"]["critic_denied"] == 1


def test_summarize_tested_with_keep():
    progress = [
        {"status": "kept", "kept": True, "batch_id": "b1"},
        {"status": "reverted", "kept": False, "batch_id": "b1"},
    ]
    s = summarize_candidate_outcomes(progress)
    assert s["outcome_class"] == "tested_with_keep"
    assert s["keeps"] == 1


def test_summarize_filters_by_batch():
    progress = [
        {"status": "kept", "kept": True, "batch_id": "b1"},
        {"status": "reverted", "kept": False, "batch_id": "b2"},
    ]
    s_b2 = summarize_candidate_outcomes(progress, batch_id="b2")
    assert s_b2["outcome_class"] == "tested_no_keep"
    assert s_b2["total"] == 1
    s_b1 = summarize_candidate_outcomes(progress, batch_id="b1")
    assert s_b1["outcome_class"] == "tested_with_keep"


def test_summarize_ignores_non_dict_rows():
    s = summarize_candidate_outcomes([None, "x", {"status": "kept", "kept": True}])  # type: ignore[list-item]
    assert s["total"] == 1
    assert s["keeps"] == 1
