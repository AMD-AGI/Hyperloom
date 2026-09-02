# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for framework-agent pure helpers: KB prior-scoring (``decision``)
and unified-diff parsing (``_audit_common``).

Both modules are pure over plain dicts / dataclasses, so they are exercised
directly here without any network, worktree, or gbrain/pr_monitor backend.
"""

from __future__ import annotations

from types import SimpleNamespace

from hyperloom.agents.framework import _audit_common as ac
from hyperloom.agents.framework.decision import (
    candidate_score,
    prior_score,
    winner_decision,
)
from hyperloom.agents.framework.models import Candidate


# --------------------------------------------------------------------------
# decision.py
# --------------------------------------------------------------------------
def test_prior_score_cold_start_and_min_samples() -> None:
    # No ledger -> cold start.
    assert prior_score({"framework": "vllm"}, ledger=[]) == 0.0
    # Records exist but none associate (framework mismatch) -> below min_samples.
    ledger = [{"framework": "sglang", "gap_canonical_id": "g1"}]
    assert (
        prior_score(
            {"framework": "vllm", "gap_canonical_id": "g1"},
            gap_canonical_id="g1",
            ledger=ledger,
        )
        == 0.0
    )


def test_prior_score_associates_and_scores() -> None:
    # Candidate as an object (exercises the getattr branch of _field).
    cand = Candidate(
        ref="pr/7",
        repo="acme/x",
        framework="vllm",
        model_class="moe",
        gpu_type="mi300x",
        precision="fp8",
        gap_canonical_id="gap-decode",
        gap_keywords=("decode", "throughput"),
        head_sha="sha7",
    )
    ledger = [
        # Exact gap + full param match + integrated outcome + positive gain.
        {
            "framework": "vllm",
            "gap_canonical_id": "gap-decode",
            "model_class": "moe",
            "gpu_type": "mi300x",
            "precision": "fp8",
            "tps_delta_pct": 30.0,
            "outcome": "integrated",
        },
        # Fuzzy keyword association + already_present outcome.
        {
            "framework": "vllm",
            "gap_canonical_id": "other",
            "gap_keywords": ["decode"],
            "tps_delta_pct": 5.0,
            "outcome": "already_present",
        },
        # Framework mismatch -> skipped.
        {"framework": "sglang", "gap_canonical_id": "gap-decode"},
        # No gap match and no keyword overlap -> skipped.
        {"framework": "vllm", "gap_canonical_id": "zzz", "gap_keywords": ["unrelated"]},
    ]
    score = prior_score(cand, ledger=ledger)
    assert 0.0 < score <= 1.0


def test_prior_score_exact_pr_url_or_sha_match() -> None:
    cand = {"framework": "vllm", "html_url": "http://gh/pr/9", "head_sha": "deadbeef"}
    ledger = [
        {"framework": "vllm", "pr_url": "http://gh/pr/9", "tps_delta_pct": 12.0, "outcome": "integrated"},
        {"framework": "vllm", "gap_canonical_id": "g", "gap_keywords": ["x"]},
    ]
    assert prior_score(cand, ledger=ledger) > 0.0


def test_candidate_score_and_winner_decision() -> None:
    req = SimpleNamespace(
        baseline=SimpleNamespace(throughput=100.0, accuracy=0.80),
        thresholds=SimpleNamespace(min_throughput_ratio=1.05, max_accuracy_drop=0.02),
    )
    # No accuracy -> plain ratio branch.
    req_no_acc = SimpleNamespace(
        baseline=SimpleNamespace(throughput=100.0, accuracy=None),
        thresholds=SimpleNamespace(min_throughput_ratio=1.05, max_accuracy_drop=0.02),
    )
    assert candidate_score(req_no_acc, 150.0, None) == 1.5
    assert candidate_score(req, None, None) == 0.0
    # Accuracy-penalized ranking.
    assert candidate_score(req, 150.0, 0.79) < 1.5

    assert winner_decision(req, None, None, "")[0] is False
    assert winner_decision(req, 150.0, 0.80, "4/4")[0] is True
    assert winner_decision(req, 101.0, 0.80, "")[0] is False  # ratio below floor


# --------------------------------------------------------------------------
# _audit_common.py
# --------------------------------------------------------------------------
def test_parse_unified_diff_with_and_without_git_header() -> None:
    patch = (
        "diff --git a/src/foo.py b/src/foo.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/src/foo.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def added():\n"
        "+    return 1\n"
        " context line\n"
        "-removed line\n"
    )
    changes = ac.parse_unified_diff(patch)
    assert len(changes) == 1
    assert changes[0].path == "src/foo.py"
    assert changes[0].is_new is True
    assert "def added():" in changes[0].added

    # No "diff --git" header: a bare "--- / +++" section still parses.
    bare = "--- a/x.py\n+++ b/x.py\n@@\n+line\n"
    bare_changes = ac.parse_unified_diff(bare)
    assert bare_changes and bare_changes[0].path == "x.py"

    # Junk before any section is tolerated (no crash, no sections).
    assert ac.parse_unified_diff("random preamble\nnothing here\n") == []


def test_strip_diff_path_variants() -> None:
    assert ac._strip_diff_path("a/src/foo.py\t2024") == "src/foo.py"
    assert ac._strip_diff_path("/dev/null") == "/dev/null"
    # No a//b/ prefix -> returned as-is.
    assert ac._strip_diff_path("plain/path.py") == "plain/path.py"
