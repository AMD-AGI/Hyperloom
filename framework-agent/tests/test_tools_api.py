# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for framework_agent.runtime.tools_api.

Hermetic - patches sources.primus_cortex / sources.github helpers so
no real HTTP traffic is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from framework_agent.runtime import tools_api
from framework_agent.sources._shared import GitHubPr


# find_relevant_prs_smart ------------------------------------------------


def test_find_relevant_prs_smart_empty_repos_returns_empty() -> None:
    """No repos means nothing to query; returns ``[]`` deterministically."""
    out = tools_api.find_relevant_prs_smart("anything", repos=[])
    assert out == []
    out2 = tools_api.find_relevant_prs_smart("anything")
    assert out2 == []


def test_find_relevant_prs_smart_primus_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """With primus_cortex_url, only primus is queried; github is skipped."""

    def fake_primus(repo_url, *, base_url, limit, state, label, timeout_sec):
        del repo_url, base_url, limit, state, label, timeout_sec
        return [GitHubPr(number=1, title="a", html_url="u1")]

    def boom_github(repo_url, *, gap_description, limit):  # noqa: ARG001
        raise AssertionError("github backend must not be called when include_github=False")

    monkeypatch.setattr(tools_api, "list_perf_prs", fake_primus)
    monkeypatch.setattr(tools_api.github_backend, "search_perf_prs", boom_github)

    out = tools_api.find_relevant_prs_smart(
        "x",
        repos=["https://github.com/sgl-project/sglang.git"],
        primus_cortex_url="http://primus",
        include_github=False,
    )
    refs = [c.ref for c in out]
    sources = {c.source for c in out}
    assert refs == ["PR:1"]
    assert sources == {"primus_cortex"}


def test_find_relevant_prs_smart_unions_primus_and_github(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both backends contribute; primus wins ties when dedup-by-(repo,ref)."""

    def fake_primus(repo_url, *, base_url, limit, state, label, timeout_sec):  # noqa: ARG001
        return [
            GitHubPr(number=1, title="a", html_url="u1"),
            GitHubPr(number=2, title="b", html_url="u2"),
        ]

    def fake_github(repo_url, *, gap_description, limit):  # noqa: ARG001
        return [
            GitHubPr(number=2, title="dup", html_url="dup"),
            GitHubPr(number=3, title="c", html_url="u3"),
        ]

    monkeypatch.setattr(tools_api, "list_perf_prs", fake_primus)
    monkeypatch.setattr(tools_api.github_backend, "search_perf_prs", fake_github)

    out = tools_api.find_relevant_prs_smart(
        "x",
        repos=["https://github.com/sgl-project/sglang.git"],
        primus_cortex_url="http://primus",
    )
    refs = [c.ref for c in out]
    by_ref = {c.ref: c.source for c in out}
    assert refs == ["PR:1", "PR:2", "PR:3"]
    assert by_ref["PR:2"] == "primus_cortex"
    assert by_ref["PR:3"] == "github"


def test_find_relevant_prs_smart_github_only_when_primus_url_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without primus_cortex_url + include_github=True: github is the only source."""

    def boom_primus(*a, **kw):
        raise AssertionError("primus must not be called without a URL")

    def fake_github(repo_url, *, gap_description, limit):  # noqa: ARG001
        return [GitHubPr(number=9, title="g", html_url="u9")]

    monkeypatch.setattr(tools_api, "list_perf_prs", boom_primus)
    monkeypatch.setattr(tools_api.github_backend, "search_perf_prs", fake_github)

    out = tools_api.find_relevant_prs_smart(
        "x",
        repos=["https://github.com/sgl-project/sglang.git"],
    )
    assert [c.source for c in out] == ["github"]
    assert [c.ref for c in out] == ["PR:9"]


# fetch_pr_audit_material -----------------------------------------------


def test_fetch_pr_audit_material_writes_two_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fetch_pr_audit_material writes pr.patches + pr_files.json."""

    def fake_patches(repo_slug, number, *, base_url, timeout_sec):  # noqa: ARG001
        return "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"

    def fake_files(repo_slug, number, *, base_url, timeout_sec):  # noqa: ARG001
        return [{"file_path": "x"}]

    monkeypatch.setattr(tools_api, "pr_patches", fake_patches)
    monkeypatch.setattr(tools_api, "pr_files", fake_files)

    out_dir = tmp_path / "audit"
    paths = tools_api.fetch_pr_audit_material(
        "https://github.com/sgl-project/sglang.git",
        42,
        out_dir=out_dir,
        primus_cortex_url="http://primus",
    )
    assert (out_dir / "pr.patches").is_file()
    assert (out_dir / "pr_files.json").is_file()
    payload = json.loads((out_dir / "pr_files.json").read_text(encoding="utf-8"))
    assert payload["number"] == 42
    assert payload["repo"] == "sgl-project/sglang"
    assert payload["files"] == [{"file_path": "x"}]
    assert paths["patches_path"].endswith("pr.patches")
    assert paths["files_json_path"].endswith("pr_files.json")


# evaluate_candidate_outcome -------------------------------------------


def test_evaluate_candidate_outcome_winner_pass() -> None:
    """Throughput >= ratio + accuracy drop <= max + completed parity -> winner."""
    out = tools_api.evaluate_candidate_outcome(
        {"throughput": 200.0, "completed": "1/1"},
        {"accuracy": 0.92},
        baseline_throughput=100.0,
        baseline_accuracy=0.9,
    )
    assert out["winner"] is True
    assert out["throughput_ratio"] == 2.0


def test_evaluate_candidate_outcome_throughput_too_low() -> None:
    """Low throughput is rejected with a ratio-shaped reason."""
    out = tools_api.evaluate_candidate_outcome(
        {"throughput": 104.0, "completed": "1/1"},
        {"accuracy": 0.9},
        baseline_throughput=100.0,
        baseline_accuracy=0.9,
    )
    assert out["winner"] is False
    assert "throughput ratio" in out["reason"]


def test_evaluate_candidate_outcome_invalid_baseline_raises() -> None:
    """Caller-side mistake of baseline_throughput<=0 must raise loudly."""
    with pytest.raises(ValueError, match="baseline_throughput"):
        tools_api.evaluate_candidate_outcome(
            {"throughput": 100.0},
            None,
            baseline_throughput=0.0,
        )


def test_evaluate_candidate_outcome_accepts_path(tmp_path: Path) -> None:
    """benchmark/accuracy may be passed as a Path; missing path -> miss-throughput."""
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"throughput": 250.0, "completed": "1/1"}), encoding="utf-8")
    out = tools_api.evaluate_candidate_outcome(
        p, None, baseline_throughput=100.0
    )
    assert out["winner"] is True
    out_missing = tools_api.evaluate_candidate_outcome(
        tmp_path / "nope.json", None, baseline_throughput=100.0
    )
    assert out_missing["winner"] is False
    assert "missing throughput" in out_missing["reason"]
