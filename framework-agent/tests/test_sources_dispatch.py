"""Tests for framework_agent.sources.enumerate_candidates dispatch.

Hermetic - monkeypatches the backend functions directly.
"""

from __future__ import annotations

import pytest

import framework_agent.sources as src
from framework_agent.models import (
    Baseline,
    ExploreRequest,
    PrimusCortexConfig,
)
from framework_agent.sources._shared import GitHubPr


def _minimal_request(**overrides) -> ExploreRequest:
    """Build a minimal valid ExploreRequest for dispatch tests."""
    base = {
        "framework": "sglang",
        "repo_url": "https://github.com/sgl-project/sglang.git",
        "work_dir": "/tmp/x",
        "baseline": {"throughput": 1.0},
    }
    base.update(overrides)
    return ExploreRequest.from_dict(base)


def test_dispatch_explicit_refs_only() -> None:
    """Without search_perf_prs, only explicit candidate_refs are returned."""
    req = _minimal_request(candidate_refs=["main", "PR:1"], search_perf_prs=False)
    out = src.enumerate_candidates(req)
    refs = [c.ref for c in out]
    sources = {c.source for c in out}
    assert refs == ["main", "PR:1"]
    assert sources == {"explicit"}


def test_dispatch_primus_missing_config_raises() -> None:
    """search_modes=['primus_cortex'] without config raises SourceConfigError."""
    req = _minimal_request(
        search_perf_prs=True,
        search_modes=["primus_cortex"],
    )
    with pytest.raises(src.SourceConfigError, match="primus_cortex"):
        src.enumerate_candidates(req)


def test_dispatch_unions_primus_and_github(monkeypatch) -> None:
    """Both backends contribute; duplicates de-duped by ref."""
    req = _minimal_request(
        search_perf_prs=True,
        search_modes=["primus_cortex", "github"],
        max_search_candidates=3,
        primus_cortex={"base_url": "http://x"},
        candidate_refs=["main"],
    )

    def fake_primus(repo_url, *, base_url, limit, label=None, timeout_sec):  # noqa: ARG001
        return [
            GitHubPr(number=1, title="a", html_url="u1"),
            GitHubPr(number=2, title="b", html_url="u2"),
        ]

    def fake_github(repo_url, *, gap_description, limit):  # noqa: ARG001
        return [
            GitHubPr(number=2, title="dup", html_url="dup"),  # dup with primus
            GitHubPr(number=3, title="c", html_url="u3"),
        ]

    monkeypatch.setattr(src, "list_perf_prs", fake_primus)
    monkeypatch.setattr(src.github_backend, "search_perf_prs", fake_github)

    out = src.enumerate_candidates(req)
    refs = [c.ref for c in out]
    # explicit first, then primus, then github (dedup keeps first occurrence)
    assert refs == ["main", "PR:1", "PR:2", "PR:3"]
    # dedup preserves primus' source for PR:2 (first seen)
    by_ref = {c.ref: c.source for c in out}
    assert by_ref["PR:2"] == "primus_cortex"
    assert by_ref["PR:3"] == "github"
