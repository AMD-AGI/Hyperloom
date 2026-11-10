# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.agents.framework.sources.enumerate_candidates dispatch. Hermetic - monkeypatches the backend functions directly."""

from __future__ import annotations

import pytest

import hyperloom.agents.framework.sources as src
from hyperloom.agents.framework.keywords import extract_keywords
from hyperloom.agents.framework.models import CandidateSearchRequest, PRMonitorConfig
from hyperloom.agents.framework.sources._shared import GitHubPr
from hyperloom.inference_optimizer import framework_registry

_PR_MONITOR = PRMonitorConfig(base_url="http://x")


def _minimal_request(**overrides) -> CandidateSearchRequest:
    """Build a minimal CandidateSearchRequest for dispatch tests."""
    return CandidateSearchRequest(**{"repo_url": "https://github.com/sgl-project/sglang.git", **overrides})


def _gap_keywords(gap: str) -> tuple[str, ...]:
    """Keywords the way the enablement mandate derives them from a gap."""
    return tuple(extract_keywords(gap))


def test_dispatch_unknown_search_mode_raises() -> None:
    """A search mode with no backend is a configuration error, not an empty result."""
    with pytest.raises(src.SourceConfigError, match="unknown search_mode"):
        src.enumerate_candidates(_minimal_request(search_modes=("gbrain_pr_kb",)))


@pytest.mark.parametrize("framework", ["sglang", "vllm", "atom"])
def test_dispatch_pr_monitor_search_per_framework(framework: str, monkeypatch) -> None:
    """PR-search backends are framework-agnostic; the framework only determines which repo gets queried."""
    req = _minimal_request(
        repo_url=framework_registry.repo_url(framework),
        search_modes=("pr_monitor",),
        max_search_candidates=2,
        pr_monitor=_PR_MONITOR,
    )

    seen_repo_urls: list[str] = []

    def fake_pr_monitor(repo_url, *, monitor, limit):
        seen_repo_urls.append(repo_url)
        return [
            GitHubPr(number=11, title=f"{framework}-pr-1", html_url="u1"),
        ]

    monkeypatch.setattr(src, "list_perf_prs", fake_pr_monitor)

    out = src.enumerate_candidates(req)
    assert seen_repo_urls == [framework_registry.repo_url(framework)]
    assert [c.ref for c in out] == ["PR:11"]


def test_dispatch_pr_monitor_without_config_raises() -> None:
    """Asking for pr_monitor without its config is a configuration error, not an empty result."""
    req = _minimal_request(search_modes=("pr_monitor",))
    with pytest.raises(src.SourceConfigError, match="pr_monitor"):
        src.enumerate_candidates(req)


def test_dispatch_unions_pr_monitor_and_github(monkeypatch) -> None:
    """Both backends contribute; duplicates de-duped by ref."""
    req = _minimal_request(
        search_modes=("pr_monitor", "github"),
        max_search_candidates=3,
        pr_monitor=_PR_MONITOR,
    )

    def fake_pr_monitor(repo_url, *, monitor, limit):
        return [
            GitHubPr(number=1, title="a", html_url="u1"),
            GitHubPr(number=2, title="b", html_url="u2"),
        ]

    def fake_github(repo_url, *, limit, terms=()):
        return [
            GitHubPr(number=2, title="dup", html_url="dup"),
            GitHubPr(number=3, title="c", html_url="u3"),
        ]

    monkeypatch.setattr(src, "list_perf_prs", fake_pr_monitor)
    monkeypatch.setattr(src.github_backend, "search_perf_prs", fake_github)

    out = src.enumerate_candidates(req)
    refs = [c.ref for c in out]
    # pr_monitor first, then github (dedup keeps first occurrence)
    assert refs == ["PR:1", "PR:2", "PR:3"]
    assert out[1].title == "b"


def test_github_searches_for_the_request_keywords(monkeypatch) -> None:
    """An enablement request names the failure, so the GitHub query must carry it."""
    captured: dict[str, object] = {}

    def fake_github(repo_url, *, limit, terms=()):
        captured["terms"] = terms
        return []

    monkeypatch.setattr(src.github_backend, "search_perf_prs", fake_github)
    src.enumerate_candidates(
        _minimal_request(search_modes=("github",), keywords=("deepseekv4", "causallm", "architectures"))
    )
    assert captured["terms"] == ("deepseekv4", "causallm", "architectures")


def test_github_backend_failure_reaches_the_caller(monkeypatch) -> None:
    """A github backend bug is not turned into an empty result."""

    def broken_github(repo_url, *, limit, terms=()):
        raise RuntimeError("github backend bug")

    monkeypatch.setattr(src.github_backend, "search_perf_prs", broken_github)
    with pytest.raises(RuntimeError, match="github backend bug"):
        src.enumerate_candidates(_minimal_request(search_modes=("github",)))


def test_github_without_keywords_searches_the_perf_terms(monkeypatch) -> None:
    """Perf discovery names no failure and keeps the perf term set."""
    captured: dict[str, object] = {}

    def fake_github(repo_url, *, limit, terms=()):
        captured["terms"] = terms
        return []

    monkeypatch.setattr(src.github_backend, "search_perf_prs", fake_github)
    src.enumerate_candidates(_minimal_request(search_modes=("github",)))
    assert captured["terms"] == src.github_backend.PERF_TERMS


def test_pr_monitor_uses_search_endpoint_when_keywords_present(monkeypatch) -> None:
    """When the request carries keywords, dispatcher uses /v1/search/prs."""
    captured: dict[str, object] = {}

    def fake_search(repo_url, *, monitor, query, limit):
        captured["called"] = "search"
        captured["query"] = query
        captured["limit"] = limit
        return [
            GitHubPr(number=10, title="MoE fp8 perf improvement", html_url="u10"),
            GitHubPr(number=11, title="random doc edit", html_url="u11"),
            GitHubPr(number=12, title="fp8 attention fusion", html_url="u12"),
        ]

    def fake_list(*_a, **_kw):
        captured["called"] = "list"
        return []

    monkeypatch.setattr(src, "search_perf_prs_via_pr_monitor_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", fake_list)

    req = _minimal_request(
        search_modes=("pr_monitor",),
        max_search_candidates=2,
        pr_monitor=_PR_MONITOR,
        keywords=_gap_keywords("improve sglang fp8 MoE on MI300X"),
    )
    out = src.enumerate_candidates(req)
    assert captured["called"] == "search", "search endpoint must be preferred when keywords present"
    # over-fetch = 3 * max_search_candidates = 6
    assert captured["limit"] == 6
    # Rerank: MoE+fp8 title first, fp8 second, doc edit last; trimmed to limit=2
    refs = [c.ref for c in out]
    assert refs[0] == "PR:10"
    assert refs[1] == "PR:12"
    assert len(refs) == 2


def test_pr_monitor_falls_back_to_list_when_search_returns_empty(monkeypatch) -> None:
    """When /v1/search/prs returns 0 candidates, fall back to list_perf_prs + client-side rerank rather than failing the run."""
    calls: list[str] = []

    def fake_search(*_a, **_kw):
        calls.append("search")
        return []

    def fake_list(repo_url, *, monitor, limit):
        calls.append("list")
        return [
            GitHubPr(number=40, title="NPU Ascend backend", html_url="u40"),
            GitHubPr(number=41, title="fp8 MoE quant", html_url="u41"),
        ]

    monkeypatch.setattr(src, "search_perf_prs_via_pr_monitor_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", fake_list)

    req = _minimal_request(
        search_modes=("pr_monitor",),
        max_search_candidates=1,
        pr_monitor=_PR_MONITOR,
        keywords=_gap_keywords("improve sglang fp8 MoE on MI300X throughput"),
    )
    out = src.enumerate_candidates(req)
    assert calls == ["search", "list"], "must try search first, then fall back to list"
    refs = [c.ref for c in out]
    assert refs == ["PR:41"], "rerank picks the fp8/MoE PR over the NPU one"


def test_pr_monitor_falls_back_to_list_when_search_unavailable(monkeypatch) -> None:
    """If the search endpoint raises PRMonitorError, fall back to list_perf_prs."""
    captured: dict[str, object] = {}

    def fake_search(*_a, **_kw):
        raise src.PRMonitorError("404 Not Found at /v1/search/prs")

    def fake_list(repo_url, *, monitor, limit):
        captured["called"] = "list"
        captured["limit"] = limit
        return [
            GitHubPr(number=20, title="NPU Ascend backend", html_url="u20"),
            GitHubPr(number=21, title="fp8 MoE quant", html_url="u21"),
        ]

    monkeypatch.setattr(src, "search_perf_prs_via_pr_monitor_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", fake_list)

    req = _minimal_request(
        search_modes=("pr_monitor",),
        max_search_candidates=2,
        pr_monitor=_PR_MONITOR,
        keywords=_gap_keywords("improve sglang fp8 MoE on MI300X"),
    )
    out = src.enumerate_candidates(req)
    assert captured["called"] == "list", "must fall back to list_perf_prs when search fails"
    # Fallback over-fetch still uses 3x
    assert captured["limit"] == 6
    refs = [c.ref for c in out]
    assert refs[0] == "PR:21"


def test_pr_monitor_no_keywords_uses_list_only_path(monkeypatch) -> None:
    """Without keywords, dispatcher uses the cheap listing endpoint."""
    captured: dict[str, object] = {}

    def fake_search(*_a, **_kw):
        captured["called"] = "search"
        return []

    def fake_list(repo_url, *, monitor, limit):
        captured["called"] = "list"
        captured["limit"] = limit
        return [
            GitHubPr(number=30, title="generic PR", html_url="u30"),
        ]

    monkeypatch.setattr(src, "search_perf_prs_via_pr_monitor_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", fake_list)

    req = _minimal_request(
        search_modes=("pr_monitor",),
        max_search_candidates=1,
        pr_monitor=_PR_MONITOR,
    )
    out = src.enumerate_candidates(req)
    assert captured["called"] == "list"
    # No over-fetch without keywords
    assert captured["limit"] == 1
    assert [c.ref for c in out] == ["PR:30"]


def test_rank_by_keyword_overlap_preserves_ties() -> None:
    """Ties in score preserve the upstream order (stable sort)."""
    prs = [
        GitHubPr(number=1, title="fp8 moe a", html_url="u1"),
        GitHubPr(number=2, title="fp8 moe b", html_url="u2"),
        GitHubPr(number=3, title="unrelated", html_url="u3"),
    ]
    out = src._rank_by_keyword_overlap(prs, ["fp8", "moe"])
    assert [pr.number for pr in out] == [1, 2, 3]


def test_resolve_keywords_empty_returns_empty() -> None:
    """No keywords -> empty list (cheapest path)."""
    assert src._resolve_keywords(_minimal_request(keywords=())) == []


def test_resolve_keywords_lowercases_explicit() -> None:
    """Keywords are lowercased to match service token shape."""
    req = _minimal_request(keywords=("MI300X", "FP8"))
    assert src._resolve_keywords(req) == ["mi300x", "fp8"]


def test_pr_monitor_uses_explicit_keywords(monkeypatch) -> None:
    """End-to-end: request keywords are sent as the PR Monitor query verbatim."""
    captured: dict[str, object] = {}

    def fake_search(repo_url, *, monitor, query, limit):
        captured["query"] = query
        return [GitHubPr(number=99, title="mi300x perf PR", html_url="u")]

    def fake_list(*_a, **_kw):
        captured["list_called"] = True
        return []

    monkeypatch.setattr(src, "search_perf_prs_via_pr_monitor_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", fake_list)

    req = _minimal_request(
        search_modes=("pr_monitor",),
        max_search_candidates=1,
        pr_monitor=_PR_MONITOR,
        keywords=("mi300x",),
    )
    out = src.enumerate_candidates(req)
    assert captured["query"] == "mi300x", "service query must be the explicit keyword"
    assert "list_called" not in captured, "non-empty search result should not trigger fallback"
    assert [c.ref for c in out] == ["PR:99"]


def test_pr25769_megamoe_demoted_at_dispatcher_for_dense_mi300x_gap(monkeypatch) -> None:
    """A dense+mi300x PR must rank ahead of PR:25769 MegaMoE at the enumerate_candidates boundary."""

    def fake_search(repo_url, *, monitor, query, limit):
        return [
            GitHubPr(
                number=25769,
                title="Enable MegaMoE for NextN with TP attn A2A scatter padding",
                html_url="u25769",
            ),
            GitHubPr(
                number=99999,
                title="optimize sglang bf16 attention prefill on mi300x throughput",
                html_url="u99999",
            ),
        ]

    monkeypatch.setattr(src, "search_perf_prs_via_pr_monitor_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", lambda *a, **kw: [])

    req = _minimal_request(
        search_modes=("pr_monitor",),
        max_search_candidates=2,
        pr_monitor=_PR_MONITOR,
        keywords=_gap_keywords("improve sglang bf16 dense throughput on mi300x"),
    )
    out = src.enumerate_candidates(req)
    refs = [c.ref for c in out]
    assert refs[0] == "PR:99999", (
        f"dense+mi300x+bf16 gap must promote relevant PR over PR:25769 MegaMoE PR; got order={refs}"
    )
    # The MegaMoE PR is not filtered, just demoted.
    assert "PR:25769" in refs


def test_pr_monitor_path_orders_candidates_by_keyword_score(monkeypatch) -> None:
    """pr_monitor candidates come back in rerank-score order: the dense+mi300x PR first, the doc edit last."""

    def fake_search(repo_url, *, monitor, query, limit):
        return [
            GitHubPr(number=10, title="optimize sglang bf16 dense attention on mi300x", html_url="u10"),
            GitHubPr(number=11, title="MegaMoE NextN A2A", html_url="u11"),
            GitHubPr(number=12, title="random doc edit", html_url="u12"),
        ]

    monkeypatch.setattr(src, "search_perf_prs_via_pr_monitor_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", lambda *a, **kw: [])

    req = _minimal_request(
        search_modes=("pr_monitor",),
        max_search_candidates=3,
        pr_monitor=_PR_MONITOR,
        keywords=_gap_keywords("improve sglang bf16 dense throughput on mi300x"),
    )
    out = src.enumerate_candidates(req)
    refs = [c.ref for c in out]
    assert refs[0] == "PR:10"
    assert refs[-1] == "PR:12"


def test_pr_monitor_no_keywords_never_calls_search(monkeypatch) -> None:
    """No keywords -> cheap listing path; the search endpoint is never hit."""

    def fake_search(*a, **kw):  # would never be called when keywords empty
        raise AssertionError("search must not be called on the no-keyword path")

    def fake_list(repo_url, *, monitor, limit):
        return [GitHubPr(number=30, title="generic PR", html_url="u30")]

    monkeypatch.setattr(src, "search_perf_prs_via_pr_monitor_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", fake_list)

    req = _minimal_request(
        search_modes=("pr_monitor",),
        max_search_candidates=1,
        pr_monitor=_PR_MONITOR,
    )
    out = src.enumerate_candidates(req)
    assert [c.ref for c in out] == ["PR:30"]


def test_anti_signal_inactive_at_dispatcher_when_no_trigger_in_gap(monkeypatch) -> None:
    """Anti rerank is a no-op when the gap carries no anti-trigger keyword."""

    def fake_search(repo_url, *, monitor, query, limit):
        return [
            GitHubPr(number=10, title="fp8 moe perf improvement", html_url="u10"),
            GitHubPr(number=11, title="fp8 attention fusion", html_url="u11"),
            GitHubPr(number=12, title="random doc edit", html_url="u12"),
        ]

    monkeypatch.setattr(src, "search_perf_prs_via_pr_monitor_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", lambda *a, **kw: [])

    # Gap with NO anti-trigger; extract_keywords -> ['attention', 'fp8'].
    req = _minimal_request(
        search_modes=("pr_monitor",),
        max_search_candidates=3,
        pr_monitor=_PR_MONITOR,
        keywords=_gap_keywords("improve fp8 attention"),
    )
    out = src.enumerate_candidates(req)
    refs = [c.ref for c in out]
    # PR:10 keeps its positive overlap despite containing ``moe``; anti is gated on the gap-side trigger.
    assert refs == ["PR:11", "PR:10", "PR:12"]
