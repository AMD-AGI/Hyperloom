# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.agents.framework.sources.enumerate_candidates dispatch. Hermetic - monkeypatches the backend functions directly."""

from __future__ import annotations

import pytest

import hyperloom.agents.framework.sources as src
from hyperloom.agents.framework.keywords import extract_keywords
from hyperloom.agents.framework.models import CandidateSearchRequest, PRMonitorConfig
from hyperloom.agents.framework.sources._shared import GitHubPr

_PR_MONITOR = PRMonitorConfig(base_url="http://x")


def _minimal_request(**overrides) -> CandidateSearchRequest:
    """Build a minimal CandidateSearchRequest for dispatch tests."""
    return CandidateSearchRequest(**{"repo_url": "https://github.com/sgl-project/sglang.git", **overrides})


def _gap_keywords(gap: str) -> tuple[str, ...]:
    """Keywords the way the enablement mandate derives them from a gap."""
    return tuple(extract_keywords(gap))


# Per-framework repo URLs to parametrise dispatch tests over every framework.
_FRAMEWORK_TO_REPO_URL: dict[str, str] = {
    "sglang": "https://github.com/sgl-project/sglang.git",
    "vllm": "https://github.com/ROCm/vllm.git",
    "atom": "https://github.com/ROCm/ATOM.git",
}


def test_pr_states_defaults_to_open() -> None:
    req = _minimal_request()
    assert req.pr_states == ("open",)


def test_dispatch_unknown_search_mode_raises() -> None:
    """A search mode with no backend is a configuration error, not an empty result."""
    with pytest.raises(src.SourceConfigError, match="unknown search_mode"):
        src.enumerate_candidates(_minimal_request(search_modes=("gbrain_pr_kb",)))


def test_pr_monitor_search_state_broadens_with_pr_states(monkeypatch) -> None:
    """pr_states=all -> pr_monitor search queried with state='all'."""
    captured: dict[str, str] = {}

    def _fake_search(repo_url, *, base_url, query, limit, state, timeout_sec):
        captured["state"] = state
        return [GitHubPr(number=7, title="perf fastpath", html_url="https://github.com/x/y/pull/7")]

    monkeypatch.setattr(src, "search_perf_prs_via_pr_monitor_search", _fake_search)
    req = _minimal_request(
        keywords=_gap_keywords("speed up decode"),
        pr_states=("all",),
        pr_monitor=PRMonitorConfig(base_url="http://pr_monitor.local"),
    )
    out = src._run_pr_monitor(req)
    assert captured["state"] == "all"
    assert out and out[0].source == "pr_monitor"


def test_pr_monitor_search_state_open_only_default(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_search(repo_url, *, base_url, query, limit, state, timeout_sec):
        captured["state"] = state
        return []

    monkeypatch.setattr(src, "search_perf_prs_via_pr_monitor_search", _fake_search)
    monkeypatch.setattr(src, "list_perf_prs", lambda *a, **k: [])
    req = _minimal_request(
        keywords=_gap_keywords("speed up decode"),
        pr_monitor=PRMonitorConfig(base_url="http://pr_monitor.local"),
    )
    src._run_pr_monitor(req)
    assert captured["state"] == "open"


@pytest.mark.parametrize("framework", ["sglang", "vllm", "atom"])
def test_dispatch_pr_monitor_search_per_framework(framework: str, monkeypatch) -> None:
    """PR-search backends are framework-agnostic; the framework only determines which repo gets queried."""
    req = _minimal_request(
        repo_url=_FRAMEWORK_TO_REPO_URL[framework],
        search_modes=("pr_monitor",),
        max_search_candidates=2,
        pr_monitor=_PR_MONITOR,
    )

    seen_repo_urls: list[str] = []

    def fake_pr_monitor(repo_url, *, base_url, limit, timeout_sec, state=None):
        seen_repo_urls.append(repo_url)
        return [
            GitHubPr(number=11, title=f"{framework}-pr-1", html_url="u1"),
        ]

    monkeypatch.setattr(src, "list_perf_prs", fake_pr_monitor)

    out = src.enumerate_candidates(req)
    assert seen_repo_urls == [_FRAMEWORK_TO_REPO_URL[framework]]
    assert any(c.source == "pr_monitor" for c in out)


def test_dispatch_pr_monitor_missing_remote_config_raises(monkeypatch) -> None:
    """Remote Recipe mode does not synthesize a missing KB Service URL."""
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.delenv("KB_STORE_URL", raising=False)
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

    def fake_pr_monitor(repo_url, *, base_url, limit, timeout_sec, state=None):
        return [
            GitHubPr(number=1, title="a", html_url="u1"),
            GitHubPr(number=2, title="b", html_url="u2"),
        ]

    def fake_github(repo_url, *, limit, states=("open",)):
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
    by_ref = {c.ref: c.source for c in out}
    assert by_ref["PR:2"] == "pr_monitor"
    assert by_ref["PR:3"] == "github"


def test_pr_monitor_uses_search_endpoint_when_keywords_present(monkeypatch) -> None:
    """When the request carries keywords, dispatcher uses /v1/search/prs."""
    captured: dict[str, object] = {}

    def fake_search(repo_url, *, base_url, query, limit, state, timeout_sec):
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

    def fake_list(repo_url, *, base_url, limit, timeout_sec, state=None):
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

    def fake_list(repo_url, *, base_url, limit, timeout_sec, state=None):
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

    def fake_list(repo_url, *, base_url, limit, timeout_sec, state=None):
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

    def fake_search(repo_url, *, base_url, query, limit, state, timeout_sec):
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


def test_rank_by_keyword_overlap_empty_keywords_is_identity() -> None:
    """An empty keyword list returns the input list unchanged."""
    prs = [
        GitHubPr(number=1, title="a", html_url="u1"),
        GitHubPr(number=2, title="b", html_url="u2"),
    ]
    out = src._rank_by_keyword_overlap(prs, [])
    assert out == prs


def test_pr25769_megamoe_demoted_at_dispatcher_for_dense_mi300x_gap(monkeypatch) -> None:
    """A dense+mi300x PR must rank ahead of PR:25769 MegaMoE at the enumerate_candidates boundary."""

    def fake_search(repo_url, *, base_url, query, limit, state, timeout_sec):
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


def test_candidate_score_field_populated_for_pr_monitor_path(monkeypatch) -> None:
    """The dispatcher transports the rerank score on every pr_monitor Candidate; order is score-descending, stable on ties."""

    def fake_search(repo_url, *, base_url, query, limit, state, timeout_sec):
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
    scores = [(c.ref, c.score) for c in out]
    # PR:10 best (positive hits, no anti); PR:12 worst (0).
    assert scores[0][0] == "PR:10"
    assert scores[0][1] > 0.0, f"top candidate must carry a positive score; got {scores}"
    # PR:11 MegaMoE has anti hits so its score should be <= top candidate.
    assert scores[1][1] <= scores[0][1]
    # Sort order must match score desc.
    assert scores == sorted(scores, key=lambda x: -x[1])


def test_candidate_score_defaults_to_zero_for_list_only_path(monkeypatch) -> None:
    """No keywords -> cheap listing path; Candidate.score defaults to 0.0 (no ranking happened)."""

    def fake_search(*a, **kw):  # would never be called when keywords empty
        raise AssertionError("search must not be called on the no-keyword path")

    def fake_list(repo_url, *, base_url, limit, timeout_sec, state=None):
        return [GitHubPr(number=30, title="generic PR", html_url="u30")]

    monkeypatch.setattr(src, "search_perf_prs_via_pr_monitor_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", fake_list)

    req = _minimal_request(
        search_modes=("pr_monitor",),
        max_search_candidates=1,
        pr_monitor=_PR_MONITOR,
    )
    out = src.enumerate_candidates(req)
    assert len(out) == 1
    assert out[0].score == 0.0
    assert out[0].ref == "PR:30"


def test_anti_signal_inactive_at_dispatcher_when_no_trigger_in_gap(monkeypatch) -> None:
    """Anti rerank is a no-op when the gap carries no anti-trigger keyword."""

    def fake_search(repo_url, *, base_url, query, limit, state, timeout_sec):
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
