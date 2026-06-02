"""Tests for framework_agent.sources.enumerate_candidates dispatch.

Hermetic - monkeypatches the backend functions directly.
"""

from __future__ import annotations

import pytest

import framework_agent.sources as src
from framework_agent.models import ExploreRequest
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


# Per-framework repo URLs used to parametrise the dispatch tests over
# every framework the framework-agent CLI accepts. This map plus the
# parametrize decorator below ensure every dispatch path is exercised
# for sglang AND vllm AND atom so a
# future regression that silently routes only one framework correctly
# fails the suite.
_FRAMEWORK_TO_REPO_URL: dict[str, str] = {
    "sglang": "https://github.com/sgl-project/sglang.git",
    "vllm":   "https://github.com/ROCm/vllm.git",
    "atom":   "https://github.com/ROCm/ATOM.git",
}


def test_dispatch_explicit_refs_only() -> None:
    """Without search_perf_prs, only explicit candidate_refs are returned."""
    req = _minimal_request(candidate_refs=["main", "PR:1"], search_perf_prs=False)
    out = src.enumerate_candidates(req)
    refs = [c.ref for c in out]
    sources = {c.source for c in out}
    assert refs == ["main", "PR:1"]
    assert sources == {"explicit"}


# Parametrise the dispatch path over every framework the CLI accepts
# so a future regression that silently routes only one framework
# correctly fails the suite. The
# parametrised guard below covers sglang / vllm / atom.
@pytest.mark.parametrize("framework", ["sglang", "vllm", "atom"])
def test_dispatch_explicit_refs_only_across_frameworks(framework: str) -> None:
    """Explicit candidate_refs must come out untouched for every
    framework — there's no framework-specific filtering at the
    dispatch layer.
    """
    req = _minimal_request(
        framework=framework,
        repo_url=_FRAMEWORK_TO_REPO_URL[framework],
        candidate_refs=["main", "PR:1"],
        search_perf_prs=False,
    )
    out = src.enumerate_candidates(req)
    assert [c.ref for c in out] == ["main", "PR:1"]
    assert {c.source for c in out} == {"explicit"}


@pytest.mark.parametrize("framework", ["sglang", "vllm", "atom"])
def test_dispatch_primus_search_per_framework(framework: str, monkeypatch) -> None:
    """PR-search backends are framework-agnostic — the framework only
    determines which repo gets queried. atom must be dispatched the
    same way sglang/vllm are.
    """
    req = _minimal_request(
        framework=framework,
        repo_url=_FRAMEWORK_TO_REPO_URL[framework],
        search_perf_prs=True,
        search_modes=["primus_cortex"],
        max_search_candidates=2,
        primus_cortex={"base_url": "http://x"},
    )

    seen_repo_urls: list[str] = []

    def fake_primus(repo_url, *, base_url, limit, label=None, timeout_sec):  # noqa: ARG001
        seen_repo_urls.append(repo_url)
        return [
            GitHubPr(number=11, title=f"{framework}-pr-1", html_url="u1"),
        ]

    monkeypatch.setattr(src, "list_perf_prs", fake_primus)

    out = src.enumerate_candidates(req)
    # The primus backend was called against the framework's own repo URL.
    assert seen_repo_urls == [_FRAMEWORK_TO_REPO_URL[framework]]
    assert any(c.source == "primus_cortex" for c in out)


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


# ---------------------------------------------------------------------------
# B2 fix: gap-aware primus_cortex routing + client-side rerank
# ---------------------------------------------------------------------------


def test_primus_uses_search_endpoint_when_gap_present(monkeypatch) -> None:
    """When gap_description yields keywords, dispatcher uses /v1/search/prs."""
    captured: dict[str, object] = {}

    def fake_search(repo_url, *, base_url, query, limit, state, timeout_sec):  # noqa: ARG001
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

    monkeypatch.setattr(src, "search_perf_prs_via_primus_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", fake_list)

    req = _minimal_request(
        search_perf_prs=True,
        search_modes=["primus_cortex"],
        max_search_candidates=2,
        primus_cortex={"base_url": "http://x"},
        gap_description="improve sglang fp8 MoE on MI300X",
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


def test_primus_falls_back_to_list_when_search_returns_empty(monkeypatch) -> None:
    """B2 v2: when /v1/search/prs returns 0 candidates (word-AND match too tight),
    fall back to list_perf_prs + client-side rerank rather than failing the run."""
    calls: list[str] = []

    def fake_search(*_a, **_kw):
        calls.append("search")
        return []  # service returns empty - the real-world failure mode

    def fake_list(repo_url, *, base_url, limit, label=None, timeout_sec):  # noqa: ARG001
        calls.append("list")
        return [
            GitHubPr(number=40, title="NPU Ascend backend", html_url="u40"),
            GitHubPr(number=41, title="fp8 MoE quant", html_url="u41"),
        ]

    monkeypatch.setattr(src, "search_perf_prs_via_primus_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", fake_list)

    req = _minimal_request(
        search_perf_prs=True,
        search_modes=["primus_cortex"],
        max_search_candidates=1,
        primus_cortex={"base_url": "http://x"},
        gap_description="improve sglang fp8 MoE on MI300X throughput",
    )
    out = src.enumerate_candidates(req)
    assert calls == ["search", "list"], "must try search first, then fall back to list"
    refs = [c.ref for c in out]
    assert refs == ["PR:41"], "rerank picks the fp8/MoE PR over the NPU one"


def test_primus_falls_back_to_list_when_search_unavailable(monkeypatch) -> None:
    """If the search endpoint raises PrimusCortexError, fall back to list_perf_prs."""
    captured: dict[str, object] = {}

    def fake_search(*_a, **_kw):
        raise src.PrimusCortexError("404 Not Found at /v1/search/prs")

    def fake_list(repo_url, *, base_url, limit, label=None, timeout_sec):  # noqa: ARG001
        captured["called"] = "list"
        captured["limit"] = limit
        return [
            GitHubPr(number=20, title="NPU Ascend backend", html_url="u20"),
            GitHubPr(number=21, title="fp8 MoE quant", html_url="u21"),
        ]

    monkeypatch.setattr(src, "search_perf_prs_via_primus_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", fake_list)

    req = _minimal_request(
        search_perf_prs=True,
        search_modes=["primus_cortex"],
        max_search_candidates=2,
        primus_cortex={"base_url": "http://x"},
        gap_description="improve sglang fp8 MoE on MI300X",
    )
    out = src.enumerate_candidates(req)
    assert captured["called"] == "list", "must fall back to list_perf_prs when search fails"
    # Fallback over-fetch still uses 3x
    assert captured["limit"] == 6
    refs = [c.ref for c in out]
    # Rerank still applied to the list_perf_prs payload: fp8/MoE first
    assert refs[0] == "PR:21"


def test_primus_no_gap_uses_label_only_path(monkeypatch) -> None:
    """When gap_description is empty, dispatcher uses the cheap label-only listing."""
    captured: dict[str, object] = {}

    def fake_search(*_a, **_kw):
        captured["called"] = "search"
        return []

    def fake_list(repo_url, *, base_url, limit, label=None, timeout_sec):  # noqa: ARG001
        captured["called"] = "list"
        captured["limit"] = limit
        return [
            GitHubPr(number=30, title="generic PR", html_url="u30"),
        ]

    monkeypatch.setattr(src, "search_perf_prs_via_primus_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", fake_list)

    req = _minimal_request(
        search_perf_prs=True,
        search_modes=["primus_cortex"],
        max_search_candidates=1,
        primus_cortex={"base_url": "http://x"},
        # gap_description omitted (defaults to empty)
    )
    out = src.enumerate_candidates(req)
    assert captured["called"] == "list"
    # No over-fetch when gap is empty (preserves old cheap path)
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


# ---------------------------------------------------------------------------
# C: explicit keyword override
# ---------------------------------------------------------------------------


def test_resolve_keywords_explicit_overrides_gap() -> None:
    """request.keywords (non-empty) wins over extract_keywords(gap_description)."""
    req = _minimal_request(
        gap_description="improve sglang fp8 MoE on MI300X",  # auto would yield ['fp8','moe','sglang']
        keywords=["mi300x"],                                  # but explicit wins
    )
    assert src._resolve_keywords(req) == ["mi300x"]


def test_resolve_keywords_fallback_to_gap_extract() -> None:
    """Empty request.keywords + non-empty gap -> auto-extract."""
    req = _minimal_request(
        gap_description="improve sglang fp8 MoE",
        keywords=[],
    )
    out = src._resolve_keywords(req)
    # extract_keywords whitelist hits sglang/fp8/moe; order = sorted
    assert "fp8" in out and "moe" in out and "sglang" in out


def test_resolve_keywords_both_empty_returns_empty() -> None:
    """Neither keywords nor gap -> empty list (cheapest path)."""
    req = _minimal_request(keywords=[], gap_description="")
    assert src._resolve_keywords(req) == []


def test_resolve_keywords_lowercases_explicit() -> None:
    """Explicit override is lowercased to match service token shape."""
    req = _minimal_request(keywords=["MI300X", "FP8"])
    assert src._resolve_keywords(req) == ["mi300x", "fp8"]


def test_primus_uses_explicit_keywords(monkeypatch) -> None:
    """End-to-end: --framework-keywords sent as Primus query verbatim,
    bypassing the gap_description auto-extract."""
    captured: dict[str, object] = {}

    def fake_search(repo_url, *, base_url, query, limit, state, timeout_sec):  # noqa: ARG001
        captured["query"] = query
        return [GitHubPr(number=99, title="mi300x perf PR", html_url="u")]

    def fake_list(*_a, **_kw):
        captured["list_called"] = True
        return []

    monkeypatch.setattr(src, "search_perf_prs_via_primus_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", fake_list)

    req = _minimal_request(
        search_perf_prs=True,
        search_modes=["primus_cortex"],
        max_search_candidates=1,
        primus_cortex={"base_url": "http://x"},
        gap_description="improve sglang fp8 MoE",   # auto would be 'fp8 moe sglang'
        keywords=["mi300x"],                         # but explicit wins
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


# ---------------------------------------------------------------------------
# B3 fix: anti-correlation rerank lands at the dispatcher boundary.
#
# These tests pin the end-to-end behaviour through enumerate_candidates so a
# regression in _rank_by_keyword_overlap or the keywords.py scorer surfaces
# in the dispatcher's contract test, not just in the keywords unit tests.
# ---------------------------------------------------------------------------


def test_pr25769_megamoe_demoted_at_dispatcher_for_dense_mi300x_gap(monkeypatch) -> None:
    """Regression for session f219629b on Qwen-Qwen3-32B (dense, bf16, mi300x).

    Before the anti-correlation fix, fa picked PR:25769 ("Enable MegaMoE for
    NextN with TP attn A2A scatter padding") because positive-only overlap
    matched ``throughput`` while the PR's MoE / NVIDIA-H20 signals were
    invisible to the ranker. With the fix, a relevant dense+mi300x PR must
    rank ahead at the enumerate_candidates boundary.
    """

    def fake_search(repo_url, *, base_url, query, limit, state, timeout_sec):  # noqa: ARG001
        # Return the real PR:25769 title alongside a hypothetical relevant PR.
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

    monkeypatch.setattr(src, "search_perf_prs_via_primus_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", lambda *a, **kw: [])

    req = _minimal_request(
        search_perf_prs=True,
        search_modes=["primus_cortex"],
        max_search_candidates=2,
        primus_cortex={"base_url": "http://x"},
        gap_description="improve sglang bf16 dense throughput on mi300x",
    )
    out = src.enumerate_candidates(req)
    refs = [c.ref for c in out]
    assert refs[0] == "PR:99999", (
        f"dense+mi300x+bf16 gap must promote relevant PR over PR:25769 MegaMoE PR; "
        f"got order={refs}"
    )
    # The MegaMoE PR is not filtered, just demoted.
    assert "PR:25769" in refs


# ---------------------------------------------------------------------------
# Candidate.score must carry the rerank score across the subprocess JSON
# boundary so the IO framework_pr arm can log it.
# ---------------------------------------------------------------------------


def test_candidate_score_field_populated_for_primus_path(monkeypatch) -> None:
    """The dispatcher must transport the anti-aware rerank score on every
    Candidate it returns from the primus_cortex path with non-empty keywords.

    This is the data plane that lets the downstream framework_pr arm
    (inference_optimizer) persist "why we picked this PR" into state.json
    without re-running the scorer. Order of candidates must match score
    descending (stable on ties).
    """

    def fake_search(repo_url, *, base_url, query, limit, state, timeout_sec):  # noqa: ARG001
        return [
            GitHubPr(number=10, title="optimize sglang bf16 dense attention on mi300x", html_url="u10"),
            GitHubPr(number=11, title="MegaMoE NextN A2A", html_url="u11"),
            GitHubPr(number=12, title="random doc edit", html_url="u12"),
        ]

    monkeypatch.setattr(src, "search_perf_prs_via_primus_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", lambda *a, **kw: [])

    req = _minimal_request(
        search_perf_prs=True,
        search_modes=["primus_cortex"],
        max_search_candidates=3,
        primus_cortex={"base_url": "http://x"},
        gap_description="improve sglang bf16 dense throughput on mi300x",
    )
    out = src.enumerate_candidates(req)
    scores = [(c.ref, c.score) for c in out]
    # PR:10 best (dense+sglang+bf16+mi300x positive hits, no anti); PR:12 worst (0).
    assert scores[0][0] == "PR:10"
    assert scores[0][1] > 0.0, f"top candidate must carry a positive score; got {scores}"
    # PR:11 MegaMoE has anti hits (dense trigger fires moe penalty) so its score
    # should be <= top candidate.
    assert scores[1][1] <= scores[0][1]
    # Sort order must match score desc (already stable from _rank_by_keyword_overlap).
    assert scores == sorted(scores, key=lambda x: -x[1])


def test_candidate_score_defaults_to_zero_for_label_only_path(monkeypatch) -> None:
    """When gap_description / keywords are empty, the dispatcher takes the
    label-only cheap path. Candidate.score must default to 0.0 then — the
    field is opt-in semantics: 0.0 means "no gap-driven ranking happened".
    """

    def fake_search(*a, **kw):  # would never be called when keywords empty
        raise AssertionError("search must not be called on the no-keyword path")

    def fake_list(repo_url, *, base_url, limit, label=None, timeout_sec):  # noqa: ARG001
        return [GitHubPr(number=30, title="generic PR", html_url="u30")]

    monkeypatch.setattr(src, "search_perf_prs_via_primus_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", fake_list)

    req = _minimal_request(
        search_perf_prs=True,
        search_modes=["primus_cortex"],
        max_search_candidates=1,
        primus_cortex={"base_url": "http://x"},
        # gap_description omitted; keywords too
    )
    out = src.enumerate_candidates(req)
    assert len(out) == 1
    assert out[0].score == 0.0
    assert out[0].ref == "PR:30"


def test_anti_signal_inactive_at_dispatcher_when_no_trigger_in_gap(monkeypatch) -> None:
    """Anti rerank must be a no-op when the gap has no anti-trigger keyword.

    Pins the ``additive only`` contract at the dispatcher boundary so this
    fix can never regress callers whose gaps happen to not carry any of the
    orthogonal-axis triggers (dense / mi300x / bf16 / ...).
    """

    def fake_search(repo_url, *, base_url, query, limit, state, timeout_sec):  # noqa: ARG001
        return [
            GitHubPr(number=10, title="fp8 moe perf improvement", html_url="u10"),
            GitHubPr(number=11, title="fp8 attention fusion", html_url="u11"),
            GitHubPr(number=12, title="random doc edit", html_url="u12"),
        ]

    monkeypatch.setattr(src, "search_perf_prs_via_primus_search", fake_search)
    monkeypatch.setattr(src, "list_perf_prs", lambda *a, **kw: [])

    # Gap with NO anti-trigger (no dense / no mi300x / no bf16 / no rocm / ...).
    # extract_keywords("improve fp8 attention") -> ['attention', 'fp8'].
    req = _minimal_request(
        search_perf_prs=True,
        search_modes=["primus_cortex"],
        max_search_candidates=3,
        primus_cortex={"base_url": "http://x"},
        gap_description="improve fp8 attention",
    )
    out = src.enumerate_candidates(req)
    refs = [c.ref for c in out]
    # Positive-only scoring (no anti-trigger fires):
    #   PR:11 "fp8 attention fusion"   -> 2 (fp8 + attention)
    #   PR:10 "fp8 moe perf improvement" -> 1 (fp8 only; moe not penalised
    #     because gap has no ``dense`` trigger to activate the anti set)
    #   PR:12 "random doc edit"        -> 0
    # The key contract pinned here is: PR:10 keeps its positive overlap
    # of 1 even though it contains ``moe``, proving anti is fully gated
    # on the gap-side trigger.
    assert refs == ["PR:11", "PR:10", "PR:12"]
