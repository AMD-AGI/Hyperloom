"""Tests for the four-stage PR discovery pipeline."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse

import pytest

from kernelforge.knowledge import pr_monitor_search as search_module
from kernelforge.knowledge.pr_monitor_client import PRMonitorClient
from kernelforge.knowledge.pr_monitor_search import (
    HIT_FILE_PATH,
    HIT_RECENT,
    HIT_SEARCH,
    PRReference,
    component_relevance,
    components_of_interest,
    discover,
    filter_references_by_relevance,
    rank_references,
)
from kernelforge.knowledge.pr_query_context import (
    REASON_CONTRACT_ERROR,
    REASON_NO_CANDIDATE,
    REASON_REPO_UNTRACKED,
    REASON_SERVICE_UNREACHABLE,
    REASON_SKIPPED_DEADLINE,
    PRQueryContext,
)

REPO = "ROCm/FlyDSL"


def _detail(
    number: int,
    *,
    worth: float | None = 0.5,
    merged: bool = True,
    status: str | None = "ok",
    components: list[str] | None = None,
    files: int = 3,
    updated: str = "2026-08-01T00:00:00Z",
    title: str = "",
) -> dict:
    """Build a /prs/{n} payload shaped like the real service response."""
    distill: dict | None = None
    if status is not None:
        distill = {
            "status": status,
            "worth_trying": worth,
            "components": components or ["fused_moe"],
            "mechanisms": ["vectorize"],
            "summary": f"distilled {number}",
            "risk_notes": "",
            "expected_gain": "",
            "head_sha": f"sha{number}",
            "schema_version": "1",
        }
    payload = {
        "summary": {
            "title": title or f"PR {number}",
            "is_merged": merged,
            "pr_updated_at": updated,
            "changed_files": None,
            "head_sha": f"sha{number}",
        },
        "files": [{"path": f"f{i}.py"} for i in range(files)],
        "commits": [],
    }
    if distill is not None:
        payload["distill"] = distill
    return payload


class _Service:
    """Fake PR Monitor that records every URL it is asked for."""

    def __init__(self) -> None:
        self.urls: list[str] = []
        self.by_file: dict[str, list[int]] = {}
        self.by_query: dict[str, list[int]] = {}
        self.recent: list[int] = []
        self.details: dict[int, dict] = {}
        self.status_for: dict[str, int] = {}

    def install(self, monkeypatch) -> None:
        """Route the client's urlopen through this fake."""
        monkeypatch.setattr(
            "kernelforge.knowledge.pr_monitor_client.urllib.request.urlopen",
            self._urlopen,
        )

    def _urlopen(self, url, timeout=None):
        self.urls.append(url)
        for fragment, code in self.status_for.items():
            if fragment in url:
                raise urllib.error.HTTPError(url, code, "boom", {}, None)
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        path = parsed.path

        if "/prs/" in path:
            number = int(path.rsplit("/", 1)[-1])
            detail = self.details.get(number)
            if detail is None:
                raise urllib.error.HTTPError(url, 404, "missing", {}, None)
            return _Body(detail)
        if path.endswith("/search/prs"):
            # A bare JSON array of {matched_field, snippet, summary}, NOT the
            # {items, page} envelope the /prs endpoints use.
            return _Body(
                _search_body(
                    self.by_query.get(params.get("q", [""])[0], []),
                    repo=(params.get("repo") or [REPO])[0],
                )
            )
        if params.get("file_path"):
            return _Body(_list_body(self.by_file.get(params["file_path"][0], [])))
        return _Body(_list_body(self.recent))

    def query_count(self, fragment: str) -> int:
        """How many recorded URLs contain a fragment."""
        return sum(1 for url in self.urls if fragment in url)


def _list_body(rows: list) -> dict:
    """Envelope shape used by /repos/{o}/{r}/prs: {"items": [...], "page": {...}}."""
    items = [row if isinstance(row, dict) else {"number": row} for row in rows]
    return {"items": items, "page": {"total": len(items), "returned": len(items)}}


def _search_body(rows: list, *, repo: str = REPO) -> list:
    """Build the bare-array response used by ``/search/prs``."""
    out = []
    for row in rows:
        if isinstance(row, dict):
            out.append(row)
            continue
        out.append(
            {
                "matched_field": "title",
                "snippet": f"...match for #{row}...",
                "summary": {"number": row, "repo_name": repo, "title": f"PR {row}"},
            }
        )
    return out


class _Body:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._raw


@pytest.fixture()
def service(monkeypatch):
    svc = _Service()
    svc.install(monkeypatch)
    return svc


@pytest.fixture()
def client():
    return PRMonitorClient("https://host/pr-monitor")


def _context(**kwargs) -> PRQueryContext:
    kwargs.setdefault("repo", REPO)
    return PRQueryContext(**kwargs)


def _ref(number: int, worth: float | None, merged: bool, **kwargs) -> PRReference:
    return PRReference(
        repo=REPO,
        number=number,
        hit_via=(HIT_FILE_PATH,),
        worth_trying=worth,
        is_merged=merged,
        **kwargs,
    )


def test_worth_trying_outranks_merge_state():
    """Rank score before merge state."""
    refs = [
        _ref(959, 0.60, True),
        _ref(892, 0.30, True),
        _ref(974, 0.05, True),
        _ref(913, 0.05, True),
        _ref(930, 0.30, False),
    ]

    order = [ref.number for ref in rank_references(refs)]

    assert order[0] == 959
    assert order.index(930) < order.index(974)
    assert order.index(930) < order.index(913)


def test_merge_state_only_breaks_ties_on_equal_worth():
    refs = [_ref(1, 0.30, False), _ref(2, 0.30, True)]

    assert [ref.number for ref in rank_references(refs)] == [2, 1]


def test_path_hits_outrank_everything_else():
    path_hit = PRReference(repo=REPO, number=1, hit_via=(HIT_FILE_PATH,), worth_trying=0.0)
    search_hit = PRReference(repo=REPO, number=2, hit_via=(HIT_SEARCH,), worth_trying=0.9)

    assert [r.number for r in rank_references([search_hit, path_hit])] == [1, 2]


def test_unknown_worth_sorts_below_the_lowest_real_score():
    refs = [_ref(1, None, True), _ref(2, 0.0, False)]

    assert [ref.number for ref in rank_references(refs)] == [2, 1]


def test_ranking_with_all_none_scores_does_not_raise():
    refs = [_ref(1, None, True), _ref(2, None, False)]

    assert len(rank_references(refs)) == 2


def test_component_relevance_outranks_worth():
    """Prefer a task-related search hit over a higher generic score."""
    matching = PRReference(
        repo=REPO,
        number=1,
        hit_via=(HIT_SEARCH,),
        worth_trying=0.1,
        components=("MXFP_MOE",),
    )
    other = PRReference(
        repo=REPO,
        number=2,
        hit_via=(HIT_SEARCH,),
        worth_trying=0.9,
        components=("attention",),
    )

    ranked = rank_references([other, matching], components_of_interest=frozenset({"mxfp_moe"}))

    assert [ref.number for ref in ranked] == [1, 2]


def test_component_relevance_orders_equal_scores():
    focused = PRReference(
        repo=REPO,
        number=1,
        hit_via=(HIT_SEARCH,),
        worth_trying=0.3,
        components=("moe_gemm", "fused_moe"),
    )
    sweeping = PRReference(
        repo=REPO,
        number=2,
        hit_via=(HIT_SEARCH,),
        worth_trying=0.3,
        components=("moe_gemm", "flash_attn", "softmax", "rmsnorm"),
    )

    ranked = rank_references([sweeping, focused], components_of_interest=frozenset({"moe"}))

    assert [ref.number for ref in ranked] == [1, 2]


def test_component_relevance_matches_sub_words_not_just_equality():
    """'moe_gemm' is a real component; exact equality against 'moe' misses it."""
    assert component_relevance(("moe_gemm",), frozenset({"moe"})) == 1.0
    assert component_relevance(("mxfp4_gemm2",), frozenset({"gemm2"})) == 1.0
    assert component_relevance(("flash_attn",), frozenset({"moe"})) == 0.0


def test_component_relevance_prices_in_a_long_component_list():
    """A sweeping refactor hitting one label out of eight is not a strong match."""
    sweeping = (
        "flash_attn",
        "rmsnorm",
        "softmax",
        "layernorm",
        "topk_gating",
        "fused_rope",
        "preshuffle_gemm",
        "mxfp_moe",
    )
    focused = ("fused_moe", "moe_gemm", "mxfp4_gemm2")
    interest = frozenset({"moe", "gemm2"})

    assert component_relevance(focused, interest) > component_relevance(sweeping, interest)


def test_component_relevance_is_zero_without_input():
    assert component_relevance((), frozenset({"moe"})) == 0.0
    assert component_relevance(("moe",), frozenset()) == 0.0


def test_zero_relevance_search_hits_are_filtered():
    """Exclude scored search hits with no task component overlap."""
    reference = PRReference(
        repo=REPO,
        number=1,
        hit_via=(HIT_SEARCH,),
        components=("rmsnorm",),
    )

    assert filter_references_by_relevance([reference], frozenset({"mha_batch_prefill"})) == []


def test_exact_path_hits_survive_a_zero_component_score():
    """Keep exact source history despite component vocabulary drift."""
    reference = PRReference(
        repo=REPO,
        number=1,
        hit_via=(HIT_FILE_PATH,),
        components=("fp8_kv_cache",),
    )

    assert filter_references_by_relevance([reference], frozenset({"mha_batch_prefill"})) == [reference]


def test_undistilled_search_hits_are_not_assumed_irrelevant():
    """Keep search hits whose component metadata is unavailable."""
    reference = PRReference(
        repo=REPO,
        number=1,
        hit_via=(HIT_SEARCH,),
        distill_absent=True,
    )

    assert filter_references_by_relevance([reference], frozenset({"mha_batch_prefill"})) == [reference]


def test_missing_query_terms_disable_relevance_filtering():
    """Avoid rejecting references when the task has no relevance terms."""
    reference = PRReference(
        repo=REPO,
        number=1,
        hit_via=(HIT_SEARCH,),
        components=("rmsnorm",),
    )

    assert filter_references_by_relevance([reference], frozenset()) == [reference]


def test_path_hit_layer_is_sorted_by_worth():
    interest = frozenset({"gemm2", "kernels", "moe", "mxfp_moe"})
    refs = [
        _ref(959, 0.60, True, components=("fused_moe", "gemm2", "mxfp4_gemm", "moe")),
        _ref(
            974,
            0.05,
            True,
            components=(
                "flash_attn",
                "rmsnorm",
                "softmax",
                "layernorm",
                "topk_gating",
                "fused_rope",
                "preshuffle_gemm",
                "mxfp_moe",
            ),
        ),
        _ref(
            913,
            0.05,
            True,
            components=("preshuffle_gemm", "fp8_gemm", "mxfp_moe", "conv3d_implicit", "tiled_mma", "im2col"),
        ),
        _ref(892, 0.30, True, components=("fused_moe", "moe_gemm", "mxfp4_gemm1", "mxfp4_gemm2")),
        _ref(
            930,
            0.30,
            False,
            components=("flash_attention", "mla_decode", "paged_attention", "fused_moe", "gemm", "softmax"),
        ),
    ]

    ranked = rank_references(refs, components_of_interest=interest)
    scores = [r.worth_trying for r in ranked]

    assert scores == sorted(scores, reverse=True)
    assert [r.number for r in ranked][:3] == [959, 892, 930]


def test_components_of_interest_comes_from_keywords_and_paths():
    terms = components_of_interest(_context(file_paths=("kernels/moe/gemm2.py",), keywords=("mxfp moe",)))

    assert {"mxfp", "moe", "kernels", "gemm2"} <= terms


def test_keywords_are_sent_one_request_each(service, client):
    service.by_query = {"mxfp moe": [1], "gemm swizzle": [2]}
    service.details = {1: _detail(1), 2: _detail(2)}

    outcome = discover(client, _context(keywords=("mxfp moe", "gemm swizzle")))

    assert service.query_count("/search/prs") == 2
    assert {ref.number for ref in outcome.references} == {1, 2}


def test_search_results_are_parsed_from_a_bare_array(service, client):
    """Parse the bare-array search response."""
    service.by_query = {"moe": [4629, 4641]}
    service.details = {4629: _detail(4629), 4641: _detail(4641)}

    outcome = discover(client, _context(keywords=("moe",)))

    assert {r.number for r in outcome.references} == {4629, 4641}
    assert all(r.hit_via == (HIT_SEARCH,) for r in outcome.references)
    assert outcome.stats["fallback_used"] is False


def test_search_rows_nest_the_number_under_summary(service, client):
    """A search row is {matched_field, snippet, summary}; the number is inside."""
    service.by_query = {
        "moe": [
            {
                "matched_field": "body",
                "snippet": "...",
                "summary": {"number": 4572, "repo_name": REPO, "title": "t"},
            }
        ]
    }
    service.details = {4572: _detail(4572)}

    outcome = discover(client, _context(keywords=("moe",)))

    assert [r.number for r in outcome.references] == [4572]


def test_search_rows_from_another_repo_are_discarded(service, client):
    """A search may run unfiltered; a foreign row must never become a candidate."""
    service.by_query = {
        "moe": [
            {
                "matched_field": "title",
                "snippet": "...",
                "summary": {"number": 99, "repo_name": "someone/else", "title": "t"},
            }
        ]
    }
    service.details = {99: _detail(99)}

    outcome = discover(client, _context(keywords=("moe",)))

    assert all(r.number != 99 for r in outcome.references)


def test_keyword_queries_are_capped(service, client):
    service.by_query = {}
    discover(client, _context(keywords=tuple(f"kw{i}" for i in range(9))))

    assert service.query_count("/search/prs") == 4


def test_path_queries_are_capped(service, client):
    discover(client, _context(file_paths=tuple(f"a/f{i}.py" for i in range(9))))

    assert service.query_count("file_path=") == 3


def test_empty_path_result_never_retries_with_a_basename(service, client):
    service.by_file = {}

    discover(client, _context(file_paths=("kernels/moe/mxfp_moe/gemm2.py",)))

    path_queries = [u for u in service.urls if "file_path=" in u]
    assert len(path_queries) == 1
    sent = urllib.parse.parse_qs(urllib.parse.urlparse(path_queries[0]).query)
    assert sent["file_path"] == ["kernels/moe/mxfp_moe/gemm2.py"]


def test_fallback_runs_only_when_both_sources_are_empty(service, client):
    service.by_file = {"a/f.py": [7]}
    service.details = {7: _detail(7)}

    discover(client, _context(file_paths=("a/f.py",)))

    assert "state=merged" not in "".join(service.urls)


def test_low_scoring_fallback_only_candidates_are_dropped(service, client):
    """Drop recent-only candidates below their score floor."""
    service.recent = [11042, 11150, 11218, 11223, 11224]
    worths = {11042: 0.02, 11150: 0.60, 11218: 0.02, 11223: 0.05, 11224: 0.10}
    service.details = {n: _detail(n, worth=w, components=["nothing"]) for n, w in worths.items()}

    outcome = discover(client, _context(keywords=("nothing",)))

    assert [ref.number for ref in outcome.references] == [11150]


def test_a_path_hit_is_never_dropped_for_a_low_score(service, client):
    """Keep path hits at the default global floor."""
    service.by_file = {"a/f.py": [1]}
    service.details = {1: _detail(1, worth=0.0)}

    outcome = discover(client, _context(file_paths=("a/f.py",)))

    assert [ref.number for ref in outcome.references] == [1]


def test_a_keyword_hit_is_never_dropped_for_a_low_score(service, client):
    service.by_query = {"moe": [1]}
    service.details = {1: _detail(1, worth=0.01)}

    outcome = discover(client, _context(keywords=("moe",)))

    assert [ref.number for ref in outcome.references] == [1]


def test_all_fallback_candidates_weak_yields_no_candidate(service, client):
    service.recent = [1, 2]
    service.details = {n: _detail(n, worth=0.05) for n in (1, 2)}

    outcome = discover(client, _context(keywords=("nothing",)))

    assert outcome.references == ()
    assert outcome.reason == REASON_NO_CANDIDATE


def test_the_fallback_floor_is_configurable(monkeypatch, service, client):
    """A repository whose distills score conservatively needs the floor moved,
    not the whole feature turned off."""
    monkeypatch.setenv("PR_KB_FALLBACK_MIN_WORTH", "0.05")
    service.recent = [1, 2]
    service.details = {
        1: _detail(1, worth=0.05, components=["nothing"]),
        2: _detail(2, worth=0.01, components=["nothing"]),
    }

    outcome = discover(client, _context(keywords=("nothing",)))

    assert [ref.number for ref in outcome.references] == [1]


def test_the_global_floor_is_disabled_by_default(service, client):
    """Keep established hits when the global floor is unset."""
    service.by_query = {"moe": [1]}
    service.details = {1: _detail(1, worth=0.0)}

    assert discover(client, _context(keywords=("moe",))).references


def test_the_global_floor_filters_established_hits_when_raised(monkeypatch, service, client):
    """PR_KB_MIN_WORTH is the opt-in that trades recall for precision."""
    monkeypatch.setenv("PR_KB_MIN_WORTH", "0.5")
    service.by_query = {"moe": [1, 2]}
    service.details = {1: _detail(1, worth=0.6), 2: _detail(2, worth=0.4)}

    outcome = discover(client, _context(keywords=("moe",)))

    assert [ref.number for ref in outcome.references] == [1]


def test_an_unparsable_floor_is_rejected(monkeypatch, service, client):
    monkeypatch.setenv("PR_KB_FALLBACK_MIN_WORTH", "not-a-number")
    service.recent = [1]
    service.details = {1: _detail(1, worth=0.05)}

    with pytest.raises(ValueError):
        discover(client, _context(keywords=("nothing",)))


def test_an_undistilled_fallback_candidate_is_dropped(service, client):
    """Unknown score plus no established link is not worth prompt space."""
    service.recent = [1]
    service.details = {1: _detail(1, status=None)}

    outcome = discover(client, _context(keywords=("nothing",)))

    assert outcome.references == ()


def test_fallback_uses_a_small_page(service, client):
    service.recent = [11, 12]
    service.details = {11: _detail(11), 12: _detail(12)}

    outcome = discover(client, _context(keywords=("nothing",)))

    fallback = [u for u in service.urls if "state=merged" in u]
    assert len(fallback) == 1
    assert "limit=5" in fallback[0]
    assert outcome.stats["fallback_used"] is True
    assert all(ref.hit_via == (HIT_RECENT,) for ref in outcome.references)


def test_multi_source_hits_keep_every_source_marker(service, client):
    service.by_file = {"a/f.py": [42]}
    service.by_query = {"moe": [42]}
    service.details = {42: _detail(42)}

    outcome = discover(client, _context(file_paths=("a/f.py",), keywords=("moe",)))

    assert len(outcome.references) == 1
    assert outcome.references[0].hit_via == (HIT_FILE_PATH, HIT_SEARCH)


def test_candidate_cap_bounds_the_enrichment_request_count(service, client):
    service.by_query = {"moe": list(range(1, 31))}
    service.details = {n: _detail(n) for n in range(1, 31)}

    discover(client, _context(keywords=("moe",)), candidate_cap=4)

    assert service.query_count("/prs/") == 4


def test_cap_prefers_path_hits_over_search_hits(service, client):
    service.by_file = {"a/f.py": [1]}
    service.by_query = {"moe": [900, 901, 902]}
    service.details = {n: _detail(n) for n in (1, 900, 901, 902)}

    discover(client, _context(file_paths=("a/f.py",), keywords=("moe",)), candidate_cap=1)

    assert service.query_count("/prs/1") == 1


def test_enrichment_is_one_hop_per_candidate(service, client):
    service.by_query = {"moe": [1, 2]}
    service.details = {1: _detail(1), 2: _detail(2)}

    outcome = discover(client, _context(keywords=("moe",)))

    assert service.query_count("/distill") == 0
    assert service.query_count("/files") == 0
    assert outcome.stats["http_calls"] == 3


@pytest.mark.parametrize("status", ["empty", "error"])
def test_distilled_but_contentless_prs_are_dropped(service, client, status):
    service.by_query = {"moe": [1]}
    service.details = {1: _detail(1, status=status)}

    outcome = discover(client, _context(keywords=("moe",)))

    assert outcome.references == ()
    assert outcome.reason == REASON_NO_CANDIDATE
    assert outcome.stats["distill_dropped"] == 1


def test_undistilled_pr_is_kept_with_an_unknown_score(service, client):
    """Not yet distilled is not the same as distilled and found empty."""
    service.by_file = {"a/f.py": [1]}
    service.details = {1: _detail(1, status=None)}

    outcome = discover(client, _context(file_paths=("a/f.py",)))

    assert len(outcome.references) == 1
    reference = outcome.references[0]
    assert reference.distill_absent is True
    assert reference.worth_trying is None
    assert outcome.stats["distill_absent"] == 1


def test_missing_pr_is_skipped_without_failing_the_batch(service, client):
    service.by_query = {"moe": [1, 2]}
    service.details = {2: _detail(2)}

    outcome = discover(client, _context(keywords=("moe",)))

    assert [ref.number for ref in outcome.references] == [2]


def test_merge_state_reads_is_merged_not_merged_at(service, client):
    """``merged_at`` does not exist; reading it would silently yield False."""
    service.by_query = {"moe": [1]}
    detail = _detail(1, merged=True)
    detail["summary"].pop("is_merged")
    detail["summary"]["is_merged"] = True
    detail["summary"]["pr_merged_at"] = "2026-08-01T00:00:00Z"
    service.details = {1: detail}

    outcome = discover(client, _context(keywords=("moe",)))

    assert outcome.references[0].is_merged is True


def test_file_count_comes_from_the_files_array(service, client):
    """``summary.changed_files`` is always null, so it must not be the source."""
    service.by_query = {"moe": [1]}
    service.details = {1: _detail(1, files=7)}

    outcome = discover(client, _context(keywords=("moe",)))

    assert outcome.references[0].n_files == 7


def test_reference_carries_the_snapshot_key_fields(service, client):
    service.by_query = {"moe": [1]}
    service.details = {1: _detail(1)}

    reference = discover(client, _context(keywords=("moe",))).references[0]

    assert (reference.repo, reference.number) == (REPO, 1)
    assert reference.head_sha == "sha1"
    assert reference.schema_version == "1"


def test_unusable_context_short_circuits_without_any_request(service, client):
    outcome = discover(client, _context(reason=REASON_REPO_UNTRACKED))

    assert outcome.reason == REASON_REPO_UNTRACKED
    assert service.urls == []
    assert outcome.stats["http_calls"] == 0


def test_no_candidate_anywhere_reports_no_candidate(service, client):
    outcome = discover(client, _context(keywords=("moe",)))

    assert outcome.reason == REASON_NO_CANDIDATE
    assert not outcome.references


def test_contract_error_is_reported_not_swallowed(service, client):
    service.status_for = {"/search/prs": 422}

    outcome = discover(client, _context(keywords=("moe",)))

    assert outcome.reason == REASON_CONTRACT_ERROR


def test_malformed_rows_are_skipped_not_fatal(service, client):
    service.by_query = {"moe": [{"number": "not-a-number"}, {"no_number": 1}, 5]}
    service.details = {5: _detail(5)}

    outcome = discover(client, _context(keywords=("moe",)))

    assert [ref.number for ref in outcome.references] == [5]


def test_absent_stage_one_response_is_not_an_error(service, client):
    """A 404 on a candidate query is normal absence, not contract breakage."""
    service.status_for = {"file_path=": 404}
    service.recent = [3]
    service.details = {3: _detail(3)}

    outcome = discover(client, _context(file_paths=("a/f.py",)))

    assert [ref.number for ref in outcome.references] == [3]


def test_contract_error_in_the_fallback_is_reported(service, client):
    service.status_for = {"state=merged": 422}

    outcome = discover(client, _context(keywords=("moe",)))

    assert outcome.reason == REASON_CONTRACT_ERROR


def test_transport_failure_in_the_fallback_is_reported(service, client):
    service.status_for = {"state=merged": 503}

    outcome = discover(client, _context(keywords=("moe",)))

    assert outcome.reason == REASON_SERVICE_UNREACHABLE


def test_contract_error_during_enrichment_is_recorded(service, client):
    service.by_query = {"moe": [1]}
    service.status_for = {"/prs/1": 400}

    outcome = discover(client, _context(keywords=("moe",)))

    assert outcome.reason == REASON_CONTRACT_ERROR


def test_partial_contract_error_is_recorded_as_degraded(service, client):
    service.by_query = {"moe": [1, 2]}
    service.status_for = {"/prs/1": 400}
    service.details = {2: _detail(2)}

    outcome = discover(client, _context(keywords=("moe",)))

    assert [ref.number for ref in outcome.references] == [2]
    assert outcome.reason == ""
    assert outcome.stats["degraded_reason"] == REASON_CONTRACT_ERROR


def test_top_k_truncates_the_ranked_list(service, client):
    service.by_query = {"moe": [1, 2, 3, 4, 5, 6]}
    service.details = {n: _detail(n, worth=n / 10) for n in range(1, 7)}

    outcome = discover(client, _context(keywords=("moe",)), top_k=2)

    assert len(outcome.references) == 2
    assert [ref.number for ref in outcome.references] == [6, 5]
    assert len(outcome.surfaced_references) == 6
    assert outcome.stats["surfaced"] == 6


def test_discovery_drops_unrelated_search_hits(service, client):
    """Filter unrelated search hits before top-k presentation."""
    service.by_query = {"mha batch prefill": [1, 2]}
    service.details = {
        1: _detail(1, worth=0.2, components=["mha_batch_prefill"]),
        2: _detail(2, worth=0.9, components=["fused_moe"]),
    }

    outcome = discover(client, _context(keywords=("mha batch prefill",)))

    assert [reference.number for reference in outcome.references] == [1]
    assert outcome.stats["relevance_dropped"] == 1


def test_stats_record_the_unique_candidate_count(service, client):
    service.by_file = {"a/f.py": [1]}
    service.by_query = {"moe": [1, 2]}
    service.details = {1: _detail(1), 2: _detail(2)}

    outcome = discover(client, _context(file_paths=("a/f.py",), keywords=("moe",)))

    assert outcome.stats["candidates"] == 2


def test_an_expired_deadline_blocks_the_recent_fallback(service, client):
    """The least precise stage never spends time the caller no longer has."""
    service.by_query = {"moe": []}
    service.recent = [11042]
    service.details = {11042: _detail(11042, worth=0.9)}

    outcome = discover(client, _context(keywords=("moe",)), deadline=time.monotonic() - 1.0)

    assert outcome.reason == REASON_SKIPPED_DEADLINE
    assert outcome.stats["fallback_used"] is False
    assert service.query_count("state=merged") == 0


def test_an_expired_deadline_blocks_enrichment(service, client, monkeypatch):
    """Candidates in hand do not license spending past the deadline."""
    service.by_query = {"moe": [1, 2]}
    service.details = {1: _detail(1), 2: _detail(2)}
    real = search_module.remaining_sec
    calls = {"n": 0}

    def expire_after_discovery(deadline):
        """Report time left for candidate collection, none after it."""
        calls["n"] += 1
        return real(deadline) if calls["n"] <= 1 else -1.0

    monkeypatch.setattr(search_module, "remaining_sec", expire_after_discovery)

    outcome = discover(client, _context(keywords=("moe",)))

    assert outcome.reason == REASON_SKIPPED_DEADLINE
    assert outcome.stats["degraded_reason"] == REASON_SKIPPED_DEADLINE
    assert service.query_count("/prs/") == 0


def test_a_caller_deadline_outranks_a_budget(service, client):
    """The absolute cutoff wins so stages cannot each restart the clock."""
    service.by_query = {"moe": [1]}
    service.details = {1: _detail(1)}

    outcome = discover(
        client,
        _context(keywords=("moe",)),
        budget_sec=300.0,
        deadline=time.monotonic() - 1.0,
    )

    assert outcome.reason == REASON_SKIPPED_DEADLINE


def test_environment_overrides_top_k_and_cap(service, client, monkeypatch):
    monkeypatch.setenv("PR_KB_TOP_K", "1")
    monkeypatch.setenv("PR_KB_CANDIDATE_CAP", "2")
    service.by_query = {"moe": [1, 2, 3]}
    service.details = {n: _detail(n) for n in (1, 2, 3)}

    outcome = discover(client, _context(keywords=("moe",)))

    assert len(outcome.references) == 1
    assert service.query_count("/prs/") == 2
