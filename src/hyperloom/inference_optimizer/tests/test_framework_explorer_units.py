# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the framework-agent explorer's pure metadata helpers.

``explorer.py`` normalizes PR-detail payloads (from pr-monitor / GitHub,
whose shapes vary) into ``Candidate`` fields, and applies ``PrFilter``. These
extractors and the filter are pure functions over plain dicts/dataclasses, so
they are covered here directly without any network or worktree setup.
"""

from __future__ import annotations

from hyperloom.agents.framework.explorer import (
    _coalesce_str,
    _extract_author,
    _extract_changed_files,
    _extract_head_sha,
    _extract_html_url,
    _extract_labels,
    _extract_title,
    _extract_updated_at,
    _passes_filter,
    _summary_of,
)
from hyperloom.agents.framework.models import Candidate, PrFilter


def test_coalesce_str_returns_first_nonempty_stripped() -> None:
    assert _coalesce_str(None, 5, "", "  hello  ") == "hello"
    assert _coalesce_str("", None, 0) == ""


def test_summary_of_defaults_to_empty_dict() -> None:
    assert _summary_of({"summary": {"k": 1}}) == {"k": 1}
    assert _summary_of({"summary": "not-a-dict"}) == {}
    assert _summary_of({}) == {}


def test_extract_head_sha_from_all_shapes() -> None:
    assert _extract_head_sha({"summary": {"head_sha": "aaa"}}) == "aaa"
    assert _extract_head_sha({"git_fetched_head": "bbb"}) == "bbb"
    # No flat sha keys -> nested head dict, then head string, then empty.
    assert _extract_head_sha({"head": {"oid": "ccc"}}) == "ccc"
    assert _extract_head_sha({"head": "ddd"}) == "ddd"
    assert _extract_head_sha({}) == ""


def test_extract_labels_from_list_and_summary() -> None:
    got = _extract_labels({"labels": ["bug", {"name": "perf"}, {"label": "kernel"}, "   ", 5]})
    assert got == ("bug", "perf", "kernel")
    # Prefer the summary block when it carries labels.
    assert _extract_labels({"summary": {"labels": ["x"]}}) == ("x",)
    assert _extract_labels({}) == ()


def test_extract_author_from_all_shapes() -> None:
    assert _extract_author({"summary": {"author_login": "alice"}}) == "alice"
    assert _extract_author({"author": "bob"}) == "bob"
    assert _extract_author({"author": {"login": "carol"}}) == "carol"
    assert _extract_author({"user": {"name": "dave"}}) == "dave"
    assert _extract_author({"login": "erin"}) == "erin"
    assert _extract_author({}) == ""


def test_extract_scalar_fields() -> None:
    assert _extract_title({"summary": {"title": "T"}}) == "T"
    assert _extract_title({"title": "T2"}) == "T2"
    assert _extract_updated_at({"updated": "2025-01-01"}) == "2025-01-01"
    assert _extract_html_url({"url": "http://x/pr/1"}) == "http://x/pr/1"


def test_extract_changed_files_prefers_payload_then_embedded() -> None:
    payload = [{"path": "a.py"}, {"filename": "b.py"}, {}]
    assert _extract_changed_files({}, payload) == ("a.py", "b.py")
    # No dedicated payload -> fall back to the embedded list (str + dict items).
    embedded = {"files": ["c.py", {"path": "d.py"}, "  ", 7]}
    assert _extract_changed_files(embedded, []) == ("c.py", "d.py")
    assert _extract_changed_files({}, []) == ()


def _cand(**kw: object) -> Candidate:
    return Candidate(ref=kw.pop("ref", "pr/1"), repo="acme/x", **kw)  # type: ignore[arg-type]


def test_passes_filter_empty_and_pass() -> None:
    assert _passes_filter(_cand(), PrFilter()) == (True, "")
    ok, reason = _passes_filter(
        _cand(labels=("perf",), author="alice", changed_files=("src/a.py",)),
        PrFilter(require_labels=("perf",), authors=("alice",), include_paths=("src/",)),
    )
    assert ok and reason == ""


def test_passes_filter_rejections() -> None:
    # Excluded label.
    ok, why = _passes_filter(_cand(labels=("bug",)), PrFilter(exclude_labels=("bug",)))
    assert not ok and "excluded label" in why
    # Author unknown / author not allowed.
    assert _passes_filter(_cand(author=""), PrFilter(authors=("alice",)))[0] is False
    assert _passes_filter(_cand(author="bob"), PrFilter(authors=("alice",)))[0] is False
    # Date window.
    assert _passes_filter(_cand(updated_at="2020"), PrFilter(since="2021"))[0] is False
    assert _passes_filter(_cand(updated_at="2022"), PrFilter(until="2021"))[0] is False
    # Path include/exclude.
    assert _passes_filter(_cand(changed_files=("src/x.py",)), PrFilter(exclude_paths=("src/",)))[0] is False
    assert _passes_filter(_cand(changed_files=("docs/y.md",)), PrFilter(include_paths=("src/",)))[0] is False
    # Changed-file counts.
    assert _passes_filter(_cand(changed_files=("a",)), PrFilter(min_changed_files=2))[0] is False
    assert _passes_filter(_cand(changed_files=("a", "b", "c")), PrFilter(max_changed_files=2))[0] is False
    # Path/count filters need enrichment metadata.
    assert _passes_filter(_cand(), PrFilter(include_paths=("src/",)))[0] is False
