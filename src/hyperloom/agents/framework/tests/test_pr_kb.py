# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for PR KB consumption (slug parity, page client, diff synthesis, source)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.common.jsonio import iter_sse_objects

from hyperloom.agents.framework import pr_kb, pr_kb_slug
from hyperloom.agents.framework.gbrain_page_client import (
    GbrainPageClient,
    GbrainPageError,
    _select_mcp_response,
    build_gbrain_page_client_from_env,
)


# --- slug -------------------------------------------------------------------


def test_repo_slug_examples():
    assert pr_kb_slug.repo_slug("ROCm/aiter") == "rocm-aiter"
    assert pr_kb_slug.repo_slug("sgl-project/sglang") == "sgl-project-sglang"
    assert pr_kb_slug.repo_slug("https://github.com/ROCm/vllm.git") == "rocm-vllm"


def test_slug_builders(monkeypatch):
    monkeypatch.delenv("PR_KB_SLUG_PREFIX", raising=False)
    assert pr_kb_slug.files_slug("ROCm/vllm", 42) == "pr-kb-files/rocm-vllm/pr/42"
    assert pr_kb_slug.index_slug("ROCm/vllm") == "pr-kb-index/rocm-vllm"


def test_slug_prefix_override(monkeypatch):
    monkeypatch.setenv("PR_KB_SLUG_PREFIX", "prkb2")
    assert pr_kb_slug.files_slug("ROCm/vllm", 1) == "prkb2-files/rocm-vllm/pr/1"


def test_normalise_repo():
    assert pr_kb_slug.normalise_repo("https://github.com/ROCm/vllm.git") == "ROCm/vllm"
    assert pr_kb_slug.normalise_repo("ROCm/vllm") == "ROCm/vllm"
    assert pr_kb_slug.normalise_repo("") == ""


def test_repo_slug_parity_with_worker():
    """Consumer repo_slug must match the PR KB writer byte-for-byte."""
    worker = Path("/path/zhanglei/Primus-Claw/pr-kb/pr_kb/slug.py")
    if not worker.is_file():
        pytest.skip("worker slug.py not present")
    import importlib.util

    spec = importlib.util.spec_from_file_location("_worker_slug", worker)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for repo in ("ROCm/aiter", "sgl-project/sglang", "ROCm/vllm", "triton-lang/triton", "ROCm/FlyDSL", "ROCm/hip"):
        assert pr_kb_slug.repo_slug(repo) == mod.repo_slug(repo), repo


# --- SSE parsing ------------------------------------------------------------


def test_iter_sse_plain_json():
    assert list(iter_sse_objects('{"a": 1}')) == [{"a": 1}]


def test_iter_sse_multi_event_join():
    raw = 'event: hb\ndata: {"id":"0"}\n\nevent: message\ndata: {"id":"1",\ndata: "result":{"ok":true}}\n\n'
    assert _select_mcp_response(raw, "1") == {"id": "1", "result": {"ok": True}}


# --- diff synthesis ---------------------------------------------------------


def test_synthesize_unified_diff_skips_omitted():
    patches = [
        {"filename": "a.py", "status": "modified", "patch": "@@ -1 +1 @@\n-x\n+y"},
        {"filename": "big.bin", "status": "modified", "patch_omitted": True, "reason": "binary"},
    ]
    diff = pr_kb.synthesize_unified_diff(patches)
    assert "diff --git a/a.py b/a.py" in diff
    assert "big.bin" not in diff
    assert diff.endswith("\n")


def test_synthesize_added_removed_dev_null():
    added = pr_kb.synthesize_unified_diff([{"filename": "n.py", "status": "added", "patch": "@@ -0,0 +1 @@\n+z"}])
    assert "--- /dev/null" in added
    removed = pr_kb.synthesize_unified_diff([{"filename": "d.py", "status": "removed", "patch": "@@ -1 +0,0 @@\n-z"}])
    assert "+++ /dev/null" in removed


# --- files page parse + fetch ----------------------------------------------


class _StubClient:
    def __init__(self, page):
        self._page = page

    def get_page(self, slug):
        return self._page


def _files_page(patches, *, truncated=False):
    fm = f"---\nrepo: ROCm/vllm\npr_number: 7\nfiles_truncated: {'true' if truncated else 'false'}\n---\n"
    body = fm + "# PR #7 file changes\n\n## Patches JSON\n\n```json\n" + json.dumps(patches) + "\n```\n"
    return {"markdown": body}


def test_parse_index_prs():
    prs = [{"number": 9, "state": "open", "title": "x"}]
    md = "---\nrepo: ROCm/vllm\n---\n# index\n\n## PRs JSON\n\n```json\n" + json.dumps(prs) + "\n```\n"
    assert pr_kb.parse_index_prs({"markdown": md})[0]["number"] == 9


# --- env builder ------------------------------------------------------------


def test_build_client_from_env_none_when_unset(monkeypatch):
    monkeypatch.delenv("GBRAIN_BASE_URL", raising=False)
    monkeypatch.delenv("GBRAIN_TOKEN", raising=False)
    assert build_gbrain_page_client_from_env() is None


def test_gbrain_client_rejects_non_http_schemes():
    with pytest.raises(GbrainPageError, match="unsupported URL scheme"):
        GbrainPageClient("file:///etc/passwd", "tok")
    with pytest.raises(GbrainPageError, match="unsupported URL scheme"):
        GbrainPageClient("ftp://gbrain.example/mcp", "tok")


def test_build_client_from_env_none_when_scheme_is_not_http(monkeypatch):
    monkeypatch.setenv("GBRAIN_BASE_URL", "file:///tmp/gbrain")
    monkeypatch.setenv("GBRAIN_TOKEN", "tok")
    assert build_gbrain_page_client_from_env() is None


def test_build_client_from_env_none_when_url_is_unparseable(monkeypatch):
    """An unterminated IPv6 literal raises ValueError from under the scheme check."""
    monkeypatch.setenv("GBRAIN_BASE_URL", "http://[::1")
    monkeypatch.setenv("GBRAIN_TOKEN", "tok")
    assert build_gbrain_page_client_from_env() is None


# --- discovery source ---------------------------------------------------

from hyperloom.agents.framework import sources  # noqa: E402
from hyperloom.agents.framework.models import Candidate, ExploreRequest  # noqa: E402
from hyperloom.agents.framework.sources import pr_kb as pr_kb_source  # noqa: E402


class _DiscoveryClient:
    def __init__(self, *, index_page=None, hits=None, list_pages=None):
        self._index = index_page
        self._hits = hits or []
        self._list_pages = list_pages or []

    def health(self):
        return True

    def get_page(self, slug):
        return self._index if slug.startswith("pr-kb-index/") else None

    def query(self, text, *, limit=20):
        return list(self._hits)

    def list_pages(self, *, page_type="", limit=100):
        return list(self._list_pages)


def _req(repo="https://github.com/ROCm/vllm.git", gap="chunked prefill", n=5):
    return ExploreRequest.from_dict(
        {
            "framework": "vllm",
            "repo_url": repo,
            "work_dir": "/tmp/x",
            "baseline": {"throughput": 1.0},
            "gap_description": gap,
            "max_search_candidates": n,
        }
    )


def _index_page(numbers):
    prs = [{"number": n, "state": "open", "title": f"pr {n}", "labels": ["perf"]} for n in numbers]
    md = "---\nrepo: ROCm/vllm\n---\n## PRs JSON\n\n```json\n" + json.dumps(prs) + "\n```\n"
    return {"markdown": md}


def test_enumerate_pr_kb_index_only(monkeypatch):
    monkeypatch.setenv("PR_KB_ENABLE", "1")
    client = _DiscoveryClient(index_page=_index_page([101, 102]))
    monkeypatch.setattr(pr_kb_source, "build_gbrain_page_client_from_env", lambda: client)
    out = pr_kb_source.enumerate_pr_kb(_req())
    assert sorted(c.ref for c in out) == ["PR:101", "PR:102"]
    c = out[0]
    assert c.source == "gbrain_pr_kb"
    assert c.pr_kb_files_slug == "pr-kb-files/rocm-vllm/pr/102"
    assert c.html_url == "https://github.com/ROCm/vllm/pull/102"


def test_enumerate_pr_kb_index_union_query(monkeypatch):
    monkeypatch.setenv("PR_KB_ENABLE", "1")
    hits = [
        {"slug": "pr-kb-meta/rocm-vllm/pr/200", "score": 0.9, "title": "moe"},
        {"slug": "pr-kb-meta/other-repo/pr/999", "score": 0.9},
        {"slug": "pr-kb-meta/rocm-vllm/pr/201", "score": 0.05},
    ]
    client = _DiscoveryClient(index_page=_index_page([101]), hits=hits)
    monkeypatch.setattr(pr_kb_source, "build_gbrain_page_client_from_env", lambda: client)
    out = pr_kb_source.enumerate_pr_kb(_req())
    # All hits for this repo's meta prefix are included regardless of score;
    # the cross-repo hit (other-repo/pr/999) is still excluded by the prefix check.
    assert sorted(c.ref for c in out) == ["PR:101", "PR:200", "PR:201"]


def test_enumerate_pr_kb_list_pages_fallback(monkeypatch):
    # No index page + no meta hits -> list_pages fallback filtered to this repo's meta prefix.
    monkeypatch.setenv("PR_KB_ENABLE", "1")
    listed = [
        {"slug": "pr-kb-files/sgl-project-sglang/pr/29881"},
        {"slug": "pr-kb-meta/sgl-project-sglang/pr/29881", "title": "moe fp8"},
        {"slug": "pr-kb-meta/sgl-project-sglang/pr/27560"},
        {"slug": "pr-kb-meta/other-repo/pr/999"},
    ]
    hits = [{"slug": "hyperloom-recipe-kb/x", "score": 0.9}]
    client = _DiscoveryClient(index_page=None, hits=hits, list_pages=listed)
    monkeypatch.setattr(pr_kb_source, "build_gbrain_page_client_from_env", lambda: client)
    out = pr_kb_source.enumerate_pr_kb(_req(repo="https://github.com/sgl-project/sglang.git"))
    assert sorted(c.ref for c in out) == ["PR:27560", "PR:29881"]
    assert all(c.source == "gbrain_pr_kb" for c in out)
    assert out[0].pr_kb_files_slug == "pr-kb-files/sgl-project-sglang/pr/29881"


def test_enumerate_pr_kb_index_skips_list_pages_fallback(monkeypatch):
    # When the index page yields candidates, the list_pages fallback is not used.
    monkeypatch.setenv("PR_KB_ENABLE", "1")
    listed = [{"slug": "pr-kb-meta/rocm-vllm/pr/777"}]
    client = _DiscoveryClient(index_page=_index_page([101]), list_pages=listed)
    monkeypatch.setattr(pr_kb_source, "build_gbrain_page_client_from_env", lambda: client)
    out = pr_kb_source.enumerate_pr_kb(_req())
    assert [c.ref for c in out] == ["PR:101"]


def test_enumerate_pr_kb_disabled(monkeypatch):
    monkeypatch.setenv("PR_KB_ENABLE", "0")
    assert pr_kb_source.enumerate_pr_kb(_req()) == []


def test_enumerate_pr_kb_unconfigured(monkeypatch):
    monkeypatch.setenv("PR_KB_ENABLE", "1")
    monkeypatch.setattr(pr_kb_source, "build_gbrain_page_client_from_env", lambda: None)
    assert pr_kb_source.enumerate_pr_kb(_req()) == []


# --- tool name + dispatch + whitelist --------------------------------------


def test_query_uses_search_tool():
    called = {}
    client = GbrainPageClient("http://x", "tok")
    client.call = lambda tool, args: called.update(tool=tool, args=args) or []  # type: ignore[method-assign]
    client.query("moe fp8", limit=7)
    assert called["tool"] == "search"
    assert called["args"] == {"query": "moe fp8", "limit": 7}


def test_gbrain_pr_kb_is_valid_search_mode():
    req = ExploreRequest.from_dict(
        {
            "framework": "vllm",
            "repo_url": "https://github.com/ROCm/vllm.git",
            "work_dir": "/tmp/x",
            "baseline": {"throughput": 1.0},
            "search_modes": ["gbrain_pr_kb", "github"],
        }
    )
    assert "gbrain_pr_kb" in req.search_modes


def test_default_search_modes_stay_legacy():
    req = ExploreRequest.from_dict(
        {
            "framework": "vllm",
            "repo_url": "https://github.com/ROCm/vllm.git",
            "work_dir": "/tmp/x",
            "baseline": {"throughput": 1.0},
        }
    )
    assert req.search_modes == ("primus_cortex", "github")


def test_enumerate_candidates_dispatches_gbrain_pr_kb(monkeypatch):
    monkeypatch.setattr(
        sources, "_run_pr_kb", lambda request: [Candidate(ref="PR:5", repo=request.repo_url, source="gbrain_pr_kb")]
    )
    req = ExploreRequest.from_dict(
        {
            "framework": "vllm",
            "repo_url": "https://github.com/ROCm/vllm.git",
            "work_dir": "/tmp/x",
            "baseline": {"throughput": 1.0},
            "search_perf_prs": True,
            "search_modes": ["gbrain_pr_kb"],
        }
    )
    out = sources.enumerate_candidates(req)
    assert [c.ref for c in out] == ["PR:5"]
    assert out[0].source == "gbrain_pr_kb"
