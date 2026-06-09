# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""v0.8 M4 — Knowledge plane + PR Monitor tests (KB_design §3.6 + §3.13 M4)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from inference_optimizer.orchestrator.knowledge_plane import (
    KnowledgePlane,
    load_domain_repos,
)
from inference_optimizer.orchestrator.pr_monitor import (
    DEFAULT_PR_MONITOR_URL,
    PRMonitorClient,
    PRMonitorError,
    PRSummary,
    _matches_keywords,
)
from inference_optimizer.orchestrator.specialist_runner import (
    CORTEX_KB_READONLY_MCP_TOOLS,
    DEFAULT_SPECIALIST_TOOLS,
    PR_MONITOR_MCP_TOOLS,
    SpecialistRunner,
)


# helpers
def _make_response(body: dict[str, Any] | list[Any], *, status: int = 200):
    """Return a context-manager that mimics ``urllib.request.urlopen``."""
    class _Resp:
        def __enter__(self_inner):
            return self_inner
        def __exit__(self_inner, *exc_info):
            return False
        def read(self_inner, *args, **kwargs):
            return json.dumps(body).encode("utf-8")
    return _Resp()


# 1. PR Monitor client — direct REST surface
def test_pr_monitor_client_from_args_default_url():
    c = PRMonitorClient.from_args()
    assert c.base_url == DEFAULT_PR_MONITOR_URL.rstrip("/")
    assert c.enabled is True


def test_pr_monitor_client_from_args_disabled():
    c = PRMonitorClient.from_args(enabled=False)
    assert c.enabled is False
    # The low-level helper raises; the high-level wrappers swallow into empty results.
    with pytest.raises(PRMonitorError):
        c._get_json("/healthz")
    assert c.healthz() is False
    assert c.list_repos() == []
    assert c.list_prs("ROCm/aiter") == []


def test_pr_monitor_healthz_handles_network_error(monkeypatch):
    c = PRMonitorClient.from_args()

    def _boom(*_args, **_kwargs):
        import urllib.error
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.pr_monitor.urllib.request.urlopen",
        _boom,
    )
    assert c.healthz() is False


def test_pr_monitor_list_repos_parses_items(monkeypatch):
    """Legacy mock shape ``{items: [{name: ...}]}`` is still supported for back-compat."""
    c = PRMonitorClient.from_args()
    payload = {"items": [
        {"name": "ROCm/aiter"},
        {"name": "sgl-project/sglang"},
    ]}
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.pr_monitor.urllib.request.urlopen",
        lambda *_a, **_kw: _make_response(payload),
    )
    repos = c.list_repos()
    assert repos == ["ROCm/aiter", "sgl-project/sglang"]


def test_pr_monitor_list_repos_parses_real_rest_shape(monkeypatch):
    """Production returns a JSON array of ``repo_name`` + ``is_active`` entries; reading ``name`` dropped every row (wildcard bug). Inactive repos are skipped."""
    c = PRMonitorClient.from_args()
    payload = [
        {
            "repo_name": "ROCm/aiter",
            "url": "https://github.com/ROCm/aiter.git",
            "is_active": True,
        },
        {
            "repo_name": "ROCm/sglang",
            "url": "https://github.com/ROCm/sglang.git",
            "is_active": False,        # ← inactive: must be skipped
        },
        {
            "repo_name": "ROCm/vllm",
            "is_active": True,
        },
    ]
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.pr_monitor.urllib.request.urlopen",
        lambda *_a, **_kw: _make_response(payload),
    )
    repos = c.list_repos()
    assert repos == ["ROCm/aiter", "ROCm/vllm"]


def test_pr_monitor_list_prs_parses_summary(monkeypatch):
    c = PRMonitorClient.from_args()
    payload = {"items": [
        {
            "number": 3067, "title": "fix attention kernel",
            "state": "merged",
            "html_url": "https://github.com/ROCm/aiter/pull/3067",
            "labels": [{"name": "kernel"}, {"name": "performance"}],
            "author": {"login": "alice"},
            "merged_at": "2026-05-19T00:00:00Z",
            "updated_at": "2026-05-19T00:00:00Z",
            "body": "Improves attention by 3x for short sequences.",
        },
    ]}
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.pr_monitor.urllib.request.urlopen",
        lambda *_a, **_kw: _make_response(payload),
    )
    prs = c.list_prs("ROCm/aiter")
    assert len(prs) == 1
    pr = prs[0]
    assert pr.number == 3067
    assert pr.title == "fix attention kernel"
    assert "kernel" in pr.labels
    assert pr.author == "alice"
    assert pr.url.endswith("/pull/3067")


def test_pr_monitor_list_prs_returns_empty_on_http_error(monkeypatch):
    c = PRMonitorClient.from_args()
    import urllib.error
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.pr_monitor.urllib.request.urlopen",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            urllib.error.HTTPError(
                url="x", code=503, msg="Service Unavailable", hdrs=None, fp=None,
            )
        ),
    )
    # Should swallow into empty list (fail-soft per KB_design §3.13 M4).
    assert c.list_prs("ROCm/aiter") == []


def test_pr_monitor_list_prs_cache_dedups_repeated_calls(monkeypatch):
    c = PRMonitorClient.from_args()
    call_count = {"n": 0}

    def _stub(*_args, **_kwargs):
        call_count["n"] += 1
        return _make_response({"items": [
            {"number": 1, "title": "t", "state": "open"},
        ]})

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.pr_monitor.urllib.request.urlopen",
        _stub,
    )
    a = c.list_prs("ROCm/aiter", state="all", limit=5)
    b = c.list_prs("ROCm/aiter", state="all", limit=5)
    assert a == b
    assert call_count["n"] == 1   # cached
    c.reset_cache()
    c.list_prs("ROCm/aiter", state="all", limit=5)
    assert call_count["n"] == 2


def test_pr_monitor_pr_feed_warm_keyword_filter_and_warning(monkeypatch):
    c = PRMonitorClient.from_args()

    def _stub(req, **_kwargs):  # noqa: ARG001
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "ROCm/aiter" in url:
            return _make_response({"items": [
                {"number": 1, "title": "kernel optimisation", "state": "merged",
                 "labels": []},
                {"number": 2, "title": "docs typo", "state": "merged",
                 "labels": []},
            ]})
        import urllib.error
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.pr_monitor.urllib.request.urlopen",
        _stub,
    )
    prs, warns = c.pr_feed_warm(
        ["ROCm/aiter", "ROCm/missing"],
        keywords=["kernel"],
        window_days=7,
    )
    # docs typo filtered out by keyword; ROCm/missing folded into warnings.
    assert [p.number for p in prs] == [1]
    assert any("ROCm/missing" in w for w in warns)


def test_matches_keywords_case_insensitive():
    pr = PRSummary(
        repo="x/y", number=1, title="Improve KV cache fp8",
        url="", state="open",
    )
    assert _matches_keywords(pr, ["kv_cache", "fp8"])
    assert _matches_keywords(pr, ["fp8"])
    assert not _matches_keywords(pr, ["allreduce"])


# 2. domain → repos yaml loader
def test_load_domain_repos_returns_six_domains():
    repos = load_domain_repos()
    assert set(repos.keys()) == {
        "kernel_switch_specialist", "serving_specialist", "comm_specialist",
        "compiler_specialist", "system_specialist", "pr_intel_specialist",
    }


def test_serving_specialist_has_expected_repos():
    repos = load_domain_repos()
    fr = repos["serving_specialist"]
    assert fr.is_wildcard is False
    assert "sgl-project/sglang" in fr.repos
    assert "vllm" in " ".join(fr.default_keywords)


def test_pr_intel_specialist_is_wildcard():
    repos = load_domain_repos()
    pr = repos["pr_intel_specialist"]
    assert pr.is_wildcard is True
    assert pr.repos == ()


def test_load_domain_repos_missing_file_returns_empty(tmp_path):
    out = load_domain_repos(tmp_path / "nope.yaml")
    assert out == {}


def test_load_domain_repos_ignores_unknown_domain(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "serving_specialist:\n  repos: [a/b]\n  default_keywords: [x]\n"
        "ghost_specialist:\n  repos: [c/d]\n",
        encoding="utf-8",
    )
    out = load_domain_repos(bad)
    assert "serving_specialist" in out
    assert "ghost_specialist" not in out


# 3. KnowledgePlane facade
@pytest.fixture
def plane_with_disabled_pr() -> KnowledgePlane:
    return KnowledgePlane.from_clients(
        cortex_kb=None,
        pr_monitor=PRMonitorClient.from_args(enabled=False),
    )


def test_plane_default_disabled_states(plane_with_disabled_pr):
    plane = plane_with_disabled_pr
    assert plane.pr_monitor_enabled is False
    assert plane.cortex_enabled is False
    assert plane.specialist_mcp_url() == ""


def test_plane_pr_feed_warm_disabled_returns_empty(plane_with_disabled_pr):
    plane = plane_with_disabled_pr
    prs, warns = plane.pr_feed_warm("serving_specialist")
    assert prs == []
    assert "pr_monitor:disabled" in warns
    assert plane.last_warnings == warns


def test_plane_pr_feed_warm_unknown_domain():
    plane = KnowledgePlane.from_clients(
        cortex_kb=None, pr_monitor=PRMonitorClient.from_args(),
    )
    prs, warns = plane.pr_feed_warm("nope_specialist")
    assert prs == []
    assert any("unknown_domain" in w for w in warns)


def test_plane_pr_feed_warm_dispatches_to_repos(monkeypatch):
    plane = KnowledgePlane.from_clients(
        cortex_kb=None, pr_monitor=PRMonitorClient.from_args(),
    )

    def _stub(req, **_kwargs):  # noqa: ARG001
        url = req.full_url if hasattr(req, "full_url") else str(req)
        return _make_response({"items": [
            {"number": 100, "title": "scheduler bug fix",
             "state": "merged", "labels": []},
        ]})

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.pr_monitor.urllib.request.urlopen",
        _stub,
    )
    prs, warns = plane.pr_feed_warm("serving_specialist")
    assert len(prs) > 0
    # No warnings on the happy path.
    assert warns == [] or all(not w.startswith("pr_monitor:exception") for w in warns)


def test_plane_pr_feed_warm_wildcard_expands_via_list_repos(monkeypatch):
    plane = KnowledgePlane.from_clients(
        cortex_kb=None, pr_monitor=PRMonitorClient.from_args(),
    )

    def _stub(req, **_kwargs):  # noqa: ARG001
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/repos?" in url or url.endswith("/repos"):
            return _make_response({"items": [{"name": "ROCm/aiter"}]})
        return _make_response({"items": [
            {"number": 1, "title": "kernel speedup", "state": "merged",
             "labels": []},
        ]})

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.pr_monitor.urllib.request.urlopen",
        _stub,
    )
    prs, warns = plane.pr_feed_warm("pr_intel_specialist")
    assert len(prs) >= 1
    # pr_intel_specialist has empty keywords → everything passes.


# 5. SpecialistRunner tool-list gating
def test_default_specialist_tools_include_all_pr_monitor_mcp_tools():
    for t in PR_MONITOR_MCP_TOOLS:
        assert t in DEFAULT_SPECIALIST_TOOLS
    # 12-tool surface per primus-cortex-pr-monitor-access.md §"Available Tools".
    assert len(PR_MONITOR_MCP_TOOLS) == 12


def test_default_specialist_tools_exclude_orphan_cortex_kb_readonly():
    """Cortex KB has no MCP surface; advertising its read-only tool names caused orphan calls, so they're kept out of the default whitelist (importable for PolicyGate)."""
    for t in CORTEX_KB_READONLY_MCP_TOOLS:
        assert t not in DEFAULT_SPECIALIST_TOOLS


def test_specialist_runner_strips_pr_monitor_when_plane_disabled():
    plane = KnowledgePlane.from_clients(
        cortex_kb=None,
        pr_monitor=PRMonitorClient.from_args(enabled=False),
    )
    runner = SpecialistRunner(
        backend_factory=lambda d: None, knowledge_plane=plane,
    )
    tools = runner._resolve_tools()
    for t in PR_MONITOR_MCP_TOOLS:
        assert t not in tools
    # Cortex KB also disabled here (no client) → cortex MCP stripped too.
    for t in CORTEX_KB_READONLY_MCP_TOOLS:
        assert t not in tools


def test_specialist_runner_keeps_pr_monitor_when_plane_enabled():
    plane = KnowledgePlane.from_clients(
        cortex_kb=None,
        pr_monitor=PRMonitorClient.from_args(enabled=True),
    )
    runner = SpecialistRunner(
        backend_factory=lambda d: None, knowledge_plane=plane,
    )
    tools = runner._resolve_tools()
    for t in PR_MONITOR_MCP_TOOLS:
        assert t in tools


def test_specialist_runner_without_plane_keeps_default_tools():
    """Back-compat: callers without a KnowledgePlane keep every default tool."""
    runner = SpecialistRunner(backend_factory=lambda d: None)
    tools = runner._resolve_tools()
    for t in DEFAULT_SPECIALIST_TOOLS:
        assert t in tools


# 6. KnowledgePlane cache reset
def test_plane_reset_round_caches():
    pr = PRMonitorClient.from_args()
    pr._cache["x"] = []
    plane = KnowledgePlane.from_clients(cortex_kb=None, pr_monitor=pr)
    plane.last_warnings = ["existing"]
    plane.reset_round_caches()
    assert pr._cache == {}
    assert plane.last_warnings == []
