# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Knowledge plane and PR Monitor tests."""

from __future__ import annotations

import json
from typing import Any

import pytest

from hyperloom.orchestrator.knowledge.knowledge_plane import (
    KnowledgePlane,
    load_domain_repos,
)
from hyperloom.orchestrator.knowledge.pr_monitor import (
    DEFAULT_PR_MONITOR_URL,
    PRMonitorClient,
    PRMonitorError,
    PRSummary,
    _matches_keywords,
)
from hyperloom.orchestrator.specialists.runner import (
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
    assert c.list_repos() == []


def test_pr_monitor_list_repos_parses_items(monkeypatch):
    """Legacy mock shape ``{items: [{name: ...}]}`` is still supported for back-compat."""
    c = PRMonitorClient.from_args()
    payload = {
        "items": [
            {"name": "ROCm/aiter"},
            {"name": "sgl-project/sglang"},
        ]
    }
    monkeypatch.setattr(
        "hyperloom.orchestrator.knowledge.pr_monitor.urllib.request.urlopen",
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
            "is_active": False,  # ← inactive: must be skipped
        },
        {
            "repo_name": "ROCm/vllm",
            "is_active": True,
        },
    ]
    monkeypatch.setattr(
        "hyperloom.orchestrator.knowledge.pr_monitor.urllib.request.urlopen",
        lambda *_a, **_kw: _make_response(payload),
    )
    repos = c.list_repos()
    assert repos == ["ROCm/aiter", "ROCm/vllm"]


def test_pr_monitor_pr_feed_warm_keyword_filter_and_warning(monkeypatch):
    c = PRMonitorClient.from_args()

    def _stub(req, **_kwargs):  # noqa: ARG001
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "ROCm/aiter" in url:
            return _make_response(
                {
                    "items": [
                        {"number": 1, "title": "kernel optimisation", "state": "merged", "labels": []},
                        {"number": 2, "title": "docs typo", "state": "merged", "labels": []},
                    ]
                }
            )
        import urllib.error

        raise urllib.error.URLError("no route")

    monkeypatch.setattr(
        "hyperloom.orchestrator.knowledge.pr_monitor.urllib.request.urlopen",
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
        repo="x/y",
        number=1,
        title="Improve KV cache fp8",
        url="",
        state="open",
    )
    assert _matches_keywords(pr, ["kv_cache", "fp8"])
    assert _matches_keywords(pr, ["fp8"])
    assert not _matches_keywords(pr, ["allreduce"])


# 2. domain → repos yaml loader
def test_load_domain_repos_returns_six_domains():
    repos = load_domain_repos()
    assert set(repos.keys()) == {
        "kernel_switch_specialist",
        "serving_specialist",
        "comm_specialist",
        "compiler_specialist",
        "system_specialist",
        "pr_intel_specialist",
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
        "serving_specialist:\n  repos: [a/b]\n  default_keywords: [x]\nghost_specialist:\n  repos: [c/d]\n",
        encoding="utf-8",
    )
    out = load_domain_repos(bad)
    assert "serving_specialist" in out
    assert "ghost_specialist" not in out


# 3. KnowledgePlane facade
@pytest.fixture
def plane_with_disabled_pr() -> KnowledgePlane:
    return KnowledgePlane.from_clients(
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
        pr_monitor=PRMonitorClient.from_args(),
    )
    prs, warns = plane.pr_feed_warm("nope_specialist")
    assert prs == []
    assert any("unknown_domain" in w for w in warns)


def test_plane_pr_feed_warm_dispatches_to_repos(monkeypatch):
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(),
    )

    def _stub(req, **_kwargs):  # noqa: ARG001
        url = req.full_url if hasattr(req, "full_url") else str(req)
        return _make_response(
            {
                "items": [
                    {"number": 100, "title": "scheduler bug fix", "state": "merged", "labels": []},
                ]
            }
        )

    monkeypatch.setattr(
        "hyperloom.orchestrator.knowledge.pr_monitor.urllib.request.urlopen",
        _stub,
    )
    prs, warns = plane.pr_feed_warm("serving_specialist")
    assert len(prs) > 0
    # No warnings on the happy path.
    assert warns == [] or all(not w.startswith("pr_monitor:exception") for w in warns)


def test_plane_pr_feed_warm_wildcard_expands_via_list_repos(monkeypatch):
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(),
    )

    def _stub(req, **_kwargs):  # noqa: ARG001
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/repos?" in url or url.endswith("/repos"):
            return _make_response({"items": [{"name": "ROCm/aiter"}]})
        return _make_response(
            {
                "items": [
                    {"number": 1, "title": "kernel speedup", "state": "merged", "labels": []},
                ]
            }
        )

    monkeypatch.setattr(
        "hyperloom.orchestrator.knowledge.pr_monitor.urllib.request.urlopen",
        _stub,
    )
    prs, warns = plane.pr_feed_warm("pr_intel_specialist")
    assert len(prs) >= 1
    # pr_intel_specialist has empty keywords → everything passes.


# 5. SpecialistRunner tool-list gating
def test_default_specialist_tools_include_all_pr_monitor_mcp_tools():
    for t in PR_MONITOR_MCP_TOOLS:
        assert t in DEFAULT_SPECIALIST_TOOLS
    # 12-tool PR-Monitor surface.
    assert len(PR_MONITOR_MCP_TOOLS) == 12


def test_default_specialist_tools_include_cortex_kb_readonly():
    """The cortex_kb MCP server (gbrain KB-graph) now backs these read-only tool
    names, so they live in the default whitelist; they are stripped at resolve
    time when the KB-graph MCP is not wired (KnowledgePlane.cortex_enabled)."""
    for t in CORTEX_KB_READONLY_MCP_TOOLS:
        assert t in DEFAULT_SPECIALIST_TOOLS


def test_specialist_runner_strips_pr_monitor_when_plane_disabled():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=False),
    )
    runner = SpecialistRunner(
        backend_factory=lambda d: None,
        knowledge_plane=plane,
    )
    tools = runner._resolve_tools()
    for t in PR_MONITOR_MCP_TOOLS:
        assert t not in tools
    # Cortex KB also disabled here (no client) → cortex MCP stripped too.
    for t in CORTEX_KB_READONLY_MCP_TOOLS:
        assert t not in tools


def test_specialist_runner_keeps_pr_monitor_when_plane_enabled():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=True),
    )
    runner = SpecialistRunner(
        backend_factory=lambda d: None,
        knowledge_plane=plane,
    )
    tools = runner._resolve_tools()
    for t in PR_MONITOR_MCP_TOOLS:
        assert t in tools


def test_specialist_runner_keeps_cortex_kb_when_mcp_wired():
    """When the KB-graph (cortex_kb) MCP URL is configured the read-only tools survive."""
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=False),
        cortex_kb_mcp_url="http://gbrain.test/mcp",
        cortex_kb_mcp_headers={"Authorization": "Bearer t"},
    )
    assert plane.cortex_enabled is True
    assert plane.cortex_specialist_mcp_url() == "http://gbrain.test/mcp"
    assert plane.cortex_specialist_mcp_headers() == {"Authorization": "Bearer t"}
    runner = SpecialistRunner(
        backend_factory=lambda d: None,
        knowledge_plane=plane,
    )
    tools = runner._resolve_tools()
    for t in CORTEX_KB_READONLY_MCP_TOOLS:
        assert t in tools


def test_specialist_runner_without_plane_keeps_default_tools():
    """Back-compat: callers without a KnowledgePlane keep every default tool."""
    runner = SpecialistRunner(backend_factory=lambda d: None)
    tools = runner._resolve_tools()
    for t in DEFAULT_SPECIALIST_TOOLS:
        assert t in tools


# 5b. Specialist MCP config writer (pr_monitor + cortex_kb servers)
def test_mcp_config_writes_cortex_kb_server_with_headers(tmp_path):
    from hyperloom.orchestrator.specialists.mcp_config import (
        SPECIALIST_MCP_CONFIG_FILENAME,
        write_specialist_mcp_config,
    )

    path = write_specialist_mcp_config(
        session_dir=tmp_path,
        pr_monitor_mcp_url="http://pr.test/mcp/",
        cortex_kb_mcp_url="http://gbrain.test/mcp",
        cortex_kb_mcp_headers={"Authorization": "Bearer secret"},
    )
    assert path is not None and path.name == SPECIALIST_MCP_CONFIG_FILENAME
    cfg = json.loads(path.read_text())
    servers = cfg["mcpServers"]
    assert servers["pr_monitor"] == {"type": "http", "url": "http://pr.test/mcp/"}
    assert servers["cortex_kb"]["type"] == "http"
    assert servers["cortex_kb"]["url"] == "http://gbrain.test/mcp"
    assert servers["cortex_kb"]["headers"] == {"Authorization": "Bearer secret"}


def test_mcp_config_omits_cortex_kb_when_url_absent(tmp_path):
    from hyperloom.orchestrator.specialists.mcp_config import (
        write_specialist_mcp_config,
    )

    path = write_specialist_mcp_config(
        session_dir=tmp_path,
        pr_monitor_mcp_url="http://pr.test/mcp/",
    )
    assert path is not None
    cfg = json.loads(path.read_text())
    assert "cortex_kb" not in cfg["mcpServers"]
    assert "pr_monitor" in cfg["mcpServers"]


def test_mcp_config_returns_none_when_nothing_wireable(tmp_path):
    from hyperloom.orchestrator.specialists.mcp_config import (
        write_specialist_mcp_config,
    )

    assert (
        write_specialist_mcp_config(session_dir=tmp_path, pr_monitor_mcp_url="")
        is None
    )


# 6. KnowledgePlane cache reset
def test_plane_reset_round_caches():
    pr = PRMonitorClient.from_args()
    pr._cache["x"] = []
    plane = KnowledgePlane.from_clients(pr_monitor=pr)
    plane.last_warnings = ["existing"]
    plane.reset_round_caches()
    assert pr._cache == {}
    assert plane.last_warnings == []
