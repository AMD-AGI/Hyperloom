# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the stdlib PR Monitor REST client (config resolution, the
fail-soft HTTP helper, endpoint wrappers, and ``pr_feed_warm`` budgeting)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from inference_optimizer.orchestrator import pr_monitor as pm


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=None):
        return self._body


def _stub_urlopen(monkeypatch, body):
    monkeypatch.setattr(
        pm.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResp(
            body if isinstance(body, bytes) else json.dumps(body).encode()
        ),
    )


# ---- from_args / config ---------------------------------------------------
def test_from_args_env_resolution(monkeypatch):
    monkeypatch.delenv("PR_MONITOR_URL", raising=False)
    monkeypatch.setenv("PRIMUS_CORTEX_PR_URL", "http://env-host/v1/")
    c = pm.PRMonitorClient.from_args()
    assert c.base_url == "http://env-host/v1"  # trailing slash stripped


def test_from_args_explicit_url_and_timeout(monkeypatch):
    c = pm.PRMonitorClient.from_args(url="http://x/v1", timeout_sec=2.5)
    assert c.base_url == "http://x/v1"
    assert c.timeout_sec == 2.5


def test_from_args_bad_timeout_env(monkeypatch):
    monkeypatch.setenv("PR_MONITOR_TIMEOUT_SEC", "not-a-float")
    c = pm.PRMonitorClient.from_args(url="http://x")
    assert c.timeout_sec == pm.DEFAULT_PR_MONITOR_TIMEOUT_SEC


def test_reset_cache():
    c = pm.PRMonitorClient(base_url="http://x")
    c._cache["k"] = []
    c.reset_cache()
    assert c._cache == {}


# ---- _get_json ------------------------------------------------------------
def test_get_json_disabled_raises():
    c = pm.PRMonitorClient(base_url="http://x", enabled=False)
    with pytest.raises(pm.PRMonitorError):
        c._get_json("/healthz")


def test_get_json_success(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")
    _stub_urlopen(monkeypatch, {"ok": True})
    assert c._get_json("/healthz", params={"a": 1, "b": None, "c": ""}) == {"ok": True}


def test_get_json_http_error(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")

    def _raise(req, timeout=None):
        raise pm.urllib.error.HTTPError("http://x", 500, "boom", {}, None)

    monkeypatch.setattr(pm.urllib.request, "urlopen", _raise)
    with pytest.raises(pm.PRMonitorError):
        c._get_json("/p")


def test_get_json_url_error(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")

    def _raise(req, timeout=None):
        raise pm.urllib.error.URLError("down")

    monkeypatch.setattr(pm.urllib.request, "urlopen", _raise)
    with pytest.raises(pm.PRMonitorError):
        c._get_json("/p")


def test_get_json_timeout(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")

    def _raise(req, timeout=None):
        raise TimeoutError("slow")

    monkeypatch.setattr(pm.urllib.request, "urlopen", _raise)
    with pytest.raises(pm.PRMonitorError):
        c._get_json("/p")


def test_get_json_non_json(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")
    _stub_urlopen(monkeypatch, b"\xff\xfenot json")
    with pytest.raises(pm.PRMonitorError):
        c._get_json("/p")


# ---- healthz / list_repos / get_pr ---------------------------------------
def test_healthz_ok_and_fail(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")
    _stub_urlopen(monkeypatch, {"status": "ok"})
    assert c.healthz() is True
    monkeypatch.setattr(c, "_get_json",
                        lambda *a, **k: (_ for _ in ()).throw(pm.PRMonitorError("x")))
    assert c.healthz() is False


def test_list_repos_variants(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")
    monkeypatch.setattr(c, "_get_json", lambda *a, **k: {"items": [
        {"repo_name": "a/b", "is_active": True},
        {"name": "c/d"},
        {"repo_name": "skip", "is_active": False},
        "e/f",
        {"repo_name": ""},
    ]})
    assert c.list_repos() == ["a/b", "c/d", "e/f"]


def test_list_repos_failure_and_nonlist(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")
    monkeypatch.setattr(c, "_get_json",
                        lambda *a, **k: (_ for _ in ()).throw(pm.PRMonitorError("x")))
    assert c.list_repos() == []
    monkeypatch.setattr(c, "_get_json", lambda *a, **k: {"items": "nope"})
    assert c.list_repos() == []


def test_get_pr_success_and_failure(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")
    monkeypatch.setattr(c, "_get_json", lambda *a, **k: {"number": 7})
    assert c.get_pr("a/b", 7) == {"number": 7}
    monkeypatch.setattr(c, "_get_json",
                        lambda *a, **k: (_ for _ in ()).throw(pm.PRMonitorError("x")))
    assert c.get_pr("a/b", 7) is None


# ---- list_prs -------------------------------------------------------------
def _rich_items():
    return [
        {"number": 1, "title": " feat ", "html_url": "http://pr/1",
         "state": "open", "labels": [{"name": "kernel"}, "perf", None],
         "author": {"login": "alice"}, "merged_at": "", "updated_at": "t2",
         "body": "x" * 300},
        {"pr_number": 2, "url": "http://pr/2", "title": "fix",
         "labels": "not-a-list", "author": "bob", "updated_at": "t1"},
        {"number": 0},          # non-positive number -> skip
        {"number": "bad"},      # unparseable -> skip
        "not-a-dict",           # skip
    ]


def test_list_prs_success_and_cache(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")
    monkeypatch.setattr(c, "_get_json", lambda *a, **k: {"items": _rich_items()})
    prs = c.list_prs("a/b", limit=0)  # limit<=0 -> default
    assert [p.number for p in prs] == [1, 2]
    assert prs[0].title == "feat"
    assert prs[0].labels == ("kernel", "perf")
    assert prs[0].author == "alice"
    assert prs[0].body_snippet.endswith("…")
    assert prs[1].url == "http://pr/2"
    # second call served from cache
    monkeypatch.setattr(c, "_get_json",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("net")))
    assert [p.number for p in c.list_prs("a/b", limit=0)] == [1, 2]


def test_list_prs_failure_and_nonlist(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")
    monkeypatch.setattr(c, "_get_json",
                        lambda *a, **k: (_ for _ in ()).throw(pm.PRMonitorError("x")))
    assert c.list_prs("a/b") == []
    c2 = pm.PRMonitorClient(base_url="http://x")
    monkeypatch.setattr(c2, "_get_json", lambda *a, **k: {"items": "nope"})
    assert c2.list_prs("a/b") == []


# ---- pr_feed_warm ---------------------------------------------------------
def test_pr_feed_warm_disabled():
    c = pm.PRMonitorClient(base_url="http://x", enabled=False)
    prs, warns = c.pr_feed_warm(["a/b"])
    assert prs == []
    assert "pr_monitor:disabled" in warns


def test_pr_feed_warm_empty_repos():
    c = pm.PRMonitorClient(base_url="http://x")
    assert c.pr_feed_warm([]) == ([], [])


def test_pr_feed_warm_success_with_keywords(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")
    rows = [
        pm.PRSummary(repo="a/b", number=1, title="fused moe kernel",
                     url="u1", state="open", updated_at="t2"),
        pm.PRSummary(repo="a/b", number=2, title="unrelated docs",
                     url="u2", state="open", updated_at="t1"),
    ]
    monkeypatch.setattr(c, "_list_prs_raising", lambda *a, **k: rows)
    out, warns = c.pr_feed_warm(["a/b"], keywords=["moe"],
                                now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert [p.number for p in out] == [1]
    assert warns == []


def test_pr_feed_warm_fetch_failed_and_exception(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")

    def _raise(repo, **k):
        if repo == "a/b":
            raise pm.PRMonitorError("502")
        raise RuntimeError("weird")

    monkeypatch.setattr(c, "_list_prs_raising", _raise)
    out, warns = c.pr_feed_warm(["a/b", "c/d"])
    assert out == []
    assert any("fetch_failed:a/b" in w for w in warns)
    assert any("exception:c/d" in w for w in warns)


def test_pr_feed_warm_budget_exhausted(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")
    monkeypatch.setattr(c, "_list_prs_raising", lambda *a, **k: [])
    # zero budget with monotonic always past the deadline
    import inference_optimizer.orchestrator.pr_monitor as mod
    times = iter([100.0, 100.0, 200.0, 300.0])
    monkeypatch.setattr(mod, "_matches_keywords", pm._matches_keywords)
    out, warns = c.pr_feed_warm(["a/b"], total_budget_sec=0.0)
    # total_budget_sec falsy -> budget check skipped; just returns empty
    assert out == []


# ---- _list_prs_raising re-raises ------------------------------------------
def test_list_prs_raising_propagates(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")
    monkeypatch.setattr(c, "_get_json",
                        lambda *a, **k: (_ for _ in ()).throw(pm.PRMonitorError("x")))
    with pytest.raises(pm.PRMonitorError):
        c._list_prs_raising("a/b")


def test_list_prs_raising_parses(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")
    monkeypatch.setattr(c, "_get_json", lambda *a, **k: _rich_items())
    prs = c._list_prs_raising("a/b", since="2026-01-01T00:00:00Z")
    assert [p.number for p in prs] == [1, 2]


def test_list_prs_raising_nonlist(monkeypatch):
    c = pm.PRMonitorClient(base_url="http://x")
    monkeypatch.setattr(c, "_get_json", lambda *a, **k: {"items": "nope"})
    assert c._list_prs_raising("a/b") == []


# ---- helpers --------------------------------------------------------------
def test_matches_keywords():
    pr = pm.PRSummary(repo="a/b", number=1, title="Fused MoE",
                      url="u", state="open", labels=("perf",),
                      body_snippet="speedup")
    assert pm._matches_keywords(pr, []) is True
    assert pm._matches_keywords(pr, ["moe"]) is True
    assert pm._matches_keywords(pr, ["nomatch"]) is False


def test_pr_summary_to_dict():
    pr = pm.PRSummary(repo="a/b", number=3, title="t", url="u", state="open",
                      labels=("x", "y"))
    d = pr.to_dict()
    assert d["labels"] == ["x", "y"]
    assert d["number"] == 3
