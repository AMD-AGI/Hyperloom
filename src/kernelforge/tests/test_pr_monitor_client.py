"""Tests for the PR Monitor REST transport."""

from __future__ import annotations

import json
import socket
import time
import urllib.error

import pytest

from kernelforge.knowledge import pr_monitor_client as client_module
from kernelforge.knowledge.pr_monitor_client import (
    BOUNDED_PAGE_LIMIT,
    PRContractError,
    PRMonitorClient,
    PRMonitorError,
    PRTransportError,
    clamp_limit,
    extract_items,
    normalize_base_url,
)


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "boom", {}, None)


def _install(monkeypatch, handler):
    """Route urlopen through a handler that receives the requested URL."""
    seen: list[str] = []

    def fake_urlopen(url, timeout=None):
        seen.append(url)
        return handler(url)

    monkeypatch.setattr(
        "kernelforge.knowledge.pr_monitor_client.urllib.request.urlopen",
        fake_urlopen,
    )
    return seen


def test_normalize_base_url_tolerates_version_suffix(monkeypatch):
    monkeypatch.delenv("KB_STORE_URL", raising=False)
    assert normalize_base_url("https://host/pr-monitor") == "https://host/pr-monitor"
    assert normalize_base_url("https://host/pr-monitor/") == "https://host/pr-monitor"
    assert normalize_base_url("https://host/pr-monitor/v1") == "https://host/pr-monitor"
    assert normalize_base_url("https://host/pr-monitor/v1/") == "https://host/pr-monitor"


def test_normalize_base_url_reads_env(monkeypatch):
    monkeypatch.setenv("KB_STORE_URL", "https://env-host/knowledge-base")
    assert normalize_base_url() == "https://env-host/knowledge-base/pr-monitor"


def test_client_requires_kb_store_url(monkeypatch):
    monkeypatch.delenv("KB_STORE_URL", raising=False)
    client = PRMonitorClient()
    with pytest.raises(PRMonitorError, match="KB_STORE_URL"):
        client.get("/repos")


def test_base_url_property_exposes_the_normalized_root():
    assert PRMonitorClient("https://host/pr-monitor/v1").base_url == "https://host/pr-monitor"


def test_clamp_limit_respects_the_local_page_ceiling():
    assert clamp_limit(1000) == BOUNDED_PAGE_LIMIT
    assert clamp_limit(0) == 1
    assert clamp_limit(5) == 5


def test_extract_items_accepts_the_items_envelope():
    assert extract_items({"items": [{"a": 1}]}) == [{"a": 1}]


def test_extract_items_accepts_a_bare_array():
    """Accept the bare arrays returned by search and repository listing."""
    assert extract_items([{"a": 1}, {"b": 2}]) == [{"a": 1}, {"b": 2}]
    assert extract_items([]) == []


@pytest.mark.parametrize(
    "payload",
    [{"total": 3}, "nope", {"items": [{"ok": 1}, "junk"]}, [{"ok": 1}, None]],
)
def test_extract_items_rejects_invalid_contracts(payload):
    with pytest.raises(PRContractError):
        extract_items(payload)


def test_list_repos_reads_the_bare_array_shape(monkeypatch):
    _install(
        monkeypatch,
        lambda url: _FakeResponse(json.dumps([{"repo_name": "ROCm/aiter"}])),
    )

    assert PRMonitorClient("https://host/pr-monitor").list_repos() == [{"repo_name": "ROCm/aiter"}]


def test_get_returns_payload_and_builds_versioned_url(monkeypatch):
    seen = _install(monkeypatch, lambda url: _FakeResponse('{"total": 1}'))
    client = PRMonitorClient("https://host/pr-monitor")

    assert client.get("/repos") == {"total": 1}
    assert seen == ["https://host/pr-monitor/v1/repos"]


def test_get_drops_none_params(monkeypatch):
    seen = _install(monkeypatch, lambda url: _FakeResponse("{}"))
    client = PRMonitorClient("https://host/pr-monitor")

    client.get("/search/prs", {"q": "moe", "repo": None, "limit": 5})

    assert "repo=" not in seen[0]
    assert "q=moe" in seen[0]


def test_missing_pr_and_missing_distill_are_both_normal_absence(monkeypatch):
    _install(monkeypatch, lambda url: (_ for _ in ()).throw(_http_error(404)))
    client = PRMonitorClient("https://host/pr-monitor")

    assert client.get("/repos/o/r/prs/1") is None
    assert client.get("/repos/o/r/prs/1/distill") is None
    assert client.get_pr("o/r", 1) is None


@pytest.mark.parametrize("code", [400, 422])
def test_contract_breakage_is_distinct_from_absence(monkeypatch, code):
    _install(monkeypatch, lambda url: (_ for _ in ()).throw(_http_error(code)))
    client = PRMonitorClient("https://host/pr-monitor")

    with pytest.raises(PRContractError):
        client.get("/repos/o/r/prs")


def test_non_json_body_is_contract_breakage(monkeypatch):
    _install(monkeypatch, lambda url: _FakeResponse("<html>gateway</html>"))
    client = PRMonitorClient("https://host/pr-monitor")

    with pytest.raises(PRContractError):
        client.get("/repos")


@pytest.mark.parametrize(
    "failure",
    [socket.timeout("slow"), urllib.error.URLError("refused"), _http_error(503)],
)
def test_timeout_connection_and_server_errors_are_transport_errors(monkeypatch, failure):
    _install(monkeypatch, lambda url: (_ for _ in ()).throw(failure))
    client = PRMonitorClient("https://host/pr-monitor")

    with pytest.raises(PRTransportError):
        client.get("/repos")


def test_pagination_is_refused_outright(monkeypatch):
    """Reject the service's lossy timestamp-only cursor."""
    seen = _install(monkeypatch, lambda url: _FakeResponse("{}"))
    client = PRMonitorClient("https://host/pr-monitor")

    with pytest.raises(PRMonitorError, match="pagination is disabled"):
        client.get("/repos/o/r/prs", {"before": "2026-08-08T05:45:25+00:00"})
    assert seen == []


def test_no_public_method_accepts_a_cursor():
    """Keep pagination out of the public client API."""
    import inspect

    for name, member in inspect.getmembers(PRMonitorClient, inspect.isfunction):
        if name.startswith("_"):
            continue
        params = set(inspect.signature(member).parameters)
        assert "before" not in params, f"{name} must not expose a cursor"
        assert "cursor" not in params, f"{name} must not expose a cursor"


def test_get_many_returns_one_outcome_per_request(monkeypatch):
    def handler(url):
        if url.endswith("/2"):
            raise _http_error(404)
        if url.endswith("/3"):
            raise _http_error(400)
        return _FakeResponse('{"number": 1}')

    _install(monkeypatch, handler)
    client = PRMonitorClient("https://host/pr-monitor")

    outcomes = client.get_many(
        [
            ("/repos/o/r/prs/1", None),
            ("/repos/o/r/prs/2", None),
            ("/repos/o/r/prs/3", None),
        ]
    )
    by_path = {o.path: o for o in outcomes}

    assert by_path["/repos/o/r/prs/1"].error is None
    assert by_path["/repos/o/r/prs/1"].payload == {"number": 1}
    assert by_path["/repos/o/r/prs/2"].payload is None
    assert by_path["/repos/o/r/prs/2"].error is None
    assert isinstance(by_path["/repos/o/r/prs/3"].error, PRContractError)


def test_get_many_on_empty_input_is_a_no_op():
    assert PRMonitorClient("https://host/pr-monitor").get_many([]) == []


def test_get_many_reports_budget_exhaustion_instead_of_hanging(monkeypatch):
    _install(monkeypatch, lambda url: _FakeResponse("{}"))
    client = PRMonitorClient("https://host/pr-monitor", budget_sec=-1.0)

    outcomes = client.get_many([("/repos/o/r/prs/1", None)])

    assert len(outcomes) == 1
    assert isinstance(outcomes[0].error, PRTransportError)


def test_get_many_accepts_a_per_call_budget(monkeypatch):
    """Allow multiple batches to share one caller budget."""
    _install(monkeypatch, lambda url: _FakeResponse("{}"))
    client = PRMonitorClient("https://host/pr-monitor", budget_sec=300.0)

    outcomes = client.get_many([("/repos/o/r/prs/1", None)], budget_sec=-1.0)

    assert isinstance(outcomes[0].error, PRTransportError)


def test_get_many_returns_when_the_budget_expires(monkeypatch):
    def slow_response(_url):
        """Return after the request budget has expired."""
        time.sleep(0.15)
        return _FakeResponse("{}")

    _install(monkeypatch, slow_response)
    client = PRMonitorClient("https://host/pr-monitor")

    started = time.monotonic()
    outcomes = client.get_many([("/repos/o/r/prs/1", None)], budget_sec=0.01)

    assert time.monotonic() - started < 0.1
    assert outcomes[0].error is not None


def test_get_many_keeps_results_a_slow_sibling_would_have_discarded(monkeypatch):
    """One slow request must not invalidate the answers already in hand."""

    def handler(url):
        if url.endswith("/1"):
            time.sleep(5.0)
        return _FakeResponse(json.dumps({"url": url}))

    _install(monkeypatch, handler)
    client = PRMonitorClient("https://host/pr-monitor")

    outcomes = client.get_many([(f"/repos/o/r/prs/{n}", None) for n in (1, 2, 3)], budget_sec=0.3)

    assert isinstance(outcomes[0].error, PRTransportError)
    assert outcomes[1].payload["url"].endswith("/2")
    assert outcomes[2].payload["url"].endswith("/3")


def test_get_many_returns_within_its_budget_despite_a_hung_request(monkeypatch):
    _install(monkeypatch, lambda url: time.sleep(5.0))
    client = PRMonitorClient("https://host/pr-monitor")

    started = time.monotonic()
    outcomes = client.get_many([("/repos/o/r/prs/1", None)], budget_sec=0.2)

    assert time.monotonic() - started < 1.0
    assert isinstance(outcomes[0].error, PRTransportError)


def test_get_many_gives_queued_requests_only_their_actual_remaining_time(
    monkeypatch,
):
    """A queued worker must not restart the batch clock when it begins."""
    monkeypatch.setattr(client_module, "_MAX_WORKERS", 1)
    timeouts: list[float] = []

    class _RecordingClient(PRMonitorClient):
        def get(self, path, params=None, *, timeout_sec=None):
            """Record the worker's budget and delay the first request."""
            timeouts.append(timeout_sec)
            if path.endswith("/1"):
                time.sleep(0.08)
            return {"path": path}

    client = _RecordingClient("https://host/pr-monitor")
    outcomes = client.get_many(
        [(f"/repos/o/r/prs/{number}", None) for number in (1, 2)],
        budget_sec=0.3,
    )

    assert all(outcome.error is None for outcome in outcomes)
    assert len(timeouts) == 2
    assert 0 < timeouts[1] < timeouts[0] - 0.04


def test_get_many_does_not_hide_an_unexpected_worker_bug(monkeypatch):
    """Only documented PR Monitor failures become per-request outcomes."""
    client = PRMonitorClient("https://host/pr-monitor")

    def bug(*_args, **_kwargs):
        raise RuntimeError("client bug")

    monkeypatch.setattr(client, "get", bug)

    with pytest.raises(RuntimeError, match="client bug"):
        client.get_many([("/repos/o/r/prs/1", None)], budget_sec=1.0)


def test_get_wraps_expected_transport_failures_only(monkeypatch):
    _install(monkeypatch, lambda _url: (_ for _ in ()).throw(OSError("offline")))
    client = PRMonitorClient("https://host/pr-monitor")

    with pytest.raises(PRTransportError, match="OSError"):
        client.get("/healthz")


def test_get_does_not_relabel_an_unexpected_bug_as_transport(monkeypatch):
    _install(
        monkeypatch,
        lambda _url: (_ for _ in ()).throw(RuntimeError("handler bug")),
    )
    client = PRMonitorClient("https://host/pr-monitor")

    with pytest.raises(RuntimeError, match="handler bug"):
        client.get("/healthz")


def test_an_exhausted_budget_issues_no_request(monkeypatch):
    """Past the deadline nothing may be started, not even a cheap call."""
    seen = _install(monkeypatch, lambda url: _FakeResponse("{}"))
    client = PRMonitorClient("https://host/pr-monitor")

    outcomes = client.get_many([("/repos/o/r/prs/1", None)], budget_sec=0.0)

    assert seen == []
    assert isinstance(outcomes[0].error, PRTransportError)


def test_a_single_request_cannot_outlive_the_remaining_budget():
    client = PRMonitorClient("https://host/pr-monitor", timeout_sec=10.0)

    assert client._request_timeout(None) == 10.0
    assert client._request_timeout(2.5) == 2.5
    assert client._request_timeout(60.0) == 10.0
    assert client._request_timeout(-1.0) == 0.0


def test_the_configured_timeout_bounds_each_socket_read(monkeypatch):
    seen: list[float] = []

    def fake_urlopen(url, timeout=None):
        seen.append(timeout)
        return _FakeResponse("{}")

    monkeypatch.setattr(
        "kernelforge.knowledge.pr_monitor_client.urllib.request.urlopen",
        fake_urlopen,
    )
    client = PRMonitorClient("https://host/pr-monitor", timeout_sec=10.0)

    client.get_many([("/repos/o/r/prs/1", None)], budget_sec=2.0)
    client.healthz(timeout_sec=1.0)

    assert 0 < seen[0] <= 2.0
    assert seen[1] == 1.0


def test_get_many_preserves_input_order(monkeypatch):
    """Preserve order when paths are identical and parameters differ."""
    _install(monkeypatch, lambda url: _FakeResponse(json.dumps({"url": url})))
    client = PRMonitorClient("https://host/pr-monitor")
    requests = [("/repos/o/r/prs", {"file_path": f"f{i}.py"}) for i in range(6)]

    outcomes = client.get_many(requests)

    assert len(outcomes) == len(requests)
    for index, outcome in enumerate(outcomes):
        assert f"f{index}.py" in outcome.payload["url"]


def test_get_without_params_omits_the_query_string(monkeypatch):
    seen = _install(monkeypatch, lambda url: _FakeResponse("{}"))
    client = PRMonitorClient("https://host/pr-monitor")

    client.get("/repos", {"state": None})

    assert seen == ["https://host/pr-monitor/v1/repos"]


def test_recent_merged_fallback_defaults_to_a_small_page(monkeypatch):
    """Bound the low-precision recent-PR query."""
    seen = _install(monkeypatch, lambda url: _FakeResponse('{"items": []}'))
    client = PRMonitorClient("https://host/pr-monitor")

    client.list_recent_prs("ROCm/aiter")

    assert "limit=5" in seen[0]
    assert "state=merged" in seen[0]


def test_recent_pr_404_is_normal_absence(monkeypatch):
    _install(monkeypatch, lambda url: (_ for _ in ()).throw(_http_error(404)))

    assert PRMonitorClient("https://host/pr-monitor").list_recent_prs("ROCm/aiter") == []


def test_healthz_uses_the_versioned_path(monkeypatch):
    seen = _install(monkeypatch, lambda url: _FakeResponse("{}"))
    client = PRMonitorClient("https://host/pr-monitor")

    assert client.healthz() is True
    assert seen == ["https://host/pr-monitor/v1/healthz"]


def test_healthz_reports_false_instead_of_raising(monkeypatch):
    _install(monkeypatch, lambda url: (_ for _ in ()).throw(urllib.error.URLError("x")))

    assert PRMonitorClient("https://host/pr-monitor").healthz() is False


def test_healthz_reports_false_on_not_found(monkeypatch):
    _install(monkeypatch, lambda url: (_ for _ in ()).throw(_http_error(404)))

    assert PRMonitorClient("https://host/pr-monitor").healthz() is False


def test_list_repos_rejects_a_non_list_body(monkeypatch):
    _install(monkeypatch, lambda url: _FakeResponse('{"unexpected": true}'))

    with pytest.raises(PRContractError):
        PRMonitorClient("https://host/pr-monitor").list_repos()


def test_get_file_patch_passes_the_path_parameter(monkeypatch):
    seen = _install(monkeypatch, lambda url: _FakeResponse('{"patch": "@@"}'))
    client = PRMonitorClient("https://host/pr-monitor")

    assert client.get_file_patch("ROCm/FlyDSL", 974, "a/b.py") == {"patch": "@@"}
    assert "path=a%2Fb.py" in seen[0]


def test_pr_request_builds_a_get_many_tuple():
    client = PRMonitorClient("https://host/pr-monitor")

    assert client.pr_request("ROCm/aiter", 3747) == ("/repos/ROCm/aiter/prs/3747", None)
