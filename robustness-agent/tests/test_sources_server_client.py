"""Unit tests for the robustness-server client + Source adapter."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest

from robustness_agent.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.sources.base import SourceUnavailable
from robustness_agent.sources.server_client import (
    RobustnessServerClient,
    RobustnessServerSource,
)


def _ctx(
    *,
    session_id: str = "sess-1",
    now_unix: float = 1_700_000_000.0,
) -> ReactorContext:
    return ReactorContext(
        tick_index=1,
        shared_state=SharedStateSnapshot(session_id=session_id),
        inbox=[],
        now_unix=now_unix,
    )


def _client(handler) -> RobustnessServerClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(
        base_url="http://server.test", transport=transport, timeout=5.0
    )
    return RobustnessServerClient("http://server.test", client=http)


# ---------------------------------------------------------------------------
# Client low-level behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_session_returns_none_on_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "session not found"})

    client = _client(handler)
    try:
        result = await client.get_session("missing")
    finally:
        await client.aclose()
    assert result is None


@pytest.mark.asyncio
async def test_get_session_returns_dict_on_200():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"session_id": "sess-1", "model_name": "qwen3-8b"},
        )

    client = _client(handler)
    try:
        result = await client.get_session("sess-1")
    finally:
        await client.aclose()
    assert result == {"session_id": "sess-1", "model_name": "qwen3-8b"}


@pytest.mark.asyncio
async def test_list_session_events_unwraps_envelope():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"events": [{"id": 1}, {"id": 2}]})

    client = _client(handler)
    try:
        events = await client.list_session_events("sess-1")
    finally:
        await client.aclose()
    assert events == [{"id": 1}, {"id": 2}]


@pytest.mark.asyncio
async def test_5xx_raises_source_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "down"})

    client = _client(handler)
    try:
        with pytest.raises(SourceUnavailable):
            await client.list_session_pods("sess-1")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_connect_error_raises_source_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = _client(handler)
    try:
        with pytest.raises(SourceUnavailable):
            await client.list_sessions(limit=10)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_metrics_window_query_params_are_unix_seconds():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"pods": []})

    client = _client(handler)
    from robustness_agent.sources.server_client import _MetricsWindow

    try:
        await client.get_session_metrics(
            "sess-1",
            _MetricsWindow(start_unix=100, end_unix=200),
            categories=["gpu", "cpu"],
            step="15s",
        )
    finally:
        await client.aclose()
    assert "start=100" in seen["url"]
    assert "end=200" in seen["url"]
    assert "categories=gpu%2Ccpu" in seen["url"] or "categories=gpu,cpu" in seen["url"]
    assert "step=15s" in seen["url"]


# ---------------------------------------------------------------------------
# Source adapter behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_source_returns_pods_events_summary():
    requests_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(str(request.url))
        if "/pods" in request.url.path:
            return httpx.Response(200, json=[{"pod": {"name": "brain-0"}}])
        if "/events" in request.url.path:
            return httpx.Response(200, json={"events": [{"kind": "ping"}]})
        if "/summary" in request.url.path:
            return httpx.Response(200, json={"pods": [], "session": {}})
        return httpx.Response(404)

    client = _client(handler)
    try:
        source = RobustnessServerSource(client)
        data = await source.fetch(_ctx())
    finally:
        await client.aclose()
    assert data.session_pods == [{"pod": {"name": "brain-0"}}]
    assert data.session_events == [{"kind": "ping"}]
    assert data.session_summary == {"pods": [], "session": {}}
    assert data.sources_used == ["robustness-server"]
    assert any("/pods" in u for u in requests_seen)
    assert any("/events" in u for u in requests_seen)
    assert any("/summary" in u for u in requests_seen)


@pytest.mark.asyncio
async def test_source_raises_when_session_id_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = _client(handler)
    try:
        source = RobustnessServerSource(client)
        with pytest.raises(SourceUnavailable):
            await source.fetch(_ctx(session_id=""))
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_source_propagates_5xx_as_source_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "boom"})

    client = _client(handler)
    try:
        source = RobustnessServerSource(client)
        with pytest.raises(SourceUnavailable):
            await source.fetch(_ctx())
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_source_skips_summary_when_now_unix_is_zero():
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if "/pods" in request.url.path:
            return httpx.Response(200, json=[])
        if "/events" in request.url.path:
            return httpx.Response(200, json={"events": []})
        return httpx.Response(404)

    client = _client(handler)
    try:
        source = RobustnessServerSource(client)
        data = await source.fetch(_ctx(now_unix=0.0))
    finally:
        await client.aclose()
    assert "/api/v1/sessions/sess-1/summary" not in requested_paths
    assert data.session_summary == {}


# ---------------------------------------------------------------------------
# M2: cluster proxy methods + RobustnessServerSource cluster_faults fetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_cluster_pod_metrics_forwards_window_and_categories():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = str(request.url.query)
        return httpx.Response(200, json={"data": {"pods": []}})

    client = _client(handler)
    from robustness_agent.sources.server_client import _MetricsWindow

    try:
        body = await client.get_cluster_pod_metrics(
            "ns1",
            "podA",
            _MetricsWindow(start_unix=100, end_unix=200),
            categories=["gpu"],
            step="15s",
        )
    finally:
        await client.aclose()
    assert seen["path"] == "/api/v1/cluster/pods/ns1/podA/metrics"
    assert "start=100" in seen["query"] and "end=200" in seen["query"]
    assert "step=15s" in seen["query"]
    assert "categories=gpu" in seen["query"]
    assert body == {"data": {"pods": []}}


@pytest.mark.asyncio
async def test_list_cluster_pod_metric_categories_unwraps_available():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "namespace": "ns1",
                "name": "podA",
                "available": [{"name": "gpu_temp", "category": "gpu"}],
            },
        )

    client = _client(handler)
    from robustness_agent.sources.server_client import _MetricsWindow

    try:
        out = await client.list_cluster_pod_metric_categories(
            "ns1", "podA", _MetricsWindow(start_unix=10, end_unix=20)
        )
    finally:
        await client.aclose()
    assert out == [{"name": "gpu_temp", "category": "gpu"}]


@pytest.mark.asyncio
async def test_get_cluster_workload_hierarchy_returns_dict():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "workload_id": "wl-1",
                "pods": [{"namespace": "ns1", "name": "podA"}],
            },
        )

    client = _client(handler)
    try:
        body = await client.get_cluster_workload_hierarchy("wl-1")
    finally:
        await client.aclose()
    assert body == {
        "workload_id": "wl-1",
        "pods": [{"namespace": "ns1", "name": "podA"}],
    }


@pytest.mark.asyncio
async def test_list_cluster_faults_forwards_filters_and_unwraps():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = str(request.url.query)
        return httpx.Response(
            200,
            json={
                "faults": [{"name": "g1-ecc", "phase": "Isolating"}],
                "total_count": 1,
            },
        )

    client = _client(handler)
    try:
        faults = await client.list_cluster_faults(
            since="1700000000", node="g1", phase="Isolating", page_size=100
        )
    finally:
        await client.aclose()
    assert seen["path"] == "/api/v1/cluster/faults"
    assert "since=1700000000" in seen["query"]
    assert "node=g1" in seen["query"]
    assert "phase=Isolating" in seen["query"]
    assert "page_size=100" in seen["query"]
    assert faults == [{"name": "g1-ecc", "phase": "Isolating"}]


@pytest.mark.asyncio
async def test_list_cluster_faults_handles_bare_array_response():
    """Be tolerant of older robust-api builds that return a list."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"name": "g1-ecc"}, {"name": "g2-ecc"}])

    client = _client(handler)
    try:
        faults = await client.list_cluster_faults()
    finally:
        await client.aclose()
    assert faults == [{"name": "g1-ecc"}, {"name": "g2-ecc"}]


@pytest.mark.asyncio
async def test_source_fetch_populates_cluster_faults():
    """The Source adapter calls /cluster/faults and surfaces them."""

    paths_hit: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths_hit.append(request.url.path)
        if "/sessions/sess-1/pods" in request.url.path:
            return httpx.Response(200, json=[])
        if "/sessions/sess-1/events" in request.url.path:
            return httpx.Response(200, json={"events": []})
        if "/sessions/sess-1/summary" in request.url.path:
            return httpx.Response(200, json={})
        if request.url.path == "/api/v1/cluster/faults":
            return httpx.Response(
                200,
                json={
                    "faults": [
                        {
                            "name": "g53-gpu_ecc",
                            "monitor_id": "gpu_ecc",
                            "node_name": "g53",
                            "phase": "Isolating",
                            "auto_repair": True,
                            "affected_workload_count": 2,
                            "affected_gpu_count": 4,
                        }
                    ],
                    "total_count": 1,
                },
            )
        return httpx.Response(404)

    client = _client(handler)
    try:
        source = RobustnessServerSource(client)
        data = await source.fetch(_ctx())
    finally:
        await client.aclose()

    assert "/api/v1/cluster/faults" in paths_hit
    assert len(data.cluster_faults) == 1
    assert data.cluster_faults[0]["phase"] == "Isolating"
    assert data.sources_used == ["robustness-server"]


@pytest.mark.asyncio
async def test_source_propagates_cluster_faults_5xx_as_source_unavailable():
    """A 5xx on /cluster/faults must trigger DegradeRouter, not be swallowed."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "/sessions/sess-1/pods" in request.url.path:
            return httpx.Response(200, json=[])
        if "/sessions/sess-1/events" in request.url.path:
            return httpx.Response(200, json={"events": []})
        if "/sessions/sess-1/summary" in request.url.path:
            return httpx.Response(200, json={})
        if request.url.path == "/api/v1/cluster/faults":
            return httpx.Response(503, json={"detail": "robust-api down"})
        return httpx.Response(404)

    client = _client(handler)
    try:
        source = RobustnessServerSource(client)
        with pytest.raises(SourceUnavailable):
            await source.fetch(_ctx())
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_source_can_disable_cluster_faults():
    paths_hit: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths_hit.append(request.url.path)
        if "/sessions/sess-1/pods" in request.url.path:
            return httpx.Response(200, json=[])
        if "/sessions/sess-1/events" in request.url.path:
            return httpx.Response(200, json={"events": []})
        if "/sessions/sess-1/summary" in request.url.path:
            return httpx.Response(200, json={})
        return httpx.Response(404)

    client = _client(handler)
    try:
        source = RobustnessServerSource(client, enable_cluster_faults=False)
        data = await source.fetch(_ctx())
    finally:
        await client.aclose()

    assert "/api/v1/cluster/faults" not in paths_hit
    assert data.cluster_faults == []
