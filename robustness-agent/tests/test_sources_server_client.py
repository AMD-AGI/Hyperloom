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
