# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the robustness-server client + Source adapter."""

from __future__ import annotations


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


# ---------------------------------------------------------------------------
# M2.5: cluster pod metrics fan-out -> SourceData.local_gpu
# ---------------------------------------------------------------------------


def _gpu_metric_response(value: float, *, gpu_id: str = "0", ts: int = 100):
    """Build a robust-api-shaped pod-metrics response for one GPU."""

    return {
        "data": {
            "pods": [
                {
                    "namespace": "ns1",
                    "name": "podA",
                    "results": [
                        {
                            "name": "rocm_temperature_celsius",
                            "category": "gpu",
                            "unit": "C",
                            "series": [
                                {
                                    "labels": {"gpu": gpu_id},
                                    "values": [
                                        {"timestamp": ts, "value": value}
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    }


@pytest.mark.asyncio
async def test_source_disables_cluster_pod_metrics_by_default():
    """The fan-out costs one HTTP call per pod; default off."""

    paths_hit: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths_hit.append(request.url.path)
        if "/sessions/sess-1/pods" in request.url.path:
            return httpx.Response(
                200,
                json=[{"pod": {"namespace": "ns1", "name": "podA"}}],
            )
        if "/sessions/sess-1/events" in request.url.path:
            return httpx.Response(200, json={"events": []})
        if "/sessions/sess-1/summary" in request.url.path:
            return httpx.Response(200, json={})
        if request.url.path == "/api/v1/cluster/faults":
            return httpx.Response(200, json={"faults": []})
        return httpx.Response(404)

    client = _client(handler)
    try:
        source = RobustnessServerSource(client)
        data = await source.fetch(_ctx())
    finally:
        await client.aclose()

    assert not any("/cluster/pods/" in p for p in paths_hit)
    assert data.local_gpu == {}


@pytest.mark.asyncio
async def test_source_fans_out_pod_metrics_when_enabled():
    """With enable_cluster_pod_metrics=True, fetch hits /cluster/pods/{ns}/{name}/metrics."""

    paths_hit: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths_hit.append(request.url.path)
        if "/sessions/sess-1/pods" in request.url.path:
            return httpx.Response(
                200,
                json=[{"pod": {"namespace": "ns1", "name": "podA"}}],
            )
        if "/sessions/sess-1/events" in request.url.path:
            return httpx.Response(200, json={"events": []})
        if "/sessions/sess-1/summary" in request.url.path:
            return httpx.Response(200, json={})
        if request.url.path == "/api/v1/cluster/faults":
            return httpx.Response(200, json={"faults": []})
        if request.url.path == "/api/v1/cluster/pods/ns1/podA/metrics":
            return httpx.Response(200, json=_gpu_metric_response(95.0))
        return httpx.Response(404)

    client = _client(handler)
    try:
        source = RobustnessServerSource(client, enable_cluster_pod_metrics=True)
        data = await source.fetch(_ctx())
    finally:
        await client.aclose()

    assert "/api/v1/cluster/pods/ns1/podA/metrics" in paths_hit
    assert data.local_gpu["tool"] == "robust-api"
    assert len(data.local_gpu["gpus"]) == 1
    assert data.local_gpu["gpus"][0]["temperature_c"] == 95.0
    assert data.local_gpu["gpus"][0]["pod_name"] == "podA"


@pytest.mark.asyncio
async def test_source_pod_metrics_5xx_propagates_for_degrade():
    """Transport / 5xx on cluster metrics still triggers DegradeRouter."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "/sessions/sess-1/pods" in request.url.path:
            return httpx.Response(
                200,
                json=[{"pod": {"namespace": "ns1", "name": "podA"}}],
            )
        if "/sessions/sess-1/events" in request.url.path:
            return httpx.Response(200, json={"events": []})
        if "/sessions/sess-1/summary" in request.url.path:
            return httpx.Response(200, json={})
        if request.url.path == "/api/v1/cluster/faults":
            return httpx.Response(200, json={"faults": []})
        if request.url.path == "/api/v1/cluster/pods/ns1/podA/metrics":
            return httpx.Response(503, text="busy")
        return httpx.Response(404)

    client = _client(handler)
    try:
        source = RobustnessServerSource(client, enable_cluster_pod_metrics=True)
        with pytest.raises(SourceUnavailable):
            await source.fetch(_ctx())
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_source_pod_metrics_dedups_repeated_pod_refs():
    """Same pod appearing twice in session_pods should fan out once."""

    seen_metrics_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "/sessions/sess-1/pods" in request.url.path:
            return httpx.Response(
                200,
                json=[
                    {"pod": {"namespace": "ns1", "name": "podA"}},
                    {"pod": {"namespace": "ns1", "name": "podA"}},
                    {"pod": {"namespace": "ns1", "name": "podB"}},
                ],
            )
        if "/sessions/sess-1/events" in request.url.path:
            return httpx.Response(200, json={"events": []})
        if "/sessions/sess-1/summary" in request.url.path:
            return httpx.Response(200, json={})
        if request.url.path == "/api/v1/cluster/faults":
            return httpx.Response(200, json={"faults": []})
        if "/api/v1/cluster/pods/" in request.url.path:
            seen_metrics_paths.append(request.url.path)
            return httpx.Response(200, json=_gpu_metric_response(80.0))
        return httpx.Response(404)

    client = _client(handler)
    try:
        source = RobustnessServerSource(client, enable_cluster_pod_metrics=True)
        await source.fetch(_ctx())
    finally:
        await client.aclose()

    assert sorted(seen_metrics_paths) == [
        "/api/v1/cluster/pods/ns1/podA/metrics",
        "/api/v1/cluster/pods/ns1/podB/metrics",
    ]


@pytest.mark.asyncio
async def test_source_pod_metrics_caps_fan_out_per_tick():
    """Sessions with too many pods must not blow the per-tick budget."""

    metrics_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal metrics_calls
        if "/sessions/sess-1/pods" in request.url.path:
            return httpx.Response(
                200,
                json=[
                    {"pod": {"namespace": "ns1", "name": f"pod-{i:02d}"}}
                    for i in range(10)
                ],
            )
        if "/sessions/sess-1/events" in request.url.path:
            return httpx.Response(200, json={"events": []})
        if "/sessions/sess-1/summary" in request.url.path:
            return httpx.Response(200, json={})
        if request.url.path == "/api/v1/cluster/faults":
            return httpx.Response(200, json={"faults": []})
        if "/api/v1/cluster/pods/" in request.url.path:
            metrics_calls += 1
            return httpx.Response(200, json=_gpu_metric_response(80.0))
        return httpx.Response(404)

    client = _client(handler)
    try:
        source = RobustnessServerSource(
            client,
            enable_cluster_pod_metrics=True,
            max_pods_per_tick=3,
        )
        await source.fetch(_ctx())
    finally:
        await client.aclose()

    assert metrics_calls == 3


@pytest.mark.asyncio
async def test_server_pod_metrics_drive_local_health_gpu_signal():
    """End-to-end: server-decoded GPU >= warn threshold fires gpu_thermal_high."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "/sessions/sess-1/pods" in request.url.path:
            return httpx.Response(
                200,
                json=[{"pod": {"namespace": "ns1", "name": "podA"}}],
            )
        if "/sessions/sess-1/events" in request.url.path:
            return httpx.Response(200, json={"events": []})
        if "/sessions/sess-1/summary" in request.url.path:
            return httpx.Response(200, json={})
        if request.url.path == "/api/v1/cluster/faults":
            return httpx.Response(200, json={"faults": []})
        if request.url.path == "/api/v1/cluster/pods/ns1/podA/metrics":
            # 95 C  -> warn (>= 90) but below crit (100) -> medium
            return httpx.Response(200, json=_gpu_metric_response(95.0))
        return httpx.Response(404)

    client = _client(handler)
    try:
        source = RobustnessServerSource(client, enable_cluster_pod_metrics=True)
        data = await source.fetch(_ctx())
    finally:
        await client.aclose()

    from robustness_agent.signals import (
        Classifier,
        SymptomSeverity,
    )

    classifier = Classifier()
    symptoms = classifier.classify(data, _ctx())
    thermal = [s for s in symptoms if s.name == "gpu_thermal_high"]
    assert len(thermal) == 1
    assert thermal[0].severity is SymptomSeverity.MEDIUM
    assert thermal[0].evidence["temperature_c"] == 95.0


# ---------------------------------------------------------------------------
# M2 multi-node: workload_uid -> /cluster/workloads/{uid}/hierarchy fan-out
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_source_workload_uid_merges_hierarchy_pods_into_session_pods():
    """Hierarchy-derived workers are added even when session_pods skipped them."""

    paths_hit: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths_hit.append(request.url.path)
        if "/sessions/sess-1/pods" in request.url.path:
            return httpx.Response(
                200,
                json=[{"pod": {"namespace": "ns1", "name": "head-pod"}}],
            )
        if "/sessions/sess-1/events" in request.url.path:
            return httpx.Response(200, json={"events": []})
        if "/sessions/sess-1/summary" in request.url.path:
            return httpx.Response(200, json={})
        if request.url.path == "/api/v1/cluster/workloads/wl-1/hierarchy":
            return httpx.Response(
                200,
                json={
                    "workload_id": "wl-1",
                    "pods": [
                        {"namespace": "ns1", "name": "head-pod"},
                        {"namespace": "ns1", "name": "worker-0"},
                        {"namespace": "ns1", "name": "worker-1"},
                    ],
                },
            )
        if request.url.path == "/api/v1/cluster/faults":
            return httpx.Response(200, json={"faults": []})
        return httpx.Response(404)

    client = _client(handler)
    try:
        source = RobustnessServerSource(client, workload_uid="wl-1")
        data = await source.fetch(_ctx())
    finally:
        await client.aclose()

    assert "/api/v1/cluster/workloads/wl-1/hierarchy" in paths_hit
    pod_refs = {
        (entry["pod"]["namespace"], entry["pod"]["name"])
        for entry in data.session_pods
    }
    assert pod_refs == {
        ("ns1", "head-pod"),
        ("ns1", "worker-0"),
        ("ns1", "worker-1"),
    }


@pytest.mark.asyncio
async def test_source_workload_uid_drives_multi_node_pod_metric_fan_out():
    """Cluster pod-metrics fans out across hierarchy-only pods, not just session_pods."""

    metric_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "/sessions/sess-1/pods" in request.url.path:
            # Session only knows the head pod; workers exist only in the
            # cluster hierarchy view.
            return httpx.Response(
                200,
                json=[{"pod": {"namespace": "ns1", "name": "head"}}],
            )
        if "/sessions/sess-1/events" in request.url.path:
            return httpx.Response(200, json={"events": []})
        if "/sessions/sess-1/summary" in request.url.path:
            return httpx.Response(200, json={})
        if request.url.path == "/api/v1/cluster/workloads/wl-1/hierarchy":
            return httpx.Response(
                200,
                json={
                    "workload_id": "wl-1",
                    "pods": [
                        {"namespace": "ns1", "name": "head"},
                        {"namespace": "ns1", "name": "worker-0"},
                    ],
                },
            )
        if request.url.path == "/api/v1/cluster/faults":
            return httpx.Response(200, json={"faults": []})
        if "/api/v1/cluster/pods/" in request.url.path:
            metric_paths.append(request.url.path)
            return httpx.Response(200, json=_gpu_metric_response(80.0))
        return httpx.Response(404)

    client = _client(handler)
    try:
        source = RobustnessServerSource(
            client,
            workload_uid="wl-1",
            enable_cluster_pod_metrics=True,
        )
        await source.fetch(_ctx())
    finally:
        await client.aclose()

    assert sorted(metric_paths) == [
        "/api/v1/cluster/pods/ns1/head/metrics",
        "/api/v1/cluster/pods/ns1/worker-0/metrics",
    ]


@pytest.mark.asyncio
async def test_source_workload_uid_hierarchy_5xx_triggers_degrade():
    """A 5xx on the hierarchy endpoint must propagate as SourceUnavailable."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "/sessions/sess-1/pods" in request.url.path:
            return httpx.Response(200, json=[])
        if "/sessions/sess-1/events" in request.url.path:
            return httpx.Response(200, json={"events": []})
        if "/sessions/sess-1/summary" in request.url.path:
            return httpx.Response(200, json={})
        if request.url.path == "/api/v1/cluster/workloads/wl-1/hierarchy":
            return httpx.Response(503, text="busy")
        return httpx.Response(404)

    client = _client(handler)
    try:
        source = RobustnessServerSource(client, workload_uid="wl-1")
        with pytest.raises(SourceUnavailable):
            await source.fetch(_ctx())
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_source_no_workload_uid_skips_hierarchy_call():
    """Single-node path keeps the existing list_session_pods behaviour."""

    paths_hit: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths_hit.append(request.url.path)
        if "/sessions/sess-1/pods" in request.url.path:
            return httpx.Response(
                200,
                json=[{"pod": {"namespace": "ns1", "name": "head"}}],
            )
        if "/sessions/sess-1/events" in request.url.path:
            return httpx.Response(200, json={"events": []})
        if "/sessions/sess-1/summary" in request.url.path:
            return httpx.Response(200, json={})
        if request.url.path == "/api/v1/cluster/faults":
            return httpx.Response(200, json={"faults": []})
        return httpx.Response(404)

    client = _client(handler)
    try:
        source = RobustnessServerSource(client)
        data = await source.fetch(_ctx())
    finally:
        await client.aclose()

    assert not any("workloads" in p for p in paths_hit)
    assert len(data.session_pods) == 1
    assert data.session_pods[0]["pod"]["name"] == "head"
