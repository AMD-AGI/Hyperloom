"""Tests for the robust-api HTTP client wrapper.

We use ``httpx.MockTransport`` to assert request shapes (URL, body)
and exercise both the happy path (parsed response) and error
propagation (transport failure → ``RobustAPIError``; non-200 →
``RobustAPIError``).
"""

from __future__ import annotations

import json

import httpx
import pytest

from robustness_server.config import Settings
from robustness_server.models import PodRef
from robustness_server.services import (
    PodMetricsRequest,
    RobustAPIClient,
    RobustAPIError,
)


def _make_client(handler) -> tuple[RobustAPIClient, list[httpx.Request]]:
    captured: list[httpx.Request] = []

    def _wrapped(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    transport = httpx.MockTransport(_wrapped)
    settings = Settings(
        robust_api_url="http://robust.test",
        apply_migrations_on_start=False,
    )
    client = RobustAPIClient(
        settings,
        client=httpx.AsyncClient(transport=transport, timeout=5.0),
    )
    return client, captured


@pytest.mark.asyncio
async def test_fetch_batch_skips_request_for_empty_pods() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("should not be called")

    client, captured = _make_client(handler)
    try:
        out = await client.fetch_batch(
            PodMetricsRequest(pods=[], start="1", end="2")
        )
        assert out.data.pods == []
        assert captured == []
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_fetch_batch_posts_pods_payload_and_parses_response() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content.decode("utf-8"))
        assert req.url.path == "/api/v1/pod-metrics/batch"
        assert body["pods"] == [{"namespace": "ns", "name": "p1"}]
        assert body["categories"] == ["cpu"]
        assert body["start"] == "100"
        assert body["end"] == "200"
        assert body["step"] == "30"
        return httpx.Response(
            200,
            json={
                "data": {
                    "pods": [
                        {
                            "namespace": "ns",
                            "name": "p1",
                            "results": [
                                {
                                    "name": "cpu_usage_cores",
                                    "category": "cpu",
                                    "unit": "cores",
                                    "series": [
                                        {
                                            "labels": {"pod": "p1"},
                                            "values": [
                                                {"timestamp": 1, "value": 0.5}
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                }
            },
        )

    client, captured = _make_client(handler)
    try:
        out = await client.fetch_batch(
            PodMetricsRequest(
                pods=[PodRef(namespace="ns", name="p1")],
                categories=["cpu"],
                start="100",
                end="200",
                step="30",
            )
        )
    finally:
        await client.aclose()

    assert len(captured) == 1
    assert len(out.data.pods) == 1
    pod = out.data.pods[0]
    assert pod.namespace == "ns" and pod.name == "p1"
    assert pod.results[0].name == "cpu_usage_cores"
    assert pod.results[0].series[0].values[0].value == 0.5


@pytest.mark.asyncio
async def test_fetch_batch_raises_on_http_error() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="boom")

    client, _ = _make_client(handler)
    try:
        with pytest.raises(RobustAPIError) as info:
            await client.fetch_batch(
                PodMetricsRequest(
                    pods=[PodRef(namespace="ns", name="p1")],
                    start="1",
                    end="2",
                )
            )
        assert "503" in str(info.value)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_list_categories_for_pod_passes_filters_and_returns_available() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/v1/pod-metrics/ns/p1/list"
        assert req.url.params["start"] == "100"
        assert req.url.params["end"] == "200"
        assert req.url.params["categories"] == "cpu,memory"
        return httpx.Response(
            200,
            json={
                "available": [
                    {"name": "cpu_usage_cores", "category": "cpu"},
                    {"name": "memory_working_set_bytes", "category": "memory"},
                ]
            },
        )

    client, _ = _make_client(handler)
    try:
        out = await client.list_categories_for_pod(
            pod=PodRef(namespace="ns", name="p1"),
            categories=["cpu", "memory"],
            start="100",
            end="200",
        )
    finally:
        await client.aclose()

    assert {c["name"] for c in out} == {
        "cpu_usage_cores",
        "memory_working_set_bytes",
    }


@pytest.mark.asyncio
async def test_list_workloads_passes_filters_and_returns_array() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/v1/workloads"
        assert req.url.params["state"] == "RUNNING"
        assert req.url.params["limit"] == "200"
        return httpx.Response(
            200,
            json=[
                {"uid": "w1", "namespace": "ns", "labels": {"k": "v"}},
                {"uid": "w2", "namespace": "ns2"},
            ],
        )

    client, _ = _make_client(handler)
    try:
        out = await client.list_workloads(state="RUNNING", limit=200)
    finally:
        await client.aclose()

    assert [w["uid"] for w in out] == ["w1", "w2"]


@pytest.mark.asyncio
async def test_list_workloads_extracts_data_array_from_dict_envelope() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"uid": "w-only"}], "meta": {"total": 1}},
        )

    client, _ = _make_client(handler)
    try:
        out = await client.list_workloads()
    finally:
        await client.aclose()

    assert [w["uid"] for w in out] == ["w-only"]


@pytest.mark.asyncio
async def test_get_workload_hierarchy_returns_pods_block() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/v1/workloads/w-1/hierarchy"
        return httpx.Response(
            200,
            json={
                "workload_id": "w-1",
                "pods": [
                    {"pod_name": "p-a", "pod_uid": "u-a"},
                    {"pod_name": "p-b"},
                ],
                "pod_count": 2,
            },
        )

    client, _ = _make_client(handler)
    try:
        out = await client.get_workload_hierarchy(workload_id="w-1")
    finally:
        await client.aclose()

    assert out["workload_id"] == "w-1"
    assert [p["pod_name"] for p in out["pods"]] == ["p-a", "p-b"]


@pytest.mark.asyncio
async def test_transport_error_is_wrapped() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client, _ = _make_client(handler)
    try:
        with pytest.raises(RobustAPIError) as info:
            await client.fetch_batch(
                PodMetricsRequest(
                    pods=[PodRef(namespace="ns", name="p1")],
                    start="1",
                    end="2",
                )
            )
        assert "transport error" in str(info.value)
    finally:
        await client.aclose()
