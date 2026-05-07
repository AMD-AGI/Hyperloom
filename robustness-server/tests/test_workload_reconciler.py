"""Tests for the workload reconciler.

Both the Robust workload lister (talks via stub ``RobustAPIClient``)
and the ``reconcile_once`` loop (side-effecting, swaps in a fake
assignments repo) are covered here. No Postgres, NATS, or Kubernetes
required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from robustness_server.config import Settings
from robustness_server.models import PodAssignmentSource, PodRef, PodRole
from robustness_server.services import (
    RobustWorkloadLister,
    WorkloadPodAssignment,
    WorkloadReconciler,
)
from robustness_server.services.workload_reconciler import (
    _coerce_role,
    _extract_hierarchy_pod_names,
)


def _settings() -> Settings:
    return Settings(apply_migrations_on_start=False)


def test_coerce_role_known_values_pass_through() -> None:
    assert _coerce_role("brain") == PodRole.BRAIN
    assert _coerce_role("hands_gpu") == PodRole.HANDS_GPU
    assert _coerce_role("hands_cpu") == PodRole.HANDS_CPU


def test_coerce_role_keyword_fallback() -> None:
    assert _coerce_role("GPU-hands") == PodRole.HANDS_GPU
    assert _coerce_role("primary-brain") == PodRole.BRAIN
    assert _coerce_role("foo") == PodRole.HANDS_CPU
    assert _coerce_role(None) == PodRole.HANDS_CPU


def test_extract_hierarchy_pod_names_canonical_and_fallbacks() -> None:
    payload = {
        "workload_id": "w1",
        "pods": [
            {"pod_name": "p1", "pod_uid": "u1"},
            {"name": "p2"},
            "p3",
            {"foo": "bar"},
        ],
    }
    assert list(_extract_hierarchy_pod_names(payload)) == ["p1", "p2", "p3"]


def test_extract_hierarchy_handles_missing_pods_block() -> None:
    assert list(_extract_hierarchy_pod_names({})) == []
    assert list(_extract_hierarchy_pod_names({"pods": None})) == []


class FakeRobustClient:
    """Stub of ``RobustAPIClient`` used by ``RobustWorkloadLister``."""

    def __init__(
        self,
        *,
        workloads: list[dict[str, Any]],
        hierarchies: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._workloads = workloads
        self._hierarchies = hierarchies or {}
        self.list_calls: list[dict[str, Any]] = []
        self.hierarchy_calls: list[str] = []

    async def list_workloads(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.list_calls.append(kwargs)
        return list(self._workloads)

    async def get_workload_hierarchy(
        self, *, workload_id: str
    ) -> dict[str, Any]:
        self.hierarchy_calls.append(workload_id)
        return dict(self._hierarchies.get(workload_id, {"pods": []}))


@pytest.mark.asyncio
async def test_lister_skips_workloads_without_session_label() -> None:
    client = FakeRobustClient(
        workloads=[
            {"uid": "w-no-label", "namespace": "ns", "labels": {"k": "v"}},
            {
                "uid": "w-claw",
                "namespace": "claw-ns",
                "labels": {
                    "primus-claw/session-id": "sess-1",
                    "primus-claw/role": "hands_gpu",
                },
            },
        ],
        hierarchies={
            "w-claw": {"pods": [{"pod_name": "p1"}, {"pod_name": "p2"}]},
        },
    )
    lister = RobustWorkloadLister(settings=_settings(), client=client)  # type: ignore[arg-type]

    out = await lister()

    assert len(out) == 2
    assert {a.pod.name for a in out} == {"p1", "p2"}
    assert all(a.session_id == "sess-1" for a in out)
    assert all(a.pod.namespace == "claw-ns" for a in out)
    assert all(a.role == PodRole.HANDS_GPU for a in out)
    # Hierarchy fetch happens only for the matched workload.
    assert client.hierarchy_calls == ["w-claw"]


@pytest.mark.asyncio
async def test_lister_namespace_falls_back_to_default() -> None:
    client = FakeRobustClient(
        workloads=[
            {
                "uid": "w-no-ns",
                "labels": {
                    "primus-claw/session-id": "sess-x",
                },
            },
        ],
        hierarchies={"w-no-ns": {"pods": [{"pod_name": "p1"}]}},
    )
    lister = RobustWorkloadLister(settings=_settings(), client=client)  # type: ignore[arg-type]
    out = await lister()
    assert out and out[0].pod.namespace == "default"
    assert out[0].role == PodRole.HANDS_CPU  # default role


@pytest.mark.asyncio
async def test_lister_swallows_list_failure() -> None:
    class FailClient(FakeRobustClient):
        async def list_workloads(self, **kwargs: Any):  # type: ignore[override]
            from robustness_server.services.robust_client import RobustAPIError

            raise RobustAPIError("boom")

    client = FailClient(workloads=[])
    lister = RobustWorkloadLister(settings=_settings(), client=client)  # type: ignore[arg-type]
    assert await lister() == []


class FakeAssignments:
    def __init__(self) -> None:
        self.opens: list[dict[str, Any]] = []
        self.expired: list[dict[str, Any]] = []

    async def open_assignment(self, **kwargs: Any) -> int:
        self.opens.append(kwargs)
        return len(self.opens)

    async def expire_stale_open(self, **kwargs: Any) -> int:
        self.expired.append(kwargs)
        return 0


@pytest.mark.asyncio
async def test_reconcile_opens_each_assignment_with_workload_source() -> None:
    repo = FakeAssignments()

    async def list_fn() -> list[WorkloadPodAssignment]:
        return [
            WorkloadPodAssignment(
                session_id="s1",
                pod=PodRef(namespace="claw", name="p1"),
                role=PodRole.HANDS_GPU,
            )
        ]

    rec = WorkloadReconciler(
        settings=_settings(),
        assignments=repo,  # type: ignore[arg-type]
        list_fn=list_fn,
        grace_period_seconds=10.0,
    )
    opened = await rec.reconcile_once()
    assert opened == 1
    assert repo.opens[0]["source"] == PodAssignmentSource.WORKLOAD_RECONCILE
    assert repo.opens[0]["role"] == PodRole.HANDS_GPU


@pytest.mark.asyncio
async def test_reconcile_expires_stale_with_workload_source_filter() -> None:
    repo = FakeAssignments()

    async def list_fn() -> list[WorkloadPodAssignment]:
        return []

    rec = WorkloadReconciler(
        settings=_settings(),
        assignments=repo,  # type: ignore[arg-type]
        list_fn=list_fn,
        grace_period_seconds=5.0,
    )
    opened = await rec.reconcile_once()
    assert opened == 0
    assert len(repo.expired) == 1
    assert (
        repo.expired[0]["source"] == PodAssignmentSource.WORKLOAD_RECONCILE
    )
    assert repo.expired[0]["last_seen_before"] < datetime.now(tz=timezone.utc)


@pytest.mark.asyncio
async def test_reconcile_swallows_list_fn_errors() -> None:
    repo = FakeAssignments()

    async def list_fn() -> list[WorkloadPodAssignment]:
        raise RuntimeError("upstream down")

    rec = WorkloadReconciler(
        settings=_settings(),
        assignments=repo,  # type: ignore[arg-type]
        list_fn=list_fn,
    )
    opened = await rec.reconcile_once()
    assert opened == 0
    assert repo.opens == []
    assert repo.expired == []
