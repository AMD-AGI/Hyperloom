"""Periodic workload reconciler.

Originally this module reconciled SaFE Workload CRDs directly. In the
core42 deployment, the data-plane cluster does not host SaFE CRDs (they
live on the management cluster), and we want to keep the
``robustness-server`` pod free of cross-cluster Kubernetes credentials.

The reconciler now polls Primus-Robust's workload API
(``GET /api/v1/workloads`` + ``GET /api/v1/workloads/{id}/hierarchy``)
which is already populated by Robust's own K8s watcher, and is the
single source of truth for "which pods belong to which workload" in the
Robust pipeline. Talking to Robust over HTTP keeps the dependency graph
shallow:

    NATS events  ─┐
                  ├─►  hyperloom-robustness-server  ──►  PG (sessions /
    Robust pods  ─┘                                     assignments)

The conceptual responsibility ("eventual consistency for session ↔ pod
pairing when a NATS event was lost") is unchanged, so the reconciler
loop and grace-period logic are reused as-is. Only the lister strategy
(``WorkloadListFn``) was swapped out.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterable

from ..config import Settings
from ..models import (
    PodAssignmentSource,
    PodRef,
    PodRole,
)
from ..store import AssignmentsRepository
from .robust_client import RobustAPIClient, RobustAPIError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkloadPodAssignment:
    """One row produced by the workload listing step."""

    session_id: str
    pod: PodRef
    role: PodRole


WorkloadListFn = Callable[[], Awaitable[list[WorkloadPodAssignment]]]
"""Strategy used by the reconciler to enumerate current assignments.

A function (rather than a method on a Workload client) keeps the
backing source pluggable: production wires ``RobustWorkloadLister``;
tests pass a lambda returning canned data.
"""


class WorkloadReconciler:
    """Periodic loop that pulls workload state into the DB."""

    def __init__(
        self,
        *,
        settings: Settings,
        assignments: AssignmentsRepository,
        list_fn: WorkloadListFn,
        grace_period_seconds: float = 30.0,
    ) -> None:
        self._settings = settings
        self._assignments = assignments
        self._list_fn = list_fn
        # Pods absent from the lister for at least this long are
        # closed. Larger than the reconcile interval so a one-tick
        # blip does not flap an assignment.
        self._grace_period = timedelta(seconds=grace_period_seconds)
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="workload-reconciler")
        logger.info(
            "Workload reconciler started interval=%.1fs grace=%.1fs",
            self._settings.workload_reconcile_interval_seconds,
            self._grace_period.total_seconds(),
        )

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        interval = self._settings.workload_reconcile_interval_seconds
        while not self._stop_event.is_set():
            try:
                await self.reconcile_once()
            except Exception:
                logger.exception("workload reconcile tick failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=interval
                )
            except asyncio.TimeoutError:
                continue

    async def reconcile_once(self) -> int:
        """Run one reconciliation pass; returns rows opened.

        Exposed for the integration tests so they can drive the loop
        deterministically without sleeping.
        """

        observed_at = datetime.now(tz=timezone.utc)
        try:
            assignments = await self._list_fn()
        except Exception:
            logger.exception("workload list failed; skipping reconcile tick")
            return 0

        opened = 0
        for entry in assignments:
            try:
                await self._assignments.open_assignment(
                    session_id=entry.session_id,
                    pod=entry.pod,
                    role=entry.role,
                    source=PodAssignmentSource.WORKLOAD_RECONCILE,
                    observed_at=observed_at,
                )
                opened += 1
            except Exception:
                logger.exception(
                    "open_assignment failed (session=%s pod=%s/%s)",
                    entry.session_id,
                    entry.pod.namespace,
                    entry.pod.name,
                )

        cutoff = observed_at - self._grace_period
        try:
            closed = await self._assignments.expire_stale_open(
                source=PodAssignmentSource.WORKLOAD_RECONCILE,
                last_seen_before=cutoff,
                closed_at=observed_at,
            )
            if closed:
                logger.info(
                    "workload reconciler closed %d stale assignment(s)",
                    closed,
                )
        except Exception:
            logger.exception("expire_stale_open failed")

        return opened


class RobustWorkloadLister:
    """Default ``WorkloadListFn`` implementation backed by Robust API.

    Walks ``GET /api/v1/workloads`` (Robust's workload catalogue,
    populated by its own k8s-watcher) and pulls hierarchy/pod lists
    on-demand. The Robust workload row already carries
    ``namespace`` / ``labels`` / ``state`` so we can filter to running
    Claw-managed sessions without touching Kubernetes.

    Filter knobs intentionally mirror Robust's query parameters so the
    server doesn't have to walk inactive history every minute.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        client: RobustAPIClient,
        state_filter: str = "RUNNING",
        page_limit: int = 200,
    ) -> None:
        self._settings = settings
        self._client = client
        self._state_filter = state_filter
        self._page_limit = page_limit

    async def __call__(self) -> list[WorkloadPodAssignment]:
        try:
            workloads = await self._client.list_workloads(
                state=self._state_filter,
                limit=self._page_limit,
            )
        except RobustAPIError:
            logger.exception("robust list_workloads failed; reconcile skipped")
            return []

        # Pre-filter to workloads that look Claw-managed before
        # spending a hierarchy fetch each.
        session_label = self._settings.claw_session_label
        role_label = self._settings.claw_role_label
        candidates: list[tuple[dict[str, Any], str, PodRole, str]] = []
        for wl in workloads:
            labels = wl.get("labels") or {}
            session_id = (
                labels.get(session_label) if isinstance(labels, dict) else None
            )
            if not session_id:
                continue
            role = _coerce_role(
                labels.get(role_label) if isinstance(labels, dict) else None
            )
            namespace = wl.get("namespace") or "default"
            workload_id = str(wl.get("uid") or wl.get("id") or "")
            if not workload_id:
                continue
            candidates.append((wl, session_id, role, namespace))

        out: list[WorkloadPodAssignment] = []
        for wl, session_id, role, namespace in candidates:
            workload_id = str(wl.get("uid") or wl.get("id"))
            try:
                hierarchy = await self._client.get_workload_hierarchy(
                    workload_id=workload_id,
                )
            except RobustAPIError:
                logger.warning(
                    "skip workload %s: hierarchy fetch failed", workload_id
                )
                continue
            for pod_name in _extract_hierarchy_pod_names(hierarchy):
                out.append(
                    WorkloadPodAssignment(
                        session_id=session_id,
                        pod=PodRef(namespace=namespace, name=pod_name),
                        role=role,
                    )
                )
        return out


def _coerce_role(raw: str | None) -> PodRole:
    if not raw:
        return PodRole.HANDS_CPU
    candidate = raw.strip().lower()
    try:
        return PodRole(candidate)
    except ValueError:
        if "gpu" in candidate:
            return PodRole.HANDS_GPU
        if "brain" in candidate:
            return PodRole.BRAIN
        return PodRole.HANDS_CPU


def _extract_hierarchy_pod_names(hierarchy: dict[str, Any]) -> Iterable[str]:
    """Pull pod names out of a robust-api workload hierarchy payload.

    The robust-api ``hierarchy`` endpoint returns
    ``{workload_id, pods: [{pod_name, pod_uid, ...}], pod_count}``. We
    accept ``pod_name`` (canonical) and ``name`` (defensive) so the
    function survives a minor field rename without breaking.
    """

    pods = hierarchy.get("pods") or []
    for entry in pods:
        if isinstance(entry, dict):
            name = entry.get("pod_name") or entry.get("name")
            if name:
                yield str(name)
        elif isinstance(entry, str):
            yield entry
