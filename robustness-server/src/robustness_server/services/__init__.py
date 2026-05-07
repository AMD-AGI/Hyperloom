"""Service layer.

Long-lived collaborators (NATS consumer, KV watcher, workload
reconciler, robust-api client) and the orchestration logic that wires
them through the repositories. Each service is a plain class with a
tight interface so unit tests can drive them with fakes / stubs.
"""

from __future__ import annotations

from .kv_watcher import BrainRegistryWatcher
from .nats_consumer import NatsEventConsumer
from .robust_client import (
    PodMetricsRequest,
    PodMetricsResponse,
    RobustAPIClient,
    RobustAPIError,
)
from .session_router import SessionRouter
from .workload_reconciler import (
    RobustWorkloadLister,
    WorkloadListFn,
    WorkloadPodAssignment,
    WorkloadReconciler,
)

__all__ = [
    "BrainRegistryWatcher",
    "NatsEventConsumer",
    "PodMetricsRequest",
    "PodMetricsResponse",
    "RobustAPIClient",
    "RobustAPIError",
    "RobustWorkloadLister",
    "SessionRouter",
    "WorkloadListFn",
    "WorkloadPodAssignment",
    "WorkloadReconciler",
]
