"""Decode robust-api raw responses into LocalProbe-equivalent schemas.

robust-api emits Prometheus-style time-series via
``pod-metrics/batch`` (proxied as ``/api/v1/cluster/pods/.../metrics``
in M2). The agent's local signals consume a flatter snapshot — one
row per device with the latest value of each metric — so this module
bridges the two so ``signals/local_health.py`` does not have to
care which source filled :data:`SourceData.local_gpu`.

We currently decode GPU metrics only; disk / log mappings can be
added the same way once we have a stable upstream catalogue.
"""

from __future__ import annotations

from typing import Any, Mapping


# Map of metric_name -> SourceData.local_gpu field. Adding a new
# metric is a one-line addition; the signal layer never sees this.
# Keys cover the three exporter conventions we currently meet on
# core42 (rocm exporter, DCGM, generic) so signals work uniformly
# across nodes regardless of who scraped the raw counter.
_GPU_METRIC_FIELD: Mapping[str, str] = {
    # rocm-exporter
    "rocm_temperature_celsius": "temperature_c",
    "rocm_temperature_edge_celsius": "temperature_c",
    "rocm_temperature_junction_celsius": "temperature_junction_c",
    "rocm_temperature_memory_celsius": "temperature_memory_c",
    "rocm_gpu_utilization": "util_gpu_pct",
    "rocm_memory_utilization": "util_mem_pct",
    "rocm_power_average_watts": "power_watts",
    # NVIDIA DCGM exporter
    "DCGM_FI_DEV_GPU_TEMP": "temperature_c",
    "DCGM_FI_DEV_MEMORY_TEMP": "temperature_memory_c",
    "DCGM_FI_DEV_GPU_UTIL": "util_gpu_pct",
    "DCGM_FI_DEV_MEM_COPY_UTIL": "util_mem_pct",
    "DCGM_FI_DEV_POWER_USAGE": "power_watts",
    # Generic / re-labelled
    "gpu_temperature_celsius": "temperature_c",
    "gpu_temperature_c": "temperature_c",
    "gpu_util_percent": "util_gpu_pct",
    "gpu_memory_util_percent": "util_mem_pct",
}


# Series labels we look at to deduce gpu_id; first match wins.
# robust-api / Prometheus exporters disagree on the label name so we
# accept all three conventions.
_GPU_ID_LABELS: tuple[str, ...] = (
    "gpu",
    "device",
    "device_id",
    "minor",
    "minor_number",
    "DCGM_FI_DRIVER_DEVICE_ID",
    "card",
)


def decode_gpu_snapshot(
    response: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Decode pod-metrics/batch response into LocalProbe local_gpu shape.

    Returns ``{"gpus": [...], "tool": "robust-api"}`` or ``{}`` when
    no GPU metric is found. Each gpu row carries the same fields the
    LocalProbe rocm-smi parser produces (``gpu_id``, ``temperature_c``,
    ``util_gpu_pct`` etc.) plus ``pod_namespace`` / ``pod_name`` so
    signals can pinpoint where the heat is coming from.
    """

    if not isinstance(response, Mapping):
        return {}
    data = response.get("data")
    if not isinstance(data, Mapping):
        return {}
    pod_results = data.get("pods")
    if not isinstance(pod_results, list):
        return {}

    by_id: dict[tuple[str, str, str], dict[str, Any]] = {}

    for pod in pod_results:
        if not isinstance(pod, Mapping):
            continue
        ns = str(pod.get("namespace") or "")
        name = str(pod.get("name") or "")
        results = pod.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, Mapping):
                continue
            metric_name = str(result.get("name") or "")
            field = _GPU_METRIC_FIELD.get(metric_name)
            if field is None:
                continue
            for series in result.get("series") or []:
                if not isinstance(series, Mapping):
                    continue
                gpu_id = _extract_gpu_id(series.get("labels"))
                latest = _latest_value(series.get("values"))
                if latest is None:
                    continue
                # Key by (ns, name, gpu_id) so two pods on the same
                # node with overlapping GPU IDs don't collide. The
                # caller can flatten this if it wants a single
                # node-wide list.
                key = (ns, name, gpu_id)
                snap = by_id.setdefault(
                    key,
                    {
                        "gpu_id": _coerce_int_id(gpu_id),
                        "pod_namespace": ns,
                        "pod_name": name,
                    },
                )
                snap[field] = latest

    if not by_id:
        return {}
    return {
        "gpus": [by_id[k] for k in sorted(by_id)],
        "tool": "robust-api",
    }


def merge_gpu_snapshots(
    snapshots: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine multiple per-pod snapshots into a single ``local_gpu``.

    Used by :class:`RobustnessServerSource` when fan-out across the
    session's pods produces one decoded snapshot each. Later writes
    win on field clashes per (pod, gpu_id), but distinct pods stay
    distinct so the signal evidence keeps the namespace / name the
    GPU lived under.
    """

    rows: list[dict[str, Any]] = []
    for snap in snapshots:
        if not isinstance(snap, Mapping):
            continue
        gpus = snap.get("gpus")
        if not isinstance(gpus, list):
            continue
        for row in gpus:
            if isinstance(row, Mapping):
                rows.append(dict(row))
    if not rows:
        return {}
    # Stable ordering: by (pod_namespace, pod_name, gpu_id) so test
    # assertions are deterministic.
    rows.sort(
        key=lambda r: (
            str(r.get("pod_namespace") or ""),
            str(r.get("pod_name") or ""),
            str(r.get("gpu_id") or ""),
        )
    )
    return {"gpus": rows, "tool": "robust-api"}


def _extract_gpu_id(labels: Any) -> str:
    if not isinstance(labels, Mapping):
        return ""
    for key in _GPU_ID_LABELS:
        if key in labels:
            return str(labels[key])
    return ""


def _coerce_int_id(raw: str) -> int | str:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return raw


def _latest_value(values: Any) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    best_ts = -1
    best_val: float | None = None
    for entry in values:
        if not isinstance(entry, Mapping):
            continue
        ts = entry.get("timestamp")
        if not isinstance(ts, (int, float)):
            continue
        if ts > best_ts:
            try:
                best_val = float(entry.get("value"))
            except (TypeError, ValueError):
                continue
            best_ts = ts
    return best_val


__all__ = ["decode_gpu_snapshot", "merge_gpu_snapshots"]
