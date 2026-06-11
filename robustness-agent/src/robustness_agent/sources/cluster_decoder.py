# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Decode robust-api raw responses into LocalProbe-equivalent schemas.

robust-api emits Prometheus-style time-series via ``pod-metrics/batch``;
this module flattens them to one row per device (latest value per
metric) so ``signals/local_health.py`` is agnostic to which source
filled :data:`SourceData.local_gpu`. GPU metrics only for now.
"""

from __future__ import annotations

from typing import Any, Mapping


# Map of metric_name -> SourceData.local_gpu field. Keys cover the three
# exporter conventions seen on core42 (rocm exporter, DCGM, generic) so
# signals work uniformly regardless of who scraped the raw counter.
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


# Series labels used to deduce gpu_id; first match wins. Exporters
# disagree on the label name so we accept all conventions.
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

    Each gpu row carries the same fields the LocalProbe rocm-smi parser
    produces (``gpu_id``, ``temperature_c``, ``util_gpu_pct`` etc.) plus
    ``pod_namespace`` / ``pod_name`` so signals can pinpoint where the
    heat is coming from. Rows are keyed by ``(namespace, name, gpu_id)``
    so two pods sharing a GPU id on one node do not collide.

    Args:
        response (Mapping[str, Any] | None): The raw
            ``pod-metrics/batch`` response, expected to nest the per-pod
            results under ``data.pods``. Any other shape yields ``{}``.

    Returns:
        dict[str, Any]: ``{"gpus": [...], "tool": "robust-api"}`` with
        one row per decoded device, or ``{}`` when no GPU metric is
        found.
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
                # Key by (ns, name, gpu_id) so same-node pods with overlapping IDs don't collide.
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

    Later writes win on field clashes per (pod, gpu_id); distinct pods
    stay distinct so signal evidence keeps the GPU's namespace / name.
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
    # Stable ordering by (pod_namespace, pod_name, gpu_id).
    rows.sort(
        key=lambda r: (
            str(r.get("pod_namespace") or ""),
            str(r.get("pod_name") or ""),
            str(r.get("gpu_id") or ""),
        )
    )
    return {"gpus": rows, "tool": "robust-api"}


def _extract_gpu_id(labels: Any) -> str:
    """Deduce a GPU id string from a Prometheus series' labels.

    Walks :data:`_GPU_ID_LABELS` in priority order so the differing
    exporter conventions (rocm / DCGM / generic) resolve to one id.

    Args:
        labels (Any): The ``labels`` mapping from a metric series.
            Non-mapping values yield an empty string.

    Returns:
        str: The first matching label's value as a string, or ``""``
        when no known label is present.
    """
    if not isinstance(labels, Mapping):
        return ""
    for key in _GPU_ID_LABELS:
        if key in labels:
            return str(labels[key])
    return ""


def _coerce_int_id(raw: str) -> int | str:
    """Coerce a GPU id to ``int`` when numeric, else keep it as a string.

    Args:
        raw (str): The raw GPU id extracted from a series label.

    Returns:
        int | str: The integer form when ``raw`` parses as an int,
        otherwise ``raw`` unchanged.
    """
    try:
        return int(raw)
    except (TypeError, ValueError):
        return raw


def _latest_value(values: Any) -> float | None:
    """Return the most recent numeric value from a metric series.

    Scans the ``values`` list for the entry with the highest
    ``timestamp`` whose ``value`` coerces to ``float``.

    Args:
        values (Any): The series ``values`` list, each entry expected
            to be a mapping with ``timestamp`` and ``value`` keys.

    Returns:
        float | None: The value at the latest timestamp, or ``None``
        when the list is empty or carries no usable entry.
    """
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
