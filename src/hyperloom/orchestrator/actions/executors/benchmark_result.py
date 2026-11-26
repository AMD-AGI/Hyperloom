# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A finished round's measurement and its validity, shared by the round executors, plus the multi-node fold of the
pods' GPU sampler CSVs into a round's directory."""

from __future__ import annotations

import logging
import csv
import os
import time
import re
from pathlib import Path
from typing import Any

from hyperloom.common.benchmark_config import read_benchmark_point
from hyperloom_common.coerce import to_float, to_int
from hyperloom.common.perf_metric import GRADED_INTVTY_P50_CHECKPOINT
from inference_suite.contracts import Report

from .gpu._gpu_metrics import gpu_metrics_from_report
from ._round_folder import RoundFolder

log = logging.getLogger(__name__)


# Multi-node GPU metrics: the GPU pods run a rocm-smi sampler (see launch_infera_node.py) streaming per-card samples
# to ``$HYPERLOOM_MN_SERVER_LOG_DIR/gpu_metrics_<host>.csv`` on shared storage.
_MN_GPU_WINDOW_SLACK_SEC: float = 2.0


def _num_from_cell(cell: Any) -> float | None:
    """Parse the first numeric token from a rocm-smi CSV cell (unit-tolerant)."""
    if cell is None:
        return None
    m = re.search(r"-?\d+\.?\d*", str(cell))
    return float(m.group(0)) if m else None


def _pod_gpu_metric_csvs() -> list[Path]:
    """The per-pod GPU sampler CSVs sitting in the shared server-log dir the pods were told to write to."""
    # Resolved exactly as cli.py forwards it to the pods (explicit env, else the $USER_DATA_PATH/server_logs
    # default) so the client reads where the pod sampler wrote, without changing forwarding logic.
    shared = os.path.expandvars(
        os.environ.get("HYPERLOOM_MN_SERVER_LOG_DIR", "").strip() or "$USER_DATA_PATH/server_logs"
    )
    if not shared.startswith("/") or "$" in shared:
        return []
    src_dir = Path(shared)
    try:
        if not src_dir.is_dir():
            return []
        return sorted(src_dir.glob("gpu_metrics_*.csv"))
    except OSError:
        return []


def _pod_role_by_ip(pd: dict[str, Any]) -> dict[str, str]:
    """Pod IP to prefill/decode role, so each merged row can be tagged with its pod's role.

    Empty unless the deployment is PD-disaggregated.
    """
    role_of: dict[str, str] = {}
    for ip in pd.get("prefill_pod_ips", []):
        role_of[str(ip)] = "prefill"
    for ip in pd.get("decode_pod_ips", []):
        role_of[str(ip)] = "decode"
    return role_of


def _write_pod_gpu_csv(
    destination: Path, pod_csvs: list[Path], role_of: dict[str, str], started_unix: float | None
) -> dict[str, Any]:
    """Tag every in-window pod row with its host and role and write them under one header as ``gpu_metrics.csv``."""
    lo = None if started_unix is None else float(started_unix) - _MN_GPU_WINDOW_SLACK_SEC
    hi = time.time() + _MN_GPU_WINDOW_SLACK_SEC
    header: list[str] | None = None
    merged: list[list[str]] = []
    for pod_csv in pod_csvs:
        host = pod_csv.stem[len("gpu_metrics_") :]
        try:
            with pod_csv.open(encoding="utf-8", errors="replace", newline="") as f:
                rows = list(csv.reader(f))
        except OSError:
            continue
        if len(rows) < 2:
            continue
        role = role_of.get(host, "")
        if header is None:
            header = (["host", "role"] if role_of else ["host"]) + rows[0]
        for row in rows[1:]:
            ts = _num_from_cell(row[0]) if row else None
            if ts is None or (lo is not None and (ts < lo or ts > hi)):
                continue
            merged.append(([host, role] if role_of else [host]) + row)
    if not header or not merged:
        return {}
    csv_path = destination / "gpu_metrics.csv"
    try:
        destination.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(merged)
    except OSError as exc:
        log.warning("benchmark_result: MN gpu_metrics.csv write failed: %s", exc)
        return {}
    return {"gpu_metrics_csv": str(csv_path), "rows": len(merged)}


def harvest_mn_gpu_metrics(
    destination: Path,
    *,
    subprocess_started_unix: float | None = None,
) -> dict[str, Any]:
    """Fold pod-side GPU sampler CSVs into ``destination`` as ``gpu_metrics.csv`` (multi-node only)."""
    # Multi-node only: a single-node round samples its own GPUs, so never touch its result path.
    # is_multi_node() is the authoritative gate (state nodes>=2 or $INFERENCE_OPTIMIZER_NODES>=2).
    from ._multi_node_env import is_multi_node, pd_topology_from_state

    if not is_multi_node():
        return {}
    pod_csvs = _pod_gpu_metric_csvs()
    if not pod_csvs:
        return {}

    role_of = _pod_role_by_ip(pd_topology_from_state())
    out = _write_pod_gpu_csv(destination, pod_csvs, role_of, subprocess_started_unix)
    if out:
        log.info("benchmark_result: harvested MN GPU metrics %s", out)
    return out


def _round_gpu_power(report: Report) -> dict[str, Any]:
    """The round's measured-window power and energy: :data:`~._gpu_power.POWER_FIELDS`, ``gpu_power_by_gpu_w`` and
    ``power_valid``; only the first two without a recording.

    The report's ``power`` section wins whenever the round has one, including one whose energy is invalid: that round
    was sampled and measured nothing trustworthy, and substituting the ``gpu_monitor`` figure would replace
    "unmeasured" with a number from a different window. The ``gpu_monitor`` block is the fallback for the mean on
    rounds no recorder ran on; such a round has no per-card figure and any per-card or total budget treats it as
    unmeasured.
    """
    from .gpu._gpu_power import round_power_fields

    if report.power is not None:
        return round_power_fields(report.power)
    monitored = gpu_metrics_from_report(report.to_document())
    return {"gpu_power_avg_w": monitored.get("avg_power_w"), "gpu_power_by_gpu_w": None}


def extract_benchmark_measurement(run_dir: Path) -> dict[str, Any]:
    """The normalized measurement of the inference_suite round in ``run_dir``, with the measured-window power its
    report's ``power`` section holds.

    Raises ``OSError``, ``ValueError`` for a report or launch off contract (:meth:`RoundFolder.read`).
    """
    folder = RoundFolder.read(run_dir)
    measurement = folder.measurement()
    measurement.update(_round_gpu_power(folder.report))
    if folder.report.agentx is not None:
        _add_agentx_timeline_fields(measurement, run_dir)
    return measurement


def _add_agentx_timeline_fields(measurement: dict[str, Any], workspace: Path | None) -> None:
    """The checkpoint median and the replayed corpus shape, both read from the aiperf exports under ``workspace``."""
    from hyperloom.inference_optimizer.agentx.report import measured_corpus_shape

    from .workload._agentx_checkpoint import workspace_checkpoint_p50

    measurement[GRADED_INTVTY_P50_CHECKPOINT] = workspace_checkpoint_p50(workspace)
    if corpus_shape := measured_corpus_shape(workspace):
        measurement["agentx_corpus_shape"] = corpus_shape


def _is_scriptable_measurement(result: dict[str, Any]) -> bool:
    """Return whether a measurement came from a scriptable (server-less) run."""
    from hyperloom.inference_optimizer import framework_registry

    if str(result.get("workload_kind") or "").strip().lower() == framework_registry.SCRIPTABLE:
        return True
    if result.get("quality_gate") is not None:
        return True
    return framework_registry.is_scriptable(result.get("framework"))


def is_valid_measurement(result: dict[str, Any] | None) -> bool:
    """Return whether a measurement reflects a usable benchmark result."""
    if not isinstance(result, dict):
        return False
    output_tput = to_float(result.get("output_throughput"))
    if output_tput is None or output_tput <= 0:
        return False
    if result.get("submission_valid") is False:
        return False
    # ``diagnostic_only`` comes only from a report mapper, whose verdict on the report stands.
    if result.get("diagnostic_only") is not None and (
        result.get("diagnostic_only") is not False or result.get("valid_measurement") is not True
    ):
        return False
    if _is_scriptable_measurement(result):
        # A scriptable run whose image-quality gate failed is not selectable, regardless of throughput.
        from ._accuracy_gate import quality_gate_passed

        qg = result.get("quality_gate")
        if not quality_gate_passed(qg, require=False):
            return False
        return True
    completed = to_int(result.get("completed_requests"))
    return completed is not None and completed > 0


def round_succeeded(measurement: dict[str, Any]) -> bool:
    """Whether a round ran to the verdict its kind accepts.

    A native AgentX round marked diagnostic (a profile) is never a valid measurement: the trace it leaves is what it is
    run for, so the round's own ``round_succeeded`` decides it. Any other round succeeds when it is valid.
    """
    return measurement.get("round_succeeded", measurement.get("valid_measurement")) is True


def served_complete_protocol(result: dict[str, Any]) -> bool:
    """Return whether the run served every request its protocol asked for.

    This is what separates a benchmark that finished from one that stopped
    early, and it is the question a non-zero exit code cannot answer on its
    own. A server that died mid-run leaves fewer completed requests than were
    requested; a wrapper that failed on its way out leaves the full count and a
    measurement taken over the same protocol as a clean round.

    Both counts come from the run's own result artifact, so this answers for
    every caller of :func:`extract_benchmark_measurement` rather than only the
    ones that happen to declare the request count in their own env layer.

    A run that recorded no request count did not get far enough to state its
    protocol, so it cannot be judged complete. Scriptable workloads drive their
    own iteration count and never record one.
    """
    requested = to_int(result.get("requested_requests"))
    if requested is None:
        return False
    completed = to_int(result.get("completed_requests"))
    return completed is not None and completed >= requested


def _find_server_logs(slot: Path) -> list[Path]:
    """Return ``server.log`` files under ``slot``, largest first."""
    try:
        logs = list(slot.rglob("server.log"))
    except OSError:
        return []

    def _size(path: Path) -> int:
        """Best-effort byte size used to rank candidate logs (0 on error)."""
        try:
            return path.stat().st_size
        except OSError:
            return 0

    return sorted(logs, key=_size, reverse=True)


def double_run_requested(params: dict | None, config_path: Path) -> bool:
    """Whether a discarded warmup round precedes the measured one; ``baseline_double_run`` decides, on by default,
    except for a point measured once (an AgentX replay), which never has one."""
    from hyperloom_common.env import is_truthy

    if read_benchmark_point(config_path).measured_once:
        return False
    if params and "baseline_double_run" in params:
        return is_truthy(params["baseline_double_run"])
    return True


__all__ = [
    "extract_benchmark_measurement",
    "harvest_mn_gpu_metrics",
    "double_run_requested",
    "is_valid_measurement",
    "round_succeeded",
    "served_complete_protocol",
]
