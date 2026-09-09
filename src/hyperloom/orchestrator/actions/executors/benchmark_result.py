# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Benchmark result parsing shared by Magpie-backed executors, plus post-run artifact harvesting and salvage helpers."""

from __future__ import annotations

import logging
import csv
import json
import os
import time
import re
import shutil
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import first_float, first_int, to_float, to_int
from hyperloom.common.jsonio import read_json

log = logging.getLogger(__name__)


# Wrapper-side files that leak outside the per-task workspace (under /workspace or env-derived roots like
# $INFERENCEX_PATH, where append_lm_eval_summary ``mv ./``-s eval output); harvest_leaked_artifacts copies fresh
# matches back.
_DEFAULT_LEAK_ARTIFACT_GLOBS: tuple[str, ...] = (
    "server.log",
    "gpu_metrics.csv",
    "profile_*.trace.json.gz",
    "inferencex_result*.json",
    "results*.json",
)
_DEFAULT_LEAK_ARTIFACT_ROOT: Path = Path("/workspace")

# Slack subtracted from ``subprocess_started_unix`` before comparing a leak's ``st_mtime``, to reject stale prior-run
# leaks without false-dropping fresh ones. 1s absorbs clock-vs-mtime / FS-granularity skew.
_MTIME_GATE_SLACK_SEC: float = 1.0


def _candidate_raw_jsons(workspace: Path) -> list[Path]:
    """Return likely InferenceX result files, preferring baseline over profile."""
    paths = [p for p in workspace.rglob("*.json") if p.name != "benchmark_report.json"]
    return sorted(
        paths,
        key=lambda p: (
            "profile" in p.name.lower(),
            "eval" in str(p).lower(),
            str(p),
        ),
    )


def _rescue_candidate_paths(
    workspace: Path,
    *,
    subprocess_started_unix: float | None = None,
) -> list[Path]:
    """Return absolute paths to known Magpie leak destinations."""
    candidates: list[Path] = []
    seen: set[Path] = set()

    def _push(path: Path) -> None:
        """Add ``path`` to the candidate list if it passes all gates."""
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        # Skip files already inside the workspace (handled by ``_candidate_raw_jsons``).
        try:
            ws_resolved = workspace.resolve()
            resolved.relative_to(ws_resolved)
            return
        except (OSError, ValueError):
            pass
        if not path.is_file():
            return
        if subprocess_started_unix is not None:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                return
            if mtime + _MTIME_GATE_SLACK_SEC < float(subprocess_started_unix):
                return
        candidates.append(path)

    env_raw = os.environ.get("INFERENCE_OPTIMIZER_RESCUE_PATHS", "").strip()
    env_entries = [part.strip() for part in env_raw.split(":") if part.strip()] if env_raw else []
    for entry in env_entries:
        p = Path(entry)
        if p.is_dir():
            try:
                for fp in sorted(p.glob("inferencex_result*.json")):
                    _push(fp)
            except OSError:
                continue
        else:
            _push(p)

    # Env-derived dirs: the InferenceX checkout ($INFERENCEX_PATH), where append_lm_eval_summary's ``mv ./`` lands,
    # plus $RESULT_DIR overrides.
    for derived in _env_derived_leak_roots():
        if derived.is_dir():
            try:
                for fp in sorted(derived.glob("inferencex_result*.json")):
                    _push(fp)
            except OSError:
                continue

    return candidates


def _materialize_rescue_into_workspace(
    rescue_path: Path,
    workspace: Path,
) -> Path | None:
    """Copy a leaked InferenceX result back into the task workspace."""
    try:
        rescue_resolved = rescue_path.resolve()
        ws_resolved = workspace.resolve()
    except OSError:
        return None
    try:
        rescue_resolved.relative_to(ws_resolved)
        return None
    except ValueError:
        pass
    destination = workspace / rescue_path.name
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rescue_path, destination)
    except OSError as exc:
        log.warning(
            "benchmark_result: failed to copy rescued result %s -> %s: %s",
            rescue_path,
            destination,
            exc,
        )
        return None
    return destination


def _env_derived_leak_roots() -> list[Path]:
    """Leak roots derived from the runtime env: the InferenceX checkout (``$INFERENCEX_PATH``), where ``append_lm_eval_summary``'s ``mv ./`` lands, plus ``$RESULT_DIR`` when an override routed results outside the workspace."""
    out: list[Path] = []
    for env_key in ("INFERENCEX_PATH", "RESULT_DIR"):
        val = (os.environ.get(env_key) or "").strip()
        if val:
            out.append(Path(val))
    return out


def _resolve_leak_roots(leak_root: Path | None) -> tuple[Path, ...]:
    """Return the directory roots to scan for wrapper-side leak files."""
    if leak_root is not None:
        return (leak_root,)
    env_raw = os.environ.get("INFERENCE_OPTIMIZER_LEAK_ROOTS", "").strip()
    if env_raw:
        parts = [Path(p.strip()) for p in env_raw.split(":") if p.strip()]
        if parts:
            return tuple(parts)
    roots: list[Path] = [_DEFAULT_LEAK_ARTIFACT_ROOT]
    seen = {_DEFAULT_LEAK_ARTIFACT_ROOT}
    for root in _env_derived_leak_roots():
        if root not in seen:
            seen.add(root)
            roots.append(root)
    return tuple(roots)


def snapshot_workspaces(root: Path) -> frozenset[Path]:
    """Return the ``benchmark_*`` workspaces present in ``root`` right now."""
    return frozenset(p.resolve() for p in root.glob("benchmark_*") if p.is_dir())


def select_run_workspace(root: Path, *, known_before: frozenset[Path]) -> Path | None:
    """Return the ``benchmark_*`` workspace this run created in ``root``."""
    fresh = [p for p in root.glob("benchmark_*") if p.is_dir() and p.resolve() not in known_before]
    return max(fresh, default=None)


def harvest_leaked_artifacts(
    destination: Path,
    *,
    subprocess_started_unix: float | None = None,
    leak_root: Path | None = None,
    extra_globs: tuple[str, ...] = (),
) -> list[tuple[Path, Path]]:
    """Copy known Magpie/InferenceX leak artifacts into ``destination``."""
    harvested: list[tuple[Path, Path]] = []
    leak_roots = _resolve_leak_roots(leak_root)
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning(
            "benchmark_result.harvest: cannot prepare destination=%s: %s",
            destination,
            exc,
        )
        return harvested
    try:
        ws_resolved = destination.resolve()
    except OSError:
        return harvested

    globs = tuple(_DEFAULT_LEAK_ARTIFACT_GLOBS) + tuple(extra_globs)
    seen: set[Path] = set()
    for root in leak_roots:
        try:
            if not root.exists() or not root.is_dir():
                continue
        except OSError:
            continue
        for pattern in globs:
            try:
                matches = sorted(root.glob(pattern))
            except OSError:
                continue
            for match in matches:
                try:
                    resolved = match.resolve()
                except OSError:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    resolved.relative_to(ws_resolved)
                    continue  # Already under the workspace — nothing to harvest.
                except ValueError:
                    pass  # Outside the workspace; fall through to harvest below.
                if not match.is_file():
                    continue
                if subprocess_started_unix is not None:
                    try:
                        mtime = match.stat().st_mtime
                    except OSError:
                        continue
                    if mtime + _MTIME_GATE_SLACK_SEC < float(subprocess_started_unix):
                        continue
                destination_path = destination / match.name
                try:
                    shutil.copy2(match, destination_path)
                except OSError as exc:
                    log.warning(
                        "benchmark_result.harvest: copy %s -> %s failed: %s",
                        match,
                        destination_path,
                        exc,
                    )
                    continue
                harvested.append((match, destination_path))
    # Multi-node: fold pod-side GPU sampler CSVs into this workspace and inject a flat gpu_monitor list into
    # benchmark_report.json (no-op single-node).
    try:
        harvest_mn_gpu_metrics(destination, subprocess_started_unix=subprocess_started_unix)
    except Exception as exc:
        log.warning("benchmark_result.harvest: MN GPU-metrics harvest failed: %s", exc)
    return harvested


# Multi-node GPU metrics: the GPU pods run a rocm-smi sampler (see launch_infera_node.py) streaming per-card samples
# to ``$HYPERLOOM_MN_SERVER_LOG_DIR/gpu_metrics_<host>.csv`` on shared storage.
_MN_GPU_SAMPLE_CAP: int = 5000
_MN_GPU_WINDOW_SLACK_SEC: float = 2.0


def _num_from_cell(cell: Any) -> float | None:
    """Parse the first numeric token from a rocm-smi CSV cell (unit-tolerant)."""
    if cell is None:
        return None
    m = re.search(r"-?\d+\.?\d*", str(cell))
    return float(m.group(0)) if m else None


def _row_to_gpu_sample(header: list[str], row: list[str]) -> dict[str, Any]:
    """Map one rocm-smi ``--csv`` data row to a flat gpu_monitor sample."""
    n = min(len(header), len(row))
    cols = [(header[i] or "").strip().lower() for i in range(n)]
    vals = [_num_from_cell(row[i]) for i in range(n)]

    def _pick(*preds: Any) -> float | None:
        """Return the first numeric cell whose column matches a predicate."""
        for pred in preds:
            for i in range(n):
                if vals[i] is not None and pred(cols[i]):
                    return vals[i]
        return None

    sample: dict[str, Any] = {}
    temp = _pick(
        lambda c: "temp" in c and "junction" in c,
        lambda c: "temp" in c and "edge" in c,
        lambda c: "temp" in c and "mem" not in c,
        lambda c: "temp" in c,
    )
    if temp is not None:
        sample["temperature_c"] = temp
    power = _pick(
        lambda c: "average" in c and "power" in c,
        lambda c: "socket" in c and "power" in c,
        lambda c: "power" in c,
    )
    if power is not None:
        sample["power_w"] = power
    clock = _pick(lambda c: "sclk" in c)
    if clock is not None:
        sample["clock_mhz"] = clock
    util = _pick(lambda c: "gpu use" in c or "gpu_use" in c or c == "gpu%")
    if util is not None:
        sample["gpu_util_pct"] = util
    vram = _pick(lambda c: "vram" in c or ("memory" in c and "use" in c))
    if vram is not None:
        sample["vram_pct"] = vram
    return sample


def _aggregate_gpu_samples_by_role(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate flat GPU samples by ``role`` (prefill / decode)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        role = str(s.get("role") or "").strip()
        if role:
            groups.setdefault(role, []).append(s)
    if not groups:
        return {}

    def _stat(rows: list[dict[str, Any]], key: str, fn: Any) -> float:
        """Reduce a numeric field across ``rows`` via ``fn`` (0.0 when empty)."""
        vals = [to_float(r.get(key)) for r in rows]
        vals = [v for v in vals if v is not None]
        return round(fn(vals), 2) if vals else 0.0

    out: dict[str, Any] = {}
    for role, rows in groups.items():
        out[role] = {
            "samples": len(rows),
            "avg_power_w": _stat(rows, "power_w", lambda v: sum(v) / len(v)),
            "max_power_w": _stat(rows, "power_w", max),
            "avg_temp_c": _stat(rows, "temperature_c", lambda v: sum(v) / len(v)),
            "max_temp_c": _stat(rows, "temperature_c", max),
            "avg_gpu_util_pct": _stat(rows, "gpu_util_pct", lambda v: sum(v) / len(v)),
            "max_gpu_util_pct": _stat(rows, "gpu_util_pct", max),
            "avg_vram_pct": _stat(rows, "vram_pct", lambda v: sum(v) / len(v)),
            "max_vram_pct": _stat(rows, "vram_pct", max),
        }
    return out


def harvest_mn_gpu_metrics(
    destination: Path,
    *,
    subprocess_started_unix: float | None = None,
) -> dict[str, Any]:
    """Fold pod-side GPU sampler CSVs into ``destination`` (multi-node only)."""
    out: dict[str, Any] = {}
    # Multi-node only: single-node uses Magpie's own client-side GPUMonitor, so never touch its result path.
    # is_multi_node() is the authoritative gate (state nodes>=2 or $INFERENCE_OPTIMIZER_NODES>=2).
    from ._multi_node_env import is_multi_node

    if not is_multi_node():
        return out
    # Resolve the shared server-log dir exactly as cli.py forwards it to the pods (explicit env, else the
    # $USER_DATA_PATH/server_logs default) so the client reads where the pod sampler wrote, without changing
    # forwarding logic.
    shared = os.path.expandvars(
        os.environ.get("HYPERLOOM_MN_SERVER_LOG_DIR", "").strip() or "$USER_DATA_PATH/server_logs"
    )
    if not shared.startswith("/") or "$" in shared:
        return out
    src_dir = Path(shared)
    try:
        if not src_dir.is_dir():
            return out
        pod_csvs = sorted(src_dir.glob("gpu_metrics_*.csv"))
    except OSError:
        return out
    if not pod_csvs:
        return out

    # PD-disaggregation: map each pod IP -> prefill/decode role so metrics can be tagged and aggregated per role
    # (empty unless disaggregated).
    from ._multi_node_env import pd_topology_from_state

    pd = pd_topology_from_state()
    role_of: dict[str, str] = {}
    for _ip in pd.get("prefill_pod_ips", []):
        role_of[str(_ip)] = "prefill"
    for _ip in pd.get("decode_pod_ips", []):
        role_of[str(_ip)] = "decode"

    lo = None
    if subprocess_started_unix is not None:
        lo = float(subprocess_started_unix) - _MN_GPU_WINDOW_SLACK_SEC
    hi = time.time() + _MN_GPU_WINDOW_SLACK_SEC

    header: list[str] | None = None
    merged: list[list[str]] = []
    samples: list[dict[str, Any]] = []
    for pod_csv in pod_csvs:
        host = pod_csv.stem[len("gpu_metrics_") :]
        try:
            with pod_csv.open(encoding="utf-8", errors="replace", newline="") as f:
                rows = list(csv.reader(f))
        except OSError:
            continue
        if len(rows) < 2:
            continue
        rocm_header = rows[0]
        role = role_of.get(host, "")
        if header is None:
            header = (["host", "role"] if role_of else ["host"]) + rocm_header
        for row in rows[1:]:
            if not row:
                continue
            ts = _num_from_cell(row[0])
            if ts is None:
                continue
            if lo is not None and (ts < lo or ts > hi):
                continue
            merged.append(([host, role] if role_of else [host]) + row)
            s = _row_to_gpu_sample(rocm_header, row)
            if s:
                if role:
                    s["role"] = role
                samples.append(s)

    if header and merged:
        try:
            destination.mkdir(parents=True, exist_ok=True)
            csv_path = destination / "gpu_metrics.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(merged)
            out["gpu_metrics_csv"] = str(csv_path)
            out["rows"] = len(merged)
        except OSError as exc:
            log.warning("benchmark_result: MN gpu_metrics.csv write failed: %s", exc)

    if samples:
        report_path = destination / "benchmark_report.json"
        if report_path.is_file():
            try:
                with report_path.open(encoding="utf-8") as f:
                    report = json.load(f)
            except (OSError, json.JSONDecodeError):
                report = None
            if isinstance(report, dict):
                if len(samples) > _MN_GPU_SAMPLE_CAP:
                    stride = max(1, len(samples) // _MN_GPU_SAMPLE_CAP)
                    report["gpu_monitor"] = samples[::stride]
                else:
                    report["gpu_monitor"] = samples
                # PD-disaggregation: surface topology + per-role GPU aggregate so downstream analysis / the specialist
                # LLM can target prefill (compute/TTFT) vs decode (bandwidth/TPOT).
                if pd:
                    report["pd"] = pd
                    by_role = _aggregate_gpu_samples_by_role(samples)
                    if by_role:
                        report["gpu_monitor_by_role"] = by_role
                        out["gpu_monitor_by_role"] = {k: v.get("samples") for k, v in by_role.items()}
                try:
                    with report_path.open("w", encoding="utf-8") as f:
                        json.dump(report, f, indent=2)
                    out["gpu_monitor_samples"] = len(report["gpu_monitor"])
                except (OSError, TypeError) as exc:
                    log.warning("benchmark_result: gpu_monitor inject failed: %s", exc)
    if out:
        log.info("benchmark_result: harvested MN GPU metrics %s", out)
    return out


def _merge_raw_result(
    measurement: dict[str, Any],
    raw: dict[str, Any],
    *,
    source_path: Path,
) -> None:
    """Fill missing measurement fields from a raw InferenceX result."""
    if measurement.get("output_throughput") is None:
        measurement["output_throughput"] = to_float(raw.get("output_throughput"))
    if measurement.get("request_throughput") is None:
        measurement["request_throughput"] = to_float(raw.get("request_throughput"))
    if measurement.get("total_token_throughput") is None:
        measurement["total_token_throughput"] = to_float(raw.get("total_token_throughput"))
    if measurement.get("completed_requests") is None:
        measurement["completed_requests"] = first_int(
            raw.get("completed_requests"),
            raw.get("completed"),
        )
    if measurement.get("duration_seconds") is None:
        measurement["duration_seconds"] = first_float(
            raw.get("duration_seconds"),
            raw.get("duration"),
        )
    if measurement.get("ttft_mean_ms") is None:
        measurement["ttft_mean_ms"] = to_float(raw.get("mean_ttft_ms"))
    if measurement.get("ttft_p99_ms") is None:
        measurement["ttft_p99_ms"] = to_float(raw.get("p99_ttft_ms"))
    if measurement.get("tpot_mean_ms") is None:
        measurement["tpot_mean_ms"] = to_float(raw.get("mean_tpot_ms"))
    if measurement.get("input_throughput") is None:
        measurement["input_throughput"] = to_float(raw.get("input_throughput"))
    if measurement.get("tpot_p90_ms") is None:
        measurement["tpot_p90_ms"] = to_float(raw.get("p90_tpot_ms"))
    if measurement.get("intvty_p90") is None:
        measurement["intvty_p90"] = to_float(raw.get("intvty_p90_tok_s_user"))
    if measurement.get("e2el_mean_ms") is None:
        measurement["e2el_mean_ms"] = first_float(
            raw.get("mean_e2el_ms"),
            raw.get("mean_latency_ms"),
        )
    if measurement.get("e2el_p99_ms") is None:
        measurement["e2el_p99_ms"] = first_float(
            raw.get("p99_e2el_ms"),
            raw.get("p99_latency_ms"),
        )
    if measurement.get("raw_result_path") is None:
        measurement["raw_result_path"] = str(source_path)
    # AgentX scenario verdict.
    if "submission_valid" in raw and "submission_valid" not in measurement:
        measurement["submission_valid"] = raw.get("submission_valid")
        reasons = raw.get("submission_invalid_reasons") or []
        measurement["submission_invalid_reasons"] = (
            [str(r) for r in reasons] if isinstance(reasons, list) else [str(reasons)]
        )


#: The latency fields whose origin is tracked. A measurement can fill each of
#: them from a different place, and which place answered is not recoverable
#: from the number afterwards -- every source writes the same key.
_LATENCY_FIELDS = ("ttft_mean_ms", "e2el_mean_ms", "tpot_mean_ms")

#: Stable labels naming where a latency number was read from. Reported by the
#: extraction itself because this is the only frame that knows: by the time the
#: measurement is on a result dict, a value the report supplied and one salvaged
#: out of a leaked raw JSON are indistinguishable.
LATENCY_FROM_REPORT = "benchmark_report"
LATENCY_FROM_RAW = "raw_result"
LATENCY_FROM_RESCUED_RAW = "rescued_raw_result"
LATENCY_DERIVED = "derived_from_e2el_ttft"
LATENCY_UNAVAILABLE = "unavailable"


def _latency_snapshot(measurement: dict[str, Any]) -> dict[str, Any]:
    """The latency fields as they stand, for comparing across a fill pass.

    Args:
        measurement: The measurement dict to read.

    Returns:
        The tracked latency fields and their current values.
    """
    return {field: measurement.get(field) for field in _LATENCY_FIELDS}


def _tag_latency_origins(
    measurement: dict[str, Any],
    origins: dict[str, str],
    *,
    label: str,
    before: dict[str, Any],
) -> None:
    """Attribute to ``label`` the latency fields this pass filled.

    Only fields that were absent and are now present are attributed: every
    fill pass leaves what it found intact, so a field it did not fill belongs
    to whichever pass did.

    Args:
        measurement: The measurement dict after the pass ran.
        origins: The origin map to record into, mutated in place.
        label: The source label for this pass.
        before: The :func:`_latency_snapshot` taken before the pass ran.
    """
    for field in _LATENCY_FIELDS:
        if before.get(field) is None and measurement.get(field) is not None:
            origins[field] = label


def extract_benchmark_measurement(
    report: dict[str, Any] | None,
    *,
    workspace: Path | None = None,
    subprocess_started_unix: float | None = None,
) -> dict[str, Any]:
    """Extract a normalized measurement from Magpie and InferenceX outputs.

    ``subprocess_started_unix`` enables an opt-in salvage pass over the
    Magpie leak destinations (see :func:`_rescue_candidate_paths`) when the
    in-workspace search fails; only leaks written after this run are adopted.

    Args:
        report: The Magpie ``benchmark_report.json`` mapping, or ``None``.
        workspace: Optional task workspace scanned for raw InferenceX results
            and (as a fallback) salvageable leaks.
        subprocess_started_unix: Optional launch time enabling the mtime-gated
            leak salvage pass.

    Returns:
        A normalized measurement dict (including ``valid_measurement``, any
        ``nonfatal_warnings``, and the ``ttft_e2el_source`` / ``tpot_source``
        provenance labels).
    """
    report = report or {}
    throughput = report.get("throughput") or {}
    latency = report.get("latency") or {}
    ttft = latency.get("ttft") or {}
    tpot = latency.get("tpot") or {}
    e2el = latency.get("e2el") or {}

    measurement: dict[str, Any] = {
        "reported_success": report.get("success") if report else None,
        "framework": report.get("framework"),
        "model": report.get("model"),
        # Scriptable (server-less) workloads tag the report with workload_kind/unit and ship a quality_gate block
        # instead of a GSM8K eval; carried through so downstream gates/reporters can branch.
        "workload_kind": report.get("workload_kind"),
        "throughput_unit": report.get("throughput_unit") or throughput.get("unit"),
        "quality_gate": report.get("quality_gate"),
        "latency_s": first_float(report.get("latency_s"), throughput.get("latency_s")),
        "request_throughput": to_float(throughput.get("request_throughput")),
        "output_throughput": to_float(throughput.get("output_throughput")),
        "total_token_throughput": to_float(throughput.get("total_token_throughput")),
        "completed_requests": first_int(
            throughput.get("completed_requests"),
            throughput.get("completed"),
            # Diffusion scripts report images produced under either key.
            throughput.get("images_generated"),
            throughput.get("num_images"),
        ),
        "duration_seconds": to_float(throughput.get("duration_seconds")),
        "ttft_mean_ms": to_float(ttft.get("mean_ms")),
        "ttft_p99_ms": to_float(ttft.get("p99_ms")),
        "tpot_mean_ms": to_float(tpot.get("mean_ms")),
        "e2el_mean_ms": to_float(e2el.get("mean_ms")),
        "e2el_p99_ms": to_float(e2el.get("p99_ms")),
        "raw_result_path": None,
        "nonfatal_warnings": [],
    }

    origins: dict[str, str] = {}
    _tag_latency_origins(
        measurement,
        origins,
        label=LATENCY_FROM_REPORT,
        before=dict.fromkeys(_LATENCY_FIELDS),
    )

    if workspace is not None:
        for raw_path in _candidate_raw_jsons(workspace):
            raw = read_json(raw_path, default=None, require_dict=True)
            if not raw or to_float(raw.get("output_throughput")) is None:
                continue
            before = _latency_snapshot(measurement)
            _merge_raw_result(measurement, raw, source_path=raw_path)
            _tag_latency_origins(measurement, origins, label=LATENCY_FROM_RAW, before=before)
            if is_valid_measurement(measurement):
                break

    warnings = measurement["nonfatal_warnings"]
    if report and report.get("success") is not True:
        warnings.append("benchmark_report_success_false")
    if workspace is not None and measurement.get("raw_result_path"):
        warnings.append("raw_inferencex_result_used")

    before = _latency_snapshot(measurement)
    _derive_tpot_if_missing(measurement, report)
    _tag_latency_origins(measurement, origins, label=LATENCY_DERIVED, before=before)
    measurement["valid_measurement"] = is_valid_measurement(measurement)

    # Second-chance salvage from Magpie leak destinations when the in-workspace search found no usable measurement
    # (mtime-gated).
    if not measurement["valid_measurement"] and workspace is not None:
        for rescue_path in _rescue_candidate_paths(
            workspace,
            subprocess_started_unix=subprocess_started_unix,
        ):
            raw = read_json(rescue_path, default=None, require_dict=True)
            if not raw or to_float(raw.get("output_throughput")) is None:
                continue
            # Copy the leak into the workspace BEFORE merging so the NFS clone stays self-contained.
            materialized = _materialize_rescue_into_workspace(
                rescue_path,
                workspace,
            )
            recorded_path = materialized if materialized is not None else rescue_path
            before = _latency_snapshot(measurement)
            _merge_raw_result(measurement, raw, source_path=recorded_path)
            _tag_latency_origins(measurement, origins, label=LATENCY_FROM_RESCUED_RAW, before=before)
            if is_valid_measurement(measurement):
                warnings.append(f"rescued_from_leaked_path:{rescue_path}")
                if materialized is None:
                    warnings.append(f"rescued_copy_into_workspace_failed: {rescue_path}")
                break
        before = _latency_snapshot(measurement)
        _derive_tpot_if_missing(measurement, report)
        _tag_latency_origins(measurement, origins, label=LATENCY_DERIVED, before=before)
        measurement["valid_measurement"] = is_valid_measurement(measurement)

    # One label for the pair, keyed on TTFT and falling back to E2EL, because
    # that is the question a reader asks of it: the two are read from the same
    # place in every path that supplies either, and TTFT is the one a latency
    # reference is anchored on.
    measurement["ttft_e2el_source"] = origins.get("ttft_mean_ms") or origins.get("e2el_mean_ms") or LATENCY_UNAVAILABLE
    # Separate from the pair: TPOT is the one latency figure that can be
    # computed rather than measured, and a derived value must not be read as
    # one the benchmark reported.
    measurement["tpot_source"] = origins.get("tpot_mean_ms") or LATENCY_UNAVAILABLE
    return measurement


def _derive_tpot_if_missing(
    measurement: dict[str, Any],
    report: dict[str, Any] | None,
) -> None:
    """Fill ``tpot_mean_ms`` from ``(e2el - ttft) / (osl - 1)`` when absent."""
    if measurement.get("tpot_mean_ms") is not None:
        return
    e2el = to_float(measurement.get("e2el_mean_ms"))
    ttft = to_float(measurement.get("ttft_mean_ms"))
    if e2el is None or ttft is None or e2el <= ttft:
        return
    osl = _resolve_osl(report)
    if osl is None or osl <= 1:
        return
    measurement["tpot_mean_ms"] = (e2el - ttft) / (osl - 1)


def _resolve_osl(report: dict[str, Any] | None) -> int | None:
    """Pull the output sequence length from common report locations."""
    if not isinstance(report, dict):
        return None
    candidates: list[Any] = [report.get("osl"), report.get("output_len")]
    for section_key in ("config", "request", "params", "workload"):
        section = report.get(section_key)
        if isinstance(section, dict):
            candidates.extend(section.get(k) for k in ("osl", "output_len", "max_tokens"))
    for value in candidates:
        n = to_int(value)
        if n is not None and n > 0:
            return n
    return None


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
    # Gated on BOTH the mode and the key's presence, and each half earns its keep.
    from ._workload_envs import agentx_enabled

    if agentx_enabled() and "submission_valid" in result:
        verdict = result.get("submission_valid")
        if verdict is False:
            return False
        if verdict is None:
            # The verdict is unknown: no --scenario was requested or the aiperf build predates the field. map_aiperf
            # writes the key unconditionally, so None arrives as a present key.
            from hyperloom.common.env import env_bool

            if not env_bool("HYPERLOOM_ALLOW_UNVERIFIED_SUBMISSION"):
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


# ── Approximate throughput for killed-overtime variants ──
_SGLANG_GEN_TPUT_RE = re.compile(
    r"gen throughput \(token/s\):\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_VLLM_GEN_TPUT_RE = re.compile(
    r"Avg generation throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens?/s",
    re.IGNORECASE,
)

# Fraction of the leading warmup samples dropped before averaging so the estimate reflects sustained decode rather
# than the cold-start climb.
_DEFAULT_WARMUP_SKIP_FRAC: float = 0.25


def _parse_server_log_gen_throughput(log_path: Path) -> list[float]:
    """Return every positive decode-throughput sample logged in ``server.log``."""
    samples: list[float] = []
    try:
        with log_path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                match = _SGLANG_GEN_TPUT_RE.search(line) or _VLLM_GEN_TPUT_RE.search(line)
                if match is None:
                    continue
                value = to_float(match.group(1))
                if value is not None:
                    samples.append(value)
    except OSError:
        return []
    return samples


def _steady_state_mean(
    samples: list[float],
    *,
    warmup_skip_frac: float = _DEFAULT_WARMUP_SKIP_FRAC,
) -> float | None:
    """Average the steady-state portion of throughput ``samples``."""
    positive = [s for s in samples if s > 0]
    if not positive:
        return None
    skip = int(len(positive) * max(0.0, min(1.0, warmup_skip_frac)))
    steady = positive[skip:] or positive
    return sum(steady) / len(steady)


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


def estimate_output_throughput_from_server_log(
    log_path: Path,
    *,
    warmup_skip_frac: float = _DEFAULT_WARMUP_SKIP_FRAC,
) -> dict[str, Any] | None:
    """Estimate sustained output throughput from one engine ``server.log``."""
    samples = _parse_server_log_gen_throughput(log_path)
    mean = _steady_state_mean(samples, warmup_skip_frac=warmup_skip_frac)
    if mean is None:
        return None
    return {
        "output_throughput": mean,
        "num_samples": sum(1 for s in samples if s > 0),
        "source_path": str(log_path),
    }


def estimate_killed_variant_throughput(
    slot: Path,
    *,
    warmup_skip_frac: float = _DEFAULT_WARMUP_SKIP_FRAC,
) -> dict[str, Any] | None:
    """Estimate output throughput for a killed-overtime variant from its logs."""
    for log_path in _find_server_logs(slot):
        estimate = estimate_output_throughput_from_server_log(
            log_path,
            warmup_skip_frac=warmup_skip_frac,
        )
        if estimate is not None:
            return estimate
    return None


__all__ = [
    "LATENCY_DERIVED",
    "LATENCY_FROM_RAW",
    "LATENCY_FROM_REPORT",
    "LATENCY_FROM_RESCUED_RAW",
    "LATENCY_UNAVAILABLE",
    "harvest_mn_gpu_metrics",
    "estimate_killed_variant_throughput",
    "estimate_output_throughput_from_server_log",
    "extract_benchmark_measurement",
    "harvest_leaked_artifacts",
    "is_valid_measurement",
    "_materialize_rescue_into_workspace",
]
