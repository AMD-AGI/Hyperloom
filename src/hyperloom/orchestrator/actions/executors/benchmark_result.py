# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Benchmark result parsing shared by Magpie-backed executors, plus post-run artifact harvesting and salvage helpers."""

from __future__ import annotations

import logging
import csv
import json
import math
import os
import time
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.coerce import first_float, first_int, to_float, to_int
from hyperloom.common.jsonio import read_json

from ._gpu_metrics import write_gpu_metrics

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
    except Exception as exc:  # noqa: BLE001 - telemetry harvest must not fail the run
        log.warning("benchmark_result.harvest: MN GPU-metrics harvest failed: %s", exc)
    # Whatever wrote the round's ``gpu_monitor`` block -- Magpie on one node, the harvest above on several -- normalise
    # it into an artifact of its own now, while the round's own workspace is the subject. Aggregating it per session
    # instead averaged baseline, explore and roofline rounds together and described none of them.
    # Best effort, and only that: the report may still be settling, in which case there is nothing to read yet and the
    # settled path writes it instead. ``write_gpu_metrics`` owns the guarantee that it never raises, so wrapping it
    # again here would only add a second, unreachable handler over the one that reports what actually went wrong.
    write_gpu_metrics(destination)
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
    if measurement.get("ttft_p50_ms") is None:
        measurement["ttft_p50_ms"] = to_float(raw.get("median_ttft_ms"))
    if measurement.get("ttft_p90_ms") is None:
        measurement["ttft_p90_ms"] = to_float(raw.get("p90_ttft_ms"))
    if measurement.get("tpot_p50_ms") is None:
        measurement["tpot_p50_ms"] = to_float(raw.get("median_tpot_ms"))
    if measurement.get("e2e_norm_intvty_p90") is None:
        measurement["e2e_norm_intvty_p90"] = to_float(raw.get("e2e_norm_intvty_p90"))
    if measurement.get("e2e_norm_intvty_p50") is None:
        measurement["e2e_norm_intvty_p50"] = to_float(raw.get("e2e_norm_intvty_p50"))
    if measurement.get("request_error_rate") is None:
        measurement["request_error_rate"] = to_float(raw.get("request_error_rate"))
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
    if measurement.get("requested_requests") is None:
        measurement["requested_requests"] = first_int(raw.get("num_prompts"))
    if measurement.get("raw_result_path") is None:
        measurement["raw_result_path"] = str(source_path)
    # AgentX scenario verdict.
    if "submission_valid" in raw and "submission_valid" not in measurement:
        measurement["scenario"] = "agentx"
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

_TRUSTED_NATIVE_AGENTX_GPU_COUNT_SOURCES = frozenset(
    {
        "magpie_report_recipe",
        "magpie_config_snapshot",
        "materialized_workload_metadata",
        "inferencex_raw",
    }
)


def _valid_recipe_fingerprint(value: Any) -> str | None:
    """Return an InferenceX recipe fingerprint only when strictly valid."""
    if not isinstance(value, str):
        return None
    fingerprint = value
    if len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
        return None
    return fingerprint


def _native_agentx_explicit_topology_fields(value: Any) -> tuple[dict[str, int], bool]:
    """Return explicit native topology fields and whether all were valid.

    InferenceX's aggregate currently serializes only ``tp`` and ``ep`` even
    though the fingerprint-bound Magpie snapshot retains TP x PP x PCP.  Keep
    partial fields available for consistency checks without ever deriving a
    GPU count from them.
    """
    if not isinstance(value, dict):
        return {}, False

    def explicit_positive_integer(*keys: str) -> tuple[int | None, bool]:
        present_values = [value[key] for key in keys if key in value]
        if not present_values:
            return None, False
        parsed_values: list[int] = []
        for raw in present_values:
            parsed = to_float(raw)
            if parsed is None or parsed <= 0 or not parsed.is_integer():
                return None, True
            parsed_values.append(int(parsed))
        if len(set(parsed_values)) != 1:
            return None, True
        return parsed_values[0], True

    aliases = {
        "tp": ("tp",),
        "pp": ("pp",),
        "pcp": ("pcp_size", "pcp-size", "pcp"),
        "ep": ("ep",),
        "gpu_count": ("gpu_count",),
    }
    fields: dict[str, int] = {}
    for canonical, keys in aliases.items():
        parsed, present = explicit_positive_integer(*keys)
        if present and parsed is None:
            return {}, False
        if parsed is not None:
            fields[canonical] = parsed
    return fields, True


def _native_agentx_topology(value: Any) -> tuple[dict[str, int], int] | None:
    """Extract an explicit AgentX topology without inventing PP/PCP defaults."""
    fields, fields_valid = _native_agentx_explicit_topology_fields(value)
    if not fields_valid:
        return None
    tp = fields.get("tp")
    pp = fields.get("pp")
    pcp = fields.get("pcp")
    explicit_gpu_count = fields.get("gpu_count")
    dimensions = (tp, pp, pcp)
    if all(dimension is not None for dimension in dimensions):
        assert tp is not None and pp is not None and pcp is not None
        calculated_gpu_count = tp * pp * pcp
        if explicit_gpu_count is not None and explicit_gpu_count != calculated_gpu_count:
            return None
        topology = {"tp": tp, "pp": pp, "pcp": pcp}
        if "ep" in fields:
            topology["ep"] = fields["ep"]
        return topology, calculated_gpu_count
    if explicit_gpu_count is not None and explicit_gpu_count > 0 and all(dimension is None for dimension in dimensions):
        topology = {"gpu_count": explicit_gpu_count}
        if "ep" in fields:
            topology["ep"] = fields["ep"]
        return topology, explicit_gpu_count
    return None


def _set_native_agentx_topology(
    measurement: dict[str, Any],
    value: Any,
    *,
    source: str,
) -> bool:
    """Record a validated native topology and its provenance."""
    parsed = _native_agentx_topology(value)
    if parsed is None:
        return False
    topology, gpu_count = parsed
    existing_gpu_count = first_int(measurement.get("agentx_gpu_count"))
    if existing_gpu_count is not None and existing_gpu_count != gpu_count:
        return False
    existing_topology = measurement.get("agentx_gpu_topology")
    if isinstance(existing_topology, dict):
        for dimension in ("tp", "pp", "pcp", "ep", "gpu_count"):
            if (
                dimension in existing_topology
                and dimension in topology
                and existing_topology[dimension] != topology[dimension]
            ):
                return False
        topology = {**existing_topology, **topology}
    measurement["agentx_gpu_topology"] = topology
    measurement["agentx_gpu_count"] = gpu_count
    measurement["agentx_gpu_count_source"] = source
    return True


def _merge_native_agentx_report(
    measurement: dict[str, Any],
    report: dict[str, Any],
) -> None:
    """Normalize Magpie's native ``agentx_metrics`` into Hyperloom fields."""
    metrics = report.get("agentx_metrics")
    if not isinstance(metrics, dict):
        return

    raw_throughput = metrics.get("throughput")
    throughput = raw_throughput if isinstance(raw_throughput, dict) else {}
    raw_requests = metrics.get("requests")
    requests = raw_requests if isinstance(raw_requests, dict) else {}
    raw_latency = metrics.get("latency_seconds")
    latency = raw_latency if isinstance(raw_latency, dict) else {}

    schema_errors: list[str] = []

    def positive_number(value: Any) -> bool:
        parsed = to_float(value)
        return parsed is not None and math.isfinite(parsed) and parsed > 0

    if not isinstance(raw_throughput, dict):
        schema_errors.append("agentx_metrics.throughput_missing")
    else:
        for key in ("output_tokens_per_second", "total_tokens_per_second", "duration_seconds"):
            if not positive_number(throughput.get(key)):
                schema_errors.append(f"agentx_metrics.throughput.{key}_invalid")
    if not isinstance(raw_requests, dict):
        schema_errors.append("agentx_metrics.requests_missing")
    else:
        successful_for_schema = first_int(requests.get("successful"))
        profiled_for_schema = first_int(requests.get("profiled_total"))
        errors_for_schema = first_int(requests.get("errors"))
        error_rate_for_schema = to_float(requests.get("error_rate"))
        if successful_for_schema is None or successful_for_schema <= 0:
            schema_errors.append("agentx_metrics.requests.successful_invalid")
        if profiled_for_schema is None or profiled_for_schema <= 0:
            schema_errors.append("agentx_metrics.requests.profiled_total_invalid")
        if errors_for_schema is None or errors_for_schema < 0:
            schema_errors.append("agentx_metrics.requests.errors_invalid")
        if (
            error_rate_for_schema is None
            or not math.isfinite(error_rate_for_schema)
            or not 0.0 <= error_rate_for_schema <= 1.0
        ):
            schema_errors.append("agentx_metrics.requests.error_rate_invalid")
    if not isinstance(raw_latency, dict):
        schema_errors.append("agentx_metrics.latency_seconds_missing")
    interactivity_for_schema = latency.get("e2e_norm_intvty")
    if not isinstance(interactivity_for_schema, dict) or not positive_number(interactivity_for_schema.get("p90")):
        schema_errors.append("agentx_metrics.latency_seconds.e2e_norm_intvty.p90_invalid")
    recipe_for_schema = metrics.get("recipe")
    recipe_fingerprint: str | None = None
    if not isinstance(recipe_for_schema, dict):
        schema_errors.append("agentx_metrics.recipe_missing")
    else:
        recipe_fingerprint = _valid_recipe_fingerprint(recipe_for_schema.get("recipe_fingerprint"))
        if recipe_fingerprint is None:
            schema_errors.append("agentx_metrics.recipe.recipe_fingerprint_invalid")
        recipe_tp = first_int(recipe_for_schema.get("tp"))
        if recipe_tp is None or recipe_tp <= 0:
            schema_errors.append("agentx_metrics.recipe.tp_invalid")
    launch_for_schema = metrics.get("launch")
    launch_fingerprint: str | None = None
    if not isinstance(launch_for_schema, dict):
        schema_errors.append("agentx_metrics.launch_missing")
    else:
        launch_fingerprint = _valid_recipe_fingerprint(launch_for_schema.get("recipe_fingerprint"))
        if launch_fingerprint is None:
            schema_errors.append("agentx_metrics.launch.recipe_fingerprint_invalid")
        elif recipe_fingerprint is not None and launch_fingerprint != recipe_fingerprint:
            schema_errors.append("agentx_metrics.recipe_launch_fingerprint_mismatch")
    if str(metrics.get("mode") or "").strip().lower() not in {"canonical", "fast"}:
        schema_errors.append("agentx_metrics.mode_invalid")
    if str(metrics.get("scenario_type") or "").strip().lower() != "agentic-coding":
        schema_errors.append("agentx_metrics.scenario_type_invalid")
    if metrics.get("recipe_fingerprint_valid") is not True:
        schema_errors.append("agentx_metrics.recipe_fingerprint_valid_invalid")
    if not isinstance(report.get("benchmark_valid"), bool):
        schema_errors.append("benchmark_valid_missing")
    if not isinstance(report.get("publishable"), bool):
        schema_errors.append("publishable_missing")
    measurement["native_agentx_schema_errors"] = schema_errors
    measurement["native_agentx_schema_valid"] = not schema_errors

    native_throughput_fields = {
        "request_throughput": "request_throughput",
        "input_throughput": "input_tokens_per_second",
        "output_throughput": "output_tokens_per_second",
        "total_token_throughput": "total_tokens_per_second",
        "duration_seconds": "duration_seconds",
    }
    for target, source in native_throughput_fields.items():
        value = to_float(throughput.get(source))
        if value is not None:
            measurement[target] = value

    successful = first_int(requests.get("successful"))
    if successful is not None:
        measurement["completed_requests"] = successful
    # AgentX is duration-bounded.  Magpie's ``profiled_total`` is successful
    # responses plus error-dropped records, not a requested-session target.
    # Treating it as ``requested_requests`` makes the generic fixed-request
    # completeness check reject an otherwise complete canonical replay.
    measurement["requested_requests"] = None
    measurement["request_errors"] = first_int(requests.get("errors"))
    native_error_rate = to_float(requests.get("error_rate"))
    # Hyperloom's legacy result contract records request error rate as a
    # percentage. Magpie's native AgentX schema uses a 0..1 ratio.
    measurement["request_error_rate"] = native_error_rate * 100.0 if native_error_rate is not None else None

    def latency_stat_ms(metric: str, stat_name: str) -> float | None:
        values = latency.get(metric)
        if not isinstance(values, dict):
            return None
        seconds = to_float(values.get(stat_name))
        return seconds * 1000.0 if seconds is not None else None

    # Magpie's generic LatencyMetrics serializer emits p99=0 even though the
    # pinned InferenceX AgentX aggregate supplies p95, not p99. Clear those
    # placeholders; a missing percentile is not a zero-latency observation.
    measurement["ttft_p99_ms"] = None
    measurement["e2el_p99_ms"] = None
    for target, metric, stat_name in (
        ("ttft_mean_ms", "ttft", "mean"),
        ("ttft_p50_ms", "ttft", "p50"),
        ("ttft_p90_ms", "ttft", "p90"),
        ("ttft_p95_ms", "ttft", "p95"),
        ("tpot_mean_ms", "tpot", "mean"),
        ("tpot_p50_ms", "tpot", "p50"),
        ("tpot_p90_ms", "tpot", "p90"),
        ("tpot_p95_ms", "tpot", "p95"),
        ("e2el_mean_ms", "e2el", "mean"),
        ("e2el_p95_ms", "e2el", "p95"),
    ):
        value = latency_stat_ms(metric, stat_name)
        if value is not None:
            measurement[target] = value

    interactivity = latency.get("e2e_norm_intvty")
    if isinstance(interactivity, dict):
        # This is a rate (1 / normalized E2E seconds), not a latency. Keep the
        # native unit; multiplying it by 1000 would corrupt Hyperloom's 2-D
        # AgentX objective.
        measurement["e2e_norm_intvty_p90"] = to_float(interactivity.get("p90"))
        measurement["e2e_norm_intvty_p50"] = to_float(interactivity.get("p50"))

    benchmark_valid = report.get("benchmark_valid")
    publishable = report.get("publishable")
    measurement["benchmark_valid"] = benchmark_valid
    measurement["publishable"] = publishable
    # Canonical Hyperloom measurements must pass both native validity and the
    # recipe-fingerprint/canonical-mode gate represented by ``publishable``.
    measurement["submission_valid"] = (
        benchmark_valid is True and publishable is True
        if isinstance(benchmark_valid, bool) and isinstance(publishable, bool)
        else None
    )
    reasons = report.get("errors") or []
    invalid_reasons = [str(reason) for reason in reasons] if isinstance(reasons, list) else [str(reasons)]
    if benchmark_valid is not True:
        invalid_reasons.append("native_agentx_benchmark_invalid")
    if publishable is not True:
        mode = str(metrics.get("mode") or "").strip().lower()
        invalid_reasons.append("native_agentx_fast_mode" if mode == "fast" else "native_agentx_not_publishable")
    measurement["submission_invalid_reasons"] = list(dict.fromkeys(invalid_reasons))
    measurement["agentx_mode"] = metrics.get("mode")
    measurement["agentx_requests"] = requests
    # Keep the compact payloads until the protocol validator has cross-bound
    # them to InferenceX's aggregate.  The normalized fields omit the complete
    # latency/accounting maps and cannot by themselves prove a single run.
    measurement["agentx_report_throughput"] = throughput
    measurement["agentx_report_latency_seconds"] = latency
    recipe = metrics.get("recipe")
    measurement["agentx_recipe"] = recipe
    measurement["agentx_launch"] = metrics.get("launch")
    measurement["agentx_request_accounting"] = metrics.get("request_accounting")
    if recipe_fingerprint is not None:
        measurement["agentx_recipe_fingerprint"] = recipe_fingerprint
    if launch_fingerprint is not None:
        measurement["agentx_launch_recipe_fingerprint"] = launch_fingerprint
    if recipe_fingerprint is not None and launch_fingerprint == recipe_fingerprint:
        # Current Magpie omits PCP from agentx_metrics.recipe. Only accept the
        # compact report itself when every parallelism dimension is explicit.
        _set_native_agentx_topology(measurement, recipe, source="magpie_report_recipe")
    measurement["agentx_report_dataset"] = metrics.get("dataset")
    measurement["agentx_dataset"] = metrics.get("dataset")


def _merge_native_agentx_workspace_topology(measurement: dict[str, Any], workspace: Path) -> None:
    """Use Magpie's resolved config snapshot as a fingerprint-bound topology."""
    expected_fingerprint = _valid_recipe_fingerprint(measurement.get("agentx_recipe_fingerprint"))
    if expected_fingerprint is None:
        return
    config_path = workspace / "config.yaml"
    if not config_path.is_file():
        return
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        measurement["nonfatal_warnings"].append("native_agentx_config_snapshot_unreadable")
        return
    if not isinstance(config, dict):
        return

    wrapped = config.get("benchmark")
    if isinstance(wrapped, dict):
        config = wrapped

    agentx = config.get("agentx")
    resolved = agentx.get("resolved") if isinstance(agentx, dict) else None
    if isinstance(resolved, dict):
        resolved_fingerprint = _valid_recipe_fingerprint(
            resolved.get("recipe-fingerprint", resolved.get("recipe_fingerprint"))
        )
        if resolved_fingerprint != expected_fingerprint:
            measurement["nonfatal_warnings"].append("native_agentx_config_recipe_fingerprint_mismatch")
            measurement["native_agentx_topology_conflict"] = True
            return
        if not _set_native_agentx_topology(measurement, resolved, source="magpie_config_snapshot"):
            measurement["nonfatal_warnings"].append("native_agentx_config_topology_invalid")
            measurement["native_agentx_topology_conflict"] = True
        return

    # Forward-compatible path for a materializer that persists an explicitly
    # fingerprint-bound workload topology in Magpie's workspace snapshot.
    workload_spec = config.get("workload_spec")
    topology = workload_spec.get("resolved_topology") if isinstance(workload_spec, dict) else None
    if not isinstance(topology, dict):
        return
    topology_fingerprint = _valid_recipe_fingerprint(
        topology.get("recipe_fingerprint", topology.get("recipe-fingerprint"))
    )
    if topology_fingerprint != expected_fingerprint:
        measurement["nonfatal_warnings"].append("native_agentx_workload_recipe_fingerprint_mismatch")
        measurement["native_agentx_topology_conflict"] = True
        return
    if not _set_native_agentx_topology(measurement, topology, source="materialized_workload_metadata"):
        measurement["nonfatal_warnings"].append("native_agentx_workload_topology_invalid")
        measurement["native_agentx_topology_conflict"] = True


def _native_distribution(value: Any) -> dict[str, int]:
    """Map InferenceX's native token distribution onto Hyperloom names."""
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for source, target in (("mean", "avg"), ("p50", "p50"), ("p75", "p75"), ("p90", "p90"), ("p95", "p95")):
        parsed = to_float(value.get(source))
        if parsed is not None and math.isfinite(parsed) and parsed >= 0:
            out[target] = int(round(parsed))
    return out


def _native_agentx_raw_matches_report(
    measurement: dict[str, Any],
    raw: dict[str, Any],
    *,
    source_path: Path,
) -> bool:
    """Validate raw aggregate identity before any of its payload is consumed."""
    warnings = measurement["nonfatal_warnings"]
    expected_fingerprint = _valid_recipe_fingerprint(measurement.get("agentx_recipe_fingerprint"))
    raw_fingerprint = _valid_recipe_fingerprint(raw.get("recipe_fingerprint"))
    if expected_fingerprint is None or raw_fingerprint is None:
        warnings.append(f"native_agentx_raw_recipe_fingerprint_missing:{source_path}")
        return False
    if raw_fingerprint != expected_fingerprint:
        warnings.append(f"native_agentx_raw_recipe_fingerprint_mismatch:{source_path}")
        return False

    raw_fields, raw_fields_valid = _native_agentx_explicit_topology_fields(raw)
    raw_has_topology_field = any(key in raw for key in ("tp", "pp", "pcp", "pcp_size", "pcp-size", "ep", "gpu_count"))
    if raw_has_topology_field and not raw_fields_valid:
        warnings.append(f"native_agentx_raw_topology_invalid:{source_path}")
        return False

    raw_topology = _native_agentx_topology(raw)
    if raw_topology is not None:
        parsed_raw_topology, raw_gpu_count = raw_topology
        expected_gpu_count = first_int(measurement.get("agentx_gpu_count"))
        if expected_gpu_count is not None and expected_gpu_count != raw_gpu_count:
            warnings.append(f"native_agentx_raw_topology_mismatch:{source_path}")
            return False
        expected_topology = measurement.get("agentx_gpu_topology")
        if isinstance(expected_topology, dict):
            for dimension in ("tp", "pp", "pcp", "ep"):
                if (
                    dimension in expected_topology
                    and dimension in parsed_raw_topology
                    and expected_topology[dimension] != parsed_raw_topology[dimension]
                ):
                    warnings.append(f"native_agentx_raw_topology_mismatch:{source_path}")
                    return False
        return True

    # Pinned InferenceX aggregates currently expose TP + EP, not PP + PCP.
    # Such a raw file may enrich corpus/cache telemetry only after Magpie's
    # fingerprint-bound snapshot established the complete physical topology.
    # In particular, TP alone must never become the per-GPU divisor.
    expected_gpu_count = first_int(measurement.get("agentx_gpu_count"))
    expected_topology = measurement.get("agentx_gpu_topology")
    expected_dimensions = (
        {
            dimension: value
            for dimension in ("tp", "pp", "pcp")
            if (value := first_int(expected_topology.get(dimension))) is not None and value > 0
        }
        if isinstance(expected_topology, dict)
        else {}
    )
    trusted_topology = (
        expected_gpu_count is not None
        and expected_gpu_count > 0
        and measurement.get("agentx_gpu_count_source") in _TRUSTED_NATIVE_AGENTX_GPU_COUNT_SOURCES
        and len(expected_dimensions) == 3
        and expected_gpu_count == expected_dimensions["tp"] * expected_dimensions["pp"] * expected_dimensions["pcp"]
    )
    if not trusted_topology:
        warnings.append(f"native_agentx_raw_topology_invalid:{source_path}")
        return False

    assert isinstance(expected_topology, dict)
    for dimension in ("tp", "pp", "pcp"):
        if dimension in raw_fields and raw_fields[dimension] != expected_dimensions[dimension]:
            warnings.append(f"native_agentx_raw_topology_mismatch:{source_path}")
            return False
    if "gpu_count" in raw_fields and raw_fields["gpu_count"] != expected_gpu_count:
        warnings.append(f"native_agentx_raw_topology_mismatch:{source_path}")
        return False
    if "ep" in raw_fields:
        expected_ep = first_int(expected_topology.get("ep"))
        if expected_ep is None:
            recipe = measurement.get("agentx_recipe")
            if isinstance(recipe, dict):
                expected_ep = first_int(recipe.get("ep"))
        if expected_ep is not None and raw_fields["ep"] != expected_ep:
            warnings.append(f"native_agentx_raw_topology_mismatch:{source_path}")
            return False
    return True


def _merge_native_agentx_raw(
    measurement: dict[str, Any],
    raw: dict[str, Any],
    *,
    source_path: Path,
) -> bool:
    """Enrich a native Magpie report from InferenceX's aggregate JSON.

    Magpie intentionally exposes a compact AgentX summary. The underlying
    aggregate retains corpus distributions, cache telemetry, and PCP topology
    that Hyperloom needs for its prompts and per-GPU comparisons.
    """
    if str(raw.get("scenario_type") or "").strip().lower() != "agentic-coding":
        return False
    if not _native_agentx_raw_matches_report(measurement, raw, source_path=source_path):
        return False
    request_metrics = raw.get("request_metrics")
    request_metrics = request_metrics if isinstance(request_metrics, dict) else {}
    tokens = request_metrics.get("tokens")
    tokens = tokens if isinstance(tokens, dict) else {}
    measurement["isl_distribution"] = _native_distribution(tokens.get("input"))
    measurement["osl_distribution"] = _native_distribution(tokens.get("output_actual"))

    dataset = raw.get("dataset")
    if isinstance(dataset, dict):
        measurement["agentx_dataset"] = dataset
        measurement["corpus_loader"] = str(dataset.get("loader") or dataset.get("name") or "")
    cache = request_metrics.get("cache")
    cache = cache if isinstance(cache, dict) else {}
    measurement["theoretical_prefix_cache_hit"] = to_float(cache.get("theoretical_cache_hit_rate"))
    server_metrics = raw.get("server_metrics")
    if isinstance(server_metrics, dict):
        measurement["agentx_server_cache"] = server_metrics.get("cache")

    _set_native_agentx_topology(measurement, raw, source="inferencex_raw")
    measurement["raw_result_path"] = str(source_path)
    return True


def _validate_native_agentx_protocol(
    measurement: dict[str, Any],
    *,
    workspace: Path | None,
    raw: dict[str, Any] | None,
    raw_path: Path | None,
) -> None:
    """Cross-bind the native report to its exact AIPerf protocol artifacts.

    Magpie's compact report is deliberately convenient, but its current
    ``benchmark_valid`` gate only proves positive throughput and an acceptable
    error ratio.  A one-second or cancelled run can therefore look publishable
    if that report is considered alone.  Native AgentX is accepted only when
    the Magpie summary, InferenceX aggregate, materialized recipe snapshot and
    AIPerf's own scenario export all describe the same complete run.
    """

    errors: list[str] = []

    def reject(code: str) -> None:
        if code not in errors:
            errors.append(code)

    def strict_number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    def strict_non_negative_int(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value if value >= 0 else None

    def metric_avg(payload: dict[str, Any], key: str) -> float | None:
        metric = payload.get(key)
        if not isinstance(metric, dict):
            return None
        return strict_number(metric.get("avg"))

    config: dict[str, Any] = {}
    if workspace is None:
        reject("workspace_missing")
    else:
        config_path = workspace / "config.yaml"
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            loaded = None
            reject("config_snapshot_unreadable")
        if isinstance(loaded, dict):
            wrapped = loaded.get("benchmark")
            config = wrapped if isinstance(wrapped, dict) else loaded
        else:
            reject("config_snapshot_invalid")

    raw_agentx = config.get("agentx")
    agentx = raw_agentx if isinstance(raw_agentx, dict) else {}
    raw_resolved = agentx.get("resolved")
    resolved = raw_resolved if isinstance(raw_resolved, dict) else {}
    if not resolved:
        reject("config_resolved_recipe_missing")
    workload = config.get("workload_spec")
    workload = workload if isinstance(workload, dict) else {}

    expected_fingerprint = _valid_recipe_fingerprint(
        resolved.get("recipe-fingerprint", resolved.get("recipe_fingerprint"))
    )
    if expected_fingerprint is None:
        reject("config_recipe_fingerprint_invalid")
    elif expected_fingerprint != _valid_recipe_fingerprint(measurement.get("agentx_recipe_fingerprint")):
        reject("config_report_recipe_fingerprint_mismatch")

    expected_concurrency = first_int(
        resolved.get("conc"),
        workload.get("concurrency"),
    )
    expected_duration = first_int(
        resolved.get("duration"),
        workload.get("duration_s"),
    )
    if expected_concurrency is None or expected_concurrency <= 0:
        reject("config_concurrency_invalid")
    if expected_duration is None or expected_duration <= 0:
        reject("config_duration_invalid")
    if str(measurement.get("agentx_mode") or "").strip().lower() == "canonical" and expected_duration != 3600:
        reject("canonical_duration_not_3600")

    report_recipe = measurement.get("agentx_recipe")
    report_recipe = report_recipe if isinstance(report_recipe, dict) else {}
    report_launch = measurement.get("agentx_launch")
    report_launch = report_launch if isinstance(report_launch, dict) else {}

    def require_same(label: str, *values: Any, fold: bool = False) -> None:
        normalized: list[Any] = []
        for value in values:
            if value is None or value == "":
                reject(f"{label}_missing")
                return
            item = str(value).strip().lower() if fold else value
            normalized.append(item)
        if len(set(normalized)) != 1:
            reject(f"{label}_mismatch")

    if raw is None or raw_path is None:
        reject("inferencex_aggregate_missing")
        raw = {}
    else:
        measurement["native_agentx_aggregate_path"] = str(raw_path)

    require_same(
        "recipe_fingerprint",
        expected_fingerprint,
        report_recipe.get("recipe_fingerprint"),
        report_launch.get("recipe_fingerprint"),
        raw.get("recipe_fingerprint"),
    )
    require_same("concurrency", expected_concurrency, report_recipe.get("conc"), raw.get("conc"))
    require_same("model", config.get("model"), resolved.get("model"), report_recipe.get("model"), raw.get("model"))
    require_same(
        "framework",
        config.get("framework"),
        resolved.get("framework"),
        report_recipe.get("framework"),
        raw.get("framework"),
        fold=True,
    )
    require_same(
        "precision",
        config.get("precision"),
        resolved.get("precision"),
        report_recipe.get("precision"),
        raw.get("precision"),
        fold=True,
    )
    require_same(
        "image",
        config.get("docker_image"),
        resolved.get("image"),
        report_recipe.get("image"),
        report_launch.get("docker_image"),
        raw.get("image"),
    )
    require_same(
        "model_prefix",
        resolved.get("model-prefix"),
        report_recipe.get("infmax_model_prefix"),
        raw.get("infmax_model_prefix"),
    )
    require_same(
        "launcher",
        config.get("benchmark_script"),
        report_launch.get("benchmark_script"),
    )
    require_same("recipe_name", agentx.get("recipe"), report_launch.get("recipe"))
    for label, config_key, report_key, raw_key in (
        ("tp", "tp", "tp", "tp"),
        ("pp", "pp", "pp", "pp"),
        ("ep", "ep", "ep", "ep"),
        ("pcp", "pcp-size", None, "pcp_size"),
    ):
        values = [resolved.get(config_key), raw.get(raw_key)]
        if report_key is not None:
            values.append(report_recipe.get(report_key))
        require_same(label, *values)

    accounting = raw.get("request_accounting")
    accounting = accounting if isinstance(accounting, dict) else {}
    records_total = strict_non_negative_int(accounting.get("records_total"))
    records_profiled = strict_non_negative_int(accounting.get("records_profiled"))
    records_dropped = strict_non_negative_int(accounting.get("records_dropped_total"))
    records_warmup = strict_non_negative_int(accounting.get("records_warmup_dropped"))
    records_errors = strict_non_negative_int(accounting.get("records_error_dropped"))
    successful = strict_non_negative_int(raw.get("num_requests_successful"))
    total = strict_non_negative_int(raw.get("num_requests_total"))
    if None in (
        records_total,
        records_profiled,
        records_dropped,
        records_warmup,
        records_errors,
        successful,
        total,
    ):
        reject("request_accounting_invalid")
    else:
        assert records_total is not None
        assert records_profiled is not None
        assert records_dropped is not None
        assert records_warmup is not None
        assert records_errors is not None
        assert successful is not None
        assert total is not None
        if records_total != records_profiled + records_dropped:
            reject("request_accounting_sum_mismatch")
        if successful != records_profiled:
            reject("request_accounting_success_mismatch")
        if total != records_total:
            reject("request_accounting_total_mismatch")
        if records_profiled <= 0:
            reject("request_accounting_empty")
        # InferenceX drops the union of warmup and error rows. A row may be in
        # both sets, so their sum need not equal records_dropped_total, but the
        # union is bounded by max(counts) and sum(counts).
        if not max(records_warmup, records_errors) <= records_dropped <= (records_warmup + records_errors):
            reject("request_accounting_drop_bounds_mismatch")
        error_categories = accounting.get("error_categories")
        if not isinstance(error_categories, dict):
            reject("request_accounting_error_categories_invalid")
        else:
            category_counts = [strict_non_negative_int(value) for value in error_categories.values()]
            if (
                any(value is None for value in category_counts)
                or sum(value for value in category_counts if value is not None) != records_errors
            ):
                reject("request_accounting_error_categories_mismatch")
        metrics_requests = measurement.get("completed_requests")
        if first_int(metrics_requests) != successful:
            reject("report_success_count_mismatch")

        report_accounting = measurement.get("agentx_request_accounting")
        if report_accounting != accounting:
            reject("report_request_accounting_mismatch")
        report_requests = measurement.get("agentx_requests")
        report_requests = report_requests if isinstance(report_requests, dict) else {}
        if strict_non_negative_int(report_requests.get("total")) != records_total:
            reject("report_request_total_mismatch")
        if strict_non_negative_int(report_requests.get("records_total")) != records_total:
            reject("report_records_total_mismatch")
        if strict_non_negative_int(report_requests.get("profiled_total")) != (successful + records_errors):
            reject("report_profiled_total_mismatch")
        if strict_non_negative_int(report_requests.get("errors")) != records_errors:
            reject("report_error_count_mismatch")
        if strict_non_negative_int(report_requests.get("warmup_dropped")) != records_warmup:
            reject("report_warmup_count_mismatch")
        error_rate_denominator = successful + records_errors
        report_error_rate = strict_number(report_requests.get("error_rate"))
        if error_rate_denominator <= 0:
            reject("report_error_rate_invalid")
        elif report_error_rate is None or not math.isclose(
            report_error_rate,
            records_errors / error_rate_denominator,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            reject("report_error_rate_mismatch")

    request_metrics = raw.get("request_metrics")
    request_metrics = request_metrics if isinstance(request_metrics, dict) else {}
    raw_throughput = request_metrics.get("throughput")
    raw_throughput = raw_throughput if isinstance(raw_throughput, dict) else {}
    raw_qps = request_metrics.get("qps")
    raw_qps = raw_qps if isinstance(raw_qps, dict) else {}
    report_throughput = measurement.get("agentx_report_throughput")
    report_throughput = report_throughput if isinstance(report_throughput, dict) else {}

    def nested_metric(payload: dict[str, Any], section: str, key: str) -> float | None:
        nested = payload.get(section)
        if not isinstance(nested, dict):
            return None
        return strict_number(nested.get(key))

    throughput_pairs = (
        (
            "request_throughput",
            strict_number(report_throughput.get("request_throughput")),
            strict_number(raw_qps.get("mean")),
        ),
        (
            "input_tokens_per_second",
            strict_number(report_throughput.get("input_tokens_per_second")),
            nested_metric(raw_throughput, "input", "tokens_per_second"),
        ),
        (
            "output_tokens_per_second",
            strict_number(report_throughput.get("output_tokens_per_second")),
            nested_metric(raw_throughput, "output", "tokens_per_second"),
        ),
        (
            "total_tokens_per_second",
            strict_number(report_throughput.get("total_tokens_per_second")),
            nested_metric(raw_throughput, "total", "tokens_per_second"),
        ),
        (
            "duration_seconds",
            strict_number(report_throughput.get("duration_seconds")),
            strict_number(raw_throughput.get("duration_seconds")),
        ),
    )
    for label, report_value, raw_value in throughput_pairs:
        if report_value is None or raw_value is None:
            reject(f"report_{label}_missing")
        elif report_value != raw_value:
            reject(f"report_{label}_mismatch")

    raw_latency = request_metrics.get("latency")
    report_latency = measurement.get("agentx_report_latency_seconds")
    if not isinstance(raw_latency, dict) or not isinstance(report_latency, dict):
        reject("report_latency_missing")
    elif report_latency != raw_latency:
        reject("report_latency_mismatch")

    aiperf: dict[str, Any] = {}
    artifact_path: Path | None = None
    if workspace is not None:
        try:
            artifacts = sorted(workspace.rglob("profile_export_aiperf.json"))
        except OSError:
            artifacts = []
        if len(artifacts) != 1:
            reject("aiperf_artifact_missing" if not artifacts else "aiperf_artifact_ambiguous")
        else:
            artifact_path = artifacts[0]
            payload = read_json(artifact_path, default=None, require_dict=True)
            if isinstance(payload, dict):
                aiperf = payload
                measurement["native_agentx_aiperf_path"] = str(artifact_path)
            else:
                reject("aiperf_artifact_unreadable")

    metadata = aiperf.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if metadata.get("scenario") != "inferencex-agentx-mvp":
        reject("aiperf_scenario_mismatch")
    if metadata.get("submission_valid") is not True:
        reject("aiperf_submission_invalid")
        raw_reasons = metadata.get("submission_invalid_reasons")
        if isinstance(raw_reasons, list):
            for reason in raw_reasons:
                reject(f"aiperf:{reason}")
    if aiperf.get("was_cancelled") is not False:
        reject("aiperf_cancelled_or_unknown")

    input_config = aiperf.get("input_config")
    input_config = input_config if isinstance(input_config, dict) else {}
    if input_config.get("scenario") != "inferencex-agentx-mvp":
        reject("aiperf_input_scenario_mismatch")
    expected_model = str(config.get("model") or "").strip()
    models = input_config.get("models")
    model_items = models.get("items") if isinstance(models, dict) else None
    if not isinstance(model_items, list) or len(model_items) != 1 or not isinstance(model_items[0], dict):
        reject("aiperf_models_invalid")
    elif str(model_items[0].get("name") or "").strip() != expected_model:
        reject("aiperf_model_mismatch")
    tokenizer = input_config.get("tokenizer")
    if not isinstance(tokenizer, dict) or not str(tokenizer.get("name") or "").strip():
        reject("aiperf_tokenizer_invalid")
    elif str(tokenizer.get("name") or "").strip() != expected_model:
        reject("aiperf_tokenizer_mismatch")
    phases = input_config.get("phases")
    profiling_phases = (
        [
            phase
            for phase in phases
            if isinstance(phase, dict) and str(phase.get("kind") or phase.get("name") or "").lower() == "profiling"
        ]
        if isinstance(phases, list)
        else []
    )
    if len(profiling_phases) != 1:
        reject("aiperf_profiling_phase_invalid")
        phase: dict[str, Any] = {}
    else:
        phase = profiling_phases[0]
        if first_int(phase.get("concurrency")) != expected_concurrency:
            reject("aiperf_concurrency_mismatch")
        phase_duration = strict_number(phase.get("duration"))
        if expected_duration is None or phase_duration != float(expected_duration):
            reject("aiperf_configured_duration_mismatch")
        if str(phase.get("timing_mode") or "").strip().lower() != "agentic_replay":
            reject("aiperf_timing_mode_mismatch")

    coverage = metadata.get("metric_duration_coverage")
    coverage_rows = coverage if isinstance(coverage, list) else []
    if len(coverage_rows) != 1 or not isinstance(coverage_rows[0], dict):
        reject("aiperf_duration_coverage_missing")
    else:
        row = coverage_rows[0]
        required_ratio = strict_number(row.get("required_ratio"))
        ttft_ratio = strict_number(row.get("ttft_ratio"))
        itl_ratio = strict_number(row.get("inter_token_latency_ratio"))
        coverage_duration = strict_number(row.get("expected_duration_seconds"))
        if profiling_phases and row.get("phase_name") != profiling_phases[0].get("name"):
            reject("aiperf_duration_coverage_phase_mismatch")
        if expected_duration is None or coverage_duration != float(expected_duration):
            reject("aiperf_duration_coverage_expected_mismatch")
        if required_ratio is None or required_ratio < 0.95 or required_ratio > 1.0:
            reject("aiperf_duration_coverage_threshold_invalid")
        if (
            required_ratio is None
            or ttft_ratio is None
            or itl_ratio is None
            or max(ttft_ratio, itl_ratio) < required_ratio
        ):
            reject("aiperf_duration_coverage_failed")

    observed_duration = metric_avg(aiperf, "benchmark_duration")
    if observed_duration is None or expected_duration is None or observed_duration < 0.95 * expected_duration:
        reject("aiperf_benchmark_duration_incomplete")
    aiperf_successes = metric_avg(aiperf, "request_count")
    if successful is None or aiperf_successes != float(successful):
        reject("aiperf_success_count_mismatch")

    aiperf_dataset = metadata.get("dataset")
    report_dataset = measurement.get("agentx_report_dataset")
    raw_dataset = raw.get("dataset")
    if not all(isinstance(value, dict) for value in (aiperf_dataset, report_dataset, raw_dataset)):
        reject("dataset_provenance_missing")
    else:
        assert isinstance(aiperf_dataset, dict)
        assert isinstance(report_dataset, dict)
        assert isinstance(raw_dataset, dict)
        if aiperf_dataset != report_dataset or aiperf_dataset != raw_dataset:
            reject("dataset_provenance_mismatch")
        if aiperf_dataset.get("source_type") != "public_dataset":
            reject("dataset_not_public")
        if first_int(aiperf_dataset.get("num_dataset_entries")) != 393:
            reject("dataset_entry_count_mismatch")
        loader = str(aiperf_dataset.get("loader") or "")
        if not loader.startswith("semianalysis_cc_traces_weka"):
            reject("dataset_loader_invalid")
        configured_corpus = str(workload.get("corpus") or "").strip()
        if configured_corpus and loader != configured_corpus:
            reject("dataset_config_mismatch")

    if measurement.get("native_agentx_aggregate_ambiguous") is True:
        reject("inferencex_aggregate_ambiguous")

    measurement["native_agentx_protocol_errors"] = errors
    measurement["native_agentx_protocol_valid"] = not errors
    reasons = measurement.get("submission_invalid_reasons")
    reasons = (
        [str(reason) for reason in reasons if not str(reason).startswith("native_agentx_protocol:")]
        if isinstance(reasons, list)
        else []
    )
    if errors:
        reasons.extend(f"native_agentx_protocol:{error}" for error in errors)
    measurement["submission_invalid_reasons"] = list(dict.fromkeys(reasons))
    measurement["submission_valid"] = bool(
        measurement.get("benchmark_valid") is True and measurement.get("publishable") is True and not errors
    )


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
        "scenario": report.get("scenario"),
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
        # What the client was asked to send, as the client recorded it. Read
        # back rather than taken from the env stack: the env is what we asked
        # for, this is what the run actually requested.
        "requested_requests": first_int(throughput.get("num_prompts")),
        "ttft_mean_ms": to_float(ttft.get("mean_ms")),
        "ttft_p99_ms": to_float(ttft.get("p99_ms")),
        "tpot_mean_ms": to_float(tpot.get("mean_ms")),
        "e2el_mean_ms": to_float(e2el.get("mean_ms")),
        "e2el_p99_ms": to_float(e2el.get("p99_ms")),
        "raw_result_path": None,
        "nonfatal_warnings": [],
    }
    if str(report.get("scenario") or "").strip().lower() == "agentx":
        measurement["native_agentx_report"] = True
        measurement["native_agentx_schema_valid"] = False
    _merge_native_agentx_report(measurement, report)
    if workspace is not None and measurement.get("native_agentx_report") is True:
        _merge_native_agentx_workspace_topology(measurement, workspace)

    origins: dict[str, str] = {}
    native_aggregate: dict[str, Any] | None = None
    native_aggregate_path: Path | None = None
    _tag_latency_origins(
        measurement,
        origins,
        label=LATENCY_FROM_REPORT,
        before=dict.fromkeys(_LATENCY_FIELDS),
    )

    if workspace is not None:
        for raw_path in _candidate_raw_jsons(workspace):
            raw = read_json(raw_path, default=None, require_dict=True)
            if not raw:
                continue
            if measurement.get("native_agentx_report") is True:
                if str(raw.get("scenario_type") or "").strip().lower() != "agentic-coding":
                    continue
                accepted = _merge_native_agentx_raw(measurement, raw, source_path=raw_path)
                if accepted:
                    if native_aggregate is not None:
                        measurement["native_agentx_aggregate_ambiguous"] = True
                        native_aggregate = None
                        native_aggregate_path = None
                        break
                    native_aggregate = raw
                    native_aggregate_path = raw_path
                continue
            if to_float(raw.get("output_throughput")) is None:
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
    if measurement.get("native_agentx_report") is True:
        _validate_native_agentx_protocol(
            measurement,
            workspace=workspace,
            raw=native_aggregate,
            raw_path=native_aggregate_path,
        )
    measurement["valid_measurement"] = is_valid_measurement(measurement)

    # Second-chance salvage from Magpie leak destinations when the in-workspace search found no usable measurement
    # (mtime-gated).
    if not measurement["valid_measurement"] and workspace is not None:
        for rescue_path in _rescue_candidate_paths(
            workspace,
            subprocess_started_unix=subprocess_started_unix,
        ):
            raw = read_json(rescue_path, default=None, require_dict=True)
            if not raw:
                continue
            native_report = measurement.get("native_agentx_report") is True
            native_raw = str(raw.get("scenario_type") or "").strip().lower() == "agentic-coding"
            if native_report and not native_raw:
                continue
            if native_report and not _native_agentx_raw_matches_report(
                measurement,
                raw,
                source_path=rescue_path,
            ):
                continue
            if not native_report and to_float(raw.get("output_throughput")) is None:
                continue
            # Copy the leak into the workspace BEFORE merging so the NFS clone stays self-contained.
            materialized = _materialize_rescue_into_workspace(
                rescue_path,
                workspace,
            )
            recorded_path = materialized if materialized is not None else rescue_path
            if native_report:
                accepted = _merge_native_agentx_raw(measurement, raw, source_path=recorded_path)
                if accepted:
                    native_aggregate = raw
                    native_aggregate_path = recorded_path
                    warnings.append(f"rescued_from_leaked_path:{rescue_path}")
                    if materialized is None:
                        warnings.append(f"rescued_copy_into_workspace_failed: {rescue_path}")
                    break
                continue
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
        if measurement.get("native_agentx_report") is True:
            _validate_native_agentx_protocol(
                measurement,
                workspace=workspace,
                raw=native_aggregate,
                raw_path=native_aggregate_path,
            )
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
    if _is_scriptable_measurement(result):
        # Scriptable workloads have their own image-quality contract and never
        # run the AgentX replay. An ambient HYPERLOOM_AGENTX left in the parent
        # shell must not make their otherwise valid result require an AgentX
        # submission verdict.
        from ._accuracy_gate import quality_gate_passed

        qg = result.get("quality_gate")
        return quality_gate_passed(qg, require=False)
    # Native reports and merged legacy AgentX raw results serialize their
    # scenario, so a resumed subprocess need not inherit HYPERLOOM_AGENTX.
    # A stray similarly named key on a synthetic result remains inert.
    from ._workload_envs import agentx_enabled

    agentx_result = str(result.get("scenario") or "").strip().lower() == "agentx" or agentx_enabled()
    if result.get("native_agentx_report") is True and result.get("native_agentx_schema_valid") is not True:
        return False
    if result.get("native_agentx_report") is True:
        if result.get("native_agentx_protocol_valid") is not True:
            return False
        gpu_count = first_int(result.get("agentx_gpu_count"))
        if (
            result.get("native_agentx_topology_conflict") is True
            or gpu_count is None
            or gpu_count <= 0
            or result.get("agentx_gpu_count_source") not in _TRUSTED_NATIVE_AGENTX_GPU_COUNT_SOURCES
        ):
            return False
    if agentx_result:
        verdict = result.get("submission_valid")
        if verdict is False:
            return False
        if verdict is not True:
            # The verdict is unknown: no --scenario was requested or the aiperf build predates the field. map_aiperf
            # writes the key unconditionally, so None arrives as a present key.
            from hyperloom.common.env import env_bool

            if not env_bool("HYPERLOOM_ALLOW_UNVERIFIED_SUBMISSION"):
                return False
    completed = to_int(result.get("completed_requests"))
    return completed is not None and completed > 0


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
    "served_complete_protocol",
    "_materialize_rescue_into_workspace",
]
