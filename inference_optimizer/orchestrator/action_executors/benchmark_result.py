# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark result parsing shared by Magpie-backed executors.

Magpie and shell wrappers can report failure after InferenceX has already
written valid throughput numbers (for example a post-benchmark cleanup error).
The optimizer should treat the measurement as usable whenever the benchmark
completed requests and produced positive throughput, while preserving the
wrapper status as diagnostics.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Default rescue path: scripts hardcoding ``--result-dir /workspace/``
# leak to ``/workspace/inferencex_result.json``. Extend/replace via
# ``$INFERENCE_OPTIMIZER_RESCUE_PATHS`` (see :func:`_rescue_candidate_paths`).
_DEFAULT_RESCUE_PATHS: tuple[Path, ...] = (
    Path("/workspace/inferencex_result.json"),
)


# Wrapper-side diagnostic files hardcoded under ``/workspace/`` (server
# log, GPU monitor CSV, profile relay trace). Unlike
# ``inferencex_result.json`` they don't feed measurement recovery, but
# they live outside the per-task workspace so the NFS clone misses them;
# :func:`harvest_leaked_artifacts` copies fresh matches in.
_DEFAULT_LEAK_ARTIFACT_GLOBS: tuple[str, ...] = (
    "server.log",
    "gpu_metrics.csv",
    "profile_*.trace.json.gz",
    "inferencex_result*.json",
)
_DEFAULT_LEAK_ARTIFACT_ROOT: Path = Path("/workspace")

# Slack subtracted from ``subprocess_started_unix`` before comparing a leak's
# ``st_mtime``, to reject stale prior-run leaks without false-dropping fresh
# ones. 1s absorbs clock-vs-mtime / FS-granularity skew (NFS ~1s) while
# staying below the multi-second gap that separates genuinely stale leaks.
_MTIME_GATE_SLACK_SEC: float = 1.0


def _to_float(value: Any) -> float | None:
    """Coerce a value to ``float``, rejecting bools and ``None``.

    Args:
        value (Any): The value to coerce.

    Returns:
        float | None: The parsed float, or ``None`` when the value is a
        bool, ``None``, or not convertible.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    """Coerce a value to ``int``, rejecting bools and ``None``.

    Args:
        value (Any): The value to coerce.

    Returns:
        int | None: The parsed int, or ``None`` when the value is a
        bool, ``None``, or not convertible.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float | None:
    """Return the first value that parses as a float.

    Args:
        *values (Any): Candidate values, tried in order.

    Returns:
        float | None: The first successfully parsed float, or ``None``.
    """
    for value in values:
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _first_int(*values: Any) -> int | None:
    """Return the first value that parses as an int.

    Args:
        *values (Any): Candidate values, tried in order.

    Returns:
        int | None: The first successfully parsed int, or ``None``.
    """
    for value in values:
        parsed = _to_int(value)
        if parsed is not None:
            return parsed
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON object from ``path``, tolerating read/parse errors.

    Args:
        path (Path): The JSON file to read.

    Returns:
        dict[str, Any] | None: The parsed mapping, or ``None`` on IO /
        decode error or when the top-level JSON is not an object.
    """
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _candidate_raw_jsons(workspace: Path) -> list[Path]:
    """Return likely InferenceX result files, preferring baseline over profile.

    Args:
        workspace (Path): The task workspace to scan recursively.

    Returns:
        list[Path]: Candidate ``*.json`` result paths (excluding
        ``benchmark_report.json``), ordered baseline-before-profile.
    """
    paths = [
        p for p in workspace.rglob("*.json")
        if p.name != "benchmark_report.json"
    ]
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
    """Return absolute paths to known Magpie leak destinations.

    Scripts hardcoding ``--result-dir /workspace/`` land the InferenceX
    result at ``/workspace/inferencex_result.json`` outside the per-task
    workspace. Order: ``$INFERENCE_OPTIMIZER_RESCUE_PATHS`` (files verbatim;
    dirs scanned for ``inferencex_result*.json``) → :data:`_DEFAULT_RESCUE_PATHS`.

    When ``subprocess_started_unix`` is given, candidates older than it
    (minus :data:`_MTIME_GATE_SLACK_SEC`) are dropped as stale prior-run
    leaks. Never raises: per-candidate I/O errors are swallowed.
    """
    candidates: list[Path] = []
    seen: set[Path] = set()

    def _push(path: Path) -> None:
        """Add ``path`` to the candidate list if it passes all gates.

        Resolves the path, skips duplicates and in-workspace files,
        requires a regular file, and (when a start time is known)
        drops stale candidates older than the subprocess launch.

        Args:
            path (Path): A candidate leak path to consider.

        Returns:
            None: Mutates the enclosing ``candidates``/``seen`` sets.
        """
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        # Skip files already inside the workspace (handled by
        # ``_candidate_raw_jsons``).
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
    env_entries = [
        part.strip() for part in env_raw.split(":") if part.strip()
    ] if env_raw else []
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

    for default in _DEFAULT_RESCUE_PATHS:
        _push(default)

    return candidates


def _materialize_rescue_into_workspace(
    rescue_path: Path,
    workspace: Path,
) -> Path | None:
    """Copy a leaked InferenceX result back into the task workspace.

    Best-effort ``shutil.copy2`` (preserving basename) so the NFS clone is
    self-contained. Returns the destination on success, or ``None`` on I/O
    error (caller falls back to the leak path) or when the source already
    lives inside the workspace.
    """
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
            rescue_path, destination, exc,
        )
        return None
    return destination


def _resolve_leak_roots(leak_root: Path | None) -> tuple[Path, ...]:
    """Return the directory roots to scan for wrapper-side leak files.

    Order: explicit ``leak_root`` kwarg (tests) →
    ``$INFERENCE_OPTIMIZER_LEAK_ROOTS`` (colon-separated) →
    :data:`_DEFAULT_LEAK_ARTIFACT_ROOT` (``/workspace``).
    """
    if leak_root is not None:
        return (leak_root,)
    env_raw = os.environ.get("INFERENCE_OPTIMIZER_LEAK_ROOTS", "").strip()
    if env_raw:
        parts = [Path(p.strip()) for p in env_raw.split(":") if p.strip()]
        if parts:
            return tuple(parts)
    return (_DEFAULT_LEAK_ARTIFACT_ROOT,)


def harvest_leaked_artifacts(
    destination: Path,
    *,
    subprocess_started_unix: float | None = None,
    leak_root: Path | None = None,
    extra_globs: tuple[str, ...] = (),
) -> list[tuple[Path, Path]]:
    """Copy known Magpie/InferenceX leak artifacts into ``destination``.

    For every glob in :data:`_DEFAULT_LEAK_ARTIFACT_GLOBS` (extensible via
    ``extra_globs``), scans each root from :func:`_resolve_leak_roots`,
    mtime-gates against ``subprocess_started_unix`` (skips stale), and
    ``shutil.copy2``-s each match (source never moved). Returns
    ``(leak_path, copy_path)`` tuples for audit; never raises (per-artifact
    errors are isolated).
    """
    harvested: list[tuple[Path, Path]] = []
    leak_roots = _resolve_leak_roots(leak_root)
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning(
            "benchmark_result.harvest: cannot prepare destination=%s: %s",
            destination, exc,
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
                    # Already under the workspace — nothing to harvest.
                    continue
                except ValueError:
                    pass
                if not match.is_file():
                    continue
                if subprocess_started_unix is not None:
                    try:
                        mtime = match.stat().st_mtime
                    except OSError:
                        continue
                    if mtime + _MTIME_GATE_SLACK_SEC < float(
                        subprocess_started_unix
                    ):
                        continue
                destination_path = destination / match.name
                try:
                    shutil.copy2(match, destination_path)
                except OSError as exc:
                    log.warning(
                        "benchmark_result.harvest: copy %s -> %s failed: %s",
                        match, destination_path, exc,
                    )
                    continue
                harvested.append((match, destination_path))
    return harvested


def _merge_raw_result(
    measurement: dict[str, Any],
    raw: dict[str, Any],
    *,
    source_path: Path,
) -> None:
    """Fill missing measurement fields from a raw InferenceX result.

    Only keys that are still ``None`` in ``measurement`` are populated,
    so an earlier (preferred) source is never overwritten.

    Args:
        measurement (dict[str, Any]): The measurement dict to fill in
            place.
        raw (dict[str, Any]): The raw InferenceX result mapping.
        source_path (Path): Path the raw result was read from; recorded
            as ``raw_result_path`` when not already set.

    Returns:
        None: ``measurement`` is mutated in place.
    """
    if measurement.get("output_throughput") is None:
        measurement["output_throughput"] = _to_float(raw.get("output_throughput"))
    if measurement.get("request_throughput") is None:
        measurement["request_throughput"] = _to_float(raw.get("request_throughput"))
    if measurement.get("total_token_throughput") is None:
        measurement["total_token_throughput"] = _to_float(
            raw.get("total_token_throughput")
        )
    if measurement.get("completed_requests") is None:
        measurement["completed_requests"] = _first_int(
            raw.get("completed_requests"),
            raw.get("completed"),
        )
    if measurement.get("duration_seconds") is None:
        measurement["duration_seconds"] = _first_float(
            raw.get("duration_seconds"),
            raw.get("duration"),
        )
    if measurement.get("ttft_mean_ms") is None:
        measurement["ttft_mean_ms"] = _to_float(raw.get("mean_ttft_ms"))
    if measurement.get("ttft_p99_ms") is None:
        measurement["ttft_p99_ms"] = _to_float(raw.get("p99_ttft_ms"))
    if measurement.get("tpot_mean_ms") is None:
        measurement["tpot_mean_ms"] = _to_float(raw.get("mean_tpot_ms"))
    if measurement.get("e2el_mean_ms") is None:
        measurement["e2el_mean_ms"] = _first_float(
            raw.get("mean_e2el_ms"),
            raw.get("mean_latency_ms"),
        )
    if measurement.get("e2el_p99_ms") is None:
        measurement["e2el_p99_ms"] = _first_float(
            raw.get("p99_e2el_ms"),
            raw.get("p99_latency_ms"),
        )
    if measurement.get("raw_result_path") is None:
        measurement["raw_result_path"] = str(source_path)


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
        "request_throughput": _to_float(throughput.get("request_throughput")),
        "output_throughput": _to_float(throughput.get("output_throughput")),
        "total_token_throughput": _to_float(
            throughput.get("total_token_throughput")
        ),
        "completed_requests": _first_int(
            throughput.get("completed_requests"),
            throughput.get("completed"),
        ),
        "duration_seconds": _to_float(throughput.get("duration_seconds")),
        "ttft_mean_ms": _to_float(ttft.get("mean_ms")),
        "ttft_p99_ms": _to_float(ttft.get("p99_ms")),
        "tpot_mean_ms": _to_float(tpot.get("mean_ms")),
        "e2el_mean_ms": _to_float(e2el.get("mean_ms")),
        "e2el_p99_ms": _to_float(e2el.get("p99_ms")),
        "raw_result_path": None,
        "nonfatal_warnings": [],
    }

    if workspace is not None:
        for raw_path in _candidate_raw_jsons(workspace):
            raw = _load_json(raw_path)
            if not raw or _to_float(raw.get("output_throughput")) is None:
                continue
            _merge_raw_result(measurement, raw, source_path=raw_path)
            if is_valid_measurement(measurement):
                break

    warnings = measurement["nonfatal_warnings"]
    if report and report.get("success") is not True:
        warnings.append("benchmark_report_success_false")
    if workspace is not None and measurement.get("raw_result_path"):
        warnings.append("raw_inferencex_result_used")

    _derive_tpot_if_missing(measurement, report)
    measurement["valid_measurement"] = is_valid_measurement(measurement)

    # Second-chance salvage from Magpie leak destinations when the
    # in-workspace search found no usable measurement (mtime-gated).
    if (
        not measurement["valid_measurement"]
        and workspace is not None
    ):
        for rescue_path in _rescue_candidate_paths(
            workspace,
            subprocess_started_unix=subprocess_started_unix,
        ):
            raw = _load_json(rescue_path)
            if not raw or _to_float(raw.get("output_throughput")) is None:
                continue
            # Copy the leak into the workspace BEFORE merging so
            # ``raw_result_path`` advertises the in-workspace copy and the
            # NFS clone stays self-contained. Best-effort: on copy failure
            # we fall back to the leak path rather than drop the measurement.
            materialized = _materialize_rescue_into_workspace(
                rescue_path, workspace,
            )
            recorded_path = materialized if materialized is not None else rescue_path
            _merge_raw_result(measurement, raw, source_path=recorded_path)
            if is_valid_measurement(measurement):
                warnings.append(f"rescued_from_leaked_path:{rescue_path}")
                if materialized is None:
                    warnings.append(
                        "rescued_copy_into_workspace_failed: "
                        f"{rescue_path}"
                    )
                break
        _derive_tpot_if_missing(measurement, report)
        measurement["valid_measurement"] = is_valid_measurement(measurement)
    return measurement


def _derive_tpot_if_missing(
    measurement: dict[str, Any],
    report: dict[str, Any] | None,
) -> None:
    """Fill ``tpot_mean_ms`` from ``(e2el - ttft) / (osl - 1)`` when absent.

    Best-effort: only derives when end-to-end and TTFT latencies are
    available and an output sequence length greater than 1 can be
    resolved from the report. Leaves the field untouched otherwise.
    """
    if measurement.get("tpot_mean_ms") is not None:
        return
    e2el = _to_float(measurement.get("e2el_mean_ms"))
    ttft = _to_float(measurement.get("ttft_mean_ms"))
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
            candidates.extend(
                section.get(k) for k in ("osl", "output_len", "max_tokens")
            )
    for value in candidates:
        n = _to_int(value)
        if n is not None and n > 0:
            return n
    return None


def is_valid_measurement(result: dict[str, Any] | None) -> bool:
    """Return whether a measurement reflects a usable benchmark result.

    A measurement is valid when it reports positive output throughput
    and at least one completed request.

    Args:
        result (dict[str, Any] | None): The measurement dict to check.

    Returns:
        bool: ``True`` if throughput and completed requests are both
        positive; ``False`` otherwise.
    """
    if not isinstance(result, dict):
        return False
    output_tput = _to_float(result.get("output_throughput"))
    completed = _to_int(result.get("completed_requests"))
    return (
        output_tput is not None
        and output_tput > 0
        and completed is not None
        and completed > 0
    )


__all__ = [
    "extract_benchmark_measurement",
    "harvest_leaked_artifacts",
    "is_valid_measurement",
    "_materialize_rescue_into_workspace",
]
