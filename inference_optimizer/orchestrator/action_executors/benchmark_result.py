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

# Default rescue path: Magpie's ``dsr1_fp8_mi300x.sh`` hardcodes
# ``--result-dir /workspace/`` so the canonical leak destination is
# ``/workspace/inferencex_result.json``. Operators can extend or
# replace the list via ``$INFERENCE_OPTIMIZER_RESCUE_PATHS`` (see
# :func:`_rescue_candidate_paths`). Kept as a list so future variants
# (e.g. ``inferencex_result_eval.json``) can be appended in one place.
_DEFAULT_RESCUE_PATHS: tuple[Path, ...] = (
    Path("/workspace/inferencex_result.json"),
)


# Magpie shell wrappers (``InferenceX/benchmarks/{single_node,multi_node}/*.sh``)
# hardcode several additional output paths under ``/workspace/`` that
# the wrapper status code does NOT depend on:
#
# * ``SERVER_LOG=/workspace/server.log`` — every ``single_node/*.sh``
#   redirects sglang/vllm server stdout+stderr here. Vital for
#   diagnosing GPU OOMs / init failures / accuracy-mode crashes.
# * ``GPU_METRICS_CSV=/workspace/gpu_metrics.csv`` — default of
#   ``benchmark_lib.sh:start_gpu_monitor`` (nvidia-smi / amd-smi
#   per-second power / temp / utilisation telemetry).
# * ``/workspace/profile_<RESULT_FILENAME>.trace.json.gz`` — the
#   PROFILE relay copy of the torch profiler trace
#   (``benchmark_lib.sh`` ``[PROFILE] Relay trace prepared``).
#
# Unlike ``inferencex_result.json`` these files are NOT consulted by
# :func:`extract_benchmark_measurement` to recover a measurement —
# they are wrapper-side diagnostics. But they live OUTSIDE the
# per-task workspace under ``<session>/runs/<action>/<task_id>/``,
# so the NFS clone of that task dir misses them entirely.
# :func:`harvest_leaked_artifacts` copies every fresh match into the
# task workspace so the canonical NFS layout is self-contained.
# Globs are evaluated via :meth:`Path.glob` against ``/workspace`` so
# new wrapper-side conventions (e.g. ``profile_<run>.trace.json.gz``)
# can be added without code changes.
_DEFAULT_LEAK_ARTIFACT_GLOBS: tuple[str, ...] = (
    "server.log",
    "gpu_metrics.csv",
    "profile_*.trace.json.gz",
    "inferencex_result*.json",
)
_DEFAULT_LEAK_ARTIFACT_ROOT: Path = Path("/workspace")

# Slack subtracted from ``subprocess_started_unix`` before comparing it
# against a candidate leak's ``st_mtime``. The cutoff exists to reject
# *stale* leaks from a previous run or an earlier grid variant, which are
# always seconds-to-hours older than the current launch. A file written
# *after* the launch can nonetheless report an ``st_mtime`` slightly
# *behind* the ``time.time()`` we snapshot as ``subprocess_started_unix``:
# filesystem mtime resolution is coarser than the monotonic-ish wall clock
# (commonly 1s on NFS, and even on local ext4 we measured the freshly
# written file's mtime trailing the pre-write clock by several ms). The old
# 1e-3 (1ms) tolerance was below that skew, so genuinely fresh salvage
# candidates were misclassified as stale and dropped — silently failing a
# benchmark whose result had actually been written. One full second is
# comfortably larger than any observed clock-vs-mtime skew / FS granularity,
# yet far smaller than the multi-second gaps that separate genuinely stale
# prior-run leaks, so it preserves the staleness guard without false drops.
_MTIME_GATE_SLACK_SEC: float = 1.0


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _to_int(value)
        if parsed is not None:
            return parsed
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _candidate_raw_jsons(workspace: Path) -> list[Path]:
    """Return likely InferenceX result files, preferring baseline over profile."""
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

    Magpie's bundled ``dsr1_fp8_mi300x.sh`` (and a handful of
    framework-specific siblings) hardcode ``--result-dir /workspace/``,
    so when the optimizer launches a benchmark with a per-task workspace
    the InferenceX result JSON lands at ``/workspace/inferencex_result.json``
    instead of inside ``workspace``. The wrapper then reports failure
    because it can't find a result file under its own workspace.

    Resolution order:

    1. ``$INFERENCE_OPTIMIZER_RESCUE_PATHS`` — colon-separated list of
       files and/or directories. Files are returned verbatim;
       directories are scanned for ``inferencex_result*.json``.
    2. Default fallback :data:`_DEFAULT_RESCUE_PATHS`
       (currently ``/workspace/inferencex_result.json``).

    When ``subprocess_started_unix`` is provided, candidates whose mtime
    is **earlier** than that timestamp (minus :data:`_MTIME_GATE_SLACK_SEC`)
    are dropped. This guards against a stale leak from a *previous* run
    masquerading as the current run's result — without the cutoff we'd risk
    re-promoting a stale 1761.6 tok/s number after a fresh run silently
    failed. The slack absorbs the skew between the wall clock used for
    ``subprocess_started_unix`` and the coarser filesystem mtime resolution
    so a leak written just after launch is not misjudged as stale.

    The function intentionally never raises: any I/O error on a single
    candidate is swallowed (the file is just skipped) so the caller's
    fast-path (no rescue) is preserved.
    """
    candidates: list[Path] = []
    seen: set[Path] = set()

    def _push(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        # Skip files that live inside the workspace already — those are
        # the responsibility of ``_candidate_raw_jsons`` and including
        # them again here would just duplicate work.
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

    Magpie scripts that hardcode ``--result-dir /workspace/`` (e.g.
    ``dsr1_fp8_mi300x.sh``) write ``inferencex_result.json`` outside
    the per-task workspace the optimizer created. When that workspace
    later gets cloned to NFS (the canonical artifact location for
    cross-host inspection), the JSON is missing — only the wrapper
    summary ``benchmark_report.json`` is present.

    This helper performs a best-effort ``shutil.copy2`` of the leaked
    file into ``workspace`` (preserving its basename so multiple
    variants — ``inferencex_result.json``,
    ``inferencex_result_eval.json``, etc. — remain distinguishable).
    On any I/O error we return ``None`` and the caller falls back to
    advertising the leak path verbatim, so this only adds capability;
    it never breaks the read path.

    Returns the destination path on success, or ``None`` on failure /
    when the source already lives inside the workspace (the latter
    means the path came through ``_candidate_raw_jsons`` already and
    we should not re-copy onto ourselves).
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

    Priority:

    1. Explicit ``leak_root`` kwarg (single path, used by unit tests
       to isolate the scan from the host's real ``/workspace``).
    2. ``$INFERENCE_OPTIMIZER_LEAK_ROOTS`` — colon-separated paths.
       Set this in production when the deployed wrapper writes leaks
       to a non-``/workspace/`` location, or in test envs to point
       at a sandbox.
    3. Default :data:`_DEFAULT_LEAK_ARTIFACT_ROOT`
       (``/workspace``), matching the destinations hardcoded into
       Magpie's bundled ``InferenceX/benchmarks/*.sh``.
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

    Magpie's bundled shell wrappers hardcode several output paths
    under ``/workspace/`` (server log, GPU monitor CSV, profile relay
    trace, InferenceX result JSON). When the optimizer launches a
    benchmark inside a per-task workspace under
    ``<session_dir>/runs/.../`` those artifacts end up outside the
    session tree — the NFS clone of the task therefore misses them
    entirely.

    For every glob in :data:`_DEFAULT_LEAK_ARTIFACT_GLOBS` (extensible
    via ``extra_globs``) this helper scans each leak root resolved by
    :func:`_resolve_leak_roots` (explicit ``leak_root`` kwarg, then
    ``$INFERENCE_OPTIMIZER_LEAK_ROOTS``, then ``/workspace``),
    mtime-gates against ``subprocess_started_unix`` (same discipline
    as :func:`_rescue_candidate_paths` — files older than the
    subprocess start are treated as stale and skipped), and
    ``shutil.copy2``-s each match into ``destination``. The source is
    never moved or deleted; the copy preserves the basename so
    artifacts remain distinguishable (``server.log``,
    ``gpu_metrics.csv``, etc.).

    Returns a list of ``(leak_path, copy_path)`` tuples for audit.
    Each step is wrapped in its own ``try`` so a permission error on
    one artifact does not block the rest of the harvest. The function
    intentionally never raises: callers always receive a (possibly
    empty) list and can decide what to surface to the prompt.
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
                    # Match is already under the workspace — nothing
                    # to harvest, it landed in the right place
                    # already.
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
    documented Magpie leak destinations (see
    :func:`_rescue_candidate_paths`) when the in-workspace search fails.
    Callers (executors) capture ``time.time()`` immediately before
    invoking the benchmark subprocess and forward it here so we only
    adopt a leaked result that was written *after* this run started.
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

    # Second-chance salvage: try documented Magpie leak destinations
    # (e.g. ``/workspace/inferencex_result.json``) when the in-workspace
    # search didn't yield a usable measurement. mtime gating inside
    # :func:`_rescue_candidate_paths` prevents stale leaks from being
    # adopted as this run's result.
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
            # Copy the leaked InferenceX result back into the task
            # workspace BEFORE merging so ``raw_result_path`` always
            # advertises the in-workspace copy. This keeps NFS clones
            # of ``<session>/runs/<action>/<task_id>/`` self-contained
            # — the canonical artifact is materialized alongside
            # ``benchmark_report.json`` rather than left at the leak
            # location (typically ``/workspace/`` outside the session
            # tree). ``_materialize_rescue_into_workspace`` is
            # best-effort: on permission / disk failure we fall back
            # to the leak path so a successful salvage measurement is
            # never discarded just because the copy step couldn't run.
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
