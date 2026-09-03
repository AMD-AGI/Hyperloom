#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TraceLens analysis tool for the resident Kernel-agent skill.

Conservative: records every step, writes a stable artifact set, supports TraceLens
capture directories, and has a dry-run path that works without TraceLens installed.
"""

import argparse
import ast
import asyncio
import contextlib
import csv
import functools
import gzip
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from hyperloom.orchestrator.framework.paths import (
        resolve_patch_target_roots as _resolve_patch_target_roots,
    )
except ImportError:
    _resolve_patch_target_roots = None

try:
    from hyperloom.orchestrator.framework.paths import (
        resolve_flydsl_source_roots as _resolve_flydsl_source_roots,
    )
except ImportError:
    _resolve_flydsl_source_roots = None

try:
    from hyperloom.orchestrator.framework.paths import (
        FRAMEWORK_SOURCE_PACKAGES as _FRAMEWORK_SOURCE_PACKAGES,
    )
    from hyperloom.orchestrator.framework.paths import (
        resolve_kernel_search_roots as _resolve_kernel_search_roots,
    )
except ImportError:
    _FRAMEWORK_SOURCE_PACKAGES = None
    _resolve_kernel_search_roots = None

try:
    from apply_kernel_patch import known_target_roots as _known_target_roots
except ImportError:
    _known_target_roots = None

try:
    from _task_group_contract import native_operation_key as _native_operation_key
except ImportError:
    _native_operation_key = None

# Unguarded like the other sibling imports below: a fallback here would silently
# turn name normalization into the identity and mis-key every kernel lookup.
from _task_group_contract import _strip_dispatch_decoration

try:
    import aiter.jit.core as _aiter_jit_core  # type: ignore[import-untyped]
except Exception:
    _aiter_jit_core = None

from tracelens_arch_benchmark import normalize_platform, populate_gpu_arch_json
from tracelens_skill_runner import (
    aggregate_by_source_function,
    discover_capture_folder,
    extract_compute_pct_from_analysis_md,
    extract_exposed_comm_pct_from_analysis_md,
    extract_idle_pct_from_analysis_md,
    normalize_upstream_category,
    parse_analysis_md,
    run_tracelens_skill,
)

from _io_utils import append_log, atomic_write_json, read_last_lines, safe_float, utc_now
from _nccl_summary_candidates import extract_collective_candidates

# Standalone-tool workspace-root resolver (cannot import hyperloom.inference_optimizer.session.paths; see _paths.py).
from _paths import workspace_root

# Capture-vs-workload trace classification, shared across routes so a sidecar is
# recognised identically whichever backend reads the profile.
from _capture_shapes import is_capture_fragment as _shared_is_capture_fragment

# Trace-health gate thresholds + warnings (high idle, low compute): shared single
# source of truth so the TraceLens and bypass routes gate on identical semantics.
from _idle_gate import (
    build_graph_under_recorded_warning as _build_graph_under_recorded_warning,
    build_high_idle_warning as _build_high_idle_warning,
    build_low_compute_warning as _build_low_compute_warning,
    resolve_idle_pct_threshold,
    resolve_min_compute_pct_threshold,
)

# Canonical roofline_source provenance enum, shared with the bypass route so both
# emit the field from one vocabulary (see _roofline_source for the value ladder).
from _roofline_source import (
    ANALYTICAL as _RL_ANALYTICAL,
    PLACEHOLDER as _RL_PLACEHOLDER,
)

# Vendor-operator-playbook registry: routes a closed-source vendor op (no
# rewritable device source) to a validated KernelForge task bundle instead of
# a source rewrite -- see KernelForge PR #88's mori dispatch/combine gap.
from _vendor_operator_playbooks import (
    match_vendor_operator_playbook,
    resolve_kernel_anchor_path,
)

# Shared with the kernel-opt side; these tools also run as standalone scripts
# outside an importable hyperloom, where the artifact is simply not written.
try:
    from hyperloom.common import kernel_source_contract as _KSC

    # This script runs as a standalone subprocess against the *installed*
    # hyperloom, which need not be the same tree as this file (cf.
    # runtime/source-mirrors/). A contract module that predates the API used
    # below would raise AttributeError at first use and abort the whole
    # analysis rather than degrade. Treat an incompatible (too-old) module as
    # absent so the ``_KSC is not None`` guards below fall back to an unwritten
    # artifact instead of crashing.
    if not all(
        hasattr(_KSC, _name)
        for _name in (
            "METHOD_ACTIVE_FINDER",
            "METHOD_SYMBOL_INDEX",
            "METHOD_CURATED",
            "METHOD_GREP",
            "METHOD_UNRESOLVED",
            "KNOWN_METHODS",
            "make_entry",
            "make_document",
            "validate_document",
        )
    ):
        _KSC = None  # type: ignore[assignment]
except ImportError:  # pragma: no cover - standalone invocation
    _KSC = None  # type: ignore[assignment]

try:
    from hyperloom.common.kernel_shape_contract import (
        REVIEW_DERIVED_PROVENANCE as _REVIEW_DERIVED_PROVENANCE,
    )
except ImportError:  # pragma: no cover - standalone invocation
    # This script also runs against an installed hyperloom that may predate the
    # constant; the literal keeps the review's dims labelled either way.
    _REVIEW_DERIVED_PROVENANCE = "review_derived"

log = logging.getLogger(__name__)

# Duplicated from kernel_source_contract.SOURCE_RESOLUTION_FILENAME: the
# standalone-script path cannot import hyperloom.common. Keep the two in sync.
_SOURCE_RESOLUTION_NAME = "kernel_source_resolution.json"


# Candidate building keeps a broad pool; dispatch grouping owns the real budget gate.
# Override with HYPERLOOM_KERNEL_CANDIDATES_TOP_K; non-positive means unbounded.
_DEFAULT_KERNEL_CANDIDATES_TOP_K = 100


def _default_top_k() -> int:
    """Resolve the default kernel-candidate pool size.

    Reads ``HYPERLOOM_KERNEL_CANDIDATES_TOP_K`` when set (``0``/negative =>
    unbounded pool, represented internally as a very large cap), otherwise
    falls back to :data:`_DEFAULT_KERNEL_CANDIDATES_TOP_K`.

    Returns:
        The candidate-build-time cap (a large number when unbounded).
    """
    raw = os.environ.get("HYPERLOOM_KERNEL_CANDIDATES_TOP_K", "").strip()
    if not raw:
        return _DEFAULT_KERNEL_CANDIDATES_TOP_K
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_KERNEL_CANDIDATES_TOP_K
    # 0 / negative => no build-time cap (dispatch layer owns the budget).
    return val if val > 0 else 1_000_000


# Statuses that mean "we have an editable device source to optimize".
_ROUTABLE_STATUS = "resolved"

# TraceLens may annotate a candidate op name with the steady-state phase it was
# observed in, e.g. ``aiter::fmoe_g1u1 (prefill)``. The mapping is keyed by the
# bare op name, so strip a trailing phase tag before the (exact) dict lookup.
_PHASE_SUFFIX_RE = re.compile(r"\s*\((?:prefill|decode|prefilldecode|mixed)\)\s*$")

# Editable source extensions: native device code plus repo-resident Triton .py.
_NATIVE_SOURCE_EXTS = (".cu", ".cuh", ".hip", ".h")
_PY_DIST_ROOT = "/usr/local/lib/python3.12/dist-packages/"

# Active-finder resolver: resolve a kernel to its editable source in the
# *currently installed* framework tree by demangling its device symbol. This is
# the deterministic op->source tier (it replaces the retired static op_to_source
# map) and self-heals across file moves/renames and vLLM/aiter/sglang version
# drift. On a miss the pipeline falls through to the trace-stack/grep/LLM tiers.
# Optional: if the finder modules are unavailable, tier-1 resolution is skipped.
try:  # package import (TraceLens route / tests)
    from . import kernel_source_index as _kernel_source_index
    from . import source_resolver as _active_finder
except ImportError:  # flat import (standalone: tools/ on sys.path)
    try:
        import kernel_source_index as _kernel_source_index  # type: ignore[no-redef]
        import source_resolver as _active_finder  # type: ignore[no-redef]
    except ImportError:
        _kernel_source_index = None  # type: ignore[assignment]
        _active_finder = None  # type: ignore[assignment]

# Owns the only AST reading of what a Triton kernel definition looks like, which
# source_type_for needs to tell a Triton ``.py`` from any other Python file.
try:  # package import (TraceLens route / tests)
    from ._bypass_source_resolver import triton_def_line as _triton_def_line
except ImportError:  # flat import (standalone: tools/ on sys.path)
    try:
        from _bypass_source_resolver import triton_def_line as _triton_def_line  # type: ignore[no-redef]
    except ImportError:
        _triton_def_line = None  # type: ignore[assignment]

_ACTIVE_FINDER_METHOD = getattr(_KSC, "METHOD_ACTIVE_FINDER", "active_finder")

# Resolution methods whose verdict is carried on the ``op_to_source_*`` fields
# and honored directly by classify_patchability (instead of re-running its
# heuristics). The active finder is the deterministic tier; ``op_to_source`` is
# retained only for backward compatibility with any externally-supplied verdict.
_CURATED_LIKE_METHODS = frozenset({"op_to_source", _ACTIVE_FINDER_METHOD, "symbol_index"})


@dataclass
class OpResolution:
    """One op's resolved editable source, as produced by the active finder.

    The finder resolves a single device symbol to a single source, so in the
    current pipeline it only ever emits ``kind="single"``.

    Attributes:
        op_name: The CPU op name that was looked up.
        kind: ``single`` / ``dispatch``.
        status: ``resolved`` / ``non_rewritable`` / ``no_kernel`` / ``unresolved``.
        patchable: The curated patchability verdict (may be ``None``).
        framework: Framework that owns the source (``aiter``/``vllm``/...).
        sources: Absolute editable source path(s) this resolution owns
            (``.cu``/``.cuh``/``.hip``/``.h`` or repo-resident ``.py``);
            empty when there is no editable source.
        reason: Skip reason (``triton``/``aten``/...) or the entry ``label``.
        matched_route: For ``dispatch``, the ``match`` glob that fired.
        resolution_method: The resolver that produced this verdict (the active
            finder, ``active_finder``); stamped onto candidates for the audit.
        target_index: Which of ``sources`` this leaf optimizes; see
            :meth:`leaf_resolutions`.
    """

    op_name: str
    kind: str
    status: str
    patchable: bool | None
    framework: str | None
    sources: list[str] = field(default_factory=list)
    reason: str = ""
    matched_route: str | None = None
    resolution_method: str = _ACTIVE_FINDER_METHOD
    target_index: int = 0

    @property
    def primary_source(self) -> str:
        """The editable kernel source this leaf optimizes, or ``""``."""
        if 0 <= self.target_index < len(self.sources):
            return self.sources[self.target_index]
        return self.sources[0] if self.sources else ""

    @property
    def is_routable(self) -> bool:
        """True when there is a resolved, patchable, editable source to optimize."""
        return self.status == _ROUTABLE_STATUS and bool(self.patchable) and bool(self.sources)

    def leaf_resolutions(self) -> list["OpResolution"]:
        """Expand into one routable leaf per editable source file to optimize.

        A routable resolution with N ``sources`` yields N leaves (one per
        editable source file); a non-routable resolution yields none.
        """
        if not self.is_routable:
            return []
        return [replace(self, target_index=i) for i in range(len(self.sources))]

    def stamp_onto(self, item: dict[str, Any]) -> None:
        """Record this verdict on a candidate (audit + classify_patchability)."""
        item["source_resolution_method"] = self.resolution_method
        item["op_to_source_status"] = self.status
        item["op_to_source_kind"] = self.kind
        item["op_to_source_patchable"] = self.patchable
        item["op_to_source_reason"] = self.reason
        if self.matched_route:
            item["op_to_source_matched_route"] = self.matched_route

    def apply_to(self, item: dict[str, Any]) -> None:
        """Override an item's source with this leaf's editable ``.cu`` (ground truth).

        Promotes the prior ``.py`` launcher to ``launcher_source_file`` and stamps
        the op's full ``.cu`` set on ``kernel_sources`` (sibling context).
        """
        launcher = item.get("source_file") or item.get("tracelens_launcher_path") or ""
        item["source_file"] = self.primary_source
        item["kernel_sources"] = list(self.sources)
        if launcher and launcher != self.primary_source:
            item["launcher_source_file"] = launcher
            item["source_promoted_from_launcher"] = True
        self.stamp_onto(item)


# HIGH_IDLE_PCT_THRESHOLD_* and the idle-gate helpers now live in _idle_gate
# (imported above) as the shared single source of truth across trace routes.

ARCH_BENCHMARK_TIMEOUT_ENV = "TRACELENS_ARCH_BENCHMARK_TIMEOUT_SEC"
ARCH_BENCHMARK_TIMEOUT_FLOOR_S = 600


def _is_safe_litellm_gateway() -> bool:
    """True when the Claude SDK targets a strict LiteLLM-style gateway (#574).

    ``LLM_GATEWAY_KEY`` is an explicit gateway signal and wins on its own; a
    deployment may front the gateway on a hostname with no protocol marker.
    Otherwise detected via the SDK's ``ANTHROPIC_BASE_URL`` / ``OPENAI_BASE_URL``
    host; other backends are left alone.
    """
    if os.environ.get("LLM_GATEWAY_KEY", "").strip():
        return True
    base_url = (os.environ.get("ANTHROPIC_BASE_URL", "") or os.environ.get("OPENAI_BASE_URL", "")).lower()
    # Generic protocol markers by default (no operator/brand strings shipped);
    # a specific deployment can add its own gateway host substrings via
    # HYPERLOOM_STRICT_GATEWAY_MARKERS (comma-separated).
    markers = tuple(
        m.strip().lower()
        for m in os.environ.get("HYPERLOOM_STRICT_GATEWAY_MARKERS", "litellm,llm-proxy").split(",")
        if m.strip()
    )
    return any(m in base_url for m in markers)


def _resolve_tracelens_model() -> str:
    """Resolve ``ANTHROPIC_MODEL`` to the gateway's dash-form id (#574).

    Only the strict SAFE/LiteLLM gateway rejects the image's dot form
    (``Claude-Opus-4.7``); for it, map to dash (``claude-opus-4-7``). Other
    backends keep the raw id; empty env yields ``""`` (SDK default).

    Returns:
        The model id to pass to the SDK, normalized only for SAFE/LiteLLM.
    """
    raw = os.environ.get("ANTHROPIC_MODEL", "").strip()
    if not raw:
        return ""
    if not _is_safe_litellm_gateway():
        return raw
    return raw.lower().replace(".", "-")


def _resolve_arch_benchmark_timeout_s() -> int:
    """Return the GPU arch microbenchmark timeout in seconds (floor 600s).

    Configured via ``TRACELENS_ARCH_BENCHMARK_TIMEOUT_SEC``. Empty, non-numeric,
    or out-of-range values fall back to the 600s floor rather than crashing the
    pipeline with a ``ValueError`` before the microbenchmark runs.

    Returns:
        The arch microbenchmark timeout in seconds (at least the 600s floor).
    """
    raw = os.environ.get(ARCH_BENCHMARK_TIMEOUT_ENV, "").strip()
    if not raw:
        return ARCH_BENCHMARK_TIMEOUT_FLOOR_S
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return ARCH_BENCHMARK_TIMEOUT_FLOOR_S
    return max(ARCH_BENCHMARK_TIMEOUT_FLOOR_S, value)


def _evaluate_high_idle_gate(idle_pct: float | None, report_path: Path) -> tuple[float, dict[str, Any] | None]:
    """Return the idle threshold plus a warning when the gate is exceeded.

    ``_build_high_idle_warning`` and ``resolve_idle_pct_threshold`` are the
    shared ``_idle_gate`` helpers imported at module top, so the threshold, gate
    semantics, and warning shape stay unified across routes.
    """
    threshold = resolve_idle_pct_threshold()
    if idle_pct is None or idle_pct <= threshold:
        return threshold, None
    return threshold, _build_high_idle_warning(
        idle_pct=idle_pct,
        threshold_pct=threshold,
        report_path=report_path,
    )


def _graph_coverage_from_raw_trace(trace_path: str | Path | None) -> dict[str, Any]:
    """Return the bypass reader's ``graph_coverage`` for the raw trace, or ``{}``.

    Reuses the tested ``_bypass_trace_reader.analyze_trace`` graph-launch
    detection so the TraceLens route can tell a cuda/HIP-graph under-recorded
    capture (profiler activity-buffer overflow → only ~1 of N replays recorded →
    inflated idle%) apart from a genuinely idle/launch-bound workload. Best
    effort: any failure returns ``{}`` so the caller falls back to the plain
    idle gate (never worse than today).

    ``graph_coverage`` is derived from the launch/kernel correlation scan and is
    independent of the returned aggregation lists, so we pass ``emit_launches=
    False`` and ``top_k=1`` to avoid materializing the per-launch rows and full
    top-N lists on large ``--skip-split`` raw traces.
    """
    if not trace_path:
        return {}
    try:
        import _bypass_trace_reader as _reader

        analyze = _reader.analyze_trace(str(trace_path), top_k=1, emit_launches=False)
        cov = analyze.get("graph_coverage") if isinstance(analyze, dict) else None
        return cov if isinstance(cov, dict) else {}
    except Exception:  # noqa: BLE001 - guard is advisory; never block on it
        return {}


def _evaluate_idle_gate_with_graph_guard(
    idle_pct: float | None,
    report_path: Path,
    trace_path: str | Path | None,
) -> tuple[float, dict[str, Any] | None, dict[str, Any] | None]:
    """Idle gate that first honors graph under-recording.

    A cuda/HIP-graph trace that under-records replays reports an unreliable
    (inflated) idle%, so gating candidates on it wrongly suppresses the whole
    hot-kernel list on a workload that is actually compute-bound (the exact
    failure the bypass route already guards against via
    ``bypass_graph_under_recorded``). When under-recording is detected we skip
    the high-idle suppression and surface the graph-under-recorded warning
    instead; otherwise the plain idle gate applies unchanged.

    Returns:
        ``(threshold, high_idle_warning, graph_under_recorded_warning)`` where at
        most one of the two warnings is non-``None``.
    """
    threshold = resolve_idle_pct_threshold()
    if idle_pct is None or idle_pct <= threshold:
        return threshold, None, None
    cov = _graph_coverage_from_raw_trace(trace_path)
    if cov.get("graph_under_recorded"):
        return (
            threshold,
            None,
            _build_graph_under_recorded_warning(
                graph_launch_count=int(cov.get("graph_launch_count", 0) or 0),
                idle_pct=float(idle_pct),
            ),
        )
    _, high_idle_warning = _evaluate_high_idle_gate(idle_pct, report_path)
    return threshold, high_idle_warning, None


def _evaluate_low_compute_gate(
    compute_pct: float | None,
    exposed_comm_pct: float | None,
    report_path: Path,
) -> tuple[float, dict[str, Any] | None]:
    """Return the compute threshold plus a warning when the compute share is too low.

    Complements :func:`_evaluate_idle_gate_with_graph_guard`. The idle gate
    cannot see a window whose wall time is consumed by a spin-waiting
    collective, because that wait is charged as GPU-busy time -- such a trace
    reports ~0% idle alongside a single-digit compute share. Since a kernel
    rewrite is bounded by the compute share, both regimes warrant the same
    response: suppress the hot-kernel list and let the Coordinator route
    elsewhere.

    Args:
        compute_pct: Compute share of trace wall time, or ``None`` when the
            source report did not expose it (gate is skipped).
        exposed_comm_pct: Exposed-communication share, for warning context.
        report_path: Path to the source report, recorded in the warning.

    Returns:
        The resolved threshold and the ``low_gpu_compute_pct`` warning, or
        ``None`` when the gate does not fire.
    """
    threshold = resolve_min_compute_pct_threshold()
    if compute_pct is None or compute_pct >= threshold:
        return threshold, None
    return threshold, _build_low_compute_warning(
        compute_pct=compute_pct,
        threshold_pct=threshold,
        report_path=report_path,
        exposed_comm_pct=exposed_comm_pct,
    )


def _build_trace_split_warning(
    *,
    trace_input: Path,
    split_dir: Path,
    split_rc: int,
    mixed_count: int,
    decode_count: int,
    prefilldecode_count: int,
) -> dict[str, Any]:
    """Build the ``trace_split_no_steady_state`` trace-health warning.

    Emitted when the TraceLens splitter produces no steady-state chunks,
    so analyzing the raw trace would risk misleading high-idle results.

    Args:
        trace_input (Path): The raw trace that was handed to the splitter.
        split_dir (Path): Directory the splitter wrote its outputs into.
        split_rc (int): Return code from the splitter subprocess.
        mixed_count (int): Number of ``mixed`` steady-state chunks produced.
        decode_count (int): Number of ``decode_only`` chunks produced.
        prefilldecode_count (int): Number of ``prefilldecode`` chunks produced.

    Returns:
        dict[str, Any]: A structured warning entry with code
            ``trace_split_no_steady_state`` and the supporting counts/message.
    """
    return {
        "code": "trace_split_no_steady_state",
        "severity": "warning",
        "trace_input": str(trace_input),
        "split_dir": str(split_dir),
        "split_returncode": split_rc,
        "mixed_count": mixed_count,
        "decode_only_count": decode_count,
        "prefilldecode_count": prefilldecode_count,
        "message": (
            "TraceLens splitter produced no steady-state chunks; refusing "
            "to analyze the raw trace because that can report misleading "
            "high idle and suppress valid kernel opportunities. Verify the "
            "profile request used TraceLens-compatible annotations and enough "
            "NUM_PROMPTS to reach the requested start_step/num_steps window."
        ),
    }


def _build_pretrim_no_steady_state_warning(
    *,
    trace_input: Path,
    steps: int,
    leading_outliers: int,
    max_dropped_steps: int,
    outlier_factor: float,
) -> dict[str, Any]:
    """Build the ``pretrim_no_steady_state`` trace-health warning.

    Emitted when so many leading steps exceed their phase median that the
    capture cannot be said to have reached steady state. One such step is the
    profiler-start transient and gets trimmed; a run of them is a workload still
    warming up, which trimming cannot fix and which every percentage derived
    from the window has to be read against.

    Args:
        trace_input (Path): The capture the pretrimmer inspected.
        steps (int): Total step annotations in the capture.
        leading_outliers (int): Leading steps found over the outlier threshold.
        max_dropped_steps (int): The ceiling that was exceeded.
        outlier_factor (float): Multiple of the phase median used as threshold.

    Returns:
        dict[str, Any]: A structured warning entry with code
            ``pretrim_no_steady_state`` and the supporting counts/message.
    """
    return {
        "code": "pretrim_no_steady_state",
        "severity": "warning",
        "trace_input": str(trace_input),
        "steps": steps,
        "leading_outliers": leading_outliers,
        "max_dropped_steps": max_dropped_steps,
        "outlier_factor": outlier_factor,
        "message": (
            f"{leading_outliers} of {steps} leading steps run more than "
            f"{outlier_factor:g}x their phase median, past the "
            f"{max_dropped_steps} a profiler-start transient accounts for, so "
            "the capture never reached steady state and the head of the window "
            "was left in place. Compute%/Comm% and every kernel's gpu_pct below "
            "are computed over a window that includes the warm-up. Re-profile "
            "with more warm-up steps before acting on them."
        ),
    }


def _check_selected_chunk_has_gpu_events(
    *,
    split_dir: Path,
    selected_chunk: Path,
    mode: str,
    available_modes: "dict[str, tuple[str, list[Path]]]",
) -> "dict[str, Any] | None":
    """Verify the ``--steady-state-mode``-selected chunk actually contains GPU events.

    Reads the splitter's ``execution_details.csv`` and returns ``None`` when the
    chunk carries real GPU work, else a ``steady_state_chunk_empty`` warning the
    caller appends and raises on.

    Args:
        split_dir: Directory holding the splitter's ``execution_details.csv``.
        selected_chunk: The chunk file selected by the steady-state mode.
        mode: The requested ``--steady-state-mode``.
        available_modes: Mapping of mode to its ``(label, chunks)`` for
            surfacing non-empty alternatives.

    Returns:
        ``None`` when the chunk has real GPU work, else a
        ``steady_state_chunk_empty`` warning dict.
    """
    details_path = split_dir / "execution_details.csv"
    if not details_path.is_file():
        # No CSV: let the chunk through (idle gate still applies).
        return None
    try:
        with details_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error):
        return None

    selected_resolved = str(selected_chunk.resolve())
    selected_row: dict[str, str] | None = None
    for row in rows:
        out_path = row.get("output_path", "")
        if not out_path:
            continue
        try:
            if str(Path(out_path).resolve()) == selected_resolved:
                selected_row = row
                break
        except (OSError, ValueError):
            continue
    if selected_row is None:
        return None

    def _f(name: str) -> float:
        """Read a numeric field from the selected splitter CSV row.

        Args:
            name (str): Column name to read from ``selected_row``.

        Returns:
            float: The parsed value, or ``0.0`` when missing/unparseable.
        """
        try:
            return float(selected_row.get(name) or "0") or 0.0
        except (TypeError, ValueError):
            return 0.0

    num_gpu_events = int(_f("num_gpu_events"))
    gpu_busy_duration = _f("gpu_busy_duration")
    if num_gpu_events > 0 and gpu_busy_duration > 0.0:
        return None  # chunk carries real GPU work -- proceed.

    # Empty: surface which other modes' chunks DO have gpu events for re-issue.
    non_empty_modes: list[str] = []
    for other_mode, (label, chunks) in available_modes.items():
        if other_mode == mode or not chunks:
            continue
        other_resolved = str(chunks[0].resolve())
        for row in rows:
            try:
                if str(Path(row.get("output_path", "")).resolve()) != other_resolved:
                    continue
            except (OSError, ValueError):
                continue
            try:
                other_events = int(float(row.get("num_gpu_events") or "0"))
                other_busy = float(row.get("gpu_busy_duration") or "0")
            except (TypeError, ValueError):
                other_events, other_busy = 0, 0.0
            if other_events > 0 and other_busy > 0.0:
                non_empty_modes.append(other_mode)
            break

    return {
        "code": "steady_state_chunk_empty",
        "severity": "blocking",
        "requested_mode": mode,
        "selected_chunk": str(selected_chunk),
        "num_gpu_events": num_gpu_events,
        "gpu_busy_duration": gpu_busy_duration,
        "non_empty_modes": non_empty_modes,
        "remediation": (
            "Re-issue roofline with env "
            "INFERENCE_OPTIMIZER_STEADY_STATE_MODE set to one of "
            f"{non_empty_modes or ['(none of the splitter outputs has GPU events; re-profile required)']}. "
            "Most common cause: short / batched workload (e.g. "
            "NUM_PROMPTS<=CONC*OSL/2) where prefill is burst-shaped so "
            "the mixed window degenerates to PD=0; switching to "
            "'prefilldecode' picks up the real GEMM/attention region."
        ),
        "message": (
            f"TraceLens splitter selected chunk ({mode}) has "
            f"num_gpu_events={num_gpu_events}, "
            f"gpu_busy_duration={gpu_busy_duration:.1f}us -- structurally "
            "empty. Refusing to feed it into TraceLens analysis (would "
            "produce a misleading high-idle Executive Summary). The "
            "coordinator should re-issue roofline with a different "
            "--steady-state-mode per the 'remediation' field."
        ),
    }


# Chunk-quality gate: a structurally-non-empty chunk can still be garbage.
# Emits ``steady_state_chunk_low_quality`` when an alternate mode is materially
# better; returns None otherwise (avoids a retry-loop).
_DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO = 0.05  # 5%
# Alternate must beat the requested mode by this margin to avoid thrashing.
_CHUNK_QUALITY_ALTERNATE_MARGIN = 0.10  # 10 ppt


def _resolve_min_busy_ratio() -> float:
    """Return the minimum chunk busy-ratio threshold for the N36 quality gate.

    Reads ``INFERENCE_OPTIMIZER_CHUNK_QUALITY_MIN_BUSY_RATIO`` and falls back
    to :data:`_DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO` when unset, out of the
    ``[0.0, 1.0]`` range, or unparseable.

    Returns:
        float: The busy-ratio threshold in the inclusive range ``[0.0, 1.0]``.
    """
    raw = os.environ.get(
        "INFERENCE_OPTIMIZER_CHUNK_QUALITY_MIN_BUSY_RATIO",
        "",
    ).strip()
    if not raw:
        return _DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO
    try:
        v = float(raw)
        return v if 0.0 <= v <= 1.0 else _DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO
    except ValueError:
        return _DEFAULT_CHUNK_QUALITY_MIN_BUSY_RATIO


def _busy_ratio(num_events: float, busy_us: float, dur_us: float) -> float | None:
    """Compute the clamped busy ratio for a chunk.

    Args:
        num_events: GPU event count for the chunk.
        busy_us: GPU busy duration in microseconds.
        dur_us: Total chunk duration in microseconds.

    Returns:
        ``busy_us / dur_us`` clamped to ``[0, 1]``, or ``None`` when undefined
        (the caller defers to the N25 structural gate).
    """
    if dur_us <= 0.0 or num_events <= 0:
        return None
    return max(0.0, min(1.0, busy_us / dur_us))


def _check_selected_chunk_has_gpu_events_quality(
    *,
    split_dir: "Path",
    selected_chunk: "Path",
    mode: str,
    available_modes: "dict[str, tuple[str, list[Path]]]",
) -> "dict[str, Any] | None":
    """Quality gate complementing the structural GPU-events gate.

    See the module-level chunk-quality comment for the gate's rationale.

    Args:
        split_dir: Directory holding the splitter's ``execution_details.csv``.
        selected_chunk: The chunk file selected by the steady-state mode.
        mode: The requested ``--steady-state-mode``.
        available_modes: Mapping of mode to its ``(label, chunks)`` for
            evaluating better alternatives.

    Returns:
        ``None`` when the chunk is acceptable (busy_ratio >= threshold), no
        alternate is materially better, or the CSV/row is absent. Otherwise a
        ``steady_state_chunk_low_quality`` warning dict (same shape as N25).
    """
    details_path = split_dir / "execution_details.csv"
    if not details_path.is_file():
        return None
    try:
        with details_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error):
        return None

    def _row_for(chunk_path: "Path") -> "dict[str, str] | None":
        """Find the splitter CSV row whose ``output_path`` is the chunk.

        Args:
            chunk_path (Path): Chunk file to match against ``output_path``.

        Returns:
            dict[str, str] | None: The matching CSV row, or ``None`` when no
                row resolves to the same path.
        """
        resolved = str(chunk_path.resolve())
        for row in rows:
            out_path = row.get("output_path", "")
            if not out_path:
                continue
            try:
                if str(Path(out_path).resolve()) == resolved:
                    return row
            except (OSError, ValueError):
                continue
        return None

    def _stats(row: "dict[str, str] | None") -> "tuple[int, float, float]":
        """Extract ``(num_gpu_events, gpu_busy_duration, gpu_duration)``.

        Args:
            row (dict[str, str] | None): A splitter CSV row, or ``None``.

        Returns:
            tuple[int, float, float]: The event count, busy duration (us),
                and total duration (us); all zero when ``row`` is ``None``.
        """
        if row is None:
            return 0, 0.0, 0.0

        def _f(k: str) -> float:
            """Read a numeric field from the CSV row.

            Args:
                k (str): Column name to read.

            Returns:
                float: The parsed value, or ``0.0`` when missing/unparseable.
            """
            try:
                return float(row.get(k) or "0") or 0.0
            except (TypeError, ValueError):
                return 0.0

        return int(_f("num_gpu_events")), _f("gpu_busy_duration"), _f("gpu_duration")

    selected_row = _row_for(selected_chunk)
    if selected_row is None:
        return None
    sel_events, sel_busy, sel_dur = _stats(selected_row)
    sel_ratio = _busy_ratio(sel_events, sel_busy, sel_dur)
    if sel_ratio is None:
        # Can't measure ratio; defer to the structural-empty gate.
        return None
    threshold = _resolve_min_busy_ratio()
    if sel_ratio >= threshold:
        return None

    # Below threshold: look for an alternate mode with materially higher busy_ratio.
    alternates: list[tuple[str, float]] = []
    for other_mode, (_label, chunks) in available_modes.items():
        if other_mode == mode or not chunks:
            continue
        other_row = _row_for(chunks[0])
        if other_row is None:
            continue
        oth_events, oth_busy, oth_dur = _stats(other_row)
        oth_ratio = _busy_ratio(oth_events, oth_busy, oth_dur)
        if oth_ratio is None:
            continue
        if oth_ratio >= threshold and (oth_ratio - sel_ratio) >= _CHUNK_QUALITY_ALTERNATE_MARGIN:
            alternates.append((other_mode, oth_ratio))
    if not alternates:
        return None  # No better mode exists; let roofline_failure_streak path handle.

    # Best alternate first (the retry path picks the head of non_empty_modes).
    alternates.sort(key=lambda mr: -mr[1])
    non_empty_modes = [m for m, _r in alternates]
    return {
        "code": "steady_state_chunk_low_quality",
        "severity": "blocking",
        "requested_mode": mode,
        "selected_chunk": str(selected_chunk),
        "num_gpu_events": sel_events,
        "gpu_busy_duration": sel_busy,
        "gpu_duration": sel_dur,
        "busy_ratio": sel_ratio,
        "threshold": threshold,
        "non_empty_modes": non_empty_modes,
        "alternate_busy_ratios": dict(alternates),
        "remediation": (
            "Re-issue roofline with env "
            "INFERENCE_OPTIMIZER_STEADY_STATE_MODE set to one of "
            f"{non_empty_modes}. The TraceLens splitter chunk for the "
            f"requested mode '{mode}' is {sel_ratio * 100:.2f}% busy "
            f"(threshold {threshold * 100:.0f}%) -- non-empty but "
            "substantively garbage. Most common cause for prefill-"
            "heavy workloads: profile window misalignment "
            "(_workload_envs.delay_iters formula only considers OSL, "
            "so high-ISL workloads land in pure-decode windows)."
        ),
        "message": (
            f"TraceLens splitter selected chunk ({mode}) busy_ratio="
            f"{sel_ratio * 100:.3f}% (events={sel_events}, "
            f"busy={sel_busy:.1f}us / dur={sel_dur:.1f}us) -- below "
            f"the {threshold * 100:.0f}% threshold and alternate "
            f"modes have higher busy_ratio. Refusing to feed it into "
            "TraceLens analysis (would produce a misleading analysis.md "
            "with reusable_native_kernel_ids=[] and stall the "
            "optimization loop, per DSR1-0528 10k/1k case)."
        ),
    }


KERNEL_HINTS = (
    "kernel",
    "triton",
    "hip",
    "cuda",
    "rocblas",
    "hipblas",
    "aiter",
    "fmha",
    "gemm",
    "attention",
    "moe",
    "rmsnorm",
    "layernorm",
)
RUNTIME_API_NAMES = {
    "hipeventsynchronize",
    "hipdevicesynchronize",
    "hipstreamsynchronize",
    "hipgraphlaunch",
    "hiplaunchkernel",
    "hipmodulelaunchkernel",
    "hipmemcpy",
    "hipmemset",
    "cudaeventsynchronize",
    "cudadevicesynchronize",
    "cudastreamsynchronize",
}
# TRACELENS_ROOT comes from env / --tracelens-root (fail loudly if absent);
# the internal extension is opt-in via TRACELENS_INTERNAL_ROOT.
DEFAULT_TRACELENS_INTERNAL_ROOT = ""


def update_status(
    status_path: Path,
    *,
    state: str,
    current_step: str,
    log_path: Path,
    artifact_paths: dict[str, str],
    run_id: str,
    started_at: str,
    error: str | None = None,
) -> None:
    """Atomically write a tracelens_analysis run-status JSON file.

    Captures the current run state, recent log tail, and (on terminal
    states) the wall-clock duration so downstream collectors can build a
    timeline event.

    Args:
        status_path (Path): Destination status JSON file.
        state (str): Current run state (e.g. ``running``, ``succeeded``).
        current_step (str): Human-readable label of the active step.
        log_path (Path): Log file whose size/tail are recorded.
        artifact_paths (dict[str, str]): Map of artifact names to paths.
        run_id (str): Unique identifier for this run.
        started_at (str): ISO-8601 start time used to compute duration.
        error (str | None): Error message recorded when the run failed.
    """
    updated_at = utc_now()
    payload: dict[str, Any] = {
        "tool": "tracelens_analysis",
        "run_id": run_id,
        "state": state,
        "current_step": current_step,
        "pid": os.getpid(),
        "started_at": started_at,
        "updated_at": updated_at,
        "log_path": str(log_path),
        "artifact_paths": artifact_paths,
        "offset_bytes": log_path.stat().st_size if log_path.exists() else 0,
        "last_lines": read_last_lines(log_path),
    }
    # Emit ended_at + duration_seconds on terminal states for the timeline.
    if state in ("succeeded", "failed", "aborted", "cancelled"):
        payload["ended_at"] = updated_at
        try:
            start_dt = datetime.fromisoformat(started_at)
            end_dt = datetime.fromisoformat(updated_at)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            payload["duration_seconds"] = max(
                0.0,
                (end_dt - start_dt).total_seconds(),
            )
        except (ValueError, TypeError):
            payload["duration_seconds"] = None
    if error:
        payload["error"] = error
    atomic_write_json(status_path, payload)


def open_json(path: Path) -> dict[str, Any]:
    """Load a JSON file, transparently handling ``.gz`` compression.

    Args:
        path (Path): JSON or gzipped-JSON file to read.

    Returns:
        dict[str, Any]: The parsed JSON payload.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


def count_gpu_kernel_events(trace_file: Path, max_events: int = 1_000_000) -> int:
    """Count GPU kernel events in a torch_profiler trace.

    Used as a pre-flight check for CPU-only traces. Counts only real GPU
    kernels via :func:`is_kernel_event`, not host-side wrappers.

    Args:
        trace_file: Path to the torch_profiler trace (JSON or ``.gz``).
        max_events: Early-exit cap on the number of events counted.

    Returns:
        The GPU kernel event count, or 0 when the trace is unreadable.
    """
    try:
        payload = open_json(trace_file)
    except Exception:
        return 0
    events = payload.get("traceEvents") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return 0
    count = 0
    for ev in events:
        if isinstance(ev, dict) and is_kernel_event(ev):
            count += 1
            if count >= max_events:
                break
    return count


#: How many times the median step *of its own phase* a leading step must exceed
#: before it counts as a profiler-start transient rather than real work. Observed
#: transients run two-to-three orders of magnitude over the median (479x on the
#: reference capture) while healthy decode steps hold inside 2% of it, so
#: anything in this range separates them with a wide margin either way.
_PRETRIM_OUTLIER_FACTOR = 10.0

#: Fewest step annotations a trace must carry before the median is worth
#: trusting. Below this a single slow step skews the median enough that the
#: comparison stops meaning anything, so the trim is skipped rather than guessed.
_PRETRIM_MIN_STEPS = 8

#: Fewest steps of the *same phase* before that phase's median is used as a
#: baseline. Prefill and decode differ by one to two orders of magnitude, so a
#: phase has to recur often enough for its own median to mean something; a
#: leading step whose phase is rarer than this is left alone rather than measured
#: against a population it does not belong to.
_PRETRIM_MIN_PHASE_STEPS = 3

#: Ceiling on how many leading steps may be dropped. A transient is one step in
#: every capture examined; needing more than a handful means the run never
#: reached steady state, which the splitter and the health gates should see
#: rather than have quietly trimmed away.
_PRETRIM_MAX_DROPPED_STEPS = 4


#: Categories that live on the device timeline. Everything else in a torch
#: trace -- ``cpu_op``, ``python_function``, ``user_annotation``,
#: ``cuda_runtime`` -- is host-side.
_GPU_TIMELINE_CATS = frozenset({"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"})


def _step_annotation_spans(events: list[Any]) -> list[tuple[float, float, str]]:
    """``(ts, dur, name)`` for every GPU-side step annotation, ordered by start.

    Args:
        events (list[Any]): ``traceEvents`` from a torch-profiler trace.

    Returns:
        list[tuple[float, float, str]]: One entry per ``step[...]`` annotation on
            the GPU timeline. Empty when the trace carries no step annotations.
    """
    spans: list[tuple[float, float, str]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("cat") != "gpu_user_annotation":
            continue
        name = ev.get("name")
        if not isinstance(name, str) or not name.startswith("step["):
            continue
        ts = ev.get("ts")
        dur = ev.get("dur")
        if isinstance(ts, (int, float)) and isinstance(dur, (int, float)):
            spans.append((float(ts), float(dur), name))
    spans.sort()
    return spans


def _host_step_starts(events: list[Any]) -> list[float]:
    """Host-timeline start of every ``step[...]`` annotation, ordered by start.

    The host runs ahead of the device -- a step's launches are issued while the
    previous step is still executing -- so a step's host span begins before its
    device span, and the two have to be paired to cut them at the right places.

    Returned positionally rather than keyed by annotation name, because the
    names are the framework's and are not unique: a build that omits the
    cumulative-sequence-length fields repeats ``step[DECODE bs=36]`` for every
    step at that batch size, and keying on it silently pairs the surviving step
    with the *first* step of the capture.

    Positional pairing needs the two timelines to hold one annotation per step,
    and that is a property of the capture rather than something this can assume.
    Across 32 measured per-rank captures only 5 held it; the other 27 carried
    128 host ``step[...]`` annotations against a single device one, with the
    step's own *children* (``scheduler.run_batch``, ``copy_result_to_cpu``)
    projected onto the device 127 times each. Whatever produces that, the
    missing entries are not a tail: the Nth host start is then some earlier
    step's, and both the count check and the ordering check downstream would
    pass while the host cut lands too early. The caller therefore requires the
    counts to match exactly and refuses the trim otherwise.

    Args:
        events (list[Any]): ``traceEvents`` from a torch-profiler trace.

    Returns:
        list[float]: Host-timeline start timestamps, ascending.
    """
    starts: list[float] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("cat") != "user_annotation":
            continue
        name = ev.get("name")
        ts = ev.get("ts")
        if not isinstance(name, str) or not name.startswith("step["):
            continue
        if not isinstance(ts, (int, float)):
            continue
        starts.append(float(ts))
    starts.sort()
    return starts


def _step_phase(name: str) -> str:
    """The phase token of a ``step[...]`` annotation name.

    ``step[DECODE bs=64 g_sk=580480]`` gives ``DECODE`` and
    ``step[EXTEND bs=1 toks=16384]`` gives ``EXTEND``.

    Args:
        name (str): A ``step[...]`` annotation name.

    Returns:
        str: The leading token inside the brackets, or ``""`` when the name
            carries none.
    """
    return name[len("step[") :].split(" ", 1)[0].rstrip("]")


def _phase_step_medians(
    spans: list[tuple[float, float, str]],
    min_phase_steps: int = _PRETRIM_MIN_PHASE_STEPS,
) -> dict[str, float]:
    """Median step duration per phase, for phases that recur often enough.

    A single median over the whole trace is not a usable baseline for a mixed
    capture: decode dominates the population at tens of milliseconds while a
    prefill step runs for the better part of a second, so every prefill step
    reads as an outlier against it and the leading ones -- exactly the window
    ``--steady-state-mode=prefilldecode`` exists to analyse -- would be trimmed
    as start-up noise. Grouping by phase keeps each step measured against work
    of its own kind.

    Args:
        spans (list[tuple[float, float, str]]): ``(ts, dur, name)`` per step.
        min_phase_steps (int): Samples a phase needs before its median is used.

    Returns:
        dict[str, float]: Phase to median step duration in microseconds. Phases
            below the sample floor, and phases whose median is not positive, are
            omitted -- callers treat a missing phase as "no baseline".
    """
    by_phase: dict[str, list[float]] = {}
    for _, dur, name in spans:
        by_phase.setdefault(_step_phase(name), []).append(dur)
    medians: dict[str, float] = {}
    for phase, durs in by_phase.items():
        if len(durs) < min_phase_steps:
            continue
        durs.sort()
        median = durs[len(durs) // 2]
        if median > 0:
            medians[phase] = median
    return medians


def pretrim_startup_transient(
    src: Path,
    dst: Path,
    *,
    outlier_factor: float = _PRETRIM_OUTLIER_FACTOR,
    min_steps: int = _PRETRIM_MIN_STEPS,
    min_phase_steps: int = _PRETRIM_MIN_PHASE_STEPS,
    max_dropped_steps: int = _PRETRIM_MAX_DROPPED_STEPS,
) -> tuple[bool, dict[str, Any]]:
    """Drop the profiler-start transient from the head of a torch trace.

    ``torch.profiler.start()`` is a local, unfenced call whose cost varies by
    seconds across ranks, and SGLang does not barrier after it. The ranks whose
    profiler comes up first therefore reach the first collective of the step and
    spin there until the slowest peer arrives. A spin-waiting collective is
    charged as GPU-busy time, so that wait lands in the trace as one enormous
    kernel at the head of the window: on the reference capture a single
    ``cross_device_reduce_2stage`` held 15747 ms of a 16805 ms window, leaving
    Compute% at 5.75% and tripping the low-compute gate that suppresses the
    entire hot-kernel candidate list.

    The collective is self-healing -- releasing it re-synchronises every rank --
    so the contamination is confined to the first step and everything after it is
    already steady (32.94 ms +/- 0.3 across the following 127 steps on the same
    capture). Cutting the leading outlier steps therefore recovers an honest
    window without touching instrumentation, and keeps ``shape_discovery`` and
    the ``with_stack`` frames its shapes ride on fully intact.

    Detection is on step duration, not on any kernel name, so the same guard
    catches other head-of-window transients (first-replay JIT, KV-pool growth)
    without knowing what they are. Durations are compared against the median of
    the step's *own phase*: a mixed capture's prefill steps run one to two orders
    of magnitude longer than its decode steps, and against a whole-trace median
    the leading prefill steps -- the window
    ``--steady-state-mode=prefilldecode`` exists to analyse -- would all read as
    start-up noise. Only a *leading* run of outliers is cut: a slow step in the
    middle is real behaviour the splitter and the health gates should still see.

    The cut is applied per timeline rather than as one timestamp, because the
    host runs a step ahead of the device and the two spans interleave across the
    boundary. See the comment at the cut for what each single-point alternative
    loses. The two timelines are paired by position, never by annotation name;
    ``_host_step_starts`` has the reasoning.

    Step count is unaffected downstream: the splitter is invoked with
    ``--num-steps`` and caps its window there, so it still hands TraceLens
    exactly that many steps -- only which ones changes. The caller refuses the
    trim outright when fewer than ``--num-steps`` would remain.

    Args:
        src (Path): Raw per-rank trace, JSON or ``.gz``.
        dst (Path): Where the trimmed trace is written. Untouched when no trim
            is performed.
        outlier_factor (float): Multiple of the phase median above which a
            leading step is treated as a transient.
        min_steps (int): Fewest step annotations required before trimming.
        min_phase_steps (int): Samples a phase needs before its median is
            trusted as a baseline.
        max_dropped_steps (int): Refuse to trim when more than this many leading
            steps look like transients.

    Returns:
        tuple[bool, dict[str, Any]]: ``(trimmed, report)``. ``report`` always
            carries a ``reason``; when ``trimmed`` is True it also carries the
            step counts and durations for the run log and the pretrim artifact.
    """
    factor = outlier_factor
    try:
        payload = open_json(src)
    except Exception as exc:  # noqa: BLE001 - unreadable trace is the splitter's problem, not ours
        return False, {"reason": "unreadable_trace", "error": str(exc)[:200]}
    if not isinstance(payload, dict):
        return False, {"reason": "unexpected_payload"}
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        return False, {"reason": "no_trace_events"}

    spans = _step_annotation_spans(events)
    if len(spans) < min_steps:
        return False, {"reason": "too_few_steps", "steps": len(spans), "min_steps": min_steps}

    medians = _phase_step_medians(spans, min_phase_steps)
    if not medians:
        # No phase recurs often enough to be a baseline; there is nothing to
        # call a step abnormal against.
        return False, {
            "reason": "no_phase_baseline",
            "steps": len(spans),
            "min_phase_steps": min_phase_steps,
        }
    phase_medians_ms = {phase: round(med / 1000.0, 3) for phase, med in sorted(medians.items())}

    dropped = 0
    ratios: list[float] = []
    while dropped < len(spans):
        _, dur, name = spans[dropped]
        # A step whose phase has no baseline stops the run rather than being
        # dropped on a comparison that does not hold.
        median_us = medians.get(_step_phase(name))
        if median_us is None or dur <= factor * median_us:
            break
        ratios.append(dur / median_us)
        dropped += 1
    if dropped == 0:
        first_median_us = medians.get(_step_phase(spans[0][2]))
        return False, {
            "reason": "no_leading_outlier",
            "steps": len(spans),
            "phase_medians_ms": phase_medians_ms,
            "first_step_phase": _step_phase(spans[0][2]),
            "first_step_ratio": (round(spans[0][1] / first_median_us, 2) if first_median_us else None),
        }
    if dropped > max_dropped_steps:
        # Not a start-up blip. Leave it visible so the health gates can act.
        return False, {
            "reason": "too_many_leading_outliers",
            "steps": len(spans),
            "leading_outliers": dropped,
            "max_dropped_steps": max_dropped_steps,
            "outlier_factor": factor,
            "phase_medians_ms": phase_medians_ms,
        }

    remaining = len(spans) - dropped
    # One cut per timeline, not one for the trace. The host runs a step ahead of
    # the device, so the first kept step's launches are issued *while the dropped
    # step is still executing on the GPU* -- on the reference capture the kept
    # step's host span starts 839 ms inside the dropped step's 15.78 s device
    # span. A single timestamp therefore cannot both drop the transient whole and
    # keep the first surviving step whole: cutting on the device boundary strips
    # that step's host ops, taking the `kernel_shape_profiler` frames its shape
    # attribution rides on with them, and cutting on the host boundary leaves the
    # dropped step's post-barrier kernels behind as orphans.
    gpu_cut = spans[dropped][0]
    host_starts = _host_step_starts(events)
    if len(host_starts) != len(spans):
        # Exact equality, not "enough host entries": a capture that carries more
        # host annotations than device ones is missing them from the middle, not
        # the tail (see `_host_step_starts`), so the Nth host start belongs to
        # some earlier step and the host cut silently lands too early. Pairing by
        # name is not an alternative either. Refuse rather than approximate --
        # the caller keeps the untrimmed trace, which is the state this whole
        # step was added to improve on but never worse than it.
        return False, {
            "reason": "timeline_step_count_mismatch",
            "steps": len(spans),
            "host_steps": len(host_starts),
        }
    cpu_cut = host_starts[dropped]
    if cpu_cut > gpu_cut:
        # The host issues a step's launches before the device runs it, so a host
        # start later than the paired device start means the two lists are not
        # aligned and the cut points cannot be trusted.
        return False, {
            "reason": "timeline_pairing_unreliable",
            "steps": len(spans),
            "host_steps": len(host_starts),
            "gpu_cut_ts": gpu_cut,
            "cpu_cut_ts": cpu_cut,
        }

    def _survives(ev: Any) -> bool:
        if not isinstance(ev, dict):
            return True
        # ph:"M" is metadata (process_name / process_labels / thread_name /
        # sort indices), not timeline work. torch stamps it with the
        # profiler-open ts, which is always before either cut, so a plain ts
        # filter would strip every one of them and leave the chunk with
        # unnamed processes and threads for TraceLens to attribute against.
        if ev.get("ph") == "M":
            return True
        ts = ev.get("ts")
        if not isinstance(ts, (int, float)):
            return True
        # Trace-level spans and markers (cat:"Trace", the ph:"i" iteration-start
        # instant) are deliberately *not* given the ph:"M" treatment: unlike
        # metadata their ts/dur describe the untrimmed window, so carrying them
        # over unchanged would have the artifact state a start and a duration
        # the events no longer support. They fall through the ts filter below.
        ph = ev.get("ph")
        if ph == "s":  # flow start: sits on the host timeline
            return ts >= cpu_cut
        if ph == "f":  # flow finish: sits on the device timeline
            return ts >= gpu_cut
        return ts >= (gpu_cut if ev.get("cat") in _GPU_TIMELINE_CATS else cpu_cut)

    kept = [ev for ev in events if _survives(ev)]

    dropped_phase = _step_phase(spans[0][2])
    report = {
        "reason": "trimmed",
        "dropped_steps": dropped,
        "dropped_ms": round(sum(dur for _, dur, _ in spans[:dropped]) / 1000.0, 3),
        "dropped_phase": dropped_phase,
        "median_step_ms": phase_medians_ms[dropped_phase],
        "phase_medians_ms": phase_medians_ms,
        # The worst of the dropped steps, not the first: with more than one
        # dropped the first is not necessarily the transient.
        "outlier_ratio": round(max(ratios), 2),
        "outlier_factor": factor,
        "remaining_steps": remaining,
        "host_steps": len(host_starts),
        "events_before": len(events),
        "events_after": len(kept),
        "gpu_cut_ts": gpu_cut,
        "cpu_cut_ts": cpu_cut,
        "source": str(src),
        "output": str(dst),
    }

    payload["traceEvents"] = kept
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if dst.suffix == ".gz" else open
        with opener(dst, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception as exc:  # noqa: BLE001 - fall back to the raw trace rather than fail the run
        # Drop the partial write. The caller keeps the untrimmed trace, and a
        # truncated file left in the split directory is only something for a
        # later step to mistake for a usable one. ``reason`` goes last so the
        # failure is what the caller reads, not the "trimmed" this report was
        # built with.
        with contextlib.suppress(OSError):
            dst.unlink(missing_ok=True)
        return False, {**report, "reason": "write_failed", "error": str(exc)[:200]}
    return True, report


#: Directory name the splitter writes its per-phase output into. Everything
#: below it is derived from a raw capture, never a capture itself.
_SPLIT_DIR_NAME = "trace_split"

#: How many discovered files the CPU-only preflight will open before giving up.
#: A cost ceiling, not the thing that makes the preflight land on the capture --
#: size ordering does that, and it holds whatever the fragments are called or
#: where they sit. Reaching this limit means every large candidate was empty,
#: which is a real capture problem rather than a selection one.
_KERNEL_PROBE_LIMIT = 8

#: Cumulative bytes of candidate traces the preflight will deserialise before
#: giving up. Only the failing path spends this: with size ordering a healthy
#: capture answers on the first probe. It exists because production rank traces
#: reach hundreds of megabytes, and eight of those would turn a failure that
#: used to take a second into one that takes minutes or exhausts memory.
_KERNEL_PROBE_BYTE_BUDGET = 512 * 1024 * 1024

#: Per-phase fragment names the splitter emits. Matched as well as the directory
#: because a flat layout would otherwise leave them in the default bucket, and a
#: capture with eight ranks would then spend the whole probe budget on fragments
#: before reaching a rank file.
_PHASE_FRAGMENT_RE = re.compile(
    r"^(?:decode_only|mixed|prefill_only|prefilldecode)\w*_steady_state\w*"
    r"_rank_\d+\.trace\.json(?:\.gz)?$"
)


def _is_capture_fragment(path: Path, root: Path | None = None) -> bool:
    """Whether a trace path is a CUDA-graph capture sidecar.

    Thin alias over :func:`_capture_shapes.is_capture_fragment`, which both
    trace-analysis routes share so the answer cannot drift between them. Both
    callers here need that same answer: discovery demotes sidecars below a real
    capture, and the preflight refuses an input made of nothing else.

    Args:
        path: The trace file path to classify.
        root: Directory the classification is relative to, when known.

    Returns:
        True when ``path`` is a graph-capture sidecar rather than a workload
        trace.
    """
    return _shared_is_capture_fragment(path, root)


def _capture_classification_root(trace_input: Path) -> Path:
    """Root to classify candidates discovered from ``trace_input`` against.

    A directory input is its own root, so a ``capture_traces/`` component below
    it counts. A file input resolves to its parent, which leaves the filename as
    the only component and keeps an unrelated ancestor directory out of the
    decision.

    Args:
        trace_input: The ``--trace-input`` path.

    Returns:
        The directory candidate paths are made relative to.
    """
    return trace_input if trace_input.is_dir() else trace_input.parent


def _is_derived_trace(path: Path, root: Path | None = None) -> bool:
    """Whether a trace path is splitter output rather than a raw capture.

    Three shapes, all derived: anything under the splitter's own output
    directory, the per-iteration annotation sidecars it writes beside a capture,
    and the per-phase fragment names themselves. They are a few hundred bytes
    each and cover one phase of one iteration, so an analysis pointed at them
    describes a sliver of the run.

    The name test is not redundant with the directory test. Production nests
    these under ``trace_split/`` today -- 276 of 276 capture directories with
    fragments -- but the demotion should not depend on a layout that the
    splitter is free to change.

    ``root`` bounds the directory test to the capture being analysed. Paths
    arrive absolute, so testing every component would demote *every* candidate
    whenever an ancestor happened to be named ``trace_split`` -- pointing
    ``--trace-input`` inside a previous split, say. With all candidates in the
    same bucket the ordering collapses back to alphabetical and the original bug
    returns, which is a lot of damage for a coincidence of naming.
    """
    relative = path
    if root is not None:
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
    if _SPLIT_DIR_NAME in relative.parts:
        return True
    if _PHASE_FRAGMENT_RE.match(path.name):
        return True
    return bool(re.search(r"trace_annotation_iteration_\d+", path.name))


def _trace_file_size(path: Path) -> int:
    """Size in bytes, or 0 when it cannot be read."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _count_kernels_if_readable(path: Path) -> tuple[bool, int]:
    """``(readable, kernel_count)`` for a candidate trace.

    :func:`count_gpu_kernel_events` answers 0 both for a trace with no GPU work
    and for one it could not parse. Those need different readers: the first is a
    profiler problem, the second is a truncated or corrupt file. Collapsing them
    is the same misdirection this module already sent people on once.

    Still counts through :func:`count_gpu_kernel_events`, so it remains the one
    place kernel events are recognised. The disambiguating parse runs only on a
    zero answer, which is the only answer that is ambiguous -- a trace with
    kernels in it demonstrably parsed.
    """
    count = count_gpu_kernel_events(path)
    if count:
        return True, count
    try:
        payload = open_json(path)
    except Exception:  # noqa: BLE001 - unreadable is a distinct answer, not a crash
        return False, 0
    if not isinstance(payload, dict) or not isinstance(payload.get("traceEvents"), list):
        return False, 0
    return True, 0


def _trace_input_sort_key(path: Path, root: Path | None = None) -> tuple[int, int, str]:
    """Compute the discovery sort key for a trace input path.

    Prefers the merged annotated trace over rank/phase shards (the TraceLens
    splitter needs the large trace).

    Splitter output sorts last, and within a bucket the largest file leads.
    Both parts exist because of the same bug: the fragments used to share the
    default bucket with the raw capture, so alphabetical order decided, and
    ``decode_only_steady_state_...`` beats ``rank_0.trace.json.gz`` on the first
    letter. Every xDiT roofline attempt therefore analysed a 938-byte phase
    fragment instead of the 910 KB capture beside it: runs whose fragment held
    no GPU kernels failed the CPU-only preflight outright, and the one model
    whose fragment happened to hold 512 produced a roofline computed from 2.6%
    of its own trace, with no ceiling.

    Size is the part that does not depend on recognising a name. A real capture
    is orders of magnitude larger than a per-phase fragment or a sidecar like
    ``execution_details.json``, so ordering by descending size puts the right
    file first even for a fragment shape nobody has seen yet.

    Args:
        path: The trace file path to rank.
        root: The capture directory being analysed, so the ``trace_split``
            component is looked for below it rather than anywhere in an
            absolute path.

    Returns:
        A ``(priority, -size, name)`` sort key (lower sorts first).
    """
    name = path.name
    size = _trace_file_size(path)
    if _is_derived_trace(path, root):
        return (4, -size, name)
    if name.startswith("merged-"):
        return (0, -size, name)
    if re.search(r"TP-\d+-DECODE\.trace\.json(?:\.gz)?$", name):
        return (2, -size, name)
    if _is_capture_fragment(path, root):
        return (3, -size, name)
    return (1, -size, name)


def discover_trace_inputs(trace_input: Path) -> tuple[str, list[Path]]:
    """Resolve a trace input path into a list of trace files.

    Accepts either a single trace file or a capture directory; directories
    are searched recursively for known trace extensions, deduplicated, and
    ordered via :func:`_trace_input_sort_key`.

    Args:
        trace_input (Path): A trace file or a capture directory.

    Returns:
        tuple[str, list[Path]]: ``("file", [path])`` for a single file or
            ``("capture_dir", paths)`` for a directory.

    Raises:
        FileNotFoundError: When the path does not exist or no trace files are
            found under a supplied directory.
    """
    if trace_input.is_file():
        return "file", [trace_input]
    if not trace_input.is_dir():
        raise FileNotFoundError(f"trace_input does not exist: {trace_input}")

    traces: list[Path] = []
    for pattern in ("*.json", "*.json.gz", "*.trace", "*.trace.json", "*.trace.json.gz"):
        traces.extend(sorted(trace_input.rglob(pattern)))
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique = []
    for trace in traces:
        if trace not in seen:
            seen.add(trace)
            unique.append(trace)
    unique.sort(key=lambda p: _trace_input_sort_key(p, trace_input))
    if not unique:
        raise FileNotFoundError(f"no trace files found under capture directory: {trace_input}")
    return "capture_dir", unique


def is_kernel_event(event: dict[str, Any]) -> bool:
    """Apply a strict GPU-kernel filter to a trace event.

    Only ``cat == 'kernel'`` events qualify, excluding host-side sync/launch
    wrappers that would eclipse real kernels.

    Args:
        event: A single trace event dict.

    Returns:
        ``True`` if the event is a real GPU kernel.
    """
    cat = str(event.get("cat") or event.get("category") or "").lower()
    if cat != "kernel":
        return False
    name = str(event.get("name") or event.get("kernel_name") or "")
    if name.lower() in RUNTIME_API_NAMES:
        return False
    return True


def extract_shape(event: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a shape annotation from a trace event when present.

    Checks both ``event['args']`` and the event top level for the first of
    several known shape keys.

    Args:
        event (dict[str, Any]): A single trace event.

    Returns:
        dict[str, Any] | None: A single-key dict ``{shape_key: value}`` for
            the first shape field found, or ``None`` when none is present.
    """
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    for key in ("shape", "shapes", "input_shape", "trace_shapes"):
        if key in args:
            return {key: args[key]}
        if key in event:
            return {key: event[key]}
    return None


def extract_source_file(event: dict[str, Any]) -> str:
    """Extract a source-file path from a trace event when present.

    Checks both ``event['args']`` and the event top level for the first of
    several known path keys.

    Args:
        event (dict[str, Any]): A single trace event.

    Returns:
        str: The first non-empty source path found, or ``""`` when none.
    """
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    for key in ("source_file", "file", "filename", "path"):
        value = args.get(key) or event.get(key)
        if value:
            return str(value)
    return ""


_FLYDSL_SOURCE_MARKERS = (
    "import flydsl",
    "from flydsl",
    "@flyc.kernel",
    "@flyc.jit",
    "flydsl.compiler",
    "flydsl.expr",
)
_FLYDSL_SCAN_BYTES = 4096


def _looks_like_flydsl_source(source_file: str) -> bool:
    """Detect whether a source file is a FlyDSL kernel source.

    Content-sniffs the first 4 KiB for FlyDSL markers.

    Args:
        source_file: Path to the candidate ``.py`` source.

    Returns:
        ``True`` when FlyDSL markers are found in the file head.
    """
    if not source_file or not source_file.endswith(".py"):
        return False
    try:
        with open(source_file, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(_FLYDSL_SCAN_BYTES)
    except OSError:
        return False
    return any(marker in head for marker in _FLYDSL_SOURCE_MARKERS)


def _defines_traced_triton_kernel(name: str, source_file: str) -> bool:
    """Whether ``source_file`` defines the Triton kernel ``name`` was traced from.

    A device symbol carries no language and frameworks import Triton through a
    shim, so both signals the name-based check reads are absent. Resolve the
    symbol to a ``@triton.jit`` def instead.

    Args:
        name: Kernel/symbol name as the trace reports it.
        source_file: Resolved source path.

    Returns:
        ``True`` when a ``@triton.jit`` def in ``source_file`` matches ``name``.
    """
    if _triton_def_line is None or not source_file or not source_file.endswith(".py"):
        return False
    for keyword in _candidate_keywords(name):
        if _triton_def_line(source_file, symbol=keyword, require_name_match=True) is not None:
            return True
    return False


_FLYDSL_PSEUDO_OP_NAME_MARKERS = (
    "pseudo_op::moe_flydsl_",
    "pseudo_op::flydsl_",
)


def source_type_for(name: str, source_file: str) -> str:
    """Classify a kernel's source type from its name and source path.

    Recognizes FlyDSL pseudo-ops, runtime-generated kernels, HIP/C++,
    Triton, FlyDSL, plain Python, and vendor-binary backends.

    Args:
        name (str): Kernel symbol/name.
        source_file (str): Resolved source-file path (may be empty).

    Returns:
        str: One of ``flydsl``, ``runtime_generated``, ``hip_cpp``,
            ``triton``, ``python``, ``vendor_binary``, or ``unknown``.
    """
    lower_name = name.lower()
    # Synthetic flydsl pseudo-ops carry no source_file; match the name prefix.
    if any(marker in lower_name for marker in _FLYDSL_PSEUDO_OP_NAME_MARKERS):
        return "flydsl"
    if is_runtime_generated_kernel(name, source_file):
        return "runtime_generated"
    if source_file.endswith((".cu", ".cuh", ".hip", ".cpp", ".h", ".hpp")):
        return "hip_cpp"
    if "triton" in lower_name and source_file.endswith(".py"):
        return "triton"
    # Ahead of the Triton proof: a FlyDSL kernel may import Triton for its own
    # reference path, and the FlyDSL identity is the one its consumers act on.
    if _looks_like_flydsl_source(source_file):
        return "flydsl"
    if _defines_traced_triton_kernel(name, source_file):
        return "triton"
    if source_file.endswith(".py"):
        return "python"
    if "hipblas" in lower_name or "rocblas" in lower_name:
        return "vendor_binary"
    return "unknown"


_RUNTIME_GENERATED_SOURCE_MARKERS = (  # nosec B108 - marker strings, not filesystem writes.
    "/tmp/torchinductor",
    "/torchinductor_",
    "/.cache/torch/inductor",
    "/.triton/cache",
    "/triton/cache",
)
_COMPILE_GENERATED_NAME_MARKERS = (
    "triton_poi_",
    "triton_red_",
    "triton_tem_",
    "torchinductor",
    "inductor",
)


@functools.lru_cache(maxsize=1)
def _framework_patch_roots() -> tuple[str, ...]:
    """Resolve framework install roots for patch-target matching.

    Roots come from ``framework_paths.resolve_patch_target_roots``; a
    lower-case variant of each (e.g. ``/app/ATOM/atom/`` ->
    ``/app/atom/atom/``) is also emitted for case-insensitive matching.

    Returns:
        The framework install roots, including lower-case variants.
    """
    try:
        if _resolve_patch_target_roots is None:
            raise ImportError
        roots = _resolve_patch_target_roots()
    except ImportError:
        if _known_target_roots is None:
            roots = []
        else:
            roots = _known_target_roots()
    out: list[str] = []
    seen: set[str] = set()
    for root in roots:
        for variant in (root, root.lower()):
            if variant and variant not in seen:
                seen.add(variant)
                out.append(variant)
    return tuple(out)


@functools.lru_cache(maxsize=1)
def _aiter_csrc_root() -> str:
    """Resolve aiter's own device-source root from the installed package.

    Cached once per process.

    Returns:
        The aiter csrc root (e.g. ``.../aiter_meta/csrc/``), or ``""`` when
        aiter is not importable.
    """
    if _aiter_jit_core is None:
        return ""
    raw = (getattr(_aiter_jit_core, "AITER_CSRC_DIR", "") or "").replace(os.sep, "/")
    return (raw.rstrip("/") + "/") if raw else ""


def _flydsl_reusable_roots() -> tuple[str, ...]:
    """Resolve FlyDSL checkout root(s) for moe_flydsl pseudo-ops.

    ``$DSL2_ROOT`` / ``$FLYDSL_ROOT`` take precedence over the WekaFS default.
    Uncached, so an env override applies without a process restart.

    Returns:
        The lower-cased FlyDSL checkout roots.
    """
    if _resolve_flydsl_source_roots is not None:
        return tuple(dict.fromkeys(r.lower() for r in _resolve_flydsl_source_roots()))
    out: list[str] = []
    for env_key in ("DSL2_ROOT", "FLYDSL_ROOT"):
        val = (os.environ.get(env_key, "") or "").strip()
        if val:
            out.append((val.rstrip("/") + "/").lower())
    for default in ("/opt/flydsl/", "/sgl-workspace/flydsl/"):
        if default not in out:
            out.append(default)
    return tuple(out)


def _reusable_roots() -> tuple[str, ...]:
    """Combine all reusable-source roots used by the patchability gate.

    Returns:
        tuple[str, ...]: Discovered framework roots plus the aiter csrc root
            and FlyDSL checkout roots, deduplicated.
    """
    roots = _framework_patch_roots()
    csrc = _aiter_csrc_root()
    if csrc and csrc not in roots:
        roots = roots + (csrc,)
    for fly in _flydsl_reusable_roots():
        if fly not in roots:
            roots = roots + (fly,)
    return roots


# Kernel-name substrings marking an op non-patchable: vendor BLAS, collectives, copies.
_NON_PATCHABLE_NAME_MARKERS: tuple[str, ...] = (
    "rocblas",
    "hipblas",
    "hipblaslt",
    "rocblaslt",
    "tensile",
    "miopen",
    "ck_kernels",
    "nccl",
    "rccl",
    "hipmemcpy",
    "__amd_rocclr_copybuffer",
    "aten::copy",
)

# Vendor BLAS / closed-source compute backends: a candidate whose runtime
# implementation is one of these has no rewritable source (the device body
# lives in a precompiled vendor binary). Matched against the ``library`` field.
_VENDOR_BACKEND_LIBRARIES: frozenset[str] = frozenset(
    {
        "tensile",
        "hipblas",
        "hipblaslt",
        "rocblas",
        "rocblaslt",
        "ck",
        "composable_kernel",
        "ck_kernels",
        "miopen",
    }
)

# torch dispatch interception shims: these forward a tensor op to a vendor
# backend and hold no rewritable device kernel, so a symbol resolving here is a
# dispatch stub. Matched as a POSIX path suffix against the resolved source_file.
_TORCH_DISPATCH_SHIM_SOURCES: tuple[str, ...] = ("vllm/model_executor/parameter.py",)


def is_torch_dispatch_shim_source(source_file: str) -> bool:
    """True when ``source_file`` is a known torch-dispatch interception shim.

    These ``__torch_function__`` / ``__torch_dispatch__`` files forward a
    tensor op to a vendor backend and hold no rewritable device kernel, so a
    symbol attributed here must be treated as a non-reusable dispatch stub
    (same handling as ``@compile_ops`` JIT stubs in ``aiter/ops/moe_op.py``).

    Args:
        source_file: The resolved source-file path.

    Returns:
        ``True`` when the file is a known torch-dispatch interception shim.
    """
    posix = str(source_file or "").replace("\\", "/")
    return any(posix.endswith(suffix) for suffix in _TORCH_DISPATCH_SHIM_SOURCES)


def is_runtime_generated_kernel(name: str, source_file: str) -> bool:
    """Detect torch.compile / Inductor / cache-generated kernels.

    These are not portable across serving runs.

    Args:
        name: The kernel symbol/name.
        source_file: The resolved source-file path.

    Returns:
        ``True`` for a runtime-generated kernel that is not a stable in-repo
        Triton source.
    """
    lower_name = (name or "").lower()
    lower_file = (source_file or "").lower()
    if any(marker in lower_file for marker in _RUNTIME_GENERATED_SOURCE_MARKERS):
        return True
    if any(marker in lower_name for marker in _COMPILE_GENERATED_NAME_MARKERS):
        # A stable in-repo Triton source can still be reusable.
        return not any(root in lower_file for root in _reusable_roots())
    return False


def classify_patchability(candidate: dict[str, Any]) -> tuple[bool, str]:
    """Return ``(reusable, skip_reason)`` for a hot-kernel candidate.

    Single source of truth for the kernel-opt routing gate. Also rejects
    vendor/collective/native-op name markers
    (:data:`_NON_PATCHABLE_NAME_MARKERS`) and library-less ``aten::*`` ops.

    Args:
        candidate: The hot-kernel candidate dict.

    Returns:
        A ``(reusable, skip_reason)`` tuple; ``skip_reason`` is empty when
        reusable, else a short audit explanation.
    """
    source_file = str(candidate.get("source_file") or "")
    name = str(candidate.get("name") or "")
    lower_name = name.lower()
    # Verdict-first: honor the active finder's symbol-based verdict (carried on
    # the op_to_source_* audit fields) when present. A routable entry is ground
    # truth and bypasses the heuristics; a non-rewritable verdict reports its
    # reason; anything else falls through to the heuristics below.
    if candidate.get("source_resolution_method") in _CURATED_LIKE_METHODS:
        patchable = candidate.get("op_to_source_patchable")
        if patchable is True and source_file:
            # aiter_asm compute-cores are hand-written assembly: the resolved .cu
            # is only a dispatcher for a prebuilt .co, so the compute core is not
            # editable from source. Skip with a clear reason.
            if str(candidate.get("kernel_kind") or "").strip().lower() == "aiter_asm":
                return False, (
                    "source: aiter_asm prebuilt assembly compute-core "
                    "(.co loaded by the .cu dispatcher; no editable .s source) "
                    "-- not rewritable, no deterministic tuner available"
                )
            return True, ""
        if patchable is False:
            reason = (
                str(candidate.get("op_to_source_reason") or "").strip()
                or str(candidate.get("op_to_source_status") or "").strip()
                or "non-rewritable"
            )
            return False, f"source: {reason}"
    # Vendor-operator-playbook check: some closed-source vendor operators (a
    # pip-installed compiled library with no rewritable device source) still
    # have a small, named set of launch-config knobs that a KernelForge
    # forge-loop task bundle has already been validated to tune (mori's EP
    # dispatch/combine is the first case -- see KernelForge PR #88). This is
    # checked before the vendor_binary/no-source-file rejections below and
    # before the ``source_file`` requirement, since a vendor-playbook
    # candidate's "source" is a pip-installed file, not something forge would
    # ever rewrite.
    if match_vendor_operator_playbook(candidate) is not None:
        return True, ""
    if not source_file:
        # A bare launch API has no kernel body to rewrite, which is a different
        # situation from a kernel whose source we merely failed to locate. Say
        # so rather than sending a reader looking for a file that cannot exist.
        if is_runtime_api_name(str(candidate.get("name") or "")):
            return False, "launch API, not a kernel (no rewritable body)"
        return False, "source file not resolved"
    if candidate.get("source_type") == "vendor_binary":
        return False, "vendor binary (no rewritable source)"
    if candidate.get("vendor_dispatch_wrapper"):
        return False, f"vendor dispatch wrapper at {source_file}"
    if is_torch_dispatch_shim_source(source_file):
        return False, (f"torch dispatch shim (no rewritable kernel body): {source_file}")
    for marker in _NON_PATCHABLE_NAME_MARKERS:
        if marker in lower_name:
            return False, (f"non-patchable kernel name marker '{marker}' in {name!r}")
    library = str(candidate.get("library") or "").strip().lower()
    if library in _VENDOR_BACKEND_LIBRARIES:
        return False, (
            f"vendor backend library {candidate.get('library')!r} (precompiled binary, no rewritable source)"
        )
    if name.startswith("aten::"):
        if not library or library in {"tensile", "pytorch native"}:
            return False, (
                f"PyTorch native op {name!r} backed by "
                f"{candidate.get('library') or 'unknown'} library "
                "(typically Tensile / vendor backend)"
            )
    if is_runtime_generated_kernel(name, source_file):
        return False, (f"runtime-generated (torch.compile / Inductor cache): {source_file}")
    lower_file = source_file.lower()
    if not any(root in lower_file for root in _reusable_roots()):
        return False, (f"source not under a reusable framework root: {source_file}")
    # cpp_itfs host-launcher guard: a .py under csrc/cpp_itfs/ is a host driver
    # whose real GPU code lives in sibling .cuh/.cpp.jinja; skip it so the
    # candidate doesn't burn a forge/geak attempt.
    if "/csrc/cpp_itfs/" in lower_file and lower_file.endswith(".py"):
        return False, (
            f"cpp_itfs host launcher (device code is in sibling .cuh/.cpp.jinja, not this .py): {source_file}"
        )
    source_type = candidate.get("source_type")
    # aiter device-source promotion: a real .cu/.cuh/.hip kernel under /aiter/ is
    # patchable even when the classifier left source_type unknown, since aiter
    # JIT-compiles each op and forge edits it in place.
    if (
        source_type not in {"hip_cpp", "triton", "python", "flydsl"}
        and "/aiter/" in lower_file
        and lower_file.endswith((".cu", ".cuh", ".hip"))
    ):
        return True, ""
    if source_type not in {"hip_cpp", "triton", "python", "flydsl"}:
        return False, (f"source_type={source_type!r} not in {{hip_cpp, triton, python, flydsl}}")
    return True, ""


# Wrapper TUs that just dispatch to a precompiled .so/.co, detected by small
# file size + content signature (conservative so real small kernels survive).
_VENDOR_DISPATCH_SIGS = (
    "ctypes.CDLL",  # pure-Python wrapper around .so
    "torch.ops.aiter.",  # registered aten op forwarding
    "_C_aiter.",  # bound C extension forwarding
    "module_name = ",  # aiter jit module loaders
    "AITER_JIT_LOAD",  # aiter macro
    "hipModuleLoad",  # raw .co loader
    "AiterAsmKernel",  # ASM dispatch wrapper
)
_VENDOR_KEYWORD_NAMES = (
    "hipblaslt",
    "rocblaslt",
    "miopen",
    "ck_kernels",
)


def is_vendor_dispatch_wrapper(name: str, source_file: str) -> bool:
    """Detect a thin dispatch wrapper around a precompiled vendor binary.

    Uses a small file size plus content signatures so that real small kernels
    survive (conservative).

    Args:
        name: The kernel symbol/name.
        source_file: The resolved source-file path.

    Returns:
        ``True`` when the source is a dispatch wrapper around a ``.so``/``.co``
        with nothing to rewrite.
    """
    nm = (name or "").lower()
    if any(kw in nm for kw in _VENDOR_KEYWORD_NAMES):
        return True
    if not source_file:
        return False
    p = Path(source_file)
    try:
        if not p.is_file():
            return False
        # >16 KB is presumed a real device kernel; wrappers/shims sit well below.
        if p.stat().st_size > 16 * 1024:
            return False
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    return any(sig in text for sig in _VENDOR_DISPATCH_SIGS)


#: Packages whose trees hold rewritable kernel source. Located at runtime so a
#: wheel install, an editable checkout and a serving image all resolve, rather
#: than only the one layout a literal happens to name.
#:
#: Taken from the orchestrator's resolver whenever that package is importable,
#: which is every path except standalone CLI use. A second literal here is what
#: let the two disagree: this tool listed ``sgl_kernel`` while the resolver it
#: defers to did not, so on a host with a standalone ``sgl_kernel`` wheel the
#: package was named in the "looked for" message and never actually searched.
#: The literal below is the standalone default only, and
#: ``test_kernel_search_roots`` fails if it drifts from the authoritative tuple.
_STANDALONE_KERNEL_SOURCE_PACKAGES: tuple[str, ...] = (
    "aiter",
    "aiter_meta",
    "sglang",
    "sgl_kernel",
    "vllm",
    "atom",
    "xfuser",
)

_KERNEL_SOURCE_PACKAGES: tuple[str, ...] = (
    _FRAMEWORK_SOURCE_PACKAGES if _FRAMEWORK_SOURCE_PACKAGES is not None else _STANDALONE_KERNEL_SOURCE_PACKAGES
)

#: Last-resort checkout layouts for a host where nothing above is importable.
#: Kept small on purpose: a pinned path cannot follow a package across
#: container images or Python versions, and a list of them going stale in
#: silence is what emptied this tier and stalled kernel-opt entirely.
_FALLBACK_SEARCH_ROOTS: tuple[str, ...] = (
    "/sgl-workspace/aiter",
    "/sgl-workspace/sglang/sgl-kernel",
    "/sgl-workspace/sglang/python/sglang",
    "/sgl-workspace/vllm",
)


def _installed_package_dir(package: str) -> str:
    """Locate a package's directory without importing it.

    Args:
        package (str): Importable package name.

    Returns:
        str: The package directory, or ``""`` when it is not on this
            interpreter's path.
    """
    if not package or not package.isidentifier():
        return ""
    try:
        spec = importlib.util.find_spec(package)
    except (AttributeError, ImportError, ValueError):
        return ""
    if spec is None:
        return ""
    for location in list(getattr(spec, "submodule_search_locations", None) or []):
        candidate = str(location).rstrip("/")
        if candidate:
            return candidate
    origin = str(getattr(spec, "origin", "") or "")
    return os.path.dirname(origin) if origin else ""


@lru_cache(maxsize=1)
def kernel_search_roots() -> tuple[str, ...]:
    """Resolve the framework trees to grep for kernel source, at runtime.

    Prefers the orchestrator's centralised resolver so this tool agrees with
    PolicyGate and patch application on where framework source lives. Falls back
    to locating each known package itself, then to the pinned checkout layouts,
    both when that package is not importable (standalone CLI use) and when it
    imported but resolved nothing -- an empty answer from the resolver used to
    end the search, which is the same silent outcome as having no roots at all.

    Non-existent roots are dropped: grepping them returns nothing and is
    indistinguishable from a kernel that genuinely has no source here.

    Cached for the duration of a run, since every candidate consults it and
    ``find_spec`` plus a directory probe per package is not free. A process that
    outlives one run -- the orchestrator imports this module -- calls
    :func:`refresh_kernel_search_roots` at run entry so a framework installed
    since the last one is still discoverable.

    Returns:
        tuple[str, ...]: Existing roots without trailing separators,
            de-duplicated in discovery order.
    """
    discovered: list[str] = []
    if _resolve_kernel_search_roots is not None:
        discovered.extend(_resolve_kernel_search_roots())
    # A chain, not an either/or. The centralised resolver is authoritative and
    # goes first, but "it imported" is not "it found something": when it returns
    # nothing, the alternative to probing here is a run that greps no directory
    # at all and reports every hot kernel as non-routable. Its answer is kept
    # whole when it has one, so this cannot widen the roots a normal host
    # searches -- it only decides between local discovery and nothing.
    if not discovered:
        discovered.extend(
            location
            for location in (_installed_package_dir(package) for package in _KERNEL_SOURCE_PACKAGES)
            if location
        )
        discovered.extend(_FALLBACK_SEARCH_ROOTS)
    roots: list[str] = []
    seen: set[str] = set()
    for root in discovered:
        normalized = str(root or "").rstrip("/")
        if not normalized or normalized in seen or not os.path.isdir(normalized):
            continue
        seen.add(normalized)
        roots.append(normalized)
    if not roots:
        log.warning(
            "no framework source root exists on this host (looked for %s); "
            "kernel source resolution will find nothing and every hot kernel "
            "will be reported as non-routable",
            ", ".join(_KERNEL_SOURCE_PACKAGES),
        )
    return tuple(roots)


def refresh_kernel_search_roots() -> tuple[str, ...]:
    """Re-run root discovery, dropping every answer derived from the old roots.

    :func:`_harness_search_bases` is cached for the same run and derived from
    these roots, so clearing one without the other reproduces the bug this
    function exists to fix, one step further on: the second analysis in a
    long-lived orchestrator would grep the freshly installed framework and still
    resolve its harnesses against the bases discovered when it was absent, so
    ``benchmark_files`` comes back empty and the invocation spec reaches the
    backend with no benchmark to run.

    Returns:
        tuple[str, ...]: The freshly discovered roots.
    """
    kernel_search_roots.cache_clear()
    _harness_search_bases.cache_clear()
    return kernel_search_roots()


# Extensions a grep hit may be admitted under. Deliberately narrow, and kept in
# lockstep with source_type_for(): a suffix admitted here but unclassified there
# lands as source_type="unknown", which classify_patchability rejects. Worse, it
# still competes in _rank_paths, where kind_score outweighs ext_score -- so a
# /csrc/ file with an unclassified suffix outranks the sibling .py and turns a
# routable candidate non-routable. Widen this only together with source_type_for.
SOURCE_EXTENSIONS = (".cuh", ".cu", ".hip", ".cpp", ".h", ".hpp", ".py")

# Extensions that make a string *look like* a path, used only to tell a real
# source_file from a producer placeholder ("Not found", "AITER (vendor)"). This
# one is permissive on purpose: rejecting an unusual-but-real path would zero a
# field the resolution tiers already filled, whereas admitting one merely leaves
# it to the classifier. Never use it as a grep admission filter.
PATH_SHAPED_EXTENSIONS = SOURCE_EXTENSIONS + (
    ".pyx",
    ".cxx",
    ".cc",
    ".hh",
    ".s",
    ".asm",
    ".jinja",
)


def _strip_template_args(symbol: str) -> str:
    """Remove C++ template argument blocks (``<...>``) from a symbol.

    Args:
        symbol (str): A possibly templated C++ symbol name.

    Returns:
        str: The symbol with all balanced ``<...>`` sections removed.
    """
    out: list[str] = []
    depth = 0
    for ch in symbol:
        if ch == "<":
            depth += 1
            continue
        if ch == ">":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


_NAMESPACE_BLOCKLIST = {
    "aiter",
    "sglang",
    "vllm",
    "torch",
    "ck_tile",
    "ck",
    "pybind",
    "RankData",
    "RankSignals",
    "Signal",
    "module",
    "namespace",
}
_TYPE_BLOCKLIST = {
    "void",
    "int",
    "float",
    "char",
    "long",
    "short",
    "bool",
    "unsigned",
    "string",
}


def _normalize_profiler_op_name(name: str) -> str:
    """Strip graph-capture / synthetic wrappers; return the original if stripping empties it."""
    original = (name or "").strip()
    if not original:
        return ""
    return _strip_dispatch_decoration(original) or original


def _candidate_keywords(name: str) -> list[str]:
    """Pick stable search keywords from a kernel symbol.

    Prefers descriptive identifiers (e.g. cross_device_reduce_2stage, gemm_a16w16)
    over namespace/type tokens (aiter, vllm, RankData) that match too widely.

    Args:
        name (str): Kernel symbol/name (possibly Itanium-mangled).

    Returns:
        list[str]: Up to three descriptive search keywords, most-specific
            first; empty when nothing usable can be extracted.
    """
    cleaned = _normalize_profiler_op_name(name)
    if cleaned.startswith("_Z"):
        # Itanium ABI uses <len><name>; slice manually so consecutive segments
        # are parsed as separate identifiers.
        tokens = []
        pos = 0
        while pos < len(cleaned):
            m = re.match(r"(\d+)", cleaned[pos:])
            if not m:
                pos += 1
                continue
            length = int(m.group(1))
            start = pos + m.end()
            ident = cleaned[start : start + length]
            if ident and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", ident):
                tokens.append(ident)
                pos = start + length
            else:
                pos = start + 1
        if not tokens:
            tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{4,}", cleaned)
    else:
        cleaned = _strip_template_args(cleaned)
        if "::" in cleaned:
            cleaned = cleaned.split("::")[-1]
        tokens = [cleaned]
    seen: set[str] = set()
    raw: list[str] = []
    for tok in tokens:
        tok = tok.rstrip("_")
        if not tok or tok in seen:
            continue
        if tok in _TYPE_BLOCKLIST:
            continue
        if len(tok) < 5:
            continue
        seen.add(tok)
        raw.append(tok)
    if not raw:
        return []
    # Prefer multi-segment identifiers and drop namespace tokens that match too widely.
    descriptive = [t for t in raw if t not in _NAMESPACE_BLOCKLIST]
    if descriptive:
        descriptive.sort(key=lambda t: (-t.count("_"), -len(t)))
        return descriptive[:3]
    raw.sort(key=lambda t: (-t.count("_"), -len(t)))
    return raw[:3]


_GREP_CACHE: dict[tuple[str, str], list[Path]] = {}


def _grep_for_keyword(keyword: str, root: Path) -> list[Path]:
    """Recursively grep ``root`` for source files containing ``keyword``.

    Results are cached per ``(keyword, root)`` and restricted to known
    source extensions. Failures (missing grep, timeout) yield ``[]``.

    Args:
        keyword (str): Literal string to search for.
        root (Path): Directory to search recursively.

    Returns:
        list[Path]: Existing source files that match, possibly empty.
    """
    if not root.exists():
        return []
    cache_key = (keyword, str(root))
    if cache_key in _GREP_CACHE:
        return _GREP_CACHE[cache_key]
    cmd = [
        "grep",
        "-rln",
        "--include=*.cuh",
        "--include=*.cu",
        "--include=*.hip",
        "--include=*.cpp",
        "--include=*.h",
        "--include=*.hpp",
        "--include=*.py",
        "--",
        keyword,
        str(root),
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=15)
    except Exception:
        _GREP_CACHE[cache_key] = []
        return []
    if proc.returncode not in (0, 1):
        _GREP_CACHE[cache_key] = []
        return []
    paths: list[Path] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        path = Path(line)
        if path.exists() and path.suffix in SOURCE_EXTENSIONS:
            paths.append(path)
    _GREP_CACHE[cache_key] = paths
    return paths


def _rank_paths(paths: list[Path], keyword: str = "") -> list[Path]:
    """Sort candidate source paths by likely relevance.

    Prefers real source repos over installed wheels and over optimized /
    build variants, then by file extension and path depth.

    Args:
        paths (list[Path]): Candidate source paths to rank.
        keyword (str): The search keyword; files whose stem contains it rank higher.

    Returns:
        list[Path]: ``paths`` sorted best-first.
    """
    kw_lower = keyword.lower()

    def score(path: Path) -> tuple[int, int, int, int]:
        """Relevance sort key for a candidate source path (lower sorts first).

        Args:
            path (Path): A candidate source path to rank.

        Returns:
            tuple[int, int, int, int]: ``(name_match, kind_score, ext_score,
            depth_penalty)`` so keyword-stem matches, impl source kinds,
            implementation extensions, and shallower paths rank ahead.
        """
        s = str(path)
        depth_penalty = s.count("/")
        kind_score = 0
        if "/csrc/" in s:
            kind_score -= 3
        if "/optimized_versions/" in s or "/build/" in s:
            kind_score += 5
        if "/site-packages/" in s:
            kind_score += 2
        ext_score = {".cuh": 0, ".cu": 0, ".hip": 0, ".cpp": 1, ".h": 2, ".hpp": 2, ".py": 3}.get(path.suffix, 4)
        # Prefer files whose stem matches the keyword over incidental mentions.
        name_match = 0 if (kw_lower and kw_lower in path.stem.lower()) else 1
        # Penalize include headers and pybind wrappers.
        if "/include/" in s:
            kind_score += 1
        if "/pybind/" in s:
            kind_score += 2
        return (name_match, kind_score, ext_score, depth_penalty)

    return sorted(paths, key=score)


def _compound_subwindow_keywords(name: str) -> list[str]:
    """Trailing snake_case sub-windows of a compound/profiler-wrapped symbol.

    A profiler-wrapped op name never appears verbatim in source, so this yields
    progressively shorter trailing windows (longest/most-specific first) so the
    embedded function symbol still resolves. The namespace/profiler prefix and a
    trailing numeric id are stripped first.

    Args:
        name: The compound/profiler-wrapped symbol name.

    Returns:
        Progressively shorter trailing snake_case windows (longest first),
        capped at six.
    """
    cleaned = _strip_template_args(_normalize_profiler_op_name(name))
    if "::" in cleaned:
        cleaned = cleaned.split("::")[-1]
    cleaned = re.sub(r"_\d+$", "", cleaned)  # drop a trailing launcher line number
    segs = [s for s in cleaned.split("_") if s]
    if len(segs) < 3:
        return []
    out: list[str] = []
    seen: set[str] = set()
    # Windows anchored at the end, dropping leading segments one at a time.
    for start in range(0, len(segs) - 1):
        window = "_".join(segs[start:])
        if len(window) >= 6 and not window.isdigit() and window not in seen:
            seen.add(window)
            out.append(window)
    return out[:6]


def _composed_name_prefix_keywords(name: str) -> list[str]:
    """Leading snake_case prefixes of a name assembled at runtime.

    The mirror image of ``_compound_subwindow_keywords``. That one handles a
    profiler prefix glued onto a real symbol, so it drops leading segments. A
    name built by an f-string has the opposite shape: the head is literal in
    source and the tail is interpolated, as in aiter's
    ``f"mfma_moe1_silu_mul_a{a_dtype}_w{b_dtype}_{out_s}"``, which no search for
    the launched name ``mfma_moe1_silu_mul_afp8_wfp8_bf16_...`` can ever match.
    So this drops trailing segments instead, longest (most specific) first.

    Args:
        name: The runtime kernel name.

    Returns:
        Progressively shorter leading snake_case prefixes (longest first),
        capped at eight.
    """
    cleaned = _strip_template_args(_normalize_profiler_op_name(name))
    if "::" in cleaned:
        cleaned = cleaned.split("::")[-1]
    segs = [s for s in cleaned.split("_") if s]
    if len(segs) < 3:
        return []
    out: list[str] = []
    # Anchored at the start, dropping trailing segments one at a time. The full
    # name is skipped: the primary pass already searched it.
    for end in range(len(segs) - 1, 1, -1):
        window = "_".join(segs[:end])
        if len(window) >= 6 and not window.isdigit():
            out.append(window)
    # The SHORTEST prefixes are the ones that can match, because everything the
    # f-string interpolated sits at the tail -- the reverse of the sub-window
    # pass, where the longest window is the likely hit. So the budget is spent
    # on the head of the name, still tried longest-first so the most specific
    # match wins.
    return out[-8:]


def _longest_literal_prefix_len(path: Path, name: str) -> int:
    """How much of ``name``'s head appears verbatim in ``path``.

    Two kernels built by neighbouring f-strings share a short prefix, so the
    keyword that found them cannot tell them apart -- aiter's
    ``mfma_moe2_a{a_dtype}...`` and ``mfma_moe2_{in_dtype}...`` both answer to
    ``mfma_moe2``. The file that spells out more of the launched name is the one
    that built it, and it is measured in characters rather than segments
    because the interpolation cuts mid-segment.

    Args:
        path: Candidate source file.
        name: The runtime kernel name.

    Returns:
        Length of the longest prefix of ``name`` found in the file, or 0.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    low = text.lower()
    target = _strip_template_args(_normalize_profiler_op_name(name)).lower()
    if "::" in target:
        target = target.split("::")[-1]
    lo, hi = 6, len(target)
    if hi < lo or target[:lo] not in low:
        return 0
    result = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if target[:mid] in low:
            result = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def _prefer_composing_file(name: str, hits: list[Path]) -> list[Path]:
    """Rank by literal prefix length; return ``[]`` when no hit scores above zero."""
    scored = sorted(
        _rank_paths(hits),
        key=lambda h: -_longest_literal_prefix_len(h, name),
    )
    if not scored or _longest_literal_prefix_len(scored[0], name) == 0:
        return []
    return scored


def _file_defines_symbol(path: Path, keyword: str) -> bool:
    """True when ``path`` *defines* ``keyword`` (vs merely mentioning it).

    Distinguishes a kernel's definition site (``def invoke_fused_moe_kernel``,
    a Triton ``@triton.jit`` function, or a C/HIP ``__global__``) from a file
    that only calls/wraps it (e.g. sglang's ``kernel_shape_profiler.py`` dispatch
    shim).

    Args:
        path: The source file to inspect.
        keyword: The symbol name to look for a definition of.

    Returns:
        ``True`` when the file defines the symbol; ``False`` on read errors or
        when it only mentions it.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    # \b cannot anchor between "_" and a letter, so a bare \bkw\b never matches
    # def _kw. Search the underscore spelling too.
    kw = re.escape(keyword)
    candidates = [kw] if keyword.startswith("_") else [kw, r"_" + kw]
    patterns: list[str] = []
    for sym in candidates:
        patterns.extend(
            [
                r"\bdef\s+" + sym + r"\b",  # Python / @triton.jit
                r"\b" + sym + r"\s*=",  # module-level assignment
                r"__global__[^\n;{]*" + sym + r"\b",  # CUDA/HIP
            ]
        )
    return any(re.search(p, text) for p in patterns)


def _prefer_symbol_definition(keyword: str, hits: list[Path]) -> list[Path]:
    """Rank definition sites of a symbol ahead of mere mentions.

    Args:
        keyword: The symbol name to prefer definitions of.
        hits: Candidate source paths to rank.

    Returns:
        The ranked paths, preferring files that define ``keyword``.
    """
    definers = [h for h in hits if _file_defines_symbol(h, keyword)]
    return _rank_paths(definers, keyword=keyword) if definers else _rank_paths(hits, keyword=keyword)


# Bare HIP/CUDA launch APIs. TraceLens emits these as standalone rows when it
# aggregates launches it could not attribute to a device kernel, but they are
# runtime entry points, not kernels. Grepping them matches wherever the API name
# merely appears -- e.g. aiter's hipify name-mapping table, which then gets
# routed to a backend as if it were rewritable kernel source.
_RUNTIME_API_NAMES: frozenset[str] = frozenset(
    {
        "cudagraphlaunch",
        "cudalaunchcooperativekernel",
        "cudalaunchkernel",
        "cudamodulelaunchkernel",
        "hipextlaunchkernel",
        "hipgraphlaunch",
        "hiplaunchcooperativekernel",
        "hiplaunchkernel",
        "hipmodulelaunchkernel",
    }
)


def is_runtime_api_name(name: str) -> bool:
    """Return whether ``name`` is a bare HIP/CUDA launch API rather than a kernel.

    Args:
        name (str): Kernel symbol/name, possibly profiler-wrapped.

    Returns:
        bool: ``True`` when the normalised name is a bare launch API.
    """
    return _normalize_profiler_op_name(name).strip().lower() in _RUNTIME_API_NAMES


def locate_source_via_grep(name: str) -> str:
    """Locate a kernel source file by grepping known repos.

    Returns "" when no confident match exists. Never fabricates a path.

    Args:
        name (str): Kernel symbol/name to locate.

    Returns:
        str: The best-ranked matching source path, or ``""`` when none.
    """
    # A bare launch API has no source of its own; grepping it only yields
    # incidental mentions (see _RUNTIME_API_NAMES).
    if is_runtime_api_name(name):
        return ""
    tried: set[str] = set()
    # Primary pass: keyword extraction + ranking. A file that DEFINES the symbol
    # outranks one that merely mentions it, the same rule the fallback pass below
    # applies: a package __init__ that re-exports the kernel scores well on path
    # shape alone and would otherwise be handed to a backend as the kernel source.
    # _prefer_symbol_definition degrades to plain ranking when nothing defines it.
    for keyword in _candidate_keywords(name):
        if not keyword or keyword in tried:
            continue
        tried.add(keyword)
        hits: list[Path] = []
        for root in kernel_search_roots():
            hits.extend(_grep_for_keyword(keyword, Path(root)))
        if hits:
            return str(_prefer_symbol_definition(keyword, hits)[0])
    # Fallback pass: trailing sub-windows of a compound/profiler-wrapped symbol
    # whose full identifier never appears verbatim in source. Prefer the file
    # that defines the embedded function over dispatch shims.
    for keyword in _compound_subwindow_keywords(name):
        if not keyword or keyword in tried:
            continue
        tried.add(keyword)
        hits = []
        for root in kernel_search_roots():
            hits.extend(_grep_for_keyword(keyword, Path(root)))
        if hits:
            return str(_prefer_symbol_definition(keyword, hits)[0])
    # Last pass: the name was assembled by an f-string, so only its head is
    # literal in source and both passes above searched text that is never
    # written down. Ranked by how much of the name a file spells out, because
    # the shortest prefixes are shared by sibling kernels in sibling files.
    for keyword in _composed_name_prefix_keywords(name):
        if not keyword or keyword in tried:
            continue
        tried.add(keyword)
        hits = []
        for root in kernel_search_roots():
            hits.extend(_grep_for_keyword(keyword, Path(root)))
        if hits:
            ranked = _prefer_composing_file(name, hits)
            if ranked:
                return str(ranked[0])
    return ""


def _inject_collective_candidates(
    tracelens_dir: Path,
    candidates: list[dict[str, Any]],
    *,
    source_roots: list[str] | None = None,
    log_path: Path | None = None,
    health_warnings: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge source-resolved NCCL rows with traced all-reduce workloads.

    Injection is best-effort: a malformed ``nccl_summary`` must not fail the
    whole analysis. It does, however, remove the only path a collective has into
    the candidate pool, so every skip is recorded in ``health_warnings`` where
    the report surfaces it -- a log line alone leaves the lane looking as if the
    workload simply had no collective.
    """
    if not isinstance(candidates, list) or any(not isinstance(item, dict) for item in candidates):
        raise ValueError("Collective candidates must be a list of mappings")
    existing = [dict(item) for item in candidates]
    roots = list(source_roots or [])
    if not roots:
        aiter_csrc = _aiter_csrc_root().rstrip("/")
        if aiter_csrc and Path(aiter_csrc).is_dir():
            roots.append(aiter_csrc)

    def _skip(code: str, detail: str, notes: list[str] | None = None) -> list[dict[str, Any]]:
        """Record a visible reason the collective lane got no candidate."""
        message = f"nccl_summary: {detail}; skipping injection"
        log.warning(message)
        if log_path is not None:
            # The per-symbol notes are the only record of WHICH symbol failed and
            # they are normally flushed at the end, which a skip never reaches.
            for note in notes or []:
                append_log(log_path, note)
            append_log(log_path, message)
        if health_warnings is not None:
            health_warnings.append(
                {
                    "code": code,
                    "severity": "warning",
                    "message": (
                        "No collective candidate was injected, so the collective "
                        f"optimization lane cannot run: {detail}"
                    ),
                }
            )
        return existing

    if not roots:
        return _skip(
            "collective_source_root_missing",
            "no bounded collective source root",
        )

    messages: list[str] = []
    scan_diagnostics: list[dict[str, Any]] = []
    try:
        extracted = extract_collective_candidates(
            tracelens_dir,
            roots,
            log_fn=messages.append,
            diagnostics=scan_diagnostics,
        )
    except ValueError as exc:
        return _skip("collective_summary_unusable", str(exc))
    for diagnostic in scan_diagnostics:
        message = str(diagnostic.get("message") or "")
        log.warning(message)
        if health_warnings is not None:
            health_warnings.append(
                {
                    "code": str(diagnostic.get("code") or ""),
                    "severity": "warning",
                    "message": message,
                    "scanned_file_limit": diagnostic.get("scanned_file_limit"),
                    "source_roots": list(diagnostic.get("source_roots") or []),
                }
            )
    if not extracted:
        # Every summary row failed device-symbol resolution. Individually those
        # are debug detail, but together they mean the trace saw collectives and
        # the lane still got nothing.
        return _skip(
            "collective_symbols_unresolved",
            "no summary row resolved to a device source under " + ", ".join(roots),
            notes=messages,
        )

    def _name(item: dict[str, Any]) -> str:
        """Return the normalized trace name for one candidate."""
        return str(item.get("name") or "").strip().lower()

    def _workload_dtypes(item: dict[str, Any]) -> list[str]:
        """Return explicit or shape-derived input dtypes for one workload."""
        values = item.get("input_dtypes") or item.get("dtypes") or []
        if not values:
            values = _dtypes_from_shapes(item.get("shapes") or [])
        return [str(value).strip() for value in values if isinstance(value, str) and value.strip()]

    def _has_workload(item: dict[str, Any]) -> bool:
        """Return whether the trace row carries driver inputs."""
        return bool(item.get("input_shapes") or item.get("shapes")) and bool(_workload_dtypes(item))

    def _is_all_reduce_workload(item: dict[str, Any]) -> bool:
        """Return whether a traced workload has all-reduce semantics."""
        contract = item.get("kernel_contract")
        if isinstance(contract, dict) and str(contract.get("kind") or "") == "collective":
            return str(contract.get("collective_op") or "") == "all_reduce"
        name = _name(item)
        return any(tag in name for tag in ("all_reduce", "allreduce", "cross_device_reduce"))

    def _workload_family(item: dict[str, Any]) -> str:
        """Collapse prefill and decode rows from one profiled wrapper."""
        name = _name(item)
        if name.startswith(("sglang_profiler::", "vllm_profiler::")):
            name = name.split("->", 1)[0]
        return re.sub(r"\s+\((?:prefill|decode)\)\s*$", "", name)

    def _first_shape_record(item: dict[str, Any]) -> dict[str, Any] | None:
        """Return the first tensor shape record from one invocation."""
        records = item.get("input_shapes")
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict) and str(record.get("shape") or "").strip():
                    return dict(record)
        shapes = item.get("shapes")
        if isinstance(shapes, list):
            for shape in shapes:
                text = str(shape or "").strip()
                if text:
                    return {
                        "call_num": int(item.get("call_count") or 1),
                        "shape": text,
                    }
        return None

    def _merge_workloads(
        target: dict[str, Any],
        donors: list[dict[str, Any]],
    ) -> None:
        """Attach distinct first-input cases from one workload family."""
        records: list[dict[str, Any]] = []
        shapes: list[str] = []
        dtypes: list[str] = []
        seen_shapes: set[str] = set()
        for donor in donors:
            record = _first_shape_record(donor)
            if record is None:
                continue
            shape = str(record["shape"]).strip()
            if shape in seen_shapes:
                continue
            seen_shapes.add(shape)
            records.append(record)
            shapes.append(shape)
            donor_dtypes = _workload_dtypes(donor)
            if donor_dtypes:
                dtypes.append(donor_dtypes[0])
        if records:
            target["input_shapes"] = records
            target["shapes"] = shapes
        if dtypes:
            target["input_dtypes"] = dtypes
        provenance = next(
            (str(donor.get("shape_provenance") or "") for donor in donors if donor.get("shape_provenance")),
            "",
        )
        if provenance:
            target["shape_provenance"] = provenance
        target["workload_source_kernels"] = [str(donor.get("name") or "") for donor in donors]
        hottest = max(
            donors,
            key=lambda donor: float(donor.get("duration_us") or 0.0),
        )
        target["workload_source_kernel"] = str(hottest.get("name") or "")

    by_name = {_name(item): item for item in existing if _name(item)}
    workload_rows = [item for item in existing if _has_workload(item) and _is_all_reduce_workload(item)]
    workload_families: dict[str, list[dict[str, Any]]] = {}
    for row in workload_rows:
        family = _workload_family(row)
        if family:
            workload_families.setdefault(family, []).append(row)
    allow_inferred_shapes = os.environ.get("HYPERLOOM_COLLECTIVE_ALLOW_INFERRED_SHAPES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    appended: list[dict[str, Any]] = []
    for item in extracted:
        exact = by_name.get(_name(item))
        target = exact or dict(item)
        for key in (
            "source_file",
            "source_line",
            "source_function",
            "source_resolution_method",
            "candidate_source",
            "collective_stream",
            "nccl_summary_total_ms",
            "duration_provenance",
        ):
            target[key] = item[key]
        if exact is not None and _has_workload(exact):
            donors = [exact]
            borrowed = False
        elif allow_inferred_shapes and len(workload_families) == 1 and _is_all_reduce_workload(item):
            # Shapes are inferred from the sole all-reduce family, valid only
            # because the symbol is itself an all-reduce. Handing the driver
            # shapes the traced kernel never ran would yield confident SNR and
            # latency for a workload that never existed, so ``shape_provenance``
            # travels with them to mark the values attributed, not observed.
            donors = next(iter(workload_families.values()))
            borrowed = True
        else:
            donors = []
            borrowed = False
        if len(donors) == 1:
            donor = donors[0]
            for key in ("input_shapes", "shapes", "input_dtypes", "dtypes", "shape_provenance"):
                if donor.get(key):
                    target[key] = donor[key]
            if not target.get("input_dtypes"):
                target["input_dtypes"] = _workload_dtypes(donor)
            target["workload_source_kernel"] = str(donor.get("name") or "")
        elif donors:
            _merge_workloads(target, donors)
        if borrowed:
            target["shape_provenance"] = "borrowed_sole_all_reduce_family"
            messages.append(
                "nccl_summary: attributing the trace's only all-reduce workload "
                f"to {item.get('source_function')!r}; shapes are inferred"
            )
        if exact is None:
            if not donors:
                messages.append(
                    "nccl_summary: no unique traced all-reduce workload for "
                    f"{item.get('source_function')!r}; dropping candidate"
                )
                continue
            appended.append(target)
            by_name[_name(target)] = target
    if log_path is not None:
        for message in messages:
            append_log(log_path, message)
        for item in appended:
            append_log(
                log_path,
                "nccl_summary: injected source-resolved collective "
                f"{item.get('source_function')!r} from {item.get('source_file')}",
            )
    return existing + appended


def find_repo_root(source_file: str) -> str:
    """Walk upward from source_file until we find a .git/ dir; return the dir.

    Returns "" when no git repo root is found.

    Args:
        source_file (str): Path to a file inside a (possibly) git repo.

    Returns:
        str: The directory containing the nearest ``.git`` ancestor, or
            ``""`` when none is found.
    """
    if not source_file:
        return ""
    p = Path(source_file).expanduser().resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return str(parent)
    return ""


_BENCHMARK_DIRS = ("op_tests", "tests", "benchmarks", "benchmark", "test", "perf")


#: Curated harness lookups, keyed by marker substrings in a kernel's name or
#: source path. Paths are *checkout-relative* on purpose: the same harness sits
#: at ``/sgl-workspace/aiter/op_tests/...`` in a serving image and is absent from
#: a wheel install, so a pinned absolute path is either right on one host or a
#: fabrication on every other.
_KNOWN_HARNESS_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    # --- Normalization ---
    (
        (
            "rmsnorm_quant",
            "add_rmsnorm_quant",
            "rmsnorm",
            "add_rmsnorm",
        ),
        (
            "aiter/op_tests/test_rmsnorm2dFusedAddQuant.py",
            "aiter/op_tests/test_rmsnorm2d.py",
            "aiter/op_tests/op_benchmarks/triton/bench_rmsnorm.py",
            "sglang/sgl-kernel/benchmark/bench_rmsnorm.py",
            "aiter/op_tests/triton_tests/normalization/test_rmsnorm.py",
            "aiter/op_tests/triton_tests/normalization/test_fused_add_rmsnorm_pad.py",
        ),
    ),
    # --- Activation ---
    (
        (
            "activation",
            "act_and_mul",
            "silu",
        ),
        (
            "aiter/op_tests/test_activation.py",
            "aiter/op_tests/op_benchmarks/triton/bench_ff_a16w16_fused.py",
            "sglang/sgl-kernel/tests/test_activation.py",
            "sglang/sgl-kernel/benchmark/bench_activation.py",
            "sglang/python/sglang/jit_kernel/tests/test_activation.py",
            "sglang/python/sglang/jit_kernel/benchmark/bench_activation.py",
        ),
    ),
    # --- Attention ---
    (
        (
            "paged_attention",
            "fmha",
            "attention",
        ),
        (
            "aiter/op_tests/test_pa.py",
            "aiter/op_tests/op_benchmarks/triton/bench_pa_decode.py",
            "aiter/op_tests/op_benchmarks/triton/bench_pa_prefill.py",
        ),
    ),
    # --- MLA decode ---
    (
        (
            "mla_decode",
            "pseudo_mla",
            "mla_persistent",
        ),
        (
            "aiter/op_tests/test_mla.py",
            "aiter/op_tests/test_mla_persistent.py",
            "aiter/op_tests/op_benchmarks/triton/bench_mla_decode.py",
        ),
    ),
    # --- MoE CK two-stage ---
    (
        (
            "ck_moe_stage",
            "moe_2stage",
            "moe_stage1",
            "moe_stage2",
        ),
        (
            "aiter/op_tests/test_moe_2stage.py",
            "aiter/op_tests/op_benchmarks/triton/bench_moe.py",
        ),
    ),
    # --- MoE FP8 blockscale (ASM) ---
    (
        (
            "fmoe_fp8_blockscale",
            "moe_blockscale",
        ),
        (
            "aiter/op_tests/test_moe_blockscale.py",
            "aiter/op_tests/triton_tests/moe/test_moe_gemm_a8w8_blockscale.py",
        ),
    ),
    # --- GEMM A8W8 blockscale ---
    (
        ("gemm_a8w8_blockscale",),
        (
            "aiter/op_tests/test_gemm_a8w8_blockscale.py",
            "aiter/op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py",
        ),
    ),
    # --- Quantization ---
    (
        (
            "dynamic_per_token_scaled_quant",
            "per_token_quant",
        ),
        (
            "aiter/op_tests/test_quant.py",
            "aiter/op_tests/triton_tests/quant/test_quant.py",
        ),
    ),
    # --- Batch-invariant addmm (Triton) ---
    (
        (
            "batch_invariant",
            "addmm",
        ),
        ("sglang/test/registered/unit/batch_invariant_ops/test_batch_invariant_ops.py",),
    ),
)


@lru_cache(maxsize=1)
def _harness_search_bases() -> tuple[str, ...]:
    """Directories a checkout-relative harness path may be joined onto.

    A hint reads ``aiter/op_tests/...``, so the base is whatever holds the
    ``aiter`` checkout. Each resolved search root contributes both itself and
    its parent, because a root is the package directory on a wheel install
    (``.../dist-packages/aiter``) and the checkout itself in a serving image
    (``/sgl-workspace/aiter``); one join is the right one and the other simply
    does not exist.

    Returns:
        tuple[str, ...]: Existing base directories, de-duplicated.
    """
    bases: list[str] = []
    seen: set[str] = set()
    for root in kernel_search_roots():
        trimmed = root.rstrip("/")
        for base in (os.path.dirname(trimmed), trimmed):
            if base and base not in seen and os.path.isdir(base):
                seen.add(base)
                bases.append(base)
    return tuple(bases)


def _known_harness_files(name: str, source_file: str) -> list[Path]:
    """Return curated benchmark/test harnesses matching a kernel.

    Resolves each checkout-relative hint against the bases that exist here and
    keeps only files actually present. A list naming paths that cannot be
    opened is worse than an empty one, because every reader downstream -- the
    dispatch prompt included -- treats a non-empty list as a harness it can run.

    Args:
        name (str): Kernel symbol/name.
        source_file (str): Resolved source-file path (may be empty).

    Returns:
        list[Path]: Existing curated harness files, possibly empty.
    """
    blob = f"{name} {source_file}".lower()
    out: list[Path] = []
    seen: set[str] = set()
    bases = _harness_search_bases()
    for markers, relatives in _KNOWN_HARNESS_HINTS:
        if not any(marker in blob for marker in markers):
            continue
        for relative in relatives:
            for base in bases:
                candidate = os.path.join(base, relative)
                if candidate not in seen and os.path.isfile(candidate):
                    seen.add(candidate)
                    out.append(Path(candidate))
                    break
    return out


_LIBRARY_TOKENS = ("aiter", "sglang", "vllm", "flashinfer", "sgl-kernel", "sgl_kernel")


def _library_token(path: str) -> str:
    """Return the library a path belongs to (aiter/sglang/...), else "".

    Used to keep kernel↔benchmark pairing within one library. sgl-kernel and
    sgl_kernel normalize to "sglang" since they are the sglang kernel package.
    """
    low = (path or "").lower()
    for tok in _LIBRARY_TOKENS:
        if f"/{tok}/" in low or f"/{tok}." in low:
            return "sglang" if tok in ("sgl-kernel", "sgl_kernel") else tok
    return ""


def find_benchmark_files(name: str, repo_root: str, source_file: str = "") -> list[str]:
    """Find test/benchmark files matching a kernel under a repo's subdirs.

    Searches ``repo_root``'s known benchmark/test subdirectories for the
    kernel keywords, returning absolute paths.

    Args:
        name: Kernel symbol/name.
        repo_root: Repo root to search; empty returns only curated harnesses.
        source_file: Resolved source-file path, used to derive extra keywords.

    Returns:
        Up to ten matching harness paths, with multi-GPU tests demoted.
    """
    known = _known_harness_files(name, source_file)
    if not repo_root:
        return [str(p) for p in known[:10]]
    keywords = _candidate_keywords(name)
    # Add the source stem (and no-underscore variant) for repos that name tests differently.
    if source_file:
        stem = Path(source_file).stem
        if stem and stem not in keywords:
            keywords.append(stem)
        no_us = stem.replace("_", "")
        if len(no_us) >= 6 and no_us not in keywords:
            keywords.append(no_us)
    if not keywords:
        return []
    root = Path(repo_root)
    found: list[Path] = list(known)
    for sub in _BENCHMARK_DIRS:
        sub_root = root / sub
        if not sub_root.exists():
            continue
        for keyword in keywords:
            try:
                proc = subprocess.run(
                    [
                        "grep",
                        "-rln",
                        "--include=*.py",
                        "--include=*.cpp",
                        "--include=*.cu",
                        "--include=*.cuh",
                        "--include=*.hip",
                        "--include=*.sh",
                        keyword,
                        str(sub_root),
                    ],
                    text=True,
                    capture_output=True,
                    timeout=15,
                )
            except Exception:
                continue
            if proc.returncode not in (0, 1):
                continue
            for line in (proc.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if not p.exists():
                    continue
                found.append(p)
    seen: set[str] = set()
    unique: list[str] = []
    for p in found:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        unique.append(s)

    # Demote multi-GPU / distributed tests to the end: backends running on a
    # single Ray worker can't satisfy them, and they tend to make agents bail.
    def _is_multigpu(path_str: str) -> bool:
        """Return whether a harness path looks multi-GPU / distributed.

        Args:
            path_str (str): Candidate harness file path.

        Returns:
            bool: ``True`` when the path contains a multi-GPU/distributed tag.
        """
        low = path_str.lower()
        return any(tag in low for tag in ("multigpu", "multi_gpu", "multinode", "/dist/", "_dist_"))

    # Same-library guard (RCA root cause 2): never pair a kernel from one library
    # with a benchmark from another (e.g. a sglang sgl-kernel .cuh with an aiter
    # op_test). Editing the kernel then "validating" against an unrelated lib's
    # op always fails the smoke test -> REVERT. Drop cross-library candidates
    # when the kernel's library is recognizable.
    src_lib = _library_token(source_file)
    if src_lib:
        same_lib = [s for s in unique if _library_token(s) in (src_lib, "")]
        if same_lib:
            unique = same_lib
    unique.sort(key=_is_multigpu)
    return unique[:10]


_PYBIND_PARENT_DIRS = ("csrc/pybind", "csrc/python", "python_bindings")


# A pybind11 registration shim has no device code; callers promote it to the real .cu/.cuh.
def _is_pybind_shim(source_file: str) -> bool:
    """Detect a tiny pybind11 registration TU (no rewritable device code).

    A shim is a small (<2 KB) ``.cu``/``.cpp``/``.cc`` file under a pybind
    directory that only contains ``PYBIND11_MODULE`` glue.

    Args:
        source_file (str): Candidate source-file path.

    Returns:
        bool: ``True`` when the file is a pybind11 registration shim.
    """
    if not source_file:
        return False
    p = Path(source_file)
    if not any(d in source_file for d in _PYBIND_PARENT_DIRS):
        return False
    if not source_file.endswith((".cu", ".cpp", ".cc")):
        return False
    try:
        if p.stat().st_size > 2048:
            return False
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    return "PYBIND11_MODULE" in text or "pybind11" in text


# Framework package-inner roots for resolving relative launcher paths
# (TraceLens emits package-dir-relative paths). Tried when
# _resolve_launcher_to_abs_source fails.
_PACKAGE_INNER_ROOTS = (
    "/sgl-workspace/aiter/aiter",
    "/sgl-workspace/sglang/python/sglang",
)


def upgrade_pybind_shim_source(source_file: str, kernel_name: str, kernel_repo: str) -> str:
    """Promote a tiny pybind11 shim to the real device source.

    When ``source_file`` is a pybind11 shim, finds the ``.cu``/``.cuh``
    implementing ``kernel_name``, preferring a same-stem file under
    ``csrc/py_itfs_cu|kernels|include`` before grepping the symbol.

    Args:
        source_file: The candidate's resolved source path.
        kernel_name: The kernel name used to locate the device source.
        kernel_repo: The resolved kernel repo root.

    Returns:
        The real device source path, or ``source_file`` unchanged when no
        better target is found.
    """
    if not _is_pybind_shim(source_file):
        return source_file
    repo = Path(kernel_repo) if kernel_repo else Path(source_file).parent.parent.parent
    if not repo.is_dir():
        return source_file
    stem = Path(source_file).stem.replace("_pybind", "").replace("_asm_pybind", "")
    # Strategy 1: same-stem file under a csrc source dir.
    for sub in ("csrc/py_itfs_cu", "csrc/kernels", "csrc/include", "csrc/asm"):
        for ext in (".cu", ".cuh", ".cpp", ".h", ".hpp"):
            candidates = list((repo / sub).glob(f"*{stem}*{ext}")) if (repo / sub).is_dir() else []
            for c in candidates:
                if _is_pybind_shim(str(c)):
                    continue
                if c.stat().st_size > 2048:
                    return str(c)
    # Strategy 2: grep the kernel symbol name.
    sym = kernel_name.split("(")[0].split("<")[0].split("::")[-1]
    if sym and len(sym) >= 4:
        for ext in ("*.cu", "*.cuh", "*.hip"):
            for f in repo.rglob(ext):
                if _is_pybind_shim(str(f)):
                    continue
                try:
                    if sym in f.read_text(encoding="utf-8", errors="replace"):
                        if f.stat().st_size > 2048:
                            return str(f)
                except Exception:
                    continue
    return source_file


def _coerce_count(value: Any) -> int | None:
    """Coerce a loosely-typed call-count value into a positive int.

    Handles ``None``/empty, numpy-repr strings (``np.float64(...)``), and
    float-like strings.

    Args:
        value (Any): The raw count value to coerce.

    Returns:
        int | None: A positive integer, or ``None`` when absent, unparseable,
            or non-positive.
    """
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.startswith("np.float64(") and text.endswith(")"):
        text = text[len("np.float64(") : -1]
    try:
        count = int(float(text))
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def _merge_shape_call(target: list[Any], shape: Any, call_num: int) -> None:
    """Merge a shape/call-count pair into an accumulator list in place.

    If an entry with the same shape already exists, its ``call_num`` is
    incremented; otherwise a new entry is appended.

    Args:
        target (list[Any]): Accumulator list of ``{"shape", "call_num"}`` dicts.
        shape (Any): The shape value to merge.
        call_num (int): Call count to add for this shape.
    """
    for entry in target:
        if isinstance(entry, dict) and entry.get("shape") == shape:
            entry["call_num"] = int(entry.get("call_num") or 0) + call_num
            return
    target.append({"call_num": call_num, "shape": shape})


def _shape_call_entries(shapes: Any, call_num: Any = None) -> list[dict[str, Any]]:
    """Normalize a shapes payload into merged ``{shape, call_num}`` entries.

    Args:
        shapes (Any): A list of shape values or ``{"shape", "call_num"}`` dicts.
        call_num (Any): Default call count applied when an entry lacks one.

    Returns:
        list[dict[str, Any]]: Deduplicated shape entries with summed call
            counts; empty when ``shapes`` is not a list.
    """
    if not isinstance(shapes, list):
        return []
    count = _coerce_count(call_num) or 1
    entries: list[dict[str, Any]] = []
    for shape in shapes:
        if isinstance(shape, dict):
            value = shape.get("shape")
            shape_count = _coerce_count(shape.get("call_num")) or count
        else:
            value = shape
            shape_count = count
        if value in (None, "", [], ()):
            continue
        _merge_shape_call(entries, value, shape_count)
    return entries


# Recognized dtype tokens appearing as the suffix on a TraceLens shape entry
# (e.g. ``(64,5120) bf16``); used to recognise a trailing dtype on a paren-less entry.
_KNOWN_DTYPE_TOKENS = frozenset(
    {
        "bf16",
        "fp16",
        "fp32",
        "f32",
        "fp8",
        "f8",
        "fp8_e4m3",
        "fp8_e5m2",
        "e4m3",
        "e5m2",
        "int8",
        "int4",
        "int32",
        "int64",
        "int",
        "uint8",
        "bool",
        "float",
        "double",
        "half",
    }
)


def _split_shape_dtype(entry: Any) -> tuple[str, str]:
    """Split a TraceLens shape entry ``"(64,5120) bf16"`` into ``("(64,5120)", "bf16")``.

    Separates the inline dtype so the per-arg dtype can be surfaced as a
    positionally aligned ``input_dtypes`` list. Non-string / dict entries return
    ``(str(value), "")``; an empty scalar ``()`` keeps an empty dtype so arity
    stays aligned with the call signature.
    """
    if isinstance(entry, dict):
        entry = entry.get("shape", "")
    s = str(entry).strip()
    if not s:
        return "", ""
    if s.startswith("("):
        close = s.find(")")
        if close != -1:
            return s[: close + 1].strip(), s[close + 1 :].strip()
        return s, ""
    parts = s.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].lower() in _KNOWN_DTYPE_TOKENS:
        return parts[0].strip(), parts[1].strip()
    return s, ""


def _dtypes_from_shapes(shapes: Any) -> list[str]:
    """Ordered per-arg dtype list parsed from a candidate's ``shapes`` strings.

    Positionally aligned with the shapes (one entry per arg, ``""`` when no dtype
    was captured). Returns ``[]`` when no dtype is present on any entry, so callers
    keep the existing empty-default behaviour when TraceLens captured none.
    """
    if not isinstance(shapes, list):
        return []
    dtypes = [_split_shape_dtype(s)[1] for s in shapes]
    return dtypes if any(dtypes) else []


def derive_kernel_category(candidate: dict[str, Any]) -> str:
    """Map a candidate to its GEAK-facing kernel category.

    Priority: explicit TraceLens category (normalized), then a kernel-name
    heuristic, then ``unknown``.

    Args:
        candidate: The hot-kernel candidate dict.

    Returns:
        The GEAK-facing kernel category string.
    """
    cat = (candidate.get("tracelens_category") or "").strip()
    if cat:
        return normalize_upstream_category(cat)
    if candidate.get("source_type") == "flydsl":
        return "FlyDSL"
    name = str(candidate.get("name") or "").lower()
    if any(
        t in name
        for t in (
            "gemm",
            "matmul",
            "rocblas",
            "hipblas",
            "cijk",
            "sgemm",
            "hgemm",
            # PyTorch op-name variants.
            "::mm",
            "::addmm",
            "::bmm",
        )
    ):
        return "GEMM"
    if any(t in name for t in ("attention", "attn", "fmha", "paged_attention", "flash")):
        return "SDPA"
    if "rmsnorm" in name or "layernorm" in name or "norm_kernel" in name:
        return "LayerNorm"
    if "act_and_mul" in name or "silu" in name or "gelu" in name or "activation" in name:
        return "Activation"
    if "moe" in name or "topk" in name or "expert" in name:
        return "MoE"
    if "softmax" in name:
        return "Softmax"
    if "embed" in name:
        return "Embedding"
    if "reduce" in name or "all_reduce" in name or "all_gather" in name:
        return "Communication"
    if "triton" in name:
        return "Triton"
    if "elementwise" in name or "binary" in name:
        return "Elementwise"
    return "unknown"


def is_multigpu_kernel(name: str, source_file: str) -> bool:
    """Heuristic: kernel is a multi-GPU collective if name/source hints it.

    Args:
        name (str): Kernel symbol/name.
        source_file (str): Resolved source-file path (may be empty).

    Returns:
        bool: ``True`` when the name/source contains a collective /
            distributed marker.
    """
    blob = f"{name} {source_file}".lower()
    return any(
        tag in blob
        for tag in (
            "all_reduce",
            "allreduce",
            "all_gather",
            "allgather",
            "reduce_scatter",
            "broadcast",
            "p2p",
            "send_recv",
            "cross_device",
            "rank_signal",
            "ranksignals",
            "/dist/",
            "dist/",
            "communicator",
        )
    )


def analyze_trace_files(
    trace_files: list[Path],
    top_k: int,
) -> list[dict[str, Any]]:
    """Aggregate GPU kernels across raw trace files into top-K candidates.

    Sums per-kernel duration and call counts across all events, takes the
    top ``top_k`` by duration, then runs :func:`_finalize_candidates`. This
    is the dry-run / test-only raw-trace path (production uses analysis.md).

    Args:
        trace_files (list[Path]): Trace files (optionally gzipped) to scan.
        top_k (int): Number of hottest kernels to keep.

    Returns:
        list[dict[str, Any]]: Finalized hot-kernel candidate dicts.
    """
    aggregates: dict[str, dict[str, Any]] = {}
    total_dur = 0.0

    for trace_file in trace_files:
        try:
            payload = open_json(trace_file)
        except Exception:
            continue

        if isinstance(payload.get("kernels"), list):
            events = payload["kernels"]
        else:
            events = payload.get("traceEvents", [])
        if not isinstance(events, list):
            continue

        for event in events:
            if not isinstance(event, dict) or not is_kernel_event(event):
                continue
            name = str(event.get("kernel_name") or event.get("name") or "unknown_kernel")
            dur = float(event.get("dur") or event.get("duration_us") or event.get("duration") or 0)
            if dur <= 0:
                continue
            total_dur += dur
            item = aggregates.setdefault(
                name,
                {
                    "name": name,
                    "duration_us": 0.0,
                    "call_count": 0,
                    "source_file": "",
                    "source_type": "unknown",
                    "shapes": [],
                },
            )
            item["duration_us"] += dur
            item["call_count"] += 1
            if not item.get("_extracted_source_checked"):
                item["source_file"] = extract_source_file(event)
                item["_extracted_source_checked"] = True
            shape = extract_shape(event)
            if shape and shape not in item["shapes"]:
                item["shapes"].append(shape)

    candidates = sorted(aggregates.values(), key=lambda x: x["duration_us"], reverse=True)
    top = candidates[:top_k]
    return _finalize_candidates(
        top,
        total_dur=total_dur,
        trace_files=trace_files,
    )


def load_op_category_map(
    perf_report_csv_dir: Path | str,
) -> dict[str, str]:
    """Read a ``{name: raw op category}`` map from the unified perf summary.

    Args:
        perf_report_csv_dir: Directory containing
            ``unified_perf_summary.csv``.

    Returns:
        A mapping of kernel name to its first non-empty TraceLens op category,
        or ``{}`` when the CSV is absent or unreadable.
    """
    csv_path = Path(perf_report_csv_dir) / "unified_perf_summary.csv"
    if not csv_path.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = str(row.get("name") or "").strip()
                cat = str(row.get("op category") or "").strip()
                if name and cat and name not in out:
                    out[name] = cat
    except (OSError, csv.Error):
        return {}
    return out


# Capture a device-kernel ``name`` and its total (preferred) or mean duration from
# TraceLens' per-row ``kernel_details_summary`` / ``trunc_kernel_details`` repr;
# the non-greedy match prefers total when present.
_KERNEL_DETAIL_RE = re.compile(
    r"'name':\s*'((?:[^'\\]|\\.)*)'.*?'(total_duration_us|mean_duration_us)':\s*(?:np\.float64\()?([0-9.eE+-]+)"
)


def load_op_dominant_kernel_map(perf_report_csv_dir: Path | str) -> dict[str, str]:
    """Read ``{op_name: dominant_device_kernel_name}`` from the unified perf summary.

    A composite profiler op fires several device kernels under one CPU op; the
    dominant (max aggregated duration) one is the real hot kernel, and surfacing
    it as ``device_kernel_name`` lets the active finder pin the single owning
    source. ``{}`` when the CSV is absent or unreadable.
    """
    csv_path = Path(perf_report_csv_dir) / "unified_perf_summary.csv"
    if not csv_path.is_file():
        return {}
    agg: dict[str, dict[str, float]] = {}
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = str(row.get("name") or "").strip()
                if not name:
                    continue
                detail = str(row.get("kernel_details_summary") or row.get("trunc_kernel_details") or "")
                if not detail:
                    continue
                per = agg.setdefault(name, {})
                for m in _KERNEL_DETAIL_RE.finditer(detail):
                    kname = m.group(1).strip()
                    try:
                        dur = float(m.group(3))
                    except (TypeError, ValueError):
                        continue
                    if kname:
                        per[kname] = per.get(kname, 0.0) + dur
    except (OSError, csv.Error):
        return {}
    return {op: max(per.items(), key=lambda kv: kv[1])[0] for op, per in agg.items() if per}


@lru_cache(maxsize=1)
def _load_finder_index() -> Any:
    """Build the live source index once per process (``None`` if unavailable).

    The index is an optimization: any failure (finder not importable, no
    framework trees discovered, scan error) degrades to curated-only resolution.
    """
    if _kernel_source_index is None:
        return None
    try:
        return _kernel_source_index.load_or_build()
    except Exception:  # noqa: BLE001 - best-effort; degrade to curated-only.
        return None


def _resolve_via_active_finder(
    op_name: str,
    framework: str | None,
    device_kernel_name: str | None,
    index: Any,
) -> "OpResolution | None":
    """Resolve via the active finder (symbol -> live index): the deterministic tier.

    Returns a routable or ``non_rewritable`` :class:`OpResolution` on a symbol
    hit, or ``None`` (a miss) so the caller falls through to the downstream
    trace-stack / grep / LLM tiers. The finder is symbol-driven, so it only runs
    when a device kernel name is available.
    """
    if _active_finder is None or not device_kernel_name:
        return None
    try:
        res = _active_finder.resolve(
            op_name,
            framework=framework or "",
            device_kernel_name=device_kernel_name,
            index=index,
        )
    except Exception:  # noqa: BLE001 - a finder failure must fall through, not raise.
        return None
    op = _PHASE_SUFFIX_RE.sub("", op_name)
    if res.source_file and res.patchable:
        return OpResolution(
            op_name=op,
            kind="single",
            status=_ROUTABLE_STATUS,
            patchable=True,
            framework=framework,
            sources=[res.source_file],
            matched_route=device_kernel_name,
            resolution_method=_ACTIVE_FINDER_METHOD,
        )
    if res.method == "non_patchable":
        return OpResolution(
            op_name=op,
            kind="single",
            status="non_rewritable",
            patchable=False,
            framework=framework,
            reason=res.reason or "non-patchable kernel (symbol-detected)",
            matched_route=device_kernel_name,
            resolution_method=_ACTIVE_FINDER_METHOD,
        )
    return None


def _expand_op_fanout(
    top: list[dict[str, Any]],
    framework: str | None = None,
    op_dominant_kernel: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve each op via the active finder and fan out one candidate per ``.cu``.

    A routable op with N editable sources becomes N candidates, each routed to
    its own GEAK run; the op's ``duration_us`` is split evenly so ``gpu_pct`` is
    not inflated. Non-routable ops and finder misses pass through unchanged, with
    their :class:`OpResolution` (or ``None``) cached on ``_op_resolution`` for
    finalize, which then applies the downstream trace/grep/LLM tiers.

    ``framework`` disambiguates which framework tree's source to prefer when a
    symbol lives in more than one; only ``vllm``/``sglang`` are honored.
    """
    framework = (framework or "").strip().lower() or None
    op_dominant_kernel = op_dominant_kernel or {}
    finder_index = _load_finder_index()
    expanded: list[dict[str, Any]] = []
    for item in top:
        op_name = str(item.get("name") or "")
        # Prefer the candidate's own device symbol; else fall back to the
        # dominant-by-time device kernel so _composite pins the single hot source.
        dkn = str(item.get("device_kernel_name") or "").strip() or op_dominant_kernel.get(op_name) or None
        # Active finder (symbol -> live installed source) is the deterministic
        # tier. On a miss, _op_resolution is None so _finalize_candidates falls
        # through to the trace-stack / grep / LLM tiers.
        res = _resolve_via_active_finder(op_name, framework, dkn, finder_index)
        if res is None:
            item["_op_resolution"] = None
            expanded.append(item)
            continue
        leaves = res.leaf_resolutions()
        if len(leaves) <= 1:
            # 1 leaf -> attach it; 0 leaves -> keep res for stamping.
            item["_op_resolution"] = leaves[0] if leaves else res
            expanded.append(item)
            continue
        orig_dur = item.get("duration_us", 0.0) or 0.0
        for idx, leaf in enumerate(leaves):
            clone = dict(item)
            clone["_op_resolution"] = leaf
            clone["op_fanout_index"] = idx
            clone["op_fanout_total"] = len(leaves)
            clone["duration_us"] = orig_dur / len(leaves)
            expanded.append(clone)
    return expanded


# Trace-anchored shape capture for the fused-MoE expert kernel: its top-level
# kernel event carries no resolvable ``Input Dims``, so the candidate emits
# ``shapes: []`` and the dispatch gate rejects it. ``ops_unique_args.csv`` still
# carries the trace-recorded operands, so recover missing shapes from it by exact
# op-name match (or fused-MoE marker match).
_FUSED_MOE_KERNEL_MARKER = "invoke_fused_moe_kernel"

# torch ``Input type`` token → compact dtype suffix used by TraceLens shape
# strings. Unmapped/empty types emit a bare shape.
_TRACE_DTYPE_SUFFIX = {
    "c10::bfloat16": "bf16",
    "bfloat16": "bf16",
    "c10::half": "f16",
    "half": "f16",
    "float16": "f16",
    "float": "f32",
    "float32": "f32",
    "double": "f64",
    "float64": "f64",
    "int": "i32",
    "int32": "i32",
    "long": "i64",
    "int64": "i64",
    "short": "i16",
    "int16": "i16",
    "char": "i8",
    "int8": "i8",
    "uint8": "u8",
    "bool": "bool",
}


def _format_trace_shape(dims: Any, dtype: Any) -> str | None:
    """Render one operand as a TraceLens-style ``(d0,d1,...) <dtype>`` string.

    Args:
        dims: The operand dimensions (list/tuple of ints).
        dtype: The operand dtype token.

    Returns:
        The formatted shape string, or ``None`` for scalar/empty operands.
    """
    if not isinstance(dims, (list, tuple)) or not dims:
        return None
    try:
        body = ",".join(str(int(d)) for d in dims)
    except (TypeError, ValueError):
        return None
    shape = f"({body},)" if len(dims) == 1 else f"({body})"
    suffix = _TRACE_DTYPE_SUFFIX.get(str(dtype or "").strip().lower())
    return f"{shape} {suffix}" if suffix else shape


def _resolve_shapes_from_ops_unique_args_csv(
    perf_report_csv_dir: Path | str | None,
    row_matches: Callable[[str], bool],
) -> list[str]:
    """Recover operand shapes from matching ``ops_unique_args.csv`` rows.

    Parses ``Input Dims`` / ``Input type`` tuple-of-tuples into TraceLens-style
    shape strings (e.g. ``(15360,2048) bf16``), deduped in first-seen order.

    Args:
        perf_report_csv_dir: Directory containing ``ops_unique_args.csv``.
        row_matches: Predicate called with the normalized row name.

    Returns:
        The recovered operand shape strings, or ``[]`` when no trusted rows
        are available.
    """
    if not perf_report_csv_dir:
        return []
    csv_path = Path(perf_report_csv_dir) / "ops_unique_args.csv"
    if not csv_path.is_file():
        return []
    shapes: list[str] = []
    seen: set[str] = set()
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = str(row.get("name") or "").strip().lower()
                if not row_matches(name):
                    continue
                try:
                    dims = ast.literal_eval(str(row.get("Input Dims") or "").strip() or "()")
                    dtypes = ast.literal_eval(str(row.get("Input type") or "").strip() or "()")
                except (ValueError, SyntaxError):
                    continue
                if not isinstance(dims, (list, tuple)):
                    continue
                if not isinstance(dtypes, (list, tuple)):
                    dtypes = ()
                for i, operand in enumerate(dims):
                    dtype = dtypes[i] if i < len(dtypes) else ""
                    s = _format_trace_shape(operand, dtype)
                    if s and s not in seen:
                        seen.add(s)
                        shapes.append(s)
    except (OSError, csv.Error):
        return []
    return shapes


def _invocation_case_from_csv_row(row: dict[str, str]) -> dict[str, Any] | None:
    """Preserve one ops_unique_args row as an exact invocation case."""
    raw_dims = str(row.get("Input Dims") or "").strip()
    raw_types = str(row.get("Input type") or "").strip()
    raw_concrete = str(row.get("Concrete Inputs") or "").strip()
    try:
        dims = ast.literal_eval(raw_dims or "()")
        dtypes = ast.literal_eval(raw_types or "()")
    except (ValueError, SyntaxError):
        return None
    if not isinstance(dims, (list, tuple)):
        return None
    if not isinstance(dtypes, (list, tuple)):
        dtypes = ()

    operands: list[str] = []
    tensor_dtypes: list[str] = []
    for index, operand in enumerate(dims):
        dtype = dtypes[index] if index < len(dtypes) else ""
        rendered = _format_trace_shape(operand, dtype)
        if rendered:
            operands.append(rendered)
            tensor_dtypes.append(str(dtype or ""))

    raw_arg_spec = {
        "input_dims": raw_dims,
        "input_type": raw_types,
        "concrete_inputs": raw_concrete,
    }
    if not operands and not any(raw_arg_spec.values()):
        return None
    return {
        "operation": str(row.get("name") or ""),
        "input_shapes": ([{"call_num": 1, "shape": "<br>".join(operands)}] if operands else []),
        "input_dtypes": tensor_dtypes,
        "raw_arg_spec": raw_arg_spec,
    }


def _resolve_invocation_cases_from_ops_unique_args_csv(
    perf_report_csv_dir: Path | str | None,
    row_matches: Callable[[str], bool],
) -> list[dict[str, Any]]:
    """Return distinct CSV invocations without flattening argument boundaries."""
    if not perf_report_csv_dir:
        return []
    csv_path = Path(perf_report_csv_dir) / "ops_unique_args.csv"
    if not csv_path.is_file():
        return []

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with csv_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                name = str(row.get("name") or "").strip().lower()
                if not row_matches(name):
                    continue
                case = _invocation_case_from_csv_row(row)
                if case is None:
                    continue
                signature = json.dumps(
                    case,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                cases.append(case)
    except (OSError, csv.Error):
        return []
    return cases


def resolve_fused_moe_shapes_from_csv(
    perf_report_csv_dir: Path | str | None,
) -> list[str]:
    """Recover the fused-MoE expert-kernel operand shapes from ``ops_unique_args.csv``."""
    return _resolve_shapes_from_ops_unique_args_csv(
        perf_report_csv_dir,
        lambda name: _FUSED_MOE_KERNEL_MARKER in name,
    )


def resolve_fused_moe_invocation_cases_from_csv(
    perf_report_csv_dir: Path | str | None,
) -> list[dict[str, Any]]:
    """Recover every fused-MoE invocation row as an independent case."""
    return _resolve_invocation_cases_from_ops_unique_args_csv(
        perf_report_csv_dir,
        lambda name: _FUSED_MOE_KERNEL_MARKER in name,
    )


def resolve_shapes_from_csv_for_op(
    perf_report_csv_dir: Path | str | None,
    op_name: str,
) -> list[str]:
    """Recover operand shapes for a candidate by exact TraceLens op name."""
    target = str(op_name or "").strip().lower()
    if not target:
        return []
    return _resolve_shapes_from_ops_unique_args_csv(
        perf_report_csv_dir,
        lambda name: name == target,
    )


def resolve_invocation_cases_from_csv(
    perf_report_csv_dir: Path | str | None,
    op_name: str,
) -> list[dict[str, Any]]:
    """Recover every exact-name invocation row without merging its arguments."""
    target = str(op_name or "").strip().lower()
    if not target:
        return []
    return _resolve_invocation_cases_from_ops_unique_args_csv(
        perf_report_csv_dir,
        lambda name: name == target,
    )


def resolve_raw_arg_spec_from_csv(
    perf_report_csv_dir: Path | str | None,
    op_name: str,
) -> dict[str, str] | None:
    """Return the matched ``ops_unique_args.csv`` row's raw arg columns verbatim.

    Unlike :func:`resolve_shapes_from_csv_for_op` (which drops scalar/empty
    operands), this preserves the full ordered argument metadata exactly as the
    trace recorded it — ``Input Dims`` / ``Input type`` / ``Concrete Inputs`` —
    so the GEAK harness builder can reconstruct the real call signature
    (tensors + scalar args in order) instead of inferring it from tensor shapes
    alone. No parsing, no reshaping: the strings are forwarded as-is.

    Args:
        perf_report_csv_dir: Directory containing ``ops_unique_args.csv``.
        op_name: Exact TraceLens op name to match (case-insensitive).

    Returns:
        ``{"input_dims", "input_type", "concrete_inputs"}`` from the first
        matching row, or ``None`` when unavailable.
    """
    target = str(op_name or "").strip().lower()
    if not target or not perf_report_csv_dir:
        return None
    csv_path = Path(perf_report_csv_dir) / "ops_unique_args.csv"
    if not csv_path.is_file():
        return None
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("name") or "").strip().lower() != target:
                    continue
                spec = {
                    "input_dims": str(row.get("Input Dims") or "").strip(),
                    "input_type": str(row.get("Input type") or "").strip(),
                    "concrete_inputs": str(row.get("Concrete Inputs") or "").strip(),
                }
                return spec if any(spec.values()) else None
    except (OSError, csv.Error):
        return None
    return None


def _is_fused_moe_candidate(item: dict[str, Any]) -> bool:
    """Detect the Triton fused-MoE expert-kernel candidate.

    Matches by op name or the ``moe_fused`` category.

    Args:
        item: The candidate dict to test.

    Returns:
        ``True`` when the candidate is the fused-MoE expert kernel.
    """
    name = str(item.get("name") or "").lower()
    if _FUSED_MOE_KERNEL_MARKER in name:
        return True
    cat = str(item.get("tracelens_category") or "").strip().lower()
    return cat == "moe_fused" and "moe" in name


# High-GPU-time "other"-bucket candidate recovery (defense-in-depth). TraceLens
# emits no reasoning-candidate block for an op filed under "other", so a dominant
# editable kernel can land in neither hot_kernels nor skipped_kernels. This net
# recovers such a high-GPU-time op from the per-op ranking sidecars so it flows
# through _finalize_candidates -> classify_patchability. analysis.md stays primary.

_OTHER_BUCKET_MIN_GPU_PCT_ENV = "HYPERLOOM_OTHER_BUCKET_MIN_GPU_PCT"
_DEFAULT_OTHER_BUCKET_MIN_GPU_PCT = 10.0

# Per-op ranking sidecars in preference order, relative to the skill output dir.
_OPS_RANKING_CSV_RELPATHS = (
    "ops_summary.csv",
    "perf_report_csvs/ops_summary.csv",
    "perf_report_csvs/unified_perf_summary.csv",
    "unified_perf_summary.csv",
)
_OPS_RANKING_JSON_RELPATHS = (
    "priority_data.json",
    "perf_report_csvs/priority_data.json",
)

_RANK_NAME_KEYS = ("name", "operation", "op", "kernel", "kernel_name", "op_name")
_RANK_CATEGORY_KEYS = (
    # ``categories`` is the real ops_summary.csv column (a list-repr string).
    "op category",
    "op_category",
    "category",
    "categories",
    "tracelens_category",
)
# GPU-time column/field names in milliseconds (most specific first).
# ``total_direct_kernel_time_ms`` is the real ops_summary.csv per-op GPU-time column.
_RANK_TIME_MS_KEYS = (
    "gpu time total (ms)",
    "total gpu time (ms)",
    "gpu time (ms)",
    "self gpu time (ms)",
    "total_direct_kernel_time_ms",
    "total_direct_kernel_time_ms_sum",
    "gpu_time_ms",
    "gpu_time_total_ms",
    "duration (ms)",
    "dur (ms)",
    "time (ms)",
    "time_ms",
    "duration_ms",
)
_RANK_TIME_US_KEYS = (
    "gpu time (us)",
    "self gpu time (us)",
    "total_direct_kernel_time_sum",
    "total_direct_kernel_time_us",
    "gpu_time_us",
    "gpu_time_total_us",
    "duration_us",
    "dur (us)",
    "time (us)",
    "time_us",
    "dur",
)
_RANK_PCT_KEYS = (
    # ``percentage (%)`` is the real ops_summary.csv % column.
    "% gpu time",
    "gpu time %",
    "gpu %",
    "gpu_pct",
    "gpu_time_pct",
    "% of compute time",
    "% of compute",
    "%e2e",
    "% e2e",
    "percentage (%)",
    "percentage",
    "percent",
    "pct",
)


def _lower_keyed(row: dict) -> dict[str, Any]:
    """Return a copy of ``row`` with keys trimmed and lower-cased.

    Args:
        row: Source mapping (e.g. a CSV/JSON record).

    Returns:
        A new dict keyed by the normalized column names.
    """
    return {str(k).strip().lower(): v for k, v in row.items()}


def _first_present(low: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-empty value among candidate keys.

    Args:
        low: Lower-keyed record to read from.
        keys: Candidate keys, in priority order.

    Returns:
        The first present, non-empty value as a stripped string, or ``""``.
    """
    for k in keys:
        v = low.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _clean_category_label(raw: str) -> str:
    """Normalize a category cell to a bare label.

    The real ``ops_summary.csv`` stores the category as a Python list-repr
    string, e.g. ``['MoE_fused']`` or ``['GEMM', 'Reduce']`` — return the first
    element (``MoE_fused`` / ``GEMM``). Plain labels pass through unchanged.

    Args:
        raw: The raw category cell value.

    Returns:
        The bare category label.
    """
    s = str(raw or "").strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)) and val:
                return str(val[0]).strip()
            if isinstance(val, (list, tuple)):
                return ""
        except (ValueError, SyntaxError):
            s = s.strip("[]")
    return s.strip().strip("'\"").strip()


def _record_gpu_us(low: dict[str, Any]) -> float | None:
    """Extract GPU time in microseconds from a per-op ranking record.

    Args:
        low: A per-op ranking record with lower-cased keys.

    Returns:
        The GPU time in microseconds, or ``None`` when no time field resolves.
    """
    for k in _RANK_TIME_MS_KEYS:
        if k in low:
            val = safe_float(low[k], default=None, strip_percent=True, strip_commas=True)
            if val is not None:
                return val * 1000.0
    for k in _RANK_TIME_US_KEYS:
        if k in low:
            val = safe_float(low[k], default=None, strip_percent=True, strip_commas=True)
            if val is not None:
                return val
    return None


def _record_gpu_pct(low: dict[str, Any]) -> float | None:
    """Extract the GPU-time percentage from a per-op ranking record.

    Args:
        low: Lower-keyed ranking record.

    Returns:
        The GPU-time percentage, or ``None`` when no recognized key parses.
    """
    for k in _RANK_PCT_KEYS:
        if k in low:
            val = safe_float(low[k], default=None, strip_percent=True, strip_commas=True)
            if val is not None:
                return val
    return None


def _ranking_record(raw: dict) -> dict[str, Any] | None:
    """Normalize a raw ranking row into a standard candidate record.

    Args:
        raw: Raw CSV/JSON ranking row.

    Returns:
        A dict with ``name``, ``category``, ``gpu_us``, and ``gpu_pct``, or
        ``None`` when the row has no usable op name.
    """
    low = _lower_keyed(raw)
    name = _first_present(low, _RANK_NAME_KEYS)
    if not name:
        return None
    return {
        "name": name,
        "category": _clean_category_label(_first_present(low, _RANK_CATEGORY_KEYS)),
        "gpu_us": _record_gpu_us(low),
        "gpu_pct": _record_gpu_pct(low),
    }


def _load_ops_ranking_csv(path: Path) -> list[dict[str, Any]]:
    """Load per-op ranking records from a CSV sidecar.

    Args:
        path: Path to the CSV file.

    Returns:
        Normalized ranking records, or ``[]`` when the file is missing or
        cannot be read.
    """
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rec = _ranking_record(row)
                if rec is not None:
                    out.append(rec)
    except (OSError, csv.Error):
        return []
    return out


def _iter_json_ranking_records(data: Any) -> list[dict]:
    """Extract the list of ranking record dicts from parsed JSON.

    Accepts either a top-level list or a dict containing the records under one
    of several known keys.

    Args:
        data: Parsed JSON value.

    Returns:
        The list of dict records, or ``[]`` when none are found.
    """
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for key in (
            "findings",
            "operations",
            "ops",
            "priorities",
            "candidates",
            "kernels",
            "items",
            "rows",
        ):
            v = data.get(key)
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
    return []


def _load_ops_ranking_json(path: Path) -> list[dict[str, Any]]:
    """Load per-op ranking records from a JSON sidecar.

    Args:
        path: Path to the JSON file.

    Returns:
        Normalized ranking records, or ``[]`` when the file is missing or
        cannot be parsed.
    """
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[dict[str, Any]] = []
    for rec_raw in _iter_json_ranking_records(data):
        rec = _ranking_record(rec_raw)
        if rec is not None:
            out.append(rec)
    return out


def load_ops_ranking(
    skill_output_dir: Path | str | None,
) -> list[dict[str, Any]]:
    """Load a per-op GPU-time ranking from TraceLens sidecars (schema-tolerant).

    Returns ``[{name, category, gpu_us, gpu_pct}]`` from the first sidecar that
    yields rows (ops_summary.csv / unified_perf_summary.csv /
    priority_data.json, under the output dir or its ``perf_report_csvs/``).
    Used only by the ``other``-bucket recovery fallback — the contracted
    candidate source remains analysis.md.

    Args:
        skill_output_dir: The TraceLens skill output directory to scan.

    Returns:
        A list of ``{name, category, gpu_us, gpu_pct}`` rows, or ``[]`` when no
        sidecar is present or parseable.
    """
    if not skill_output_dir:
        return []
    root = Path(skill_output_dir)
    for rel in _OPS_RANKING_CSV_RELPATHS:
        rows = _load_ops_ranking_csv(root / rel)
        if rows:
            return rows
    for rel in _OPS_RANKING_JSON_RELPATHS:
        rows = _load_ops_ranking_json(root / rel)
        if rows:
            return rows
    return []


def _resolve_other_bucket_min_gpu_pct() -> float:
    """Return the minimum GPU-time percentage for other-bucket recovery.

    Returns:
        The value from ``HYPERLOOM_OTHER_BUCKET_MIN_GPU_PCT`` when set to a
        valid non-negative number, otherwise the default threshold.
    """
    raw = os.environ.get(_OTHER_BUCKET_MIN_GPU_PCT_ENV, "").strip()
    if raw:
        val = safe_float(raw, default=None, strip_percent=True, strip_commas=True)
        if val is not None and val >= 0:
            return val
    return _DEFAULT_OTHER_BUCKET_MIN_GPU_PCT


# Provenance for a candidate's ``gpu_pct``. Only ``e2e_window`` is comparable
# with the percentages ``parse_analysis_md`` lifts out of analysis.md; the other
# two are best-effort and are recorded so a consumer can tell them apart instead
# of silently ranking incomparable numbers against each other.
GPU_PCT_BASIS_E2E = "e2e_window"
GPU_PCT_BASIS_SIDECAR = "sidecar_reported"
GPU_PCT_BASIS_LISTED_OPS = "listed_ops_sum"


def recover_other_bucket_candidates(
    skill_output_dir: Path | str | None,
    existing_candidates: list[dict[str, Any]],
    *,
    top_k: int = 10,
    min_gpu_pct: float | None = None,
    total_window_us: float | None = None,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Recover HIGH-GPU-time ops missing from analysis.md (any category).

    Defense-in-depth fallback for the candidate-extraction gap: surfaces ops
    that have no analysis.md reasoning-candidate block (so ``parse_analysis_md``
    dropped them) as raw candidates, ranked by per-op sidecar GPU time, so each
    flows through ``_finalize_candidates`` -> ``classify_patchability``.

    Any high-GPU-time op missing from ``existing_candidates`` is eligible (e.g.
    ``MoE_fused`` whose roofline is null). The patchability gate downstream still
    rejects vendor / native ops, so widening the net does not route non-patchable
    kernels to GEAK.

    Fires for ops that are (a) absent from ``existing_candidates`` by name and
    (b) at or above ``min_gpu_pct`` of total GPU time.

    Whenever ``total_window_us`` is known, ``gpu_pct`` is normalized against it
    -- the same denominator ``parse_analysis_md`` uses -- so both candidate
    sources stay comparable inside one ranked ``hot_kernels[]``. The sidecar's
    own percentage column is preferred only as a fallback, because its
    denominator is not the trace wall span: ``ops_summary.csv`` normalizes
    against total *op* time, which excludes collectives, so on a comm-dominated
    trace its percentages run an order of magnitude above the end-to-end share.
    Mixing the two bases in one field silently produced unrankable candidate
    lists (a fallback op at "23.5% GPU" outranking an analysis.md op at "4.7%
    GPU" when the true e2e shares were 0.5% and 4.7%). The basis actually used
    is recorded on each candidate as ``gpu_pct_basis``.

    Note this bypasses rather than repairs the ambiguity: ``_RANK_PCT_KEYS``
    still maps both ``% of compute time`` and ``%e2e`` onto one ``gpu_pct``, so
    a run with no ``gpu_timeline.csv`` to supply ``total_window_us`` falls back
    to whatever the sidecar reported and carries the original mixed-basis
    hazard -- now labelled ``sidecar_reported`` rather than silent. Splitting
    those aliases into separate fields is the real repair and is not done here.

    Args:
        skill_output_dir: The TraceLens skill output directory to scan.
        existing_candidates: Candidates already extracted from analysis.md.
        top_k: Maximum number of recovered candidates to return.
        min_gpu_pct: Minimum GPU-time percentage to qualify; defaults to the
            ``HYPERLOOM_OTHER_BUCKET_MIN_GPU_PCT`` env value (10%). Compared
            against the sidecar's own share, not against ``gpu_pct``, so
            admission is unchanged by the reporting basis (see
            ``_admission_pct``).
        total_window_us: Wall span of the analyzed trace, in microseconds. When
            omitted or non-positive the function falls back to the sidecar's
            reported percentage, then to a share of the listed ops, and marks
            the candidates accordingly.
        log: Optional logging callable for diagnostics.

    Returns:
        The recovered candidate dicts, or ``[]`` when no sidecar is available
        or nothing qualifies (so analysis.md stays primary).
    """
    ranking = load_ops_ranking(skill_output_dir)
    if not ranking:
        return []
    if min_gpu_pct is None:
        min_gpu_pct = _resolve_other_bucket_min_gpu_pct()

    have = {str(c.get("name") or "").strip().lower() for c in existing_candidates if isinstance(c, dict)}
    listed_ops_us = sum(r["gpu_us"] for r in ranking if r.get("gpu_us") is not None) or 0.0
    window_us = float(total_window_us) if total_window_us is not None and total_window_us > 0 else 0.0

    def _admission_pct(rec: dict[str, Any]) -> float | None:
        """Return the share ``min_gpu_pct`` has always been compared against.

        Deliberately the pre-existing quantity -- the sidecar's own percentage,
        else the record's share of the rows we can see -- and NOT the e2e share
        below. ``min_gpu_pct`` is a tuned constant (10%) whose meaning is "a
        large slice of the GPU work we know about"; re-pointing it at the wall
        span would silently raise the bar by the reciprocal of the compute
        share, and on an ordinary compute-bound trace (60% compute, work spread
        over eight kernels) that takes this whole recovery path from eight
        candidates to zero. Admission stays on the old basis; only the number we
        publish moves.

        Args:
            rec: A ranking record.

        Returns:
            The admission share, or ``None`` when the record carries neither a
            reported percentage nor a usable GPU time.
        """
        if rec.get("gpu_pct") is not None:
            return float(rec["gpu_pct"])
        gpu_us = rec.get("gpu_us")
        if listed_ops_us > 0 and gpu_us is not None:
            return gpu_us / listed_ops_us * 100.0
        return None

    def _reported_pct(rec: dict[str, Any]) -> tuple[float, str] | None:
        """Return the percentage to publish and the basis it was computed on.

        Args:
            rec: A ranking record.

        Returns:
            A ``(percentage, basis)`` pair, or ``None`` when the record carries
            neither a usable GPU time nor a reported percentage.
        """
        gpu_us = rec.get("gpu_us")
        if window_us > 0 and gpu_us is not None:
            return gpu_us / window_us * 100.0, GPU_PCT_BASIS_E2E
        # No trace wall span available (e.g. a JSON sidecar with no
        # gpu_timeline.csv beside it): fall back to whatever the sidecar
        # reported, then to a share of the rows we can see.
        if rec.get("gpu_pct") is not None:
            return float(rec["gpu_pct"]), GPU_PCT_BASIS_SIDECAR
        if listed_ops_us > 0 and gpu_us is not None:
            return gpu_us / listed_ops_us * 100.0, GPU_PCT_BASIS_LISTED_OPS
        return None

    qualifying: list[tuple[float, str, dict[str, Any]]] = []
    for rec in ranking:
        name_l = str(rec.get("name") or "").strip().lower()
        if not name_l or name_l in have:
            continue
        # Gate on any high-GPU-time op missing from existing candidates.
        admission = _admission_pct(rec)
        if admission is None or admission < min_gpu_pct:
            continue
        scored = _reported_pct(rec)
        if scored is None:
            continue
        qualifying.append((scored[0], scored[1], rec))

    if not qualifying:
        return []
    qualifying.sort(key=lambda t: t[0], reverse=True)

    out: list[dict[str, Any]] = []
    for pct, pct_basis, rec in qualifying[: max(1, top_k)]:
        gpu_us = rec.get("gpu_us")
        out.append(
            {
                "name": rec["name"],
                "duration_us": float(gpu_us) if gpu_us is not None else 0.0,
                "call_count": 0,
                "source_file": "",
                "source_type": "unknown",
                "shapes": [],
                "tracelens_category": rec.get("category") or "other",
                "gpu_pct": round(pct, 3),
                "gpu_pct_basis": pct_basis,
                "candidate_source": "other_bucket_fallback",
            }
        )
        if log is not None:
            log(
                f"candidate recovery fallback (#514/#515): recovered "
                f"high-GPU-time op {rec['name']!r} (~{pct:.1f}% GPU time, "
                f"basis={pct_basis}, "
                f"category={rec.get('category') or 'other'!r}) that has no "
                f"analysis.md reasoning-candidate block; routing through "
                f"classify_patchability so a reusable native kernel still "
                f"reaches GEAK"
            )
    # One batch on two bases is the state this labelling exists to prevent: the
    # percentages are then not comparable with each other, let alone with the
    # analysis.md candidates they get ranked against. It is reachable when only
    # some ranking rows carry an absolute GPU time, so say so rather than leave
    # the evidence buried in a field nothing reads yet.
    bases = {c["gpu_pct_basis"] for c in out}
    if len(bases) > 1 and log is not None:
        log(
            f"candidate recovery fallback: gpu_pct spans {sorted(bases)} in one "
            f"batch, so the recovered candidates are not mutually comparable; "
            f"rank them on gpu_pct_basis-matched subsets only"
        )
    return out


# A resolved source_file always carries a source extension, whether absolute
# ("/sgl-workspace/.../fused_moe.py"), package-relative ("sgl_kernel/moe.py"), or
# TraceLens' frame form ("path.py(124): fn"). Producer placeholders never do --
# "Not found", "N/A", "AITER (vendor)", "unknown".
# Derived from PATH_SHAPED_EXTENSIONS, not from the grep admission list: this
# gate only answers "is this a path or a placeholder". Longest first -- the
# alternation is left-biased, and while ``\b`` already forces a retry on ".hpp"
# vs ".h", ordering makes the intent explicit.
_SOURCE_EXT_RE = re.compile(
    r"\.(?:"
    + "|".join(re.escape(ext.lstrip(".")) for ext in sorted(PATH_SHAPED_EXTENSIONS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def looks_like_source_path(value: str) -> bool:
    """Return whether ``value`` has the shape of a source-file path.

    Args:
        value (str): Candidate ``source_file`` value.

    Returns:
        bool: ``True`` when it carries a recognized source extension.
    """
    # Normalize separators first, matching is_torch_dispatch_shim_source: a
    # Windows-style path is still a path, and rejecting it would zero a real
    # source_file as if it were a placeholder.
    text = (value or "").strip().replace("\\", "/")
    return bool(_SOURCE_EXT_RE.search(text))


def reject_non_path_source(item: dict[str, Any]) -> bool:
    """Zero a ``source_file`` that is a producer placeholder, not a path.

    A real source_file always looks like a path (absolute, or package-relative
    such as "sgl_kernel/moe.py"); a bare word is what the producer wrote for
    "unresolved". Admitting one is how classify_patchability ends up reporting
    the nonsensical "source not under a reusable framework root: Not found".
    Keyed on shape rather than on-disk presence so an analysis host that lacks
    the serving container's filesystem still classifies normally.

    Must run before the trace/grep tiers: they are gated on an empty
    ``source_file``, so a surviving placeholder silently skips them.

    Returns:
        True when a placeholder was rejected.
    """
    source_file = str(item.get("source_file") or "").strip()
    if not source_file or looks_like_source_path(source_file):
        return False
    item["source_file_rejected"] = source_file
    item["source_file"] = ""
    item["source_resolution_method"] = "rejected_non_path_sentinel"
    return True


def _stamp_candidate_metadata(item: dict[str, Any], op_cat_map: dict[str, str] | None) -> None:
    """Stamp routing, backend, and category metadata onto a finalized candidate."""
    # Placeholders are already rejected by reject_non_path_source() at the top of
    # _finalize_candidates -- early enough for the trace and grep tiers to run.
    # What is left to check here is the resolved path's presence on disk.
    source_file = str(item.get("source_file") or "").strip()
    if source_file and not os.path.isfile(source_file):
        # Path-shaped but absent: keep it (the resolution may target the serving
        # container) and leave a breadcrumb for triage.
        item["source_file_missing_on_disk"] = True
    reusable, skip_reason = classify_patchability(item)
    item["reusable_native_kernel"] = reusable
    item["skip_reason"] = skip_reason
    playbook = match_vendor_operator_playbook(item)
    if playbook is not None:
        item["patch_strategy"] = "vendor_playbook"
        item["vendor_operator_playbook"] = playbook
        item["vendor_playbook_role"] = playbook.get("role", "")
        # kernel_optimization.py's CLI gates on a non-empty, path-shaped
        # source_file before it will dispatch to any backend; a vendor
        # playbook candidate has no rewritable device source, so point that
        # field at the task bundle's anchor file instead of leaving it
        # empty (which would otherwise fall through as "missing_native_source").
        #
        # The anchor also overrides whatever the grep tier guessed, and does so
        # whether or not the guess was right. A registry match is a curated
        # statement that this operator is tuned through a task bundle, so the
        # device source is not the file a backend should be handed even when
        # the guess found it -- and the guess can just as easily be a same-word
        # collision (``mori::EpDispatchCombineOp::dispatch`` reduces to the
        # keyword "dispatch" and lands on an unrelated vendor header).
        # Unconditional, and safe to be: the registry refuses an entry with no
        # ``kernel_anchor``, so ``resolve_kernel_anchor_path`` cannot come back
        # empty for a playbook that matched. Guarding it here instead cost two
        # commits and produced a real deadlock -- the guard's marker gated on
        # ``source_file``, so a row with neither anchor nor path read as
        # anchor-backed, went into ``protected_ids``, and the row most in need of
        # the review was the one refused it. The shape it defended against does
        # not occur in the data; the shape it created did.
        anchor = resolve_kernel_anchor_path(playbook)
        if source_file and source_file != anchor:
            # Keep the displaced path for triage: an unconditional override is
            # only auditable if the value it replaced is still recorded.
            item["source_file_superseded_by_playbook"] = source_file
        item["source_file"] = anchor
    item["benchmark_files"] = find_benchmark_files(
        item["name"], item.get("kernel_repo", ""), item.get("source_file", "")
    )
    item["is_multigpu"] = is_multigpu_kernel(item["name"], item.get("source_file", ""))
    item["num_gpus_recommended"] = 2 if item["is_multigpu"] else 1
    item["recommended_backends"] = recommend_backends(item)
    item["optimization_notes"] = build_notes(item)
    if op_cat_map and not str(item.get("tracelens_category") or "").strip():
        csv_cat = op_cat_map.get(str(item.get("name") or ""))
        if csv_cat:
            item["tracelens_category"] = csv_cat
    item["kernel_category"] = derive_kernel_category(item)
    item.setdefault("source_path", item.get("source_file", ""))


#: Populated once per run from the CLI args so both model tiers see the same
#: serving configuration. Empty when the analysis runs without that context.
_RUNTIME_CONTEXT: dict[str, Any] = {}


def set_runtime_context(
    *,
    model_path: str = "",
    server_args: str = "",
    framework: str = "",
    precision: str = "",
) -> None:
    """Record the serving configuration the resolution tiers should reason with.

    Which file implements a kernel depends on the running configuration -- the
    same MoE operator dispatches differently under ``--moe-runner-backend
    triton`` and ``aiter``. Set once at analysis start; both tiers read it.
    """
    _RUNTIME_CONTEXT["model_path"] = str(model_path or "")
    _RUNTIME_CONTEXT["server_args"] = str(server_args or "")
    _RUNTIME_CONTEXT["framework"] = str(framework or "")
    _RUNTIME_CONTEXT["precision"] = str(precision or "")


def _runtime_server_args_from_config(config_path: str) -> str:
    """Read materialized EXTRA_*_ARGS without sending the config to a model."""
    if not str(config_path or "").strip():
        return ""
    try:
        import yaml  # type: ignore[import-untyped]  # noqa: PLC0415

        payload = yaml.safe_load(Path(config_path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - runtime context is advisory
        log.warning("could not read source runtime config: %s", type(exc).__name__)
        return ""
    benchmark = payload.get("benchmark") if isinstance(payload, dict) else None
    envs = benchmark.get("envs") if isinstance(benchmark, dict) else None
    if not isinstance(envs, dict):
        return ""
    return " ".join(
        str(value).strip()
        for key, value in envs.items()
        if str(key).startswith("EXTRA_") and str(key).endswith("_ARGS") and str(value).strip()
    )


def _source_context_block() -> str:
    """Render the shared runtime context for a model tier, or "" when absent."""
    try:
        from _llm_source_context import build_context_block  # noqa: PLC0415

        return build_context_block(
            model_path=_RUNTIME_CONTEXT.get("model_path") or "",
            server_args=_RUNTIME_CONTEXT.get("server_args") or "",
            framework_roots=kernel_search_roots(),
            framework=_RUNTIME_CONTEXT.get("framework") or "",
            precision=_RUNTIME_CONTEXT.get("precision") or "",
        )
    except Exception as exc:  # noqa: BLE001 - context is an aid, never required
        log.warning("could not build source-resolution context: %r", exc)
        return ""


def _forward_to_log(message: str) -> None:
    """Adapter so the resolution tiers' own diagnostics reach the logger.

    Both tiers accept a ``log=`` callable and emit detailed progress through it.
    Without this the messages are produced and discarded.
    """
    log.info("%s", message)


def _append_resolution_reason(item: dict[str, Any], reason: str) -> None:
    """Append a distinct resolution outcome without masking an earlier tier."""
    current = str(item.get("source_resolution_reason") or "").strip()
    if not current:
        item["source_resolution_reason"] = reason
    elif reason not in current:
        item["source_resolution_reason"] = f"{current}; {reason}"


@functools.lru_cache(maxsize=64)
def _package_parent_dir(package: str) -> str:
    """Directory holding ``package``'s own directory, resolved at runtime.

    Uses ``find_spec`` so the package is located without importing it (the same
    approach ``_bypass_source_resolver`` takes for aiter). Returns ``""`` when
    the name is not a package on this interpreter's path.
    """
    if not package or not package.isidentifier():
        return ""
    try:
        spec = importlib.util.find_spec(package)
    except (AttributeError, ImportError, ValueError):
        return ""
    if spec is None:
        return ""
    for location in list(getattr(spec, "submodule_search_locations", None) or []):
        parent = os.path.dirname(str(location).rstrip("/"))
        if parent:
            return parent
    origin = str(getattr(spec, "origin", "") or "")
    return os.path.dirname(os.path.dirname(origin)) if origin else ""


def absolutize_launcher_path(path: str) -> str:
    """Best-effort absolute path for a trace-relative launcher frame.

    torch profiler records a frame path relative to the ``sys.path`` entry the
    module was imported from (``aiter/dist/x.py``), whereas patchability keys on
    an absolute framework root. The relative path already starts with the package
    name, so it joins against each package's *parent*.

    Args:
        path (str): Launcher path from the trace, absolute or relative.

    Returns:
        str: The absolutized path when one exists on disk, else ``path``.
    """
    if not path or os.path.isabs(path):
        return path
    # The relative path starts with the package name ("vllm/models/..."), so
    # locate that package where it actually lives. A pinned list cannot do this:
    # the same package sits under /sgl-workspace in the serving image and under
    # dist-packages on a wheel install, and the Python version moves too.
    parent = _package_parent_dir(path.split("/", 1)[0])
    if parent:
        candidate = os.path.join(parent, path)
        if os.path.isfile(candidate):
            return candidate
    # Fall back to the pinned checkout layouts (editable installs whose package
    # dir is not importable from this process).
    for inner_root in _PACKAGE_INNER_ROOTS:
        candidate = os.path.join(os.path.dirname(inner_root), path)
        if os.path.isfile(candidate):
            return candidate
    return path


def _source_paths_match(left: str, right: str) -> bool:
    """Return whether two source paths identify the same normalized file."""
    if not left or not right:
        return False
    left_path = os.path.normcase(os.path.normpath(left))
    right_path = os.path.normcase(os.path.normpath(right))
    if left_path == right_path:
        return True
    if not os.path.isabs(left_path) or not os.path.isabs(right_path):
        return False
    return os.path.realpath(left_path) == os.path.realpath(right_path)


def _trace_launcher_key(item: dict[str, Any]) -> str:
    """Device kernel symbol to look this candidate up by in the trace."""
    name = str(item.get("device_kernel_name") or "").strip()
    if not name:
        name = _normalize_profiler_op_name(str(item.get("name") or ""))
    return name.strip()


def _resolve_trace_launchers(
    candidates: list[dict[str, Any]],
    trace_files: list[Path] | None,
    log_path: Path | str | None = None,
) -> dict[str, Any]:
    """Batch-resolve launcher frames for candidates that still lack a source.

    One trace scan for the whole candidate list; returns ``{}`` when there is
    nothing to look up or the trace cannot be read, so the grep fallback stays
    in charge.
    """
    if not trace_files:
        return {}
    wanted = {
        key
        for item in candidates
        # Shape, not truthiness: an upstream placeholder ("AITER (vendor)",
        # "Not found") is a non-empty string, and treating it as a resolved
        # source would skip the very candidates this resolver exists for.
        if not looks_like_source_path(str(item.get("source_file") or ""))
        and (key := _trace_launcher_key(item))
        and not is_runtime_api_name(key)
    }
    if not wanted:
        return {}
    try:
        from _trace_launcher_resolver import resolve_launchers_from_trace  # noqa: PLC0415

        file_errors: list[str] = []
        found = resolve_launchers_from_trace(
            [Path(p) for p in trace_files],
            wanted,
            log=_forward_to_log,
            file_errors=file_errors,
        )
        if file_errors:
            # A per-file failure previously surfaced only as "0 resolved",
            # which reads the same as "no candidate needed a launcher".
            reason = f"trace_resolver_error: {'; '.join(file_errors[:2])}"
            log.warning("trace launcher tier hit %d unreadable file(s)", len(file_errors))
            for item in candidates:
                if _trace_launcher_key(item) in wanted:
                    item.setdefault("source_resolution_reason", reason)
        summary = f"trace launcher tier: resolved {len(found)}/{len(wanted)} unresolved candidate(s)"
        log.info("%s", summary)
        # Also to the run log: that is where an operator looks when candidates
        # come out unresolved, and a logger record does not survive there.
        if log_path:
            append_log(log_path, summary)
        return found
    except Exception as exc:  # noqa: BLE001 - advisory resolution, never fatal
        # Stay fail-soft into the grep tier, but leave a trail. A bare ``{}``
        # is indistinguishable from "no candidate needed a launcher", which
        # hides unreadable traces, import errors and parse failures alike.
        reason = f"trace_resolver_error: {type(exc).__name__}: {exc}"
        log.warning(
            "trace launcher resolution failed for %d candidate(s) (%s); falling back to grep",
            len(wanted),
            reason,
        )
        for item in candidates:
            if _trace_launcher_key(item) in wanted:
                item.setdefault("source_resolution_reason", reason)
        return {}


def _finalize_candidates(
    top: list[dict[str, Any]],
    *,
    total_dur: float | None = None,
    perf_report_csv_dir: Path | str | None = None,
    framework: str | None = None,
    trace_files: list[Path] | None = None,
    log_path: Path | str | None = None,
    source_resolution_out: Path | str | None = None,
    model_name: str = "",
) -> list[dict[str, Any]]:
    """Apply shared post-processing to parsed candidate rows.

    Resolves source files, promotes pybind/launcher shims, recommends
    backends, classifies patchability, and attaches notes; mutates ``top`` in
    place.

    Args:
        top: The parsed candidate rows to finalize (mutated in place).
        total_dur: Total GPU duration for percentage computation; summed from
            ``top`` when omitted.
        perf_report_csv_dir: Optional CSV directory used to populate each
            item's ``tracelens_category``.
        framework: Explicit trace framework, threaded into the resolver so a
            dual-framework image routes to the correct container's source. Only
            ``vllm``/``sglang`` steer routing; other values fall back to on-disk
            presence then default ordering.
        trace_files: Optional raw trace files. When given, Python launcher
            frames are retained as evidence and accepted as source only when
            name grep independently resolves the same file.
        model_name: Runtime model identity recorded in the resolution artifact.

    Returns:
        The finalized candidate list (the same ``top`` object).
    """
    op_cat_map = load_op_category_map(perf_report_csv_dir) if perf_report_csv_dir is not None else {}
    op_dom_map = load_op_dominant_kernel_map(perf_report_csv_dir) if perf_report_csv_dir is not None else {}
    # Dict-first: resolve each op to its editable .cu and expand composite
    # fan-out into one candidate per sub-kernel before finalizing.
    top = _expand_op_fanout(top, framework=framework, op_dominant_kernel=op_dom_map)
    # Drop upstream placeholders up front. They are non-empty strings, so every
    # later "did we already resolve this?" check would read them as a resolved
    # source and skip the resolution tiers entirely.
    for item in top:
        if isinstance(item, dict):
            reject_non_path_source(item)
    trace_launchers = _resolve_trace_launchers(top, trace_files, log_path=log_path)
    sum_dur = total_dur if total_dur is not None else sum(it.get("duration_us", 0.0) for it in top)
    sum_dur = sum_dur or 1.0
    # Recover the fused-MoE expert kernel's operand shapes (its trace event carries
    # no Input Dims) once from the sidecar and graft onto candidates lacking shapes.
    _fused_moe_shapes: list[str] | None = None
    _fused_moe_invocation_cases: list[dict[str, Any]] | None = None
    for idx, item in enumerate(top, 1):
        item.pop("_extracted_source_checked", None)
        item.setdefault("source_file", "")
        item.setdefault("source_type", "unknown")
        item.setdefault("shapes", [])
        had_observed_shapes = bool(item.get("shapes"))
        operation = str(item.get("name") or "")
        if _is_fused_moe_candidate(item):
            if _fused_moe_shapes is None:
                _fused_moe_shapes = resolve_fused_moe_shapes_from_csv(perf_report_csv_dir)
            if _fused_moe_invocation_cases is None:
                _fused_moe_invocation_cases = resolve_fused_moe_invocation_cases_from_csv(perf_report_csv_dir)
            invocation_cases = list(_fused_moe_invocation_cases or [])
            if not item.get("shapes") and _fused_moe_shapes:
                item["shapes"] = list(_fused_moe_shapes)
                item["shape_provenance"] = "torch_trace"
        else:
            invocation_cases = resolve_invocation_cases_from_csv(
                perf_report_csv_dir,
                operation,
            )
        if not item.get("shapes"):
            csv_shapes = resolve_shapes_from_csv_for_op(
                perf_report_csv_dir,
                operation,
            )
            if csv_shapes:
                item["shapes"] = csv_shapes
                item["shape_provenance"] = "torch_trace"
        if invocation_cases:
            if had_observed_shapes:
                invocation_cases = [
                    {
                        "operation": operation,
                        "input_shapes": (item.get("input_shapes") or item.get("shapes") or []),
                        "input_dtypes": item.get("input_dtypes") or [],
                        "raw_arg_spec": item.get("raw_arg_spec") or {},
                    },
                    *invocation_cases,
                ]
            item["invocation_cases"] = invocation_cases
        # Mark trace-extracted shapes so the dispatch-time validator knows their provenance.
        if item.get("shapes"):
            item.setdefault("shape_provenance", "torch_trace")
        # Forward the full ordered arg metadata so GEAK's harness builder can
        # reconstruct the exact call signature.
        if not item.get("raw_arg_spec"):
            raw_spec = (
                invocation_cases[0].get("raw_arg_spec")
                if invocation_cases
                else resolve_raw_arg_spec_from_csv(
                    perf_report_csv_dir,
                    str(item.get("name") or ""),
                )
            )
            if raw_spec:
                item["raw_arg_spec"] = raw_spec
        item["kernel_id"] = f"k{idx:03d}"
        if not item.get("gpu_pct"):
            item["gpu_pct"] = round(item["duration_us"] / sum_dur * 100.0, 3)
        item["duration_us"] = round(item["duration_us"], 3)
        # Dict-first source resolution. A routable verdict overrides source_file
        # with the curated .cu; a non-rewritable verdict is stamped; a miss or
        # unresolved route leaves the .py-launcher pipeline below intact.
        res = item.pop("_op_resolution", None)
        if res is not None and res.is_routable:
            res.apply_to(item)
            item["kernel_repo"] = find_repo_root(item.get("source_file", ""))
            item["source_type"] = source_type_for(item["name"], item.get("source_file", ""))
            item["runtime_generated_kernel"] = False
            _stamp_candidate_metadata(item, op_cat_map)
            continue
        if res is not None and res.status in {"non_rewritable", "no_kernel"}:
            # Curated verdict: not rewritable. Keep the .py launcher as context.
            res.stamp_onto(item)
        # Legacy fallback resolution runs ONLY for a dictionary miss or an
        # unresolved dispatch. An in-dict non-rewritable verdict is authoritative,
        # so we keep its .py launcher as context and do NOT grep/promote.
        if res is None or res.status == "unresolved":
            frame = None
            trace_source = ""
            if not item.get("source_file"):
                frame = trace_launchers.get(_trace_launcher_key(item))
                if frame is not None:
                    trace_source = absolutize_launcher_path(frame.source_file)
                    item["trace_launcher_file"] = trace_source
                    item["trace_launcher_line"] = frame.line
                    item["trace_launcher_function"] = frame.function
            if not item.get("source_file"):
                grep_source = locate_source_via_grep(item["name"])
                if grep_source:
                    item["source_file"] = grep_source
                    if frame is not None and _source_paths_match(trace_source, grep_source):
                        item["source_line"] = frame.line
                        item["source_function"] = frame.function
                        item["source_resolution_method"] = "trace_python_stack"
                        _append_resolution_reason(
                            item,
                            "trace launcher corroborated by name grep",
                        )
                    else:
                        item.pop("source_line", None)
                        item.pop("source_function", None)
                        item["source_resolution_method"] = getattr(_KSC, "METHOD_GREP", "name_grep")
                        if trace_source:
                            _append_resolution_reason(
                                item,
                                f"trace launcher differs from grep source: {trace_source}",
                            )
                elif trace_source:
                    _append_resolution_reason(
                        item,
                        f"trace launcher unconfirmed by name grep: {trace_source}",
                    )
            # An unresolved candidate stops here. The agent review pass runs
            # once over the finished table rather than per kernel, so it can
            # weigh a blank against the rest of the evidence instead of
            # guessing from a symbol alone.
            # Promote a tiny pybind shim TU to the real device code.
            item["kernel_repo"] = find_repo_root(item.get("source_file", ""))
            item["source_file"] = upgrade_pybind_shim_source(
                item.get("source_file", ""), item["name"], item.get("kernel_repo", "")
            )
            # Re-resolve repo in case the upgraded path lives in a different repo.
            item["kernel_repo"] = find_repo_root(item.get("source_file", "")) or item["kernel_repo"]
            item["source_type"] = source_type_for(item["name"], item.get("source_file", ""))
            # FlyDSL pseudo-ops carry no real source_file; inject the real FlyDSL
            # MoE kernel source before the patchability gate.
            if item["source_type"] == "flydsl":
                _sf = str(item.get("source_file") or "").strip()
                if (not _sf) or (not os.path.isfile(_sf)):
                    _fb = _resolve_flydsl_source_fallback()
                    if _fb:
                        item["source_file"] = _fb
                        item["kernel_repo"] = find_repo_root(_fb) or item.get("kernel_repo", "")
                        item["flydsl_source_from_fallback"] = True
        else:
            # In-dict non-routable verdict: keep the launcher as context.
            item["kernel_repo"] = find_repo_root(item.get("source_file", ""))
            item["source_type"] = source_type_for(item["name"], item.get("source_file", ""))
        # Downgrade thin dispatch wrappers to vendor_binary so recommend_backends drops them.
        if item["source_type"] != "vendor_binary" and is_vendor_dispatch_wrapper(
            item["name"], item.get("source_file", "")
        ):
            item["source_type"] = "vendor_binary"
            item["vendor_dispatch_wrapper"] = True
        item["runtime_generated_kernel"] = is_runtime_generated_kernel(item["name"], item.get("source_file", ""))
        _stamp_candidate_metadata(item, op_cat_map)
    _apply_vendor_operator_playbook_grouping(top)
    if source_resolution_out and _KSC is not None:
        write_source_resolution_artifact(
            top,
            source_resolution_out,
            framework=framework or "",
            model_name=model_name,
            log_path=log_path,
        )
    return top


def _apply_vendor_operator_playbook_grouping(top: list[dict[str, Any]]) -> None:
    """Sum GPU share across a vendor playbook's sibling roles; mutates ``top``.

    mori's dispatch and combine are two separate kernel launches under the
    hood, but this deliberately invokes them as **one** Forge task/session
    (see KernelForge PR #88): each is gated on the *sum* of dispatch's +
    combine's ``gpu_pct``, since together they are one logical round trip.
    Each member keeps its own candidate entry (so either one can be picked as
    the ``--kernel-id`` the orchestrator dispatches); ``forge_submit``'s
    vendor-playbook path de-duplicates so only one forge-loop session actually
    runs per group per analysis session.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in top:
        if not isinstance(item, dict) or item.get("patch_strategy") != "vendor_playbook":
            continue
        playbook = item.get("vendor_operator_playbook")
        group_id = str(playbook.get("id") or "") if isinstance(playbook, dict) else ""
        if not group_id:
            continue
        groups.setdefault(group_id, []).append(item)
    for group_id, members in groups.items():
        aggregate = round(sum(float(m.get("gpu_pct") or 0.0) for m in members), 3)
        member_ids = [str(m.get("kernel_id") or "") for m in members]
        floor: float | None = None
        for member in members:
            playbook = member.get("vendor_operator_playbook")
            if isinstance(playbook, dict) and playbook.get("min_gpu_pct_floor") is not None:
                try:
                    floor = float(playbook["min_gpu_pct_floor"])
                except (TypeError, ValueError):
                    floor = None
                break
        for member in members:
            member["vendor_playbook_group_id"] = group_id
            # Namespaced (not bare "aggregate_gpu_pct"): that name is already
            # an existing task_group-level concept (the sum of GPU% across a
            # task_group's rows -- see tracelens_skill_runner.py and
            # _bypass_report.py), stamped on task_group dicts, not candidate
            # rows. Candidate rows don't carry it today, so reusing the name
            # here is currently harmless, but any future code that flattens
            # a task_group onto its candidate rows would silently change
            # every ordinary group's gating behavior the moment it did (PR
            # #1191 review finding #7).
            member["vendor_playbook_aggregate_gpu_pct"] = aggregate
            member["vendor_playbook_group_kernel_ids"] = list(member_ids)
            if floor is not None:
                # Consumed by effective_hot_kernel_min_gpu_pct() in the
                # orchestrator's hot-kernel gate (_kernel_decisions.py /
                # request_handlers.py) so a heavier forge-loop-session
                # playbook isn't dispatched below its own floor even when
                # HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT is loosened elsewhere.
                member["vendor_playbook_min_gpu_pct_floor"] = floor


def _candidate_resolution_method(item: dict[str, Any]) -> str:
    """Map a finalized candidate onto the artifact's method vocabulary.

    The candidate carries the method only when a tier stamped one; a path that
    arrived from grep has none, and an absent path means nothing resolved it.
    """
    stamped = str(item.get("source_resolution_method") or "").strip()
    if stamped in getattr(_KSC, "KNOWN_METHODS", frozenset()):
        return stamped
    if str(item.get("source_file") or "").strip():
        # No tier claimed it but a path is present: grep is the only tier that
        # resolves without stamping.
        return getattr(_KSC, "METHOD_GREP", "name_grep")
    return getattr(_KSC, "METHOD_UNRESOLVED", "unresolved")


def build_source_resolution_entries(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project finalized candidates onto source-resolution entries."""
    entries: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        method = _candidate_resolution_method(item)
        line = item.get("source_line")
        entry = _KSC.make_entry(
            kernel_id=str(item.get("kernel_id") or ""),
            name=str(item.get("name") or ""),
            gpu_pct=float(item.get("gpu_pct") or 0.0),
            source_file=str(item.get("source_file") or ""),
            source_line=int(line) if isinstance(line, (int, float)) else None,
            source_function=str(item.get("source_function") or ""),
            method=method,
            confidence=item.get("source_resolution_confidence"),
            reason=str(item.get("source_resolution_reason") or ""),
            rejected_value=str(item.get("source_file_rejected") or ""),
            # The names ``apply_revisions`` actually writes. The
            # ``source_resolution_previous_*`` spelling read here before has no
            # producer anywhere in the tree, so the audit's previous-path
            # columns were empty for every row the review rewrote -- the one
            # case they exist to record.
            previous_source_file=str(item.get("previous_source_file") or ""),
            previous_method=str(item.get("previous_method") or ""),
        )
        audit = item.get("source_resolution_llm_audit")
        if isinstance(audit, dict):
            entry["llm_audit"] = dict(audit)
        entries.append(entry)
    return entries


#: Metadata describing WHICH source a candidate points at, as opposed to facts
#: about the kernel itself. Every one of these is derived from the old path, so
#: a rewrite that leaves them in place produces a candidate describing two
#: different sources at once -- and the downstream readers disagree about which
#: one wins. ``forge_submit._resolve_framework`` consults ``source_framework``
#: before it ever looks at ``source_file``, and ``classify_patchability`` reads
#: ``kernel_kind`` to decide a kernel is prebuilt assembly, so a stale value
#: silently misroutes or skips the new source.
_SOURCE_DERIVED_METADATA = (
    "kernel_sources",
    "kernel_kind",
    "prebuilt_binary",
    "source_framework",
    "runtime_backend",
    "launcher_source_file",
    "source_promoted_from_launcher",
    "tracelens_launcher_path",
    "kernel_path",
    "vendor_dispatch_wrapper",
    "runtime_generated_kernel",
    "flydsl_source_from_fallback",
    "source_resolution_confidence",
    "op_to_source_status",
    "op_to_source_kind",
    "op_to_source_patchable",
    "op_to_source_reason",
    "op_to_source_matched_route",
    "source_file_missing_on_disk",
    # Names the path the playbook anchor displaced. Derived from the old
    # source_file like the rest, so a review that moves the path leaves it
    # asserting that some third file was superseded -- and it is restamped in
    # the same pass, since _stamp_candidate_metadata re-runs the playbook match.
    "source_file_superseded_by_playbook",
)


def _is_curated_resolution(item: dict[str, Any]) -> bool:
    """Whether this candidate's source came from a deterministic, authoritative tier.

    The active finder demangles the device symbol and pins the actual editable
    source in the installed tree; a model shown a path and forty lines cannot
    outrank that, so finder resolutions (like the retired curated map before
    them) are not open to LLM rewriting.

    A vendor-playbook match qualifies on the same grounds and regardless of
    which tier resolved the path it replaced: the registry states that the
    operator is tuned through a task bundle, and ``source_file`` holds that
    bundle's anchor. Leaving it unprotected would let a review proposal point
    the field back at framework source and route the candidate to a backend
    that has nothing to rewrite there.

    Unconditional for a playbook match, because a matched playbook always has
    an anchor: :func:`load_vendor_operator_playbooks` refuses an entry without
    one. Scoping this on a per-row marker instead is what let a row with no
    anchor and no path read as protected.
    """
    if str(item.get("patch_strategy") or "").strip() == "vendor_playbook":
        return True
    status = str(item.get("op_to_source_status") or "").strip()
    method = str(item.get("source_resolution_method") or "").strip()
    authoritative = {
        getattr(_KSC, "METHOD_ACTIVE_FINDER", "active_finder"),
        getattr(_KSC, "METHOD_SYMBOL_INDEX", "symbol_index"),
        getattr(_KSC, "METHOD_CURATED", "op_to_source"),
    }
    return method in authoritative and status in {
        _ROUTABLE_STATUS,
        "non_rewritable",
        "no_kernel",
    }


def write_source_resolution_artifact(
    candidates: list[dict[str, Any]],
    out_path: Path | str,
    *,
    framework: str = "",
    model_name: str = "",
    log_path: Path | str | None = None,
) -> Path | None:
    """Write the source-resolution artifact next to the candidate report.

    One file answering "where does each hot kernel live, and how do we know",
    so a reviewer does not have to reconstruct it from candidate internals.
    Failure is non-fatal: the artifact is for humans and the review tier, and
    the candidates themselves already carry the same fields.
    """
    try:
        entries = build_source_resolution_entries(candidates)
        doc = _KSC.make_document(
            entries,
            generated_by="tracelens_analysis",
            model_name=model_name,
            framework=framework,
        )
        problems = _KSC.validate_document(doc)
        if problems:
            log.warning(
                "source-resolution artifact failed its own contract (%d issue(s)): %s",
                len(problems),
                "; ".join(problems[:3]),
            )
            return None
        path = Path(out_path)
        atomic_write_json(path, doc)
        final_entries = doc.get("entries") or []
        resolved = sum(1 for entry in final_entries if entry.get("source_file"))
        summary = f"source resolution: {resolved}/{len(final_entries)} kernel(s) located -> {path.name}"
        log.info("%s", summary)
        if log_path:
            append_log(log_path, summary)
        return path
    except Exception as exc:  # noqa: BLE001 - reporting aid, never fails the run
        log.warning("could not write source-resolution artifact: %r", exc)
        return None


def recommend_backends(candidate: dict[str, Any]) -> list[str]:
    """Recommend a backend ladder for a reusable native kernel.

    Recommends the forge backend.

    Args:
        candidate: The hot-kernel candidate dict.

    Returns:
        The recommended backend names, or ``[]`` for unresolved source,
        non-reusable, vendor-binary, or runtime-generated kernels.
    """
    source_type = candidate.get("source_type")
    if not candidate.get("source_file"):
        return []
    if not candidate.get("reusable_native_kernel", classify_patchability(candidate)[0]):
        return []
    if source_type == "vendor_binary":
        return []
    if source_type == "runtime_generated":
        return []
    return ["forge"]


def build_notes(candidate: dict[str, Any]) -> str:
    """Build a short human-readable optimization note for a candidate.

    Args:
        candidate (dict[str, Any]): A finalized hot-kernel candidate row.

    Returns:
        str: A note describing whether/why the candidate is routable
            (resolved source vs. an explanation of why it was skipped).
    """
    if not candidate.get("source_file"):
        return "source file not resolved; backend dispatch will be skipped"
    if candidate.get(
        "runtime_generated_kernel",
        is_runtime_generated_kernel(str(candidate.get("name") or ""), str(candidate.get("source_file") or "")),
    ):
        return "runtime-generated torch.compile/Inductor kernel; not reusable, kernel-opt disabled"
    if not candidate.get("reusable_native_kernel", classify_patchability(candidate)[0]):
        return "not a reusable native source; kernel-opt disabled"
    return f"resolved source: {candidate['source_file']}"


#: Mirrors of the registry, used only when that package is not importable
#: (standalone invocation). Kept identical to the bypass route's copies; tests
#: assert every one of them against the registry.
_STANDALONE_SCRIPTABLE = frozenset({"xdit", "custom"})
_STANDALONE_DENOISER_CONFIG = frozenset({"xdit"})


def _is_scriptable_framework(framework: str | None) -> bool:
    """Return whether ``framework`` is a server-less scriptable image framework.

    Scriptable frameworks (e.g. xDiT diffusion) have no LLM decode steady-state
    phase, so trace analysis uses the plain pytorch perf report + skips the
    steady-state splitter. Prefers the canonical ``framework_registry``; falls
    back to a name check so the tool stays usable when run standalone (outside
    an importable ``inference_optimizer`` package).

    Args:
        framework: Framework name (matched case-insensitively).

    Returns:
        bool: ``True`` for scriptable image frameworks.
    """
    try:
        from hyperloom.inference_optimizer.framework_registry import is_scriptable

        return is_scriptable(framework)
    except ImportError:  # standalone invocation without the package installed.
        return str(framework or "").strip().lower() in _STANDALONE_SCRIPTABLE


def _has_diffusion_ceiling(framework: str | None) -> bool:
    """Return whether an analytic diffusion ceiling is meaningful for ``framework``.

    Scriptable does not imply diffusion: ``custom`` runs an operator-supplied
    entrypoint whose model Hyperloom never inspects, so the config-derived
    geometry the ceiling needs cannot be resolved, and a guessed one is worse
    than none. Read from the registry rather than matched against a name, so the
    next framework is classified when it is added rather than when someone
    remembers this call site.

    Args:
        framework: Framework name (matched case-insensitively).

    Returns:
        bool: ``True`` for frameworks shipping a readable denoiser config.
    """
    try:
        from hyperloom.inference_optimizer.framework_registry import has_denoiser_config

        return has_denoiser_config(framework)
    except ImportError:  # standalone invocation without the package installed.
        return str(framework or "").strip().lower() in _STANDALONE_DENOISER_CONFIG


def _load_gpu_timeline_rows(output_dir: Path) -> list[dict[str, str]]:
    """Read all rows from ``perf_report_csvs/gpu_timeline.csv``, empty if absent."""
    csv_path = output_dir / "perf_report_csvs" / "gpu_timeline.csv"
    if not csv_path.exists():
        return []
    try:
        with open(csv_path, encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except (OSError, csv.Error):
        return []


#: Column spellings for the duration cell of a gpu_timeline row. TraceLens has
#: renamed this header across versions, and a spelling this tuple does not carry
#: reads as "value 0" instead of "value unknown".
_GPU_TIMELINE_MS_COLUMNS = ("time ms", "time (ms)", "time_ms", "ms")


def _gpu_timeline_cell(row: dict[str, str], columns: tuple[str, ...]) -> float | None:
    """Read one numeric cell from a gpu_timeline row, or ``None`` if unusable.

    Absent, blank and unparseable all collapse to ``None`` rather than to a
    number. A defaulted ``0`` is not a neutral answer here: the trace total time
    is the ``gpu_pct`` denominator, so a renamed or truncated column would read
    as "this trace took 0 ms" and skew every reported share, silently and with
    ``status`` still ``ok``. The caller must be able to distinguish "the profile
    says zero" from "this file did not tell us".

    Args:
        row: A parsed ``gpu_timeline.csv`` row.
        columns: Accepted column spellings, in priority order.

    Returns:
        The cell value, or ``None`` when no accepted column carries a number.
    """
    lowered = {str(k).strip().lower(): v for k, v in row.items() if k}
    for column in columns:
        if column not in lowered:
            continue
        raw = lowered[column]
        if raw is None or str(raw).strip() == "":
            continue
        try:
            # csv.DictReader hands back None for a short row, and float(None)
            # raises TypeError rather than ValueError.
            return float(str(raw).strip().rstrip("%"))
        except (TypeError, ValueError):
            continue
    return None


def _extract_total_time_us_from_gpu_timeline(output_dir: Path) -> float | None:
    """Read the trace total_time from gpu_timeline.csv (ms -> us).

    Args:
        output_dir: The analysis output directory holding
            ``perf_report_csvs/gpu_timeline.csv``.

    Returns:
        The trace total time in microseconds, or ``None`` when the CSV is
        absent or has no usable total-time row.
    """
    for row in _load_gpu_timeline_rows(output_dir):
        if (row.get("type") or "").strip().lower() == "total_time":
            ms = _gpu_timeline_cell(row, _GPU_TIMELINE_MS_COLUMNS)
            return None if ms is None else ms * 1000.0
    return None


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None,
    log_path: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
) -> int:
    """Run a subprocess, tee its output to a log, and return the exit code.

    Args:
        cmd: Command and arguments to execute.
        cwd: Working directory, or ``None`` to inherit the current one.
        log_path: Log file the command line and output are appended to.
        timeout_s: Subprocess timeout in seconds.
        env: Optional environment for the child process.

    Returns:
        The process return code.
    """
    append_log(log_path, f"$ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    append_log(log_path, proc.stdout or "")
    append_log(log_path, f"[exit_code] {proc.returncode}")
    return proc.returncode


# Defaults kept in sync with src/hyperloom/agents/kernel/scripts/install.sh (TRACELENS_REPO /
# TRACELENS_REF). Overridable via env so a run can pin its own SHA.
_TRACELENS_REPO_DEFAULT = "https://github.com/AMD-AGI/TraceLens.git"
# Head of release/hyperloom_integration_v1.0.
_TRACELENS_REF_DEFAULT = "a59a9c165bb64c7c416fd7cf79149803d552e43c"


def _default_tracelens_root() -> Path:
    """Installer-managed default checkout path (mirrors install.sh /
    hyperloom.inference_optimizer.session.paths.deps_cache_root)."""
    from hyperloom.inference_optimizer.session.paths import deps_cache_root

    return deps_cache_root() / "TraceLens"


def _is_default_tracelens_root(tl_root: Path) -> bool:
    """True when tl_root is the installer-managed default (not an operator
    override). Only the default path is self-healed; an explicit
    --tracelens-root / TRACELENS_ROOT is operator-maintained and fails fast."""
    try:
        return Path(tl_root).resolve() == _default_tracelens_root().resolve()
    except OSError:
        return False


def _tracelens_checkout_complete(tl_root: Path) -> bool:
    """A checkout is usable when it has TraceLens metadata or the skill tree.

    A real checkout normally has ``.git`` (which may be a file in worktrees),
    but tests and copied source bundles can be valid without VCS metadata. The
    TraceLens skill file is the runtime contract this wrapper needs before pip
    install and analysis, so accept that layout too.
    """
    if (tl_root / ".git").exists():
        return True
    return (tl_root / "TraceLens/Agent/Analysis/skills/analysis-orchestrator/SKILL.md").exists()


def _rmtree_quiet(path: Path) -> None:
    """Best-effort recursive delete; never raises (used on cleanup paths)."""
    import shutil

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError:
        return


def _ensure_tracelens_checkout(tl_root: Path, *, log_path: Path) -> None:
    """Idempotently rebuild the TraceLens checkout if it vanished mid-run.

    The pod-local checkout can disappear when a concurrent install re-clones it.
    Take the shared pod-local ``.install.lock``, double-check under the lock, then
    clone into a temp sibling and atomically rename into place so a partial clone
    is never observed. Keep in lockstep with install.sh (ensure_tracelens).
    """
    tl_root = Path(tl_root)
    if _tracelens_checkout_complete(tl_root):
        return
    repo = os.environ.get("TRACELENS_REPO") or _TRACELENS_REPO_DEFAULT
    ref = os.environ.get("TRACELENS_REF") or _TRACELENS_REF_DEFAULT
    # .install.lock lives at the open-source root, the same path both installers lock.
    lock_dir = tl_root.parent
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / ".install.lock"
    append_log(log_path, f"TraceLens root missing/incomplete; self-healing checkout at {tl_root}")
    import fcntl

    with lock_path.open("w") as lock_fh:
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
        except OSError as exc:
            # Lock unavailable: proceed unlocked (the atomic rename still avoids a torn checkout).
            append_log(log_path, f"TraceLens self-heal: flock failed ({exc}); proceeding without lock")
        # Re-check completeness under the lock (a concurrent healer may have finished).
        if _tracelens_checkout_complete(tl_root):
            append_log(log_path, "TraceLens checkout completed by a concurrent healer; reusing")
            return
        # A stale/partial tree blocks the atomic rename below; move it aside first.
        if tl_root.exists():
            stale = tl_root.parent / f".{tl_root.name}.stale.{uuid.uuid4().hex[:8]}"
            os.replace(tl_root, stale)
            _rmtree_quiet(stale)
        tmp_dir = tl_root.parent / f".{tl_root.name}.heal.{uuid.uuid4().hex[:8]}"
        try:
            rc = run_command(
                ["git", "clone", "--depth", "1", repo, str(tmp_dir)],
                cwd=None,
                log_path=log_path,
                timeout_s=600,
            )
            if rc != 0:
                raise FileNotFoundError(
                    f"TraceLens root not found and self-heal clone failed (repo={repo}); tried to rebuild at {tl_root}"
                )
            # Pin to the requested SHA; a failed fetch/checkout must not ship the
            # clone's default HEAD (an unpinned tree).
            if ref and ref != "HEAD":
                rc = run_command(
                    ["git", "-C", str(tmp_dir), "fetch", "--depth", "1", "origin", ref],
                    cwd=None,
                    log_path=log_path,
                    timeout_s=600,
                )
                if rc == 0:
                    rc = run_command(
                        ["git", "-C", str(tmp_dir), "checkout", "-q", "FETCH_HEAD"],
                        cwd=None,
                        log_path=log_path,
                        timeout_s=120,
                    )
                if rc != 0:
                    raise FileNotFoundError(
                        f"TraceLens self-heal could not pin ref={ref} (repo={repo}); "
                        f"refusing to install an unpinned checkout at {tl_root}"
                    )
            os.replace(tmp_dir, tl_root)
        except BaseException:
            _rmtree_quiet(tmp_dir)
            raise
        append_log(log_path, f"TraceLens checkout self-healed at {tl_root}")


def roofline_match_key(name: str) -> str:
    """Normalize trace and rocprof names enough to join roofline data.

    Args:
        name (str): A kernel name from a trace or rocprof report.

    Returns:
        str: A canonical match key (e.g. ``hipblaslt_gemm``, ``attention``),
            falling back to the first 80 lower-cased chars.
    """
    lower = (name or "").lower()
    if "cijk_" in lower:
        return "hipblaslt_gemm"
    if "gemm_a16w16_asm" in lower or "a16w16" in lower:
        return "aiter_asm_gemm"
    if "attn_fwd" in lower or "flash_attn" in lower:
        return "attention"
    if "moe_ck2stages" in lower or "moe_ck_tile" in lower:
        return "moe_gemm"
    if "vectorized_layer_norm" in lower or "rms_norm" in lower:
        return "rms_norm"
    if "topk" in lower:
        return "topk"
    if "rope" in lower or "rotary" in lower:
        return "rope"
    if "nccl" in lower or "allreduce" in lower:
        return "allreduce"
    if "copy" in lower or "memcpy" in lower:
        return "memcpy"
    if "softmax" in lower:
        return "softmax"
    if "skinny" in lower:
        return "skinny_gemm"
    return lower[:80]


def load_roofline_results(path: str | None) -> dict[str, dict[str, Any]]:
    """Load roofline results JSON keyed by normalized kernel match key.

    Args:
        path (str | None): Path to a roofline results JSON file (a list of
            rows or a dict with a ``results`` list); may be empty/``None``.

    Returns:
        dict[str, dict[str, Any]]: Map of :func:`roofline_match_key` to row;
            empty when the path is missing or unparseable.
    """
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("name"):
            out[roofline_match_key(str(row["name"]))] = row
    return out


def merge_roofline_into_candidates(
    candidates: list[dict[str, Any]],
    roofline_by_name: dict[str, dict[str, Any]],
) -> None:
    """Merge roofline metrics into candidate rows in place.

    For each candidate, looks up roofline data by normalized name and copies
    bottleneck / utilization / suggestion fields; candidates without a match
    get conservative ``None``/default placeholders.

    Args:
        candidates (list[dict[str, Any]]): Hot-kernel candidate rows to enrich.
        roofline_by_name (dict[str, dict[str, Any]]): Roofline rows keyed by
            :func:`roofline_match_key`.
    """
    for item in candidates:
        if not isinstance(item, dict):
            continue
        roofline = roofline_by_name.get(roofline_match_key(str(item.get("name") or "")))
        if roofline:
            item["bottleneck"] = roofline.get("bottleneck", "unknown")
            item["arithmetic_intensity"] = roofline.get("arithmetic_intensity")
            item["compute_utilization_pct"] = roofline.get("compute_utilization_pct", 0.0)
            item["bandwidth_utilization_pct"] = roofline.get("bandwidth_utilization_pct", 0.0)
            item["suggestion"] = roofline.get("suggestion", "")
            item["recommended_actions"] = roofline.get("recommended_actions") or []
            item["roofline_name"] = roofline.get("name")
        else:
            item.setdefault("bottleneck", "unknown")
            item.setdefault("arithmetic_intensity", None)
            item.setdefault("compute_utilization_pct", None)
            item.setdefault("bandwidth_utilization_pct", None)
            item.setdefault("recommended_actions", [])


def _first_non_empty(*values: Any) -> Any:
    """Return the first argument that is neither ``None`` nor empty string.

    Args:
        *values (Any): Candidate values in priority order.

    Returns:
        Any: The first value that is not ``None`` or ``""``, else ``None``.
    """
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _kernel_roofline_row(candidate: dict[str, Any]) -> dict[str, Any]:
    """Project one hot-kernel candidate into the kernel-roofline view.

    Args:
        candidate (dict[str, Any]): A finalized hot-kernel candidate row.

    Returns:
        dict[str, Any]: A flattened roofline row with identity, cost, and
            (possibly ``None``) utilization/bottleneck fields.
    """
    arithmetic_intensity = _first_non_empty(
        candidate.get("arithmetic_intensity"),
        candidate.get("flops_per_byte"),
    )
    return {
        "kernel_id": candidate.get("kernel_id"),
        "name": candidate.get("name"),
        "gpu_pct": candidate.get("gpu_pct"),
        "duration_us": candidate.get("duration_us"),
        "call_count": candidate.get("call_count"),
        "kernel_category": candidate.get("kernel_category"),
        "source_file": candidate.get("source_file"),
        "bottleneck": _first_non_empty(
            candidate.get("bottleneck"),
            candidate.get("bound_type"),
        ),
        "bound_type": candidate.get("bound_type"),
        "arithmetic_intensity": arithmetic_intensity,
        "flops_per_byte": candidate.get("flops_per_byte"),
        "efficiency_percent": candidate.get("efficiency_percent"),
        "compute_utilization_pct": candidate.get("compute_utilization_pct"),
        "bandwidth_utilization_pct": candidate.get("bandwidth_utilization_pct"),
        "suggestion": candidate.get("suggestion") or "",
        "roofline_name": candidate.get("roofline_name"),
        "recommended_actions": list(candidate.get("recommended_actions") or []),
        "reusable_native_kernel": bool(candidate.get("reusable_native_kernel")),
        "rocprof_roofline": candidate.get("rocprof_roofline"),
        # Contract alignment (F6): emit the shared roofline_source provenance enum
        # so both routes carry it. TraceLens rows come from its per-op perf model
        # (analytical) when that model produced numbers, else placeholder.
        "roofline_source": _RL_ANALYTICAL
        if any(
            candidate.get(k) is not None
            for k in (
                "arithmetic_intensity",
                "flops_per_byte",
                "efficiency_percent",
                "compute_utilization_pct",
                "bandwidth_utilization_pct",
            )
        )
        else _RL_PLACEHOLDER,
    }


def build_kernel_roofline_payload(
    *,
    trace_input: str,
    trace_input_type: str,
    analysis_md_path: str,
    kernel_candidates_path: str,
    roofline_json_path: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the per-kernel roofline sidecar payload.

    A view over the candidates plus an optional ``--roofline-json``; missing
    counters stay null.

    Args:
        trace_input: The trace input path recorded in the payload.
        trace_input_type: Whether ``trace_input`` is a file or capture dir.
        analysis_md_path: Path to the source analysis report.
        kernel_candidates_path: Path to the candidates JSON.
        roofline_json_path: Optional path to an external roofline JSON.
        candidates: The finalized hot-kernel candidate rows.

    Returns:
        The kernel-roofline sidecar payload dict.
    """
    rows = [_kernel_roofline_row(candidate) for candidate in candidates if isinstance(candidate, dict)]
    return {
        "schema_version": 1,
        "source": "tracelens_analysis",
        "trace_input": trace_input,
        "trace_input_type": trace_input_type,
        "analysis_md_path": analysis_md_path,
        "kernel_candidates_path": kernel_candidates_path,
        "roofline_json_path": roofline_json_path,
        "kernels": rows,
    }


def kernel_roofline_path_for_run(run_dir: Path, filename: str = "kernel_roofline.json") -> Path:
    """Return the stable session-level kernel roofline report path."""
    try:
        session_sub = run_dir.parent
        runs_dir = session_sub.parent
        kernel_agent_dir = runs_dir.parent
    except (IndexError, AttributeError):
        return run_dir / "reports" / filename
    if runs_dir.name == "runs" and kernel_agent_dir.name == "kernel-agent":
        session_dir = kernel_agent_dir.parent
        return session_dir / "reports" / filename
    # Backward-compatible flat run-dir layout.
    try:
        runs_dir_legacy = run_dir.parent
        kernel_agent_dir_legacy = runs_dir_legacy.parent
    except (IndexError, AttributeError):
        return run_dir / "reports" / filename
    if runs_dir_legacy.name == "runs" and kernel_agent_dir_legacy.name == "kernel-agent":
        session_dir = kernel_agent_dir_legacy.parent
        return session_dir / "reports" / filename
    return run_dir / "reports" / filename


def _candidate_model_config_paths(model_name: str) -> list[Path]:
    """Enumerate candidate ``config.json`` paths for a model name/path.

    Considers the value as a direct JSON path, a directory containing
    ``config.json``, and locations under ``$HYPERLOOM_MODELS_ROOT``.

    Args:
        model_name (str): A model name or filesystem path.

    Returns:
        list[Path]: Deduplicated candidate config paths in priority order;
            empty when ``model_name`` is blank.
    """
    text = str(model_name or "").strip()
    if not text:
        return []
    raw = Path(text).expanduser()
    candidates: list[Path] = []
    if raw.suffix == ".json":
        candidates.append(raw)
    candidates.append(raw / "config.json")
    # Optional models-root override (env-only; no hardcoded default). Standalone
    # kernel-agent tools cannot import hyperloom.common (the Ray/subprocess
    # sys.path contract in hyperloom.common.__init__), so this mirrors the shared
    # resolver's strategy (local path + HF hub cache) independently.
    _root = os.environ.get("HYPERLOOM_MODELS_ROOT", "").strip()
    if _root:
        candidates.append(Path(_root) / text / "config.json")
        candidates.append(Path(_root) / raw.name / "config.json")
    # HF hub cache (what vLLM/SGLang populate when given a repo id): the snapshot
    # commit-hash segment is not derivable by string, so let huggingface_hub
    # locate it (honoring HF_HOME / HF_HUB_CACHE). Best-effort; skipped when the
    # optional dep is absent or nothing is cached.
    try:
        from huggingface_hub import try_to_load_from_cache

        _hit = try_to_load_from_cache(repo_id=text, filename="config.json")
        if isinstance(_hit, str):
            candidates.append(Path(_hit))
    except Exception:
        pass
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def load_model_kernel_params(model_name: str) -> dict[str, Any]:
    """Read HF config.json and return attention parameters relevant to GEAK.

    Args:
        model_name (str): A model name or path used to locate ``config.json``.

    Returns:
        dict[str, Any]: Attention/MLA params (e.g. ``HEAD_SIZE``,
            ``NUM_ATTENTION_HEADS``) plus ``MODEL_CONFIG_PATH``; empty when no
            readable config is found.
    """
    for config_path in _candidate_model_config_paths(model_name):
        if not config_path.is_file():
            continue
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        params: dict[str, Any] = {
            "MODEL_CONFIG_PATH": str(config_path),
        }
        has_mla_dims = any(cfg.get(key) is not None for key in ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"))
        if cfg.get("head_dim") is not None:
            params["HEAD_SIZE"] = cfg.get("head_dim")
        elif not has_mla_dims:
            hidden = cfg.get("hidden_size")
            heads = cfg.get("num_attention_heads")
            if isinstance(hidden, int) and isinstance(heads, int) and heads > 0 and hidden % heads == 0:
                params["HEAD_SIZE"] = hidden // heads
        for src, dst in (
            ("qk_nope_head_dim", "QK_NOPE_HEAD_DIM"),
            ("qk_rope_head_dim", "QK_ROPE_HEAD_DIM"),
            ("v_head_dim", "V_HEAD_DIM"),
            ("kv_lora_rank", "KV_LORA_RANK"),
            ("num_attention_heads", "NUM_ATTENTION_HEADS"),
            ("num_key_value_heads", "NUM_KEY_VALUE_HEADS"),
            ("hidden_size", "HIDDEN_SIZE"),
        ):
            if cfg.get(src) is not None:
                params[dst] = cfg[src]
        return params
    return {}


_FLYDSL_TARGET_ARCH_BY_PLATFORM = {
    "mi300x": "gfx942",
    "mi308x": "gfx942",
    "mi325x": "gfx942",
    "mi355x": "gfx950",
}
_FLYDSL_SMEM_MARKERS = ("SmemAllocator", "SmemPtr", "smem_alloc")
_FLYDSL_BUFFER_LOAD_MARKERS = (
    "make_buffer_tensor",
    "BufferCopy",
    "rocdl",
    "buffer_load",
)


def _resolve_flydsl_source_fallback() -> str:
    """Resolve the real FlyDSL MoE kernel source for synthetic pseudo-ops.

    Used for FlyDSL pseudo-ops, looking for
    ``$DSL2_ROOT/kernels/moe_gemm_2stage.py`` and known fallback roots.

    Returns:
        The first existing FlyDSL MoE kernel source path, or ``""`` when none
        exists.
    """
    roots = [
        os.environ.get("DSL2_ROOT", "").strip(),
        os.environ.get("FLYDSL_ROOT", "").strip(),
        "/opt/FlyDSL",
        "/sgl-workspace/flydsl",
    ]
    for root in roots:
        if not root:
            continue
        cand = os.path.join(root, "kernels", "moe_gemm_2stage.py")
        if os.path.isfile(cand):
            return cand
    return ""


def _flydsl_kernel_params(
    source_file: str,
    target_platform: str,
) -> dict[str, Any]:
    """Build FlyDSL-specific kernel params.

    Captures target arch, JIT cache state, and smem/buffer-load usage;
    best-effort and never raises.

    Args:
        source_file: The FlyDSL kernel source path.
        target_platform: The target GPU platform name.

    Returns:
        The FlyDSL kernel-params dict (possibly partial).
    """
    params: dict[str, Any] = {}
    arch = _FLYDSL_TARGET_ARCH_BY_PLATFORM.get(
        (target_platform or "").strip().lower(),
    )
    if arch:
        params["FLYDSL_TARGET_ARCH"] = arch
    cache_dir = os.environ.get("FLYDSL_AUTOTUNE_CACHE_DIR", "").strip()
    if cache_dir:
        params["FLYDSL_AUTOTUNE_CACHE_DIR"] = cache_dir
    enable_cache = os.environ.get("FLYDSL_RUNTIME_ENABLE_CACHE", "").strip()
    if enable_cache:
        params["FLYDSL_RUNTIME_ENABLE_CACHE"] = enable_cache
    if source_file:
        try:
            with open(source_file, "r", encoding="utf-8", errors="replace") as fh:
                head = fh.read(_FLYDSL_SCAN_BYTES)
        except OSError:
            head = ""
        if head:
            if any(m in head for m in _FLYDSL_SMEM_MARKERS):
                params["FLYDSL_USES_SMEM"] = True
            if any(m in head for m in _FLYDSL_BUFFER_LOAD_MARKERS):
                params["FLYDSL_USES_BUFFER_LOAD"] = True
    return params


def enrich_candidates_with_runtime_metadata(
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Attach stable runtime metadata fields before GEAK prompt generation.

    Mutates each candidate in place, filling framework, shapes/dtypes,
    model and FlyDSL kernel params, and runtime flags so the downstream
    prompt builder sees a consistent schema.

    Args:
        candidates (list[dict[str, Any]]): Hot-kernel candidate rows to enrich.
        args (argparse.Namespace): Parsed CLI args carrying framework, model
            name, target platform, and runtime flags.
    """
    framework = str(getattr(args, "framework", "") or "").strip()
    model_params = load_model_kernel_params(str(getattr(args, "model_name", "") or ""))
    target_platform = str(getattr(args, "target_platform", "") or "")
    runtime_flags = {
        "analysis_mode": getattr(args, "analysis_mode", ""),
        "runtime_env": getattr(args, "runtime_env", ""),
        "target_platform": target_platform,
    }
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if framework:
            item.setdefault("framework", framework)
            item.setdefault("backend", framework)
        if "input_shapes" not in item:
            item["input_shapes"] = _shape_call_entries(
                item.get("shapes", []) or [],
                item.get("call_count"),
            )
            item["_input_shapes_synthetic"] = True
        item.setdefault("output_shapes", [])
        # Per-arg dtypes: TraceLens records them INLINE in each ``shapes`` string
        # ("(64,5120) bf16"); surface them as a clean, positionally-aligned list so
        # the GEAK harness allocates the correct-dtype tensors (fp8 weight vs bf16
        # activation). Falls back to any explicit ``dtypes`` field, then ``[]``.
        existing_dtypes = item.get("input_dtypes") or item.get("dtypes") or []
        if not existing_dtypes:
            existing_dtypes = _dtypes_from_shapes(item.get("shapes", []) or [])
        item["input_dtypes"] = existing_dtypes
        item.setdefault("output_dtypes", [])
        item.setdefault("runtime_args", {})
        item.setdefault("env_vars", {})
        item.setdefault("kernel_params", {})
        if model_params:
            params = item["kernel_params"]
            if isinstance(params, dict):
                for key, value in model_params.items():
                    params.setdefault(key, value)
        if item.get("source_type") == "flydsl":
            # FlyDSL pseudo-ops carry no real source_file; inject the FlyDSL MoE kernel source.
            _sf2 = str(item.get("source_file") or "").strip()
            if (not _sf2) or (not os.path.isfile(_sf2)):
                fb = _resolve_flydsl_source_fallback()
                if fb:
                    item["source_file"] = fb
            flydsl_params = _flydsl_kernel_params(
                str(item.get("source_file") or ""),
                target_platform,
            )
            params = item["kernel_params"]
            if isinstance(params, dict):
                for key, value in flydsl_params.items():
                    params.setdefault(key, value)
        flags = item.get("runtime_flags")
        if not isinstance(flags, dict):
            flags = {}
            item["runtime_flags"] = flags
        for key, value in runtime_flags.items():
            if value not in (None, ""):
                flags.setdefault(key, value)
        flags.setdefault("is_multigpu", bool(item.get("is_multigpu")))
        flags.setdefault("num_gpus_recommended", item.get("num_gpus_recommended"))
        # Kernel-class CONTRACT enrichment so GEAK's harness is faithful for
        # non-MoE kernels too (collectives need a real distributed reference;
        # attention needs causal/kv layout). Without these the harness can pass
        # a tautological or wrong-regime correctness check and a "win" won't
        # transfer to E2E. Derived from the kernel name + model config.
        _enrich_kernel_contract(item, model_params)


def _enrich_kernel_contract(item: dict[str, Any], model_params: dict[str, Any] | None) -> None:
    """Populate per-kernel-class contract fields used to build a faithful harness.

    - Collectives (all_reduce / all_gather / reduce_scatter / ...): record
      ``collective_op`` + ``world_size``/``tp_size`` (from model_params) so the
      harness can use ``torch.distributed.<op>`` as a real reference instead of
      a self-compare, and so a reviewer knows the kernel is comm-bound.
    - Attention (mha / flash / paged): record ``causal`` (decode/prefill attn
      is causal), ``kv_layout``, ``head_dim``/``num_heads`` from model_params,
      and the ``seqlen_regime`` so the harness benchmarks the served regime.
    These are advisory metadata; absent fields are simply not set.
    """
    name = str(item.get("name") or "").lower()
    mp = model_params or {}
    contract = item.get("kernel_contract")
    if not isinstance(contract, dict):
        contract = {}
    # --- collectives ---
    _COLL = (
        ("all_reduce", "allreduce"),
        ("all_gather", "allgather"),
        ("reduce_scatter", "reducescatter"),
        ("all_to_all", "alltoall"),
        ("broadcast",),
        # aiter cross_device_reduce_* kernels use all-reduce semantics.
        ("cross_device_reduce",),
        ("reduce",),
    )
    _OPMAP = {
        "all_reduce": "all_reduce",
        "allreduce": "all_reduce",
        "all_gather": "all_gather",
        "allgather": "all_gather",
        "reduce_scatter": "reduce_scatter",
        "reducescatter": "reduce_scatter",
        "all_to_all": "all_to_all",
        "alltoall": "all_to_all",
        "broadcast": "broadcast",
        "cross_device_reduce": "all_reduce",
        "reduce": "reduce",
    }
    if bool(item.get("is_multigpu")) or any(tag in name for grp in _COLL for tag in grp):
        op = next((_OPMAP[t] for grp in _COLL for t in grp if t in name), "all_reduce")
        contract["kind"] = "collective"
        contract["collective_op"] = op
        ws = (
            mp.get("WORLD_SIZE")
            or mp.get("TP_SIZE")
            or mp.get("TENSOR_PARALLEL_SIZE")
            or item.get("num_gpus_recommended")
        )
        if ws:
            contract["world_size"] = int(ws)
        if mp.get("TP_SIZE") or mp.get("TENSOR_PARALLEL_SIZE"):
            contract["tp_size"] = int(mp.get("TP_SIZE") or mp.get("TENSOR_PARALLEL_SIZE"))
        contract.setdefault("reduce_op", "sum")
        contract["reference"] = f"torch.distributed.{op}"
        contract["e2e_note"] = (
            "comm-bound collective: a 1-GPU GEAK slot cannot "
            "reproduce inter-GPU traffic; needs KERNEL_AGENT_NUM_GPUS>=world_size"
        )
    # --- attention ---
    elif any(t in name for t in ("mha", "flash", "attn", "attention", "paged")):
        contract["kind"] = "attention"
        contract["causal"] = True  # autoregressive serving attention is causal
        for src, dst in (
            ("HEAD_SIZE", "head_dim"),
            ("NUM_ATTENTION_HEADS", "num_heads"),
            ("NUM_KEY_VALUE_HEADS", "num_kv_heads"),
        ):
            if mp.get(src) is not None:
                contract[dst] = mp.get(src)
        contract.setdefault("kv_layout", "unknown")  # to be confirmed from trace; flag for reviewer
        contract["seqlen_regime"] = "prefill" if "prefill" in name else ("decode" if "decode" in name else "mixed")
        contract["reference"] = "torch.nn.functional.scaled_dot_product_attention"
    # E2E-TRANSFER honesty flag: warn when a kernel-level win is unlikely to move
    # serving E2E so neither the pipeline nor a reviewer over-trusts a micro-speedup.
    #  - sealed-ASM launchers: asm_*.cu dispatches a prebuilt .co HSACO; only the
    #    host-side launcher is editable, so wins are tiny (observed E2E #1: 1.09x
    #    kernel -> 1.0008x E2E).
    #  - collectives on a 1-GPU GEAK slot: cannot reproduce inter-GPU traffic.
    src = str(item.get("source_file") or "").lower()
    reasons = []
    if "/py_itfs_cu/asm_" in src or (os.path.basename(src).startswith("asm_") and src.endswith(".cu")):
        reasons.append("sealed-ASM launcher (.co binary not editable; only host launcher)")
    if contract.get("kind") == "collective":
        reasons.append("comm-bound collective (needs multi-GPU GEAK slot to reproduce E2E traffic)")
    if reasons:
        item["e2e_transferable"] = False
        item["e2e_transfer_note"] = "; ".join(reasons)
    else:
        item.setdefault("e2e_transferable", True)
    if contract:
        item["kernel_contract"] = contract


def build_task_groups(
    candidates: list[dict[str, Any]],
    *,
    source_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate reusable candidates by AST-resolved source function.

    A wrapper over :func:`aggregate_by_source_function`; only
    ``reusable_native_kernel`` candidates are grouped.

    Args:
        candidates: The hot-kernel candidate rows.
        source_root: Optional root to resolve relative source paths against.

    Returns:
        The task-group dicts, or ``[]`` when none carries a parseable launcher
        path (callers fall through to per-kernel dispatch).
    """
    reusable = [c for c in candidates if isinstance(c, dict) and c.get("reusable_native_kernel")]
    if not reusable:
        return []
    return aggregate_by_source_function(reusable, source_root=source_root)


def build_audit_summary(
    candidates: list[dict[str, Any]],
    *,
    trace_input: str,
    framework: str = "",
    target_platform: str = "",
    task_groups: list[dict[str, Any]] | None = None,
    trace_health_warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the ``tracelens/summary.json`` payload.

    Splits candidates into routable ``tasks`` and ``skipped`` (each with a
    ``skip_reason``), preserving priority order. Pure function.

    Args:
        candidates: The hot-kernel candidate rows.
        trace_input: The trace input path recorded in the summary.
        framework: Optional framework name recorded in the summary.
        target_platform: Optional target platform recorded in the summary.
        task_groups: Optional task-group projections to include.
        trace_health_warnings: Optional trace-health warnings to include.

    Returns:
        The ``summary.json`` payload dict.
    """
    tasks: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        reusable = bool(cand.get("reusable_native_kernel"))
        compact = {
            "kernel_id": cand.get("kernel_id"),
            "name": cand.get("name"),
            "source_file": cand.get("source_file") or "",
            "source_type": cand.get("source_type") or "",
            "kernel_category": cand.get("kernel_category") or "",
            "gpu_pct": cand.get("gpu_pct"),
            "duration_us": cand.get("duration_us"),
            "call_count": cand.get("call_count"),
            "tracelens_pitem_rank": cand.get("tracelens_pitem_rank"),
            "tracelens_pitem_title": cand.get("tracelens_pitem_title"),
            "bound_type": cand.get("bound_type") or "",
        }
        if reusable:
            compact["recommended_backends"] = list(cand.get("recommended_backends") or [])
            tasks.append(compact)
        else:
            compact["skip_reason"] = cand.get("skip_reason") or "unknown"
            skipped.append(compact)
    # Compact task_group projections for the audit view.
    group_entries: list[dict[str, Any]] = []
    for group in task_groups or []:
        if not isinstance(group, dict):
            continue
        group_entries.append(
            {
                "task_group_id": group.get("task_group_id"),
                "source_path": group.get("source_path"),
                "definition_line": group.get("definition_line"),
                "function_name": group.get("function_name"),
                "ast_resolved": bool(group.get("ast_resolved")),
                "primary_kernel_id": group.get("primary_kernel_id"),
                "kernel_ids": list(group.get("kernel_ids") or []),
                "row_count": len(group.get("rows") or []),
                "aggregate_duration_us": group.get("aggregate_duration_us"),
                "aggregate_call_count": group.get("aggregate_call_count"),
                "aggregate_gpu_pct": group.get("aggregate_gpu_pct"),
            }
        )
    return {
        "generated_at": utc_now(),
        "trace_input": trace_input,
        "framework": framework,
        "target_platform": target_platform,
        "task_count": len(tasks),
        "skipped_count": len(skipped),
        "task_group_count": len(group_entries),
        "tasks": tasks,
        "skipped": skipped,
        "task_groups": group_entries,
        # Trace-quality findings (empty = healthy; non-empty explains an empty ``tasks``).
        "trace_health_warnings": list(trace_health_warnings or []),
    }


def _with_demangled_symbol(candidate: dict[str, Any]) -> dict[str, Any]:
    """Copy ``candidate`` with the demangled device symbol added when it differs.

    ``native_operation_key`` already owns this normalization for the task-group
    identity: it strips the CPU-side launch call, a return type and template
    parameters, then demangles an Itanium symbol. Only added when it actually
    changes the string, so a row whose symbol was never mangled does not grow a
    field restating it.

    Args:
        candidate (dict[str, Any]): One finalized candidate row.

    Returns:
        dict[str, Any]: A shallow copy, possibly carrying
            ``device_kernel_name_demangled``.
    """
    row = dict(candidate)
    if _native_operation_key is None:
        return row
    raw = str(row.get("device_kernel_name") or row.get("name") or "").strip()
    if not raw:
        return row
    try:
        demangled = _native_operation_key(raw)
    except Exception:  # noqa: BLE001 - a reading aid must not fail the stage
        return row
    if demangled and demangled != raw:
        row["device_kernel_name_demangled"] = demangled
    return row


def run_candidate_review_stage(
    run_dir: Path,
    *,
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
    log_path: Path | str | None = None,
    trace_health_warnings: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Run the review stage, converting any unexpected fault into a warning.

    The stage is advisory by construction, and it sits at the end of an
    analysis that a multi-hour benchmark paid for. An unforeseen fault in it
    must cost the audit, not the run, so nothing escapes this boundary.
    """
    try:
        return _run_candidate_review_stage(
            run_dir,
            candidates=candidates,
            args=args,
            log_path=log_path,
            trace_health_warnings=trace_health_warnings,
        )
    except Exception as exc:  # noqa: BLE001 - never let the audit fail the run
        log.warning("candidate review stage failed (%r); keeping the deterministic table", exc)
        if trace_health_warnings is not None:
            trace_health_warnings.append(
                {
                    "code": "candidate_review_failed",
                    "severity": "error",
                    "status": "internal_error",
                    "detail": type(exc).__name__,
                    "message": (
                        "The candidate review stage raised "
                        f"{type(exc).__name__}; kernel_candidates.json is the "
                        "unreviewed deterministic result."
                    ),
                }
            )
        return {}


#: Everything the review can stage without moving ``source_file``. The
#: re-derivation is skipped for rows that did not change, and a path is only one
#: of the things that can: operand dims are most often supplied for a kernel the
#: deterministic tiers already located, so keying the check on the path alone
#: drops exactly the proposals that were hardest to obtain.
_REVIEW_STAGED_PROPOSALS = (
    "review_shapes",
    "review_input_dtypes",
    "review_reusable_hint",
    "review_benchmark_files",
)

#: Rebuilt from ``shapes``, so they must not outlive the dims they described.
#: ``enrich_candidates_with_runtime_metadata`` runs after this stage and refills
#: ``input_shapes`` from whatever ``shapes`` now holds, marking it synthetic
#: again -- which is the whole reason clearing it is safe.
#:
#: ``invocation_cases`` and ``raw_arg_spec`` are deliberately NOT here. Nothing
#: rebuilds them: both are produced only by ``_finalize_candidates``, from the
#: perf CSV, long before the review runs, and no later pass can re-derive an
#: ordered scalar argument list from a list of operand dims. They also do not
#: describe the dims being replaced -- a CSV row can carry a raw arg spec while
#: yielding no tensor operands at all, which is precisely a row whose ``shapes``
#: is empty and therefore the row a review supplies dims for. Clearing them
#: there dropped the scalar signature and collapsed multi-case task groups
#: (``_expanded_group_rows`` expands ``invocation_cases`` into one workload case
#: each), so a stage that exists to add operand evidence removed some.
_REVIEW_STALE_SHAPE_FIELDS = (
    "input_shapes",
    "_input_shapes_synthetic",
)


def _adopt_reviewed_shapes(item: dict[str, Any]) -> None:
    """Take the operand dims the review supplied, if it supplied any.

    Only fires where the deterministic stage came up empty. A recorded shape
    outranks a reviewed one even when the review is confident, because the
    reviewed dims can be arithmetic over the serving configuration and nothing
    downstream re-measures them; the integration benchmark hours later is the
    first thing that would notice they were wrong.

    The alternate representations are dropped rather than translated. They
    describe the previous dims, and a harness built from a mix of the two would
    be wrong in a way that still benchmarks cleanly.
    """
    proposed = item.get("review_shapes")
    if not isinstance(proposed, list) or not proposed:
        return
    if item.get("shapes"):
        return
    item["shapes"] = list(proposed)
    item["shape_provenance"] = str(item.get("review_shape_provenance") or _REVIEW_DERIVED_PROVENANCE)
    reviewed_dtypes = item.get("review_input_dtypes")
    if isinstance(reviewed_dtypes, list) and reviewed_dtypes:
        item["input_dtypes"] = list(reviewed_dtypes)
    for key in _REVIEW_STALE_SHAPE_FIELDS:
        item.pop(key, None)


def _accept_review_proposals(
    item: dict[str, Any],
    op_cat_map: dict[str, str] | None = None,
) -> None:
    """Take what the review supplied for a candidate whose source did not move.

    Restamping is still required, because the alternate shape representations
    are rebuilt from ``shapes`` and adopting reviewed dims invalidates the ones
    describing the old set. What it must not do is disturb the source judgments:
    they describe the same path they were computed from, so there is nothing to
    recompute and nothing stale to clear.
    """
    _adopt_reviewed_shapes(item)
    _stamp_candidate_metadata(item, op_cat_map)
    # Stamping recomputes benchmark_files from the curated marker table, which
    # is coarser than a session that went and looked. Its verified answer wins.
    # Only a non-empty one: an empty list means the session named harnesses and
    # none of them exist, which says its proposal was wrong, not that the
    # curated table's answer is. Letting it through would strip a runnable
    # harness from the invocation spec the backend is handed.
    reviewed_harnesses = item.get("review_benchmark_files")
    if isinstance(reviewed_harnesses, list) and reviewed_harnesses:
        item["benchmark_files"] = list(reviewed_harnesses)
    # A restrictive hint is honoured, a permissive one is not. The reviewer can
    # veto a kernel it knows is not worth a tuning session, but it cannot talk
    # the gate into dispatching something the deterministic rules rejected.
    if item.get("review_reusable_hint") is False and item.get("reusable_native_kernel"):
        item["reusable_native_kernel"] = False
        item["skip_reason"] = (
            str(item.get("review_skip_reason") or "").strip()
            or f"review: {item.get('review_reason') or 'not worth a tuning session'}"
        )


def _rederive_after_review(item: dict[str, Any], op_cat_map: dict[str, str] | None = None) -> None:
    """Recompute everything that follows from ``source_file`` after it moved.

    The review returns a location, not a verdict. Re-running the deterministic
    stamping keeps :func:`classify_patchability` the only gate that decides
    routability, so the vendor-binary, dispatch-wrapper and runtime-generated
    rejections still apply to a path the model supplied.

    Only for a revision that moved the path. Clearing
    :data:`_SOURCE_DERIVED_METADATA` is sound when the values describe a source
    the candidate no longer names, and unsound otherwise: eighteen of those
    nineteen keys have no producer in this pass -- the finder that filled them
    read the demangled device symbol and the binary's exports, and stamping
    cannot reconstruct that from a path -- so clearing them where the source
    stood still would drop them for good. Whether any of them is populated today
    is not the point; a field that cannot be rebuilt must not be cleared on a
    revision that gave no reason to doubt it.
    """
    new_source = str(item.get("source_file") or "")
    for key in _SOURCE_DERIVED_METADATA:
        item.pop(key, None)
    item["source_path"] = new_source
    item["kernel_repo"] = find_repo_root(new_source) if new_source else ""
    item["source_type"] = source_type_for(item.get("name", ""), new_source)
    if item["source_type"] != "vendor_binary" and is_vendor_dispatch_wrapper(item.get("name", ""), new_source):
        item["source_type"] = "vendor_binary"
        item["vendor_dispatch_wrapper"] = True
    item["runtime_generated_kernel"] = is_runtime_generated_kernel(item.get("name", ""), new_source)
    _accept_review_proposals(item, op_cat_map)


def _run_candidate_review_stage(
    run_dir: Path,
    *,
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
    log_path: Path | str | None = None,
    trace_health_warnings: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Audit the deterministic candidate table with one agent session.

    The deterministic tiers resolve a kernel from its symbol alone; they cannot
    tell a file that defines a kernel from one that merely launches it, and they
    have no view of the model or how it is being served. This hands that table
    to an agent together with the paths of everything the run already produced,
    and folds back the revisions it can verify.

    Mandatory on the agent route, but never fatal: a definitive failure records
    an ``error``-severity trace-health warning and leaves the deterministic
    table standing. Losing the audit costs some candidates; failing the run
    would cost the hours of benchmarking that produced the trace.

    Args:
        run_dir: The per-run output directory.
        candidates: The finalized candidate rows, revised in place.
        args: Parsed CLI args (model name, framework, source root).
        log_path: Optional log file for diagnostics.
        trace_health_warnings: Warning sink surfaced to the Coordinator.

    Returns:
        dict[str, str]: Artifact paths produced by this stage.
    """
    artifacts: dict[str, str] = {}
    warnings = trace_health_warnings if trace_health_warnings is not None else []

    def _note(message: str) -> None:
        log.info("%s", message)
        if log_path:
            append_log(log_path, message)

    try:
        from _candidate_review_agent import (  # noqa: PLC0415
            RAW_CANDIDATES_FILENAME,
            REVISIONS_FILENAME,
            apply_revisions,
            run_candidate_review,
        )
    except ImportError as exc:  # pragma: no cover - packaging fault
        warnings.append(
            {
                "code": "candidate_review_unavailable",
                "severity": "error",
                "message": (
                    "The candidate review agent could not be imported "
                    f"({type(exc).__name__}); the candidate table is the "
                    "unreviewed deterministic result."
                ),
            }
        )
        return artifacts

    tracelens_dir = run_dir / "tracelens"
    raw_path = run_dir / RAW_CANDIDATES_FILENAME
    # Demangled here rather than in the session. Demangling a vendor symbol was
    # the one job that wanted a shell, and a shell cannot be confined to the run
    # directory -- so the host does it and the session keeps a read-only tool
    # surface. Written onto copies: the demangled name is an aid for the reader
    # of this table, not a candidate field, and the reviewed table downstream
    # must stay diffable against the deterministic one.
    payload_rows = [_with_demangled_symbol(c) for c in candidates if isinstance(c, dict)]
    routable = [c for c in payload_rows if c.get("reusable_native_kernel") is True]
    atomic_write_json(
        raw_path,
        {
            "model_name": args.model_name,
            "framework": args.framework,
            "source": "tracelens_analysis:deterministic",
            "hot_kernels": payload_rows,
            "routable_kernels": routable,
        },
    )
    artifacts["kernel_candidates_raw"] = str(raw_path)

    # Only ``analysis.md`` is a supported TraceLens output; everything else in
    # that directory is internal and may be removed without notice. The rest of
    # the list is Hyperloom's own or the model's, so it is ours to offer.
    #
    # Little is lost by not pointing at the sidecars: for every operator they
    # describe, ``analysis.md`` carries the same operand dims and launcher in
    # its own table, and for a graph-launched operator neither has anything --
    # the replay has no CPU-side parent op, so nothing recorded the arguments.
    reference_paths = {
        "source resolution audit": str(run_dir / _SOURCE_RESOLUTION_NAME),
        "tracelens report": str(tracelens_dir / "analysis.md"),
        "trace input manifest": str(run_dir / "trace_input_manifest.json"),
        "model directory": str(_RUNTIME_CONTEXT.get("model_path") or ""),
    }
    outcome = run_candidate_review(
        run_dir=run_dir,
        raw_candidates_path=raw_path,
        reference_paths={k: v for k, v in reference_paths.items() if v},
        framework_roots=kernel_search_roots(),
        context_block=_source_context_block(),
        log=_forward_to_log,
    )

    if not outcome.ok:
        warnings.append(
            {
                "code": "candidate_review_failed",
                "severity": "error",
                "status": outcome.status,
                "detail": outcome.detail,
                "message": (
                    f"The mandatory candidate review did not complete "
                    f"({outcome.status}: {outcome.detail}). kernel_candidates.json "
                    "is the unreviewed deterministic result; a wrongly resolved "
                    "kernel will not have been caught."
                ),
            }
        )
        _note(f"candidate review failed ({outcome.status}): {outcome.detail}")
        return artifacts

    op_cat_map = load_op_category_map(tracelens_dir / "perf_report_csvs")
    before_state = {str(c.get("kernel_id") or ""): str(c.get("source_file") or "") for c in candidates}
    protected_ids = {
        str(c.get("kernel_id") or "") for c in candidates if isinstance(c, dict) and _is_curated_resolution(c)
    }
    notes = apply_revisions(
        candidates,
        outcome.revisions,
        framework_roots=kernel_search_roots(),
        protected_ids=protected_ids,
    )
    changed = 0
    for item in candidates:
        if not isinstance(item, dict):
            continue
        kernel_id = str(item.get("kernel_id") or "")
        source_moved = str(item.get("source_file") or "") != before_state.get(kernel_id)
        staged = any(item.get(key) is not None for key in _REVIEW_STAGED_PROPOSALS)
        if not source_moved and not staged:
            continue
        # Two different revisions. A moved path invalidates everything derived
        # from the old one, so those are cleared and recomputed. A path that
        # stood still invalidates nothing about the source, and most of what the
        # finder recorded about it has no producer here -- so the proposals are
        # taken and the source judgments are left alone.
        if source_moved:
            _rederive_after_review(item, op_cat_map)
        else:
            _accept_review_proposals(item, op_cat_map)
        changed += 1

    revisions_path = run_dir / REVISIONS_FILENAME
    atomic_write_json(
        revisions_path,
        {
            "status": outcome.status,
            "revisions": outcome.revisions,
            "applied_notes": notes,
            "candidates_changed": changed,
            "raw_candidates": str(raw_path),
        },
    )
    artifacts["kernel_candidates_revisions"] = str(revisions_path)

    # Unconditionally, not only when ``changed`` is non-zero. The audit was
    # written during finalize, before this stage ran, so a review that moved a
    # ``source_file`` left the public artifact naming the old path while
    # kernel_candidates.json named the new one -- two answers to "where does
    # this kernel live", and the one a human reads was the stale one. Rebuilding
    # on every completed review keeps the two derived from the same table, and
    # costs one projection over rows already in memory.
    source_resolution_path = run_dir / _SOURCE_RESOLUTION_NAME
    if write_source_resolution_artifact(
        candidates,
        source_resolution_path,
        framework=args.framework or "",
        model_name=args.model_name or "",
        log_path=log_path,
    ):
        artifacts["kernel_source_resolution"] = str(source_resolution_path)

    _note(f"candidate review applied {changed} change(s) over {len(outcome.revisions)} revision(s)")
    for line in notes:
        _note(f"  review: {line}")
    return artifacts


def write_reports(
    run_dir: Path,
    *,
    trace_input_type: str,
    trace_files: list[Path],
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
    existing_report_path: Path | None = None,
    trace_health_warnings: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Write Hyperloom-owned sidecar JSONs and surface the upstream ``analysis.md``.

    ``analysis.md`` is owned by the TraceLens SDK orchestrator and not copied/aliased.

    Args:
        run_dir: The per-run output directory to write sidecars into.
        trace_input_type: Whether the trace input is a file or capture dir.
        trace_files: The resolved trace files for the manifest.
        candidates: The finalized hot-kernel candidate rows.
        args: Parsed CLI args carrying model/framework/platform settings.
        existing_report_path: Optional pre-existing report path to surface.
        trace_health_warnings: Optional trace-health warnings to record.

    Returns:
        A mapping of report name to its written/surfaced path.

    Raises:
        RuntimeError: When the orchestrator did not produce ``analysis.md``
            (rather than fabricating a report).
    """
    tracelens_dir = run_dir / "tracelens"
    (tracelens_dir / "system_findings").mkdir(parents=True, exist_ok=True)
    (tracelens_dir / "category_findings").mkdir(parents=True, exist_ok=True)
    enrich_candidates_with_runtime_metadata(candidates, args)

    manifest = {
        "trace_input": str(Path(args.trace_input).resolve()),
        "trace_input_type": trace_input_type,
        "trace_files": [str(p) for p in trace_files],
        "created_at": utc_now(),
    }
    # Aggregate reusable candidates into source-function task groups.
    source_root_str = getattr(args, "source_root", None)
    task_groups = build_task_groups(candidates, source_root=source_root_str)
    report = {
        "model_name": args.model_name,
        "framework": args.framework,
        "target_platform": args.target_platform,
        "analysis_mode": args.analysis_mode,
        "runtime_env": args.runtime_env,
        "trace_input_type": trace_input_type,
        "hot_kernels": candidates,
        "task_groups": task_groups,
        "source": "tracelens_analysis",
        "dry_run": args.dry_run,
    }
    atomic_write_json(run_dir / "trace_input_manifest.json", manifest)
    atomic_write_json(tracelens_dir / "tracelens_report.json", report)
    kernel_candidates_path = run_dir / "kernel_candidates.json"
    # ``hot_kernels`` is always the full ranked set; the reusable dispatch subset
    # is exposed as ``routable_kernels`` and non-routable dicts as ``skipped_kernels``.
    routable_candidates = [c for c in candidates if isinstance(c, dict) and c.get("reusable_native_kernel") is True]
    skipped_kernels = [c for c in candidates if isinstance(c, dict) and c.get("reusable_native_kernel") is not True]
    atomic_write_json(
        kernel_candidates_path,
        {
            **report,
            "hot_kernels": candidates,
            "routable_kernels": routable_candidates,
            "skipped_kernels": skipped_kernels,
            "task_groups": task_groups,
        },
    )

    # Per-run audit sidecar (tasks routed vs skipped w/ reason).
    summary = build_audit_summary(
        candidates,
        trace_input=str(Path(args.trace_input).resolve()),
        framework=str(args.framework or ""),
        target_platform=str(args.target_platform or ""),
        task_groups=task_groups,
        trace_health_warnings=trace_health_warnings,
    )
    summary_path = tracelens_dir / "summary.json"
    atomic_write_json(summary_path, summary)

    missing_trace_report = existing_report_path is None or not existing_report_path.exists()
    trace_quality_blocked = any(
        isinstance(w, dict) and w.get("code") == "trace_split_no_steady_state" for w in (trace_health_warnings or [])
    )
    if missing_trace_report:
        if not getattr(args, "dry_run", False):
            if trace_quality_blocked:
                # Refused to run on a raw/non-steady trace; leave trace_report_path empty.
                existing_report_path = None
            else:
                raise RuntimeError(
                    "TraceLens SDK orchestrator did not produce analysis.md "
                    f"(expected at {existing_report_path}); refusing to "
                    "fabricate a Markdown report. Inspect the TraceLens skill "
                    "log and report upstream if this is reproducible."
                )
        else:
            # ``--dry-run``: synthesize a tiny stub so existence checks pass.
            stub_md = tracelens_dir / "analysis.md"
            stub_md.write_text(
                "# TraceLens dry-run stub (no SDK orchestrator output)\n",
                encoding="utf-8",
            )
            existing_report_path = stub_md

    kernel_roofline_path = kernel_roofline_path_for_run(
        run_dir,
        filename=getattr(args, "roofline_output_name", "kernel_roofline.json") or "kernel_roofline.json",
    )
    kernel_roofline_payload = build_kernel_roofline_payload(
        trace_input=str(Path(args.trace_input).resolve()),
        trace_input_type=trace_input_type,
        analysis_md_path=(str(existing_report_path) if existing_report_path else ""),
        kernel_candidates_path=str(kernel_candidates_path),
        roofline_json_path=(str(Path(args.roofline_json).expanduser()) if getattr(args, "roofline_json", "") else ""),
        candidates=candidates,
    )
    atomic_write_json(kernel_roofline_path, kernel_roofline_payload)

    # Diffusion / scriptable workload-level roofline: aggregate the per-kernel
    # roofline into an end-to-end workload roofline. Best-effort sidecar; never
    # blocks the per-kernel report.
    diffusion_roofline_path = ""
    if _is_scriptable_framework(getattr(args, "framework", "")):
        try:
            tools_dir = str(Path(__file__).resolve().parent)
            if tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
            from diffusion_roofline import build_report as _build_diffusion_roofline  # noqa: WPS433
            from _denoise_steps import count_profiler_steps, resolve_perstep_divisor  # noqa: WPS433

            # Per-step divisor: an operator-declared count wins over the one
            # inferred from the trace, matching the bypass route.
            _num_steps = resolve_perstep_divisor(
                requested_steps=int(getattr(args, "num_denoise_steps", 0) or 0),
                inferred_steps=count_profiler_steps(getattr(args, "trace_input", "") or ""),
            )
            _diff_report = _build_diffusion_roofline(
                tracelens_dir / "perf_report_csvs",
                _num_steps,
                int(getattr(args, "top_k", 10) or 10),
            )
            # A-priori analytic compute ceiling (config/safetensors derived),
            # giving the workload roofline an absolute ideal-ms floor. Best-effort,
            # and only for frameworks whose denoiser config Hyperloom can read --
            # the trace-derived totals above need no such config and always ship.
            if _has_diffusion_ceiling(getattr(args, "framework", "")):
                try:
                    _model_dir = str(getattr(args, "model_path", "") or "").strip()
                    if not _model_dir:
                        _mn = str(getattr(args, "model_name", "") or "").strip()
                        if _mn:
                            for _cfg in _candidate_model_config_paths(_mn):
                                if Path(_cfg).is_file():
                                    _model_dir = str(Path(_cfg).parent)
                                    break
                    if _model_dir and Path(_model_dir).is_dir():
                        import diffusion_flops as _dflops  # noqa: WPS433

                        _gpu = str(getattr(args, "target_platform", "") or "mi355x").strip() or "mi355x"
                        _prec = str(getattr(args, "precision", "") or "bf16").strip() or "bf16"
                        _h = int(getattr(args, "height", 0) or 0)
                        _w = int(getattr(args, "width", 0) or 0)
                        _cfg_batch = int(getattr(args, "cfg_batch", 0) or 0)
                        _est = _dflops.analytic_ceiling(
                            _model_dir,
                            gpu_type=_gpu,
                            precision=_prec,
                            height=_h or 1024,
                            width=_w or 1024,
                            num_steps=_num_steps or None,
                            cfg_batch=_cfg_batch or None,
                        )
                        if _est:
                            _diff_report["analytic_ceiling"] = _est
                            _actual_us = float(_diff_report.get("totals", {}).get("sigma_actual_kernel_us", 0.0) or 0.0)
                            if _est.get("ideal_ms") and _actual_us > 0:
                                _diff_report["analytic_within_pct"] = round(
                                    _est["ideal_ms"] / (_actual_us / 1e3) * 100.0, 2
                                )
                except Exception as _exc:  # noqa: BLE001 — analytic ceiling is best-effort
                    _diff_report["analytic_ceiling_error"] = f"{type(_exc).__name__}: {_exc}"
            out = run_dir / "diffusion_roofline.json"
            atomic_write_json(out, _diff_report)
            diffusion_roofline_path = str(out)
            print(
                "[diffusion_roofline] "
                f"kernel_eff={_diff_report['totals']['kernel_roofline_efficiency']:.3f} "
                f"gpu_busy_ratio={_diff_report.get('gpu_busy_ratio')} "
                f"-> {out}"
            )
        except FileNotFoundError as exc:
            print(f"[diffusion_roofline] skipped: {exc}")
        except Exception as exc:  # noqa: BLE001 - best-effort sidecar
            print(f"[diffusion_roofline] skipped: {type(exc).__name__}: {exc}")

    artifact_paths = {
        "trace_input_manifest": str(run_dir / "trace_input_manifest.json"),
        "kernel_candidates": str(kernel_candidates_path),
        "kernel_roofline": str(kernel_roofline_path),
        "tracelens_report_json": str(tracelens_dir / "tracelens_report.json"),
        # Canonical Markdown exit is the orchestrator's analysis.md (surfaced, not aliased).
        "trace_report_path": str(existing_report_path) if existing_report_path else "",
        "tracelens_summary": str(summary_path),
    }
    if diffusion_roofline_path:
        artifact_paths["diffusion_roofline"] = diffusion_roofline_path
    return artifact_paths


def _default_workspace_path() -> str:
    """Resolve the default workspace root for ``--workspace-path``.

    Fallback order: ``$USER_DATA_PATH``, then legacy ``$WORKSPACE_PATH``, then
    ``_paths.workspace_root()`` (which warns once when ``$USER_DATA_PATH`` is unset).

    Returns:
        The resolved default workspace path.
    """
    user_data = os.environ.get("USER_DATA_PATH")
    if user_data:
        return user_data
    workspace = os.environ.get("WORKSPACE_PATH")
    if workspace:
        return workspace
    # Neither env set: route through the shared helper so the one-shot
    # "USER_DATA_PATH unset" warning fires.
    return workspace_root()


def main() -> int:
    """CLI entry point for the TraceLens analysis tool.

    Parses arguments, optionally splits the trace into a steady-state chunk,
    runs the TraceLens SDK orchestrator, extracts hot-kernel candidates,
    merges roofline data, and writes the run's report sidecars and status.

    Returns:
        int: ``0`` on success, ``1`` when the run failed (the error is also
            written to status and printed as JSON).
    """
    parser = argparse.ArgumentParser(description="Kernel-agent TraceLens analysis tool")
    parser.add_argument("--trace-input", required=True)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--framework", default="")
    parser.add_argument("--target-platform", default="MI355X")
    parser.add_argument("--analysis-mode", default="default")
    parser.add_argument("--runtime-env", default="local")
    parser.add_argument(
        "--source-root",
        default=os.environ.get("TRACELENS_SOURCE_ROOT", "") or None,
        help=(
            "Optional root directory against which TraceLens launcher "
            "paths (e.g. ``aiter/ops/rmsnorm.py(76): rmsnorm``) are "
            "resolved for AST-based function-line lookup. Used only by "
            "the PR-B source-function aggregation pass; absolute paths "
            "in the report don't need this. Defaults to "
            "$TRACELENS_SOURCE_ROOT when set."
        ),
    )
    parser.add_argument(
        "--workspace-path",
        default=_default_workspace_path(),
        help=(
            "Root the tool writes under (output lands at "
            "<workspace_path>/kernel-agent/runs/<session_id>/...). "
            "Defaults to $USER_DATA_PATH so every kernel-agent artefact "
            "stays inside the session dir; falls back to $WORKSPACE_PATH "
            "for legacy launchers, then to /workspace/hyperloom."
        ),
    )
    parser.add_argument(
        "--tracelens-root",
        default=os.environ.get("TRACELENS_ROOT", ""),
        help="TraceLens public checkout (TRACELENS_ROOT). Required: "
        "src/hyperloom/agents/kernel/scripts/install.sh exports it from "
        "kernel-agent.env.sh; pass --tracelens-root only when "
        "running outside the installer-managed env.",
    )
    parser.add_argument(
        "--tracelens-internal-root",
        default=os.environ.get("TRACELENS_INTERNAL_ROOT", DEFAULT_TRACELENS_INTERNAL_ROOT),
        help="Optional TraceLens-internal checkout (TRACELENS_INTERNAL_ROOT). "
        "Rehydration module; plumbed to run_tracelens_skill. "
        "Leave empty for the open-source-only report.",
    )
    parser.add_argument("--roofline-json", default="")
    parser.add_argument(
        "--num-denoise-steps",
        type=int,
        default=int(os.environ.get("HYPERLOOM_NUM_DENOISE_STEPS", "0") or 0),
        help=(
            "Denoise steps captured in the profiled window (scriptable/xDiT "
            "diffusion only). Enables per-denoise-step timings in the workload "
            "roofline sidecar. Env: HYPERLOOM_NUM_DENOISE_STEPS."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", ""),
        help=(
            "Local model directory. Selected config.json fields inform bounded "
            "kernel source resolution for every framework. For scriptable/xDiT "
            "diffusion it also enables the config/safetensors-derived analytic "
            "compute ceiling in the workload roofline sidecar. Falls back to "
            "resolving --model-name; env: MODEL_PATH."
        ),
    )
    parser.add_argument(
        "--precision",
        default="",
        help="Diffusion analytic-ceiling precision (bf16/fp8/fp16); default bf16.",
    )
    parser.add_argument(
        "--runtime-config",
        default="",
        help=("Materialized workload YAML used only to recover EXTRA_*_ARGS for bounded source-resolution context."),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Diffusion image height for the analytic ceiling (0 = estimator default).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Diffusion image width for the analytic ceiling (0 = estimator default).",
    )
    parser.add_argument(
        "--cfg-batch",
        type=int,
        default=0,
        help="Forwards per denoise step for the analytic ceiling (0 = family default).",
    )
    parser.add_argument(
        "--roofline-output-name",
        default="kernel_roofline.json",
        help="Report file name under reports/. Pass kernel_roofline_opt.json "
        "for the post-kernel-opt snapshot so it does not overwrite baseline.",
    )
    parser.add_argument(
        "--capture-folder",
        default=os.environ.get("TRACELENS_CAPTURE_FOLDER", ""),
        help=(
            "Optional graph-capture folder for TraceLens inference graph "
            "replay analysis. Also accepts env TRACELENS_CAPTURE_FOLDER."
        ),
    )
    parser.add_argument("--budget-minutes", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    default_llm_orchestrator = os.environ.get(
        "KERNEL_AGENT_USE_LLM_ORCHESTRATOR",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    parser.add_argument(
        "--use-llm-orchestrator",
        dest="use_llm_orchestrator",
        action="store_true",
        default=default_llm_orchestrator,
        help=(
            "Run TraceLens analysis-orchestrator skill through "
            "claude_agent_sdk to produce analysis.md, then parse hot kernels "
            "from it (default: env KERNEL_AGENT_USE_LLM_ORCHESTRATOR, on)."
        ),
    )
    parser.add_argument(
        "--no-llm-orchestrator",
        dest="use_llm_orchestrator",
        action="store_false",
        help=(
            "Disable the Claude SDK TraceLens skill runner. Production runs "
            "will fail rather than falling back to intermediate/CSV candidate "
            "parsers; --dry-run still uses the test-only raw parser."
        ),
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help=(
            "Disable TraceLens trace splitting (#127). When set, the raw "
            "filtered trace is fed directly to TraceLens; useful for debugging "
            "or when the splitter binary isn't available."
        ),
    )
    parser.add_argument(
        "--split-num-steps",
        type=int,
        default=int(os.environ.get("TRACELENS_SPLIT_NUM_STEPS", "32") or 32),
        help=(
            "Number of steady-state iterations for the splitter to extract "
            "(#127). Maps to --num-steps on TraceLens.TraceUtils."
            "split_inference_trace_annotation."
        ),
    )
    parser.add_argument(
        "--split-conc",
        default=os.environ.get("TRACELENS_SPLIT_CONC", "") or os.environ.get("CONC", ""),
        help=("Expected peak concurrency for the splitter (#127). Maps to --CONC. Defaults to $CONC when set."),
    )
    parser.add_argument(
        "--split-osl",
        default=os.environ.get("TRACELENS_SPLIT_OSL", "") or os.environ.get("OSL", ""),
        help=("Maximum output sequence length hint for the splitter (#127). Maps to --OSL. Defaults to $OSL when set."),
    )
    parser.add_argument(
        "--split-r",
        default=(os.environ.get("TRACELENS_SPLIT_R", "") or os.environ.get("RANDOM_RANGE_RATIO", "")),
        help=(
            "OSL window ratio R for the splitter (#194 §3). Maps to "
            "--R on TraceLens.TraceUtils.split_inference_trace_annotation. "
            "Pairs with --CONC / --OSL so mixed-window selection uses the "
            "benchmark-contract PD ratio instead of an empirical default. "
            "Defaults to $RANDOM_RANGE_RATIO when set; leave empty to let "
            "the splitter fall back to its built-in heuristic."
        ),
    )
    parser.add_argument(
        "--steady-state-mode",
        choices=("mixed", "decode_only", "prefilldecode"),
        default=(os.environ.get("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", "").strip() or "mixed"),
        help=(
            "Which of TraceLens splitter's three steady-state chunks to "
            "consume for the perf report. The splitter always produces all "
            "three (mixed / decode_only / prefilldecode); this flag picks "
            "ONE per TraceLens's design that the chunks are parallel "
            "view-of-the-same-trace, not a fallback ladder. "
            "Defaults to 'mixed' (representative DO:PD mix at ~max "
            "concurrency) which matches roofline-v2's default profiling "
            "intent. Switch to 'prefilldecode' when the workload is "
            "short / batched (NUM_PROMPTS << CONC*OSL) so prefill is "
            "burst-shaped and the mixed window degenerates to PD=0 -- "
            "TP=1 + CUDA-graph traces frequently hit this corner case "
            "because the decode region's GPU work is fully inside the "
            "graph replay and rocprofiler-sdk doesn't emit aggregate "
            "Dispatch Task events outside TP-multi-stream contexts, so "
            "the mixed chunk looks 99%% idle while the prefilldecode "
            "chunk carries the real GEMM / attention kernels. "
            "Switch to 'decode_only' when you specifically want the "
            "longest pure-decode region (decode-perf comparison runs). "
            "May also be set via env "
            "INFERENCE_OPTIMIZER_STEADY_STATE_MODE so the coordinator "
            "can re-issue roofline with a different mode after a "
            "steady_state_chunk_empty warning lands."
        ),
    )
    args = parser.parse_args()

    args.target_platform = normalize_platform(args.target_platform)

    # Give the model tiers the serving configuration. Which implementation a
    # kernel reaches depends on it, and forty lines of a candidate file cannot
    # convey a backend flag. EXTRA_*_ARGS is how the harness passes them; the
    # raw string is collected here but never forwarded, because it also carries
    # credentials and paths. _llm_source_context reduces it to allowlisted
    # backend selectors before anything reaches a model.
    inherited_server_args = " ".join(
        value
        for key, value in os.environ.items()
        if key.startswith("EXTRA_") and key.endswith("_ARGS") and value.strip()
    )
    materialized_server_args = _runtime_server_args_from_config(str(getattr(args, "runtime_config", "") or ""))
    set_runtime_context(
        model_path=str(getattr(args, "model_path", "") or ""),
        server_args=" ".join(value for value in (inherited_server_args, materialized_server_args) if value),
        framework=str(getattr(args, "framework", "") or ""),
        precision=str(getattr(args, "precision", "") or ""),
    )

    session_id = args.session_id or uuid.uuid4().hex[:12]
    run_id = f"tl-{uuid.uuid4().hex[:8]}"
    started_at = utc_now()
    # Keep each TraceLens invocation's artifacts in its own run subdirectory.
    ts_compact = started_at.replace("-", "").replace(":", "").split(".")[0]
    if not ts_compact.endswith("Z"):
        ts_compact = ts_compact + "Z"
    sub_dir = f"{ts_compact}_{run_id}"
    root = Path(args.workspace_path) / "kernel-agent"
    run_dir = root / "runs" / session_id / sub_dir
    log_path = run_dir / "logs" / "tracelens_analysis" / f"{run_id}.log"
    status_path = run_dir / "status" / "tracelens_analysis" / f"{run_id}.json"
    artifacts: dict[str, str] = {}
    agent_candidates: list[dict[str, Any]] | None = None
    agent_report_path: Path | None = None
    allow_empty_candidates = False
    orchestrator_mode = "inline"
    orchestrator_error = ""
    # Structured trace-health findings surfaced to the Coordinator.
    trace_health_warnings: list[dict[str, Any]] = []
    # Discovery is cached per run, so re-run it here: a process that outlives a
    # single run would otherwise keep the roots it saw when it first imported
    # this module, and miss a framework installed since.
    search_roots = refresh_kernel_search_roots()
    # Without a single searchable root every kernel resolves to "" and the whole
    # run reports zero routable candidates -- a host misconfiguration that reads
    # exactly like a trace with nothing worth optimizing. Say so up front.
    if not search_roots:
        trace_health_warnings.append(
            {
                "code": "no_framework_source_root",
                "severity": "error",
                "packages": list(_KERNEL_SOURCE_PACKAGES),
                "message": (
                    "No framework source root exists on this host (looked for "
                    f"{', '.join(_KERNEL_SOURCE_PACKAGES)}). Source resolution "
                    "cannot grep anything, so every hot kernel will be reported "
                    "as non-routable and kernel-opt will have nothing to "
                    "dispatch. Install the framework in this interpreter's "
                    "environment or point $FRAMEWORK_REPO_PATH at its checkout."
                ),
            }
        )

    try:
        update_status(
            status_path,
            state="running",
            current_step="discover_trace_input",
            log_path=log_path,
            artifact_paths=artifacts,
            run_id=run_id,
            started_at=started_at,
        )
        trace_input = Path(args.trace_input).expanduser().resolve()
        trace_input_type, trace_files = discover_trace_inputs(trace_input)
        append_log(log_path, f"trace_input_type={trace_input_type}")
        append_log(log_path, f"trace_files={len(trace_files)}")
        # The file the analysis will actually read. Discovery order picks the
        # default; the preflight below promotes whichever candidate it proves
        # carries GPU kernels, because passing the check on one file and then
        # analysing another is how an empty trace reaches TraceLens silently.
        #
        # Unconditional: discover_trace_inputs returns [trace_input] for a file
        # and raises FileNotFoundError for a directory with no traces, so the
        # list is never empty. Guarding it would type this as Path | None and
        # push that None through every downstream call for a branch that cannot
        # be taken.
        analysis_trace_path = trace_files[0]

        # Fail-fast when the input is nothing but CUDA-graph capture sidecars.
        #
        # Ahead of the kernel probe below because the two answer different
        # questions and only this one is reliable here. A capture records the
        # graph being built, so it carries a handful of stray kernels rather
        # than none -- 2 out of 1.49M events on the run that prompted this --
        # and a probe that asks "are there any kernels" therefore passes it and
        # hands the splitter a file with no iteration loop in it. The splitter
        # then fails with ``trace_split_no_steady_state``, which reads as "the
        # window was too short" and sends the next person to lengthen a capture
        # that was never a workload timeline to begin with.
        #
        # ``all`` rather than ``any``: a healthy profile writes the annotated
        # trace *beside* its sidecars, and discovery already sorts those first.
        # Only an input with nothing else in it is unanalysable.
        if not args.dry_run and trace_files:
            capture_root = _capture_classification_root(trace_input)
            if all(_is_capture_fragment(p, capture_root) for p in trace_files):
                raise RuntimeError(
                    "trace_input_capture_only: "
                    f"all {len(trace_files)} trace file(s) under {trace_input} "
                    "are CUDA-graph capture sidecars. "
                    "A capture records graph construction rather than "
                    "execution, so it holds no per-iteration annotations for "
                    "the steady-state splitter to cut on. "
                    "The upstream profile produced no annotated workload trace: "
                    "re-profile so the run writes one beside the sidecars. "
                    "Sidecars are a supported auxiliary input -- pass them as "
                    "--capture-folder alongside a real trace, never as "
                    "--trace-input."
                )

        # Fail-fast on CPU-only traces.
        #
        # Probes candidates in discovery order rather than only the first. A
        # single-file probe reports the capture directory as CPU-only whenever
        # the leading file happens to be a fragment with no kernels, and the
        # error it raises then blames the profiler for a capture that is sitting
        # in the same directory with thirty thousand kernel events in it.
        #
        # With size ordering the first candidate is normally the capture, so this
        # loop exits on one probe and the promotion below stays quiet. It earns
        # its keep on the layouts where it does not: a multi-rank capture whose
        # leading rank recorded nothing.
        if not args.dry_run and trace_files:
            kernel_event_count = 0
            probed: list[str] = []
            spent_bytes = 0
            for candidate in trace_files[:_KERNEL_PROBE_LIMIT]:
                # Each probe deserialises the whole file, so a directory of large
                # empty captures could otherwise turn a fast failure into a slow
                # one. Ordering puts the likeliest candidate first, so stopping
                # on a byte budget costs the unlikely tail, not the answer.
                if probed and spent_bytes >= _KERNEL_PROBE_BYTE_BUDGET:
                    probed.append(f"(stopped after {spent_bytes} bytes probed)")
                    break
                spent_bytes += _trace_file_size(candidate)
                readable, kernel_event_count = _count_kernels_if_readable(candidate)
                if not readable:
                    # Distinct from an empty trace on purpose: "unreadable" sends
                    # a reader to the file, "no kernels" sends them to the
                    # profiler, and conflating them is the misdirection this
                    # whole change exists to remove.
                    probed.append(f"{candidate.name}=unreadable")
                    continue
                probed.append(f"{candidate.name}={kernel_event_count}")
                if kernel_event_count:
                    analysis_trace_path = candidate
                    break
            append_log(
                log_path,
                f"trace_gpu_kernel_events={kernel_event_count} (probed={', '.join(probed)})",
            )
            if analysis_trace_path != trace_files[0]:
                promotion_warning: dict[str, Any] = {
                    "code": "trace_analysis_input_promoted",
                    "severity": "info",
                    "leading_candidate": trace_files[0].name,
                    "analysed": analysis_trace_path.name,
                    "probed": list(probed),
                    "detail": (
                        "the leading candidate carried no GPU kernel events or "
                        "could not be read; the analysis ran on the first "
                        "candidate that did"
                    ),
                }
                # Structured as well as logged: a run that changed its own input
                # has to be explicable from the artifacts, not only from a tool
                # log nobody keeps.
                trace_health_warnings.append(promotion_warning)
                artifacts["tracelens_analysis_input"] = str(analysis_trace_path)
                append_log(
                    log_path,
                    "trace_analysis_input promoted from "
                    f"{trace_files[0].name} to {analysis_trace_path.name} "
                    f"(probed={', '.join(probed)})",
                )
            if kernel_event_count == 0:
                raise RuntimeError(
                    "Trace contains zero GPU kernel events in any of "
                    f"{len(probed)} probed file(s) under {trace_input}: "
                    f"{', '.join(probed)}. Either the upstream profile run "
                    "captured CPU-only activity -- re-run profile with the "
                    "torch.profiler GPU activities enabled (no LD_PRELOAD "
                    "competing for ROCprofiler-SDK) -- or the traces listed as "
                    "'unreadable' above are truncated or corrupt, which is a "
                    "different problem in the same place."
                )

        if not args.dry_run:
            update_status(
                status_path,
                state="running",
                current_step="install_tracelens",
                log_path=log_path,
                artifact_paths=artifacts,
                run_id=run_id,
                started_at=started_at,
            )
            tl_root_arg = (args.tracelens_root or "").strip()
            if not tl_root_arg:
                raise SystemExit(
                    "TraceLens root not provided: set TRACELENS_ROOT in env "
                    "(src/hyperloom/agents/kernel/scripts/install.sh writes it to "
                    "kernel-agent.env.sh) or pass --tracelens-root."
                )
            tl_root = Path(tl_root_arg)
            # Internal extension is opt-in (non-empty --tracelens-internal-root / env).
            internal_root_arg = (args.tracelens_internal_root or "").strip()
            tl_internal_root: Path | None = Path(internal_root_arg) if internal_root_arg else None
            if not _tracelens_checkout_complete(tl_root) and _is_default_tracelens_root(tl_root):
                # The installer-managed pod-local checkout can vanish mid-run;
                # self-heal the default path. An operator override fails fast below.
                _ensure_tracelens_checkout(tl_root, log_path=log_path)
            if not tl_root.exists():
                raise FileNotFoundError(
                    f"TraceLens root not found: {tl_root} (set TRACELENS_ROOT or pass --tracelens-root)"
                )
            # A dir that exists but has neither git metadata nor the TraceLens
            # skill tree is unusable; fail fast.
            if not _tracelens_checkout_complete(tl_root):
                raise FileNotFoundError(
                    f"TraceLens root incomplete (not a git checkout): {tl_root} "
                    "(set TRACELENS_ROOT or pass --tracelens-root to a valid checkout)"
                )
            if tl_internal_root is not None and not tl_internal_root.exists():
                append_log(
                    log_path,
                    f"TraceLens-internal root not found: {tl_internal_root}; "
                    "falling back to open-source-only "
                    "(provide an existing internal checkout to enable)",
                )
                tl_internal_root = None
            if tl_internal_root is None:
                append_log(
                    log_path,
                    "TraceLens-internal: not provided (open-source-only; set TRACELENS_INTERNAL_ROOT to enable)",
                )
                os.environ.pop("TL_EXTENSION", None)
            run_command(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                cwd=tl_root,
                log_path=log_path,
                timeout_s=max(60, int(args.budget_minutes * 60)),
            )
            # Read and follow the analysis-orchestrator skill entry point.
            skill = tl_root / "TraceLens/Agent/Analysis/skills/analysis-orchestrator/SKILL.md"
            if not skill.exists():
                raise FileNotFoundError(f"TraceLens analysis-orchestrator skill not found: {skill}")
            append_log(log_path, f"TraceLens skill: {skill}")

            tracelens_dir = run_dir / "tracelens"
            tracelens_dir.mkdir(parents=True, exist_ok=True)

            # Backfill MAF for the open-source TraceLens path: measure the
            # missing arch/MAF spec on an idle GPU before report generation,
            # unless the internal extension (which backfills MAF) is enabled.
            update_status(
                status_path,
                state="running",
                current_step="populate_gpu_arch_json",
                log_path=log_path,
                artifact_paths=artifacts,
                run_id=run_id,
                started_at=started_at,
            )
            arch_benchmark_timeout_s = _resolve_arch_benchmark_timeout_s()
            gpu_arch_path = populate_gpu_arch_json(
                tracelens_root=tl_root,
                platform=args.target_platform,
                internal_extension_enabled=tl_internal_root is not None,
                log=lambda msg: append_log(log_path, msg),
                run_command=lambda cmd, *, cwd, timeout_s, env=None: run_command(
                    cmd,
                    cwd=cwd,
                    log_path=log_path,
                    timeout_s=timeout_s,
                    env=env,
                ),
                timeout_s=arch_benchmark_timeout_s,
            )
            if gpu_arch_path is not None:
                artifacts["tracelens_gpu_arch_json"] = str(gpu_arch_path)

            # Split the full-window filtered trace into steady-state chunks via
            # TraceLens's own splitter, since the perf report expects a single
            # steady-state chunk.
            # Whichever candidate the preflight proved has GPU kernels, which is
            # trace_files[0] unless it was promoted. Analysing a different file
            # from the one that passed the check would let an empty rank through
            # on a sibling's evidence.
            cli_trace_path = analysis_trace_path
            # The un-split source trace: analysis runs on the steady-state chunk
            # (cli_trace_path is reassigned below), but graph-capture health is a
            # whole-run property and must be read from the original trace -- the
            # chunk may drop the graph-launch runtime events the detector needs.
            raw_trace_path = analysis_trace_path
            trace_split_blocked = False
            if not args.skip_split:
                update_status(
                    status_path,
                    state="running",
                    current_step="split_trace",
                    log_path=log_path,
                    artifact_paths=artifacts,
                    run_id=run_id,
                    started_at=started_at,
                )
                split_dir = tracelens_dir / "trace_split"
                split_dir.mkdir(parents=True, exist_ok=True)
                # Cut the profiler-start transient before the splitter sees the
                # trace. --find-steady-state selects on load composition, not on
                # timing, so it will happily hand back a window whose first step
                # is a multi-second rank-arrival barrier -- and every downstream
                # percentage is then computed against that inflated denominator.
                # Trimming here keeps the splitter and TraceLens untouched.
                trimmed_trace = split_dir / (analysis_trace_path.name.replace(".trace.json", ".pretrimmed.trace.json"))
                split_num_steps = max(8, int(args.split_num_steps or 32))
                did_trim, pretrim_report = pretrim_startup_transient(
                    analysis_trace_path,
                    trimmed_trace,
                )
                pretrim_summary = dict(pretrim_report)
                pretrim_summary["applied"] = False
                # Only the splitter's input moves to the trimmed copy.
                # analysis_trace_path stays on the real capture: capture-folder
                # discovery resolves the graph-capture sidecar from the trace
                # file's own directory, and the split warnings name it as the
                # capture the operator profiled. Both would point into
                # trace_split/ if this were reassigned.
                split_input_path = analysis_trace_path
                if did_trim and pretrim_report["remaining_steps"] < split_num_steps:
                    # Trimming would leave the splitter fewer steps than it
                    # was asked for. A silently short window reads as a clean
                    # measurement; the untrimmed one at least shows the damage.
                    append_log(
                        log_path,
                        f"pretrim: only {pretrim_report['remaining_steps']} step(s) would "
                        f"remain < --split-num-steps {split_num_steps}; keeping untrimmed trace",
                    )
                    pretrim_summary["reason"] = "insufficient_remaining_steps"
                    pretrim_summary["min_remaining_steps"] = split_num_steps
                    # The rejected copy is nearly the size of the capture; it is
                    # not the input to anything now, so do not leave it behind.
                    with contextlib.suppress(OSError):
                        trimmed_trace.unlink(missing_ok=True)
                elif did_trim:
                    append_log(
                        log_path,
                        f"pretrim: dropped {pretrim_report['dropped_steps']} leading "
                        f"{pretrim_report['dropped_phase']} step(s), "
                        f"{pretrim_report['dropped_ms']:.1f} ms, "
                        f"{pretrim_report['outlier_ratio']:.0f}x phase median "
                        f"{pretrim_report['median_step_ms']:.1f} ms; "
                        f"{pretrim_report['remaining_steps']} step(s) remain",
                    )
                    pretrim_summary["applied"] = True
                    split_input_path = trimmed_trace
                    artifacts["tracelens_pretrimmed_trace"] = str(trimmed_trace)
                elif pretrim_report.get("reason") != "no_leading_outlier":
                    append_log(log_path, f"pretrim: not applied ({pretrim_report.get('reason')})")
                if pretrim_report.get("reason") == "too_many_leading_outliers":
                    # A run of leading outliers is not a transient to trim but a
                    # workload that never settled. Route it to the health
                    # warnings so it reaches the report, not just the run log.
                    trace_health_warnings.append(
                        _build_pretrim_no_steady_state_warning(
                            trace_input=analysis_trace_path,
                            steps=int(pretrim_report["steps"]),
                            leading_outliers=int(pretrim_report["leading_outliers"]),
                            max_dropped_steps=int(pretrim_report["max_dropped_steps"]),
                            outlier_factor=float(pretrim_report["outlier_factor"]),
                        )
                    )
                # Persist as an artifact, not just a log line: the status
                # file is rewritten by every later step, so a diagnostic
                # parked there is gone by the time anyone reads the report.
                pretrim_path = tracelens_dir / "pretrim.json"
                atomic_write_json(pretrim_path, pretrim_summary)
                artifacts["tracelens_pretrim"] = str(pretrim_path)
                # --find-steady-state writes the three *_steady_state_* chunks; --R feeds PD-ratio selection.
                split_cmd = [
                    sys.executable,
                    "-m",
                    "TraceLens.TraceUtils.split_inference_trace_annotation",
                    str(split_input_path),
                    "-o",
                    str(split_dir),
                    "--find-steady-state",
                    "--num-steps",
                    str(split_num_steps),
                ]
                conc = args.split_conc or os.environ.get("CONC", "").strip()
                if str(conc).strip():
                    split_cmd += ["--CONC", str(conc).strip()]
                osl = args.split_osl or os.environ.get("OSL", "").strip()
                if str(osl).strip():
                    split_cmd += ["--OSL", str(osl).strip()]
                # Only pass --R when provided so the splitter's default keeps working.
                r_raw = args.split_r or os.environ.get(
                    "RANDOM_RANGE_RATIO",
                    "",
                )
                r_str = str(r_raw).strip()
                if r_str:
                    try:
                        float(r_str)
                    except ValueError:
                        append_log(
                            log_path,
                            f"split_trace: ignoring non-numeric --R={r_str!r}",
                        )
                    else:
                        split_cmd += ["--R", r_str]
                split_rc = run_command(
                    split_cmd,
                    cwd=tl_root,
                    log_path=log_path,
                    timeout_s=max(60, int(args.budget_minutes * 60)),
                )

                # The three chunks are parallel views; the consumer picks ONE via
                # --steady-state-mode and we hard-fail when it is missing/empty.
                def _collect(prefix: str) -> list[Path]:
                    """Collect splitter chunk files for a steady-state prefix.

                    Args:
                        prefix (str): Chunk prefix (``mixed``, ``decode_only``,
                            or ``prefilldecode``).

                    Returns:
                        list[Path]: Sorted chunk files matching the prefix
                            across known trace extensions.
                    """
                    out: list[Path] = []
                    for ext in ("trace.json.gz", "json.gz", "trace.json", "json"):
                        out.extend(sorted(split_dir.rglob(f"{prefix}_steady_state_*.{ext}")))
                    return out

                mixed_chunks = _collect("mixed")
                decode_chunks = _collect("decode_only")
                prefill_chunks = _collect("prefilldecode")
                # Splitter produced nothing -> trace_split_no_steady_state failure.
                if split_rc != 0 or not (mixed_chunks or decode_chunks or prefill_chunks):
                    warning = _build_trace_split_warning(
                        trace_input=analysis_trace_path,
                        split_dir=split_dir,
                        split_rc=split_rc,
                        mixed_count=len(mixed_chunks),
                        decode_count=len(decode_chunks),
                        prefilldecode_count=len(prefill_chunks),
                    )
                    trace_health_warnings.append(warning)
                    append_log(
                        log_path,
                        f"WARNING: trace split unavailable "
                        f"(rc={split_rc}, mixed={len(mixed_chunks)}, "
                        f"decode_only={len(decode_chunks)}, "
                        f"prefilldecode={len(prefill_chunks)}); "
                        "refusing raw-trace fallback and returning "
                        "trace_split_no_steady_state warning",
                    )
                    raise RuntimeError(
                        "trace_split_no_steady_state: TraceLens splitter "
                        "produced no steady-state chunks; refusing to run "
                        "TraceLens analysis on the raw trace"
                    )

                _mode_to_chunks = {
                    "mixed": ("mixed_steady_state", mixed_chunks),
                    "decode_only": ("decode_only_steady_state", decode_chunks),
                    "prefilldecode": ("prefilldecode_steady_state", prefill_chunks),
                }
                chunk_label, selected_chunks = _mode_to_chunks[args.steady_state_mode]
                if not selected_chunks:
                    # Requested mode produced no chunk; emit a warning for re-issue.
                    warning = {
                        "code": "steady_state_chunk_missing",
                        "severity": "blocking",
                        "requested_mode": args.steady_state_mode,
                        "requested_chunk_label": chunk_label,
                        "available_modes": [m for m, (_, ch) in _mode_to_chunks.items() if ch],
                        "remediation": (
                            "Re-issue roofline with env "
                            "INFERENCE_OPTIMIZER_STEADY_STATE_MODE set to one "
                            "of the available_modes (or pass --steady-state-mode "
                            "directly when invoking tracelens_analysis.py)."
                        ),
                        "trace_input": str(analysis_trace_path),
                        "split_dir": str(split_dir),
                    }
                    trace_health_warnings.append(warning)
                    append_log(
                        log_path,
                        f"ERROR: --steady-state-mode={args.steady_state_mode} "
                        f"requested but no {chunk_label}_*.json[.gz] in "
                        f"{split_dir} (mixed={len(mixed_chunks)}, "
                        f"decode_only={len(decode_chunks)}, "
                        f"prefilldecode={len(prefill_chunks)}); refusing "
                        "silent fallback per TraceLens parallel-chunk design",
                    )
                    raise RuntimeError(
                        f"steady_state_chunk_missing: requested "
                        f"--steady-state-mode={args.steady_state_mode} but "
                        f"splitter produced no matching chunk under "
                        f"{split_dir}"
                    )

                # Data-validity gate: the selected chunk must have observable GPU work.
                cli_trace_path = selected_chunks[0]
                empty_chunk_warning = _check_selected_chunk_has_gpu_events(
                    split_dir=split_dir,
                    selected_chunk=cli_trace_path,
                    mode=args.steady_state_mode,
                    available_modes=_mode_to_chunks,
                )
                if empty_chunk_warning is not None:
                    trace_health_warnings.append(empty_chunk_warning)
                    append_log(
                        log_path,
                        f"ERROR: --steady-state-mode={args.steady_state_mode} "
                        f"selected chunk {cli_trace_path.name} has "
                        f"num_gpu_events={empty_chunk_warning['num_gpu_events']} "
                        f"/ gpu_busy_duration={empty_chunk_warning['gpu_busy_duration']}"
                        f"; refusing to feed an empty chunk to TraceLens "
                        "analysis (would produce misleading "
                        "'Compute %=~0, Idle %=~100' Executive Summary)",
                    )
                    raise RuntimeError(
                        f"steady_state_chunk_empty: requested "
                        f"--steady-state-mode={args.steady_state_mode} but the "
                        f"selected chunk has zero GPU events; available "
                        f"non-empty modes: "
                        f"{empty_chunk_warning['non_empty_modes']}"
                    )

                # Quality gate: a non-empty but low-busy chunk emits
                # steady_state_chunk_low_quality for the same retry path.
                low_quality_warning = _check_selected_chunk_has_gpu_events_quality(
                    split_dir=split_dir,
                    selected_chunk=cli_trace_path,
                    mode=args.steady_state_mode,
                    available_modes=_mode_to_chunks,
                )
                if low_quality_warning is not None:
                    trace_health_warnings.append(low_quality_warning)
                    append_log(
                        log_path,
                        f"ERROR: --steady-state-mode={args.steady_state_mode} "
                        f"selected chunk {cli_trace_path.name} is "
                        f"non-empty but low-quality: busy_ratio="
                        f"{low_quality_warning['busy_ratio'] * 100:.3f}% "
                        f"(threshold "
                        f"{low_quality_warning['threshold'] * 100:.0f}%); "
                        f"alternate modes with higher busy_ratio: "
                        f"{low_quality_warning['non_empty_modes']}. "
                        "Refusing to analyze (would yield misleading "
                        "high-idle Executive Summary). "
                        "The roofline executor will auto-retry trace_analyze "
                        "with an adjusted steady-state window — this is a "
                        "self-healing step on the same captured trace, not a hang.",
                    )
                    raise RuntimeError(
                        f"steady_state_chunk_low_quality: requested "
                        f"--steady-state-mode={args.steady_state_mode} chunk "
                        f"busy_ratio="
                        f"{low_quality_warning['busy_ratio'] * 100:.3f}%; "
                        f"better alternates: "
                        f"{low_quality_warning['non_empty_modes']}"
                    )

                artifacts["tracelens_trace_split_dir"] = str(split_dir)
                artifacts["tracelens_steady_state_trace"] = str(cli_trace_path)
                append_log(
                    log_path,
                    f"trace split OK: mixed={len(mixed_chunks)} "
                    f"decode_only={len(decode_chunks)} "
                    f"prefilldecode={len(prefill_chunks)}; "
                    f"--steady-state-mode={args.steady_state_mode} -> "
                    f"using {cli_trace_path.name} for perf report",
                )

            # Graph-capture folder for TraceLens inference graph replay analysis.
            trace_input_path = Path(args.trace_input).expanduser().resolve()
            capture_folder: Path | None = (
                Path(args.capture_folder).expanduser().resolve()
                if args.capture_folder
                # The analysed trace, not the leading candidate: the helper looks
                # for capture_traces/ beside the file it is given, and after a
                # cross-directory promotion those are different places.
                else discover_capture_folder(trace_input_path, [analysis_trace_path])
            )
            if capture_folder:
                append_log(
                    log_path,
                    f"capture_folder resolved: {capture_folder} (exists={capture_folder.is_dir()})",
                )

            if args.use_llm_orchestrator and not trace_split_blocked:
                update_status(
                    status_path,
                    state="running",
                    current_step="run_tracelens_sdk_orchestrator",
                    log_path=log_path,
                    artifact_paths=artifacts,
                    run_id=run_id,
                    started_at=started_at,
                )
                try:
                    skill_result = asyncio.run(
                        run_tracelens_skill(
                            skill_path=skill,
                            trace_path=cli_trace_path,
                            output_dir=tracelens_dir,
                            tracelens_root=tl_root,
                            tracelens_internal_root=tl_internal_root,
                            platform=args.target_platform,
                            framework=args.framework,
                            analysis_mode=args.analysis_mode,
                            capture_folder=capture_folder,
                            budget_minutes=args.budget_minutes,
                            model=_resolve_tracelens_model(),
                            log=lambda msg: append_log(log_path, msg),
                        )
                    )
                    artifacts.update(skill_result.artifact_paths)
                    agent_report_path = skill_result.report_path
                    orchestrator_mode = skill_result.runner

                    raw_agent_candidates = []
                    report_source = ""
                    idle_pct_value = extract_idle_pct_from_analysis_md(
                        skill_result.report_path,
                    )
                    idle_pct_threshold, high_idle_warning, graph_under_recorded_warning = (
                        _evaluate_idle_gate_with_graph_guard(
                            idle_pct_value,
                            skill_result.report_path,
                            raw_trace_path,
                        )
                    )
                    compute_pct_value = extract_compute_pct_from_analysis_md(
                        skill_result.report_path,
                    )
                    exposed_comm_pct_value = extract_exposed_comm_pct_from_analysis_md(
                        skill_result.report_path,
                    )
                    compute_pct_threshold, low_compute_warning = _evaluate_low_compute_gate(
                        compute_pct_value,
                        exposed_comm_pct_value,
                        skill_result.report_path,
                    )
                    if graph_under_recorded_warning is not None:
                        # Under-recording deflates every recorded share alike, so
                        # the compute share is as unreliable as idle% here.
                        low_compute_warning = None
                        trace_health_warnings.append(graph_under_recorded_warning)
                        append_log(
                            log_path,
                            "TraceLens high idle% is a graph under-recording "
                            "artifact (profiler captured ~1 of N graph replays); "
                            "skipping the idle/compute gates and keeping hot_kernels[].",
                        )
                    if high_idle_warning is not None or low_compute_warning is not None:
                        agent_candidates = []
                        allow_empty_candidates = True
                        # Both gates can fire on one window (95% idle AND 3%
                        # compute is a real shape), so accumulate rather than
                        # let the second suppression erase the first, matching
                        # the "+".join the non-suppressed path already uses.
                        skipped_sources: list[str] = []
                        if high_idle_warning is not None:
                            assert idle_pct_value is not None
                            trace_health_warnings.append(high_idle_warning)
                            skipped_sources.append("skipped:high_gpu_idle_pct")
                            append_log(
                                log_path,
                                f"TraceLens Executive Summary reports "
                                f"Idle % = {idle_pct_value:.2f}% (threshold "
                                f"{idle_pct_threshold:.2f}%); suppressing "
                                "hot_kernels[] — kernel rewriting cannot move "
                                "end-to-end latency in the high-idle regime. "
                                "Coordinator will see this in "
                                "trace_health_warnings[] and route to "
                                "parameter optimization.",
                            )
                        if low_compute_warning is not None:
                            assert compute_pct_value is not None
                            trace_health_warnings.append(low_compute_warning)
                            skipped_sources.append("skipped:low_gpu_compute_pct")
                            append_log(
                                log_path,
                                f"TraceLens Executive Summary reports "
                                f"Compute % = {compute_pct_value:.2f}% "
                                f"(threshold {compute_pct_threshold:.2f}%); "
                                "suppressing hot_kernels[] — a kernel rewrite "
                                "is bounded by the compute share and cannot "
                                "move end-to-end latency here. Coordinator "
                                "will see this in trace_health_warnings[] and "
                                "route to comm/parameter optimization.",
                            )
                        report_source = "+".join(skipped_sources)
                    else:
                        if idle_pct_value is not None and graph_under_recorded_warning is None:
                            append_log(
                                log_path,
                                f"TraceLens Executive Summary: "
                                f"Idle % = {idle_pct_value:.2f}% "
                                f"(threshold {idle_pct_threshold:.2f}%) — "
                                "below gate, continuing with kernel "
                                "candidate extraction",
                            )
                        report_cands = parse_analysis_md(
                            skill_result.report_path,
                            _default_top_k(),
                        )
                        # Defense-in-depth: recover any high-GPU-time op that
                        # TraceLens filed without a reasoning-candidate block from
                        # the per-op ranking sidecar. analysis.md stays primary.
                        fallback_cands = recover_other_bucket_candidates(
                            skill_result.output_dir,
                            report_cands,
                            top_k=_default_top_k(),
                            total_window_us=_extract_total_time_us_from_gpu_timeline(
                                skill_result.output_dir,
                            ),
                            log=lambda msg: append_log(log_path, msg),
                        )
                        if fallback_cands:
                            report_cands = report_cands + fallback_cands
                        raw_agent_candidates = _inject_collective_candidates(
                            skill_result.output_dir,
                            report_cands,
                            log_path=log_path,
                            health_warnings=trace_health_warnings,
                        )
                        collective_injected = len(raw_agent_candidates) > len(report_cands)
                        if raw_agent_candidates:
                            source_parts = ["analysis.md"]
                            if fallback_cands:
                                source_parts.append("other_bucket_fallback")
                            if collective_injected:
                                source_parts.append("nccl_summary")
                            report_source = "+".join(source_parts)
                        else:
                            agent_candidates = []
                            allow_empty_candidates = True
                            append_log(
                                log_path,
                                "TraceLens analysis.md had no Detailed "
                                "Analysis compute candidate blocks "
                                "(v0.3 contract: analysis.md is the single "
                                "source of truth) and the other-bucket "
                                "fallback found no high-GPU-time op to "
                                "recover. Producing empty hot_kernels[] — "
                                "downstream Coordinator will route to "
                                "params/backends.",
                            )

                    if raw_agent_candidates:
                        # Use whole-trace GPU time as the gpu_pct denominator,
                        # falling back to the candidate sum only when
                        # gpu_timeline.csv is missing.
                        total_dur = _extract_total_time_us_from_gpu_timeline(skill_result.output_dir) or sum(
                            float(c.get("duration_us") or 0) for c in raw_agent_candidates
                        )
                        agent_candidates = _finalize_candidates(
                            raw_agent_candidates,
                            total_dur=total_dur or None,
                            perf_report_csv_dir=(skill_result.output_dir / "perf_report_csvs"),
                            framework=args.framework or None,
                            trace_files=trace_files,
                            log_path=log_path,
                            source_resolution_out=(run_dir / _SOURCE_RESOLUTION_NAME),
                            model_name=args.model_name,
                        )
                        append_log(
                            log_path,
                            f"TraceLens SDK orchestrator produced "
                            f"{len(agent_candidates)} hot kernels "
                            f"(source={report_source})",
                        )
                except Exception as exc:  # noqa: BLE001
                    orchestrator_error = f"{type(exc).__name__}: {exc}"
                    append_log(
                        log_path,
                        f"WARNING: TraceLens SDK orchestrator failed; "
                        f"not falling back to intermediate/CSV candidate "
                        f"parsers: {type(exc).__name__}: {exc}",
                    )

            if agent_candidates is None:
                raise RuntimeError(
                    "TraceLens analysis.md was not produced; refusing to "
                    "fall back to priority_data/category_data/CSV candidate "
                    "parsers because analysis.md is the single source of truth."
                )
        else:
            append_log(log_path, "[dry-run] skipping TraceLens install and external CLI")

        update_status(
            status_path,
            state="running",
            current_step="extract_hot_kernels",
            log_path=log_path,
            artifact_paths=artifacts,
            run_id=run_id,
            started_at=started_at,
        )
        # Production candidate extraction is analysis.md-only.
        candidates = agent_candidates
        if candidates:
            append_log(
                log_path,
                f"hot kernels from TraceLens SDK orchestrator ({len(candidates)})",
            )
        if not candidates:
            if allow_empty_candidates:
                # Routing signal (high idle / TraceLens failure): keep candidates
                # empty so the Coordinator pivots to params/backends.
                candidates = []
                append_log(
                    log_path,
                    "TraceLens produced no kernel candidates; returning "
                    "empty hot_kernels[] without fallback so params/backends "
                    "optimization can continue.",
                )
            elif args.dry_run:
                # Test-only path: parse the raw trace so unit tests can exercise extraction.
                append_log(
                    log_path,
                    "dry-run: parsing raw trace for hot kernels (production code path raises here — see #203)",
                )
                candidates = analyze_trace_files(trace_files, _default_top_k())
            else:
                raise RuntimeError(
                    "No hot-kernel candidates produced by any TraceLens "
                    "analysis.md path. Refusing intermediate/CSV/raw-trace "
                    "fallbacks because analysis.md is the single source of "
                    "truth. Inspect the TraceLens skill log and report "
                    "upstream if reproducible."
                )
        roofline_by_name = load_roofline_results(args.roofline_json)
        if roofline_by_name:
            append_log(log_path, f"merged roofline results: {len(roofline_by_name)} kernels")
        merge_roofline_into_candidates(candidates, roofline_by_name)
        source_resolution_path = run_dir / _SOURCE_RESOLUTION_NAME
        if _KSC is not None:
            if not source_resolution_path.is_file():
                write_source_resolution_artifact(
                    candidates,
                    source_resolution_path,
                    framework=args.framework or "",
                    model_name=args.model_name or "",
                    log_path=log_path,
                )
            if source_resolution_path.is_file():
                artifacts["kernel_source_resolution"] = str(source_resolution_path)
        if args.dry_run:
            # A dry run plans; it must not spend an agent session, wait out the
            # session timeout, or read the framework tree to audit a table
            # nobody will dispatch from.
            append_log(log_path, "candidate review skipped: --dry-run")
        else:
            artifacts.update(
                run_candidate_review_stage(
                    run_dir,
                    candidates=candidates,
                    args=args,
                    log_path=log_path,
                    trace_health_warnings=trace_health_warnings,
                )
            )
        artifacts.update(
            write_reports(
                run_dir,
                trace_input_type=trace_input_type,
                trace_files=trace_files,
                candidates=candidates,
                args=args,
                existing_report_path=agent_report_path,
                trace_health_warnings=trace_health_warnings,
            )
        )
        if args.roofline_json:
            artifacts["roofline_json"] = str(Path(args.roofline_json).expanduser())
        artifacts["cli_log_path"] = str(log_path)
        artifacts["status_path"] = str(status_path)

        # Surface the contracted ``analysis.md`` exit path alongside hot_kernels.
        analysis_report_path = ""
        for cand_key in ("tracelens_agent_report", "trace_report_path"):
            if artifacts.get(cand_key):
                analysis_report_path = str(artifacts[cand_key])
                break

        result = {
            "tool": "tracelens_analysis",
            "session_id": session_id,
            "run_id": run_id,
            "trace_input_type": trace_input_type,
            "hot_kernels": candidates,
            "trace_report_path": artifacts["trace_report_path"],
            "analysis_report_path": analysis_report_path,
            "cli_log_path": str(log_path),
            "status_path": str(status_path),
            "artifact_paths": artifacts,
            "orchestrator_mode": orchestrator_mode,
            "orchestrator_error": orchestrator_error,
            # Trace-quality findings surfaced to the Coordinator (empty = nothing wrong).
            "trace_health_warnings": trace_health_warnings,
        }
        atomic_write_json(
            run_dir / "session_state.json",
            {
                "session_id": session_id,
                "last_tool": "tracelens_analysis",
                "last_run_id": run_id,
                "updated_at": utc_now(),
                "model_name": args.model_name,
                "framework": args.framework,
            },
        )
        update_status(
            status_path,
            state="succeeded",
            current_step="done",
            log_path=log_path,
            artifact_paths=artifacts,
            run_id=run_id,
            started_at=started_at,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        append_log(log_path, f"[error] {type(exc).__name__}: {exc}")
        update_status(
            status_path,
            state="failed",
            current_step="failed",
            log_path=log_path,
            artifact_paths=artifacts,
            run_id=run_id,
            started_at=started_at,
            error=f"{type(exc).__name__}: {exc}",
        )
        # Include trace_health_warnings accumulated pre-exception for auto-recovery.
        print(
            json.dumps(
                {
                    "tool": "tracelens_analysis",
                    "session_id": session_id,
                    "run_id": run_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "cli_log_path": str(log_path),
                    "status_path": str(status_path),
                    "trace_health_warnings": trace_health_warnings,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
