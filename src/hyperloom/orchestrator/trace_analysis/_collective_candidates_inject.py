# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Collective (RCCL/NCCL) kernels as candidates the analysis can act on."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from common.env import env_flag

from ._io_utils import append_log
from ._kernel_classification import _aiter_csrc_root
from ._nccl_summary_candidates import extract_collective_candidates
from ._trace_shapes import _dtypes_from_shapes

log = logging.getLogger(__name__)


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
                return {"call_num": int(item.get("call_count") or 1), "shape": text}
    return None


def _workload_dtypes(item: dict[str, Any]) -> list[str]:
    """Return explicit or shape-derived input dtypes for one workload."""
    values = item.get("input_dtypes") or _dtypes_from_shapes(item.get("shapes") or [])
    return [str(value).strip() for value in values if isinstance(value, str) and value.strip()]


def _trace_name(item: dict[str, Any]) -> str:
    """Return the normalized trace name for one candidate."""
    return str(item.get("name") or "").strip().lower()


def _has_workload(item: dict[str, Any]) -> bool:
    """Return whether the trace row carries driver inputs."""
    return bool(item.get("input_shapes") or item.get("shapes")) and bool(_workload_dtypes(item))


def _is_all_reduce_name(item: dict[str, Any]) -> bool:
    """Return whether a trace name carries all-reduce semantics."""
    return any(tag in _trace_name(item) for tag in ("all_reduce", "allreduce", "cross_device_reduce"))


def _workload_family(item: dict[str, Any]) -> str:
    """Collapse prefill and decode rows from one profiled wrapper."""
    name = _trace_name(item)
    if name.startswith(("sglang_profiler::", "vllm_profiler::")):
        name = name.split("->", 1)[0]
    return re.sub(r"\s+\((?:prefill|decode)\)\s*$", "", name)


def _attach_workload(target: dict[str, Any], donors: list[dict[str, Any]]) -> None:
    """Attach the donors' traced workload: one donor lends all of it, several their distinct first inputs."""
    if len(donors) == 1:
        donor = donors[0]
        for key in ("input_shapes", "shapes", "input_dtypes", "shape_provenance"):
            if donor.get(key):
                target[key] = donor[key]
        if not target.get("input_dtypes"):
            target["input_dtypes"] = _workload_dtypes(donor)
        target["workload_source_kernel"] = str(donor.get("name") or "")
        return
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
        (str(donor.get("shape_provenance") or "") for donor in donors if donor.get("shape_provenance")), ""
    )
    if provenance:
        target["shape_provenance"] = provenance
    target["workload_source_kernels"] = [str(donor.get("name") or "") for donor in donors]
    hottest = max(donors, key=lambda donor: float(donor.get("duration_us") or 0.0))
    target["workload_source_kernel"] = str(hottest.get("name") or "")


def _attach_collective_workloads(
    existing: list[dict[str, Any]],
    extracted: list[dict[str, Any]],
    note: Callable[[str], None],
) -> list[dict[str, Any]]:
    """Give each source-resolved collective its traced workload; return the rows to append.

    A collective that matches a trace row by name updates that row in place. Otherwise it is appended only when it
    can borrow the trace's sole all-reduce workload family, which ``HYPERLOOM_COLLECTIVE_ALLOW_INFERRED_SHAPES``
    must opt into.
    """
    by_name = {_trace_name(item): item for item in existing if _trace_name(item)}
    families: dict[str, list[dict[str, Any]]] = {}
    for row in existing:
        if _has_workload(row) and _is_all_reduce_name(row):
            families.setdefault(_workload_family(row), []).append(row)
    sole_family: list[dict[str, Any]] = []
    if len(families) == 1 and env_flag("HYPERLOOM_COLLECTIVE_ALLOW_INFERRED_SHAPES"):
        sole_family = next(iter(families.values()))
    appended: list[dict[str, Any]] = []
    for item in extracted:
        exact = by_name.get(_trace_name(item))
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
            donors, borrowed = [exact], False
        elif sole_family and _is_all_reduce_name(item):
            # Shapes are inferred from the sole all-reduce family, valid only
            # because the symbol is itself an all-reduce. Handing the driver
            # shapes the traced kernel never ran would yield confident SNR and
            # latency for a workload that never existed, so ``shape_provenance``
            # travels with them to mark the values attributed, not observed.
            donors, borrowed = sole_family, True
        else:
            donors, borrowed = [], False
        if donors:
            _attach_workload(target, donors)
        if borrowed:
            target["shape_provenance"] = "borrowed_sole_all_reduce_family"
            note(
                "nccl_summary: attributing the trace's only all-reduce workload "
                f"to {item.get('source_function')!r}; shapes are inferred"
            )
        if exact is None:
            if not donors:
                note(
                    "nccl_summary: no unique traced all-reduce workload for "
                    f"{item.get('source_function')!r}; dropping candidate"
                )
                continue
            appended.append(target)
            by_name[_trace_name(target)] = target
    return appended


def _inject_collective_candidates(
    tracelens_dir: Path,
    candidates: list[dict[str, Any]],
    *,
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
    existing = [dict(item) for item in candidates]
    aiter_csrc = _aiter_csrc_root().rstrip("/")
    roots = [aiter_csrc] if aiter_csrc and Path(aiter_csrc).is_dir() else []
    messages: list[str] = []

    def _skip(code: str, detail: str) -> list[dict[str, Any]]:
        """Record a visible reason the collective lane got no candidate."""
        message = f"nccl_summary: {detail}; skipping injection"
        log.warning(message)
        if log_path is not None:
            # The per-symbol notes are the only record of WHICH symbol failed and
            # they are normally flushed at the end, which a skip never reaches.
            for note in messages:
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
        )
    appended = _attach_collective_workloads(existing, extracted, messages.append)
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
