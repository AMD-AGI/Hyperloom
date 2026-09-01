#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel optimization tool for the resident Kernel-agent skill."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

# Self-location so sibling modules import both as package members and as scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _io_utils import (  # noqa: E402
    append_jsonl,
    append_log,
    atomic_write_json,
    read_last_lines,
    safe_float,
    source_text_looks_complete,
    utc_now,
)
from _invocation_spec import (  # noqa: E402
    build_invocation_spec,
    invocation_spec_filename,
    write_invocation_spec,
)
from _paths import workspace_root  # noqa: E402
from _task_group_contract import task_group_shape_cases  # noqa: E402

sys.path.pop(0)

try:
    from hyperloom.common.kernel_shape_contract import (
        REVIEW_SHAPE_PROVENANCE as _REVIEW_SHAPE_PROVENANCE,
    )
except ImportError:  # pragma: no cover - standalone invocation
    # Runs as a subprocess against the installed hyperloom, which may predate
    # the constant. An unrecognised provenance then reads as measured, which is
    # the pre-existing wording rather than a new claim.
    _REVIEW_SHAPE_PROVENANCE = frozenset({"review_backfill", "review_derived"})


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
    """Persist a status snapshot for the current run.

    Args:
        status_path (Path): Destination status JSON file.
        state (str): Current run state (e.g. ``running``, ``succeeded``).
        current_step (str): Human-readable label of the active step.
        log_path (Path): Log file whose size/tail are recorded.
        artifact_paths (dict[str, str]): Map of artifact names to paths.
        run_id (str): Unique identifier for this run.
        started_at (str): ISO-8601 start time of the run.
        error (str | None): Error message recorded when the run failed.
    """
    payload: dict[str, Any] = {
        "tool": "kernel_optimization",
        "run_id": run_id,
        "state": state,
        "current_step": current_step,
        "pid": os.getpid(),
        "started_at": started_at,
        "updated_at": utc_now(),
        "log_path": str(log_path),
        "artifact_paths": artifact_paths,
        "offset_bytes": log_path.stat().st_size if log_path.exists() else 0,
        "last_lines": read_last_lines(log_path),
    }
    if error:
        payload["error"] = error
    atomic_write_json(status_path, payload)


def resolve_candidates_path(run_dir: Path) -> Path:
    """Resolve the latest per-run ``kernel_candidates.json`` or the flat fallback."""
    flat = run_dir / "kernel_candidates.json"
    if flat.is_file():
        return flat
    if run_dir.is_dir():
        sub_candidates = [
            child / "kernel_candidates.json"
            for child in run_dir.iterdir()
            if child.is_dir() and (child / "kernel_candidates.json").is_file()
        ]
        if sub_candidates:
            # Zero-padded timestamp prefix makes lexical order == time order.
            return max(sub_candidates, key=lambda p: p.parent.name)
    return flat


def load_candidates(path: Path) -> list[dict[str, Any]]:
    """Load kernel candidates from JSON, normalizing legacy shapes.

    Returns the union of ``hot_kernels`` (routable) + ``skipped_kernels`` so id
    lookup still resolves non-routable kernels; legacy flat-list /
    ``kernel_candidates`` shapes are respected.

    Args:
        path: Path to the ``kernel_candidates.json`` file.

    Returns:
        The list of candidate dicts, with trace-report paths backfilled.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"kernel candidates file is neither an object nor a list: {path}")
    candidates = list(payload.get("hot_kernels") or payload.get("kernel_candidates") or [])
    skipped = payload.get("skipped_kernels") or []
    if isinstance(skipped, list):
        seen_ids = {c.get("kernel_id") for c in candidates if isinstance(c, dict) and c.get("kernel_id")}
        for entry in skipped:
            if not isinstance(entry, dict):
                continue
            kid = entry.get("kernel_id")
            if kid and kid in seen_ids:
                continue
            candidates.append(entry)
            if kid:
                seen_ids.add(kid)
    artifact_paths = payload.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        artifact_paths = {}
    report_path = payload.get("trace_report_path") or artifact_paths.get("trace_report_path")
    if report_path:
        for candidate in candidates:
            if isinstance(candidate, dict):
                candidate.setdefault("trace_report_path", str(report_path))
    return candidates


def load_candidate_input(path: str, kernel_id: str) -> dict[str, Any] | None:
    """Load a serialized dispatch candidate, including task-group context."""
    if not path:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    candidate_id = str(payload.get("kernel_id") or "")
    if candidate_id and candidate_id != str(kernel_id):
        return None
    return payload


def task_group_result_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return stable task-group identity and shape-contract metadata."""
    group = candidate.get("task_group")
    if not isinstance(group, dict):
        return {}
    cases = task_group_shape_cases(candidate)
    return {
        "task_group_id": str(group.get("task_group_id") or ""),
        "task_group_key": str(group.get("task_group_key") or ""),
        "task_group_kernel_ids": [str(item) for item in (group.get("kernel_ids") or []) if str(item)],
        "task_group_primary_kernel_id": str(group.get("primary_kernel_id") or candidate.get("kernel_id") or ""),
        "task_group_shape_case_ids": [
            str(case.get("case_id") or "") for case in cases if str(case.get("case_id") or "")
        ],
        "task_group_shape_case_count": len(cases),
    }


def _normalize_kernel_id(value: str) -> str:
    """Normalize a kernel id for tolerant comparison.

    Folds hallucinated ``kn``/``rn`` prefixes onto the real ``k`` numbering and
    lower-cases the value.

    Args:
        value: The raw kernel id.

    Returns:
        The normalized kernel id.
    """
    s = value.strip().lower()
    for prefix in ("kn", "rn"):
        if s.startswith(prefix) and s[len(prefix) :].isdigit():
            return "k" + s[len(prefix) :]
    return s


def find_candidate(candidates: list[dict[str, Any]], kernel_id: str) -> dict[str, Any] | None:
    """Resolve a candidate by id with progressively looser matching.

    Tries exact ``kernel_id`` first, then a unique routable ``name`` match,
    then a normalized id (``kn``/``rn`` → ``k``).

    Args:
        candidates: The candidate dicts to search.
        kernel_id: The kernel id (or name) to resolve.

    Returns:
        The matching candidate, or ``None`` when nothing matches (the caller
        skips gracefully).
    """
    for candidate in candidates:
        if candidate.get("kernel_id") == kernel_id:
            return candidate
    # Names aren't stable ids; accept only a unique routable match.
    name_matches = [
        candidate
        for candidate in candidates
        if candidate.get("name") == kernel_id
        and candidate.get("reusable_native_kernel") is not False
        and candidate.get("source_file")
    ]
    if len(name_matches) == 1:
        return name_matches[0]
    target = _normalize_kernel_id(kernel_id)
    for candidate in candidates:
        if _normalize_kernel_id(str(candidate.get("kernel_id") or "")) == target:
            return candidate
    return None


def existing_path(value: str) -> str:
    """Resolve ``value`` to an absolute path if it exists, else empty string.

    Args:
        value (str): A filesystem path (possibly ``~``-prefixed) or empty.

    Returns:
        str: The resolved absolute path when the file/dir exists; otherwise
            an empty string (including when ``value`` is falsy).
    """
    if not value:
        return ""
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else ""


def has_benchmark(args: argparse.Namespace, candidate: dict[str, Any]) -> bool:
    """Report whether any usable benchmark file exists for a kernel.

    Args:
        args (argparse.Namespace): Parsed CLI args carrying ``benchmark_file``.
        candidate (dict[str, Any]): Kernel candidate dict.

    Returns:
        bool: True when at least one referenced benchmark file exists on disk.
    """
    bench_files = candidate.get("benchmark_files") or []
    return bool(
        existing_path(args.benchmark_file)
        or existing_path(str(candidate.get("benchmark_file") or ""))
        or any(existing_path(str(p)) for p in bench_files)
    )


def _resolve_source_file(
    llm_source: str,
    candidate: dict[str, Any],
    kernel_id: str,
    log_path: Path | None = None,
) -> str:
    """Resolve the effective source file, preferring TraceLens (candidate).

    Candidate wins over the LLM's ``--source-file`` (which can mismatch the
    kernel); a differing LLM path emits a ``[source-override]`` warning. Falls
    back to the LLM path when the candidate has no source_file.

    Args:
        llm_source: The LLM-supplied ``--source-file`` path.
        candidate: The TraceLens candidate dict (source of truth).
        kernel_id: The kernel id, used in log messages.
        log_path: Optional path to append override/fallback notes.

    Returns:
        The effective source file path to use.
    """
    cand_source = str((candidate or {}).get("source_file") or "").strip()
    llm = str(llm_source or "").strip()
    if not cand_source:
        return llm

    # A candidate "source" can be a profiler frame label, not a real file;
    # prefer the caller's explicit source_file when it's readable.
    def _is_real_file(p: str) -> bool:
        """Return True when ``p`` is a non-empty path pointing at a real file.

        Args:
            p (str): Candidate filesystem path to test.

        Returns:
            bool: True if ``p`` is truthy and refers to an existing file;
                False on any OS/runtime error or non-file path.
        """
        try:
            return bool(p) and Path(p).is_file()
        except (OSError, RuntimeError):
            return False

    if not _is_real_file(cand_source) and _is_real_file(llm):
        if log_path is not None:
            append_log(
                log_path,
                f"[source-fallback] kernel_id={kernel_id} candidate "
                f"source_file={cand_source!r} is not a readable file "
                f"(likely a pseudo-op frame label); using explicit "
                f"source_file={llm!r}",
            )
        return llm

    if llm and Path(cand_source) != Path(llm):
        try:
            differ = Path(cand_source).resolve(strict=False) != Path(llm).resolve(strict=False)
        except (OSError, RuntimeError):
            differ = True
        if differ and log_path is not None:
            append_log(
                log_path,
                f"[source-override] kernel_id={kernel_id} "
                f"LLM passed source_file={llm!r} but TraceLens candidate resolves to "
                f"{cand_source!r}; using TraceLens (source of truth)",
            )
    return cand_source


# Kernel-name → benchmark-name priority patterns (specific families first).
_BENCHMARK_PATTERNS: list[tuple["re.Pattern[str]", list["re.Pattern[str]"]]] = [
    # Flash / multi-head attention (before paged-attn so fmha doesn't hit test_pa.py).
    (
        re.compile(r"(fmha|^mha|::mha|flash[_-]?attn|multi[_-]?head)", re.IGNORECASE),
        [
            re.compile(r"^(test|bench)_.*mha", re.IGNORECASE),
            re.compile(r"^(test|bench)_.*flash.*attn", re.IGNORECASE),
        ],
    ),
    # Paged attention (matches both ``paged_attn`` and ``paged_attention``)
    (
        re.compile(r"(paged[_-]?att(?:n|ention)|^pa_|::pa_)", re.IGNORECASE),
        [
            re.compile(r"^(test|bench)_pa\b", re.IGNORECASE),
            re.compile(r"^(test|bench)_.*paged", re.IGNORECASE),
        ],
    ),
    # MoE / fused-MoE
    (
        re.compile(r"(fmoe|fused[_-]?moe|::moe|^moe_)", re.IGNORECASE),
        [re.compile(r"^(test|bench)_.*moe", re.IGNORECASE)],
    ),
    # GEMM / matmul / linear
    (
        re.compile(r"(gemm|matmul|^linear|::linear|_mm_)", re.IGNORECASE),
        [
            re.compile(r"^(test|bench)_.*gemm", re.IGNORECASE),
            re.compile(r"^(test|bench)_.*matmul", re.IGNORECASE),
        ],
    ),
    # RMSNorm / LayerNorm
    (
        re.compile(r"(rmsnorm|layernorm|_norm\b|norm$)", re.IGNORECASE),
        [re.compile(r"^(test|bench)_.*norm", re.IGNORECASE)],
    ),
]


def _match_benchmark_for_kernel(
    kernel_name: str,
    bench_files: list[Any],
) -> list[str]:
    """Reorder ``bench_files`` so semantically-matching benchmarks come first.

    Scans :data:`_BENCHMARK_PATTERNS` in priority order; the first matching
    kernel-name regex hoists that family's bench files to the front (earlier
    patterns win). No match preserves the original order. Prevents picking an
    off-topic benchmark (e.g. fmha → test_pa.py stalling the benchmark step).

    Args:
        kernel_name: The kernel name to match patterns against.
        bench_files: Candidate benchmark file paths.

    Returns:
        The benchmark paths reordered so the matching family comes first.
    """
    existing = [p for p in (bench_files or []) if isinstance(p, str) and p]
    if not existing:
        return []
    name = str(kernel_name or "")
    for kernel_re, bench_res in _BENCHMARK_PATTERNS:
        if not kernel_re.search(name):
            continue

        def _priority(path: str, _bench_res=bench_res) -> int:
            """Return the sort rank of a benchmark path within its family.

            Args:
                path (str): Benchmark file path whose basename is matched.
                _bench_res (list[re.Pattern[str]]): Priority-ordered bench
                    patterns for the matched kernel family (bound default).

            Returns:
                int: Index of the first matching pattern, or ``len(_bench_res)``
                    when none match (sorts after all matched files).
            """
            base = Path(path).name
            for idx, br in enumerate(_bench_res):
                if br.search(base):
                    return idx
            return len(_bench_res)

        return sorted(existing, key=_priority)
    return existing


def parse_backends(backends: str) -> list[str]:
    """Parse and validate a comma-separated backend list.

    Args:
        backends (str): Comma-separated backend names (case-insensitive),
            e.g. ``"forge"``.

    Returns:
        list[str]: The normalized (lowercased, trimmed) backend names in the
        order given.

    Raises:
        ValueError: When any backend is outside the allowed set
            (``forge``).
    """
    raw = str(backends or "").strip()
    # Recover inner tokens when handed the repr() of a list instead of a string.
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].replace("'", "").replace('"', "")
    parsed = [b.strip().lower() for b in raw.split(",") if b.strip()]
    allowed = {"forge"}
    invalid = [b for b in parsed if b not in allowed]
    if invalid:
        raise ValueError(f"unsupported backend(s): {', '.join(invalid)} (allowed: {sorted(allowed)})")
    return parsed


def choose_backends(args: argparse.Namespace, candidate: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Select the backend ladder for a kernel-opt run.

    Per-kernel Forge is opt-in only. The default KERNEL_AGENT path is the
    coordinator-owned GEAK phase delegate, so the CLI returns ``[]`` unless
    exact ``KERNEL_OPT_BACKEND_ORDER=forge`` is set.

    Args:
        args (argparse.Namespace): Parsed CLI args carrying ``backends`` and
            the benchmark path.
        candidate (dict[str, Any]): Kernel candidate dict used for benchmark
            availability.

    Returns:
        tuple[list[str], dict[str, Any]]: The selected backend ladder and a
            notes dict describing the selection (benchmark availability, etc.).
    """
    from hyperloom.common.env import forge_explicitly_enabled

    forge_enabled = forge_explicitly_enabled()
    user_backends = parse_backends(args.backends) if forge_enabled else []
    if forge_enabled and not user_backends:
        user_backends = ["forge"]
    benchmark_available = has_benchmark(args, candidate)
    notes: dict[str, Any] = {
        "user_specified_backends": bool(user_backends),
        "benchmark_available": benchmark_available,
    }

    if user_backends:
        return user_backends, notes

    return [], notes


_GEAK_KERNEL_TYPE = {
    "triton": "triton",
    "hip_cpp": "hip",
    "flydsl": "flydsl",
    "python": "other",
    "vendor_binary": "other",
    "unknown": "other",
}


_GPU_HW: dict[str, dict[str, Any]] = {
    "mi300x": {
        "name": "MI300X",
        "arch": "gfx942",
        "uarch": "CDNA3",
        "cus": 304,
        "mem": "HBM3 (~5.3 TB/s peak), 256 MB Infinity Cache",
        "build_flag": "--offload-arch=gfx942",
    },
    "mi308x": {
        "name": "MI308X",
        "arch": "gfx942",
        "uarch": "CDNA3",
        "cus": 304,
        "mem": "HBM3 (~5.3 TB/s peak), 256 MB Infinity Cache",
        "build_flag": "--offload-arch=gfx942",
    },
    "mi325x": {
        "name": "MI325X",
        "arch": "gfx942",
        "uarch": "CDNA3",
        "cus": 304,
        "mem": "HBM3E (~6.0 TB/s peak), 256 MB Infinity Cache",
        "build_flag": "--offload-arch=gfx942",
    },
    "mi355x": {
        "name": "MI355X",
        "arch": "gfx950",
        "uarch": "CDNA4",
        "cus": 256,
        "mem": "HBM3E (~8.0 TB/s peak)",
        "build_flag": "--offload-arch=gfx950",
    },
}


def _normalize_target_platform(value: str) -> str:
    """Normalize a target-platform string for ``_GPU_HW`` lookups.

    Args:
        value (str): Raw platform name (e.g. ``"MI300X"``) or empty.

    Returns:
        str: The lowercased, whitespace-stripped platform key.
    """
    return str(value or "").strip().lower()


def _hardware_prompt_blocks(target_platform: str) -> tuple[str, str]:
    """Build the intro and hardware-notes prompt blocks for a target GPU.

    Looks up the platform in :data:`_GPU_HW`; when unknown, returns generic
    blocks instructing the agent to inspect the runtime device.

    Args:
        target_platform (str): Target GPU platform name (e.g. ``"mi300x"``).

    Returns:
        tuple[str, str]: A ``(intro, notes)`` pair of prompt text — the
            optimization intro line and the hardware-notes block.
    """
    platform = _normalize_target_platform(target_platform)
    hw = _GPU_HW.get(platform)
    if not hw:
        intro = (
            "Optimize this GPU kernel for the active AMD Instinct GPU "
            "inference serving. Produce an actual edited kernel file with "
            "measurable speedup; do NOT just analyze and submit unchanged."
        )
        notes = "\n".join(
            [
                "Hardware notes (target platform unknown):",
                "- Before benchmarking, query the runtime environment for the ROCm arch ",
                "(hipDeviceGetName/rocminfo), visible GPU IDs (ROCR_VISIBLE_DEVICES), " + "and memory size/bandwidth.",
                "- Record those values in the result and choose --offload-arch=<arch> "
                + "accordingly; replace <arch> with the inspected ROCm arch before running.",
            ]
        )
        return intro, notes

    intro = (
        f"Optimize this GPU kernel for **AMD Instinct {hw['name']} "
        f"({hw['arch']}, {hw['uarch']})** inference serving. Produce an actual "
        "edited kernel file with measurable speedup; do NOT just analyze and "
        "submit unchanged."
    )
    notes = "\n".join(
        [
            f"Hardware notes (target platform: `{platform}`):",
            f"- {hw['cus']} CUs, {hw['uarch']}, ROCm arch `{hw['arch']}`",
            f"- {hw['mem']}",
            f"- Build flag: `{hw['build_flag']}`",
            f"- Use optimizations compatible with `{hw['arch']}` and verify runtime "
            "device properties before benchmarking.",
        ]
    )
    return intro, notes


def _target_build_flag(target_platform: str) -> str:
    """Return the ``--offload-arch`` build flag for a target platform.

    Args:
        target_platform (str): Target GPU platform name (e.g. ``"mi355x"``).

    Returns:
        str: The platform's build flag (e.g. ``--offload-arch=gfx950``), or
            the ``--offload-arch=<arch>`` placeholder when unknown.
    """
    platform = _normalize_target_platform(target_platform)
    hw = _GPU_HW.get(platform)
    return str(hw["build_flag"]) if hw else "--offload-arch=<arch>"


def _env_target_platform() -> str:
    """Read the target GPU platform from the environment.

    Returns:
        str: The value of ``TARGET_GPU_TYPE``, falling back to ``GPU_TYPE``,
            or an empty string when neither is set.
    """
    return os.environ.get("TARGET_GPU_TYPE", "") or os.environ.get("GPU_TYPE", "")


def _format_shapes_for_case(shapes: Any) -> str:
    """Render a candidate row's ``shapes`` field as a comma-joined line.

    Args:
        shapes: A shapes value (string, list, or list of ``{call_num, shape}``
            dicts).

    Returns:
        The rendered single-line shapes string, or empty when none.
    """
    if not shapes:
        return ""
    if isinstance(shapes, str):
        return shapes
    if isinstance(shapes, (list, tuple)):
        parts: list[str] = []
        for entry in shapes:
            if isinstance(entry, str):
                parts.append(entry)
            elif isinstance(entry, dict):
                shape = entry.get("shape") or entry.get("Args") or ""
                call_num = entry.get("call_num")
                if shape:
                    parts.append(f"{shape}" + (f" (x{call_num})" if call_num else ""))
            else:
                parts.append(str(entry))
        return ", ".join(p for p in parts if p)
    return str(shapes)


_SHAPE_ARG_RE = re.compile(r"^\s*\((?P<dims>[^)]*)\)\s*(?P<dtype>[A-Za-z0-9_]+)?\s*$")


def _split_shape_fragments(shape_text: Any) -> list[str]:
    """Split TraceLens ``Args`` text into per-argument shape fragments.

    Args:
        shape_text: The raw TraceLens ``Args`` text (any type; coerced to str).

    Returns:
        The non-empty per-argument shape fragments.
    """
    text = str(shape_text or "").strip()
    if not text:
        return []
    return [frag.strip() for frag in re.split(r"\s*(?:<br\s*/?>|\n)\s*", text, flags=re.IGNORECASE) if frag.strip()]


def _parse_shape_arg(raw: Any, *, index: int) -> dict[str, Any]:
    """Parse one shape fragment such as ``(15360,8,768) bf16``.

    Args:
        raw: The raw shape fragment text (any type; coerced to str).
        index: The argument index recorded on the parsed result.

    Returns:
        A dict with ``index`` / ``raw`` and, when parseable, ``shape`` (dims)
        and ``dtype``.
    """
    text = str(raw or "").strip()
    out: dict[str, Any] = {"index": index, "raw": text}
    match = _SHAPE_ARG_RE.match(text)
    if not match:
        return out
    dims: list[int | str] = []
    dims_text = match.group("dims").strip()
    if dims_text:
        for part in dims_text.split(","):
            item = part.strip()
            if not item:
                continue
            try:
                dims.append(int(item))
            except ValueError:
                dims.append(item)
    out["shape"] = dims
    dtype = (match.group("dtype") or "").strip()
    if dtype:
        out["dtype"] = dtype
    return out


def _shape_case_from_value(
    value: Any,
    *,
    call_count: Any = None,
    primary: bool = False,
) -> dict[str, Any]:
    """Build one structured benchmark shape case from TraceLens shape data.

    Args:
        value: The shape data (dict, list/tuple, or scalar) to convert.
        call_count: Fallback call count when the value carries none.
        primary: Whether this case is the primary benchmark case.

    Returns:
        A shape-case dict with ``primary`` / ``call_count`` / ``raw`` / ``args``
        keys.
    """
    if isinstance(value, dict):
        structured_args = value.get("args")
        raw_shape = value.get("shape") or value.get("Args") or value.get("args") or ""
        case_count = value.get("call_num", value.get("call_count", call_count))
    elif isinstance(value, (list, tuple)):
        structured_args = None
        fragments: list[str] = []
        case_count = call_count
        for item in value:
            if isinstance(item, dict):
                shape = item.get("shape") or item.get("Args") or item.get("args") or ""
                if case_count is None:
                    case_count = item.get("call_num", item.get("call_count"))
            else:
                shape = item
            if shape not in (None, "", [], ()):
                fragments.append(str(shape))
        raw_shape = "<br>".join(fragments)
    else:
        structured_args = None
        raw_shape = value
        case_count = call_count
    try:
        parsed_count = int(float(case_count or 1))
    except (TypeError, ValueError):
        parsed_count = 1
    if isinstance(structured_args, list):
        args = list(structured_args)
    else:
        fragments = _split_shape_fragments(raw_shape)
        args = [_parse_shape_arg(fragment, index=idx) for idx, fragment in enumerate(fragments)]
    return {
        "primary": bool(primary),
        "call_count": parsed_count,
        "raw": str(raw_shape or "").strip(),
        "args": args,
    }


def _structured_benchmark_shape_cases(candidate: dict[str, Any]) -> dict[str, Any]:
    """Expose primary/supplementary serving shapes in machine-readable form.

    Args:
        candidate: The kernel candidate dict, possibly carrying a
            ``task_group`` or ``input_shapes``.

    Returns:
        A dict with ``primary_shape`` and ``supplementary_shapes``, or ``{}``
        when no usable shapes are present.
    """
    group = candidate.get("task_group")
    rows = group.get("rows") if isinstance(group, dict) else None
    grouped_shape_cases = group.get("shape_cases") if isinstance(group, dict) else None
    cases: list[dict[str, Any]] = []
    input_shapes = candidate.get("input_shapes")
    is_synthetic = bool(candidate.get("_input_shapes_synthetic"))
    if isinstance(grouped_shape_cases, list) and grouped_shape_cases:
        for index, grouped_case in enumerate(grouped_shape_cases):
            if not isinstance(grouped_case, dict):
                continue
            case = _shape_case_from_value(
                grouped_case.get("input_shapes"),
                call_count=grouped_case.get("call_count"),
                primary=index == 0,
            )
            case.update(
                {
                    "case_id": str(grouped_case.get("case_id") or ""),
                    "selector": dict(grouped_case.get("selector") or {}),
                    "operation": str(grouped_case.get("operation") or ""),
                    "source": "task_group_shape_cases",
                }
            )
            if case["raw"] or case["args"]:
                cases.append(case)
    elif isinstance(rows, list) and rows:
        # Prefer task_group rows so the prompt keeps supplementary shapes.
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            case = _shape_case_from_value(
                row.get("shapes"),
                call_count=row.get("call_count"),
                primary=idx == 0,
            )
            case.update(
                {
                    "operation": str(row.get("name") or ""),
                    "aggregate_time_ms": _safe_float(row.get("duration_us")) / 1000.0,
                    "percent_e2e": row.get("percent_of_total"),
                    "bound": str(row.get("bound_type") or ""),
                    "source": "task_group",
                }
            )
            if case["raw"] or case["args"]:
                cases.append(case)
    # Synthetic input_shapes still carry real dims when shape_provenance is
    # ``torch_trace`` (only tensor values are synthesized).
    dims_real = str(candidate.get("shape_provenance") or "").strip().lower() == "torch_trace"
    if not cases and isinstance(input_shapes, list) and input_shapes and (not is_synthetic or dims_real):
        for idx, entry in enumerate(input_shapes):
            case = _shape_case_from_value(entry, primary=idx == 0)
            case["source"] = "input_shapes"
            if case["raw"] or case["args"]:
                cases.append(case)
    if not cases:
        return {}
    cases[0]["primary"] = True
    for case in cases[1:]:
        case["primary"] = False
    return {
        "primary_shape": cases[0],
        "supplementary_shapes": cases[1:],
    }


def _build_captured_shapes_block(candidate: dict[str, Any]) -> str:
    """Fallback shapes block when no TraceLens ``task_group`` is attached.

    Pins the harness to the candidate's argument shapes so the backend does not
    pick its own. The heading states where the dims came from: a graph replay
    records no arguments, so the hottest kernels of a captured model can only be
    given dims the candidate review recovered or computed. Telling the backend
    those were measured would misrepresent the one thing worth knowing when the
    tuned kernel later fails to move end-to-end throughput.

    Args:
        candidate: The kernel candidate dict, possibly carrying captured
            shapes.

    Returns:
        The shapes prompt block, or ``""`` when the candidate has no shapes.
    """
    shapes = candidate.get("shapes") or candidate.get("kernel_shapes")
    rendered = _format_shapes_for_case(shapes)
    if not rendered:
        return ""
    bound = str(candidate.get("bound_type") or candidate.get("bound") or "").strip()
    bound_line = f" (bound: {bound})" if bound else ""
    provenance = str(candidate.get("shape_provenance") or "").strip().lower()
    if provenance in _REVIEW_SHAPE_PROVENANCE:
        heading = "Benchmark shapes (reconstructed for this serving run)"
        origin = (
            "The profiler recorded no arguments for this kernel because it runs\n"
            "inside a captured graph; these dims were reconstructed from the run's\n"
            f"own artifacts and serving configuration (provenance: {provenance}).\n"
            "They are the workload this kernel actually serves, so optimizing\n"
            "against them is what produces an end-to-end gain -- whereas shapes you\n"
            "choose yourself cannot account for the serving configuration:\n"
        )
    else:
        heading = "Benchmark shapes (TraceLens-captured from the serving run)"
        origin = (
            "They are what the kernel saw during sglang/vLLM serving, so optimizing\n"
            "against them is what produces an end-to-end gain on the workload:\n"
        )
    return (
        f"\n## {heading}\n\n"
        "Build your harness shape sweep / `get_inputs()` from EXACTLY these\n"
        f"argument shapes{bound_line} -- do NOT invent shapes.\n" + origin + f"- args: {rendered}\n"
        "Correctness golden: the ORIGINAL kernel's output on these shapes "
        "(baseline / `fn=` injection); do not hand-derive a reference from scratch.\n"
        + _build_kernel_contract_block(candidate)
    )


def _build_kernel_contract_block(candidate: dict[str, Any]) -> str:
    """Render the kernel-class CONTRACT (collective / attention) into the prompt.

    TraceLens enrichment attaches a ``kernel_contract`` dict for non-MoE kernel
    classes that need extra info to build a faithful harness. Returns "" when absent.
    """
    c = candidate.get("kernel_contract")
    if not isinstance(c, dict) or not c:
        return ""
    kind = c.get("kind", "")
    # Lead with the literal ``USER TASK CONTEXT`` marker the GEAK harness-generator keys on.
    rendered = _format_shapes_for_case(candidate.get("shapes") or candidate.get("kernel_shapes"))
    lines = [
        "\n## USER TASK CONTEXT (authoritative — overrides any discovered benchmark/test file)\n",
        "Build the harness to EXACTLY this contract. Use these shapes verbatim (do NOT\n",
        "sweep/scale them, do NOT read shapes from a discovered benchmark file), build the\n",
        "stated reference, and write `user_task:production` to harness_shapes_source.txt.\n",
    ]
    if rendered:
        lines.append(f"- Exact serving shapes: {rendered}\n")
    if kind == "collective":
        lines.append(
            f"- This is a COLLECTIVE ({c.get('collective_op')}). The shapes above are the\n"
            f"  per-rank tensor. Correctness reference MUST be `{c.get('reference')}` "
            f"(reduce_op={c.get('reduce_op', 'sum')}, world_size={c.get('world_size', '?')})\n"
            "  computed independently -- NEVER `return run_kernel(inputs)` (a self-compare is\n"
            "  a tautology). Initialize the process group and shard inputs per rank.\n"
            f"- NOTE: {c.get('e2e_note', '')}\n"
        )
    elif kind == "attention":
        extra = ", ".join(
            f"{k}={c[k]}"
            for k in ("head_dim", "num_heads", "num_kv_heads", "kv_layout", "seqlen_regime")
            if c.get(k) is not None
        )
        lines.append(
            f"- This is ATTENTION (causal={c.get('causal', True)}). Correctness reference MUST be\n"
            f"  `{c.get('reference')}` with is_causal={c.get('causal', True)} -- NEVER a self-compare.\n"
            f"- Serving contract: {extra}. Benchmark the '{c.get('seqlen_regime', 'mixed')}' regime\n"
            "  (decode=seqlen 1 per step vs prefill=full); do not mix regimes in one config.\n"
        )
    else:
        return ""
    return "".join(lines)


def _build_benchmark_cases_block(candidate: dict[str, Any]) -> str:
    """Render the multi-row benchmark cases section for a task_group.

    Falls back to :func:`_build_captured_shapes_block` when
    ``candidate["task_group"]`` is absent/empty so captured shapes still reach
    GEAK. With a task_group, emits one bullet per TraceLens row (sorted by
    aggregate time descending) surfacing operation, args, aggregate time,
    percent E2E, count, per-call ms, flops/byte, efficiency, and bound (bound +
    per-call ms drive backend dispatch).

    Args:
        candidate: The kernel candidate dict, optionally with a ``task_group``.

    Returns:
        The rendered benchmark-cases prompt block.
    """
    group = candidate.get("task_group")
    rows = group.get("rows") if isinstance(group, dict) else None
    if not (isinstance(rows, list) and rows):
        # No task_group: fall back to captured serving shapes.
        return _build_captured_shapes_block(candidate)
    function_name = str(group.get("function_name") or "")
    source_path = str(group.get("source_path") or "")
    definition_line = group.get("definition_line")
    ast_resolved = bool(group.get("ast_resolved"))
    location = f"{source_path}:{definition_line}" if source_path and definition_line else ""

    lines: list[str] = [
        "",
        "## Benchmark cases (TraceLens, sorted by aggregate time)",
        "",
    ]
    if len(rows) > 1:
        lines.extend(
            [
                f"This kernel resolves to the same source function across "
                f"{len(rows)} TraceLens rows ("
                f"{function_name or '<unknown function>'}"
                + (f" at {location}" if location else "")
                + (", AST-resolved" if ast_resolved else "")
                + "). Optimize the source function once; the patch applies "
                "to all rows below. Use the first row as the primary",
                "benchmark case; treat the rest as supplementary shape coverage.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"This kernel maps to a single TraceLens row in "
                f"{function_name or '<unknown function>'}"
                + (f" at {location}" if location else "")
                + ". The case below is the primary benchmark target.",
                "",
            ]
        )

    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        op = str(row.get("name") or "").strip()
        shapes = _format_shapes_for_case(row.get("shapes"))
        try:
            duration_us = float(row.get("duration_us") or 0.0)
        except (TypeError, ValueError):
            duration_us = 0.0
        aggregate_time_ms = duration_us / 1000.0
        try:
            count = int(row.get("call_count") or 0)
        except (TypeError, ValueError):
            count = 0
        per_call_ms = (aggregate_time_ms / count) if count else 0.0
        percent_e2e = row.get("percent_of_total")
        flops_per_byte = row.get("flops_per_byte")
        bound = str(row.get("bound_type") or "").strip() or "unknown"
        eff_pct = row.get("efficiency_percent")
        eff_peak_val = row.get("efficiency_peak_value")
        eff_peak_unit = str(row.get("efficiency_peak_unit") or "").strip()
        if eff_pct and eff_peak_val and eff_peak_unit:
            efficiency = f"{eff_pct:.2f}% of {eff_peak_val} {eff_peak_unit}"
        elif eff_pct:
            efficiency = f"{eff_pct:.2f}%"
        else:
            efficiency = "unknown"
        lines.append(
            f"- Case {idx}: operation={op}; args={shapes or '-'}; "
            f"aggregate_time_ms={aggregate_time_ms:.3f}; "
            f"percent_e2e={percent_e2e if percent_e2e is not None else '-'}; "
            f"count={count}; per_call_ms={per_call_ms:.6f}; "
            f"flops_per_byte={flops_per_byte if flops_per_byte is not None else '-'}; "
            f"efficiency={efficiency}; bound={bound}"
        )
    # Surface the kernel-class contract on the task_group path too.
    return "\n".join(lines) + _build_kernel_contract_block(candidate)


# Optimization directions keyed by bound type; first lever matches the bottleneck.
_PRIORITY_BULLETS: dict[str, list[str]] = {
    "memory": [
        (
            "1. **Memory traffic reduction** (primary lever for memory-bound rows): "
            + "improve coalescing / vectorization, fuse with neighbouring ops to "
            + "amortize global loads, reduce intermediate writes, and avoid extra "
            + "global-memory round trips."
        ),
        (
            "2. **Shape-aware tuning**: specialize block sizes and grid indexing "
            + "for the dominant TraceLens Args. Memory-bound kernels are especially "
            + "sensitive to load-coalescing alignment on the dominant shape."
        ),
        (
            "3. **Launch amortization** for tiny high-count decode shapes: "
            + "persistent / batched handling or wrapper-level batching when source "
            + "and harness allow."
        ),
        (
            "4. **Structural simplification**: hoist loop-invariant computations, "
            + "remove redundant address arithmetic, collapse dual-pass logic."
        ),
        (
            "5. **Compute utilization** (rarely the bottleneck here, but check): "
            + "MFMA tile choice, occupancy, register / shared-memory balance."
        ),
    ],
    "compute": [
        (
            "1. **Compute utilization** (primary lever for compute-bound rows): "
            + "improve MFMA tile choice, occupancy, and register / shared-memory "
            + "balance so the same FLOPs issue under a better-utilized pipeline."
        ),
        (
            "2. **Shape-aware tuning**: specialize block sizes and grid indexing "
            + "for the dominant TraceLens Args. Compute-bound kernels often hit "
            + "different efficiency ceilings on K-major vs N-major shapes."
        ),
        (
            "3. **Structural simplification**: hoist loop-invariant computations, "
            + "remove redundant address arithmetic, collapse dual-pass logic."
        ),
        (
            "4. **Memory traffic reduction** (secondary): coalescing / "
            + "vectorization, fewer intermediate writes — rarely the bottleneck "
            + "here but worth measuring after a compute-side change."
        ),
        (
            "5. **Launch amortization** for tiny high-count decode shapes: "
            + "persistent / batched handling or wrapper-level batching."
        ),
    ],
    "unknown": [
        (
            "1. **Structural simplification**: hoist loop-invariant computations, "
            + "remove redundant address arithmetic, collapse dual-pass logic."
        ),
        ("2. **Shape-aware tuning**: specialize block sizes and grid indexing " + "for the dominant TraceLens Args."),
        (
            "3. **Memory traffic reduction**: improve coalescing / vectorization, "
            + "reduce intermediate writes, avoid extra global-memory round trips."
        ),
        "4. **Launch amortization** for tiny high-count decode shapes.",
        ("5. **Compute utilization**: improve MFMA tile choice, occupancy, " + "register / shared-memory balance."),
    ],
}


def _classify_bound(bound_type: str) -> str:
    """Classify a TraceLens ``bound`` string into a coarse bucket.

    Args:
        bound_type: The TraceLens bound description.

    Returns:
        One of ``"memory"``, ``"compute"``, or ``"unknown"``.
    """
    text = (bound_type or "").lower()
    if "memory" in text or "bandwidth" in text or "hbm" in text:
        return "memory"
    if "compute" in text or "arithmetic" in text or "flops" in text:
        return "compute"
    return "unknown"


def _build_priority_block(candidate: dict[str, Any]) -> str:
    """Render the bound-keyed optimization priority list.

    Uses the primary row's bound when the candidate carries a ``task_group``.

    Args:
        candidate: The kernel candidate dict.

    Returns:
        The priority-list prompt block, or ``""`` when ``bound_type`` is
        missing and no ``task_group`` is attached.
    """
    group = candidate.get("task_group")
    bound_type = str(candidate.get("bound_type") or "").strip()
    if not bound_type and isinstance(group, dict):
        rows = group.get("rows") or []
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            bound_type = str(rows[0].get("bound_type") or "").strip()
    if not bound_type:
        return ""
    bucket = _classify_bound(bound_type)
    label = bound_type or "unknown"
    header_line = f"## Optimization priorities (TraceLens bound: `{label}`)"
    intro = (
        "The list below orders optimization levers by expected payoff for "
        "this kernel's bottleneck. Try lever 1 first; only move to lever 2 "
        "if profiling shows lever 1 is exhausted or not applicable."
    )
    lines = ["", header_line, "", intro, ""]
    lines.extend(_PRIORITY_BULLETS[bucket])
    return "\n".join(lines)


# Shared numeric coercion (see _io_utils.safe_float).
_safe_float = safe_float


def _format_impact_range(
    low_ms: float,
    low_e2e: float,
    high_ms: float,
    high_e2e: float,
) -> str:
    """One-line impact range formatter; empty string when both ends zero.

    Args:
        low_ms (float): Low-end estimated savings in milliseconds.
        low_e2e (float): Low-end savings as a percent of end-to-end time.
        high_ms (float): High-end estimated savings in milliseconds.
        high_e2e (float): High-end savings as a percent of end-to-end time.

    Returns:
        str: The formatted impact-range line, or empty string when both
            ``low_ms`` and ``high_ms`` are zero.
    """
    if not (low_ms or high_ms):
        return ""
    return (
        f"**Estimated impact range:** low {low_ms:.2f} ms savings "
        f"({low_e2e:.2f}% E2E), high {high_ms:.2f} ms savings "
        f"({high_e2e:.2f}% E2E). These are TraceLens roofline estimates, "
        "not measured speedups — confirm with a real benchmark."
    )


def _build_extra_context_block(candidate: dict[str, Any]) -> str:
    """Render authoritative workload context for the candidate, if supplied.

    Returns ``""`` unless a caller has pre-attached ``extra_dispatch_context``
    to the candidate (no in-tree producer today). When present it is injected
    verbatim as authoritative workload context (serving config, E2E/Amdahl
    framing, roofline specifics) so harness-gen can pin the true decode shapes.

    Args:
        candidate: The kernel candidate dict, optionally carrying
            ``extra_dispatch_context`` (a pre-rendered free-form string).

    Returns:
        The workload-context prompt block, or ``""`` when absent.
    """
    ctx = candidate.get("extra_dispatch_context")
    if not isinstance(ctx, str) or not ctx.strip():
        return ""
    return "\n## WORKLOAD CONTEXT (authoritative — use to build the harness; do NOT guess these)\n" + ctx.strip() + "\n"


def _build_hypothesis_block(candidate: dict[str, Any]) -> str:
    """Render a TraceLens hypothesis section for the candidate, if any.

    Empty when no prose fields are present. A multi-P-item ``task_group`` renders
    every P-item's prose under a ``### P{rank}`` header; otherwise a single block.
    The reasoning/resolution prose is labelled a hypothesis to validate (it is
    itself LLM-generated); the numeric impact range is roofline arithmetic.

    Args:
        candidate: The kernel candidate dict, optionally with a ``task_group``
            carrying P-item prose.

    Returns:
        The hypothesis prompt block, or ``""`` when no prose is present.
    """
    group = candidate.get("task_group")
    all_prose: list[Any] = []
    if isinstance(group, dict):
        raw = group.get("all_pitem_prose")
        if isinstance(raw, list):
            all_prose = [e for e in raw if isinstance(e, dict)]

    if len(all_prose) > 1:
        lines: list[str] = [
            "",
            "## TraceLens Hypothesis [validate before acting]",
            "",
            "This source function appears across MULTIPLE TraceLens P-items;",
            "each subsection below is the analysis-orchestrator's hypothesis",
            "for the corresponding P-item. Treat them as starting points —",
            "verify each against the source / a quick micro-benchmark before",
            "committing to a direction. If your measurements contradict any",
            "hypothesis, follow the data and record the discrepancy in your",
            "final summary.",
            "",
        ]
        for entry in all_prose:
            rank = entry.get("rank") or 0
            title = str(entry.get("title") or "").strip()
            header = f"### P{rank}" if rank else "### (un-ranked TraceLens entry)"
            if title:
                header += f" — {title}"
            lines.extend([header, ""])
            ident = str(entry.get("identification") or "").strip()
            reason = str(entry.get("reasoning_for_slowdown") or "").strip()
            resol = str(entry.get("resolution") or "").strip()
            if ident:
                lines.extend(["**Identification (TraceLens context):**", ident, ""])
            if reason:
                lines.extend(["**Reasoning for slowdown (hypothesis):**", reason, ""])
            if resol:
                lines.extend(["**Recommended direction (hypothesis):**", resol, ""])
            impact = _format_impact_range(
                _safe_float(entry.get("impact_low_ms")),
                _safe_float(entry.get("impact_low_e2e_pct")),
                _safe_float(entry.get("impact_high_ms")),
                _safe_float(entry.get("impact_high_e2e_pct")),
            )
            if impact:
                lines.extend([impact, ""])
        return "\n".join(lines).rstrip()

    # Single/no-P-item path: read prose from the candidate directly.
    identification = str(candidate.get("identification") or "").strip()
    reasoning = str(candidate.get("reasoning_for_slowdown") or "").strip()
    resolution = str(candidate.get("resolution") or "").strip()
    low_ms = _safe_float(candidate.get("impact_low_ms"))
    low_e2e = _safe_float(candidate.get("impact_low_e2e_pct"))
    high_ms = _safe_float(candidate.get("impact_high_ms"))
    high_e2e = _safe_float(candidate.get("impact_high_e2e_pct"))
    if not (identification or reasoning or resolution or low_ms or high_ms):
        return ""
    lines = [
        "",
        "## TraceLens Hypothesis [validate before acting]",
        "",
        "The lines below are the TraceLens analysis-orchestrator's",
        "hypothesis for this kernel. Treat them as a starting point —",
        "verify the reasoning against the source / a quick micro-benchmark",
        "before committing to the recommended direction. If your",
        "measurements contradict the hypothesis, follow the data and",
        "record the discrepancy in your final summary.",
        "",
    ]
    if identification:
        lines.extend(["**Identification (TraceLens context):**", identification, ""])
    if reasoning:
        lines.extend(["**Reasoning for slowdown (hypothesis):**", reasoning, ""])
    if resolution:
        lines.extend(["**Recommended direction (hypothesis):**", resolution, ""])
    impact = _format_impact_range(low_ms, low_e2e, high_ms, high_e2e)
    if impact:
        lines.append(impact)
    return "\n".join(lines)


def _coerce_cli_value(value: str | bool) -> Any:
    """Coerce a CLI token to a bool/int/float, falling back to the string.

    Args:
        value (str | bool): A raw CLI value (already a bool, or a string
            token such as ``"true"``, ``"42"``, or ``"0.5"``).

    Returns:
        Any: The bool/int/float parsed from ``value``, or the original
            string when it is not a recognized scalar.
    """
    if isinstance(value, bool):
        return value
    text = str(value)
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_extra_server_args(extra_args: str) -> dict[str, Any]:
    """Parse selected SGLang flags from an EXTRA_SGLANG_ARGS-style string.

    Splits the string with shell rules, turning ``--flag value`` pairs into
    coerced dict entries and bare ``--flag`` tokens into ``True``. The raw
    string is preserved under the ``raw`` key.

    Args:
        extra_args (str): A shell-style argument string (e.g.
            ``"--page-size 16 --disable-cuda-graph"``).

    Returns:
        dict[str, Any]: Parsed flags keyed by normalized name (dashes →
            underscores), always including ``raw`` with the original text;
            an empty dict when ``extra_args`` is blank.
    """
    if not extra_args.strip():
        return {}
    try:
        tokens = shlex.split(extra_args)
    except ValueError:
        return {"raw": extra_args}
    parsed: dict[str, Any] = {"raw": extra_args}
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if not token.startswith("--"):
            idx += 1
            continue
        flag = token[2:].replace("-", "_")
        value: str | bool = True
        if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
            value = tokens[idx + 1]
            idx += 1
        parsed[flag] = _coerce_cli_value(value)
        idx += 1
    return parsed


def _shape_call_entries(shapes: Any, call_num: Any = None) -> list[dict[str, Any]]:
    """Normalize a shapes list into ``{call_num, shape}`` entries.

    Args:
        shapes (Any): Shapes value; only a list is processed (each item may
            be a ``{call_num, shape}`` dict or a bare shape).
        call_num (Any): Default call count applied to bare-shape entries;
            coerced to int, defaulting to 1.

    Returns:
        list[dict[str, Any]]: One ``{"call_num", "shape"}`` dict per
            non-empty shape; empty list when ``shapes`` is not a list.
    """
    if not isinstance(shapes, list):
        return []
    try:
        count = int(float(call_num or 1))
    except (TypeError, ValueError):
        count = 1
    entries: list[dict[str, Any]] = []
    for shape in shapes:
        if isinstance(shape, dict) and "shape" in shape:
            entries.append(
                {
                    "call_num": int(shape.get("call_num") or count),
                    "shape": shape["shape"],
                }
            )
        elif shape not in (None, "", [], ()):
            entries.append({"call_num": count, "shape": shape})
    return entries


def build_kernel_metadata(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Build structured runtime context for GEAK task prompts.

    Merges the candidate's shape/dtype/runtime fields with parsed
    ``extra_server_args`` so GEAK receives a single normalized metadata
    dict (kernel path/name, input/output shapes and dtypes, runtime args
    and flags, kernel params, env vars, etc.).

    Args:
        candidate (dict[str, Any]): Kernel candidate dict supplying shapes,
            dtypes, runtime flags/args, and kernel params.
        args (argparse.Namespace): Parsed CLI args; provides overrides such
            as ``source_file`` and ``extra_server_args``.

    Returns:
        dict[str, Any]: The structured kernel-metadata dict consumed when
            rendering GEAK task prompts.
    """
    source_file = getattr(args, "source_file", "") or candidate.get("source_file", "")
    kernel_name = str(candidate.get("name") or getattr(args, "kernel_id", ""))
    input_shapes = candidate.get("input_shapes")
    if input_shapes is None:
        input_shapes = _shape_call_entries(candidate.get("shapes", []), candidate.get("call_count"))
    input_dtypes = candidate.get("input_dtypes")
    if input_dtypes is None:
        input_dtypes = candidate.get("dtypes", [])
    benchmark_shape_cases = _structured_benchmark_shape_cases(candidate)

    runtime_flags: dict[str, Any] = {}
    if isinstance(candidate.get("runtime_flags"), dict):
        runtime_flags.update(candidate["runtime_flags"])
    runtime_flags.setdefault("is_multigpu", bool(candidate.get("is_multigpu")))
    runtime_flags.setdefault("num_gpus_recommended", candidate.get("num_gpus_recommended"))
    extra_server_args = (
        getattr(args, "extra_server_args", "")
        or candidate.get("extra_server_args", "")
        or candidate.get("candidate_extra_server_args", "")
    )
    parsed_sglang_args = parse_extra_server_args(str(extra_server_args))
    for key in (
        "attention_backend",
        "decode_attention_backend",
        "prefill_attention_backend",
        "disable_cuda_graph",
        "disable_radix_cache",
        "enable_torch_compile",
        "enable_dp_attention",
    ):
        if key in parsed_sglang_args:
            runtime_flags.setdefault(key, parsed_sglang_args[key])

    runtime_args = candidate.get("runtime_args") if isinstance(candidate.get("runtime_args"), dict) else {}
    runtime_args = dict(runtime_args)
    if parsed_sglang_args:
        runtime_args.setdefault("extra_server_args", parsed_sglang_args.get("raw", str(extra_server_args)))
    for key in (
        "kv_cache_dtype",
        "page_size",
        "block_size",
        "cuda_graph_max_bs",
        "num_continuous_decode_steps",
        "triton_attention_num_kv_splits",
        "triton_attention_split_tile_size",
    ):
        if key in parsed_sglang_args:
            runtime_args.setdefault(key, parsed_sglang_args[key])

    raw_params = candidate.get("kernel_params") if isinstance(candidate.get("kernel_params"), dict) else {}
    kernel_params = dict(raw_params)
    if "kv_cache_dtype" in parsed_sglang_args:
        kernel_params.setdefault("KV_DTYPE", parsed_sglang_args["kv_cache_dtype"])
    if "page_size" in parsed_sglang_args:
        kernel_params.setdefault("BLOCK_SIZE", parsed_sglang_args["page_size"])
    if "block_size" in parsed_sglang_args:
        kernel_params.setdefault("BLOCK_SIZE", parsed_sglang_args["block_size"])
    for key in ("KV_DTYPE", "BLOCK_SIZE", "HEAD_SIZE"):
        kernel_params.setdefault(key, candidate.get(key))

    metadata = {
        "kernel_path": str(source_file or ""),
        "kernel_name": kernel_name,
        "input_shapes": input_shapes or [],
        "output_shapes": candidate.get("output_shapes") or [],
        "input_dtypes": input_dtypes or [],
        "output_dtypes": candidate.get("output_dtypes") or [],
        "backend": candidate.get("backend") or candidate.get("framework"),
        "runtime_backend": str(candidate.get("runtime_backend") or ""),
        "runtime_args": runtime_args,
        "runtime_flags": runtime_flags,
        "env_vars": candidate.get("env_vars") or {},
        "kernel_params": kernel_params,
        # launcher_source_file is the @compile_ops wrapper; kernel_path above is
        # the device source to rewrite.
        "launcher_source_file": str(candidate.get("launcher_source_file", "") or ""),
        "source_promoted_from_launcher": bool(
            candidate.get("source_promoted_from_launcher"),
        ),
        # The device kernel symbol that disambiguated dispatch, and the full .cu
        # set this op spans (each .cu is optimized in its own GEAK run).
        "device_kernel_name": str(candidate.get("device_kernel_name", "") or ""),
        "kernel_sources": list[Any](candidate.get("kernel_sources") or []),
    }
    if benchmark_shape_cases:
        metadata["benchmark_shape_cases"] = benchmark_shape_cases
    return metadata


def _embeddable_source_file(source_file: str) -> Path | None:
    """Return the candidate path whose contents may be pasted into the prompt.

    Embedding puts file contents into the prompt and into the on-disk prompt
    artifact, so the path is bounded to a framework source root here. The
    returned path is the one that passed the check, never the raw input, so a
    relative value cannot be validated as in-root and then read from the CWD.

    Args:
        source_file (str): The resolved or trace-cited source path.

    Returns:
        Path | None: The in-root file to embed, or ``None`` when no candidate
            form is both a real file and inside a known root.
    """
    from hyperloom.orchestrator.framework.paths import (
        resolve_patch_target_roots,
        resolved_within,
        source_file_candidates,
    )

    roots = resolve_patch_target_roots()
    for candidate in source_file_candidates(source_file):
        path = Path(candidate)
        if path.is_file() and any(resolved_within(candidate, root) for root in roots):
            return path
    return None


def build_prompt(
    candidate: dict[str, Any],
    args: argparse.Namespace,
    *,
    backend: str | None = None,
) -> str:
    """Render the optimization prompt handed to a rewrite backend.

    Two shapes, because the backends own different amounts of the run. GEAK is
    handed the whole harness: budget protocol, sandbox rules, the deliverable
    contract Hyperloom later parses, and the A/B recipes it needs because it
    brings no benchmark of its own. Forge brings its own driver, clock, gate and
    artifact export, so it is handed only what it cannot derive -- the trace
    evidence and the source-attribution guards. See the ``backend == "forge"``
    branch for why the omitted sections are not merely redundant there.

    Args:
        candidate (dict[str, Any]): Kernel candidate dict supplying source,
            benchmarks, shapes, and TraceLens context.
        args (argparse.Namespace): Parsed CLI args (source override, GPU
            count, kernel id, etc.).
        backend (str | None): Target backend name. ``"forge"`` selects the slim
            prompt; every other value (including None) renders the full one.

    Returns:
        str: The rendered prompt text for the rewrite backend.
    """
    source_file = args.source_file or candidate.get("source_file", "")
    source_block = ""
    embeddable = _embeddable_source_file(str(source_file)) if source_file else None
    if embeddable is not None:
        content = embeddable.read_text(encoding="utf-8", errors="replace")
        source_block = f"\nSource content:\n```\n{content[:12000]}\n```"
    kernel_repo = str(candidate.get("kernel_repo") or "")
    bench_files = candidate.get("benchmark_files") or []
    if isinstance(bench_files, str):
        bench_files = [bench_files]
    # Sort by semantic match so relevant benchmarks head the [:8]-clipped list.
    bench_files = _match_benchmark_for_kernel(str(candidate.get("name") or ""), bench_files)
    is_multigpu = bool(candidate.get("is_multigpu"))
    # GPU count: CLI override, then candidate hint, then 1.
    num_gpus = max(1, int(getattr(args, "num_gpus", 0) or 0) or int(candidate.get("num_gpus_recommended") or 1))
    # Map source_type to GEAK's kernel_type vocabulary.
    geak_kernel_type = _GEAK_KERNEL_TYPE.get(str(candidate.get("source_type", "unknown")), "other")
    kernel_name = str(candidate.get("name", args.kernel_id))
    kernel_metadata = build_kernel_metadata(candidate, args)
    budget_protocol_block = (
        "## BUDGET PROTOCOL (read this FIRST, before any tool call):\n"
        "Every `mini-swe-agent step N ($X.XX)` header shows CUMULATIVE LLM TOKEN COST\n"
        "in dollars. This is **TELEMETRY**, NOT a budget signal. The per-task LLM\n"
        "cost_limit has been disabled (`--cost-limit 0.0`); you will NOT be terminated\n"
        "by the cost meter at $2, $5, $10, $20, $50, or $100. The ONLY budget that\n"
        "ends your task is the wall-clock timeout managed by the runner.\n"
        "\n"
        "Prior failed runs have been observed to exit at step ~3 with ~$2 spend,\n"
        "declaring 'budget exhausted' WITHOUT making any code changes. Every one of\n"
        "those runs threw away 90%+ of the available wall-clock budget. **DO NOT\n"
        "REPEAT THAT MISTAKE.**\n"
        "\n"
        "Successful runs typically use 30-60 tool calls and $15-$40 of token spend\n"
        "across the full wall-clock budget. Plan for THAT scale. Read the target\n"
        "kernel, write an optimization, rebuild, test, iterate. If you see a low\n"
        "step / low $ telemetry header and your impulse is 'submit now to be safe'\n"
        "— that impulse is WRONG. Make the edit. Run the test. Iterate.\n"
    )
    # Hard-rule notice when the source was promoted from a @compile_ops wrapper.
    promotion_block = ""
    launcher_source = str(candidate.get("launcher_source_file", "") or "").strip()
    if candidate.get("source_promoted_from_launcher") and launcher_source:
        promotion_block = (
            "\n>>> SOURCE ATTRIBUTION NOTE — READ FIRST <<<\n"
            f"This kernel (`{kernel_name}`) was originally traced at the Python launcher:\n"
            f"  {launcher_source}\n"
            "which is a thin `@compile_ops` wrapper. The wrapper does NOT contain the\n"
            "compute path — at runtime the `@compile_ops` decorator dispatches to a\n"
            "JIT-compiled `.so` under `<aiter>/jit/build/module_*/` and bypasses the\n"
            "Python wrapper entirely. Patching the wrapper has ZERO runtime effect.\n"
            "\n"
            "Your rewrite target is the DEVICE SOURCE shown above as `kernel_url`:\n"
            f"  {source_file}\n"
            "\n"
            "Hard rules for this kernel:\n"
            "1. DO NOT modify the Python wrapper at the launcher path above. Patches\n"
            "   there are silently bypassed by the @compile_ops .so loader and the\n"
            "   integrate baseline will measure -0% E2E gain followed by REVERT.\n"
            "2. The device source may be a CODEGEN ENTRY (e.g. `gemm_moe_ck2stages.cu`)\n"
            "   that hipcc compiles into per-(dtype, quant, act) `module_*.so` instances\n"
            "   under `<aiter>/jit/build/`. The orchestrator clears the matching jit/\n"
            "   build/ entries before rebuild so your patch actually takes effect on\n"
            "   next import (no manual cache invalidation needed on your side).\n"
            "3. Preserve function names, signatures, host entry points, and the\n"
            "   `aiter` namespace exactly as in the original — the apply step rejects\n"
            "   patches that drop required host entry functions or that submit a\n"
            "   standalone `PYBIND11_MODULE` / `TORCH_LIBRARY` block absent from the\n"
            "   target file.\n"
        )
    # Name the exact device kernel symbol so the rewrite targets the right
    # __global__ in a multi-kernel file.
    device_symbol_block = ""
    device_kernel_name = str(candidate.get("device_kernel_name", "") or "").strip()
    if (
        candidate.get("source_resolution_method") in ("op_to_source", "active_finder", "symbol_index")
        and device_kernel_name
    ):
        device_symbol_block = (
            "\n>>> DEVICE KERNEL FOCUS <<<\n"
            f"This op (`{kernel_name}`) dispatches to the device kernel symbol:\n"
            f"  {device_kernel_name}\n"
            f"resolved to the editable source: {source_file}\n"
            "If the file defines multiple `__global__` kernels, focus your rewrite on\n"
            "the one matching the symbol above; preserve all other kernels verbatim.\n"
        )
    budget_min = int(getattr(args, "budget_minutes", 60) or 60)
    target_platform = getattr(args, "target_platform", "") or _env_target_platform()
    platform_intro, hardware_notes = _hardware_prompt_blocks(target_platform)
    platform_build_flag = _target_build_flag(target_platform)
    hypothesis_block = _build_hypothesis_block(candidate)
    extra_context_block = _build_extra_context_block(candidate)
    benchmark_cases_block = _build_benchmark_cases_block(candidate)
    priority_block = _build_priority_block(candidate)
    # Surface the discovered benchmark/test files to the backend.
    bench_block = ""
    if bench_files:
        bench_block = "\nKnown benchmark/test files (also copied into your workspace as -f):\n"
        for b in bench_files[:8]:
            bench_block += f"- {b}\n"
        if is_multigpu and num_gpus >= 2:
            bench_block += (
                f"\nNOTE: This is a multi-GPU collective kernel and you HAVE {num_gpus} GPUs "
                "available in this sandbox (Ray/ROCR_VISIBLE_DEVICES already set). "
                f"To run a real benchmark use `torchrun --nproc_per_node={num_gpus} <bench>.py` "
                f"or `mpirun -n {num_gpus} ...` so torch.distributed init_process_group "
                "(backend='nccl' / 'rccl') succeeds. Do NOT fall back to a single-GPU "
                "rank-slice surrogate; the speedup numbers from a single-GPU surrogate are "
                "NOT comparable across attempts.\n"
            )
        elif is_multigpu:
            bench_block += (
                "\nNOTE: This is a multi-GPU collective kernel but you only have 1 GPU. "
                "Write a single-GPU rank-slice micro-bench (clearly labelled as a "
                "surrogate) for compute/IO improvement signal only.\n"
            )
    repo_block = ""
    if kernel_repo:
        repo_block = (
            f"\nKernel repo root: {kernel_repo}\n"
            f"You may READ any file under {kernel_repo} (it is on the local filesystem)."
        )
    safety = (
        "\nIMPORTANT — sandbox rules:\n"
        f"- Do NOT modify files under {kernel_repo or '/sgl-workspace'} or any system path.\n"
        "- Write all new/optimized kernel code, benchmarks, and reports under the\n"
        "  current working directory (your isolated workspace) ONLY.\n"
        "- DO NOT run `find /` or any unbounded filesystem scan. The host mounts\n"
        "  WekaFS, so a single `find / ...` typically takes 30–60 minutes and\n"
        "  burns the entire budget. The kernel source is at `kernel_url` above and\n"
        "  the repo root is `repo` above — use those EXACT paths. If you need to\n"
        "  search inside the repo, scope to the repo root: `find <repo> -name ...`\n"
        "  or `rg ... <repo>`, NEVER `find /`.\n"
        "\n"
        "GOAL & TIME BUDGET:\n"
        # Emit the explicit --mode token so GEAK's parser locks in the right preset.
        f"- Run mode: {'full' if budget_min >= 120 else 'quick'} "
        f"(--mode {'full' if budget_min >= 120 else 'quick'}).\n"
        f"- Hard wall-clock budget: ~{budget_min} minutes. Iterate up to minute "
        f"{int(budget_min * 0.85)},\n"
        "  then STOP iterating and finalize the report with your best so-far measured\n"
        "  speedup. The runner will SIGTERM at minute "
        f"{budget_min}; any in-flight work not on disk is lost.\n"
        "- Always print the final number in the form `speedup: X.XXx` (lowercase `x`)\n"
        "  at the END of `optimization_report.md` so the runner can extract it; if you\n"
        "  cannot measure, write `speedup: N/A`.\n"
        "- End `optimization_report.md` with machine-readable markers on separate lines:\n"
        "  `[CORRECTNESS] PASS` or `[CORRECTNESS] FAIL`, and\n"
        "  `[MICRO_SPEEDUP] X.XXx` or `[MICRO_SPEEDUP] N/A`.\n"
        "- Write the final optimized implementation as a COMPLETE source file under\n"
        "  `optimized_versions/` with the SAME extension as `kernel_url` (for example\n"
        "  `.cu` stays `.cu`, `.py` stays `.py`). Do NOT submit markdown, a diff, or\n"
        "  an excerpt as the optimized artifact; integration replaces the target file\n"
        "  byte-for-byte and will reject non-source artifacts.\n"
        "- The optimized source must be an IN-PLACE replacement for `kernel_url`:\n"
        "  start from the original file, preserve its namespace, exported host entry\n"
        "  functions, registration macros, includes, and public signatures. Do NOT\n"
        "  create a standalone `torch.utils.cpp_extension`/`PYBIND11_MODULE` module\n"
        "  unless the original file already uses that pattern.\n"
        "\n"
        "PRIORITY ORDER for picking an optimization path — check IN ORDER, use the\n"
        "FIRST that applies. Do NOT default to the C++ source you were given:\n"
        "(priority 0) IF kernel_url ends with `.cu`/`.cuh` AND the file is mostly\n"
        "  host-side ASM-kernel dispatch (it contains `hipModuleLoad`,\n"
        "  `AiterAsmKernel`, `.co`, `kernelName`, or `cfg_*`), then DO NOT try to\n"
        "  rewrite the C++ host code — the actual compute lives in pre-compiled\n"
        "  `.co` ASM artifacts you cannot rebuild. Instead, search for an\n"
        f"  equivalent Triton implementation under `{kernel_repo or '/sgl-workspace/aiter'}/aiter/ops/triton/...`\n"
        "  matching the kernel name (e.g. `aiter::gemm_a16w16` →\n"
        "  `aiter/ops/triton/gemm/basic/gemm_a16w16.py`) and optimize THAT Triton\n"
        "  kernel. This is how a 1.30x+ speedup is typically achieved on ASM-backed\n"
        "  kernels.\n"
        "\n"
        "How to do A/B benchmarking WITHOUT rebuilding aiter (which is forbidden):\n"
        "(option 1) TRITON path (preferred when available). If you took priority 0,\n"
        f"  write your version as a NEW Triton .py under ./optimized_versions/, then:\n"
        "  `from aiter.ops.triton.<path> import <fn> as baseline; "
        "from your_v3 import <fn> as optimized` — Triton is JIT-compiled, NO rebuild.\n"
        "(option 2) STANDALONE HIP/CUDA program. Write a single .hip/.cu under\n"
        "  ./benchmarks/ that #include's BOTH the aiter baseline header (e.g.\n"
        f'  `#include "{kernel_repo}/csrc/include/<the_target>.cuh"`) AND your\n'
        "  optimized .cuh from ./optimized_versions/, then build with:\n"
        f"  `hipcc -O3 -std=c++17 -DUSE_ROCM -I{kernel_repo or '/sgl-workspace/aiter'}/csrc/include "
        f"{platform_build_flag} -o ./benchmarks/bench ./benchmarks/bench.hip`.\n"
        "  Run as a single-process program; for multi-GPU collectives simulate ranks\n"
        "  with `std::thread` + `std::barrier` (no MPI/torchrun needed).\n"
        "(option 3) PYTORCH cpp_extension.load(). Build a .so from your modified\n"
        "  .cu/.cuh entirely under ./optimized_versions/, then `import` it and\n"
        "  compare against the original Python entry point. Concrete template:\n"
        "    ```python\n"
        "    import os, torch\n"
        "    from torch.utils.cpp_extension import load\n"
        "    HERE = os.path.dirname(os.path.abspath(__file__))\n"
        f"    AITER_INC = '{kernel_repo or '/sgl-workspace/aiter'}/csrc/include'\n"
        "    opt = load(\n"
        "        name='opt_kernel',\n"
        "        sources=[os.path.join(HERE, 'v1_my_kernel.cu')],\n"
        "        extra_include_paths=[AITER_INC],\n"
        "        extra_cuda_cflags=['-O3', '-std=c++17', '-DUSE_ROCM',\n"
        f"                           '{platform_build_flag}'],\n"
        "        verbose=False,\n"
        "    )\n"
        "    out_opt = opt.my_kernel(*args)            # YOUR optimized version\n"
        "    out_ref = aiter.<original_entry>(*args)   # baseline (unmodified)\n"
        "    torch.testing.assert_close(out_opt, out_ref)  # correctness\n"
        "    # then time both with torch.cuda.Event for speedup\n"
        "    ```\n"
        "  This is the ONLY way to A/B an ASM-backed C++ kernel without\n"
        "  rebuilding aiter (which is forbidden). Do NOT skip this\n"
        "  and write `speedup: N/A`.\n"
        "Pick whichever option matches the kernel; do NOT just measure baseline\n"
        "and write `speedup: N/A` — that wastes the run.\n"
    )
    # Multi-node sandbox is GPU-less: direct the LLM to delegate compile + execution
    # to a GPU-bearing pod. Trust only the in-process $INFERENCE_OPTIMIZER_NODES
    # signal the optimizer CLI exports at launch; never read a co-tenant-writable
    # state file, so a planted multi_node_state.json cannot force multi-node
    # fan-out guidance (same hardening as apply_kernel_patch._is_multi_node).
    try:
        is_multinode_run = int(os.environ.get("INFERENCE_OPTIMIZER_NODES", "0") or 0) >= 2
    except ValueError:
        is_multinode_run = False
    # Held separately from ``safety``: the GPU-less sandbox applies to whichever
    # backend runs, while the rest of ``safety`` describes the GEAK harness only.
    #
    # Two shapes, for the same reason the prompt has two. The constraint -- no
    # GPU on this node -- is a fact about the host and belongs to both backends.
    # The procedure below it is not: it routes measurements through
    # ``kernel-bench`` and names ``optimized_versions/`` and
    # ``optimization_report.md``, which are the two paths forge's workspace guard
    # refuses and the reason the deliverable contract was dropped in the first
    # place. Handing forge the whole block put them straight back, so a
    # multi-node forge run lost its first iteration exactly as before.
    multinode_notice = ""
    multinode_block = ""
    if is_multinode_run:
        multinode_notice = (
            "\nMULTI-NODE SANDBOX: this node has no GPU. Nothing you run here can\n"
            "compile against or measure a device -- `hipcc`, `torch.cuda.*` and\n"
            "`torch.utils.cpp_extension.load` will hang or crash. Do not stand up a\n"
            "measurement of your own; the harness that owns benchmarking dispatches\n"
            "it to a GPU-bearing pod.\n"
        )
        multinode_block = (
            "\nMULTI-NODE SANDBOX (no local GPU): every compile + benchmark\n"
            "step MUST be dispatched to a GPU-bearing RayJob pod. Do NOT\n"
            "call `hipcc`, `torch.cuda.*`, or `torch.utils.cpp_extension.load`\n"
            "directly; they have no GPU here and will hang or crash.\n"
            "Instead, for each A/B benchmark iteration:\n"
            "  1. Write the bench script (and any deps) under your\n"
            "     workspace ($WORKSPACE/benchmarks/, $WORKSPACE/optimized_versions/).\n"
            "  2. Invoke:\n"
            "       python3 -m hyperloom.inference_optimizer.multi_node kernel-bench \\\n"
            "         --workspace /tmp/kbench_$KERNEL_ID \\\n"
            "         --bench-command 'cd /tmp/kbench_$KERNEL_ID && bash bench.sh' \\\n"
            '         --files-b64-json \'<{"bench.sh":"<b64>","v1.cu":"<b64>",...}>\' \\\n'
            "         --result-glob 'result*.json'\n"
            "  3. Parse the printed JSON document's `result.stdout_tail`\n"
            "     and `result.artifacts[].content` for the speedup number;\n"
            "     write it into optimization_report.md as `[MICRO_SPEEDUP]`.\n"
            "Helper script to construct the b64 map cleanly:\n"
            '    python3 -c \'import base64,json,glob;print(json.dumps({p:base64.b64encode(open(p,"rb").read()).decode() for p in glob.glob("**/*",recursive=True) if __import__("os").path.isfile(p)}))\'\n'
            "Treat `kernel-bench` as your only measurement gate; everything\n"
            "else (code edits, correctness reasoning) still happens locally.\n"
        )
    safety += multinode_block
    if not is_multigpu:
        safety += "- Use the provided benchmark/test files above for correctness/perf measurement.\n"
    elif num_gpus >= 2:
        safety += (
            f"- Run REAL multi-GPU benchmarks via `torchrun --nproc_per_node={num_gpus}`. "
            "Save the bench script under `./benchmarks/` and the per-shape latency "
            "numbers in `./optimization_report.md`.\n"
        )
    else:
        safety += (
            "- Write a SINGLE-GPU micro-bench using torch tensors that exercises ONE "
            "rank's slice of the algorithm (e.g. local reduce + memcpy) so you can "
            "still measure compute/IO improvements.\n"
        )
    tracelens_context_block = ""
    # Fall back to the full analysis.md only when no hypothesis_block was rendered.
    if not hypothesis_block.strip():
        report_path_str = str(candidate.get("trace_report_path") or "")
        report_path = Path(report_path_str) if report_path_str else None
        if report_path and report_path.exists():
            try:
                full_report = report_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                full_report = ""
            if full_report:
                from hyperloom.inference_optimizer.tracelens_md import strip_base64_data_urls

                full_report = strip_base64_data_urls(full_report)
                rank = candidate.get("tracelens_pitem_rank")
                title = candidate.get("tracelens_pitem_title", "")
                if rank:
                    focus_line = (
                        f"Focus on **P{rank}: {title}** in the report below. "
                        "Other P-items are context only — do not optimize them.\n"
                    )
                else:
                    focus_line = "Use the report below as full context for this kernel.\n"
                tracelens_context_block = "\n## TraceLens Context\n\n" + focus_line + "\n" + full_report
    if (backend or "").strip().lower() == "forge":
        # Forge already owns everything the sections below describe, and two of
        # them fight it. The deliverables (``optimization_report.md``, a copy
        # under ``optimized_versions/``) arrive as new untracked paths that its
        # workspace guard refuses, which costs the iteration that wrote them --
        # while Hyperloom writes both itself from forge's published manifest and
        # git diff, so nothing reads the agent's copies. The A/B recipes
        # (standalone hipcc program, ``cpp_extension.load``) tell the agent to
        # stand up a second benchmark beside the driver its own in-session gate
        # scores, and a number from the wrong benchmark is worse than no number.
        # The budget protocol narrates a mini-swe-agent cost meter forge has no
        # header for, and the runtime-metadata block is labelled for GEAK's
        # parser -- forge reads the invocation spec instead, which carries the
        # same operands in more detail.
        #
        # What is left is what forge cannot derive: the trace evidence, and the
        # two source-attribution guards that keep a rewrite off a @compile_ops
        # wrapper and on the right ``__global__``.
        forge_sections = [
            f"# TASK: Optimize the `{kernel_name}` kernel",
            f"kernel_name: {kernel_name}\nkernel_url: {source_file}",
            # The target architecture, which forge cannot derive: it is told
            # which kernel to rewrite, not which card the measurement runs on,
            # and a gfx942 intrinsic emitted for a gfx950 host does not compile.
            platform_intro,
            hardware_notes,
            promotion_block,
            device_symbol_block,
            hypothesis_block,
            extra_context_block,
            benchmark_cases_block,
            # The harnesses the trace resolved, and the only channel by which a
            # review-verified ``benchmark_files`` reaches this backend at all.
            bench_block,
            priority_block,
            "Preserve function name, signature, decorators, and numerical behavior.",
            repo_block,
            multinode_notice,
            # The same fallback GEAK gets, for the same reason: the hypothesis
            # block is built from TraceLens p-items, and a kernel the trace
            # ranked without one renders it empty. Building it after the forge
            # return meant those kernels reached forge with no trace context at
            # all -- so the shape that keeps "the trace evidence forge cannot
            # derive" dropped exactly the rows where that evidence only exists
            # in analysis.md.
            tracelens_context_block,
        ]
        # Most of these blocks render empty for any given kernel; joining them
        # unfiltered leaves runs of blank lines where a section was skipped.
        return "\n\n".join(part.strip() for part in forge_sections if part.strip())
    # Use GEAK task_parser field names so its parser can extract them.
    return "\n".join(
        [
            f"# TASK: Optimize the `{kernel_name}` kernel",
            "",
            budget_protocol_block,
            platform_intro,
            "",
            f"kernel_name: {kernel_name}",
            f"kernel_url: {source_file}",
            f"kernel_type: {geak_kernel_type}",
            f"repo: {kernel_repo}",
            f"GPU percent: {candidate.get('gpu_pct', 'unknown')}",
            f"Shapes: {json.dumps(candidate.get('shapes', []), sort_keys=True)}",
            (
                "Shape contract: when `benchmark_shape_cases` is present in the "
                "metadata, benchmark its `primary_shape` first and use "
                "`supplementary_shapes` only as additional coverage. Do not invent "
                "shapes or reorder tensor arguments."
                if kernel_metadata.get("benchmark_shape_cases")
                else ""
            ),
            promotion_block,
            device_symbol_block,
            "",
            "Kernel runtime metadata (structured context for GEAK; unknown fields are null, empty arrays, or empty objects):",
            "```json",
            json.dumps(kernel_metadata, indent=2, sort_keys=True),
            "```",
            "",
            hardware_notes,
            hypothesis_block,
            extra_context_block,
            benchmark_cases_block,
            priority_block,
            "",
            "Preserve function name, signature, decorators, and numerical behavior.",
            "Return complete optimized code plus explanation of correctness assumptions.",
            repo_block,
            bench_block,
            safety,
            source_block,
            tracelens_context_block,
        ]
    )


def ray_available() -> bool:
    """Report whether the optional ``ray`` dependency can be imported.

    Returns:
        bool: True when ``import ray`` succeeds; False on any import error.
    """
    try:
        import ray  # noqa: F401

        return True
    except Exception:
        return False


def _backends_module_dir() -> Path:
    """Return the directory holding the per-backend submitter modules.

    Returns:
        Path: The ``backends`` directory next to this module.
    """
    return Path(__file__).resolve().parent / "backends"


def _import_backend(name: str):
    """Dynamically import a per-backend submitter module.

    The ``backends`` directory is added to ``sys.path`` so its submodules can
    cross-import each other.

    Args:
        name: The backend module name (without ``.py``).

    Returns:
        The imported backend module.
    """
    backends_dir = _backends_module_dir()
    if str(backends_dir) not in sys.path:
        sys.path.insert(0, str(backends_dir))
    import importlib

    return importlib.import_module(name)


def _kernel_agent_root() -> Path:
    """Resolve the kernel-agent tools output root.

    Uses :func:`workspace_root` (which warns once when ``$USER_DATA_PATH`` is
    unset).

    Returns:
        The ``$USER_DATA_PATH/kernel-agent`` output root path.
    """
    return Path(workspace_root()) / "kernel-agent"


def _forge_output_dir(session_id: str, prompt_file: Path) -> Path:
    """Return the per-attempt output directory for a Forge run.

    Per-attempt so Forge artifacts (forge_loop.log, forge_experiments/,
    optimization_report, optimized_versions/) are scoped under their own
    ``forge/`` root.

    Args:
        session_id: Session identifier for the run.
        prompt_file: Prompt file whose stem scopes the attempt directory.

    Returns:
        The created ``.../forge/<session_id>/<prompt_stem>`` directory.
    """
    out = _kernel_agent_root() / "forge" / session_id / prompt_file.stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def _git_checkout_fallback(kernel_repo: str, log_path: Path) -> None:
    """Run a best-effort ``git checkout -- .`` to undo rogue agent writes.

    Idempotent and safe to call when the repo has no changes.

    Args:
        kernel_repo: Path to the kernel repo to clean.
        log_path: Path to append checkout diagnostics.
    """
    if not kernel_repo:
        return
    git_dir = Path(kernel_repo) / ".git"
    if not git_dir.exists():
        return
    try:
        proc = subprocess.run(
            ["git", "-C", kernel_repo, "checkout", "--", "."],
            capture_output=True,
            text=True,
            timeout=60,
        )
        append_log(log_path, f"[git-fallback] checkout rc={proc.returncode}")
        if proc.stderr.strip():
            append_log(log_path, f"[git-fallback] stderr: {proc.stderr.strip()[:400]}")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        append_log(log_path, f"[git-fallback] failed: {type(exc).__name__}: {exc}")


def invoke_backend(
    backend: str,
    prompt_file: Path,
    source_file: str,
    args: argparse.Namespace,
    candidate: dict[str, Any] | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Run a backend via the self-contained submitters.

    Returns a normalized dict: returncode, stdout_tail, stderr_tail, stdout,
    gpu_ids, elapsed_s, cmd, optimized_path (optional), cli_workspace (forge).

    Args:
        backend (str): Backend name to run; only ``forge`` is supported (any
            other value returns an unknown-backend result with returncode 2).
            ``geak`` is a whole-pipeline phase delegate, not a value accepted
            here.
        prompt_file (Path): File containing the rendered optimization prompt.
        source_file (str): Path to the kernel source to be rewritten.
        args (argparse.Namespace): Parsed CLI args carrying backend settings.
        candidate (dict[str, Any] | None): Kernel candidate dict, when known.
        log_path (Path | None): Optional run log for backend diagnostics.

    Returns:
        dict[str, Any]: A normalized result dict (returncode, stdout/stderr
            tails, gpu_ids, elapsed_s, cmd, and optional optimized_path /
            cli_workspace).
    """
    budget_min = float(getattr(args, "budget_minutes", 60) or 60)
    timeout_s = max(60, int(budget_min * 60))
    prefer_ray = ray_available()
    candidate = candidate or {}
    kernel_repo = str(candidate.get("kernel_repo") or "")
    # GPU count: CLI override, then candidate hint, then 1.
    num_gpus = max(1, int(getattr(args, "num_gpus", 0) or 0) or int(candidate.get("num_gpus_recommended") or 1))

    try:
        if backend == "forge":
            # Kernel-Forge backend: runs inside a git worktree of kernel_repo and
            # emits optimized_versions/ + optimization_report.md.
            forge = _import_backend("forge_submit")
            out_dir = _forge_output_dir(args.session_id, prompt_file)
            invocation_spec_path = out_dir / invocation_spec_filename(candidate)
            try:
                invocation_spec = build_invocation_spec(candidate, source_file=source_file)
                write_invocation_spec(invocation_spec_path, invocation_spec)
                if log_path is not None:
                    append_log(log_path, f"[invocation_spec] wrote {invocation_spec_path}")
            except Exception as exc:
                # Evidence extraction must not block an otherwise runnable Forge
                # attempt while the spec is not yet a required CLI input.
                if log_path is not None:
                    append_log(
                        log_path,
                        f"[invocation_spec] failed: {type(exc).__name__}: {exc}",
                    )
            result = forge.submit(
                source_file=source_file,
                prompt_file=prompt_file,
                output_dir=out_dir,
                source_type=str((candidate or {}).get("source_type") or "unknown"),
                candidate=candidate or {},
                num_gpus=num_gpus,
                timeout_s=timeout_s,
                prefer_ray=prefer_ray,
                kernel_repo=kernel_repo,
                invocation_spec_file=(str(invocation_spec_path) if invocation_spec_path.is_file() else ""),
            )
            result["output_dir"] = str(out_dir)
            if invocation_spec_path.is_file():
                result["invocation_spec_path"] = str(invocation_spec_path)
            # submit() decides the FlyDSL rewrite route and records the verdict
            # on its result, but nothing persisted that dict, so opting in with
            # HYPERLOOM_FORGE_REWRITE_BY_FLYDSL and silently getting plain
            # forge-loop was indistinguishable from the route running. This is
            # the one channel the attempt log is known to surface.
            if log_path is not None:
                route = result.get("flydsl_rewrite")
                if isinstance(route, dict):
                    verdict = "eligible" if route.get("eligible") else f"declined:{route.get('reason')}"
                    append_log(log_path, f"flydsl_rewrite_route={verdict} {route.get('detail') or ''}".rstrip())
            return result
        return {
            "returncode": 2,
            "stdout_tail": f"unknown backend: {backend}",
            "stderr_tail": "",
            "stdout": "",
            "gpu_ids": "",
            "elapsed_s": 0.0,
            "cmd": [],
        }
    finally:
        # Undo rogue writes under the kernel repo. Skip for forge: it manages its
        # own restore, which a blanket `git checkout -- .` would overwrite.
        if log_path is not None and backend != "forge":
            _git_checkout_fallback(kernel_repo, log_path)


def run_attempt(
    backend: str,
    *,
    args: argparse.Namespace,
    candidate: dict[str, Any],
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Run a single backend optimization attempt and record its artifacts.

    Renders the prompt, invokes the backend (or a dry-run placeholder),
    captures stdout to a durable log, locates any optimized-source artifact,
    and returns a structured attempt record.

    Args:
        backend (str): Backend name to run for this attempt.
        args (argparse.Namespace): Parsed CLI args controlling the attempt.
        candidate (dict[str, Any]): Kernel candidate dict being optimized.
        run_dir (Path): Run directory for prompts/optimized/log outputs.
        log_path (Path): Run log appended with attempt progress.

    Returns:
        dict[str, Any]: An attempt record (id, backend, status, returncode,
            elapsed, optimized_path, backend_paths, stdout tail, etc.).
    """
    attempt_id = f"{backend}-{uuid.uuid4().hex[:8]}"
    prompt_dir = run_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / f"{attempt_id}.md"
    prompt_file.write_text(build_prompt(candidate, args, backend=backend), encoding="utf-8")

    source_file = args.source_file or str(candidate.get("source_file") or "")
    started = time.time()
    append_log(log_path, f"[attempt {attempt_id}] backend={backend}")

    source_suffix = Path(source_file).suffix if source_file else ".txt"
    # Dry-run emits a source-suffixed placeholder; real runs capture stdout to a
    # `.log` so _extract_source_block scans for fenced code, not the log itself.
    if args.dry_run:
        optimized_path = run_dir / "optimized" / f"{attempt_id}_optimized{source_suffix or '.txt'}"
    else:
        optimized_path = run_dir / "optimized" / f"{attempt_id}_stdout.log"
    optimized_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        status = "completed"
        returncode = 0
        stdout_tail = "[dry-run] backend execution skipped"
        full_stdout = stdout_tail
        if source_suffix == ".py":
            placeholder = "def optimized_kernel_placeholder():\n    return None\n"
        else:
            placeholder = 'extern "C" __global__ void optimized_kernel_placeholder() {}\n'
        optimized_path.write_text(placeholder, encoding="utf-8")
        result = {}
    else:
        append_log(log_path, f"$ invoke_backend({backend})")
        result = invoke_backend(backend, prompt_file, source_file, args, candidate=candidate, log_path=log_path)
        returncode = int(result.get("returncode", 1))
        full_stdout = result.get("stdout") or result.get("stdout_tail") or ""
        stdout_tail = (full_stdout or "")[-4000:] or result.get("stderr_tail", "")
        if returncode == 0:
            status = "completed"
        elif returncode == 124:
            status = "timeout"
        else:
            status = "failed"
        # Materialise the stdout `.log` (audit trail + code-fence extraction fallback).
        if full_stdout.strip():
            optimized_path.write_text(full_stdout, encoding="utf-8")
        append_log(log_path, stdout_tail)

    elapsed = round(time.time() - started, 3)

    backend_paths: dict[str, str] = {}
    if not args.dry_run:
        if isinstance(result, dict):
            forge_workspace = str(result.get("forge_workspace") or "")
            if forge_workspace:
                backend_paths["forge_workspace"] = forge_workspace
            artifacts = result.get("artifacts")
            if isinstance(artifacts, list):
                forge_patch = next(
                    (str(path) for path in artifacts if str(path).endswith(("forge.patch", ".diff"))),
                    "",
                )
                if forge_patch:
                    backend_paths["forge_patch"] = forge_patch
            changed_files = result.get("changed_files")
            if isinstance(changed_files, list):
                backend_paths["forge_changed_files"] = json.dumps(changed_files)
            for result_key, backend_key in (
                ("best_manifest", "forge_best_manifest"),
                ("canonical_patch_path", "forge_canonical_patch"),
                ("canonical_files_root", "forge_canonical_files_root"),
                ("forge_workspace", "forge_workspace"),
            ):
                value = str(result.get(result_key) or "")
                if value:
                    backend_paths[backend_key] = value
        out_dir = result.get("output_dir") if isinstance(result, dict) else ""
        if out_dir:
            backend_paths["output_dir"] = out_dir
            invocation_spec_path = result.get("invocation_spec_path") or "" if isinstance(result, dict) else ""
            if invocation_spec_path:
                backend_paths["invocation_spec"] = str(invocation_spec_path)
            checkpoint_path = result.get("checkpoint_path") or "" if isinstance(result, dict) else ""
            if checkpoint_path:
                backend_paths["forge_checkpoint"] = str(checkpoint_path)
            cli_workspace = (result.get("cli_workspace") or "") if isinstance(result, dict) else ""
            session_id_oob = (result.get("session_id") or "") if isinstance(result, dict) else ""
            cli_log = ""
            if cli_workspace:
                exec_log = Path(cli_workspace) / "execution.log"
                if exec_log.exists():
                    cli_log = str(exec_log)
                # Scan for partial outputs even when returncode != 0.
                opt_dir = Path(cli_workspace) / "optimized_versions"
                if opt_dir.is_dir():
                    files = sorted(opt_dir.iterdir(), key=lambda p: p.stat().st_mtime)
                    if files:
                        backend_paths["partial_optimized_count"] = str(len(files))
                        backend_paths["partial_latest_optimized"] = str(files[-1])
                report = Path(cli_workspace) / "optimization_report.md"
                if report.exists():
                    backend_paths["partial_report"] = str(report)
                forge_patch = opt_dir / "forge.patch"
                if forge_patch.is_file():
                    backend_paths.setdefault("forge_patch", str(forge_patch))
            # Rescue: surface fresh ~/optimized_versions/ files when the
            # workspace's dir is empty.
            home_opt = Path("/home/user/optimized_versions")
            if (
                cli_workspace
                and (
                    not (Path(cli_workspace) / "optimized_versions").is_dir()
                    or not list((Path(cli_workspace) / "optimized_versions").iterdir())
                )
                and home_opt.is_dir()
            ):
                rescued = sorted(
                    [p for p in home_opt.iterdir() if p.is_file() and p.stat().st_mtime >= started],
                    key=lambda p: p.stat().st_mtime,
                )
                if rescued:
                    backend_paths["partial_optimized_count"] = str(len(rescued))
                    backend_paths["partial_latest_optimized"] = str(rescued[-1])
                    backend_paths["partial_optimized_rescued_from"] = str(home_opt)
                    home_report = Path("/home/user/optimization_report.md")
                    if (
                        home_report.is_file()
                        and home_report.stat().st_mtime >= started
                        and "partial_report" not in backend_paths
                    ):
                        backend_paths["partial_report"] = str(home_report)
            if cli_workspace:
                backend_paths["cli_workspace"] = cli_workspace
            if cli_log:
                backend_paths["cli_execution_log"] = cli_log
            if session_id_oob:
                backend_paths["kernel_session_id"] = session_id_oob
            # Promote a timed-out / failed attempt with on-disk artifacts to
            # "partial", except on a persistent inner-LLM auth loop.
            partial_evidence_keys = (
                "partial_latest_optimized",
                "partial_report",
            )
            unrecoverable_timeout = bool(
                isinstance(result, dict) and result.get("timed_out") and not result.get("salvaged")
            )
            auth_loop_hits = _count_auth_failures(full_stdout)
            if auth_loop_hits >= _AUTH_RETRY_THRESHOLD:
                backend_paths["auth_failure_count"] = str(auth_loop_hits)
                backend_paths["auth_failure_marker"] = "persistent_inner_llm_401_loop_no_partial_promotion"
                # Force a non-partial terminal state so make_proposal REVERTs.
                if status == "timeout":
                    status = "failed"
            elif (
                not unrecoverable_timeout
                and status in {"timeout", "failed"}
                and any(k in backend_paths for k in partial_evidence_keys)
            ):
                status = "partial"

    return {
        "attempt_id": attempt_id,
        "backend": backend,
        "status": status,
        "error_type": status if status in {"backend_not_installed", "timeout"} else "",
        "returncode": returncode,
        # Structured backend self-skip marker so the classifier labels the
        # kernel ``skip`` instead of a failure without parsing free-text stdout.
        "skipped": bool(result.get("skipped")) if isinstance(result, dict) else False,
        "timed_out": bool(result.get("timed_out")) if isinstance(result, dict) else False,
        "salvaged": bool(result.get("salvaged")) if isinstance(result, dict) else False,
        "best_commit": str(result.get("best_commit") or "") if isinstance(result, dict) else "",
        "flydsl_applyback": (
            dict(result.get("flydsl_applyback") or {})
            if isinstance(result, dict) and isinstance(result.get("flydsl_applyback"), dict)
            else {}
        ),
        "forge_result": (
            dict(result.get("forge_result") or {})
            if isinstance(result, dict) and isinstance(result.get("forge_result"), dict)
            else {}
        ),
        "kb_experience": (
            dict(result.get("kb_experience") or {})
            if isinstance(result, dict) and isinstance(result.get("kb_experience"), dict)
            else {}
        ),
        "pristine_baseline_ms": (result.get("pristine_baseline_ms") if isinstance(result, dict) else None),
        "search_start_ms": (result.get("search_start_ms") if isinstance(result, dict) else None),
        "best_ms": result.get("best_ms") if isinstance(result, dict) else None,
        "mean_case_speedup": (result.get("mean_case_speedup") if isinstance(result, dict) else None),
        "search_start_mean_case_speedup": (
            result.get("search_start_mean_case_speedup") if isinstance(result, dict) else None
        ),
        "total_improved": (bool(result.get("total_improved")) if isinstance(result, dict) else False),
        "incremental_improved": (bool(result.get("incremental_improved")) if isinstance(result, dict) else False),
        "improved": bool(result.get("improved")) if isinstance(result, dict) else False,
        "improved_during_search": (bool(result.get("improved_during_search")) if isinstance(result, dict) else False),
        "elapsed_s": elapsed,
        "prompt_path": str(prompt_file),
        "optimized_path": str(optimized_path) if optimized_path.exists() else "",
        "stdout_tail": stdout_tail,
        "created_at": utc_now(),
        "backend_paths": backend_paths,
    }


_SPEEDUP_PATTERNS = [
    # Match `speedup: 1.28x` / `Speedup: **1.076x**` / `avg=1.044x`.
    re.compile(r"(?im)^\s*\[micro_speedup\]\s*([0-9]+(?:\.[0-9]+)?)\s*[xX]\b"),
    re.compile(r"(?i)\bspeedup\b[^\n]{0,40}?([0-9]+(?:\.[0-9]+)?)\s*[xX]"),
    re.compile(r"(?i)\bavg(?:erage)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*[xX]\s+(?:speedup|across)"),
    re.compile(r"(?i)\b([0-9]+(?:\.[0-9]+)?)\s*[xX]\s+(?:speedup|faster)"),
]


# Inner-LLM auth failure markers; >= AUTH_RETRY_THRESHOLD signals a dead-end.
_AUTH_FAILURE_PATTERNS = [
    re.compile(r"\b401\b[^\n]{0,80}(unauthor|forbidden|client\s*error)", re.IGNORECASE),
    re.compile(r"HTTP/\d\.\d\s+401\b"),
    re.compile(r"Authentication\s*Error|Invalid\s*API\s*Key|invalid[._]api[._]key", re.IGNORECASE),
    re.compile(r"Subscription[- ]Key[^\n]{0,80}(missing|invalid|not\s*present)", re.IGNORECASE),
    re.compile(r"Primus\.00009\s+token\s+not\s+present", re.IGNORECASE),
]
_AUTH_RETRY_THRESHOLD = 3


def _count_auth_failures(text: str) -> int:
    """Count inner-LLM auth-failure markers in captured text.

    Distinguishes a transient 401 from an unrecoverable loop.

    Args:
        text: The captured backend output to scan.

    Returns:
        The number of auth-failure markers found.
    """
    if not text:
        return 0
    total = 0
    for pat in _AUTH_FAILURE_PATTERNS:
        total += sum(1 for _ in pat.finditer(text))
    return total


def _read_text_file(path: str | Path, *, errors: str | None = "replace") -> str | None:
    """Read a text file, returning ``None`` when missing or unreadable."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        if errors is None:
            return p.read_text(encoding="utf-8")
        return p.read_text(encoding="utf-8", errors=errors)
    except Exception:
        return None


def _extract_speedup_from_report(report_path: str | Path) -> float | None:
    """Scan an external ``optimization_report.md`` for a speedup figure.

    Averages the top-3 reported speedups to dampen cherry-picked best-shape
    numbers.

    Args:
        report_path: Path to the optimization report.

    Returns:
        The estimated speedup, or ``None`` when the report is missing or no
        plausible figure is found.
    """
    text = _read_text_file(report_path)
    if text is None:
        return None
    found: list[float] = []
    for pat in _SPEEDUP_PATTERNS:
        for m in pat.finditer(text):
            try:
                v = float(m.group(1))
                # Reject obvious junk.
                if 0.3 <= v <= 50.0:
                    found.append(v)
            except ValueError:
                continue
    if not found:
        return None
    found.sort(reverse=True)
    top = found[:3]
    return round(sum(top) / len(top), 4)


def _extract_correctness_from_report(report_path: str | Path) -> bool | None:
    """Best-effort correctness signal from backend markdown/json reports.

    Looks for an explicit ``[correctness] pass/fail`` marker first, then
    falls back to scanning for known pass/fail phrasings.

    Args:
        report_path (str | Path): Path to a backend report (markdown/text).

    Returns:
        bool | None: True/False when a correctness signal is found, or None
            when the file is missing/unreadable or no signal is present.
    """
    text = _read_text_file(report_path)
    if text is None:
        return None
    lower = text.lower()
    marker = re.search(r"(?im)^\s*\[correctness\]\s*(pass|passed|fail|failed)\b", text)
    if marker:
        return marker.group(1).lower().startswith("pass")
    fail_markers = (
        "correctness failed",
        "incorrect",
        "mismatch",
        "assert_close failed",
        "not close",
        "wrong output",
        "validation failed",
        "correctness: fail",
        "correctness: failed",
        "does not match",
        "do not match",
        "failed against reference",
        "reference check failed",
    )
    pass_markers = (
        "correctness passed",
        "correctness: pass",
        "all tests passed",
        "assert_close passed",
        "torch.testing.assert_close passed",
        "validation passed",
        "matches reference",
        "match reference",
        "matched reference",
        "matches the reference",
        "verified against original",
        "verified against the original",
        "validated against original",
        "validated against the original",
        "validated against baseline",
        "outputs match",
        "output matches",
        "numerically matches",
        "reference comparison passed",
    )
    if any(marker in lower for marker in fail_markers):
        return False
    if any(marker in lower for marker in pass_markers):
        return True
    return None


_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".hip",
    ".py",
}
_FENCE_RE = re.compile(
    r"```(?P<lang>[A-Za-z0-9_+.-]*)\s*\n(?P<body>.*?)\n```",
    re.DOTALL,
)


# Shared source-completeness heuristic (see _io_utils.source_text_looks_complete).
_source_text_looks_complete = source_text_looks_complete


def _extract_source_block(text_path: Path, target_suffix: str, output_path: Path) -> str:
    """Extract a complete fenced source block from a backend stdout log.

    Scans ``text_path`` for fenced code blocks whose language hint matches
    ``target_suffix`` and that look like complete source; the last such
    block is written to ``output_path``.

    Args:
        text_path (Path): File (typically backend stdout) to scan for fences.
        target_suffix (str): Target source suffix used to filter fences and
            validate completeness (e.g. ``.cu``, ``.py``).
        output_path (Path): Where the extracted source block is written.

    Returns:
        str: The string path of the written artifact, or empty string when
            no suitable code block is found.
    """
    try:
        text = text_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lang_hints = {
        ".py": {"python", "py"},
        ".cu": {"cuda", "cu", "cpp", "c++"},
        ".cuh": {"cuda", "cu", "cpp", "c++"},
        ".hip": {"hip", "cpp", "c++"},
        ".cpp": {"cpp", "c++"},
        ".cc": {"cpp", "c++"},
        ".c": {"c"},
        ".h": {"c", "cpp", "c++"},
        ".hpp": {"cpp", "c++"},
    }.get(target_suffix, set())
    candidates: list[str] = []
    for match in _FENCE_RE.finditer(text):
        lang = match.group("lang").strip().lower()
        body = match.group("body").strip()
        if lang_hints and lang and lang not in lang_hints:
            continue
        if _source_text_looks_complete(body, target_suffix):
            candidates.append(body)
    if not candidates:
        return ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(candidates[-1].rstrip() + "\n", encoding="utf-8")
    return str(output_path)


def _candidate_artifact_paths(
    attempt: dict[str, Any],
    target_suffix: str,
    *,
    source_file: str = "",
    kernel_repo: str = "",
) -> list[Path]:
    """Collect candidate optimized-artifact paths for an attempt, in priority.

    Gathers partial/patch outputs, ``optimized_versions`` directories, and the
    recorded ``optimized_path`` into a priority-ordered list for downstream
    source selection.

    Args:
        attempt (dict[str, Any]): Attempt record carrying ``backend_paths``
            and ``optimized_path``.
        target_suffix (str): Target source suffix (used by callers to match).
        source_file (str): Original kernel source path, for worktree mapping.
        kernel_repo (str): Kernel repo root, for worktree-relative resolution.

    Returns:
        list[Path]: Candidate artifact paths ordered most- to least-precise.
    """
    paths: list[Path] = []
    bp = attempt.get("backend_paths") or {}
    value = bp.get("partial_latest_optimized")
    if value:
        paths.append(Path(value))
    cli_workspace = bp.get("cli_workspace")
    if cli_workspace:
        opt_dir = Path(cli_workspace) / "optimized_versions"
        if opt_dir.is_dir():
            paths.extend(
                sorted(
                    (p for p in opt_dir.iterdir() if p.is_file()),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            )
    out_dir = bp.get("output_dir")
    if out_dir:
        opt_dir = Path(out_dir) / "optimized_versions"
        if opt_dir.is_dir():
            paths.extend(
                sorted(
                    (p for p in opt_dir.iterdir() if p.is_file()),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            )
    optimized_path = attempt.get("optimized_path")
    if optimized_path:
        paths.append(Path(optimized_path))

    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


_BACKUP_SUFFIXES = {".orig", ".rej", ".bak"}


def _unquote_diff_path(raw: str) -> str:
    """Decode a git diff header path, handling C-style quoting.

    Git emits paths with special characters as C-quoted strings (e.g.
    ``"b/\\303\\251.py"``). A naive ``..``/absolute check on the still-quoted
    string can be bypassed, so decode it first.

    Args:
        raw (str): The raw token following ``+++ `` / ``--- `` (already
            whitespace-trimmed of a trailing tab-timestamp by the caller).

    Returns:
        str: The decoded path, or the input unchanged when it is not quoted.
    """
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        try:
            return raw[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape")
        except (UnicodeDecodeError, ValueError):
            return raw[1:-1]
    return raw


def _strip_diff_prefix(path: str, strip: int) -> str:
    """Drop the first ``strip`` path components, mirroring ``-p<N>``.

    Empty components (from ``//`` or a leading ``/``) are dropped first so the
    strip count operates on real path segments and cannot resurrect an absolute
    path via an empty component.

    Args:
        path (str): A diff target path (e.g. ``b/dir/file.py``).
        strip (int): Number of leading components to remove.

    Returns:
        str: The post-strip relative path, or ``""`` when nothing remains.
    """
    parts = [p for p in path.split("/") if p]
    return "/".join(parts[strip:]) if len(parts) > strip else ""


def _select_patch_section(patch_text: str, target_file: str) -> tuple[str, str] | None:
    """Slice out the per-file diff section that targets ``target_file``.

    Reconstruction only has the single original kernel on disk, so extract just
    the matched file's section and reject anything unsafe or unusable.

    Args:
        patch_text (str): The full unified diff.
        target_file (str): Original kernel path the section must target.

    Returns:
        tuple[str, str] | None: ``(section_text, raw_target_path)`` for the
            matched file, or ``None`` when no usable/safe section matches.
    """
    if not patch_text.strip():
        return None
    # Split into per-file blocks: prefer ``diff --git`` boundaries, else ``---``/``+++`` pairs.
    if re.search(r"(?m)^diff --git ", patch_text):
        blocks = [b for b in re.split(r"(?m)^(?=diff --git )", patch_text) if b.strip()]
    else:
        blocks = [b for b in re.split(r"(?m)^(?=--- )", patch_text) if b.strip()]

    want = Path(target_file).name
    want_parts = Path(target_file).parts
    matches: list[tuple[int, str, str]] = []  # (tail_len, raw_path, block)
    for block in blocks:
        m = re.search(r"(?m)^\+\+\+ (.+)$", block)
        if not m:
            continue
        raw = _unquote_diff_path(m.group(1).split("\t", 1)[0].strip())
        if raw == "/dev/null":
            continue
        # Strip a leading a/ or b/ for tail comparison only.
        cmp_path = re.sub(r"^[ab]/", "", raw)
        cmp_parts = Path(cmp_path).parts
        if not cmp_parts or Path(cmp_path).name != want:
            continue
        if Path(cmp_path).suffix.lower() in _BACKUP_SUFFIXES:
            continue
        # Longest common path-tail length (basename match is the floor).
        tail = 0
        for a, b in zip(reversed(cmp_parts), reversed(want_parts)):
            if a != b:
                break
            tail += 1
        matches.append((tail, raw, block))

    if not matches:
        return None
    best_tail = max(t for t, _, _ in matches)
    best = [(r, b) for t, r, b in matches if t == best_tail]
    if len(best) != 1:
        return None  # genuine ambiguity: two distinct targets share the tail
    raw, block = best[0]
    if "\n@@" not in ("\n" + block):
        return None  # rename/mode-only/binary section: no hunk to apply
    # Reject if any usable strip level yields an absolute or ``..``-containing path.
    stripped = [_strip_diff_prefix(raw, s) for s in (1, 0, 2)]
    if not any(stripped):
        return None
    if any(rel and (os.path.isabs(rel) or ".." in Path(rel).parts) for rel in stripped):
        return None
    return block, raw


def _reconstruct_source_from_patch(
    patch_path: Path,
    target_file: str,
    output_path: Path,
) -> str:
    """Reconstruct a complete source file by applying a unified diff.

    When no full-source artifact is found, the original kernel at ``target_file``
    plus the patch deterministically reconstruct the optimized source. The
    matched file's section is applied inside an isolated temp dir (so an untrusted
    patch header cannot escape it), via ``git apply`` then ``patch``.

    Args:
        patch_path (Path): Unified diff produced by the backend.
        target_file (str): Original (pre-patch) kernel source path.
        output_path (Path): Where the reconstructed source is written.

    Returns:
        str: The string path of the reconstructed source, or empty string when
            the original is missing or the patch does not apply cleanly.
    """
    original = Path(target_file)
    if not patch_path.is_file() or not original.is_file():
        return ""
    try:
        original_text = original.read_text(encoding="utf-8", errors="replace")
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    selected = _select_patch_section(patch_text, target_file)
    if selected is None:
        return ""
    section_text, raw_target = selected

    for strip in (1, 0, 2):
        rel = _strip_diff_prefix(raw_target, strip)
        if not rel or os.path.isabs(rel) or ".." in Path(rel).parts:
            continue
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            work = tmp / rel
            section = tmp / "section.patch"
            # Containment guard: confirm the write path stays inside the temp dir.
            try:
                work.resolve().relative_to(tmp.resolve())
            except ValueError:
                continue
            try:
                work.parent.mkdir(parents=True, exist_ok=True)
                work.write_text(original_text, encoding="utf-8")
                section.write_text(section_text, encoding="utf-8")
            except OSError:
                continue
            applied = False
            # git apply contained to tmp; git refuses any path escape itself.
            try:
                rc = subprocess.run(
                    ["git", "apply", f"-p{strip}", "section.patch"],
                    capture_output=True,
                    text=True,
                    cwd=str(tmp),
                    check=False,
                )
                applied = rc.returncode == 0 and work.is_file()
            except (OSError, ValueError):
                applied = False
            if not applied:
                # patch with an explicit file arg only writes ``work``.
                try:
                    with section.open(encoding="utf-8", errors="replace") as pf:
                        rc = subprocess.run(
                            ["patch", f"-p{strip}", "--force", "--no-backup-if-mismatch", str(work)],
                            stdin=pf,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                    applied = rc.returncode == 0
                except (OSError, ValueError):
                    applied = False
            if not applied:
                continue
            try:
                patched_text = work.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if (
                patched_text
                and patched_text != original_text
                and _source_text_looks_complete(patched_text, output_path.suffix.lower())
            ):
                try:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(patched_text, encoding="utf-8")
                except OSError:
                    return ""
                return str(output_path)
    return ""


def build_patch_snapshot(
    patch_path: str,
    *,
    worktree: Path | None,
    kernel_repo: str,
    clean_base: str,
    out_dir: Path,
) -> dict[str, Any] | None:
    """Stage byte-exact final contents for every file a patch writes.

    Parse the patch as a manifest, then for each non-deleted path materialise its
    exact final bytes into ``out_dir`` (mirrored at the same repo-relative path).
    Content is sourced, in priority order, from the backend's ``worktree`` and,
    failing that, by reconstructing from ``clean_base`` + the patch in a contained
    temp dir. Only manifest paths are staged, never the whole worktree.

    Args:
        patch_path (str): The winning unified diff.
        worktree (Path | None): The backend worktree of final files, if any.
        kernel_repo (str): Repo root, for worktree-relative resolution.
        clean_base (str): Repo root holding pristine pre-patch sources, used to
            reconstruct content when the worktree lacks a path.
        out_dir (Path): Snapshot staging dir to populate.

    Returns:
        dict[str, Any] | None: ``{"snapshot_dir", "descriptors", "patch_path"}``
            when every write path was materialised byte-exact, else ``None``
            (caller treats this as non-deployable -> hard fail downstream).
    """
    import apply_kernel_patch as _akp

    try:
        patch_text = Path(patch_path).read_text(encoding="utf-8", errors="replace")
        descriptors = _akp.parse_patch_manifest(patch_text)
    except (OSError, ValueError):
        return None
    if not descriptors:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    for desc in descriptors:
        if desc["op"] != "write":
            continue
        rel = desc["path"]
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        sourced = False
        # 1) worktree ground-truth file at the same relative path.
        if worktree is not None:
            cand = worktree / rel
            if cand.is_file() or cand.is_symlink():
                try:
                    import shutil as _shutil

                    _shutil.copy2(cand, dst, follow_symlinks=False)
                    sourced = True
                except OSError:
                    sourced = False
        # 2) reconstruct from clean base + patch (single-file slice).
        if not sourced and clean_base:
            base_file = Path(clean_base) / rel
            reconstructed = (
                _reconstruct_source_from_patch(Path(patch_path), str(base_file), dst.with_suffix(dst.suffix + ".recon"))
                if base_file.is_file()
                else ""
            )
            if reconstructed:
                try:
                    Path(reconstructed).replace(dst)
                    sourced = True
                except OSError:
                    sourced = False
        if not sourced:
            # No byte-exact content for this path -> non-deployable (hard fail).
            return None

    return {
        "snapshot_dir": str(out_dir),
        "descriptors": descriptors,
        "patch_path": str(patch_path),
        "repo_root": str(kernel_repo or ""),
    }


def prepare_deploy_patch(
    patch_path: str,
    *,
    output_path: Path,
) -> str:
    """Remove generated Python cache entries from a source deployment patch."""
    import apply_kernel_patch as _akp

    try:
        patch_text = Path(patch_path).read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return ""
    if re.search(r"(?m)^diff --git ", patch_text):
        blocks = [block for block in re.split(r"(?m)^(?=diff --git )", patch_text) if block.strip()]
    else:
        blocks = [block for block in re.split(r"(?m)^(?=--- )", patch_text) if block.strip()]
    kept: list[str] = []
    for block in blocks:
        try:
            descriptors = _akp.parse_patch_manifest(block)
        except ValueError:
            return ""
        generated = any(
            "__pycache__" in Path(str(desc.get("path") or "")).parts
            or Path(str(desc.get("path") or "")).suffix.lower() in {".pyc", ".pyo"}
            for desc in descriptors
        )
        if not generated:
            kept.append(block)
    if not kept:
        return ""
    deploy_text = "".join(kept)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(deploy_text, encoding="utf-8")
    return str(output_path)


def resolve_deploy_repo_root(
    target_file: str,
    descriptors: list[dict[str, Any]],
    *,
    explicit_root: str = "",
) -> str:
    """Resolve the consumer repo root for repo-relative patch paths.

    Producer worktree paths are not portable to installed framework trees.
    Resolve against an explicit root when supplied; then anchor on the traced
    source, whose absolute path already names the root once the entry that
    describes it is stripped off its tail; and only then fall back to walking
    ancestors and requiring exactly one whose existing preimage files match
    every non-new patch entry.

    The anchor exists because the ancestor walk cannot break a tie it has no
    information about. A producer rooted below the consumer's package -- aiter
    puts a copy of the same file under both ``ops/triton`` and
    ``ops/triton/_triton_kernels``, and forge's worktree sits at the deeper one
    -- exports ``gemm/basic/<kernel>.py``, which then resolves under two
    ancestors of the traced source. The walk finds two matches, refuses, and a
    correct rewrite is reported as an unresolvable artifact. Refusing was right:
    the two files differ, and deploying to the wrong one measures nothing at a
    full re-baseline's cost. But the tie was never real -- the trace resolved
    which of them defines the kernel, and that answer is in ``target_file``.

    The anchor concludes nothing the walk would not have to confirm anyway: the
    derived root still has to satisfy the same preimage check, so a descriptor
    set that does not belong to the traced tree falls through to the walk.
    """

    def _matches(root: Path, *, require_preimage: bool) -> bool:
        checked = 0
        for desc in descriptors:
            rel = Path(str(desc.get("path") or ""))
            if not rel.parts or rel.is_absolute() or ".." in rel.parts:
                return False
            if desc.get("is_new") and desc.get("op") == "write":
                continue
            checked += 1
            if not (root / rel).exists():
                return False
        return checked > 0 or not require_preimage

    if explicit_root:
        root = Path(explicit_root).resolve()
        if root.is_dir() and _matches(root, require_preimage=False):
            return str(root)

    if not target_file:
        return ""
    try:
        target = Path(target_file).resolve()
    except OSError:
        return ""

    # Anchor on the traced source: the descriptor that names it is a tail of its
    # absolute path, so removing that tail leaves the root the producer was
    # rooted at. Longest tail first, so a descriptor that is merely a basename
    # collision cannot claim the anchor ahead of the full relative path.
    anchors = sorted(
        (Path(str(desc.get("path") or "")) for desc in descriptors if str(desc.get("path") or "")),
        key=lambda rel: len(rel.parts),
        reverse=True,
    )
    for rel in anchors:
        if rel.is_absolute() or ".." in rel.parts:
            continue
        depth = len(rel.parts)
        if depth >= len(target.parts) or target.parts[-depth:] != rel.parts:
            continue
        anchored = Path(*target.parts[:-depth])
        if anchored.is_dir() and _matches(anchored, require_preimage=True):
            return str(anchored)

    matches: list[Path] = []
    for root in target.parents:
        if _matches(root, require_preimage=True):
            matches.append(root)
    if len(matches) != 1:
        return ""
    return str(matches[0])


def _select_source_artifact(
    attempt: dict[str, Any],
    *,
    target_file: str,
    run_dir: Path | None = None,
    kernel_repo: str = "",
) -> tuple[str, str, str]:
    """Return (artifact_path, source, error) for a complete source artifact.

    Tries suffix-matching complete-source candidates first, then falls back
    to extracting a fenced code block from text/log/patch candidates.

    Args:
        attempt (dict[str, Any]): Attempt record to source artifacts from.
        target_file (str): Original kernel source path; its suffix selects
            the expected artifact type.
        run_dir (Path | None): Directory used to write extracted blocks;
            defaults to the optimized-path parent when None.
        kernel_repo (str): Kernel repo root, for worktree-relative resolution.

    Returns:
        tuple[str, str, str]: ``(artifact_path, source, error)`` where
            ``source`` is one of ``source_file`` / ``extracted_code_block`` /
            ``missing`` / ``unsupported`` and ``error`` describes failures.
    """
    target_suffix = Path(target_file).suffix.lower()
    if target_suffix not in _SOURCE_SUFFIXES:
        return "", "unsupported", f"unsupported target suffix: {target_suffix or '<none>'}"

    candidates = _candidate_artifact_paths(
        attempt,
        target_suffix,
        source_file=target_file,
        kernel_repo=kernel_repo,
    )
    for path in candidates:
        suffix = path.suffix.lower()
        if suffix == target_suffix and _source_text_looks_complete(
            path.read_text(encoding="utf-8", errors="replace"),
            target_suffix,
        ):
            return str(path), "source_file", ""

    extraction_root = run_dir or Path(attempt.get("optimized_path") or tempfile.gettempdir()).parent
    for path in candidates:
        if path.suffix.lower() not in {".txt", ".md", ".markdown", ".log", ".patch", ".diff"}:
            continue
        extracted = _extract_source_block(
            path,
            target_suffix,
            extraction_root / f"{attempt.get('attempt_id', 'attempt')}_extracted{target_suffix}",
        )
        if extracted:
            return extracted, "extracted_code_block", ""

    # Final fallback: a backend's best artifact is often a unified diff. The
    # original kernel + the patch reconstruct the optimized source
    # deterministically.
    for path in candidates:
        if path.suffix.lower() not in {".patch", ".diff"}:
            continue
        reconstructed = _reconstruct_source_from_patch(
            path,
            target_file,
            extraction_root / f"{attempt.get('attempt_id', 'attempt')}_patched{target_suffix}",
        )
        if reconstructed:
            return reconstructed, "reconstructed_from_patch", ""

    tried = ", ".join(str(p) for p in candidates[:6])
    return "", "missing", f"no complete {target_suffix} source artifact found; tried: {tried}"


def _framework_applyback_evidence(applyback: dict[str, Any]) -> dict[str, Any]:
    """Summarize a validated apply-back for the attempt ledger and integrate.

    Args:
        applyback (dict[str, Any]): The backend's validated apply-back record.

    Returns:
        dict[str, Any]: The provenance downstream needs, or ``{}`` when this
            attempt produced no framework apply-back.
    """
    if not applyback.get("artifact_kind"):
        return {}
    return {
        "artifact_kind": str(applyback.get("artifact_kind") or ""),
        "artifact_schema_version": applyback.get("artifact_schema_version"),
        "validation_scope": str(applyback.get("validation_scope") or ""),
        "reference_correctness_passed": bool(applyback.get("reference_correctness_passed")),
        "reference_snr_db": applyback.get("reference_snr_db"),
        "integration_validation_required": bool(applyback.get("integration_validation_required")),
        "integration_validation_status": str(applyback.get("integration_validation_status") or ""),
        "commit": str(applyback.get("best_commit") or ""),
        "commit_ref": str(applyback.get("commit_ref") or ""),
        "builder_symbol": str(applyback.get("builder_symbol") or ""),
        "changed_files": [str(item) for item in (applyback.get("changed_files") or [])],
    }


def build_verification(
    args: argparse.Namespace, attempts: list[dict[str, Any]], benchmark_available: bool
) -> dict[str, Any]:
    """Summarize attempt results into a verification record.

    Selects the best usable attempt (by extracted speedup, falling back to the
    first usable one) and reports whether a real speedup was measured.

    Args:
        args: Parsed CLI arguments for the run.
        attempts: Per-attempt result records.
        benchmark_available: Whether a benchmark/harness was available to
            measure speedups.

    Returns:
        A verification dict describing the best attempt and measured speedup.
    """
    # Usable = completed cleanly OR killed-but-left-artifacts (status=partial).
    usable = [a for a in attempts if a.get("status") in {"completed", "partial"}]
    best = None
    best_speedup = 0.0
    best_speedup_source = "report_scan"
    measured = False
    # Prefer the highest extracted speedup; else fall back to the first usable attempt.
    for a in usable:
        bp = a.get("backend_paths") or {}
        report = bp.get("partial_report") or bp.get("report") or ""
        sp = None
        is_forge = a.get("backend") == "forge"
        speedup_source = "report_scan"
        if is_forge:
            try:
                candidate_speedup = float(a.get("mean_case_speedup"))
                if a.get("total_improved") is True and math.isfinite(candidate_speedup) and candidate_speedup > 1.0:
                    sp = candidate_speedup
            except (TypeError, ValueError):
                sp = None
            if sp is not None:
                speedup_source = "forge_mean_case_result"
        else:
            try:
                pristine_ms = float(a.get("pristine_baseline_ms"))
                result_best_ms = float(a.get("best_ms"))
                if (
                    a.get("improved") is True
                    and math.isfinite(pristine_ms)
                    and math.isfinite(result_best_ms)
                    and pristine_ms > 0
                    and 0 < result_best_ms < pristine_ms
                ):
                    sp = pristine_ms / result_best_ms
                    speedup_source = "structured_timing_result"
            except (TypeError, ValueError):
                sp = None
        if sp is None and not is_forge:
            sp = _extract_speedup_from_report(report)
            speedup_source = "report_scan"
        if sp is not None:
            measured = True
            if sp > best_speedup:
                best_speedup = sp
                best_speedup_source = speedup_source
                best = a
    if best is None and usable:
        best = usable[0]
    compile_passed = bool(best)
    best_artifact_path = ""
    artifact_source = "missing"
    artifact_error = "no usable backend attempt"
    if best is not None:
        target_file = str(getattr(args, "source_file", "") or "")
        # kernel_repo lets worktree recovery map absolute source to GEAK's relative path.
        kernel_repo = str(getattr(args, "kernel_repo", "") or getattr(args, "repo", "") or "")
        run_dir = None
        optimized_path = best.get("optimized_path")
        if optimized_path:
            run_dir = Path(optimized_path).parent
        best_artifact_path, artifact_source, artifact_error = _select_source_artifact(
            best,
            target_file=target_file,
            run_dir=run_dir,
            kernel_repo=kernel_repo,
        )
    artifact_valid = bool(best_artifact_path)

    # When the winning attempt produced a unified diff, stage byte-exact final
    # contents for the whole patch now so deploy lands all files atomically.
    deploy_snapshot_dir = ""
    deploy_patch_path = ""
    deploy_repo_root = ""
    best_artifact_bundle: dict[str, Any] = {}
    if best is not None and artifact_valid:
        bp = best.get("backend_paths") or {}
        winning_patch = ""
        for key in ("forge_patch", "patch", "partial_report", "report"):
            cand = str(bp.get(key) or "")
            if cand.endswith((".patch", ".diff")):
                winning_patch = cand
                break
        if winning_patch and Path(winning_patch).is_file():
            snap_out = (run_dir or Path(winning_patch).parent) / f"{best.get('attempt_id', 'attempt')}_deploy_snapshot"
            deploy_patch = prepare_deploy_patch(
                winning_patch,
                output_path=snap_out.parent / f"{best.get('attempt_id', 'attempt')}_deploy.patch",
            )
            # Prefer the retained forge workspace: it is the live tree the patch
            # was produced against, so it carries every file the diff touches.
            # Fall back to the exported artifact copies, which outlive a workspace
            # that has already been reaped. Both are probed with is_dir() so a
            # stale path falls through instead of yielding an empty snapshot.
            snapshot_worktree = None
            canonical_files_root = str(bp.get("forge_canonical_files_root") or "")
            forge_workspace = str(bp.get("forge_workspace") or "")
            if canonical_files_root and Path(canonical_files_root).is_dir():
                snapshot_worktree = Path(canonical_files_root)
            elif forge_workspace and Path(forge_workspace).is_dir():
                snapshot_worktree = Path(forge_workspace)
            else:
                for root_key in ("output_dir", "cli_workspace"):
                    output_root = str(bp.get(root_key) or "")
                    files_root = Path(output_root) / "optimized_versions" / "files" if output_root else None
                    if files_root is not None and files_root.is_dir():
                        snapshot_worktree = files_root
                        break
            snap = (
                build_patch_snapshot(
                    deploy_patch,
                    worktree=snapshot_worktree,
                    kernel_repo=kernel_repo,
                    clean_base=kernel_repo,
                    out_dir=snap_out,
                )
                if deploy_patch
                else None
            )
            if snap is not None:
                descriptors = list(snap.get("descriptors") or [])
                deploy_repo_root = resolve_deploy_repo_root(
                    target_file,
                    descriptors,
                    explicit_root=str(snap.get("repo_root") or ""),
                )
                if deploy_repo_root:
                    deploy_snapshot_dir = snap["snapshot_dir"]
                    deploy_patch_path = snap["patch_path"]
                    best_artifact_bundle = {
                        "type": "patch_snapshot",
                        "producer": "kernelforge",
                        "producer_manifest": str(bp.get("forge_best_manifest") or ""),
                        "snapshot_dir": deploy_snapshot_dir,
                        "patch_path": deploy_patch_path,
                        "repo_root": deploy_repo_root,
                        "descriptors": descriptors,
                        "write_paths": [d["path"] for d in descriptors if d.get("op") == "write" and d.get("path")],
                        "delete_paths": [d["path"] for d in descriptors if d.get("op") == "delete" and d.get("path")],
                    }
            if not best_artifact_bundle:
                artifact_valid = False
                artifact_error = (
                    "Forge produced a patch, but its complete snapshot or consumer repo root could not be resolved"
                )
    applyback = dict((best or {}).get("flydsl_applyback") or {})
    correctness_signal = getattr(args, "correctness_passed", None)
    correctness_source = "cli_override" if correctness_signal is not None else "missing"
    # Read ahead of the report scan: the validated manifest is stronger evidence
    # than prose, and would otherwise be overwritten by it.
    if correctness_signal is None and applyback.get("reference_correctness_passed") is True:
        correctness_signal = True
        correctness_source = "forge_rewrite_reference"
    if correctness_signal is None and best is not None:
        bp = best.get("backend_paths") or {}
        correctness_signal = _extract_correctness_from_report(bp.get("partial_report") or bp.get("report") or "")
        if correctness_signal is not None:
            correctness_source = "report_scan"
    if correctness_signal is None and getattr(args, "accuracy_passed", None) is True:
        correctness_signal = True
        correctness_source = "accuracy_override"
    correctness_passed = bool(best and correctness_signal is True)
    if args.micro_speedup is not None:
        micro_speedup = float(args.micro_speedup)
        speedup_source = "cli_override"
    elif measured:
        micro_speedup = best_speedup
        speedup_source = best_speedup_source
    elif getattr(args, "dry_run", False):
        # Dry-run placeholder so CI can exercise KEEP/REVIEW paths.
        micro_speedup = 1.05 if best else 0.0
        speedup_source = "dry_run_placeholder"
    else:
        # No parseable speedup: leave it at 1.0 so PolicyGate routes to PARTIAL.
        micro_speedup = 1.0 if best else 0.0
        speedup_source = "default_unmeasured"
    e2e_gain_pct = args.e2e_gain_pct
    accuracy_passed = args.accuracy_passed
    return {
        "compile_passed": compile_passed,
        "correctness_passed": correctness_passed,
        "correctness_source": correctness_source,
        "benchmark_available": benchmark_available,
        "micro_speedup": micro_speedup,
        "micro_speedup_source": speedup_source,
        "e2e_gain_pct": e2e_gain_pct,
        "accuracy_passed": accuracy_passed,
        "verification_status": "complete" if correctness_passed and e2e_gain_pct is not None else "deferred",
        "integration_validation_status": str(applyback.get("integration_validation_status") or ""),
        "framework_applyback": _framework_applyback_evidence(applyback),
        "best_attempt_id": best["attempt_id"] if best else "",
        "best_backend": best["backend"] if best else "",
        "best_artifact_path": best_artifact_path,
        "artifact_valid": artifact_valid,
        "artifact_source": artifact_source,
        "artifact_error": "" if artifact_valid else artifact_error,
        "best_artifact_bundle": best_artifact_bundle,
        "deploy_snapshot_dir": deploy_snapshot_dir,
        "deploy_patch_path": deploy_patch_path,
        "deploy_repo_root": deploy_repo_root,
    }


def make_proposal(verification: dict[str, Any]) -> dict[str, Any]:
    """Turn a verification result into a KEEP/REVERT/PARTIAL/REVIEW decision.

    Applies the policy gates (compile, correctness, artifact validity,
    measured speedup vs the KEEP threshold, E2E/accuracy signals) to choose
    a disposition and the reasons behind it.

    Args:
        verification (dict[str, Any]): The dict returned by
            :func:`build_verification`.

    Returns:
        dict[str, Any]: A proposal dict with a ``decision`` (one of
            ``KEEP`` / ``REVERT`` / ``PARTIAL`` / ``NEEDS_REVIEW``) and a
            ``reasons`` list.
    """
    reasons: list[str] = []
    if not verification["compile_passed"]:
        # artifact_error distinguishes a real compile failure from a dispatch failure.
        err = (verification.get("artifact_error") or "").strip()
        if err and verification.get("best_attempt_id", "") == "":
            return {"decision": "REVERT", "reasons": [f"backend dispatch failed: {err}"]}
        return {"decision": "REVERT", "reasons": ["compile failed"]}
    if not verification["correctness_passed"]:
        reasons.append("correctness evidence missing or failed")
    if not verification.get("artifact_valid"):
        reasons.append("optimized source artifact missing or invalid")
    # default_unmeasured (no speedup found) => PARTIAL, not REVERT.
    src = verification.get("micro_speedup_source", "default_unmeasured")
    if src == "default_unmeasured":
        reasons.append("no measurable speedup found in any backend report")
        return {"decision": "PARTIAL", "reasons": reasons}
    if verification["micro_speedup"] <= 1.0:
        return {"decision": "REVERT", "reasons": ["microbench did not improve"]}
    KEEP_THRESHOLD = 1.10
    if verification["micro_speedup"] < KEEP_THRESHOLD:
        reasons.append(f"speedup {verification['micro_speedup']:.3f}x below KEEP threshold {KEEP_THRESHOLD:.2f}x")
    if verification["e2e_gain_pct"] is not None and verification["e2e_gain_pct"] < 0:
        return {"decision": "REVERT", "reasons": ["E2E regressed"]}
    if verification["accuracy_passed"] is False:
        return {"decision": "REVERT", "reasons": ["accuracy gate failed"]}
    if reasons and verification["e2e_gain_pct"] is None:
        reasons.append("E2E evidence missing")
    if reasons and verification["accuracy_passed"] is None:
        reasons.append("accuracy evidence missing")

    if reasons:
        return {"decision": "NEEDS_REVIEW", "reasons": reasons}
    if verification["e2e_gain_pct"] is None or verification["accuracy_passed"] is None:
        # A reference-verified apply-back is a micro KEEP by design: only the
        # serving run can prove the framework integration, and refusing to keep
        # it here would stop the patch ever reaching that run.
        if verification.get("integration_validation_status") == "pending":
            return {
                "decision": "KEEP",
                "reasons": ["framework apply-back reference-verified; framework E2E/accuracy deferred to integrate"],
            }
        return {
            "decision": "KEEP",
            "reasons": ["kernel artifact ready; E2E/accuracy deferred to integrate"],
        }
    return {"decision": "KEEP", "reasons": ["all required evidence passed"]}


def main() -> int:
    """CLI entry point for the kernel-optimization tool.

    Parses command-line arguments, runs the configured backend ladder for
    the requested kernel, builds the verification result and proposal, and
    persists status/artifacts.

    Returns:
        int: Process exit code (0 on success, non-zero on failure).
    """
    parser = argparse.ArgumentParser(description="Kernel-agent optimization tool")
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--session-id", default="")
    parser.add_argument(
        "--workspace-path",
        default=workspace_root(),
        help=(
            "Root the tool writes under (output lands at "
            "<workspace_path>/kernel-agent/runs/<session_id>/...). "
            "Defaults to $USER_DATA_PATH."
        ),
    )
    parser.add_argument("--candidates-path", default="")
    parser.add_argument(
        "--candidate-json",
        default="",
        help="Serialized dispatch candidate; preserves task-group context.",
    )
    parser.add_argument("--backends", default="")
    parser.add_argument("--benchmark-file", default="")
    parser.add_argument("--source-file", default="")
    parser.add_argument("--target-platform", default=_env_target_platform())
    parser.add_argument("--extra-sglang-args", default="")
    parser.add_argument(
        "--budget-minutes",
        type=float,
        default=60.0,
        help="Per-attempt wall-clock budget for forge.",
    )
    parser.add_argument("--micro-speedup", type=float, default=None)
    parser.add_argument("--e2e-gain-pct", type=float, default=None)
    parser.add_argument("--correctness-passed", choices=["true", "false", "unknown"], default="unknown")
    parser.add_argument("--accuracy-passed", choices=["true", "false", "unknown"], default="unknown")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=int(os.environ.get("KERNEL_AGENT_NUM_GPUS", "0")),
        help="Per-task GPU reservation; 0 means follow the "
        "candidate's num_gpus_recommended (1 for compute "
        "kernels, 2 for communication kernels).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    session_id = args.session_id or uuid.uuid4().hex[:12]
    run_id = f"ko-{uuid.uuid4().hex[:8]}"
    started_at = utc_now()
    root = Path(args.workspace_path) / "kernel-agent"
    run_dir = root / "runs" / session_id
    log_path = run_dir / "logs" / "kernel_optimization" / f"{run_id}.log"
    status_path = run_dir / "status" / "kernel_optimization" / f"{run_id}.json"
    artifacts: dict[str, str] = {}

    try:
        update_status(
            status_path,
            state="running",
            current_step="load_candidate",
            log_path=log_path,
            artifact_paths=artifacts,
            run_id=run_id,
            started_at=started_at,
        )
        candidates_path = Path(args.candidates_path) if args.candidates_path else resolve_candidates_path(run_dir)
        all_candidates = load_candidates(candidates_path)
        candidate = load_candidate_input(args.candidate_json, args.kernel_id)
        if candidate is None:
            candidate = find_candidate(all_candidates, args.kernel_id)
        if candidate is None:
            # kernel_id matches no candidate; skip cleanly instead of crashing.
            known = [str(c.get("kernel_id") or "") for c in all_candidates]
            msg = (
                f"kernel_id {args.kernel_id!r} not found among TraceLens "
                f"candidates {known}; skipping (no fabricated target)"
            )
            append_log(log_path, f"[skip] {msg}")
            update_status(
                status_path,
                state="skipped",
                current_step="skipped",
                log_path=log_path,
                artifact_paths=artifacts,
                run_id=run_id,
                started_at=started_at,
                error=msg,
            )
            print(
                json.dumps(
                    {
                        "tool": "kernel_optimization",
                        "session_id": session_id,
                        "run_id": run_id,
                        "kernel_id": args.kernel_id,
                        "status": "skipped",
                        "reason": "kernel_id_not_in_candidates",
                        "error_class": "invalid_kernel_id",
                        "known_kernel_ids": known,
                        "cli_log_path": str(log_path),
                        "status_path": str(status_path),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if candidate.get("reusable_native_kernel") is False or not candidate.get("source_file"):
            reason = (
                candidate.get("skip_reason")
                or candidate.get("optimization_notes")
                or "candidate is not a reusable native kernel"
            )
            msg = (
                f"kernel_id {args.kernel_id!r} resolved to non-routable "
                f"TraceLens candidate {candidate.get('kernel_id')!r}: {reason}"
            )
            append_log(log_path, f"[skip] {msg}")
            update_status(
                status_path,
                state="skipped",
                current_step="skipped",
                log_path=log_path,
                artifact_paths=artifacts,
                run_id=run_id,
                started_at=started_at,
                error=msg,
            )
            print(
                json.dumps(
                    {
                        "tool": "kernel_optimization",
                        "session_id": session_id,
                        "run_id": run_id,
                        "kernel_id": candidate.get("kernel_id") or args.kernel_id,
                        "requested_kernel_id": args.kernel_id,
                        "resolved_kernel_id": candidate.get("kernel_id"),
                        "kernel_name": candidate.get("name"),
                        "status": "skipped",
                        "decision": "REVERT",
                        "error_class": (
                            "missing_native_source" if not candidate.get("source_file") else "non_reusable_kernel"
                        ),
                        "reason": "non_routable_candidate",
                        "skip_reason": reason,
                        "verification": {
                            "micro_speedup": 0.0,
                            "best_artifact_path": "",
                        },
                        "proposal": {
                            "decision": "REVERT",
                            "reasons": [reason],
                        },
                        "cli_log_path": str(log_path),
                        "status_path": str(status_path),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        # TraceLens is source of truth; overrides a disagreeing LLM path.
        resolved_source = _resolve_source_file(args.source_file, candidate, args.kernel_id, log_path)
        args.source_file = resolved_source
        # Forward the candidate's repo root for worktree artifact recovery.
        if not getattr(args, "kernel_repo", None):
            args.kernel_repo = str(candidate.get("kernel_repo") or "")
        if not args.dry_run and not resolved_source:
            raise RuntimeError(
                f"source file not resolved for kernel {args.kernel_id}; "
                "skipping backend dispatch (no fabricated source allowed)"
            )
        selected_backends, backend_notes = choose_backends(args, candidate)
        benchmark_available = bool(backend_notes["benchmark_available"])
        append_log(log_path, f"kernel_id={args.kernel_id}")
        append_log(log_path, f"resolved_source={resolved_source or 'NONE'}")
        append_log(log_path, f"selected_backends={','.join(selected_backends) or 'none'}")

        update_status(
            status_path,
            state="running",
            current_step="run_backends",
            log_path=log_path,
            artifact_paths=artifacts,
            run_id=run_id,
            started_at=started_at,
        )
        attempts: list[dict[str, Any]] = []
        for backend in selected_backends:
            attempt = run_attempt(backend, args=args, candidate=candidate, run_dir=run_dir, log_path=log_path)
            attempt.update(backend_notes)
            attempts.append(attempt)
            append_jsonl(run_dir / "optimization_attempts.jsonl", attempt)

        update_status(
            status_path,
            state="running",
            current_step="verify_and_propose",
            log_path=log_path,
            artifact_paths=artifacts,
            run_id=run_id,
            started_at=started_at,
        )
        accuracy = None if args.accuracy_passed == "unknown" else args.accuracy_passed == "true"
        args.accuracy_passed = accuracy
        correctness = None if args.correctness_passed == "unknown" else args.correctness_passed == "true"
        args.correctness_passed = correctness
        verification = build_verification(args, attempts, benchmark_available)
        proposal = make_proposal(verification)

        verification_path = run_dir / "verification" / f"{args.kernel_id}.json"
        atomic_write_json(verification_path, verification)
        result_path = run_dir / "results" / f"{args.kernel_id}.json"
        result = {
            "tool": "kernel_optimization",
            "session_id": session_id,
            "run_id": run_id,
            "kernel_id": args.kernel_id,
            "source_file": resolved_source,
            "best_artifact_path": verification.get("best_artifact_path", ""),
            "selected_backends": selected_backends,
            "backend_selection": backend_notes,
            "attempts": attempts,
            "rag_hits": [],
            "xs_memory_hits": [],
            "verification": verification,
            "proposal": proposal,
            "cli_log_path": str(log_path),
            "status_path": str(status_path),
            "artifact_paths": {
                "verification": str(verification_path),
                "result": str(result_path),
                "cli_log_path": str(log_path),
                "status_path": str(status_path),
            },
        }
        result.update(task_group_result_metadata(candidate))
        atomic_write_json(result_path, result)
        artifacts.update(result["artifact_paths"])
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
        print(
            json.dumps(
                {
                    "tool": "kernel_optimization",
                    "session_id": session_id,
                    "run_id": run_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "cli_log_path": str(log_path),
                    "status_path": str(status_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
