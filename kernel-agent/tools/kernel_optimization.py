#!/usr/bin/env python3
"""Kernel optimization tool for the resident Kernel Agent skill."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Sibling import: kernel name → multi-GPU collective detection. Used by
# `invoke_backend` to decide between `torchrun --nproc=N` and plain
# `python` for the GEAK test-command, and to keep the
# `parallel_e2e_runner` decision consistent here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _collective_names import kernel_name_implies_multigpu  # noqa: E402
sys.path.pop(0)


def utc_now() -> str:
    """Return the current UTC time as an ISO8601 string."""
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON to ``path`` using a temp file then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as tmp:
        json.dump(data, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    """Append one JSON object as a line to the given JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(data, sort_keys=True) + "\n")


def append_log(log_path: Path, message: str) -> None:
    """Append a log line to ``log_path`` (ensuring parent dirs exist)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def read_last_lines(log_path: Path, limit: int = 20) -> list[str]:
    """Return the last ``limit`` lines of a log file, empty if missing."""
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]


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
    """Persist a status snapshot for the current run."""
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


def load_candidates(path: Path) -> list[dict[str, Any]]:
    """Load kernel candidates from JSON, normalizing legacy shapes.

    Per AMD-AGI/Hyperloom#314 the canonical ``hot_kernels`` field now only
    carries kernels that ``classify_patchability`` marked routable, with the
    rejected ones moved to ``skipped_kernels`` (full candidate dicts, not a
    compact audit projection). Batch dispatchers (parallel_e2e_runner,
    ``_batch_kernel_candidates``) read ``hot_kernels`` directly and benefit
    from the filter. The kernel-opt CLI's direct lookup path
    (``find_candidate(load_candidates(...), kid)``) still needs to be able
    to resolve a non-routable kernel by id — both for operator debugging
    ("what does the backend selector say about k001?") and so the
    dispatcher's ``_validate_reusable_native_kernel`` guard fires with the
    real ``reusable_native_kernel=False`` candidate instead of an empty
    "missing native source" stub. Return the union so both call sites
    work; the older flat-list / ``kernel_candidates`` legacy shapes are
    still respected.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    candidates = list(payload.get("hot_kernels") or payload.get("kernel_candidates") or [])
    skipped = payload.get("skipped_kernels") or []
    if isinstance(skipped, list):
        seen_ids = {
            c.get("kernel_id") for c in candidates
            if isinstance(c, dict) and c.get("kernel_id")
        }
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
    report_path = (
        payload.get("trace_report_path")
        or artifact_paths.get("trace_report_path")
    )
    if report_path:
        for candidate in candidates:
            if isinstance(candidate, dict):
                candidate.setdefault("trace_report_path", str(report_path))
    return candidates


def _normalize_kernel_id(value: str) -> str:
    """Fold hallucinated synthetic prefixes (``kn``/``rn``) onto the real
    ``k`` numbering and lower-case for a tolerant comparison.

    The Orchestration LLM sometimes returns ``kn001`` / ``rn010`` instead of
    the TraceLens ``k001`` / ``k010`` it was offered; collapsing the leading
    letter run to a single ``k`` recovers the intended candidate without
    guessing across unrelated kernels.
    """
    s = value.strip().lower()
    for prefix in ("kn", "rn"):
        if s.startswith(prefix) and s[len(prefix):].isdigit():
            return "k" + s[len(prefix):]
    return s


def find_candidate(
    candidates: list[dict[str, Any]], kernel_id: str
) -> dict[str, Any] | None:
    """Resolve a candidate by ``kernel_id`` / ``name``.

    Resolution order: exact ``kernel_id`` match, then a unique routable
    ``name`` match, then a
    normalized ``kernel_id`` match (case-insensitive, ``kn``/``rn`` prefix
    folded to ``k``). Returns ``None`` when nothing matches so the caller can
    skip the kernel gracefully instead of crashing the whole run with a
    ``KeyError`` on an LLM-hallucinated id.
    """
    for candidate in candidates:
        if candidate.get("kernel_id") == kernel_id:
            return candidate
    # Operator names are not stable identifiers: several TraceLens candidates
    # can share names like ``aten::mm``. Accept a name only when it uniquely
    # identifies a routable candidate; otherwise treat it as an invalid
    # kernel_id so the caller can skip instead of optimizing the wrong target.
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
    if not value:
        return ""
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else ""


def has_benchmark(args: argparse.Namespace, candidate: dict[str, Any]) -> bool:
    bench_files = candidate.get("benchmark_files") or []
    return bool(
        existing_path(args.benchmark_file)
        or existing_path(args.test_harness_path)
        or existing_path(str(candidate.get("benchmark_file") or ""))
        or existing_path(str(candidate.get("test_harness_path") or ""))
        or any(existing_path(str(p)) for p in bench_files)
    )


def _resolve_source_file(
    llm_source: str,
    candidate: dict[str, Any],
    kernel_id: str,
    log_path: Path | None = None,
) -> str:
    """Resolve the effective source file, preferring TraceLens (candidate).

    TraceLens analyzes the trace and produces the authoritative
    ``kernel_id → source_file`` mapping in ``kernel_candidates.json``.
    The Orchestration LLM can also pass ``--source-file`` via payload,
    but it occasionally confuses kernel IDs (e.g. picks fmoe ``k001``'s
    source for fmha ``k003``) and supplies a path that no longer matches
    the kernel being optimized. The legacy ``args.source_file or
    candidate.source_file`` order let any LLM-supplied string silently
    override TraceLens, which on DeepSeek-R1 routed an MHA kernel's
    rewrite at ``fused_moe.py``.

    Policy: candidate wins. If the LLM's path resolves to a different
    absolute location than candidate's, emit a ``[source-override]``
    warning to the run log so the discrepancy is visible in postmortem,
    then return the candidate path. When candidate has no source_file
    (legacy / synthetic fixtures), fall back to the LLM-supplied path.
    """
    cand_source = str((candidate or {}).get("source_file") or "").strip()
    llm = str(llm_source or "").strip()
    if not cand_source:
        return llm

    # A TraceLens "launcher path" can be a profiler *frame label* such as
    # ``aiter/fused_moe.py(986): fused_moe_2stages`` rather than a real file.
    # This is the norm for synthetic pseudo-ops (e.g. TraceLens PR #668's
    # ``pseudo_op::moe_flydsl_stage1/2``, which inherit a frame-label launcher
    # path from the ``aiter::fused_moe_`` donor). GEAK cannot open a frame
    # label, so when the candidate source is not a readable file but the
    # caller supplied a real one, prefer the explicit source_file. This is the
    # source-resolution analogue of the pseudo-op fast path in
    # ``tracelens_analysis.source_type_for``.
    def _is_real_file(p: str) -> bool:
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


# Kernel-name → benchmark-name priority patterns. Listed in priority order
# (more specific kernel families first). Each entry pairs a kernel-name
# regex with a priority-ordered list of benchmark-filename regexes; when
# the kernel matches, benchmarks whose basename matches any of the
# patterns are hoisted to the front of the candidate list (preserving the
# pattern order). Patterns are intentionally conservative so a missing
# kernel family degrades to "preserve original order" rather than picking
# an off-topic benchmark.
_BENCHMARK_PATTERNS: list[tuple["re.Pattern[str]", list["re.Pattern[str]"]]] = [
    # Flash / multi-head attention (must come BEFORE paged-attn so a kernel
    # name like ``fmha_v3_varlen_fwd`` does not accidentally match a generic
    # ``attn`` rule that also hits ``test_pa.py``).
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

    TraceLens populates ``candidate.benchmark_files`` with every test/bench
    file it found under the kernel's repo, in an order driven by repo
    enumeration rather than kernel semantics. On DeepSeek-R1 this surfaced
    a real failure: the fmha kernel ``aiter::fmha_v3_varlen_fwd``'s
    benchmark list was led by ``test_pa.py`` (PagedAttention), so the
    legacy ``for bf in bench_files: ... break`` selector in
    :func:`invoke_backend` picked a benchmark that doesn't exercise the
    kernel — and whose 90-config × 3-replay default matrix stalled the
    GEAK Step-5 profiling for hours.

    Policy: scan :data:`_BENCHMARK_PATTERNS` in declared priority order;
    for the first kernel-name regex that matches, sort the bench list so
    items matching that family's bench patterns come first (within the
    matched group, earlier patterns win). When no kernel pattern matches,
    return the original order — never invent a preference.
    """
    existing = [p for p in (bench_files or []) if isinstance(p, str) and p]
    if not existing:
        return []
    name = str(kernel_name or "")
    for kernel_re, bench_res in _BENCHMARK_PATTERNS:
        if not kernel_re.search(name):
            continue

        def _priority(path: str, _bench_res=bench_res) -> int:
            base = Path(path).name
            for idx, br in enumerate(_bench_res):
                if br.search(base):
                    return idx
            return len(_bench_res)

        return sorted(existing, key=_priority)
    return existing


def _profile_timeout_sec() -> int:
    """Per-subprocess profiling timeout (seconds) for GEAK's Step 5.

    The GEAK preprocessor (vendored ``minisweagent.preprocess``) runs the
    rendered ``test_command`` under Metrix instrumentation with
    ``num_replays=3`` and then captures a second baseline pass — neither
    call carries a subprocess timeout. With a default-matrix benchmark
    such as aiter's ``test_pa.py`` (2 dtypes × 5 head configs × 9 ctx_len
    = 90 cases per replay), Step 5 can stall for hours and burn the entire
    GEAK budget before any patch is attempted.

    We bound this by injecting ``timeout <N>`` as the prefix of the
    ``test_command`` we hand to GEAK. The benchmark subprocess SIGTERMs at
    ``N`` seconds and returns exit 124, which Metrix surfaces as a normal
    profiling failure; the preprocessor's existing ``except Exception``
    handler logs a warning and continues to Step 6/7 instead of hanging.

    Default 600s; override via ``KERNEL_OPT_PROFILE_TIMEOUT_SEC``. Floors
    at 1 so a misconfigured ``0`` cannot disable the guard entirely.
    """
    try:
        value = int(os.environ.get("KERNEL_OPT_PROFILE_TIMEOUT_SEC", "600"))
    except (ValueError, TypeError):
        return 600
    return max(1, value)


def _render_geak_test_command(
    kernel_name: str,
    bench_files: list[Any],
    is_multigpu: bool,
    num_gpus: int,
    timeout_sec: int,
) -> str:
    """Render the ``--test-command`` GEAK receives, with timeout + match.

    Picks the first existing ``test_*.py`` / ``bench*.py`` from the
    semantically-ordered bench list, prefixes ``timeout <N>``, and wraps
    multi-GPU collectives in ``torchrun --nproc_per_node=<num_gpus>`` so
    GEAK's subprocess can ``init_process_group`` correctly. Returns ``""``
    when no usable benchmark exists; the caller leaves
    ``--test-command`` blank so GEAK falls back to its own discovery.
    """
    ordered = _match_benchmark_for_kernel(kernel_name, bench_files)
    for bf in ordered:
        path = Path(bf)
        if not bf.endswith(".py") or not path.exists():
            continue
        name = path.name
        if "test_" not in name and "bench" not in name:
            continue
        if is_multigpu and num_gpus >= 2:
            return f"timeout {timeout_sec} torchrun --nproc_per_node={num_gpus} {bf}"
        return f"timeout {timeout_sec} python {bf}"
    return ""


def parse_backends(backends: str) -> list[str]:
    parsed = [b.strip().lower() for b in backends.split(",") if b.strip()]
    allowed = {"geak", "claude", "codex", "cursor"}
    invalid = [b for b in parsed if b not in allowed]
    if invalid:
        raise ValueError(f"unsupported backend(s): {', '.join(invalid)} "
                         f"(allowed: {sorted(allowed)}; the 'llm' single-shot "
                         "backend was removed because max_tokens=2048 truncates "
                         "any non-trivial kernel)")
    return parsed


def choose_backends(args: argparse.Namespace, candidate: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Select the backend ladder for a kernel-opt run.

    Policy (#144 last comment Layer 1, broadened): every kernel
    Claude/Codex can rewrite, GEAK can rewrite too. Include GEAK in
    every default ladder, FIRST (high-priority handoff) — Claude/Codex
    follow as fallbacks if GEAK times out or rejects.

    When no benchmark/test harness is available, GEAK still attempts
    the rewrite but ``geak_without_benchmark=True`` is flagged so
    downstream KEEP gates / operators know verification confidence is
    reduced. Aligns the auto-pick with the SKILL.md "allow the attempt
    but mark" contract that previously only applied to user-specified
    backends.

    Only ``[]`` returns: vendor binaries (nothing rewritable upstream).
    """
    user_backends = parse_backends(args.backends)
    benchmark_available = has_benchmark(args, candidate)
    source_type = str(candidate.get("source_type") or "unknown")
    # Cursor backend requires CURSOR_API_KEY (Cursor's own gateway, not the
    # AMD LiteLLM gateway). When the operator has not provisioned a Cursor
    # key, skip cursor from auto-selected defaults to avoid wasted 401
    # attempts. User-specified `--backends cursor` still wins (respects
    # explicit intent; the missing key surfaces as a clear backend failure).
    cursor_key_present = bool(os.environ.get("CURSOR_API_KEY", "").strip())
    notes: dict[str, Any] = {
        "user_specified_backends": bool(user_backends),
        "benchmark_available": benchmark_available,
        "geak_without_benchmark": False,
        "cursor_key_present": cursor_key_present,
    }

    if user_backends:
        if "geak" in user_backends and not benchmark_available:
            notes["geak_without_benchmark"] = True
        return user_backends, notes

    if source_type == "vendor_binary":
        return [], notes

    # Unified ladder per #144 last comment Layer 1 (broadened): every
    # kernel Claude/Codex can rewrite, GEAK can rewrite too. GEAK is
    # FIRST (high-priority handoff). When no benchmark/test harness is
    # present, GEAK still attempts but ``geak_without_benchmark=True``
    # is flagged so downstream KEEP gates know verification confidence
    # is reduced — matches the SKILL.md "allow but mark" contract that
    # previously only applied to user-specified backends.
    selected = ["geak", "claude", "codex"]
    if not benchmark_available:
        notes["geak_without_benchmark"] = True
    return selected, notes


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
    return str(value or "").strip().lower()


def _hardware_prompt_blocks(target_platform: str) -> tuple[str, str]:
    platform = _normalize_target_platform(target_platform)
    hw = _GPU_HW.get(platform)
    if not hw:
        intro = (
            "Optimize this GPU kernel for the active AMD Instinct GPU "
            "inference serving. Produce an actual edited kernel file with "
            "measurable speedup; do NOT just analyze and submit unchanged."
        )
        notes = "\n".join([
            "Hardware notes (target platform unknown):",
            "- Before benchmarking, query the runtime environment for the ROCm arch ",
            "(hipDeviceGetName/rocminfo), visible GPU IDs (ROCR_VISIBLE_DEVICES), "
            "and memory size/bandwidth.",
            "- Record those values in the result and choose --offload-arch=<arch> "
            "accordingly; replace <arch> with the inspected ROCm arch before running.",
        ])
        return intro, notes

    intro = (
        f"Optimize this GPU kernel for **AMD Instinct {hw['name']} "
        f"({hw['arch']}, {hw['uarch']})** inference serving. Produce an actual "
        "edited kernel file with measurable speedup; do NOT just analyze and "
        "submit unchanged."
    )
    notes = "\n".join([
        f"Hardware notes (target platform: `{platform}`):",
        f"- {hw['cus']} CUs, {hw['uarch']}, ROCm arch `{hw['arch']}`",
        f"- {hw['mem']}",
        f"- Build flag: `{hw['build_flag']}`",
        f"- Use optimizations compatible with `{hw['arch']}` and verify runtime "
        "device properties before benchmarking.",
    ])
    return intro, notes


def _target_build_flag(target_platform: str) -> str:
    platform = _normalize_target_platform(target_platform)
    hw = _GPU_HW.get(platform)
    return str(hw["build_flag"]) if hw else "--offload-arch=<arch>"


def _env_target_platform() -> str:
    return os.environ.get("TARGET_GPU_TYPE", "") or os.environ.get("GPU_TYPE", "")


def _format_shapes_for_case(shapes: Any) -> str:
    """Render a candidate row's ``shapes`` field as one comma-joined line.

    Rows in a ``task_group`` come straight from TraceLens's 9-column
    Detailed Analysis Data table where ``Args`` is a ``<br>``-joined
    list of ``(shape) dtype`` strings. The ``_row_to_candidate`` parser
    has already split them into a Python list; here we collapse back
    to one line so the case bullet stays single-line.
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
                # Some rows carry {call_num, shape} dicts; render the
                # shape verbatim and tag the call_num if present.
                shape = entry.get("shape") or entry.get("Args") or ""
                call_num = entry.get("call_num")
                if shape:
                    parts.append(f"{shape}" + (f" (x{call_num})" if call_num else ""))
            else:
                parts.append(str(entry))
        return ", ".join(p for p in parts if p)
    return str(shapes)


def _build_benchmark_cases_block(candidate: dict[str, Any]) -> str:
    """Render the multi-row benchmark cases section for a task_group.

    Returns the empty string when ``candidate["task_group"]`` is absent
    so the prompt body stays byte-identical for legacy per-kernel
    dispatch. When present, emits one bullet per TraceLens Operation
    row sorted by aggregate time (descending). Each bullet carries
    ``operation``, ``args``, ``aggregate_time_ms``, ``percent_e2e``,
    ``count``, ``per_call_ms``, ``flops_per_byte``, ``efficiency``,
    and ``bound``.

    The two most useful fields for backend dispatch decisions are
    ``bound`` (memory vs compute drives which optimization lens to
    apply — see ``_build_priority_block``) and ``per_call_ms``
    (separates "high-count tiny-shape decode launch overhead" from
    "fat per-invocation prefill cost"). Both are surfaced verbatim
    rather than buried inside the kernel_metadata JSON.
    """
    group = candidate.get("task_group")
    if not isinstance(group, dict):
        return ""
    rows = group.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return ""
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
        lines.extend([
            f"This kernel resolves to the same source function across "
            f"{len(rows)} TraceLens rows ("
            f"{function_name or '<unknown function>'}"
            + (f" at {location}" if location else "")
            + (", AST-resolved" if ast_resolved else "")
            + "). Optimize the source function once; the patch applies "
            f"to all rows below. Use the first row as the primary",
            "benchmark case; treat the rest as supplementary shape coverage.",
            "",
        ])
    else:
        lines.extend([
            f"This kernel maps to a single TraceLens row in "
            f"{function_name or '<unknown function>'}"
            + (f" at {location}" if location else "")
            + ". The case below is the primary benchmark target.",
            "",
        ])

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
    return "\n".join(lines)


# PR-B §3: ordered optimization directions, keyed by bound type so the
# agent's first lever matches the kernel's actual bottleneck. The order
# below mirrors the feature branch's ``tracelens_geak_task_parser``
# Optimization directions section; ``compute`` flips the top two
# entries, ``unknown`` falls back to the feature-branch default order.
# Each entry is one already-formatted bullet line.
# Each bullet is a SINGLE string broken across lines for readability —
# wrap in ``( ... )`` so adjacent-literal concatenation is explicit and
# a forgotten trailing comma can't silently merge two bullets together
# (defensive against the github-code-quality "Implicit string
# concatenation in a list" lint; byte-identical to the un-wrapped form
# at parse time).
_PRIORITY_BULLETS: dict[str, list[str]] = {
    "memory": [
        (
            "1. **Memory traffic reduction** (primary lever for memory-bound rows): "
            "improve coalescing / vectorization, fuse with neighbouring ops to "
            "amortize global loads, reduce intermediate writes, and avoid extra "
            "global-memory round trips."
        ),
        (
            "2. **Shape-aware tuning**: specialize block sizes and grid indexing "
            "for the dominant TraceLens Args. Memory-bound kernels are especially "
            "sensitive to load-coalescing alignment on the dominant shape."
        ),
        (
            "3. **Launch amortization** for tiny high-count decode shapes: "
            "persistent / batched handling or wrapper-level batching when source "
            "and harness allow."
        ),
        (
            "4. **Structural simplification**: hoist loop-invariant computations, "
            "remove redundant address arithmetic, collapse dual-pass logic."
        ),
        (
            "5. **Compute utilization** (rarely the bottleneck here, but check): "
            "MFMA tile choice, occupancy, register / shared-memory balance."
        ),
    ],
    "compute": [
        (
            "1. **Compute utilization** (primary lever for compute-bound rows): "
            "improve MFMA tile choice, occupancy, and register / shared-memory "
            "balance so the same FLOPs issue under a better-utilized pipeline."
        ),
        (
            "2. **Shape-aware tuning**: specialize block sizes and grid indexing "
            "for the dominant TraceLens Args. Compute-bound kernels often hit "
            "different efficiency ceilings on K-major vs N-major shapes."
        ),
        (
            "3. **Structural simplification**: hoist loop-invariant computations, "
            "remove redundant address arithmetic, collapse dual-pass logic."
        ),
        (
            "4. **Memory traffic reduction** (secondary): coalescing / "
            "vectorization, fewer intermediate writes — rarely the bottleneck "
            "here but worth measuring after a compute-side change."
        ),
        (
            "5. **Launch amortization** for tiny high-count decode shapes: "
            "persistent / batched handling or wrapper-level batching."
        ),
    ],
    "unknown": [
        (
            "1. **Structural simplification**: hoist loop-invariant computations, "
            "remove redundant address arithmetic, collapse dual-pass logic."
        ),
        (
            "2. **Shape-aware tuning**: specialize block sizes and grid indexing "
            "for the dominant TraceLens Args."
        ),
        (
            "3. **Memory traffic reduction**: improve coalescing / vectorization, "
            "reduce intermediate writes, avoid extra global-memory round trips."
        ),
        "4. **Launch amortization** for tiny high-count decode shapes.",
        (
            "5. **Compute utilization**: improve MFMA tile choice, occupancy, "
            "register / shared-memory balance."
        ),
    ],
}


def _classify_bound(bound_type: str) -> str:
    """Map TraceLens ``bound`` strings to one of the three priority keys.

    TraceLens emits values like ``memory-bound`` / ``Memory-Bound`` /
    ``compute-bound`` / ``mixed`` / ``-`` / empty. Normalise to one of
    ``memory`` / ``compute`` / ``unknown`` so the priority block stays
    deterministic.
    """
    text = (bound_type or "").lower()
    if "memory" in text or "bandwidth" in text or "hbm" in text:
        return "memory"
    if "compute" in text or "arithmetic" in text or "flops" in text:
        return "compute"
    return "unknown"


def _build_priority_block(candidate: dict[str, Any]) -> str:
    """Render the bound-keyed optimization priority list.

    Pulls ``bound_type`` from the candidate (set by ``_row_to_candidate``
    when the TraceLens v0.3 report carries a ``Bound`` column).
    Returns the empty string when ``bound_type`` is missing AND no
    ``task_group`` is attached — the section is purely additive context;
    skipping it on legacy candidates keeps the prompt body byte-identical
    to PR-A.

    When a ``task_group`` is present, the bound classification of the
    *primary* row is used; non-primary rows are typically the same
    Operation called from different shapes and share the same bound.
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
    header_line = (
        f"## Optimization priorities (TraceLens bound: `{label}`)"
    )
    intro = (
        "The list below orders optimization levers by expected payoff for "
        "this kernel's bottleneck. Try lever 1 first; only move to lever 2 "
        "if profiling shows lever 1 is exhausted or not applicable."
    )
    lines = ["", header_line, "", intro, ""]
    lines.extend(_PRIORITY_BULLETS[bucket])
    return "\n".join(lines)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Mirror of ``tracelens_skill_runner._safe_float`` — coerce
    ``int / float / numeric str`` to ``float``; everything else (None,
    empty string, malformed) → ``default``. Used by hypothesis-block
    helpers so per-row impact numbers from ``all_pitem_prose`` survive
    JSON round-trips that may have already converted them to strings."""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_impact_range(
    low_ms: float, low_e2e: float, high_ms: float, high_e2e: float,
) -> str:
    """One-line impact range formatter; empty string when both ends zero."""
    if not (low_ms or high_ms):
        return ""
    return (
        f"**Estimated impact range:** low {low_ms:.2f} ms savings "
        f"({low_e2e:.2f}% E2E), high {high_ms:.2f} ms savings "
        f"({high_e2e:.2f}% E2E). These are TraceLens roofline estimates, "
        "not measured speedups — confirm with a real benchmark."
    )


def _build_hypothesis_block(candidate: dict[str, Any]) -> str:
    """Render a TraceLens hypothesis section for the candidate, if any.

    Returns empty string when none of the prose fields are present so the
    block is invisible for candidates whose TraceLens P-item lacked
    Reasoning / Resolution / Impact sections.

    When the candidate carries a ``task_group`` whose
    ``all_pitem_prose`` list contains MORE than one entry — i.e. the
    same source function legitimately appears across multiple
    TraceLens P-items (e.g. memory-bound at decode shapes, compute-
    bound at prefill shapes) — every P-item's prose is rendered with
    a ``### P{rank}`` header so GEAK sees all framings, not just the
    primary's. Single-P-item / no-P-item candidates fall back to the
    legacy single-block layout (reads from candidate's prose fields
    directly).

    Framing notes:

    * TraceLens's ``Reasoning for Slowdown`` / ``Resolution`` are
      themselves LLM-generated by the analysis-orchestrator skill —
      anchoring GEAK to them as ground truth would degrade optimization
      quality when the hypothesis is wrong. We label the block
      explicitly as a hypothesis to *validate*, not an imperative.
    * Numeric ``Impact estimate`` (low / high ms savings + %E2E) is
      pure roofline arithmetic and is safer to surface directly; the
      agent still needs a real measurement before declaring success.
    """
    # Multi-P-item case (Q2): the same source function spans multiple
    # TraceLens P-items. Each P-item contributes its own prose tuple;
    # render them all so GEAK sees every framing, not just the
    # primary's.
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
            "hypothesis, follow the data and document the discrepancy in",
            "`optimization_report.md`.",
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

    # Single-P-item / no-P-item path: read prose from the candidate
    # directly. Backward-compatible with raw-trace / csv-fallback
    # candidates that have no ``task_group`` attached.
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
        "document the discrepancy in `optimization_report.md`.",
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
    """Parse selected SGLang flags from an EXTRA_SGLANG_ARGS-style string."""
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
    if not isinstance(shapes, list):
        return []
    try:
        count = int(float(call_num or 1))
    except (TypeError, ValueError):
        count = 1
    entries: list[dict[str, Any]] = []
    for shape in shapes:
        if isinstance(shape, dict) and "shape" in shape:
            entries.append({
                "call_num": int(shape.get("call_num") or count),
                "shape": shape["shape"],
            })
        elif shape not in (None, "", [], ()):
            entries.append({"call_num": count, "shape": shape})
    return entries


def build_kernel_metadata(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Build structured runtime context for GEAK task prompts."""
    source_file = getattr(args, "source_file", "") or candidate.get("source_file", "")
    kernel_name = str(candidate.get("name") or getattr(args, "kernel_id", ""))
    input_shapes = candidate.get("input_shapes")
    if input_shapes is None:
        input_shapes = _shape_call_entries(candidate.get("shapes", []), candidate.get("call_count"))
    input_dtypes = candidate.get("input_dtypes")
    if input_dtypes is None:
        input_dtypes = candidate.get("dtypes", [])

    runtime_flags: dict[str, Any] = {}
    if isinstance(candidate.get("runtime_flags"), dict):
        runtime_flags.update(candidate["runtime_flags"])
    runtime_flags.setdefault("is_multigpu", bool(candidate.get("is_multigpu")))
    runtime_flags.setdefault("num_gpus_recommended", candidate.get("num_gpus_recommended"))
    # The canonical payload key is ``extra_server_args`` (renamed from
    # the legacy ``extra_sglang_args``); the local shim accepts both
    # shapes on read so envelopes still carrying the legacy key work.
    # The kernel-agent ``tools/`` directory is added to sys.path (not
    # used as a package), so import the shim by bare module name.
    from _payload_aliases import (  # type: ignore[import-not-found]
        read_extra_server_args as _read_eserver,
    )
    extra_server_args = (
        getattr(args, "extra_server_args", "")
        or _read_eserver(candidate)
        or candidate.get("candidate_extra_server_args", "")
        or candidate.get("candidate_extra_sglang_args", "")
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

    return {
        "kernel_path": str(source_file or ""),
        "kernel_name": kernel_name,
        "input_shapes": input_shapes or [],
        "output_shapes": candidate.get("output_shapes") or [],
        "input_dtypes": input_dtypes or [],
        "output_dtypes": candidate.get("output_dtypes") or [],
        "backend": candidate.get("backend") or candidate.get("framework"),
        "runtime_args": runtime_args,
        "runtime_flags": runtime_flags,
        "env_vars": candidate.get("env_vars") or {},
        "kernel_params": kernel_params,
        # PR-K: source attribution. ``launcher_source_file`` is the python
        # @compile_ops wrapper TraceLens originally attributed the kernel
        # to (e.g. ``aiter/ops/moe_op.py``); ``kernel_path`` above is the
        # device source the LLM must actually rewrite (e.g.
        # ``csrc/.../gemm_moe_ck2stages.cu``). Both fields are empty /
        # False when the candidate was NOT promoted, so this metadata
        # block is a non-event for vendor-style kernels whose trace
        # source already pointed at the device file.
        "launcher_source_file": str(candidate.get("launcher_source_file", "") or ""),
        "source_promoted_from_launcher": bool(
            candidate.get("source_promoted_from_launcher"),
        ),
    }


def build_prompt(
    candidate: dict[str, Any],
    args: argparse.Namespace,
    *,
    backend: str | None = None,
) -> str:
    source_file = args.source_file or candidate.get("source_file", "")
    source_block = ""
    if source_file and Path(str(source_file)).exists():
        content = Path(str(source_file)).read_text(encoding="utf-8", errors="replace")
        source_block = f"\nSource content:\n```\n{content[:12000]}\n```"
    kernel_repo = str(candidate.get("kernel_repo") or "")
    bench_files = candidate.get("benchmark_files") or []
    if isinstance(bench_files, str):
        bench_files = [bench_files]
    # Sort by semantic match against the kernel name so the most-relevant
    # benchmarks head the list GEAK reads (the prompt clips to ``[:8]``
    # below, so an off-topic ``test_*.py`` would otherwise crowd out the
    # right one on kernels with long bench listings).
    bench_files = _match_benchmark_for_kernel(
        str(candidate.get("name") or ""), bench_files
    )
    is_multigpu = bool(candidate.get("is_multigpu"))
    # Resolve how many GPUs the executor will give this attempt: CLI override
    # wins, then candidate hint, then 1 (single-GPU compute kernel).
    num_gpus = max(1, int(getattr(args, "num_gpus", 0) or 0)
                   or int(candidate.get("num_gpus_recommended") or 1))
    # Map our source_type to GEAK's kernel_type vocabulary so its task_parser
    # can route to the right agent (hip / triton / flydsl / other).
    geak_kernel_type = _GEAK_KERNEL_TYPE.get(str(candidate.get("source_type", "unknown")), "other")
    kernel_name = str(candidate.get("name", args.kernel_id))
    kernel_metadata = build_kernel_metadata(candidate, args)
    # Budget-protocol preamble. mini-swe-agent renders a ``step N ($X.XX)``
    # header before every tool call where ``$X.XX`` is the cumulative LLM
    # token spend. With GEAK's per-task cost-limit disabled (``--cost-limit
    # 0.0``) this number is pure telemetry — it does NOT terminate the
    # agent. In the May 2026 M2.5 N36 run multiple GEAK rounds nonetheless
    # exited at step 3 / ~$2 with a ``budget exhausted`` panic-submit and
    # zero code edits, throwing away ~90% of the 60-minute wall-clock
    # budget that actually governs the task. Re-iterate the contract in
    # the prompt so the LLM treats the header as a cost meter, not a
    # stop sign. Wall-clock budget is sourced from the runner's
    # ``geak_budget_minutes`` (default 60); the prompt repeats it verbatim
    # rather than re-deriving so it stays accurate if the operator lowers
    # the budget via ``RUN_OPT_GEAK_BUDGET_MINUTES``.
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
    # PR-K: source attribution note. When TraceLens originally attributed a
    # kernel to a python ``@compile_ops`` wrapper (e.g. ``aiter/ops/moe_op.py``
    # for ``ck_moe_stage1``) and tracelens_analysis promoted it to the device
    # source, render a hard-rule notice at the top of the prompt so the LLM
    # rewrites the device file and not the bypassed wrapper. Empty for
    # un-promoted kernels — does not bloat the legacy prompt by even a byte.
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
    # Quote the per-backend wall-clock so GEAK v3.2.0's LLM task-mode parser
    # (mini.py:435 task_extracted_mode) infers the right mode (>=120min→full,
    # else quick), instead of always seeing the OOB 60min default.
    if backend == "geak":
        budget_min = int(getattr(args, "geak_budget_min", 130) or 130)
    else:
        budget_min = int(getattr(args, "budget_minutes", 60) or 60)
    target_platform = (
        getattr(args, "target_platform", "") or _env_target_platform()
    )
    platform_intro, hardware_notes = _hardware_prompt_blocks(target_platform)
    platform_build_flag = _target_build_flag(target_platform)
    hypothesis_block = _build_hypothesis_block(candidate)
    benchmark_cases_block = _build_benchmark_cases_block(candidate)
    priority_block = _build_priority_block(candidate)
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
        # GEAK v3.2.0 LLM-parses prompt for `--mode full` / `mode=quick` etc.
        # (prompts.py:73-76 trigger list). Emit the explicit token so the
        # parser locks in the right preset (yaml run.budgets.<mode>) instead
        # of leaking off other prompt phrases like "quick micro-benchmark".
        f"- Run mode: {'full' if budget_min >= 120 else 'quick'} "
        f"(--mode {'full' if budget_min >= 120 else 'quick'}).\n"
        f"- Hard wall-clock budget: ~{budget_min} minutes. Iterate up to minute "
        f"{int(budget_min*0.85)},\n"
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
        "  kernels (claude r19 pattern).\n"
        "\n"
        "How to do A/B benchmarking WITHOUT rebuilding aiter (which is forbidden):\n"
        "(option 1) TRITON path (preferred when available). If you took priority 0,\n"
        f"  write your version as a NEW Triton .py under ./optimized_versions/, then:\n"
        "  `from aiter.ops.triton.<path> import <fn> as baseline; "
        "from your_v3 import <fn> as optimized` — Triton is JIT-compiled, NO rebuild.\n"
        "(option 2) STANDALONE HIP/CUDA program. Write a single .hip/.cu under\n"
        "  ./benchmarks/ that #include's BOTH the aiter baseline header (e.g.\n"
        f"  `#include \"{kernel_repo}/csrc/include/<the_target>.cuh\"`) AND your\n"
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
        "  rebuilding aiter (which is forbidden). Codex tends to skip this\n"
        "  and write `speedup: N/A` — DO NOT do that.\n"
        "Pick whichever option matches the kernel; do NOT just measure baseline\n"
        "and write `speedup: N/A` — that wastes the run.\n"
    )
    # Multi-node sandbox is GPU-less: any local `hipcc` / `torch.cuda.*` /
    # `torch.utils.cpp_extension.load` call WILL fail. Direct the LLM to
    # delegate compile + execution to a GPU-bearing pod via the
    # `inference_optimizer.multi_node kernel-bench` subcommand (head pod,
    # single-GPU actor); LLM still iterates locally on source, just
    # off-loads each measurement step. The CLI base64-encodes any
    # supporting files, stages them under --workspace on the pod, runs
    # the bench inside that workspace with the GPU, and returns
    # stdout/stderr + any matching result*.json artifacts.
    mn_state_file = Path("/tmp/multi_node_state.json")
    is_multinode_run = False
    try:
        if mn_state_file.is_file():
            _st = json.loads(mn_state_file.read_text(encoding="utf-8"))
            is_multinode_run = int(_st.get("nodes") or 0) >= 2
    except (OSError, ValueError):
        is_multinode_run = False
    if is_multinode_run:
        safety += (
            "\nMULTI-NODE SANDBOX (no local GPU): every compile + benchmark\n"
            "step MUST be dispatched to a GPU-bearing RayJob pod. Do NOT\n"
            "call `hipcc`, `torch.cuda.*`, or `torch.utils.cpp_extension.load`\n"
            "directly; they have no GPU here and will hang or crash.\n"
            "Instead, for each A/B benchmark iteration:\n"
            "  1. Write the bench script (and any deps) under your\n"
            "     workspace ($WORKSPACE/benchmarks/, $WORKSPACE/optimized_versions/).\n"
            "  2. Invoke:\n"
            "       python3 -m inference_optimizer.multi_node kernel-bench \\\n"
            "         --workspace /tmp/kbench_$KERNEL_ID \\\n"
            "         --bench-command 'cd /tmp/kbench_$KERNEL_ID && bash bench.sh' \\\n"
            "         --files-b64-json '<{\"bench.sh\":\"<b64>\",\"v1.cu\":\"<b64>\",...}>' \\\n"
            "         --result-glob 'result*.json'\n"
            "  3. Parse the printed JSON document's `result.stdout_tail`\n"
            "     and `result.artifacts[].content` for the speedup number;\n"
            "     write it into optimization_report.md as `[MICRO_SPEEDUP]`.\n"
            "Helper script to construct the b64 map cleanly:\n"
            "    python3 -c 'import base64,json,glob;print(json.dumps({p:base64.b64encode(open(p,\"rb\").read()).decode() for p in glob.glob(\"**/*\",recursive=True) if __import__(\"os\").path.isfile(p)}))'\n"
            "Treat `kernel-bench` as your only measurement gate; everything\n"
            "else (code edits, correctness reasoning) still happens locally.\n"
        )
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
    # Per AMD-AGI/Hyperloom#307: the per-kernel TraceLens prose
    # (Identification / Reasoning / Recommended direction / Impact
    # estimate) is already extracted into ``hypothesis_block`` above
    # via ``_build_hypothesis_block``, and the bound-specific lever
    # list is in ``priority_block``. Dumping the full analysis.md on
    # top of that bloats the prompt by ~300 lines (~40% of the body
    # in a real Qwen3-32B run) and surfaces other P-items that this
    # very prompt tells the agent NOT to optimize. Only fall back to
    # the full-report dump when no per-kernel hypothesis could be
    # rendered (raw-trace / csv-fallback path with empty prose), so
    # the agent still has *some* TraceLens grounding in that edge
    # case.
    if not hypothesis_block.strip():
        report_path_str = str(candidate.get("trace_report_path") or "")
        report_path = Path(report_path_str) if report_path_str else None
        if report_path and report_path.exists():
            try:
                full_report = report_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                full_report = ""
            if full_report:
                from inference_optimizer.tracelens_md import strip_base64_data_urls

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
                tracelens_context_block = (
                    "\n## TraceLens Context\n\n"
                    + focus_line
                    + "\n"
                    + full_report
                )
    # Use GEAK task_parser field names (kernel_name/kernel_url/kernel_type/repo)
    # so its LLM-based parser can extract them; OOB agents read the same body
    # as a normal natural-language prompt.
    return "\n".join([
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
        promotion_block,
        "",
        "Kernel runtime metadata (structured context for GEAK; unknown fields are null, empty arrays, or empty objects):",
        "```json",
        json.dumps(kernel_metadata, indent=2, sort_keys=True),
        "```",
        "",
        hardware_notes,
        hypothesis_block,
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
    ])


def ray_available() -> bool:
    try:
        import ray  # noqa: F401
        return True
    except Exception:
        return False


def _backends_module_dir() -> Path:
    return Path(__file__).resolve().parent / "backends"


def _import_backend(name: str):
    """Dynamically load kernel-agent/tools/backends/<name>.py.

    The submodules are not part of a Python package on disk; we add their
    directory to sys.path before import so they can also import each other.
    """
    backends_dir = _backends_module_dir()
    if str(backends_dir) not in sys.path:
        sys.path.insert(0, str(backends_dir))
    import importlib
    return importlib.import_module(name)


def _kernel_agent_root() -> Path:
    """Output root for kernel-agent tools.

    Lands at ``$USER_DATA_PATH/kernel-agent`` (the per-session tool-output
    namespace; sibling of ``$USER_DATA_PATH/kernel-agent-workspace``
    which keeps cross-task GEAK/OOB artefacts keyed by kernel_id).
    Legacy default was ``$WORKSPACE_PATH/kernel-agent``; the env was
    removed during the all-artefacts-under-USER_DATA_PATH migration.
    """
    return Path(os.environ.get("USER_DATA_PATH", "/workspace/hyperloom")) / "kernel-agent"


def _geak_output_dir(session_id: str, prompt_file: Path) -> Path:
    out = _kernel_agent_root() / "geak" / session_id / prompt_file.stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def _set_yaml_tools_rag(text: str, enabled: bool) -> str:
    """Return YAML text with tools.rag set without mutating the source config."""
    value = "true" if enabled else "false"
    lines = text.splitlines()
    out: list[str] = []
    in_tools = False
    tools_indent = 0
    saw_tools = False
    wrote_rag = False

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        is_comment = line.lstrip().startswith("#")
        if re.match(r"\s*tools\s*:\s*(?:#.*)?$", line) and not is_comment:
            saw_tools = True
            in_tools = True
            tools_indent = indent
            out.append(line)
            continue
        if in_tools:
            if stripped and indent <= tools_indent and not is_comment:
                if not wrote_rag:
                    out.append(f"{' ' * (tools_indent + 2)}rag: {value}")
                    wrote_rag = True
                in_tools = False
            elif re.match(r"\s*rag\s*:", line):
                out.append(f"{' ' * indent}rag: {value}")
                wrote_rag = True
                continue
        out.append(line)

    if in_tools and not wrote_rag:
        out.append(f"{' ' * (tools_indent + 2)}rag: {value}")
    if not saw_tools:
        if out and out[-1].strip():
            out.append("")
        out.extend(["tools:", f"  rag: {value}"])
    return "\n".join(out) + "\n"


_DEFAULT_GEAK_FALLBACK_TIMEOUT_SEC = 3600


def _ensure_yaml_env_timeout(text: str, *, timeout: int = _DEFAULT_GEAK_FALLBACK_TIMEOUT_SEC) -> str:
    """Inject ``env.timeout`` if the GEAK config doesn't already define one.

    mini-swe-agent's ``LocalEnvironmentConfig.timeout`` defaults to 30 seconds.
    Without an explicit override, GEAK's test command dies with
    ``Test command timed out`` after 30s. This helper ensures the config
    always has a reasonable timeout (default 3600s).
    """
    timeout = max(60, int(timeout))
    has_env = re.search(r"^env\s*:\s*(?:#.*)?$", text, flags=re.MULTILINE)
    if has_env:
        # Only mutate the timeout line; leave the rest of the env block alone.
        m = re.search(
            r"^(env\s*:\s*(?:#.*)?\n(?:[ \t]+.*\n)*)",
            text,
            flags=re.MULTILINE,
        )
        if not m:
            return text
        block = m.group(1)
        timeout_re = re.compile(r"^([ \t]+)timeout\s*:\s*(\d+)\s*$", flags=re.MULTILINE)
        existing = timeout_re.search(block)
        if existing:
            current = int(existing.group(2))
            if current >= timeout:
                return text
            new_block = timeout_re.sub(f"{existing.group(1)}timeout: {timeout}", block, count=1)
        else:
            indent = "  "
            for line in block.splitlines()[1:]:
                stripped = line.lstrip(" \t")
                if stripped and not stripped.startswith("#"):
                    indent = line[: len(line) - len(stripped)] or indent
                    break
            new_block = block.rstrip("\n") + f"\n{indent}timeout: {timeout}\n"
        return text.replace(block, new_block, 1)
    addition = (
        "\n"
        "# Injected by Hyperloom kernel_optimization: mini-swe-agent\n"
        "# LocalEnvironmentConfig defaults timeout=30 which kills every\n"
        "# patch test inside the auto-generated unittest harness.\n"
        "env:\n"
        "  env:\n"
        "    PAGER: cat\n"
        "    MANPAGER: cat\n"
        "    LESS: -R\n"
        "    PIP_PROGRESS_BAR: 'off'\n"
        "    TQDM_DISABLE: '1'\n"
        f"  timeout: {timeout}\n"
    )
    if not text.endswith("\n"):
        text += "\n"
    return text + addition


def _geak_config_for_run(
    args: argparse.Namespace,
    prompt_file: Path,
) -> str:
    """Create a per-run GEAK config only when runtime overrides need it."""
    base_config = os.environ.get("GEAK_CONFIG", "")
    if not base_config or not Path(base_config).is_file():
        return base_config
    text = Path(base_config).read_text(encoding="utf-8", errors="replace")
    new_text = text
    if getattr(args, "disable_rag", False):
        new_text = _set_yaml_tools_rag(new_text, enabled=False)
    new_text = _ensure_yaml_env_timeout(new_text, timeout=_DEFAULT_GEAK_FALLBACK_TIMEOUT_SEC)
    if new_text == text:
        return base_config
    override = prompt_file.parent / f"{prompt_file.stem}.geak-config.yaml"
    override.write_text(new_text, encoding="utf-8")
    return str(override)


def _extract_py_path(test_command: str) -> str | None:
    """Extract the .py file path from a test command string."""
    try:
        for part in shlex.split(test_command):
            if part.endswith(".py"):
                return part
    except ValueError:
        pass
    return None


def _try_generate_harness(
    test_command: str,
    candidate: dict,
    source_file: str,
    out_dir: Path,
    kernel_repo: str,
    log_path: Path | None,
) -> str | None:
    """Try to auto-generate a GEAK-compatible harness from a benchmark file.

    Returns a new test_command pointing to the generated harness, or None.
    """
    bench_py = _extract_py_path(test_command)
    if not bench_py or not Path(bench_py).is_file():
        return None

    try:
        tools_dir = str(Path(__file__).resolve().parent)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        from harness_generator import maybe_generate_harness

        hr = maybe_generate_harness(
            benchmark_file=bench_py,
            candidate=candidate,
            source_file=source_file,
            out_dir=out_dir,
            kernel_repo=kernel_repo,
            log_fn=(lambda msg: append_log(log_path, msg)) if log_path else None,
        )
        if hr is not None:
            return hr.test_command
    except Exception as exc:
        if log_path:
            append_log(log_path, f"[harness_gen] failed: {exc}")
    return None


def _apply_geak_env_overrides(
    args: argparse.Namespace,
    prompt_file: Path,
) -> dict[str, str | None]:
    """Temporarily tune GEAK env for this attempt; caller must restore."""
    keys = ("GEAK_CONFIG", "GEAK_USE_KNOWLEDGE_BASE", "GEAK_SAVE_TO_KNOWLEDGE_BASE")
    previous = {key: os.environ.get(key) for key in keys}
    config = _geak_config_for_run(args, prompt_file)
    if config:
        os.environ["GEAK_CONFIG"] = config
    if getattr(args, "disable_xs_memory", False):
        os.environ["GEAK_USE_KNOWLEDGE_BASE"] = "0"
        os.environ["GEAK_SAVE_TO_KNOWLEDGE_BASE"] = "0"
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _oob_output_dir(session_id: str) -> Path:
    out = _kernel_agent_root() / "oob" / session_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _mirror_path_link(run_dir: Path, mirror: Path) -> None:
    """Create a relative symlink inside the run dir pointing at the mirror."""
    try:
        link_dir = run_dir / mirror.parent.name  # geak / oob
        link_dir.mkdir(parents=True, exist_ok=True)
        link = link_dir / mirror.name
        if link.exists() or link.is_symlink():
            return
        link.symlink_to(mirror, target_is_directory=True)
    except OSError:
        pass


def _git_checkout_fallback(kernel_repo: str, log_path: Path) -> None:
    """Best-effort `git checkout -- .` to undo any rogue agent writes under
    the kernel repo (e.g. claude ignoring the soft safety prompt). Idempotent
    and safe to call after every backend attempt."""
    if not kernel_repo:
        return
    git_dir = Path(kernel_repo) / ".git"
    if not git_dir.exists():
        return
    try:
        proc = subprocess.run(
            ["git", "-C", kernel_repo, "checkout", "--", "."],
            capture_output=True, text=True, timeout=60,
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
    gpu_ids, elapsed_s, cmd, optimized_path (optional), cli_workspace (oob).
    """
    # GEAK needs more wall-clock than claude/codex: a single sub-agent task
    # already takes 5-10 min (baseline + LLM patch generation + per-patch
    # benchmark), and the orchestrator typically dispatches 4-9 tasks per
    # round + a select_patch round at the end. 130 min is the default so
    # the prompt-quoted budget triggers GEAK v3.2.0's mode=full path
    # (yaml run.budgets.full.total_s=7200s + finalize_grace + kill_buffer);
    # override via --geak-budget-min.
    if backend == "geak":
        budget_min = float(getattr(args, "geak_budget_min", 0)
                           or getattr(args, "budget_minutes", 60) or 60)
    else:
        budget_min = float(getattr(args, "budget_minutes", 60) or 60)
    timeout_s = max(60, int(budget_min * 60))
    prefer_ray = ray_available()
    candidate = candidate or {}
    kernel_repo = str(candidate.get("kernel_repo") or "")
    bench_files: list[str] = list(candidate.get("benchmark_files") or [])
    # Resolve per-task GPU count: CLI override wins, then candidate hint,
    # then 1.
    num_gpus = max(1, int(getattr(args, "num_gpus", 0) or 0)
                   or int(candidate.get("num_gpus_recommended") or 1))
    try:
        if backend == "geak":
            geak = _import_backend("geak_submit")
            out_dir = _geak_output_dir(args.session_id, prompt_file)
            # test_command: prefer external --test-command generated by the
            # unittest skill; otherwise derive from benchmark files with the
            # kernel-aware selector/timeout wrapper.
            test_command = getattr(args, "test_command", "").strip()
            cand_name = str((candidate or {}).get("name") or "")
            is_multigpu = (
                bool((candidate or {}).get("is_multigpu"))
                or kernel_name_implies_multigpu(cand_name)
            )
            if not test_command:
                # GEAK preprocess may invent a non-existent bench path when
                # --test-command is empty. Always provide a real test command.
                test_command = _render_geak_test_command(
                    kernel_name=cand_name,
                    bench_files=bench_files,
                    is_multigpu=is_multigpu,
                    num_gpus=num_gpus,
                    timeout_sec=_profile_timeout_sec(),
                )
            # Auto-generate GEAK-compatible harness when needed (e.g. raw
            # AITER op_test without 4-mode CLI contract).
            if test_command:
                _harness_cmd = _try_generate_harness(
                    test_command, candidate, source_file, out_dir,
                    kernel_repo, log_path,
                )
                if _harness_cmd:
                    test_command = _harness_cmd
            if log_path is not None and test_command:
                append_log(log_path, f"[geak] test_command={test_command}")
            if test_command:
                import shutil as _shutil
                harness_dir = out_dir / "unittest"
                harness_dir.mkdir(parents=True, exist_ok=True)
                for _tc_part in test_command.split("&&"):
                    for _w in _tc_part.strip().split():
                        if _w.endswith(".py") and Path(_w).exists():
                            _dst = harness_dir / Path(_w).name
                            if _dst.exists() and _dst.resolve() == Path(_w).resolve():
                                continue
                            _shutil.copy2(_w, _dst)
            previous_env = _apply_geak_env_overrides(args, prompt_file)
            try:
                result = geak.submit(
                    prompt_file=prompt_file,
                    output_dir=out_dir,
                    kernel_path=source_file,
                    cost_limit=args.geak_cost_limit,
                    timeout_s=timeout_s,
                    num_gpus=num_gpus,
                    prefer_ray=prefer_ray,
                    kernel_repo=kernel_repo,
                    test_command=test_command,
                )
            finally:
                _restore_env(previous_env)
            result["stdout"] = result.get("stdout_tail", "")
            result["output_dir"] = str(out_dir)
            if test_command:
                result["test_command"] = test_command
            # Surface GEAK partial outputs (final_report.json / results dir)
            # so a SIGTERM'd attempt with patches on disk still gets
            # promoted to "partial" by the run_attempt scanner below.
            final_report = out_dir / "final_report.json"
            if final_report.is_file():
                result["geak_final_report"] = str(final_report)
            results_dir = out_dir / "results"
            if results_dir.is_dir():
                # Count any *.patch under results/ as evidence of partial work.
                patches = sorted(results_dir.rglob("*.patch"))
                if patches:
                    result["geak_results_dir"] = str(results_dir)
                    result["geak_patch_count"] = len(patches)
                    result["geak_latest_patch"] = str(patches[-1])
                # Per-task best_results.json: GEAK's heterogeneous orchestrator
                # writes one per sub-agent task with `best_patch_speedup` from
                # an LLM-judged comparison of patches against the baseline.
                # Aggregate them here so the driver can extract a real speedup
                # even when the run is SIGTERM'd before the top-level
                # final_report.json (select_patch round) finishes (observed in
                # r38: 60min budget burned by 9 sub-agent tasks; select_patch
                # never started). Take the max across tasks.
                best_jsons = sorted(results_dir.rglob("best_results.json"))
                if best_jsons:
                    best_speedup = 0.0
                    best_task = ""
                    best_patch_path = ""
                    for bj in best_jsons:
                        try:
                            d = json.loads(bj.read_text(encoding="utf-8"))
                            sp = float(d.get("best_patch_speedup") or 0.0)
                        except Exception:
                            continue
                        if sp > best_speedup:
                            best_speedup = sp
                            best_task = bj.parent.name
                            best_patch_path = str(d.get("best_patch_file") or "")
                    if best_speedup > 0:
                        result["geak_per_task_best_speedup"] = best_speedup
                        result["geak_per_task_best_task"] = best_task
                        if best_patch_path:
                            result["geak_per_task_best_patch"] = best_patch_path
                        # Surface the worktree directory holding the actual
                        # rewritten files that produced this best patch.
                        # GEAK's homogeneous orchestrator lays out each
                        # sub-agent's slot as
                        # ``results/round_<R>/parallel_<M>/`` for patches +
                        # ``results/round_<R>/worktrees/slot_<M>/`` for the
                        # checked-out repo it edited. Without this surface
                        # the downstream artifact extractor only sees the
                        # ``.patch`` (unified diff, often mixed with JIT
                        # cache binary deltas) and fails to recover a real
                        # ``.py`` source — see _select_source_artifact /
                        # _candidate_artifact_paths for the consumer.
                        wt = _geak_best_worktree(best_patch_path)
                        if wt:
                            result["geak_per_task_best_worktree"] = str(wt)
            return result
        if backend in {"claude", "codex", "cursor"}:
            oob = _import_backend("oob_submit")
            out_dir = _oob_output_dir(args.session_id)
            is_multigpu = bool((candidate or {}).get("is_multigpu"))
            if is_multigpu:
                keep = [f for f in bench_files if not Path(f).name.startswith("test_")]
                extras = keep[:3]
            else:
                extras = bench_files[:3]
            result = oob.submit(
                agent=backend,
                prompt_file=prompt_file,
                output_dir=out_dir,
                source_file=source_file,
                max_turns=args.oob_max_turns,
                timeout_s=timeout_s,
                num_gpus=num_gpus,
                prefer_ray=prefer_ray,
                extra_files=extras,
                kernel_repo=kernel_repo,
            )
            result["output_dir"] = str(out_dir)
            return result
        return {
            "returncode": 2,
            "stdout_tail": f"unknown backend: {backend}",
            "stderr_tail": "", "stdout": "", "gpu_ids": "",
            "elapsed_s": 0.0, "cmd": [],
        }
    finally:
        # Always undo any rogue writes under the kernel repo, regardless of
        # the backend's exit code (claude has been observed to ignore the
        # soft safety prompt and edit files in /sgl-workspace/aiter directly).
        if log_path is not None:
            _git_checkout_fallback(kernel_repo, log_path)


def env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def run_attempt(
    backend: str,
    *,
    args: argparse.Namespace,
    candidate: dict[str, Any],
    run_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    attempt_id = f"{backend}-{uuid.uuid4().hex[:8]}"
    prompt_dir = run_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / f"{attempt_id}.md"
    prompt_file.write_text(build_prompt(candidate, args, backend=backend), encoding="utf-8")

    source_file = args.source_file or str(candidate.get("source_file") or "")
    started = time.time()
    append_log(log_path, f"[attempt {attempt_id}] backend={backend}")

    source_suffix = Path(source_file).suffix if source_file else ".txt"
    # In dry-run we still emit a synthetic source-suffixed placeholder so the
    # downstream verification path can pick it up as a real artifact (preserved
    # for back-compat with the dry-run smoke tests). For real backend runs we
    # capture the raw subprocess stdout to a `.log` file instead of pretending
    # it is the optimized CU/PY source: GEAK / claude / codex stdout is the
    # mini-swe-agent / OOB conversation log, not a kernel source. Writing it
    # with a `.cu` suffix and then handing it to `_select_source_artifact`
    # made `_source_text_looks_complete` false-positive match generic
    # English text containing markers like "void " or "extern " and report
    # the log file as `artifact_source=source_file` (observed on Qwen3-8B
    # k007 rmsnorm_quant and k013 silu_and_mul, 2026-05-20). The new `.log`
    # suffix routes the stdout through `_extract_source_block` instead — that
    # path scans for fenced code blocks (Claude/Codex sometimes emit the full
    # optimized CU as ```cuda```) and only returns a real artifact when one
    # exists, otherwise verification falls back cleanly to GEAK patches /
    # `optimized_versions/` artefacts via the canonical backend_paths keys.
    #
    # Downstream-consumer contract: see ``kernel-agent/SKILL.md`` § *Per-
    # attempt stdout file naming* — external scripts (dashboards, breakdown
    # collector, etc.) MUST either read ``attempt["optimized_path"]`` from
    # ``optimization_attempts.jsonl`` or glob ``<attempt_id>*`` under
    # ``runs/<sid>/optimized/`` so both legacy ``_optimized.<suffix>`` and the
    # new ``_stdout.log`` are picked up transparently.
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
            placeholder = "extern \"C\" __global__ void optimized_kernel_placeholder() {}\n"
        optimized_path.write_text(placeholder, encoding="utf-8")
        result = {}
    else:
        append_log(log_path, f"$ invoke_backend({backend})")
        result = invoke_backend(backend, prompt_file, source_file, args,
                                candidate=candidate, log_path=log_path)
        returncode = int(result.get("returncode", 1))
        full_stdout = result.get("stdout") or result.get("stdout_tail") or ""
        stdout_tail = (full_stdout or "")[-4000:] or result.get("stderr_tail", "")
        if returncode == 0:
            status = "completed"
        elif returncode == 124:
            status = "timeout"
        else:
            status = "failed"
        # Always materialise the stdout `.log` (even on non-zero returncode);
        # this is the durable audit trail for the attempt and the source for
        # the code-fence extraction fallback.
        if full_stdout.strip():
            optimized_path.write_text(full_stdout, encoding="utf-8")
        append_log(log_path, stdout_tail)

    elapsed = round(time.time() - started, 3)

    backend_paths: dict[str, str] = {}
    if not args.dry_run:
        out_dir = result.get("output_dir") if isinstance(result, dict) else ""
        if out_dir:
            backend_paths["output_dir"] = out_dir
            # Prefer the workspace path emitted in oob's init ndjson event;
            # the previous mtime-based heuristic mis-attributed concurrent
            # replicas to each other (claude-rep0 → codex-rep1's dir).
            cli_workspace = (result.get("cli_workspace") or "") if isinstance(result, dict) else ""
            session_id_oob = (result.get("session_id") or "") if isinstance(result, dict) else ""
            cli_log = ""
            if cli_workspace:
                exec_log = Path(cli_workspace) / "execution.log"
                if exec_log.exists():
                    cli_log = str(exec_log)
                # Scan for partial outputs even when returncode != 0 so a
                # timed-out attempt doesn't get marked as 0-product:
                opt_dir = Path(cli_workspace) / "optimized_versions"
                if opt_dir.is_dir():
                    files = sorted(opt_dir.iterdir(), key=lambda p: p.stat().st_mtime)
                    if files:
                        backend_paths["partial_optimized_count"] = str(len(files))
                        backend_paths["partial_latest_optimized"] = str(files[-1])
                report = Path(cli_workspace) / "optimization_report.md"
                if report.exists():
                    backend_paths["partial_report"] = str(report)
            # /home/user/ rescue: claude occasionally ignores the absolute-
            # path system_prompt and writes to ~/optimized_versions/ instead
            # of the workspace cwd (observed pre-Fix-3 in r12-r17, recurs
            # rarely after). When the cli_workspace's optimized_versions/ is
            # empty but /home/user/optimized_versions/ has fresh files newer
            # than this attempt's start time, surface them so the report
            # is not silently lost.
            home_opt = Path("/home/user/optimized_versions")
            if (cli_workspace
                    and (not (Path(cli_workspace) / "optimized_versions").is_dir()
                         or not list((Path(cli_workspace) / "optimized_versions").iterdir()))
                    and home_opt.is_dir()):
                rescued = sorted(
                    [p for p in home_opt.iterdir() if p.is_file() and p.stat().st_mtime >= started],
                    key=lambda p: p.stat().st_mtime,
                )
                if rescued:
                    backend_paths["partial_optimized_count"] = str(len(rescued))
                    backend_paths["partial_latest_optimized"] = str(rescued[-1])
                    backend_paths["partial_optimized_rescued_from"] = str(home_opt)
                    home_report = Path("/home/user/optimization_report.md")
                    if (home_report.is_file()
                            and home_report.stat().st_mtime >= started
                            and "partial_report" not in backend_paths):
                        backend_paths["partial_report"] = str(home_report)
            if cli_workspace:
                backend_paths["cli_workspace"] = cli_workspace
            if cli_log:
                backend_paths["cli_execution_log"] = cli_log
            if session_id_oob:
                backend_paths["oob_session_id"] = session_id_oob
            test_cmd_used = (result.get("test_command") or "") if isinstance(result, dict) else ""
            if test_cmd_used:
                backend_paths["test_command"] = test_cmd_used
            # GEAK partial-output surface (forwarded by invoke_backend on
            # the geak branch). final_report.json / per-round patches.
            geak_final = (result.get("geak_final_report") or "") if isinstance(result, dict) else ""
            if geak_final:
                backend_paths["geak_final_report"] = geak_final
            geak_patch = (result.get("geak_latest_patch") or "") if isinstance(result, dict) else ""
            if geak_patch:
                backend_paths["geak_latest_patch"] = geak_patch
                backend_paths["geak_patch_count"] = str(result.get("geak_patch_count") or 0)
            # Per-task best speedup salvage (when select_patch round didn't
            # finish before SIGTERM). build_verification picks this up.
            per_task_sp = (result.get("geak_per_task_best_speedup")
                           if isinstance(result, dict) else None)
            if per_task_sp:
                backend_paths["geak_per_task_best_speedup"] = str(per_task_sp)
                bt = result.get("geak_per_task_best_task")
                if bt:
                    backend_paths["geak_per_task_best_task"] = str(bt)
                bp = result.get("geak_per_task_best_patch")
                if bp:
                    backend_paths["geak_per_task_best_patch"] = str(bp)
                wt = result.get("geak_per_task_best_worktree")
                if wt:
                    # Forward the worktree directory so build_verification's
                    # _candidate_artifact_paths can recover the rewritten
                    # ``.py`` (or ``.cu``) file under it instead of trying
                    # to scrape source out of GEAK's diff-with-binary-blobs
                    # ``.patch``.
                    backend_paths["geak_per_task_best_worktree"] = str(wt)
            # Promote any timed-out / failed attempt that left artifacts on
            # disk to "partial" so build_verification + make_proposal can
            # distinguish "killed but useful" from "truly empty failure".
            #
            # EXCEPTION: refuse promotion when stdout shows persistent
            # inner-LLM auth failure (>= _AUTH_RETRY_THRESHOLD 401-style
            # markers). An auth-loop typically leaves an empty
            # optimized_versions/ that fools the evidence check; without
            # this guard we ship PARTIAL and the orchestrator never
            # retires the kernel. See _AUTH_FAILURE_PATTERNS comment.
            partial_evidence_keys = (
                "partial_latest_optimized", "partial_report",
                "geak_final_report", "geak_latest_patch",
            )
            auth_loop_hits = _count_auth_failures(full_stdout)
            if auth_loop_hits >= _AUTH_RETRY_THRESHOLD:
                backend_paths["auth_failure_count"] = str(auth_loop_hits)
                backend_paths["auth_failure_marker"] = (
                    "persistent_inner_llm_401_loop_no_partial_promotion"
                )
                # Force status to a non-partial terminal state so
                # build_verification's `usable` filter excludes this
                # attempt and make_proposal returns REVERT.
                if status == "timeout":
                    status = "failed"
                # else: status is already "failed"; leave it alone.
            elif status in {"timeout", "failed"} and any(
                k in backend_paths for k in partial_evidence_keys
            ):
                status = "partial"

    return {
        "attempt_id": attempt_id,
        "backend": backend,
        "status": status,
        "error_type": status if status in {"backend_not_installed", "timeout"} else "",
        "returncode": returncode,
        "elapsed_s": elapsed,
        "prompt_path": str(prompt_file),
        "optimized_path": str(optimized_path) if optimized_path.exists() else "",
        "stdout_tail": stdout_tail,
        "created_at": utc_now(),
        "backend_paths": backend_paths,
    }


_SPEEDUP_PATTERNS = [
    # Match `speedup: 1.28x` / `Speedup: **1.076x**` / `avg=1.044x` etc.
    re.compile(r"(?im)^\s*\[micro_speedup\]\s*([0-9]+(?:\.[0-9]+)?)\s*[xX]\b"),
    re.compile(r"(?i)\bspeedup\b[^\n]{0,40}?([0-9]+(?:\.[0-9]+)?)\s*[xX]"),
    re.compile(r"(?i)\bavg(?:erage)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*[xX]\s+(?:speedup|across)"),
    re.compile(r"(?i)\b([0-9]+(?:\.[0-9]+)?)\s*[xX]\s+(?:speedup|faster)"),
]


# Persistent inner-LLM auth failure markers. When a backend's stdout
# contains >= AUTH_RETRY_THRESHOLD distinct matches we treat the run as
# a credential dead-end and refuse to promote `timeout`/`failed` to
# `partial`. Without this guard, GEAK's mini-swe-agent SelectPatchAgent
# can loop on a wrong-issuer gateway (observed: the inner agent hit
# `https://llm-api.amd.com/Anthropic` which expects a different
# `AMD_LLM_API_KEY` than the SAFE_API_KEY the outer GEAK CLI uses), leave
# an empty `optimized_versions/` on disk, trigger the partial-evidence
# path below, and ship back PARTIAL — which the orchestrator never
# retired (see inference_optimizer.shared_state.record_kernel_opt for
# the matching reject-on-partial change).
_AUTH_FAILURE_PATTERNS = [
    re.compile(r"\b401\b[^\n]{0,80}(unauthor|forbidden|client\s*error)", re.IGNORECASE),
    re.compile(r"HTTP/\d\.\d\s+401\b"),
    re.compile(r"Authentication\s*Error|Invalid\s*API\s*Key|invalid[._]api[._]key", re.IGNORECASE),
    re.compile(r"Subscription[- ]Key[^\n]{0,80}(missing|invalid|not\s*present)", re.IGNORECASE),
    re.compile(r"Primus\.00009\s+token\s+not\s+present", re.IGNORECASE),
]
_AUTH_RETRY_THRESHOLD = 3


def _count_auth_failures(text: str) -> int:
    """Count distinct inner-LLM auth-failure markers in *text*.

    The threshold-based gate in :func:`run_attempt` uses this to
    distinguish "a single transient 401 that retried successfully" from
    "every single retry hit 401 because the wrong gateway is being
    talked to and there is no recoverable path".
    """
    if not text:
        return 0
    total = 0
    for pat in _AUTH_FAILURE_PATTERNS:
        total += sum(1 for _ in pat.finditer(text))
    return total


def _extract_speedup_from_report(report_path: str | Path) -> float | None:
    """Best-effort scan of an OOB optimization_report.md for a speedup figure.

    Picks the MAX value across all matches (agents often report per-shape
    numbers and an aggregate; we want the headline). Returns None if nothing
    parseable is found or the file does not exist.
    """
    if not report_path:
        return None
    p = Path(report_path)
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    found: list[float] = []
    for pat in _SPEEDUP_PATTERNS:
        for m in pat.finditer(text):
            try:
                v = float(m.group(1))
                # Reject obvious junk (e.g. "100x faster" hyperbole)
                if 0.3 <= v <= 50.0:
                    found.append(v)
            except ValueError:
                continue
    if not found:
        return None
    # Use median-of-top-3 to dodge cherry-picked best-shape numbers; agents
    # tend to also print regression shapes which we don't want to filter out.
    found.sort(reverse=True)
    top = found[:3]
    return round(sum(top) / len(top), 4)


def _extract_speedup_from_geak(final_report_path: str | Path) -> float | None:
    """Pull best_speedup from a GEAK final_report.json if present and >0."""
    if not final_report_path:
        return None
    p = Path(final_report_path)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        v = float(d.get("best_speedup") or 0.0)
        return v if v > 0 else None
    except Exception:
        return None


def _extract_correctness_from_report(report_path: str | Path) -> bool | None:
    """Best-effort correctness signal from backend markdown/json reports."""
    if not report_path:
        return None
    p = Path(report_path)
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    lower = text.lower()
    marker = re.search(r"(?im)^\s*\[correctness\]\s*(pass|passed|fail|failed)\b", text)
    if marker:
        return marker.group(1).lower().startswith("pass")
    fail_markers = (
        "correctness failed", "incorrect", "mismatch", "assert_close failed",
        "not close", "wrong output", "validation failed", "correctness: fail",
        "correctness: failed", "does not match", "do not match",
        "failed against reference", "reference check failed",
    )
    pass_markers = (
        "correctness passed", "correctness: pass", "all tests passed",
        "assert_close passed", "torch.testing.assert_close passed",
        "validation passed", "matches reference", "match reference",
        "matched reference", "matches the reference", "verified against original",
        "verified against the original", "validated against original",
        "validated against the original", "validated against baseline",
        "outputs match", "output matches", "numerically matches",
        "reference comparison passed",
    )
    if any(marker in lower for marker in fail_markers):
        return False
    if any(marker in lower for marker in pass_markers):
        return True
    return None


def _trust_geak_correctness() -> bool:
    """Treat GEAK ``status=complete`` + measured speedup as sufficient
    correctness evidence by default.

    GEAK's ``save_and_test`` only verifies that the patch compiles and
    that ``import aiter`` succeeds; it does NOT exercise the kernel's
    numerical output (e.g. ck_moe_stage1 with a a8w8 blockscale harness
    only prints the aiter import banner). Without trusting GEAK, every
    GEAK KEEP candidate degrades to NEEDS_REVIEW because
    ``correctness_source == 'missing'`` and the patch never reaches
    integrate.

    Default ON: the integrate stage's E2E magpie benchmark is the
    ground-truth functional check, and operators can layer
    ``RUN_EVAL=true`` for an accuracy gate on top. Historical data
    (5 Qwen3-30B-A3B-Base sessions) shows GEAK 0/4 KEEP without this
    trust gate; with it, real shape-specific kernels like
    ck_moe_stage1's 1.30x patch reach the integrate REVERT/KEEP
    decision instead of being silently dropped.

    Set ``HYPERLOOM_TRUST_GEAK_CORRECTNESS=0`` to restore the
    conservative behaviour (every GEAK KEEP -> NEEDS_REVIEW) for
    operators that want human review before integrate.
    """
    raw = os.environ.get("HYPERLOOM_TRUST_GEAK_CORRECTNESS", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def _extract_correctness_from_geak(final_report_path: str | Path) -> bool | None:
    """Read correctness from GEAK-style JSON reports when present."""
    if not final_report_path:
        return None
    p = Path(final_report_path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    found: list[bool] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if "correct" in lk or "valid" in lk:
                    if isinstance(v, bool):
                        found.append(v)
                    elif isinstance(v, str):
                        lv = v.lower()
                        if lv in {"pass", "passed", "true", "ok", "success"}:
                            found.append(True)
                        elif lv in {"fail", "failed", "false", "error"}:
                            found.append(False)
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    if False in found:
        return False
    if True in found:
        return True
    return None


_SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".hip", ".py",
}
_FENCE_RE = re.compile(
    r"```(?P<lang>[A-Za-z0-9_+.-]*)\s*\n(?P<body>.*?)\n```",
    re.DOTALL,
)


def _source_text_looks_complete(text: str, suffix: str) -> bool:
    stripped = text.strip()
    if not stripped or "```" in stripped:
        return False
    if suffix == ".py":
        try:
            compile(stripped + "\n", "<optimized_kernel>", "exec")
        except SyntaxError:
            return False
        return any(
            marker in stripped
            for marker in ("def ", "class ", "import ", "@triton.jit", "torch.")
        )
    if suffix in {".cu", ".cuh", ".hip", ".cpp", ".cc", ".c", ".h", ".hpp"}:
        return any(
            marker in stripped
            for marker in (
                "#include", "__global__", "__device__", "extern ", "namespace ",
                "template", "void ", "int ", "float ", "half", "torch::",
            )
        )
    return False


def _extract_source_block(text_path: Path, target_suffix: str, output_path: Path) -> str:
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


def _geak_best_worktree(best_patch_path: str) -> Path | None:
    """Map a GEAK best-patch file path to the worktree slot it edited.

    GEAK's homogeneous orchestrator lays out::

        <patch_output_dir>/results/round_<R>/parallel_<M>/patch_<N>.patch
        <patch_output_dir>/results/round_<R>/worktrees/slot_<M>/<repo files>

    The ``parallel_<M>`` directory holding the patch and the
    ``worktrees/slot_<M>`` directory holding the actual rewritten
    files share their integer suffix ``M``. Recover the worktree
    directory from the patch path so callers can pick up the real
    source file instead of trying to scrape ``.py`` out of a unified
    diff. Returns ``None`` when the layout doesn't match the expected
    shape (e.g. a fixture / future GEAK reorg) so callers can fail
    soft and fall back to existing patch-based recovery.
    """
    if not best_patch_path:
        return None
    parent = Path(best_patch_path).parent
    parallel_name = parent.name
    if not parallel_name.startswith("parallel_"):
        return None
    slot_id = parallel_name[len("parallel_"):]
    if not slot_id:
        return None
    worktree = parent.parent / "worktrees" / f"slot_{slot_id}"
    if not worktree.is_dir():
        return None
    return worktree


def _worktree_source_paths(
    worktree: Path,
    *,
    source_file: str,
    kernel_repo: str,
) -> list[Path]:
    """Return concrete files under ``worktree`` that mirror ``source_file``.

    Resolution order (most precise first):

    1. ``source_file`` relative to ``kernel_repo`` → join with ``worktree``.
       This is the canonical mapping for kernels Hyperloom dispatched
       via TraceLens (we always have both the absolute source path and
       the repo root).
    2. Same file basename as ``source_file`` reachable anywhere within
       ``worktree`` (bounded recursive search). Defends against minor
       layout shifts when the source path has multiple plausible repo
       roots (e.g. aiter's ``aiter/ops/rmsnorm.py`` could also live at
       ``aiter/ops/triton/normalization/rmsnorm.py``).

    Each returned path is verified to exist on disk. Empty list when
    no match is found so the caller falls back to its other candidate
    sources.
    """
    if not worktree.is_dir() or not source_file:
        return []
    out: list[Path] = []
    if kernel_repo:
        try:
            rel = Path(source_file).resolve().relative_to(Path(kernel_repo).resolve())
        except (OSError, ValueError):
            rel = None
        if rel is not None:
            cand = worktree / rel
            if cand.is_file():
                out.append(cand)
    basename = Path(source_file).name
    if basename:
        for match in sorted(worktree.rglob(basename)):
            if match.is_file() and match not in out:
                out.append(match)
    return out


def _candidate_artifact_paths(
    attempt: dict[str, Any],
    target_suffix: str,
    *,
    source_file: str = "",
    kernel_repo: str = "",
) -> list[Path]:
    paths: list[Path] = []
    bp = attempt.get("backend_paths") or {}
    # GEAK worktree files first: when we have the worktree slot from
    # ``geak_per_task_best_worktree`` plus the original source path +
    # repo root, those rewritten files are the ground-truth artifact
    # the LLM actually edited. They precede ``.patch`` candidates so
    # the first-pass (suffix-match + compile) succeeds without ever
    # falling back to fence extraction on a diff-with-binary blobs.
    worktree_dir = bp.get("geak_per_task_best_worktree")
    if worktree_dir:
        paths.extend(
            _worktree_source_paths(
                Path(worktree_dir),
                source_file=source_file,
                kernel_repo=kernel_repo,
            )
        )
    for key in (
        "partial_latest_optimized",
        "geak_per_task_best_patch",
        "geak_latest_patch",
    ):
        value = bp.get(key)
        if value:
            paths.append(Path(value))
    cli_workspace = bp.get("cli_workspace")
    if cli_workspace:
        opt_dir = Path(cli_workspace) / "optimized_versions"
        if opt_dir.is_dir():
            paths.extend(sorted(
                (p for p in opt_dir.iterdir() if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ))
    out_dir = bp.get("output_dir")
    if out_dir:
        opt_dir = Path(out_dir) / "optimized_versions"
        if opt_dir.is_dir():
            paths.extend(sorted(
                (p for p in opt_dir.iterdir() if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ))
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


def _select_source_artifact(
    attempt: dict[str, Any],
    *,
    target_file: str,
    run_dir: Path | None = None,
    kernel_repo: str = "",
) -> tuple[str, str, str]:
    """Return (artifact_path, source, error) for a complete source artifact."""
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

    extraction_root = run_dir or Path(attempt.get("optimized_path") or "/tmp").parent
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

    tried = ", ".join(str(p) for p in candidates[:6])
    return "", "missing", f"no complete {target_suffix} source artifact found; tried: {tried}"


def build_verification(args: argparse.Namespace, attempts: list[dict[str, Any]], benchmark_available: bool) -> dict[str, Any]:
    # An attempt is usable if it either completed cleanly OR was killed past
    # the budget but left optimized_versions/ + report on disk (status=partial).
    usable = [a for a in attempts if a.get("status") in {"completed", "partial"}]
    best = None
    best_speedup = 0.0
    measured = False
    # Prefer the attempt with the highest extracted speedup; if none has a
    # measurable number, fall back to the first usable attempt with a 0.0 hint.
    for a in usable:
        bp = a.get("backend_paths") or {}
        report = bp.get("partial_report") or bp.get("report") or ""
        sp = _extract_speedup_from_report(report)
        if sp is None:
            sp = _extract_speedup_from_geak(bp.get("geak_final_report", ""))
        # Fallback: GEAK timed out before select_patch round wrote
        # final_report.json, but per-task best_results.json files have
        # speedups. Use the max across tasks (already aggregated by
        # invoke_backend → backend_paths["geak_per_task_best_speedup"]).
        if sp is None:
            try:
                per_task = bp.get("geak_per_task_best_speedup")
                if per_task is not None:
                    sp = float(per_task)
                    if sp <= 0:
                        sp = None
            except (ValueError, TypeError):
                sp = None
        if sp is not None:
            measured = True
            if sp > best_speedup:
                best_speedup = sp
                best = a
    if best is None and usable:
        best = usable[0]
    compile_passed = bool(best)
    best_artifact_path = ""
    artifact_source = "missing"
    artifact_error = "no usable backend attempt"
    if best is not None:
        target_file = str(getattr(args, "source_file", "") or "")
        # kernel_repo lets the worktree-recovery branch in
        # _candidate_artifact_paths map a TraceLens-style absolute
        # source path back to the relative path GEAK actually edited.
        kernel_repo = str(
            getattr(args, "kernel_repo", "")
            or getattr(args, "repo", "")
            or ""
        )
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
    correctness_signal = getattr(args, "correctness_passed", None)
    correctness_source = "cli_override" if correctness_signal is not None else "missing"
    if correctness_signal is None and best is not None:
        bp = best.get("backend_paths") or {}
        correctness_signal = _extract_correctness_from_report(
            bp.get("partial_report") or bp.get("report") or ""
        )
        if correctness_signal is not None:
            correctness_source = "report_scan"
    if correctness_signal is None and best is not None:
        bp = best.get("backend_paths") or {}
        correctness_signal = _extract_correctness_from_geak(
            bp.get("geak_final_report", "")
        )
        if correctness_signal is not None:
            correctness_source = "geak_report"
    # PR-E (default ON): trust GEAK's status=complete + measured speedup
    # as correctness=True even when the harness was an import-only test
    # (e.g. test_moe_gemm_a8w8_blockscale.py for an aiter ck_moe_stage1
    # kernel -- the harness loads aiter but does not exercise the kernel,
    # so patch_*_test.txt is empty and the standard extractors return
    # missing). GEAK's per-task save_and_test still confirms compile +
    # import succeed; the integrate stage's E2E magpie benchmark is the
    # ground-truth functional check (and operators can layer RUN_EVAL=true
    # for an accuracy gate on top). Set
    # ``HYPERLOOM_TRUST_GEAK_CORRECTNESS=0`` to disable.
    if (
        correctness_signal is None
        and best is not None
        and best.get("backend") == "geak"
        and measured
        and best_speedup >= 1.0
        and _trust_geak_correctness()
    ):
        bp_geak = (best.get("backend_paths") or {}).get("geak_final_report", "")
        geak_status = ""
        if bp_geak and Path(bp_geak).is_file():
            try:
                geak_status = str(
                    json.loads(Path(bp_geak).read_text(encoding="utf-8"))
                    .get("status") or ""
                ).lower()
            except Exception:  # noqa: BLE001
                geak_status = ""
        if geak_status in {"complete", "succeeded", "ok"}:
            correctness_signal = True
            correctness_source = "geak_assumed_pass"
    if correctness_signal is None and getattr(args, "accuracy_passed", None) is True:
        correctness_signal = True
        correctness_source = "accuracy_override"
    correctness_passed = bool(best and correctness_signal is True)
    if args.micro_speedup is not None:
        micro_speedup = float(args.micro_speedup)
        speedup_source = "cli_override"
    elif measured:
        micro_speedup = best_speedup
        speedup_source = "report_scan"
    elif getattr(args, "dry_run", False):
        # Dry-run is a smoke-test of the pipeline, not a real measurement.
        # Keep the legacy 1.05 placeholder so CI can exercise KEEP/REVIEW
        # paths without needing a real backend.
        micro_speedup = 1.05 if best else 0.0
        speedup_source = "dry_run_placeholder"
    else:
        # Real run with no parseable speedup → don't fake "improved";
        # leave it at 1.0 so PolicyGate can route to PARTIAL.
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
        "best_attempt_id": best["attempt_id"] if best else "",
        "best_backend": best["backend"] if best else "",
        "best_artifact_path": best_artifact_path,
        "artifact_valid": artifact_valid,
        "artifact_source": artifact_source,
        "artifact_error": "" if artifact_valid else artifact_error,
    }


def make_proposal(verification: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if not verification["compile_passed"]:
        # ``compile_passed`` is computed as ``bool(best)`` in build_verification,
        # so a False value can mean either "we compiled and it failed" OR
        # "we never had a usable backend attempt to compile from". Distinguish
        # them via ``artifact_error`` so operators can tell apart a real compile
        # regression (action: fix the kernel) from a backend-dispatch failure
        # (action: fix Ray / network / auth and retry).
        err = (verification.get("artifact_error") or "").strip()
        if err and verification.get("best_attempt_id", "") == "":
            return {"decision": "REVERT",
                    "reasons": [f"backend dispatch failed: {err}"]}
        return {"decision": "REVERT", "reasons": ["compile failed"]}
    if not verification["correctness_passed"]:
        reasons.append("correctness evidence missing or failed")
    if not verification.get("artifact_valid"):
        reasons.append("optimized source artifact missing or invalid")
    # Distinguish "we have artifacts but didn't measure a speedup" (PARTIAL,
    # human review can salvage) from "we measured and it's a regression"
    # (REVERT). The signal is verification["micro_speedup_source"]:
    #   * report_scan / cli_override → real number
    #   * default_unmeasured        → no number found, don't punish
    src = verification.get("micro_speedup_source", "default_unmeasured")
    if src == "default_unmeasured":
        # Real run with backend artifacts on disk but no measurable speedup;
        # don't punish as REVERT (regression) and don't lie as KEEP — leave
        # it at PARTIAL so a human reviewer can salvage from the report.
        reasons.append("no measurable speedup found in any backend report")
        return {"decision": "PARTIAL", "reasons": reasons}
    if verification["micro_speedup"] <= 1.0:
        return {"decision": "REVERT", "reasons": ["microbench did not improve"]}
    # Goal threshold: 1.10x lets modest but real shape-specific wins
    # (claude r19 GEMM 1.32x, GEAK r39 rms_norm 1.18x, codex r25 GEMM 1.66x)
    # through to human KEEP review. Below 1.10x is treated as noise / not
    # worth the production risk and routed to NEEDS_REVIEW with reason.
    # Originally 1.50 (overly strict), then 1.20; lowered to 1.10 May 2026.
    KEEP_THRESHOLD = 1.10
    if verification["micro_speedup"] < KEEP_THRESHOLD:
        reasons.append(
            f"speedup {verification['micro_speedup']:.3f}x below KEEP "
            f"threshold {KEEP_THRESHOLD:.2f}x"
        )
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
        return {
            "decision": "KEEP",
            "reasons": ["kernel artifact ready; E2E/accuracy deferred to integrate"],
        }
    return {"decision": "KEEP", "reasons": ["all required evidence passed"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Kernel Agent optimization tool")
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--session-id", default="")
    parser.add_argument(
        "--workspace-path",
        default=os.environ.get("USER_DATA_PATH", "/workspace/hyperloom"),
        help=(
            "Root the tool writes under (output lands at "
            "<workspace_path>/kernel-agent/runs/<session_id>/...). "
            "Defaults to $USER_DATA_PATH."
        ),
    )
    parser.add_argument("--candidates-path", default="")
    parser.add_argument("--backends", default="")
    parser.add_argument("--benchmark-file", default="")
    parser.add_argument("--test-harness-path", default="")
    parser.add_argument("--source-file", default="")
    parser.add_argument("--target-platform", default=_env_target_platform())
    parser.add_argument("--extra-sglang-args", default="")
    parser.add_argument("--budget-minutes", type=float, default=60.0,
                        help="Per-attempt wall-clock budget for claude/codex "
                             "OOB backends. GEAK uses --geak-budget-min.")
    # Default tracks $GEAK_RUN_MODE (exported by install.sh / env.sh):
    # quick (yaml total_s=3600s) -> 70 min, full (yaml total_s=7200s) -> 130 min.
    # Both sit above their yaml total_s + finalize_grace + kill_buffer + safety,
    # so the prompt-quoted budget triggers the matching mode (mini.py:435).
    _geak_budget_default = 70.0 if os.environ.get("GEAK_RUN_MODE", "full").strip().lower() == "quick" else 130.0
    parser.add_argument("--geak-budget-min", type=float, default=_geak_budget_default,
                        help="Per-attempt wall-clock budget for GEAK only "
                             "(default tracks $GEAK_RUN_MODE: full -> 130, "
                             "quick -> 70; both aligned with yaml "
                             "run.budgets.<mode>.total_s + finalize_grace + "
                             "kill_buffer + safety so the prompt-quoted "
                             "budget triggers the matching GEAK mode at "
                             "mini.py:435 task_extracted_mode).")
    parser.add_argument("--micro-speedup", type=float, default=None)
    parser.add_argument("--e2e-gain-pct", type=float, default=None)
    parser.add_argument("--correctness-passed", choices=["true", "false", "unknown"], default="unknown")
    parser.add_argument("--accuracy-passed", choices=["true", "false", "unknown"], default="unknown")
    parser.add_argument("--oob-max-turns", type=int, default=int(os.environ.get("KERNEL_AGENT_OOB_MAX_TURNS", "100")))
    # GEAK cost limit semantics:
    #   * GEAK's bundled ``config/geak.yaml`` declares ``cost_limit: 0.``
    #     (= unlimited) — that is the design contract the GEAK team picked.
    #   * GEAK's sub-agent spawn path (``parallel_agent`` → ``DefaultAgent``)
    #     does NOT honour that yaml entry; it falls back to
    #     ``AgentConfig.cost_limit = 3.0`` (``minisweagent/agents/default.py``).
    #     Observed 2026-05-15 on Qwen3-32B: every sub-agent died at $3.08
    #     after ~50 steps, well before producing a real optimisation.
    #   * The only externally addressable lever is GEAK's ``-l/--cost-limit``
    #     CLI option (``minisweagent/run/mini.py:194``) which writes
    #     ``config["agent"]["cost_limit"]`` and is honoured by every child
    #     agent spawned from that config.
    # We therefore default to ``0.0`` so Hyperloom matches GEAK's stated
    # geak.yaml contract instead of inheriting the dataclass-default $3
    # via the sub-agent fallback path. Operators can pin a finite cap with
    # ``HYPERLOOM_GEAK_COST_LIMIT`` or ``--geak-cost-limit`` when they want
    # a budget guardrail (e.g. CI smoke runs).
    parser.add_argument(
        "--geak-cost-limit",
        type=float,
        default=float(os.environ.get("HYPERLOOM_GEAK_COST_LIMIT", "0.0")),
        help=(
            "Per-attempt GEAK cost cap in USD; 0 means unlimited (mirrors "
            "GEAK's geak.yaml `cost_limit: 0.`). Set via "
            "$HYPERLOOM_GEAK_COST_LIMIT or this flag for CI budgets."
        ),
    )
    parser.add_argument("--disable-rag", action="store_true",
                        help="Run GEAK with tools.rag disabled for this request.")
    parser.add_argument("--disable-xs-memory", action="store_true",
                        help="Disable GEAK cross-session memory retrieval/write-back for this request.")
    parser.add_argument("--test-command", type=str, default="",
                        help="Test command from unittest skill. "
                             "Passed to GEAK as --test-command.")
    parser.add_argument("--num-gpus", type=int,
                        default=int(os.environ.get("KERNEL_AGENT_NUM_GPUS", "0")),
                        help="Per-task GPU reservation; 0 means follow the "
                             "candidate's num_gpus_recommended (1 for compute "
                             "kernels, 2 for communication kernels).")
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
        update_status(status_path, state="running", current_step="load_candidate",
                      log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                      started_at=started_at)
        candidates_path = Path(args.candidates_path) if args.candidates_path else run_dir / "kernel_candidates.json"
        all_candidates = load_candidates(candidates_path)
        candidate = find_candidate(all_candidates, args.kernel_id)
        if candidate is None:
            # The Orchestration LLM supplied a kernel_id that matches no
            # TraceLens candidate (e.g. a hallucinated operator name). Skip
            # this kernel cleanly instead of crashing the whole subprocess
            # with a KeyError, so the orchestrator can move on to the next
            # decision rather than burning the run.
            known = [str(c.get("kernel_id") or "") for c in all_candidates]
            msg = (
                f"kernel_id {args.kernel_id!r} not found among TraceLens "
                f"candidates {known}; skipping (no fabricated target)"
            )
            append_log(log_path, f"[skip] {msg}")
            update_status(status_path, state="skipped", current_step="skipped",
                          log_path=log_path, artifact_paths=artifacts,
                          run_id=run_id, started_at=started_at, error=msg)
            print(json.dumps({
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
            }, indent=2, sort_keys=True))
            return 0
        if (
            candidate.get("reusable_native_kernel") is False
            or not candidate.get("source_file")
        ):
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
            update_status(status_path, state="skipped", current_step="skipped",
                          log_path=log_path, artifact_paths=artifacts,
                          run_id=run_id, started_at=started_at, error=msg)
            print(json.dumps({
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
                    "missing_native_source"
                    if not candidate.get("source_file")
                    else "non_reusable_kernel"
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
            }, indent=2, sort_keys=True))
            return 0
        # TraceLens is the source of truth for kernel_id → source_file.
        # ``_resolve_source_file`` overrides any LLM-supplied path that
        # disagrees with ``candidate.source_file`` and logs the override,
        # so a kernel-ID confusion at the Orchestration layer (e.g. fmoe
        # k001's source attached to fmha k003) no longer routes GEAK's
        # rewrite at the wrong file.
        resolved_source = _resolve_source_file(
            args.source_file, candidate, args.kernel_id, log_path
        )
        args.source_file = resolved_source
        # Forward the candidate's repo root onto args so build_verification's
        # GEAK-worktree artifact recovery can map ``source_file`` (an
        # absolute path produced by the TraceLens resolver) back to the
        # repo-relative path GEAK edited inside its worktree slot.
        # Empty when the candidate didn't carry a repo (e.g. legacy
        # CSV-only fixtures); the worktree recovery falls back to a
        # basename-based rglob in that case.
        if not getattr(args, "kernel_repo", None):
            args.kernel_repo = str(candidate.get("kernel_repo") or "")
        if not args.dry_run and not resolved_source:
            raise RuntimeError(
                f"source file not resolved for kernel {args.kernel_id}; "
                "skipping backend dispatch (no fabricated source allowed)"
            )
        selected_backends, backend_notes = choose_backends(args, candidate)
        backend_notes["rag_enabled"] = not args.disable_rag
        backend_notes["xs_memory_enabled"] = not args.disable_xs_memory
        benchmark_available = bool(backend_notes["benchmark_available"])
        append_log(log_path, f"kernel_id={args.kernel_id}")
        append_log(log_path, f"resolved_source={resolved_source or 'NONE'}")
        append_log(log_path, f"selected_backends={','.join(selected_backends) or 'none'}")

        update_status(status_path, state="running", current_step="run_backends",
                      log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                      started_at=started_at)
        attempts: list[dict[str, Any]] = []
        for backend in selected_backends:
            attempt = run_attempt(backend, args=args, candidate=candidate,
                                  run_dir=run_dir, log_path=log_path)
            attempt.update(backend_notes)
            attempts.append(attempt)
            append_jsonl(run_dir / "optimization_attempts.jsonl", attempt)

        update_status(status_path, state="running", current_step="verify_and_propose",
                      log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                      started_at=started_at)
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
        atomic_write_json(result_path, result)
        artifacts.update(result["artifact_paths"])
        update_status(status_path, state="succeeded", current_step="done",
                      log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                      started_at=started_at)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        append_log(log_path, f"[error] {type(exc).__name__}: {exc}")
        update_status(status_path, state="failed", current_step="failed",
                      log_path=log_path, artifact_paths=artifacts, run_id=run_id,
                      started_at=started_at, error=f"{type(exc).__name__}: {exc}")
        print(json.dumps({
            "tool": "kernel_optimization",
            "session_id": session_id,
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "cli_log_path": str(log_path),
            "status_path": str(status_path),
        }, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
