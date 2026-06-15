#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

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

# Sibling import: kernel name → multi-GPU collective detection (torchrun vs python).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _collective_names import kernel_name_implies_multigpu  # noqa: E402
from _paths import workspace_root  # noqa: E402
sys.path.pop(0)


def utc_now() -> str:
    """Return the current UTC time as an ISO8601 string.

    Returns:
        str: The current UTC timestamp in ISO-8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON to ``path`` using a temp file then rename.

    Args:
        path (Path): Destination JSON file; parent dirs are created.
        data (dict[str, Any]): JSON-serializable payload to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as tmp:
        json.dump(data, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    """Append one JSON object as a line to the given JSONL file.

    Args:
        path (Path): Destination JSONL file; parent dirs are created.
        data (dict[str, Any]): JSON-serializable object to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(data, sort_keys=True) + "\n")


def append_log(log_path: Path, message: str) -> None:
    """Append a log line to ``log_path`` (ensuring parent dirs exist).

    Args:
        log_path (Path): Log file to append to.
        message (str): Text to append; trailing whitespace is stripped and a
            newline is added.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def read_last_lines(log_path: Path, limit: int = 20) -> list[str]:
    """Return the last ``limit`` lines of a log file, empty if missing.

    Args:
        log_path (Path): Log file to read.
        limit (int): Maximum number of trailing lines to return.

    Returns:
        list[str]: The trailing lines, or an empty list when the file does
            not exist.
    """
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
    """Locate the ``kernel_candidates.json`` TraceLens wrote for this session.

    PR-C (``tracelens_analysis.py``) moved every TraceLens invocation's
    outputs into a per-run sub-directory
    ``runs/<session_id>/<compact_ts>_<run_id>/`` so successive watermark
    refreshes no longer overwrite each other. ``kernel_optimization.py``
    still keys its own artifacts off the flat ``runs/<session_id>/`` root,
    so the candidates file is no longer where the pre-PR-C lookup expected
    it. Resolve it here with the same "latest pointer" semantics the
    roofline sidecar uses:

    * honour the flat legacy path (``run_dir/kernel_candidates.json``) when
      present so pre-PR-C sessions / callers that drop the file at the
      session root keep working;
    * otherwise descend into the newest ``<ts>_<run_id>`` sub-directory that
      actually carries a ``kernel_candidates.json`` — the compact-timestamp
      prefix sorts chronologically, so ``max()`` is the most recent run.

    Falls back to the flat path (which ``load_candidates`` will surface as a
    clean ``FileNotFoundError``) when nothing matches, preserving the
    "no fabricated target" failure mode for genuinely missing analyses.
    """
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
            # Sort by the parent sub-dir name (``<compact_ts>_<run_id>``);
            # the zero-padded timestamp prefix makes lexical order == time
            # order, so the last element is the most recent TraceLens run.
            return max(sub_candidates, key=lambda p: p.parent.name)
    return flat


def load_candidates(path: Path) -> list[dict[str, Any]]:
    """Load kernel candidates from JSON, normalizing legacy shapes.

    Per Hyperloom#314 returns the union of ``hot_kernels`` (routable) + ``skipped_kernels`` so id lookup still resolves non-routable kernels; legacy flat-list / ``kernel_candidates`` shapes respected.
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
    """Fold hallucinated ``kn``/``rn`` prefixes onto the real ``k`` numbering, lower-cased, for tolerant comparison."""
    s = value.strip().lower()
    for prefix in ("kn", "rn"):
        if s.startswith(prefix) and s[len(prefix):].isdigit():
            return "k" + s[len(prefix):]
    return s


def find_candidate(
    candidates: list[dict[str, Any]], kernel_id: str
) -> dict[str, Any] | None:
    """Resolve a candidate by exact ``kernel_id``, then unique routable ``name``, then normalized id (``kn``/``rn``→``k``); ``None`` if nothing matches (caller skips gracefully)."""
    for candidate in candidates:
        if candidate.get("kernel_id") == kernel_id:
            return candidate
    # Names aren't stable ids (``aten::mm`` is shared); accept only a unique routable match.
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
    """Report whether any usable benchmark/test harness exists for a kernel.

    Checks the CLI-supplied benchmark/harness paths and the candidate's own
    ``benchmark_file`` / ``test_harness_path`` / ``benchmark_files`` fields,
    treating only paths that resolve on disk as available.

    Args:
        args (argparse.Namespace): Parsed CLI args carrying ``benchmark_file``
            and ``test_harness_path``.
        candidate (dict[str, Any]): Kernel candidate dict that may declare
            benchmark/harness paths.

    Returns:
        bool: True when at least one referenced benchmark or harness file
            exists on disk.
    """
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

    Candidate wins over the LLM's ``--source-file`` (which can mismatch the
    kernel, e.g. DeepSeek-R1 routed an MHA rewrite at ``fused_moe.py``); a
    differing LLM path emits a ``[source-override]`` warning. Falls back to the
    LLM path when the candidate has no source_file.
    """
    cand_source = str((candidate or {}).get("source_file") or "").strip()
    llm = str(llm_source or "").strip()
    if not cand_source:
        return llm

    # A candidate "source" can be a profiler frame label (pseudo-ops, TraceLens PR #668),
    # not a real file; prefer the caller's explicit source_file when it's a readable file.
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


# Kernel-name → benchmark-name priority patterns (specific families first). Each maps a
# kernel regex to priority-ordered bench-filename regexes; a non-match preserves original order.
_BENCHMARK_PATTERNS: list[tuple["re.Pattern[str]", list["re.Pattern[str]"]]] = [
    # Flash / multi-head attention (BEFORE paged-attn so fmha doesn't hit test_pa.py).
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
    off-topic benchmark (e.g. fmha → test_pa.py stalling GEAK Step-5).
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


def _profile_timeout_sec() -> int:
    """Per-subprocess profiling timeout (seconds) for GEAK's Step 5.

    Injected as a ``timeout <N>`` prefix on the test_command so a default-matrix
    benchmark (e.g. aiter test_pa.py) can't stall Step 5 for hours; SIGTERM at N
    surfaces as a normal profiling failure. Default 600s, override via
    ``KERNEL_OPT_PROFILE_TIMEOUT_SEC``, floored at 1.
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

    Args:
        kernel_name (str): Kernel name used to order benchmark candidates.
        bench_files (list[Any]): Candidate benchmark/test file paths.
        is_multigpu (bool): True when the kernel implies a multi-GPU
            collective and needs ``torchrun``.
        num_gpus (int): Number of GPUs to pass to ``--nproc_per_node`` when
            multi-GPU.
        timeout_sec (int): Per-subprocess timeout prefixed via ``timeout``.

    Returns:
        str: The rendered ``--test-command`` string, or empty string when no
            usable benchmark file is found.
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
    """Parse and validate a comma-separated backend list.

    Args:
        backends (str): Comma-separated backend names (case-insensitive),
            e.g. ``"geak,claude"``.

    Returns:
        list[str]: The normalized (lowercased, trimmed) backend names in the
            order given.

    Raises:
        ValueError: When any backend is outside the allowed set
            (``geak``, ``claude``, ``codex``, ``cursor``).
    """
    parsed = [b.strip().lower() for b in backends.split(",") if b.strip()]
    # `forge` is the Kernel-Forge autonomous-loop backend; it is first in the
    # default ladder (choose_backends) and falls through to geak/claude/codex
    # when it skips a non-triton candidate or misses a KEEP. See
    # claw-dev/docs-zh/forge-as-hyperloom-backend-integration.md.
    allowed = {"geak", "claude", "codex", "cursor", "forge"}
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

    Args:
        args (argparse.Namespace): Parsed CLI args carrying ``backends`` and
            benchmark/harness paths.
        candidate (dict[str, Any]): Kernel candidate dict, used for
            ``source_type`` and benchmark availability.

    Returns:
        tuple[list[str], dict[str, Any]]: The selected backend ladder and a
            notes dict describing the selection (benchmark availability,
            ``geak_without_benchmark`` flag, cursor key presence, etc.).
    """
    user_backends = parse_backends(args.backends)
    # Honor the coordinator's KERNEL_OPT_BACKEND_ORDER / KERNEL_OPT_BACKENDS env
    # when no explicit --backends was passed: the single-kernel subprocess used to
    # ignore it and fall back to the full default ladder, so a forge-only run
    # (KERNEL_OPT_BACKEND_ORDER=forge) still fired geak/claude/codex. Mirror the
    # handler's _backend_order precedence here so the subprocess agrees.
    if not user_backends:
        env_order = (os.environ.get("KERNEL_OPT_BACKEND_ORDER")
                     or os.environ.get("KERNEL_OPT_BACKENDS") or "").strip()
        if env_order:
            user_backends = parse_backends(env_order)
    benchmark_available = has_benchmark(args, candidate)
    source_type = str(candidate.get("source_type") or "unknown")
    # Skip cursor from auto-selected defaults when CURSOR_API_KEY is unset (explicit --backends still wins).
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

    # Unified ladder: forge FIRST (Kernel-Forge autonomous loop; falls through to
    # geak/claude/codex when forge skips a non-triton candidate or misses a KEEP),
    # then GEAK, then claude/codex. Without a benchmark GEAK still attempts but flags
    # geak_without_benchmark=True so KEEP gates know confidence is reduced.
    selected = ["forge", "geak", "claude", "codex"]
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
        notes = "\n".join([
            "Hardware notes (target platform unknown):",
            "- Before benchmarking, query the runtime environment for the ROCm arch ",
            "(hipDeviceGetName/rocminfo), visible GPU IDs (ROCR_VISIBLE_DEVICES), "
            + "and memory size/bandwidth.",
            "- Record those values in the result and choose --offload-arch=<arch> "
            + "accordingly; replace <arch> with the inspected ROCm arch before running.",
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
    """Render a candidate row's ``shapes`` field as one comma-joined line."""
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
                # Some rows carry {call_num, shape} dicts.
                shape = entry.get("shape") or entry.get("Args") or ""
                call_num = entry.get("call_num")
                if shape:
                    parts.append(f"{shape}" + (f" (x{call_num})" if call_num else ""))
            else:
                parts.append(str(entry))
        return ", ".join(p for p in parts if p)
    return str(shapes)


_SHAPE_ARG_RE = re.compile(
    r"^\s*\((?P<dims>[^)]*)\)\s*(?P<dtype>[A-Za-z0-9_]+)?\s*$"
)


def _split_shape_fragments(shape_text: Any) -> list[str]:
    """Split TraceLens ``Args`` text into per-argument shape fragments."""
    text = str(shape_text or "").strip()
    if not text:
        return []
    return [
        frag.strip()
        for frag in re.split(r"\s*(?:<br\s*/?>|\n)\s*", text, flags=re.IGNORECASE)
        if frag.strip()
    ]


def _parse_shape_arg(raw: Any, *, index: int) -> dict[str, Any]:
    """Parse one shape fragment such as ``(15360,8,768) bf16``."""
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
    """Build one structured benchmark shape case from TraceLens shape data."""
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
        args = [
            _parse_shape_arg(fragment, index=idx)
            for idx, fragment in enumerate(fragments)
        ]
    return {
        "primary": bool(primary),
        "call_count": parsed_count,
        "raw": str(raw_shape or "").strip(),
        "args": args,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce a TraceLens numeric field, returning ``default`` on drift."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _structured_benchmark_shape_cases(candidate: dict[str, Any]) -> dict[str, Any]:
    """Expose primary/supplementary serving shapes in machine-readable form."""
    group = candidate.get("task_group")
    rows = group.get("rows") if isinstance(group, dict) else None
    cases: list[dict[str, Any]] = []
    input_shapes = candidate.get("input_shapes")
    is_synthetic = bool(candidate.get("_input_shapes_synthetic"))
    if isinstance(rows, list) and rows:
        # A task_group represents one dispatch covering multiple observed
        # shapes for the same source function. Prefer its rows so the prompt
        # keeps supplementary shapes instead of only the primary candidate.
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            case = _shape_case_from_value(
                row.get("shapes"),
                call_count=row.get("call_count"),
                primary=idx == 0,
            )
            case.update({
                "operation": str(row.get("name") or ""),
                "aggregate_time_ms": _safe_float(row.get("duration_us")) / 1000.0,
                "percent_e2e": row.get("percent_of_total"),
                "bound": str(row.get("bound_type") or ""),
                "source": "task_group",
            })
            if case["raw"] or case["args"]:
                cases.append(case)
    if (
        not cases
        and isinstance(input_shapes, list)
        and input_shapes
        and not is_synthetic
    ):
        # Only use input_shapes when they come from a real program output
        # (TraceLens / runtime enrichment), not from the synthetic
        # legacy-shapes conversion in enrich_candidates_with_runtime_metadata.
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

    Surfaces the candidate's TraceLens-captured argument shapes so GEAK binds
    its (self-generated) harness to the EXACT shapes the kernel saw during
    serving -- the optimization signal must match the workload or a kernel-level
    speedup will not translate to an end-to-end gain. Generic: applies to any
    candidate carrying captured shapes; returns ``""`` when none exist so the
    prompt stays byte-identical to legacy in that case.
    """
    shapes = candidate.get("shapes") or candidate.get("kernel_shapes")
    rendered = _format_shapes_for_case(shapes)
    if not rendered:
        return ""
    bound = str(candidate.get("bound_type") or candidate.get("bound") or "").strip()
    bound_line = f" (bound: {bound})" if bound else ""
    return (
        "\n## Benchmark shapes (TraceLens-captured from the serving run)\n\n"
        "Build your harness shape sweep / `get_inputs()` from EXACTLY these\n"
        f"captured argument shapes{bound_line} -- do NOT invent shapes. They are what\n"
        "the kernel saw during sglang/vLLM serving, so optimizing against them is\n"
        "what produces an end-to-end gain on the workload:\n"
        f"- args: {rendered}\n"
        "Correctness golden: the ORIGINAL kernel's output on these shapes "
        "(baseline / `fn=` injection); do not hand-derive a reference from scratch.\n"
    )


def _build_benchmark_cases_block(candidate: dict[str, Any]) -> str:
    """Render the multi-row benchmark cases section for a task_group.

    Falls back to :func:`_build_captured_shapes_block` when ``candidate["task_group"]`` is absent/empty so captured shapes still reach GEAK. With a task_group, emits one bullet per TraceLens row (sorted by aggregate time desc) surfacing operation/args/aggregate_time_ms/percent_e2e/count/per_call_ms/flops_per_byte/efficiency/bound (bound + per_call_ms drive backend dispatch).
    """
    group = candidate.get("task_group")
    rows = group.get("rows") if isinstance(group, dict) else None
    if not (isinstance(rows, list) and rows):
        # No task_group (or no rows): still surface the captured serving
        # shapes so GEAK's harness is bound to the real workload (generic
        # fallback). Returns "" when the candidate carries no shapes either.
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
        lines.extend([
            f"This kernel resolves to the same source function across "
            f"{len(rows)} TraceLens rows ("
            f"{function_name or '<unknown function>'}"
            + (f" at {location}" if location else "")
            + (", AST-resolved" if ast_resolved else "")
            + "). Optimize the source function once; the patch applies "
            "to all rows below. Use the first row as the primary",
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


# PR-B §3: ordered optimization directions keyed by bound type so the first lever
# matches the kernel's bottleneck (``compute`` flips the top two; ``unknown`` is the default order).
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
        (
            "2. **Shape-aware tuning**: specialize block sizes and grid indexing "
            + "for the dominant TraceLens Args."
        ),
        (
            "3. **Memory traffic reduction**: improve coalescing / vectorization, "
            + "reduce intermediate writes, avoid extra global-memory round trips."
        ),
        "4. **Launch amortization** for tiny high-count decode shapes.",
        (
            "5. **Compute utilization**: improve MFMA tile choice, occupancy, "
            + "register / shared-memory balance."
        ),
    ],
}


def _classify_bound(bound_type: str) -> str:
    """Map TraceLens ``bound`` strings to one of ``memory`` / ``compute`` / ``unknown``."""
    text = (bound_type or "").lower()
    if "memory" in text or "bandwidth" in text or "hbm" in text:
        return "memory"
    if "compute" in text or "arithmetic" in text or "flops" in text:
        return "compute"
    return "unknown"


def _build_priority_block(candidate: dict[str, Any]) -> str:
    """Render the bound-keyed optimization priority list (uses the primary row's bound for a task_group).

    Empty string when ``bound_type`` is missing and no ``task_group`` is attached.
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
    JSON round-trips that may have already converted them to strings.

    Args:
        value (Any): Value to coerce to ``float``.
        default (float): Fallback returned when ``value`` is None/empty or
            cannot be parsed.

    Returns:
        float: The parsed float, or ``default`` on any failure.
    """
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_impact_range(
    low_ms: float, low_e2e: float, high_ms: float, high_e2e: float,
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


def _build_hypothesis_block(candidate: dict[str, Any]) -> str:
    """Render a TraceLens hypothesis section for the candidate, if any.

    Empty when no prose fields are present. A multi-P-item ``task_group`` renders
    every P-item's prose under a ``### P{rank}`` header; otherwise a single block.
    The reasoning/resolution prose is labelled a hypothesis to validate (it is
    itself LLM-generated); the numeric impact range is roofline arithmetic.
    """
    # Multi-P-item case (Q2): render every P-item's prose so GEAK sees all framings.
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

    # Single-P-item / no-P-item path: read prose from the candidate directly.
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
            entries.append({
                "call_num": int(shape.get("call_num") or count),
                "shape": shape["shape"],
            })
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
    # Canonical key is ``extra_server_args`` (legacy ``extra_sglang_args`` still read by the shim).
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

    metadata = {
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
        # PR-K: source attribution. launcher_source_file is the @compile_ops wrapper;
        # kernel_path above is the device source to rewrite. Both empty/False when un-promoted.
        "launcher_source_file": str(candidate.get("launcher_source_file", "") or ""),
        "source_promoted_from_launcher": bool(
            candidate.get("source_promoted_from_launcher"),
        ),
    }
    if benchmark_shape_cases:
        metadata["benchmark_shape_cases"] = benchmark_shape_cases
    return metadata


def build_prompt(
    candidate: dict[str, Any],
    args: argparse.Namespace,
    *,
    backend: str | None = None,
) -> str:
    """Render the full optimization prompt handed to a rewrite backend.

    Assembles the hardware/budget/source-attribution preamble, the source
    listing, semantically-ordered benchmark references, and the TraceLens
    benchmark-cases / priority / hypothesis blocks into one prompt string.

    Args:
        candidate (dict[str, Any]): Kernel candidate dict supplying source,
            benchmarks, shapes, and TraceLens context.
        args (argparse.Namespace): Parsed CLI args (source override, GPU
            count, kernel id, etc.).
        backend (str | None): Target backend name, used to tailor backend-
            specific prompt sections; None for the generic prompt.

    Returns:
        str: The fully rendered prompt text for the rewrite backend.
    """
    source_file = args.source_file or candidate.get("source_file", "")
    source_block = ""
    if source_file and Path(str(source_file)).exists():
        content = Path(str(source_file)).read_text(encoding="utf-8", errors="replace")
        source_block = f"\nSource content:\n```\n{content[:12000]}\n```"
    kernel_repo = str(candidate.get("kernel_repo") or "")
    bench_files = candidate.get("benchmark_files") or []
    if isinstance(bench_files, str):
        bench_files = [bench_files]
    # Sort by semantic match so the most-relevant benchmarks head the [:8]-clipped list.
    bench_files = _match_benchmark_for_kernel(
        str(candidate.get("name") or ""), bench_files
    )
    is_multigpu = bool(candidate.get("is_multigpu"))
    # GPU count: CLI override, then candidate hint, then 1.
    num_gpus = max(1, int(getattr(args, "num_gpus", 0) or 0)
                   or int(candidate.get("num_gpus_recommended") or 1))
    # Map source_type to GEAK's kernel_type vocabulary for task_parser routing.
    geak_kernel_type = _GEAK_KERNEL_TYPE.get(str(candidate.get("source_type", "unknown")), "other")
    kernel_name = str(candidate.get("name", args.kernel_id))
    kernel_metadata = build_kernel_metadata(candidate, args)
    # Budget-protocol preamble: tell the LLM the ``step N ($X.XX)`` header is a cost meter,
    # not a stop sign (cost-limit is disabled; only the wall-clock timeout ends the task).
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
    # PR-K: render a hard-rule notice when the source was promoted from a @compile_ops
    # wrapper to the device file, so the LLM rewrites the device file. Empty if un-promoted.
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
    # Quote the per-backend wall-clock so GEAK's task-mode parser infers the right mode (>=120min→full).
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
    # Honour $MULTI_NODE_STATE_FILE (default /tmp/multi_node_state.json), same
    # resolution as apply_kernel_patch._mn_state_path / _multi_node_env. This
    # keeps test isolation intact: a stale real /tmp state file no longer
    # misclassifies a single-node run as multi-node.
    mn_state_file = Path(
        os.environ.get("MULTI_NODE_STATE_FILE", "/tmp/multi_node_state.json")
    )
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
    # Hyperloom#307: only fall back to dumping the full analysis.md when no per-kernel
    # hypothesis_block could be rendered (else it bloats the prompt and surfaces other P-items).
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
    # Use GEAK task_parser field names so its parser can extract them; OOB reads the same body as prose.
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
        (
            "Shape contract: when `benchmark_shape_cases` is present in the "
            "metadata, benchmark its `primary_shape` first and use "
            "`supplementary_shapes` only as additional coverage. Do not invent "
            "shapes or reorder tensor arguments."
            if kernel_metadata.get("benchmark_shape_cases")
            else ""
        ),
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
    """Dynamically load kernel-agent/tools/backends/<name>.py (dir added to sys.path so submodules cross-import)."""
    backends_dir = _backends_module_dir()
    if str(backends_dir) not in sys.path:
        sys.path.insert(0, str(backends_dir))
    import importlib
    return importlib.import_module(name)


def _kernel_agent_root() -> Path:
    """Output root for kernel-agent tools at ``$USER_DATA_PATH/kernel-agent`` (via workspace_root, which warns once when unset)."""
    return Path(workspace_root()) / "kernel-agent"


def _geak_output_dir(session_id: str, prompt_file: Path) -> Path:
    """Return (creating if needed) the GEAK output dir for a run.

    Args:
        session_id (str): Per-session identifier namespacing the output.
        prompt_file (Path): Prompt file whose stem names the run subdir.

    Returns:
        Path: The created ``.../geak/<session_id>/<prompt_stem>`` directory.
    """
    out = _kernel_agent_root() / "geak" / session_id / prompt_file.stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def _set_yaml_tools_rag(text: str, enabled: bool) -> str:
    """Return YAML text with tools.rag set without mutating the source config.

    Args:
        text (str): The original GEAK YAML config text.
        enabled (bool): Whether ``tools.rag`` should be ``true`` or ``false``.

    Returns:
        str: A new YAML string with the ``tools.rag`` value set/inserted.
    """
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
    """Inject ``env.timeout`` (default 3600s) if absent; mini-swe-agent defaults to 30s and would kill the test command."""
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
    """Create a per-run GEAK config only when runtime overrides need it.

    Reads the base ``GEAK_CONFIG`` file and, when CLI overrides (RAG
    disable, env timeout) change it, writes a per-run override file next to
    the prompt; otherwise the original config path is returned unchanged.

    Args:
        args (argparse.Namespace): Parsed CLI args (e.g. ``disable_rag``).
        prompt_file (Path): Prompt file whose directory hosts any override.

    Returns:
        str: Path to the config GEAK should use — the base config when no
            override is needed, otherwise the per-run override file.
    """
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
    """Extract the .py file path from a test command string.

    Args:
        test_command (str): A shell-style test command string.

    Returns:
        str | None: The first ``.py`` token found, or None when none is
            present or the command cannot be split.
    """
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

    Args:
        test_command (str): Existing test command whose ``.py`` path is the
            benchmark source for generation.
        candidate (dict): Kernel candidate dict passed to the generator.
        source_file (str): Path to the kernel source under optimization.
        out_dir (Path): Directory the generated harness is written to.
        kernel_repo (str): Repo root used to resolve imports for the harness.
        log_path (Path | None): Optional run log for generator diagnostics.

    Returns:
        str | None: A new ``test_command`` pointing at the generated harness,
            or None when generation is not possible or fails.
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


def _rocprof_roofline_enabled() -> bool:
    """Return whether rocprof roofline profiling is enabled.

    Returns:
        ``True`` unless ``HYPERLOOM_ROCPROF_ROOFLINE`` is set to a falsy value.
    """
    value = os.environ.get("HYPERLOOM_ROCPROF_ROOFLINE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _rocprof_timeout_sec() -> int:
    """Return the per-kernel rocprof roofline timeout in seconds.

    Returns:
        The value from ``HYPERLOOM_ROCPROF_ROOFLINE_TIMEOUT_SEC`` (floored at
        60s), or ``1800`` when unset or invalid.
    """
    try:
        return max(60, int(os.environ.get("HYPERLOOM_ROCPROF_ROOFLINE_TIMEOUT_SEC", "1800")))
    except ValueError:
        return 1800


def _rocprof_workdir(candidate: dict[str, Any], source_file: str, out_dir: Path) -> Path:
    """Choose a working directory for the rocprof profiling run.

    Args:
        candidate: Candidate metadata containing ``kernel_repo``.
        source_file: Path to the kernel source file.
        out_dir: Fallback directory when no candidate path resolves.

    Returns:
        The first existing directory derived from the candidate/source paths,
        else ``out_dir``.
    """
    for raw in (candidate.get("kernel_repo"), source_file):
        if not raw:
            continue
        path = Path(str(raw))
        if path.is_file():
            path = path.parent
        if path.is_dir():
            return path
    return out_dir


def _rocprof_profile_command(test_command: str) -> str:
    """Convert a harness correctness command into a profiling command.

    Args:
        test_command: Original harness test command.

    Returns:
        The command with a single ``--correctness`` swapped for ``--profile``
        when it targets a generated harness; otherwise the command unchanged.
    """
    if "--correctness" not in test_command:
        return test_command
    if "/unittest/harness_" not in test_command and " harness_" not in test_command:
        return test_command
    return test_command.replace("--correctness", "--profile", 1)


def _compact_rocprof_prompt(payload: dict[str, Any]) -> str:
    """Render a compact roofline-evidence addendum for the agent prompt.

    Args:
        payload: Structured rocprof roofline payload.

    Returns:
        A Markdown snippet summarizing up to three kernels' roofline signals,
        or an empty string when there are no results.
    """
    rows = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        return ""
    lines = [
        "",
        "## rocprof-compute Roofline Evidence",
        "",
        "Use these measured roofline signals to choose the kernel optimization direction.",
    ]
    for row in rows[:3]:
        if not isinstance(row, dict):
            continue
        roof = row.get("rocprof_roofline") or {}
        if not isinstance(roof, dict):
            roof = {}
        lines.extend([
            f"- Device kernel: {row.get('matched_kernel_name') or row.get('name') or 'unknown'}",
            f"  - Bound type: {roof.get('bound_type') or row.get('bottleneck') or 'unknown'}",
            f"  - Roofline efficiency: {roof.get('roofline_efficiency_pct')}",
            f"  - AI HBM: {roof.get('ai_hbm')}",
            f"  - Compute util pct: {roof.get('compute_utilization_pct')}",
            f"  - Bandwidth util pct: {roof.get('bandwidth_utilization_pct')}",
        ])
    return "\n".join(lines) + "\n"


def _run_rocprof_roofline(
    *,
    test_command: str,
    candidate: dict[str, Any],
    source_file: str,
    out_dir: Path,
    prompt_file: Path,
    log_path: Path | None,
) -> dict[str, Any]:
    """Run rocprof roofline before GEAK and append evidence to the prompt.

    Profiles the kernel via the ``rocprof_roofline.py`` helper, writes the
    JSON/text artifacts, and appends a compact evidence section to the prompt.

    Args:
        test_command: Harness command used to exercise the kernel.
        candidate: Candidate kernel metadata.
        source_file: Path to the kernel source file.
        out_dir: Output directory for this attempt.
        prompt_file: Prompt file to append roofline evidence to.
        log_path: Optional log file for progress messages.

    Returns:
        A status dict with the run outcome and artifact paths (``status`` is
        ``skipped``/``failed``/``ok``).
    """
    roof_dir = out_dir / "rocprof_roofline"
    roof_dir.mkdir(parents=True, exist_ok=True)
    out_json = roof_dir / "before.json"
    out_txt = roof_dir / "before.txt"
    if not _rocprof_roofline_enabled():
        return {"status": "skipped", "reason": "disabled_by_env"}
    if not test_command:
        return {"status": "skipped", "reason": "missing_test_command"}

    tool = Path(__file__).resolve().parent / "rocprof_roofline.py"
    workdir = _rocprof_workdir(candidate, source_file, out_dir)
    profiling_command = _rocprof_profile_command(test_command)
    target_kernel = str(candidate.get("name") or "").strip()
    cmd = [
        sys.executable,
        str(tool),
        "--workdir", str(workdir),
        "--cmd", profiling_command,
        "--out-json", str(out_json),
        "--out-txt", str(out_txt),
        "--timeout-sec", str(_rocprof_timeout_sec()),
    ]
    if target_kernel:
        cmd.extend(["--target-kernel", target_kernel])
    if log_path is not None:
        append_log(log_path, f"[rocprof_roofline] workdir={workdir}")
        append_log(log_path, "[rocprof_roofline] running before GEAK")
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=_rocprof_timeout_sec() + 30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        payload = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        atomic_write_json(out_json, payload)
        out_txt.write_text(payload["error"] + "\n", encoding="utf-8")
        return {"status": "failed", "json_path": str(out_json), "txt_path": str(out_txt), "error": payload["error"]}

    if log_path is not None and proc.stdout.strip():
        append_log(log_path, "[rocprof_roofline] " + proc.stdout.strip()[-1000:])
    try:
        payload = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception:
        payload = {"status": "failed", "error": proc.stdout.strip() or "missing rocprof roofline JSON"}
        atomic_write_json(out_json, payload)
    prompt_addendum = _compact_rocprof_prompt(payload)
    if prompt_addendum:
        with prompt_file.open("a", encoding="utf-8") as fh:
            fh.write(prompt_addendum)
    status = "ok" if proc.returncode == 0 and payload.get("status") == "ok" else payload.get("status", "failed")
    return {
        "status": status,
        "json_path": str(out_json),
        "txt_path": str(out_txt),
        "returncode": proc.returncode,
        "num_results": len(payload.get("results") or []) if isinstance(payload.get("results"), list) else 0,
    }


def _rocprof_kernel_matches(row: dict[str, Any], target_kernel: str) -> bool:
    """Return whether a roofline result row matches the target kernel.

    Args:
        row: Result row with ``matched_kernel_name`` / ``name`` fields.
        target_kernel: Kernel name to match; empty matches any row.

    Returns:
        ``True`` if the row corresponds to ``target_kernel``.
    """
    if not target_kernel:
        return True
    target = target_kernel.strip()
    names = (
        str(row.get("matched_kernel_name") or "").strip(),
        str(row.get("name") or "").strip(),
    )
    return any(name == target for name in names)


def _rocprof_sidecar_from_payload(payload: dict[str, Any], txt_path: str, json_path: str) -> dict[str, Any]:
    """Project a roofline payload into a sidecar record for one kernel.

    Args:
        payload: Structured rocprof roofline payload.
        txt_path: Path to the text report, recorded on the result.
        json_path: Path to the JSON report, recorded on the result.

    Returns:
        A roofline record for the matched (or first) kernel, or a
        skipped/failed status record when no match is found.
    """
    rows = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        return {
            "status": payload.get("status", "failed") if isinstance(payload, dict) else "failed",
            "report_path": txt_path,
            "json_path": json_path,
        }
    target_kernel = str(payload.get("target_kernel") or "").strip()
    first = None
    if target_kernel:
        for row in rows:
            if isinstance(row, dict) and _rocprof_kernel_matches(row, target_kernel):
                first = row
                break
        if first is None:
            return {
                "status": "skipped",
                "reason": "target_kernel_not_matched",
                "target_kernel": target_kernel,
                "matched_kernel_names": [
                    str(row.get("matched_kernel_name") or row.get("name") or "")
                    for row in rows
                    if isinstance(row, dict)
                ],
                "report_path": txt_path,
                "json_path": json_path,
            }
    else:
        first = rows[0] if isinstance(rows[0], dict) else {}
    roof = dict(first.get("rocprof_roofline") or {})
    roof.update({
        "status": first.get("status") or payload.get("status") or "matched",
        "matched_kernel_name": first.get("matched_kernel_name") or first.get("name"),
        "target_kernel": target_kernel,
        "report_path": txt_path,
        "json_path": json_path,
    })
    return roof


def _rocprof_phase_has_measurement(phase_data: dict[str, Any]) -> bool:
    """Return whether a roofline phase contains real numeric measurements.

    Args:
        phase_data: A before/after roofline phase record.

    Returns:
        ``True`` only when the phase did not fail/skip and carries at least one
        numeric roofline metric (metadata alone does not count).
    """
    status = str(phase_data.get("status") or "").lower()
    if status in {"failed", "skipped"}:
        return False
    # Only numeric roofline metrics count as real measurement. matched_kernel_name
    # is metadata and may be present even when rocprof produced no roofline values.
    measured_keys = (
        "roofline_efficiency_pct",
        "compute_utilization_pct",
        "bandwidth_utilization_pct",
        "ai_hbm",
        "perf_gflops",
        "hbm_actual_gbps",
    )
    return any(phase_data.get(key) not in (None, "") for key in measured_keys)


def _update_kernel_roofline_sidecar(
    *,
    workspace_path: str,
    kernel_id: str,
    rocprof_json_path: str,
    rocprof_txt_path: str,
    log_path: Path | None,
    rocprof_status: str = "",
    rocprof_reason: str = "",
    phase: str = "before_kernel_opt",
) -> None:
    """Mirror per-attempt rocprof artifacts into ``reports/kernel_roofline.json``.

    ``phase`` controls which sub-key is written:
      - ``"before_kernel_opt"``: pre-optimization snapshot (default, written by
        ``invoke_backend`` for every backend).
      - ``"after_kernel_opt"``: post-optimization snapshot (written after
        integrate succeeds).

    The outer ``rocprof_roofline`` dict is now::

        "rocprof_roofline": {
            "before_kernel_opt": {...},   # pre-opt rocprof
            "after_kernel_opt":  null,    # post-opt rocprof (null until available)
        }

    Even when ``_run_rocprof_roofline`` skipped (e.g. no ``test_command``)
    or failed, we still write a tagged entry so the dashboard can distinguish
    "considered but skipped/failed" from "not yet evaluated" (``null``).
    """
    sidecar_path = Path(workspace_path) / "reports" / "kernel_roofline.json"
    if not sidecar_path.is_file():
        return
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        if log_path is not None:
            append_log(log_path, f"[rocprof_roofline] sidecar update skipped: {exc}")
        return
    kernels = payload.get("kernels")
    if not isinstance(kernels, list):
        return

    phase_data: dict[str, Any]
    if rocprof_json_path and Path(rocprof_json_path).is_file():
        try:
            rocprof_payload = json.loads(Path(rocprof_json_path).read_text(encoding="utf-8"))
        except Exception as exc:
            if log_path is not None:
                append_log(log_path, f"[rocprof_roofline] sidecar payload load failed: {exc}")
            phase_data = {
                "status": rocprof_status or "failed",
                "reason": rocprof_reason or f"json_load_error: {type(exc).__name__}",
                "report_path": rocprof_txt_path or "",
                "json_path": rocprof_json_path or "",
            }
        else:
            phase_data = _rocprof_sidecar_from_payload(rocprof_payload, rocprof_txt_path, rocprof_json_path)
            if rocprof_status and rocprof_status != "ok":
                phase_data.setdefault("status", rocprof_status)
                if rocprof_reason:
                    phase_data.setdefault("reason", rocprof_reason)
    else:
        if not rocprof_status:
            return
        phase_data = {"status": rocprof_status}
        if rocprof_reason:
            phase_data["reason"] = rocprof_reason

    changed = False
    for row in kernels:
        if not isinstance(row, dict) or str(row.get("kernel_id") or "") != str(kernel_id):
            continue
        # Ensure the outer rocprof_roofline is a dict with both sub-keys
        outer = row.get("rocprof_roofline")
        if not isinstance(outer, dict):
            outer = {"before_kernel_opt": None, "after_kernel_opt": None}
        outer[phase] = phase_data
        row["rocprof_roofline"] = outer

        # Mirror key metrics from before_kernel_opt to the row level for
        # fast dashboard rendering.
        if phase == "before_kernel_opt":
            if phase_data.get("bound_type"):
                row["bottleneck"] = row.get("bottleneck") or phase_data["bound_type"]
                row["bound_type"] = row.get("bound_type") or phase_data["bound_type"]
            if phase_data.get("ai_hbm") is not None:
                row["arithmetic_intensity"] = row.get("arithmetic_intensity") or phase_data["ai_hbm"]
            if phase_data.get("roofline_efficiency_pct") is not None:
                row["efficiency_percent"] = phase_data["roofline_efficiency_pct"]
            if phase_data.get("compute_utilization_pct") is not None:
                row["compute_utilization_pct"] = phase_data["compute_utilization_pct"]
            if phase_data.get("bandwidth_utilization_pct") is not None:
                row["bandwidth_utilization_pct"] = phase_data["bandwidth_utilization_pct"]
        changed = True
    if changed:
        if _rocprof_phase_has_measurement(phase_data):
            source = str(payload.get("source") or "tracelens_analysis")
            if "rocprof_roofline" not in source:
                payload["source"] = f"{source}+rocprof_roofline" if source else "rocprof_roofline"
        atomic_write_json(sidecar_path, payload)
        if log_path is not None:
            append_log(log_path, f"[rocprof_roofline] updated {sidecar_path} [{phase}]")


def _apply_geak_env_overrides(
    args: argparse.Namespace,
    prompt_file: Path,
) -> dict[str, str | None]:
    """Temporarily tune GEAK env for this attempt; caller must restore.

    Sets ``GEAK_CONFIG`` (and disables the knowledge base when requested)
    for the duration of one attempt, returning the prior values so the
    caller can restore them via :func:`_restore_env`.

    Args:
        args (argparse.Namespace): Parsed CLI args (e.g. ``disable_xs_memory``).
        prompt_file (Path): Prompt file used to derive any per-run config.

    Returns:
        dict[str, str | None]: The previous values of the mutated env vars
            (None for vars that were previously unset).
    """
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
    """Restore environment variables captured by ``_apply_geak_env_overrides``.

    Args:
        previous (dict[str, str | None]): Map of env var name to its prior
            value; a None value means the var was unset and is removed.
    """
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _oob_output_dir(session_id: str, prompt_file: Path) -> Path:
    """Return the per-attempt output directory for an OOB run.

    Args:
        session_id: Session identifier for the run.
        prompt_file: Prompt file whose stem scopes the attempt directory.

    Returns:
        The created ``.../oob/<session_id>/<prompt_stem>`` directory.
    """
    # Per-attempt, mirroring _geak_output_dir. A session-level dir would let
    # concurrent OOB attempts share artifacts AND the per-attempt compile
    # caches (isolated_compile_cache_env keys off output_dir), reintroducing
    # the stale-lock / cache-clobber race this scoping is meant to avoid.
    out = _kernel_agent_root() / "oob" / session_id / prompt_file.stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def _mirror_path_link(run_dir: Path, mirror: Path) -> None:
    """Create a relative symlink inside the run dir pointing at the mirror.

    Best-effort: failures (unsupported filesystem, existing link) are
    swallowed so artifact mirroring never breaks a run.

    Args:
        run_dir (Path): The run directory the symlink is created under.
        mirror (Path): The target directory the symlink should point at.
    """
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
    """Best-effort `git checkout -- .` to undo rogue agent writes under the kernel repo. Idempotent."""
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

    Args:
        backend (str): Backend name to run (e.g. ``geak``, ``claude``).
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
    # GEAK needs more wall-clock than claude/codex; 130min default triggers GEAK's mode=full path.
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
    # GPU count: CLI override, then candidate hint, then 1.
    num_gpus = max(1, int(getattr(args, "num_gpus", 0) or 0)
                   or int(candidate.get("num_gpus_recommended") or 1))

    # ---------------------------------------------------------------------------
    # Common test-command + before_kernel_opt rocprof — runs for ALL backends
    # so every optimization attempt (GEAK, Claude, Codex, Cursor) gets the same
    # pre-optimization roofline snapshot without duplicating the logic.
    # ---------------------------------------------------------------------------
    cand_name = str(candidate.get("name") or "")
    is_multigpu_common = (
        bool(candidate.get("is_multigpu"))
        or kernel_name_implies_multigpu(cand_name)
    )
    # Derive a shared test_command that GEAK will use and rocprof will profile.
    # OOB backends don't accept --test-command but we still want the rocprof
    # snapshot, so we compute it here unconditionally.
    common_test_command = getattr(args, "test_command", "").strip()
    if not common_test_command:
        common_test_command = _render_geak_test_command(
            kernel_name=cand_name,
            bench_files=bench_files,
            is_multigpu=is_multigpu_common,
            num_gpus=num_gpus,
            timeout_sec=_profile_timeout_sec(),
        )
    # Shared temp out_dir for the before_kernel_opt rocprof artifact
    # (each backend will further scope its own out_dir below).
    _shared_out_dir = (
        _geak_output_dir(args.session_id, prompt_file)
        if backend == "geak"
        else _oob_output_dir(args.session_id, prompt_file)
    )
    if common_test_command:
        _harness_cmd = _try_generate_harness(
            common_test_command, candidate, source_file, _shared_out_dir,
            kernel_repo, log_path,
        )
        if _harness_cmd:
            common_test_command = _harness_cmd
    rocprof_before = {}
    if common_test_command:
        rocprof_before = _run_rocprof_roofline(
            test_command=common_test_command,
            candidate=candidate,
            source_file=source_file,
            out_dir=_shared_out_dir,
            prompt_file=prompt_file,
            log_path=log_path,
        )

    try:
        if backend == "geak":
            geak = _import_backend("geak_submit")
            out_dir = _geak_output_dir(args.session_id, prompt_file)
            # Use the common test_command (already derived + harness-patched above).
            test_command = common_test_command
            is_multigpu = is_multigpu_common
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
            if rocprof_before:
                result["rocprof_before_kernel_opt_status"] = str(rocprof_before.get("status") or "")
                if rocprof_before.get("reason"):
                    result["rocprof_before_kernel_opt_reason"] = str(rocprof_before["reason"])
                if rocprof_before.get("json_path"):
                    result["rocprof_before_kernel_opt_json"] = str(rocprof_before["json_path"])
                if rocprof_before.get("txt_path"):
                    result["rocprof_before_kernel_opt_txt"] = str(rocprof_before["txt_path"])
            # Surface GEAK partial outputs so a SIGTERM'd attempt with patches is still promoted to "partial".
            final_report = out_dir / "final_report.json"
            if final_report.is_file():
                result["geak_final_report"] = str(final_report)
            results_dir = out_dir / "results"
            if results_dir.is_dir():
                # Any *.patch under results/ is evidence of partial work.
                patches = sorted(results_dir.rglob("*.patch"))
                if patches:
                    result["geak_results_dir"] = str(results_dir)
                    result["geak_patch_count"] = len(patches)
                    result["geak_latest_patch"] = str(patches[-1])
                # Aggregate per-task best_results.json (max best_patch_speedup) so a real speedup
                # survives a SIGTERM before the top-level final_report.json (observed r38).
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
                        # Surface the worktree dir with the rewritten files, so the artifact
                        # extractor can recover a real .py source (not just the .patch diff).
                        wt = _geak_best_worktree(best_patch_path)
                        if wt:
                            result["geak_per_task_best_worktree"] = str(wt)
            return result
        if backend in {"claude", "codex", "cursor"}:
            oob = _import_backend("oob_submit")
            out_dir = _oob_output_dir(args.session_id, prompt_file)
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
            if common_test_command:
                result["test_command"] = common_test_command
            if rocprof_before:
                result["rocprof_before_kernel_opt_status"] = str(rocprof_before.get("status") or "")
                if rocprof_before.get("reason"):
                    result["rocprof_before_kernel_opt_reason"] = str(rocprof_before["reason"])
                if rocprof_before.get("json_path"):
                    result["rocprof_before_kernel_opt_json"] = str(rocprof_before["json_path"])
                if rocprof_before.get("txt_path"):
                    result["rocprof_before_kernel_opt_txt"] = str(rocprof_before["txt_path"])
            return result
        if backend == "forge":
            # Kernel-Forge autonomous-loop backend. Runs entirely inside a git
            # worktree of kernel_repo (never mutates the live repo) and emits the
            # same artifacts as OOB (optimized_versions/ + optimization_report.md),
            # so the downstream verify/propose/integrate path is unchanged.
            forge = _import_backend("forge_submit")
            out_dir = _oob_output_dir(args.session_id, prompt_file)
            result = forge.submit(
                source_file=source_file,
                prompt_file=prompt_file,
                output_dir=out_dir,
                test_command=common_test_command,
                source_type=str((candidate or {}).get("source_type") or "unknown"),
                candidate=candidate or {},
                num_gpus=num_gpus,
                timeout_s=timeout_s,
                prefer_ray=prefer_ray,
                kernel_repo=kernel_repo,
            )
            result["output_dir"] = str(out_dir)
            if common_test_command:
                result["test_command"] = common_test_command
            if rocprof_before:
                result["rocprof_before_kernel_opt_status"] = str(rocprof_before.get("status") or "")
                if rocprof_before.get("reason"):
                    result["rocprof_before_kernel_opt_reason"] = str(rocprof_before["reason"])
                if rocprof_before.get("json_path"):
                    result["rocprof_before_kernel_opt_json"] = str(rocprof_before["json_path"])
                if rocprof_before.get("txt_path"):
                    result["rocprof_before_kernel_opt_txt"] = str(rocprof_before["txt_path"])
            return result
        return {
            "returncode": 2,
            "stdout_tail": f"unknown backend: {backend}",
            "stderr_tail": "", "stdout": "", "gpu_ids": "",
            "elapsed_s": 0.0, "cmd": [],
        }
    finally:
        # Always undo rogue writes under the kernel repo, regardless of exit code.
        # Skip for forge: it manages its own restore (per-file write-back on a
        # temp branch); a blanket `git checkout -- .` here would overwrite the
        # dirty-file state that forge just carefully restored.
        if log_path is not None and backend != "forge":
            _git_checkout_fallback(kernel_repo, log_path)


def env_first(*names: str) -> str:
    """Return the first non-empty environment variable among ``names``.

    Args:
        *names (str): Environment variable names to check, in priority order.

    Returns:
        str: The value of the first set, non-empty variable, or an empty
            string when none are set.
    """
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
    # Dry-run emits a synthetic source-suffixed placeholder (back-compat); real runs capture raw
    # stdout to a `.log` (not a `.cu`) so _extract_source_block scans for fenced code rather than
    # false-positiving the conversation log as kernel source. Consumers read attempt["optimized_path"]
    # or glob <attempt_id>* under runs/<sid>/optimized/ (see kernel-agent/SKILL.md).
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
        # Always materialise the stdout `.log` (audit trail + code-fence extraction fallback).
        if full_stdout.strip():
            optimized_path.write_text(full_stdout, encoding="utf-8")
        append_log(log_path, stdout_tail)

    elapsed = round(time.time() - started, 3)

    backend_paths: dict[str, str] = {}
    if not args.dry_run:
        out_dir = result.get("output_dir") if isinstance(result, dict) else ""
        if out_dir:
            backend_paths["output_dir"] = out_dir
            # Use the workspace path from oob's init event (mtime heuristic mis-attributed concurrent replicas).
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
                    # Scan for partial outputs even on returncode != 0.
                    files = sorted(opt_dir.iterdir(), key=lambda p: p.stat().st_mtime)
                    if files:
                        backend_paths["partial_optimized_count"] = str(len(files))
                        backend_paths["partial_latest_optimized"] = str(files[-1])
                report = Path(cli_workspace) / "optimization_report.md"
                if report.exists():
                    backend_paths["partial_report"] = str(report)
            # /home/user/ rescue: claude sometimes writes to ~/optimized_versions/ instead of the
            # workspace; surface fresh files there when the workspace's optimized_versions/ is empty.
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
            rocprof_json = (result.get("rocprof_before_kernel_opt_json") or "") if isinstance(result, dict) else ""
            rocprof_txt = (result.get("rocprof_before_kernel_opt_txt") or "") if isinstance(result, dict) else ""
            rocprof_status = (result.get("rocprof_before_kernel_opt_status") or "") if isinstance(result, dict) else ""
            rocprof_reason = (result.get("rocprof_before_kernel_opt_reason") or "") if isinstance(result, dict) else ""
            if rocprof_status:
                backend_paths["rocprof_before_kernel_opt_status"] = rocprof_status
            if rocprof_reason:
                backend_paths["rocprof_before_kernel_opt_reason"] = rocprof_reason
            if rocprof_json:
                backend_paths["rocprof_before_kernel_opt_json"] = rocprof_json
            if rocprof_txt:
                backend_paths["rocprof_before_kernel_opt_txt"] = rocprof_txt
            # Mirror status/reason into the dashboard sidecar even without a JSON artifact so the row distinguishes considered/skipped/failed from not-yet-evaluated (null).
            if rocprof_status:
                _update_kernel_roofline_sidecar(
                    workspace_path=str(getattr(args, "workspace_path", "")),
                    kernel_id=str(candidate.get("kernel_id") or args.kernel_id),
                    rocprof_json_path=rocprof_json,
                    rocprof_txt_path=rocprof_txt,
                    log_path=log_path,
                    rocprof_status=rocprof_status,
                    rocprof_reason=rocprof_reason,
                )
            # GEAK partial-output surface (forwarded by invoke_backend): final_report.json / patches.
            geak_final = (result.get("geak_final_report") or "") if isinstance(result, dict) else ""
            if geak_final:
                backend_paths["geak_final_report"] = geak_final
            geak_patch = (result.get("geak_latest_patch") or "") if isinstance(result, dict) else ""
            if geak_patch:
                backend_paths["geak_latest_patch"] = geak_patch
                backend_paths["geak_patch_count"] = str(result.get("geak_patch_count") or 0)
            # Per-task best speedup salvage (when select_patch didn't finish before SIGTERM).
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
                    # Forward the worktree dir so artifact recovery reads the rewritten file, not the diff.
                    backend_paths["geak_per_task_best_worktree"] = str(wt)
            # Promote a timed-out / failed attempt with on-disk artifacts to "partial".
            # EXCEPTION: refuse promotion on a persistent inner-LLM auth loop (>= _AUTH_RETRY_THRESHOLD),
            # which leaves an empty optimized_versions/ that would falsely ship PARTIAL.
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
                # Force a non-partial terminal state so build_verification excludes it and make_proposal REVERTs.
                if status == "timeout":
                    status = "failed"
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


# Persistent inner-LLM auth failure markers. >= AUTH_RETRY_THRESHOLD matches => credential
# dead-end; refuse to promote timeout/failed to partial (an auth loop leaves an empty
# optimized_versions/ that would falsely ship PARTIAL the orchestrator never retires).
_AUTH_FAILURE_PATTERNS = [
    re.compile(r"\b401\b[^\n]{0,80}(unauthor|forbidden|client\s*error)", re.IGNORECASE),
    re.compile(r"HTTP/\d\.\d\s+401\b"),
    re.compile(r"Authentication\s*Error|Invalid\s*API\s*Key|invalid[._]api[._]key", re.IGNORECASE),
    re.compile(r"Subscription[- ]Key[^\n]{0,80}(missing|invalid|not\s*present)", re.IGNORECASE),
    re.compile(r"Primus\.00009\s+token\s+not\s+present", re.IGNORECASE),
]
_AUTH_RETRY_THRESHOLD = 3


def _count_auth_failures(text: str) -> int:
    """Count distinct inner-LLM auth-failure markers in *text* (distinguishes a transient 401 from an unrecoverable loop)."""
    if not text:
        return 0
    total = 0
    for pat in _AUTH_FAILURE_PATTERNS:
        total += sum(1 for _ in pat.finditer(text))
    return total


def _extract_speedup_from_report(report_path: str | Path) -> float | None:
    """Best-effort scan of an OOB optimization_report.md for a speedup figure (median-of-top-3; None if absent)."""
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
                # Reject obvious junk (e.g. "100x faster").
                if 0.3 <= v <= 50.0:
                    found.append(v)
            except ValueError:
                continue
    if not found:
        return None
    # Median-of-top-3 to dodge cherry-picked best-shape numbers.
    found.sort(reverse=True)
    top = found[:3]
    return round(sum(top) / len(top), 4)


def _extract_speedup_from_geak(final_report_path: str | Path) -> float | None:
    """Pull best_speedup from a GEAK final_report.json if present and >0.

    Args:
        final_report_path (str | Path): Path to a GEAK ``final_report.json``.

    Returns:
        float | None: The reported ``best_speedup`` when positive, else None
            (also None when the file is missing or unparseable).
    """
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
    """Best-effort correctness signal from backend markdown/json reports.

    Looks for an explicit ``[correctness] pass/fail`` marker first, then
    falls back to scanning for known pass/fail phrasings.

    Args:
        report_path (str | Path): Path to a backend report (markdown/text).

    Returns:
        bool | None: True/False when a correctness signal is found, or None
            when the file is missing/unreadable or no signal is present.
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
    """Treat GEAK ``status=complete`` + measured speedup as sufficient correctness evidence (default ON).

    GEAK's save_and_test only checks compile + import, not numerical output, so without this
    every GEAK KEEP would degrade to NEEDS_REVIEW; integrate's E2E magpie benchmark is the
    ground-truth check. Set ``HYPERLOOM_TRUST_GEAK_CORRECTNESS=0`` to restore the conservative behaviour.
    """
    raw = os.environ.get("HYPERLOOM_TRUST_GEAK_CORRECTNESS", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def _extract_correctness_from_geak(final_report_path: str | Path) -> bool | None:
    """Read correctness from GEAK-style JSON reports when present.

    Recursively walks the JSON looking for correctness/validity keys; any
    False seen wins over True (fail-safe).

    Args:
        final_report_path (str | Path): Path to a GEAK-style JSON report.

    Returns:
        bool | None: False if any correctness key is falsy, True if only
            truthy ones are found, or None when nothing relevant is present.
    """
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
        """Recursively collect correctness/validity booleans from ``obj``.

        Args:
            obj (Any): A JSON value (dict, list, or scalar) to traverse;
                matching keys append their resolved bool to ``found``.
        """
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
    """Heuristically decide whether ``text`` is a complete source file.

    For ``.py`` it must parse and contain Python markers; for C/C++/CUDA
    suffixes it must contain typical source markers. Text containing a code
    fence is rejected (handled by :func:`_extract_source_block` instead).

    Args:
        text (str): Candidate file contents.
        suffix (str): File suffix that selects the language heuristic.

    Returns:
        bool: True when ``text`` plausibly is a complete source file for the
            given suffix; False otherwise.
    """
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


def _geak_best_worktree(best_patch_path: str) -> Path | None:
    """Map a GEAK best-patch file path to the ``worktrees/slot_<M>`` it edited (shares ``parallel_<M>``'s suffix).

    Lets callers pick up the real source file instead of scraping a diff; ``None`` on layout mismatch.
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
    """Return existing files under ``worktree`` mirroring ``source_file``.

    Tries source_file relative to kernel_repo first, then a basename rglob within
    the worktree. Empty list when nothing matches.
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
    """Collect candidate optimized-artifact paths for an attempt, in priority.

    Gathers GEAK worktree files, partial/patch outputs, ``optimized_versions``
    directories, and the recorded ``optimized_path`` into a priority-ordered
    list for downstream source selection.

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
    # GEAK worktree files first (ground-truth edited source), before .patch candidates.
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
    measured = False
    # Prefer the highest extracted speedup; else fall back to the first usable attempt.
    for a in usable:
        bp = a.get("backend_paths") or {}
        report = bp.get("partial_report") or bp.get("report") or ""
        sp = _extract_speedup_from_report(report)
        if sp is None:
            sp = _extract_speedup_from_geak(bp.get("geak_final_report", ""))
        # Fallback to the per-task best speedup (aggregated by invoke_backend) when final_report.json is absent.
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
        # kernel_repo lets worktree recovery map an absolute source path to GEAK's edited relative path.
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
    # PR-E (default ON): trust GEAK status=complete + measured speedup as correctness=True
    # for import-only harnesses (HYPERLOOM_TRUST_GEAK_CORRECTNESS=0 to disable). See _trust_geak_correctness.
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
        # compile_passed == bool(best); use artifact_error to distinguish a real compile
        # failure from a backend-dispatch failure (no usable attempt to compile from).
        err = (verification.get("artifact_error") or "").strip()
        if err and verification.get("best_attempt_id", "") == "":
            return {"decision": "REVERT",
                    "reasons": [f"backend dispatch failed: {err}"]}
        return {"decision": "REVERT", "reasons": ["compile failed"]}
    if not verification["correctness_passed"]:
        reasons.append("correctness evidence missing or failed")
    if not verification.get("artifact_valid"):
        reasons.append("optimized source artifact missing or invalid")
    # default_unmeasured (no speedup found) => PARTIAL, not REVERT (don't punish unmeasured as a regression).
    src = verification.get("micro_speedup_source", "default_unmeasured")
    if src == "default_unmeasured":
        reasons.append("no measurable speedup found in any backend report")
        return {"decision": "PARTIAL", "reasons": reasons}
    if verification["micro_speedup"] <= 1.0:
        return {"decision": "REVERT", "reasons": ["microbench did not improve"]}
    # 1.05x KEEP threshold (issue #442); below is routed to NEEDS_REVIEW.
    KEEP_THRESHOLD = 1.05
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
    """CLI entry point for the kernel-optimization tool.

    Parses command-line arguments, runs the configured backend ladder for
    the requested kernel, builds the verification result and proposal, and
    persists status/artifacts.

    Returns:
        int: Process exit code (0 on success, non-zero on failure).
    """
    parser = argparse.ArgumentParser(description="Kernel Agent optimization tool")
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
    parser.add_argument("--backends", default="")
    parser.add_argument("--benchmark-file", default="")
    parser.add_argument("--test-harness-path", default="")
    parser.add_argument("--source-file", default="")
    parser.add_argument("--target-platform", default=_env_target_platform())
    parser.add_argument("--extra-sglang-args", default="")
    parser.add_argument("--budget-minutes", type=float, default=60.0,
                        help="Per-attempt wall-clock budget for claude/codex "
                             "OOB backends. GEAK uses --geak-budget-min.")
    # Default tracks $GEAK_RUN_MODE: quick -> 70 min, full -> 130 min.
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
    # GEAK cost limit: yaml cost_limit:0. (unlimited) isn't honoured by the sub-agent path
    # (falls back to $3.0); the only working lever is GEAK's -l/--cost-limit CLI option.
    # Default 0.0 to match GEAK's geak.yaml; pin a cap via $HYPERLOOM_GEAK_COST_LIMIT / --geak-cost-limit.
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
        candidates_path = (
            Path(args.candidates_path)
            if args.candidates_path
            else resolve_candidates_path(run_dir)
        )
        all_candidates = load_candidates(candidates_path)
        candidate = find_candidate(all_candidates, args.kernel_id)
        if candidate is None:
            # kernel_id matches no candidate (hallucinated id); skip cleanly instead of crashing.
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
        # TraceLens is source of truth: _resolve_source_file overrides a disagreeing LLM path.
        resolved_source = _resolve_source_file(
            args.source_file, candidate, args.kernel_id, log_path
        )
        args.source_file = resolved_source
        # Forward the candidate's repo root so worktree artifact recovery can map source_file to GEAK's relative path.
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
