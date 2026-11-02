# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Splitting an oversized trace into chunks, and choosing the chunk the run analyses."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from hyperloom.inference_optimizer.session.paths import workspace_root

from ._io_utils import append_log
from ._trace_gates import _check_chunk_quality
from ._tracelens_checkout import run_command


def _default_workspace_path() -> str:
    """Resolve the default workspace root for ``--workspace-path``.

    Fallback order: ``$USER_DATA_PATH``, then legacy ``$WORKSPACE_PATH``, then
    ``session.paths.workspace_root()`` (which warns once when ``$USER_DATA_PATH`` is unset).

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
    return str(workspace_root())


def _build_split_cmd(
    args,
    split_input_path: "Path",
    split_dir: "Path",
    split_num_steps: int,
    log_path: "Path",
) -> "list[str]":
    """Assemble the TraceLens splitter CLI argv."""
    split_cmd = [
        sys.executable,
        "-m",
        "TraceLens.TraceUtils.split_trace.main",
        str(split_input_path),
        "-o",
        str(split_dir),
        "--find-steady-state",
        "--num-steps",
        str(split_num_steps),
    ]
    if args.split_llm_inference:
        split_cmd += ["--llm-inference"]
        conc = str(args.split_conc).strip()
        if conc:
            split_cmd += ["--CONC", conc]
        osl = str(args.split_osl).strip()
        if osl:
            split_cmd += ["--OSL", osl]
        r_str = str(args.split_r).strip()
        if r_str:
            try:
                float(r_str)
            except ValueError:
                append_log(log_path, f"split_trace: ignoring non-numeric --R={r_str!r}")
            else:
                split_cmd += ["--R", r_str]
    return split_cmd


def _collect_split_chunks(
    split_dir: "Path",
    steady_state_mode: str,
    analysis_trace_path: "Path",
    health_warnings: "list[dict]",
) -> "tuple[Path, dict, dict]":
    """Collect splitter output chunks and select the requested mode."""

    def _collect(pattern: str) -> "list[Path]":
        out: list[Path] = []
        for ext in ("trace.json.gz", "json.gz", "trace.json", "json"):
            out.extend(sorted(split_dir.rglob(f"{pattern}.{ext}")))
        return out

    mixed_chunks = _collect("mixed_steady_state_*")
    decode_chunks = _collect("decode_only_steady_state_*")
    prefill_chunks = _collect("prefilldecode_steady_state_*")
    generic_chunks = _collect("steady_state_*")

    split_meta = {
        "chunks_by_mode": {
            "mixed": len(mixed_chunks),
            "decode_only": len(decode_chunks),
            "prefilldecode": len(prefill_chunks),
            "generic": len(generic_chunks),
        },
        "chunks_extracted": len(mixed_chunks) + len(decode_chunks) + len(prefill_chunks) + len(generic_chunks),
    }

    if not (mixed_chunks or decode_chunks or prefill_chunks or generic_chunks):
        health_warnings.append(
            {
                "code": "trace_split_no_steady_state",
                "severity": "blocking",
                "trace_input": str(analysis_trace_path),
                "split_dir": str(split_dir),
                "chunks_by_mode": split_meta["chunks_by_mode"],
                "message": "TraceLens splitter produced no steady-state chunks; refusing to analyze the raw trace.",
            }
        )
        raise RuntimeError(
            "trace_split_no_steady_state: TraceLens splitter "
            "produced no steady-state chunks; refusing to run "
            "TraceLens analysis on the raw trace"
        )

    mode_to_chunks = {
        "mixed": ("mixed_steady_state", mixed_chunks),
        "decode_only": ("decode_only_steady_state", decode_chunks),
        "prefilldecode": ("prefilldecode_steady_state", prefill_chunks),
        "generic": ("steady_state", generic_chunks),
    }
    chunk_label, selected_chunks = mode_to_chunks[steady_state_mode]
    if not selected_chunks:
        available = [m for m, (_, ch) in mode_to_chunks.items() if ch]
        warning = {
            "code": "steady_state_chunk_missing",
            "severity": "blocking",
            "requested_mode": steady_state_mode,
            "available_modes": available,
            "remediation": (
                "Re-issue roofline with env "
                "INFERENCE_OPTIMIZER_STEADY_STATE_MODE set to one of "
                f"{available} (or pass --steady-state-mode directly)."
            ),
            "trace_input": str(analysis_trace_path),
            "split_dir": str(split_dir),
        }
        health_warnings.append(warning)
        raise RuntimeError(
            f"steady_state_chunk_missing: requested "
            f"--steady-state-mode={steady_state_mode} but "
            f"splitter produced no matching chunk under {split_dir}"
        )

    selected_chunk = selected_chunks[0]
    split_meta["selected_mode"] = steady_state_mode
    split_meta["chunk_label"] = chunk_label
    split_meta["selected_chunk"] = str(selected_chunk)
    split_meta["selected_chunk_count"] = len(selected_chunks)
    split_meta["available_modes"] = [m for m, (_, ch) in mode_to_chunks.items() if ch]
    return selected_chunk, split_meta, mode_to_chunks


def _validate_selected_chunk(
    split_dir: "Path",
    selected_chunk: "Path",
    mode: str,
    mode_to_chunks: "dict[str, tuple[str, list[Path]]]",
    health_warnings: "list[dict]",
) -> None:
    """Run the low-quality busy-ratio gate on the selected chunk.

    Appends a structured warning dict to ``health_warnings`` before raising so
    the retry consumer in the roofline executor can find it even when the
    exception propagates past the caller.
    """
    low_quality_warning = _check_chunk_quality(
        split_dir=split_dir,
        selected_chunk=selected_chunk,
        mode=mode,
        available_modes=mode_to_chunks,
    )
    if low_quality_warning is not None:
        health_warnings.append(low_quality_warning)
        raise RuntimeError(
            f"steady_state_chunk_low_quality: requested "
            f"--steady-state-mode={mode} chunk "
            f"busy_ratio={low_quality_warning['busy_ratio'] * 100:.3f}%; "
            f"better alternates: {low_quality_warning['non_empty_modes']}"
        )


@dataclass(frozen=True)
class TraceSplitJob:
    """The input trace, output directory and TraceLens checkout one splitter run works with."""

    analysis_trace_path: Path
    split_dir: Path
    tl_root: Path


def _run_trace_split(
    args,
    job: TraceSplitJob,
    log_path: "Path",
    health_warnings: "list[dict]",
) -> "tuple[Path, dict]":
    """Run the TraceLens splitter and return the selected chunk.

    Builds the splitter command, runs it, collects chunks, and validates
    the selected chunk. This is the complete splitting pipeline extracted
    from ``main()`` for testability.

    ``health_warnings`` is mutated in place: structured warning dicts are
    appended before any RuntimeError is raised so the caller's try/except
    can include them in the result even on failure.

    Returns:
        ``(selected_chunk, split_meta)``

    Raises:
        RuntimeError: On splitter failure, no chunks, missing mode, or
            low-quality chunk with a better alternate.
    """
    analysis_trace_path = job.analysis_trace_path
    split_dir = job.split_dir
    split_dir.mkdir(parents=True, exist_ok=True)
    split_num_steps = max(8, int(args.split_num_steps or 32))
    split_input_path = analysis_trace_path
    split_cmd = _build_split_cmd(
        args,
        split_input_path,
        split_dir,
        split_num_steps,
        log_path,
    )
    split_rc = run_command(
        split_cmd,
        cwd=job.tl_root,
        log_path=log_path,
        timeout_s=max(60, int(getattr(args, "budget_minutes", 60) * 60)),
    )
    if split_rc != 0:
        raise RuntimeError(
            f"trace_split_failed: TraceLens splitter exited with code {split_rc}; see {log_path} for subprocess output."
        )

    selected_chunk, split_meta, mode_to_chunks = _collect_split_chunks(
        split_dir,
        args.steady_state_mode,
        analysis_trace_path,
        health_warnings,
    )
    split_meta.update(
        {
            "split_input": str(split_input_path),
            "split_dir": str(split_dir),
            "num_steps": split_num_steps,
            "returncode": split_rc,
        }
    )

    _validate_selected_chunk(
        split_dir,
        selected_chunk,
        args.steady_state_mode,
        mode_to_chunks,
        health_warnings,
    )
    return selected_chunk, split_meta
