# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``combine_by_node`` multi-trace TraceLens run config of a PD profile, one job per server node."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from TraceLens.Agent.Analysis.utils.config_utils import SCOPE_MULTI_TRACE
from TraceLens.Reporting.aggregate_perf_reports import POLICY_BY_NODE

from .tracelens_skill_runner import write_checked_config

_TRACE_TP_RANK_RE = re.compile(r"-TP-(\d+)(?:\D|$)")
_CAPTURE_RANK_RE = re.compile(r"_rank(\d+)\.json(?:\.gz)?$")


def _rank_scoped_capture_folder(output_dir: Path, trace_path: Path, capture_folder: Path, label: str) -> Path:
    """Narrow ``capture_folder`` to the graph captures of the TP rank an SGLang trace names (``-TP-<n>``).

    TraceLens picks a replay's captures by the ``rank_<n>`` of its file name only, so a ``-TP-<n>`` trace is matched
    against every rank's captures. The narrowed folder is ``output_dir/cache/<label>_rank<n>``, symlinks to that rank's
    capture files.

    Args:
        output_dir: The TraceLens output directory the narrowed folder is created under.
        trace_path: The replay trace the captures are merged into.
        capture_folder: The graph-capture folder of the trace's engine.
        label: Names the narrowed folder, distinct per job of one config.

    Returns:
        Path: The narrowed folder; ``capture_folder`` itself when the trace names no TP rank, or the folder holds
            captures of fewer than two ranks or none of the trace's.
    """
    match = _TRACE_TP_RANK_RE.search(trace_path.name)
    if match is None or not capture_folder.is_dir():
        return capture_folder
    by_rank: dict[int, list[Path]] = {}
    for entry in capture_folder.iterdir():
        rank_match = _CAPTURE_RANK_RE.search(entry.name)
        if rank_match and entry.is_file():
            by_rank.setdefault(int(rank_match.group(1)), []).append(entry)
    rank = int(match.group(1))
    if len(by_rank) < 2 or rank not in by_rank:
        return capture_folder
    scoped = output_dir / "cache" / f"{label}_rank{rank}"
    if scoped.exists():
        shutil.rmtree(scoped)
    scoped.mkdir(parents=True)
    for entry in by_rank[rank]:
        (scoped / entry.name).symlink_to(entry.resolve())
    return scoped


def write_multi_trace_run_config(*, output_dir: Path, jobs: list[dict[str, Any]], analysis_mode: str) -> Path:
    """Write the ``combine_by_node`` multi-trace run config for ``jobs``, the first job being the focus.

    Checked with TraceLens's ``check_config`` under the ``multi_trace`` scope; job paths are made absolute. A rejected
    config raises ``ConfigError``.
    """
    resolved = []
    for index, job in enumerate(jobs):
        entry = {**job, "trace_path": str(Path(job["trace_path"]).resolve())}
        if job.get("capture_folder"):
            scoped = _rank_scoped_capture_folder(
                output_dir, Path(job["trace_path"]), Path(job["capture_folder"]).resolve(), f"capture_{index}"
            )
            entry["capture_folder"] = str(scoped.resolve())
        resolved.append(entry)
    config = {
        "analysis_mode": analysis_mode,
        "output_dir": str(output_dir.resolve()),
        "aggregation": {"policy": POLICY_BY_NODE},
        "jobs": resolved,
    }
    return write_checked_config(config, SCOPE_MULTI_TRACE, output_dir)
