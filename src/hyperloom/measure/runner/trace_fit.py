# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fit a profile test's traces under the result's size cap before the output is inventoried.

A profile of a large model at TP4 can write more trace than the result may carry (``max_total_bytes``, 4 GiB by
default), and the whole result was then refused as ``output_too_large`` after the test had held its cards for the whole
run. Before the inventory, the traces are made to fit, in two steps, each only when the output is still over the cap:

1. every trace written uncompressed is gzipped (the trace analysis reads ``.json.gz`` as it reads ``.json``);
2. only the traces of the primary rank (rank 0 of each parallel dimension a file name states) are kept: the steady-state
   window every rank records is the same program, and the analysis reads one rank.

What was done is recorded in ``trace_fit.json`` beside the traces' directory and in the result's notes. An output still
over the cap after both is left to the inventory, which refuses it.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
from pathlib import Path

from hyperloom.measure.profile_traces import PROFILE_DIR, is_trace_file

__all__ = ["TRACE_FIT_FILE", "fit_traces"]

#: The record of what was done to fit the traces, below the output directory (not a trace: its name ends otherwise).
TRACE_FIT_FILE = "trace_fit.record"
_GZIP_SUFFIX = ".gz"
#: A rank a trace file's name states: sglang names them ``<id>-TP-<rank>[-DP-<rank>][-EP-<rank>]...``.
_RANK = re.compile(r"-(?:TP|DP|EP|PP|RANK)-(\d+)", re.IGNORECASE)


def fit_traces(files_dir: Path, max_total_bytes: int) -> str | None:
    """Fit the traces below ``files_dir/profile`` so the output is at most ``max_total_bytes``, as far as compressing them
    and keeping the primary rank's can; a note of what was done, ``None`` when nothing was needed."""
    profile = Path(files_dir) / PROFILE_DIR
    before = _total_bytes(files_dir)
    if before <= max_total_bytes or not profile.is_dir() or profile.is_symlink():
        return None
    record: dict[str, object] = {"max_total_bytes": max_total_bytes, "bytes_before": before}
    record["compressed"] = _compress(profile)
    if _total_bytes(files_dir) > max_total_bytes:
        record["dropped_secondary_ranks"] = _drop_secondary_ranks(profile)
    after = _total_bytes(files_dir)
    record["bytes_after"] = after
    (Path(files_dir) / TRACE_FIT_FILE).write_text(json.dumps(record, sort_keys=True, indent=1), encoding="utf-8")
    dropped = len(record.get("dropped_secondary_ranks", []))
    fits = "fits" if after <= max_total_bytes else "still does not fit"
    return (
        f"the profile's output was {before} bytes, over the cap of {max_total_bytes}: {len(record['compressed'])} "
        f"trace(s) compressed, {dropped} trace(s) of secondary ranks dropped, {after} bytes now ({fits}; {TRACE_FIT_FILE})"
    )


def _regular_files(root: Path) -> list[Path]:
    """Every regular file below ``root``, links neither followed nor counted (the inventory refuses them)."""
    found = []
    for current, _, names in os.walk(root):
        for name in names:
            path = Path(current, name)
            if path.is_file() and not path.is_symlink():
                found.append(path)
    return found


def _total_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in _regular_files(Path(root)))


def _compress(profile: Path) -> list[str]:
    """Gzip every uncompressed trace below ``profile`` in place; the relative paths of those compressed."""
    done = []
    for path in sorted(_regular_files(profile)):
        if not is_trace_file(path.name) or path.name.endswith(_GZIP_SUFFIX):
            continue
        target = path.with_name(path.name + _GZIP_SUFFIX)
        with path.open("rb") as source, gzip.open(target, "wb", compresslevel=6) as sink:
            shutil.copyfileobj(source, sink, length=1 << 20)
        path.unlink()
        done.append(str(path.relative_to(profile)))
    return done


def _is_primary(name: str) -> bool:
    """Whether a trace file is of rank 0 in every parallel dimension its name states (a name that states none is)."""
    return all(int(rank) == 0 for rank in _RANK.findall(name))


def _drop_secondary_ranks(profile: Path) -> list[str]:
    """Remove the traces below ``profile`` that are not the primary rank's, when one that is remains; those removed."""
    traces = [path for path in _regular_files(profile) if is_trace_file(path.name)]
    if not any(_is_primary(path.name) for path in traces):
        return []
    dropped = []
    for path in sorted(traces):
        if not _is_primary(path.name):
            path.unlink()
            dropped.append(str(path.relative_to(profile)))
    return dropped
