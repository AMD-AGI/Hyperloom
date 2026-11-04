# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A profile's traces over the result's size cap are compressed, then cut to the primary rank's, before the inventory."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from hyperloom.measure.runner.inventory import OutputRejected, inventory
from hyperloom.measure.runner.trace_fit import TRACE_FIT_FILE, fit_traces

_RANKS = 4
_EVENT = b'{"ph": "X", "name": "gemm_a8w8_blockscale_bpreshuffle", "ts": 1, "dur": 2},\n'


def _traces(files: Path, *, gzipped: bool, size: int) -> list[Path]:
    """One trace per TP rank below ``files/profile/torch_trace`` plus a rank-0 capture sidecar, ``size`` bytes each."""
    made = []
    for name in [f"run-TP-{rank}-EP-{rank}.trace.json" for rank in range(_RANKS)] + [
        "capture_traces/bs_1-TP-0.trace.json"
    ]:
        path = files / "profile" / "torch_trace" / (name + (".gz" if gzipped else ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(os.urandom(size) if gzipped else _EVENT * (size // len(_EVENT)))
        made.append(path)
    return made


def _names(files: Path) -> set[str]:
    return {str(p.relative_to(files / "profile")) for p in (files / "profile").rglob("*") if p.is_file()}


def test_an_output_within_the_cap_is_left_as_it_is(tmp_path: Path) -> None:
    _traces(tmp_path, gzipped=False, size=10_000)
    before = _names(tmp_path)

    assert fit_traces(tmp_path, 10**9) is None
    assert _names(tmp_path) == before and not (tmp_path / TRACE_FIT_FILE).exists()


def test_uncompressed_traces_are_gzipped_first_and_kept_whole_when_that_fits(tmp_path: Path) -> None:
    traces = _traces(tmp_path, gzipped=False, size=200_000)
    cap = 500_000  # 1 MB of text, a few kB once compressed

    note = fit_traces(tmp_path, cap)

    assert note is not None and "5 trace(s) compressed, 0 trace(s) of secondary ranks dropped" in note
    assert _names(tmp_path) == {str(p.relative_to(tmp_path / "profile")) + ".gz" for p in traces}
    with gzip.open(traces[1].with_name(traces[1].name + ".gz"), "rb") as handle:
        assert handle.read(len(_EVENT)) == _EVENT
    record = json.loads((tmp_path / TRACE_FIT_FILE).read_text(encoding="utf-8"))
    assert record["bytes_after"] <= cap < record["bytes_before"] and "dropped_secondary_ranks" not in record
    inventory(tmp_path, max_total_bytes=cap)


def test_traces_that_do_not_compress_keep_only_the_primary_ranks(tmp_path: Path) -> None:
    _traces(tmp_path, gzipped=True, size=100_000)
    cap = 250_000  # five incompressible traces of 100 kB; rank 0's step trace and its sidecar fit

    note = fit_traces(tmp_path, cap)

    assert note is not None and "3 trace(s) of secondary ranks dropped" in note and "(fits;" in note
    assert _names(tmp_path) == {
        "torch_trace/run-TP-0-EP-0.trace.json.gz",
        "torch_trace/capture_traces/bs_1-TP-0.trace.json.gz",
    }
    record = json.loads((tmp_path / TRACE_FIT_FILE).read_text(encoding="utf-8"))
    assert sorted(record["dropped_secondary_ranks"]) == [
        f"torch_trace/run-TP-{r}-EP-{r}.trace.json.gz" for r in (1, 2, 3)
    ]
    inventory(tmp_path, max_total_bytes=cap)


def test_an_output_that_still_does_not_fit_is_left_for_the_inventory_to_refuse(tmp_path: Path) -> None:
    _traces(tmp_path, gzipped=True, size=100_000)

    note = fit_traces(tmp_path, 150_000)

    assert note is not None and "still does not fit" in note
    with pytest.raises(OutputRejected) as refused:
        inventory(tmp_path, max_total_bytes=150_000)
    assert refused.value.error_class == "output_too_large"
