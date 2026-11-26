# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom.inspector.history import MetricsHistory, history_path


def test_history_path_is_per_session_and_never_inside_it(tmp_path):
    a, b = tmp_path / "x" / "run", tmp_path / "y" / "run"
    a.mkdir(parents=True), b.mkdir(parents=True)
    pa, pb = history_path(a, tmp_path / "hist"), history_path(b, tmp_path / "hist")
    assert pa.parent == pb.parent == (tmp_path / "hist").resolve()
    assert pa != pb and pa.name.startswith("run-")
    for inside in (a, a / "reports"):
        with pytest.raises(ValueError, match="inside the session"):
            history_path(a, inside)


def test_ring_keeps_the_newest_samples_and_skips_a_torn_line(tmp_path):
    h = MetricsHistory(tmp_path / "h.jsonl", keep=3)
    for i in range(7):
        h.append({"t": float(i), "memory": {"used_bytes": i}})
    # Rotation at 2 * keep lines leaves the newest keep, then one more append.
    assert len((tmp_path / "h.jsonl").read_text().splitlines()) == 4
    with (tmp_path / "h.jsonl").open("a") as fh:
        fh.write('{"t": 7, "mem')
    assert [r["t"] for r in MetricsHistory(tmp_path / "h.jsonl", keep=3).load()] == [5.0, 6.0]
    assert MetricsHistory(tmp_path / "missing.jsonl").load() == []


def test_a_write_error_disables_the_history(tmp_path):
    (tmp_path / "file").write_text("")
    h = MetricsHistory(tmp_path / "file" / "h.jsonl")
    h.append({"t": 1.0})
    assert h.path is None
    h.append({"t": 2.0})
