# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The report route names what a second tracer kept of a profile it traced, beside TraceLens' shares, which keep
their shape. The module table and the check are written here, in the form ``hyperloom.measure.second_tracer`` says a
tracer keeps them."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from hyperloom.measure.second_tracer import CHECK_FILE, MODULE_SHARES_FILE, OUTPUT_DIR
from hyperloom_multibranch.toolbox.tools import tracelens_report as report
from hyperloom_multibranch.toolbox.tools.op_shares import MODULE_SHARES_KEY
from hyperloom_multibranch.toolbox.tools.tests import final_probe as probe
from hyperloom_multibranch.toolbox.tools.tests.test_tracelens_report import Rig, _run, rig  # noqa: F401 (fixture)

COLUMNS = ("kind", "module", "gpu_ms", "share_percent", "steps")
ROWS = [
    {"kind": "decode", "module": "RadixAttention", "gpu_ms": 6.0, "share_percent": 60.0, "steps": 64},
    {"kind": "decode", "module": "RowParallelLinear", "gpu_ms": 3.0, "share_percent": 30.0, "steps": 64},
    {"kind": "decode", "module": "unattributed", "gpu_ms": 1.0, "share_percent": 10.0, "steps": 64},
    {"kind": "prefill", "module": "RowParallelLinear", "gpu_ms": 2.0, "share_percent": 100.0, "steps": 2},
]


def _kept(profile: Path, *, check: bool | None = True, rows=ROWS) -> Path:
    """What a tracer keeps beside ``profile``: its module table, and its check's verdict unless ``check`` is ``None``."""
    kept = profile.parent / OUTPUT_DIR
    kept.mkdir(parents=True, exist_ok=True)
    with (kept / MODULE_SHARES_FILE).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if check is not None:
        (kept / CHECK_FILE).write_text(json.dumps({"ok": check, "criteria": {}}), encoding="utf-8")
    return kept


def test_a_profile_a_tracer_traced_names_its_table_check_and_decode_share(tmp_path):
    kept = _kept(tmp_path / "profile")

    assert report.tracer_fields(tmp_path / "profile") == {
        MODULE_SHARES_KEY: str(kept / MODULE_SHARES_FILE),
        "tracer_status": "ok",
        "tracer_check_ok": True,
        "tracer_decode_attributed_share": 0.9,
    }
    deep = tmp_path / "profile" / "torch_trace" / "1.-TP-0.trace.json.gz"
    assert report.tracer_fields(deep)[MODULE_SHARES_KEY] == str(kept / MODULE_SHARES_FILE), "a trace file below it"


def test_a_failed_or_missing_check_and_a_table_without_decode_are_said(tmp_path):
    kept = _kept(tmp_path / "profile", check=False)
    assert report.tracer_fields(tmp_path / "profile")["tracer_status"] == "check_failed"
    (kept / CHECK_FILE).unlink()
    _kept(tmp_path / "profile", check=None, rows=ROWS[3:])
    fields = report.tracer_fields(tmp_path / "profile")
    assert (fields["tracer_status"], fields["tracer_check_ok"], fields["tracer_decode_attributed_share"]) == (
        "unchecked",
        False,
        None,
    )


def test_a_table_that_cannot_be_read_is_said_without_its_path(tmp_path):
    kept = _kept(tmp_path / "profile")
    (kept / MODULE_SHARES_FILE).write_text("kind,module,gpu_ms\ndecode,RadixAttention,not-a-number\n", encoding="utf-8")

    assert report.tracer_fields(tmp_path / "profile") == {
        MODULE_SHARES_KEY: "",
        "tracer_status": "unreadable",
        "tracer_check_ok": True,
    }


def test_a_profile_no_tracer_traced_adds_nothing(tmp_path):
    assert report.tracer_fields(tmp_path / "profile") == {}
    assert report.tracer_fields(tmp_path / "elsewhere" / "trace.json") == {}
    (tmp_path / OUTPUT_DIR).mkdir()
    assert report.tracer_fields(tmp_path / "profile") == {}, "a directory with no table is not a report"


@pytest.mark.parametrize("traced", [True, False])
async def test_the_report_route_carries_the_fields_and_its_shares_keep_their_shape(rig: Rig, tmp_path: Path, traced):
    probe.install_capture(rig.trace_input / "torch_trace" / "graph_capture_profile")
    if traced:
        _kept(rig.trace_input)

    _, pointer, record = await _run(tmp_path, probe.GRADED, rig)

    result = record["result"]
    assert (MODULE_SHARES_KEY in result) is traced
    if traced:
        assert (result["tracer_status"], result["tracer_decode_attributed_share"]) == ("ok", 0.9)
    assert "RadixAttention" not in pointer.op_shares, "the op shares are TraceLens' kernel shares, never a tracer's"
    assert sum(pointer.op_shares.values()) == pytest.approx(100.0)
