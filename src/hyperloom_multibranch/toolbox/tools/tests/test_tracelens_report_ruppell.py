# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The report route names what ruppell kept of a profile it traced, beside TraceLens' shares, which keep their shape.

The check is the real one of W243's ruppell run (Qwen3-14B-FP8 TP1 on MI355X, board ``hyperloom-multibranch#150``); the
module table is written by the backend's own writer.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hyperloom.measure import ruppell
from hyperloom_multibranch.toolbox.tools import tracelens_report as report
from hyperloom_multibranch.toolbox.tools.op_shares import MODULE_SHARES_KEY
from hyperloom_multibranch.toolbox.tools.tests import final_probe as probe
from hyperloom_multibranch.toolbox.tools.tests.test_tracelens_report import Rig, _run, rig  # noqa: F401 (fixture)

DATA = Path(ruppell.__file__).parent / "tests" / "data" / "ruppell"
ROWS = [
    {"kind": "decode", "module": "RadixAttention", "gpu_ms": 6.0, "share_percent": 60.0, "steps": 64},
    {"kind": "decode", "module": "RowParallelLinear", "gpu_ms": 3.0, "share_percent": 30.0, "steps": 64},
    {"kind": "decode", "module": "unattributed", "gpu_ms": 1.0, "share_percent": 10.0, "steps": 64},
    {"kind": "prefill", "module": "RowParallelLinear", "gpu_ms": 2.0, "share_percent": 100.0, "steps": 2},
]


def _kept(profile: Path, *, check: str | None = "w243_check.json", rows=ROWS) -> Path:
    kept = profile.parent / "ruppell"
    kept.mkdir(parents=True, exist_ok=True)
    ruppell.write_module_shares(rows, kept / "module_shares.csv")
    if check is not None:
        shutil.copyfile(DATA / check, kept / "check.json")
    return kept


def test_a_profile_ruppell_traced_names_its_table_check_and_decode_share(tmp_path):
    kept = _kept(tmp_path / "profile")

    assert report.ruppell_fields(tmp_path / "profile") == {
        MODULE_SHARES_KEY: str(kept / "module_shares.csv"),
        "ruppell_status": "ok",
        "ruppell_check_ok": True,
        "ruppell_decode_attributed_share": 0.9,
    }
    deep = tmp_path / "profile" / "torch_trace" / "1.-TP-0.trace.json.gz"
    assert report.ruppell_fields(deep)[MODULE_SHARES_KEY] == str(kept / "module_shares.csv"), "a trace file below it"


def test_a_failed_or_missing_check_and_a_table_without_decode_are_said(tmp_path):
    kept = _kept(tmp_path / "profile")
    (kept / "check.json").write_text(json.dumps({"ok": False, "criteria": {}}))
    assert report.ruppell_fields(tmp_path / "profile")["ruppell_status"] == "check_failed"
    (kept / "check.json").unlink()
    _kept(tmp_path / "profile", check=None, rows=ROWS[3:])
    fields = report.ruppell_fields(tmp_path / "profile")
    assert (fields["ruppell_status"], fields["ruppell_check_ok"], fields["ruppell_decode_attributed_share"]) == (
        "unchecked",
        False,
        None,
    )


def test_a_profile_ruppell_did_not_trace_adds_nothing(tmp_path):
    assert report.ruppell_fields(tmp_path / "profile") == {}
    assert report.ruppell_fields(tmp_path / "elsewhere" / "trace.json") == {}
    (tmp_path / "ruppell").mkdir()
    assert report.ruppell_fields(tmp_path / "profile") == {}, "a directory with no table is not a report"


@pytest.mark.parametrize("traced", [True, False])
async def test_the_report_route_carries_the_fields_and_its_shares_keep_their_shape(rig: Rig, tmp_path: Path, traced):
    probe.install_capture(rig.trace_input / "torch_trace" / "graph_capture_profile")
    if traced:
        _kept(rig.trace_input)

    _, pointer, record = await _run(tmp_path, probe.GRADED, rig)

    result = record["result"]
    assert (MODULE_SHARES_KEY in result) is traced
    if traced:
        assert (result["ruppell_status"], result["ruppell_decode_attributed_share"]) == ("ok", 0.9)
    assert "RadixAttention" not in pointer.op_shares, "the op shares are TraceLens' kernel shares, never ruppell's"
    assert sum(pointer.op_shares.values()) == pytest.approx(100.0)
