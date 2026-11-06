# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import subprocess

import pytest

from hyperloom_common import amd_smi
from hyperloom_common.amd_smi import (
    AmdSmiError,
    gfx_clock_limits,
    gpu_rows,
    metric_value,
    read_amd_smi_json,
    run_amd_smi,
    single_clock_limit,
)


def _answer(monkeypatch, returncode=0, stdout="", stderr=""):
    monkeypatch.setattr(
        amd_smi.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
    )


def test_a_failed_run_names_the_last_line_amd_smi_printed(monkeypatch):
    _answer(monkeypatch, returncode=1, stderr="Traceback (most recent call last):\n  ...\nPermissionError: elevation\n")
    with pytest.raises(AmdSmiError, match=r"amd-smi set exited 1: PermissionError: elevation$"):
        run_amd_smi(["set"])


def test_an_amd_smi_that_cannot_start_is_an_amd_smi_error(monkeypatch):
    def _missing(cmd, **kw):
        raise FileNotFoundError("amd-smi")

    monkeypatch.setattr(amd_smi.subprocess, "run", _missing)
    with pytest.raises(AmdSmiError, match="could not run"):
        run_amd_smi(["list"])


def test_output_that_is_not_json_is_an_amd_smi_error(monkeypatch):
    _answer(monkeypatch, stdout="not json")
    with pytest.raises(AmdSmiError, match="printed no JSON"):
        read_amd_smi_json("metric", "--clock")


def test_rows_and_values_read_both_amd_smi_shapes():
    rows = [{"gpu": 0}, {"gpu": "1"}, "x"]
    assert gpu_rows({"gpu_data": rows}) == gpu_rows(rows) == [{"gpu": 0}]
    assert metric_value({"value": 750, "unit": "W"}) == metric_value(750) == 750.0
    assert metric_value("N/A") is None and metric_value(True) is None


def test_the_clock_limit_is_the_one_the_filled_gfx_slots_agree_on():
    clock = {
        "gfx_0": {"clk": {"value": 140}, "max_clk": {"value": 2400}},
        "gfx_1": {"clk": {"value": 150}, "max_clk": {"value": 2400}},
        "gfx_2": {"clk": "N/A", "max_clk": {"value": 1800}},
        "mem_0": {"clk": {"value": 900}, "max_clk": {"value": 1300}},
    }
    assert gfx_clock_limits(clock) == {2400.0}
    assert single_clock_limit(gfx_clock_limits(clock)) == 2400.0
    assert single_clock_limit({2400.0, 2100.0}) is None
    assert gfx_clock_limits("N/A") == frozenset()
