# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure-function tests for ``scripts/platform_audit.py``.

The script sits outside ``[tool.coverage.run].source`` because ``scripts/`` is
not shipped as a package, but its output is consumed as a configuration verdict
-- so the logic that produces that verdict is tested here. Everything covered
below is hardware-independent; nothing in this module reads real sysfs, loads a
core, or shells out.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "platform_audit.py"


def _load():
    spec = importlib.util.spec_from_file_location("platform_audit", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pa = _load()


# -------------------------------------------------------------- verdicts

def test_determinism_is_recorded_not_judged():
    """The one knob this tool cannot read, only infer, must not gate anything.

    Power remains the setting to want (58011 §4.2.2, "maximum performance of any
    individual system"). But the OS layer infers it from per-core frequency
    spread, and five consecutive runs on one unchanged EPYC 9575F measured 7.9,
    21.0, 18.1, 20.9 and 15.1 MHz -- the host's own jitter straddles the
    threshold. Gating on that flips the exit code on a machine nobody touched.
    """
    assert "determinism" not in pa.CHECKED
    assert "determinism" in pa.RECORDED
    row = {r["key"]: r for r in pa.build_rows({"determinism": "performance"})}["determinism"]
    assert row["verdict"] == "RECORD"
    assert pa.exit_code(pa.build_rows({"determinism": "performance"})) != pa.EXIT_FAIL


def test_determinism_row_says_it_was_inferred():
    """A deduction presented like a reading invites a BIOS change on a heuristic.

    A uniformly binned part running Power determinism produces the same low
    spread as a part running Performance determinism, so the row must carry the
    caveat rather than look like a setting that was read.
    """
    assert pa.RECORDED["determinism"].get("inferred") is True
    row = {r["key"]: r for r in pa.build_rows({"determinism": "performance"})}["determinism"]
    assert row["inferred"] is True
    # A knob read straight from sysfs/MSR carries no such caveat.
    boost = {r["key"]: r for r in pa.build_rows({"core_performance_boost": "enabled"})}
    assert boost["core_performance_boost"]["inferred"] is False


def test_every_recorded_knob_explains_its_silence():
    """"Not checked" is a claim, so each recorded knob carries its reason."""
    for key, spec in pa.RECORDED.items():
        assert spec["why"].strip(), f"{key} records no reason for having no verdict"
    row = {r["key"]: r for r in pa.build_rows({})}["nps"]
    assert row["why"], "the reason must reach --json, not only this module"


def test_verdict_is_exact_not_substring():
    """A value merely containing a target word is not a pass."""
    assert pa.verdict("cpufreq_governor", "performance") == "PASS"
    assert pa.verdict("cpufreq_governor", "schedutil-performance-ish") == "FAIL"
    assert pa.verdict("core_performance_boost", "enabled") == "PASS"
    assert pa.verdict("core_performance_boost", "disabled") == "FAIL"


def test_verdict_normalizes_case_and_whitespace():
    assert pa.verdict("cpufreq_governor", "  PERFORMANCE  ") == "PASS"
    assert pa.verdict("core_performance_boost", "  Enabled ") == "PASS"


@pytest.mark.parametrize("value", ["", "unknown", "auto", "n/a", "none", None])
def test_unresolved_values_are_unknown_not_pass(value):
    """An unresolved knob must never be reported as satisfied."""
    assert pa.verdict("core_performance_boost", value) == "UNKNOWN"


# -------------------------------------------------------------- determinism

def test_infer_determinism_normalizes_value_and_separates_the_caveat():
    value, note = pa.infer_determinism(25.0)
    assert value == "power" and "25.0" in note

    value, note = pa.infer_determinism(0.5)
    assert value == "performance"
    # The caveat survives, but only as prose -- never as the reported value.
    assert "uniformly binned" in note


def test_infer_determinism_declines_when_ambiguous_or_unmeasured():
    assert pa.infer_determinism(None)[0] is None
    assert pa.infer_determinism(6.0)[0] is None  # between the two thresholds


# -------------------------------------------------------------- exit codes

def _rows(**verdicts):
    return [{"key": k, "verdict": v, "knob": k, "value": "", "target": "", "note": ""}
            for k, v in verdicts.items()]


def test_exit_code_ranks_fail_above_unknown():
    assert pa.exit_code(_rows(a="PASS", b="PASS")) == pa.EXIT_OK
    assert pa.exit_code(_rows(a="PASS", b="UNKNOWN")) == pa.EXIT_UNKNOWN
    assert pa.exit_code(_rows(a="UNKNOWN", b="FAIL")) == pa.EXIT_FAIL


def test_recorded_knobs_never_affect_the_exit_code():
    """SMT being enabled is the fleet norm and must not turn CI red."""
    assert pa.exit_code(_rows(a="PASS", smt="RECORD", nps="RECORD")) == pa.EXIT_OK


def test_quick_and_full_agree_on_the_same_host():
    """--quick must reach the same verdicts, or it is not a usable gate.

    No judged knob needs load generation any more, so the only thing --quick
    gives up is a recorded value. A fast run and a full run on one healthy host
    must therefore return the same exit code.
    """
    common = {
        "core_performance_boost": "enabled",
        "cpufreq_governor": "performance",
        "smt": "enabled",
        "nps": "NPS1",
    }
    quick = pa.build_rows({**common, "determinism": "unknown", "quick": True})
    full = pa.build_rows({**common, "determinism": "power"})
    assert pa.exit_code(quick) == pa.exit_code(full) == pa.EXIT_OK
    assert {r["key"]: r["verdict"] for r in quick}["determinism"] == "RECORD"


def test_quick_mode_still_fails_on_a_knob_it_can_read():
    """An unmeasured recorded knob must not mask a knob quick mode can see."""
    osl = {
        "core_performance_boost": "disabled",
        "cpufreq_governor": "performance",
        "determinism": "unknown",
        "smt": "enabled",
        "nps": "NPS1",
        "quick": True,
    }
    assert pa.exit_code(pa.build_rows(osl)) == pa.EXIT_FAIL


# -------------------------------------------------------------- rows

def test_build_rows_marks_recorded_knobs_and_judges_the_rest():
    osl = {
        "core_performance_boost": "enabled",
        "cpufreq_governor": "schedutil",
        "determinism": "power",
        "smt": "enabled",
        "nps": "NPS4",
    }
    by_key = {r["key"]: r for r in pa.build_rows(osl)}
    assert by_key["core_performance_boost"]["verdict"] == "PASS"
    assert by_key["cpufreq_governor"]["verdict"] == "FAIL"
    assert by_key["determinism"]["verdict"] == "RECORD"
    assert by_key["smt"]["verdict"] == "RECORD"
    assert by_key["nps"]["verdict"] == "RECORD"


def test_every_judged_knob_cites_its_basis():
    """A target without a stated reason is how the profile drifted in the first place."""
    for key, spec in pa.CHECKED.items():
        assert spec["basis"].strip(), f"{key} has no basis"
        assert spec["target"], f"{key} has no target"


# -------------------------------------------------------------- epyc parsing

@pytest.mark.parametrize(
    "model,expected",
    [
        ("AMD EPYC 9575F 64-Core Processor", "EPYC 9005 (Turin)"),
        ("AMD EPYC 9755 128-Core Processor", "EPYC 9005 (Turin)"),
        ("AMD EPYC 9654 96-Core Processor", "EPYC 9004 (Genoa)"),
        ("AMD EPYC 7763 64-Core Processor", "EPYC 7003 (Milan)"),
    ],
)
def test_epyc_generation_reads_first_and_last_digit(model, expected):
    """The series is the first and last digit, not the middle two."""
    assert pa.epyc_generation(model) == expected


def test_epyc_generation_declines_on_non_numeric_skus():
    """Cloud SKUs like EPYC 9V84 must decline rather than guess a generation."""
    assert pa.epyc_generation("AMD EPYC 9V84") == "unknown"
    assert pa.epyc_generation("Intel Xeon Platinum 8480+") == "unknown"


# -------------------------------------------------------------- topology

def test_sample_cores_spreads_across_the_topology(monkeypatch):
    monkeypatch.setattr(pa, "physical_cores", lambda: list(range(0, 64)))
    assert pa.sample_cores(4) == [0, 16, 32, 48]


def test_sample_cores_degrades_on_small_parts(monkeypatch):
    """A part with fewer cores than requested must not index out of range."""
    monkeypatch.setattr(pa, "physical_cores", lambda: [0, 1])
    assert pa.sample_cores(4) == [0, 1]
    monkeypatch.setattr(pa, "physical_cores", lambda: [])
    assert pa.sample_cores(4) == []


def test_os_layer_survives_a_host_with_no_cpu_sysfs(monkeypatch):
    """"Never raises" is a promise, and a missing /sys tree is how it breaks.

    A container without /sys/devices/system/cpu is the ordinary case here, not
    an exotic one: it must degrade to "unknown" rather than throw out of a
    function every caller treats as total.
    """
    def _no_such_tree(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(pa.os, "listdir", _no_such_tree)
    monkeypatch.setattr(pa, "read", lambda _path: "")
    monkeypatch.setattr(pa, "read_hwcr", lambda: None)

    osl = pa.os_layer(quick=True)
    assert osl["identity"]["sockets"] is None
    assert osl["identity"]["nps"] == "unknown", "NPS0 is a value no BIOS can hold"
    assert pa.physical_cores() == []
    # Still a usable record: unresolved, not absent.
    assert {r["key"] for r in pa.build_rows(osl)} >= set(pa.CHECKED) | set(pa.RECORDED)
