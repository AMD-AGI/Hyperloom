# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the --help capability probe.

The probe exists because aiter moved a tuner script and changed its argparse
surface, and forge kept sending a flag the old path never accepted (14 runs,
0 output). The tests below pin the three behaviours that make the probe an
improvement rather than a new failure mode:

1. a rejected *required* flag fails the run instead of silently degrading it,
2. a probe that cannot run vetoes nothing,
3. probing is cached, because each one costs ~6-7s of ``import aiter``.
"""

from __future__ import annotations

import subprocess

import pytest

from kernelforge.gemm_tune import script_probe as sp


class TestNegativeNumbersAreValuesNotFlags:
    """``-1.0`` is an argument, not an option.

    Testing ``isdigit()`` alone called every non-integer negative a flag, which
    splits an option from its own value: the number then reads as an unsupported
    flag and the option reads as having been passed nothing.
    """

    def test_numeric_forms(self):
        for tok in ("-1", "-1.0", "-1e-3", "-1.5E+2", "+2", "-.5", "0", "3.25"):
            assert not sp._is_flag(tok), tok

    def test_flag_forms(self):
        for tok in ("--libtype", "-v", "-o2", "--with-hipblaslt", "-k"):
            assert sp._is_flag(tok), tok

    def test_a_negative_value_stays_with_its_option(self):
        surface = sp.ScriptSurface("s", frozenset({"--min_improvement_pct"}), True)
        out = sp.filter_args(["--min_improvement_pct", "-1.5"], surface)
        assert out.args == ["--min_improvement_pct", "-1.5"]
        assert out.dropped == [] and out.rejected_required == []

    def test_a_dropped_option_takes_its_negative_value_with_it(self):
        surface = sp.ScriptSurface("s", frozenset({"--untune_file"}), True)
        out = sp.filter_args(["--untune_file", "x.csv", "--iters", "-2.5"], surface)
        assert out.args == ["--untune_file", "x.csv"]
        assert out.dropped == ["--iters"]


def test_corrupt_probe_cache_costs_a_reprobe_not_a_crash(tmp_path, monkeypatch):
    # A truncated or hand-edited cache can decode to a list or a string just as
    # validly as to a dict, and .get on those raises rather than missing.
    monkeypatch.setenv("FORGE_SCRIPT_PROBE_CACHE", str(tmp_path))
    for payload in ("[1, 2, 3]", '"nope"', "null", "17"):
        (tmp_path / "deadbeef.json").write_text(payload, encoding="utf-8")
        assert sp._read_cache("deadbeef") is None


_HELP = """usage: gemm_a16w16_tune.py [-h] [-i INPUT] [-o OUTPUT] [--libtype {all,asm}]
                           [--with-hipblaslt] [--mp MP] [-v]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        untuned csv
  --libtype {all,asm,flydsl,hipblaslt}
                        libtypes to search (hipblaslt requires --with-hipblaslt)
  --with-hipblaslt      enable the hipblaslt candidate generator
  --mp MP               parallel workers
  -v, --verbose         verbose
"""


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_SCRIPT_PROBE_CACHE", str(tmp_path / "probe_cache"))
    sp._MEMO.clear()
    yield
    sp._MEMO.clear()


def _script(tmp_path, name="gemm_a16w16_tune.py", body="# stub"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _fake_run(stdout="", stderr="", rc=0, calls=None):
    def _run(cmd, **kwargs):
        if calls is not None:
            calls.append(cmd)
        return subprocess.CompletedProcess(cmd, rc, stdout, stderr)

    return _run


class TestParseHelpFlags:
    def test_extracts_long_and_short_flags(self):
        flags = sp.parse_help_flags(_HELP)
        assert {"--libtype", "--with-hipblaslt", "--mp", "-v", "-i", "--input"} <= flags

    def test_does_not_split_on_hyphenated_flag(self):
        # "--with-hipblaslt" must not be truncated to "--with".
        assert "--with-hipblaslt" in sp.parse_help_flags("  --with-hipblaslt  enable it")

    def test_empty_input_is_empty(self):
        assert sp.parse_help_flags("") == frozenset()


class TestProbeScript:
    def test_reads_flags_from_help(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sp.subprocess, "run", _fake_run(stdout=_HELP))
        surface = sp.probe_script(_script(tmp_path))
        assert surface.probed is True
        assert surface.supports("--with-hipblaslt")
        assert not surface.supports("--mxfp4-flydsl")

    def test_nonzero_rc_still_usable_when_flags_parsed(self, tmp_path, monkeypatch):
        # aiter's import side effects can make --help exit non-zero; the same
        # lesson as the tuner's own exit code -- judge by output, not by rc.
        monkeypatch.setattr(sp.subprocess, "run", _fake_run(stderr=_HELP, rc=1))
        assert sp.probe_script(_script(tmp_path)).probed is True

    def test_caches_by_digest_across_calls(self, tmp_path, monkeypatch):
        calls: list = []
        monkeypatch.setattr(sp.subprocess, "run", _fake_run(stdout=_HELP, calls=calls))
        script = _script(tmp_path)
        sp.probe_script(script)
        sp._MEMO.clear()  # force the on-disk cache to be exercised
        sp.probe_script(script)
        assert len(calls) == 1, "probe re-ran despite an unchanged script"

    def test_edited_script_is_reprobed(self, tmp_path, monkeypatch):
        calls: list = []
        monkeypatch.setattr(sp.subprocess, "run", _fake_run(stdout=_HELP, calls=calls))
        script = _script(tmp_path)
        sp.probe_script(script)
        script.write_text("# aiter moved on", encoding="utf-8")
        sp._MEMO.clear()
        sp.probe_script(script)
        assert len(calls) == 2, "cache keyed on path only, not content"


class TestProbeFailureIsPermissive:
    """A probe that cannot run must never veto a call that would have worked."""

    def test_missing_script(self, tmp_path):
        surface = sp.probe_script(tmp_path / "gone.py")
        assert surface.probed is False and surface.supports("--anything")

    def test_subprocess_error(self, tmp_path, monkeypatch):
        def _boom(cmd, **kwargs):
            raise OSError("no interpreter")

        monkeypatch.setattr(sp.subprocess, "run", _boom)
        assert sp.probe_script(_script(tmp_path)).supports("--libtype")

    def test_timeout(self, tmp_path, monkeypatch):
        def _slow(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 1)

        monkeypatch.setattr(sp.subprocess, "run", _slow)
        assert sp.probe_script(_script(tmp_path)).supports("--libtype")

    def test_unparseable_help(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sp.subprocess, "run", _fake_run(stdout="Segmentation fault"))
        surface = sp.probe_script(_script(tmp_path))
        assert surface.probed is False and surface.supports("--libtype")


def _surface(*flags):
    return sp.ScriptSurface("s.py", frozenset(flags), True)


class TestFilterArgs:
    def test_supported_args_pass_through(self):
        args = ["-i", "in.csv", "--libtype", "hipblaslt", "--with-hipblaslt"]
        out = sp.filter_args(args, _surface("-i", "--libtype", "--with-hipblaslt"))
        assert out.ok and out.args == args and out.dropped == []

    def test_required_flag_rejected_is_reported(self):
        out = sp.filter_args(["--libtype", "hipblaslt", "--with-hipblaslt"], _surface("--libtype"))
        assert not out.ok
        assert out.rejected_required == ["--with-hipblaslt"]

    def test_droppable_flag_is_dropped_with_its_value(self):
        out = sp.filter_args(
            ["-i", "in.csv", "--iters", "20", "--libtype", "all"],
            _surface("-i", "--libtype"),
        )
        assert out.ok
        assert out.args == ["-i", "in.csv", "--libtype", "all"]
        assert out.dropped == ["--iters"]
        assert "20" not in out.args, "dropped flag left its value behind"

    def test_unknown_unsupported_flag_is_kept_on_purpose(self):
        # Keeping it makes the script emit "unrecognized arguments: --wat",
        # which the call-time guard turns into a precise failure. Guessing here
        # would only hide which flag was wrong.
        out = sp.filter_args(["--wat", "1", "-i", "in.csv"], _surface("-i"))
        assert out.ok and out.dropped == []
        assert out.args == ["--wat", "1", "-i", "in.csv"]

    def test_unprobed_surface_keeps_everything(self):
        args = ["--libtype", "all", "--iters", "20", "--wat"]
        out = sp.filter_args(args, sp.ScriptSurface("s.py", frozenset(), False))
        assert out.ok and out.args == args and out.dropped == []

    def test_negative_numbers_are_values_not_flags(self):
        out = sp.filter_args(["--min_improvement_pct", "-1"], _surface("--min_improvement_pct"))
        assert out.args == ["--min_improvement_pct", "-1"]

    def test_flag_without_value_at_end(self):
        out = sp.filter_args(["-i", "in.csv", "-v"], _surface("-i"))
        assert out.ok and out.dropped == ["-v"] and out.args == ["-i", "in.csv"]
