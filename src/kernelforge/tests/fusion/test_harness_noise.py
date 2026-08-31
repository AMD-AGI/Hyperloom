# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The KEEP bar and the inherited floor assume measurements are stable.

Nothing had ever checked that. If run-to-run spread on a real GPU is comparable
to the 3% improvement margin, then "beat the previous result by 3%" is partly
deciding on noise -- and it decides whether a result is recorded at all. This
diagnostic measures the spread so the assumption can be checked per machine.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from kernelforge.fusion import command as cli
from kernelforge.fusion.command import main
from kernelforge.fusion.validate import BenchOutcome


def _run(tmp_path, monkeypatch, benches, repeat=None):
    harness = tmp_path / "kernel_harness.py"
    harness.write_text("print('{}')\n", encoding="utf-8")
    calls = iter(benches)

    class FakeRunner:
        def __init__(self, *_args, **_kwargs):
            pass

        def microbench(self, _recipe):
            return next(calls)

    monkeypatch.setattr(cli, "HarnessKernelRunner", FakeRunner)
    result = CliRunner().invoke(
        main,
        [
            "--harness-noise",
            str(harness),
            "--harness-noise-repeat",
            str(repeat if repeat is not None else len(benches)),
        ],
    )
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def _bench(eager, fused):
    return BenchOutcome(eager_us=eager, fused_us=fused, skipped=False, skip_reason="")


def test_a_steady_machine_puts_the_bar_well_outside_noise(tmp_path, monkeypatch):
    """Half a percent of spread leaves 3% comfortably decidable."""
    report = _run(
        tmp_path,
        monkeypatch,
        [
            _bench(100.0, 50.00),
            _bench(100.0, 50.10),
            _bench(100.0, 49.90),
            _bench(100.0, 50.05),
            _bench(100.0, 49.95),
        ],
    )
    assert report["usable"] == 5
    assert report["speedup_cv"] < 0.01
    assert report["bar_in_sigmas"] > 2.0
    assert report["verdict"] == "the 3% bar is outside noise"


def test_a_noisy_machine_is_called_out(tmp_path, monkeypatch):
    """When the spread rivals the margin, the floor is deciding on noise."""
    report = _run(
        tmp_path,
        monkeypatch,
        [
            _bench(100.0, 50.0),
            _bench(100.0, 56.0),
            _bench(100.0, 45.0),
            _bench(100.0, 53.0),
            _bench(100.0, 47.0),
        ],
    )
    assert report["speedup_cv"] > 0.05
    assert report["bar_in_sigmas"] < 2.0
    assert report["verdict"] == "the 3% bar is within noise"


def test_skipped_and_timing_less_runs_are_counted_not_averaged(tmp_path, monkeypatch):
    report = _run(
        tmp_path,
        monkeypatch,
        [
            _bench(100.0, 50.0),
            BenchOutcome(eager_us=None, fused_us=None, skipped=True, skip_reason="no gpu"),
            _bench(100.0, 50.0),
        ],
    )
    assert (report["usable"], report["failed"]) == (2, 1)


def test_a_single_usable_run_reports_no_statistics(tmp_path, monkeypatch):
    """One sample has no spread; reporting a stdev of zero would be a lie."""
    report = _run(
        tmp_path,
        monkeypatch,
        [
            _bench(100.0, 50.0),
            BenchOutcome(eager_us=None, fused_us=None, skipped=True, skip_reason="no gpu"),
        ],
    )
    assert report["usable"] == 1
    assert "speedup_cv" not in report


def test_the_reported_spread_matches_the_samples(tmp_path, monkeypatch):
    report = _run(
        tmp_path,
        monkeypatch,
        [
            _bench(100.0, 40.0),  # 2.50x
            _bench(100.0, 50.0),  # 2.00x
        ],
    )
    assert report["speedup_min"] == 2.0
    assert report["speedup_max"] == 2.5
    assert report["speedup_mean"] == 2.25
    assert report["spread_pct"] == round(0.5 / 2.25 * 100, 2)
