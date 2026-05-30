"""Tests for the plugin interface and custom plugin."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

import pytest

from hyperloom.plugins.base import BenchResult, AccuracyResult
from hyperloom.plugins.custom import CustomBenchmarkPlugin, CustomAccuracyPlugin


@pytest.fixture
def json_bench_script(tmp_path: Path) -> str:
    """Create a benchmark script that outputs JSON."""
    script = tmp_path / "bench.sh"
    script.write_text('#!/bin/bash\necho \'{"throughput": 1234.5, "latency_mean_ms": 3.2}\'\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.fixture
def last_line_bench_script(tmp_path: Path) -> str:
    """Create a benchmark script that outputs a single number."""
    script = tmp_path / "bench.sh"
    script.write_text("#!/bin/bash\necho 'warming up...'\necho '5678.9'\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.fixture
def failing_script(tmp_path: Path) -> str:
    """Create a script that exits with error."""
    script = tmp_path / "fail.sh"
    script.write_text("#!/bin/bash\nexit 1\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


class TestCustomBenchmarkPlugin:
    def test_json_output(self, json_bench_script: str):
        plugin = CustomBenchmarkPlugin(
            script=json_bench_script,
            output_format="json",
            throughput_key="throughput",
        )
        result = plugin.run({})
        assert result.throughput == 1234.5
        assert result.latency_mean_ms == 3.2

    def test_last_line_output(self, last_line_bench_script: str):
        plugin = CustomBenchmarkPlugin(
            script=last_line_bench_script,
            output_format="last_line",
        )
        result = plugin.run({})
        assert result.throughput == 5678.9

    def test_failing_script(self, failing_script: str):
        plugin = CustomBenchmarkPlugin(script=failing_script)
        result = plugin.run({})
        assert not result.success
        assert result.throughput == 0.0

    def test_validate_missing_script(self):
        plugin = CustomBenchmarkPlugin(script="/nonexistent/script.sh")
        errors = plugin.validate_config({})
        assert len(errors) > 0
        assert "not found" in errors[0]


class TestCustomAccuracyPlugin:
    def test_json_output_passes(self, tmp_path: Path):
        script = tmp_path / "eval.sh"
        script.write_text('#!/bin/bash\necho \'{"score": 0.95}\'\n')
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        plugin = CustomAccuracyPlugin(
            script=str(script),
            output_format="json",
            score_key="score",
            threshold=0.9,
        )
        result = plugin.run({})
        assert result.passed
        assert result.score == 0.95

    def test_json_output_fails_threshold(self, tmp_path: Path):
        script = tmp_path / "eval.sh"
        script.write_text('#!/bin/bash\necho \'{"score": 0.5}\'\n')
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        plugin = CustomAccuracyPlugin(
            script=str(script),
            output_format="json",
            score_key="score",
            threshold=0.9,
        )
        result = plugin.run({})
        assert not result.passed
        assert result.score == 0.5
