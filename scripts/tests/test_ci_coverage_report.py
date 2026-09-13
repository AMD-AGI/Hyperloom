# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the single-pass CI coverage report and deferred gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import coverage
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("ci_coverage_report", _ROOT / "scripts" / "ci_coverage_report.py")
assert _SPEC and _SPEC.loader
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)

_REPORT_TEXT = """Name                                  Stmts   Miss   Cover
----------------------------------------------------------
src/hyperloom/example.py                  10      1  90.00%
/checkout/OOB/example.py                  10      0 100.00%
----------------------------------------------------------
TOTAL                                    20      1  95.00%
"""
_CONFIG = """[tool.coverage.run]
source = ["src/hyperloom", "OOB", "src/kernelforge"]
[tool.coverage.report]
fail_under = 90
precision = 2
"""


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(_CONFIG, encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    monkeypatch.setenv("PYTEST_OUTCOME", "success")
    monkeypatch.setenv("COVERAGE_FILE", str(tmp_path / ".coverage"))
    monkeypatch.delenv("REPORT_EXIT_CODE", raising=False)
    monkeypatch.delenv("COVERAGE_RELAX_FAIL_UNDER", raising=False)
    return tmp_path


@pytest.mark.parametrize("exit_code", [0, 1, 2, 7, -9, None])
def test_report_runs_once_and_defers_native_status(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], exit_code: int | None
) -> None:
    (workspace / ".coverage").touch()
    (workspace / "summary").write_text("### Shard results\n", encoding="utf-8")
    (workspace / "outputs").write_text("existing=true\n", encoding="utf-8")

    def coverage_main(argv: list[str]) -> int | None:
        sys.stdout.write(_REPORT_TEXT)
        sys.stderr.write("coverage warning without trailing newline")
        return exit_code

    invoke = Mock(side_effect=coverage_main)
    monkeypatch.setattr(report, "coverage_main", invoke)

    assert report.main([]) == 0
    invoke.assert_called_once_with(["report", "--skip-empty"])
    captured = capsys.readouterr()
    assert captured.out == "=== Full coverage report (step log) ===\n" + _REPORT_TEXT
    assert captured.err == "coverage warning without trailing newline"
    assert (workspace / "outputs").read_text(encoding="utf-8") == (
        f"existing=true\nreport_exit_code={exit_code or 0}\n"
    )
    assert (workspace / "summary").read_text(encoding="utf-8") == (
        "### Shard results\n"
        "## Coverage (UT)\n\n"
        f"Python {sys.version_info.major}.{sys.version_info.minor}; "
        "roots and CI pytest argv from `pyproject.toml`. Combined across sharded jobs.\n\n"
        "### Combined measured source (all configured trees)\n"
        "**TOTAL (line): 95.00%**\n\n"
        "### Per-tree (narrow)\n"
        "| Tree | Line coverage (TOTAL) |\n"
        "|------|----------------------|\n"
        "| `src/hyperloom` | 90.00% |\n"
        "| `OOB` | 100.00% |\n"
        "| `src/kernelforge` | n/a |\n"
    )


@pytest.mark.parametrize("outcome", ["failure", "unknown"])
def test_failed_tests_still_render_complete_report_with_caveat(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    (workspace / ".coverage").touch()
    monkeypatch.setenv("PYTEST_OUTCOME", outcome)

    def coverage_main(argv: list[str]) -> int:
        print(_REPORT_TEXT, end="")
        return 2

    invoke = Mock(side_effect=coverage_main)
    monkeypatch.setattr(report, "coverage_main", invoke)
    assert report.main([]) == 0
    invoke.assert_called_once()
    assert "Tests did not all pass; coverage can be **partial or misleading**." in (workspace / "summary").read_text(
        encoding="utf-8"
    )
    assert (workspace / "outputs").read_text(encoding="utf-8") == "report_exit_code=2\n"


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_missing_data_skips_report_and_records_failure(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    monkeypatch.setenv("PYTEST_OUTCOME", outcome)
    invoke = Mock()
    monkeypatch.setattr(report, "coverage_main", invoke)
    assert report.main([]) == 0
    invoke.assert_not_called()
    assert (workspace / "summary").read_text(encoding="utf-8") == "## Coverage (UT)\n\nNo `.coverage` data.\n"
    assert (workspace / "outputs").read_text(encoding="utf-8") == "report_exit_code=1\n"


def test_unexpected_exception_keeps_diagnostics_and_does_not_publish_success(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (workspace / ".coverage").touch()

    def broken_report(argv: list[str]) -> int:
        sys.stdout.write("partial report")
        sys.stderr.write("original diagnostic")
        raise RuntimeError("unexpected report error")

    invoke = Mock(side_effect=broken_report)
    monkeypatch.setattr(report, "coverage_main", invoke)
    with pytest.raises(RuntimeError, match="unexpected report error"):
        report.main([])
    invoke.assert_called_once()
    captured = capsys.readouterr()
    assert captured.out.endswith("partial report")
    assert captured.err == "original diagnostic"
    assert not (workspace / "outputs").exists()
    assert report.main(["gate"]) == 1


@pytest.mark.parametrize("exit_code", [0, 1, 2, 7, -9])
@pytest.mark.parametrize("relax", ["", "false", "0", "1", "true", " YES ", "on"])
def test_gate_relaxes_only_threshold_failure(monkeypatch: pytest.MonkeyPatch, exit_code: int, relax: str) -> None:
    monkeypatch.setenv("REPORT_EXIT_CODE", str(exit_code))
    monkeypatch.setenv("COVERAGE_RELAX_FAIL_UNDER", relax)
    invoke = Mock()
    monkeypatch.setattr(report, "coverage_main", invoke)
    expected = 0 if exit_code == 2 and relax.strip().lower() in {"1", "true", "yes", "on"} else exit_code
    assert report.main(["gate"]) == expected
    invoke.assert_not_called()


@pytest.mark.parametrize("status", [None, "", " ", "not-an-exit-code", "2.0"])
@pytest.mark.parametrize("relax", ["false", "true"])
def test_gate_fails_closed_without_valid_status(
    monkeypatch: pytest.MonkeyPatch, status: str | None, relax: str
) -> None:
    if status is None:
        monkeypatch.delenv("REPORT_EXIT_CODE", raising=False)
    else:
        monkeypatch.setenv("REPORT_EXIT_CODE", status)
    monkeypatch.setenv("COVERAGE_RELAX_FAIL_UNDER", relax)
    assert report.main(["gate"]) == 1


def test_summary_without_github_files_prints_to_log(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY")
    monkeypatch.delenv("GITHUB_OUTPUT")
    assert report.main([]) == 0
    assert "## Coverage (UT)\n\nNo `.coverage` data." in capsys.readouterr().out


def test_report_parsers_preserve_path_spaces_and_source_aliases() -> None:
    text = """Name Stmts Miss Cover
----------------------------------------
C:/checkout with spaces/src/hyperloom/a.py 10 2 80.00%
/venv/site-packages/agent_mcp_server/a.py 20 1 95.00%
OOB/a.py 10 2 80.00%
src/kernelforge/a.py 10 0 100.00%
invalid row
TOTAL 50 5 90.00%
"""
    assert report.extract_total(text) == "90.00%"
    assert report.extract_total("No data to report.") == "unknown"
    assert report.per_tree_totals(text) == {"src/hyperloom": "80.00%", "OOB": "90.00%"}
    assert report.per_tree_totals("No data to report.") == {"src/hyperloom": "n/a", "OOB": "n/a"}


@pytest.mark.parametrize(
    ("statements", "covered", "precision", "threshold", "expected", "total"),
    [
        (10, 9, 2, 90, 0, "90.00%"),
        (10, 8, 2, 90, 2, "80.00%"),
        (1001, 900, 2, 90, 2, "89.91%"),
        (1001, 900, 0, 90, 0, "90%"),
        (1001, 900, 2, 89.9, 0, "89.91%"),
        (1001, 900, 2, 89.92, 2, "89.91%"),
    ],
)
def test_real_coverage_owns_threshold_and_precision(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    statements: int,
    covered: int,
    precision: int,
    threshold: float,
    expected: int,
    total: str,
) -> None:
    (workspace / "pyproject.toml").write_text(
        f'[tool.coverage.run]\nsource = ["."]\n[tool.coverage.report]\n'
        f"precision = {precision}\nfail_under = {threshold}\n",
        encoding="utf-8",
    )
    source = workspace / "tiny.py"
    source.write_text("value = 1\n" * statements, encoding="utf-8")
    data = coverage.CoverageData(basename=str(workspace / ".coverage"))
    data.add_lines({str(source): set(range(1, covered + 1))})
    data.write()
    invoke = Mock(wraps=report.coverage_main)
    monkeypatch.setattr(report, "coverage_main", invoke)

    assert report.main([]) == 0
    invoke.assert_called_once_with(["report", "--skip-empty"])
    assert (workspace / "outputs").read_text(encoding="utf-8") == f"report_exit_code={expected}\n"
    assert f"**TOTAL (line): {total}**" in (workspace / "summary").read_text(encoding="utf-8")
    monkeypatch.setenv("REPORT_EXIT_CODE", str(expected))
    assert report.main(["gate"]) == expected
    monkeypatch.setenv("COVERAGE_RELAX_FAIL_UNDER", "true")
    assert report.main(["gate"]) == 0
    invoke.assert_called_once()


def test_real_coverage_operational_error_cannot_be_relaxed(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = coverage.CoverageData(basename=str(workspace / ".coverage"))
    data.add_lines({str(workspace / "missing_source.py"): {1}})
    data.write()

    assert report.main([]) == 0
    assert (workspace / "outputs").read_text(encoding="utf-8") == "report_exit_code=1\n"
    assert "No source for code" in capsys.readouterr().out
    monkeypatch.setenv("REPORT_EXIT_CODE", "1")
    monkeypatch.setenv("COVERAGE_RELAX_FAIL_UNDER", "true")
    assert report.main(["gate"]) == 1
