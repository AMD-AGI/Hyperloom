# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The number carried out of a run, and the verdicts that must survive to reach it."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.sikl import report, workspace
from hyperloom.sikl.schema import find, load
from hyperloom.sikl.tests.conftest import CASES, SCHEMA_NAME

KERNEL = '"""Start."""\n\nimport torch\n\n\ndef run(*, a, b, trans_b, bias, out_dtype, out):\n    torch.matmul(a, b.t(), out=out)\n'


@pytest.fixture
def staged(sikl_root: Path, tmp_path: Path):
    schema = load(find(sikl_root, SCHEMA_NAME))
    run = workspace.materialize(
        schema, tmp_path / "run", sikl_root=sikl_root, language="flydsl", starting_kernel=KERNEL
    )
    return schema, run


def _result(case_id: str, *, candidate=0.01, baseline=0.02, bound=1.5, incorrect=()) -> report.CaseResult:
    return report.CaseResult(
        case_id=case_id, incorrect=tuple(incorrect), candidate_ms=candidate, baseline_ms=baseline, speedup_bound=bound
    )


def test_finalize_scores_the_kernel_the_workspace_holds(staged):
    """The loop leaves its kept kernel in the workspace; that is the one SIKL scores."""
    schema, run = staged
    summary = report.finalize(schema, run).read_text(encoding="utf-8")
    assert all(f"`{case_id}`" in summary for case_id in CASES)
    assert "Geometric mean speedup:** 0.9000x over 2 scored cases" in summary


def test_a_driver_that_writes_no_outcomes_is_a_failure_to_measure(staged):
    schema, run = staged
    run.driver.write_text("raise SystemExit(1)\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no outcomes"):
        report.finalize(schema, run)


def test_the_table_states_the_unit_its_numbers_are_in(staged):
    schema, _run = staged
    summary = report.build_summary(schema, [_result(case_id) for case_id in CASES])
    assert "| case | candidate us | baseline us | speedup | lower bound | served |" in summary
    assert "| `m=8` | 10.00 | 20.00 | 2.000x | 1.500x | yes |" in summary


def test_a_workload_is_served_only_correct_and_past_sikls_bound(staged):
    schema, _run = staged
    results = [_result("m=8", bound=1.02), _result("m=16", incorrect=("numeric_mismatch",))]
    summary = report.build_summary(schema, results)
    assert "Cases SIKL would serve** (speedup lower bound ≥ 1.03): 0/2" in summary
    assert "- `m=16`: SIKL found it wrong: numeric_mismatch" in summary


def test_a_case_without_a_baseline_is_named_not_averaged(staged):
    schema, _run = staged
    results = [_result("m=8"), _result("m=16", baseline=None, bound=None)]
    summary = report.build_summary(schema, results)
    assert "over 1 scored cases" in summary
    assert "- `m=16`: SIKL timed no baseline" in summary


def test_a_losing_case_is_listed(staged):
    schema, _run = staged
    summary = report.build_summary(schema, [_result("m=8", candidate=0.04, bound=0.4), _result("m=16")])
    assert "- `m=8`: 0.500x" in summary
