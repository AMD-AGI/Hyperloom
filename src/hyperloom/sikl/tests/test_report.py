# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The number carried out of a run, and the verdicts that must survive to reach it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.sikl import report, workspace
from hyperloom.sikl.definition import load
from hyperloom.sikl.tests.conftest import CASES, DEFINITION_NAME, SOURCE_JUDGING_SIKL, install_harness

# Writes the record shape `sikl.benchmarks.api --json-report` produces, then exits with
# the code the harness uses for that outcome.
FAKE_SIKL = """import json, sys
path = sys.argv[sys.argv.index("--json-report") + 1]
records = []
for index, (uuid, rows) in enumerate({cases!r}):
    evaluation = {{"status": {status!r}}}
    if {timed!r}:
        evaluation["performance"] = {{
            "latency_ms": 0.01 * (index + 1),
            "reference_latency_ms": 0.01 * (index + 1) * {ratio!r},
        }}
    records.append({{"workload": {{"uuid": uuid, "axes": {{"m": rows}}}}, "evaluation": evaluation}})
open(path, "w").write(json.dumps(records))
sys.exit({exit_code!r})
"""


def _stub_sikl(root: Path, **kwargs) -> Path:
    """Install a `sikl.benchmarks.api` the report module can shell out to."""
    payload = {"cases": list(CASES), "status": "PASSED", "timed": True, "ratio": 2.0, "exit_code": 0}
    payload.update(kwargs)
    install_harness(root, FAKE_SIKL.format(**payload))
    return root


REJECTED_SOURCE = "def run(a, b, bias=None):\n    return 'rejected'\n"


@pytest.fixture
def staged(bundle: Path, tmp_path: Path):
    """A materialized run whose SIKL root (on `_read_verdict`'s PYTHONPATH) holds the stand-in harness,
    with a rejected kernel published last, as the driver leaves it when the final measurement is not kept."""
    definition = load(bundle, DEFINITION_NAME, language="triton")
    run = workspace.materialize(definition, bundle, tmp_path / "run", sikl_root=tmp_path / "sikl")
    published = workspace.candidate_solution(definition, run)
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_text(
        json.dumps(
            {
                "name": workspace.candidate_name(definition),
                "definition": definition.name,
                "author": workspace.CANDIDATE_AUTHOR,
                "spec": {"language": "triton", "entry_point": "kernel.py::run", "dependencies": ["triton==3.4.0"]},
                "sources": [{"path": "kernel.py", "content": REJECTED_SOURCE}],
            }
        ),
        encoding="utf-8",
    )
    sikl_root = tmp_path / "sikl"
    sikl_root.mkdir(exist_ok=True)
    return definition, run, sikl_root


def test_a_candidate_that_misses_the_gate_still_produces_a_summary(staged):
    """The harness exits nonzero for a candidate that misses the gate, the run a reviewer most needs
    the table for; that is the verdict to record, not a failure."""
    definition, run, sikl_root = staged
    _stub_sikl(sikl_root, ratio=0.5, exit_code=1)

    summary = report.finalize(definition, run, sikl_root=sikl_root)

    assert summary.is_file()
    assert (run.root / report.VERDICT_FILENAME).is_file()
    text = summary.read_text(encoding="utf-8")
    assert "0.5000x" in text
    losing = text.split("## Cases this kernel loses", 1)[1].split("##", 1)[0]
    assert "- `m=1`: 0.500x" in losing and "- `m=8`: 0.500x" in losing


def test_an_absent_report_is_a_failure_to_measure(staged):
    """A harness that produced no verdict is not a verdict of zero."""
    definition, run, sikl_root = staged
    install_harness(sikl_root, "import sys\nsys.exit(2)\n")

    with pytest.raises(SystemExit, match="wrote no report"):
        report.finalize(definition, run, sikl_root=sikl_root)


def test_the_table_states_the_unit_its_numbers_are_in(staged):
    """A column headed `ms` holding microseconds is an error a reviewer cannot see, since every row shares it."""
    definition, run, sikl_root = staged
    _stub_sikl(sikl_root)

    summary = report.finalize(definition, run, sikl_root=sikl_root).read_text(encoding="utf-8")
    verdict = json.loads((run.root / report.VERDICT_FILENAME).read_text(encoding="utf-8"))

    first_ms = verdict[0]["evaluation"]["performance"]["latency_ms"]
    assert "candidate us" in summary and "baseline us" in summary
    assert f"| {first_ms * 1000:.2f} |" in summary


def test_geometric_mean_is_reported_over_the_scored_cases(staged):
    definition, run, sikl_root = staged
    _stub_sikl(sikl_root, ratio=2.0)

    summary = report.finalize(definition, run, sikl_root=sikl_root).read_text(encoding="utf-8")

    assert f"2.0000x over {len(CASES)} scored cases" in summary
    assert f"{len(CASES)}/{len(CASES)}" in summary


def test_an_incorrect_case_is_counted_and_never_scored(staged):
    """A case the harness failed carries no speedup, and must not be averaged into one."""
    definition, run, sikl_root = staged
    _stub_sikl(sikl_root, status="INCORRECT_NUMERICAL", timed=False, exit_code=1)

    summary = report.finalize(definition, run, sikl_root=sikl_root).read_text(encoding="utf-8")

    assert f"**Cases failing correctness:** {len(CASES)}" in summary
    assert f"**Cases not scored:** {len(CASES)}" in summary
    assert "0.0000x over 0 scored cases" in summary


def test_the_kernel_scored_is_the_one_the_loop_kept(staged):
    """With lanes, the driver's last publication is often a candidate the loop rejected, not the kernel it kept."""
    definition, run, sikl_root = staged
    run.kernel.write_text("def run(a, b, bias=None):\n    return 'kept'\n", encoding="utf-8")
    install_harness(sikl_root, SOURCE_JUDGING_SIKL.format(passing="kept", cases=list(CASES)))

    summary = report.finalize(definition, run, sikl_root=sikl_root).read_text(encoding="utf-8")

    assert "**Cases failing correctness:** 0" in summary
    published = json.loads(workspace.candidate_solution(definition, run).read_text(encoding="utf-8"))
    assert published["spec"]["dependencies"] == ["triton==3.4.0"], "the spec stays as the driver measured it"


def test_a_report_left_by_an_earlier_pass_is_not_read_as_this_ones(staged):
    """A verdict on disk from a previous pass must not stand in for a harness that failed."""
    definition, run, sikl_root = staged
    _stub_sikl(sikl_root)
    report.finalize(definition, run, sikl_root=sikl_root)
    install_harness(sikl_root, "import sys\nsys.exit(2)\n")

    with pytest.raises(SystemExit, match="wrote no report"):
        report.finalize(definition, run, sikl_root=sikl_root)


# The first case's baseline fails to run, which SIKL reports as a zero reference latency.
BASELINE_LOST_SIKL = """import json, sys
path = sys.argv[sys.argv.index("--json-report") + 1]
records = []
for index, (uuid, rows) in enumerate({cases!r}):
    reference = 0.0 if index == 0 else 0.04
    records.append({{"workload": {{"uuid": uuid, "axes": {{"m": rows}}}},
                    "evaluation": {{"status": "PASSED",
                                   "performance": {{"latency_ms": 0.02, "reference_latency_ms": reference}}}}}})
open(path, "w").write(json.dumps(records))
"""


def test_a_case_whose_baseline_did_not_run_is_named_not_averaged(staged):
    """A missing baseline drops the case from the mean, so the summary names which case left it and why."""
    definition, run, sikl_root = staged
    install_harness(sikl_root, BASELINE_LOST_SIKL.format(cases=list(CASES)))

    summary = report.finalize(definition, run, sikl_root=sikl_root).read_text(encoding="utf-8")

    assert "2.0000x over 1 scored cases" in summary
    unscored = summary.split("## Cases not scored", 1)[1].split("##", 1)[0]
    assert "- `m=1`: the baseline produced no timing" in unscored
    assert "m=8" not in unscored
