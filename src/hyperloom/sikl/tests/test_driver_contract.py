# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The driver's output is read by KernelForge's own parsers, so assert with those: a line they skip
loses its case silently."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

from hyperloom.sikl import workspace
from hyperloom.sikl.definition import load
from hyperloom.sikl.tests.conftest import CASES, DEFINITION_NAME, install_harness
from kernelforge.mcp_server.tools.bench import parse_case_timings

# The harness stand-in: emits the record shape `sikl.benchmarks.api --json-report` writes.
FAKE_SIKL = """import json, sys
report = sys.argv[sys.argv.index("--json-report") + 1]
timed = "--time-baseline" in sys.argv
records = []
for index, (uuid, rows) in enumerate({cases!r}):
    evaluation = {{"status": "PASSED", "correctness": {{"max_absolute_error": 0.5 * (index + 1)}}}}
    if timed:
        evaluation["performance"] = {{"latency_ms": 0.01 * (index + 1), "reference_latency_ms": 0.02 * (index + 1)}}
    records.append({{"workload": {{"uuid": uuid, "axes": {{"m": rows}}}}, "evaluation": evaluation}})
open(report, "w").write(json.dumps(records))
"""


def _materialize(bundle: Path, tmp_path: Path):
    definition = load(bundle, DEFINITION_NAME, language="triton")
    return definition, workspace.materialize(definition, bundle, tmp_path / "run", sikl_root=tmp_path / "sikl")


def _run_driver(run, fake_sikl: Path, *args: str, expect_rc: int = 0) -> str:
    """Invoke the driver with `sikl.benchmarks.api` replaced by the stand-in."""
    stub_root = install_harness(fake_sikl.parent, fake_sikl.read_text(encoding="utf-8"))
    completed = subprocess.run(
        [sys.executable, str(run.driver), *args],
        cwd=run.workspace,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(stub_root)},
    )
    assert completed.returncode == expect_rc, completed.stdout + completed.stderr
    return completed.stdout


@pytest.fixture
def driver_run(bundle: Path, tmp_path: Path):
    definition, run = _materialize(bundle, tmp_path)
    # The driver pins SIKL_ROOT at materialization; point it at the stand-in package.
    stub_root = tmp_path / "stub"
    stub_root.mkdir()
    text = run.driver.read_text(encoding="utf-8").replace(str(tmp_path / "sikl"), str(stub_root))
    run.driver.write_text(text, encoding="utf-8")
    fake = stub_root / "fake_api.py"
    fake.write_text(FAKE_SIKL.format(cases=list(CASES)), encoding="utf-8")
    return definition, run, fake


def test_bench_output_parses_into_every_case(driver_run):
    """`parse_case_timings` must find every scored workload, none unscored."""
    _definition, run, fake = driver_run
    stdout = _run_driver(run, fake, "--bench-mode", "--warmup", "1", "--iters", "2")
    case_times, unscored, duplicates = parse_case_timings(stdout)
    assert set(case_times) == {uuid for uuid, _ in CASES}
    assert not unscored and not duplicates
    assert all(value > 0 for value in case_times.values())


def test_correctness_output_carries_the_allclose_fallback(driver_run):
    """SIKL reports a matched ratio, not an SNR; the loop reads allclose instead."""
    _definition, run, fake = driver_run
    stdout = _run_driver(
        run,
        fake,
    )
    assert re.search(r"allclose:\s*(True|False)", stdout)
    assert re.search(r"max_diff:\s*([\d.eE+-]+)", stdout)
    assert "SNR:" not in stdout


def test_bench_records_the_in_window_baseline(driver_run):
    """The harness times the baseline beside the candidate; keep that number."""
    _definition, run, fake = driver_run
    _run_driver(run, fake, "--bench-mode", "--warmup", "1", "--iters", "2")
    record = json.loads((run.bundle.parent / "sikl_in_window_baseline.json").read_text(encoding="utf-8"))
    assert set(record) == {uuid for uuid, _ in CASES}


def test_driver_is_importable_without_a_gpu(driver_run):
    """forge-loop hashes the driver before it ever runs; importing must not need ROCm."""
    _definition, run, _fake = driver_run
    spec = importlib.util.spec_from_file_location("sikl_forge_driver", run.driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DEFINITION == DEFINITION_NAME


def _has_metadata(distribution: str) -> bool:
    try:
        metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return False
    return True


@pytest.mark.skipif(not _has_metadata("triton"), reason="the driver records the installed triton; none is installed")
def test_candidate_records_the_language_version_it_was_measured_with(driver_run):
    """A kernel may not load under another version of its language, so a candidate that declares
    none cannot be re-measured by its reviewer."""
    definition, run, fake = driver_run
    _run_driver(run, fake, "--bench-mode", "--warmup", "1", "--iters", "2")
    published = (
        run.bundle
        / "solutions"
        / "forge-candidate"
        / definition.op_type
        / definition.name
        / f"{definition.name}__forge.json"
    )
    spec = json.loads(published.read_text(encoding="utf-8"))["spec"]
    assert spec["language"] == "triton"
    assert spec["entry_point"] == "kernel.py::run"
    assert [item for item in spec["dependencies"] if item.startswith("triton==")], spec["dependencies"]


FAILING_SIKL = """import json, sys
report = sys.argv[sys.argv.index("--json-report") + 1]
records = [{{"workload": {{"uuid": uuid, "axes": {{"m": rows}}}},
            "evaluation": {{"status": "INCORRECT_NUMERICAL",
                           "correctness": {{"max_absolute_error": 2.0}}}}}}
           for uuid, rows in {cases!r}]
open(report, "w").write(json.dumps(records))
sys.exit(1)
"""


def test_a_rejected_candidate_is_a_verdict_not_a_driver_crash(driver_run):
    """The harness exits nonzero for a candidate that does not qualify, which is a
    `correctness_failure` to the loop, not a `driver_error`."""
    _definition, run, fake = driver_run
    fake.write_text(FAILING_SIKL.format(cases=list(CASES)), encoding="utf-8")
    stdout = _run_driver(run, fake)
    assert "allclose: False" in stdout
    assert "max_diff: 2" in stdout


def test_bench_emits_the_suite_aggregate(driver_run):
    """Without an aggregate the prepare phase reads the task as unbenchmarkable and rewrites the
    driver, after which the measurement is no longer SIKL's."""
    _definition, run, fake = driver_run
    stdout = _run_driver(run, fake, "--bench-mode", "--warmup", "1", "--iters", "2")
    case_times, _unscored, _duplicates = parse_case_timings(stdout)
    aggregate = re.search(r"(median_ms|mean_ms):\s*([\d.eE+-]+)", stdout)
    assert aggregate, stdout
    assert float(aggregate.group(2)) == pytest.approx(sum(case_times.values()))
