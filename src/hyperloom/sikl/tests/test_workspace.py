# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a materialized run guarantees to forge-loop."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.sikl import handoff, workspace
from hyperloom.sikl.definition import BundleError, load
from hyperloom.sikl.tests.conftest import CASES, DEFINITION_NAME, SOURCE_JUDGING_SIKL, install_harness


def _materialize(bundle: Path, tmp_path: Path):
    definition = load(bundle, DEFINITION_NAME, language="triton")
    run = workspace.materialize(definition, bundle, tmp_path / "run", sikl_root=tmp_path / "sikl")
    handoff.write(definition, run.handoff, run.program_md, kernel_backend="triton", starting_point="reference")
    workspace.commit(run)
    return definition, run


def test_bundle_lives_outside_the_loops_workspace(bundle: Path, tmp_path: Path):
    """The driver rewrites a candidate on every measurement; inside the workspace, the guard would
    read that as tampering and fail the analysis rather than the candidate."""
    _definition, run = _materialize(bundle, tmp_path)
    assert run.bundle.exists()
    assert run.workspace not in run.bundle.parents


def test_initial_commit_contains_every_workspace_document(bundle: Path, tmp_path: Path):
    """The loop reverts to this commit; a document written after it would be lost."""
    _definition, run = _materialize(bundle, tmp_path)
    tracked = subprocess.run(
        ["git", "-C", str(run.workspace), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    assert {"kernel.py", "driver.py", "program.md", ".gitignore"} <= set(tracked)


def test_driver_carries_no_unfilled_placeholder(bundle: Path, tmp_path: Path):
    _definition, run = _materialize(bundle, tmp_path)
    text = run.driver.read_text(encoding="utf-8")
    assert "@BUNDLE@" not in text and "@DEFINITION@" not in text
    assert str(run.bundle) in text


def test_starting_kernel_is_the_definitions_reference(bundle: Path, tmp_path: Path):
    """The harness times only cases that pass correctness, and the reference passes every one."""
    definition, run = _materialize(bundle, tmp_path)
    source = run.kernel.read_text(encoding="utf-8")
    assert source.endswith(definition.reference_source)

    namespace: dict = {}
    exec(compile(source, str(run.kernel), "exec"), namespace)
    assert namespace["run"](3, 4) == 12


def _judged_run(bundle: Path, tmp_path: Path, harness: str):
    definition = load(bundle, DEFINITION_NAME, language="triton")
    sikl_root = install_harness(tmp_path / "sikl", harness)
    return definition, workspace.materialize(definition, bundle, tmp_path / "run", sikl_root=sikl_root)


@pytest.mark.parametrize(("passing", "expected"), [("a * b", "reference"), ("a @ b.T", "baseline")])
def test_the_starting_kernel_is_the_first_that_passes_every_workload(bundle, tmp_path, passing, expected):
    """A reference that syncs with the host fails under the harness's graph capture; its baseline may not."""
    harness = SOURCE_JUDGING_SIKL.format(passing=passing, cases=list(CASES))
    definition, run = _judged_run(bundle, tmp_path, harness)
    assert workspace.choose_starting_kernel(definition, run) == expected
    assert passing in run.kernel.read_text(encoding="utf-8")


def test_no_starting_kernel_that_passes_stops_the_run(bundle, tmp_path):
    harness = SOURCE_JUDGING_SIKL.format(passing="neither", cases=list(CASES))
    definition, run = _judged_run(bundle, tmp_path, harness)
    with pytest.raises(workspace.StartingKernelError, match="neither the reference nor the baseline"):
        workspace.choose_starting_kernel(definition, run)


def test_a_harness_without_a_verdict_is_not_read_as_a_failing_kernel(bundle, tmp_path):
    """Falling back would blame the reference for a harness that never ran."""
    definition, run = _judged_run(bundle, tmp_path, "import sys\nsys.exit(3)\n")
    with pytest.raises(workspace.StartingKernelError, match="no verdict"):
        workspace.choose_starting_kernel(definition, run)
    assert "a * b" in run.kernel.read_text(encoding="utf-8")


def test_definition_document_lists_every_scored_workload(bundle: Path, tmp_path: Path):
    definition, run = _materialize(bundle, tmp_path)
    text = (run.handoff / handoff.DEFINITION_FILENAME).read_text(encoding="utf-8")
    for uuid, _rows in CASES:
        assert uuid in text
    assert str(len(definition.workloads)) in text


def test_missing_definition_names_what_is_absent(bundle: Path):
    with pytest.raises(BundleError, match="no definition"):
        load(bundle, "not_a_definition", language="triton")


def test_a_definition_whose_name_prefixes_another_gets_its_own_baseline(bundle: Path):
    """SIKL ships both `gemm_a16w16_nt_n7168_k3584` and `gemm_a16w16_nt_n7168_k35840`."""
    own = next((bundle / "solutions" / "baseline").rglob(f"{DEFINITION_NAME}__baseline.json"))
    longer = own.parents[1] / f"{DEFINITION_NAME}0" / f"{DEFINITION_NAME}0__baseline.json"
    longer.parent.mkdir()
    longer.write_text(own.read_text(encoding="utf-8").replace("a @ b.T", "longer"), encoding="utf-8")
    assert "a @ b.T" in load(bundle, DEFINITION_NAME, language="triton").baseline_source
