# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a run directory holds before forge-loop sees it, and which schema it is built from."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from hyperloom.sikl import handoff, workspace
from hyperloom.sikl.schema import SchemaError, find, load
from hyperloom.sikl.tests.conftest import CASES, SCHEMA_NAME, install_sikl, schema_payload, write_schema

KERNEL = '"""Start."""\n\nimport torch\n\n\ndef run(*, a, b, trans_b, bias, out_dtype, out):\n    torch.matmul(a, b.t(), out=out)\n'


def _materialize(sikl_root: Path, tmp_path: Path, kernel: str = KERNEL) -> workspace.Run:
    schema = load(find(sikl_root, SCHEMA_NAME))
    return workspace.materialize(
        schema, tmp_path / "run", sikl_root=sikl_root, language="flydsl", starting_kernel=kernel
    )


def test_what_the_driver_writes_lives_outside_the_loops_workspace(sikl_root: Path, tmp_path: Path):
    """The driver writes SIKL's reports on every measurement; inside the workspace the guard reads that as tampering."""
    run = _materialize(sikl_root, tmp_path)
    assert run.schema_json.is_file()
    assert run.workspace not in run.schema_json.parents
    assert run.workspace not in run.validate_dir.parents


def test_initial_commit_contains_every_workspace_document(sikl_root: Path, tmp_path: Path):
    run = _materialize(sikl_root, tmp_path)
    schema = load(find(sikl_root, SCHEMA_NAME))
    handoff.write(schema, run.handoff, run.program_md, kernel_backend="flydsl", sikl_root=sikl_root)
    workspace.commit(run)
    tracked = subprocess.run(
        ["git", "-C", str(run.workspace), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    assert sorted(tracked) == [".gitignore", "driver.py", "kernel.py", "program.md"]


def test_driver_carries_no_unfilled_placeholder(sikl_root: Path, tmp_path: Path):
    assert not re.search(r"@[A-Z_]+@", _materialize(sikl_root, tmp_path).driver.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("rank", "statement"), [(2, "torch.matmul(a, b.t()"), (3, "torch.bmm(a, b.transpose(1, 2)")])
def test_a_gemm_schema_starts_from_torch_with_its_contracts_call_keys(tmp_path: Path, rank, statement):
    root = install_sikl(tmp_path / "sikl")
    schema = load(write_schema(root, schema_payload(rank=rank)))
    kernel = workspace.torch_starting_kernel(schema, root)
    assert "def run(*, a, b, trans_b, bias, out_dtype, out):" in kernel
    assert statement in kernel
    compile(kernel, "kernel.py", "exec")


def test_another_op_has_no_built_in_starting_kernel(tmp_path: Path):
    root = install_sikl(tmp_path / "sikl")
    schema = load(write_schema(root, schema_payload("mla_demo", op_type="mla")))
    with pytest.raises(workspace.StartingKernelError, match="--starting-kernel"):
        workspace.torch_starting_kernel(schema, root)


def test_a_starting_kernel_sikl_passes_is_kept(sikl_root: Path, tmp_path: Path):
    workspace.check_starting_kernel(_materialize(sikl_root, tmp_path))


def test_a_starting_kernel_sikl_fails_stops_the_run_with_its_findings(sikl_root: Path, tmp_path: Path):
    run = _materialize(
        sikl_root, tmp_path, kernel='"""Wrong."""\n\n\ndef run(*, a, b, trans_b, bias, out_dtype, out):\n    pass\n'
    )
    with pytest.raises(workspace.StartingKernelError, match="numeric_mismatch"):
        workspace.check_starting_kernel(run)


def test_sikl_without_a_verdict_is_not_read_as_a_failing_kernel(sikl_root: Path, tmp_path: Path):
    run = _materialize(sikl_root, tmp_path)
    (sikl_root / "sikl" / "__main__.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
    with pytest.raises(workspace.StartingKernelError, match="no verdict"):
        workspace.check_starting_kernel(run)


def test_the_schema_found_by_name_is_the_kernel_less_one(sikl_root: Path):
    kernel = [{"path": "kernel.py", "content": "def run():\n    pass\n"}]
    write_schema(sikl_root, schema_payload(sources=kernel), uuid="fedcba9876543210")
    assert not load(find(sikl_root, SCHEMA_NAME)).path.name.startswith("fedcba")


def test_a_missing_schema_names_where_it_looked(sikl_root: Path):
    with pytest.raises(SchemaError, match="no kernel-less schema 'absent'"):
        find(sikl_root, "absent")


def test_two_kernel_less_schemas_of_one_name_are_refused(sikl_root: Path):
    write_schema(sikl_root, schema_payload(), uuid="fedcba9876543210")
    with pytest.raises(SchemaError, match="expected one"):
        find(sikl_root, SCHEMA_NAME)


def test_schema_document_lists_every_scored_workload(sikl_root: Path, tmp_path: Path):
    run = _materialize(sikl_root, tmp_path)
    handoff.write(
        load(find(sikl_root, SCHEMA_NAME)), run.handoff, run.program_md, kernel_backend="flydsl", sikl_root=sikl_root
    )
    document = (run.handoff / handoff.SCHEMA_FILENAME).read_text(encoding="utf-8")
    assert all(f"`{case_id}`" in document for case_id in CASES)
    assert "bitwise" in document
