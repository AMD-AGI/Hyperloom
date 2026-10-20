# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The command surface: what an operator types and what it leaves on disk."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.cli import _COMMANDS
from hyperloom.sikl import cli
from hyperloom.sikl.tests.conftest import SCHEMA_NAME, install_sikl, schema_payload, write_schema


def _prepare(sikl_root: Path, tmp_path: Path, *extra: str) -> int:
    return cli.main(
        [
            "--sikl-root",
            str(sikl_root),
            "--schema",
            SCHEMA_NAME,
            "--run-dir",
            str(tmp_path / "run"),
            "--prepare-only",
            *extra,
        ]
    )


def test_subcommand_is_registered():
    module, prefix, help_text = _COMMANDS["sikl"]
    assert module == "hyperloom.sikl.cli"
    assert prefix == ()
    assert "SIKL" in help_text


def test_prepare_only_materializes_without_dispatching(sikl_root: Path, tmp_path: Path, capsys):
    run_dir = tmp_path / "run"
    assert _prepare(sikl_root, tmp_path) == 0
    for relative in ("workspace/kernel.py", "workspace/driver.py", "workspace/program.md", "handoff/schema.md"):
        assert (run_dir / relative).is_file(), relative
    assert (run_dir / "workspace" / ".git").is_dir()
    assert "scored workloads: 2" in capsys.readouterr().out
    program = (run_dir / "workspace" / "program.md").read_text(encoding="utf-8")
    assert "Every GPU kernel `run` launches is one `kernel.py` defines with `@flydsl.compiler.kernel`" in program


def test_a_schema_file_from_another_checkout_is_read_with_this_ones_contracts(sikl_root: Path, tmp_path: Path):
    other = write_schema(install_sikl(tmp_path / "other"), schema_payload())
    status = cli.main(
        [
            "--sikl-root",
            str(sikl_root),
            "--schema-file",
            str(other),
            "--run-dir",
            str(tmp_path / "run"),
            "--prepare-only",
        ]
    )
    assert status == 0


def test_a_backend_whose_kernels_the_driver_cannot_attribute_is_refused(sikl_root: Path, tmp_path: Path, capsys):
    with pytest.raises(SystemExit) as exited:
        _prepare(sikl_root, tmp_path, "--kernel-backend", "hip")
    assert exited.value.code == 2
    assert "invalid choice: 'hip'" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_a_starting_kernel_sikl_fails_exits_two_before_the_first_commit(sikl_root: Path, tmp_path: Path, capsys):
    wrong = tmp_path / "wrong.py"
    wrong.write_text('"""Wrong."""\n\n\ndef run(*, a, b, trans_b, bias, out_dtype, out):\n    pass\n', encoding="utf-8")
    assert _prepare(sikl_root, tmp_path, "--starting-kernel", str(wrong)) == 2
    assert "does not pass the starting kernel" in capsys.readouterr().err
    assert not (tmp_path / "run" / "workspace" / ".git").exists()


def test_an_op_without_a_built_in_start_needs_one_given(tmp_path: Path, capsys):
    root = install_sikl(tmp_path / "sikl")
    write_schema(root, schema_payload("mla_demo", op_type="mla"))
    status = cli.main(
        ["--sikl-root", str(root), "--schema", "mla_demo", "--run-dir", str(tmp_path / "run"), "--prepare-only"]
    )
    assert status == 2
    assert "pass --starting-kernel" in capsys.readouterr().err


def test_unknown_schema_exits_two(sikl_root: Path, tmp_path: Path, capsys):
    status = cli.main(["--sikl-root", str(sikl_root), "--schema", "absent", "--run-dir", str(tmp_path / "run")])
    assert status == 2
    assert "no kernel-less schema 'absent'" in capsys.readouterr().err


def test_a_run_directory_another_run_left_files_in_is_refused(sikl_root: Path, tmp_path: Path, capsys):
    """The initial commit takes the whole workspace, so a leftover would become the starting point."""
    leftover = tmp_path / "run" / "workspace" / "graph_harness.py"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("# another run's file\n", encoding="utf-8")
    assert _prepare(sikl_root, tmp_path) == 2
    assert "is not empty" in capsys.readouterr().err
