# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The command surface: what an operator types and what it leaves on disk."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.cli import _COMMANDS
from hyperloom.sikl import cli, forge, workspace
from hyperloom.sikl.definition import load
from hyperloom.sikl.tests.conftest import CASES, DEFINITION_NAME, SOURCE_JUDGING_SIKL, install_harness


def _prepare(bundle: Path, tmp_path: Path, passing: str) -> int:
    install_harness(tmp_path / "sikl", SOURCE_JUDGING_SIKL.format(passing=passing, cases=list(CASES)))
    return cli.main(
        [
            "--bundle",
            str(bundle),
            "--definition",
            DEFINITION_NAME,
            "--sikl-root",
            str(tmp_path / "sikl"),
            "--run-dir",
            str(tmp_path / "run"),
            "--prepare-only",
        ]
    )


def test_subcommand_is_registered():
    module, prefix, help_text = _COMMANDS["sikl"]
    assert module == "hyperloom.sikl.cli"
    assert prefix == ()
    assert "SIKL" in help_text


def test_prepare_only_materializes_without_dispatching(bundle: Path, tmp_path: Path, capsys):
    run_dir = tmp_path / "run"
    assert _prepare(bundle, tmp_path, passing="a @ b.T") == 0
    for relative in ("workspace/kernel.py", "workspace/driver.py", "workspace/program.md", "handoff/definition.md"):
        assert (run_dir / relative).is_file(), relative
    out = capsys.readouterr().out
    assert "scored workloads: 2" in out
    assert "starting kernel: baseline" in out
    program = (run_dir / "workspace" / "program.md").read_text(encoding="utf-8")
    assert "starts as the definition's baseline" in program
    assert "Every GPU kernel `run` launches is one `kernel.py` defines with `@triton.jit`" in program


def test_a_backend_whose_kernels_the_driver_cannot_attribute_is_refused(bundle: Path, tmp_path: Path, capsys):
    with pytest.raises(SystemExit) as exited:
        cli.main(
            [
                "--bundle",
                str(bundle),
                "--definition",
                DEFINITION_NAME,
                "--sikl-root",
                str(tmp_path / "sikl"),
                "--run-dir",
                str(tmp_path / "run"),
                "--kernel-backend",
                "hip",
            ]
        )
    assert exited.value.code == 2
    assert "invalid choice: 'hip'" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_no_starting_kernel_that_passes_exits_two_before_the_first_commit(bundle: Path, tmp_path: Path, capsys):
    assert _prepare(bundle, tmp_path, passing="neither") == 2
    assert "neither the reference nor the baseline" in capsys.readouterr().err
    assert not (tmp_path / "run" / "workspace" / ".git").exists()


def test_unknown_definition_exits_two(bundle: Path, tmp_path: Path, capsys):
    status = cli.main(
        [
            "--bundle",
            str(bundle),
            "--definition",
            "absent",
            "--sikl-root",
            str(tmp_path / "sikl"),
            "--run-dir",
            str(tmp_path / "run"),
            "--prepare-only",
        ]
    )
    assert status == 2
    assert "no definition" in capsys.readouterr().err


def test_a_run_directory_another_run_left_files_in_is_refused(bundle: Path, tmp_path: Path, capsys):
    """The initial commit takes the whole workspace, so a leftover would become the starting point."""
    run_dir = tmp_path / "run"
    leftover = run_dir / "workspace" / "graph_harness.py"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("# another run's file\n", encoding="utf-8")

    status = cli.main(
        [
            "--bundle",
            str(bundle),
            "--definition",
            DEFINITION_NAME,
            "--sikl-root",
            str(tmp_path / "sikl"),
            "--run-dir",
            str(run_dir),
            "--prepare-only",
        ]
    )

    assert status == 2
    assert "not empty" in capsys.readouterr().err
    assert leftover.read_text(encoding="utf-8") == "# another run's file\n"
    assert not (run_dir / "workspace" / "kernel.py").exists()


def test_forge_command_pins_the_driver_and_workspace(bundle: Path, tmp_path: Path):
    """forge-loop treats the driver as the measurement boundary; pass it explicitly."""
    definition = load(bundle, DEFINITION_NAME, language="triton")
    run = workspace.materialize(definition, bundle, tmp_path / "run", sikl_root=tmp_path / "sikl")
    command = forge.build_command(
        run, kernel_backend="triton", gpu_target="gfx950", max_hours=1.0, git_branch="forge-optimize"
    )
    assert "forge-loop" in command
    assert command[command.index("--driver") + 1] == str(run.driver)
    assert command[command.index("--workspace") + 1] == str(run.workspace)
    assert "--model" not in command
