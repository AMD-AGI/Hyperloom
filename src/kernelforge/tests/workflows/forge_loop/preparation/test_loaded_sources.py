# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The driver preflight refuses a driver that ran a source file of the campaign from outside its workspace."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kernelforge.workflows.forge_loop.bootstrap import run as forge_run
from kernelforge.workflows.forge_loop.bootstrap.errors import ForgeLoopError
from kernelforge.workflows.forge_loop.preparation import task_preflight
from kernelforge.workflows.forge_loop.preparation.loaded_sources import declared_source_modules
from kernelforge.workflows.forge_loop.preparation.task_preflight import PROBE_SOURCE_VIOLATION, DriverContract

_BY_NAME = "import pkgx.kern\n"
#: A driver that puts the installed copy ahead of the campaign's before importing it.
_INSTALLED_FIRST = "import sys\nsys.path.insert(0, {site!r})\nimport pkgx.kern\n"


def _trees(tmp_path: Path) -> tuple[Path, Path]:
    """A campaign workspace that is the package ``pkgx``, and the copy of that package installed on the host."""
    workspace = tmp_path / "root" / "pkgx"
    installed = tmp_path / "site-packages" / "pkgx"
    for package in (workspace, installed):
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "kern.py").write_text("VALUE = 1\n")
    return workspace, installed


def _probe(workspace: Path, driver_source: str) -> tuple[int, str]:
    driver = workspace / "driver.py"
    driver.write_text(driver_source)
    contract = DriverContract(sources=declared_source_modules(str(workspace), [str(workspace / "kern.py")]))
    return task_preflight.asyncio.run(
        task_preflight._count_graph_replays(str(driver), 1, 1, timeout_sec=120, contract=contract)
    )


def test_a_driver_that_ran_the_installed_copy_of_the_kernel_is_refused(tmp_path: Path, monkeypatch) -> None:
    workspace, installed = _trees(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(workspace.parent))

    replays, detail = _probe(workspace, _INSTALLED_FIRST.format(site=str(installed.parent)))

    assert replays == PROBE_SOURCE_VIOLATION
    assert f"pkgx.kern from {installed / 'kern.py'}" in detail


def test_a_driver_that_ran_the_workspace_kernel_passes_the_source_check(tmp_path: Path, monkeypatch) -> None:
    workspace, _installed = _trees(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(workspace.parent))

    replays, _detail = _probe(workspace, _BY_NAME)

    assert replays == 0


def test_source_modules_are_named_as_the_workspace_is_imported(tmp_path: Path, monkeypatch) -> None:
    """Below the workspace's own name when the directory holding it is first on the import path, else from its root."""
    workspace, _installed = _trees(tmp_path)
    (workspace / "ops").mkdir()
    sources = [workspace / "kern.py", workspace / "__init__.py", workspace / "ops" / "k.hip", tmp_path / "other.py"]

    monkeypatch.setenv("PYTHONPATH", str(workspace.parent))
    by_name = declared_source_modules(str(workspace), map(str, sources))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "elsewhere"))
    from_root = declared_source_modules(str(workspace), map(str, sources))

    assert by_name == {"pkgx.kern": str(workspace / "kern.py"), "pkgx": str(workspace / "__init__.py")}
    assert from_root == {"kern": str(workspace / "kern.py")}


def test_import_name_puts_the_directory_holding_the_workspace_first_on_the_import_path(
    tmp_path: Path, monkeypatch
) -> None:
    workspace, _installed = _trees(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/inherited")

    forge_run._put_workspace_first_on_import_path("pkgx", workspace)

    assert os.environ["PYTHONPATH"].split(os.pathsep) == [str(workspace.parent), "/inherited"]


def test_import_name_has_to_be_the_name_of_the_workspace_directory(tmp_path: Path, monkeypatch) -> None:
    workspace, _installed = _trees(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/inherited")

    with pytest.raises(ForgeLoopError, match="--import-name vllm"):
        forge_run._put_workspace_first_on_import_path("vllm", workspace)

    assert os.environ["PYTHONPATH"] == "/inherited"
