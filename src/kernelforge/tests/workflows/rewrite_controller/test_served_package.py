# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A campaign on a snapshot of an installed package measures its own checkout, not the installed copy."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kernelforge.knowledge.kb_store.identity.kernel_recipe import (
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)
from kernelforge.tests.workflows.rewrite_controller.conftest import _git
from kernelforge.workflows.rewrite_controller import ControllerLayout, parse_task_payload
from kernelforge.workflows.rewrite_controller.forge_runner import _child_environment
from kernelforge.workflows.rewrite_controller.paths import operator_directory_name
from kernelforge.workflows.rewrite_controller.served_package import ServedPackageError, shadow_served_package
from kernelforge.workflows.rewrite_controller.worktree import (
    create_operator_worktree,
    export_patch_from_base,
    untracked_paths,
)

_IDENTITY = {
    "producer": "forge-loop",
    "kernel_name": "kern",
    "framework": "vllm",
    "framework_version": "0.29.1",
    "backend": "triton",
    "gpu": "w7900",
}
#: What a driver does: import the kernel by the package's name and use it.
_DRIVER = "import pkgx.kern, pkgx; print(pkgx.kern.VALUE, pkgx.kern.__file__, sep='|')"


def _installed_package(tmp_path: Path) -> Path:
    """A package as a wheel installs it: sources, a subpackage and compiled extension modules, no history."""
    package = tmp_path / "site-packages" / "pkgx"
    (package / "ops").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kern.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "ops" / "__init__.py").write_text("", encoding="utf-8")
    (package / "_ext.abi3.so").write_bytes(b"\x7fELF top-level extension")
    (package / "ops" / "_fast.so").write_bytes(b"\x7fELF nested extension")
    return package


def _snapshot(tmp_path: Path, package: Path) -> tuple[Path, str]:
    """The host's snapshot of ``package``: its sources at the repository root, its extension modules left out."""
    repo = tmp_path / "snapshot"
    repo.mkdir()
    _git(repo, "init")
    for source in sorted(package.rglob("*.py")):
        target = repo / source.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "snapshot")
    return repo, _git(repo, "rev-parse", "HEAD")


def _task(tmp_path: Path, repo: Path, base_commit: str):
    operator_id = kernel_recipe_canonical_id(KernelRecipeIdentity.from_mapping(_IDENTITY))
    task_dir = tmp_path / "output" / "controller" / "tasks" / operator_directory_name(operator_id)
    task_dir.mkdir(parents=True)
    (task_dir / "driver.py").write_text("print('SNR: 100 dB')\n", encoding="utf-8")
    payload = {
        "identity": _IDENTITY,
        "base_commit": base_commit,
        "repo_root": str(repo),
        "kernel_path": "kern.py",
        "operator_name": "kern",
        "driver_path": "driver.py",
        "source_files": ["kern.py"],
        "target_functions": ["kern"],
        "shape_cases": [],
        "priority": 0,
        "reason": "",
        "evidence": [],
    }
    return parse_task_payload(payload, task_dir=task_dir)


def _campaign_worktree(tmp_path: Path, package: Path, *, served: bool = True):
    repo, base_commit = _snapshot(tmp_path, package)
    layout = ControllerLayout(tmp_path / "output")
    return create_operator_worktree(_task(tmp_path, repo, base_commit), layout, package if served else None)


def _run_driver(worktree, package: Path) -> tuple[str, Path]:
    """Run the driver as forge-loop's subprocesses do, with the installed package importable as on a real host."""
    env = _child_environment(worktree.import_root)
    env["PYTHONPATH"] = os.pathsep.join([env["PYTHONPATH"], str(package.parent)])
    done = subprocess.run([sys.executable, "-c", _DRIVER], env=env, capture_output=True, text=True, check=True)
    value, origin = done.stdout.strip().split("|")
    return value, Path(origin).resolve()


def test_a_driver_importing_the_package_by_name_measures_the_campaign_checkout(tmp_path: Path) -> None:
    package = _installed_package(tmp_path)
    worktree = _campaign_worktree(tmp_path, package)
    worktree.kernel_path.write_text("VALUE = 2\n", encoding="utf-8")

    value, origin = _run_driver(worktree, package)

    assert value == "2"
    assert origin == worktree.kernel_path.resolve()
    assert (package / "kern.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_without_the_served_tree_the_driver_measures_the_installed_package(tmp_path: Path) -> None:
    """The failure the served tree exists to prevent: the edit is made and never loaded."""
    package = _installed_package(tmp_path)
    worktree = _campaign_worktree(tmp_path, package, served=False)
    worktree.kernel_path.write_text("VALUE = 2\n", encoding="utf-8")

    value, origin = _run_driver(worktree, package)

    assert worktree.import_root is None
    assert value == "1"
    assert origin == (package / "kern.py").resolve()


def test_the_checkout_carries_the_installed_extension_modules(tmp_path: Path) -> None:
    package = _installed_package(tmp_path)

    worktree = _campaign_worktree(tmp_path, package)

    for relative in ("_ext.abi3.so", "ops/_fast.so"):
        linked = worktree.workspace / relative
        assert linked.is_symlink()
        assert linked.resolve() == (package / relative).resolve()


def test_what_importing_the_checkout_adds_stays_out_of_its_status_and_its_patch(tmp_path: Path) -> None:
    package = _installed_package(tmp_path)
    worktree = _campaign_worktree(tmp_path, package)
    worktree.kernel_path.write_text("VALUE = 2\n", encoding="utf-8")
    _run_driver(worktree, package)

    assert untracked_paths(worktree.workspace) == frozenset()
    _git(worktree.workspace, "commit", "-am", "candidate")
    patch = export_patch_from_base(worktree, best_commit=_git(worktree.workspace, "rev-parse", "HEAD"))

    assert "+VALUE = 2" in patch
    assert ".so" not in patch
    assert "__pycache__" not in patch


def test_a_served_tree_nothing_imports_by_name_is_not_shadowed(tmp_path: Path) -> None:
    package = _installed_package(tmp_path)
    (package / "__init__.py").unlink()

    worktree = _campaign_worktree(tmp_path, package)

    assert worktree.import_root is None


def test_a_checkout_the_package_name_does_not_resolve_to_is_refused(tmp_path: Path) -> None:
    package = _installed_package(tmp_path)
    repo, _ = _snapshot(tmp_path, package)
    (repo / "__init__.py").unlink()

    with pytest.raises(ServedPackageError, match="does not resolve to the campaign checkout"):
        shadow_served_package(repo, package, tmp_path / "imports")


def test_a_campaign_that_cannot_measure_its_checkout_leaves_no_worktree(tmp_path: Path, monkeypatch) -> None:
    package = _installed_package(tmp_path)
    repo, base_commit = _snapshot(tmp_path, package)
    layout = ControllerLayout(tmp_path / "output")
    task = _task(tmp_path, repo, base_commit)

    def _refuse(*_args, **_kwargs):
        raise ServedPackageError("the campaign would measure a tree it does not edit")

    monkeypatch.setattr("kernelforge.workflows.rewrite_controller.worktree.shadow_served_package", _refuse)

    with pytest.raises(ServedPackageError):
        create_operator_worktree(task, layout, package)

    assert not layout.workspace_dir(task.operator_id).exists()
