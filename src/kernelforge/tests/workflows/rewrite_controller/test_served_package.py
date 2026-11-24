# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A campaign on a snapshot of an installed package measures its own checkout, not the installed copy."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from kernelforge.knowledge.kb_store.identity.kernel_recipe import (
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)
from kernelforge.tests.workflows.rewrite_controller.conftest import _git
from kernelforge.utils.import_root import import_path_first
from kernelforge.workflows.rewrite_controller import ControllerLayout, TaskStateStore, dispatcher
from kernelforge.workflows.rewrite_controller.forge_runner import ForgeLoopOutcome
from kernelforge.workflows.rewrite_controller.paths import operator_directory_name
from kernelforge.workflows.rewrite_controller.recovery import RecoveryResult
from kernelforge.workflows.rewrite_controller.task import load_task
from kernelforge.workflows.rewrite_controller.worktree import (
    OperatorWorktree,
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
#: A package reading a directory installed beside it, found from where the import system says the package is and from
#: where the filesystem resolves it to, as aiter finds ``aiter_meta`` both ways.
_BESIDE = (
    "from pathlib import Path\n"
    "NAMED = (Path(__file__).absolute().parents[1] / 'pkgx_meta' / 'data.txt').read_text()\n"
    "REAL = (Path(__file__).resolve().parents[1] / 'pkgx_meta' / 'data.txt').read_text()\n"
)
_READ_BESIDE = "import pkgx.beside as b; print(b.NAMED, b.REAL, sep='|')"
#: What the host's snapshot of an installed tree keeps out of its repository.
_SNAPSHOT_EXCLUDES = ("__pycache__/", "*.pyc", "*.pyo", "*.so", "*.so.*")


def _installed(tmp_path: Path) -> Path:
    """A package as a wheel installs it, with compiled extension modules and a data directory beside it."""
    site = tmp_path / "site-packages"
    package = site / "pkgx"
    (package / "ops").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kern.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "beside.py").write_text(_BESIDE, encoding="utf-8")
    (package / "ops" / "__init__.py").write_text("", encoding="utf-8")
    (package / "_ext.abi3.so").write_bytes(b"\x7fELF top-level extension")
    (package / "ops" / "_fast.so").write_bytes(b"\x7fELF nested extension")
    (site / "pkgx_meta").mkdir()
    (site / "pkgx_meta" / "data.txt").write_text("installed", encoding="utf-8")
    return package


def _snapshot(tmp_path: Path, tree: Path) -> Path:
    """The host's snapshot of ``tree``: its files at the repository root, under the excludes that repository carries."""
    repo = tmp_path / "tree_snapshots" / f"{tree.name}.campaign"
    shutil.copytree(tree, repo)
    _git(repo, "init")
    (repo / ".git" / "info" / "exclude").write_text("".join(f"{line}\n" for line in _SNAPSHOT_EXCLUDES))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "snapshot")
    return repo


def _host(tmp_path: Path) -> tuple[ControllerLayout, dict[str, Path]]:
    """A controller given the snapshots of an installed package and of the directory installed beside it."""
    package = _installed(tmp_path)
    snapshots = {tree.name: _snapshot(tmp_path, tree) for tree in (package, package.parent / "pkgx_meta")}
    served = {snapshots[name]: package.parent / name for name in snapshots}
    return ControllerLayout(tmp_path / "output", served_trees=served), snapshots


def _publish_task(layout: ControllerLayout, repo: Path, kernel_path: str) -> Path:
    operator_id = kernel_recipe_canonical_id(KernelRecipeIdentity.from_mapping(_IDENTITY))
    task_dir = layout.tasks_root / operator_directory_name(operator_id)
    task_dir.mkdir(parents=True)
    (task_dir / "driver.py").write_text("print('SNR: 100 dB')\n", encoding="utf-8")
    payload = {
        "identity": _IDENTITY,
        "base_commit": _git(repo, "rev-parse", "HEAD"),
        "repo_root": str(repo),
        "kernel_path": kernel_path,
        "operator_name": "kern",
        "driver_path": "driver.py",
        "source_files": [kernel_path],
        "target_functions": ["kern"],
        "shape_cases": [],
        "priority": 0,
        "reason": "",
        "evidence": [],
    }
    (task_dir / "task.json").write_text(json.dumps(payload), encoding="utf-8")
    return task_dir


def _campaign_worktree(layout: ControllerLayout, repo: Path, kernel_path: str = "kern.py") -> OperatorWorktree:
    task = load_task(_publish_task(layout, repo, kernel_path), record_state=False).task
    return create_operator_worktree(task, layout)


def _run(worktree: OperatorWorktree, code: str) -> list[str]:
    """Run ``code`` as the campaign runs its driver: the directory holding the checkout ahead of the host's packages."""
    site = worktree.repo_root.parents[1] / "site-packages"
    env = {**os.environ, "PYTHONPATH": import_path_first(worktree.workspace.parent, str(site))}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    return done.stdout.strip().split("|")


def test_a_driver_importing_the_package_by_name_measures_the_campaign_checkout(tmp_path: Path) -> None:
    layout, snapshots = _host(tmp_path)
    worktree = _campaign_worktree(layout, snapshots["pkgx"])
    worktree.kernel_path.write_text("VALUE = 2\n", encoding="utf-8")

    value, origin = _run(worktree, _DRIVER)

    assert worktree.import_name == worktree.workspace.name == "pkgx"
    assert value == "2"
    assert Path(origin) == worktree.kernel_path
    assert (tmp_path / "site-packages" / "pkgx" / "kern.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_the_checkout_carries_the_installed_extension_modules(tmp_path: Path) -> None:
    layout, snapshots = _host(tmp_path)

    worktree = _campaign_worktree(layout, snapshots["pkgx"])

    for relative in ("_ext.abi3.so", "ops/_fast.so"):
        linked = worktree.workspace / relative
        assert linked.is_symlink()
        assert linked.resolve() == (tmp_path / "site-packages" / "pkgx" / relative).resolve()


def test_what_importing_the_checkout_adds_stays_out_of_its_status_and_its_patch(tmp_path: Path) -> None:
    layout, snapshots = _host(tmp_path)
    worktree = _campaign_worktree(layout, snapshots["pkgx"])
    worktree.kernel_path.write_text("VALUE = 2\n", encoding="utf-8")
    _run(worktree, _DRIVER)

    assert untracked_paths(worktree.workspace) == frozenset()
    _git(worktree.workspace, "commit", "-am", "candidate")
    patch = export_patch_from_base(worktree, best_commit=_git(worktree.workspace, "rev-parse", "HEAD"))

    assert "+VALUE = 2" in patch
    assert ".so" not in patch
    assert "__pycache__" not in patch


def test_a_package_finds_beside_the_checkout_what_is_installed_beside_it(tmp_path: Path) -> None:
    """The checkout is a directory of the package's own name, so the package has one location to look beside."""
    layout, snapshots = _host(tmp_path)

    worktree = _campaign_worktree(layout, snapshots["pkgx"])

    assert _run(worktree, _READ_BESIDE) == ["installed", "installed"]


def test_a_campaign_on_what_is_installed_beside_a_package_is_what_that_package_reads(tmp_path: Path) -> None:
    """A tree nothing imports by name is reached through the package beside it, which is linked beside the checkout."""
    layout, snapshots = _host(tmp_path)
    worktree = _campaign_worktree(layout, snapshots["pkgx_meta"], kernel_path="data.txt")
    worktree.kernel_path.write_text("edited", encoding="utf-8")

    named, _real = _run(worktree, _READ_BESIDE)

    assert worktree.workspace.name == "pkgx_meta"
    assert named == "edited"


def test_the_controller_runs_forge_loop_on_the_checkout_under_the_package_name(tmp_path: Path, monkeypatch) -> None:
    """From a published task to the forge-loop command: the checkout stands under the package's name, forge-loop is
    told that name, and the task's recorded workspace is where the result is later recovered from."""
    layout, snapshots = _host(tmp_path)
    task_dir = _publish_task(layout, snapshots["pkgx"], "kern.py")
    commands: list[tuple[str, ...]] = []

    def _run_forge_loop(invocation, **_kwargs) -> ForgeLoopOutcome:
        commands.append(invocation.command)
        return ForgeLoopOutcome(returncode=1, stdout="", stderr="stop", result=None, timed_out=False, command=())

    monkeypatch.setattr(dispatcher, "run_forge_loop", _run_forge_loop)
    monkeypatch.setattr(
        dispatcher,
        "recover_task_result",
        lambda *_args, **_kwargs: RecoveryResult(operator_id="none", published=False, patch_dir=None, best_commit=""),
    )

    result = dispatcher.dispatch_single_task(task_dir, layout=layout, deadline_unix=10**12)

    checkout = layout.workspace_dir(result.task.operator_id) / "pkgx"
    (command,) = commands
    assert command[command.index("--workspace") + 1] == str(checkout)
    assert command[command.index("--import-name") + 1] == "pkgx"
    assert TaskStateStore(task_dir).load().workspace_dir == str(checkout)
    assert (checkout.parent / "pkgx_meta").resolve() == (tmp_path / "site-packages" / "pkgx_meta").resolve()
