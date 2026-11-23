# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A lane of a workspace that is imported by its directory's name is laid out, and imported, the same way."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from kernelforge.tests.utils.test_lane_aiter_cache import (
    _campaign,
    _implementer_through,
    _lane_factory,
    _recording_claude_backend,
)
from kernelforge.utils.git import git
from kernelforge.workflows.forge_loop.iteration.fanout import SERIALIZED_DRIVER_NAME, LanePlan, run_lanes

_PRINT_VALUE = "from workspace.src import kernel; print(kernel.VALUE)"


def _value_imported_by(env: dict[str, str]) -> str:
    """What ``workspace.src.kernel`` holds for a process started with ``env``."""
    return subprocess.run(
        [sys.executable, "-c", _PRINT_VALUE], env=env, capture_output=True, text=True, check=True
    ).stdout.strip()


def _commit(repository: Path) -> None:
    git("init", "--quiet", cwd=repository)
    git("config", "user.email", "lane@test", cwd=repository)
    git("config", "user.name", "lane", cwd=repository)
    git("add", "-A", cwd=repository)
    git("commit", "-m", "baseline", cwd=repository)


def _lane_copy(tmp_path: Path, lane_id: str, name: str) -> Path:
    """A lane copy as the fan-out leaves it for a workspace imported as ``name``: its own repository, under that name."""
    lane_dir = tmp_path / "lanes" / lane_id / name
    (lane_dir / "src").mkdir(parents=True)
    (lane_dir / "src" / "kernel.py").write_text(f"VALUE = {lane_id}\n")
    _commit(lane_dir)
    return lane_dir


async def test_a_lane_session_imports_the_lane_by_the_name_its_workspace_is_imported_by(tmp_path, monkeypatch) -> None:
    """Otherwise a driver importing the package by name times the canonical workspace in every lane."""
    config, workspace = _campaign(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(workspace.parent), str(elsewhere))))
    spawned: list[dict[str, str]] = []
    factory = _lane_factory(config, workspace, _implementer_through(_recording_claude_backend(spawned)))

    for lane_id in ("1", "2"):
        lane_dir = _lane_copy(tmp_path, lane_id, workspace.name)
        session = factory(str(lane_dir), str(lane_dir / SERIALIZED_DRIVER_NAME))
        await session(str(lane_dir / "src" / "kernel.py"), f"plan {lane_id}")

    assert [_value_imported_by(env) for env in spawned] == ["1", "2"]
    assert _value_imported_by(dict(os.environ)) == "0"


async def test_a_lane_of_a_workspace_not_imported_by_name_inherits_the_import_path(tmp_path, monkeypatch) -> None:
    config, workspace = _campaign(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "elsewhere"))
    spawned: list[dict[str, str]] = []
    factory = _lane_factory(config, workspace, _implementer_through(_recording_claude_backend(spawned)))
    lane_dir = _lane_copy(tmp_path, "1", "")

    await factory(str(lane_dir), str(lane_dir / SERIALIZED_DRIVER_NAME))(str(lane_dir / "src" / "kernel.py"), "plan")

    assert [env["PYTHONPATH"] for env in spawned] == [str(tmp_path / "elsewhere")]


async def test_a_round_copies_such_a_workspace_under_its_name_with_what_is_linked_beside_it(
    tmp_path, monkeypatch
) -> None:
    """The lane copy is the package by the same name, and finds beside itself what the workspace finds beside itself."""
    import_root = tmp_path / "root"
    workspace = import_root / "served"
    workspace.mkdir(parents=True)
    (workspace / "kernel.py").write_text("VALUE = 0\n")
    (workspace / "forge_driver.py").write_text("pass\n")
    _commit(workspace)
    installed_beside = tmp_path / "site-packages" / "served_meta"
    installed_beside.mkdir(parents=True)
    (import_root / "served_meta").symlink_to(installed_beside, target_is_directory=True)
    monkeypatch.setenv("PYTHONPATH", str(import_root))
    seen: list[tuple[Path, Path]] = []

    async def session(_lane: LanePlan, lane_dir: Path, _driver: Path) -> None:
        seen.append((lane_dir, (lane_dir.parent / "served_meta").resolve()))

    await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "first"), LanePlan("2", "second")],
        session=session,
        parent_dir=str(import_root),
        driver="forge_driver.py",
    )

    assert [lane_dir.name for lane_dir, _beside in seen] == ["served", "served"]
    assert len({lane_dir.parent for lane_dir, _beside in seen}) == 2
    assert {beside for _lane_dir, beside in seen} == {installed_beside.resolve()}
