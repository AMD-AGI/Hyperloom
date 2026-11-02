# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A measurement site moved onto another checkout of the same repository."""

from __future__ import annotations

from pathlib import Path

from kernelforge.workflows.forge_loop.iteration.measurement_site import MeasurementSite


def _site(workspace: Path) -> MeasurementSite:
    return MeasurementSite(
        workspace_dir=str(workspace),
        driver_script=str(workspace / "scripts" / "driver.py"),
        build_dir=str(workspace / "build"),
        declared_sources=(str(workspace / "kernel.py"), "src/helper.py"),
    )


def test_a_rebased_site_names_the_same_task_paths_in_the_other_checkout(tmp_path):
    workspace, start = tmp_path / "workspace", tmp_path / "starts" / "abc"
    workspace.mkdir()

    rebased = _site(workspace).rebased(str(start))

    assert rebased.workspace_dir == str(start)
    assert rebased.driver_script == str(start / "scripts" / "driver.py")
    assert rebased.build_dir == str(start / "build")
    assert rebased.declared_sources == (str(start / "kernel.py"), str(start / "src" / "helper.py"))


def test_a_path_outside_the_workspace_is_the_same_file_from_every_checkout(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shared_build = tmp_path / "shared-build"
    site = MeasurementSite(str(workspace), str(workspace / "driver.py"), str(shared_build), ())

    assert site.rebased(str(tmp_path / "start")).build_dir == str(shared_build)
