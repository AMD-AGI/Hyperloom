# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The sgl-kernel recipe: placed by the rule pack, budgeted and preflighted by the single-lineage runner's recipe."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.measure.runner.native import recipes
from hyperloom.measure.runner.native.recipes import NATIVE_BUILD_REFUSE, sgl_kernel_recipe
from hyperloom.orchestrator.enablement.runtime.build_actions import BuildResult
from hyperloom.orchestrator.enablement.runtime.targeted_build import SglKernelBuild


def test_the_sgl_kernel_recipe_builds_where_the_rule_pack_places_the_package() -> None:
    recipe = sgl_kernel_recipe()

    assert (recipe.source_dir, recipe.package_dir, recipe.outputs) == (
        "python/sglang/kernels/aot",
        "python",
        ("sgl_kernel/*.so",),
    )
    assert (recipe.budget_sec, recipe.max_jobs) == (60 * 60, 8)
    assert recipe.argv("/py") == ["/py", "setup_rocm.py", "build_ext", "--inplace"]
    assert recipe.layout().spec.refused_by("python/sglang/kernels/aot/setup_rocm.py") == "setup_rocm.py"
    assert set(NATIVE_BUILD_REFUSE) == set(recipe.layout().refuse)


def test_a_failed_runner_preflight_is_the_builders_problem(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    failed = BuildResult(ok=False, failure_class="preflight_disk", failure_summary="2.0 GB free, 8.0 GB needed")
    monkeypatch.setattr(SglKernelBuild, "preflight", lambda self: failed)

    checked = sgl_kernel_recipe().preflight(tmp_path, "gfx942")

    assert (checked.problem, checked.toolchain) == ("preflight_disk: 2.0 GB free, 8.0 GB needed", {})


def test_a_passed_runner_preflight_reports_the_toolchain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def passed(self: SglKernelBuild) -> None:
        assert self.gpu_arch == "gfx942"
        self.abi = {"torch_version": "2.9.0+rocm7.0", "hip_version": "7.0.1"}

    monkeypatch.setattr(SglKernelBuild, "preflight", passed)

    checked = recipes.sgl_kernel_recipe().preflight(tmp_path, "gfx942")

    assert (checked.problem, dict(checked.toolchain)) == (None, {"torch": "2.9.0+rocm7.0", "hip": "7.0.1"})
