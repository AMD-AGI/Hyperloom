# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The sgl-kernel and vLLM recipes: placed by the rule pack, budgeted and preflighted by the single-lineage runner's
recipes."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.measure.runner.native import recipes
from hyperloom.measure.runner.native.recipes import (
    NATIVE_BUILD_REFUSE,
    RECIPE_FACTORIES,
    native_recipes,
    rocm_variables,
    sgl_kernel_recipe,
    vllm_recipe,
)
from hyperloom.orchestrator.enablement.runtime.build_actions import BuildResult
from hyperloom.orchestrator.enablement.runtime.targeted_build import SglKernelBuild, VllmBuild


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


def test_the_vllm_recipe_builds_at_the_top_of_the_repository_where_the_rule_pack_places_the_package() -> None:
    recipe = vllm_recipe()

    assert (recipe.component, recipe.import_name, recipe.source_dir, recipe.package_dir, recipe.outputs) == (
        "vllm",
        "vllm",
        "",
        "",
        ("vllm/*.so",),
    )
    assert (recipe.budget_sec, recipe.max_jobs) == (5400, VllmBuild.default_max_jobs)
    assert recipe.argv("/py") == ["/py", "setup.py", "build_ext", "--inplace"]
    for definition in ("setup.py", "CMakeLists.txt", "cmake/utils.cmake"):
        assert recipe.layout().spec.refused_by(definition) is not None, definition


def test_the_vllm_compile_is_given_the_arch_and_the_rocm_root_as_the_single_lineage_build_names_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SECRET_TOKEN", "not for the compile")
    for name in ("ROCM_HOME", "CUDA_HOME", "ROCM_PATH", "HIP_PATH"):
        monkeypatch.delenv(name, raising=False)

    environment = vllm_recipe().environment("gfx950", tmp_path)

    assert environment["PYTORCH_ROCM_ARCH"] == "gfx950" and environment["MAX_JOBS"] == "8"
    assert {environment[n] for n in ("CUDA_HOME", "ROCM_HOME", "ROCM_PATH", "HIP_HOME")} == {"/opt/rocm"}
    assert "SECRET_TOKEN" not in environment


@pytest.mark.parametrize(
    ("environ", "root"),
    [
        ({"ROCM_PATH": "/r/path", "CUDA_HOME": "/r/cuda"}, "/r/cuda"),
        ({"ROCM_HOME": " ", "HIP_PATH": "/r/hip"}, "/r/hip"),
        ({}, "/opt/rocm"),
    ],
    ids=["first-set-wins", "blank-is-unset", "default"],
)
def test_the_rocm_root_is_the_first_variable_set(environ: dict[str, str], root: str) -> None:
    assert rocm_variables(environ) == dict.fromkeys(("CUDA_HOME", "ROCM_HOME", "ROCM_PATH", "HIP_HOME"), root)


def test_a_failed_vllm_runner_preflight_is_the_builders_problem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failed = BuildResult(ok=False, failure_class="preflight_disk", failure_summary="5.0 GB free, 20.0 GB needed")
    monkeypatch.setattr(VllmBuild, "preflight", lambda self: failed)

    checked = vllm_recipe().preflight(tmp_path, "gfx950")

    assert (checked.problem, checked.toolchain) == ("preflight_disk: 5.0 GB free, 20.0 GB needed", {})


def _vllm_runner_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    def passed(self: VllmBuild) -> None:
        assert (self.gpu_arch, self.disk_gb) == ("gfx950", 20.0)
        self.abi = {"torch_version": "2.10.0+git8514f05", "hip_version": "7.2.53211"}

    monkeypatch.setattr(VllmBuild, "preflight", passed)


def test_a_passed_vllm_preflight_reports_the_toolchain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _vllm_runner_passes(monkeypatch)
    monkeypatch.setattr(recipes.shutil, "which", lambda tool: f"/usr/bin/{tool}")

    checked = vllm_recipe().preflight(tmp_path, "gfx950")

    assert (checked.problem, dict(checked.toolchain)) == (None, {"torch": "2.10.0+git8514f05", "hip": "7.2.53211"})


def test_a_vllm_builder_without_cmake_or_ninja_fails_as_the_machine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _vllm_runner_passes(monkeypatch)
    monkeypatch.setattr(recipes.shutil, "which", lambda tool: None if tool == "ninja" else f"/usr/bin/{tool}")

    checked = vllm_recipe().preflight(tmp_path, "gfx950")

    assert checked.problem == "preflight_toolchain: ninja not on PATH; vLLM's build runs them"


def test_every_factory_makes_its_components_recipe() -> None:
    made = native_recipes()

    assert set(made) == set(RECIPE_FACTORIES) == {"sgl_kernel", "vllm"}
    assert {component: recipe.component for component, recipe in made.items()} == {c: c for c in made}
