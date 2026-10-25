# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What compiling one component's native tree takes: where it builds, with what, within which budget, and what it makes.

The sgl-kernel recipe reuses the single-lineage runner's (:class:`~hyperloom.orchestrator.enablement.runtime.targeted_build.
SglKernelBuild`): its disk and toolchain preflight, its ``MAX_JOBS`` default, its wall-clock budget and the variable it
names the GPU architecture by. Its front and back are new: the tree is a copy of the pinned upstream checkout with the
stack laid over it, the compile builds the extension in place instead of installing into a venv, and the outputs are
harvested by glob.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from hyperloom.common.outside_facts import component_layouts
from hyperloom.measure.compile import ComponentLayout
from hyperloom.orchestrator.enablement.runtime.build_actions import TargetedBuildAction
from hyperloom.orchestrator.enablement.runtime.targeted_build import SglKernelBuild, default_budget_sec

__all__ = ["NATIVE_BUILD_REFUSE", "NativeRecipe", "Preflight", "sgl_kernel_recipe"]

#: Files a native stack may not change in v1: the build definition runs arbitrary code at build time, with the widest
#: blast radius; matched on the file name, as a component layout's ``refuse`` is.
NATIVE_BUILD_REFUSE = ("CMakeLists.txt", "*.cmake", "setup.py", "setup_rocm.py", "pyproject.toml", "setup.cfg")
PYTHON_PLACEHOLDER = "{python}"
_SGL_KERNEL = "sgl_kernel"
_SGL_KERNEL_RECIPE_VERSION = "sgl_kernel/1"
_SGL_KERNEL_MAX_OUTPUT_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class Preflight:
    """The outcome of a recipe's machine checks: ``problem`` is an INFRA failure, ``toolchain`` the facts to record."""

    problem: str | None
    toolchain: Mapping[str, str]


@dataclass(frozen=True)
class NativeRecipe:
    """One component's compile. Paths are POSIX paths in the upstream tree; ``outputs`` are globs below ``package_dir``."""

    component: str
    import_name: str
    version: str
    source_dir: str
    package_dir: str
    outputs: tuple[str, ...]
    command: tuple[str, ...]
    budget_sec: int
    max_jobs: int
    max_output_bytes: int
    preflight: Callable[[Path, str], Preflight]
    gfx_variable: str = "AMDGPU_TARGET"

    def layout(self) -> ComponentLayout:
        """The native section: the build directory of the upstream tree, the build definition refused."""
        return ComponentLayout(self.import_name, self.source_dir, refuse=NATIVE_BUILD_REFUSE)

    def argv(self, python: str) -> list[str]:
        return [python if part == PYTHON_PLACEHOLDER else part for part in self.command]

    def environment(self, gfx_arch: str, scratch: Path) -> dict[str, str]:
        """The compile's whole environment: no variable of the service (none of its credentials) reaches it."""
        return {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": str(scratch),
            "TMPDIR": str(scratch),
            "LC_ALL": "C.UTF-8",
            "MAX_JOBS": str(self.max_jobs),
            self.gfx_variable: gfx_arch,
        }


def sgl_kernel_recipe() -> NativeRecipe:
    """sgl-kernel's in-place extension build, in the directory the rule pack places its Python package in."""
    package = PurePosixPath(component_layouts()[_SGL_KERNEL].sections[0].prefix)
    return NativeRecipe(
        component=_SGL_KERNEL,
        import_name=_SGL_KERNEL,
        version=_SGL_KERNEL_RECIPE_VERSION,
        source_dir=str(package.parent.parent),
        package_dir=package.parent.name,
        outputs=(f"{package.name}/*.so",),
        command=(PYTHON_PLACEHOLDER, "setup_rocm.py", "build_ext", "--inplace"),
        budget_sec=default_budget_sec(_SGL_KERNEL),
        max_jobs=SglKernelBuild.default_max_jobs,
        max_output_bytes=_SGL_KERNEL_MAX_OUTPUT_BYTES,
        preflight=_sgl_kernel_preflight,
    )


def _sgl_kernel_preflight(scratch: Path, gfx_arch: str) -> Preflight:
    """The single-lineage recipe's own preflight: disk for one candidate, the ROCm toolchain, a ROCm torch, the arch."""
    action = TargetedBuildAction(
        gap_id="native-build", framework="sglang", component=_SGL_KERNEL, capability="native", gpu_arch=gfx_arch
    )
    recipe = SglKernelBuild(action, str(scratch / "preflight"), run=subprocess.run, git=None, disk_preflight_fn=None)
    failed = recipe.preflight()
    if failed is not None:
        return Preflight(f"{failed.failure_class}: {failed.failure_summary}", {})
    abi = recipe.abi
    return Preflight(None, {"torch": str(abi.get("torch_version", "")), "hip": str(abi.get("hip_version", ""))})
