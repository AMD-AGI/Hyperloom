# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What compiling one component's native tree takes: where it builds, with what, within which budget, and what it makes.

The sgl-kernel and vLLM recipes reuse the single-lineage runner's (:class:`~hyperloom.orchestrator.enablement.runtime.
targeted_build.SglKernelBuild` and :class:`~hyperloom.orchestrator.enablement.runtime.targeted_build.VllmBuild`): their
disk and toolchain preflight, their ``MAX_JOBS`` default, their wall-clock budget and the variables they name the GPU
architecture and the ROCm root by. Their front and back are new: the tree is a copy of the pinned upstream checkout with
the stack laid over it, the compile builds the extensions in place instead of installing into a venv, and the outputs
are harvested by glob.

:data:`RECIPE_FACTORIES` names every recipe's factory as the build runner's child imports it; :func:`native_recipes`
makes them all.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath

from hyperloom.common.outside_facts import component_layouts
from hyperloom.measure.compile import ComponentLayout
from hyperloom.orchestrator.enablement.runtime.build_actions import TargetedBuildAction
from hyperloom.orchestrator.enablement.runtime.targeted_build import SglKernelBuild, VllmBuild, default_budget_sec

__all__ = [
    "NATIVE_BUILD_REFUSE",
    "RECIPE_FACTORIES",
    "NativeRecipe",
    "Preflight",
    "native_recipes",
    "rocm_variables",
    "runner_preflight",
    "sgl_kernel_recipe",
    "vllm_recipe",
]

#: Files a native stack may not change in v1: the build definition runs arbitrary code at build time, with the widest
#: blast radius; matched on the file name, as a component layout's ``refuse`` is.
NATIVE_BUILD_REFUSE = ("CMakeLists.txt", "*.cmake", "setup.py", "setup_rocm.py", "pyproject.toml", "setup.cfg")
PYTHON_PLACEHOLDER = "{python}"
_SGL_KERNEL = "sgl_kernel"
_SGL_KERNEL_RECIPE_VERSION = "sgl_kernel/1"
_VLLM = "vllm"
_VLLM_RECIPE_VERSION = "vllm/1"
#: The single-lineage runner's name of the vLLM build, which its budgets are kept under.
_VLLM_SOURCE = "vllm_source"
#: The tools vLLM's ``setup.py`` drives the compile with; with no egress in the build they must be in the image.
_VLLM_BUILD_TOOLS = ("cmake", "ninja")
_MAX_OUTPUT_BYTES = 2 * 1024**3
_IN_PLACE = ("build_ext", "--inplace")
#: The variables that name the ROCm root, in the order the single-lineage vLLM build reads them, and where it is when
#: none is set.
_ROCM_ROOT_VARIABLES = ("ROCM_HOME", "CUDA_HOME", "ROCM_PATH", "HIP_PATH")
_ROCM_ROOT_SET = ("CUDA_HOME", "ROCM_HOME", "ROCM_PATH", "HIP_HOME")
_DEFAULT_ROCM_ROOT = "/opt/rocm"


@dataclass(frozen=True)
class Preflight:
    """The outcome of a recipe's machine checks: ``problem`` is an INFRA failure, ``toolchain`` the facts to record."""

    problem: str | None
    toolchain: Mapping[str, str]


@dataclass(frozen=True)
class NativeRecipe:
    """One component's compile. Paths are POSIX paths in the upstream tree, ``""`` its top (``source_dir``) or the top of
    the build directory (``package_dir``); ``outputs`` are globs below ``package_dir``."""

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
    #: The recipe's own variables of the compile, read when the compile starts.
    variables: Callable[[], Mapping[str, str]] = dict
    #: Whether an output equal to the image's own file is left out of the record (the farm then lays the image's); a
    #: recipe whose outputs replace no file of the image keeps every one.
    against_image: bool = True

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
            **self.variables(),
            self.gfx_variable: gfx_arch,
        }


def _relative(path: PurePosixPath) -> str:
    return "" if path == PurePosixPath(".") else path.as_posix()


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
        command=(PYTHON_PLACEHOLDER, "setup_rocm.py", *_IN_PLACE),
        budget_sec=default_budget_sec(_SGL_KERNEL),
        max_jobs=SglKernelBuild.default_max_jobs,
        max_output_bytes=_MAX_OUTPUT_BYTES,
        preflight=partial(runner_preflight, SglKernelBuild, "sglang", _SGL_KERNEL),
    )


def vllm_recipe() -> NativeRecipe:
    """vLLM's in-place build of its extensions (``_C``, ``_moe_C``, ``_rocm_C`` and whatever else the ROCm build makes
    beside the package), at the top of the repository, where the rule pack places its package.

    The compile is vLLM's own ``setup.py build_ext --inplace``: CMake over ``csrc``, its outputs copied into the package.
    """
    package = PurePosixPath(component_layouts()[_VLLM].sections[0].prefix)
    return NativeRecipe(
        component=_VLLM,
        import_name=_VLLM,
        version=_VLLM_RECIPE_VERSION,
        source_dir=_relative(package.parent),
        package_dir="",
        outputs=(f"{package.name}/*.so",),
        command=(PYTHON_PLACEHOLDER, "setup.py", *_IN_PLACE),
        budget_sec=default_budget_sec(_VLLM_SOURCE),
        max_jobs=VllmBuild.default_max_jobs,
        max_output_bytes=_MAX_OUTPUT_BYTES,
        preflight=_vllm_preflight,
        gfx_variable="PYTORCH_ROCM_ARCH",
        variables=rocm_variables,
    )


#: Each recipe's factory, as ``module:function``: the build runner's child imports the recipe by it.
RECIPE_FACTORIES: Mapping[str, str] = {
    _SGL_KERNEL: f"{__name__}:{sgl_kernel_recipe.__name__}",
    _VLLM: f"{__name__}:{vllm_recipe.__name__}",
}


def native_recipes() -> dict[str, NativeRecipe]:
    """Every component's recipe, by component, made by its factory in :data:`RECIPE_FACTORIES`."""
    recipes: dict[str, NativeRecipe] = {}
    for component, factory in RECIPE_FACTORIES.items():
        module, _, name = factory.partition(":")
        recipes[component] = getattr(importlib.import_module(module), name)()
    return recipes


def rocm_variables(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The ROCm root as the single-lineage vLLM build names it: vLLM's ROCm build reads the toolchain root from
    ``CUDA_HOME`` as well. The first of ``ROCM_HOME``, ``CUDA_HOME``, ``ROCM_PATH`` and ``HIP_PATH`` set, else
    ``/opt/rocm``."""
    environ = os.environ if environ is None else environ
    root = next((environ[name].strip() for name in _ROCM_ROOT_VARIABLES if environ.get(name, "").strip()), "")
    return dict.fromkeys(_ROCM_ROOT_SET, root or _DEFAULT_ROCM_ROOT)


def _vllm_preflight(scratch: Path, gfx_arch: str) -> Preflight:
    """The single-lineage vLLM recipe's preflight, then the build tools vLLM's ``setup.py`` runs, which the build cannot
    fetch: a missing one is the machine's, never the candidate's compile error."""
    checked = runner_preflight(VllmBuild, _VLLM, _VLLM_SOURCE, scratch, gfx_arch)
    missing = [tool for tool in _VLLM_BUILD_TOOLS if shutil.which(tool) is None]
    if checked.problem is None and missing:
        return Preflight(f"preflight_toolchain: {', '.join(missing)} not on PATH; vLLM's build runs them", {})
    return checked


def runner_preflight(
    recipe_class: type[SglKernelBuild] | type[VllmBuild], framework: str, component: str, scratch: Path, gfx_arch: str
) -> Preflight:
    """A single-lineage recipe's own preflight: disk for one candidate, the ROCm toolchain, a ROCm torch, the arch."""
    action = TargetedBuildAction(
        gap_id="native-build", framework=framework, component=component, capability="native", gpu_arch=gfx_arch
    )
    recipe = recipe_class(action, str(scratch / "preflight"), run=subprocess.run, git=None, disk_preflight_fn=None)
    failed = recipe.preflight()
    if failed is not None:
        return Preflight(f"{failed.failure_class}: {failed.failure_summary}", {})
    abi = recipe.abi
    return Preflight(None, {"torch": str(abi.get("torch_version", "")), "hip": str(abi.get("hip_version", ""))})
