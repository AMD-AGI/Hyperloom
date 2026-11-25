# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What compiling one component's native tree takes: where it builds, with what, within which budget, and what it makes.

The sgl-kernel and vLLM recipes reuse the single-lineage runner's (:class:`~hyperloom.orchestrator.enablement.runtime.
targeted_build.SglKernelBuild` and :class:`~hyperloom.orchestrator.enablement.runtime.targeted_build.VllmBuild`): their
disk and toolchain preflight, their ``MAX_JOBS`` default, their wall-clock budget and the variables they name the GPU
architecture and the ROCm root by. Their front and back are new: the tree is a copy of the pinned upstream checkout with
the stack laid over it, the compile builds the extensions in place instead of installing into a venv, and the outputs
are harvested by glob.

The llama.cpp recipe builds no Python extension and needs no torch: its preflight is
:func:`~hyperloom.measure.runner.native.preflight.native_preflight`, and it makes static executables from a checkout
the operator pinned (:attr:`CommitSource.PINNED_CHECKOUT`), whose commit the build key names: the server, and
``test-backend-ops``, which times a change of its kernels one operation at a time before a test.

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
from enum import Enum
from functools import partial
from pathlib import Path, PurePosixPath

from hyperloom.common.outside_facts import component_layouts
from hyperloom.measure.compile import ComponentLayout
from hyperloom.measure.runner.native.preflight import Preflight, native_preflight
from hyperloom.orchestrator.enablement.runtime.build_actions import TargetedBuildAction
from hyperloom.orchestrator.enablement.runtime.targeted_build import SglKernelBuild, VllmBuild, default_budget_sec

__all__ = [
    "NATIVE_BUILD_REFUSE",
    "RECIPE_FACTORIES",
    "CommitSource",
    "NativeRecipe",
    "llamacpp_recipe",
    "native_recipes",
    "rocm_variables",
    "runner_preflight",
    "sgl_kernel_recipe",
    "vllm_recipe",
]

#: Files a native stack may not change in v1: the build definition runs arbitrary code at build time, with the widest
#: blast radius; matched on the file name, as a component layout's ``refuse`` pattern without a slash is.
NATIVE_BUILD_REFUSE = ("CMakeLists.txt", "*.cmake", "setup.py", "setup_rocm.py", "pyproject.toml", "setup.cfg")
PYTHON_PLACEHOLDER = "{python}"
GFX_PLACEHOLDER = "{gfx}"
JOBS_PLACEHOLDER = "{jobs}"
#: The upstream commit the tree is a checkout of: the build is told it, never reads it from a ``.git`` it may not have.
COMMIT_PLACEHOLDER = "{commit}"
_SGL_KERNEL = "sgl_kernel"
_SGL_KERNEL_RECIPE_VERSION = "sgl_kernel/1"
_VLLM = "vllm"
_VLLM_RECIPE_VERSION = "vllm/1"
#: The single-lineage runner's name of the vLLM build, which its budgets are kept under.
_VLLM_SOURCE = "vllm_source"
_CMAKE = "cmake"
#: The tools vLLM's ``setup.py`` drives the compile with; with no egress in the build they must be in the image.
_VLLM_BUILD_TOOLS = (_CMAKE, "ninja")
_MAX_OUTPUT_BYTES = 2 * 1024**3
_IN_PLACE = ("build_ext", "--inplace")
#: The variables that name the ROCm root, in the order the single-lineage vLLM build reads them, and where it is when
#: none is set.
_ROCM_ROOT_VARIABLES = ("ROCM_HOME", "CUDA_HOME", "ROCM_PATH", "HIP_PATH")
_ROCM_ROOT_SET = ("CUDA_HOME", "ROCM_HOME", "ROCM_PATH", "HIP_HOME")
_DEFAULT_ROCM_ROOT = "/opt/rocm"
_LLAMACPP = "llamacpp"
_LLAMACPP_RECIPE_VERSION = "llamacpp/3"
_LLAMACPP_BUILD = "build"
_LLAMACPP_TARGETS = ("llama-server", "test-backend-ops")
_LLAMACPP_MAX_JOBS = 32
_LLAMACPP_DISK_GB = 10.0
#: Static, so no shared library has to sit beside the executables (the harvest takes no symlink). Nothing is fetched
#: (the compile may have no network): no FetchContent, no prebuilt web UI, which the build would download. The commit
#: and build number are given, since the copied tree's ``.git`` may be unreadable to the compile's user. The tests
#: are configured only so that ``test-backend-ops`` is a target; no other test is built. Position independent: where
#: the distribution's linker makes a PIE by default (Ubuntu 24.04), HIP objects compiled without it do not link.
_LLAMACPP_CONFIGURE = (
    _CMAKE,
    "-S",
    ".",
    "-B",
    _LLAMACPP_BUILD,
    "-G",
    "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    "-DGGML_HIP=ON",
    "-DGGML_HIP_GRAPHS=ON",
    "-DLLAMA_CURL=OFF",
    "-DGGML_NATIVE=OFF",
    "-DLLAMA_BUILD_TESTS=ON",
    "-DLLAMA_USE_PREBUILT_UI=OFF",
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
    f"-DAMDGPU_TARGETS={GFX_PLACEHOLDER}",
    f"-DLLAMA_BUILD_COMMIT={COMMIT_PLACEHOLDER}",
    "-DLLAMA_BUILD_NUMBER=0",
)
_LLAMACPP_COMPILE = (_CMAKE, "--build", _LLAMACPP_BUILD, "--target", *_LLAMACPP_TARGETS, "-j", JOBS_PLACEHOLDER)


class CommitSource(Enum):
    """Where a component's upstream commit comes from: the distribution the image installs, or a checkout the operator
    pinned (which no image digest stands for)."""

    IMAGE_DIST = "image_dist"
    PINNED_CHECKOUT = "pinned_checkout"


def _filled(part: str, values: Mapping[str, str]) -> str:
    for placeholder, value in values.items():
        part = part.replace(placeholder, value)
    return part


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
    #: The compile's commands, run in order in the tree; each may name the placeholders :meth:`commands` fills.
    steps: tuple[tuple[str, ...], ...]
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
    commit_source: CommitSource = CommitSource.IMAGE_DIST
    #: Paths below ``package_dir`` the compile must leave as executable files, harvested beside ``outputs``. The image
    #: has none of them to stand in for one left out, so a recipe that names any sets ``against_image`` false and every one is kept.
    executables: tuple[str, ...] = ()

    def layout(self) -> ComponentLayout:
        """The native section: the build directory of the upstream tree, the build definition refused."""
        return ComponentLayout(self.import_name, self.source_dir, refuse=NATIVE_BUILD_REFUSE)

    def build_version(self, upstream_ref: str) -> str:
        """The recipe version a build key carries: a pinned checkout's commit is no image's, so the key names it."""
        if self.commit_source is CommitSource.PINNED_CHECKOUT:
            return f"{self.version}+{upstream_ref}"
        return self.version

    def commands(self, python: str, gfx_arch: str, upstream_ref: str) -> list[list[str]]:
        """Each step's command line, its placeholders filled; a step that exits non-zero ends the compile."""
        values = {
            PYTHON_PLACEHOLDER: python,
            GFX_PLACEHOLDER: gfx_arch,
            JOBS_PLACEHOLDER: str(self.max_jobs),
            COMMIT_PLACEHOLDER: upstream_ref,
        }
        return [[_filled(part, values) for part in step] for step in self.steps]

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
        steps=((PYTHON_PLACEHOLDER, "setup_rocm.py", *_IN_PLACE),),
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
        steps=((PYTHON_PLACEHOLDER, "setup.py", *_IN_PLACE),),
        budget_sec=default_budget_sec(_VLLM_SOURCE),
        max_jobs=VllmBuild.default_max_jobs,
        max_output_bytes=_MAX_OUTPUT_BYTES,
        preflight=_vllm_preflight,
        gfx_variable="PYTORCH_ROCM_ARCH",
        variables=rocm_variables,
    )


def llamacpp_recipe() -> NativeRecipe:
    """llama.cpp's ``llama-server`` and ``test-backend-ops``, configured and built with CMake at the top of the pinned
    checkout, which no image installs: its commit is part of the build's key."""
    return NativeRecipe(
        component=_LLAMACPP,
        import_name=_LLAMACPP,
        version=_LLAMACPP_RECIPE_VERSION,
        source_dir="",
        package_dir=_LLAMACPP_BUILD,
        outputs=(),
        steps=(_LLAMACPP_CONFIGURE, _LLAMACPP_COMPILE),
        budget_sec=default_budget_sec(_LLAMACPP),
        max_jobs=_LLAMACPP_MAX_JOBS,
        max_output_bytes=_MAX_OUTPUT_BYTES,
        preflight=_llamacpp_preflight,
        variables=_llamacpp_variables,
        against_image=False,
        commit_source=CommitSource.PINNED_CHECKOUT,
        executables=tuple(f"bin/{target}" for target in _LLAMACPP_TARGETS),
    )


#: Each recipe's factory, as ``module:function``: the build runner's child imports the recipe by it.
RECIPE_FACTORIES: Mapping[str, str] = {
    _SGL_KERNEL: f"{__name__}:{sgl_kernel_recipe.__name__}",
    _VLLM: f"{__name__}:{vllm_recipe.__name__}",
    _LLAMACPP: f"{__name__}:{llamacpp_recipe.__name__}",
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


def _hip_clang(root: str) -> str:
    return f"{root}/llvm/bin/clang++"


def _llamacpp_variables(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The ROCm root (:func:`rocm_variables`) and, as ``HIPCXX``, its clang, which CMake compiles HIP with."""
    variables = rocm_variables(environ)
    return {**variables, "HIPCXX": _hip_clang(variables["ROCM_PATH"])}


def _llamacpp_preflight(scratch: Path, gfx_arch: str) -> Preflight:
    """No torch: disk and the versions of CMake, Ninja and the ROCm root's clang."""
    programs = {"cmake": _CMAKE, "ninja": "ninja", "hip_clang": _llamacpp_variables()["HIPCXX"]}
    tools = {name: (program, "--version") for name, program in programs.items()}
    return native_preflight(tools, _LLAMACPP_DISK_GB, scratch)


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
