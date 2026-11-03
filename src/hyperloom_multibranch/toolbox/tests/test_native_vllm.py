# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""#125 stage 4c with a stand-in compiler: a stack that edits vLLM's ``csrc`` is planned by vLLM's own recipe over the
rule pack's layout of vLLM, built out of process from ``<sources>/vllm/<ref>``, laid in the farm and imported from it. A
native source inside the ``vllm`` package, which the package compiles at run time, is an ordinary package file."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass
from hyperloom.contracts.recipe import ComponentPatches, Recipe, RecipeDelta
from hyperloom.measure.compile import EnvironmentCompiler, PatchApplyError, UnsupportedComponentError
from hyperloom.measure.farm import prepend_pythonpath, verify_farm
from hyperloom.measure.jit.native.needs import BuildNotReady
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.tests._recipes import IMAGE
from hyperloom_multibranch.toolbox.tests import native_support
from hyperloom_multibranch.toolbox.tests import vllm_support as vllm
from hyperloom_multibranch.toolbox.tests.test_native_builds import Hooks, Rig, farm_of

EDITED_CU = "__global__ void reshape_and_cache() { /* faster */ }\n"
KERNEL = '#include <hip/hip_runtime.h>\nextern "C" __global__ void osk_gemm() {}\n'


def plain_compiler(rig: Rig) -> EnvironmentCompiler:
    """The compiler of a session with no native builds configured: the rule pack's layout of vLLM alone."""
    return EnvironmentCompiler(rig.blobs, IMAGE, lambda layout: rig.image.package(), {"vllm": vllm.LAYOUT})


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path, kit=vllm)


async def test_a_vllm_cu_edit_waits_for_its_build_and_is_laid_in_the_farm_and_imported_from_it(rig: Rig) -> None:
    stack = rig.recipe(
        rig.patch(
            {vllm.ACT_CU: EDITED_CU, vllm.PY_FILE: "SCALE = 2\n"},
            {vllm.ACT_CU: vllm.BASE_CU, vllm.PY_FILE: vllm.BASE_PY},
        )
    )
    compiled = rig.compiler.compile(stack, RecipeDelta())
    (root,) = compiled.roots
    assert (root.id, root.kind, [e.path for e in root.entries]) == ("vllm", "runtime", ["envs.py"])
    with pytest.raises(BuildNotReady):
        compiled.to_document()

    hooks = Hooks()
    builds = await rig.wait(AcceptedMeasurement(compiled, ()), hooks)
    assert hooks.ready == ["r1"] and hooks.ended == []
    accepted = builds.resolved(AcceptedMeasurement(compiled, ()))

    environment, farm = farm_of(rig, accepted)
    record = rig.store.read(compiled.builds[0].inputs)
    extension = native_support.extension().name
    assert [o.path for o in record.outputs] == [f"vllm/{extension}"], "harvested beside the package, as vllm/_C*.so"
    output = farm.site / "vllm" / extension
    assert output.read_bytes() == native_support.compiled(EDITED_CU) and not output.is_symlink()
    assert (farm.site / "vllm" / "envs.py").read_text() == "SCALE = 2\n"
    envs: dict = {}
    prepend_pythonpath(envs, farm.site)
    verify_farm(farm, envs, python=rig.image.python, cwd=rig.tmp)
    assert [m.name for m in farm.native] == [vllm.module_name()]


def test_a_header_edit_alone_is_a_build_with_no_package_files(rig: Rig) -> None:
    compiled = rig.compiler.compile(
        rig.recipe(rig.patch({"csrc/ops.h": "void act(int);\n"}, {"csrc/ops.h": "void act();\n"})), RecipeDelta()
    )

    ((root,), (need,)) = compiled.roots, compiled.builds
    assert (root.kind, root.entries, need.inputs.component) == ("runtime", (), "vllm")


async def test_a_broken_vllm_cu_is_the_candidates_compile_error_naming_the_cu_line(rig: Rig) -> None:
    compiled = rig.compiler.compile(rig.recipe(rig.patch({vllm.ACT_CU: "int a;\n#error W207 broken\n"})), RecipeDelta())

    hooks = Hooks()
    await rig.wait(AcceptedMeasurement(compiled, ()), hooks)

    ((request_id, end),) = hooks.ended
    assert request_id == "r1" and end.failure_class is FailureClass.CANDIDATE
    assert end.reason.startswith("compile_error: vllm ")
    assert "csrc/cache_kernels.cu:2:2 (csrc/cache_kernels.hip:4:2): error: W207 broken" in end.reason


@pytest.mark.parametrize(
    "path",
    ["CMakeLists.txt", "setup.py", "cmake/utils.cmake", "cmake/hipify.py"],
    ids=["cmakelists", "setup", "cmake-file", "cmake-dir"],
)
def test_a_change_to_vllms_build_definition_is_refused_at_submission_naming_the_path(rig: Rig, path: str) -> None:
    patch = rig.patch({vllm.ACT_CU: EDITED_CU, path: "changed\n"}, {vllm.ACT_CU: vllm.BASE_CU, path: "original\n"})

    with pytest.raises(PatchApplyError, match=f"{path!r} is .*build definition.* of vllm"):
        rig.compiler.compile(rig.recipe(patch), RecipeDelta())


def test_without_a_native_plan_a_vllm_cu_edit_is_refused_by_the_rule_packs_layout(rig: Rig) -> None:
    with pytest.raises(
        PatchApplyError, match=r"'csrc/cache_kernels.cu' is a native or binary file \(csrc/\*\.cu\) of vllm"
    ):
        plain_compiler(rig).compile(rig.recipe(rig.patch({vllm.ACT_CU: EDITED_CU})), RecipeDelta())


@pytest.mark.parametrize("path", ["vllm/_osk_gemm.hip", "vllm/kernels/osk.cu", "vllm/kernels/osk.cpp"])
def test_a_source_the_package_compiles_at_run_time_is_laid_as_a_package_file_and_builds_nothing(
    rig: Rig, path: str
) -> None:
    """T23's keep: ``vllm/_osk_gemm.hip`` beside the Python that hipcc-compiles it when it runs."""
    stack = rig.recipe(rig.patch({path: KERNEL, vllm.PY_FILE: "SCALE = 2\n"}, {vllm.PY_FILE: vllm.BASE_PY}))
    relative = path.removeprefix("vllm/")

    for compiler in (rig.compiler, plain_compiler(rig)):
        compiled = compiler.compile(stack, RecipeDelta())

        assert [(r.kind, sorted(e.path for e in r.entries)) for r in compiled.roots] == [
            ("package", sorted([relative, "envs.py"]))
        ]
        assert compiled.builds == ()


@pytest.mark.parametrize(
    ("path", "pattern"),
    [("csrc/rocm/osk_gemm.hip", "csrc/*.hip"), ("csrc/osk.cu", "csrc/*.cu"), ("csrc/osk.cpp", "csrc/*.cpp")],
)
def test_a_source_below_csrc_is_a_native_build_and_without_a_native_plan_is_refused(
    rig: Rig, path: str, pattern: str
) -> None:
    stack = rig.recipe(rig.patch({path: KERNEL}, {vllm.PY_FILE: vllm.BASE_PY}))

    compiled = rig.compiler.compile(stack, RecipeDelta())

    ((root,), (need,)) = compiled.roots, compiled.builds
    assert (root.kind, root.entries, need.inputs.component) == ("runtime", (), "vllm")
    with pytest.raises(PatchApplyError, match=re.escape(f"{path!r} is a native or binary file ({pattern}) of vllm")):
        plain_compiler(rig).compile(stack, RecipeDelta())


@pytest.mark.parametrize(
    ("path", "pattern"),
    [
        ("vllm/.osk_cache/osk_gemm.co", "*.co"),
        ("vllm/_osk_gemm.hsaco", "*.hsaco"),
        ("vllm/_C.abi3.so", "*.so"),
        ("csrc/prebuilt.so", "*.so"),
        ("CMakeLists.txt", "CMakeLists.txt"),
    ],
)
def test_without_a_native_plan_a_compiled_or_build_file_is_refused_wherever_it_lies(
    rig: Rig, path: str, pattern: str
) -> None:
    stack = rig.recipe(rig.patch({path: "text\n", vllm.PY_FILE: "SCALE = 2\n"}, {vllm.PY_FILE: vllm.BASE_PY}))

    with pytest.raises(PatchApplyError, match=re.escape(f"{path!r} is a native or binary file ({pattern}) of vllm")):
        plain_compiler(rig).compile(stack, RecipeDelta())


def test_a_python_only_vllm_stack_stays_a_package_root_with_no_build(rig: Rig) -> None:
    stack = rig.recipe(rig.patch({vllm.PY_FILE: "SCALE = 2\n"}, {vllm.PY_FILE: vllm.BASE_PY}))

    compiled = rig.compiler.compile(stack, RecipeDelta())

    assert [(r.kind, [e.path for e in r.entries]) for r in compiled.roots] == [("package", ["envs.py"])]
    assert compiled.builds == ()


def test_a_vllm_ref_with_no_checkout_under_sources_vllm_is_refused(rig: Rig) -> None:
    (patch,) = rig.recipe(rig.patch({vllm.ACT_CU: EDITED_CU})).components[0].patches
    other = Recipe(
        base_image_digest=IMAGE,
        components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", "e" * 40, (patch,)),),
    )

    with pytest.raises(UnsupportedComponentError, match=f"vllm at {'e' * 40} has no upstream checkout"):
        rig.compiler.compile(other, RecipeDelta())
