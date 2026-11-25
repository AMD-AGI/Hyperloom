# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A stack of the stand-in native component built end to end into the store: key, compile, harvest, publish, read."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.measure.compile import PatchApplyError
from hyperloom.measure.runner.native import builder as builder_module
from hyperloom.measure.runner.native.builder import BuildRequest, compile_tree, plan, tree_dir
from hyperloom.measure.jit.native.record import FAILED, LOG_NAME, READY, RECORD_FILES
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.runner.native.isolated import IsolatedBuilds, Isolation
from hyperloom.measure.runner.native.planning import Planned
from hyperloom.measure.runner.native.recipes import CommitSource
from hyperloom.measure.runner.native.tests import _native as native_bench
from hyperloom.measure.runner.native.tests._native import COMPONENT, KERNEL, OUTPUT, Bench, compiled, stepped_recipe

PATCHED = "int kernel(void) { return 2; }\n"


@pytest.fixture
def bench(tmp_path: Path) -> Bench:
    return Bench(tmp_path)


def _build(bench: Bench, *patches):
    return bench.build(*patches)


def test_patched_build_publishes_the_output_that_differs_from_the_image(bench: Bench, tmp_path: Path) -> None:
    ensured = _build(bench, bench.patch({KERNEL: PATCHED}))

    record = ensured.record
    assert (ensured.reused, record.status, [o.path for o in record.outputs]) == (False, READY, [OUTPUT])
    entry = bench.store.entry_dir(record.inputs)
    assert sorted(p.name for p in entry.iterdir()) == sorted([*RECORD_FILES[READY], LOG_NAME])
    bench.store.fetch(record, tmp_path / "farm")
    assert (tmp_path / "farm" / OUTPUT).read_bytes() == compiled(PATCHED)
    assert record.toolchain == {"compiler": "stand-in", "gfx": "gfx942", "max_jobs": "1"}


def test_a_second_request_of_the_key_reuses_the_entry_without_compiling(bench: Bench) -> None:
    patch = bench.patch({KERNEL: PATCHED})
    first, second = _build(bench, patch), _build(bench, patch)

    assert (first.reused, second.reused, len(bench.preflight.calls)) == (False, True, 1)
    assert second.record == first.record


def test_stacks_that_yield_the_same_sources_share_one_key(bench: Bench) -> None:
    direct = bench.patch({KERNEL: PATCHED})
    detour = bench.patch({KERNEL: "int kernel(void) { return 3; }\n"})
    back = bench.patch({KERNEL: PATCHED}, before={KERNEL: "int kernel(void) { return 3; }\n"})

    key_direct = plan(bench.request(direct), bench.blobs)[0].key
    assert plan(bench.request(detour, back), bench.blobs)[0].key == key_direct
    assert plan(bench.request(detour), bench.blobs)[0].key != key_direct


def test_unpatched_build_equal_to_the_image_lists_no_output(bench: Bench) -> None:
    record = _build(bench).record

    assert (record.status, record.outputs) == (READY, ())


def test_a_compile_error_is_stored_signed_and_answered_again_without_a_rebuild(bench: Bench) -> None:
    patch = bench.patch({KERNEL: "#error\n"})
    first, second = _build(bench, patch), _build(bench, patch)

    assert (first.record.status, second.reused, len(bench.preflight.calls)) == (FAILED, True, 1)
    assert first.record.failure == "compile_error: the compile exited 1: src/kernel.c:1: error: stop"
    entry = bench.store.entry_dir(first.record.inputs)
    assert sorted(p.name for p in entry.iterdir()) == sorted([*RECORD_FILES[FAILED], LOG_NAME])
    assert "error: stop" in (entry / LOG_NAME).read_text()


def test_the_compilers_error_is_kept_when_later_output_buries_it(bench: Bench) -> None:
    record = _build(bench, bench.patch({KERNEL: "#noisy\n"})).record

    assert record.failure == "compile_error: the compile exited 1: src/kernel.c:3:2: error: noisy failure"
    log = (bench.store.entry_dir(record.inputs) / LOG_NAME).read_text()
    assert log.startswith("src/kernel.c:3:2: error: noisy failure\n")


ACT_CU, ACT_HIP = "native/src/act.cu", "native/src/act.hip"
HIP_INCLUDE = '#include "hip/hip_runtime.h"\n'


@pytest.mark.parametrize(
    ("cu", "named"),
    [
        ("int a;\n#error act broken\n", "src/act.cu:2:2 (src/act.hip:4:2): error: act broken"),
        (f"{HIP_INCLUDE}#error act broken\n", "src/act.cu:2:2 (src/act.hip:3:2): error: act broken"),
    ],
)
def test_an_error_in_the_hip_file_hipify_generated_names_the_patched_cu(bench: Bench, cu: str, named: str) -> None:
    record = _build(bench, bench.patch({ACT_CU: cu})).record

    assert record.failure == f"compile_error: the compile exited 1: {named}"
    assert (bench.store.entry_dir(record.inputs) / LOG_NAME).read_text().startswith(f"{named}\n")


def test_an_error_in_a_hip_file_the_upstream_tree_holds_names_that_file(bench: Bench) -> None:
    (bench.upstream / ACT_HIP).write_text("int a;\n#error act tracked\n")
    record = _build(bench, bench.patch({ACT_CU: "int a;\n"})).record

    assert record.failure == "compile_error: the compile exited 1: src/act.hip:2:2: error: act tracked"


def test_an_error_in_a_hip_file_the_stack_adds_names_that_file(bench: Bench) -> None:
    record = _build(bench, bench.patch({ACT_CU: "int a;\n", ACT_HIP: "#error act laid\n"})).record

    assert record.failure == "compile_error: the compile exited 1: src/act.hip:1:2: error: act laid"


def test_an_error_in_a_generated_hip_file_with_no_cu_beside_it_names_that_file(bench: Bench) -> None:
    record = _build(bench, bench.patch({KERNEL: "#orphanhip\n"})).record

    assert record.failure == "compile_error: the compile exited 1: src/act.hip:1:2: error: orphan"


@pytest.mark.parametrize(
    ("marker", "reason"),
    [
        ("#symlink", "an output must be a regular file reached without a symlink (symlink or not a directory"),
        ("#notelf", "an output must be an ELF file"),
        ("#big", "is more than the 1024 an output may be"),
        ("#none", "the build made none of its declared outputs"),
    ],
)
def test_outputs_not_as_declared_are_the_candidates_compile_error(bench: Bench, marker: str, reason: str) -> None:
    record = _build(bench, bench.patch({KERNEL: f"{marker}\n"})).record

    assert record.status == FAILED
    assert record.failure.startswith("compile_error: ") and reason in record.failure


def test_a_failed_preflight_is_infra_and_stores_nothing(bench: Bench) -> None:
    patch = bench.patch({KERNEL: PATCHED})
    bench.preflight.problem = "preflight_disk: 1 GB free"
    with pytest.raises(BuildInfraError, match="preflight_disk"):
        _build(bench, patch)
    inputs = plan(bench.request(patch), bench.blobs)[0]
    assert not bench.store.entry_dir(inputs).exists()

    bench.preflight.problem = None
    assert _build(bench, patch).reused is False


def test_a_compiler_killed_by_a_signal_is_infra_and_stores_nothing(bench: Bench) -> None:
    patch = bench.patch({KERNEL: "#signal\n"})
    with pytest.raises(BuildInfraError, match="timed out or was killed"):
        _build(bench, patch)
    assert not bench.store.entry_dir(plan(bench.request(patch), bench.blobs)[0]).exists()


def test_a_stack_that_changes_the_build_definition_is_refused(bench: Bench) -> None:
    patch = bench.patch(
        {"native/CMakeLists.txt": "project(evil)\n"}, before={"native/CMakeLists.txt": "project(fake)\n"}
    )
    with pytest.raises(
        PatchApplyError, match="'native/CMakeLists.txt' is a native or binary file \\(CMakeLists.txt\\)"
    ):
        plan(bench.request(patch), bench.blobs)


def test_the_compile_sees_no_variable_of_the_service(bench: Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HL_TEST_SERVICE_SECRET", "x")
    record = _build(bench, bench.patch({KERNEL: PATCHED})).record

    names = (bench.store.entry_dir(record.inputs) / LOG_NAME).read_text().split()
    assert "HL_TEST_SERVICE_SECRET" not in names
    assert {"AMDGPU_TARGET", "MAX_JOBS", "PATH"} <= set(names)


def _with_steps(request: BuildRequest, *steps: tuple[str, ...]) -> BuildRequest:
    return replace(request, recipe=replace(request.recipe, steps=steps))


class _Planner:
    def __init__(self, planned: Planned) -> None:
        self._planned = planned

    def planned(self, inputs: object) -> Planned:
        return self._planned


def test_the_build_child_runs_the_steps_in_order_with_the_arch_the_jobs_and_the_commit_filled(
    bench: Bench, tmp_path: Path
) -> None:
    request = _with_steps(bench.request(bench.patch({KERNEL: PATCHED})), *stepped_recipe().steps)
    inputs, root = plan(request, bench.blobs)
    runner = IsolatedBuilds(
        _Planner(Planned(request, root)),  # type: ignore[arg-type]
        bench.blobs,
        Isolation((), None),
        {COMPONENT: f"{native_bench.__name__}:{stepped_recipe.__name__}"},
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    staged = runner(inputs)(scratch)

    assert staged.status == READY, staged.failure
    assert f"step -DT=gfx942 -j 1 -DC={request.stack.upstream_ref}" in staged.log
    assert [output.path for output in staged.outputs] == [OUTPUT]


def test_a_step_that_fails_ends_the_compile_as_the_candidates_error(bench: Bench, tmp_path: Path) -> None:
    request = _with_steps(
        bench.request(bench.patch({KERNEL: PATCHED})),
        ("{python}", "-c", "import sys; sys.exit(3)"),
        ("{python}", "build.py"),
    )
    inputs, root = plan(request, bench.blobs)

    staged = compile_tree(request, root, bench.blobs, tmp_path / "scratch")

    assert (staged.status, staged.failure) == (FAILED, "compile_error: the compile exited 3")
    assert not (tree_dir(tmp_path / "scratch") / "python" / OUTPUT).exists(), "the later step never ran"


def test_a_recipes_only_step_gets_the_whole_budget_and_a_later_one_what_is_left(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    given: list[int] = []
    run = builder_module.run_argv
    monkeypatch.setattr(
        builder_module, "run_argv", lambda argv, **kwargs: given.append(kwargs["timeout_sec"]) or run(argv, **kwargs)
    )
    one = bench.request(bench.patch({KERNEL: PATCHED}))
    two = _with_steps(one, ("{python}", "-c", "import time; time.sleep(1.1)"), *one.recipe.steps)

    for number, request in enumerate((one, two)):
        staged = compile_tree(request, plan(request, bench.blobs)[1], bench.blobs, tmp_path / f"scratch-{number}")
        assert staged.status == READY, staged.failure

    budget = one.recipe.budget_sec
    assert given[:2] == [budget, budget], "a one-step recipe's timeout is its budget, as before steps"
    assert given[2] < budget, "the second step has what the first left"


@pytest.mark.parametrize(
    ("source", "separate"), [(CommitSource.PINNED_CHECKOUT, True), (CommitSource.IMAGE_DIST, False)]
)
def test_a_pinned_checkouts_commit_is_part_of_the_key(bench: Bench, source: CommitSource, separate: bool) -> None:
    request = bench.request(bench.patch({KERNEL: PATCHED}))
    request = replace(request, recipe=replace(request.recipe, commit_source=source))
    moved = replace(request, stack=replace(request.stack, upstream_ref="e" * 40))

    first, second = plan(request, bench.blobs)[0], plan(moved, bench.blobs)[0]

    assert (first.key != second.key) is separate
    if separate:
        assert first.recipe_version == f"fake/1+{request.stack.upstream_ref}"


def _executable_request(bench: Bench, *steps: tuple[str, ...]) -> BuildRequest:
    request = bench.request(bench.patch({KERNEL: PATCHED}))
    recipe = replace(request.recipe, outputs=(), executables=(OUTPUT,), against_image=False, steps=steps)
    return replace(request, recipe=recipe)


def test_a_declared_executable_left_plain_is_the_candidates_compile_error(bench: Bench, tmp_path: Path) -> None:
    request = _executable_request(bench, ("{python}", "build.py"))
    inputs, root = plan(request, bench.blobs)

    staged = compile_tree(request, root, bench.blobs, tmp_path / "scratch")

    assert (staged.status, staged.failure) == (FAILED, f"compile_error: the build made no executable {OUTPUT}")
