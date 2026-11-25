# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session's native builds through their one entry, over the real store, planner and build runner child: the pinned
commit built as it is, a stack built over it, each answered as built, the candidate's compile error or the machine's
failure, and the environment a test of a built stack runs in."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import ComponentPatches, PatchRef
from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    CompileError,
    ComponentLayout,
    LocalBlobStore,
    PatchApplyError,
)
from hyperloom.measure.jit.native.config import BuildStoreConfigError
from hyperloom.measure.jit.native.needs import BuildNotReady
from hyperloom.measure.jit.native.record import FAILED, LOG_NAME
from hyperloom.measure.runner.native import isolated as isolated_module
from hyperloom.measure.runner.native import recipes as recipes_module
from hyperloom.measure.runner.native.builder import BuildMachine
from hyperloom.measure.runner.native.isolated import (
    BUILD_CPUS_ENV,
    BUILD_CPUS_KEY,
    Isolation,
    Unisolated,
    build_cpus,
)
from hyperloom.measure.runner.native.planning import NativeSite
from hyperloom.measure.runner.native.recipes import NativeRecipe, llamacpp_recipe, native_recipes
from hyperloom.measure.runner.native.session import (
    BuildOutcome,
    BuildStatus,
    NativeBuildSession,
    PinnedCheckout,
    open_session,
    pinned_checkouts,
)
from hyperloom.measure.runner.native.tests import _native as native_bench
from hyperloom.measure.runner.native.tests._native import (
    BASE_SOURCE,
    BUILDER_IMAGE,
    COMPONENT,
    KERNEL,
    Bench,
    affinity_recipe,
    executable_recipe,
    image_root,
    pinned_affinity_recipe,
    stepped_recipe,
    upstream_tree,
)
from hyperloom.measure.tests._native_llamacpp import git, llama_checkout
from hyperloom.measure.tests._recipes import COMMIT, IMAGE, git_patch, put_patch, write_tree

LAYOUT = ComponentLayout(COMPONENT, f"native/python/{COMPONENT}", native=("*.c",))
LLAMACPP_COMMIT = "e" * 40


@pytest.fixture()
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Bench:
    monkeypatch.setitem(COMPONENT_LAYOUTS, COMPONENT, LAYOUT)
    monkeypatch.setitem(
        recipes_module.RECIPE_FACTORIES, COMPONENT, f"{native_bench.__name__}:{stepped_recipe.__name__}"
    )
    return Bench(tmp_path)


def _session(
    bench: Bench,
    isolation: Isolation | Unisolated = Isolation((), None),
    recipe: NativeRecipe | None = None,
    cpus: str = "",
) -> NativeBuildSession:
    upstream_tree(bench.tmp / "sources" / COMPONENT / COMMIT)
    image_root(bench.tmp / "site-packages")
    machine = BuildMachine(IMAGE, BUILDER_IMAGE, "gfx942")
    recipes = {COMPONENT: recipe or stepped_recipe()}
    sources, image = bench.tmp / "sources", bench.tmp / "site-packages"
    site = NativeSite(machine, recipes, sources, image, {COMPONENT: COMMIT}, cpus)
    return open_session(bench.store, bench.signer, site, bench.blobs, lambda: isolation)


def _stack(*patches: PatchRef) -> ComponentPatches:
    return ComponentPatches(COMPONENT, "https://example.invalid/fake.git", COMMIT, patches)


def test_a_stack_that_does_not_compile_is_the_candidates_error_and_stays_so(bench: Bench) -> None:
    session = _session(bench)
    stack = _stack(bench.patch({KERNEL: "#error\n" + BASE_SOURCE}))

    first, again = session.build(stack), session.build(stack)

    assert (first.status, first.reused, again.status, again.reused) == (
        BuildStatus.COMPILE_ERROR,
        False,
        BuildStatus.COMPILE_ERROR,
        True,
    )
    assert first.detail.startswith("compile_error: the compile exited 1")
    assert first.record is not None and first.record.status == FAILED


def test_a_host_that_cannot_isolate_the_compile_fails_the_build_as_the_machines_and_stores_nothing(
    bench: Bench,
) -> None:
    session = _session(bench, Unisolated("no tier (a stand-in)"))

    outcome = session.build(_stack(bench.patch({KERNEL: "int kernel(void) { return 2; }\n"})))

    assert (outcome.status, outcome.record) == (BuildStatus.INFRA, None)
    assert "no tier (a stand-in)" in outcome.detail
    assert bench.store.read(outcome.inputs) is None


def test_a_test_of_a_stack_runs_in_its_build_once_the_build_is_ready(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(bench)
    monkeypatch.syspath_prepend(str(bench.tmp / "site-packages"))
    stack = _stack(bench.patch({KERNEL: "int kernel(void) { return 3; }\n"}))
    with pytest.raises(BuildNotReady, match="is not ready in the store"):
        session.environment(stack)

    built = session.build(stack)
    environment = session.environment(stack)

    assert built.status is BuildStatus.BUILT
    assert [(ref.root_id, ref.key) for ref in environment.build_refs] == [(COMPONENT, built.inputs.key)]
    assert [root.kind for root in environment.roots] == ["runtime"]


def test_a_stack_that_changes_no_native_file_asks_for_no_build(bench: Bench) -> None:
    session = _session(bench)
    before = {f"native/python/{COMPONENT}/__init__.py": ""}
    stack = _stack(bench.patch({f"native/python/{COMPONENT}/__init__.py": "X = 1\n"}, before))

    with pytest.raises(CompileError, match="changes no native file"):
        session.build(stack)


def test_a_stack_may_change_a_file_outside_the_package_beside_a_native_source(bench: Bench) -> None:
    """As before a checkout could be built: the file outside the package is in the build's tree and in no root."""
    session = _session(bench)
    readme = "native/README.md"
    stack = _stack(bench.patch({KERNEL: "int kernel(void) { return 5; }\n", readme: "notes\n"}))

    planned = session.planner(stack)

    assert planned is not None and readme not in {entry.path for entry in planned.entries}
    assert session.build(stack).status is BuildStatus.BUILT


@pytest.fixture()
def executables(bench: Bench, monkeypatch: pytest.MonkeyPatch) -> NativeBuildSession:
    """A session of the stand-in component as a build that makes an executable and no package."""
    factory = f"{native_bench.__name__}:{executable_recipe.__name__}"
    monkeypatch.setitem(recipes_module.RECIPE_FACTORIES, COMPONENT, factory)
    return _session(bench, recipe=executable_recipe())


def test_a_stack_of_a_build_that_makes_executables_runs_in_an_executable_root(
    bench: Bench, executables: NativeBuildSession
) -> None:
    stack = _stack(bench.patch({KERNEL: "int kernel(void) { return 4; }\n"}))

    built = executables.build(stack)
    environment = executables.environment(stack)

    assert built.status is BuildStatus.BUILT
    assert [(r.kind, r.import_name, r.dist_version, r.entries) for r in environment.roots] == [
        ("executable", COMPONENT, COMMIT, ())
    ]
    assert [(ref.root_id, ref.key) for ref in environment.build_refs] == [(COMPONENT, built.inputs.key)]


def test_the_pinned_commit_as_it_is_runs_in_its_build_when_the_build_makes_executables(
    bench: Bench, executables: NativeBuildSession
) -> None:
    with pytest.raises(BuildNotReady, match="is not ready in the store"):
        executables.base_environment(COMPONENT)

    built = executables.base(COMPONENT)
    environment = executables.base_environment(COMPONENT)
    again = executables.base(COMPONENT)

    assert [(r.id, r.kind) for r in environment.roots] == [(COMPONENT, "executable")]
    assert [ref.key for ref in environment.build_refs] == [built.inputs.key]
    assert not environment.is_base
    assert (built.reused, again.status, again.reused) == (False, BuildStatus.BUILT, True)


def _affinity(bench: Bench, monkeypatch: pytest.MonkeyPatch, recipe: Callable[[], NativeRecipe]) -> BuildOutcome:
    """The build of the pinned commit by ``recipe``, which prints its CPUs and priority, on a site whose build CPUs are
    CPU 0."""
    monkeypatch.setitem(recipes_module.RECIPE_FACTORIES, COMPONENT, f"{native_bench.__name__}:{recipe.__name__}")
    return _session(bench, recipe=recipe(), cpus="0").base(COMPONENT)


def _log(bench: Bench, built: BuildOutcome) -> str:
    return (bench.store.entry_dir_of(COMPONENT, built.inputs.key) / LOG_NAME).read_text()


def test_a_pinned_checkouts_compile_runs_at_the_lowest_priority_on_the_build_cpus_the_record_names(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _affinity(bench, monkeypatch, pinned_affinity_recipe)

    assert built.status is BuildStatus.BUILT
    assert built.record.toolchain[BUILD_CPUS_KEY] == "0"
    assert "cpus [0] nice 19" in _log(bench, built)


def test_any_other_compile_keeps_this_process_cpus_and_priority_and_its_record_names_no_build_cpus(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolated_module.shutil, "which", lambda tool, *args, **kwargs: None)

    built = _affinity(bench, monkeypatch, affinity_recipe)

    assert built.status is BuildStatus.BUILT, "it needs neither nice nor taskset"
    assert BUILD_CPUS_KEY not in built.record.toolchain
    assert f"cpus {sorted(os.sched_getaffinity(0))} nice {os.nice(0)}" in _log(bench, built)


def test_a_host_without_nice_fails_a_pinned_checkouts_compile_as_the_machines(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolated_module.shutil, "which", lambda tool, *args, **kwargs: None)

    built = _affinity(bench, monkeypatch, pinned_affinity_recipe)

    assert (built.status, built.record) == (BuildStatus.INFRA, None)
    assert "taskset, nice is not on PATH" in built.detail


def test_the_build_cpus_are_a_cpu_list_of_this_host_and_anything_else_is_refused() -> None:
    own = min(os.sched_getaffinity(0))

    assert build_cpus({BUILD_CPUS_ENV: f" {own} "}) == str(own)
    assert build_cpus({}) == ""
    with pytest.raises(BuildStoreConfigError, match="HYPERLOOM_BUILD_CPUS: 'all' is not a CPU list"):
        build_cpus({BUILD_CPUS_ENV: "all"})
    with pytest.raises(
        BuildStoreConfigError, match="HYPERLOOM_BUILD_CPUS: .* a compile cannot run on here, the lowest CPU 100000"
    ):
        build_cpus({BUILD_CPUS_ENV: f"{own},100000-100001"})


# --- llama.cpp: a pinned checkout -----------------------------------------------------------------------------------


def _llamacpp(tmp_path: Path) -> NativeBuildSession:
    """A session whose host cannot isolate a compile: what is planned and keyed is real, nothing is compiled."""
    checkout = tmp_path / "sources" / "llamacpp" / LLAMACPP_COMMIT
    write_tree(checkout, {"CMakeLists.txt": "", "ggml/src/ggml.c": "a\n", "ggml/CMakeLists.txt": ""})
    machine = BuildMachine(IMAGE, BUILDER_IMAGE, "gfx1100")
    recipe = llamacpp_recipe()
    site = NativeSite(machine, {"llamacpp": recipe}, tmp_path / "sources", tmp_path, {"llamacpp": LLAMACPP_COMMIT})
    blobs = LocalBlobStore(tmp_path / "blobs")
    return open_session(Bench(tmp_path / "bench").store, Bench(tmp_path / "keys").signer, site, blobs, _unisolated)


def _unisolated() -> Unisolated:
    return Unisolated("no tier (a stand-in)")


def _llamacpp_stack(session: NativeBuildSession, tmp_path: Path, *paths: str) -> ComponentPatches:
    data = git_patch(tmp_path / "patch", dict.fromkeys(paths, "a\n"), dict.fromkeys(paths, "b\n"))
    patch = put_patch(session.compiler.blobs, "p.patch", data)
    return ComponentPatches("llamacpp", "https://github.com/ggml-org/llama.cpp", LLAMACPP_COMMIT, (patch,))


def test_llamacpps_pinned_commit_is_part_of_the_key_of_its_build_as_it_is_and_of_a_stack(tmp_path: Path) -> None:
    session = _llamacpp(tmp_path)

    base = session.base("llamacpp")
    stacked = session.build(_llamacpp_stack(session, tmp_path, "ggml/src/ggml.c"))

    assert (
        base.inputs.recipe_version == stacked.inputs.recipe_version == llamacpp_recipe().build_version(LLAMACPP_COMMIT)
    )
    assert base.inputs.key != stacked.inputs.key
    assert (base.status, stacked.status) == (BuildStatus.INFRA, BuildStatus.INFRA)
    assert session.planner.base("llamacpp").kind == "executable", "llama.cpp's build makes a server, no package"


@pytest.mark.parametrize("native", [(), ("ggml/src/ggml.c",)], ids=["alone", "beside-a-native-source"])
def test_a_llamacpp_stack_that_changes_what_no_build_of_it_takes_is_refused_naming_the_path(
    tmp_path: Path, native: tuple[str, ...]
) -> None:
    session = _llamacpp(tmp_path)
    path = "tools/llama-bench/llama-bench.cpp"

    with pytest.raises(PatchApplyError) as refused:
        session.build(_llamacpp_stack(session, tmp_path, *native, path))

    assert f"{path!r} is outside the checkout's sections (ggml/, src/, include/, common/" in str(refused.value)


# --- the pinned checkout --------------------------------------------------------------------------------------------


def test_a_component_built_from_a_pinned_checkout_is_built_at_the_head_of_the_framework_checkout(
    tmp_path: Path,
) -> None:
    checkout = llama_checkout(tmp_path / "llama.cpp")
    (checkout / "out-gfx1100").mkdir()
    (checkout / "out-gfx1100" / "llama-server").write_text("an operator's own build, untracked")

    found = pinned_checkouts(native_recipes(), {"FRAMEWORK_REPO_PATH": str(checkout)})

    assert found == [PinnedCheckout("llamacpp", git(checkout, "rev-parse", "HEAD"), checkout.resolve())]


def test_no_component_is_pinned_without_a_framework_checkout_that_is_its(tmp_path: Path) -> None:
    other = write_tree(tmp_path / "sglang", {"python/sglang/__init__.py": ""})

    assert pinned_checkouts(native_recipes(), {"FRAMEWORK_REPO_PATH": str(other)}) == []
