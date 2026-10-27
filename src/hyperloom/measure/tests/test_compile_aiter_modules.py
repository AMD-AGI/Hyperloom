# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Acceptance with the CPU build of aiter's touched modules (#125 stage 2b): which modules the build is asked for, the
build the compiled environment then needs, and the time limit counted in CPU build time for the modules it makes and in
card time for the rest. Without the plan everything is as in ``test_compile_aiter_meta`` (2a)."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.recipe import LaunchConfig, RecipeDelta
from hyperloom.measure.compile import EnvironmentCompiler, LocalBlobStore, PackageBase, TooManyModulesError
from hyperloom.measure.jit.modules import ESTIMATED_MODULE_SECONDS, MAX_REBUILD_SECONDS
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.needs import ModuleBuild
from hyperloom.measure.runner.native.aiter_modules import AiterModules, buildable_modules
from hyperloom.measure.tests._recipes import IMAGE, git_patch, put_patch, recipe_with
from hyperloom.measure.tests.test_compile_aiter_meta import LIMIT_MODULES, NEW, aiter_image
from hyperloom.measure.tests.test_compile_aiter_tables import VARIABLE, table_image


@dataclass
class Plan:
    """A module plan that builds every candidate that is a config entry, each in ``each`` CPU seconds."""

    each: float
    asked: list[frozenset[str]] = field(default_factory=list)

    def __call__(self, roots: tuple, config: Mapping[str, Any], candidates: Collection[str]) -> ModuleBuild | None:
        self.asked.append(frozenset(candidates))
        built = frozenset(name for name in candidates if name in config)
        inputs = BuildInputs(IMAGE, IMAGE, "aiter", "gfx942", "0" * 64, "aiter_jit/1")
        return ModuleBuild(inputs, built) if built else None

    def seconds(self, modules: Collection[str]) -> float:
        return len(modules) * self.each


def compile_with(tmp_path: Path, base: PackageBase, files: dict, plan: Plan | None, envs: dict | None = None):
    blobs = LocalBlobStore(tmp_path / "blobs")
    before = {path: (base.path.parent / path).read_text() for path in files}
    patch = put_patch(blobs, "p.patch", git_patch(tmp_path / "up", before, files))
    recipe = recipe_with((patch,), component="aiter", launch=LaunchConfig(envs=envs or {}))
    compiler = EnvironmentCompiler(blobs, IMAGE, lambda layout: base, modules=plan)
    return compiler.compile(recipe, RecipeDelta())


def test_a_touched_module_is_asked_of_the_cpu_build_and_the_environment_needs_that_build(tmp_path) -> None:
    plan = Plan(each=10.0)

    compiled = compile_with(tmp_path, aiter_image(tmp_path / "image"), {"csrc/kernels/m0.cu": NEW}, plan)

    assert plan.asked == [frozenset({"module_m0"})]
    ((need),) = compiled.builds
    assert (need.root_id, need.inputs.component) == ("aiter", "aiter")


def test_a_stack_that_touches_no_module_needs_no_build(tmp_path) -> None:
    plan = Plan(each=10.0)

    compiled = compile_with(tmp_path, aiter_image(tmp_path / "image"), {"aiter/ops/mha.py": NEW}, plan)

    assert plan.asked == [] and compiled.builds == ()


def test_too_many_modules_for_the_card_fit_the_cpu_build_when_it_is_fast_enough(tmp_path) -> None:
    modules = LIMIT_MODULES + 1
    plan = Plan(each=MAX_REBUILD_SECONDS / (2 * modules))

    compiled = compile_with(tmp_path, aiter_image(tmp_path / "image", modules), {"csrc/include/common.h": NEW}, plan)

    assert len(plan.asked[0]) == modules and len(compiled.builds) == 1


def test_a_cpu_build_that_would_not_fit_its_time_is_refused_naming_where_it_builds(tmp_path) -> None:
    modules = LIMIT_MODULES + 1
    base = aiter_image(tmp_path / "image", modules)

    with pytest.raises(TooManyModulesError) as refused:
        compile_with(tmp_path, base, {"csrc/include/common.h": NEW}, Plan(each=float(ESTIMATED_MODULE_SECONDS)))

    message = str(refused.value)
    assert f"touches {modules} of aiter's prebuilt modules" in message
    assert f"about {modules * ESTIMATED_MODULE_SECONDS} s to build again on a CPU build runner" in message


def test_a_module_whose_table_the_launch_names_is_left_to_the_card_and_counted_there(tmp_path) -> None:
    base = table_image(tmp_path / "image", LIMIT_MODULES + 1)
    plan = Plan(each=1.0)

    with pytest.raises(TooManyModulesError, match="to build again on the first test's card"):
        compile_with(tmp_path, base, {"csrc/kernels/m0.cu": NEW}, plan, envs={VARIABLE: "/tables/mine.csv"})

    assert plan.asked == [], "only the server reads the table its variable names"


def test_only_entries_aiter_builds_by_name_alone_are_built_on_cpu() -> None:
    config = {
        "module_a": {"srcs": []},
        "module_tp": {"third_party": "['hipkittens']"},
        "module_standalone": {"is_standalone": "True"},
        "module_lib": {"is_python_module": "False"},
    }

    built = buildable_modules(config, {*config, "mha_varlen_fwd_bf16"})

    assert built == {"module_a"}, "a variant aiter names at run time has no entry of its own"


@dataclass
class _Record:
    status: str
    outputs: tuple
    build_seconds: float


def test_the_cpu_estimate_is_the_median_cpu_seconds_of_a_module_spread_over_max_jobs() -> None:
    history = [_Record("ready", ("a", "b"), 160.0), _Record("ready", ("a",), 40.0), _Record("failed", (), 1.0)]
    plan = AiterModules(planner=None, history=lambda: history)  # type: ignore[arg-type]

    assert plan.seconds({"x", "y", "z"}) == 3 * 60.0 / 8, "per module 80 and 40 s of CPU, median 60, MAX_JOBS 8"


def test_with_no_build_of_the_image_yet_a_module_is_estimated_as_on_a_card() -> None:
    plan = AiterModules(planner=None)  # type: ignore[arg-type]

    assert plan.seconds({"x", "y"}) == 2 * ESTIMATED_MODULE_SECONDS
