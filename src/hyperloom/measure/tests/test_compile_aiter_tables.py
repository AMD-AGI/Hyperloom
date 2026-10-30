# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Acceptance counts the modules generated from a tuned table the stack may change, with the modules its patch touches,
against the time their first build is given (the image of ``test_compile_aiter_meta``, its modules generated from the
table of ``AITER_CONFIG_GEMM_A8W8`` as aiter's GEMM modules are)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import LaunchConfig, RecipeDelta
from hyperloom.measure.compile import LocalBlobStore, PackageBase, TooManyModulesError, compile_environment
from hyperloom.measure.tests._recipes import IMAGE, git_patch, put_patch, recipe_with, write_tree
from hyperloom.measure.tests.test_compile_aiter_meta import LIMIT_MODULES, NEW, aiter_image, module_entry

VARIABLE = "AITER_CONFIG_GEMM_A8W8"
TABLE = "aiter/configs/a8w8_tuned_gemm.csv"
CORE = (
    f'{VARIABLE} = os.getenv("{VARIABLE}", f"{{AITER_ROOT_DIR}}/{TABLE}")\n'
    "class AITER_CONFIG:\n"
    "    @property\n"
    f"    def {VARIABLE}_FILE(self):\n"
    f'        return self.get_config_file("{VARIABLE}", {VARIABLE}, "a8w8_tuned_gemm")\n'
)
COMMAND = (
    f"f'{{AITER_CSRC_DIR}}/ck/gen_instances.py --working_path {{{{}}}} --tune_file {{AITER_CONFIGS.{VARIABLE}_FILE}}'"
)


def table_image(root: Path, modules: int) -> PackageBase:
    """``modules`` prebuilt modules, all generated from the variable's table."""
    base = aiter_image(root, modules)
    config = {f"module_m{index}": {**module_entry(index), "blob_gen_cmd": COMMAND} for index in range(modules)}
    write_tree(
        root,
        {"aiter/jit/optCompilerConfig.json": json.dumps(config), "aiter/jit/core.py": CORE, TABLE: "M,N,K\n1,2,3\n"},
    )
    return base


def compile_stack(tmp_path: Path, base: PackageBase, files: dict[str, str], envs: dict[str, str] | None = None):
    blobs = LocalBlobStore(tmp_path / "blobs")
    root = base.path.parent
    before = {path: (root / path).read_text() for path in files}
    patch = put_patch(blobs, "p.patch", git_patch(tmp_path / "up", before, files))
    recipe = recipe_with((patch,), component="aiter", launch=LaunchConfig(envs=envs or {}))
    return compile_environment(recipe, RecipeDelta(), blobs=blobs, active_image_digest=IMAGE, locator=lambda _: base)


def test_a_patch_to_a_table_that_too_many_modules_are_generated_from_is_refused_with_the_count(tmp_path) -> None:
    base = table_image(tmp_path / "image", LIMIT_MODULES + 1)

    with pytest.raises(TooManyModulesError, match=f"touches {LIMIT_MODULES + 1} of aiter's prebuilt modules"):
        compile_stack(tmp_path, base, {TABLE: "M,N,K\n7,8,9\n"})


def test_a_variable_the_launch_sets_counts_its_modules_with_the_ones_a_source_patch_touches(tmp_path) -> None:
    base = table_image(tmp_path / "image", LIMIT_MODULES + 1)

    with pytest.raises(TooManyModulesError, match=f"touches {LIMIT_MODULES + 1} of aiter's prebuilt modules"):
        compile_stack(tmp_path, base, {"csrc/kernels/m0.cu": NEW}, envs={VARIABLE: "/tables/mine.csv"})


def test_the_same_stack_without_the_variable_touches_only_the_module_its_patch_reaches(tmp_path) -> None:
    base = table_image(tmp_path / "image", LIMIT_MODULES + 1)

    environment = compile_stack(tmp_path, base, {"csrc/kernels/m0.cu": NEW}, envs={"AITER_CONFIG_OTHER": "/t.csv"})

    assert [entry.path for root in environment.roots for entry in root.entries] == ["csrc/kernels/m0.cu"]
