# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A test whose server sets a build variable that only aiter's module config reads, through the whole runner (the shapes
of ``test_run_jit``).

The image carries two prebuilt modules: ``module_gemm``, whose config entry passes
``-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT={os.environ.get("CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT", 2)}`` (the flag of the image
aiter's ``module_mha_batch_prefill``, 285579b62), and ``module_other``, which reads none. No Python of the build path reads
the variable: aiter evaluates the entry when it builds the module. A candidate that only sets it must neither share the
image's cache entries nor be given the image's ``module_gemm``, which was built under the image's value; a variable no entry
and no code names changes nothing.
"""

from __future__ import annotations

import json

import pytest

from hyperloom.measure.runner.tests.test_run_jit import (  # noqa: F401 - fixtures
    CORE,
    Rig,
    _arm,
    _keys,
    _kind,
    env,
    key,
    rig,
)

VARIABLE = "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT"
FLAG = "f'-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT={os.environ.get(\"CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT\", 2)}'"
IMAGE_MODULE = "image build"
TOUCHED = "the build variables touch 1 aiter modules, built when first called (module_gemm)"


@pytest.fixture
def image(rig: Rig) -> Rig:
    config = {
        "module_gemm": {"srcs": [], "extra_include": [], "flags_extra_hip": [FLAG], "blob_gen_cmd": "''"},
        "module_other": {"srcs": [], "extra_include": [], "blob_gen_cmd": "''"},
    }
    files = {
        "aiter/jit/core.py": CORE,
        "aiter/jit/optCompilerConfig.json": json.dumps(config),
        "aiter/jit/module_gemm.so": IMAGE_MODULE,
        "aiter/jit/module_other.so": IMAGE_MODULE,
    }
    for relative, text in files.items():
        path = rig.bases / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return rig


def test_a_candidate_that_only_sets_a_variable_the_module_config_reads_misses_the_cache_and_builds_the_module(
    image,
) -> None:
    base, base_status, _ = image.run("base", _arm(image))
    rounding, status, _ = image.run("rounding", _arm(image, envs={VARIABLE: "3"}))

    assert _keys(rounding)["aiter_jit"] != _keys(base)["aiter_jit"], "it does not read what was built under 2"
    assert _keys(rounding)["triton"] == _keys(base)["triton"]
    assert TOUCHED in status["message"] and TOUCHED not in base_status["message"]
    contributed = _kind(rounding, "aiter_jit").contribution
    assert contributed is not None and contributed.files == 1, "module_gemm.so is the test's; module_other.so a link"
    assert (image.bases / "aiter/jit/module_gemm.so").read_text() == IMAGE_MODULE, "the image keeps its module"
    base_contribution = _kind(base, "aiter_jit").contribution
    assert base_contribution is None or base_contribution.files == 0, "the base test runs the image's module"


def test_the_value_the_runner_already_has_is_the_images_and_changes_nothing(image) -> None:
    inherited = {VARIABLE: "3"}
    base, _, _ = image.run("base", _arm(image), extra_env=inherited)
    same, status, _ = image.run("same", _arm(image, envs={VARIABLE: "3"}), extra_env=inherited)

    assert _keys(same) == _keys(base)
    assert "aiter modules, built when first called" not in status["message"]


def test_a_variable_no_entry_and_no_code_names_changes_nothing(image) -> None:
    base, _, _ = image.run("base", _arm(image))
    unrelated, status, _ = image.run("unrelated", _arm(image, envs={"CK_TILE_NOT_READ_BY_ANY_BUILD": "3"}))

    assert _keys(unrelated) == _keys(base)
    assert "aiter modules, built when first called" not in status["message"]
