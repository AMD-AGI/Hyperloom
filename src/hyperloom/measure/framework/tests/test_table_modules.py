# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which prebuilt modules a test's tuned tables make stale.

The image is an aiter package whose ``jit/core.py`` declares two table variables the way aiter's does, with their default
tables below ``configs`` and a per-model table below ``configs/model_configs``. The module entries are aiter's own
(``optCompilerConfig.json`` of the measured image, e7d2453f): two GEMM modules generated from one variable each, one
(``module_deepgemm_opus``) that also names files below ``configs`` directly, and a tuning module generated from no table.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hyperloom.measure.framework.rows import DEFAULT_TABLE_PLACEHOLDER
from hyperloom.measure.framework.table_modules import named_table_modules, stale_table_modules, table_modules

A8W8, BPRESHUFFLE = "AITER_CONFIG_GEMM_A8W8", "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"
GEMM, BLOCK, OPUS, TUNE = (
    "module_gemm_a8w8",
    "module_gemm_a8w8_blockscale_bpreshuffle",
    "module_deepgemm_opus",
    "module_gemm_a8w8_tune",
)
CONFIG = {
    GEMM: {
        "blob_gen_cmd": "f'{AITER_CSRC_DIR}/ck_gemm_a8w8/gen_instances.py --working_path {{}} "
        "--tune_file {AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE}'"
    },
    BLOCK: {
        "blob_gen_cmd": "f'{AITER_CSRC_DIR}/ck_gemm_a8w8_blockscale_bpreshuffle/gen_instances.py --working_path {{}} "
        "--tune_file {AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE}'"
    },
    OPUS: {
        "blob_gen_cmd": "f'{AITER_CSRC_DIR}/opus_gemm/gen_instances.py --working_path {{0}} --tune_files "
        "{AITER_ROOT_DIR}/aiter/configs/bf16_tuned_gemm.csv:{AITER_ROOT_DIR}/aiter/configs/model_configs/"
        "*_bf16_tuned_gemm.csv:{AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE} "
        "--compiled_kids_sidecar={{0}}/compiled_kids_opus.json'"
    },
    TUNE: {"blob_gen_cmd": "f'{AITER_CSRC_DIR}/ck_gemm_a8w8/gen_instances.py --working_path {{}} --tune'"},
    "module_plain": {"srcs": ["f'{AITER_CSRC_DIR}/pybind/plain.cu'"], "blob_gen_cmd": "''"},
}
CORE = """
import os
AITER_CONFIG_GEMM_A8W8 = os.getenv("AITER_CONFIG_GEMM_A8W8", f"{AITER_ROOT_DIR}/aiter/configs/a8w8_tuned_gemm.csv")
AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE = os.getenv(
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    f"{AITER_ROOT_DIR}/aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
)
class AITER_CONFIG:
    @property
    def AITER_CONFIG_GEMM_A8W8_FILE(self):
        return self.get_config_file("AITER_CONFIG_GEMM_A8W8", AITER_CONFIG_GEMM_A8W8, "a8w8_tuned_gemm")
    @property
    def AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
            AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE,
            "a8w8_blockscale_bpreshuffle_tuned_gemm",
        )
"""
TABLES = {
    "configs/a8w8_tuned_gemm.csv": "M,N,K,kernelName\n1,2,3,a8w8_k1\n",
    "configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv": "M,N,K,kernelName\n1,2,3,bp_k1\n",
    "configs/model_configs/dsv3_a8w8_blockscale_bpreshuffle_tuned_gemm.csv": "M,N,K,kernelName\n4,5,6,bp_k2\n",
    "configs/model_configs/untuned_dsv3_a8w8_blockscale_bpreshuffle_tuned_gemm.csv": "M,N,K\n4,5,6\n",
    "configs/bf16_tuned_gemm.csv": "M,N,K,kernelName\n1,2,3,bf16_k1\n",
}
OTHER_ROWS = "M,N,K,kernelName\n7,8,9,a8w8_k9\n"


def _package(root: Path, changes: dict[str, str | None] | None = None) -> Path:
    files = {"jit/core.py": CORE, **TABLES, **(changes or {})}
    for relative, text in files.items():
        if text is not None:
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def image(tmp_path) -> Path:
    return _package(tmp_path / "image" / "aiter")


def _farm(tmp_path: Path, image: Path, changes: dict[str, str | None]) -> Path:
    """The test's copy of the package (what a farm lays for a patched aiter), with ``changes`` laid over it."""
    farm = tmp_path / "farm" / "aiter"
    shutil.copytree(image, farm)
    for relative, text in changes.items():
        if text is None:
            (farm / relative).unlink()
        else:
            (farm / relative).parent.mkdir(parents=True, exist_ok=True)
            (farm / relative).write_text(text, encoding="utf-8")
    return farm


def _table(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / "tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_the_modules_generated_from_a_table_are_read_from_their_commands() -> None:
    uses = table_modules(CONFIG)

    assert set(uses) == {GEMM, BLOCK, OPUS}, "the tuning module and a module with no generator read no table"
    assert uses[GEMM].variables == {A8W8} and uses[BLOCK].variables == {BPRESHUFFLE}
    assert uses[OPUS].variables == {BPRESHUFFLE}
    assert uses[OPUS].files == ("configs/bf16_tuned_gemm.csv", "configs/model_configs/*_bf16_tuned_gemm.csv")


def test_a_variable_that_names_another_table_makes_its_module_stale_and_leaves_the_image_as_it_was(
    tmp_path, image
) -> None:
    before = (image / "configs/a8w8_tuned_gemm.csv").read_bytes()
    environment = {A8W8: _table(tmp_path, "mine.csv", OTHER_ROWS)}

    assert stale_table_modules(CONFIG, image, image, [environment]) == {GEMM}
    assert (image / "configs/a8w8_tuned_gemm.csv").read_bytes() == before


def test_a_variable_that_names_a_copy_of_the_images_table_elsewhere_makes_nothing_stale(tmp_path, image) -> None:
    environment = {A8W8: _table(tmp_path, "copy.csv", TABLES["configs/a8w8_tuned_gemm.csv"])}

    assert stale_table_modules(CONFIG, image, image, [environment]) == frozenset()


def test_a_variable_set_to_the_images_default_and_per_model_tables_by_path_makes_nothing_stale(image) -> None:
    listed = [image / "configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv"]
    listed.append(image / "configs/model_configs/dsv3_a8w8_blockscale_bpreshuffle_tuned_gemm.csv")
    environment = {BPRESHUFFLE: ":".join(str(path) for path in listed)}

    assert stale_table_modules(CONFIG, image, image, [environment]) == frozenset()


def test_a_patch_to_a_default_table_makes_the_modules_of_its_variable_stale(tmp_path, image) -> None:
    farm = _farm(tmp_path, image, {"configs/a8w8_tuned_gemm.csv": OTHER_ROWS})

    assert stale_table_modules(CONFIG, image, farm, [{}]) == {GEMM}


def test_a_patch_that_adds_a_per_model_table_makes_every_module_of_that_variable_stale(tmp_path, image) -> None:
    farm = _farm(tmp_path, image, {"configs/model_configs/qwen_a8w8_blockscale_bpreshuffle_tuned_gemm.csv": OTHER_ROWS})

    assert stale_table_modules(CONFIG, image, farm, [{}]) == {BLOCK, OPUS}


def test_a_patch_to_a_file_a_command_names_directly_makes_that_module_stale(tmp_path, image) -> None:
    farm = _farm(tmp_path, image, {"configs/bf16_tuned_gemm.csv": OTHER_ROWS})

    assert stale_table_modules(CONFIG, image, farm, [{}]) == {OPUS}


def test_an_untuned_table_beside_the_per_model_tables_is_not_one_aiter_merges(tmp_path, image) -> None:
    farm = _farm(
        tmp_path, image, {"configs/model_configs/untuned_q_a8w8_blockscale_bpreshuffle_tuned_gemm.csv": "M\n1\n"}
    )

    assert stale_table_modules(CONFIG, image, farm, [{}]) == frozenset()


def test_a_patched_package_whose_tables_are_the_images_makes_nothing_stale(tmp_path, image) -> None:
    farm = _farm(tmp_path, image, {"ops/mha.py": "X = 1\n"})

    assert stale_table_modules(CONFIG, image, farm, [{}]) == frozenset()


def test_a_kernel_tunes_rows_over_the_images_default_table_make_its_modules_stale(tmp_path, image) -> None:
    rows = _table(tmp_path, "k-0123456789abcdef", OTHER_ROWS)

    tuned = {BPRESHUFFLE: f"{DEFAULT_TABLE_PLACEHOLDER}:{rows}"}
    default_only = {BPRESHUFFLE: DEFAULT_TABLE_PLACEHOLDER}

    assert stale_table_modules(CONFIG, image, image, [tuned]) == {BLOCK, OPUS}
    assert stale_table_modules(CONFIG, image, image, [default_only]) == frozenset(), (
        "the placeholder alone is the image's default table and its per-model tables"
    )


def test_a_kernel_tunes_whole_table_makes_its_modules_stale(tmp_path, image) -> None:
    environment = {A8W8: _table(tmp_path, "k-fedcba9876543210", OTHER_ROWS)}

    assert stale_table_modules(CONFIG, image, image, [environment]) == {GEMM}


def test_a_table_that_cannot_be_read_counts_as_different(tmp_path, image) -> None:
    environment = {A8W8: str(tmp_path / "missing.csv")}

    assert stale_table_modules(CONFIG, image, image, [environment]) == {GEMM}


def test_a_variable_the_package_does_not_declare_counts_as_different_when_it_is_set(tmp_path, image) -> None:
    config = {"module_x": {"blob_gen_cmd": "f'{AITER_CSRC_DIR}/g.py --tune_file {AITER_CONFIGS.AITER_CONFIG_X_FILE}'"}}
    copy = _table(tmp_path, "x.csv", TABLES["configs/a8w8_tuned_gemm.csv"])

    assert stale_table_modules(config, image, image, [{"AITER_CONFIG_X": copy}]) == {"module_x"}
    assert stale_table_modules(config, image, image, [{}]) == frozenset()


def test_a_command_that_names_the_tables_in_another_way_is_stale_whenever_any_table_differs(tmp_path, image) -> None:
    config = {**CONFIG, "module_odd": {"blob_gen_cmd": "f'{AITER_CSRC_DIR}/g.py --tune_file {AITER_CONFIGS.odd()}'"}}
    other = {A8W8: _table(tmp_path, "mine.csv", OTHER_ROWS)}
    same = {A8W8: _table(tmp_path, "copy.csv", TABLES["configs/a8w8_tuned_gemm.csv"])}

    assert stale_table_modules(config, image, image, [other]) == {GEMM, "module_odd"}
    assert stale_table_modules(config, image, image, [same]) == frozenset()
    assert stale_table_modules(config, image, _farm(tmp_path, image, {"configs/new.csv": "a\n"}), [{}]) == {
        "module_odd"
    }


def test_a_table_that_differs_in_any_point_makes_its_module_stale(tmp_path, image) -> None:
    second = {A8W8: _table(tmp_path, "mine.csv", OTHER_ROWS)}

    assert stale_table_modules(CONFIG, image, image, [{}, second]) == {GEMM}


def test_before_a_test_runs_every_table_it_names_counts(image) -> None:
    assert named_table_modules(CONFIG, image, [A8W8], []) == {GEMM}
    assert named_table_modules(CONFIG, image, [], ["configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv"]) == {
        BLOCK,
        OPUS,
    }
    assert named_table_modules(
        CONFIG, image, [], ["configs/model_configs/q_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"]
    )
    assert named_table_modules(CONFIG, image, [], ["configs/bf16_tuned_gemm.csv"]) == {OPUS}
    untuned = "configs/model_configs/untuned_q_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
    assert named_table_modules(CONFIG, image, ["AITER_CONFIG_OTHER"], [untuned, "ops/mha.py"]) == frozenset()
