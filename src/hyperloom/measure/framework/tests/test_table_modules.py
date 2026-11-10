# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which prebuilt modules a test's tuned tables make stale.

The image is an aiter package whose ``jit/core.py`` declares two table variables the way aiter's does, with their default
tables below ``configs`` and a per-model table below ``configs/model_configs``. The module entries are aiter's own
(``optCompilerConfig.json`` of the measured image, e7d2453f): two GEMM modules generated from one variable each, one
(``module_deepgemm_opus``) that also names files below ``configs`` directly, and a tuning module generated from no table.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import yaml

from hyperloom.measure.framework.base import framework_envs
from hyperloom.measure.framework.copies import private_framework
from hyperloom.measure.framework.rows import DEFAULT_TABLE_PLACEHOLDER
from hyperloom.measure.framework.table_modules import (
    _shares_a_key,
    named_table_modules,
    stale_table_modules,
    table_modules,
)
from hyperloom.measure.socket import TestCall

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


#: Three more per-model tables of the bpreshuffle variable, so the order they are listed in can differ from the sorted one.
MODEL_TABLES = {
    f"configs/model_configs/{name}_a8w8_blockscale_bpreshuffle_tuned_gemm.csv": f"M,N,K,kernelName\n{m},5,6,bp_{name}\n"
    for m, name in ((7, "glm5"), (8, "kimi"), (9, "qwen"))
}


def _listing(paths: list[Path]) -> str:
    return os.pathsep.join(str(path) for path in paths)


def test_the_images_tables_listed_in_another_order_make_nothing_stale(tmp_path) -> None:
    image = _package(tmp_path / "image" / "aiter", MODEL_TABLES)
    copies = tmp_path / "copies"
    shutil.copytree(image / "configs", copies)
    tuned = (copies / "model_configs").glob("*_a8w8_blockscale_bpreshuffle_tuned_gemm.csv")
    models = sorted((path for path in tuned if "untuned" not in path.name), reverse=True)
    reordered = {BPRESHUFFLE: _listing([copies / "a8w8_blockscale_bpreshuffle_tuned_gemm.csv", *models])}
    one_left_out = {BPRESHUFFLE: _listing([copies / "a8w8_blockscale_bpreshuffle_tuned_gemm.csv", *models[1:]])}

    assert stale_table_modules(CONFIG, image, image, [reordered]) == frozenset()
    assert stale_table_modules(CONFIG, image, image, [one_left_out]) == {BLOCK, OPUS}, "other content stays stale"


def test_the_private_copies_a_test_is_redirected_to_make_nothing_stale_whatever_order_the_directory_lists(
    tmp_path, monkeypatch
) -> None:
    """The real redirect: the session's snapshot of the image's tables, and the value ``private_framework`` gives the
    server for a variable the test leaves unset, its private copies in the order the shared directory listed them."""
    for name in [name for name in os.environ if name.startswith(("AITER_CONFIG_", "HYPERLOOM_FRAMEWORK_"))]:
        monkeypatch.delenv(name)
    image = _package(tmp_path / "image" / "aiter", MODEL_TABLES)
    for name, value in framework_envs(tmp_path / "framework", str(image / "configs")).items():
        monkeypatch.setenv(name, value)
    manifest = tmp_path / "framework" / "snapshot.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["model_order"] = sorted(document["model_order"], reverse=True)  # as an ext4 or overlay directory may list
    manifest.write_text(json.dumps(document), encoding="utf-8")
    point = tmp_path / "point.yaml"
    point.write_text(yaml.safe_dump({"benchmark": {"envs": {"TP": 1}}}), encoding="utf-8")
    call = TestCall(
        points=(point,), output_dir=tmp_path / "out", timeout_sec=60, cwd=str(tmp_path), magpie_python="", task_id="t1"
    )

    with private_framework(call) as private:
        served = dict(private.call.env_overrides)
        listed = [Path(part).name for part in served[BPRESHUFFLE].split(os.pathsep)]
        judged = stale_table_modules(CONFIG, image, image, [served])

    assert listed[1:] == sorted(listed[1:], reverse=True), "the server is told the copies in the directory's order"
    assert judged == frozenset()


def _merge_order(models: Path, pattern: str) -> list[Path]:
    """The per-model tables below ``models`` that ``pattern`` matches, in the order the directory lists them."""
    return [path for path in models.glob(pattern) if "untuned" not in path.name]


def test_the_images_tables_with_one_changed_under_its_own_name_are_other_tables(tmp_path) -> None:
    """The same names and as many files, one of them holding another kernel: compared by content, not by name."""
    image = _package(tmp_path / "image" / "aiter", MODEL_TABLES)
    copies = tmp_path / "copies"
    shutil.copytree(image / "configs", copies)
    models = sorted(_merge_order(copies / "model_configs", "*_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"))
    environment = {BPRESHUFFLE: _listing([copies / "a8w8_blockscale_bpreshuffle_tuned_gemm.csv", *models])}
    unchanged = stale_table_modules(CONFIG, image, image, [environment])
    kimi = copies / "model_configs" / "kimi_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
    kimi.write_text(kimi.read_text(encoding="utf-8").replace("bp_kimi", "bp_kimx"), encoding="utf-8")

    assert unchanged == frozenset()
    assert stale_table_modules(CONFIG, image, image, [environment]) == {BLOCK, OPUS}, "same names, same count"


def test_the_same_table_listed_twice_is_not_the_table_listed_once(image) -> None:
    default = image / "configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
    dsv3 = image / "configs/model_configs/dsv3_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"

    assert stale_table_modules(CONFIG, image, image, [{BPRESHUFFLE: _listing([default, dsv3, dsv3])}]) == {BLOCK, OPUS}


def test_a_listed_table_that_is_not_a_file_is_left_out_as_aiter_leaves_it_out(tmp_path, image) -> None:
    default = image / "configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
    dsv3 = image / "configs/model_configs/dsv3_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
    environment = {BPRESHUFFLE: _listing([default, tmp_path / "missing.csv", dsv3])}

    assert stale_table_modules(CONFIG, image, image, [environment]) == frozenset()


#: A GEMM table's columns: the shape a generator keys a row by, ``q_dtype_w`` that aiter's merge keys a row by as well,
#: the libtype a generator may keep the rows of, and the kernel.
SHAPE_HEADER = "gfx,cu_num,M,N,K,q_dtype_w,libtype,kernelId,kernelName\n"


def _one_shape(variable: str, tuned: str, kinds: tuple[str, str]) -> dict[str, str | None]:
    """``variable``'s default table (``tuned``) and two per-model tables (glm5, kimi) holding one shape with another
    ``q_dtype_w`` and kernel, of the libtypes ``kinds``: aiter's merge takes both rows, and a generator that keys by the
    shape compiles the kernel of the row merged last."""
    first, second = kinds
    return {
        f"configs/{tuned}.csv": SHAPE_HEADER + f"gfx950,256,16,1536,5120,torch.float8_e4m3fn,{first},2,{variable}_k2\n",
        f"configs/model_configs/glm5_{tuned}.csv": SHAPE_HEADER
        + f"gfx950,256,64,1536,5120,torch.float8_e4m3fn,{first},0,{variable}_k0\n",
        f"configs/model_configs/kimi_{tuned}.csv": SHAPE_HEADER
        + f"gfx950,256,64,1536,5120,torch.float8_e4m3fnuz,{second},1,{variable}_k1\n",
    }


#: The bpreshuffle variable's per-model table of the shared fixture, left out where the key matters: of its columns it has
#: only M, N and K, and a key is cut to the columns every table with rows has.
DSV3 = "configs/model_configs/dsv3_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"


def test_tables_that_hold_one_shape_count_in_the_order_aiter_merges_them(tmp_path, monkeypatch) -> None:
    """The image's directory lists its per-model tables against their names here; aiter merges them in that order."""
    listing = Path.glob
    monkeypatch.setattr(Path, "glob", lambda self, *a, **k: iter(sorted(listing(self, *a, **k), reverse=True)))
    tables = _one_shape("bp", "a8w8_blockscale_bpreshuffle_tuned_gemm", ("ck", "ck"))
    image = _package(tmp_path / "image" / "aiter", {**tables, DSV3: None})
    default = image / "configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
    glm5, kimi = (
        image / "configs/model_configs" / f"{name}_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
        for name in ("glm5", "kimi")
    )
    as_the_image = {BPRESHUFFLE: _listing([default, kimi, glm5])}
    by_name = {BPRESHUFFLE: _listing([default, glm5, kimi])}

    assert stale_table_modules(CONFIG, image, image, [as_the_image]) == frozenset()
    assert stale_table_modules(CONFIG, image, image, [by_name]) == {BLOCK, OPUS}, "the other row is merged last"


@pytest.mark.parametrize("content", ["", "\n", "M,N,K,kernelName\n"], ids=["empty", "blank-line", "short-header"])
def test_a_table_without_rows_shares_no_key_whatever_its_header(tmp_path, content) -> None:
    """aiter merges a header-only table with fewer columns than the others; its missing key columns hold no row."""
    header = "gfx,cu_num,M,N,K,libtype,kernelName\n"
    first, second, bare = tmp_path / "first.csv", tmp_path / "second.csv", tmp_path / "bare.csv"
    first.write_text(header + "gfx950,256,64,1536,5120,ck,a\n")
    second.write_text(header + "gfx950,256,128,1536,5120,ck,b\n")
    bare.write_text(content)
    key = ("gfx", "cu_num", "M", "N", "K", "libtype")

    assert _shares_a_key([str(first), str(second), str(bare)], [key]) is False
    second.write_text(header + "gfx950,256,64,1536,5120,ck,b\n")
    assert _shares_a_key([str(first), str(bare), str(second)], [key]) is True


def test_rows_of_another_libtype_hold_no_shape_of_a_generator_that_keeps_one_libtype(tmp_path) -> None:
    """``ck_gemm_a8w8_blockscale_bpreshuffle`` keeps the ``ck`` rows; ``ck_gemm_a8w8`` keeps every row."""
    tables = {
        **_one_shape("bp", "a8w8_blockscale_bpreshuffle_tuned_gemm", ("ck", "cktile")),
        **_one_shape("a8w8", "a8w8_tuned_gemm", ("ck", "cktile")),
    }
    image = _package(tmp_path / "image" / "aiter", {**tables, DSV3: None})
    configs = image / "configs"
    reversed_bp = [
        configs / "a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
        *reversed(_merge_order(configs / "model_configs", "*a8w8_blockscale_bpreshuffle_tuned_gemm*.csv")),
    ]
    reversed_a8w8 = [
        configs / "a8w8_tuned_gemm.csv",
        *reversed(_merge_order(configs / "model_configs", "*a8w8_tuned_gemm*.csv")),
    ]

    assert stale_table_modules(CONFIG, image, image, [{BPRESHUFFLE: _listing(reversed_bp)}]) == frozenset()
    assert stale_table_modules(CONFIG, image, image, [{A8W8: _listing(reversed_a8w8)}]) == {GEMM}


def test_tables_a_generator_not_known_reads_count_in_their_order_whatever_rows_they_hold(tmp_path) -> None:
    config = {
        **CONFIG,
        "module_new": {
            "blob_gen_cmd": "f'{AITER_CSRC_DIR}/new_gemm/gen_instances.py --working_path {{}} "
            "--tune_file {AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE}'"
        },
    }
    image = _package(tmp_path / "image" / "aiter", MODEL_TABLES)
    configs = image / "configs"
    models = _merge_order(configs / "model_configs", "*a8w8_blockscale_bpreshuffle_tuned_gemm*.csv")
    default = configs / "a8w8_blockscale_bpreshuffle_tuned_gemm.csv"

    assert stale_table_modules(config, image, image, [{BPRESHUFFLE: _listing([default, *models])}]) == frozenset()
    assert stale_table_modules(config, image, image, [{BPRESHUFFLE: _listing([default, *reversed(models)])}]) == {
        BLOCK,
        OPUS,
        "module_new",
    }
    assert stale_table_modules(CONFIG, image, image, [{BPRESHUFFLE: _listing([default, *reversed(models)])}]) == (
        frozenset()
    ), "the generators known here hold no shape of these tables twice"


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
