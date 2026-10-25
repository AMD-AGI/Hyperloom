# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which prebuilt modules a patch to aiter's sources makes stale, on a tree and a config shaped like the measured image's:
config values are Python expressions (``f'{AITER_CSRC_DIR}/...'``), a source may be a directory or a header, a header is
found beside the including file or in ``extra_include`` or ``csrc/include``, a generator is a script below ``csrc`` or
``hsa/codegen.py -m <name>``, and some prebuilt names are made at run time."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.measure.jit.modules import SourceTree, config_path, prebuilt_names, touched_entries, touched_modules
from hyperloom.measure.tests._recipes import write_tree

CSRC = "f'{AITER_CSRC_DIR}/"
CONFIG = {
    "module_activation": {
        "srcs": [CSRC + "pybind/activation_pybind.cu'", CSRC + "kernels/activation_kernels.cu'"],
        "extra_include": [],
        "blob_gen_cmd": "''",
    },
    "module_gemm_a8w8": {
        "srcs": [CSRC + "ck_gemm_a8w8/gemm_a8w8.cu'"],
        "extra_include": [CSRC + "ck_gemm_a8w8/include'"],
        "blob_gen_cmd": CSRC + "ck_gemm_a8w8/gen_instances.py --working_path {{}} --tune_file {AITER_CONFIGS.F}'",
    },
    "module_moe_topk": {"srcs": [CSRC + "kernels/topk_gating'"], "extra_include": [], "blob_gen_cmd": "''"},
    "module_pa": {
        "srcs": [CSRC + "py_itfs_cu/asm_pa.cu'"],
        "extra_include": [],
        "blob_gen_cmd": "f'{AITER_META_DIR}/hsa/codegen.py -m pa --output_dir {{}}'",
    },
    "module_mla": {
        "srcs": [CSRC + "py_itfs_cu/asm_mla.cu'"],
        "extra_include": [],
        "blob_gen_cmd": "f'{AITER_META_DIR}/hsa/codegen.py -m mla --output_dir {{}}'",
    },
    "module_moe_ck2stages": {"srcs": [CSRC + "ck_moe/moe.cu'"], "extra_include": [], "blob_gen_cmd": "''"},
    "module_mha_varlen_fwd": {"srcs": [CSRC + "py_itfs_ck/mha_varlen.cu'"], "extra_include": [], "blob_gen_cmd": "''"},
    "module_untouched": {"srcs": [CSRC + "kernels/other.cu'"], "extra_include": [], "blob_gen_cmd": "''"},
}
FILES = {
    "csrc/pybind/activation_pybind.cu": '#include "rocm_ops.hpp"\n',
    "csrc/kernels/activation_kernels.cu": '#include "aiter_hip_common.h"\n#include <opus/opus.hpp>\n',
    "csrc/include/rocm_ops.hpp": "// ops\n",
    "csrc/include/aiter_hip_common.h": '#include "aiter_stream.h"\n',
    "csrc/include/aiter_stream.h": "// stream\n",
    "csrc/include/opus/opus.hpp": "// opus\n",
    "csrc/ck_gemm_a8w8/gemm_a8w8.cu": '#include "gemm_common.cuh"\n#include "local_helper.h"\n',
    "csrc/ck_gemm_a8w8/local_helper.h": "// beside\n",
    "csrc/ck_gemm_a8w8/include/gemm_common.cuh": "// common\n",
    "csrc/ck_gemm_a8w8/gen_instances.py": "# generator\n",
    "csrc/ck_gemm_a8w8/template.jinja": "// template\n",
    "csrc/kernels/topk_gating/a.cu": '#include "topk_common.h"\n',
    "csrc/kernels/topk_gating/topk_common.h": "// dir source\n",
    "csrc/py_itfs_cu/asm_pa.cu": "// pa\n",
    "csrc/py_itfs_cu/asm_mla.cu": "// mla\n",
    "csrc/ck_moe/moe.cu": "// moe\n",
    "csrc/py_itfs_ck/mha_varlen.cu": "// mha\n",
    "csrc/kernels/other.cu": '#include "gemm_common.cuh"\n',
    "csrc/other_dir/gemm_common.cuh": "// a header of the same name nobody includes\n",
    "hsa/codegen.py": "# codegen\n",
    "hsa/gfx942/pa/pa.csv": "a,b\n",
    "hsa/gfx942/mla/mla.csv": "a,b\n",
}
PREBUILT = {
    "module_activation",
    "module_gemm_a8w8",
    "module_moe_topk",
    "module_pa",
    "module_mla",
    "module_untouched",
    "module_moe_ck2stages_b16_silu_Stage1",
    "mha_varlen_fwd_bf16_nlogits_nbias",
}
ORPHAN = "orphan_module"


@pytest.fixture()
def tree(tmp_path: Path) -> SourceTree:
    return SourceTree(write_tree(tmp_path / "meta", FILES))


def touched(tree: SourceTree, patched, *, overlay=None, after=None, before=None, prebuilt=PREBUILT) -> set[str]:
    tree = SourceTree(tree.root, overlay or {})
    result = touched_modules(before or CONFIG, after or CONFIG, tree, patched, prebuilt)
    assert result.rebuilt == result.names & set(prebuilt)
    return set(result.names)


def test_a_patched_source_touches_the_module_that_builds_it_and_nothing_else(tree):
    assert touched(tree, ["csrc/kernels/activation_kernels.cu"]) == {"module_activation"}


def test_a_patched_header_touches_the_modules_whose_includes_reach_it_through_other_headers(tree):
    assert touched(tree, ["csrc/include/aiter_stream.h"]) == {"module_activation"}


def test_an_angle_bracket_include_and_one_beside_the_including_file_are_followed(tree):
    assert touched(tree, ["csrc/include/opus/opus.hpp"]) == {"module_activation"}
    assert touched(tree, ["csrc/ck_gemm_a8w8/local_helper.h"]) == {"module_gemm_a8w8"}


def test_a_header_is_found_in_the_modules_own_extra_include_not_in_a_directory_that_merely_has_that_name(tree):
    assert touched(tree, ["csrc/ck_gemm_a8w8/include/gemm_common.cuh"]) == {"module_gemm_a8w8"}
    assert touched(tree, ["csrc/other_dir/gemm_common.cuh"]) == set()


def test_a_header_the_patch_adds_is_followed_and_one_it_deletes_still_touches(tree):
    added = {"csrc/include/new_helper.h": b"// new\n", "csrc/include/rocm_ops.hpp": b'#include "new_helper.h"\n'}
    assert touched(tree, ["csrc/include/new_helper.h", "csrc/include/rocm_ops.hpp"], overlay=added) == {
        "module_activation"
    }
    assert touched(tree, ["csrc/include/aiter_stream.h"], overlay={"csrc/include/aiter_stream.h": None}) == {
        "module_activation"
    }


def test_a_file_below_a_source_directory_and_a_file_a_source_there_includes_touch_the_module(tree):
    assert touched(tree, ["csrc/kernels/topk_gating/new.cu"]) == {"module_moe_topk"}
    assert touched(tree, ["csrc/kernels/topk_gating/topk_common.h"]) == {"module_moe_topk"}


def test_the_generator_script_and_the_files_beside_it_touch_the_module(tree):
    assert touched(tree, ["csrc/ck_gemm_a8w8/gen_instances.py"]) == {"module_gemm_a8w8"}
    assert touched(tree, ["csrc/ck_gemm_a8w8/template.jinja"]) == {"module_gemm_a8w8"}


@pytest.mark.parametrize(
    ("path", "modules"),
    [
        ("hsa/gfx942/pa/pa.csv", {"module_pa"}),
        ("hsa/gfx942/mla/mla.csv", {"module_mla"}),
        ("hsa/codegen.py", {"module_pa", "module_mla"}),
        ("hsa/gfx942/unlisted/x.csv", {"module_pa", "module_mla"}),
    ],
)
def test_an_hsa_table_touches_the_modules_its_directory_names_and_an_unknown_one_all_generated_from_hsa(
    tree, path, modules
):
    assert touched(tree, [path]) == modules


def test_a_module_whose_config_entry_changed_or_is_new_is_touched_with_no_patched_path(tree):
    after = {**CONFIG, "module_untouched": {**CONFIG["module_untouched"], "flags_extra_hip": ["'-O1'"]}}
    after["module_new"] = CONFIG["module_untouched"]

    assert touched(tree, [], after=after) == {"module_untouched", "module_new"}


def test_a_module_whose_entry_cannot_be_expanded_is_touched(tree):
    odd = {"srcs": ["os.path.join(ROOT, 'x.cu')"], "extra_include": [], "blob_gen_cmd": "''"}
    unknown = {"srcs": ["f'{SOMETHING}/x.cu'"], "extra_include": [], "blob_gen_cmd": "''"}
    after = {**CONFIG, "module_odd": odd, "module_unknown": unknown}

    assert touched(tree, ["csrc/kernels/activation_kernels.cu"], before=after, after=after) == {
        "module_activation",
        "module_odd",
        "module_unknown",
    }


def test_an_entry_that_is_not_a_mapping_is_touched_and_does_not_break_the_reading_of_the_others(tree):
    after = {**CONFIG, "flags": ["-O0"]}

    assert touched(tree, ["csrc/kernels/activation_kernels.cu"], before=after, after=after) == {
        "module_activation",
        "flags",
    }


def test_a_name_aiter_makes_at_run_time_follows_the_entry_it_starts_with(tree):
    assert touched(tree, ["csrc/ck_moe/moe.cu"]) == {"module_moe_ck2stages", "module_moe_ck2stages_b16_silu_Stage1"}
    assert touched(tree, ["csrc/py_itfs_ck/mha_varlen.cu"]) == {
        "module_mha_varlen_fwd",
        "mha_varlen_fwd_bf16_nlogits_nbias",
    }


def test_a_prebuilt_name_that_follows_no_entry_is_touched_whenever_anything_is(tree):
    with_orphan = PREBUILT | {ORPHAN}

    assert ORPHAN in touched(tree, ["csrc/kernels/activation_kernels.cu"], prebuilt=with_orphan)
    assert touched(tree, [], prebuilt=with_orphan) == set()


def test_an_entry_generated_from_a_table_that_differs_is_touched_with_no_patched_path_and_so_are_its_runtime_names(
    tree,
):
    found = touched_modules(CONFIG, CONFIG, tree, [], PREBUILT | {ORPHAN}) | touched_entries(
        {"module_gemm_a8w8", "module_moe_ck2stages"}, CONFIG, PREBUILT | {ORPHAN}
    )

    assert found.names == {
        "module_gemm_a8w8",
        "module_moe_ck2stages",
        "module_moe_ck2stages_b16_silu_Stage1",
        ORPHAN,
    }, "a prebuilt name that follows no entry is touched whenever anything is"
    assert found.rebuilt == found.names & (PREBUILT | {ORPHAN})


def test_the_prebuilt_modules_are_the_so_files_of_the_jit_directory(tmp_path):
    package = write_tree(
        tmp_path / "aiter", {"jit/module_a.so": "", "jit/module_b.so": "", "jit/core.py": "", "jit/build/x.so": ""}
    )

    assert prebuilt_names(package) == {"module_a", "module_b"}
    assert prebuilt_names(tmp_path / "missing") == frozenset()
    assert config_path(package) == package / "jit" / "optCompilerConfig.json"


def test_the_config_shape_the_expansion_reads_is_the_shape_the_json_has(tmp_path):
    document = json.loads(json.dumps(CONFIG))

    assert document["module_gemm_a8w8"]["blob_gen_cmd"].startswith("f'{AITER_CSRC_DIR}/")
