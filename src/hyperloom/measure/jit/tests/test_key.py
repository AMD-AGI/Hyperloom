# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the key of a compiled-code cache is made of: every input that decides what the compilers produce, and nothing else.
The tests separate the two directions: a changed input must change the key (or a test would be given code built from
other inputs), and an input that cannot change the code must leave it alone (or the cache would never be shared)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.measure.jit import key as key_module
from hyperloom.measure.jit.key import (
    BuildSurface,
    Machine,
    aiter_key,
    build_surface,
    entry_env_names,
    env_names_in,
    scan_env_names,
    touches_build_surface,
    tree_identity,
    triton_key,
    without_card_arch,
)

MACHINE = Machine("sha256:" + "1" * 64, "gfx950", "MI355X", "SPX/NPS1")
TREE = tree_identity([], None, [])
SURFACE = BuildSurface({"AITER_CONFIG_GEMM_A8W8": "sha256:aa"}, {"ENABLE_CK": "1"})


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def checkout(tmp_path) -> Path:
    """A miniature aiter checkout: the package, the code generators beside it, and a call-time op that reads a switch."""
    package = tmp_path / "aiter"
    _write(
        package,
        "jit/core.py",
        'import os\nA = os.getenv("AITER_CONFIG_GEMM_A8W8", "x")\nB = os.environ.get("ENABLE_CK", "1")\n',
    )
    _write(
        package, "jit/utils/chip_info.py", 'import os\nif "GPU_ARCHS" in os.environ:\n    C = os.environ["CU_NUM"]\n'
    )
    _write(tmp_path, "csrc/ck/gen_instances.py", "import os\nD = os.environ.get('ENABLE_ROPE_POSITIONS_INT32')\n")
    _write(tmp_path, "hsa/codegen.py", 'import os\nE = os.environ["AITER_GPU_ARCHS"]\n')
    _write(package, "ops/mha.py", 'import os\nF = os.environ.get("AITER_DISABLE_FMHA_OPUS", "0")\n')
    return package


def test_the_variables_the_build_reads_are_found_in_the_source_that_builds_and_generates_code(checkout) -> None:
    assert scan_env_names([checkout]) == {
        "AITER_CONFIG_GEMM_A8W8",
        "ENABLE_CK",
        "GPU_ARCHS",
        "CU_NUM",
        "ENABLE_ROPE_POSITIONS_INT32",
        "AITER_GPU_ARCHS",
    }, "the three ways of reading, the generators and codegen.py: and nothing a call-time op reads"


def test_a_switch_read_only_by_a_call_time_op_is_not_a_build_input(checkout) -> None:
    """The op picks which module is called; the module is named for what it was asked."""
    assert "AITER_DISABLE_FMHA_OPUS" not in scan_env_names([checkout])


def test_a_variable_a_patch_adds_to_the_build_path_is_found_in_the_patched_tree_too(checkout, tmp_path) -> None:
    patched = tmp_path / "farm" / "aiter"
    _write(patched, "jit/core.py", 'import os\nX = os.getenv("AITER_NEW_BUILD_SWITCH")\n')

    assert "AITER_NEW_BUILD_SWITCH" in scan_env_names([patched, checkout])


def test_a_variable_a_patch_adds_to_csrc_or_hsa_is_found_in_the_farms_meta_directory_and_nowhere_else(
    checkout, tmp_path
) -> None:
    """A farm with no rebuilt package keeps its meta directory elsewhere: it is read where it is named."""
    package = tmp_path / "farm" / "aiter"
    meta = tmp_path / "meta_copy"
    _write(package, "jit/core.py", "")
    _write(meta, "csrc/cpp_itfs/pa/pa.py", 'import os\nX = os.getenv("AITER_PATCHED_CSRC_SWITCH")\n')
    _write(meta, "hsa/codegen.py", 'import os\nY = os.environ.get("AITER_PATCHED_HSA_SWITCH")\n')
    _write(meta, "op_tests/test_x.py", 'import os\nZ = os.getenv("AITER_NOT_A_BUILD_INPUT")\n')

    assert scan_env_names([package, checkout], [meta]) >= {"AITER_PATCHED_CSRC_SWITCH", "AITER_PATCHED_HSA_SWITCH"}
    assert "AITER_PATCHED_CSRC_SWITCH" not in scan_env_names([package, checkout])
    assert "AITER_NOT_A_BUILD_INPUT" not in scan_env_names([package], [meta])


def test_a_link_back_above_itself_is_read_once_and_ends_the_walk(tmp_path, monkeypatch) -> None:
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "sub" / "a.py").write_text('import os\nos.getenv("A_VAR")\n')
    (tmp_path / "pkg" / "sub" / "loop").symlink_to(tmp_path / "pkg", target_is_directory=True)
    read: list[str] = []
    real = key_module._names_in
    monkeypatch.setattr(key_module, "_names_in", lambda path: read.append(path.name) or real(path))

    assert env_names_in([tmp_path / "pkg"]) == {"A_VAR"}
    assert read == ["a.py"], "each directory once, however many links lead to it"


def test_no_build_path_is_no_names(tmp_path) -> None:
    assert scan_env_names([tmp_path / "missing"]) == frozenset()


def test_the_surface_is_the_content_of_tables_and_the_value_of_the_variables_the_build_reads() -> None:
    names = {"AITER_CONFIG_GEMM_A8W8", "ENABLE_CK", "GPU_ARCHS", "AITER_REBUILD"}

    surface = build_surface(names, {"ENABLE_CK": "0", "SGLANG_X": "1"}, {"AITER_CONFIG_GEMM_A8W8": "sha256:aa"})

    assert surface.tables == {"AITER_CONFIG_GEMM_A8W8": "sha256:aa"}
    assert surface.environment == {"ENABLE_CK": "0"}, (
        "a variable that is unset is not an input, and SGLANG_X is not read"
    )
    assert not surface.rebuild


@pytest.mark.parametrize(
    "name",
    [
        "AITER_JIT_DIR",
        "AITER_ROOT_DIR",
        "TRITON_CACHE_DIR",
        "HOME",
        "USERPROFILE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_RUNTIME_DIR",
        "MAX_JOBS",
        "PREBUILD_THREAD_NUM",
        "AITER_JIT_VERBOSE",
    ],
)
def test_a_variable_that_names_a_location_or_changes_only_speed_or_noise_is_not_an_input(name) -> None:
    surface = build_surface({name}, {name: "whatever"}, {})

    assert surface == BuildSurface()


def test_two_sandboxes_that_differ_only_in_where_they_keep_things_share_a_key_and_a_build_input_still_splits_it(
    checkout, tmp_path
) -> None:
    """Every local sandbox has a home and a scratch directory of its own, and aiter reads ``HOME`` (``cpp_itfs/utils.py``):
    the scan finds it, and the key must still be the same."""
    _write(
        tmp_path, "csrc/cpp_itfs/utils.py", 'import os\nH = os.environ.get("HOME") or os.environ.get("USERPROFILE")\n'
    )
    _write(tmp_path, "csrc/ck/tmp.py", 'import os\nT = os.environ.get("TMPDIR")\nC = os.environ["XDG_CACHE_HOME"]\n')
    names = scan_env_names([checkout])
    assert {"HOME", "USERPROFILE", "TMPDIR", "XDG_CACHE_HOME", "ENABLE_CK"} <= names, "the discovery keeps them"

    def key_of(environment: dict[str, str]) -> str:
        return aiter_key(MACHINE, TREE, build_surface(names, environment, {}))

    one = {"HOME": "/work/a/home", "TMPDIR": "/work/a/tmp", "XDG_CACHE_HOME": "/work/a/cache", "ENABLE_CK": "1"}
    two = {"HOME": "/work/b/home", "TMPDIR": "/work/b/tmp", "XDG_CACHE_HOME": "/work/b/cache", "ENABLE_CK": "1"}
    assert key_of(one) == key_of(two)
    assert key_of(one) != key_of({**one, "ENABLE_CK": "0"}), (
        "a variable that selects what is built still splits the key"
    )


@pytest.mark.parametrize(("value", "rebuilds"), [("1", True), ("2", True), ("0", False), ("", False), ("junk", False)])
def test_aiter_rebuild_marks_a_test_that_discards_what_the_cache_gives(value, rebuilds) -> None:
    assert build_surface({"AITER_REBUILD"}, {"AITER_REBUILD": value}, {}).rebuild is rebuilds


def test_the_key_changes_with_every_input_the_build_context_has() -> None:
    base = aiter_key(MACHINE, TREE, SURFACE)
    other_machine = [
        Machine("sha256:" + "2" * 64, "gfx950", "MI355X", "SPX/NPS1"),
        Machine(MACHINE.image_digest, "gfx942", "MI355X", "SPX/NPS1"),
        Machine(MACHINE.image_digest, "gfx950", "MI300X", "SPX/NPS1"),
        Machine(MACHINE.image_digest, "gfx950", "MI355X", "CPX/NPS4"),
    ]

    keys = {aiter_key(m, TREE, SURFACE) for m in other_machine}
    keys.add(aiter_key(MACHINE, tree_identity([("jit/core.py", "upsert", "sha256:x", "0644")], None, []), SURFACE))
    keys.add(aiter_key(MACHINE, tree_identity([], "9" * 64, []), SURFACE))
    keys.add(aiter_key(MACHINE, tree_identity([], None, [("sglang", "a.py", "sha256:x")]), SURFACE))
    keys.add(aiter_key(MACHINE, TREE, BuildSurface({"AITER_CONFIG_GEMM_A8W8": "sha256:bb"}, {"ENABLE_CK": "1"})))
    keys.add(aiter_key(MACHINE, TREE, BuildSurface({"AITER_CONFIG_GEMM_A8W8": "sha256:aa"}, {"ENABLE_CK": "0"})))

    assert base not in keys and len(keys) == 9, "nine different inputs, nine different keys, none the base"


def test_the_same_inputs_give_the_same_key_whatever_order_they_were_found_in() -> None:
    a = tree_identity([("b", "upsert", "x", "0644"), ("a", "delete", "", "")], None, [("p", "1", "x"), ("p", "0", "y")])
    b = tree_identity([("a", "delete", "", ""), ("b", "upsert", "x", "0644")], None, [("p", "0", "y"), ("p", "1", "x")])

    assert a == b
    first = BuildSurface({"X": "1", "Y": "2"}, {"A": "1", "B": "2"})
    second = BuildSurface({"Y": "2", "X": "1"}, {"B": "2", "A": "1"})
    assert aiter_key(MACHINE, a, first) == aiter_key(MACHINE, b, second)


def test_the_triton_key_is_the_image_and_the_architecture_and_nothing_else() -> None:
    assert triton_key(MACHINE) == triton_key(Machine(MACHINE.image_digest, "gfx950", "other board", None))
    assert triton_key(MACHINE) != triton_key(Machine("sha256:" + "3" * 64, "gfx950", "MI355X", "SPX/NPS1"))
    assert triton_key(MACHINE) != triton_key(Machine(MACHINE.image_digest, "gfx942", "MI355X", "SPX/NPS1"))
    assert triton_key(MACHINE) != aiter_key(MACHINE, TREE, BuildSurface()), "the kinds never share a key"


@pytest.mark.parametrize(
    "text",
    [
        "import os\nos.environ['ENABLE_CK'] = '0'\n",
        "x = os.getenv('GPU_ARCHS')\n",
        "os.environ['AITER_ANYTHING'] = '1'\n",
        "import aiter.jit.core as core\ncore.AITER_REBUILD = 1\n",
        "from aiter.jit.core import build_module\n",
        "from aiter.jit.core import compile_ops\n",
        "os.putenv('X', '1')\n",
        "os.unsetenv('X')\n",
        "os.environ.update({'SGLANG_X': '1'})\n",
        "os.environ.setdefault('SGLANG_X', '1')\n",
        "os.environ.pop('SGLANG_X')\n",
        "os.environ[name] = value\n",
        "os.environ[f'SGLANG_{i}'] = value\n",
    ],
)
def test_a_framework_patch_that_can_change_the_build_context_from_inside_the_server_is_part_of_the_key(text) -> None:
    assert touches_build_surface(text, {"ENABLE_CK", "GPU_ARCHS"})


@pytest.mark.parametrize(
    "text",
    [
        "def schedule(batch):\n    return sorted(batch)\n",
        "x = os.environ['SGLANG_MAX_BATCH']\n",
        "x = os.environ.get('SGLANG_MAX_BATCH', '8')\n",
        "os.environ['SGLANG_MAX_BATCH'] = '8'\n",
        "if os.environ[name] == 'x':\n    pass\n",
        "value = os.environ[name]\n",
        "attention_backend = 'aiter' if use_aiter else 'triton'\n",
    ],
)
def test_a_patch_that_only_changes_what_the_framework_asks_for_is_not(text) -> None:
    assert not touches_build_surface(text, {"ENABLE_CK", "GPU_ARCHS"})


# Five entries of aiter/jit/optCompilerConfig.json, copied verbatim from ROCm/aiter e7d2453f25e5aaeea72a3a40b284ba453e026624,
# the aiter of the SGLang ROCm image these tests model. On the AITER_COMMIT that image states (285579b62) these five
# entries read the same variables. aiter's get_args_of_build evaluates every string of an entry as Python when it builds
# the module, so these read the server's environment then.
IMAGE_AITER_CONFIG = Path(__file__).parent / "data" / "aiter-e7d2453f-optCompilerConfig.json"
#: What each entry reads on that commit. ``module_deepgemm_opus`` asks ``'gfx1250' in os.environ.get('GPU_ARCHS', '')``:
#: ``GPU_ARCHS`` is read, ``gfx1250`` is a value. (On aiter main the CK-tile GEMM also reads
#: ``CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT``; the image's does not.)
IMAGE_ENTRY_READS = {
    "module_gemm_a8w8_bpreshuffle": set(),
    "module_deepgemm_opus": {"GPU_ARCHS"},
    "module_gemm_a8w8_bpreshuffle_cktile": {"FLATMM_HIP_CLANG_PATH"},
    "module_rmsnorm_quant": {"OPUS_FP32_to_BF16_DEFAULT"},
    "module_mha_batch_prefill": {
        "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT",
        "CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT",
        "CK_TILE_ATTENTION_USE_SOFTSIGN_ASM",
    },
}


@pytest.fixture
def image_aiter(tmp_path) -> Path:
    """The image's aiter package: a builder that reads one variable of its own, and the module config of the image."""
    package = tmp_path / "aiter"
    _write(package, "jit/core.py", 'import os\nB = os.environ.get("ENABLE_CK", "1")\n')
    _write(package, "jit/optCompilerConfig.json", IMAGE_AITER_CONFIG.read_text(encoding="utf-8"))
    return package


def test_each_entry_of_the_images_module_config_reads_the_variables_its_expressions_name() -> None:
    config = json.loads(IMAGE_AITER_CONFIG.read_text(encoding="utf-8"))

    assert {name: set(entry_env_names(entry)) for name, entry in config.items()} == IMAGE_ENTRY_READS


def test_the_scan_finds_the_variables_only_the_module_config_reads(image_aiter) -> None:
    """None of them is in the Python of the build path, so a scan of ``.py`` files alone misses every one."""
    from_config = set().union(*IMAGE_ENTRY_READS.values())

    assert scan_env_names([image_aiter]) == {"ENABLE_CK", *from_config}
    assert env_names_in([image_aiter / "jit"]) == {"ENABLE_CK"}


@pytest.mark.parametrize(
    "variable",
    ["CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT", "FLATMM_HIP_CLANG_PATH", "OPUS_FP32_to_BF16_DEFAULT", "GPU_ARCHS"],
)
def test_a_candidate_that_only_sets_a_variable_the_module_config_reads_gets_a_key_of_its_own(
    image_aiter, variable
) -> None:
    """It would otherwise read, and add to, modules built under the image's value: a CK-tile attention module built with
    another bf16 rounding, or by another compiler, measured as if it were its own."""
    names = scan_env_names([image_aiter])

    def key_of(environment: dict[str, str]) -> str:
        return aiter_key(MACHINE, TREE, build_surface(names, environment, {}))

    base = {"ENABLE_CK": "1"}
    assert key_of({**base, variable: "3"}) != key_of(base)
    assert key_of({**base, "CK_TILE_NOT_READ_BY_ANY_BUILD": "3", "SGLANG_X": "1"}) == key_of(base), (
        "a variable no code of the build and no entry names still leaves the key alone"
    )


def test_a_module_config_that_cannot_be_read_names_nothing_and_leaves_the_scan_of_the_python(tmp_path) -> None:
    package = tmp_path / "aiter"
    _write(package, "jit/core.py", 'import os\nB = os.environ.get("ENABLE_CK", "1")\n')
    _write(package, "jit/optCompilerConfig.json", "{ not json")

    assert scan_env_names([package]) == {"ENABLE_CK"}


def test_a_value_tested_against_a_variable_is_not_a_variable_and_does_not_hide_the_one_read(tmp_path) -> None:
    _write(
        tmp_path, "a.py", "import os\nX = 'gfx950' in os.environ.get('GPU_ARCHS', '')\nY = 'ENABLE_CK' in os.environ\n"
    )

    assert env_names_in([tmp_path / "a.py"]) == {"GPU_ARCHS", "ENABLE_CK"}


def test_an_aiter_key_is_not_one_made_before_the_module_configs_variables_were_in_it() -> None:
    """An entry stored under the earlier key may hold a module built under a variable that key did not name."""
    earlier = key_module._hash(
        {
            "schema": key_module.SCHEMA,
            "kind": "aiter",
            "machine": MACHINE.document(),
            "tree": TREE,
            "tables": dict(sorted(SURFACE.tables.items())),
            "environment": dict(sorted(SURFACE.environment.items())),
        }
    )

    assert aiter_key(MACHINE, TREE, SURFACE) != earlier
    assert triton_key(MACHINE) == key_module._hash(
        {"schema": 2, "kind": "triton", "image": MACHINE.image_digest, "gfx": MACHINE.gfx_arch}
    ), "Triton's key is unchanged: its cache never depended on aiter's variables"


@pytest.mark.parametrize(
    ("environment", "gfx_arch", "expected"),
    [
        ({"GPU_ARCHS": "native"}, "gfx950", {}),
        ({"GPU_ARCHS": "native"}, None, {}),
        ({"GPU_ARCHS": "gfx950"}, "gfx950", {}),
        ({"GPU_ARCHS": "gfx950"}, None, {"GPU_ARCHS": "gfx950"}),
        ({"GPU_ARCHS": "gfx942"}, "gfx950", {"GPU_ARCHS": "gfx942"}),
        ({"GPU_ARCHS": "gfx942;gfx950"}, "gfx950", {"GPU_ARCHS": "gfx942;gfx950"}),
        ({"ENABLE_CK": "1", "GPU_ARCHS": "native"}, "gfx950", {"ENABLE_CK": "1"}),
        ({"ENABLE_CK": "1"}, None, {"ENABLE_CK": "1"}),
        ({}, None, {}),
    ],
)
def test_gpu_archs_is_left_out_when_it_only_names_the_card_the_build_is_for(environment, gfx_arch, expected):
    assert without_card_arch(environment, gfx_arch) == expected


def test_the_environment_given_is_not_changed():
    environment = {"GPU_ARCHS": "native", "ENABLE_CK": "1"}

    assert without_card_arch(environment, "gfx950") == {"ENABLE_CK": "1"}
    assert environment == {"GPU_ARCHS": "native", "ENABLE_CK": "1"}
