# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the key of a compiled-code cache is made of: every input that decides what the compilers produce, and nothing else
(4.21). The tests separate the two directions: a changed input must change the key (or a test would be given code built from
other inputs), and an input that cannot change the code must leave it alone (or the cache would never be shared)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.measure.jit.key import (
    BuildSurface,
    Machine,
    aiter_key,
    build_surface,
    scan_env_names,
    touches_build_surface,
    tree_identity,
    triton_key,
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
    """The op picks which module is called; the module is named for what it was asked (design 4.21, the argument)."""
    assert "AITER_DISABLE_FMHA_OPUS" not in scan_env_names([checkout])


def test_a_variable_a_patch_adds_to_the_build_path_is_found_in_the_patched_tree_too(checkout, tmp_path) -> None:
    patched = tmp_path / "farm" / "aiter"
    _write(patched, "jit/core.py", 'import os\nX = os.getenv("AITER_NEW_BUILD_SWITCH")\n')

    assert "AITER_NEW_BUILD_SWITCH" in scan_env_names([patched, checkout])


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
    "name", ["AITER_JIT_DIR", "AITER_ROOT_DIR", "MAX_JOBS", "PREBUILD_THREAD_NUM", "AITER_JIT_VERBOSE"]
)
def test_a_variable_that_moves_a_cache_or_changes_only_speed_or_noise_is_not_an_input(name) -> None:
    surface = build_surface({name}, {name: "whatever"}, {})

    assert surface == BuildSurface()


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
