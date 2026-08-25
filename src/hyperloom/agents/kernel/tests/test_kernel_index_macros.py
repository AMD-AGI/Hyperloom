###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for attribute-aware ``__global__`` kernel definition scanning.

Pure text / no GPU / no docker. These cover the tokeniser that recovers kernel
names when an attribute carrying parentheses (notably
``__launch_bounds__(NUM_THREADS)``, present on ~40% of aiter kernels, and
``__attribute__((...))``) sits between ``__global__`` and the name. The prior
regex captured the *attribute* as the kernel name; these tests lock the fix in.

The environment has no pytest, so every check is also runnable via a plain
``python3 path/to/test.py`` ``__main__`` block that prints PASS/FAIL.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hyperloom.agents.kernel.tools import kernel_source_index as ksi


def _names(text: str) -> list[str]:
    return [name for name, _npos in ksi._iter_global_defs(text)]


# --- plain definitions ------------------------------------------------------
def test_plain_global_def() -> None:
    text = "__global__ void act_and_mul_kernel(float* x, int n) {}"
    assert _names(text) == ["act_and_mul_kernel"]


def test_return_type_on_separate_line() -> None:
    text = "__global__\n    void\n    reshape_and_cache_kernel(int* p) {\n}"
    assert _names(text) == ["reshape_and_cache_kernel"]


# --- launch_bounds attribute (the core regression) -------------------------
def test_launch_bounds_is_skipped_not_captured() -> None:
    text = "__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(\n    float* out) {}"
    got = _names(text)
    assert got == ["paged_attention_ll4mi_reduce_kernel"], got
    assert "__launch_bounds__" not in got


def test_launch_bounds_multi_arg() -> None:
    text = "__global__ __launch_bounds__(256, 4) void my_kernel(int* p) {}"
    assert _names(text) == ["my_kernel"]


def test_template_kernel_with_launch_bounds() -> None:
    text = (
        "template <typename scalar_t, int BLOCK_SIZE, int NUM_THREADS>\n"
        "__global__ __launch_bounds__(NUM_THREADS) void "
        "paged_attention_ll4mi_QKV_mfma16_kernel(\n"
        "    const scalar_t* __restrict__ q) {}"
    )
    assert _names(text) == ["paged_attention_ll4mi_QKV_mfma16_kernel"]


# --- __attribute__((...)) with nested parens -------------------------------
def test_attribute_nested_parens_is_skipped() -> None:
    text = "__global__ __attribute__((amdgpu_flat_work_group_size(256, 256))) void fused_kernel(int* p) {}"
    assert _names(text) == ["fused_kernel"]


# --- namespaced / qualified heads ------------------------------------------
def test_extern_c_and_static_qualifiers() -> None:
    text = 'extern "C" __global__ static void wrapped_kernel(int* p) {}'
    assert _names(text) == ["wrapped_kernel"]


# --- multiple defs + line accuracy -----------------------------------------
def test_multiple_defs_and_line_numbers() -> None:
    text = (
        "// header\n"
        "__global__ void first_kernel(int* a) {}\n"
        "\n"
        "__global__ __launch_bounds__(128) void second_kernel(int* b) {}\n"
    )
    defs = list(ksi._iter_global_defs(text))
    names = [d[0] for d in defs]
    assert names == ["first_kernel", "second_kernel"], names
    # name_pos -> 1-based line numbers
    lines = [text.count("\n", 0, npos) + 1 for _n, npos in defs]
    assert lines == [2, 4], lines


# --- _scan_file wiring ------------------------------------------------------
def test_scan_file_records_real_names() -> None:
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "pa.cuh"
        f.write_text(
            "template <int NUM_THREADS>\n"
            "__global__ __launch_bounds__(NUM_THREADS) void "
            "paged_attention_ll4mi_reduce_kernel(float* o) {}\n",
            encoding="utf-8",
        )
        recs = ksi._scan_file(f)
        assert recs == [("paged_attention_ll4mi_reduce_kernel", 2)], recs


# --- definitions only (declarations / comments / strings rejected) ----------
def test_forward_declaration_is_not_indexed() -> None:
    """A ``__global__`` forward declaration (``);``) has no body and is skipped."""
    text = "__global__ void paged_attn_kernel(const float*, float*);\n"
    assert _names(text) == []


def test_commented_and_stringified_defs_are_ignored() -> None:
    """A ``__global__`` inside a comment or string literal is not a definition."""
    text = '// __global__ void dead_kernel(int*);\nconst char* s = "__global__ void str_kernel(int*)";\n'
    assert _names(text) == []


def test_declaration_then_definition_indexes_only_the_definition() -> None:
    text = (
        "__global__ void mm_kernel(const float*, float*);\n"
        "__global__ void mm_kernel(const float* a, float* o) { o[0] = a[0]; }\n"
    )
    defs = list(ksi._iter_global_defs(text))
    names = [d[0] for d in defs]
    assert names == ["mm_kernel"], names
    lines = [text.count("\n", 0, npos) + 1 for _n, npos in defs]
    assert lines == [2], lines  # the definition line, not the declaration


# --- no false positives -----------------------------------------------------
def test_no_global_no_defs() -> None:
    text = "void not_a_kernel(int* p) { launch_bounds(3); }"
    assert _names(text) == []


_TESTS = [
    test_plain_global_def,
    test_return_type_on_separate_line,
    test_launch_bounds_is_skipped_not_captured,
    test_launch_bounds_multi_arg,
    test_template_kernel_with_launch_bounds,
    test_attribute_nested_parens_is_skipped,
    test_extern_c_and_static_qualifiers,
    test_multiple_defs_and_line_numbers,
    test_scan_file_records_real_names,
    test_forward_declaration_is_not_indexed,
    test_commented_and_stringified_defs_are_ignored,
    test_declaration_then_definition_indexes_only_the_definition,
    test_no_global_no_defs,
]


def _run_all() -> int:
    """Run every check, print PASS/FAIL per test, return a process exit code."""
    failures = 0
    for test in _TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report any unexpected error.
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    total = len(_TESTS)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
