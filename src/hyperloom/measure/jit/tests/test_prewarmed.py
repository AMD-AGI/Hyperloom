# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A module hyperloom's prewarm built into the image is given only to a server that would have built the same one: for
the card the entry names, under the build variables it records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.measure.jit.prewarmed import PREWARM_MANIFEST, PREWARM_SCHEMA, build_record, unlike_prewarmed

GFX, CU = "gfx950", 256
MODULE = "module_gemm_a8w8_bpreshuffle_cktile"
#: What the module's build reads, as aiter e7d2453f's entry and ``jit`` code do (a subset), with a location and a table.
READS = (
    "FLATMM_HIP_CLANG_PATH",
    "ENABLE_CK",
    "CU_NUM",
    "GPU_ARCHS",
    "AITER_META_DIR",
    "HOME",
    "AITER_CONFIG_GEMM_A8W8",
)
#: The environment the prewarm built under: the image's variables and the card's CU count.
BUILT_UNDER = {"ENABLE_CK": "1", "CU_NUM": str(CU), "HOME": "/root", "ROCM_PATH": "/opt/rocm"}


def jit_with(tmp_path: Path, **entry: object) -> Path:
    jit = tmp_path / "jit"
    jit.mkdir()
    (jit / f"{MODULE}.so").write_bytes(b"\x7fELF prewarmed")
    document = {"schema": PREWARM_SCHEMA, "modules": [{"name": MODULE, **entry}]}
    (jit / PREWARM_MANIFEST).write_text(json.dumps(document), encoding="utf-8")
    return jit


@pytest.fixture
def jit(tmp_path: Path) -> Path:
    return jit_with(tmp_path, gfx=GFX, cu_num=CU, **build_record(READS, BUILT_UNDER))


def test_the_record_names_what_the_build_reads_and_the_values_less_the_target_and_places() -> None:
    assert build_record(READS, BUILT_UNDER) == {
        "build_variables": ["AITER_META_DIR", "CU_NUM", "ENABLE_CK", "FLATMM_HIP_CLANG_PATH", "GPU_ARCHS"],
        "build_environment": {"ENABLE_CK": "1"},
    }


@pytest.mark.parametrize(
    "server",
    [{}, {"ENABLE_CK": "1"}, {"GPU_ARCHS": "native"}, {"GPU_ARCHS": GFX}, {"CU_NUM": str(CU)}, {"HOME": "/x"}],
    ids=["nothing-set", "same-value", "native", "the-card", "the-cards-count", "a-location"],
)
def test_a_server_that_would_build_the_same_module_is_given_it(jit: Path, server: dict[str, str]) -> None:
    """``ENABLE_CK`` unset is aiter's default, which the stand-in records as ``1``: only the recorded value counts."""
    base = {"ENABLE_CK": "1", **server}
    assert unlike_prewarmed(jit, [base], GFX) == frozenset()


@pytest.mark.parametrize(
    "server",
    [
        {"CU_NUM": "304"},
        {"GPU_ARCHS": "gfx942"},
        {"GPU_ARCHS": f"gfx942;{GFX}"},
        {"FLATMM_HIP_CLANG_PATH": "/opt/llvm/bin"},
        {"ENABLE_CK": "0"},
        {"AITER_META_DIR": "/elsewhere"},
    ],
    ids=[
        "another-cu-count",
        "another-arch",
        "two-archs",
        "an-orchestrator-only-clang",
        "another-value",
        "another-meta",
    ],
)
def test_a_server_that_would_build_another_module_is_not_given_it(jit: Path, server: dict[str, str]) -> None:
    servers = [{"ENABLE_CK": "1"}, {"ENABLE_CK": "1", **server}]
    assert unlike_prewarmed(jit, servers, GFX) == {MODULE}


def test_a_value_the_build_had_and_the_server_has_not_differs(jit: Path) -> None:
    assert unlike_prewarmed(jit, [{}], GFX) == {MODULE}, (
        "ENABLE_CK=1 was the build's; a server without it reads default"
    )


@pytest.mark.parametrize(
    ("entry", "gfx_arch"),
    [
        ({"gfx": "gfx942", "cu_num": CU}, GFX),
        ({"gfx": GFX, "cu_num": CU}, None),
        ({"gfx": GFX, "cu_num": CU, "build_variables": list(READS)}, GFX),
    ],
    ids=["another-card", "card-unknown", "no-record-of-values"],
)
def test_a_module_whose_card_or_variables_cannot_be_matched_is_not_given(tmp_path, entry, gfx_arch) -> None:
    jit = jit_with(tmp_path, **entry)
    assert unlike_prewarmed(jit, [{"ENABLE_CK": "1"}], gfx_arch) == {MODULE}


def test_a_listed_module_the_directory_does_not_hold_is_nothing_to_leave_out(jit: Path) -> None:
    (jit / f"{MODULE}.so").unlink()
    assert unlike_prewarmed(jit, [{"CU_NUM": "304"}], GFX) == frozenset()
    assert unlike_prewarmed(jit.parent / "no-jit", [{}], GFX) == frozenset()
