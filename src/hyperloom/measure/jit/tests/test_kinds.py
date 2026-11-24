# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which files of FlyDSL's cache directory are ever stored: the launches, nothing FlyDSL keeps beside them."""

from __future__ import annotations

import pytest

from hyperloom.measure.jit_kinds import FLYDSL, KINDS, kind_named

MANAGER, CALL = "0123456789abcdef0123456789abcdef", "fedcba9876543210"


@pytest.mark.parametrize(
    "relative",
    [
        f"launch_{MANAGER}/{CALL}.pkl",
        f"_moe_gemm1_k31_wide_{MANAGER}/{CALL}.pkl",
        f"gemm_a16w16_gfx950_256x128_{MANAGER}/{CALL}.pkl",
    ],
)
def test_a_launch_flydsl_wrote_is_admitted(relative) -> None:
    assert FLYDSL.admits(relative)


@pytest.mark.parametrize(
    "relative",
    [
        f"launch_{MANAGER}/{CALL}.lock",
        f"launch_{MANAGER}/.{CALL}.pkl.tmp",
        f"launch_{MANAGER}/{CALL}.pkl.tmp",
        f"launch_{MANAGER.upper()}/{CALL}.pkl",
        f"launch_{MANAGER[:-1]}/{CALL}.pkl",
        f"launch_{MANAGER}/{CALL}0.pkl",
        f"{MANAGER}/{CALL}.pkl",
        f"launch_{MANAGER}/sub/{CALL}.pkl",
        f"../launch_{MANAGER}/{CALL}.pkl",
        f"launch-x_{MANAGER}/{CALL}.pkl",
        f"{CALL}.pkl",
        f"launch_{MANAGER}/{CALL}.py",
    ],
)
def test_anything_else_in_flydsls_directory_is_refused(relative) -> None:
    assert not FLYDSL.admits(relative)


def test_the_kind_is_known_by_name_and_redirects_flydsls_variable() -> None:
    assert FLYDSL in KINDS and kind_named("flydsl") is FLYDSL
    assert (FLYDSL.env_var, FLYDSL.content, FLYDSL.located) == ("FLYDSL_RUNTIME_CACHE_DIR", "", None)
