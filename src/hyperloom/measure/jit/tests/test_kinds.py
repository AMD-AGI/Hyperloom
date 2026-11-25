# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which files of FlyDSL's cache directory are ever stored: the launches, in the layout of FlyDSL 0.4.x or 0.5.x, and
nothing FlyDSL keeps beside them."""

from __future__ import annotations

import pytest

from hyperloom.measure.jit_kinds import FLYDSL, KINDS, kind_named

#: FlyDSL 0.4.x (v0.4.2): ``<function>_<32 hex of the manager key>/<16 hex of the call key>.pkl``.
MANAGER, CALL = "0123456789abcdef0123456789abcdef", "fedcba9876543210"
#: FlyDSL 0.5.x (v0.5.0 to v0.5.4): ``<function>/<sha256 of FlyDSL's key and the artifact key>.pkl``.
DISK = "0123456789abcdef" * 4


@pytest.mark.parametrize(
    "relative",
    [
        f"launch_{MANAGER}/{CALL}.pkl",
        f"_moe_gemm1_k31_wide_{MANAGER}/{CALL}.pkl",
        f"gemm_a16w16_gfx950_256x128_{MANAGER}/{CALL}.pkl",
        f"launch/{DISK}.pkl",
        f"_moe_gemm1_k31_wide/{DISK}.pkl",
        f"gemm_a16w16_gfx950_256x128/{DISK}.pkl",
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
        f"launch/{DISK}.lock",
        f"launch/.{DISK}.pkl.k3x9q_2a.tmp",
        f"launch/{DISK.upper()}.pkl",
        f"launch/{DISK[:-1]}.pkl",
        f"launch/{DISK}0.pkl",
        f"launch/sub/{DISK}.pkl",
        f"{DISK}.pkl",
        f"../launch/{DISK}.pkl",
        f"launch-x/{DISK}.pkl",
        f"launch/{CALL}.pkl",
        f"launch_{MANAGER}/{DISK}0.pkl",
    ],
)
def test_anything_else_in_flydsls_directory_is_refused(relative) -> None:
    assert not FLYDSL.admits(relative)


def test_the_kind_is_known_by_name_and_redirects_flydsls_variable() -> None:
    assert FLYDSL in KINDS and kind_named("flydsl") is FLYDSL
    assert (FLYDSL.env_var, FLYDSL.content, FLYDSL.located) == ("FLYDSL_RUNTIME_CACHE_DIR", "", None)
    assert FLYDSL.nested_prebuilt and not any(kind.nested_prebuilt for kind in KINDS if kind is not FLYDSL)
