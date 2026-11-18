# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session's benchmark runs share one JIT cache directory per framework cache env."""

from __future__ import annotations

from pathlib import Path

from hyperloom.inference_optimizer.session.session_paths import jit_cache_dir
from hyperloom.orchestrator.actions.executors._aiter_jit import apply_session_jit_cache


def test_sglang_caches_land_under_the_session(tmp_path):
    env: dict[str, str] = {}

    apply_session_jit_cache(env, framework="sglang", session_dir=tmp_path)

    root = jit_cache_dir(tmp_path)
    assert env == {
        "SGLANG_CACHE_DIR": str(root / "sglang_cache_dir"),
        "AMD_COMGR_CACHE_DIR": str(root / "amd_comgr_cache_dir"),
    }
    assert all(Path(value).is_dir() for value in env.values())


def test_two_runs_of_one_session_get_the_same_directories(tmp_path):
    first: dict[str, str] = {}
    second: dict[str, str] = {}

    apply_session_jit_cache(first, framework="sglang", session_dir=tmp_path)
    apply_session_jit_cache(second, framework="sglang", session_dir=tmp_path)

    assert first == second


def test_an_operator_set_location_is_left_alone(tmp_path):
    env = {"SGLANG_CACHE_DIR": "/mnt/fast/sglang"}

    apply_session_jit_cache(env, framework="sglang", session_dir=tmp_path)

    assert env["SGLANG_CACHE_DIR"] == "/mnt/fast/sglang"
    assert env["AMD_COMGR_CACHE_DIR"] == str(jit_cache_dir(tmp_path) / "amd_comgr_cache_dir")


def test_a_framework_without_an_established_cache_location_is_untouched(tmp_path):
    env: dict[str, str] = {}

    apply_session_jit_cache(env, framework="xdit", session_dir=tmp_path)

    assert env == {}
    assert not jit_cache_dir(tmp_path).exists()
