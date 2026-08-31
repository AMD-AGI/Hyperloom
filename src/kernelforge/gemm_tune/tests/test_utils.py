# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for utils module."""

import json
import sys
import types

from kernelforge.gemm_tune.utils import (
    resolve_aiter_root,
    find_tuner_script,
    emit_result_json,
    RESULT_SENTINEL_BEGIN,
    RESULT_SENTINEL_END,
    TUNER_ENV_VARS,
)


class TestResolveAiterRoot:
    def test_returns_path(self):
        root = resolve_aiter_root()
        # In our test env, aiter should be available
        if root is not None:
            assert root.is_dir()
            assert (root / "csrc").is_dir() or (root / "aiter").is_dir()

    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AITER_ROOT_DIR", str(tmp_path))
        assert resolve_aiter_root() == tmp_path

    def test_resolves_aiter_meta_sibling_package(self, tmp_path, monkeypatch):
        dist_packages = tmp_path / "site-packages"
        aiter_package = dist_packages / "aiter"
        aiter_package.mkdir(parents=True)
        aiter_init = aiter_package / "__init__.py"
        aiter_init.write_text("", encoding="utf-8")
        aiter_meta = dist_packages / "aiter_meta"
        (aiter_meta / "csrc").mkdir(parents=True)
        monkeypatch.delenv("AITER_ROOT_DIR", raising=False)
        monkeypatch.setitem(
            sys.modules,
            "aiter",
            types.SimpleNamespace(__file__=str(aiter_init)),
        )

        assert resolve_aiter_root() == aiter_meta


class TestFindTunerScript:
    def test_known_tuner(self):
        script = find_tuner_script("fmoe_ck")
        if script is not None:
            assert script.is_file()
            assert "gemm_moe_tune.py" in script.name

    def test_unknown_tuner(self):
        assert find_tuner_script("nonexistent_tuner") is None


class TestEmitResultJson:
    def test_sentinel_wrapping(self, capsys):
        emit_result_json({"status": "ok", "value": 42})
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert lines[0] == RESULT_SENTINEL_BEGIN
        assert lines[-1] == RESULT_SENTINEL_END
        payload = json.loads("\n".join(lines[1:-1]))
        assert payload["status"] == "ok"
        assert payload["value"] == 42


class TestTunerEnvVars:
    def test_all_tuners_have_env_var(self):
        expected = [
            "fmoe_ck",
            "a8w8",
            "a8w8_blockscale",
            "a8w8_bpreshuffle",
            "a8w8_blockscale_bpreshuffle",
            "a4w4_blockscale",
            "vllm_moe_triton",
            "vllm_dense_tunableop",
            "sglang_dense_bf16",
        ]
        for name in expected:
            assert name in TUNER_ENV_VARS, f"Missing env var for {name}"
            assert TUNER_ENV_VARS[name], f"Empty env var for {name}"

    def test_a4w4_env_var_matches_aiter_serving_name(self):
        # Regression guard: aiter reads fp4/mxfp4 (gfx950-only) GEMM configs via
        # AITER_CONFIG_GEMM_A4W4, NOT the "_BLOCKSCALE" variant. A mismatch here
        # silently drops all tuned fp4 configs at serving (aiter falls back to its
        # bundled default CSV). See aiter/jit/core.py.
        assert TUNER_ENV_VARS["a4w4_blockscale"] == "AITER_CONFIG_GEMM_A4W4"

    def test_a4w4_env_var_is_read_by_installed_aiter(self):
        # When aiter is importable, verify the name against ground truth rather
        # than a hard-coded literal, so this tracks aiter if it ever renames.
        import importlib.util

        if importlib.util.find_spec("aiter") is None:
            import pytest

            pytest.skip("aiter not installed")
        from aiter.jit import core as aiter_core

        assert hasattr(aiter_core, "AITER_CONFIG_GEMM_A4W4")
        assert TUNER_ENV_VARS["a4w4_blockscale"] == "AITER_CONFIG_GEMM_A4W4"
