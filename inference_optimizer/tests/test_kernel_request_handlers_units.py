# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the small helpers inside ``kernel_request_handlers``."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

from inference_optimizer.orchestrator import kernel_request_handlers as krh
from inference_optimizer.orchestrator.shared_state import SharedState


class TestForgeGemmHelperCoverage:
    def test_resolve_backend_payload_env_and_default(self, monkeypatch):
        monkeypatch.delenv("GEMM_TUNING_BACKEND", raising=False)
        assert krh._resolve_gemm_tuning_backend({}) == "forge"
        monkeypatch.setenv("GEMM_TUNING_BACKEND", "geak")
        assert krh._resolve_gemm_tuning_backend({}) == "geak"
        assert krh._resolve_gemm_tuning_backend({"gemm_tuning_backend": "forge"}) == "forge"
        # Unknown values fall back to the default instead of surfacing an invalid backend.
        assert krh._resolve_gemm_tuning_backend({"gemm_tuning_backend": "unknown"}) == "forge"

    def test_parse_forge_gemm_sentinel(self):
        payload = {"status": "ok", "micro_decision": "candidate"}
        text = "noise\nFORGE_GEMM_TUNE_RESULT_BEGIN\n" + json.dumps(payload) + "\nFORGE_GEMM_TUNE_RESULT_END\n"
        assert krh._parse_forge_gemm_sentinel(text) == payload
        assert krh._parse_forge_gemm_sentinel("no sentinel") is None
        assert (
            krh._parse_forge_gemm_sentinel(
                "FORGE_GEMM_TUNE_RESULT_BEGIN\nnot-json\nFORGE_GEMM_TUNE_RESULT_END"
            )
            is None
        )

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_env_value_true(self, value):
        assert krh._truthy_env_value(value) is True

    @pytest.mark.parametrize("value", ["", "0", "false", "off", None])
    def test_truthy_env_value_false(self, value):
        assert krh._truthy_env_value(value) is False

    def test_resolve_forge_server_log_priority(self, tmp_path):
        state = SharedState()
        baseline = tmp_path / "baseline"
        current = tmp_path / "current"
        baseline.mkdir()
        current.mkdir()
        (baseline / "server.log").write_text("baseline", encoding="utf-8")
        (current / "server.log").write_text("current", encoding="utf-8")
        state.last_baseline = {"workspace": str(baseline)}
        state.current_best = {"workspace": str(current)}

        assert krh._resolve_forge_server_log(state, tmp_path) == str(current / "server.log")

    def test_resolve_forge_server_log_bounded_runs_fallback(self, tmp_path):
        state = SharedState()
        log = tmp_path / "runs" / "explore" / "abc" / "server.log"
        log.parent.mkdir(parents=True)
        log.write_text("x", encoding="utf-8")

        assert krh._resolve_forge_server_log(state, tmp_path) == str(log)

    def test_resolve_forge_precision_payload_override(self):
        state = SharedState(precision="bf16")
        assert krh._resolve_forge_precision_and_quant(
            state,
            {"precision": "fp8", "quant_type": "blockscale"},
        ) == ("fp8", "blockscale")

    def test_resolve_forge_precision_from_runtime_fp4(self):
        state = SharedState(precision="bf16")
        state.current_best = {"extra_server_args": "--quantization fp4", "extra_envs": {}}

        assert krh._resolve_forge_precision_and_quant(state, {}) == ("fp4", "fp4")

    def test_resolve_forge_precision_per_token_from_reference_env(self):
        state = SharedState(precision="bf16")
        state.current_best = {"extra_server_args": "--quantization fp8", "extra_envs": {}}
        state.reference_envs = {"SGLANG_USE_AITER_FP8_PER_TOKEN": "true"}

        assert krh._resolve_forge_precision_and_quant(state, {}) == ("fp8", "per_token")

    @staticmethod
    def _write_cfg(model_dir: Path, cfg: dict) -> str:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        return str(model_dir)

    def test_resolve_fp8_quant_type(self, tmp_path):
        block = self._write_cfg(
            tmp_path / "block",
            {"hidden_size": 7168, "quantization_config": {"weight_block_size": [128, 128]}},
        )
        method_block = self._write_cfg(
            tmp_path / "mblock",
            {"quantization_config": {"quant_method": "fp8_block"}},
        )
        plain = self._write_cfg(tmp_path / "plain", {"hidden_size": 2048})
        assert krh._resolve_fp8_quant_type(block) == "blockscale"
        assert krh._resolve_fp8_quant_type(method_block) == "blockscale"
        assert krh._resolve_fp8_quant_type(plain) == "per_token"
        # Multimodal: quantization_config nested under text_config is detected.
        nested_block = self._write_cfg(
            tmp_path / "nblock",
            {"text_config": {"quantization_config": {"weight_block_size": [128, 128]}}},
        )
        assert krh._resolve_fp8_quant_type(nested_block) == "blockscale"
        # Unreadable / missing config -> auto (forge sniffs the runtime log).
        assert krh._resolve_fp8_quant_type(str(tmp_path / "missing")) == "auto"
        assert krh._resolve_fp8_quant_type("") == "auto"

    def test_resolve_forge_precision_fp8_plain_model_routes_per_token(self, tmp_path):
        model = self._write_cfg(tmp_path / "plain", {"hidden_size": 2048})
        state = SharedState(precision="bf16", model_path=model)
        state.current_best = {"extra_server_args": "--quantization fp8", "extra_envs": {}}
        assert krh._resolve_forge_precision_and_quant(state, {}) == ("fp8", "per_token")

    def test_resolve_forge_precision_fp8_block_model_routes_blockscale(self, tmp_path):
        model = self._write_cfg(
            tmp_path / "block",
            {"hidden_size": 7168, "quantization_config": {"weight_block_size": [128, 128]}},
        )
        state = SharedState(precision="bf16", model_path=model)
        state.current_best = {"extra_server_args": "--quantization fp8", "extra_envs": {}}
        assert krh._resolve_forge_precision_and_quant(state, {}) == ("fp8", "blockscale")

    def test_resolve_forge_precision_fp8_unreadable_config_keeps_auto(self):
        # Matches the legacy contract: with no readable config, do not force a
        # tuner from Hyperloom -- let forge sniff the kernel_signature_log.
        state = SharedState(precision="bf16", model_path="/models/does-not-exist")
        state.current_best = {"extra_server_args": "--quantization fp8", "extra_envs": {}}
        assert krh._resolve_forge_precision_and_quant(state, {}) == ("fp8", "auto")

    def test_forge_gemm_tune_available_by_path_and_import(self, monkeypatch):
        monkeypatch.setattr(krh.shutil, "which", lambda _name: "/usr/bin/forge-gemm-tune")
        assert krh._forge_gemm_tune_available() is True

        monkeypatch.setattr(krh.shutil, "which", lambda _name: None)
        monkeypatch.setattr(krh.importlib.util, "find_spec", lambda _name: object())
        assert krh._forge_gemm_tune_available() is True

        monkeypatch.setattr(krh.importlib.util, "find_spec", lambda _name: None)
        assert krh._forge_gemm_tune_available() is False

    @pytest.mark.asyncio
    async def test_run_forge_gemm_tuning_reports_missing_cli(self, tmp_path, monkeypatch):
        state = SharedState(
            precision="bf16",
            framework="sglang",
            model_path="/models/qwen",
            gpu_type="mi300x",
            tp=1,
            conc=256,
        )
        state.save(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: False)

        result = await krh._run_forge_gemm_tuning({}, session_dir=tmp_path)

        assert result["status"] == "failed"
        assert result["error_class"] == "forge_gemm_tune_not_found"
        assert result["backend"] == "forge"

    def test_forge_gemm_tune_available_swallows_find_spec_error(self, monkeypatch):
        monkeypatch.setattr(krh.shutil, "which", lambda _name: None)

        def _boom(_name):
            raise ValueError("ambiguous spec")

        monkeypatch.setattr(krh.importlib.util, "find_spec", _boom)
        assert krh._forge_gemm_tune_available() is False

    def test_resolve_forge_precision_falls_back_to_bf16(self, monkeypatch):
        # Empty session precision + no fp8/fp4 quantization → bf16/auto default.
        state = SharedState(precision="")
        state.current_best = {"extra_server_args": "", "extra_envs": {}}
        import inference_optimizer.orchestrator.roofline_ceiling as rc

        def _raise(*_a, **_k):
            raise RuntimeError("no runtime workload")

        monkeypatch.setattr(rc, "resolve_runtime_workload", _raise)
        assert krh._resolve_forge_precision_and_quant(state, {}) == ("bf16", "auto")

    def test_resolve_forge_server_log_uses_baseline_when_no_current_best(self, tmp_path):
        state = SharedState()
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "server.log").write_text("baseline", encoding="utf-8")
        state.last_baseline = {"workspace": str(baseline)}

        assert krh._resolve_forge_server_log(state, tmp_path) == str(baseline / "server.log")

    def test_resolve_forge_shapes_reads_artifact_paths_dict(self, tmp_path):
        state = SharedState()
        shapes = tmp_path / "gemm_shapes.json"
        shapes.write_text(json.dumps({"shapes": [{"m": 1, "n": 2, "k": 3}]}), encoding="utf-8")
        state.last_trace_analyze = {"artifact_paths": {"gemm_shapes_json": str(shapes)}}

        assert krh._resolve_forge_shapes(state, tmp_path) == str(shapes)

    def test_resolve_forge_shapes_skips_incompatible_candidate(self, tmp_path):
        state = SharedState()
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([{"only": "noMNK"}]), encoding="utf-8")
        state.last_trace_analyze = {"shapes_json": str(bad)}

        assert krh._resolve_forge_shapes(state, tmp_path) == ""

    def test_is_forge_compatible_shapes_json_rejects_non_dict_sample(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([123, 456]), encoding="utf-8")
        assert krh._is_forge_compatible_shapes_json(bad) is False

    @staticmethod
    def _write_aiter_csv(session_dir: Path, hash_id: str, fname: str, rows: str) -> Path:
        cfg = session_dir / "runs" / "specialist" / hash_id / "worktree" / "aiter" / "configs"
        cfg.mkdir(parents=True, exist_ok=True)
        path = cfg / fname
        path.write_text(rows, encoding="utf-8")
        return path

    def test_resolve_forge_untuned_csv_fp8_blockscale(self, tmp_path):
        # fp8 auto -> blockscale CSV recorded by the specialist phase.
        expected = self._write_aiter_csv(
            tmp_path, "abc", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,1536,7168\n"
        )
        assert krh._resolve_forge_untuned_csv(tmp_path, "fp8", "auto") == str(expected)
        assert krh._resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale") == str(expected)

    def test_resolve_forge_untuned_csv_per_token(self, tmp_path):
        expected = self._write_aiter_csv(
            tmp_path, "abc", "a8w8_untuned_gemm.csv", "M,N,K,q_dtype_w\n16,1536,7168,fp8\n"
        )
        assert krh._resolve_forge_untuned_csv(tmp_path, "fp8", "per_token") == str(expected)

    def test_resolve_forge_untuned_csv_skips_header_only(self, tmp_path):
        # Header-only / empty files must not be passed as a real shape source.
        self._write_aiter_csv(tmp_path, "abc", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n")
        assert krh._resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale") == ""

    def test_resolve_forge_untuned_csv_picks_newest_nonempty(self, tmp_path):
        old = self._write_aiter_csv(
            tmp_path, "old", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n1,2,3\n"
        )
        new = self._write_aiter_csv(
            tmp_path, "new", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n4,5,6\n"
        )
        import os

        os.utime(old, (1, 1))
        os.utime(new, (10_000_000, 10_000_000))
        assert krh._resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale") == str(new)

    def test_resolve_forge_untuned_csv_bf16_returns_empty(self, tmp_path):
        # bf16 dense derives shapes from config.json; no CSV needed.
        self._write_aiter_csv(tmp_path, "abc", "bf16_untuned_gemm.csv", "M,N,K\n1,2,3\n")
        assert krh._resolve_forge_untuned_csv(tmp_path, "bf16", "none") == ""

    def test_resolve_forge_untuned_csv_no_specialist_dir(self, tmp_path):
        assert krh._resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale") == ""

    @staticmethod
    def _write_model_config(model_dir: Path, hidden_size: int) -> str:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(
            json.dumps({"hidden_size": hidden_size}), encoding="utf-8"
        )
        return str(model_dir)

    def test_resolve_forge_untuned_csv_rejects_model_mismatch(self, tmp_path):
        # Repro: specialist CSV carries the AITER default DeepSeek shapes
        # (K=7168) while the model under test has hidden_size=2048. The CSV must
        # be rejected so forge derives correct per-model shapes from config.json.
        self._write_aiter_csv(
            tmp_path, "abc", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,1536,7168\n"
        )
        model_path = self._write_model_config(tmp_path / "model", hidden_size=2048)
        assert (
            krh._resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale", model_path) == ""
        )

    def test_resolve_forge_untuned_csv_accepts_model_match(self, tmp_path):
        # A CSV whose K column includes the model hidden_size is the real
        # per-model shape set and must be accepted.
        expected = self._write_aiter_csv(
            tmp_path,
            "abc",
            "a8w8_blockscale_untuned_gemm.csv",
            "M,N,K\n16,6144,2048\n16,2048,8192\n",
        )
        model_path = self._write_model_config(tmp_path / "model", hidden_size=2048)
        assert krh._resolve_forge_untuned_csv(
            tmp_path, "fp8", "blockscale", model_path
        ) == str(expected)

    def test_resolve_forge_untuned_csv_no_model_path_keeps_legacy(self, tmp_path):
        # Backward compatible: without a model_path the resolver cannot validate
        # and keeps the legacy behaviour of returning the newest non-empty CSV.
        expected = self._write_aiter_csv(
            tmp_path, "abc", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,1536,7168\n"
        )
        assert krh._resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale") == str(expected)

    def test_resolve_forge_untuned_csv_unreadable_config_keeps_csv(self, tmp_path):
        # When config.json is missing/unreadable we cannot validate; preserve the
        # legacy behaviour rather than dropping a possibly-valid CSV.
        expected = self._write_aiter_csv(
            tmp_path, "abc", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,1536,7168\n"
        )
        assert krh._resolve_forge_untuned_csv(
            tmp_path, "fp8", "blockscale", str(tmp_path / "no_such_model")
        ) == str(expected)

    def test_csv_matches_model_helpers(self, tmp_path):
        csv_mismatch = self._write_aiter_csv(
            tmp_path, "h1", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,1536,7168\n"
        )
        csv_match = self._write_aiter_csv(
            tmp_path, "h2", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,6144,2048\n"
        )
        model_path = self._write_model_config(tmp_path / "m", hidden_size=2048)
        assert krh._model_hidden_size(model_path) == 2048
        assert krh._csv_k_values(csv_mismatch) == {7168}
        assert krh._csv_k_values(csv_match) == {2048}
        assert krh._csv_matches_model(csv_mismatch, model_path) is False
        assert krh._csv_matches_model(csv_match, model_path) is True
        # No model_path / unreadable config -> cannot validate -> accept.
        assert krh._csv_matches_model(csv_mismatch, "") is True

    def test_read_forge_result_json(self, tmp_path):
        (tmp_path / "result.json").write_text(
            json.dumps({"status": "skipped", "tuners_skipped": [{"tuner": "a8w8"}]}),
            encoding="utf-8",
        )
        out = krh._read_forge_result_json(tmp_path)
        assert out["status"] == "skipped"
        assert krh._read_forge_result_json(tmp_path / "missing") == {}

    def test_derive_gemm_skip_reason(self):
        skipped = [
            {"tuner": "a8w8_blockscale", "skip_reason": "needs csv"},
            {"tuner": "fmoe_ck", "skip_reason": ""},
            {"tuner": "x"},
        ]
        assert krh._derive_gemm_skip_reason(skipped) == "a8w8_blockscale: needs csv"
        assert krh._derive_gemm_skip_reason(None) == ""
        assert krh._derive_gemm_skip_reason([]) == ""

    def test_path_is_existing_file_handles_too_long(self):
        # The production crash: an inline JSON list handed in as a "path".
        inline = "[{'M': 64, 'N': 16384, 'K': 3072, 'dtype': 'bf16'}]" * 6
        assert len(inline) > 255
        assert krh._path_is_existing_file(inline) is False  # must not raise OSError(36)

    def test_path_is_existing_file_true(self, tmp_path):
        f = tmp_path / "real.csv"
        f.write_text("M,N,K\n", encoding="utf-8")
        assert krh._path_is_existing_file(str(f)) is True

    def test_normalize_forge_shapes_json_existing_path(self, tmp_path):
        f = tmp_path / "shapes.json"
        f.write_text("[{\"M\":1,\"N\":2,\"K\":3}]", encoding="utf-8")
        assert krh._normalize_forge_shapes_json(str(f), tmp_path) == str(f)

    def test_normalize_forge_shapes_json_inline_string(self, tmp_path):
        # The exact production payload shape: a Python-repr list (single quotes).
        inline = "[{'M': 64, 'N': 16384, 'K': 3072, 'dtype': 'bf16'}]"
        out = krh._normalize_forge_shapes_json(inline, tmp_path)
        assert out == str(tmp_path / "forge_shapes.json")
        data = json.loads(Path(out).read_text())
        assert data[0]["M"] == 64

    def test_normalize_forge_shapes_json_inline_list(self, tmp_path):
        out = krh._normalize_forge_shapes_json([{"M": 1, "N": 2, "K": 3}], tmp_path)
        assert Path(out).is_file()
        assert json.loads(Path(out).read_text())[0]["N"] == 2

    def test_normalize_forge_shapes_json_empty_and_garbage(self, tmp_path):
        assert krh._normalize_forge_shapes_json("", tmp_path) == ""
        assert krh._normalize_forge_shapes_json(None, tmp_path) == ""
        # Non-JSON, non-existent path string -> unusable.
        assert krh._normalize_forge_shapes_json("not_a_real_file.json", tmp_path) == ""

    def test_normalize_tokens_list_and_bracketed_string(self):
        # The production bug: tokens passed as a list or its string form.
        assert krh._normalize_tokens([4, 8, 64]) == "4,8,64"
        assert krh._normalize_tokens("[4, 8, 64]") == "4,8,64"
        assert krh._normalize_tokens("[64]") == "64"
        assert krh._normalize_tokens("4,8,64") == "4,8,64"

    def test_normalize_tokens_empty_and_garbage(self):
        assert krh._normalize_tokens(None) == ""
        assert krh._normalize_tokens("") == ""
        assert krh._normalize_tokens([]) == ""
        assert krh._normalize_tokens("[abc, 16]") == "16"

    def test_resolve_forge_shapes_returns_empty_for_non_dict_trace(self):
        state = SharedState()
        state.last_trace_analyze = ["not", "a", "dict"]
        assert krh._resolve_forge_shapes(state, Path("/tmp")) == ""

    def test_resolve_forge_shapes_finds_file_beside_candidates(self, tmp_path):
        state = SharedState()
        candidates = tmp_path / "candidates.json"
        candidates.write_text("[]", encoding="utf-8")
        shapes = tmp_path / "shapes.json"
        shapes.write_text(json.dumps([{"M": 1, "N": 2, "K": 3}]), encoding="utf-8")
        state.last_trace_analyze = {"candidates_path": str(candidates)}

        assert krh._resolve_forge_shapes(state, tmp_path) == str(shapes)

    @pytest.mark.asyncio
    async def test_run_forge_gemm_tuning_requires_model_path(self, tmp_path, monkeypatch):
        state = SharedState(precision="bf16", framework="sglang")
        state.save(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.delenv("MODEL_PATH", raising=False)

        result = await krh._run_forge_gemm_tuning({}, session_dir=tmp_path)

        assert result["status"] == "failed"
        assert result["error_class"] == "model_path_missing"

    @pytest.mark.asyncio
    async def test_run_forge_gemm_tuning_maps_failed_micro_decision(self, tmp_path, monkeypatch):
        state = SharedState(
            precision="bf16",
            framework="sglang",
            model_path="/models/qwen",
            gpu_type="mi300x",
            tp=1,
            conc=64,
        )
        state.save(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)

        sentinel = (
            "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
            + json.dumps({"micro_decision": "failed"})
            + "\nFORGE_GEMM_TUNE_RESULT_END\n"
        )

        async def _fake_subprocess(cmd, *, timeout_sec):
            return 1, sentinel, ""

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        result = await krh._run_forge_gemm_tuning({}, session_dir=tmp_path)

        assert result["decision"] == "REVERT"
        assert result["status"] == "failed"
        assert result["backend"] == "forge"

    @pytest.mark.asyncio
    async def test_run_forge_gemm_tuning_tags_engine_forge(self, tmp_path, monkeypatch):
        """Forge runs must carry ``engine='forge'`` so the breakdown attributes
        them to the forge source instead of the ``geak`` default."""
        state = SharedState(
            precision="bf16",
            framework="sglang",
            model_path="/models/qwen",
            gpu_type="mi300x",
            tp=1,
            conc=64,
        )
        state.save(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)

        sentinel = (
            "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
            + json.dumps({"status": "ok", "micro_decision": "skipped"})
            + "\nFORGE_GEMM_TUNE_RESULT_END\n"
        )

        async def _fake_subprocess(cmd, *, timeout_sec):
            return 0, sentinel, ""

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        result = await krh._run_forge_gemm_tuning({}, session_dir=tmp_path)

        assert result["engine"] == "forge"


def _ensure_torch_module(monkeypatch):
    try:
        import torch
    except ModuleNotFoundError:
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(device_count=lambda: 0),
        )
        monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


# _coerce_runtime_value
class TestCoerceRuntimeValue:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("42", 42),
            ("  17  ", 17),
            ("3.14", pytest.approx(3.14)),
            ("not-a-number", "not-a-number"),
            ("3.14.invalid", "3.14.invalid"),
            (5, 5),
            (3.5, 3.5),
            (None, None),
        ],
    )
    def test_roundtrips(self, value, expected):
        assert krh._coerce_runtime_value(value) == expected


# _backend_order
class TestBackendOrder:
    def test_documented_kernel_opt_backends_env_is_honored(self, monkeypatch):
        monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.setenv("KERNEL_OPT_BACKENDS", "geak")

        assert krh._backend_order({}) == ["geak"]

    def test_documented_kernel_opt_backends_env_is_case_normalized(self, monkeypatch):
        monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.setenv("KERNEL_OPT_BACKENDS", " GEAK , CoDeX ")

        assert krh._backend_order({}) == ["geak", "codex"]


# _candidate_env_allowed
class TestCandidateEnvAllowed:
    @pytest.mark.parametrize("name", ["AWS_SECRET_ACCESS_KEY", "ANTHROPIC_API_KEY"])
    def test_sensitive_env_blocked(self, name):
        assert krh._candidate_env_allowed(name) is False

    def test_known_prefix_allowed(self):
        # Probe one prefix without depending on the product-internal allowlist.
        prefixes = krh._CANDIDATE_ENV_PREFIXES
        assert prefixes  # registry not empty
        sample = next(iter(prefixes))
        assert krh._candidate_env_allowed(sample + "FOO") is True

    def test_explicit_allowlisted_key(self):
        keys = krh._CANDIDATE_ENV_KEYS
        if not keys:
            pytest.skip("no explicit allowlist entries in build")
        sample = next(iter(keys))
        assert krh._candidate_env_allowed(sample) is True


# _is_runtime_generated_kernel
class TestRuntimeGeneratedKernel:
    def test_runtime_generated_path_treats_as_generated(self):
        markers = krh._RUNTIME_GENERATED_SOURCE_MARKERS
        if not markers:
            pytest.skip("no runtime markers in build")
        marker = next(iter(markers))
        assert krh._is_runtime_generated_kernel("kernel", f"/tmp/{marker}_x.py") is True

    def test_reusable_source_root_overrides_compile_marker(self):
        markers = krh._COMPILE_GENERATED_NAME_MARKERS
        roots = krh._reusable_source_roots()
        if not markers or not roots:
            pytest.skip("required tables empty in build")
        marker = next(iter(markers))
        reusable_root = next(iter(roots))
        # Name matches but source lives under a reusable root → False.
        assert krh._is_runtime_generated_kernel(marker, f"{reusable_root}/foo.py") is False


# _split_server_args
class TestSplitServerArgs:
    def test_empty_returns_empty(self):
        assert krh._split_server_args("") == []

    def test_split_uses_shlex(self):
        argv = krh._split_server_args("--foo 1 --bar 'x y'")
        assert argv == ["--foo", "1", "--bar", "x y"]

    def test_unterminated_quote_returns_empty(self):
        # shlex.split raises ValueError on bad input; helper returns [].
        argv = krh._split_server_args('--foo "unterminated')
        assert argv == []


# _load_candidate_metadata
class TestLoadCandidateMetadata:
    def test_uses_inline_candidate(self):
        out = krh._load_candidate_metadata({"candidate": {"kernel_id": "x"}})
        assert out == {"kernel_id": "x"}

    def test_returns_empty_when_no_kernel_id(self):
        assert krh._load_candidate_metadata({}) == {}
        assert krh._load_candidate_metadata({"candidates_path": "x"}) == {}

    def test_reads_kernel_from_disk(self, tmp_path):
        candidates = tmp_path / "hot.json"
        candidates.write_text(
            json.dumps(
                {
                    "hot_kernels": [
                        {"kernel_id": "k0", "name": "first"},
                        {"kernel_id": "k1", "name": "second"},
                    ],
                }
            )
        )
        out = krh._load_candidate_metadata(
            {
                "candidates_path": str(candidates),
                "kernel_id": "k1",
            }
        )
        assert out["name"] == "second"

    def test_returns_empty_on_missing_kernel(self, tmp_path):
        candidates = tmp_path / "hot.json"
        candidates.write_text(json.dumps({"hot_kernels": []}))
        assert (
            krh._load_candidate_metadata(
                {
                    "candidates_path": str(candidates),
                    "kernel_id": "missing",
                }
            )
            == {}
        )

    def test_returns_empty_on_bad_json(self, tmp_path):
        candidates = tmp_path / "hot.json"
        candidates.write_text("{not json")
        assert (
            krh._load_candidate_metadata(
                {
                    "candidates_path": str(candidates),
                    "kernel_id": "x",
                }
            )
            == {}
        )


# _load_materialized_workload_metadata
class TestLoadMaterializedWorkloadMetadata:
    def test_empty_when_no_path(self):
        assert krh._load_materialized_workload_metadata("") == {}

    def test_empty_when_path_missing(self, tmp_path):
        assert krh._load_materialized_workload_metadata(str(tmp_path / "no.yaml")) == {}

    def test_parses_sglang_metadata(self, tmp_path):
        cfg = tmp_path / "magpie.yaml"
        cfg.write_text(
            "benchmark:\n"
            "  framework: sglang\n"
            "  model: /weights/m\n"
            "  precision: bf16\n"
            "  envs:\n"
            "    TP: 1\n"
            "    CONC: 16\n"
            "    ISL: 1024\n"
            "    OSL: 512\n"
            "    EXTRA_SGLANG_ARGS: '--foo 1'\n"
        )
        out = krh._load_materialized_workload_metadata(str(cfg))
        runtime = out["runtime_args"]
        assert runtime["framework"] == "sglang"
        assert runtime["server_args"] == "--foo 1"
        assert runtime["server_args_argv"] == ["--foo", "1"]
        workload = runtime["workload"]
        assert workload["tp"] == 1
        assert workload["conc"] == 16
        assert "TP" in out["env_vars"]

    @pytest.mark.parametrize(
        "framework,env_name,expected_args",
        [
            ("sglang", "EXTRA_SGLANG_ARGS", "--mem-fraction-static=0.8"),
            ("vllm", "EXTRA_VLLM_ARGS", "--gpu-memory-utilization 0.9"),
            ("atom", "EXTRA_ATOM_ARGS", "--trust-remote-code --level 2 --enable-expert-parallel"),
        ],
    )
    def test_server_args_read_from_per_framework_env_key(
        self,
        tmp_path,
        framework,
        env_name,
        expected_args,
    ):
        """The handler reads the per-framework ``EXTRA_<FRAMEWORK>_ARGS`` slot, not always ``EXTRA_SGLANG_ARGS``."""
        cfg = tmp_path / f"magpie_{framework}.yaml"
        cfg.write_text(
            "benchmark:\n"
            f"  framework: {framework}\n"
            "  model: /weights/m\n"
            "  precision: bf16\n"
            "  envs:\n"
            "    TP: 4\n"
            "    CONC: 32\n"
            "    ISL: 1024\n"
            "    OSL: 1024\n"
            f"    {env_name}: '{expected_args}'\n"
        )
        out = krh._load_materialized_workload_metadata(str(cfg))
        runtime = out["runtime_args"]
        assert runtime["framework"] == framework
        assert runtime["server_args"] == expected_args, (
            f"framework={framework!r} expected server_args={expected_args!r}; got {runtime['server_args']!r}."
        )

    def test_atom_server_args_not_read_from_extra_sglang_args(self, tmp_path):
        """When an atom YAML carries both EXTRA_ATOM_ARGS and a stray EXTRA_SGLANG_ARGS, the atom slot wins."""
        cfg = tmp_path / "magpie_atom_mixed.yaml"
        cfg.write_text(
            "benchmark:\n"
            "  framework: atom\n"
            "  model: /weights/m\n"
            "  precision: fp8\n"
            "  envs:\n"
            "    TP: 4\n"
            "    CONC: 32\n"
            "    ISL: 1024\n"
            "    OSL: 1024\n"
            "    EXTRA_SGLANG_ARGS: '--should-be-ignored'\n"
            "    EXTRA_ATOM_ARGS: '--trust-remote-code --level 2'\n"
        )
        out = krh._load_materialized_workload_metadata(str(cfg))
        runtime = out["runtime_args"]
        assert runtime["framework"] == "atom"
        assert runtime["server_args"] == "--trust-remote-code --level 2"
        assert "--should-be-ignored" not in runtime["server_args"]


# enrichment helpers
class TestEnrichCandidate:
    def test_enrich_candidate_runtime_metadata_setdefault_semantics(self):
        candidates = [{"kernel_id": "k", "env_vars": {"TP": "8"}}]
        metadata = {"env_vars": {"TP": "1", "CONC": "16"}, "runtime_args": {"framework": "sglang"}}
        krh._enrich_candidate_runtime_metadata(candidates, metadata)
        assert candidates[0]["env_vars"] == {"TP": "8", "CONC": "16"}
        assert candidates[0]["runtime_args"]["framework"] == "sglang"

    def test_enrich_candidate_runtime_metadata_ignores_non_dict_items(self):
        candidates = ["not a dict", {"kernel_id": "x"}]
        krh._enrich_candidate_runtime_metadata(candidates, {"env_vars": {"A": "B"}})
        assert candidates[1].get("env_vars") == {"A": "B"}

    def test_enrich_candidate_trace_report_skips_blank_path(self):
        candidates = [{"kernel_id": "k"}]
        krh._enrich_candidate_trace_report(candidates, "")
        assert "trace_report_path" not in candidates[0]

    def test_enrich_candidates_artifact_noop_when_missing_path(self):
        krh._enrich_candidates_artifact("", {"env_vars": {}}, trace_report_path="")


# atom-aware reusable kernel detection
class TestReusableSourceRootsAtom:
    """atom layout prefixes participate in cross-task kernel reuse
    alongside aiter/sglang/vllm."""

    def test_includes_atom_editable_path(self):
        # The matcher lowercases its source-file input, so the stored prefix is
        # lowercase ``/app/atom/atom/`` even though the real path is ``/app/ATOM/atom/``.
        assert any("/app/atom/atom/" in r.lower() for r in krh._reusable_source_roots())

    def test_includes_atom_site_packages_python_3_10(self):
        assert any("/opt/venv/lib/python3.10/site-packages/atom/" in r for r in krh._reusable_source_roots())

    def test_includes_atom_site_packages_python_3_12(self):
        assert any("/opt/venv/lib/python3.12/site-packages/atom/" in r for r in krh._reusable_source_roots())

    def test_atom_path_classified_as_reusable(self):
        """An atom-owned kernel source under /app/ATOM/atom/ is NOT runtime-generated even if its name matches a compile marker."""
        markers = krh._COMPILE_GENERATED_NAME_MARKERS
        if not markers:
            pytest.skip("compile markers empty in build")
        marker = next(iter(markers))
        result = krh._is_runtime_generated_kernel(
            marker,
            "/app/ATOM/atom/model_engine/model_runner.py",
        )
        assert result is False

    def test_non_framework_path_under_app_is_not_reusable(self):
        """A non-atom path under /app/ must NOT match the atom reusable-source-root prefix."""
        markers = krh._COMPILE_GENERATED_NAME_MARKERS
        if not markers:
            pytest.skip("compile markers empty in build")
        marker = next(iter(markers))
        # Under /app/ but not /app/ATOM/atom/ → runtime-generated (not reusable).
        result = krh._is_runtime_generated_kernel(
            marker,
            "/app/session_dir/runs/baseline/foo.py",
        )
        assert result is True


# run_gemm_tuning_handler
class TestRunGemmTuningHandler:
    def test_skips_non_fp8_without_kernel_agent_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMM_TUNING_BACKEND", "geak")
        state = SharedState(precision="bf16", framework="sglang")
        state.save(tmp_path)

        result = asyncio.run(krh.run_gemm_tuning_handler({}, session_dir=tmp_path))

        assert result["status"] == "skipped"
        assert result["error_class"] == "fp8_only_action"

    def test_builds_task_file_input_not_task_argv(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMM_TUNING_BACKEND", "geak")
        root = tmp_path / "kernel-agent"
        tool = root / "tools" / "gemm_tuning.py"
        tool.parent.mkdir(parents=True)
        tool.write_text("# placeholder\n")
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(root))

        state = SharedState(
            precision="fp8",
            framework="sglang",
            model_path="/models/qwen-fp8",
            gpu_type="mi355x",
            tp=1,
            conc=64,
            isl=1024,
            osl=1024,
            baseline_tput=4479.0,
        )
        state.save(tmp_path)
        captured: dict[str, object] = {}

        async def fake_run(cmd: list[str], *, timeout_sec: int):
            captured["cmd"] = cmd
            captured["timeout_sec"] = timeout_sec
            input_path = cmd[cmd.index("--input-json") + 1]
            data = json.loads(Path(input_path).read_text())
            assert data["framework"] == "sglang"
            assert data["precision"] == "fp8"
            return (
                0,
                json.dumps(
                    {
                        "status": "ok",
                        "decision": "KEEP",
                        "best_speedup": 1.2,
                        "tuned_file": "/tmp/a8w8_blockscale_tuned_gemm.csv",
                    }
                ),
                "",
            )

        monkeypatch.setattr(krh, "_run_subprocess", fake_run)

        result = asyncio.run(
            krh.run_gemm_tuning_handler(
                {
                    "benchmark_script": "/workspace/run_sglang_test.sh",
                    "dry_run": True,
                    "task_id": "t1",
                },
                session_dir=tmp_path,
            )
        )

        assert result["status"] == "ok"
        cmd_text = " ".join(captured["cmd"])  # type: ignore[arg-type]
        assert "run_sglang_test" not in cmd_text
        assert "gemm_a8w8_blockscale_tune" not in cmd_text
        assert "--input-json" in captured["cmd"]  # type: ignore[operator]

    def test_generates_isolated_benchmark_script_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMM_TUNING_BACKEND", "geak")
        root = tmp_path / "kernel-agent"
        tool = root / "tools" / "gemm_tuning.py"
        tool.parent.mkdir(parents=True)
        tool.write_text("# placeholder\n")
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(root))

        state = SharedState(
            precision="fp8",
            framework="sglang",
            model_path="/models/qwen-fp8",
            gpu_type="mi355x",
            tp=1,
            conc=64,
            isl=1024,
            osl=1024,
            baseline_tput=4479.0,
        )
        state.save(tmp_path)

        async def fake_run(cmd: list[str], *, timeout_sec: int):
            input_path = cmd[cmd.index("--input-json") + 1]
            data = json.loads(Path(input_path).read_text())
            bench = Path(data["benchmark_script"])
            text = bench.read_text()
            assert bench.name == "geak_gemm_benchmark.sh"
            assert 'PORT="${PORT:-18888}"' in text
            assert "pgrep" not in text
            assert data["benchmark_script"].endswith("geak_gemm_benchmark.sh")
            return (
                0,
                json.dumps(
                    {
                        "status": "ok",
                        "decision": "KEEP",
                        "best_speedup": 1.1,
                        "tuned_file": "/tmp/tuned.csv",
                    }
                ),
                "",
            )

        monkeypatch.setattr(krh, "_run_subprocess", fake_run)

        result = asyncio.run(
            krh.run_gemm_tuning_handler(
                {"dry_run": True, "task_id": "auto"},
                session_dir=tmp_path,
            )
        )

        assert result["status"] == "ok"

    def test_forge_uses_runtime_fp8_blockscale_for_aiter_backend(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMM_TUNING_BACKEND", "forge")
        state = SharedState(
            precision="bf16",
            framework="sglang",
            model_path="/models/qwen",
            gpu_type="mi300x",
            tp=1,
            conc=256,
        )
        state.current_best = {
            "extra_server_args": "--quantization fp8 --fp8-gemm-backend aiter",
            "extra_envs": {},
        }
        state.save(tmp_path)
        captured: dict[str, object] = {}

        async def fake_run(cmd: list[str], *, timeout_sec: int):
            captured["cmd"] = cmd
            return (
                0,
                "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
                + json.dumps(
                    {
                        "status": "ok",
                        "micro_decision": "candidate",
                        "recommended_env": {"AITER_CONFIG_FMOE": "/tmp/fmoe.csv"},
                        "tuners_run": [{"best_micro_speedup": 1.1}],
                    }
                )
                + "\nFORGE_GEMM_TUNE_RESULT_END\n",
                "",
            )

        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(krh, "_run_subprocess", fake_run)

        result = asyncio.run(
            krh.run_gemm_tuning_handler({"task_id": "forge"}, session_dir=tmp_path)
        )

        cmd = captured["cmd"]  # type: ignore[assignment]
        input_path = cmd[cmd.index("--input-json") + 1]
        data = json.loads(Path(input_path).read_text())
        assert data["precision"] == "fp8"
        # Do not force blockscale from Hyperloom. Forge should inspect
        # kernel_signature_log when available; without a log it defaults to
        # blockscale internally.
        assert data["quant_type"] == "auto"
        assert data["conc"] == 256
        assert result["extra_envs"] == {"AITER_CONFIG_FMOE": "/tmp/fmoe.csv"}

    def test_handler_writes_gemm_tuning_audit_row(self, tmp_path, monkeypatch):
        """run_gemm_tuning_handler appends a source-attribution audit row that
        the Langfuse emitter backfills as a ``gemm_tuning:<engine>`` span."""
        from inference_optimizer.session_paths import gemm_tuning_steps_path

        monkeypatch.setenv("GEMM_TUNING_BACKEND", "forge")
        state = SharedState(
            precision="bf16",
            framework="sglang",
            model_path="/models/qwen",
            gpu_type="mi300x",
            tp=1,
            conc=256,
        )
        state.save(tmp_path)

        async def fake_run(cmd: list[str], *, timeout_sec: int):
            return (
                0,
                "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
                + json.dumps(
                    {
                        "status": "ok",
                        "micro_decision": "candidate",
                        "recommended_env": {"AITER_CONFIG_FMOE": "/tmp/fmoe.csv"},
                        "tuners_run": [{"tuner": "fmoe_ck", "best_micro_speedup": 1.1}],
                    }
                )
                + "\nFORGE_GEMM_TUNE_RESULT_END\n",
                "",
            )

        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(krh, "_run_subprocess", fake_run)

        asyncio.run(krh.run_gemm_tuning_handler({"task_id": "forge"}, session_dir=tmp_path))

        rows = [
            json.loads(line)
            for line in gemm_tuning_steps_path(tmp_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        row = rows[0]
        assert row["kind"] == "gemm_tuning"
        assert row["engine"] == "forge"
        assert row["decision"] == "KEEP"
        assert row["tuners_run"][0]["tuner"] == "fmoe_ck"

    def test_forge_uses_per_token_only_for_explicit_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMM_TUNING_BACKEND", "forge")
        state = SharedState(
            precision="bf16",
            framework="sglang",
            model_path="/models/qwen",
            gpu_type="mi300x",
            tp=1,
            conc=256,
        )
        state.current_best = {
            "extra_server_args": "--quantization fp8 --fp8-gemm-backend aiter",
            "extra_envs": {"SGLANG_USE_AITER_FP8_PER_TOKEN": "1"},
        }
        state.save(tmp_path)
        captured: dict[str, object] = {}

        async def fake_run(cmd: list[str], *, timeout_sec: int):
            captured["cmd"] = cmd
            return (
                0,
                "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
                + json.dumps(
                    {
                        "status": "skipped",
                        "micro_decision": "skipped",
                        "recommended_env": {},
                    }
                )
                + "\nFORGE_GEMM_TUNE_RESULT_END\n",
                "",
            )

        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(krh, "_run_subprocess", fake_run)

        asyncio.run(krh.run_gemm_tuning_handler({"task_id": "forge"}, session_dir=tmp_path))

        cmd = captured["cmd"]  # type: ignore[assignment]
        input_path = cmd[cmd.index("--input-json") + 1]
        data = json.loads(Path(input_path).read_text())
        assert data["precision"] == "fp8"
        assert data["quant_type"] == "per_token"

    def test_forge_fallback_to_session_precision_when_no_quantization(self, tmp_path, monkeypatch):
        """When current_best has no --quantization, fall back to state.precision."""
        monkeypatch.setenv("GEMM_TUNING_BACKEND", "forge")
        state = SharedState(
            precision="bf16",
            framework="sglang",
            model_path="/models/moe",
            gpu_type="mi300x",
            tp=1,
            conc=256,
        )
        state.current_best = {"extra_server_args": "", "extra_envs": {}}
        state.save(tmp_path)
        captured: dict[str, object] = {}

        async def fake_run(cmd: list[str], *, timeout_sec: int):
            captured["cmd"] = cmd
            return (0, json.dumps({"status": "ok", "micro_decision": "skipped"}), "")

        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(krh, "_run_subprocess", fake_run)

        asyncio.run(krh.run_gemm_tuning_handler({"task_id": "forge"}, session_dir=tmp_path))

        cmd = captured["cmd"]  # type: ignore[assignment]
        input_path = cmd[cmd.index("--input-json") + 1]
        data = json.loads(Path(input_path).read_text())
        assert data["precision"] == "bf16"
        assert data["quant_type"] == "auto"

    def test_forge_shapes_json_schema_validation(self, tmp_path):
        good = tmp_path / "good_shapes.json"
        good.write_text(json.dumps({"shapes": [{"M": 1, "N": 2, "K": 3}]}))
        bad_empty = tmp_path / "bad_empty.json"
        bad_empty.write_text(json.dumps({"shapes": []}))
        bad_trace_shape = tmp_path / "bad_trace_shape.json"
        bad_trace_shape.write_text(json.dumps([{"shape": [1, 2, 3]}]))

        assert krh._is_forge_compatible_shapes_json(good) is True
        assert krh._is_forge_compatible_shapes_json(bad_empty) is False
        assert krh._is_forge_compatible_shapes_json(bad_trace_shape) is False
        assert krh._is_forge_compatible_shapes_json(tmp_path / "missing.json") is False

    def test_resolve_forge_shapes_prefers_compatible_artifact(self, tmp_path):
        session_dir = tmp_path / "session"
        shapes = tmp_path / "shapes.json"
        shapes.write_text(json.dumps([{"m": 4, "n": 5, "k": 6}]))
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([{"shape": [1, 2, 3]}]))

        state = SharedState()
        state.last_trace_analyze = {
            "shapes_json": str(bad),
            "artifact_paths": {"gemm_shapes_json": str(shapes)},
        }

        assert krh._resolve_forge_shapes(state, session_dir) == str(shapes)


# _default_geak_budget_minutes / _geak_budget_minutes — orchestrator-side mirror
# of the kernel-agent default (PR #301); the legacy 90 forced quick-mode timing.
class TestDefaultGeakBudgetMinutes:
    @pytest.mark.parametrize(
        "geak_run_mode, expected",
        [
            (None, 130.0),  # unset -> full default
            ("", 130.0),  # empty -> full default
            ("full", 130.0),
            ("FULL", 130.0),  # case-insensitive
            ("  full  ", 130.0),  # whitespace tolerated
            ("garbage", 130.0),  # unknown values fall back to full
            ("quick", 70.0),
            ("QUICK", 70.0),
            ("  quick  ", 70.0),
        ],
    )
    def test_tracks_geak_run_mode(self, monkeypatch, geak_run_mode, expected):
        if geak_run_mode is None:
            monkeypatch.delenv("GEAK_RUN_MODE", raising=False)
        else:
            monkeypatch.setenv("GEAK_RUN_MODE", geak_run_mode)
        assert krh._default_geak_budget_minutes() == expected


class TestGeakBudgetMinutes:
    def test_payload_override_wins(self, monkeypatch):
        monkeypatch.setenv("GEAK_RUN_MODE", "quick")
        monkeypatch.setenv("HYPERLOOM_GEAK_BUDGET_MIN", "500")
        assert krh._geak_budget_minutes({"geak_budget_min": 100}) == 100.0

    def test_env_override_beats_default(self, monkeypatch):
        monkeypatch.setenv("GEAK_RUN_MODE", "quick")
        monkeypatch.setenv("HYPERLOOM_GEAK_BUDGET_MIN", "115")
        assert krh._geak_budget_minutes({}) == 115.0

    @pytest.mark.parametrize(
        "geak_run_mode, expected",
        [
            ("full", 130.0),
            ("quick", 70.0),
        ],
    )
    def test_falls_through_to_helper_when_no_overrides(
        self,
        monkeypatch,
        geak_run_mode,
        expected,
    ):
        monkeypatch.delenv("HYPERLOOM_GEAK_BUDGET_MIN", raising=False)
        monkeypatch.setenv("GEAK_RUN_MODE", geak_run_mode)
        assert krh._geak_budget_minutes({}) == expected

    def test_empty_env_value_falls_through_to_helper(self, monkeypatch):
        # Pre-fix code would let "" propagate into ``float("")`` and raise.
        monkeypatch.setenv("HYPERLOOM_GEAK_BUDGET_MIN", "")
        monkeypatch.delenv("GEAK_RUN_MODE", raising=False)
        assert krh._geak_budget_minutes({}) == 130.0


# _default_kernel_batch_parallel — adaptive batch fanout; the legacy 8
# over-admitted on smaller pods (4-GPU labs, partial-node CI shards).
class TestDefaultKernelBatchParallel:
    @pytest.fixture
    def patch_torch(self, monkeypatch):
        """Returns a setter that overrides ``torch.cuda.device_count`` and
        ``$KERNEL_AGENT_NUM_GPUS`` for the helper under test."""
        torch = _ensure_torch_module(monkeypatch)

        def _set(n_gpus, per_task=None):
            monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpus)
            if per_task is None:
                monkeypatch.delenv("KERNEL_AGENT_NUM_GPUS", raising=False)
            else:
                monkeypatch.setenv("KERNEL_AGENT_NUM_GPUS", str(per_task))

        return _set

    @pytest.mark.parametrize(
        "n_gpus, per_task, expected",
        [
            # Exact full-node match (8 GPU, 1 GPU/task) -> cap kicks in at 8.
            (8, 1, 8),
            # Partial node -> floor at the visible-GPU count.
            (4, 1, 4),
            # 8-GPU node with 4-GPU GEAK reservations -> 2 concurrent.
            (8, 4, 2),
            # 4-GPU pod with 2-GPU per task -> 2 concurrent.
            (4, 2, 2),
            # Larger-than-cap node -> cap still kicks in.
            (16, 1, 8),
            # Per-task larger than visible -> floor at 1 (don't stall the
            # batch with semaphore=0).
            (1, 4, 1),
        ],
    )
    def test_scales_with_visible_gpus(
        self,
        patch_torch,
        n_gpus,
        per_task,
        expected,
    ):
        patch_torch(n_gpus, per_task=per_task)
        assert krh._default_kernel_batch_parallel() == expected

    def test_per_task_unset_defaults_to_one(self, patch_torch):
        patch_torch(4, per_task=None)
        assert krh._default_kernel_batch_parallel() == 4

    def test_per_task_invalid_falls_back_to_one(self, patch_torch):
        patch_torch(4, per_task="not-an-int")
        assert krh._default_kernel_batch_parallel() == 4

    def test_zero_visible_gpus_returns_legacy_fallback(self, patch_torch):
        patch_torch(0)
        assert krh._default_kernel_batch_parallel() == krh._DEFAULT_KERNEL_BATCH_PARALLEL

    def test_torch_failure_returns_legacy_fallback(self, monkeypatch):
        torch = _ensure_torch_module(monkeypatch)

        def _boom():
            raise RuntimeError("driver init failed")

        monkeypatch.setattr(torch.cuda, "device_count", _boom)
        monkeypatch.delenv("KERNEL_AGENT_NUM_GPUS", raising=False)
        assert krh._default_kernel_batch_parallel() == krh._DEFAULT_KERNEL_BATCH_PARALLEL


# ---------------------------------------------------------------------------
# _should_parallelize_backends
#
# GPU-rich mode: race GEAK against the OOB ladder per kernel whenever the
# node can fit ONE kernel's GEAK + OOB ladder side-by-side
# (``visible_gpus >= 2 * per_task``). The decision is independent of
# ``num_candidates`` -- batch width is throttled separately by the batch
# handler's concurrency cap. Below ``2 * per_task`` there is no room for a
# second ladder, so keep the sequential GEAK-first / OOB-fallback ladder.
# Operators / tests can force the decision via payload or env.
# ---------------------------------------------------------------------------


class TestShouldParallelizeBackends:
    @pytest.fixture
    def patch_torch(self, monkeypatch):
        """Override ``torch.cuda.device_count`` + ``$KERNEL_AGENT_NUM_GPUS``
        and clear the env override so the GPU-aware math is exercised."""
        torch = _ensure_torch_module(monkeypatch)

        def _set(n_gpus, per_task=None):
            monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpus)
            if per_task is None:
                monkeypatch.delenv("KERNEL_AGENT_NUM_GPUS", raising=False)
            else:
                monkeypatch.setenv("KERNEL_AGENT_NUM_GPUS", str(per_task))
            monkeypatch.delenv("KERNEL_OPT_PARALLEL_BACKENDS", raising=False)

        return _set

    @pytest.mark.parametrize(
        "n_gpus, per_task, num_candidates, expected",
        [
            # 1 GPU/task: need room for both ladders -> visible_gpus >= 2.
            # The kernel count is irrelevant (batch width is capped elsewhere).
            (8, 1, 3, True),  # 8 >= 2
            (8, 1, 7, True),  # 8 >= 2 (kernel count no longer gates)
            (8, 1, 100, True),  # 8 >= 2 even when candidates >> gpus
            (2, 1, 1, True),  # 2 >= 2 boundary
            (1, 1, 1, False),  # 1 < 2 -> no room for a second ladder
            # Multi-GPU reservations: need room for TWO per_task backends.
            (8, 4, 1, True),  # 8 >= 8 boundary
            (8, 4, 5, True),  # 8 >= 8 (candidate count irrelevant)
            (8, 8, 1, False),  # 8 < 16 -> can't fit a 2nd 8-GPU backend
            (16, 8, 1, True),  # 16 >= 16
        ],
    )
    def test_gpu_aware_threshold(
        self,
        patch_torch,
        n_gpus,
        per_task,
        num_candidates,
        expected,
    ):
        patch_torch(n_gpus, per_task=per_task)
        assert krh._should_parallelize_backends({}, num_candidates) is expected

    def test_non_positive_candidates_is_false(self, patch_torch):
        patch_torch(64, per_task=1)  # plenty of GPUs
        assert krh._should_parallelize_backends({}, 0) is False
        assert krh._should_parallelize_backends({}, -1) is False

    def test_zero_visible_gpus_is_false(self, patch_torch):
        patch_torch(0, per_task=1)
        assert krh._should_parallelize_backends({}, 1) is False

    def test_torch_unknown_is_false(self, monkeypatch):
        torch = _ensure_torch_module(monkeypatch)

        def _boom():
            raise RuntimeError("driver init failed")

        monkeypatch.setattr(torch.cuda, "device_count", _boom)
        monkeypatch.delenv("KERNEL_OPT_PARALLEL_BACKENDS", raising=False)
        assert krh._should_parallelize_backends({}, 1) is False

    def test_payload_override_enables_below_threshold(self, patch_torch):
        patch_torch(1, per_task=1)  # GPU-aware math is False (1 < 2*1)
        assert (
            krh._should_parallelize_backends(
                {"parallel_backends": True},
                5,
            )
            is True
        )
        assert (
            krh._should_parallelize_backends(
                {"parallel_backends": "on"},
                5,
            )
            is True
        )

    def test_payload_override_disables_above_threshold(self, patch_torch):
        patch_torch(64, per_task=1)  # GPU-aware math would say True
        assert (
            krh._should_parallelize_backends(
                {"parallel_backends": False},
                1,
            )
            is False
        )
        assert (
            krh._should_parallelize_backends(
                {"parallel_backends": "no"},
                1,
            )
            is False
        )

    def test_env_override(self, patch_torch, monkeypatch):
        patch_torch(1, per_task=1)  # GPU-aware math is False (1 < 2*1)
        monkeypatch.setenv("KERNEL_OPT_PARALLEL_BACKENDS", "1")
        assert krh._should_parallelize_backends({}, 5) is True
        monkeypatch.setenv("KERNEL_OPT_PARALLEL_BACKENDS", "0")
        assert krh._should_parallelize_backends({}, 1) is False


# ---------------------------------------------------------------------------
# _run_optimization_batch concurrency cap (parallel-backends mode)
#
# Each parallel kernel launches TWO before_kernel_opt rocprof subprocesses
# (GEAK + OOB) *before* entering Ray, bypassing the Ray GPU lease. The batch
# caps concurrent kernels to ``visible_gpus // (2 * per_task)`` so those
# pre-Ray profilers (and the Ray tasks that follow) stay within the real GPU
# budget even when ``max_parallel`` is set higher.
# ---------------------------------------------------------------------------


class TestBatchParallelConcurrencyCap:
    def test_caps_concurrency_to_gpu_budget(self, tmp_path, monkeypatch):
        # 8 visible GPUs, 1 GPU/task -> safe concurrency = 8 // (2*1) = 4.
        monkeypatch.setattr(krh, "_visible_gpu_count", lambda: 8)
        monkeypatch.setenv("KERNEL_AGENT_NUM_GPUS", "1")  # per_task = 1
        monkeypatch.delenv("KERNEL_OPT_PARALLEL_BACKENDS", raising=False)

        state = {"in_flight": 0, "peak": 0}

        async def fake_sequence(
            base_payload,
            candidate,
            *,
            session_dir,
            parallel_backends=False,
        ):
            assert parallel_backends is True
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
            try:
                await asyncio.sleep(0.02)  # hold the slot so siblings overlap
            finally:
                state["in_flight"] -= 1
            return {
                "status": "ok",
                "kernel_id": candidate["kernel_id"],
                "source_file": candidate.get("source_file"),
                "proposal": {"decision": "REVERT"},
                "verification": {"micro_speedup": 1.0},
            }

        monkeypatch.setattr(krh, "_run_kernel_backend_sequence", fake_sequence)
        candidates = [
            {"kernel_id": f"k{i}", "source_file": f"/p/{i}.py", "reusable_native_kernel": True} for i in range(10)
        ]
        out = asyncio.run(
            krh._run_optimization_batch(
                payload={"candidates_path": "/dummy", "backend_order": "geak,claude,codex", "max_parallel": 10},
                candidates=candidates,
                session_dir=tmp_path,
            )
        )

        assert out["parallel_backends"] is True
        # max_parallel echoes the capped value (10 -> 4).
        assert out["max_parallel"] == 4
        # Cap binds: 4 kernels * 2 ladders = 8 pre-Ray profilers == 8 GPUs.
        assert state["peak"] == 4, state["peak"]

    def test_no_cap_when_gpu_count_unknown(self, tmp_path, monkeypatch):
        # torch can't report a count (None) -> cap math is skipped so the
        # operator-supplied max_parallel is preserved (matches CI / mocks).
        monkeypatch.setattr(krh, "_visible_gpu_count", lambda: None)
        monkeypatch.setenv("KERNEL_OPT_PARALLEL_BACKENDS", "1")  # force parallel

        async def fake_sequence(
            base_payload,
            candidate,
            *,
            session_dir,
            parallel_backends=False,
        ):
            return {
                "status": "ok",
                "kernel_id": candidate["kernel_id"],
                "source_file": candidate.get("source_file"),
                "proposal": {"decision": "REVERT"},
                "verification": {"micro_speedup": 1.0},
            }

        monkeypatch.setattr(krh, "_run_kernel_backend_sequence", fake_sequence)
        candidates = [
            {"kernel_id": f"k{i}", "source_file": f"/p/{i}.py", "reusable_native_kernel": True} for i in range(3)
        ]
        out = asyncio.run(
            krh._run_optimization_batch(
                payload={"candidates_path": "/dummy", "backend_order": "geak,claude,codex", "max_parallel": 7},
                candidates=candidates,
                session_dir=tmp_path,
            )
        )

        assert out["parallel_backends"] is True
        assert out["max_parallel"] == 7  # uncapped (visible GPU count unknown)


# ---------------------------------------------------------------------------
# _reconcile_kernel_id
class TestReconcileKernelId:
    CANDS = [
        {"kernel_id": "k001", "name": "aten::mm"},
        {"kernel_id": "k010", "name": "aiter::rmsnorm"},
    ]

    def test_exact_id_kept(self):
        assert krh._reconcile_kernel_id("k010", self.CANDS) == "k010"

    def test_name_match_kept(self):
        # An exact operator-name match is canonicalized to the candidate id so
        # downstream lifecycle/results are keyed by the stable k00x id.
        assert krh._reconcile_kernel_id("aten::mm", self.CANDS) == "k001"

    def test_normalized_prefix_resolves_to_real_id(self):
        assert krh._reconcile_kernel_id("kn001", self.CANDS) == "k001"
        assert krh._reconcile_kernel_id("rn010", self.CANDS) == "k010"

    def test_missing_id_falls_back_to_first(self):
        assert krh._reconcile_kernel_id("", self.CANDS) == "k001"
        assert krh._reconcile_kernel_id(None, self.CANDS) == "k001"

    def test_hallucinated_id_is_left_for_guard_or_cli_skip(self):
        # Non-empty ids are never guessed. A pure hallucination should flow to
        # the reusable-native guard / CLI skip path rather than being mapped to
        # an unrelated candidate.
        assert krh._reconcile_kernel_id("aiter.silu_and_mul", self.CANDS) == "aiter.silu_and_mul"
        assert (
            krh._reconcile_kernel_id("framework_sglang_silu_and_mul_m64", self.CANDS)
            == "framework_sglang_silu_and_mul_m64"
        )


# _resolve_candidate_id / _all_kernel_candidates — canonicalizes an aliased id
# against the full hot ∪ skipped set (no fallback) so the reusable-native guard
# rejects the real k00x rather than the raw hallucinated alias.
class TestResolveCandidateId:
    SKIPPED = [
        {"kernel_id": "k001", "name": "aten::mm", "reusable_native_kernel": False, "source_file": ""},
        {"kernel_id": "k003", "name": "aten::mm", "reusable_native_kernel": False, "source_file": ""},
        {"kernel_id": "k010", "name": "aiter::rmsnorm", "reusable_native_kernel": False, "source_file": ""},
    ]

    def test_exact_id(self):
        assert krh._resolve_candidate_id("k003", self.SKIPPED) == "k003"

    def test_kn_prefix_alias_canonicalized(self):
        assert krh._resolve_candidate_id("kn001", self.SKIPPED) == "k001"
        assert krh._resolve_candidate_id("rn010", self.SKIPPED) == "k010"

    def test_non_unique_or_nonroutable_name_not_resolved(self):
        # ``aten::mm`` is non-unique and non-routable -> cannot disambiguate,
        # so leave it untouched (returns "") rather than guess a k00x.
        assert krh._resolve_candidate_id("aten::mm", self.SKIPPED) == ""

    def test_pure_hallucination_returns_empty(self):
        assert krh._resolve_candidate_id("aiter.silu_and_mul", self.SKIPPED) == ""

    def test_empty_request_returns_empty(self):
        assert krh._resolve_candidate_id("", self.SKIPPED) == ""
        assert krh._resolve_candidate_id(None, self.SKIPPED) == ""


class TestAllKernelCandidates:
    def test_union_of_hot_and_skipped(self, tmp_path):
        cp = tmp_path / "kc.json"
        cp.write_text(
            json.dumps(
                {
                    "hot_kernels": [{"kernel_id": "k005", "name": "moe"}],
                    "skipped_kernels": [{"kernel_id": "k001", "name": "aten::mm"}],
                }
            ),
            encoding="utf-8",
        )
        out = krh._all_kernel_candidates({"candidates_path": str(cp)})
        assert {c["kernel_id"] for c in out} == {"k005", "k001"}

    def test_missing_path_returns_empty(self):
        assert krh._all_kernel_candidates({}) == []


class TestBatchKernelCandidatesRetryBudget:
    def _write_candidates(self, tmp_path: Path) -> Path:
        cp = tmp_path / "kc.json"
        cp.write_text(
            json.dumps(
                {
                    "hot_kernels": [
                        {
                            "kernel_id": "k001",
                            "gpu_pct": 12.0,
                            "reusable_native_kernel": True,
                            "source_file": "/p/moe_op.py",
                        }
                    ],
                    "reusable_native_kernel_ids": ["k001"],
                }
            ),
            encoding="utf-8",
        )
        return cp

    def test_retryable_failed_kernel_remains_batch_eligible_by_default(self, tmp_path):
        cp = self._write_candidates(tmp_path)
        state = SharedState.load_or_init(tmp_path)
        state.kernel_opt_attempts = {
            "k001": {
                "attempts": 1,
                "attempts_per_source": {"/p/moe_op.py": 1},
                "failure_count": 1,
                "last_status": "failed",
                "last_decision": "",
                "rejected_reason": "",
            }
        }
        state.save(tmp_path)

        out = krh._batch_kernel_candidates({"candidates_path": str(cp)}, session_dir=tmp_path)

        assert [item["kernel_id"] for item in out] == ["k001"]

    def test_exhausted_failed_kernel_is_not_batch_eligible_by_default(self, tmp_path):
        cp = self._write_candidates(tmp_path)
        state = SharedState.load_or_init(tmp_path)
        state.kernel_opt_attempts = {
            "k001": {
                "attempts": 2,
                "attempts_per_source": {"/p/moe_op.py": 2},
                "failure_count": 2,
                "last_status": "failed",
                "last_decision": "",
                "rejected_reason": "max_failures_2_without_keep",
            }
        }
        state.rejected_kernel_ids = ["k001"]
        state.save(tmp_path)

        out = krh._batch_kernel_candidates({"candidates_path": str(cp)}, session_dir=tmp_path)

        assert out == []

    def test_partial_kernel_stays_single_dispatch_by_default(self, tmp_path):
        cp = self._write_candidates(tmp_path)
        state = SharedState.load_or_init(tmp_path)
        state.kernel_opt_attempts = {
            "k001": {
                "attempts": 1,
                "attempts_per_source": {"/p/moe_op.py": 1},
                "partial_count": 1,
                "last_decision": "PARTIAL",
                "last_status": "ok",
                "rejected_reason": "",
            }
        }
        state.save(tmp_path)

        out = krh._batch_kernel_candidates({"candidates_path": str(cp)}, session_dir=tmp_path)

        assert out == []

    def test_retryable_failed_kernel_respects_max_failures_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES", "3")
        cp = self._write_candidates(tmp_path)
        state = SharedState.load_or_init(tmp_path)
        state.kernel_opt_attempts = {
            "k001": {
                "attempts": 2,
                "attempts_per_source": {"/p/moe_op.py": 2},
                "failure_count": 2,
                "last_status": "failed",
                "last_decision": "",
                "rejected_reason": "",
            }
        }
        state.save(tmp_path)

        out = krh._batch_kernel_candidates({"candidates_path": str(cp)}, session_dir=tmp_path)

        assert [item["kernel_id"] for item in out] == ["k001"]
