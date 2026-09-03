# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the small helpers inside ``kernel.request_handlers``."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from hyperloom.common.codex_session import (
    CODEX_SANDBOX_MODE_ENV,
    DEFAULT_CODEX_SANDBOX_MODE,
)
from hyperloom.common.env import is_truthy
from hyperloom.orchestrator.kernel import _kernel_decisions as kd
from hyperloom.orchestrator.kernel import request_handlers as krh
from hyperloom.orchestrator.roles.agent_role import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
)
from hyperloom.orchestrator.state.shared_state import SharedState


_FUSION_PROVIDER_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "CLAUDE_MODEL",
    "CODEX_MODEL",
    "FORGE_CLAUDE_MODEL",
    "FORGE_CODEX_MODEL",
    CODEX_SANDBOX_MODE_ENV,
)
_OPENAI_ONLY_ENV = {
    "OPENAI_BASE_URL": "https://gateway.invalid/Unified/v1",
    "OPENAI_API_KEY": "openai-side-key",
}
_ANTHROPIC_ONLY_ENV = {
    "ANTHROPIC_BASE_URL": "https://gateway.invalid/Anthropic",
    "ANTHROPIC_API_KEY": "anthropic-side-key",
}


def _pin_fusion_provider_env(monkeypatch, shape):
    """Set one exact provider shape without inheriting the developer's gateway."""
    for key in _FUSION_PROVIDER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in shape.items():
        monkeypatch.setenv(key, value)


#: A candidate server.log must now carry an aiter dispatch line; these tests
#: are about priority order, so every candidate gets one.
_AITER_LINE = (
    "shape is M:128, N:512, K:4096 dtype='torch.bfloat16' otype='torch.bfloat16' "
    "bias=False, scaleAB=False, bpreshuffle=False found padded_M: 128, N:512, "
    "K:4096 is tuned on cu_num = 256 in /tmp/aiter_configs/bf16_tuned_gemm.csv, "
    "libtype is opus, kernel name is opus_gemm\n"
)


class TestForgeGemmHelperCoverage:
    def test_resolve_backend_requires_exact_kernel_order_forge(self, monkeypatch):
        monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
        monkeypatch.delenv("GEMM_TUNING_BACKEND", raising=False)
        assert krh._resolve_gemm_tuning_backend({}) == "geak"
        monkeypatch.setenv("GEMM_TUNING_BACKEND", "forge")
        assert krh._resolve_gemm_tuning_backend({}) == "geak"
        assert krh._resolve_gemm_tuning_backend({"gemm_tuning_backend": "forge"}) == "geak"
        assert krh._resolve_gemm_tuning_backend({"gemm_tuning_backend": "unknown"}) == "geak"
        monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
        assert krh._resolve_gemm_tuning_backend({}) == "forge"

    def test_parse_forge_gemm_sentinel(self):
        payload = {"status": "ok", "micro_decision": "candidate"}
        text = "noise\nFORGE_GEMM_TUNE_RESULT_BEGIN\n" + json.dumps(payload) + "\nFORGE_GEMM_TUNE_RESULT_END\n"
        assert krh._parse_forge_gemm_sentinel(text) == payload
        assert krh._parse_forge_gemm_sentinel("no sentinel") is None
        assert (
            krh._parse_forge_gemm_sentinel("FORGE_GEMM_TUNE_RESULT_BEGIN\nnot-json\nFORGE_GEMM_TUNE_RESULT_END") is None
        )

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_env_value_true(self, value):
        assert is_truthy(value) is True

    @pytest.mark.parametrize("value", ["", "0", "false", "off", None])
    def test_truthy_env_value_false(self, value):
        assert is_truthy(value) is False

    def test_resolve_forge_server_log_priority(self, tmp_path):
        state = SharedState()
        baseline = tmp_path / "baseline"
        current = tmp_path / "current"
        baseline.mkdir()
        current.mkdir()
        (baseline / "server.log").write_text("baseline\n" + _AITER_LINE, encoding="utf-8")
        (current / "server.log").write_text("current\n" + _AITER_LINE, encoding="utf-8")
        state.last_baseline = {"workspace": str(baseline)}
        state.current_best = {"workspace": str(current)}

        assert krh._resolve_forge_server_log(state, tmp_path) == str(current / "server.log")

    def test_resolve_forge_server_log_bounded_runs_fallback(self, tmp_path):
        state = SharedState()
        log = tmp_path / "runs" / "explore" / "abc" / "server.log"
        log.parent.mkdir(parents=True)
        log.write_text(_AITER_LINE, encoding="utf-8")

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
        # Unreadable / missing config -> auto.
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
        # No readable config: do not force a tuner; let forge sniff the log.
        state = SharedState(precision="bf16", model_path="/models/does-not-exist")
        state.current_best = {"extra_server_args": "--quantization fp8", "extra_envs": {}}
        assert krh._resolve_forge_precision_and_quant(state, {}) == ("fp8", "auto")

    def test_forge_gemm_tune_available_probes_the_command_it_will_run(self, monkeypatch):
        # The probe must be the same invocation the tool makes, in the same
        # interpreter. Vendoring forge in-tree retires the cross-venv failure
        # this was written for, but not the other one: an importable
        # kernelforge.gemm_tune says nothing about whether `gemm-tune` is
        # registered on the CLI, and that is what the run needs.
        seen: list[list[str]] = []

        def _fake_run(cmd, **_kwargs):
            seen.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, "usage: ...", "")

        monkeypatch.setattr(krh.subprocess, "run", _fake_run)
        assert krh._forge_gemm_tune_available() is True
        assert seen == [krh._forge_gemm_tune_probe_cmd()]
        assert seen[0][0] == sys.executable

    def test_forge_gemm_tune_available_false_when_subcommand_missing(self, monkeypatch):
        # A tree whose kernelforge imports fine but never registered the
        # gemm-tune group; click exits 2 on an unknown command.
        monkeypatch.setattr(
            krh.subprocess,
            "run",
            lambda cmd, **_k: subprocess.CompletedProcess(cmd, 2, "", "Error: No such command 'gemm-tune'."),
        )
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

    @pytest.mark.parametrize(
        "exc",
        [
            OSError("no interpreter"),
            subprocess.SubprocessError("bad subprocess"),
        ],
    )
    def test_forge_gemm_tune_available_swallows_probe_failures(self, monkeypatch, exc):
        def _boom(_cmd, **_kwargs):
            raise exc

        monkeypatch.setattr(krh.subprocess, "run", _boom)
        assert krh._forge_gemm_tune_available() is False

    def test_forge_gemm_tune_available_logs_probe_timeout(self, monkeypatch, caplog):
        def _boom(_cmd, **_kwargs):
            raise subprocess.TimeoutExpired(
                cmd="probe",
                timeout=krh._FORGE_GEMM_PREFLIGHT_TIMEOUT_SEC,
            )

        monkeypatch.setattr(krh.subprocess, "run", _boom)
        with caplog.at_level("WARNING"):
            assert krh._forge_gemm_tune_available() is False

        assert "preflight timed out" in caplog.text
        assert str(krh._FORGE_GEMM_PREFLIGHT_TIMEOUT_SEC) in caplog.text

    def test_resolve_forge_precision_falls_back_to_bf16(self, monkeypatch):
        # Empty session precision + no fp8/fp4 quantization -> bf16/auto default.
        state = SharedState(precision="")
        state.current_best = {"extra_server_args": "", "extra_envs": {}}
        import hyperloom.orchestrator.kernel.roofline_ceiling as rc

        def _raise(*_a, **_k):
            raise RuntimeError("no runtime workload")

        monkeypatch.setattr(rc, "resolve_runtime_workload", _raise)
        assert krh._resolve_forge_precision_and_quant(state, {}) == ("bf16", "auto")

    def test_resolve_forge_server_log_uses_baseline_when_no_current_best(self, tmp_path):
        state = SharedState()
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "server.log").write_text("baseline\n" + _AITER_LINE, encoding="utf-8")
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
        expected = self._write_aiter_csv(tmp_path, "abc", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,1536,7168\n")
        assert krh._resolve_forge_untuned_csv(tmp_path, "fp8", "auto") == str(expected)
        assert krh._resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale") == str(expected)

    def test_resolve_forge_untuned_csv_per_token(self, tmp_path):
        expected = self._write_aiter_csv(
            tmp_path, "abc", "a8w8_untuned_gemm.csv", "M,N,K,q_dtype_w\n16,1536,7168,fp8\n"
        )
        assert krh._resolve_forge_untuned_csv(tmp_path, "fp8", "per_token") == str(expected)

    def test_resolve_forge_untuned_csv_skips_header_only(self, tmp_path):
        # Header-only / empty files are not a valid shape source.
        self._write_aiter_csv(tmp_path, "abc", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n")
        assert krh._resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale") == ""

    def test_resolve_forge_untuned_csv_picks_newest_nonempty(self, tmp_path):
        old = self._write_aiter_csv(tmp_path, "old", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n1,2,3\n")
        new = self._write_aiter_csv(tmp_path, "new", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n4,5,6\n")
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
        (model_dir / "config.json").write_text(json.dumps({"hidden_size": hidden_size}), encoding="utf-8")
        return str(model_dir)

    def test_resolve_forge_untuned_csv_rejects_model_mismatch(self, tmp_path):
        # CSV carries K=7168 shapes but the model has hidden_size=2048: reject it
        # so forge derives per-model shapes from config.json.
        self._write_aiter_csv(tmp_path, "abc", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,1536,7168\n")
        model_path = self._write_model_config(tmp_path / "model", hidden_size=2048)
        assert krh._resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale", model_path) == ""

    def test_resolve_forge_untuned_csv_accepts_model_match(self, tmp_path):
        # A CSV whose K column includes the model hidden_size is accepted.
        expected = self._write_aiter_csv(
            tmp_path,
            "abc",
            "a8w8_blockscale_untuned_gemm.csv",
            "M,N,K\n16,6144,2048\n16,2048,8192\n",
        )
        model_path = self._write_model_config(tmp_path / "model", hidden_size=2048)
        assert krh._resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale", model_path) == str(expected)

    def test_resolve_forge_untuned_csv_no_model_path_keeps_legacy(self, tmp_path):
        # Without a model_path the resolver cannot validate; returns newest non-empty CSV.
        expected = self._write_aiter_csv(tmp_path, "abc", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,1536,7168\n")
        assert krh._resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale") == str(expected)

    def test_resolve_forge_untuned_csv_unreadable_config_keeps_csv(self, tmp_path):
        # Missing/unreadable config.json: cannot validate, so keep the CSV.
        expected = self._write_aiter_csv(tmp_path, "abc", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,1536,7168\n")
        assert krh._resolve_forge_untuned_csv(tmp_path, "fp8", "blockscale", str(tmp_path / "no_such_model")) == str(
            expected
        )

    def test_csv_matches_model_helpers(self, tmp_path):
        csv_mismatch = self._write_aiter_csv(
            tmp_path, "h1", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,1536,7168\n"
        )
        csv_match = self._write_aiter_csv(tmp_path, "h2", "a8w8_blockscale_untuned_gemm.csv", "M,N,K\n16,6144,2048\n")
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
        # An inline JSON list handed in as a "path".
        inline = "[{'M': 64, 'N': 16384, 'K': 3072, 'dtype': 'bf16'}]" * 6
        assert len(inline) > 255
        assert krh._path_is_existing_file(inline) is False  # must not raise OSError(36)

    def test_path_is_existing_file_true(self, tmp_path):
        f = tmp_path / "real.csv"
        f.write_text("M,N,K\n", encoding="utf-8")
        assert krh._path_is_existing_file(str(f)) is True

    def test_normalize_forge_shapes_json_existing_path(self, tmp_path):
        f = tmp_path / "shapes.json"
        f.write_text('[{"M":1,"N":2,"K":3}]', encoding="utf-8")
        assert krh._normalize_forge_shapes_json(str(f), tmp_path) == str(f)

    def test_normalize_forge_shapes_json_inline_string(self, tmp_path):
        # A Python-repr list (single quotes).
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
        # Tokens passed as a list or its string form.
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

    def test_forge_fusion_skip_guard_ignores_warm_replay_fusion_flags(self):
        state = SharedState(
            current_best={
                "action": "warm_replay",
                "engine": "",
                "extra_envs": {
                    "SGLANG_USE_AITER": "1",
                    "ZAYA_FUSED_QK": "1",
                    "ZAYA_FUSED_RESIDUAL": "1",
                },
            },
        )

        assert krh._active_forge_fusion_env_flags(state) == {}

    def test_parse_forge_fusion_sentinel(self):
        payload = {"status": "ok", "decision": "KEEP", "kept": True}
        text = "noise\nFORGE_FUSION_RESULT_BEGIN\n" + json.dumps(payload) + "\nFORGE_FUSION_RESULT_END\n"

        assert krh._parse_forge_fusion_sentinel(text) == payload
        assert krh._parse_forge_fusion_sentinel("no marker") is None
        assert krh._parse_forge_fusion_sentinel("FORGE_FUSION_RESULT_BEGIN\nnot-json\nFORGE_FUSION_RESULT_END") is None

    def test_resolve_fusion_decode_trace_prefers_payload_and_newest(self, tmp_path):
        state = SharedState()
        state_dir = tmp_path / "state_trace"
        payload_dir = tmp_path / "payload_trace"
        state_dir.mkdir()
        payload_dir.mkdir()
        state_trace = state_dir / "old.trace.json.gz"
        payload_old = payload_dir / "old.trace.json.gz"
        payload_new = payload_dir / "new.trace.json"
        state_trace.write_text("state", encoding="utf-8")
        payload_old.write_text("old", encoding="utf-8")
        payload_new.write_text("new", encoding="utf-8")
        import os

        os.utime(payload_old, (1, 1))
        os.utime(payload_new, (10, 10))
        state.last_profile_trace = str(state_dir)

        assert krh._resolve_fusion_decode_trace(state, {"trace_path": str(payload_dir)}) == str(payload_new)
        assert krh._resolve_fusion_decode_trace(state, {}) == str(state_trace)
        assert krh._resolve_fusion_decode_trace(state, {"trace_path": "/missing"}) == str(state_trace)

    def test_forge_fusion_available_probes_the_fusion_subpackage(self, monkeypatch):
        probed: list[str] = []

        def spec(name):
            probed.append(name)
            return object()

        monkeypatch.setattr(krh.importlib.util, "find_spec", spec)
        assert krh._forge_fusion_available() is True
        # Probing kernelforge alone would pass on a KernelForge predating the
        # fusion absorption and only fail once the subprocess rejected forge-fuse.
        assert probed == ["kernelforge.fusion"]

        monkeypatch.setattr(krh.importlib.util, "find_spec", lambda _name: None)
        assert krh._forge_fusion_available() is False

    def test_forge_fusion_available_survives_an_unimportable_parent(self, monkeypatch):
        def boom(_name):
            raise ModuleNotFoundError("kernelforge")

        monkeypatch.setattr(krh.importlib.util, "find_spec", boom)
        assert krh._forge_fusion_available() is False

    def test_materialize_unified_patch_snapshot(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "model.py").write_text("old = 1\n", encoding="utf-8")
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "add", "model.py"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "init",
            ],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        patch = tmp_path / "fusion.patch"
        patch.write_text(
            "diff --git a/model.py b/model.py\n"
            "index 5626abf..f6e7663 100644\n"
            "--- a/model.py\n"
            "+++ b/model.py\n"
            "@@ -1 +1 @@\n"
            "-old = 1\n"
            "+new = 2\n",
            encoding="utf-8",
        )

        snapshot = Path(
            krh.materialize_unified_patch_snapshot(
                patch_path=patch,
                repo_root=repo,
            )
        )

        assert (snapshot / "model.py").read_text(encoding="utf-8") == "new = 2\n"

    def test_materialize_unified_patch_snapshot_nongit_repo(self, tmp_path):
        """Repro: fusion patch against a NON-git repo_root (e.g. vLLM installed
        under site-packages/dist-packages).

        forge-fusion (PR #75) emits a patch for non-git frameworks so KEPT
        fusions reach e2e integrate, but the consumer resolved base content via
        ``git show HEAD:<path>`` which fails outside a git repo -> the snapshot
        never gets the base file -> ``git apply`` fails with
        ``<path>: No such file or directory`` (observed on Qwen3-0.6B / Llama
        vLLM sessions). The base must fall back to the on-disk source.
        """
        repo = tmp_path / "site-packages"
        (repo / "vllm" / "model_executor" / "models").mkdir(parents=True)
        target = repo / "vllm" / "model_executor" / "models" / "qwen3.py"
        target.write_text("old = 1\n", encoding="utf-8")
        # NOTE: deliberately NOT a git repo (no git init) -- mirrors dist-packages.

        patch = tmp_path / "fusion.patch"
        patch.write_text(
            "diff --git a/vllm/model_executor/models/qwen3.py b/vllm/model_executor/models/qwen3.py\n"
            "--- a/vllm/model_executor/models/qwen3.py\n"
            "+++ b/vllm/model_executor/models/qwen3.py\n"
            "@@ -1 +1 @@\n"
            "-old = 1\n"
            "+new = 2\n",
            encoding="utf-8",
        )

        snapshot = Path(
            krh.materialize_unified_patch_snapshot(
                patch_path=patch,
                repo_root=repo,
            )
        )

        assert (snapshot / "vllm" / "model_executor" / "models" / "qwen3.py").read_text(encoding="utf-8") == "new = 2\n"

    def test_materialize_unified_patch_snapshot_accepts_missing_final_newline(
        self,
        tmp_path,
    ):
        """Legacy KB patches may omit the final newline but remain valid diffs."""
        repo = tmp_path / "site-packages"
        (repo / "vllm").mkdir(parents=True)
        target = repo / "vllm" / "model.py"
        target.write_text("old = 1\n", encoding="utf-8")
        patch = tmp_path / "legacy.patch"
        patch.write_text(
            "diff --git a/vllm/model.py b/vllm/model.py\n"
            "--- a/vllm/model.py\n"
            "+++ b/vllm/model.py\n"
            "@@ -1 +1 @@\n"
            "-old = 1\n"
            "+new = 2",
            encoding="utf-8",
        )

        snapshot = Path(
            krh.materialize_unified_patch_snapshot(
                patch_path=patch,
                repo_root=repo,
            )
        )

        assert (snapshot / "vllm" / "model.py").read_text(encoding="utf-8") == "new = 2\n"

    def test_materialize_unified_patch_snapshot_nongit_new_file_timestamped(self, tmp_path):
        """A created file whose ``+++`` line carries a tab-suffixed timestamp
        must still be recognized as a create on a NON-git root.

        The new-file path is normalized exactly like ``parse_patch_manifest``
        (strip the ``\\t<timestamp>`` suffix and the ``b/`` prefix); otherwise it
        is misclassified as a modify, pre-seeded from disk, and ``git apply``
        fails "already exists". Pairs a modify (base from disk) with a create in
        the same patch.
        """
        repo = tmp_path / "site-packages"
        (repo / "vllm").mkdir(parents=True)
        (repo / "vllm" / "existing.py").write_text("old = 1\n", encoding="utf-8")
        # NOTE: deliberately NOT a git repo -- mirrors dist-packages.

        patch = tmp_path / "fusion.patch"
        patch.write_text(
            "diff --git a/vllm/existing.py b/vllm/existing.py\n"
            "--- a/vllm/existing.py\n"
            "+++ b/vllm/existing.py\n"
            "@@ -1 +1 @@\n"
            "-old = 1\n"
            "+new = 2\n"
            "diff --git a/vllm/fused_new.py b/vllm/fused_new.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/vllm/fused_new.py\t2026-07-31 17:00:00.000000000 +0800\n"
            "@@ -0,0 +1 @@\n"
            "+created = 3\n",
            encoding="utf-8",
        )

        snapshot = Path(
            krh.materialize_unified_patch_snapshot(
                patch_path=patch,
                repo_root=repo,
            )
        )

        assert (snapshot / "vllm" / "existing.py").read_text(encoding="utf-8") == "new = 2\n"
        assert (snapshot / "vllm" / "fused_new.py").read_text(encoding="utf-8") == "created = 3\n"

    def test_materialize_unified_patch_snapshot_nongit_new_file_quoted(self, tmp_path):
        """A created file whose header path is C-quoted (git quotes paths with
        spaces) must be recognized as a create via the shared
        ``parse_patch_manifest`` normalization, not pre-seeded, and produced by
        ``git apply``. Regression guard for the quoted-path branch.
        """
        repo = tmp_path / "site-packages"
        (repo / "vllm").mkdir(parents=True)
        # NOTE: deliberately NOT a git repo -- mirrors dist-packages.

        patch = tmp_path / "fusion.patch"
        patch.write_text(
            'diff --git "a/vllm/fused new.py" "b/vllm/fused new.py"\n'
            "new file mode 100644\n"
            "--- /dev/null\n"
            '+++ "b/vllm/fused new.py"\n'
            "@@ -0,0 +1 @@\n"
            "+created = 1\n",
            encoding="utf-8",
        )

        snapshot = Path(
            krh.materialize_unified_patch_snapshot(
                patch_path=patch,
                repo_root=repo,
            )
        )

        assert (snapshot / "vllm" / "fused new.py").read_text(encoding="utf-8") == "created = 1\n"

    def test_materialize_unified_patch_snapshot_modify_base_missing_raises(self, tmp_path):
        """A modify whose base is neither in git HEAD nor on disk must fail with
        a precise error instead of the opaque ``git apply`` "No such file or
        directory".
        """
        repo = tmp_path / "site-packages"
        repo.mkdir(parents=True)
        # NOTE: NOT a git repo, and the target file does not exist on disk.

        patch = tmp_path / "fusion.patch"
        patch.write_text(
            "diff --git a/vllm/missing.py b/vllm/missing.py\n"
            "--- a/vllm/missing.py\n"
            "+++ b/vllm/missing.py\n"
            "@@ -1 +1 @@\n"
            "-old = 1\n"
            "+new = 2\n",
            encoding="utf-8",
        )

        with pytest.raises(FileNotFoundError, match="patch base missing"):
            krh.materialize_unified_patch_snapshot(patch_path=patch, repo_root=repo)

    def test_materialize_unified_patch_snapshot_rejects_bad_inputs(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        patch = tmp_path / "empty.patch"
        patch.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="empty patch|no file operations"):
            krh.materialize_unified_patch_snapshot(patch_path=patch, repo_root=repo)

        with pytest.raises(FileNotFoundError):
            krh.materialize_unified_patch_snapshot(
                patch_path=tmp_path / "missing.patch",
                repo_root=repo,
            )

    @pytest.mark.asyncio
    async def test_run_forge_fusion_skips_when_current_best_is_forge_fusion(self, tmp_path, monkeypatch):
        state = SharedState(
            framework="sglang",
            model_path="/models/zaya",
            current_best={
                "action": "fusion",
                "engine": "forge_fusion",
                "extra_envs": {
                    "ZAYA_FUSED_CCA_NORMALIZE_QK": "1",
                    "ZAYA_FUSED_CCA_GROUPED_QK_MEANS": "1",
                },
            },
        )
        state.save(tmp_path)

        def _should_not_check_available():
            raise AssertionError("forge-fusion availability should not be checked")

        monkeypatch.setattr(krh, "_forge_fusion_available", _should_not_check_available)
        result = await krh._run_forge_fusion({}, session_dir=tmp_path)

        assert result["status"] == "complete"
        assert result["backend"] == "forge"
        assert result["engine"] == "forge_fusion"
        assert result["micro_decision"] == "already_active"
        assert result["kept"] is False
        assert result["requires_e2e_validation"] is False
        assert result["active_env_flags"] == {
            "ZAYA_FUSED_CCA_NORMALIZE_QK": "1",
            "ZAYA_FUSED_CCA_GROUPED_QK_MEANS": "1",
        }

    @pytest.mark.parametrize(
        ("shape", "expected_backend", "expected_model"),
        [
            (
                {**_OPENAI_ONLY_ENV, "CODEX_MODEL": "gpt-fusion"},
                "codex",
                "gpt-fusion",
            ),
            (
                {**_ANTHROPIC_ONLY_ENV, "CLAUDE_MODEL": "claude-fusion"},
                "claude",
                "claude-fusion",
            ),
            (
                {
                    **_OPENAI_ONLY_ENV,
                    **_ANTHROPIC_ONLY_ENV,
                    "CODEX_MODEL": "gpt-fusion",
                    "CLAUDE_MODEL": "claude-fusion",
                },
                "claude",
                "claude-fusion",
            ),
        ],
    )
    def test_resolve_forge_agent_follows_provider_shape(
        self,
        monkeypatch,
        shape,
        expected_backend,
        expected_model,
    ):
        """Provider shape selects one matching backend and its model variable."""
        _pin_fusion_provider_env(monkeypatch, shape)

        assert krh._resolve_forge_agent({}) == (
            expected_backend,
            expected_model,
        )

    @pytest.mark.parametrize(
        ("shape", "expected"),
        [
            (_OPENAI_ONLY_ENV, ("codex", DEFAULT_CODEX_MODEL)),
            (_ANTHROPIC_ONLY_ENV, ("claude", DEFAULT_CLAUDE_MODEL)),
        ],
    )
    def test_resolve_forge_agent_uses_established_model_defaults(
        self,
        monkeypatch,
        shape,
        expected,
    ):
        _pin_fusion_provider_env(monkeypatch, shape)

        assert krh._resolve_forge_agent({}) == expected

    def test_resolve_forge_agent_honors_valid_explicit_override(self, monkeypatch):
        _pin_fusion_provider_env(
            monkeypatch,
            {**_OPENAI_ONLY_ENV, **_ANTHROPIC_ONLY_ENV},
        )

        assert krh._resolve_forge_agent(
            {
                "agent_backend": "codex",
                "llm_model": "gpt-explicit",
            }
        ) == ("codex", "gpt-explicit")

    def test_resolve_forge_agent_prefers_forge_claude_model_over_claude_model(self, monkeypatch):
        """FORGE_CLAUDE_MODEL mirrors GEAK_CLAUDE_MODEL for the Claude forge path."""
        _pin_fusion_provider_env(
            monkeypatch,
            {
                **_ANTHROPIC_ONLY_ENV,
                "CLAUDE_MODEL": "claude-orchestration",
                "FORGE_CLAUDE_MODEL": "claude-forge-only",
            },
        )

        assert krh._resolve_forge_agent({}) == (
            "claude",
            "claude-forge-only",
        )

    def test_resolve_forge_agent_prefers_forge_codex_model_over_codex_model(self, monkeypatch):
        """FORGE_CODEX_MODEL overrides CODEX_MODEL when the forge backend is Codex."""
        _pin_fusion_provider_env(
            monkeypatch,
            {
                **_OPENAI_ONLY_ENV,
                "CODEX_MODEL": "gpt-orchestration",
                "FORGE_CODEX_MODEL": "gpt-forge-only",
            },
        )

        assert krh._resolve_forge_agent({}) == (
            "codex",
            "gpt-forge-only",
        )

    def test_resolve_forge_agent_forge_model_loses_to_payload_llm_model(self, monkeypatch):
        """Request ``llm_model`` still outranks the forge-specific env knobs."""
        _pin_fusion_provider_env(
            monkeypatch,
            {
                **_ANTHROPIC_ONLY_ENV,
                "CLAUDE_MODEL": "claude-orchestration",
                "FORGE_CLAUDE_MODEL": "claude-forge-only",
            },
        )

        assert krh._resolve_forge_agent({"llm_model": "claude-payload"}) == ("claude", "claude-payload")

    def test_resolve_forge_agent_ignores_other_backend_forge_model(self, monkeypatch):
        """A Codex forge override must not leak onto the Claude forge path."""
        _pin_fusion_provider_env(
            monkeypatch,
            {
                **_ANTHROPIC_ONLY_ENV,
                "CLAUDE_MODEL": "claude-orchestration",
                "FORGE_CODEX_MODEL": "gpt-forge-only",
            },
        )

        assert krh._resolve_forge_agent({}) == (
            "claude",
            "claude-orchestration",
        )

    def test_resolve_forge_agent_rejects_invalid_explicit_backend(self, monkeypatch):
        _pin_fusion_provider_env(
            monkeypatch,
            {**_OPENAI_ONLY_ENV, **_ANTHROPIC_ONLY_ENV},
        )

        with pytest.raises(ValueError, match="agent_backend"):
            krh._resolve_forge_agent({"agent_backend": "anthropic"})

    def test_resolve_forge_agent_rejects_unconfigured_provider(self, monkeypatch):
        _pin_fusion_provider_env(monkeypatch, {})

        with pytest.raises(RuntimeError, match="no .*provider.*configured"):
            krh._resolve_forge_agent({})

    def test_resolve_forge_fusion_codex_sandbox_defaults_to_workspace_write(self, monkeypatch):
        _pin_fusion_provider_env(monkeypatch, _OPENAI_ONLY_ENV)

        assert (
            krh._resolve_forge_fusion_sandbox_mode(
                {},
                agent_backend="codex",
            )
            == DEFAULT_CODEX_SANDBOX_MODE
        )

    def test_resolve_forge_fusion_codex_sandbox_accepts_bypass(self, monkeypatch):
        _pin_fusion_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
        monkeypatch.setenv(CODEX_SANDBOX_MODE_ENV, "bypass")

        assert (
            krh._resolve_forge_fusion_sandbox_mode(
                {"agent_sandbox_mode": "bypass"},
                agent_backend="codex",
            )
            == "bypass"
        )

    def test_resolve_forge_fusion_ignores_retired_external_sandbox_env(self, monkeypatch):
        _pin_fusion_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
        monkeypatch.setenv("HYPERLOOM_CODEX_EXTERNAL_SANDBOX", "1")

        assert (
            krh._resolve_forge_fusion_sandbox_mode(
                {},
                agent_backend="codex",
            )
            == DEFAULT_CODEX_SANDBOX_MODE
        )

    def test_resolve_forge_fusion_claude_sandbox_uses_audit_default_or_override(self, monkeypatch):
        _pin_fusion_provider_env(monkeypatch, _ANTHROPIC_ONLY_ENV)
        # Claude's audit default is stable even if the Codex operator default is narrower.
        monkeypatch.setenv(CODEX_SANDBOX_MODE_ENV, "read-only")

        assert (
            krh._resolve_forge_fusion_sandbox_mode(
                {},
                agent_backend="claude",
            )
            == DEFAULT_CODEX_SANDBOX_MODE
        )
        assert (
            krh._resolve_forge_fusion_sandbox_mode(
                {"agent_sandbox_mode": "read-only"},
                agent_backend="claude",
            )
            == "read-only"
        )

    def test_resolve_forge_fusion_sandbox_rejects_invalid_mode(self, monkeypatch):
        _pin_fusion_provider_env(monkeypatch, _OPENAI_ONLY_ENV)

        with pytest.raises(RuntimeError, match="unknown Codex sandbox mode"):
            krh._resolve_forge_fusion_sandbox_mode(
                {"agent_sandbox_mode": "unconfined"},
                agent_backend="codex",
            )

    @pytest.mark.asyncio
    async def test_run_forge_fusion_openai_only_writes_codex_input(self, tmp_path, monkeypatch):
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        trace_file = trace_dir / "decode.trace.json.gz"
        trace_file.write_text("{}", encoding="utf-8")
        state = SharedState(
            framework="sglang",
            model_path="/models/zaya",
            last_profile_trace=str(trace_dir),
        )
        state.save(tmp_path)
        _pin_fusion_provider_env(
            monkeypatch,
            {**_OPENAI_ONLY_ENV, "CODEX_MODEL": "gpt-fusion"},
        )
        monkeypatch.setattr(krh, "_forge_fusion_available", lambda: True)
        monkeypatch.setattr(
            krh,
            "_kernel_agent_tool_path",
            lambda name: tmp_path / "tools" / name,
        )
        calls: list[tuple[list[str], int]] = []

        async def _fake_subprocess(cmd, *, timeout_sec):
            calls.append((cmd, timeout_sec))
            result = {
                "status": "ok",
                "decision": "KEEP",
                "kept": True,
                "env_flags": {"ZAYA_FUSED_CCA_NORMALIZE_QK": "1"},
            }
            return (
                0,
                "FORGE_FUSION_RESULT_BEGIN\n" + json.dumps(result) + "\nFORGE_FUSION_RESULT_END\n",
                "",
            )

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        result = await krh._run_forge_fusion(
            {"task_id": "fusion_task", "max_turns": 7, "timeout": 123},
            session_dir=tmp_path,
        )

        assert result["status"] == "ok"
        assert result["backend"] == "forge"
        assert result["engine"] == "forge_fusion"
        assert result["workspace"] == str(tmp_path / "runs" / "fusion" / "fusion_task")
        assert calls[0][1] == krh._forge_fusion_wrapper_timeout_sec(123)
        input_payload = json.loads(
            (tmp_path / "runs" / "fusion" / "fusion_task" / "forge_fusion_input.json").read_text(encoding="utf-8")
        )
        assert input_payload["trace_path"] == str(trace_file)
        assert input_payload["model_path"] == "/models/zaya"
        assert input_payload["max_turns"] == 7
        assert input_payload["timeout"] == 123
        assert input_payload["agent_backend"] == "codex"
        assert input_payload["llm_model"] == "gpt-fusion"
        assert input_payload["agent_sandbox_mode"] == DEFAULT_CODEX_SANDBOX_MODE

    def test_forge_fusion_timeout_invalid_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("FORGE_FUSION_TIMEOUT", "not-an-int")

        assert krh._forge_fusion_timeout_sec({}) == 7200

    def test_forge_fusion_timeout_infinite_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("FORGE_FUSION_TIMEOUT", "inf")

        assert krh._forge_fusion_timeout_sec({}) == 7200

    def test_forge_fusion_wrapper_timeout_adds_reap_grace(self):
        assert krh._forge_fusion_wrapper_timeout_sec(123) == 153

    @pytest.mark.asyncio
    async def test_run_forge_fusion_invalid_timeout_env_uses_default(self, tmp_path, monkeypatch):
        trace = tmp_path / "decode.trace.json.gz"
        trace.write_text("{}", encoding="utf-8")
        SharedState(
            framework="sglang",
            model_path="/models/zaya",
            last_profile_trace=str(trace),
        ).save(tmp_path)
        _pin_fusion_provider_env(monkeypatch, _ANTHROPIC_ONLY_ENV)
        monkeypatch.setenv("FORGE_FUSION_TIMEOUT", "not-an-int")
        monkeypatch.setattr(krh, "_forge_fusion_available", lambda: True)
        monkeypatch.setattr(krh, "_kernel_agent_tool_path", lambda name: Path(name))
        calls: list[int] = []

        async def _fake_subprocess(cmd, *, timeout_sec):
            calls.append(timeout_sec)
            result = {"status": "complete", "decision": "REVERT", "kept": False}
            return (
                0,
                "FORGE_FUSION_RESULT_BEGIN\n" + json.dumps(result) + "\nFORGE_FUSION_RESULT_END\n",
                "",
            )

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        result = await krh._run_forge_fusion({"task_id": "fusion_task"}, session_dir=tmp_path)

        assert result["status"] == "complete"
        assert calls == [krh._forge_fusion_wrapper_timeout_sec(7200)]
        input_payload = json.loads(
            (tmp_path / "runs" / "fusion" / "fusion_task" / "forge_fusion_input.json").read_text(encoding="utf-8")
        )
        assert input_payload["timeout"] == 7200

    @pytest.mark.asyncio
    async def test_run_forge_fusion_unconfigured_provider_fails_before_subprocess(
        self,
        tmp_path,
        monkeypatch,
    ):
        trace = tmp_path / "decode.trace.json.gz"
        trace.write_text("{}", encoding="utf-8")
        SharedState(
            framework="sglang",
            model_path="/models/zaya",
            last_profile_trace=str(trace),
        ).save(tmp_path)
        _pin_fusion_provider_env(monkeypatch, {})
        monkeypatch.setattr(krh, "_forge_fusion_available", lambda: True)

        async def _should_not_run(*_args, **_kwargs):
            raise AssertionError("forge-fusion must not run without a provider")

        monkeypatch.setattr(krh, "_run_subprocess", _should_not_run)

        result = await krh._run_forge_fusion({}, session_dir=tmp_path)

        assert result["status"] == "failed"
        assert result["error_class"] == "llm_provider_unconfigured"
        assert result["decision"] == "REVERT"
        assert result["kept"] is False

    @pytest.mark.asyncio
    async def test_run_forge_fusion_failure_branches(self, tmp_path, monkeypatch):
        state = SharedState(framework="sglang", model_path="/models/zaya")
        state.save(tmp_path)
        monkeypatch.setattr(krh, "_forge_fusion_available", lambda: False)
        not_found = await krh._run_forge_fusion({}, session_dir=tmp_path)
        assert not_found["error_class"] == "forge_fusion_not_found"

        monkeypatch.setattr(krh, "_forge_fusion_available", lambda: True)
        state.model_path = ""
        state.save(tmp_path)
        monkeypatch.delenv("MODEL_PATH", raising=False)
        missing_model = await krh._run_forge_fusion({}, session_dir=tmp_path)
        assert missing_model["error_class"] == "model_path_missing"

        state.model_path = "/models/zaya"
        state.last_profile_trace = ""
        state.save(tmp_path)
        missing_trace = await krh._run_forge_fusion({}, session_dir=tmp_path)
        assert missing_trace["error_class"] == "decode_trace_missing"

    @pytest.mark.asyncio
    async def test_run_forge_fusion_timeout_is_shaped(self, tmp_path, monkeypatch):
        trace = tmp_path / "trace.json.gz"
        trace.write_text("{}", encoding="utf-8")
        SharedState(
            framework="sglang",
            model_path="/models/zaya",
            last_profile_trace=str(trace),
        ).save(tmp_path)
        _pin_fusion_provider_env(monkeypatch, _ANTHROPIC_ONLY_ENV)
        monkeypatch.setattr(krh, "_forge_fusion_available", lambda: True)
        monkeypatch.setattr(krh, "_kernel_agent_tool_path", lambda name: Path(name))

        async def _timeout(cmd, *, timeout_sec):
            raise subprocess.TimeoutExpired(cmd, timeout_sec)

        monkeypatch.setattr(krh, "_run_subprocess", _timeout)

        result = await krh._run_forge_fusion({"timeout": 60}, session_dir=tmp_path)

        assert result["status"] == "failed"
        assert result["error_class"] == "subprocess_timeout"
        assert result["backend"] == "forge"
        assert result["kept"] is False

    @pytest.mark.asyncio
    async def test_run_forge_fusion_passes_session_serve_args(self, tmp_path, monkeypatch):
        model = tmp_path / "MiniMax-M3-MXFP4"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"sparse_attention_config": {"sparse_block_size": 128}}),
            encoding="utf-8",
        )
        trace = tmp_path / "trace.json.gz"
        trace.write_text("{}", encoding="utf-8")
        SharedState(
            framework="vllm",
            model_path=str(model),
            last_profile_trace=str(trace),
            tp=8,
            max_model_len=13312,
        ).save(tmp_path)
        _pin_fusion_provider_env(monkeypatch, _ANTHROPIC_ONLY_ENV)
        monkeypatch.setattr(krh, "_forge_fusion_available", lambda: True)
        monkeypatch.setattr(krh, "_kernel_agent_tool_path", Path)

        async def _fake_subprocess(cmd, *, timeout_sec):
            result = {"status": "complete", "decision": "REVERT", "kept": False}
            return (
                0,
                "FORGE_FUSION_RESULT_BEGIN\n" + json.dumps(result) + "\nFORGE_FUSION_RESULT_END\n",
                "",
            )

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        await krh._run_forge_fusion({"task_id": "fusion_task"}, session_dir=tmp_path)
        input_payload = json.loads(
            (tmp_path / "runs" / "fusion" / "fusion_task" / "forge_fusion_input.json").read_text(encoding="utf-8")
        )
        assert input_payload["tp"] == 8
        assert input_payload["block_size"] == 128
        assert input_payload["max_model_len"] == 13312

    @pytest.mark.asyncio
    async def test_run_forge_fusion_timeout_salvages_keep(self, tmp_path, monkeypatch):
        trace = tmp_path / "trace.json.gz"
        trace.write_text("{}", encoding="utf-8")
        SharedState(
            framework="sglang",
            model_path="/models/zaya",
            last_profile_trace=str(trace),
        ).save(tmp_path)
        _pin_fusion_provider_env(monkeypatch, _ANTHROPIC_ONLY_ENV)
        monkeypatch.setattr(krh, "_forge_fusion_available", lambda: True)
        monkeypatch.setattr(krh, "_kernel_agent_tool_path", Path)

        async def _timeout(cmd, *, timeout_sec):
            workspace = tmp_path / "runs" / "fusion" / "kernel_entry_fusion"
            (workspace / "kernel_keep_checkpoint.json").write_text(
                json.dumps(
                    {
                        "kept": True,
                        "kernel_speedup": 2.69,
                        "env_flag": "QWEN_FUSED",
                        "source_file": "/fw/model.py",
                        "repo_root": "/fw",
                    }
                ),
                encoding="utf-8",
            )
            (workspace / "fusion.patch").write_text(
                "diff --git a/model.py b/model.py\n",
                encoding="utf-8",
            )
            raise subprocess.TimeoutExpired(cmd, timeout_sec)

        monkeypatch.setattr(krh, "_run_subprocess", _timeout)

        result = await krh._run_forge_fusion({"timeout": 60}, session_dir=tmp_path)

        assert result["kept"] is True
        assert result["decision"] == "KEEP"
        assert result["requires_e2e_validation"] is True
        assert result["salvaged"] is True
        assert result["error_class"] == "subprocess_timeout"
        assert result["kernel_speedup"] == 2.69

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
    async def test_run_forge_gemm_tuning_resolves_hf_repo_for_subprocess(
        self,
        tmp_path,
        monkeypatch,
    ):
        from hyperloom.inference_optimizer import model_config_utils

        snapshot = tmp_path / "models--amd--DeepSeek-V4-Pro-MXFP4" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v3"}),
            encoding="utf-8",
        )
        state = SharedState(
            precision="mxfp4",
            framework="vllm",
            model_path="amd/DeepSeek-V4-Pro-MXFP4",
            gpu_type="mi355x",
            tp=8,
            conc=64,
        )
        state.save(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(
            model_config_utils,
            "resolve_local_model_dir",
            lambda _model_path: snapshot,
        )
        monkeypatch.setattr(krh, "_resolve_aiter_root_for_forge", lambda: "/opt/aiter")
        durable = {}

        def _capture_durable_envs(extra_envs, *, model_path, session_dir):
            durable["model_path"] = model_path
            return extra_envs, ""

        monkeypatch.setattr(
            krh,
            "_persist_forge_gemm_csv_durably",
            _capture_durable_envs,
        )

        subprocess_cmd = []

        async def _fake_subprocess(_cmd, *, timeout_sec):
            subprocess_cmd.extend(_cmd)
            sentinel = (
                "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
                + json.dumps(
                    {
                        "status": "ok",
                        "micro_decision": "candidate",
                        "recommended_env": {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "/tmp/tuned.csv"},
                    }
                )
                + "\nFORGE_GEMM_TUNE_RESULT_END\n"
            )
            return 0, sentinel, ""

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)
        payload = {"task_id": "deepseek-gemm"}

        result = await krh._run_forge_gemm_tuning(payload, session_dir=tmp_path)

        workspace = krh._gemm_tuning_workspace(payload, session_dir=tmp_path)
        written = json.loads((workspace / "forge_gemm_tuning_input.json").read_text(encoding="utf-8"))
        assert written["model_path"] == str(snapshot)
        assert subprocess_cmd[:2] == ["env", "AITER_ROOT_DIR=/opt/aiter"]
        assert subprocess_cmd[2] == sys.executable
        assert result["model_path"] == "amd/DeepSeek-V4-Pro-MXFP4"
        assert durable["model_path"] == "amd/DeepSeek-V4-Pro-MXFP4"

    @pytest.mark.asyncio
    async def test_run_forge_gemm_tuning_rejects_uncached_hf_repo(
        self,
        tmp_path,
        monkeypatch,
    ):
        from hyperloom.inference_optimizer import model_config_utils

        SharedState(
            precision="mxfp4",
            framework="vllm",
            model_path="amd/Unavailable-Model",
            gpu_type="mi355x",
        ).save(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(
            model_config_utils,
            "resolve_local_model_dir",
            lambda _model_path: None,
        )
        subprocess_called = False

        async def _unexpected_subprocess(_cmd, *, timeout_sec):
            nonlocal subprocess_called
            subprocess_called = True
            return 1, "", ""

        monkeypatch.setattr(krh, "_run_subprocess", _unexpected_subprocess)

        result = await krh._run_forge_gemm_tuning({}, session_dir=tmp_path)

        # Skipped, not failed: forge never started, so it has no verdict.
        assert result["status"] == "skipped"
        assert result["error_class"] == "model_path_unavailable"
        assert result["skip_reason"]
        assert subprocess_called is False

    @pytest.mark.asyncio
    async def test_run_forge_gemm_tuning_rejects_missing_absolute_model_path(
        self,
        tmp_path,
        monkeypatch,
    ):
        missing_model_dir = tmp_path / "missing-model"
        SharedState(
            precision="mxfp4",
            framework="vllm",
            model_path=str(missing_model_dir),
            gpu_type="mi355x",
        ).save(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        subprocess_called = False

        async def _unexpected_subprocess(_cmd, *, timeout_sec):
            nonlocal subprocess_called
            subprocess_called = True
            return 1, "", ""

        monkeypatch.setattr(krh, "_run_subprocess", _unexpected_subprocess)

        result = await krh._run_forge_gemm_tuning({}, session_dir=tmp_path)

        assert missing_model_dir.is_absolute()
        assert result["status"] == "skipped"
        assert result["error_class"] == "model_path_unavailable"
        assert subprocess_called is False

    @pytest.mark.asyncio
    async def test_run_forge_gemm_tuning_maps_failed_micro_decision(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "qwen"
        model_dir.mkdir()
        state = SharedState(
            precision="bf16",
            framework="sglang",
            model_path=str(model_dir),
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

    # ---- forge wording -> coordinator decision ------------------------------
    # forge reports seven micro_decision wordings; the bridge handled four, and
    # the three it missed read in the breakdown like a genuine no_improvement.

    @pytest.mark.asyncio
    async def test_a_partial_wording_is_reverted_and_named(self, tmp_path, monkeypatch):
        """A barren ``partial_failure`` is a REVERT that still names itself."""
        self._moe_state(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        sentinel = (
            "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
            + json.dumps({"status": "ok", "micro_decision": "partial_failure"})
            + "\nFORGE_GEMM_TUNE_RESULT_END\n"
        )

        async def _fake_subprocess(cmd, *, timeout_sec):
            return 0, sentinel, ""

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        result = await krh._run_forge_gemm_tuning({}, session_dir=tmp_path)

        assert result["decision"] == "REVERT"
        assert result["error_class"] == "forge_partial_failure"

    @pytest.mark.asyncio
    async def test_an_empty_run_is_reverted_and_says_so(self, tmp_path, monkeypatch):
        """Writing zero rows is not the same outcome as finding nothing."""
        self._moe_state(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        sentinel = (
            "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
            + json.dumps({"status": "ok", "micro_decision": "empty_output"})
            + "\nFORGE_GEMM_TUNE_RESULT_END\n"
        )

        async def _fake_subprocess(cmd, *, timeout_sec):
            return 0, sentinel, ""

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        result = await krh._run_forge_gemm_tuning({}, session_dir=tmp_path)

        assert result["decision"] == "REVERT"
        assert result["error_class"], "an empty run must name itself"
        assert "empty" in result["error_class"]

    @pytest.mark.asyncio
    async def test_a_genuine_no_improvement_stays_unadorned(self, tmp_path, monkeypatch):
        """The distinction only works if the ordinary case stays ordinary."""
        self._moe_state(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        sentinel = (
            "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
            + json.dumps({"status": "ok", "micro_decision": "no_improvement"})
            + "\nFORGE_GEMM_TUNE_RESULT_END\n"
        )

        async def _fake_subprocess(cmd, *, timeout_sec):
            return 0, sentinel, ""

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        result = await krh._run_forge_gemm_tuning({}, session_dir=tmp_path)

        assert result["decision"] == "REVERT"
        assert not result.get("error_class")

    @pytest.mark.asyncio
    async def test_a_tuner_error_class_reaches_the_envelope(self, tmp_path, monkeypatch):
        """A crash a tuner named must be visible where the breakdown reads."""
        self._moe_state(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        sentinel = (
            "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
            + json.dumps(
                {
                    "status": "failed",
                    "micro_decision": "failed",
                    "tuners_run": [
                        {
                            "tuner": "fmoe_ck",
                            "status": "failed",
                            "error_class": "codegen_unsupported_dtype",
                            "error": "Unsupported data type combination",
                        }
                    ],
                }
            )
            + "\nFORGE_GEMM_TUNE_RESULT_END\n"
        )

        async def _fake_subprocess(cmd, *, timeout_sec):
            return 1, sentinel, ""

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        result = await krh._run_forge_gemm_tuning({}, session_dir=tmp_path)

        assert result["error_class"] == "codegen_unsupported_dtype"
        assert "Unsupported data type" in str(result.get("error") or "")

    @pytest.mark.asyncio
    async def test_a_candidate_keeps_both_the_env_and_a_sibling_crash(self, tmp_path, monkeypatch):
        """One tuner crashed, another delivered: forge reports ``candidate``, so
        the env is measured and the crash is still named. Promotability keys on
        ``status``, so a named crash must not demote the run."""
        self._moe_state(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(krh, "_persist_forge_gemm_csv_durably", lambda envs, **_kw: (dict(envs), ""))
        sentinel = (
            "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
            + json.dumps(
                {
                    "status": "ok",
                    "micro_decision": "candidate",
                    "recommended_env": {"AITER_CONFIG_FMOE": "/ws/tuned_fmoe.csv"},
                    "tuners_run": [
                        {"tuner": "a8w8", "status": "failed", "error_class": "codegen_crash"},
                        {"tuner": "fmoe_ck", "status": "ok"},
                    ],
                }
            )
            + "\nFORGE_GEMM_TUNE_RESULT_END\n"
        )

        async def _fake_subprocess(cmd, *, timeout_sec):
            return 0, sentinel, ""

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        result = await krh._run_forge_gemm_tuning({}, session_dir=tmp_path)

        assert result["decision"] == "KEEP"
        assert result["requires_e2e_validation"] is True
        assert result["error_class"] == "codegen_crash"
        # status decides promotability; a named crash must not demote the run.
        assert result["status"] != "failed"

    @pytest.mark.asyncio
    async def test_a_malformed_tuners_run_does_not_break_the_run(self, tmp_path, monkeypatch):
        """``tuners_run`` is forge's JSON and may be any shape; lifting a reason
        out of it must not turn a run that happened into a reported crash."""
        self._moe_state(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)

        for malformed in (5, "not-a-list", {"tuner": "fmoe_ck"}, [None, 7]):
            sentinel = (
                "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
                + json.dumps(
                    {
                        "status": "ok",
                        "micro_decision": "no_improvement",
                        "tuners_run": malformed,
                    }
                )
                + "\nFORGE_GEMM_TUNE_RESULT_END\n"
            )

            async def _fake_subprocess(cmd, *, timeout_sec, _s=sentinel):
                return 0, _s, ""

            monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

            result = await krh._run_forge_gemm_tuning(
                {"task_id": f"malformed-{type(malformed).__name__}"},
                session_dir=tmp_path,
            )

            # The verdict still lands, and no exception class leaks in as a cause.
            assert result["decision"] == "REVERT", malformed
            assert result.get("error_class") != "TypeError", malformed

    @pytest.mark.asyncio
    async def test_an_absent_micro_decision_is_left_alone(self, tmp_path, monkeypatch):
        """No wording at all is not a verdict; the bridge must not invent one."""
        self._moe_state(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        sentinel = "FORGE_GEMM_TUNE_RESULT_BEGIN\n" + json.dumps({"status": "ok"}) + "\nFORGE_GEMM_TUNE_RESULT_END\n"

        async def _fake_subprocess(cmd, *, timeout_sec):
            return 0, sentinel, ""

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        result = await krh._run_forge_gemm_tuning({}, session_dir=tmp_path)

        assert "decision" not in result

    # ---- MoE runtime key: log -> CSV -> payload -> forge argv ---------------
    # Both ends were covered (the CSV writer, and KernelForge's preference for a
    # caller-supplied CSV); the handoff between them was not, and deleting any
    # link in it left the suite green.

    #: A real dispatch line, gfx field included. Fixtures that dropped the gfx
    #: field once let a regex that could never match production pass its tests.
    _REAL_MOE_DISPATCH = (
        "(Worker_TP0 pid=1) [aiter] [fused_moe] using 2stage default for "
        "('gfx950', 256, 256, 4096, 512, 256, 6, 'ActivationType.Silu', "
        "'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float4_e2m1fn_x2', "
        "'QuantType.per_1x32', True, False)"
    )

    @staticmethod
    def _moe_state(tmp_path):
        model_dir = tmp_path / "moe-model"
        model_dir.mkdir(exist_ok=True)
        SharedState(
            precision="fp8",
            framework="sglang",
            model_path=str(model_dir),
            gpu_type="mi355x",
            tp=1,
            conc=64,
        ).save(tmp_path)
        return model_dir

    @staticmethod
    def _sentinel() -> str:
        return (
            "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
            + json.dumps({"status": "ok", "micro_decision": "skipped"})
            + "\nFORGE_GEMM_TUNE_RESULT_END\n"
        )

    @pytest.mark.asyncio
    async def test_moe_key_travels_from_the_log_into_the_forge_payload(self, tmp_path, monkeypatch):
        """The values must come from the log, not from the config: a
        config-derived key is what aiter would never look up."""
        self._moe_state(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        log = tmp_path / "server.log"
        log.write_text(self._REAL_MOE_DISPATCH + "\n", encoding="utf-8")

        argv: dict[str, list[str]] = {}

        async def _fake_subprocess(cmd, *, timeout_sec):
            argv["cmd"] = list(cmd)
            return 0, self._sentinel(), ""

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)
        payload = {"task_id": "moe-key", "kernel_signature_log": str(log)}

        await krh._run_forge_gemm_tuning(payload, session_dir=tmp_path)

        workspace = krh._gemm_tuning_workspace(payload, session_dir=tmp_path)
        written = json.loads((workspace / "forge_gemm_tuning_input.json").read_text(encoding="utf-8"))
        csv_path = Path(written["moe_untuned_csv"])
        assert csv_path.is_file(), "the payload must name a CSV that exists"

        rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
        header = rows[0].split(",")
        values = dict(zip(header, rows[1].split(",")))
        # Straight off the log line, not inferred from the model config.
        assert values["inter_dim"] == "512"
        assert values["model_dim"] == "4096"
        assert values["expert"] == "256"
        assert values["topk"] == "6"
        assert values["q_dtype_a"] == "torch.float8_e4m3fn"
        assert values["q_dtype_w"] == "torch.float4_e2m1fn_x2"
        assert values["q_type"] == "QuantType.per_1x32"

    def test_the_token_column_comes_from_the_workload(self, tmp_path):
        """``tokens`` arrives as forge's comma-separated string, not a list.

        Every prior case passed a list, so the tests agreed with the annotation
        instead of with the only production caller.
        """
        log = tmp_path / "server.log"
        log.write_text(self._REAL_MOE_DISPATCH + "\n", encoding="utf-8")

        for tokens, expected in (
            ("1,32,64", ["1", "32", "64"]),
            ("64", ["64"]),
            ([1, 32, 64], ["1", "32", "64"]),
            ("", ["1"]),
            ("  16 , 16 ,bad,-8, 0 ", ["16"]),
        ):
            csv_path, _report = krh._write_fmoe_untuned_csv_from_log(
                str(log), tokens, tmp_path / f"ws_{str(tokens)[:12].strip()}"
            )
            assert csv_path, f"no CSV for tokens={tokens!r}"
            rows = Path(csv_path).read_text(encoding="utf-8").strip().splitlines()
            got = [r.split(",")[0] for r in rows[1:]]
            assert got == expected, f"tokens={tokens!r} -> {got}"

    @pytest.mark.asyncio
    async def test_the_moe_csv_reaches_the_forge_argv(self, tmp_path, monkeypatch):
        """Deriving the CSV is useless if the option never reaches forge."""
        self._moe_state(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        log = tmp_path / "server.log"
        log.write_text(self._REAL_MOE_DISPATCH + "\n", encoding="utf-8")

        argv: dict[str, list[str]] = {}

        async def _fake_subprocess(cmd, *, timeout_sec):
            argv["cmd"] = list(cmd)
            return 0, self._sentinel(), ""

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)
        payload = {"task_id": "moe-argv", "kernel_signature_log": str(log)}

        await krh._run_forge_gemm_tuning(payload, session_dir=tmp_path)

        # The handler hands the tool an input JSON; the tool builds forge's argv
        # from it. Assert the field the tool reads is the CSV that was derived.
        workspace = krh._gemm_tuning_workspace(payload, session_dir=tmp_path)
        written = json.loads((workspace / "forge_gemm_tuning_input.json").read_text(encoding="utf-8"))
        assert written["moe_untuned_csv"].endswith("untuned_fmoe_from_runtime.csv")
        assert str(workspace) in written["moe_untuned_csv"]

    @pytest.mark.asyncio
    async def test_a_caller_supplied_moe_csv_wins(self, tmp_path, monkeypatch):
        self._moe_state(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        log = tmp_path / "server.log"
        log.write_text(self._REAL_MOE_DISPATCH + "\n", encoding="utf-8")
        supplied = tmp_path / "operator_moe.csv"
        supplied.write_text(krh._FMOE_UNTUNED_CSV_HEADER + "\n", encoding="utf-8")

        async def _fake_subprocess(cmd, *, timeout_sec):
            return 0, self._sentinel(), ""

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)
        payload = {
            "task_id": "moe-supplied",
            "kernel_signature_log": str(log),
            "moe_untuned_csv": str(supplied),
        }

        await krh._run_forge_gemm_tuning(payload, session_dir=tmp_path)

        workspace = krh._gemm_tuning_workspace(payload, session_dir=tmp_path)
        written = json.loads((workspace / "forge_gemm_tuning_input.json").read_text(encoding="utf-8"))
        assert written["moe_untuned_csv"] == str(supplied)

    @pytest.mark.asyncio
    async def test_a_stale_moe_csv_path_falls_back_to_the_log(self, tmp_path, monkeypatch):
        """A path that no longer exists must not be forwarded to forge."""
        self._moe_state(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        log = tmp_path / "server.log"
        log.write_text(self._REAL_MOE_DISPATCH + "\n", encoding="utf-8")

        async def _fake_subprocess(cmd, *, timeout_sec):
            return 0, self._sentinel(), ""

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)
        payload = {
            "task_id": "moe-stale",
            "kernel_signature_log": str(log),
            "moe_untuned_csv": str(tmp_path / "gone.csv"),
        }

        await krh._run_forge_gemm_tuning(payload, session_dir=tmp_path)

        workspace = krh._gemm_tuning_workspace(payload, session_dir=tmp_path)
        written = json.loads((workspace / "forge_gemm_tuning_input.json").read_text(encoding="utf-8"))
        assert written["moe_untuned_csv"] != str(tmp_path / "gone.csv")
        assert Path(written["moe_untuned_csv"]).is_file()

    @pytest.mark.asyncio
    async def test_vllm_block_fp8_prefers_traced_shapes_over_profile_capture(self, tmp_path, monkeypatch):
        """vLLM block-FP8 must tune the device-side traced shapes.

        Decode steps replay inside a CUDA Graph, so they emit no Kineto op
        events and the block-FP8 profile capture can only ever report prefill M
        (it reported M=2095 alone on the session that then lost 18.45% E2E).
        The TraceLens candidates carry the decode M, so they win and the capture
        pass is not needed at all.
        """
        candidates = tmp_path / "kernel_candidates.json"
        candidates.write_text(
            json.dumps(
                {
                    "hot_kernels": [
                        {
                            "name": "aiter::gemm_a8w8_blockscale_ck",
                            "input_shapes": [
                                {"call_num": 1440, "shape": "(64,3072) fp8"},
                                {"call_num": 1440, "shape": "(10240,3072) fp8"},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        model_dir = tmp_path / "qwen"
        model_dir.mkdir()
        state = SharedState(
            precision="fp8",
            framework="vllm",
            model_path=str(model_dir),
            gpu_type="mi355x",
            tp=1,
            conc=64,
            last_trace_analyze={"candidates_path": str(candidates)},
        )
        state.save(tmp_path)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(krh, "_profile_shapes_are_fresh", lambda _state: True)

        captured = {"ran": False}

        async def _record_capture(**_kwargs):
            captured["ran"] = True
            return {"status": "failed"}

        monkeypatch.setattr(krh, "_capture_vllm_tunableop_shapes", _record_capture)

        async def _fake_subprocess(cmd, *, timeout_sec):
            return 1, "", ""

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        # task_id keeps the workspace deterministic (it otherwise falls back to
        # a wall-clock suffix, which the assertion below could not re-derive).
        payload = {"precision": "fp8", "quant_type": "blockscale", "task_id": "gemm-1"}
        await krh._run_forge_gemm_tuning(payload, session_dir=tmp_path)

        assert captured["ran"] is False, "profile capture ran despite traced shapes"
        workspace = krh._gemm_tuning_workspace(payload, session_dir=tmp_path)
        written = json.loads((workspace / "forge_gemm_tuning_input.json").read_text(encoding="utf-8"))
        shapes = json.loads(Path(written["shapes_json"]).read_text(encoding="utf-8"))
        # Traced shapes are re-keyed onto aiter's power-of-two lookup ladder.
        assert {(row["N"], row["K"]) for row in shapes} == {(10240, 3072)}
        assert {row["M"] for row in shapes} == {16, 32, 64}

    @pytest.mark.asyncio
    async def test_run_forge_gemm_tuning_tags_engine_forge(self, tmp_path, monkeypatch):
        """Forge runs must carry ``engine='forge'`` so the breakdown attributes them correctly."""
        model_dir = tmp_path / "qwen"
        model_dir.mkdir()
        state = SharedState(
            precision="bf16",
            framework="sglang",
            model_path=str(model_dir),
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

    @pytest.mark.parametrize(
        "quant_type",
        [
            "blockscale_bpreshuffle",
            "a8w8_blockscale_bpreshuffle",
            "blockscale+bpreshuffle",
        ],
    )
    def test_resolve_forge_untuned_csv_blockscale_bpreshuffle(
        self,
        tmp_path,
        quant_type,
    ):
        expected = self._write_aiter_csv(
            tmp_path,
            "abc",
            "a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
            "M,N,K\n16,1536,7168\n",
        )
        assert krh._resolve_forge_untuned_csv(
            tmp_path,
            "fp8",
            quant_type,
        ) == str(expected)

    def test_resolve_forge_untuned_csv_rejects_unknown_fp8_quant(self, tmp_path):
        self._write_aiter_csv(
            tmp_path,
            "abc",
            "a8w8_blockscale_untuned_gemm.csv",
            "M,N,K\n16,1536,7168\n",
        )

        assert (
            krh._resolve_forge_untuned_csv(
                tmp_path,
                "fp8",
                "misspelled_quant_type",
            )
            == ""
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "source",
        ["fresh_profile", "explicit_benchmark", "stale_profile"],
    )
    async def test_sglang_shape_source_priority(
        self,
        tmp_path,
        monkeypatch,
        source,
    ):
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "hidden_size": 5120,
                    "intermediate_size": 17408,
                    "quantization_config": {"weight_block_size": [128, 128]},
                }
            )
        )
        profile_shapes = tmp_path / "profile_shapes.json"
        profile_shapes.write_text(json.dumps([{"M": 1024, "N": 34816, "K": 5120}]))
        specialist_csv = tmp_path / "a8w8_blockscale_untuned_gemm.csv"
        specialist_csv.write_text("M,N,K\n4096,34816,5120\n")
        state = SharedState(
            precision="fp8",
            framework="sglang",
            model_path=str(model),
            gpu_type="mi300x",
            tp=1,
            conc=64,
            last_profile_status="succeeded",
            last_trace_analyze={"shapes_json": str(profile_shapes)},
        )
        state.last_profile_workload = state.profile_workload_context()
        if source == "stale_profile":
            state.last_profile_workload["conc"] = 32
        state.save(tmp_path)
        captured: dict = {}
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(
            krh,
            "_resolve_forge_untuned_csv",
            lambda *args, **kwargs: str(specialist_csv),
        )

        async def _fake_subprocess(cmd, *, timeout_sec):
            input_path = Path(cmd[cmd.index("--input-json") + 1])
            captured.update(json.loads(input_path.read_text()))
            return (
                0,
                "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
                + json.dumps({"status": "ok", "micro_decision": "skipped"})
                + "\nFORGE_GEMM_TUNE_RESULT_END\n",
                "",
            )

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        payload = {"untuned_csv": str(specialist_csv)} if source == "explicit_benchmark" else {}
        await krh._run_forge_gemm_tuning(payload, session_dir=tmp_path)

        if source in {"explicit_benchmark", "stale_profile"}:
            assert captured["untuned_csv"] == str(specialist_csv)
            assert captured["shapes_json"] == ""
        else:
            assert captured["untuned_csv"] == ""
            # The fresh profile wins over the specialist CSV; the path handed to
            # forge is that profile re-keyed onto aiter's lookup ladder.
            assert captured["shapes_json"].endswith("forge_shapes.aiter_aligned.json")
            assert json.loads(Path(captured["shapes_json"]).read_text(encoding="utf-8"))


def _collective_candidate(**overrides):
    """Build a valid all-reduce candidate with optional field overrides."""
    candidate = {
        "kernel_id": "all-reduce-1",
        "name": "all_reduce_kernel",
        "source_file": "/workspace/kernels/all_reduce.py",
        "source_function": "all_reduce",
        "input_shapes": [[4, 1024]],
        "input_dtypes": ["float16"],
        "gpu_pct": 12.5,
        "kernel_repo": "/workspace",
        "reusable_native_kernel": True,
        "candidate_source": "nccl_summary",
        "kernel_contract": {
            "kind": "collective",
            "collective_op": "all_reduce",
        },
    }
    candidate.update(overrides)
    return candidate


class TestForgeCollectiveCoverage:
    def test_parse_forge_collective_sentinel(self):
        """Parse a valid collective wrapper result."""
        payload = {
            "engine": "forge_collective",
            "status": "complete",
            "decision": "KEEP",
            "kept": True,
            "requires_e2e_validation": True,
        }
        stdout = (
            "wrapper noise\nFORGE_COLLECTIVE_RESULT_BEGIN\n"
            + json.dumps(payload)
            + "\nFORGE_COLLECTIVE_RESULT_END\ntrailing noise"
        )

        assert krh._parse_forge_collective_sentinel(stdout) == payload

    @pytest.mark.parametrize(
        "payload, error",
        [
            pytest.param("no-sentinel", "no result sentinel", id="missing-sentinel"),
            pytest.param("malformed", "malformed result JSON", id="malformed-json"),
            pytest.param([], "must be a JSON object", id="non-object"),
            pytest.param(
                {
                    "engine": "other",
                    "status": "complete",
                    "decision": "REVERT",
                    "kept": False,
                    "requires_e2e_validation": False,
                },
                "invalid engine",
                id="engine",
            ),
            pytest.param(
                {
                    "engine": "forge_collective",
                    "status": 1,
                    "decision": "REVERT",
                    "kept": False,
                    "requires_e2e_validation": False,
                },
                "invalid status",
                id="status-type",
            ),
            pytest.param(
                {
                    "engine": "forge_collective",
                    "status": "",
                    "decision": "REVERT",
                    "kept": False,
                    "requires_e2e_validation": False,
                },
                "invalid status",
                id="status-empty",
            ),
            pytest.param(
                {
                    "engine": "forge_collective",
                    "status": "complete",
                    "decision": "SKIP",
                    "kept": False,
                    "requires_e2e_validation": False,
                },
                "invalid decision",
                id="decision",
            ),
            pytest.param(
                {
                    "engine": "forge_collective",
                    "status": "complete",
                    "decision": "KEEP",
                    "kept": 1,
                    "requires_e2e_validation": True,
                },
                "invalid kept",
                id="kept",
            ),
            pytest.param(
                {
                    "engine": "forge_collective",
                    "status": "complete",
                    "decision": "REVERT",
                    "kept": False,
                    "requires_e2e_validation": None,
                },
                "invalid requires_e2e_validation",
                id="e2e-type",
            ),
            pytest.param(
                {
                    "engine": "forge_collective",
                    "status": "complete",
                    "decision": "KEEP",
                    "kept": False,
                    "requires_e2e_validation": False,
                },
                "inconsistent decision",
                id="decision-consistency",
            ),
            pytest.param(
                {
                    "engine": "forge_collective",
                    "status": "complete",
                    "decision": "KEEP",
                    "kept": True,
                    "requires_e2e_validation": False,
                },
                "inconsistent E2E gate",
                id="e2e-consistency",
            ),
        ],
    )
    def test_parse_forge_collective_sentinel_rejects_contract_errors(
        self,
        payload,
        error,
    ):
        """Reject every malformed collective sentinel contract."""
        if payload == "no-sentinel":
            stdout = "wrapper emitted no markers"
        elif payload == "malformed":
            stdout = "FORGE_COLLECTIVE_RESULT_BEGIN\nnot-json\nFORGE_COLLECTIVE_RESULT_END"
        else:
            stdout = "FORGE_COLLECTIVE_RESULT_BEGIN\n" + json.dumps(payload) + "\nFORGE_COLLECTIVE_RESULT_END"

        with pytest.raises(ValueError, match=error):
            krh._parse_forge_collective_sentinel(stdout)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("candidate", [[], {}], ids=["list", "empty"])
    async def test_run_forge_collective_rejects_invalid_candidate(
        self,
        tmp_path,
        candidate,
    ):
        """Reject explicitly supplied empty or non-object candidates."""
        result = await krh._run_forge_collective(
            {"candidate": candidate},
            session_dir=tmp_path,
        )

        assert result["status"] == "failed"
        assert result["error_class"] == "invalid_collective_candidate"
        assert result["decision"] == "REVERT"
        assert result["kept"] is False

    @pytest.mark.asyncio
    async def test_run_forge_collective_rejects_invalid_candidate_artifact(
        self,
        tmp_path,
    ):
        """Shape a malformed candidate artifact as a skipped result."""
        candidates = tmp_path / "collective_candidates.json"
        candidates.write_text("not-json", encoding="utf-8")
        SharedState(last_trace_analyze={"candidates_path": str(candidates)}).save(tmp_path)

        result = await krh._run_forge_collective({}, session_dir=tmp_path)

        assert result["status"] == "skipped"
        assert result["error_class"] == "invalid_collective_candidate_artifact"
        assert result["analysis_key"] == str(candidates)

    @pytest.mark.asyncio
    async def test_run_forge_collective_skips_without_candidate(self, tmp_path):
        """Skip safely when the latest analysis has no collective candidate."""
        SharedState().save(tmp_path)

        result = await krh._run_forge_collective({}, session_dir=tmp_path)

        assert result["status"] == "skipped"
        assert result["error_class"] == "no_collective_candidate"
        assert result["analysis_key"] == ""
        assert result["requires_e2e_validation"] is False

    @pytest.mark.asyncio
    async def test_run_forge_collective_rejects_unsupported_contract(self, tmp_path):
        """Reject collectives with no distributed reference in the driver."""
        candidate = _collective_candidate(kernel_contract={"kind": "collective", "collective_op": "all_to_all"})

        result = await krh._run_forge_collective(
            {"candidate": candidate},
            session_dir=tmp_path,
        )

        assert result["error_class"] == "unsupported_collective_contract"
        assert result["decision"] == "REVERT"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "field, value",
        [
            ("kernel_id", ""),
            ("source_file", ""),
            ("source_function", ""),
            ("input_shapes", []),
            ("input_dtypes", []),
            ("gpu_pct", True),
            ("gpu_pct", float("inf")),
            ("gpu_pct", -1),
        ],
    )
    async def test_run_forge_collective_rejects_invalid_required_fields(
        self,
        tmp_path,
        field,
        value,
    ):
        """Reject missing and invalid fields required by the driver."""
        candidate = _collective_candidate(**{field: value})

        result = await krh._run_forge_collective(
            {"candidate": candidate},
            session_dir=tmp_path,
        )

        assert result["error_class"] == "invalid_collective_candidate"
        assert field in result["error"]

    @pytest.mark.asyncio
    async def test_run_forge_collective_rejects_non_string_kernel_repo(
        self,
        tmp_path,
    ):
        """Reject a candidate whose repository root is not a string."""
        candidate = _collective_candidate(kernel_repo=123)

        result = await krh._run_forge_collective(
            {"candidate": candidate},
            session_dir=tmp_path,
        )

        assert result["error_class"] == "invalid_collective_candidate"
        assert "kernel_repo must be a string" in result["error"]

    @pytest.mark.asyncio
    async def test_run_forge_collective_requires_resolvable_repo_root(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Reject a source path with no explicit or discoverable repository."""
        candidate = _collective_candidate(source_file=str(tmp_path / "outside" / "all_reduce.py"))
        candidate.pop("kernel_repo")
        monkeypatch.setattr(krh, "_find_repo_root_for_source", lambda _path: "")

        result = await krh._run_forge_collective(
            {"candidate": candidate},
            session_dir=tmp_path,
        )

        assert result["error_class"] == "kernel_repo_missing"
        assert candidate["source_file"] in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tp",
        [True, 1.5, "invalid", 1],
        ids=["bool", "float", "non-numeric", "single-rank"],
    )
    async def test_run_forge_collective_rejects_invalid_world_size(
        self,
        tmp_path,
        monkeypatch,
        tp,
    ):
        """Reject non-integral and single-rank collective world sizes."""
        monkeypatch.delenv("TP", raising=False)

        result = await krh._run_forge_collective(
            {"candidate": _collective_candidate(), "tp": tp},
            session_dir=tmp_path,
        )

        assert result["error_class"] == "invalid_collective_world_size"
        assert result["decision"] == "REVERT"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "budget",
        [
            {"timeout": True},
            {"timeout": 1.5},
            {"timeout": "invalid"},
            {"timeout": -1},
            {"timeout": 10000, "max_hours": True},
        ],
        ids=["bool", "float", "non-numeric", "negative", "max-hours"],
    )
    async def test_run_forge_collective_rejects_invalid_budget(
        self,
        tmp_path,
        monkeypatch,
        budget,
    ):
        """Reject malformed collective timeout and campaign budgets."""
        monkeypatch.delenv("FORGE_COLLECTIVE_TIMEOUT", raising=False)
        payload = {
            "candidate": _collective_candidate(),
            "tp": 2,
            **budget,
        }

        result = await krh._run_forge_collective(payload, session_dir=tmp_path)

        assert result["error_class"] == "invalid_collective_budget"
        assert result["decision"] == "REVERT"

    @pytest.mark.asyncio
    async def test_run_forge_collective_skips_insufficient_budget(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Skip when the wrapper budget cannot fit a minimum campaign."""
        monkeypatch.delenv("FORGE_COLLECTIVE_TIMEOUT", raising=False)

        result = await krh._run_forge_collective(
            {
                "candidate": _collective_candidate(),
                "tp": 2,
                "timeout": 100,
            },
            session_dir=tmp_path,
        )

        assert result["status"] == "skipped"
        assert result["error_class"] == "insufficient_collective_budget"
        assert result["analysis_key"] == ""

    @pytest.mark.asyncio
    async def test_run_forge_collective_writes_input_and_applies_defaults(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Run the collective happy path with a fully isolated wrapper."""
        candidates = tmp_path / "collective_candidates.json"
        fallback = _collective_candidate(
            kernel_id="fallback",
            candidate_source="trace",
            gpu_pct=90.0,
        )
        selected = _collective_candidate(
            kernel_id="resolved",
            source_file=str(tmp_path / "kernels" / "all_reduce.py"),
            kernel_repo=str(tmp_path),
            gpu_pct=10.0,
        )
        candidates.write_text(
            json.dumps(
                {
                    "hot_kernels": [
                        _collective_candidate(reusable_native_kernel=False),
                        _collective_candidate(
                            kernel_contract={
                                "kind": "collective",
                                "collective_op": "all_to_all",
                            }
                        ),
                        fallback,
                        selected,
                    ]
                }
            ),
            encoding="utf-8",
        )
        SharedState(
            gpu_type="MI300X",
            tp=8,
            last_trace_analyze={"candidates_path": str(candidates)},
        ).save(tmp_path)
        monkeypatch.setenv("FORGE_COLLECTIVE_AGENT_TIMEOUT", "900")
        _pin_fusion_provider_env(
            monkeypatch,
            {**_ANTHROPIC_ONLY_ENV, "CLAUDE_MODEL": "claude-test"},
        )
        monkeypatch.setattr(krh.time, "time_ns", lambda: 123)
        tool_path = tmp_path / "tools" / "forge_collective.py"
        monkeypatch.setattr(
            krh,
            "_kernel_agent_tool_path",
            lambda name: tmp_path / "tools" / name,
        )
        captured = {}

        async def _fake_subprocess(cmd, *, timeout_sec):
            """Capture the wrapper invocation and return a valid sentinel."""
            captured["cmd"] = cmd
            captured["timeout_sec"] = timeout_sec
            input_path = Path(cmd[cmd.index("--input-json") + 1])
            captured["input"] = json.loads(input_path.read_text(encoding="utf-8"))
            wrapper_result = {
                "status": "complete",
                "engine": "forge_collective",
                "decision": "REVERT",
                "kept": False,
                "requires_e2e_validation": False,
            }
            return (
                0,
                "FORGE_COLLECTIVE_RESULT_BEGIN\n" + json.dumps(wrapper_result) + "\nFORGE_COLLECTIVE_RESULT_END\n",
                "",
            )

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        result = await krh._run_forge_collective(
            {
                "task_id": "collective-task",
                "tp": "4",
                "timeout": 10000,
                "max_hours": 1,
            },
            session_dir=tmp_path,
        )

        workspace = tmp_path / "runs" / "collective" / "collective-task" / "attempt-123"
        assert captured["cmd"] == [
            "python3",
            str(tool_path),
            "--input-json",
            str(workspace / "forge_collective_input.json"),
        ]
        assert captured["timeout_sec"] == 7200
        assert captured["input"] == {
            "agent_backend": "claude",
            "agent_timeout_sec": "900",
            "candidate": selected,
            "experience_id": "attempt-123",
            "finalize_grace_sec": 300,
            "framework": "",
            "gpu_target": "MI300X",
            "kernel_repo": str(tmp_path),
            "llm_model": "claude-test",
            "max_hours": 1.0,
            "operator_name": "all_reduce",
            "output_dir": str(workspace),
            "source_file": selected["source_file"],
            "source_files": [selected["source_file"]],
            "target_functions": ["all_reduce"],
            "timeout": 6900,
            "tp": 4,
        }
        assert result["backend"] == "forge"
        assert result["workspace"] == str(workspace)
        assert result["kernel_id"] == "resolved"
        assert result["kernel_name"] == "all_reduce_kernel"
        assert result["source_file"] == selected["source_file"]
        assert result["kernel_repo"] == str(tmp_path)
        assert result["gpu_pct"] == 10.0
        assert result["collective_op"] == "all_reduce"
        assert result["world_size"] == 4
        assert result["source"] == "forge_collective"

    @pytest.mark.asyncio
    async def test_run_forge_collective_shapes_invalid_sentinel(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Convert an invalid wrapper sentinel into a revert result."""
        _pin_fusion_provider_env(monkeypatch, _ANTHROPIC_ONLY_ENV)
        monkeypatch.setattr(krh.time, "time_ns", lambda: 456)
        monkeypatch.setattr(
            krh,
            "_kernel_agent_tool_path",
            lambda name: tmp_path / name,
        )

        async def _fake_subprocess(cmd, *, timeout_sec):
            """Return wrapper output without a result sentinel."""
            return 9, "wrapper output", "wrapper error"

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        result = await krh._run_forge_collective(
            {
                "candidate": _collective_candidate(),
                "tp": 2,
                "timeout": 10000,
                "max_hours": 1,
            },
            session_dir=tmp_path,
        )

        assert result["status"] == "failed"
        assert result["error_class"] == "invalid_collective_result"
        assert result["returncode"] == 9
        assert result["stdout_tail"] == "wrapper output"
        assert result["stderr_tail"] == "wrapper error"
        assert result["world_size"] == 2

    @pytest.mark.asyncio
    async def test_run_forge_collective_shapes_subprocess_timeout(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Convert wrapper timeout into a bounded revert result."""
        _pin_fusion_provider_env(monkeypatch, _ANTHROPIC_ONLY_ENV)
        monkeypatch.setattr(
            krh,
            "_kernel_agent_tool_path",
            lambda name: tmp_path / name,
        )

        async def _timeout(cmd, *, timeout_sec):
            """Raise the timeout produced by the subprocess wrapper."""
            raise subprocess.TimeoutExpired(["python3", "collective-worker"], timeout_sec)

        monkeypatch.setattr(krh, "_run_subprocess", _timeout)

        result = await krh._run_forge_collective(
            {
                "candidate": _collective_candidate(),
                "tp": 2,
                "timeout": 10000,
                "max_hours": 1,
            },
            session_dir=tmp_path,
        )

        assert result["status"] == "failed"
        assert result["error_class"] == "subprocess_timeout"
        assert "TimeoutExpired after 7200s" in result["error"]
        assert "collective-worker" in result["error"]
        assert result["engine"] == "forge_collective"
        assert result["requires_e2e_validation"] is False

    @pytest.mark.asyncio
    async def test_run_forge_collective_prefers_forge_claude_model(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Collective must honor FORGE_CLAUDE_MODEL like fusion and rewrite."""
        SharedState(tp=2).save(tmp_path)
        _pin_fusion_provider_env(
            monkeypatch,
            {
                **_ANTHROPIC_ONLY_ENV,
                "CLAUDE_MODEL": "claude-orchestration",
                "FORGE_CLAUDE_MODEL": "claude-forge-only",
            },
        )
        monkeypatch.setattr(krh.time, "time_ns", lambda: 789)
        monkeypatch.setattr(
            krh,
            "_kernel_agent_tool_path",
            lambda name: tmp_path / "tools" / name,
        )
        captured = {}

        async def _fake_subprocess(cmd, *, timeout_sec):
            input_path = Path(cmd[cmd.index("--input-json") + 1])
            captured["input"] = json.loads(input_path.read_text(encoding="utf-8"))
            wrapper_result = {
                "status": "complete",
                "engine": "forge_collective",
                "decision": "REVERT",
                "kept": False,
                "requires_e2e_validation": False,
            }
            return (
                0,
                "FORGE_COLLECTIVE_RESULT_BEGIN\n" + json.dumps(wrapper_result) + "\nFORGE_COLLECTIVE_RESULT_END\n",
                "",
            )

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        result = await krh._run_forge_collective(
            {
                "candidate": _collective_candidate(kernel_repo=str(tmp_path)),
                "tp": 2,
                "timeout": 10000,
                "max_hours": 1,
            },
            session_dir=tmp_path,
        )

        assert result["status"] == "complete"
        assert captured["input"]["agent_backend"] == "claude"
        assert captured["input"]["llm_model"] == "claude-forge-only"

    @pytest.mark.asyncio
    async def test_run_forge_collective_openai_only_selects_codex(
        self,
        tmp_path,
        monkeypatch,
    ):
        """OpenAI-only collective must pin Codex and FORGE_CODEX_MODEL."""
        SharedState(tp=2).save(tmp_path)
        _pin_fusion_provider_env(
            monkeypatch,
            {
                **_OPENAI_ONLY_ENV,
                "CODEX_MODEL": "gpt-orchestration",
                "FORGE_CODEX_MODEL": "gpt-forge-only",
            },
        )
        monkeypatch.setattr(krh.time, "time_ns", lambda: 790)
        monkeypatch.setattr(
            krh,
            "_kernel_agent_tool_path",
            lambda name: tmp_path / "tools" / name,
        )
        captured = {}

        async def _fake_subprocess(cmd, *, timeout_sec):
            input_path = Path(cmd[cmd.index("--input-json") + 1])
            captured["input"] = json.loads(input_path.read_text(encoding="utf-8"))
            wrapper_result = {
                "status": "complete",
                "engine": "forge_collective",
                "decision": "REVERT",
                "kept": False,
                "requires_e2e_validation": False,
            }
            return (
                0,
                "FORGE_COLLECTIVE_RESULT_BEGIN\n" + json.dumps(wrapper_result) + "\nFORGE_COLLECTIVE_RESULT_END\n",
                "",
            )

        monkeypatch.setattr(krh, "_run_subprocess", _fake_subprocess)

        result = await krh._run_forge_collective(
            {
                "candidate": _collective_candidate(kernel_repo=str(tmp_path)),
                "tp": 2,
                "timeout": 10000,
                "max_hours": 1,
            },
            session_dir=tmp_path,
        )

        assert result["status"] == "complete"
        assert captured["input"]["agent_backend"] == "codex"
        assert captured["input"]["llm_model"] == "gpt-forge-only"

    @pytest.mark.asyncio
    async def test_run_forge_collective_unconfigured_provider_fails_before_subprocess(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Collective must fail closed before spawning when no LLM side is set."""
        SharedState(tp=2).save(tmp_path)
        _pin_fusion_provider_env(monkeypatch, {})

        async def _should_not_run(*_args, **_kwargs):
            raise AssertionError("forge-collective must not run without a provider")

        monkeypatch.setattr(krh, "_run_subprocess", _should_not_run)

        result = await krh._run_forge_collective(
            {
                "candidate": _collective_candidate(kernel_repo=str(tmp_path)),
                "tp": 2,
                "timeout": 10000,
                "max_hours": 1,
            },
            session_dir=tmp_path,
        )

        assert result["status"] == "failed"
        assert result["error_class"] == "llm_provider_unconfigured"
        assert result["decision"] == "REVERT"
        assert result["kept"] is False
        assert result["engine"] == "forge_collective"


def _ensure_torch_module(monkeypatch):
    try:
        import torch
    except ModuleNotFoundError:
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(device_count=lambda: 0),
        )
        monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


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


class TestBackendOrder:
    def test_kernel_opt_backends_alias_does_not_enable_forge(self, monkeypatch):
        monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)

        assert krh._backend_order({}) == []

    def test_kernel_opt_backends_alias_with_mixed_values_stays_geak_only(self, monkeypatch):
        monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)

        assert krh._backend_order({}) == []


class TestCandidateEnvAllowed:
    @pytest.mark.parametrize("name", ["AWS_SECRET_ACCESS_KEY", "ANTHROPIC_API_KEY"])
    def test_sensitive_env_blocked(self, name):
        assert krh._candidate_env_allowed(name) is False

    def test_known_prefix_allowed(self):
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


class TestRuntimeGeneratedKernel:
    def test_runtime_generated_path_treats_as_generated(self):
        markers = krh._RUNTIME_GENERATED_SOURCE_MARKERS
        if not markers:
            pytest.skip("no runtime markers in build")
        marker = next(iter(markers))
        assert krh._is_runtime_generated_kernel("kernel_agent", f"/tmp/{marker}_x.py") is True

    def test_reusable_source_root_overrides_compile_marker(self):
        markers = krh._COMPILE_GENERATED_NAME_MARKERS
        roots = krh._reusable_source_roots()
        if not markers or not roots:
            pytest.skip("required tables empty in build")
        marker = next(iter(markers))
        reusable_root = next(iter(roots))
        # Name matches but source lives under a reusable root -> False.
        assert krh._is_runtime_generated_kernel(marker, f"{reusable_root}/foo.py") is False


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
        """The handler reads the per-framework ``EXTRA_<FRAMEWORK>_ARGS`` slot."""
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

    def test_atom_server_args_ignore_stray_sglang_env(self, tmp_path):
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


class TestReusableSourceRootsAtom:
    """atom layout prefixes participate in cross-task kernel reuse
    alongside aiter/sglang/vllm."""

    def test_includes_atom_editable_path(self):
        # The matcher lowercases its source-file input, so the stored prefix is
        # lowercase ``/app/atom/atom/``.
        assert any("/app/atom/atom/" in r.lower() for r in krh._reusable_source_roots())

    def test_includes_atom_site_packages_python_3_10(self):
        assert any("/opt/venv/lib/python3.10/site-packages/atom/" in r for r in krh._reusable_source_roots())

    def test_includes_atom_site_packages_python_3_12(self):
        assert any("/opt/venv/lib/python3.12/site-packages/atom/" in r for r in krh._reusable_source_roots())

    def test_atom_path_classified_as_reusable(self):
        """An atom-owned kernel source under /app/ATOM/atom/ is NOT runtime-generated."""
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
        # Under /app/ but not /app/ATOM/atom/ -> runtime-generated (not reusable).
        result = krh._is_runtime_generated_kernel(
            marker,
            "/app/session_dir/runs/baseline/foo.py",
        )
        assert result is True


class TestRunGemmTuningHandler:
    def test_resolves_split_aiter_meta_root_for_forge(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        package_dir = tmp_path / "site-packages" / "aiter_meta"
        (package_dir / "csrc").mkdir(parents=True)
        spec = SimpleNamespace(submodule_search_locations=[str(package_dir)])
        monkeypatch.setattr(
            krh.importlib.util,
            "find_spec",
            lambda name: spec if name == "aiter_meta" else None,
        )

        assert krh._resolve_aiter_root_for_forge() == str(package_dir)

    def test_vllm_shape_capture_merges_per_device_native_rows(self, tmp_path, monkeypatch):
        from hyperloom.orchestrator.actions.executors import baseline as baseline_module

        baseline_config = tmp_path / "baseline.yaml"
        baseline_config.write_text("benchmark:\n  framework: vllm\n")
        state = SharedState(
            framework="vllm",
            model_path="/models/qwen",
            gpu_type="mi300x",
            tp=1,
            conc=64,
            isl=1024,
            osl=512,
            max_model_len=4096,
            baseline_config_path=str(baseline_config),
            baseline_eager_fallback=True,
            current_best={
                "extra_server_args": "--enforce-eager --port 8888",
                "extra_envs": {
                    "CONC": "128",
                    "KEEP_ME": "1",
                    "PYTORCH_TUNABLEOP_TUNING": "1",
                    "HL_TUNABLEOP_MODE": "candidate",
                },
            },
        )
        captured: dict = {}

        class FakeBaselineExecutor:
            def __init__(self, **kwargs):
                captured["init"] = kwargs

            async def __call__(self, ctx):
                captured["task"] = ctx.task
                envs = ctx.task.params["extra_envs"]
                base = Path(envs["PYTORCH_TUNABLEOP_UNTUNED_FILENAME"])
                base.with_name(f"{base.stem}0{base.suffix}").write_text(
                    "GemmTunableOp_BFloat16_NT,nt_5120_4149_17408_ld_5120_17408_5120\nnot-a-native-row\n"
                )
                base.with_name(f"{base.stem}1{base.suffix}").write_text(
                    "GemmTunableOp_BFloat16_NT,nt_5120_4149_17408_ld_5120_17408_5120\n"
                    "GemmTunableOp_BFloat16_NN,nn_17408_1083_5120_ld_17408_5120_17408\n"
                    "ScaledGemmTunableOp_Float8_e4m3fnuz_NN,"
                    "nn_5120_4149_17408_ld_5120_17408_5120_rw_0_bias_None,"
                    "BLAS_PARAMS,algo_1\n"
                )
                return {"status": "succeeded"}

        monkeypatch.setattr(baseline_module, "BaselineExecutor", FakeBaselineExecutor)
        monkeypatch.setattr(krh, "_pick_shape_capture_port", lambda: 19001)

        result = asyncio.run(
            krh._capture_vllm_tunableop_shapes(
                state=state,
                session_dir=tmp_path,
                payload={"task_id": "capture", "tp": 2},
                workspace=tmp_path / "runs" / "gemm_tuning" / "capture",
            )
        )

        assert result["status"] == "ok"
        assert result["shape_count"] == 3
        merged = Path(result["tunableop_input"])
        assert len(merged.read_text().splitlines()) == 3
        task = captured["task"]
        assert task.kind == "gemm_shape_capture"
        assert task.params["baseline_double_run"] is False
        assert task.params["disable_run_eval"] is True
        assert task.params["extra_server_args"] == "--enforce-eager --port 8888"
        assert "--port" in task.params["remove_args"]
        assert task.params["extra_envs"]["KEEP_ME"] == "1"
        assert task.params["extra_envs"]["PYTORCH_TUNABLEOP_ENABLED"] == "1"
        assert task.params["extra_envs"]["PYTORCH_TUNABLEOP_TUNING"] == "0"
        assert task.params["extra_envs"]["PYTORCH_TUNABLEOP_RECORD_UNTUNED"] == "1"
        assert task.params["extra_envs"]["HL_TUNABLEOP_MODE"] == ""
        assert task.params["extra_envs"]["PORT"] == "19001"
        assert task.params["extra_envs"]["TP"] == "2"
        assert task.params["extra_envs"]["CONC"] == "128"
        assert task.params["extra_envs"]["ISL"] == "1024"
        assert task.params["extra_envs"]["OSL"] == "512"
        assert task.params["extra_envs"]["MAX_MODEL_LEN"] == "4096"
        assert "HL_TUNABLEOP_MODE" in task.params["unset_envs"]
        assert captured["init"]["shared_state"] is not state
        assert captured["init"]["shared_state"].baseline_eager_fallback is False
        assert state.baseline_eager_fallback is True

    def test_vllm_block_fp8_profile_capture_extracts_runtime_shapes(self, tmp_path, monkeypatch):
        import gzip

        from hyperloom.orchestrator.actions.executors import roofline as roofline_module

        state = SharedState(
            framework="vllm",
            model_path="/models/qwen",
            gpu_type="mi325x",
            tp=1,
            conc=2,
            isl=128,
            osl=16,
            max_model_len=512,
            current_best={
                "extra_envs": {
                    "VLLM_ROCM_USE_AITER": "1",
                    "VLLM_ROCM_USE_AITER_LINEAR": "1",
                    "PYTORCH_TUNABLEOP_ENABLED": "1",
                }
            },
        )
        captured: dict = {}

        class FakeRooflineExecutor:
            def __init__(self, *, shared_state):
                captured["init"] = {
                    "shared_state": shared_state,
                }

            async def __call__(self, ctx):
                captured["task"] = ctx.task
                trace_dir = Path(ctx.task.params["output_dir"]) / "tracelens"
                trace_dir.mkdir(parents=True)
                trace = {
                    "traceEvents": [
                        {
                            "name": "vllm::w8a8_triton_block_scaled_mm_func",
                            "args": {
                                "Input Dims": [
                                    [4149, 5120],
                                    [34816, 5120],
                                    [33, 40],
                                    [272, 40],
                                ]
                            },
                        },
                        {
                            "name": "vllm::w8a8_triton_block_scaled_mm_func",
                            "args": {"Input Dims": [[1083, 17408], [5120, 17408]]},
                        },
                    ]
                }
                steady_trace = trace_dir / "mixed_steady_state.trace.json.gz"
                with gzip.open(steady_trace, "wt", encoding="utf-8") as stream:
                    json.dump(trace, stream)
                return {
                    "status": "succeeded",
                    "steady_state_trace": str(steady_trace),
                }

        monkeypatch.setattr(roofline_module, "RooflineExecutor", FakeRooflineExecutor)

        result = asyncio.run(
            krh._capture_vllm_tunableop_shapes(
                state=state,
                session_dir=tmp_path,
                payload={
                    "task_id": "capture",
                    "_shape_capture_mode": "block_fp8_profile",
                },
                workspace=tmp_path / "runs" / "gemm_tuning" / "capture",
            )
        )

        assert result["status"] == "ok"
        assert result["capture_mode"] == "block_fp8_profile"
        assert result["shape_count"] == 2
        assert json.loads(Path(result["shapes_json"]).read_text()) == [
            {"K": 17408, "M": 1083, "N": 5120},
            {"K": 5120, "M": 4149, "N": 34816},
        ]
        task = captured["task"]
        assert "profiler-config.delay_iterations" not in task.params["extra_server_args"]
        assert "profiler-config.max_iterations" not in task.params["extra_server_args"]
        assert "analysis_route" not in task.params
        assert "steady_state_mode" not in task.params
        assert task.params["extra_envs"]["VLLM_ROCM_USE_AITER"] == "1"
        assert task.params["extra_envs"]["VLLM_ROCM_USE_AITER_LINEAR"] == "1"
        assert task.params["extra_envs"]["PYTORCH_TUNABLEOP_ENABLED"] == "1"
        assert "PYTORCH_TUNABLEOP_ENABLED" not in task.params["unset_envs"]
        assert captured["init"]["shared_state"] is state

    def test_vllm_block_fp8_profile_capture_uses_standard_roofline(self, tmp_path, monkeypatch):
        import gzip

        from hyperloom.orchestrator.actions.executors import baseline as baseline_module
        from hyperloom.orchestrator.actions.executors import roofline as roofline_module

        steady_trace = tmp_path / "mixed_steady_state.trace.json.gz"
        with gzip.open(steady_trace, "wt", encoding="utf-8") as stream:
            json.dump(
                {
                    "traceEvents": [
                        {
                            "name": "vllm::w8a8_triton_block_scaled_mm_func",
                            "args": {"Input Dims": [[4149, 5120], [34816, 5120]]},
                        }
                    ]
                },
                stream,
            )
        state = SharedState(
            framework="vllm",
            precision="fp8",
            model_path="/models/qwen",
            current_best={
                "extra_envs": {
                    "VLLM_ROCM_USE_AITER_LINEAR": "1",
                    "PYTORCH_TUNABLEOP_ENABLED": "1",
                },
            },
            last_baseline={"benchmark_script": "vllm_custom.sh"},
        )
        captured: dict = {}

        class FailBaselineExecutor:
            def __init__(self, **kwargs):
                pass

            async def __call__(self, ctx):
                raise AssertionError("block-FP8 profile capture must use RooflineExecutor")

        class FakeRooflineExecutor:
            def __init__(self, *, shared_state):
                captured["shared_state"] = shared_state

            async def __call__(self, ctx):
                captured["task"] = ctx.task
                return {
                    "status": "succeeded",
                    "steady_state_trace": str(steady_trace),
                }

        monkeypatch.setattr(baseline_module, "BaselineExecutor", FailBaselineExecutor)
        monkeypatch.setattr(roofline_module, "RooflineExecutor", FakeRooflineExecutor)

        result = asyncio.run(
            krh._capture_vllm_tunableop_shapes(
                state=state,
                session_dir=tmp_path,
                payload={
                    "task_id": "capture",
                    "_shape_capture_mode": "block_fp8_profile",
                },
                workspace=tmp_path / "runs" / "gemm_tuning" / "capture",
            )
        )

        assert result["status"] == "ok"
        assert result["capture_mode"] == "block_fp8_profile"
        assert result["shape_count"] == 1
        assert result["source_profile_trace"] == str(steady_trace)
        assert captured["shared_state"] is state
        task = captured["task"]
        assert "config_path" not in task.params
        assert "timeout_sec" not in task.params
        assert "analysis_route" not in task.params
        assert "steady_state_mode" not in task.params
        assert "reason" not in task.params
        assert task.params["benchmark_script"] == "vllm_custom.sh"
        assert task.params["extra_envs"]["VLLM_ROCM_USE_AITER_LINEAR"] == "1"
        assert "VLLM_ROCM_USE_AITER" not in task.params["extra_envs"]
        assert task.params["extra_envs"]["PYTORCH_TUNABLEOP_ENABLED"] == "1"
        assert "PYTORCH_TUNABLEOP_ENABLED" not in task.params["unset_envs"]
        assert "profiler-config.delay_iterations" not in task.params["extra_server_args"]
        assert "profiler-config.max_iterations" not in task.params["extra_server_args"]

    def test_vllm_block_fp8_profile_excludes_capture_sidecars(self, tmp_path):
        import gzip

        trace_root = tmp_path / "profile"
        main_trace = trace_root / "torch_trace" / "main.trace.json.gz"
        capture_trace = trace_root / "capture_traces" / "graph_capture.trace.json.gz"
        main_trace.parent.mkdir(parents=True)
        capture_trace.parent.mkdir(parents=True)

        def write_trace(path, *, m, n, k):
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(
                    {
                        "traceEvents": [
                            {
                                "name": "vllm::w8a8_triton_block_scaled_mm_func",
                                "args": {"Input Dims": [[m, k], [n, k]]},
                            }
                        ]
                    },
                    stream,
                )

        write_trace(main_trace, m=4149, n=34816, k=5120)
        write_trace(capture_trace, m=256, n=34816, k=5120)

        shapes_json, shape_count = krh._extract_vllm_block_fp8_profile_shapes(trace_root)

        assert shape_count == 1
        assert json.loads(Path(shapes_json).read_text()) == [{"K": 5120, "M": 4149, "N": 34816}]
        assert krh._extract_vllm_block_fp8_profile_shapes(capture_trace) == ("", 0)

    def test_vllm_block_fp8_profile_capture_without_trace_is_explicit_failure(
        self,
        tmp_path,
        monkeypatch,
    ):
        from hyperloom.orchestrator.actions.executors import roofline as roofline_module

        state = SharedState(
            framework="vllm",
            model_path="/models/qwen",
        )

        class FakeRooflineExecutor:
            def __init__(self, *, shared_state, persist_state=True):
                pass

            async def __call__(self, ctx):
                return {"status": "succeeded"}

        monkeypatch.setattr(roofline_module, "RooflineExecutor", FakeRooflineExecutor)

        result = asyncio.run(
            krh._capture_vllm_tunableop_shapes(
                state=state,
                session_dir=tmp_path,
                payload={
                    "task_id": "capture",
                    "_shape_capture_mode": "block_fp8_profile",
                },
                workspace=tmp_path / "runs" / "gemm_tuning" / "capture",
            )
        )

        assert result["status"] == "failed"
        assert result["decision"] == "REVERT"
        assert result["error_class"] == "shape_capture_failed"
        assert result["capture_mode"] == "block_fp8_profile"
        assert result["shape_count"] == 0

    def test_extract_block_fp8_shapes_rejects_empty_trace_path(self, tmp_path, monkeypatch):
        """``Path("")`` normalizes to the CWD; it must never be walked."""
        stale = tmp_path / "unrelated" / "old_session" / "stale.trace.json"
        stale.parent.mkdir(parents=True)
        stale.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {
                            "name": "vllm::w8a8_triton_block_scaled_mm_func",
                            "args": {"Input Dims": [[999, 111], [222, 111]]},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        assert krh._extract_vllm_block_fp8_profile_shapes(Path("")) == ("", 0)
        assert krh._extract_vllm_block_fp8_profile_shapes(Path(".")) == ("", 0)

    def test_vllm_block_fp8_profile_capture_without_trace_ignores_cwd_traces(
        self,
        tmp_path,
        monkeypatch,
    ):
        """A succeeded Roofline with no steady trace must not harvest CWD shapes."""
        from hyperloom.orchestrator.actions.executors import roofline as roofline_module

        stale = tmp_path / "unrelated" / "old_session" / "stale.trace.json"
        stale.parent.mkdir(parents=True)
        stale.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {
                            "name": "vllm::w8a8_triton_block_scaled_mm_func",
                            "args": {"Input Dims": [[999, 111], [222, 111]]},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        session_dir = tmp_path / "session"
        session_dir.mkdir()
        state = SharedState(framework="vllm", model_path="/models/qwen")

        class FakeRooflineExecutor:
            def __init__(self, *, shared_state, persist_state=True):
                pass

            async def __call__(self, ctx):
                return {"status": "succeeded", "steady_state_trace": ""}

        monkeypatch.setattr(roofline_module, "RooflineExecutor", FakeRooflineExecutor)

        result = asyncio.run(
            krh._capture_vllm_tunableop_shapes(
                state=state,
                session_dir=session_dir,
                payload={
                    "task_id": "capture",
                    "_shape_capture_mode": "block_fp8_profile",
                },
                workspace=session_dir / "runs" / "gemm_tuning" / "capture",
            )
        )

        assert result["status"] == "failed"
        assert result["error_class"] == "shape_capture_failed"
        assert result["shape_count"] == 0
        assert "shapes_json" not in result

    def test_multi_node_vllm_block_fp8_preserves_existing_no_capture_behavior(self, monkeypatch):
        from hyperloom.orchestrator.actions.executors import _multi_node_env

        monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: True)

        assert (
            krh._vllm_block_fp8_profile_capture_required(
                framework="vllm",
                precision="fp8",
                quant_type="blockscale",
                shapes_json="",
                tunableop_input="",
                dry_run=False,
            )
            is False
        )

    def test_shape_capture_kill_switch_disables_dense_and_block_fp8(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_GEMM_SHAPE_CAPTURE", "0")
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"hidden_size": 5120, "intermediate_size": 17408}))

        assert (
            krh._vllm_dense_shape_capture_required(
                framework="vllm",
                model_path=str(model),
                shapes_json="",
                tunableop_input="",
                dry_run=False,
            )
            is False
        )
        assert (
            krh._vllm_block_fp8_profile_capture_required(
                framework="vllm",
                precision="fp8",
                quant_type="blockscale",
                shapes_json="",
                tunableop_input="",
                dry_run=False,
            )
            is False
        )

    @pytest.mark.parametrize("port", [0, -1, 8888, 65536, "bad"])
    def test_shape_capture_rejects_unsafe_explicit_port(self, port):
        with pytest.raises(ValueError):
            krh._resolve_shape_capture_port(port)

    def test_vllm_shape_capture_empty_output_is_explicit_failure(self, tmp_path, monkeypatch):
        from hyperloom.orchestrator.actions.executors import baseline as baseline_module

        baseline_config = tmp_path / "baseline.yaml"
        baseline_config.write_text("benchmark:\n  framework: vllm\n")
        state = SharedState(
            framework="vllm",
            model_path="/models/qwen",
            baseline_config_path=str(baseline_config),
        )

        class FakeBaselineExecutor:
            def __init__(self, **kwargs):
                pass

            async def __call__(self, ctx):
                return {"status": "succeeded"}

        monkeypatch.setattr(baseline_module, "BaselineExecutor", FakeBaselineExecutor)

        result = asyncio.run(
            krh._capture_vllm_tunableop_shapes(
                state=state,
                session_dir=tmp_path,
                payload={"task_id": "capture"},
                workspace=tmp_path / "runs" / "gemm_tuning" / "capture",
            )
        )

        assert result["status"] == "failed"
        assert result["error_class"] == "shape_capture_failed"
        assert result["shape_count"] == 0

    @pytest.mark.parametrize(
        ("framework", "shapes_json", "tunableop_input", "dry_run"),
        [
            ("sglang", "", "", False),
            ("vllm", "present", "", False),
            ("vllm", "", "/tmp/tunableop.csv", False),
            ("vllm", "", "", True),
        ],
    )
    def test_vllm_shape_capture_does_not_change_existing_input_paths(
        self,
        tmp_path,
        framework,
        shapes_json,
        tunableop_input,
        dry_run,
    ):
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"hidden_size": 5120, "intermediate_size": 17408}))
        if shapes_json:
            path = tmp_path / "shapes.json"
            path.write_text(json.dumps([{"M": 1, "N": 2, "K": 3}]))
            shapes_json = str(path)

        assert (
            krh._vllm_dense_shape_capture_required(
                framework=framework,
                model_path=str(model),
                shapes_json=shapes_json,
                tunableop_input=tunableop_input,
                dry_run=dry_run,
            )
            is False
        )

    def test_multi_node_vllm_preserves_existing_no_capture_behavior(self, tmp_path, monkeypatch):
        from hyperloom.orchestrator.actions.executors import _multi_node_env

        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"hidden_size": 5120, "intermediate_size": 17408}))
        monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: True)

        assert (
            krh._vllm_dense_shape_capture_required(
                framework="vllm",
                model_path=str(model),
                shapes_json="",
                tunableop_input="",
                dry_run=False,
            )
            is False
        )

    def test_vllm_moe_does_not_trigger_dense_shape_capture(self, tmp_path):
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "hidden_size": 5120,
                    "intermediate_size": 17408,
                    "num_experts": 128,
                }
            )
        )

        assert (
            krh._vllm_dense_shape_capture_required(
                framework="vllm",
                model_path=str(model),
                shapes_json="",
                tunableop_input="",
                dry_run=False,
            )
            is False
        )

    def test_explicit_tunableop_input_bypasses_vllm_shape_capture(self, tmp_path, monkeypatch):
        root = tmp_path / "kernel-agent"
        tool = root / "tools" / "forge_gemm_tuning.py"
        tool.parent.mkdir(parents=True)
        tool.write_text("# placeholder\n")
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(root))
        monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")

        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"hidden_size": 5120, "intermediate_size": 17408}))
        explicit_input = tmp_path / "operator-provided.csv"
        explicit_input.write_text("GemmTunableOp_BFloat16_NN,nn_5120_64_5120_ld_5120_5120_5120\n")
        SharedState(
            precision="fp8",
            framework="vllm",
            model_path=str(model),
            gpu_type="mi300x",
        ).save(tmp_path)
        captured_input: dict = {}

        async def fail_capture(**kwargs):
            raise AssertionError("explicit tunableop_input must bypass automatic capture")

        async def fake_run(cmd: list[str], *, timeout_sec: int):
            input_path = Path(cmd[cmd.index("--input-json") + 1])
            captured_input.update(json.loads(input_path.read_text()))
            return (
                0,
                "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
                + json.dumps({"status": "ok", "micro_decision": "no_improvement"})
                + "\nFORGE_GEMM_TUNE_RESULT_END\n",
                "",
            )

        monkeypatch.setattr(krh, "_capture_vllm_tunableop_shapes", fail_capture)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(krh, "_run_subprocess", fake_run)

        result = asyncio.run(
            krh.run_gemm_tuning_handler(
                {
                    "task_id": "explicit",
                    "tunableop_input": str(explicit_input),
                },
                session_dir=tmp_path,
            )
        )

        assert captured_input["tunableop_input"] == str(explicit_input)
        assert result["status"] == "ok"

    def test_vllm_dense_without_shapes_captures_tunableop_input(self, tmp_path, monkeypatch):
        root = tmp_path / "kernel-agent"
        tool = root / "tools" / "forge_gemm_tuning.py"
        tool.parent.mkdir(parents=True)
        tool.write_text("# placeholder\n")
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(root))
        monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")

        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3ForCausalLM"],
                    "hidden_size": 5120,
                    "intermediate_size": 17408,
                }
            )
        )
        SharedState(
            precision="fp8",
            framework="vllm",
            model_path=str(model),
            gpu_type="mi300x",
            tp=1,
            conc=64,
            isl=1024,
            osl=1024,
        ).save(tmp_path)

        capture_calls: list[dict] = []
        captured_input: dict = {}

        async def fake_capture(*, state, session_dir, payload, workspace):
            capture_calls.append(
                {
                    "state": state,
                    "session_dir": session_dir,
                    "payload": payload,
                    "workspace": workspace,
                }
            )
            path = tmp_path / "tunableop_untuned.csv"
            path.write_text("GemmTunableOp_BFloat16_NT,nt_5120_4149_17408_ld_5120_17408_5120\n")
            return {"status": "ok", "tunableop_input": str(path)}

        async def fake_run(cmd: list[str], *, timeout_sec: int):
            input_path = Path(cmd[cmd.index("--input-json") + 1])
            captured_input.update(json.loads(input_path.read_text()))
            return (
                0,
                "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
                + json.dumps(
                    {
                        "status": "ok",
                        "micro_decision": "no_improvement",
                        "recommended_env": {},
                        "tuners_run": [{"tuner_name": "vllm_dense_tunableop"}],
                    }
                )
                + "\nFORGE_GEMM_TUNE_RESULT_END\n",
                "",
            )

        monkeypatch.setattr(krh, "_capture_vllm_tunableop_shapes", fake_capture, raising=False)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(krh, "_run_subprocess", fake_run)

        result = asyncio.run(krh.run_gemm_tuning_handler({"task_id": "capture"}, session_dir=tmp_path))

        assert len(capture_calls) == 1
        assert captured_input["tunableop_input"] == str(tmp_path / "tunableop_untuned.csv")
        assert result["status"] == "ok"

    def test_vllm_dense_capture_failure_stops_before_forge(self, tmp_path, monkeypatch):
        root = tmp_path / "kernel-agent"
        tool = root / "tools" / "forge_gemm_tuning.py"
        tool.parent.mkdir(parents=True)
        tool.write_text("# placeholder\n")
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(root))
        monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")

        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"hidden_size": 5120, "intermediate_size": 17408}))
        SharedState(
            precision="bf16",
            framework="vllm",
            model_path=str(model),
            gpu_type="mi300x",
        ).save(tmp_path)

        async def fake_capture(**kwargs):
            return {
                "status": "failed",
                "decision": "REVERT",
                "error_class": "shape_capture_failed",
                "error": "no complete workload recording",
            }

        async def fail_run(*args, **kwargs):
            raise AssertionError("Forge must not run after shape capture fails")

        monkeypatch.setattr(krh, "_capture_vllm_tunableop_shapes", fake_capture)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(krh, "_run_subprocess", fail_run)

        result = asyncio.run(
            krh.run_gemm_tuning_handler(
                {"task_id": "capture-failed"},
                session_dir=tmp_path,
            )
        )

        assert result["status"] == "failed"
        assert result["decision"] == "REVERT"
        assert result["error_class"] == "shape_capture_failed"
        assert result["backend"] == "forge"
        assert result["engine"] == "forge"

    def test_vllm_block_fp8_reuses_roofline_trace_without_recapture(self, tmp_path, monkeypatch):
        import gzip

        root = tmp_path / "kernel-agent"
        tool = root / "tools" / "forge_gemm_tuning.py"
        tool.parent.mkdir(parents=True)
        tool.write_text("# placeholder\n")
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(root))
        monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")

        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3ForCausalLM"],
                    "hidden_size": 5120,
                    "intermediate_size": 17408,
                    "quantization_config": {
                        "quant_method": "fp8",
                        "weight_block_size": [128, 128],
                    },
                }
            )
        )
        roofline_trace = tmp_path / "roofline.trace.json.gz"
        steady_trace = tmp_path / "mixed_steady_state.trace.json.gz"
        with gzip.open(roofline_trace, "wt", encoding="utf-8") as stream:
            json.dump(
                {
                    "traceEvents": [
                        {
                            "name": "vllm::w8a8_triton_block_scaled_mm_func",
                            "args": {"Input Dims": [[1, 5120], [34816, 5120]]},
                        }
                    ]
                },
                stream,
            )
        with gzip.open(steady_trace, "wt", encoding="utf-8") as stream:
            json.dump(
                {
                    "traceEvents": [
                        {
                            "name": "vllm::w8a8_triton_block_scaled_mm_func",
                            "args": {
                                "Input Dims": [
                                    [4149, 5120],
                                    [34816, 5120],
                                    [33, 40],
                                    [272, 40],
                                ]
                            },
                        }
                    ]
                },
                stream,
            )
        state = SharedState(
            precision="fp8",
            framework="vllm",
            model_path=str(model),
            gpu_type="mi325x",
            tp=1,
            conc=64,
            isl=1024,
            osl=1024,
            max_model_len=4096,
            last_profile_trace=str(roofline_trace),
            last_profile_status="succeeded",
            last_trace_analyze={
                "trace_input": str(roofline_trace),
                "steady_state_trace": str(steady_trace),
            },
        )
        state.last_profile_workload = state.current_profile_workload_context()
        state.save(tmp_path)
        captured_input: dict = {}

        async def fail_capture(**kwargs):
            raise AssertionError("valid Roofline shapes must bypass a second profile capture")

        async def fake_run(cmd: list[str], *, timeout_sec: int):
            input_path = Path(cmd[cmd.index("--input-json") + 1])
            captured_input.update(json.loads(input_path.read_text()))
            return (
                0,
                "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
                + json.dumps(
                    {
                        "status": "ok",
                        "micro_decision": "no_improvement",
                        "tuners_run": [{"tuner_name": "a8w8_blockscale"}],
                    }
                )
                + "\nFORGE_GEMM_TUNE_RESULT_END\n",
                "",
            )

        monkeypatch.setattr(krh, "_capture_vllm_tunableop_shapes", fail_capture)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(krh, "_run_subprocess", fake_run)

        result = asyncio.run(
            krh.run_gemm_tuning_handler(
                {"task_id": "reuse-roofline"},
                session_dir=tmp_path,
            )
        )

        assert captured_input["framework"] == "vllm-aiter"
        shapes_path = Path(captured_input["shapes_json"])
        # The traced M=4149 is re-keyed onto the padded M aiter looks up, plus a
        # power-of-two ladder so the rows stay reachable for the M values the
        # profile never sampled.
        aligned = json.loads(shapes_path.read_text())
        assert {row["N"] for row in aligned} == {34816}
        assert {row["M"] for row in aligned} == {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 4224, 8192}
        assert result["shape_alignment"]["applied"] is True
        assert captured_input["untuned_csv"] == ""
        assert result["shape_capture"]["capture_mode"] == "roofline_profile_reuse"
        assert result["shape_capture"]["source_profile_trace"] == str(steady_trace)
        assert result["status"] == "ok"

    @pytest.mark.parametrize("stale_reason", ["legacy_metadata", "missing_steady_trace"])
    def test_vllm_block_fp8_handler_runs_one_standard_roofline_fallback(
        self,
        tmp_path,
        monkeypatch,
        stale_reason,
    ):
        import gzip

        from hyperloom.orchestrator.actions.executors import roofline as roofline_module

        root = tmp_path / "kernel-agent"
        tool = root / "tools" / "forge_gemm_tuning.py"
        tool.parent.mkdir(parents=True)
        tool.write_text("# placeholder\n")
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(root))
        monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3ForCausalLM"],
                    "hidden_size": 5120,
                    "intermediate_size": 17408,
                    "quantization_config": {
                        "quant_method": "fp8",
                        "weight_block_size": [128, 128],
                    },
                }
            )
        )
        old_trace = tmp_path / "old.trace.json"
        old_trace.write_text(json.dumps({"traceEvents": []}))
        old_selected_trace = tmp_path / "old_mixed_steady_state.trace.json"
        old_selected_trace.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {
                            "name": "vllm::w8a8_triton_block_scaled_mm_func",
                            "args": {"Input Dims": [[1, 5120], [34816, 5120]]},
                        }
                    ]
                }
            )
        )
        selected_trace = tmp_path / "decode_only_steady_state.trace.json.gz"
        with gzip.open(selected_trace, "wt", encoding="utf-8") as stream:
            json.dump(
                {
                    "traceEvents": [
                        {
                            "name": "vllm::w8a8_triton_block_scaled_mm_func",
                            "args": {"Input Dims": [[64, 5120], [34816, 5120]]},
                        }
                    ]
                },
                stream,
            )
        state = SharedState(
            precision="fp8",
            framework="vllm",
            model_path=str(model),
            gpu_type="mi325x",
            tp=1,
            conc=64,
            isl=1024,
            osl=1024,
            max_model_len=4096,
            last_profile_trace=str(old_trace),
            last_profile_status="succeeded",
            last_trace_analyze={
                "trace_input": str(old_trace),
                **({"steady_state_trace": str(old_selected_trace)} if stale_reason == "legacy_metadata" else {}),
            },
            current_best={
                "extra_envs": {"VLLM_ROCM_USE_AITER_LINEAR": "1"},
            },
        )
        if stale_reason == "missing_steady_trace":
            state.last_profile_workload = state.current_profile_workload_context()
        state.save(tmp_path)
        roofline_calls = 0
        captured_input: dict = {}

        class FakeRooflineExecutor:
            def __init__(self, *, shared_state):
                self.shared_state = shared_state

            async def __call__(self, ctx):
                nonlocal roofline_calls
                roofline_calls += 1
                return {
                    "status": "succeeded",
                    "steady_state_trace": str(selected_trace),
                }

        async def fake_run(cmd: list[str], *, timeout_sec: int):
            input_path = Path(cmd[cmd.index("--input-json") + 1])
            captured_input.update(json.loads(input_path.read_text()))
            return (
                0,
                "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
                + json.dumps({"status": "ok", "micro_decision": "no_improvement"})
                + "\nFORGE_GEMM_TUNE_RESULT_END\n",
                "",
            )

        monkeypatch.setattr(roofline_module, "RooflineExecutor", FakeRooflineExecutor)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(krh, "_run_subprocess", fake_run)

        result = asyncio.run(
            krh.run_gemm_tuning_handler(
                {"task_id": f"fallback-{stale_reason}"},
                session_dir=tmp_path,
            )
        )

        assert roofline_calls == 1
        assert captured_input["framework"] == "vllm-aiter"
        captured_shapes = json.loads(Path(captured_input["shapes_json"]).read_text())
        assert {(row["N"], row["K"]) for row in captured_shapes} == {(34816, 5120)}
        assert {row["M"] for row in captured_shapes} == {16, 32, 64}
        assert result["shape_capture"]["capture_mode"] == "block_fp8_profile"
        assert result["shape_capture"]["source_profile_trace"] == str(selected_trace)

    @pytest.mark.parametrize("profile_status", ["failed", ""])
    def test_vllm_block_fp8_rejects_unusable_profile_state(
        self,
        tmp_path,
        profile_status,
    ):
        profile_trace = tmp_path / "profile.trace.json"
        stale_trace = tmp_path / "stale.trace.json"
        steady_trace = tmp_path / "mixed_steady_state.trace.json"
        for path in (profile_trace, stale_trace, steady_trace):
            path.write_text(json.dumps({"traceEvents": []}))
        state = SharedState(
            framework="vllm",
            precision="fp8",
            model_path="/models/qwen",
            last_profile_trace=str(profile_trace),
            last_profile_status=profile_status or "succeeded",
            last_trace_analyze={
                "trace_input": (str(profile_trace) if profile_status else str(stale_trace)),
                "steady_state_trace": str(steady_trace),
            },
        )
        state.last_profile_workload = state.current_profile_workload_context()

        assert (
            krh._reuse_vllm_block_fp8_roofline_shapes(
                state,
                workspace=tmp_path / "gemm",
            )
            is None
        )

    @pytest.mark.parametrize(
        ("field", "profile_value"),
        [
            ("model_path", "/models/old-qwen"),
            ("tp", 2),
            ("conc", 32),
            ("isl", 512),
            ("osl", 512),
            ("max_model_len", 2048),
        ],
    )
    def test_vllm_block_fp8_rejects_roofline_trace_after_workload_change(
        self,
        tmp_path,
        field,
        profile_value,
    ):
        roofline_trace = tmp_path / "roofline.trace.json"
        roofline_trace.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {
                            "name": "vllm::w8a8_triton_block_scaled_mm_func",
                            "args": {"Input Dims": [[4149, 5120], [34816, 5120]]},
                        }
                    ]
                }
            )
        )
        state = SharedState(
            precision="fp8",
            framework="vllm",
            model_path="/models/qwen",
            tp=1,
            conc=64,
            isl=1024,
            osl=1024,
            max_model_len=4096,
            last_profile_trace=str(roofline_trace),
            last_profile_status="succeeded",
            last_trace_analyze={
                "trace_input": str(roofline_trace),
                "steady_state_trace": str(roofline_trace),
            },
        )
        state.last_profile_workload = state.current_profile_workload_context()
        state.last_profile_workload[field] = profile_value

        assert (
            krh._reuse_vllm_block_fp8_roofline_shapes(
                state,
                workspace=tmp_path / "gemm",
            )
            is None
        )

    @pytest.mark.parametrize(
        ("recorded_runtime", "current_best"),
        [
            (
                {"base_extra_args": "--attention-backend TRITON"},
                {"extra_server_args": "--attention-backend AITER"},
            ),
            (
                {"base_extra_envs": {"VLLM_ROCM_USE_AITER_LINEAR": "0"}},
                {"extra_envs": {"VLLM_ROCM_USE_AITER_LINEAR": "1"}},
            ),
            (
                {"base_remove_args": ["--old-backend"]},
                {"remove_args": ["--other-backend"]},
            ),
            (
                {"base_unset_envs": ["OLD_BACKEND"]},
                {"unset_envs": ["OTHER_BACKEND"]},
            ),
            (
                {"base_args_mode": "append"},
                {"args_mode": "replace"},
            ),
        ],
    )
    def test_vllm_block_fp8_rejects_roofline_trace_after_backend_change(
        self,
        tmp_path,
        recorded_runtime,
        current_best,
    ):
        roofline_trace = tmp_path / "roofline.trace.json"
        roofline_trace.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {
                            "name": "vllm::w8a8_triton_block_scaled_mm_func",
                            "args": {"Input Dims": [[4149, 5120], [34816, 5120]]},
                        }
                    ]
                }
            )
        )
        state = SharedState(
            precision="fp8",
            framework="vllm",
            model_path="/models/qwen",
            tp=1,
            conc=64,
            isl=1024,
            osl=1024,
            max_model_len=4096,
            last_profile_trace=str(roofline_trace),
            last_profile_status="succeeded",
            last_trace_analyze={
                "trace_input": str(roofline_trace),
                "steady_state_trace": str(roofline_trace),
            },
        )
        state.last_profile_workload = state.profile_workload_context(recorded_runtime)
        state.current_best = current_best

        assert (
            krh._reuse_vllm_block_fp8_roofline_shapes(
                state,
                workspace=tmp_path / "gemm",
                current_workload=state.current_profile_workload_context(),
            )
            is None
        )

    def test_vllm_block_fp8_logs_when_roofline_has_no_shapes(self, tmp_path, caplog):
        roofline_trace = tmp_path / "mixed_steady_state.trace.json"
        roofline_trace.write_text(json.dumps({"traceEvents": [{"name": "vllm::unrelated_op", "args": {}}]}))
        state = SharedState(
            precision="fp8",
            framework="vllm",
            model_path="/models/qwen",
            tp=1,
            conc=64,
            isl=1024,
            osl=1024,
            max_model_len=4096,
            last_profile_trace=str(roofline_trace),
            last_profile_status="succeeded",
            last_trace_analyze={
                "trace_input": str(roofline_trace),
                "steady_state_trace": str(roofline_trace),
            },
        )
        state.last_profile_workload = state.profile_workload_context()
        caplog.set_level(20, logger=krh.__name__)

        assert (
            krh._reuse_vllm_block_fp8_roofline_shapes(
                state,
                workspace=tmp_path / "gemm",
            )
            is None
        )
        assert "contains no reusable block-FP8 shapes" in caplog.text

    def test_vllm_block_fp8_reuses_decode_only_selected_by_roofline(self, tmp_path):
        decode_trace = tmp_path / "decode_only_steady_state.trace.json"
        decode_trace.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {
                            "name": "vllm::w8a8_triton_block_scaled_mm_func",
                            "args": {"Input Dims": [[64, 5120], [34816, 5120]]},
                        }
                    ]
                }
            )
        )
        state = SharedState(
            precision="fp8",
            framework="vllm",
            model_path="/models/qwen",
            tp=1,
            conc=64,
            isl=1024,
            osl=1024,
            max_model_len=4096,
            last_profile_trace=str(decode_trace),
            last_profile_status="succeeded",
            last_trace_analyze={
                "trace_input": str(decode_trace),
                "steady_state_trace": str(decode_trace),
            },
        )
        state.last_profile_workload = state.current_profile_workload_context()

        reused = krh._reuse_vllm_block_fp8_roofline_shapes(
            state,
            workspace=tmp_path / "gemm",
        )

        assert reused is not None
        assert reused["source_profile_trace"] == str(decode_trace)
        assert reused["shape_count"] == 1

    def test_vllm_block_fp8_routes_profile_shapes_to_aiter(self, tmp_path, monkeypatch):
        root = tmp_path / "kernel-agent"
        tool = root / "tools" / "forge_gemm_tuning.py"
        tool.parent.mkdir(parents=True)
        tool.write_text("# placeholder\n")
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(root))
        monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")

        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3ForCausalLM"],
                    "hidden_size": 5120,
                    "intermediate_size": 17408,
                    "quantization_config": {
                        "quant_method": "fp8",
                        "weight_block_size": [128, 128],
                    },
                }
            )
        )
        unrelated_trace = tmp_path / "roofline.trace.json"
        unrelated_trace.write_text(json.dumps({"traceEvents": [{"name": "vllm::unrelated_op", "args": {}}]}))
        SharedState(
            precision="fp8",
            framework="vllm",
            model_path=str(model),
            gpu_type="mi325x",
            tp=1,
            conc=64,
            isl=1024,
            osl=1024,
            last_profile_trace=str(unrelated_trace),
        ).save(tmp_path)
        shapes_path = tmp_path / "profile_shapes.json"
        shapes_path.write_text(json.dumps([{"M": 4149, "N": 34816, "K": 5120}]))
        stale_untuned = tmp_path / "stale_untuned.csv"
        stale_untuned.write_text("M,N,K\n1,5120,5120\n")
        captured_input: dict = {}

        async def fake_capture(*, state, session_dir, payload, workspace):
            return {
                "status": "ok",
                "shapes_json": str(shapes_path),
                "shape_count": 1,
                "shape_capture_workspace": str(workspace / "shape_capture"),
            }

        async def fake_run(cmd: list[str], *, timeout_sec: int):
            input_path = Path(cmd[cmd.index("--input-json") + 1])
            captured_input.update(json.loads(input_path.read_text()))
            return (
                0,
                "FORGE_GEMM_TUNE_RESULT_BEGIN\n"
                + json.dumps(
                    {
                        "status": "ok",
                        "micro_decision": "no_improvement",
                        "tuners_run": [{"tuner_name": "a8w8_blockscale"}],
                    }
                )
                + "\nFORGE_GEMM_TUNE_RESULT_END\n",
                "",
            )

        monkeypatch.setattr(krh, "_capture_vllm_tunableop_shapes", fake_capture)
        monkeypatch.setattr(krh, "_forge_gemm_tune_available", lambda: True)
        monkeypatch.setattr(
            krh,
            "_resolve_forge_untuned_csv",
            lambda *args, **kwargs: str(stale_untuned),
        )
        monkeypatch.setattr(krh, "_run_subprocess", fake_run)

        result = asyncio.run(krh.run_gemm_tuning_handler({"task_id": "capture"}, session_dir=tmp_path))

        assert captured_input["framework"] == "vllm-aiter"
        # Captured shapes are routed to the aiter family, then re-keyed onto the
        # padded M values the runtime actually looks up.
        assert result["shape_alignment"]["source_shapes_json"] == str(shapes_path)
        aligned = json.loads(Path(captured_input["shapes_json"]).read_text())
        assert {row["M"] for row in aligned} == {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 4224, 8192}
        assert captured_input["untuned_csv"] == ""
        assert captured_input["tunableop_input"] == ""
        assert result["shape_capture"]["shape_count"] == 1
        assert result["status"] == "ok"

    def test_handler_passes_non_fp8_geak_to_next_hyperloom_prereq(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMM_TUNING_BACKEND", "geak")
        monkeypatch.delenv("HYPERLOOM_KERNEL_AGENT_ROOT", raising=False)
        state = SharedState(precision="bf16", framework="sglang")
        state.save(tmp_path)

        result = asyncio.run(krh.run_gemm_tuning_handler({}, session_dir=tmp_path))

        assert result["status"] == "failed"
        assert result["error_class"] == "kernel_agent_root_missing"

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
            krh._run_geak_gemm_tuning(
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
            krh._run_geak_gemm_tuning(
                {"dry_run": True, "task_id": "auto"},
                session_dir=tmp_path,
            )
        )

        assert result["status"] == "ok"

    def test_geak_without_config_does_not_fall_back_to_forge(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMM_TUNING_BACKEND", "geak")
        monkeypatch.delenv("GEAK_CONFIG", raising=False)
        root = tmp_path / "kernel-agent"
        root.mkdir()
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

        async def fail_geak_subprocess(cmd, *, timeout_sec):
            raise AssertionError("legacy GEAK subprocess should not run without config")

        monkeypatch.setattr(krh, "_run_subprocess", fail_geak_subprocess)

        result = asyncio.run(krh._run_geak_gemm_tuning({"task_id": "legacy-geak"}, session_dir=tmp_path))

        assert result["status"] == "skipped"
        assert result["backend"] == "geak"
        assert result["decision"] == "REVERT"
        assert result["error_class"] == "legacy_geak_config_missing"

    def test_forge_uses_runtime_fp8_blockscale_for_aiter_backend(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
        monkeypatch.setenv("GEMM_TUNING_BACKEND", "forge")
        model_dir = tmp_path / "qwen"
        model_dir.mkdir()
        state = SharedState(
            precision="bf16",
            framework="sglang",
            model_path=str(model_dir),
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

        result = asyncio.run(krh.run_gemm_tuning_handler({"task_id": "forge"}, session_dir=tmp_path))

        cmd = captured["cmd"]  # type: ignore[assignment]
        input_path = cmd[cmd.index("--input-json") + 1]
        data = json.loads(Path(input_path).read_text())
        assert data["precision"] == "fp8"
        # Do not force blockscale from Hyperloom; forge decides (auto).
        assert data["quant_type"] == "auto"
        assert data["conc"] == 256
        assert result["extra_envs"] == {"AITER_CONFIG_FMOE": "/tmp/fmoe.csv"}

    def test_handler_writes_gemm_tuning_audit_row(self, tmp_path, monkeypatch):
        """run_gemm_tuning_handler appends a source-attribution audit row backfilled as a ``gemm_tuning:<engine>`` span."""
        from hyperloom.inference_optimizer.session.session_paths import gemm_tuning_steps_path

        monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
        monkeypatch.setenv("GEMM_TUNING_BACKEND", "forge")
        model_dir = tmp_path / "qwen"
        model_dir.mkdir()
        state = SharedState(
            precision="bf16",
            framework="sglang",
            model_path=str(model_dir),
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
        # A clean run gains nothing it did not carry before, bar status.
        assert "error" not in row["tuners_run"][0]
        assert "error_class" not in row["tuners_run"][0]

    def test_forge_uses_per_token_only_for_explicit_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
        monkeypatch.setenv("GEMM_TUNING_BACKEND", "forge")
        model_dir = tmp_path / "qwen"
        model_dir.mkdir()
        state = SharedState(
            precision="bf16",
            framework="sglang",
            model_path=str(model_dir),
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
        monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
        monkeypatch.setenv("GEMM_TUNING_BACKEND", "forge")
        model_dir = tmp_path / "moe"
        model_dir.mkdir()
        state = SharedState(
            precision="bf16",
            framework="sglang",
            model_path=str(model_dir),
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

    @pytest.mark.parametrize(
        "quant_type",
        [
            "bpreshuffle",
            "a8w8_bpreshuffle",
            "blockscale_bpreshuffle",
            "a8w8_blockscale_bpreshuffle",
        ],
    )
    def test_vllm_block_fp8_rejects_sglang_bpreshuffle_types(self, quant_type):
        assert krh._is_vllm_block_fp8("fp8", quant_type) is False

    def test_resolve_forge_shapes_extracts_only_latest_trace(self, tmp_path):
        session_dir = tmp_path / "session"
        runs = session_dir / "kernel-agent" / "runs"
        old_candidates = runs / "old" / "kernel_candidates.json"
        latest_candidates = runs / "latest" / "kernel_candidates.json"
        old_candidates.parent.mkdir(parents=True)
        latest_candidates.parent.mkdir(parents=True)

        def _write_candidates(path: Path, m: int) -> None:
            path.write_text(
                json.dumps(
                    {
                        "hot_kernels": [
                            {
                                "name": "aiter::gemm_a8w8_blockscale",
                                "input_shapes": [{"shape": f"({m},5120) fp8<br>(34816,5120) fp8"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

        _write_candidates(old_candidates, 16384)
        _write_candidates(latest_candidates, 4096)
        state = SharedState(last_trace_analyze={"candidates_path": str(latest_candidates)})

        shapes_path = krh._resolve_forge_shapes(state, session_dir)

        assert json.loads(Path(shapes_path).read_text(encoding="utf-8")) == [{"M": 4096, "N": 34816, "K": 5120}]

    def test_extract_gemm_shapes_tolerates_whitespace_after_comma(self, tmp_path):
        # TraceLens may render tuples with a space after the comma
        # ("(1024, 5120)"); the extractor must still parse M/N/K instead of
        # silently dropping every shape and falling back to config defaults.
        candidates = tmp_path / "kernel_candidates.json"
        candidates.write_text(
            json.dumps(
                {
                    "hot_kernels": [
                        {
                            "name": "aiter::gemm_a8w8_blockscale",
                            "input_shapes": [{"shape": "(1024, 5120) fp8<br>(34816, 5120) fp8"}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        out = krh._extract_gemm_shapes_from_candidates(str(candidates), tmp_path)

        assert json.loads(Path(out).read_text(encoding="utf-8")) == [{"M": 1024, "N": 34816, "K": 5120}]

    def test_extract_gemm_shapes_parses_per_tensor_input_shapes(self, tmp_path):
        # Current TraceLens emits one entry per tensor ({"call_num": .., "shape":
        # "(M,K) fp8"}) instead of a single <br>-joined string. The old parser
        # skipped every such entry and returned no shapes, so GEMM tuning fell
        # back to a lossy profile capture that recorded only a large prefill M
        # and missed the throughput-dominant decode M (observed on
        # Qwen3.5-122B-A10B-FP8: tuned M=2095 only -> -18.45% E2E).
        candidates = tmp_path / "kernel_candidates.json"
        candidates.write_text(
            json.dumps(
                {
                    "hot_kernels": [
                        {
                            "name": "aiter::gemm_a8w8_blockscale_ck",
                            "input_shapes": [
                                {"call_num": 48, "shape": "(3126,512) fp8"},
                                {"call_num": 48, "shape": "(3072,512) fp8"},
                                {"call_num": 48, "shape": "(0,) fp32"},
                            ],
                        },
                        {
                            "name": "aiter::gemm_a8w8_blockscale_ck",
                            "input_shapes": [
                                {"call_num": 1440, "shape": "(64,3072) fp8"},
                                {"call_num": 1440, "shape": "(10240,3072) fp8"},
                                {"call_num": 1440, "shape": "(0,) fp32"},
                            ],
                        },
                        {
                            # Matrix-vector head: highest call count, but N==1 is
                            # not a tunable GEMM tile and must be dropped.
                            "name": "vllm::rocm_unquantized_gemm",
                            "input_shapes": [
                                {"call_num": 9999, "shape": "(64,3072) bf16"},
                                {"call_num": 9999, "shape": "(1,3072) bf16"},
                            ],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        out = krh._extract_gemm_shapes_from_candidates(str(candidates), tmp_path)

        assert out, "per-tensor input_shapes yielded no GEMM shapes"
        shapes = json.loads(Path(out).read_text(encoding="utf-8"))
        # The decode shape is present (it was dropped entirely before) and, being
        # the most-called, is tuned first when the tuner runs out of budget.
        assert shapes == [
            {"M": 64, "N": 10240, "K": 3072},
            {"M": 3126, "N": 3072, "K": 512},
        ]

    @staticmethod
    def _mixed_dtype_candidates(path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "hot_kernels": [
                        {
                            "name": "aiter::gemm_a8w8_blockscale_ck",
                            "input_shapes": [
                                {"call_num": 360, "shape": "(64,3072) fp8"},
                                {"call_num": 360, "shape": "(8704,3072) fp8"},
                            ],
                        },
                        {
                            # BF16 router head: most-called, so without dtype
                            # scoping it displaces the FP8 shape being tuned.
                            "name": "vllm::rocm_unquantized_gemm",
                            "input_shapes": [
                                {"call_num": 1440, "shape": "(64,3072) bf16"},
                                {"call_num": 1440, "shape": "(256,3072) bf16"},
                            ],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

    def test_extract_gemm_shapes_scopes_to_tuned_precision(self, tmp_path):
        candidates = tmp_path / "kernel_candidates.json"
        self._mixed_dtype_candidates(candidates)

        out = krh._extract_gemm_shapes_from_candidates(str(candidates), tmp_path, precision="fp8")

        assert json.loads(Path(out).read_text(encoding="utf-8")) == [{"M": 64, "N": 8704, "K": 3072}]

    def test_extract_gemm_shapes_scopes_to_bf16_precision(self, tmp_path):
        candidates = tmp_path / "kernel_candidates.json"
        self._mixed_dtype_candidates(candidates)

        out = krh._extract_gemm_shapes_from_candidates(str(candidates), tmp_path, precision="bf16")

        assert json.loads(Path(out).read_text(encoding="utf-8")) == [{"M": 64, "N": 256, "K": 3072}]

    def test_extract_gemm_shapes_keeps_every_dtype_without_precision(self, tmp_path):
        candidates = tmp_path / "kernel_candidates.json"
        self._mixed_dtype_candidates(candidates)

        out = krh._extract_gemm_shapes_from_candidates(str(candidates), tmp_path)

        assert json.loads(Path(out).read_text(encoding="utf-8")) == [
            {"M": 64, "N": 256, "K": 3072},
            {"M": 64, "N": 8704, "K": 3072},
        ]

    @pytest.mark.parametrize(
        ("precision", "traced"),
        [
            ("fp8", "fp8_e4m3"),
            ("fp8", "e4m3fnuz"),
            ("fp8", "torch.float8_e4m3fn"),
            ("fp16", "f16"),  # what this repo's _TRACE_DTYPE_SUFFIX emits
            ("bf16", "b16"),
            ("mxfp4", "fp4x2"),
            ("fp4", "mxfp4"),
        ],
    )
    def test_extract_gemm_shapes_matches_dtype_aliases(self, tmp_path, precision, traced):
        """A precision and a traced token spell the same dtype many ways; exact
        string matching would drop shapes that do belong to the tuned family."""
        candidates = tmp_path / "kernel_candidates.json"
        candidates.write_text(
            json.dumps(
                {
                    "hot_kernels": [
                        {
                            "name": "aiter::gemm",
                            "input_shapes": [
                                {"call_num": 8, "shape": f"(64,3072) {traced}"},
                                {"call_num": 8, "shape": f"(8704,3072) {traced}"},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        out = krh._extract_gemm_shapes_from_candidates(str(candidates), tmp_path, precision=precision)

        assert out, f"{precision!r} rejected the equivalent traced token {traced!r}"
        assert json.loads(Path(out).read_text(encoding="utf-8")) == [{"M": 64, "N": 8704, "K": 3072}]

    def test_extract_gemm_shapes_keeps_the_highest_call_count(self, tmp_path):
        """The same (M,N,K) can be reported by several kernels; the hot sighting
        must win, otherwise a rare first one buries the real decode hotspot."""
        candidates = tmp_path / "kernel_candidates.json"
        candidates.write_text(
            json.dumps(
                {
                    "hot_kernels": [
                        {
                            "name": "aiter::gemm_cold_first",
                            "input_shapes": [
                                {"call_num": 10, "shape": "(64,3072) fp8"},
                                {"call_num": 10, "shape": "(8704,3072) fp8"},
                            ],
                        },
                        {
                            "name": "aiter::gemm_other",
                            "input_shapes": [
                                {"call_num": 100, "shape": "(64,3072) fp8"},
                                {"call_num": 100, "shape": "(10240,3072) fp8"},
                            ],
                        },
                        {
                            "name": "aiter::gemm_same_shape_hot",
                            "input_shapes": [
                                {"call_num": 1000, "shape": "(64,3072) fp8"},
                                {"call_num": 1000, "shape": "(8704,3072) fp8"},
                            ],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        out = krh._extract_gemm_shapes_from_candidates(str(candidates), tmp_path)

        assert json.loads(Path(out).read_text(encoding="utf-8")) == [
            {"M": 64, "N": 8704, "K": 3072},  # 1000, not the first sighting's 10
            {"M": 64, "N": 10240, "K": 3072},  # 100
        ]

    @pytest.mark.parametrize("sep", ["<br>", "<br/>", "<BR/>", "<BR>"])
    def test_extract_gemm_shapes_accepts_legacy_separator_spellings(self, tmp_path, sep):
        candidates = tmp_path / "kernel_candidates.json"
        candidates.write_text(
            json.dumps(
                {
                    "hot_kernels": [
                        {
                            "name": "aiter::gemm_a8w8_blockscale",
                            "input_shapes": [{"shape": f"(1024, 5120) fp8{sep}(34816, 5120) fp8"}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        out = krh._extract_gemm_shapes_from_candidates(str(candidates), tmp_path)

        assert out, f"separator {sep!r} was not recognised"
        assert json.loads(Path(out).read_text(encoding="utf-8")) == [{"M": 1024, "N": 34816, "K": 5120}]

    def test_resolve_forge_shapes_prefers_scoped_candidates_over_untyped_artifact(self, tmp_path):
        """A pre-rendered shapes artifact carries no dtype, so it must not win
        over candidates that were actually scoped to the tuned precision."""
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        artifact = tmp_path / "shapes.json"  # recorded from the BF16 head
        artifact.write_text(json.dumps([{"M": 64, "N": 256, "K": 3072}]), encoding="utf-8")
        candidates = tmp_path / "kernel_candidates.json"
        candidates.write_text(
            json.dumps(
                {
                    "hot_kernels": [
                        {
                            "name": "aiter::gemm_a8w8_blockscale_ck",
                            "input_shapes": [
                                {"call_num": 360, "shape": "(64,3072) fp8"},
                                {"call_num": 360, "shape": "(8704,3072) fp8"},
                            ],
                        },
                        {
                            "name": "vllm::rocm_unquantized_gemm",
                            "input_shapes": [
                                {"call_num": 1440, "shape": "(64,3072) bf16"},
                                {"call_num": 1440, "shape": "(256,3072) bf16"},
                            ],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        state = SharedState(
            last_trace_analyze={
                "shapes_json": str(artifact),
                "candidates_path": str(candidates),
            }
        )

        resolved = krh._resolve_forge_shapes(state, session_dir, precision="fp8")

        assert json.loads(Path(resolved).read_text(encoding="utf-8")) == [{"M": 64, "N": 8704, "K": 3072}]

    def test_resolve_forge_shapes_still_uses_artifact_when_unscoped(self, tmp_path):
        """Without a precision to scope by, the explicit artifact keeps priority."""
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        artifact = tmp_path / "shapes.json"
        artifact.write_text(json.dumps([{"M": 64, "N": 256, "K": 3072}]), encoding="utf-8")
        state = SharedState(last_trace_analyze={"shapes_json": str(artifact)})

        assert krh._resolve_forge_shapes(state, session_dir) == str(artifact)


# _default_kernel_batch_parallel — adaptive batch fanout scaling with visible GPUs.
class TestDefaultKernelBatchParallel:
    @pytest.fixture
    def patch_torch(self, monkeypatch):
        """Returns a setter that overrides ``torch.cuda.device_count`` and ``$KERNEL_AGENT_NUM_GPUS``."""
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
            # Full-node match: cap kicks in at 8.
            (8, 1, 8),
            # Partial node -> floor at the visible-GPU count.
            (4, 1, 4),
            # 8-GPU node with 4-GPU reservations -> 2 concurrent.
            (8, 4, 2),
            # 4-GPU pod with 2-GPU per task -> 2 concurrent.
            (4, 2, 2),
            # Larger-than-cap node -> cap still kicks in.
            (16, 1, 8),
            # Per-task larger than visible -> floor at 1.
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


# _should_parallelize_backends — backends never auto-parallelize; False unless
# forced via payload ``parallel_backends`` or env ``KERNEL_OPT_PARALLEL_BACKENDS``.


class TestShouldParallelizeBackends:
    @pytest.fixture
    def patch_torch(self, monkeypatch):
        """Override ``torch.cuda.device_count`` + ``$KERNEL_AGENT_NUM_GPUS`` and clear the env override."""
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
        "n_gpus, per_task, num_candidates",
        [
            # No auto-parallelize regardless of GPU count: without an explicit
            # override the decision is always sequential (False).
            (8, 1, 3),
            (8, 1, 100),
            (2, 1, 1),
            (1, 1, 1),
            (16, 8, 1),
        ],
    )
    def test_no_auto_parallelize_without_override(
        self,
        patch_torch,
        n_gpus,
        per_task,
        num_candidates,
    ):
        patch_torch(n_gpus, per_task=per_task)
        assert krh._should_parallelize_backends({}, num_candidates) is False

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


class TestReconcileKernelId:
    CANDS = [
        {"kernel_id": "k001", "name": "aten::mm"},
        {"kernel_id": "k010", "name": "aiter::rmsnorm"},
    ]

    def test_exact_id_kept(self):
        assert krh._reconcile_kernel_id("k010", self.CANDS) == "k010"

    def test_name_match_kept(self):
        # An exact operator-name match is canonicalized to the stable k00x id.
        assert krh._reconcile_kernel_id("aten::mm", self.CANDS) == "k001"

    def test_normalized_prefix_resolves_to_real_id(self):
        assert krh._reconcile_kernel_id("kn001", self.CANDS) == "k001"
        assert krh._reconcile_kernel_id("rn010", self.CANDS) == "k010"

    def test_missing_id_falls_back_to_first(self):
        assert krh._reconcile_kernel_id("", self.CANDS) == "k001"
        assert krh._reconcile_kernel_id(None, self.CANDS) == "k001"

    def test_hallucinated_id_is_left_for_guard_or_cli_skip(self):
        # Non-empty ids are never guessed; a pure hallucination is left untouched.
        assert krh._reconcile_kernel_id("aiter.silu_and_mul", self.CANDS) == "aiter.silu_and_mul"
        assert (
            krh._reconcile_kernel_id("framework_sglang_silu_and_mul_m64", self.CANDS)
            == "framework_sglang_silu_and_mul_m64"
        )


class TestReconcileKernelIdForSingleBatch:
    CANDS = [
        {
            "kernel_id": "k002",
            "name": "_fwd_grouped_kernel_stage1",
            "shape_provenance": "launch_grid",
        },
    ]

    def test_pins_mismatched_id_to_sole_candidate(self):
        kid, pinned = krh._reconcile_kernel_id_for_single_batch("k003", self.CANDS)
        assert kid == "k002"
        assert pinned is True

    def test_keeps_exact_match(self):
        kid, pinned = krh._reconcile_kernel_id_for_single_batch("k002", self.CANDS)
        assert kid == "k002"
        assert pinned is False


# _resolve_candidate_id / _all_kernel_candidates — canonicalizes an aliased id
# against the full hot ∪ skipped set (no fallback).
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
        # ``aten::mm`` is non-unique and non-routable -> leave untouched ("").
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

    def test_dedups_skipped_subset_of_hot(self, tmp_path):
        # ``hot_kernels`` is the full ranked set and ``skipped_kernels`` its
        # non-routable subset (they overlap); each kernel must be counted once.
        cp = tmp_path / "kc.json"
        hot = [
            {"kernel_id": "k001", "name": "moe", "reusable_native_kernel": True},
            {"kernel_id": "k002", "name": "aten::mm", "reusable_native_kernel": False},
            {"kernel_id": "k003", "name": "aiter::rmsnorm", "reusable_native_kernel": False},
        ]
        skipped = [dict(c) for c in hot if not c["reusable_native_kernel"]]  # subset of hot
        cp.write_text(json.dumps({"hot_kernels": hot, "skipped_kernels": skipped}), encoding="utf-8")
        out = krh._all_kernel_candidates({"candidates_path": str(cp)})
        # Each kernel exactly once, hot order preserved.
        assert [c["kernel_id"] for c in out] == ["k001", "k002", "k003"]
        assert len(out) == 3

    def test_dedups_by_name_when_kernel_id_missing(self, tmp_path):
        # Fall back to ``name`` when ``kernel_id`` is absent; a row with neither
        # id nor name is never dropped.
        cp = tmp_path / "kc.json"
        cp.write_text(
            json.dumps(
                {
                    "hot_kernels": [{"name": "moe"}, {"name": "aten::mm"}, {"gpu_pct": 1.0}],
                    "skipped_kernels": [{"name": "aten::mm"}, {"gpu_pct": 2.0}],
                }
            ),
            encoding="utf-8",
        )
        out = krh._all_kernel_candidates({"candidates_path": str(cp)})
        names = [c.get("name") for c in out]
        # "aten::mm" appears once; the two identity-less rows are both kept.
        assert names.count("aten::mm") == 1
        assert names.count("moe") == 1
        assert sum(1 for c in out if not (c.get("kernel_id") or c.get("name"))) == 2


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

    def test_geometry_only_shape_excluded_from_batch(self, tmp_path):
        # A reusable kernel with a resolved source but shape_dispatchable=False
        # fails the kernel-opt gate; it must be dropped before dispatch.
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
                            "shape_dispatchable": False,
                        },
                        {
                            "kernel_id": "k002",
                            "gpu_pct": 11.0,
                            "reusable_native_kernel": True,
                            "source_file": "/p/attn_op.py",
                            "shape_dispatchable": True,
                        },
                    ],
                    "reusable_native_kernel_ids": ["k001", "k002"],
                }
            ),
            encoding="utf-8",
        )
        out = krh._batch_kernel_candidates({"candidates_path": str(cp)}, session_dir=tmp_path)
        assert [item["kernel_id"] for item in out] == ["k002"]

    def test_missing_shape_dispatchable_stays_batch_eligible(self, tmp_path):
        # TraceLens candidates omit shape_dispatchable; absent field must not be
        # filtered so the main path is preserved.
        cp = self._write_candidates(tmp_path)  # no shape_dispatchable key
        out = krh._batch_kernel_candidates({"candidates_path": str(cp)}, session_dir=tmp_path)
        assert [item["kernel_id"] for item in out] == ["k001"]


class TestTracelensRootResolution:
    """TraceLens root is resolved/validated independently of inherited env."""

    def test_resolve_uses_explicit_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path / "tl"))
        assert krh._resolve_tracelens_root() == tmp_path / "tl"

    def test_resolve_derives_from_open_source_root_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TRACELENS_ROOT", raising=False)
        monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "podlocal"))
        expected = tmp_path / "podlocal" / "TraceLens"
        assert krh._resolve_tracelens_root() == expected

    def test_root_error_none_when_present(self, tmp_path):
        tl = tmp_path / "tl"
        (tl / ".git").mkdir(parents=True)  # usable git checkout
        assert krh._tracelens_root_error(tl) is None

    def test_root_error_message_when_missing(self, tmp_path):
        err = krh._tracelens_root_error(tmp_path / "ghost")
        assert err is not None
        assert "TraceLens root not found" in err

    def test_root_error_message_when_incomplete(self, tmp_path):
        # Dir exists but is not a git checkout (no .git) -> unusable.
        tl = tmp_path / "tl"
        tl.mkdir()
        err = krh._tracelens_root_error(tl)
        assert err is not None
        assert "incomplete" in err

    def test_trace_analyze_handler_selfheals_default_root_then_fails_if_unrecovered(self, tmp_path, monkeypatch):
        # Default root missing: handler attempts self-heal before failing.
        # Stub the heal to a no-op so the handler returns the structured error.
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(tmp_path))
        monkeypatch.delenv("TRACELENS_ROOT", raising=False)
        monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "no-tracelens-here"))
        called = {"n": 0}

        def _fake_heal(root, *, log=None):
            called["n"] += 1

        monkeypatch.setattr(krh, "_maybe_selfheal_tracelens_root", _fake_heal)
        out = asyncio.run(
            krh.trace_analyze_handler(
                {"trace_input": str(tmp_path / "trace"), "analysis_route": "agent"}, session_dir=tmp_path
            )
        )
        assert called["n"] == 1  # self-heal was attempted
        assert out["status"] == "failed"
        assert out["error_class"] == "tracelens_root_missing"

    def test_trace_analyze_handler_selfheals_incomplete_default_root(self, tmp_path, monkeypatch):
        # an incomplete default checkout (dir present, no .git) must still
        # trigger self-heal.
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(tmp_path))
        monkeypatch.delenv("TRACELENS_ROOT", raising=False)
        monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "podlocal"))
        # Create an incomplete default checkout: the dir exists but has no .git.
        incomplete = tmp_path / "podlocal" / "TraceLens"
        incomplete.mkdir(parents=True)
        (incomplete / "partial").write_text("half", encoding="utf-8")
        called = {"n": 0}

        def _fake_heal(root, *, log=None):
            called["n"] += 1
            # Simulate an unrecoverable heal so the handler fail-fasts here.
            shutil.rmtree(root, ignore_errors=True)

        monkeypatch.setattr(krh, "_maybe_selfheal_tracelens_root", _fake_heal)
        out = asyncio.run(
            krh.trace_analyze_handler(
                {"trace_input": str(tmp_path / "trace"), "analysis_route": "agent"}, session_dir=tmp_path
            )
        )
        assert called["n"] == 1  # self-heal attempted despite the dir existing
        assert out["status"] == "failed"
        assert out["error_class"] == "tracelens_root_missing"

    def test_trace_analyze_handler_failfast_on_incomplete_override(self, tmp_path, monkeypatch):
        # an incomplete non-default operator override (dir present, no .git)
        # must fail fast — never adopted, never auto-cloned.
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(tmp_path))
        monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "podlocal"))
        override = tmp_path / "operator-tl"
        override.mkdir()
        (override / "partial").write_text("half", encoding="utf-8")  # no .git
        monkeypatch.setenv("TRACELENS_ROOT", str(override))
        heal_called = {"n": 0}
        monkeypatch.setattr(
            krh,
            "_maybe_selfheal_tracelens_root",
            lambda *_a, **_k: heal_called.__setitem__("n", heal_called["n"] + 1),
        )
        out = asyncio.run(
            krh.trace_analyze_handler(
                {"trace_input": str(tmp_path / "trace"), "analysis_route": "agent"}, session_dir=tmp_path
            )
        )
        assert out["status"] == "failed"
        assert out["error_class"] == "tracelens_root_missing"

    def test_selfheal_skips_non_default_override(self, tmp_path, monkeypatch):
        # An operator override at a NON-default path is never auto-cloned, even
        # though TRACELENS_ROOT is set in env. Inject a counting fake module so a
        # regression that reaches _ensure_tracelens_checkout would trip the assert.
        monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "podlocal"))
        override = tmp_path / "operator-tl"
        monkeypatch.setenv("TRACELENS_ROOT", str(override))
        called = {"n": 0}

        def _fake_ensure(root, *, log_path=None):
            called["n"] += 1

        import sys as _sys
        import types as _types

        fake_mod = _types.ModuleType("tracelens_analysis")
        fake_mod._ensure_tracelens_checkout = _fake_ensure  # type: ignore[attr-defined]
        _sys.modules["tracelens_analysis"] = fake_mod
        monkeypatch.setattr(
            krh, "_kernel_agent_tool_path", lambda *_a, **_k: tmp_path / "tools" / "tracelens_analysis.py"
        )
        try:
            krh._maybe_selfheal_tracelens_root(override)
        finally:
            _sys.modules.pop("tracelens_analysis", None)
        assert called["n"] == 0

    def test_selfheal_runs_on_default_path_even_when_env_set(self, tmp_path, monkeypatch):
        # the default path is persisted as TRACELENS_ROOT in
        # kernel-agent.env.sh, so "env set" must NOT be treated as an override.
        # A missing default path must still attempt self-heal.
        monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "podlocal"))
        default_root = tmp_path / "podlocal" / "TraceLens"
        monkeypatch.setenv("TRACELENS_ROOT", str(default_root))
        called = {"n": 0, "root": None}

        def _fake_ensure(root, *, log_path=None):
            called["n"] += 1
            called["root"] = Path(root)

        # Route _kernel_agent_tool_path to a fake module exposing
        # _ensure_tracelens_checkout.
        import sys as _sys
        import types as _types

        fake_mod = _types.ModuleType("tracelens_analysis")
        fake_mod._ensure_tracelens_checkout = _fake_ensure  # type: ignore[attr-defined]
        _sys.modules["tracelens_analysis"] = fake_mod
        monkeypatch.setattr(
            krh, "_kernel_agent_tool_path", lambda *_a, **_k: tmp_path / "tools" / "tracelens_analysis.py"
        )
        try:
            krh._maybe_selfheal_tracelens_root(default_root)
        finally:
            _sys.modules.pop("tracelens_analysis", None)
        assert called["n"] == 1
        assert called["root"] == default_root

    def test_selfheal_runs_on_pinned_at_sha_default(self, tmp_path, monkeypatch):
        # install.sh clones the default checkout as TraceLens@<sha>; a vanished
        # per-revision default must still self-heal, not be misread as an
        # operator override (the bare-only default_root check missed @sha dirs).
        monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "podlocal"))
        default_root = tmp_path / "podlocal" / "TraceLens@deadbeef"
        monkeypatch.setenv("TRACELENS_ROOT", str(default_root))
        called = {"n": 0, "root": None}

        def _fake_ensure(root, *, log_path=None):
            called["n"] += 1
            called["root"] = Path(root)

        import sys as _sys
        import types as _types

        fake_mod = _types.ModuleType("tracelens_analysis")
        fake_mod._ensure_tracelens_checkout = _fake_ensure  # type: ignore[attr-defined]
        _sys.modules["tracelens_analysis"] = fake_mod
        monkeypatch.setattr(
            krh, "_kernel_agent_tool_path", lambda *_a, **_k: tmp_path / "tools" / "tracelens_analysis.py"
        )
        try:
            krh._maybe_selfheal_tracelens_root(default_root)
        finally:
            _sys.modules.pop("tracelens_analysis", None)
        assert called["n"] == 1
        assert called["root"] == default_root


class TestKernelOptArtifactBundleRecording:
    def test_materialize_unified_patch_snapshot_for_forge_fusion_patch(self, tmp_path):
        repo = tmp_path / "framework"
        repo.mkdir()
        (repo / "model.py").write_text("old = 1\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.email=a@b.c", "-c", "user.name=t", "commit", "-qm", "base"],
            check=True,
        )
        # The live tree is already dirty; snapshot materialization must start from HEAD.
        (repo / "model.py").write_text("new = 2\n", encoding="utf-8")
        (repo / "model_fused.py").write_text("fused = True\n", encoding="utf-8")
        patch = tmp_path / "fusion.patch"
        patch.write_text(
            "\n".join(
                [
                    "diff --git a/model.py b/model.py",
                    "--- a/model.py",
                    "+++ b/model.py",
                    "@@ -1 +1 @@",
                    "-old = 1",
                    "+new = 2",
                    "diff --git a/model_fused.py b/model_fused.py",
                    "new file mode 100644",
                    "index 0000000..1111111",
                    "--- /dev/null",
                    "+++ b/model_fused.py",
                    "@@ -0,0 +1 @@",
                    "+fused = True",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        snap = Path(
            krh.materialize_unified_patch_snapshot(
                patch_path=patch,
                repo_root=repo,
                snapshot_dir=tmp_path / "snapshot",
            )
        )

        assert (snap / "model.py").read_text(encoding="utf-8") == "new = 2\n"
        assert (snap / "model_fused.py").read_text(encoding="utf-8") == "fused = True\n"
        assert (repo / "model.py").read_text(encoding="utf-8") == "new = 2\n"

    def test_record_kernel_opt_persists_best_artifact_bundle(self):
        state = SharedState()
        bundle = {
            "type": "patch_snapshot",
            "snapshot_dir": "/tmp/snap",
            "patch_path": "/tmp/best.patch",
            "repo_root": "/repo",
            "write_paths": ["aiter/ops/moe.py", "benchmarks/bench_moe.py"],
            "delete_paths": [],
        }
        kd.record_kernel_opt(
            state,
            {
                "status": "completed",
                "kernel_id": "k001",
                "source_file": "/repo/aiter/ops/moe.py",
                "verification": {
                    "micro_speedup": 1.25,
                    "compile_passed": True,
                    "correctness_passed": True,
                    "best_backend": "forge",
                    "best_artifact_path": "/repo/aiter/ops/moe.py",
                    "best_artifact_bundle": bundle,
                    "deploy_snapshot_dir": "/tmp/snap",
                    "deploy_patch_path": "/tmp/best.patch",
                    "deploy_repo_root": "/repo",
                },
                "proposal": {"decision": "KEEP", "reasons": ["ok"]},
            },
        )

        assert state.kernel_opt_attempts["k001"]["last_artifact_bundle"] == bundle
        assert state.last_kernel_opt["best_artifact_bundle"] == bundle

    def test_resolve_integrate_payload_uses_last_kernel_artifact_bundle(self, tmp_path):
        bundle = {
            "type": "patch_snapshot",
            "snapshot_dir": "/tmp/snap",
            "patch_path": "/tmp/best.patch",
            "repo_root": "/repo",
        }
        state = SharedState.load_or_init(tmp_path)
        state.last_kernel_opt = {
            "kernel_id": "k001",
            "source_file": "/repo/aiter/ops/moe.py",
            "best_artifact_bundle": bundle,
        }
        state.save(tmp_path)

        resolved, error = krh._resolve_integrate_payload({"kernel_id": "k001"}, session_dir=tmp_path)

        assert error is None
        assert resolved["snapshot_dir"] == "/tmp/snap"
        assert resolved["patch_path"] == "/tmp/best.patch"
        assert resolved["kernel_repo"] == "/repo"
        assert resolved["source_file"] == "/repo/aiter/ops/moe.py"

    def test_resolve_integrate_payload_uses_per_kernel_artifact_bundle(self, tmp_path):
        bundle = {
            "type": "patch_snapshot",
            "snapshot_dir": "/tmp/snap2",
            "patch_path": "/tmp/queued.patch",
            "repo_root": "/repo2",
        }
        state = SharedState.load_or_init(tmp_path)
        state.last_kernel_opt = {"kernel_id": "other"}
        state.kernel_opt_attempts = {
            "k002": {
                "last_source_file": "/repo2/aiter/ops/queued.py",
                "last_artifact_bundle": bundle,
            }
        }
        state.save(tmp_path)

        resolved, error = krh._resolve_integrate_payload({"kernel_id": "k002"}, session_dir=tmp_path)

        assert error is None
        assert resolved["snapshot_dir"] == "/tmp/snap2"
        assert resolved["patch_path"] == "/tmp/queued.patch"
        assert resolved["kernel_repo"] == "/repo2"
        assert resolved["source_file"] == "/repo2/aiter/ops/queued.py"


class TestBuildTraceAnalyzeCmd:
    """argv golden for ``_build_trace_analyze_cmd``: the splitter (non-scriptable)
    and diffusion (scriptable) surfaces, plus the bypass vs TraceLens difference."""

    def _common(self, monkeypatch, tmp_path):
        monkeypatch.delenv("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", raising=False)
        monkeypatch.delenv("MODEL_PATH", raising=False)
        monkeypatch.setattr(krh, "_kernel_agent_tool_path", lambda name: Path("/tools") / name)
        state = SharedState()
        return state, tmp_path / "sess"

    def test_tracelens_splitter_cmd(self, monkeypatch, tmp_path):
        state, session_dir = self._common(monkeypatch, tmp_path)
        cmd, steady = krh._build_trace_analyze_cmd(
            {"session_id": "sid", "trace_input": "/t/trace", "split_conc": "64", "split_osl": "1024"},
            session_dir=session_dir,
            state=state,
            workspace_path="/ws",
            trace_input="/t/trace",
            tracelens_root=Path("/tl"),
            is_bypass=False,
            scriptable=False,
            workload={"conc": 8, "osl": 256, "random_range_ratio": "0.5"},
            model_name="Qwen",
            framework="sglang",
            target_platform="MI300X",
            analysis_mode="inference",
        )
        assert cmd == [
            "python3",
            "/tools/tracelens_analysis.py",
            "--trace-input",
            "/t/trace",
            "--session-id",
            "sid",
            "--workspace-path",
            "/ws",
            "--tracelens-root",
            "/tl",
            "--model-name",
            "Qwen",
            "--framework",
            "sglang",
            "--target-platform",
            "MI300X",
            "--analysis-mode",
            "inference",
            # splitter hints: payload override wins over workload metadata.
            "--split-conc",
            "64",
            "--split-osl",
            "1024",
            "--split-r",
            "0.5",
        ]
        assert steady == ""

    def test_sglang_cmd_forwards_model_context(self, monkeypatch, tmp_path):
        """Non-scriptable analysis receives model config and precision context."""
        state, session_dir = self._common(monkeypatch, tmp_path)
        state.model_path = "/models/sglang-model"
        state.precision = "fp8"
        state.baseline_config_path = "/session/materialized.yaml"
        cmd, _steady = krh._build_trace_analyze_cmd(
            {"trace_input": "/t/trace"},
            session_dir=session_dir,
            state=state,
            workspace_path="/ws",
            trace_input="/t/trace",
            tracelens_root=Path("/tl"),
            is_bypass=False,
            scriptable=False,
            workload={"precision": "mxfp8"},
            model_name="MiniMax",
            framework="sglang",
            target_platform="MI355X",
            analysis_mode="default",
        )
        assert cmd[cmd.index("--model-path") + 1] == "/models/sglang-model"
        assert cmd[cmd.index("--precision") + 1] == "fp8"
        assert cmd[cmd.index("--runtime-config") + 1] == "/session/materialized.yaml"

    def test_bypass_scriptable_cmd(self, monkeypatch, tmp_path):
        state, session_dir = self._common(monkeypatch, tmp_path)
        state.model_path = "/models/flux"
        cmd, steady = krh._build_trace_analyze_cmd(
            {
                "session_id": "sid",
                "trace_input": "/t/trace",
                "num_denoise_steps": 30,
                "precision": "bf16",
                "steady_state_mode": "auto",
                "dry_run": True,
            },
            session_dir=session_dir,
            state=state,
            workspace_path="/ws",
            trace_input="/t/trace",
            tracelens_root=None,
            is_bypass=True,
            scriptable=True,
            workload={},
            model_name="",
            framework="",
            target_platform="",
            analysis_mode="",
        )
        # bypass tool name; no --tracelens-root and no --skip-split.
        assert cmd[1] == "/tools/bypass_trace_analysis.py"
        assert "--tracelens-root" not in cmd
        assert "--skip-split" not in cmd
        # scriptable forwards denoise/model/precision; not the splitter hints.
        assert "--num-denoise-steps" in cmd and cmd[cmd.index("--num-denoise-steps") + 1] == "30"
        assert "--model-path" in cmd and cmd[cmd.index("--model-path") + 1] == "/models/flux"
        assert "--precision" in cmd and cmd[cmd.index("--precision") + 1] == "bf16"
        assert "--split-conc" not in cmd
        assert "--runtime-config" not in cmd
        assert "--steady-state-mode" in cmd and cmd[cmd.index("--steady-state-mode") + 1] == "auto"
        assert cmd[-1] == "--dry-run"
        assert steady == "auto"

    def test_steady_state_mode_from_env(self, monkeypatch, tmp_path):
        state, session_dir = self._common(monkeypatch, tmp_path)
        monkeypatch.setenv("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", "median")
        cmd, steady = krh._build_trace_analyze_cmd(
            {"trace_input": "/t/trace"},
            session_dir=session_dir,
            state=state,
            workspace_path="/ws",
            trace_input="/t/trace",
            tracelens_root=Path("/tl"),
            is_bypass=False,
            scriptable=False,
            workload={},
            model_name="",
            framework="",
            target_platform="",
            analysis_mode="",
        )
        assert steady == "median"
        assert cmd[cmd.index("--steady-state-mode") + 1] == "median"
        # session-id falls back to the session dir name when payload omits it.
        assert cmd[cmd.index("--session-id") + 1] == session_dir.name


def test_forge_loop_budget_bounds_are_read_not_copied():
    """A local copy of an upstream bound plans against a budget forge-loop ignores."""
    from hyperloom.orchestrator.kernel import request_handlers as rh

    # Read from a module that is always importable, so the test does not depend
    # on KernelForge being installed alongside.
    assert rh._forge_loop_constant("math", "pi", 0.0) == pytest.approx(3.14159, abs=1e-4)
    assert rh._forge_loop_constant("no.such.module", "X", 7.0) == 7.0
    assert rh._forge_loop_constant("math", "no_such_name", 7.0) == 7.0
    assert rh._forge_loop_constant("math", "isfinite", 7.0) == 7.0


def test_a_patched_rebaseline_gets_the_cold_start_budget(monkeypatch):
    """Applying a patch moves the JIT cache aside, so the next boot recompiles."""
    from hyperloom.orchestrator.actions.executors import _aiter_jit as aiter_jit
    from hyperloom.orchestrator.kernel import request_handlers as rh

    monkeypatch.setattr(
        aiter_jit,
        "probe_aiter_jit_cache",
        lambda: {"probe_status": "found", "is_cold": True, "kernel_count": 0},
    )

    assert rh._cold_start_rebaseline_timeout(600) == aiter_jit.BASELINE_COLD_START_TIMEOUT_SEC


def test_a_warm_cache_keeps_the_resolved_timeout(monkeypatch):
    """Only an empty cache justifies the cold cap; a warm boot keeps its budget."""
    from hyperloom.orchestrator.actions.executors import _aiter_jit as aiter_jit
    from hyperloom.orchestrator.kernel import request_handlers as rh

    for probe in (
        {"probe_status": "found", "is_cold": False, "kernel_count": 90},
        {"probe_status": "not_found", "is_cold": None, "kernel_count": 0},
        {"probe_status": "error", "is_cold": None, "kernel_count": 0},
    ):
        monkeypatch.setattr(aiter_jit, "probe_aiter_jit_cache", lambda p=probe: p)
        assert rh._cold_start_rebaseline_timeout(600) == 600


def test_a_longer_explicit_budget_is_never_shortened(monkeypatch):
    """The cap is a floor for a cold boot, not a ceiling on the operator's budget."""
    from hyperloom.orchestrator.actions.executors import _aiter_jit as aiter_jit
    from hyperloom.orchestrator.kernel import request_handlers as rh

    monkeypatch.setattr(
        aiter_jit,
        "probe_aiter_jit_cache",
        lambda: {"probe_status": "found", "is_cold": True, "kernel_count": 0},
    )
    generous = aiter_jit.BASELINE_COLD_START_TIMEOUT_SEC + 1200

    assert rh._cold_start_rebaseline_timeout(generous) == generous
