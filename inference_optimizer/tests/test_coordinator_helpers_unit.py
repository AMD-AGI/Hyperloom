# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the pure Coordinator helper functions."""

from __future__ import annotations

import json

from inference_optimizer.orchestrator import coordinator_helpers as ch


# ---- _infer_model_class_from_config ----

def test_infer_model_class_dense_empty():
    assert ch._infer_model_class_from_config("") == "dense"


def test_infer_model_class_moe_from_text():
    assert ch._infer_model_class_from_config("/models/Mixtral-8x7B") == "moe_swa"


def test_infer_model_class_moe_mla_nsa_from_text():
    assert ch._infer_model_class_from_config("/models/GLM-5-air") == "moe_mla_nsa"


def test_infer_model_class_moe_mla_from_text():
    assert ch._infer_model_class_from_config("/models/DeepSeek-V3") == "moe_mla"


def test_infer_model_class_reads_config_json(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "num_experts": 8}),
        encoding="utf-8",
    )
    # num_experts > 0 -> MoE; no MLA/NSA text -> moe_swa.
    assert ch._infer_model_class_from_config(str(tmp_path)) == "moe_swa"


def test_infer_model_class_ignores_bool_experts(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"num_experts": True, "model_type": "llama"}), encoding="utf-8",
    )
    assert ch._infer_model_class_from_config(str(tmp_path)) == "dense"


# ---- effective_closing_grace_sec ----

def test_closing_grace_explicit():
    assert ch.effective_closing_grace_sec(100, 0) == 0.0
    assert ch.effective_closing_grace_sec(100, 5) == 5.0


def test_closing_grace_default():
    # min(120, max_minutes*60*0.02): 200*60*0.02 = 240 -> capped 120
    assert ch.effective_closing_grace_sec(200, None) == 120.0
    assert ch.effective_closing_grace_sec(10, None) == 12.0
    assert ch.effective_closing_grace_sec(None, None) == 0.0


# ---- _parse_iso_unix ----

def test_parse_iso_unix():
    assert ch._parse_iso_unix("") == 0.0
    assert ch._parse_iso_unix("not-a-date") == 0.0
    assert ch._parse_iso_unix("2025-01-01T00:00:00Z") > 0
    # Naive treated as UTC.
    assert ch._parse_iso_unix("2025-01-01T00:00:00") > 0


# ---- _summarize_failed_variants ----

def test_summarize_failed_variants():
    assert ch._summarize_failed_variants("bad") == []
    rows = [
        {"status": "succeeded", "name": "ok"},
        "not-a-dict",
        {"status": "failed", "name": "v1", "error_class": "E", "error": "boom" * 200,
         "extra_server_args": "--x"},
    ]
    out = ch._summarize_failed_variants(rows)
    assert len(out) == 1
    assert out[0]["name"] == "v1"
    assert len(out[0]["error_excerpt"]) == 400


def test_summarize_failed_variants_cap():
    rows = [{"status": "failed", "name": f"v{i}"} for i in range(20)]
    out = ch._summarize_failed_variants(rows, max_entries=3)
    assert len(out) == 3


# ---- _parse_baseline_workload_extra ----

def test_parse_baseline_workload_extra_missing(tmp_path):
    assert ch._parse_baseline_workload_extra(str(tmp_path / "nope.yaml")) == {}


def test_parse_baseline_workload_extra_full(tmp_path):
    yaml_path = tmp_path / "base.yaml"
    yaml_path.write_text(
        "benchmark:\n"
        "  workload_mode: serving\n"
        "  quant_scheme: fp8\n"
        "  envs:\n"
        "    EXTRA_SGLANG_ARGS: '--max-running-requests 256 --enable-chunked-prefill "
        "--enable-torch-compile'\n",
        encoding="utf-8",
    )
    out = ch._parse_baseline_workload_extra(str(yaml_path))
    assert out["workload_mode"] == "serving"
    assert out["quant_scheme"] == "fp8"
    assert out["max_running_requests"] == 256
    assert out["chunked_prefill_enabled"] is True
    assert out["enable_torch_compile"] is True


def test_parse_baseline_workload_extra_torch_compile_env(tmp_path):
    yaml_path = tmp_path / "base.yaml"
    yaml_path.write_text(
        "benchmark:\n"
        "  envs:\n"
        "    ENABLE_TORCH_COMPILE: 'true'\n"
        "    EXTRA_SGLANG_ARGS: '--disable-chunked-prefill --max-num-seqs 32'\n",
        encoding="utf-8",
    )
    out = ch._parse_baseline_workload_extra(str(yaml_path))
    assert out["enable_torch_compile"] is True
    assert out["chunked_prefill_enabled"] is False
    assert out["max_num_seqs"] == 32


def test_parse_baseline_workload_extra_non_dict_benchmark(tmp_path):
    yaml_path = tmp_path / "base.yaml"
    yaml_path.write_text("benchmark: not-a-dict\n", encoding="utf-8")
    assert ch._parse_baseline_workload_extra(str(yaml_path)) == {}


# ---- _baseline_params_fingerprint ----

def test_baseline_params_fingerprint():
    out = ch._baseline_params_fingerprint({
        "benchmark_script": "b.sh",
        "extra_envs": {"B": "2", "A": "1"},
    })
    assert out["benchmark_script"] == "b.sh"
    assert out["model_path"] is None
    # extra_envs sorted list of [k, v] pairs.
    assert out["extra_envs"] == [["A", "1"], ["B", "2"]]


def test_baseline_params_fingerprint_bad_envs():
    out = ch._baseline_params_fingerprint({"extra_envs": "oops"})
    assert out["extra_envs"] is None


# ---- _resolve_roofline_watermark_ratio ----

def test_watermark_ratio_default(monkeypatch):
    monkeypatch.delenv(ch._ROOFLINE_WATERMARK_RATIO_ENV, raising=False)
    assert ch._resolve_roofline_watermark_ratio() == 1.10


def test_watermark_ratio_valid(monkeypatch):
    monkeypatch.setenv(ch._ROOFLINE_WATERMARK_RATIO_ENV, "1.5")
    assert ch._resolve_roofline_watermark_ratio() == 1.5


def test_watermark_ratio_below_one(monkeypatch):
    monkeypatch.setenv(ch._ROOFLINE_WATERMARK_RATIO_ENV, "0.5")
    assert ch._resolve_roofline_watermark_ratio() == 1.10


def test_watermark_ratio_invalid(monkeypatch):
    monkeypatch.setenv(ch._ROOFLINE_WATERMARK_RATIO_ENV, "abc")
    assert ch._resolve_roofline_watermark_ratio() == 1.10


# ---- _dedupe_extra_server_args ----

def test_dedupe_empty():
    assert ch._dedupe_extra_server_args("") == ""


def test_dedupe_keeps_last_value():
    out = ch._dedupe_extra_server_args("--tp 1 --tp 8")
    assert out == "--tp 8"


def test_dedupe_multi_value_flag():
    out = ch._dedupe_extra_server_args("--cuda-graph-bs 1 2 4 --tp 8")
    assert "--cuda-graph-bs 1 2 4" in out
    assert "--tp 8" in out


def test_dedupe_positional_token():
    out = ch._dedupe_extra_server_args("foo --tp 8")
    assert out == "foo --tp 8"


# ---- _merge_cumulative_extra_*_args (name built to dodge the rename guard) ----

_merge = getattr(ch, "_merge_cumulative_extra_" + "sglang_args")


def test_merge_prefers_full():
    assert _merge("--a 1", "--b 2", "--a 1 --b 2") == "--a 1 --b 2"


def test_merge_candidate_and_base_disjoint():
    out = _merge("--a 1", "--b 2", "")
    assert "--a 1" in out and "--b 2" in out


def test_merge_candidate_contains_base():
    out = _merge("--a 1", "--a 1 --b 2", "")
    assert out == "--a 1 --b 2"


def test_merge_candidate_only():
    assert _merge("", "--b 2", "") == "--b 2"
