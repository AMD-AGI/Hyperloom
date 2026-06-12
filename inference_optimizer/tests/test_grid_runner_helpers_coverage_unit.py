# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Supplemental coverage for _grid_runner pure helpers: multi-node cuda-graph
advisory, compatibility filter model-class drop, runtime override env branches,
report parsing, and per-variant yaml env injection."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from inference_optimizer.orchestrator.action_executors import _grid_runner as gr
from inference_optimizer.orchestrator.action_executors import _multi_node_env


def _variant(name: str, args: str = "", envs: dict | None = None) -> gr.GridVariant:
    return gr.GridVariant(name=name, extra_server_args=args, extra_envs=envs or {})


# -- annotate_multi_node_cuda_graph_max_bs --------------------------------
def test_annotate_multi_node_not_multi_node(monkeypatch) -> None:
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: False)
    assert gr.annotate_multi_node_cuda_graph_max_bs([_variant("v")]) == []


def test_annotate_multi_node_invalid_conc_defaults(monkeypatch) -> None:
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: True)
    monkeypatch.setenv("CONC", "not-an-int")
    grid = [_variant("v", "--cuda-graph-max-bs 8")]
    notes = gr.annotate_multi_node_cuda_graph_max_bs(grid)
    # falls back to CONC=64; 8 < 64 -> advisory emitted
    assert len(notes) == 1 and notes[0]["source"] == "multi_node_advisory"


def test_annotate_multi_node_nonpositive_conc(monkeypatch) -> None:
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: True)
    monkeypatch.setenv("CONC", "0")
    assert gr.annotate_multi_node_cuda_graph_max_bs([_variant("v", "--cuda-graph-max-bs 8")]) == []


# -- apply_compatibility_filter -------------------------------------------
def test_compatibility_filter_drops_on_model_class(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PATH", "meta-llama-3-8b")
    monkeypatch.setattr(gr, "_detect_model_class", lambda mp: (False, False))
    monkeypatch.setattr(gr, "_probe_server_help_text", lambda fw: "")
    grid = [_variant("mla", "--enable-flashinfer-mla"), _variant("plain", "")]
    kept, dropped = gr.apply_compatibility_filter(grid)
    assert [v.name for v in kept] == ["plain"]
    assert len(dropped) == 1
    assert dropped[0]["source"] == "compatibility_filter"
    assert "MLA" in dropped[0]["reason"]


def test_compatibility_filter_drops_on_missing_help_flag(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PATH", "deepseek-moe")
    monkeypatch.setattr(gr, "_detect_model_class", lambda mp: (True, True))
    monkeypatch.setattr(gr, "_probe_server_help_text", lambda fw: "--some-other-flag")
    grid = [_variant("moe", "--enable-ep-moe")]
    kept, dropped = gr.apply_compatibility_filter(grid)
    assert kept == []
    assert "too old" in dropped[0]["reason"]


def test_compatibility_filter_no_model_path_assumes_compatible(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setattr(gr, "_probe_server_help_text", lambda fw: "--enable-ep-moe")
    grid = [_variant("moe", "--enable-ep-moe")]
    kept, dropped = gr.apply_compatibility_filter(grid)
    assert [v.name for v in kept] == ["moe"] and dropped == []


# -- apply_runtime_benchmark_overrides ------------------------------------
def test_runtime_overrides_model_precision_and_gpu_no_framework(monkeypatch) -> None:
    monkeypatch.setenv("PRECISION", "fp8")
    for k in ("ISL", "OSL", "MAX_MODEL_LEN", "TP", "CONC", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(k, raising=False)
    bench: dict = {}  # no framework -> benchmark_script popped
    gr.apply_runtime_benchmark_overrides(
        bench, model_path="/models/x", gpu_type="mi355x",
    )
    assert bench["model"] == "/models/x"
    assert bench["precision"] == "fp8"
    assert bench["runner_type"] == "mi355x"
    assert "benchmark_script" not in bench


def test_runtime_overrides_framework_pins_generic_script(monkeypatch) -> None:
    for k in ("PRECISION", "ISL", "OSL", "MAX_MODEL_LEN", "TP", "CONC", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(k, raising=False)
    bench = {"framework": "sglang"}
    gr.apply_runtime_benchmark_overrides(bench, gpu_type="mi300x")
    assert bench["benchmark_script"] == "sglang_mi300x.sh"


def test_runtime_overrides_env_ints_and_rocr_autofill(monkeypatch) -> None:
    for k in ("PRECISION",):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ISL", "128")
    monkeypatch.setenv("TP", "2")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    bench: dict = {}
    envs = gr.apply_runtime_benchmark_overrides(bench)
    assert envs["ISL"] == 128
    assert envs["TP"] == 2
    # TP>1 with no explicit ROCR -> auto-filled device list
    assert envs["ROCR_VISIBLE_DEVICES"] == "0,1"


def test_runtime_overrides_explicit_benchmark_script_wins(monkeypatch) -> None:
    for k in ("PRECISION", "ISL", "OSL", "MAX_MODEL_LEN", "TP", "CONC", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(k, raising=False)
    bench = {"framework": "vllm"}
    gr.apply_runtime_benchmark_overrides(
        bench, gpu_type="mi300x", benchmark_script="custom.sh",
    )
    assert bench["benchmark_script"] == "custom.sh"


# -- _parse_report ---------------------------------------------------------
def test_parse_report_missing(tmp_path: Path) -> None:
    assert gr._parse_report(tmp_path) is None


def test_parse_report_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "benchmark_report.json").write_text("{bad", encoding="utf-8")
    assert gr._parse_report(tmp_path) is None


def test_parse_report_non_dict(tmp_path: Path) -> None:
    (tmp_path / "benchmark_report.json").write_text("[1,2,3]", encoding="utf-8")
    assert gr._parse_report(tmp_path) is None


def test_parse_report_valid(tmp_path: Path) -> None:
    (tmp_path / "benchmark_report.json").write_text(
        json.dumps({"output_throughput": 100.0}), encoding="utf-8",
    )
    assert gr._parse_report(tmp_path) == {"output_throughput": 100.0}


# -- _build_variant_yaml ---------------------------------------------------
def test_build_variant_yaml_injects_extra_envs(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {}}}),
        encoding="utf-8",
    )
    variant = _variant("v1", "--foo 1", envs={"USE_AITER": "1"})
    out = gr._build_variant_yaml(
        base, "", variant, output_subdir=tmp_path / "v1",
    )
    cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
    envs = cfg["benchmark"]["envs"]
    assert envs["USE_AITER"] == "1"
    # variant + base server args merged into the framework's args env
    arg_key = gr.server_args_env_name("sglang")
    assert "--foo 1" in envs[arg_key]


def test_build_variant_yaml_dedupes_repeated_flags(tmp_path: Path) -> None:
    """#520: when base YAML + base_extra_args + variant all set the same flag,
    the materialized YAML must contain each flag only once (last wins)."""
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({
            "benchmark": {
                "framework": "vllm",
                "envs": {"EXTRA_VLLM_ARGS": "--attention-backend ROCM_ATTN"},
            },
        }),
        encoding="utf-8",
    )
    variant = _variant("v1", "--attention-backend ROCM_AITER_FA")
    out = gr._build_variant_yaml(
        base, "", variant, output_subdir=tmp_path / "v1",
    )
    cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
    args = cfg["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert args.count("--attention-backend") == 1, f"duplicate flag: {args}"
    assert "ROCM_AITER_FA" in args, "last-wins should keep variant value"


def test_build_variant_yaml_preserves_json_arg(tmp_path: Path) -> None:
    """Dedupe must not break JSON-valued args like --json-model-override-args."""
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({
            "benchmark": {
                "framework": "sglang",
                "envs": {
                    "EXTRA_SGLANG_ARGS": (
                        '--json-model-override-args \'{"rope_scaling":null}\''
                        " --context-length 8192"
                    ),
                },
            },
        }),
        encoding="utf-8",
    )
    variant = _variant("v1", "--context-length 4096")
    out = gr._build_variant_yaml(
        base, "", variant, output_subdir=tmp_path / "v1",
    )
    cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
    args = cfg["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"]
    assert "--json-model-override-args" in args
    assert "rope_scaling" in args, f"JSON value mangled: {args}"
    assert args.count("--context-length") == 1, f"duplicate flag: {args}"
    assert "4096" in args, "last-wins should keep variant value"
