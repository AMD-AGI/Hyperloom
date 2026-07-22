"""Tests for durable Forge invocation-spec extraction and propagation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS_DIR))
import _invocation_spec as invocation_spec  # noqa: E402
import kernel_optimization  # noqa: E402

_BACKENDS_DIR = _TOOLS_DIR / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import forge_submit  # noqa: E402


def _candidate(tmp_path: Path) -> dict:
    repo = tmp_path / "repo"
    benchmark = repo / "tests" / "test_scaled_gemm.py"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text(
        """
@perftest(num_iters=100)
def run_torch(x, weight, x_scale, w_scale, dtype):
    return F.linear(x.float(), weight.float()).to(dtype)

@perftest(num_iters=100)
def run_gemm(x, weight, x_scale, w_scale, dtype):
    return aiter.gemm_a8w8_blockscale(x, weight, x_scale, w_scale, dtype)

@benchmark()
def test_gemm(dtype, m, n, k):
    return run_gemm(x, weight, x_scale, w_scale, dtype)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["ExampleForCausalLM"],
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 40960,
                "quantization_config": {
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "kernel_id": "k002",
        "name": "scaled_gemm",
        "kernel_category": "gemm",
        "source_type": "python",
        "kernel_kind": "triton",
        "source_file": "kernels/internal.py",
        "kernel_repo": str(repo),
        "kernel_sources": ["kernels/internal.py"],
        "device_kernel_name": "_scaled_gemm_kernel",
        "device_kernel_names": ["_scaled_gemm_kernel"],
        "launcher_source_file": "launchers/op.py(42): public_op",
        "tracelens_launcher_path": "launchers/op.py(42): public_op",
        "input_shapes": [
            {"call_num": 1240, "shape": "(64,17408) fp8"},
            {"call_num": 1240, "shape": "(5120,17408) fp8"},
            {"call_num": 1240, "shape": "(64,136) fp32"},
            {"call_num": 1240, "shape": "(40,136) fp32"},
        ],
        "input_dtypes": [
            "c10::Float8_e4m3fnuz",
            "c10::Float8_e4m3fnuz",
            "float",
            "float",
        ],
        "output_shapes": [[64, 5120]],
        "output_dtypes": ["c10::BFloat16"],
        "benchmark_files": ["tests/test_scaled_gemm.py"],
        "runtime_args": {
            "model": str(model),
            "materialized_config": "runtime/baseline.yaml",
            "precision": "bf16",
            "api_token": "must-not-leak",
            "workload": {
                "conc": 64,
                "num_prompts": 320,
                "num_warmups": 8,
                "tp": 1,
                "isl": 1024,
                "osl": 1024,
                "max_model_len": 6144,
            },
        },
        "runtime_flags": {"target_platform": "MI325X"},
        "framework": "sglang",
        "shape_provenance": "torch_trace",
    }


def test_builds_compact_operator_contract_with_absolute_paths(tmp_path):
    candidate = _candidate(tmp_path)
    repo = Path(candidate["kernel_repo"])
    spec = invocation_spec.build_invocation_spec(
        candidate,
        test_command="python generated/driver.py --correctness",
    )

    assert invocation_spec.invocation_spec_filename(candidate) == "invocation_spec_scaled_gemm.json"
    assert spec["status"] == "complete"
    assert [row["shape"] for row in spec["invocation"]["arguments"]] == [
        [64, 17408],
        [5120, 17408],
        [64, 136],
        [40, 136],
    ]
    assert [row["dtype"] for row in spec["invocation"]["arguments"]] == [
        "fp8",
        "fp8",
        "fp32",
        "fp32",
    ]
    assert spec["invocation"]["outputs"][0] == {
        "path": "outputs[0]",
        "position": 0,
        "shape": [64, 5120],
        "dtype": "bf16",
        "dtype_raw": "c10::BFloat16",
        "raw": "[64, 5120]",
        "source_row": 0,
    }
    assert spec["edit_target"]["source_file"] == str(repo / "kernels" / "internal.py")
    assert spec["invocation"]["launcher_source_file"] == str(repo / "launchers" / "op.py")
    primary = spec["tests"]["primary_benchmark"]
    assert primary["kernel_function"] == "run_gemm"
    assert primary["reference_function"] == "run_torch"
    assert primary["public_call_targets"] == ["aiter.gemm_a8w8_blockscale"]
    assert primary["reference_call_targets"] == ["F.linear"]
    assert spec["tests"]["related_files"] == [str(repo / "tests" / "test_scaled_gemm.py")]
    assert str(repo / "generated" / "driver.py") in spec["tests"]["selected_test_command"]
    assert spec["execution"] == {
        "framework": "sglang",
        "precision": "bf16",
        "target_platform": "MI325X",
        "is_multigpu": False,
    }
    deployment = spec["deployment"]
    assert deployment["batch"]["serving_concurrency"] == 64
    assert deployment["sequence"]["request_tokens"] == 2048
    assert deployment["model"]["config_summary"]["hidden_size"] == 5120
    assert "config" not in deployment["model"]
    assert "runtime" not in spec
    assert "benchmark_evidence" not in spec["tests"]
    assert "must-not-leak" not in json.dumps(spec)


def test_recovers_complete_source_and_runtime_symbols(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("def _scaled_gemm_kernel(x):\n    return x\n", encoding="utf-8")
    tracelens_dir = tmp_path / "analysis-run" / "tracelens"
    tracelens_dir.mkdir(parents=True)
    analysis = tracelens_dir / "analysis.md"
    analysis.write_text("# Analysis\n", encoding="utf-8")
    trace_dir = tmp_path / "profile" / "torch_trace"
    trace_dir.mkdir(parents=True)
    (tracelens_dir.parent / "trace_input_manifest.json").write_text(
        json.dumps({"trace_input": str(trace_dir)}),
        encoding="utf-8",
    )
    full_symbol = "_scaled_gemm_kernel_BLOCK_M_128_BLOCK_N_128_BLOCK_K_64"
    (trace_dir.parent / "benchmark_report.json").write_text(
        json.dumps({"kernel_summary": [{"name": full_symbol}]}),
        encoding="utf-8",
    )

    spec = invocation_spec.build_invocation_spec(
        {
            "name": "scaled_gemm",
            "source_file": str(source),
            "device_kernel_name": "_scaled_gemm_kernel_BLOCK_M_128...",
            "device_kernel_names": ["_scaled_gemm_kernel_BLOCK_M_128..."],
            "trace_report_path": str(analysis),
        }
    )

    assert spec["edit_target"]["source_symbol"] == "_scaled_gemm_kernel"
    assert spec["edit_target"]["runtime_symbols"] == [full_symbol]
    assert "unresolved_runtime_symbol_prefixes" not in spec["edit_target"]
    assert "..." not in json.dumps(spec["edit_target"])


def test_missing_optional_context_is_fail_soft(tmp_path):
    spec = invocation_spec.build_invocation_spec(
        {
            "name": "broken/context",
            "kernel_repo": str(tmp_path),
            "input_shapes": [{"shape": "(dynamic,K) unknown"}],
            "benchmark_files": [None, {"not": "a path"}],
            "device_kernel_names": 123,
            "runtime_args": {"model": "remote/model-id", "workload": {"isl": "unknown"}},
        },
        test_command="'unterminated",
    )

    assert spec["status"] == "partial"
    assert spec["deployment"]["model"]["model_id"] == "remote/model-id"
    assert "sequence" not in spec["deployment"]


def test_forge_invoke_persists_and_passes_operator_spec(tmp_path, monkeypatch):
    source = tmp_path / "kernel.py"
    source.write_text("def kernel(x):\n    return x\n", encoding="utf-8")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Task\n", encoding="utf-8")
    out_dir = tmp_path / "forge-output"
    candidate = _candidate(tmp_path)
    candidate["source_file"] = str(source)
    candidate["kernel_repo"] = str(tmp_path)
    captured: dict = {}

    def fake_submit(**kwargs):
        captured.update(kwargs)
        return {"returncode": 0, "stdout": "ok", "stdout_tail": "ok", "stderr_tail": ""}

    monkeypatch.setattr(kernel_optimization, "_forge_output_dir", lambda *_args: out_dir)
    monkeypatch.setattr(kernel_optimization, "ray_available", lambda: False)
    monkeypatch.setattr(
        kernel_optimization,
        "_import_backend",
        lambda _name: SimpleNamespace(submit=fake_submit),
    )

    result = kernel_optimization.invoke_backend(
        "forge",
        prompt,
        str(source),
        argparse.Namespace(
            budget_minutes=60,
            num_gpus=1,
            test_command="",
            session_id="session",
        ),
        candidate=candidate,
        log_path=tmp_path / "run.log",
    )

    spec_path = out_dir / "invocation_spec_scaled_gemm.json"
    assert spec_path.is_file()
    assert result["invocation_spec_path"] == str(spec_path)
    assert captured["invocation_spec_file"] == str(spec_path)


def test_forge_loop_cli_receives_absolute_spec_path(tmp_path, monkeypatch):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("def kernel(x):\n    return x\n", encoding="utf-8")
    driver = tmp_path / "driver.py"
    driver.write_text("# driver\n", encoding="utf-8")
    spec_path = tmp_path / "invocation_spec_scaled_gemm.json"
    spec_path.write_text('{"schema_version": 1}\n', encoding="utf-8")
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(
            returncode=0,
            stdout='__FORGE_RESULT__{"baseline_ms": 1.0, "best_ms": 0.9, "improved": true}',
            stderr="",
        )

    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")
    monkeypatch.setattr(forge_submit, "_apply_fellow_env", lambda _env: None)
    monkeypatch.setattr(forge_submit.subprocess, "run", fake_run)

    result = forge_submit._run_loop_via_cli(
        worktree_kernel=str(kernel),
        driver=str(driver),
        workspace=str(tmp_path),
        shapes={"primary": {"M": 64}},
        snr_threshold=30.0,
        max_iters=8,
        max_hours=1.0,
        branch="forge/session/scaled_gemm",
        gpu_target="gfx942",
        fellow="triton-fellow",
        program_md_file="",
        invocation_spec_file=str(spec_path),
        experiments_dir=tmp_path / "experiments",
        forge_log=tmp_path / "forge.log",
        timeout_s=60,
    )

    cmd = captured["cmd"]
    option_index = cmd.index("--invocation-spec-file")
    assert cmd[option_index + 1] == str(spec_path.resolve())
    assert (result[0], result[1], result[2], result[4]) == (1.0, 0.9, True, None)
