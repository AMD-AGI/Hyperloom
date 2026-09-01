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
from _task_group_contract import (  # noqa: E402
    native_operation_key,
    task_group_shape_cases,
)
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
    spec = invocation_spec.build_invocation_spec(candidate)

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
    assert spec["implementation"]["symbols"] == ["_scaled_gemm_kernel"]
    assert "unresolved_runtime_symbol_prefixes" not in spec["edit_target"]
    assert "..." not in json.dumps(spec["edit_target"])


def test_invocation_spec_adds_logical_and_implementation_provenance(tmp_path):
    repo = tmp_path / "repo"
    implementation = repo / "aiter" / "ops" / "triton" / "attention.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text(
        "@triton.jit\ndef unified_attention_kernel(x):\n    return x\n",
        encoding="utf-8",
    )
    wrapper = repo / "vllm" / "attention" / "wrapper.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("def unified_attention(x):\n    return x\n", encoding="utf-8")
    candidate = {
        "name": "vllm::fallback_name",
        "operation": "vllm::fallback_operation",
        "framework": "vllm",
        "kernel_repo": str(repo),
        "source_file": str(wrapper),
        "kernel_sources": [str(implementation)],
        "kernel_kind": "triton",
        "device_kernel_names": ["unified_attention_kernel"],
        "runtime_backend": "ROCM_ATTN",
        "task_group": {"operator_identity": {"operation": "vllm :: unified_attention_with_output"}},
    }

    spec = invocation_spec.build_invocation_spec(candidate)

    assert spec["logical_operator"] == "vllm::unified_attention_with_output"
    assert spec["source_framework"] == "aiter"
    assert spec["execution"]["framework"] == "vllm"
    assert spec["implementation"] == {
        "sources": [str(wrapper), str(implementation)],
        "kernel_kind": "triton",
        "symbols": ["unified_attention_kernel"],
        "runtime_backend": "ROCM_ATTN",
    }
    assert spec["execution"]["runtime_backend"] == "ROCM_ATTN"
    assert spec["edit_target"]["kernel_sources"] == [str(implementation)]


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
    )

    assert spec["status"] == "partial"
    assert spec["deployment"]["model"]["model_id"] == "remote/model-id"
    assert "sequence" not in spec["deployment"]


def test_non_finite_sequence_context_omits_request_token_total(tmp_path):
    spec = invocation_spec.build_invocation_spec(
        {
            "name": "non_finite_sequence",
            "kernel_repo": str(tmp_path),
            "runtime_args": {
                "workload": {
                    "isl": float("nan"),
                    "osl": 1024,
                }
            },
        }
    )

    assert "input_tokens" not in spec["deployment"]["sequence"]
    assert spec["deployment"]["sequence"]["output_tokens"] == 1024
    assert "request_tokens" not in spec["deployment"]["sequence"]


def test_preserves_raw_argument_order_alongside_tensor_projection(tmp_path):
    candidate = _candidate(tmp_path)
    candidate["raw_arg_spec"] = {
        "input_dims": "((64, 17408), (5120, 17408), (), ())",
        "input_type": "(c10::Float8_e4m3fnuz, c10::Float8_e4m3fnuz, int, bool)",
        "concrete_inputs": "(None, None, 128, True)",
    }

    spec = invocation_spec.build_invocation_spec(candidate)

    assert spec["invocation"]["arguments"]
    assert spec["invocation"]["raw_arg_spec"] == candidate["raw_arg_spec"]


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

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["popen_kwargs"] = kwargs
            self.returncode = 0
            self.pid = 123

        def communicate(self, timeout=None):
            return (
                '__FORGE_RESULT__{"baseline_ms": 1.0, '
                '"best_ms": 1.1, "mean_case_speedup": 1.2, '
                '"search_start_mean_case_speedup": 1.0, '
                '"total_improved": true, "incremental_improved": true}',
                "",
            )

    monkeypatch.setattr(forge_submit, "_apply_kernel_backend_env", lambda _env: None)
    monkeypatch.setattr(forge_submit.subprocess, "Popen", FakePopen)

    result = forge_submit._run_loop_via_cli(
        worktree_kernel=str(kernel),
        driver=str(driver),
        workspace=str(tmp_path),
        snr_threshold=30.0,
        max_hours=1.0,
        branch="forge/session/scaled_gemm",
        gpu_target="gfx942",
        gpu_type="mi300x",
        kernel_backend="triton",
        program_md_file="",
        invocation_spec_file=str(spec_path),
        experiments_dir=tmp_path / "experiments",
        forge_log=tmp_path / "forge.log",
        timeout_s=60,
        deadline_unix=9_999_999_999.0,
        experience_id="forge-attempt-1",
    )

    cmd = captured["cmd"]
    option_index = cmd.index("--invocation-spec-file")
    assert cmd[option_index + 1] == str(spec_path.resolve())
    assert cmd[cmd.index("--experiment-id") + 1] == "hyperloom"
    assert cmd[cmd.index("--experience-id") + 1] == "forge-attempt-1"
    assert cmd[cmd.index("--deadline-unix") + 1] == "9999999999.0"
    assert captured["popen_kwargs"]["start_new_session"] is True
    assert (result[0], result[1], result[2], result[4]) == (1.0, 1.1, True, None)
    assert result.mean_case_speedup == 1.2
    assert result.pristine_baseline_ms == 1.0
    assert result.search_start_ms == 1.0
    assert result.improved_during_search is True
    assert result.structured_result == {
        "baseline_ms": 1.0,
        "best_ms": 1.1,
        "mean_case_speedup": 1.2,
        "search_start_mean_case_speedup": 1.0,
        "total_improved": True,
        "incremental_improved": True,
    }


def test_forge_loop_timeout_returns_persisted_checkpoint(tmp_path, monkeypatch):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("def kernel(x):\n    return x\n", encoding="utf-8")
    driver = tmp_path / "driver.py"
    driver.write_text("# driver\n", encoding="utf-8")
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    checkpoint = {
        "state": "best_committed",
        "baseline_ms": 1.0,
        "best_ms": 0.8,
        "improved": True,
    }
    (experiments_dir / "hyperloom.json").write_text(
        json.dumps({"experiment_id": "hyperloom", "checkpoint": checkpoint})
    )

    class TimeoutPopen:
        def __init__(self, *_args, **_kwargs):
            self.returncode = None
            self.pid = 123

        def communicate(self, timeout=None):
            raise forge_submit.subprocess.TimeoutExpired(
                cmd=["forge-loop"],
                timeout=timeout,
            )

    monkeypatch.setattr(forge_submit, "_apply_kernel_backend_env", lambda _env: None)
    monkeypatch.setattr(forge_submit.subprocess, "Popen", TimeoutPopen)

    def terminate_with_checkpoint(_proc):
        (experiments_dir / "hyperloom.json").write_text(
            json.dumps(
                {
                    "experiment_id": "hyperloom",
                    "checkpoint": checkpoint,
                }
            )
        )
        return "partial stdout", ""

    monkeypatch.setattr(
        forge_submit,
        "_terminate_forge_process",
        terminate_with_checkpoint,
    )

    result = forge_submit._run_loop_via_cli(
        worktree_kernel=str(kernel),
        driver=str(driver),
        workspace=str(tmp_path),
        snr_threshold=30.0,
        max_hours=1.0,
        branch="forge/session/scaled_gemm",
        gpu_target="gfx942",
        gpu_type="mi300x",
        kernel_backend="triton",
        program_md_file="",
        invocation_spec_file="",
        experiments_dir=experiments_dir,
        forge_log=tmp_path / "forge.log",
        timeout_s=60,
    )

    assert result.timed_out is True
    assert result.checkpoint == checkpoint
    assert result.baseline_ms is None
    assert result.best_ms is None
    assert isinstance(result.error, RuntimeError)


def test_serialized_candidate_preserves_task_group(tmp_path):
    path = tmp_path / "candidate_tg001.json"
    candidate = _candidate(tmp_path)
    candidate["task_group"] = {
        "task_group_id": "tg001",
        "primary_kernel_id": "k002",
        "kernel_ids": ["k001", "k002"],
        "rows": [
            {
                "kernel_id": "k001",
                "name": "scaled_gemm",
                "input_shapes": [
                    {"shape": "(64,5120) fp8"},
                    {"shape": "(5120,5120) fp8"},
                ],
            },
            {
                "kernel_id": "k002",
                "name": "scaled_gemm",
                "input_shapes": [
                    {"shape": "(64,17408) fp8"},
                    {"shape": "(5120,17408) fp8"},
                ],
            },
        ],
    }
    path.write_text(json.dumps(candidate), encoding="utf-8")

    loaded = kernel_optimization.load_candidate_input(str(path), "k002")
    spec = invocation_spec.build_invocation_spec(loaded)

    assert loaded["task_group"]["task_group_id"] == "tg001"
    assert spec["workload"]["task_group"]["kernel_ids"] == ["k001", "k002"]
    assert len(spec["workload"]["task_group"]["cases"]) == 2
    assert spec["tests"]["driver_contract"] == {
        "shape_argument": "--shape",
        "case_selector_key": "CASE_ID",
        "requires_all_cases": True,
        "case_selectors": [
            {"CASE_ID": "case_001", "M": 64, "N": 5120, "K": 17408},
            {"CASE_ID": "case_002", "M": 64, "N": 5120, "K": 5120},
        ],
    }
    assert kernel_optimization.load_candidate_input(str(path), "k999") is None


def test_forge_shapes_include_every_grouped_gemm_case(tmp_path):
    candidate = _candidate(tmp_path)
    candidate["task_group"] = {
        "task_group_id": "tg001",
        "primary_kernel_id": "k002",
        "kernel_ids": ["k001", "k002", "k003", "k004"],
        "rows": [
            {
                "kernel_id": "k001",
                "name": "scaled_gemm",
                "input_shapes": [
                    {"shape": "(64,5120) fp8"},
                    {"shape": "(5120,5120) fp8"},
                ],
            },
            {
                "kernel_id": "k002",
                "name": "scaled_gemm",
                "input_shapes": [
                    {"shape": "(64,17408) fp8"},
                    {"shape": "(5120,17408) fp8"},
                ],
            },
            {
                "kernel_id": "k003",
                "name": "scaled_gemm",
                "input_shapes": [
                    {"shape": "(64,5120) fp8"},
                    {"shape": "(7168,5120) fp8"},
                ],
            },
            {
                "kernel_id": "k004",
                "name": "scaled_gemm",
                "input_shapes": [
                    {"shape": "(64,5120) fp8"},
                    {"shape": "(34816,5120) fp8"},
                ],
            },
        ],
    }

    shapes = forge_submit._shapes_from_candidate(candidate)

    assert shapes["primary"] == {
        "CASE_ID": "case_001",
        "M": 64,
        "N": 5120,
        "K": 17408,
    }
    assert shapes["validation"] == [
        {"CASE_ID": "case_001", "M": 64, "N": 5120, "K": 17408},
        {"CASE_ID": "case_002", "M": 64, "N": 5120, "K": 5120},
        {"CASE_ID": "case_003", "M": 64, "N": 7168, "K": 5120},
        {"CASE_ID": "case_004", "M": 64, "N": 34816, "K": 5120},
    ]


def test_forge_shapes_keep_all_generic_operator_cases():
    candidate = {
        "kernel_id": "k001",
        "name": "rms_norm",
        "task_group": {
            "task_group_id": "tg001",
            "primary_kernel_id": "k001",
            "kernel_ids": ["k001", "k002"],
            "rows": [
                {
                    "kernel_id": "k001",
                    "name": "rms_norm",
                    "input_shapes": [{"shape": "(64,5120) bf16"}],
                },
                {
                    "kernel_id": "k002",
                    "name": "rms_norm",
                    "input_shapes": [{"shape": "(8,5120) bf16"}],
                },
            ],
        },
    }

    shapes = forge_submit._shapes_from_candidate(candidate)

    assert shapes["validation"] == [
        {"CASE_ID": "case_001"},
        {"CASE_ID": "case_002"},
    ]


def test_forge_shapes_deduplicate_identical_observations():
    candidate = {
        "kernel_id": "k001",
        "name": "rms_norm",
        "task_group": {
            "task_group_id": "tg001",
            "primary_kernel_id": "k001",
            "kernel_ids": ["k001", "k002"],
            "rows": [
                {
                    "kernel_id": "k001",
                    "name": "rms_norm",
                    "call_count": 10,
                    "input_shapes": [{"call_num": 10, "shape": "(64,5120) bf16"}],
                },
                {
                    "kernel_id": "k002",
                    "name": "rms_norm",
                    "call_count": 20,
                    "input_shapes": [{"call_num": 20, "shape": "(64,5120) bf16"}],
                },
            ],
        },
    }

    cases = task_group_shape_cases(candidate)

    assert len(cases) == 1
    assert cases[0]["kernel_ids"] == ["k001", "k002"]
    assert cases[0]["call_count"] == 30


def test_group_cases_treat_malformed_duplicate_call_count_as_zero():
    candidate = {
        "kernel_id": "k001",
        "name": "rms_norm",
        "task_group": {
            "task_group_id": "tg001",
            "primary_kernel_id": "k001",
            "kernel_ids": ["k001", "k002"],
            "rows": [
                {
                    "kernel_id": "k001",
                    "name": "rms_norm",
                    "call_count": 10,
                    "input_shapes": [{"shape": "(64,5120) bf16"}],
                },
                {
                    "kernel_id": "k002",
                    "name": "rms_norm",
                    "call_count": "not-an-integer",
                    "input_shapes": [{"shape": "(64,5120) bf16"}],
                },
            ],
        },
    }

    cases = task_group_shape_cases(candidate)

    assert len(cases) == 1
    assert cases[0]["call_count"] == 10


def test_group_cases_expand_csv_invocation_boundaries():
    candidate = {
        "kernel_id": "k001",
        "name": "fused_moe",
        "task_group": {
            "task_group_id": "tg001",
            "primary_kernel_id": "k001",
            "kernel_ids": ["k001"],
            "rows": [
                {
                    "kernel_id": "k001",
                    "name": "fused_moe",
                    "invocation_cases": [
                        {
                            "operation": "fused_moe_gate",
                            "input_shapes": [{"shape": "(64,2048) bf16"}],
                            "raw_arg_spec": {"concrete_inputs": "(1,)"},
                        },
                        {
                            "operation": "fused_moe_down",
                            "input_shapes": [{"shape": "(512,768) bf16"}],
                            "raw_arg_spec": {"concrete_inputs": "(2,)"},
                        },
                    ],
                }
            ],
        },
    }

    cases = task_group_shape_cases(candidate)

    assert len(cases) == 2
    assert [case["operation"] for case in cases] == [
        "fused_moe_gate",
        "fused_moe_down",
    ]
    assert [case["selector"]["CASE_ID"] for case in cases] == [
        "case_001",
        "case_002",
    ]


def test_native_operation_key_normalizes_graph_wrapped_mangled_symbols():
    assert (
        native_operation_key("hipGraphLaunch->_ZN5aiter24add_rmsnorm_quant_kernelIDF16bEEv.kd")
        == "aiter::add_rmsnorm_quant_kernel"
    )


def _rewrite_candidate(rows: list[dict], **extra) -> dict:
    candidate = {
        "name": "fused_gemm",
        "operation": "vllm::fused_gemm",
        "source_symbol": "matmul",
        "task_group": {
            "task_group_id": "tg001",
            "primary_kernel_id": "k001",
            "kernel_ids": [row["kernel_id"] for row in rows],
            "rows": rows,
        },
    }
    candidate.update(extra)
    return candidate


# --- dims the candidate review recovered must reach the driver -------------


def test_spec_reports_missing_inputs_when_the_trace_recorded_no_arguments():
    """A graph replay has no CPU-side parent op, so a correctly resolved kernel
    can still arrive with no dims at all -- and the spec has to say so rather
    than emit an argument list built from nothing.
    """
    spec = invocation_spec.build_invocation_spec(
        {"kernel_id": "k001", "name": "kernel_paged_attention_2d", "shapes": []},
        source_file="/repo/attn.py",
    )
    # Empty members are compacted away, so an argument list is absent rather
    # than present and empty.
    assert not spec.get("invocation", {}).get("arguments")
    assert "inputs" in spec["missing_fields"]


def test_reviewed_dims_reach_the_invocation_spec():
    """The spec is what the backend builds its driver from, so dims recovered by
    the candidate review have to arrive here or the correction changes nothing
    about what actually gets benchmarked.

    ``shapes`` is the field the review revises; ``build_invocation_spec`` reads
    it as the fallback for ``input_shapes``, which is what carries it through.
    """
    spec = invocation_spec.build_invocation_spec(
        {
            "kernel_id": "k001",
            "name": "kernel_paged_attention_2d",
            "shapes": ["(8192,6144) bf16", "(6144,1536) fp4"],
            "input_dtypes": ["bf16", "fp4"],
            "shape_provenance": "review_derived",
        },
        source_file="/repo/attn.py",
    )
    arguments = spec["invocation"]["arguments"]
    assert [record["shape"] for record in arguments] == [[8192, 6144], [6144, 1536]]
    assert [record["dtype"] for record in arguments] == ["bf16", "fp4"]
    assert "inputs" not in spec["missing_fields"]


def test_only_the_reviewed_table_can_be_resolved_as_the_candidate_source(tmp_path):
    """The pre-review baseline sits in the same directory under its own name.

    Resolving that instead would hand the backend the dims the review corrected,
    which is the one outcome the two-artifact split exists to make impossible.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "kernel_candidates.raw.json").write_text(
        json.dumps({"hot_kernels": [{"kernel_id": "k001", "shapes": []}]}),
        encoding="utf-8",
    )
    reviewed = {"hot_kernels": [{"kernel_id": "k001", "shapes": ["(8192,6144) bf16"]}]}
    (run_dir / "kernel_candidates.json").write_text(json.dumps(reviewed), encoding="utf-8")

    resolved = kernel_optimization.resolve_candidates_path(run_dir)
    assert resolved.name == "kernel_candidates.json"
    assert kernel_optimization.load_candidates(resolved)[0]["shapes"] == ["(8192,6144) bf16"]
