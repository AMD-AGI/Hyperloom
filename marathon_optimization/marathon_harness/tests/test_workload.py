"""Unit tests for InferenceXWorkload — constructors, config parsing, script discovery."""

import json
import tempfile
from pathlib import Path

from marathon_harness.workload import InferenceXWorkload, BenchmarkResult


def test_basic_constructor():
    wl = InferenceXWorkload(
        inferencex_path="/tmp/ix",
        model="/models/test",
        tp=4,
    )
    assert wl.model == "/models/test"
    assert wl.tp == 4
    assert wl.isl == 1024
    assert wl.osl == 1024
    assert wl.concurrency == 64


def test_from_sprint_handoff():
    with tempfile.TemporaryDirectory() as td:
        ix_dir = Path(td) / "InferenceX" / "benchmarks"
        ix_dir.mkdir(parents=True)

        handoff = Path(td) / "handoff"
        handoff.mkdir()
        config = {
            "model_path": "/models/deepseek-r1",
            "tp": 8,
            "framework": "sglang",
            "launch_flags": ["--mem-fraction-static", "0.88", "--chunked-prefill-size", "32768"],
            "env_vars": {"SGLANG_USE_AITER": "1", "NCCL_ALGO": "Ring"},
            "benchmark_params": {
                "input_len": 2048,
                "output_len": 512,
                "max_concurrency": 32,
            },
        }
        (handoff / "config.json").write_text(json.dumps(config))

        wl = InferenceXWorkload.from_sprint_handoff(
            handoff, str(Path(td) / "InferenceX"))

        assert wl.model == "/models/deepseek-r1"
        assert wl.tp == 8
        assert "--mem-fraction-static" in wl.extra_launch_flags
        assert wl.env_vars["SGLANG_USE_AITER"] == "1"
        assert wl.isl == 2048
        assert wl.osl == 512
        assert wl.concurrency == 32


def test_from_sprint_handoff_discovers_launch_script():
    with tempfile.TemporaryDirectory() as td:
        ix_dir = Path(td) / "InferenceX" / "benchmarks"
        ix_dir.mkdir(parents=True)

        handoff = Path(td) / "handoff"
        handoff.mkdir()
        (handoff / "config.json").write_text(json.dumps({
            "model_path": "/m", "tp": 8,
        }))
        (handoff / "launch_server.sh").write_text(
            "#!/bin/bash\npython3 -m sglang.launch_server --model-path /m\n"
        )

        wl = InferenceXWorkload.from_sprint_handoff(
            handoff, str(Path(td) / "InferenceX"))
        assert hasattr(wl, '_sprint_launch_script')
        assert "launch_server.sh" in wl._sprint_launch_script


def test_from_sprint_repo():
    """Test constructing from standalone Agentic-InferenceX repo layout."""
    with tempfile.TemporaryDirectory() as td:
        ix_dir = Path(td) / "InferenceX" / "benchmarks"
        ix_dir.mkdir(parents=True)

        repo = Path(td) / "DeepSeek-R1-0528-optimized"
        scripts = repo / "scripts"
        scripts.mkdir(parents=True)

        # Write a realistic launch_server.sh
        (scripts / "launch_server.sh").write_text(
            '#!/bin/bash\n'
            'MODEL="${MODEL:-/shared_nfs/models/DeepSeek-R1-0528}"\n'
            'TP="${TP:-8}"\n'
            'python3 -m sglang.launch_server --model-path $MODEL\n'
        )
        (scripts / "run_benchmark.sh").write_text(
            '#!/bin/bash\npython3 benchmark_serving --num-prompts 256\n'
        )
        (repo / "results").mkdir()

        wl = InferenceXWorkload.from_sprint_repo(
            repo, str(Path(td) / "InferenceX"))

        assert wl.model == "/shared_nfs/models/DeepSeek-R1-0528"
        assert wl.tp == 8
        assert hasattr(wl, '_sprint_launch_script')
        assert "launch_server.sh" in wl._sprint_launch_script
        assert hasattr(wl, '_sprint_benchmark_script')
        assert "run_benchmark.sh" in wl._sprint_benchmark_script


def test_from_sprint_repo_with_handoff_subdir():
    """Test that from_sprint_repo also picks up handoff/ if present."""
    with tempfile.TemporaryDirectory() as td:
        ix_dir = Path(td) / "InferenceX" / "benchmarks"
        ix_dir.mkdir(parents=True)

        repo = Path(td) / "model-optimized"
        scripts = repo / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "launch_server.sh").write_text('#!/bin/bash\nMODEL="${MODEL:-/m}"\nTP="${TP:-4}"\n')

        handoff = repo / "handoff"
        handoff.mkdir()
        (handoff / "config.json").write_text(json.dumps({
            "model_path": "/models/actual",
            "tp": 8,
            "env_vars": {"KEY": "VAL"},
            "launch_flags": ["--flag"],
            "benchmark_params": {"input_len": 4096},
        }))

        wl = InferenceXWorkload.from_sprint_repo(
            repo, str(Path(td) / "InferenceX"))

        # handoff config should override script-parsed values
        assert wl.model == "/models/actual"
        assert wl.tp == 8
        assert wl.env_vars["KEY"] == "VAL"
        assert wl.isl == 4096


def test_parse_result():
    with tempfile.TemporaryDirectory() as td:
        wl = InferenceXWorkload(inferencex_path=td, model="m", tp=8)
        result_file = Path(td) / "result.json"
        result_file.write_text(json.dumps({
            "output_throughput": 800.0,
            "total_token_throughput": 1600.0,
            "request_throughput": 50.0,
            "mean_ttft_ms": 100.0,
            "mean_tpot_ms": 12.5,
            "p99_ttft_ms": 200.0,
            "p99_tpot_ms": 25.0,
            "num_prompts": 256,
        }))

        br = wl.parse_result(str(result_file))
        assert br.output_throughput == 800.0
        assert br.tput_per_gpu == 100.0  # 800 / 8
        assert br.mean_tpot_ms == 12.5
        assert br.num_prompts == 256


def test_parse_result_missing_file():
    wl = InferenceXWorkload(inferencex_path="/tmp", model="m", tp=8)
    br = wl.parse_result("/nonexistent/result.json")
    assert br.tput_per_gpu == 0.0
