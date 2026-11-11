# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A context-copying speculative-decoding draft is refused as a candidate on a synthetic benchmark.

Covers the rule (which methods copy, what counts as introducing or changing one), the explore wiring that refuses such a
variant before it runs, and the eval-start marker that opens the eval phase of ``kv_metrics.json``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors.explore import ExploreExecutor
from hyperloom.orchestrator.actions.executors._spec_decode_context_copy import (
    CONTEXT_COPY_REFUSAL,
    context_copy_method,
    context_copy_refusal,
)
from hyperloom.orchestrator.bus.resource_lock import SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import TaskRegistry, TaskSpec


# --------------------------------------------------------------------------
# which methods copy from the context
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        pytest.param('--speculative-config {"method":"ngram","num_speculative_tokens":7}', "ngram", id="vllm-ngram"),
        pytest.param('--speculative-config \'{"method": "ngram_gpu"}\'', "ngram_gpu", id="vllm-ngram-gpu-quoted"),
        pytest.param('--speculative-config={"method":"suffix"}', "suffix", id="vllm-suffix-attached"),
        pytest.param('--speculative-config {"model":"[ngram]","num_speculative_tokens":4}', "ngram", id="vllm-model"),
        pytest.param("--speculative-config {method:ngram,prompt_lookup_max:8}", "ngram", id="vllm-unquoted-json"),
        pytest.param("--speculative-config.method ngram", "ngram", id="vllm-dotted"),
        pytest.param("--speculative_config.method=suffix", "suffix", id="vllm-dotted-underscore"),
        pytest.param("--speculative-algorithm NGRAM --speculative-num-draft-tokens 8", "NGRAM", id="sglang-ngram"),
        pytest.param("--speculative-algorithm ngram", "NGRAM", id="sglang-lowercase"),
        pytest.param('--speculative-config {"method":"eagle3","model":"/m/eagle"}', "", id="vllm-eagle3"),
        pytest.param('--speculative-config {"method":"mtp","num_speculative_tokens":1}', "", id="vllm-mtp"),
        pytest.param('--speculative-config {"model":"/m/draft"}', "", id="vllm-draft-model"),
        pytest.param("--speculative-config {method:mtp,draft_model:ngram}", "", id="key-suffix-is-not-the-key"),
        pytest.param('--speculative-config {"method":"ngram"} --speculative-config.method mtp', "", id="dotted-wins"),
        pytest.param(
            '--speculative-config {"method":"mtp"} --speculative-config {"method":"ngram"}', "ngram", id="last-wins"
        ),
        pytest.param(
            '--speculative-config {"method":"ngram"} --speculative-config {"model":"/m/draft"}',
            "",
            id="replaces-object",
        ),
        pytest.param("--speculative-algorithm EAGLE3 --speculative-num-steps 3", "", id="sglang-eagle3"),
        pytest.param("--speculative-algorithm NEXTN", "", id="sglang-nextn"),
        pytest.param("--max-num-seqs 64 --kv-cache-dtype fp8", "", id="no-draft"),
        pytest.param("", "", id="empty"),
    ],
)
def test_context_copy_method_names_the_copying_methods_only(argv, expected):
    assert context_copy_method(argv, {}) == expected


def test_a_copying_draft_in_a_server_args_env_is_found():
    assert context_copy_method("", {"EXTRA_VLLM_ARGS": '--speculative-config {"method":"ngram"}'}) == "ngram"
    assert context_copy_method("", {"VLLM_ROCM_USE_AITER": "1"}) == ""


# --------------------------------------------------------------------------
# what introducing or changing one means
# --------------------------------------------------------------------------
_NGRAM = '--speculative-config {"method":"ngram","num_speculative_tokens":7}'
_EAGLE = '--speculative-config {"method":"eagle3","model":"/m/eagle","num_speculative_tokens":3}'


@pytest.mark.parametrize(
    ("base", "candidate", "refused"),
    [
        pytest.param("", _NGRAM, True, id="introduces"),
        pytest.param(_EAGLE, _NGRAM, True, id="switches-to"),
        pytest.param(_NGRAM, _NGRAM.replace(":7", ":4"), True, id="tunes-the-base-draft"),
        pytest.param(_NGRAM, f"{_NGRAM} --speculative-config.prompt_lookup_max 8", True, id="adds-a-dotted-key"),
        pytest.param(_NGRAM, f"--max-num-seqs 64 {_NGRAM}", False, id="a-lever-on-a-copying-base"),
        pytest.param(_NGRAM, "--max-num-seqs 64", False, id="drops-the-draft"),
        pytest.param(_NGRAM, _EAGLE, False, id="switches-to-a-model-draft"),
        pytest.param("", _EAGLE, False, id="introduces-a-model-draft"),
        pytest.param("", "--max-num-seqs 64", False, id="no-draft"),
    ],
)
def test_a_candidate_is_refused_when_it_introduces_or_changes_a_copying_draft(base, candidate, refused):
    reason = context_copy_refusal(base, {}, candidate, {})
    assert bool(reason) is refused
    if refused:
        assert "copying from the context" in reason


def test_a_speculative_env_change_on_a_copying_base_is_a_change():
    base_envs = {"SGLANG_ENABLE_SPEC_V2": "0"}
    argv = "--speculative-algorithm NGRAM"
    assert context_copy_refusal(argv, base_envs, argv, {"SGLANG_ENABLE_SPEC_V2": "1"})
    assert not context_copy_refusal(argv, base_envs, argv, {**base_envs, "SGLANG_USE_AITER": "1"})


# --------------------------------------------------------------------------
# explore
# --------------------------------------------------------------------------
@pytest.fixture
def sub_agent_runner(tmp_path):
    db = SqliteConnection(tmp_path / "db.db")
    runner = SubAgentRunner(SqliteLeaseBackend(db), TaskRegistry(db))
    yield runner, TaskRegistry(db)
    db.close()


@pytest.fixture(autouse=True)
def _isolate(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(tmp_path_factory.mktemp("leak_root")))
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_NOISE_PCT", raising=False)
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.explore.resolve_lifecycle_params",
        lambda _config_path: {"eligible": False, "framework": "vllm", "port": 30000, "reason": "test"},
    )


def _base_yaml(path: Path) -> Path:
    cfg = {
        "benchmark": {
            "framework": "vllm",
            "model": "/path/models/Qwen-Qwen3-14B-FP8",
            "precision": "fp8",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 64, "ISL": 1024, "OSL": 1024},
            "benchmark_script": "vllm_mi355x.sh",
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        },
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _write_round(slot: Path, tput: float) -> None:
    workspace = slot / "benchmark_vllm_20261014_205059"
    workspace.mkdir(parents=True)
    (workspace / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "vllm",
                "model": "/path/models/Qwen-Qwen3-14B-FP8",
                "throughput": {
                    "request_throughput": tput / 1024,
                    "output_throughput": tput,
                    "total_token_throughput": tput * 2,
                    "completed_requests": 320,
                    "duration_seconds": 80.0,
                },
                "latency": {"ttft": {"mean_ms": 140.0, "p99_ms": 160.0}, "e2el": {"mean_ms": 2500.0, "p99_ms": 2560.0}},
            }
        ),
        encoding="utf-8",
    )


_NGRAM_K7 = (
    '--speculative-config {"method":"ngram","num_speculative_tokens":7,"prompt_lookup_min":4,"prompt_lookup_max":8}'
)
_ARGS = {
    "ngram_k7": _NGRAM_K7,
    "eagle3": _EAGLE,
    "lever": "--max-num-seqs 64",
    "drop_draft": "--max-num-seqs 64",
}
_ENTRIES = {"drop_draft": {"remove_args": ["--speculative-config"]}}
_BASE_TPUT = 3909.10


async def _explore(sub_agent_runner, tmp_path, names: list[str], *, entry: dict | None = None, **params):
    """Run one explore over the named variants (args from ``_ARGS``), each measuring +95.7%; return (launched, result)."""
    sub, tr = sub_agent_runner
    state = SharedState(framework="vllm")
    state.baseline_tput = _BASE_TPUT
    state.benchmark_mode = params.pop("benchmark_mode", "synthetic")
    sub.shared_state = state
    launched: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        name = next(n for n in names if slot.parent.name.endswith(n))
        launched.append(name)
        _write_round(slot, 7648.85)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(_base_yaml(tmp_path / "base.yaml")),
                "output_dir": str(tmp_path / "explore"),
                "base_tput": _BASE_TPUT,
                "grid": [
                    {"name": name, "extra_args": _ARGS[name], **_ENTRIES.get(name, {}), **(entry or {})}
                    for name in names
                ],
                **params,
            },
            idempotency_key="ex-spec",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)
    return launched, res.result


def _refused(out: dict) -> dict[str, str]:
    return {sd["name"]: sd["detail"] for sd in out["skipped_dup"] if sd["reason"] == CONTEXT_COPY_REFUSAL}
