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
from hyperloom.orchestrator.actions.executors import _subprocess_kill as sk
from hyperloom.orchestrator.actions.executors._spec_decode_context_copy import (
    CONTEXT_COPY_REFUSAL,
    context_copy_method,
    context_copy_refusal,
)
from hyperloom.orchestrator.bus.resource_lock import SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import TaskRegistry, TaskSpec


# --------------------------------------------------------------------------
# eval-start marker
# --------------------------------------------------------------------------
def test_eval_marker_in_magpies_nested_stderr_opens_the_eval_phase(tmp_path):
    """The engine logs to the round's own server.log while Magpie's stderr, which carries the marker, is nested.

    Reached only through a nested server.log, the stderr was never scanned and the eval was folded into the
    measured phase (a 12h vLLM session's warmup round recorded only ``measured`` across its whole GSM8K eval).
    """
    primary = tmp_path / "server.log"
    offsets = sk._stale_scan_log_sizes(str(primary))
    primary.write_text("Application startup complete\n", encoding="utf-8")
    nested = tmp_path / "benchmark_vllm_20261014_204322"
    nested.mkdir()
    (nested / "benchmark_stderr.log").write_text("+ run_eval\nHYPERLOOM_EVAL_START\n", encoding="utf-8")

    scan = sk._LogCursor(offsets).advance(str(primary))
    assert scan.saw_eval_start


def test_a_previous_runs_nested_stderr_is_not_activity(tmp_path):
    """A stderr that was already there when the round started belongs to an earlier run and must stay silent."""
    nested = tmp_path / "benchmark_vllm_old"
    nested.mkdir()
    (nested / "benchmark_stderr.log").write_text("HYPERLOOM_EVAL_START\n", encoding="utf-8")
    offsets = sk._stale_scan_log_sizes(str(tmp_path / "server.log"))
    scan = sk._LogCursor(offsets).advance(str(tmp_path / "server.log"))
    assert not scan.saw_eval_start


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


@pytest.mark.asyncio
async def test_an_ngram_variant_is_refused_before_it_runs(sub_agent_runner, tmp_path):
    """The measured case: ngram k=7 would have been KEPT at +95.7% on the synthetic benchmark."""
    launched, out = await _explore(sub_agent_runner, tmp_path, ["ngram_k7", "eagle3", "lever"])
    assert launched == ["eagle3", "lever"]
    assert list(_refused(out)) == ["ngram_k7"]
    assert "'ngram'" in _refused(out)["ngram_k7"]
    assert "ngram_k7" not in {w["name"] for w in out["winners"]}
    assert {row["variant_name"]: row["reason"] for row in out["per_variant_outcomes"]}["ngram_k7"] == (
        CONTEXT_COPY_REFUSAL
    )


@pytest.mark.asyncio
async def test_an_ngram_draft_set_through_the_server_args_env_is_refused(sub_agent_runner, tmp_path):
    """EXTRA_VLLM_ARGS reaches the launch after composition; the materialized launch is what is read."""
    launched, out = await _explore(
        sub_agent_runner,
        tmp_path,
        ["lever"],
        entry={"extra_envs": {"EXTRA_VLLM_ARGS": f"--max-num-seqs 64 {_NGRAM_K7}"}},
    )
    assert launched == []
    assert list(_refused(out)) == ["lever"]


@pytest.mark.asyncio
async def test_a_lever_on_a_base_that_already_copies_runs(sub_agent_runner, tmp_path):
    """The base's own draft is the operator's choice; a candidate that leaves it alone is graded as before."""
    launched, out = await _explore(sub_agent_runner, tmp_path, ["lever"], base_extra_args=_NGRAM_K7)
    assert launched == ["lever"]
    assert _refused(out) == {}


@pytest.mark.asyncio
async def test_a_copying_draft_is_judged_against_the_stack_a_keep_in_the_same_round_left(sub_agent_runner, tmp_path):
    """Dropping the base's ngram is KEPT; restoring it next in the same round is a change against that stack."""
    launched, out = await _explore(sub_agent_runner, tmp_path, ["drop_draft", "ngram_k7"], base_extra_args=_NGRAM_K7)
    assert [w["name"] for w in out["winners"]] == ["drop_draft"]
    assert launched == ["drop_draft"]
    assert list(_refused(out)) == ["ngram_k7"]


@pytest.mark.asyncio
async def test_an_agentx_session_is_not_filtered(sub_agent_runner, tmp_path):
    launched, out = await _explore(sub_agent_runner, tmp_path, ["ngram_k7"], benchmark_mode="agentx")
    assert launched == ["ngram_k7"]
    assert _refused(out) == {}


@pytest.mark.asyncio
async def test_a_stack_revalidation_is_not_filtered(sub_agent_runner, tmp_path):
    """It re-measures the stack as it stands, which may already carry the operator's draft."""
    from hyperloom.orchestrator.actions.executors.explore import STACK_REVALIDATE_SOURCE

    launched, out = await _explore(sub_agent_runner, tmp_path, ["ngram_k7"], source=STACK_REVALIDATE_SOURCE)
    assert launched == ["ngram_k7"]
    assert _refused(out) == {}


@pytest.mark.asyncio
async def test_a_variant_that_cannot_be_materialized_is_judged_on_its_own_args(sub_agent_runner, tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import explore

    def _unreadable(*_args, **_kwargs):
        raise OSError("probe")

    monkeypatch.setattr(explore, "_materialized_launch", _unreadable)
    launched, out = await _explore(sub_agent_runner, tmp_path, ["ngram_k7", "lever"])
    assert launched == ["lever"]
    assert list(_refused(out)) == ["ngram_k7"]
