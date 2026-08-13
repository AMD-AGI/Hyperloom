# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the baseline cold-start "warmup artifact".

Covers the cold+hot double-run and its server-lifecycle reuse, the pre-start /
teardown cleanup around the reused port, the local InferenceX mirror, and the
subprocess-failure classifier.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors.baseline import (
    BaselineExecutor,
)
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    run_grid,
)
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.trace.task_progress import progress_scope

from .conftest import chatty_child, enable_multi_node, suppression_window_s


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root_warmup")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT", "8")


def _write_yaml(path: Path, *, framework: str = "vllm") -> None:
    cfg: dict = {
        "benchmark": {
            "framework": framework,
            "model": "/path/models/Qwen-Qwen3-8B",
            "precision": "fp8",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 64, "ISL": 1024, "OSL": 1024},
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        }
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def _fake_workspace(slot: Path, *, tput: float) -> Path:
    ws = slot / "benchmark_vllm_20260602_010101"
    ws.mkdir(parents=True)
    (ws / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "vllm",
                "model": "/path/models/Qwen-Qwen3-8B",
                "throughput": {
                    "request_throughput": tput / 1024,
                    "output_throughput": tput,
                    "total_token_throughput": tput * 2,
                    "completed_requests": 64,
                    "duration_seconds": 25.0,
                },
                "latency": {
                    "ttft": {"mean_ms": 100.0, "p99_ms": 120.0},
                    "e2el": {"mean_ms": 2000.0, "p99_ms": 2300.0},
                },
            }
        )
    )
    return ws


def _make_ctx(params: dict) -> SimpleNamespace:
    task = SimpleNamespace(task_id="t-baseline-warmup", params=params)
    return SimpleNamespace(task=task, extra={})


def _run(coro):
    return asyncio.run(coro)


_COLD_TPUT = 270.9
_HOT_TPUT = 4701.6


def _cold_then_hot_fake_run(captured: list | None = None):
    """Return a ``run_with_session_kill`` stand-in that emits a cold throughput
    on its first call and a hot throughput thereafter."""
    state = {"calls": 0}

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        if captured is not None:
            cfg_idx = cmd.index("--benchmark-config")
            cfg = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
            captured.append(cfg)
        tput = _COLD_TPUT if state["calls"] == 0 else _HOT_TPUT
        state["calls"] += 1
        _fake_workspace(slot, tput=tput)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    return fake_run, state


def _executor(
    base: Path,
    tmp_path: Path,
    *,
    baseline_double_run: bool = True,
) -> BaselineExecutor:
    return BaselineExecutor(
        magpie_python=sys.executable,
        default_config_path=base,
        session_dir=tmp_path,
        shared_state=SimpleNamespace(baseline_double_run=baseline_double_run),
    )


def test_baseline_discards_cold_first_round_via_lifecycle(tmp_path, monkeypatch):
    """The double-run reports the HOT second-round throughput."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert result["output_throughput"] == pytest.approx(_HOT_TPUT)
    assert result.get("warmup_round_tput") == pytest.approx(_COLD_TPUT)
    assert "baseline_double_run_discarded_first" in result["nonfatal_warnings"]

    assert len(captured) == 2
    warmup_lc = captured[0]["benchmark"]["server_lifecycle"]
    measure_lc = captured[1]["benchmark"]["server_lifecycle"]
    assert warmup_lc["enabled"] is True and measure_lc["enabled"] is True
    assert warmup_lc["cleanup"] is False
    assert measure_lc["cleanup"] is True
    assert warmup_lc["pid_dir"] == measure_lc["pid_dir"] == str(output_dir)
    assert captured[0]["benchmark"]["envs"]["PORT"] == (captured[1]["benchmark"]["envs"]["PORT"])
    assert captured[0]["benchmark"]["benchmark_script"] == "vllm_mi300x.sh"


def _run_capturing_rounds(executor, ctx, notes):
    """Run ``executor`` and record which round notes existed at each launch."""
    at_launch: list[list[str]] = []
    inner, _state = _cold_then_hot_fake_run()

    def fake_run(cmd, *args, **kwargs):
        at_launch.append([n["label"] for n in notes])
        return inner(cmd, *args, **kwargs)

    with (
        progress_scope(_sink_into(notes)),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        return _run(executor(ctx)), at_launch


def _sink_into(notes: list):
    """Build an ambient progress sink that appends every note to ``notes``."""

    async def _sink(**note):
        notes.append(note)

    return _sink


def test_each_double_run_round_reports_before_it_blocks(tmp_path):
    """A round that boots a server and never returns must still have said it started."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    notes: list[dict] = []
    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx({"output_dir": str(tmp_path / "ws"), "timeout_sec": 10, "gpu_type": "mi300x"})

    result, at_launch = _run_capturing_rounds(executor, ctx, notes)

    assert result["status"] == "succeeded"
    assert at_launch == [["warmup"], ["warmup", "warmup", "measure"]]
    assert [(n["label"], n["status"]) for n in notes] == [
        ("warmup", "started"),
        ("warmup", "succeeded"),
        ("measure", "started"),
    ]


def test_the_single_round_path_reports_too(tmp_path):
    """The non-double-run baseline used to report nothing at all."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    notes: list[dict] = []
    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx({"output_dir": str(tmp_path / "ws"), "timeout_sec": 10, "gpu_type": "mi300x"})

    result, at_launch = _run_capturing_rounds(executor, ctx, notes)

    assert result["status"] == "succeeded"
    assert at_launch == [["single"]]


def test_a_round_is_handed_the_liveness_callback_its_heartbeat_needs(tmp_path):
    """A round outlives its start report; only child output can extend it."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    seen: list = []
    inner, _state = _cold_then_hot_fake_run()

    def fake_run(cmd, *args, **kwargs):
        seen.append(kwargs.get("on_output"))
        return inner(cmd, *args, **kwargs)

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx({"output_dir": str(tmp_path / "ws"), "timeout_sec": 10, "gpu_type": "mi300x"})
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert [callable(cb) for cb in seen] == [True]


def _cadence_ctx(tmp_path) -> SimpleNamespace:
    """A single-round baseline context for the cadence tests."""
    return _make_ctx({"output_dir": str(tmp_path / "ws"), "timeout_sec": 10, "gpu_type": "mi300x"})


def test_a_round_keeps_reporting_while_its_benchmark_blocks(tmp_path, progress_cadence):
    """A round blocks for the better part of an hour; entry markers cannot cover that."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    inner, _state = _cold_then_hot_fake_run()
    executor = _executor(base, tmp_path, baseline_double_run=False)

    with (
        progress_scope(progress_cadence.sink()),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=chatty_child(progress_cadence, inner, blocks_for_s=600.0, line_every_s=30.0),
        ),
    ):
        result = _run(executor(_cadence_ctx(tmp_path)))

    assert result["status"] == "succeeded"
    assert progress_cadence.widest_silence() < suppression_window_s()


def test_the_multi_node_warmup_pass_keeps_reporting_too(tmp_path, monkeypatch, progress_cadence):
    """The discarded MN warmup is a full benchmark pass and blocks just as long."""
    enable_multi_node(monkeypatch)
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    inner, state = _cold_then_hot_fake_run()
    executor = _executor(base, tmp_path, baseline_double_run=False)

    with (
        progress_scope(progress_cadence.sink()),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=chatty_child(progress_cadence, inner, blocks_for_s=600.0, line_every_s=30.0),
        ),
    ):
        result = _run(executor(_cadence_ctx(tmp_path)))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2  # the discarded warmup pass, then the measured one
    assert progress_cadence.widest_silence() < suppression_window_s()


def test_a_failing_warmup_round_still_reported_that_it_started(tmp_path):
    """The failure path returns early; only the entry report covers it."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    notes: list[dict] = []
    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx({"output_dir": str(tmp_path / "ws"), "timeout_sec": 10, "gpu_type": "mi300x"})

    with (
        progress_scope(_sink_into(notes)),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 1, "", "boom"),
        ),
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert [(n["label"], n["status"]) for n in notes] == [("warmup", "started")]


def _prelude_shared_state(*, spent_sec: float, usable_sec: float) -> SimpleNamespace:
    """A PRELUDE session state with an explicit clock, as the budget policy reads it."""
    return SimpleNamespace(
        baseline_double_run=True,
        phase="PRELUDE",
        max_minutes=180,
        phase_elapsed_totals={"PRELUDE": spent_sec},
        phase_started_unix=0.0,
        session_budget_usable_sec=lambda: usable_sec,
    )


def _run_double_run_baseline(tmp_path, shared_state) -> dict:
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    fake_run, state = _cold_then_hot_fake_run()
    executor = BaselineExecutor(
        magpie_python=sys.executable,
        default_config_path=base,
        session_dir=tmp_path,
        shared_state=shared_state,
    )
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "gpu_type": "mi300x"})
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))
    result["_rounds_run"] = state["calls"]
    return result


def test_measured_round_is_dropped_when_preparation_has_spent_the_budget(tmp_path):
    """The MiniMax-M2 shape: a 66-minute warmup, then a second round the session cannot use.

    The warmup already carries accuracy and a throughput figure, so dropping
    the measured round leaves the single-round baseline the codebase supports
    rather than a half-measured state.
    """
    result = _run_double_run_baseline(
        tmp_path,
        _prelude_shared_state(spent_sec=10_000.0, usable_sec=500.0),
    )

    assert result["status"] == "succeeded"
    assert result["_rounds_run"] == 1
    assert result["output_throughput"] == pytest.approx(_COLD_TPUT)
    assert "baseline_measure_round_dropped_low_budget" in result["nonfatal_warnings"]
    assert result["measure_round_dropped"]["bound"] == "prelude_ceiling"


def test_measured_round_survives_a_budget_that_still_covers_it(tmp_path):
    """The guard must not turn every double-run into a single one."""
    result = _run_double_run_baseline(
        tmp_path,
        _prelude_shared_state(spent_sec=300.0, usable_sec=10_000.0),
    )

    assert result["_rounds_run"] == 2
    assert result["output_throughput"] == pytest.approx(_HOT_TPUT)
    assert "measure_round_dropped" not in result


def test_deferred_accuracy_skips_eval_when_hot_throughput_regresses(
    tmp_path,
):
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "defer_accuracy_until_after_measure": True,
            "post_measure_accuracy_min_tput": _HOT_TPUT + 1,
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert all(
        cfg["benchmark"]["envs"]["RUN_EVAL"] == "false"
        for cfg in captured
    )
    assert result["accuracy_stage"]["status"] == "skipped"
    assert (
        result["accuracy_stage"]["reason"]
        == "throughput_below_threshold"
    )


def test_deferred_accuracy_reuses_hot_server_after_throughput_passes(
    tmp_path,
):
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    captured: list = []
    state = {"calls": 0}

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        cfg_idx = cmd.index("--benchmark-config")
        cfg = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
        captured.append(cfg)
        tput = _COLD_TPUT if state["calls"] == 0 else _HOT_TPUT
        state["calls"] += 1
        _fake_workspace(slot, tput=tput)
        if cfg["benchmark"]["envs"].get("RUN_EVAL") == "true":
            (slot / "results_gsm8k.json").write_text(
                json.dumps(
                    {
                        "results": {
                            "gsm8k": {
                                "exact_match,strict-match": 0.9,
                            }
                        }
                    }
                )
            )
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "defer_accuracy_until_after_measure": True,
            "post_measure_accuracy_min_tput": _HOT_TPUT - 1,
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 3
    assert [
        cfg["benchmark"]["envs"]["RUN_EVAL"]
        for cfg in captured
    ] == ["false", "false", "true"]
    assert [
        cfg["benchmark"]["server_lifecycle"]["cleanup"]
        for cfg in captured
    ] == [False, False, True]
    assert result["accuracy"] == pytest.approx(0.9)
    assert result["accuracy_stage"]["status"] == "succeeded"


def test_deferred_accuracy_is_cancelled_by_no_eval(tmp_path):
    """The staged accuracy round is an eval, so ``--no-eval`` drops it."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)

    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "defer_accuracy_until_after_measure": True,
            "post_measure_accuracy_min_tput": _HOT_TPUT - 1,
        }
    )
    ctx.extra["shared_state"] = SimpleNamespace(eval_disabled=True, baseline_double_run=True)

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert [cfg["benchmark"]["envs"]["RUN_EVAL"] for cfg in captured] == ["false", "false"]
    assert "accuracy_stage" not in result


def test_deferred_accuracy_single_round_keeps_eval_enabled(tmp_path):
    """Ineligible lifecycle fallback must retain accuracy in its only round."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    captured: list = []

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        cfg_idx = cmd.index("--benchmark-config")
        cfg = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
        captured.append(cfg)
        _fake_workspace(slot, tput=_HOT_TPUT)
        (slot / "results_gsm8k.json").write_text(
            json.dumps(
                {
                    "results": {
                        "gsm8k": {
                            "exact_match,strict-match": 0.9,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "defer_accuracy_until_after_measure": True,
            "post_measure_accuracy_min_tput": _HOT_TPUT - 1,
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert len(captured) == 1
    assert captured[0]["benchmark"]["envs"]["RUN_EVAL"] == "true"
    assert result["accuracy"] == pytest.approx(0.9)
    assert "accuracy_stage" not in result


def test_baseline_double_run_by_default(tmp_path, monkeypatch):
    """Baseline defaults to cold+hot rounds to match EXPLORE warm-decision."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_BASELINE_DOUBLE_RUN", raising=False)
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert result["output_throughput"] == pytest.approx(_HOT_TPUT)
    assert result.get("warmup_round_tput") == pytest.approx(_COLD_TPUT)
    assert "baseline_double_run_discarded_first" in result["nonfatal_warnings"]
    assert captured[0]["benchmark"]["server_lifecycle"]["cleanup"] is False
    assert captured[1]["benchmark"]["server_lifecycle"]["cleanup"] is True


def test_baseline_double_run_can_be_disabled_by_task_param(tmp_path, monkeypatch):
    """Focused callers may explicitly opt out of the default cold+hot baseline."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "baseline_double_run": False,
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 1
    assert result["output_throughput"] == pytest.approx(_COLD_TPUT)
    assert "server_lifecycle" not in captured[0]["benchmark"]


def test_baseline_double_run_loads_persisted_session_opt_out(tmp_path):
    """A fresh executor process can recover a session-level opt-out from SharedState."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    state = SharedState.load_or_init(session_dir)
    state.baseline_double_run = False
    state.save(session_dir)

    executor = BaselineExecutor(
        magpie_python=sys.executable,
        session_dir=session_dir,
        shared_state=None,
    )

    assert executor._double_run_enabled() is False


def test_run_grid_discards_cold_first_round_via_lifecycle(tmp_path, monkeypatch):
    """The shared grid runner reports the HOT measured round when lifecycle reuse is eligible."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "1")
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "grid"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)

    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        results = _run(
            run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant(name="candidate")],
                output_root=output_dir,
                magpie_python=sys.executable,
                variant_timeout_sec=10,
                gpu_type="mi300x",
            )
        )

    assert state["calls"] == 2
    assert len(results) == 1
    result = results[0]
    assert result.status == "succeeded"
    assert result.output_throughput == pytest.approx(_HOT_TPUT)
    assert "run_grid_warmup_discarded_first" in result.nonfatal_warnings

    assert len(captured) == 2
    warmup_lc = captured[0]["benchmark"]["server_lifecycle"]
    measure_lc = captured[1]["benchmark"]["server_lifecycle"]
    assert warmup_lc["cleanup"] is False
    assert measure_lc["cleanup"] is True
    assert warmup_lc["pid_dir"] == measure_lc["pid_dir"] == str(output_dir / "variant_00_candidate")


def test_run_grid_single_round_when_warmup_disabled(tmp_path, monkeypatch):
    """``INFERENCE_OPTIMIZER_RUN_GRID_WARMUP=0`` keeps the legacy single-round grid path."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "grid"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)

    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        results = _run(
            run_grid(
                base_yaml_path=base,
                base_extra_args="",
                grid=[GridVariant(name="candidate")],
                output_root=output_dir,
                magpie_python=sys.executable,
                variant_timeout_sec=10,
                gpu_type="mi300x",
            )
        )

    assert state["calls"] == 1
    assert len(results) == 1
    result = results[0]
    assert result.status == "succeeded"
    assert result.output_throughput == pytest.approx(_COLD_TPUT)
    assert "run_grid_warmup_discarded_first" not in result.nonfatal_warnings
    assert "server_lifecycle" not in captured[0]["benchmark"]


def test_baseline_single_round_when_script_not_builtin(tmp_path):
    """A non-builtin benchmark script falls back to one round even with double-run on."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "benchmark_script": "dsr1_fp8_mi300x.sh",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 1
    assert result["output_throughput"] == pytest.approx(_COLD_TPUT)
    assert "server_lifecycle" not in captured[0]["benchmark"]


def test_baseline_warmup_round_failure_short_circuits(tmp_path, monkeypatch):
    """A failed warmup round returns immediately and does NOT run a second round."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    state = {"calls": 0}

    def fake_run(cmd, *args, **kwargs):
        state["calls"] += 1
        return subprocess.CompletedProcess(cmd, 1, "", "boom: server crashed")

    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert state["calls"] == 1
    assert "baseline_warmup_round_failed" in result.get("nonfatal_warnings", [])


def test_baseline_no_workspace_persists_stderr_to_file(tmp_path):
    """When Magpie exits nonzero before creating a benchmark_* workspace, the
    executor must persist the captured stderr to ``baseline_stderr.log`` so the
    failure leaves an on-disk artifact that survives the NFS clone / S3 archive."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="sglang")
    output_dir = tmp_path / "ws"
    crash_text = "torch.OutOfMemoryError: HIP out of memory (workspace_buffer)"

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", crash_text)

    executor = _executor(base, tmp_path)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert result["error_class"] == "subprocess_nonzero"
    log_path = result.get("stderr_log_path")
    assert log_path is not None, result
    saved = Path(log_path)
    assert saved.exists() and saved.name == "baseline_stderr.log"
    assert crash_text in saved.read_text(encoding="utf-8")


def test_baseline_classifies_vllm_engine_init_as_server_init_dead(
    tmp_path,
    monkeypatch,
):
    """A vLLM engine-core bootstrap failure (server.log carries ``Engine core
    initialization failed`` while Magpie exits nonzero without a benchmark_*
    workspace) is classified ``server_init_dead`` with the server.log root cause
    surfaced in ``error``."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "server.log").write_text(
            "(APIServer pid=16160)   File '.../vllm/v1/engine/utils.py', "
            "line 1057, in wait_for_engine_startup\n"
            "(APIServer pid=16160) RuntimeError: Engine core initialization "
            "failed. See root cause above. Failed core proc(s): {}\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 1, "", "magpie wrapper noise")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert result["error_class"] == "server_init_dead", result
    assert "Engine core initialization failed" in result["error"]


def test_baseline_server_dead_returncode_classifies_server_init_dead(
    tmp_path,
    monkeypatch,
):
    """When the liveness watchdog reaps a hung server
    (``SERVER_DEAD_RETURNCODE``), baseline classifies it ``server_init_dead``
    even when no server.log marker is independently visible."""
    from hyperloom.orchestrator.actions.executors._subprocess_kill import (
        SERVER_DEAD_RETURNCODE,
    )

    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="sglang")
    output_dir = tmp_path / "ws"

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, SERVER_DEAD_RETURNCODE, "", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert result["error_class"] == "server_init_dead", result


def test_baseline_invalid_measurement_with_server_death_marker_is_dead(
    tmp_path,
    monkeypatch,
):
    """When Magpie creates a benchmark_* workspace with no valid measurement, a
    server.log death marker takes precedence — the failure is classified
    ``server_init_dead`` and the real engine fault is surfaced in ``error``."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        # Workspace exists but has no report, so the measurement is invalid.
        (slot / "benchmark_vllm_20260602_010101").mkdir(parents=True)
        (slot / "server.log").write_text(
            "(APIServer pid=42) RuntimeError: Engine core initialization "
            "failed. See root cause above. Failed core proc(s): {} "
            "OPENAI_API_KEY=ak-invalid-measurement-secret\n",
            encoding="utf-8",
        )
        # Classification must be driven by the server.log marker, not returncode.
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert result["error_class"] == "server_init_dead", result
    assert "Engine core initialization failed" in result["error"]
    assert "invalid-measurement-secret" not in result["error"]
    assert "[REDACTED]" in result["error"]


def test_baseline_clears_stale_server_log_before_run(tmp_path, monkeypatch):
    """A stale server.log death marker in a reused output_dir must NOT bias a
    fresh attempt's classification. The executor clears the prior log before
    launching, so an attempt that boots but yields no report is classified by
    its own outcome (``no_report``), never ``server_init_dead``."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    output_dir.mkdir(parents=True)
    (output_dir / "server.log").write_text(
        "(APIServer pid=1) RuntimeError: Engine core initialization failed. Failed core proc(s): {}\n",
        encoding="utf-8",
    )

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        # Boots fine, produces no report, does NOT rewrite a death marker.
        (slot / "benchmark_vllm_20260602_010101").mkdir(parents=True)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert result["error_class"] != "server_init_dead", result
    assert result["error_class"] == "no_report", result


def test_baseline_rejects_stale_workspace_on_crash(tmp_path, monkeypatch):
    """A stale benchmark_* workspace from a prior attempt must not be adopted
    as a successful result when the current subprocess crashes (rc=1)."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    stale = output_dir / "benchmark_vllm_20260101_000000"
    stale.mkdir(parents=True)
    (stale / "benchmark_report.json").write_text(
        json.dumps({
            "success": True,
            "framework": "vllm",
            "throughput": {
                "output_throughput": 9999.0,
                "completed_requests": 64,
                "duration_seconds": 25.0,
            },
            "latency": {"ttft": {"mean_ms": 100.0}, "e2el": {"mean_ms": 2000.0}},
        }),
        encoding="utf-8",
    )
    old = 1735689600.0
    os.utime(stale / "benchmark_report.json", (old, old))
    os.utime(stale, (old, old))

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "HIP out of memory")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "gpu_type": "mi300x"})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed", result
    assert result.get("output_throughput") is None


def test_baseline_rejects_stale_workspace_on_silent_exit(tmp_path, monkeypatch):
    """A stale benchmark_* workspace must not be adopted when the subprocess
    exits 0 without producing any new workspace (silent no-op)."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    stale = output_dir / "benchmark_vllm_20260101_000000"
    stale.mkdir(parents=True)
    (stale / "benchmark_report.json").write_text(
        json.dumps({
            "success": True,
            "framework": "vllm",
            "throughput": {
                "output_throughput": 9999.0,
                "completed_requests": 64,
                "duration_seconds": 25.0,
            },
            "latency": {"ttft": {"mean_ms": 100.0}, "e2el": {"mean_ms": 2000.0}},
        }),
        encoding="utf-8",
    )
    old = 1735689600.0
    os.utime(stale / "benchmark_report.json", (old, old))
    os.utime(stale, (old, old))

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "gpu_type": "mi300x"})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed", result
    assert result.get("output_throughput") is None


def test_baseline_rejects_stale_workspace_when_the_run_produced_none(tmp_path, monkeypatch):
    """A crashed run adopts no workspace, however high the stale one sorts."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    stale = output_dir / "benchmark_vllm_29991231_235959"
    stale.mkdir(parents=True)
    (stale / "benchmark_report.json").write_text(
        json.dumps({
            "success": True,
            "framework": "vllm",
            "throughput": {
                "output_throughput": 9999.0,
                "completed_requests": 64,
                "duration_seconds": 25.0,
            },
            "latency": {"ttft": {"mean_ms": 100.0}, "e2el": {"mean_ms": 2000.0}},
        }),
        encoding="utf-8",
    )
    old = 1735689600.0
    os.utime(stale / "benchmark_report.json", (old, old))
    os.utime(stale, (old, old))

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "gpu_type": "mi300x"})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed", result
    assert result.get("output_throughput") is None


def test_baseline_picks_fresh_workspace_sorting_before_a_stale_one(tmp_path, monkeypatch):
    """The fresh workspace wins even when the stale one sorts last.

    Every other case here creates a fresh name that also sorts last, so they
    pass against the lexicographic-last glob this replaced.
    """
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    stale = output_dir / "benchmark_vllm_29991231_235959"
    stale.mkdir(parents=True)
    (stale / "benchmark_report.json").write_text(
        json.dumps({
            "success": True,
            "framework": "vllm",
            "throughput": {"output_throughput": 9999.0, "completed_requests": 64},
        }),
        encoding="utf-8",
    )
    old = 1735689600.0
    os.utime(stale / "benchmark_report.json", (old, old))
    os.utime(stale, (old, old))

    def fake_run(cmd, *args, **kwargs):
        _fake_workspace(Path(cmd[cmd.index("--output-dir") + 1]), tput=4000.0)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "gpu_type": "mi300x"})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded", result
    assert result.get("output_throughput") == pytest.approx(4000.0)


def test_baseline_fresh_workspace_succeeds_despite_stale_peer(tmp_path, monkeypatch):
    """A new workspace with valid throughput produced by the current run succeeds
    even when an older stale workspace is present in the same output_dir."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    stale = output_dir / "benchmark_vllm_20260101_000000"
    stale.mkdir(parents=True)
    (stale / "benchmark_report.json").write_text(
        json.dumps({
            "success": True,
            "framework": "vllm",
            "throughput": {"output_throughput": 1.0, "completed_requests": 1},
        }),
        encoding="utf-8",
    )
    old = 1735689600.0
    os.utime(stale / "benchmark_report.json", (old, old))
    os.utime(stale, (old, old))

    def fake_run(cmd, *args, **kwargs):
        _fake_workspace(Path(cmd[cmd.index("--output-dir") + 1]), tput=4000.0)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "gpu_type": "mi300x"})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded", result
    assert result.get("output_throughput") == pytest.approx(4000.0)


def test_ensure_local_inferencex_noop_for_local_path(tmp_path, monkeypatch):
    """A checkout already on a local filesystem is returned unchanged."""
    from hyperloom.orchestrator.actions.executors import baseline as bl

    src = tmp_path / "InferenceX"
    (src / "benchmarks").mkdir(parents=True)
    (src / "benchmarks" / "benchmark_lib.sh").write_text("# stub")
    monkeypatch.setattr(bl, "_is_network_fs", lambda p: False)

    assert bl._ensure_local_inferencex(str(src)) == str(src)


def test_ensure_local_inferencex_mirrors_network_path(tmp_path, monkeypatch):
    """A checkout on a simulated network mount is mirrored to local disk and the
    returned path points at the local copy, not the original."""
    from hyperloom.orchestrator.actions.executors import baseline as bl

    src = tmp_path / "wekafs_InferenceX"
    (src / "benchmarks").mkdir(parents=True)
    (src / "benchmarks" / "benchmark_lib.sh").write_text("# patched lib")
    (src / "utils").mkdir()
    (src / "utils" / "marker.txt").write_text("payload")

    local_root = tmp_path / "local_cache"
    monkeypatch.setattr(bl, "_is_network_fs", lambda p: True)
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT",
        str(local_root),
    )

    dest = bl._ensure_local_inferencex(str(src))

    assert dest != str(src)
    assert str(local_root) in dest
    assert (Path(dest) / "benchmarks" / "benchmark_lib.sh").read_text() == ("# patched lib")
    assert (Path(dest) / "utils" / "marker.txt").read_text() == "payload"


def test_ensure_local_inferencex_isolates_per_task_mirrors(
    tmp_path,
    monkeypatch,
):
    """Callers can include a task/output-dir key in the mirror hash so two
    overlapping baselines sharing one wekafs checkout never rmtree/replace a
    directory that another server is currently ``cd``-ed into."""
    from hyperloom.orchestrator.actions.executors import baseline as bl

    src = tmp_path / "wekafs_InferenceX"
    (src / "benchmarks").mkdir(parents=True)
    (src / "benchmarks" / "benchmark_lib.sh").write_text("# patched lib")
    local_root = tmp_path / "local_cache"
    monkeypatch.setattr(bl, "_is_network_fs", lambda p: True)
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT",
        str(local_root),
    )

    dest_a = bl._ensure_local_inferencex(str(src), mirror_key="task-a")
    dest_b = bl._ensure_local_inferencex(str(src), mirror_key="task-b")

    assert dest_a != dest_b
    assert (Path(dest_a) / "benchmarks" / "benchmark_lib.sh").is_file()
    assert (Path(dest_b) / "benchmarks" / "benchmark_lib.sh").is_file()


def test_ensure_local_inferencex_disabled_by_env(tmp_path, monkeypatch):
    """The relocation can be opted out of via env even on a network mount."""
    from hyperloom.orchestrator.actions.executors import baseline as bl

    src = tmp_path / "wekafs_InferenceX"
    (src / "benchmarks").mkdir(parents=True)
    (src / "benchmarks" / "benchmark_lib.sh").write_text("# stub")
    monkeypatch.setattr(bl, "_is_network_fs", lambda p: True)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_LOCAL_INFERENCEX", "1")

    assert bl._ensure_local_inferencex(str(src)) == str(src)


def test_ensure_local_inferencex_falls_back_on_copy_failure(
    tmp_path,
    monkeypatch,
):
    """When the mirror copy itself fails (e.g. local disk full), the helper
    degrades to the original network-mount path instead of raising, so the run
    still proceeds rather than aborting."""
    from hyperloom.orchestrator.actions.executors import baseline as bl

    src = tmp_path / "wekafs_InferenceX"
    (src / "benchmarks").mkdir(parents=True)
    (src / "benchmarks" / "benchmark_lib.sh").write_text("# patched")
    local_root = tmp_path / "local_cache"
    monkeypatch.setattr(bl, "_is_network_fs", lambda p: True)
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT",
        str(local_root),
    )

    def _boom(*_a, **_k):
        raise OSError("no space left on device")

    monkeypatch.setattr(bl.shutil, "copytree", _boom)

    assert bl._ensure_local_inferencex(str(src)) == str(src)


def test_ensure_local_inferencex_falls_back_when_mirror_incomplete(
    tmp_path,
    monkeypatch,
):
    """If the copy lands but the mirror is missing the load-bearing
    ``benchmarks/benchmark_lib.sh``, the helper rejects it and returns the
    original path rather than handing Magpie a broken ``cd`` target."""
    from hyperloom.orchestrator.actions.executors import baseline as bl

    src = tmp_path / "wekafs_InferenceX"
    (src / "utils").mkdir(parents=True)
    (src / "utils" / "marker.txt").write_text("payload")
    local_root = tmp_path / "local_cache"
    monkeypatch.setattr(bl, "_is_network_fs", lambda p: True)
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT",
        str(local_root),
    )

    assert bl._ensure_local_inferencex(str(src)) == str(src)
    assert not [p for p in local_root.iterdir() if p.is_dir()]


def test_baseline_points_magpie_at_local_inferencex(tmp_path, monkeypatch):
    """When INFERENCEX_PATH is on a network mount, the local mirror is what
    Magpie actually ``cd``-s into. Asserts both channels:

    * the materialized YAML's ``benchmark.inferencex_path`` (the field Magpie's
      ``_build_local_command`` honours — the real ``cd`` target), and
    * the ``MAGPIE_INFERENCEX_PATH`` env fallback.
    """
    from hyperloom.orchestrator.actions.executors import baseline as bl

    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="sglang")
    output_dir = tmp_path / "ws"

    ix_src = tmp_path / "wekafs_InferenceX"
    (ix_src / "benchmarks").mkdir(parents=True)
    # This test is about which InferenceX dir Magpie cd-s into, but the launch
    # path runs the real patcher, which refuses to start an eval whose patches
    # cannot be applied. So the stub has to carry the anchors a checkout carries.
    (ix_src / "benchmarks" / "benchmark_lib.sh").write_text(
        "# patched\n"
        "run_eval() {\n"
        '    export EVAL_RESULT_DIR="$results_dir"\n'
        "}\n"
        "append_lm_eval_summary() {\n"
        '    mv -f "$jf" ./ || echo "WARN: failed to move ${jf}" >&2\n'
        "}\n"
    )
    local_root = tmp_path / "local_cache"
    monkeypatch.setattr(bl, "_is_network_fs", lambda p: True)
    monkeypatch.setenv("INFERENCEX_PATH", str(ix_src))
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT",
        str(local_root),
    )

    seen: dict = {}

    def fake_run(cmd, *args, **kwargs):
        seen["env"] = kwargs.get("env")
        cfg_idx = cmd.index("--benchmark-config")
        seen["materialized_cfg"] = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=_HOT_TPUT)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    yaml_ix = seen["materialized_cfg"]["benchmark"]["inferencex_path"]
    assert yaml_ix != str(ix_src), seen["materialized_cfg"]
    assert str(local_root) in yaml_ix
    magpie_ix = seen["env"]["MAGPIE_INFERENCEX_PATH"]
    assert magpie_ix != str(ix_src), seen["env"]
    assert str(local_root) in magpie_ix
    # Relocation is task-local; process-wide env stays the original source path.
    assert os.environ["INFERENCEX_PATH"] == str(ix_src)


def test_baseline_anchors_server_cwd_to_output_dir(tmp_path, monkeypatch):
    """The Magpie parent subprocess cwd is anchored to the stable task
    output_dir (never the default ``/tmp``) as defence-in-depth."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    seen: dict = {}

    def fake_run(cmd, *args, **kwargs):
        seen["cwd"] = kwargs.get("cwd")
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=_HOT_TPUT)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = _executor(base, tmp_path, baseline_double_run=False)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert seen["cwd"] is not None
    assert seen["cwd"] != "/tmp"
    assert str(output_dir) in seen["cwd"]


def test_atom_engages_double_run_like_vllm_sglang(tmp_path, monkeypatch):
    """Atom baseline engages the lifecycle double-run like vllm/sglang."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="atom")
    output_dir = tmp_path / "ws"

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert result["output_throughput"] == pytest.approx(_HOT_TPUT)
    assert captured[0]["benchmark"]["benchmark_script"] == "atom_mi300x.sh"
    assert captured[0]["benchmark"]["server_lifecycle"]["enabled"] is True


def test_double_run_runtime_anchor_is_full_warmup_round(tmp_path, monkeypatch):
    """The overtime-kill anchor must reflect round 1's FULL run, not round 2's reuse time."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    state = {"calls": 0}

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        if state["calls"] == 0:
            time.sleep(0.6)
            tput = _COLD_TPUT
        else:
            tput = _HOT_TPUT
        state["calls"] += 1
        _fake_workspace(slot, tput=tput)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2
    assert result["subprocess_runtime_sec"] >= 0.5
    assert "measure_round_runtime_sec" in result
    assert result["measure_round_runtime_sec"] < result["subprocess_runtime_sec"]


def test_double_run_pre_start_cleanup_kills_zombie_and_clears_stale_meta(
    tmp_path,
    monkeypatch,
):
    """When the reuse port is occupied by a zombie (healthy but no metadata),
    pre-start cleanup must (a) unlink stale pid/json without sending signals to
    potentially-recycled PIDs, and (b) invoke _kill_stale_servers() to reap the
    zombie listener."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    output_dir.mkdir(parents=True)
    (output_dir / "vllm_8888.pid").write_text("2147483646")
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._server_lifecycle._pick_free_port",
        lambda: 8888,
    )

    kill_calls = {"n": 0}

    def fake_kill():
        kill_calls["n"] += 1

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with (
        patch.object(
            type(executor),
            "_port_healthy",
            return_value=True,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline._kill_stale_servers",
            side_effect=fake_kill,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert kill_calls["n"] == 1
    assert not (output_dir / "vllm_8888.pid").exists()
    assert state["calls"] == 2
    assert result["output_throughput"] == pytest.approx(_HOT_TPUT)
    warmup_lc = captured[0]["benchmark"]["server_lifecycle"]
    measure_lc = captured[1]["benchmark"]["server_lifecycle"]
    assert warmup_lc["cleanup"] is False
    assert measure_lc["cleanup"] is True
    assert warmup_lc["pid_dir"] == measure_lc["pid_dir"] == str(output_dir)


def test_pre_start_cleanup_no_kill_when_port_free(tmp_path, monkeypatch):
    """When the port is NOT occupied (no zombie), _kill_stale_servers must
    NOT fire — avoids killing unrelated servers sharing the pod."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    output_dir.mkdir(parents=True)
    (output_dir / "vllm_8888.pid").write_text("2147483646")
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._server_lifecycle._pick_free_port",
        lambda: 8888,
    )

    kill_calls = {"n": 0}

    def fake_kill():
        kill_calls["n"] += 1

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with (
        patch.object(
            type(executor),
            "_port_healthy",
            return_value=False,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline._kill_stale_servers",
            side_effect=fake_kill,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert kill_calls["n"] == 0
    assert not (output_dir / "vllm_8888.pid").exists()
    assert state["calls"] == 2


def test_pre_start_cleanup_no_kill_when_metadata_existed(tmp_path, monkeypatch):
    """A healthy port with matching metadata is not a zombie signal.

    The global stale-server reaper must not fire for a likely legitimate
    server. File preservation is covered by the direct pre-start test below;
    this full double-run path later removes files in final teardown.
    """
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"
    output_dir.mkdir(parents=True)
    (output_dir / "vllm_8888.pid").write_text("2147483646")
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._server_lifecycle._pick_free_port",
        lambda: 8888,
    )
    (output_dir / "vllm_8888.json").write_text("{}")

    kill_calls = {"n": 0}

    def fake_kill():
        kill_calls["n"] += 1

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with (
        patch.object(
            type(executor),
            "_port_healthy",
            return_value=True,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline._kill_stale_servers",
            side_effect=fake_kill,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert kill_calls["n"] == 0
    assert state["calls"] == 2


def test_pre_start_cleanup_preserves_metadata_when_reuse_target_healthy(
    tmp_path,
):
    """Direct guard: do not create port-occupied/no-metadata state."""
    output_dir = tmp_path / "ws"
    output_dir.mkdir(parents=True)
    pid_file = output_dir / "vllm_8888.pid"
    meta_file = output_dir / "vllm_8888.json"
    pid_file.write_text("2147483646")
    meta_file.write_text("{}")

    executor = _executor(tmp_path / "base.yaml", tmp_path)
    kill_calls = {"n": 0}

    def fake_kill():
        kill_calls["n"] += 1

    with (
        patch.object(
            type(executor),
            "_port_healthy",
            return_value=True,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline._kill_stale_servers",
            side_effect=fake_kill,
        ),
    ):
        executor._pre_start_cleanup(
            pid_dir=output_dir,
            framework="vllm",
            port=8888,
        )

    assert kill_calls["n"] == 0
    assert pid_file.exists()
    assert meta_file.exists()


def test_pre_start_cleanup_failure_does_not_break_double_run(tmp_path, monkeypatch):
    """The pre-start cleanup is best-effort: a raising _kill_stale_servers()
    must not abort the run — the double-run proceeds and still succeeds."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    def boom():
        raise RuntimeError("proc scan blew up")

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path, baseline_double_run=True)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
        }
    )

    with (
        patch.object(
            type(executor),
            "_port_healthy",
            return_value=True,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline._kill_stale_servers",
            side_effect=boom,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 2


def test_pre_start_cleanup_skipped_when_single_round(tmp_path, monkeypatch):
    """Single-round (double-run disabled) keeps legacy behaviour: the
    pre-start deep clean is a double-run-only concern and must not fire."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    output_dir = tmp_path / "ws"

    kill_calls = {"n": 0}

    def fake_kill():
        kill_calls["n"] += 1

    captured: list = []
    fake_run, state = _cold_then_hot_fake_run(captured)
    executor = _executor(base, tmp_path)
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "gpu_type": "mi300x",
            "baseline_double_run": False,
        }
    )

    with (
        patch(
            "hyperloom.orchestrator.actions.executors.baseline._kill_stale_servers",
            side_effect=fake_kill,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state["calls"] == 1
    assert kill_calls["n"] == 0


def test_teardown_lifecycle_server_removes_state_files(tmp_path):
    """The defensive teardown unlinks stale pid/meta files without raising."""
    executor = _executor(tmp_path / "base.yaml", tmp_path)
    _write_yaml(tmp_path / "base.yaml", framework="vllm")
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    (pid_dir / "vllm_8888.pid").write_text("2147483646")
    (pid_dir / "vllm_8888.json").write_text("{}")

    executor._teardown_lifecycle_server(
        pid_dir=pid_dir,
        framework="vllm",
        port=8888,
    )

    assert not (pid_dir / "vllm_8888.pid").exists()
    assert not (pid_dir / "vllm_8888.json").exists()


# _classify_subprocess_error unit tests

from hyperloom.orchestrator.actions.executors.baseline import (
    _classify_subprocess_error,
)


def test_classify_fast_exit_unknown_backend():
    assert (
        _classify_subprocess_error(5.0, "ValueError: Unknown attention backend: 'ROCM_FLASH'") == "fast_exit_arg_error"
    )


def test_classify_fast_exit_unrecognized_args():
    assert _classify_subprocess_error(2.0, "error: unrecognized arguments: --bogus-flag") == "fast_exit_arg_error"


def test_classify_slow_failure_not_arg_error():
    """A slow failure (>30s) with the same stderr pattern must NOT be
    classified as arg error — it could be a real inference crash."""
    assert _classify_subprocess_error(120.0, "ValueError: some runtime error") == "subprocess_nonzero"


def test_classify_fast_exit_without_pattern():
    """A fast exit without arg-error patterns stays subprocess_nonzero."""
    assert _classify_subprocess_error(3.0, "Segmentation fault (core dumped)") == "subprocess_nonzero"


def test_classify_fast_runtime_value_error_not_arg_error():
    """A generic fast runtime ValueError is not enough for arg-error routing."""
    assert _classify_subprocess_error(3.0, "ValueError: tensor shape mismatch during warmup") == "subprocess_nonzero"


def test_classify_value_error_with_argv_dump_not_arg_error():
    """A command/argv dump containing flags is not arg validation by itself."""
    assert (
        _classify_subprocess_error(
            3.0,
            "ValueError: tensor shape mismatch during warmup\nargv: vllm serve --model /models/foo --tp 8",
        )
        == "subprocess_nonzero"
    )


def test_classify_subprocess_error_none_tail_does_not_crash():
    # A slow failure with no captured stderr must not raise.
    assert _classify_subprocess_error(600.0, None) == "subprocess_nonzero"


def test_classify_kv_cache_oom_after_weight_load():
    # KV-cache OOM must be detected regardless of elapsed time.
    tail = (
        "ValueError: Loaded weights leave no GPU memory for the KV cache "
        "under --mem-fraction-static=0.7. Raise --mem-fraction-static above 0.737"
    )
    assert _classify_subprocess_error(600.0, tail) == "kv_cache_oom"


def test_classify_kv_cache_oom_fast_exit():
    tail = "no GPU memory for the KV cache"
    assert _classify_subprocess_error(3.0, tail) == "kv_cache_oom"


def test_classify_non_kv_oom_still_nonzero():
    assert _classify_subprocess_error(600.0, "some other runtime failure") == "subprocess_nonzero"
