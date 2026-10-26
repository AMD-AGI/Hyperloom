# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A cycle-bench batch: its arms in order, the env each runs, and the verdicts its two bases license."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors import cycle_bench_arm as cba
from hyperloom.orchestrator.measurement import cycle_bench as cb

_WEIGHTS = {1: 0.5, 2: 0.5}


def _arm(c1: float, c2: float, error: str = "") -> cba.ArmResult:
    return cba.ArmResult(cycle_ms={1: c1, 2: c2}, accept_len={1: 3.6, 2: 3.6}, error=error)


def test_candidates_are_graded_against_the_mean_of_two_agreeing_bases():
    arms = {"base": _arm(10.0, 20.0), "faster": _arm(9.5, 19.0), "slower": _arm(10.4, 20.8), "base2": _arm(10.0, 20.0)}

    summary = cba.summarize_batch(arms, ["faster", "slower"], _WEIGHTS, band_pct=2.0)

    assert summary["valid"] is True
    assert {name: v["verdict"] for name, v in summary["verdicts"].items()} == {
        "faster": cb.VERDICT_PASS,
        "slower": cb.VERDICT_REGRESS,
    }
    assert summary["verdicts"]["faster"]["gain_pct"] == pytest.approx(5.0)


def test_bases_that_disagree_void_the_batch_and_grade_nothing():
    arms = {"base": _arm(10.0, 20.0), "c": _arm(9.0, 18.0), "base2": _arm(10.5, 21.0)}

    summary = cba.summarize_batch(arms, ["c"], _WEIGHTS, band_pct=2.0)

    assert (summary["valid"], summary["verdicts"]) == (False, {})


def test_a_candidate_whose_server_failed_is_broken_not_graded():
    arms = {
        "base": _arm(10.0, 20.0),
        "c": _arm(0.0, 0.0, error="server did not become ready"),
        "base2": _arm(10.0, 20.0),
    }

    summary = cba.summarize_batch(arms, ["c"], _WEIGHTS, band_pct=2.0)

    assert summary["verdicts"] == {"c": {"verdict": cb.VERDICT_BROKEN, "error": "server did not become ready"}}


def _batch(tmp_path: Path) -> cba.BatchSpec:
    return cba.BatchSpec(
        argv=["python3", "-m", "sglang.launch_server", "--port", "8888", "--enable-hierarchical-cache"],
        env={"HIP_VISIBLE_DEVICES": "0,1,2,3", "SGLANG_SIMULATE_ACC_LEN": "3.61"},
        weights=_WEIGHTS,
        candidates=[("lever-a", {"HL_A": "1"})],
        out_dir=tmp_path,
        port=9100,
        prompt_tokens=1000,
        vocab_size=50000,
        band_pct=2.0,
    )


def test_a_batch_runs_its_bases_around_the_candidates_on_the_bench_shaped_launch(tmp_path, monkeypatch):
    ran: list[tuple[str, list[str], dict[str, str]]] = []

    def _run_arm(spec):
        ran.append(
            (spec.out_dir.name, spec.argv, {k: spec.env.get(k, "") for k in ("HL_A", "SGLANG_SIMULATE_ACC_TOKEN_MODE")})
        )
        return _arm(10.0, 20.0)

    monkeypatch.setattr(cba, "run_arm", _run_arm)

    summary = cba.run_batch(_batch(tmp_path))

    bench_argv = ["python3", "-m", "sglang.launch_server", "--port", "9100"]
    assert ran == [
        ("base", bench_argv, {"HL_A": "", "SGLANG_SIMULATE_ACC_TOKEN_MODE": "fixed"}),
        ("lever-a", bench_argv, {"HL_A": "1", "SGLANG_SIMULATE_ACC_TOKEN_MODE": "fixed"}),
        ("base2", bench_argv, {"HL_A": "", "SGLANG_SIMULATE_ACC_TOKEN_MODE": "fixed"}),
    ]
    assert summary["valid"] is True


def test_a_batch_stops_when_its_first_base_fails(tmp_path, monkeypatch):
    ran: list[str] = []

    def _run_arm(spec):
        ran.append(spec.out_dir.name)
        return _arm(0.0, 0.0, error="server did not become ready")

    monkeypatch.setattr(cba, "run_arm", _run_arm)

    summary = cba.run_batch(_batch(tmp_path))

    assert ran == ["base"]
    assert summary["valid"] is False


def test_the_session_batch_reads_the_baselines_launch_log_corpus_and_vocabulary(tmp_path):
    workspace = tmp_path / "baseline" / "benchmark_sglang_1"
    workspace.mkdir(parents=True)
    launch = {"effective_argv": ["python3", "-m", "sglang.launch_server"], "runtime_environment": {"X": "1"}}
    (workspace / "agentx_server_launch.json").write_text(json.dumps(launch), encoding="utf-8")
    decode = "[t TP0] Decode batch, #running-req: {n}, #token: 1, accept len: 3.60, gen throughput (token/s): 300.00, #queue-req: 0"
    (workspace / "server.log").write_text("\n".join([decode.format(n=1)] * 3 + [decode.format(n=2)]), encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"vocab_size": 154880}), encoding="utf-8")
    state = SimpleNamespace(
        baseline_workspace=str(tmp_path / "baseline"),
        model_path=str(model),
        agentx_corpus_shape={"isl": {"p50": 108000}},
    )

    spec = cba.session_batch(state, [("lever-a", {"HL_A": "1"})], out_dir=tmp_path / "out", band_pct=2.0, port=9100)

    assert (spec.argv, spec.env) == (launch["effective_argv"], {"X": "1"})
    assert (spec.weights, spec.prompt_tokens, spec.vocab_size) == ({1: 0.75, 2: 0.25, 4: 0.0}, 108000, 154880)
    assert spec.candidates == [("lever-a", {"HL_A": "1"})]


@pytest.mark.asyncio
async def test_an_explore_task_carrying_cycle_arms_runs_the_batch_instead_of_a_replay_round(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import explore

    seen: dict[str, object] = {}

    def _session_batch(state, candidates, *, out_dir, band_pct, port):
        seen.update(candidates=candidates, band_pct=band_pct, state=state)
        return "spec"

    monkeypatch.setattr(cba, "session_batch", _session_batch)
    monkeypatch.setattr(cba, "run_batch", lambda spec: {"valid": True, "verdicts": {"lever-a": {"verdict": "pass"}}})
    monkeypatch.setattr(explore, "_ExploreRound", None)
    state = SimpleNamespace()
    ctx = SimpleNamespace(
        task=SimpleNamespace(
            task_id="t-cycle",
            params={cba.CYCLE_ARMS_PARAM: [{"name": "lever-a", "envs": {"HL_A": "1"}}], "band_pct": 2.0},
        ),
        extra={"shared_state": state, "workspace": str(tmp_path)},
    )

    result = await explore.ExploreExecutor(session_dir=tmp_path)(ctx)

    assert result == {
        "status": "cycle_screened",
        cba.CYCLE_ARMS_PARAM: [{"name": "lever-a", "envs": {"HL_A": "1"}}],
        "valid": True,
        "verdicts": {"lever-a": {"verdict": "pass"}},
    }
    assert seen == {"candidates": [("lever-a", {"HL_A": "1"})], "band_pct": 2.0, "state": state}


@pytest.mark.asyncio
async def test_a_cycle_batch_without_a_baseline_launch_fails_named(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import explore

    def _missing(*_a, **_k):
        raise FileNotFoundError("no agentx_server_launch.json under the baseline workspace /nowhere")

    monkeypatch.setattr(cba, "session_batch", _missing)
    ctx = SimpleNamespace(
        task=SimpleNamespace(
            task_id="t-cycle", params={cba.CYCLE_ARMS_PARAM: [{"name": "a", "envs": {}}], "band_pct": 2.0}
        ),
        extra={"shared_state": SimpleNamespace(), "workspace": str(tmp_path)},
    )

    result = await explore.ExploreExecutor(session_dir=tmp_path)(ctx)

    assert (result["status"], result["error_class"]) == ("failed", "cycle_bench_unavailable")
