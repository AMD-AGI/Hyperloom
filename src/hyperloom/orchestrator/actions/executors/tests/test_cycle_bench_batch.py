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
_BANDS = cb.Bands(batch_pct=1.0, per_bs_pct=3.0, regress_pct=1.5)


def _arm(c1: float, c2: float, error: str = "") -> cba.ArmResult:
    return cba.ArmResult(cycle_ms={1: c1, 2: c2}, accept_len={1: 3.6, 2: 3.6}, error=error)


def test_candidates_are_graded_against_the_mean_of_two_agreeing_bases():
    arms = {"base": _arm(10.0, 20.0), "faster": _arm(9.5, 19.0), "slower": _arm(10.4, 20.8), "base2": _arm(10.0, 20.0)}

    summary = cba.summarize_batch(arms, ["faster", "slower"], _WEIGHTS, bands=_BANDS)

    assert summary["valid"] is True
    assert {name: v["verdict"] for name, v in summary["verdicts"].items()} == {
        "faster": cb.VERDICT_PASS,
        "slower": cb.VERDICT_REGRESS,
    }
    assert summary["verdicts"]["faster"]["gain_pct"] == pytest.approx(5.0)


def test_bases_that_disagree_void_the_batch_and_grade_nothing():
    arms = {"base": _arm(10.0, 20.0), "c": _arm(9.0, 18.0), "base2": _arm(10.5, 21.0)}

    summary = cba.summarize_batch(arms, ["c"], _WEIGHTS, bands=_BANDS)

    assert (summary["valid"], summary["verdicts"]) == (False, {})


def test_a_candidate_whose_server_failed_is_broken_not_graded():
    arms = {
        "base": _arm(10.0, 20.0),
        "c": _arm(0.0, 0.0, error="server did not become ready"),
        "base2": _arm(10.0, 20.0),
    }

    summary = cba.summarize_batch(arms, ["c"], _WEIGHTS, bands=_BANDS)

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
        bands=_BANDS,
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


def _referenced_batch(tmp_path: Path, monkeypatch, bases: list[tuple[float, float]]) -> tuple[list[str], dict]:
    """Run a batch whose previous base2 read 10/20 ms; each base boot reads the next of *bases*."""
    ran: list[str] = []
    readings = iter(bases)

    def _run_arm(spec):
        ran.append(spec.out_dir.name)
        return _arm(*next(readings)) if spec.out_dir.name.startswith("base") else _arm(9.0, 18.0)

    monkeypatch.setattr(cba, "run_arm", _run_arm)
    spec = _batch(tmp_path)
    summary = cba.run_batch(cba.BatchSpec(**{**spec.__dict__, "reference": {1: 10.0, 2: 20.0}}))
    return ran, summary


def test_a_base_that_agrees_with_the_previous_base2_runs_once(tmp_path, monkeypatch):
    ran, summary = _referenced_batch(tmp_path, monkeypatch, [(10.05, 20.1), (10.0, 20.0)])

    assert ran == ["base", "lever-a", "base2"]
    assert (summary["valid"], summary["parity_broken"]) == (True, False)


def test_a_base_off_the_previous_base2_is_rerun_and_the_rerun_grades_the_batch(tmp_path, monkeypatch):
    ran, summary = _referenced_batch(tmp_path, monkeypatch, [(10.4, 20.8), (10.0, 20.0), (10.0, 20.0)])

    assert ran == ["base", "base-rerun", "lever-a", "base2"]
    assert (summary["valid"], summary["parity_broken"]) == (True, False)
    assert summary["verdicts"]["lever-a"]["gain_pct"] == 10.0


def test_a_base_off_the_previous_base2_twice_refuses_the_batch_before_its_candidates(tmp_path, monkeypatch):
    ran, summary = _referenced_batch(tmp_path, monkeypatch, [(10.4, 20.8), (10.3, 20.6)])

    assert ran == ["base", "base-rerun"]
    assert (summary["valid"], summary["parity_broken"], summary["verdicts"]) == (False, True, {})
    assert summary["reason"] == "the switched-off tree runs -3.00% against the previous batch's base2"


def _referenced_state(tmp_path: Path, reference: dict | None, stack: list) -> SimpleNamespace:
    decode = "[t TP0] Decode batch, #running-req: 1, #token: 1, accept len: 3.60, gen throughput (token/s): 300.00, #queue-req: 0"
    state = _workspace(tmp_path, decode, {"effective_argv": ["x"]})
    state.cycle_reference = reference
    state.optimization_stack = stack
    return state


def test_a_session_batch_takes_the_previous_base2_of_the_same_stack_as_its_reference(tmp_path):
    state = _referenced_state(tmp_path, {"stack_len": 1, "cycle_ms": {"1": 10.0, "4": 30.0}}, [{"kind": "config"}])

    spec = cba.session_batch(state, [], out_dir=tmp_path / "o", port=1)

    assert spec.reference == {1: 10.0, 4: 30.0}


@pytest.mark.parametrize("reference", [None, {}, {"stack_len": 0, "cycle_ms": {"1": 10.0}}])
def test_a_session_batch_has_no_reference_from_another_stack_or_none_at_all(tmp_path, reference):
    state = _referenced_state(tmp_path, reference, [{"kind": "config"}])

    assert cba.session_batch(state, [], out_dir=tmp_path / "o", port=1).reference is None


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

    spec = cba.session_batch(state, [("lever-a", {"HL_A": "1"})], out_dir=tmp_path / "out", port=9100)

    assert (spec.argv, spec.env) == (launch["effective_argv"], {"X": "1"})
    assert (spec.weights, spec.prompt_tokens, spec.vocab_size) == ({1: 0.75, 2: 0.25, 4: 0.0}, 108000, 154880)
    assert spec.candidates == [("lever-a", {"HL_A": "1"})]


@pytest.mark.asyncio
async def test_an_explore_task_carrying_cycle_arms_runs_the_batch_instead_of_a_replay_round(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import explore

    seen: dict[str, object] = {}

    def _session_batch(state, candidates, *, out_dir, port):
        seen.update(candidates=candidates, state=state)
        return "spec"

    monkeypatch.setattr(cba, "session_batch", _session_batch)
    monkeypatch.setattr(cba, "run_batch", lambda spec: {"valid": True, "verdicts": {"lever-a": {"verdict": "pass"}}})
    monkeypatch.setattr(explore, "_ExploreRound", None)
    state = SimpleNamespace()
    ctx = SimpleNamespace(
        task=SimpleNamespace(
            task_id="t-cycle",
            params={cba.CYCLE_ARMS_PARAM: [{"name": "lever-a", "envs": {"HL_A": "1"}}]},
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
    assert seen == {"candidates": [("lever-a", {"HL_A": "1"})], "state": state}


@pytest.mark.asyncio
async def test_a_cycle_batch_without_a_baseline_launch_fails_named(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import explore

    def _missing(*_a, **_k):
        raise FileNotFoundError("no agentx_server_launch.json under the baseline workspace /nowhere")

    monkeypatch.setattr(cba, "session_batch", _missing)
    ctx = SimpleNamespace(
        task=SimpleNamespace(task_id="t-cycle", params={cba.CYCLE_ARMS_PARAM: [{"name": "a", "envs": {}}]}),
        extra={"shared_state": SimpleNamespace(), "workspace": str(tmp_path)},
    )

    result = await explore.ExploreExecutor(session_dir=tmp_path)(ctx)

    assert (result["status"], result["error_class"]) == ("failed", "cycle_bench_unavailable")


def test_a_batch_with_no_weighted_batch_size_grades_nothing():
    arms = {"base": _arm(10.0, 20.0), "c": _arm(5.0, 10.0), "base2": _arm(12.0, 24.0)}

    summary = cba.summarize_batch(arms, ["c"], {}, bands=_BANDS)

    assert (summary["valid"], summary["verdicts"]) == (False, {})


def _workspace(tmp_path: Path, log: str, launch: dict) -> SimpleNamespace:
    workspace = tmp_path / "baseline" / "benchmark_sglang_1"
    workspace.mkdir(parents=True)
    (workspace / "agentx_server_launch.json").write_text(json.dumps(launch), encoding="utf-8")
    (workspace / "server.log").write_text(log, encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"vocab_size": 100}), encoding="utf-8")
    return SimpleNamespace(
        baseline_workspace=str(tmp_path / "baseline"), model_path=str(model), agentx_corpus_shape={"isl": {"p50": 10}}
    )


def test_a_session_batch_refuses_a_baseline_log_with_no_decode_line(tmp_path):
    state = _workspace(tmp_path, "[2026] Decode batch, unparseable", {"effective_argv": ["x"]})

    with pytest.raises(ValueError, match="no decode line"):
        cba.session_batch(state, [], out_dir=tmp_path / "o", port=1)


@pytest.mark.parametrize("state", [None, SimpleNamespace(baseline_workspace="", model_path="/m")])
def test_a_session_batch_refuses_a_session_without_a_baseline_workspace(tmp_path, monkeypatch, state):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "stray").mkdir()
    (tmp_path / "stray" / "agentx_server_launch.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="no baseline workspace"):
        cba.session_batch(state, [], out_dir=tmp_path / "o", port=1)


def test_a_session_batch_runs_the_current_best_stack_once_a_round_moved_it(tmp_path):
    decode = "[t TP0] Decode batch, #running-req: {n}, #token: 1, accept len: 3.60, gen throughput (token/s): 300.00, #queue-req: 0"
    state = _workspace(tmp_path, decode.format(n=1), {"effective_argv": ["base"], "effective_env": {"HL_A": "0"}})
    best = tmp_path / "runs" / "explore" / "v00" / "benchmark_sglang_2"
    best.mkdir(parents=True)
    launch = {"effective_argv": ["kept"], "effective_env": {"HL_A": "1"}}
    (best / "agentx_server_launch.json").write_text(json.dumps(launch), encoding="utf-8")
    (best / "server.log").write_text(decode.format(n=4), encoding="utf-8")
    state.current_best = {"workspace": str(best)}

    spec = cba.session_batch(state, [], out_dir=tmp_path / "o", port=1)

    assert (spec.argv, spec.env, spec.weights) == (["kept"], {"HL_A": "1"}, {1: 0.0, 2: 0.0, 4: 1.0})


def test_a_session_batch_refuses_a_current_best_without_its_launch_rather_than_bench_the_baseline(tmp_path):
    decode = "[t TP0] Decode batch, #running-req: 1, #token: 1, accept len: 3.60, gen throughput (token/s): 300.00, #queue-req: 0"
    state = _workspace(tmp_path, decode, {"effective_argv": ["base"]})
    best = tmp_path / "runs" / "integrate" / "benchmark_sglang_3"
    best.mkdir(parents=True)
    state.current_best = {"workspace": str(best)}

    with pytest.raises(FileNotFoundError, match="no agentx_server_launch.json under the current best"):
        cba.session_batch(state, [], out_dir=tmp_path / "o", port=1)


def test_a_session_batch_runs_the_launchs_effective_env_over_its_runtime_env(tmp_path):
    decode = "[t TP0] Decode batch, #running-req: 1, #token: 1, accept len: 3.60, gen throughput (token/s): 300.00, #queue-req: 0"
    launch = {
        "effective_argv": ["x"],
        "runtime_environment": {"A": "1", "B": "1"},
        "effective_env": {"B": "2", "C": "3"},
    }
    state = _workspace(tmp_path, decode, launch)

    spec = cba.session_batch(state, [], out_dir=tmp_path / "o", port=1)

    assert spec.env == {"A": "1", "B": "2", "C": "3"}
