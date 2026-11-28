# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The optional GPU-specialist rebench helper: the rounds it runs and the JSON result it prints.

The contract: a rebench measures the way the integrate gate does. Where its point warms up it tests the point on one
server on the port it was given and reports -- a discarded bench that boots the server and never evaluates, then the
measured bench with the eval the point asks for -- and stops that server once; a discarded bench that failed is the
rebench's failure and nothing is measured; a point that does not warm up is one round that owns its server on that
port; an MLPerf point warms up too, in two harness runs without an eval; on a multi-node host it is the cluster's
restart, one discarded pass and the measured pass, both against the cluster's own address.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.common.ports import DEV_SANDBOX_PORTS
from hyperloom.orchestrator.actions.executors import _multi_node_env
from hyperloom.orchestrator.actions.executors._suite import rounds as suite_rounds
from hyperloom.orchestrator.actions.executors._suite.config import MEASURED_SEED, WARMUP_SEED
from hyperloom.orchestrator.actions.executors._suite.rounds import RUN_DIR_PREFIX
from hyperloom.orchestrator.actions.executors.grid import _grid_runner
from hyperloom.orchestrator.actions.executors.grid._grid_base import VariantResult
from hyperloom.orchestrator.actions.executors.tests.test_grid_suite_round import _COLD, _HOT, suite, unheld  # noqa: F401
from hyperloom.orchestrator.actions.executors.tests.test_suite_config import _mlperf_point
from hyperloom.orchestrator.rehearsal import READY, LaunchAttempt, LaunchScenario, ScriptedRounds, VirtualClock
from hyperloom.orchestrator.specialists import rebench as sr
from inference_suite.contracts import FixedWorkload, LifecycleMode, WorkloadKind
from inference_suite.tests.test_compose import _launch

_PORT = 30123
_POINT_PORT = 8123
_SCORE = {"exact_match,strict-match": 0.9}


def _variant_result(**overrides) -> VariantResult:
    """A real ``VariantResult``, which is what ``run_grid`` returns."""
    fields = {
        "name": "rebench",
        "extra_server_args": "",
        "extra_envs": {},
        "status": "succeeded",
        "output_throughput": 1234.5,
        "ttft_mean_ms": 10.0,
        "tpot_mean_ms": 2.0,
        "error": "",
        "nonfatal_warnings": ["w1"],
    }
    fields.update(overrides)
    return VariantResult(**fields)


def test_resolve_port_auto_and_explicit() -> None:
    ephemeral_start = int(Path("/proc/sys/net/ipv4/ip_local_port_range").read_text(encoding="utf-8").split()[0])
    for auto in (None, 0):
        port = sr._resolve_port(auto)
        assert port < ephemeral_start and port not in DEV_SANDBOX_PORTS
    assert sr._resolve_port(12345) == 12345
    assert sr._resolve_port(8888) == 8888


def test_current_leased_cards_precedence(monkeypatch) -> None:
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    assert sr._current_leased_cards() == ""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5")
    assert sr._current_leased_cards() == "4,5"
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "6,7")
    assert sr._current_leased_cards() == "6,7"


def test_parse_env_pairs() -> None:
    assert sr._parse_env_pairs(None) == {}
    assert sr._parse_env_pairs(["A=1", "B=x=y", "malformed", "=nope", "  C =3"]) == {"A": "1", "B": "x=y", "C": "3"}


@pytest.mark.asyncio
async def test_run_specialist_rebench_success(tmp_path, monkeypatch) -> None:
    fake = _variant_result(workspace=str(tmp_path / "ws"))
    seen: dict = {}

    async def _fake_run_grid(**kwargs):
        seen.update(kwargs)
        return [fake]

    monkeypatch.setattr(sr, "default_baseline_config", lambda: str(tmp_path / "base.yaml"))
    monkeypatch.setattr(sr, "prepare_grid_benchmark_config", lambda *a, **k: tmp_path / "mat.yaml")
    monkeypatch.setattr(sr, "run_grid", _fake_run_grid)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")

    res = await sr.run_specialist_rebench(
        output_dir=str(tmp_path / "out"),
        benchmark=sr.RebenchBenchmark(config_path=None),
        server=sr.RebenchServer(base_extra_args="--kv-cache-dtype fp8_e4m3", extra_envs={"E": "1"}, port=12321),
    )
    assert res["ok"] is True
    assert res["status"] == "succeeded"
    assert res["output_throughput"] == 1234.5
    assert res["port"] == 12321
    assert res["gpu_ids"] == "4,5,6,7"
    assert "w1" in res["warnings"]
    assert seen["recipe"].base.extra_server_args == "--kv-cache-dtype fp8_e4m3"
    assert seen["grid"][0].extra_server_args == ""


@pytest.mark.asyncio
async def test_run_specialist_rebench_failed_status(tmp_path, monkeypatch) -> None:
    fake = _variant_result(
        status="failed",
        output_throughput=None,
        ttft_mean_ms=None,
        tpot_mean_ms=None,
        workspace="",
        error="boom",
        nonfatal_warnings=[],
    )

    async def _fake_run_grid(**kwargs):
        return [fake]

    monkeypatch.setattr(sr, "prepare_grid_benchmark_config", lambda *a, **k: tmp_path / "mat.yaml")
    monkeypatch.setattr(sr, "run_grid", _fake_run_grid)

    res = await sr.run_specialist_rebench(
        output_dir=str(tmp_path / "o"), benchmark=sr.RebenchBenchmark(config_path=str(tmp_path / "c.yaml"))
    )
    assert res["ok"] is False
    assert any("rebench_failed" in w for w in res["warnings"])


@pytest.mark.asyncio
async def test_run_specialist_rebench_no_result(tmp_path, monkeypatch) -> None:
    async def _fake_run_grid(**kwargs):
        return []

    monkeypatch.setattr(sr, "prepare_grid_benchmark_config", lambda *a, **k: tmp_path / "m.yaml")
    monkeypatch.setattr(sr, "run_grid", _fake_run_grid)

    res = await sr.run_specialist_rebench(
        output_dir=str(tmp_path / "o"), benchmark=sr.RebenchBenchmark(config_path=str(tmp_path / "c.yaml"))
    )
    assert res["ok"] is False
    assert "no result" in res["error"]


@pytest.mark.asyncio
async def test_run_specialist_rebench_exception_surfaced(tmp_path, monkeypatch) -> None:
    async def _boom(**kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(sr, "prepare_grid_benchmark_config", lambda *a, **k: tmp_path / "m.yaml")
    monkeypatch.setattr(sr, "run_grid", _boom)

    res = await sr.run_specialist_rebench(
        output_dir=str(tmp_path / "o"), benchmark=sr.RebenchBenchmark(config_path=str(tmp_path / "c.yaml"))
    )
    assert res["ok"] is False
    assert "kaboom" in res["error"]


def test_main_success_prints_json(tmp_path, monkeypatch, capsys) -> None:
    async def _fake(**kwargs):
        return {"ok": True, "output_throughput": 5.0}

    monkeypatch.setattr(sr, "run_specialist_rebench", _fake)
    rc = sr.main(["--output", str(tmp_path / "o")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True


def test_main_failure_return_code(tmp_path, monkeypatch, capsys) -> None:
    async def _fake(**kwargs):
        return {"ok": False, "error": "x"}

    monkeypatch.setattr(sr, "run_specialist_rebench", _fake)
    rc = sr.main(["--output", str(tmp_path / "o"), "--env", "A=1", "--extra-args=--foo bar"])
    assert rc == 1


@pytest.fixture
def point(suite: SimpleNamespace) -> Path:
    """The suite's point, naming a port of its own: a rebench runs on the port it was given, not on that one."""
    document = yaml.safe_load(suite.point.read_text(encoding="utf-8"))
    document["benchmark"]["envs"]["PORT"] = _POINT_PORT
    suite.point.write_text(yaml.safe_dump(document), encoding="utf-8")
    return suite.point


def _rebench(monkeypatch, point: Path, tmp_path: Path, *attempts: LaunchAttempt, **envs: str):
    """Rebench ``point`` on ``_PORT`` over played rounds; its result, each round's config and the variant's slot."""
    rounds = ScriptedRounds(scenario=LaunchScenario(attempts=attempts), clock=VirtualClock(), launch=_launch())
    monkeypatch.setattr(suite_rounds, "run_round", rounds.run_round)
    result = asyncio.run(
        sr.run_specialist_rebench(
            output_dir=tmp_path / "rebench",
            benchmark=sr.RebenchBenchmark(config_path=point),
            server=sr.RebenchServer(extra_envs=envs, port=_PORT),
        )
    )
    return result, [call.config for call in rounds.calls], tmp_path / "rebench" / "variant_00_specialist-rebench"


def _round_shape(config) -> tuple:
    return (config.lifecycle.mode, config.lifecycle.pid_dir, config.fixed.seed, config.eval.enabled)


@pytest.mark.parametrize("run_eval", ["true", "false"])
def test_a_rebench_discards_a_bench_then_measures_and_evaluates_as_its_point_asks_on_one_server_on_its_port(
    monkeypatch,
    suite,
    point,
    tmp_path,
    run_eval,
):
    measured = LaunchAttempt(outcome=READY, metrics=_HOT, eval_scores=_SCORE)

    result, (warmup, bench), slot = _rebench(
        monkeypatch, point, tmp_path, LaunchAttempt(outcome=READY, metrics=_COLD), measured, RUN_EVAL=run_eval
    )

    assert _round_shape(warmup) == (LifecycleMode.BOOT, str(slot), WARMUP_SEED, False)
    assert _round_shape(bench) == (LifecycleMode.REUSE, str(slot), MEASURED_SEED, run_eval == "true")
    assert warmup.server.port == bench.server.port == result["port"] == _PORT
    assert Path(warmup.paths.run_dir).parent == slot / "warmup_round"
    assert suite.stopped == [slot]
    assert result["ok"] is True and result["status"] == "succeeded", result
    assert result["output_throughput"] == pytest.approx(_HOT["output_throughput"])
    assert result["workspace"] == str(slot / f"{RUN_DIR_PREFIX}0")
    assert result["warnings"] == [
        "run_grid_warmup_discarded_first",
        f"warmup_round_tput:{_COLD['output_throughput']:.1f}",
    ]
    assert {"ttft_ms", "itl_ms", "e2el_mean_ms", "gpu_ids", "error"} <= set(result)


def test_a_rebench_whose_discarded_bench_failed_reports_the_failure_and_measures_nothing(
    monkeypatch,
    suite,
    point,
    tmp_path,
):
    failed = LaunchAttempt(outcome=READY, metrics=_HOT, returncode=1)

    result, (warmup,), slot = _rebench(monkeypatch, point, tmp_path, failed, RUN_EVAL="false")

    assert warmup.lifecycle.mode is LifecycleMode.BOOT
    assert result["ok"] is False and result["status"] == "failed", result
    assert result["output_throughput"] is None
    assert result["workspace"] == str(slot / "warmup_round" / f"{RUN_DIR_PREFIX}0")
    assert "run_grid_warmup_round_failed" in result["warnings"]
    assert any(warning.startswith("rebench_failed:") for warning in result["warnings"])
    assert suite.stopped == [slot]


def test_a_rebench_of_a_point_that_does_not_warm_up_is_one_round_that_owns_its_server_on_its_port(
    monkeypatch,
    suite,
    unheld,
    point,
    tmp_path,
):
    result, (bench,), slot = _rebench(
        monkeypatch, point, tmp_path, LaunchAttempt(outcome=READY, metrics=_HOT), RUN_EVAL="false"
    )

    assert _round_shape(bench) == (LifecycleMode.OWN, None, MEASURED_SEED, False)
    assert bench.server.port == result["port"] == _PORT
    assert result["ok"] is True, result
    assert result["output_throughput"] == pytest.approx(_HOT["output_throughput"])
    assert result["warnings"] == []
    assert suite.stopped == [slot]


def test_a_rebench_of_an_mlperf_point_runs_the_gates_pair_without_an_eval(monkeypatch, suite, tmp_path):
    """An MLPerf point warms up, as at the integrate gate; its harness takes no seed and scores accuracy inline."""
    suite.point.write_text(yaml.safe_dump(_mlperf_point()), encoding="utf-8")
    rounds = ScriptedRounds(
        scenario=LaunchScenario(
            attempts=(LaunchAttempt(outcome=READY, metrics=_COLD), LaunchAttempt(outcome=READY, metrics=_HOT))
        ),
        clock=VirtualClock(),
        launch=_launch(),
    )
    composed = []

    def as_a_fixed_load(config, *, cancel=None):  # ScriptedRounds plays no MLPerf harness
        composed.append((config.workload, config.lifecycle.mode, config.lifecycle.pid_dir, config.eval.enabled))
        load = FixedWorkload(8, 256, 256, 1.0, 64, 2, MEASURED_SEED, WARMUP_SEED, "auto", False)
        return rounds.run_round(replace(config, workload=WorkloadKind.FIXED, fixed=load, mlperf=None), cancel=cancel)

    monkeypatch.setattr(suite_rounds, "run_round", as_a_fixed_load)

    result = asyncio.run(
        sr.run_specialist_rebench(
            output_dir=tmp_path / "rebench",
            benchmark=sr.RebenchBenchmark(config_path=suite.point),
            server=sr.RebenchServer(extra_envs={"RUN_EVAL": "false"}, port=_PORT),
        )
    )

    slot = tmp_path / "rebench" / "variant_00_specialist-rebench"
    assert composed == [
        (WorkloadKind.MLPERF, LifecycleMode.BOOT, str(slot), False),
        (WorkloadKind.MLPERF, LifecycleMode.REUSE, str(slot), False),
    ]
    assert result["ok"] is True, result
    assert result["output_throughput"] == pytest.approx(_HOT["output_throughput"])
    assert suite.stopped == [slot]


def test_a_multi_node_rebench_restarts_the_cluster_and_measures_it_at_its_own_address_after_one_discarded_pass(
    monkeypatch,
    suite,
    point,
    tmp_path,
):
    monkeypatch.setattr(suite_rounds, "client_only_server_url", lambda: "http://10.0.0.2:30000")
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: True)
    monkeypatch.setattr(_multi_node_env, "mn_bench_warmup_enabled", lambda: True)
    restarts: list[str] = []

    async def restart(run, at, rounds):
        restarts.append(at.variant.name)

    monkeypatch.setattr(_grid_runner, "_restart_multi_node_server", restart)

    result, (warm, bench), slot = _rebench(
        monkeypatch,
        point,
        tmp_path,
        LaunchAttempt(outcome=READY, metrics=_COLD, eval_scores=_SCORE),
        LaunchAttempt(outcome=READY, metrics=_HOT, eval_scores=_SCORE),
        RUN_EVAL="true",
    )

    assert restarts == ["specialist-rebench"]
    # Both passes run the rebench's own point, eval included, as a multi-node variant of the gate does.
    assert _round_shape(warm) == (LifecycleMode.EXTERNAL, None, WARMUP_SEED, True)
    assert _round_shape(bench) == (LifecycleMode.EXTERNAL, None, MEASURED_SEED, True)
    assert {(config.server.host, config.server.port) for config in (warm, bench)} == {("10.0.0.2", 30000)}
    assert Path(warm.paths.run_dir).parent == slot / "mn_warmup"
    assert result["ok"] is True and result["port"] == _PORT, result
    assert result["output_throughput"] == pytest.approx(_HOT["output_throughput"])
    assert result["warnings"] == []
    assert suite.stopped == [slot]
