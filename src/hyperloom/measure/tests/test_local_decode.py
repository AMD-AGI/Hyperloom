# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The decode pass on the local backend: after the measured round, on a server booted for it, never a guessed output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from hyperloom.measure import local
from hyperloom.measure.decode import OUTPUTS_PATH, parse_outputs, plan_pass
from hyperloom.measure.local import LocalMeasurementBackend, TestCallRefused
from hyperloom.measure.socket import TestCall
from hyperloom.measure.tests.decode_fixtures import expected_tokens, fake_server
from hyperloom.measure.tests.test_local_backend import _Grid, _call, grid, magpie_backend  # noqa: F401  (fixtures)

ISL, OSL, CONC = 12, 1024, 3


def _model(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"vocab_size": 5000}')
    return model


def _point(tmp_path: Path) -> Path:
    envs = {"TP": 1, "CONC": CONC, "ISL": ISL, "OSL": OSL, "RUN_EVAL": "true"}
    path = tmp_path / "decode-point.yaml"
    config = {"benchmark": {"framework": "sglang", "model": str(_model(tmp_path)), "envs": envs}}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class _Teardowns:
    """The lifecycle helpers: the server of the point listens on ``port`` and every teardown is recorded."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, port: int) -> None:
        self.count = 0
        monkeypatch.setattr(
            local,
            "resolve_lifecycle_params",
            lambda path: {"eligible": True, "port": port, "framework": "sglang", "reason": ""},
        )
        monkeypatch.setattr(local, "teardown_lifecycle_server", lambda **kw: setattr(self, "count", self.count + 1))


def _decode_call(tmp_path: Path, **kwargs: Any) -> TestCall:
    return _call(tmp_path, _point(tmp_path), accuracy=True, decode_pass=True, **kwargs)


def _run(call: TestCall):
    return LocalMeasurementBackend().run_test(call)


def _boot_only(grid: _Grid) -> None:
    grid.answers["decode_server"] = grid.result("decode_server", note="server_lifecycle_boot_only")


def _warnings(outcome) -> list[str]:
    return [w for r in outcome.rounds for w in r.nonfatal_warnings]


def test_the_pass_runs_after_the_measured_round_on_a_booted_server_and_leaves_its_outputs_in_the_bundle(
    grid, monkeypatch, tmp_path
):
    _boot_only(grid)
    with fake_server() as server:
        teardowns = _Teardowns(monkeypatch, server.port)
        outcome = _run(_decode_call(tmp_path))

    assert [c["grid"][0].name for c in grid.calls] == ["test", "decode_server"]
    measured, boot = grid.calls
    assert boot["lifecycle_boot_only"] is True and boot["server_lifecycle"]["cleanup"] is False
    assert boot["base_extra_envs"] == {"RUN_EVAL": "false", "NUM_PROMPTS": "1"}
    assert "base_extra_envs" not in measured, "the measured round is the one that evaluates, as the call asked"
    assert grid.configs[0]["benchmark"]["envs"]["RUN_EVAL"] == "true"
    plan = plan_pass(ISL, OSL, 5000)
    record = parse_outputs((tmp_path / "out" / OUTPUTS_PATH).read_text())
    assert record.identity == plan.identity
    assert record.outputs == tuple(tuple(expected_tokens(p, 256)) for p in plan.prompt_ids())
    assert len(record.outputs) == 32 and server.peak <= CONC
    assert teardowns.count == 1, "the decode server is reclaimed"
    assert outcome.returncode == 0 and _warnings(outcome) == []


def test_a_test_that_does_not_ask_for_the_pass_boots_nothing_more(grid, monkeypatch, tmp_path):
    _Teardowns(monkeypatch, 1)
    outcome = _run(_call(tmp_path, _point(tmp_path), accuracy=True))

    assert [c["grid"][0].name for c in grid.calls] == ["test"]
    assert not (tmp_path / "out" / "decode").exists() and outcome.returncode == 0


def test_a_measured_round_that_failed_is_followed_by_no_pass(grid, monkeypatch, tmp_path):
    grid.answers["test"] = grid.result("test", status="failed", returncode=1, error="server died")
    _Teardowns(monkeypatch, 1)

    outcome = _run(_decode_call(tmp_path))

    assert [c["grid"][0].name for c in grid.calls] == ["test"] and outcome.returncode == 1


@pytest.mark.parametrize(("mode", "reason"), [("error", "http_500"), ("no_ids", "no_token_ids_in_response")])
def test_a_pass_that_fails_leaves_the_measurement_as_it_is_and_a_warning_and_no_output(
    grid, monkeypatch, tmp_path, mode, reason
):
    _boot_only(grid)
    with fake_server(mode) as server:
        teardowns = _Teardowns(monkeypatch, server.port)
        outcome = _run(_decode_call(tmp_path))

    assert outcome.returncode == 0 and outcome.failure_class is None
    assert [r.phase for r in outcome.rounds] == ["measure", "accuracy"]
    assert _warnings(outcome) == [f"decode_pass_failed:{reason}"]
    assert not (tmp_path / "out" / "decode").exists(), "no output is invented"
    assert teardowns.count == 1


def test_a_decode_server_that_does_not_boot_is_a_warning_not_a_failed_test(grid, monkeypatch, tmp_path):
    grid.answers["decode_server"] = grid.result("decode_server", status="failed", returncode=1, error="died")
    teardowns = _Teardowns(monkeypatch, 1)

    outcome = _run(_decode_call(tmp_path))

    assert outcome.returncode == 0 and _warnings(outcome) == ["decode_pass_failed:server_did_not_boot"]
    assert teardowns.count == 1


def test_a_point_that_does_not_say_its_lengths_is_a_warning_and_boots_nothing(grid, monkeypatch, tmp_path):
    _Teardowns(monkeypatch, 1)
    point = _point(tmp_path)
    config = yaml.safe_load(point.read_text())
    del config["benchmark"]["envs"]["ISL"]
    point.write_text(yaml.safe_dump(config))

    outcome = _run(_call(tmp_path, point, accuracy=True, decode_pass=True))

    assert [c["grid"][0].name for c in grid.calls] == ["test"]
    assert _warnings(outcome) == ["decode_pass_failed:point_has_no_isl_osl_or_vocab"]


@pytest.mark.parametrize("kwargs", [{"kind": "profile"}, {"prep": "conc_sweep"}])
def test_only_a_single_point_test_or_accuracy_test_may_ask_for_the_pass(grid, tmp_path, kwargs):
    with pytest.raises(TestCallRefused, match="single-point test or accuracy test"):
        _run(_call(tmp_path, _point(tmp_path), decode_pass=True, **kwargs))
    assert grid.calls == []
