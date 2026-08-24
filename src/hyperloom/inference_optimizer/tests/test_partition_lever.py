# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The compute-partition lever, end to end around one scriptable benchmark.

``amd-smi`` is faked so these run anywhere, but the benchmark script is a real
bash process: the point of most of these cases is what the script does or does
not see in its environment, and whether it ran at all.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer import cli
from hyperloom.orchestrator.actions.executors import _partition_lever as pl
from hyperloom.orchestrator.actions.executors import bypass_scriptable as bs
from hyperloom.orchestrator.actions.executors._latency_budget import (
    REASON_OVER_BUDGET,
    REASON_UNMEASURED,
    latency_keep_block,
    resolve_latency_budget_ms,
)


class _FakeSmi:
    """An ``amd-smi`` whose partition state the test drives."""

    def __init__(self, mode: str = "SPX", *, settable: bool = True):
        self.modes = {i: mode for i in range(8)}
        self.settable = settable
        self.history: list[str] = []

    def __call__(self, cmd, **kwargs):
        if cmd[:2] == ["amd-smi", "set"]:
            gpu_id = int(cmd[cmd.index("-g") + 1])
            mode = cmd[cmd.index("--compute-partition") + 1]
            if not self.settable:
                return subprocess.CompletedProcess(cmd, 1, "", "permission denied")
            self.modes[gpu_id] = mode
            self.history.append(mode)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        rows = [
            {"gpu_id": gid, "memory": "NPS1", "accelerator_type": m}
            for gid, m in sorted(self.modes.items())
        ]
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"current_partition": rows}), "")


def _scripts_dir(tmp_path: Path) -> Path:
    """A benchmark entrypoint that dumps its partition env and records that it ran."""
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "custom_mi355x.sh").write_text(
        "#!/bin/bash\n"
        'echo "ran" > "$RESULT_DIR/ran.marker"\n'
        'env | grep -E "^HYPERLOOM_PARTITION" | sort > "$RESULT_DIR/partition.env" || true\n',
        encoding="utf-8",
    )
    return scripts


def _run(tmp_path: Path, monkeypatch, envs: dict | None = None):
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(_scripts_dir(tmp_path)))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    rc, error = bs.run_scriptable(
        framework="custom",
        runner_type="mi355x",
        inferencex_root=str(tmp_path / "InferenceX"),
        bench={"model": "/models/hwmirror", "envs": dict(envs or {})},
        workspace=workspace,
        timeout_s=60.0,
    )
    partition_env = workspace / "partition.env"
    seen = dict(
        line.split("=", 1)
        for line in (partition_env.read_text(encoding="utf-8").splitlines() if partition_env.is_file() else [])
        if "=" in line
    )
    return rc, error, (workspace / "ran.marker").is_file(), seen


@pytest.fixture(autouse=True)
def _clean_lever_env(monkeypatch):
    """No ambient lever: these cases each state their own."""
    for name in (
        pl.PARTITION_MODE_ENV,
        pl.PARTITION_MODES_ENV,
        pl.STREAMS_PER_PARTITION_ENV,
        pl.PARTITION_GPU_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def test_lever_off_leaves_the_run_untouched(tmp_path, monkeypatch):
    fake = _FakeSmi()
    monkeypatch.setattr(subprocess, "run", fake)
    rc, error, ran, seen = _run(tmp_path, monkeypatch)
    assert (rc, error, ran) == (0, None, True)
    # Nothing published, and above all nothing set: a session that did not ask
    # for partitioning must not touch a shared card.
    assert seen == {}
    assert fake.history == []


def test_a_requested_mode_is_established_published_and_restored(tmp_path, monkeypatch):
    fake = _FakeSmi()
    monkeypatch.setattr(subprocess, "run", fake)
    rc, error, ran, seen = _run(tmp_path, monkeypatch, {pl.PARTITION_MODE_ENV: "cpx"})
    assert (rc, error, ran) == (0, None, True)
    assert seen[pl.RUNTIME_MODE_ENV] == "CPX"
    assert seen[pl.RUNTIME_COUNT_ENV] == "8"
    assert seen[pl.RUNTIME_CU_ENV] == "32"
    # Two per partition by default, so sixteen streams in total.
    assert seen[pl.RUNTIME_STREAMS_ENV] == "2"
    assert seen[pl.RUNTIME_TOTAL_STREAMS_ENV] == "16"
    # Held for the run, then handed back in the mode it was found in.
    assert fake.history == ["CPX", "SPX"]
    assert fake.modes[0] == "SPX"


def test_streams_per_partition_flows_through_to_the_benchmark(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _FakeSmi())
    monkeypatch.setenv(pl.STREAMS_PER_PARTITION_ENV, "3")
    _, _, _, seen = _run(tmp_path, monkeypatch, {pl.PARTITION_MODE_ENV: "DPX"})
    assert seen[pl.RUNTIME_STREAMS_ENV] == "3"
    assert seen[pl.RUNTIME_TOTAL_STREAMS_ENV] == "6"


def test_an_unsettable_mode_aborts_before_the_benchmark_runs(tmp_path, monkeypatch):
    # No privilege to repartition. The alternative to failing here is measuring
    # the topology that happens to be present and labelling it CPX.
    monkeypatch.setattr(subprocess, "run", _FakeSmi(settable=False))
    rc, error, ran, _ = _run(tmp_path, monkeypatch, {pl.PARTITION_MODE_ENV: "CPX"})
    assert rc == 2
    assert "compute-partition lever" in error
    assert not ran


def test_an_unknown_mode_aborts_before_the_benchmark_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _FakeSmi())
    rc, error, ran, _ = _run(tmp_path, monkeypatch, {pl.PARTITION_MODE_ENV: "NPS2"})
    assert rc == 2
    assert "unknown compute-partition mode" in error
    assert not ran


def test_a_variant_mode_overrides_the_ambient_one(monkeypatch):
    monkeypatch.setenv(pl.PARTITION_MODE_ENV, "SPX")
    assert pl.requested_mode({pl.PARTITION_MODE_ENV: "cpx"}) == "CPX"
    assert pl.requested_mode({}) == "SPX"


def test_partition_plan_is_empty_when_no_mode_is_requested():
    assert pl.plan_partition_run({}, gpu_type="mi355x") == ("", {})


def test_latency_budget_is_off_by_default():
    # The gate must not change behaviour for the sessions that never set it.
    assert latency_keep_block(1211.0, budget_ms=0.0) == (False, "")
    assert resolve_latency_budget_ms({}, None) == 0.0


def test_latency_budget_blocks_the_partition_trade_it_exists_for():
    # CPX-16 measured 13.07 fwd/s at 1211 ms against SPX-2's 10.90 at 183 ms:
    # a +20% throughput gain the old gate would have kept.
    blocked, reason = latency_keep_block(1211.0, budget_ms=300.0)
    assert blocked and REASON_OVER_BUDGET in reason and "4.04x" in reason
    assert latency_keep_block(183.0, budget_ms=300.0) == (False, "")


def test_latency_budget_fails_closed_on_an_unmeasured_candidate():
    blocked, reason = latency_keep_block(None, budget_ms=300.0)
    assert blocked and REASON_UNMEASURED in reason
    # Nonsense readings are treated as no reading, not as a pass.
    assert latency_keep_block(0.0, budget_ms=300.0)[0]
    assert latency_keep_block(float("nan"), budget_ms=300.0)[0]


def test_latency_budget_precedence_is_most_specific_first(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_MAX_LATENCY_MS", "250")

    class _State:
        latency_budget_ms = 400.0

    assert resolve_latency_budget_ms({"latency_budget_ms": 500}, _State()) == 500.0
    assert resolve_latency_budget_ms({}, _State()) == 400.0
    assert resolve_latency_budget_ms({}, None) == 250.0


class TestResumeContract:
    """The lever must survive a resume, because it is part of how a number was got.

    A resume that quietly dropped the mode or the streams count would compare
    candidates measured under partitioning against a baseline that was not, and
    the comparison would look ordinary. These lock the precedence chain (CLI
    flag > exported env > archived state) and the write-back that keeps the
    manifest agreeing with what the executors were handed.
    """

    @staticmethod
    def _args(**over):
        ns = argparse.Namespace(
            compute_partition_modes=None, streams_per_partition=None, max_latency_ms=None
        )
        for key, value in over.items():
            setattr(ns, key, value)
        return ns

    @staticmethod
    def _state(modes=(), streams=2, budget=0.0):
        state = SimpleNamespace(
            compute_partition_modes=list(modes),
            streams_per_partition=streams,
            latency_budget_ms=budget,
        )
        return state

    def test_archive_supplies_the_lever_when_the_resume_omits_it(self, monkeypatch):
        monkeypatch.delenv(pl.PARTITION_MODES_ENV, raising=False)
        monkeypatch.delenv(pl.STREAMS_PER_PARTITION_ENV, raising=False)
        monkeypatch.delenv("HYPERLOOM_MAX_LATENCY_MS", raising=False)
        args = self._args()
        cli._restore_partition_lever_from_state(args, self._state(("DPX", "CPX"), 3, 400.0))
        assert args.compute_partition_modes == "DPX,CPX"
        assert args.streams_per_partition == 3
        assert args.max_latency_ms == 400.0

    def test_an_exported_env_outranks_the_archive(self, monkeypatch):
        # Someone who exports the budget for this resume means it, and the
        # archived value is older information.
        monkeypatch.setenv(pl.PARTITION_MODES_ENV, "QPX")
        monkeypatch.setenv("HYPERLOOM_MAX_LATENCY_MS", "250")
        args = self._args()
        cli._restore_partition_lever_from_state(args, self._state(("DPX",), 2, 400.0))
        assert args.compute_partition_modes == "QPX"
        assert args.max_latency_ms == 250.0

    def test_a_re_passed_flag_outranks_both(self, monkeypatch):
        monkeypatch.setenv(pl.PARTITION_MODES_ENV, "QPX")
        monkeypatch.setenv("HYPERLOOM_MAX_LATENCY_MS", "250")
        args = self._args(compute_partition_modes="cpx", max_latency_ms=99.0)
        cli._restore_partition_lever_from_state(args, self._state(("DPX",), 2, 400.0))
        assert args.compute_partition_modes == "cpx"
        assert args.max_latency_ms == 99.0

    def test_streams_two_is_distinguishable_from_streams_unset(self, monkeypatch):
        # The reason --streams-per-partition defaults to None: with a default of
        # 2 a resume cannot tell "not passed" from "passed 2", and would
        # overwrite a persisted 4 every time.
        monkeypatch.delenv(pl.STREAMS_PER_PARTITION_ENV, raising=False)
        unset = self._args()
        cli._restore_partition_lever_from_state(unset, self._state(("DPX",), 4, 0.0))
        assert unset.streams_per_partition == 4

        explicit = self._args(streams_per_partition=2)
        cli._restore_partition_lever_from_state(explicit, self._state(("DPX",), 4, 0.0))
        assert explicit.streams_per_partition == 2

    def test_persist_writes_the_live_contract_back_onto_state(self, monkeypatch):
        monkeypatch.setenv(pl.PARTITION_MODES_ENV, "DPX,CPX")
        monkeypatch.setenv(pl.STREAMS_PER_PARTITION_ENV, "3")
        monkeypatch.setenv("HYPERLOOM_MAX_LATENCY_MS", "275.5")
        state = self._state()
        cli._persist_partition_lever(state)
        assert state.compute_partition_modes == ["DPX", "CPX"]
        assert state.streams_per_partition == 3
        assert state.latency_budget_ms == 275.5

    def test_persist_records_the_lever_being_off(self, monkeypatch):
        monkeypatch.delenv(pl.PARTITION_MODES_ENV, raising=False)
        monkeypatch.delenv(pl.STREAMS_PER_PARTITION_ENV, raising=False)
        monkeypatch.delenv("HYPERLOOM_MAX_LATENCY_MS", raising=False)
        state = self._state(("DPX",), 3, 400.0)
        cli._persist_partition_lever(state)
        assert state.compute_partition_modes == []
        assert state.latency_budget_ms == 0.0


def test_streams_per_partition_parses_to_none_when_not_passed():
    """Locks the parser half of the resume contract.

    ``_restore_partition_lever_from_state`` can only tell "not passed" from
    "passed 2" if the flag's default stays None. Giving it a default of 2 --
    which reads as harmless, since 2 is the documented default -- would make
    every resume overwrite a persisted streams count with 2 and quietly change
    the experiment.
    """
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    args = _build_parser().parse_args(["optimize", "--model", "/tmp/m"])
    assert args.streams_per_partition is None
    assert args.compute_partition_modes is None
    assert args.max_latency_ms is None

    passed = _build_parser().parse_args(
        ["optimize", "--model", "/tmp/m", "--streams-per-partition", "2"]
    )
    assert passed.streams_per_partition == 2


def test_read_session_lever_tolerates_a_malformed_budget(monkeypatch):
    # The env is machine-written, but a hand-edited resume script is not, and a
    # crash in the seed path would lose the session rather than the value.
    monkeypatch.setenv("HYPERLOOM_MAX_LATENCY_MS", "not-a-number")
    monkeypatch.setenv(pl.PARTITION_MODES_ENV, " DPX , , CPX ")
    modes, streams, budget = pl.read_session_lever()
    assert modes == ("DPX", "CPX")
    assert budget == 0.0
    assert streams >= 1
