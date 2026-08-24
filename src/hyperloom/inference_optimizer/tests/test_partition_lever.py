# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The compute-partition lever, end to end around one scriptable benchmark.

``amd-smi`` is faked so these run anywhere, but the benchmark script is a real
bash process: the point of most of these cases is what the script does or does
not see in its environment, and whether it ran at all.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

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
