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
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common import gpu_partition as gp
from hyperloom.inference_optimizer import cli
from hyperloom.orchestrator.actions.executors import _partition_lever as pl
from hyperloom.orchestrator.actions.executors import bypass_scriptable as bs
from hyperloom.orchestrator.actions.executors import explore
from hyperloom.orchestrator.actions.executors._latency_budget import (
    LATENCY_BUDGET_ENV,
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
        rows = [{"gpu_id": gid, "memory": "NPS1", "accelerator_type": m} for gid, m in sorted(self.modes.items())]
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


#: Every variable the lever reads or publishes. Restored around each case
#: because ``_export_partition_lever`` writes ``os.environ`` directly, which
#: ``monkeypatch`` cannot undo on its behalf -- and a budget or a mode list left
#: behind here silently changes the KEEP gate for every test that runs after.
_LEVER_ENV = (
    pl.PARTITION_MODE_ENV,
    pl.PARTITION_MODES_ENV,
    pl.STREAMS_PER_PARTITION_ENV,
    pl.PARTITION_GPU_ENV,
    LATENCY_BUDGET_ENV,
)


@pytest.fixture(autouse=True)
def _clean_lever_env():
    """No ambient lever: these cases each state their own, and leak none."""
    saved = {name: os.environ.get(name) for name in _LEVER_ENV}
    for name in _LEVER_ENV:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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
        ns = argparse.Namespace(compute_partition_modes=None, streams_per_partition=None, max_latency_ms=None)
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

    passed = _build_parser().parse_args(["optimize", "--model", "/tmp/m", "--streams-per-partition", "2"])
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


class TestModeAxis:
    """Turning the session's mode list into variants the search actually runs.

    Declaring the modes is not the same as trying them. These cover the
    expansion, the order it has to keep, and the one framework combination that
    must not produce variants at all.
    """

    @staticmethod
    def _state(modes):
        return SimpleNamespace(compute_partition_modes=list(modes))

    def test_the_session_list_becomes_one_variant_per_mode(self):
        grid = pl.partition_lever_grid({"compute_partition_modes": "spx,dpx,cpx"}, None, framework="custom")
        assert [v["name"] for v in grid] == [
            "partition_spx",
            "partition_dpx",
            "partition_cpx",
        ]
        # Env-only, and carrying nothing but the selector: a mode variant that
        # also moved a server flag would confound the two.
        assert [v["extra_envs"] for v in grid] == [
            {pl.PARTITION_MODE_ENV: "SPX"},
            {pl.PARTITION_MODE_ENV: "DPX"},
            {pl.PARTITION_MODE_ENV: "CPX"},
        ]
        assert {v["extra_args"] for v in grid} == {""}
        assert {v["provenance"] for v in grid} == {"partition_lever"}

    def test_the_operators_order_is_preserved(self):
        # The list is a search order, not a set: the stack advances mode by mode,
        # so reordering changes which mode each later one has to beat.
        grid = pl.partition_lever_grid({"compute_partition_modes": "cpx,spx"}, None, framework="xdit")
        assert [v["name"] for v in grid] == ["partition_cpx", "partition_spx"]

    def test_no_modes_seeds_nothing(self):
        assert pl.partition_lever_grid({}, None, framework="custom") == []
        assert pl.partition_lever_grid(None, self._state([]), framework="custom") == []

    def test_a_serving_framework_gets_no_variants(self):
        """The mislabelling guard, at the grid rather than the card.

        Only the scriptable runner calls ``plan_partition_run``. On a serving
        framework the env would ride along, no partition would be established,
        and the result would be filed under the requested mode -- a number that
        is wrong in a way nothing downstream can detect.
        """
        for framework in ("sglang", "vllm", ""):
            assert pl.partition_lever_grid({"compute_partition_modes": "dpx"}, None, framework=framework) == []

    def test_an_unusable_mode_list_seeds_nothing_rather_than_raising(self, monkeypatch):
        # Grid assembly is not the place to end a session; the CLI already
        # refused this at launch, so reaching here means the env was edited.
        monkeypatch.setenv(pl.PARTITION_MODES_ENV, "spx,nope")
        assert pl.partition_lever_grid(None, None, framework="custom") == []

    def test_mode_precedence_is_most_specific_first(self, monkeypatch):
        monkeypatch.setenv(pl.PARTITION_MODES_ENV, "cpx")
        state = self._state(["QPX"])
        assert pl.resolve_session_modes({"compute_partition_modes": "dpx"}, state) == ("DPX",)
        assert pl.resolve_session_modes({}, state) == ("QPX",)
        assert pl.resolve_session_modes({}, None) == ("CPX",)


class TestModeAxisOrdering:
    """The mode has to be decided before the knobs that are tuned against it.

    Explore stacks a KEEP'd variant's envs onto every variant after it, so
    putting the mode axis first is what makes the rest of the grid get
    re-explored inside the winning partition. Landing it at the end instead
    would measure every knob against the old topology and then change the
    topology afterwards.
    """

    def test_seeded_variants_land_in_front(self):
        grid, fresh = explore._prepend_fresh_variants(
            [{"name": "llm_flag_a"}, {"name": "llm_flag_b"}],
            [{"name": "partition_dpx"}],
        )
        assert [v["name"] for v in grid] == [
            "partition_dpx",
            "llm_flag_a",
            "llm_flag_b",
        ]
        assert [v["name"] for v in fresh] == ["partition_dpx"]

    def test_a_name_the_grid_already_uses_is_left_alone(self):
        # An operator or specialist who named the variant said something more
        # specific than the generated default; the generated one steps aside.
        pinned = {"name": "partition_dpx", "extra_envs": {"CUSTOM": "1"}}
        grid, fresh = explore._prepend_fresh_variants([pinned], [{"name": "partition_dpx", "extra_envs": {}}])
        assert grid == [pinned]
        assert fresh == []

    def test_nothing_to_add_leaves_the_grid_untouched(self):
        original = [{"name": "llm_flag_a"}]
        grid, fresh = explore._prepend_fresh_variants(original, [])
        assert grid == original
        assert fresh == []


class TestHbmCapacity:
    """Reading how much memory a partition would actually get.

    ``layout_for`` cannot size a partition's memory without the card's total,
    and the total is not tabled because boards sharing an ISA do not share a
    capacity. These cover the parse and the one state where the reading means
    something other than what it says.
    """

    @staticmethod
    def _smi(monkeypatch, *, mode: str = "SPX", vram: dict | None = None):
        def fake_run(cmd, **kwargs):
            if "--vram" in cmd:
                payload = {"gpu_data": [{"gpu": 0, "vram": {"size": vram} if vram else {}}]}
            else:
                payload = {"current_partition": [{"gpu_id": 0, "accelerator_type": mode}]}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

        monkeypatch.setattr(gp.subprocess, "run", fake_run)

    def test_mebibytes_labelled_mb_are_read_as_mebibytes(self, monkeypatch):
        # An MI355X with 288 GiB reports 294896 and calls it "MB". Taking that
        # decimally would understate the card by 7% and prune modes that fit.
        self._smi(monkeypatch, vram={"value": 294896, "unit": "MB"})
        assert gp.read_hbm_gib(0) == pytest.approx(288.0, abs=0.1)

    def test_a_partitioned_card_reports_unknown(self, monkeypatch):
        """The reading is per device, and under a split mode a device is a partition.

        Nothing in the payload says whether the figure is the card or a slice of
        it, and dividing an already-divided number by the partition count again
        would understate capacity eightfold -- pruning every mode that fits.
        """
        self._smi(monkeypatch, mode="CPX", vram={"value": 36864, "unit": "MB"})
        assert gp.read_hbm_gib(0) is None

    def test_an_unparseable_size_is_unknown_rather_than_zero(self, monkeypatch):
        # Zero would read as "nothing fits" and silently empty the mode list.
        self._smi(monkeypatch, vram={"value": "N/A", "unit": "MB"})
        assert gp.read_hbm_gib(0) is None
        self._smi(monkeypatch, vram={"value": 288, "unit": "furlongs"})
        assert gp.read_hbm_gib(0) is None

    def test_no_amd_smi_is_unknown(self, monkeypatch):
        monkeypatch.setattr(gp.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
        assert gp.read_hbm_gib(0) is None


class TestFootprintPruning:
    """Refusing a mode whose partitions cannot hold what would run on them.

    A partition gets its share of HBM while every stream on it keeps a full copy
    of the weights, so the narrow modes exhaust memory first. Measuring that
    costs a run per mode and returns an OOM instead of a number.
    """

    def test_a_mode_too_small_for_its_streams_is_dropped(self):
        # 288 GiB card: CPX gives 36 GiB/partition, and two 20.7 GiB streams
        # need 41.4 GiB. QPX's 72 GiB holds them.
        kept, reasons = pl.prune_infeasible_modes(
            ("SPX", "DPX", "QPX", "CPX"),
            gpu_type="mi355x",
            hbm_gib=288.0,
            footprint_gib=20.7,
            streams=2,
        )
        assert kept == ("SPX", "DPX", "QPX")
        assert len(reasons) == 1
        # The arithmetic travels with the verdict; "CPX dropped" alone does not
        # tell an operator whether to lower streams or pick a bigger mode.
        assert "CPX" in reasons[0] and "41.4" in reasons[0] and "36.0" in reasons[0]

    def test_streams_per_partition_decides_it(self):
        """One stream fitting says nothing about two.

        This is the case that makes gating on the single-stream figure worse than
        not gating: the configuration is declared feasible and dies at the second
        worker, after the first has already been measured.
        """
        args = dict(gpu_type="mi355x", hbm_gib=288.0, footprint_gib=20.7)
        assert pl.prune_infeasible_modes(("CPX",), streams=1, **args)[0] == ("CPX",)
        assert pl.prune_infeasible_modes(("CPX",), streams=2, **args)[0] == ()

    def test_an_unknown_capacity_drops_nothing(self):
        kept, reasons = pl.prune_infeasible_modes(
            ("SPX", "CPX"), gpu_type="mi355x", hbm_gib=None, footprint_gib=20.7, streams=2
        )
        assert kept == ("SPX", "CPX")
        assert reasons == ()

    def test_an_unknown_footprint_drops_nothing(self):
        # Wrongly dropping a mode costs the optimization the configuration that
        # would have won, and leaves no trace that it was ever a candidate.
        kept, _ = pl.prune_infeasible_modes(
            ("SPX", "CPX"), gpu_type="mi355x", hbm_gib=288.0, footprint_gib=0.0, streams=2
        )
        assert kept == ("SPX", "CPX")

    def test_an_unsizeable_board_keeps_the_mode(self):
        # Refusing here would report an unknown board as a memory verdict; the
        # apply path raises with the real reason attached.
        kept, reasons = pl.prune_infeasible_modes(
            ("CPX",), gpu_type="nvidia-h100", hbm_gib=288.0, footprint_gib=999.0, streams=2
        )
        assert kept == ("CPX",)
        assert reasons == ()

    def test_the_grid_drops_the_infeasible_mode(self, monkeypatch):
        monkeypatch.setenv(pl.STREAMS_PER_PARTITION_ENV, "2")
        grid = pl.partition_lever_grid(
            {
                "compute_partition_modes": "spx,cpx",
                "gpu_type": "mi355x",
                "peak_gib_per_stream": 20.7,
            },
            None,
            framework="custom",
            hbm_gib=288.0,
        )
        assert [v["name"] for v in grid] == ["partition_spx"]

    def test_a_model_that_fits_nowhere_seeds_nothing(self, monkeypatch):
        monkeypatch.setenv(pl.STREAMS_PER_PARTITION_ENV, "2")
        grid = pl.partition_lever_grid(
            {
                "compute_partition_modes": "dpx,cpx",
                "gpu_type": "mi355x",
                "peak_gib_per_stream": 200.0,
            },
            None,
            framework="custom",
            hbm_gib=288.0,
        )
        assert grid == []


class TestFootprintSource:
    """Which number the partitions get sized against, and how sure it is."""

    def test_a_measured_peak_wins(self):
        state = SimpleNamespace(current_best={"peak_gib_per_stream": 20.7})
        assert pl.per_stream_footprint_gib(None, state) == (20.7, "measured")

    def test_params_override_the_baseline(self):
        state = SimpleNamespace(current_best={"peak_gib_per_stream": 20.7})
        assert pl.per_stream_footprint_gib({"peak_gib_per_stream": 30.0}, state) == (30.0, "measured")

    def test_weights_are_the_fallback_and_a_lower_bound(self, tmp_path):
        """Read from the checkpoint, so it needs no run to be known.

        A lower bound is the right thing to prune on: the true footprint adds
        activations and never subtracts weights, so "does not fit by the weights
        alone" is a proof while "fits" is no evidence at all.
        """
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"num_hidden_layers": 4, "hidden_size": 512, "torch_dtype": "float16"}),
            encoding="utf-8",
        )
        (model / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 150 * 1024**3}}), encoding="utf-8"
        )
        gib, source = pl.per_stream_footprint_gib(None, SimpleNamespace(model_path=str(model)))
        assert source == "weights"
        assert gib == pytest.approx(150.0)

    def test_no_measurement_and_no_readable_model_is_unknown(self, tmp_path):
        assert pl.per_stream_footprint_gib(None, SimpleNamespace(model_path=str(tmp_path / "gone"))) == (0.0, "")
        assert pl.per_stream_footprint_gib(None, None) == (0.0, "")

    def test_a_large_checkpoint_rules_out_the_narrow_modes(self, tmp_path, monkeypatch):
        """The case the weights bound is actually for.

        Two streams of a 60 GiB checkpoint need 120 GiB per partition, which
        QPX's 72 GiB cannot hold whatever the activations do -- and that is
        knowable from the checkpoint before the first run.
        """
        monkeypatch.setenv(pl.STREAMS_PER_PARTITION_ENV, "2")
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"torch_dtype": "float16"}), encoding="utf-8")
        (model / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 60 * 1024**3}}), encoding="utf-8"
        )
        grid = pl.partition_lever_grid(
            {"compute_partition_modes": "spx,dpx,qpx,cpx", "gpu_type": "mi355x"},
            SimpleNamespace(model_path=str(model)),
            framework="custom",
            hbm_gib=288.0,
        )
        assert [v["name"] for v in grid] == ["partition_spx", "partition_dpx"]


def test_launch_refuses_the_lever_on_a_serving_framework(capsys):
    """Fail at launch, not by quietly seeding an empty axis.

    An operator who passed the flag expects modes to be tried. Dropping them at
    grid time with only a log line would look like the lever ran and found
    nothing worth keeping.
    """
    with pytest.raises(SystemExit) as excinfo:
        cli._export_partition_lever(
            modes_raw="dpx",
            streams_per_partition=2,
            max_latency_ms=400.0,
            framework="sglang",
        )
    assert excinfo.value.code == 2
    assert "scriptable framework" in capsys.readouterr().err


def test_launch_refuses_the_lever_on_a_multi_node_session(capsys):
    """One card, no cluster coordination -- and the report is silent there.

    Left to run it would mutate one node's GPU and file the result as a property
    of the whole topology, with the report saying nothing about the privileged
    change because it skips multi-node sessions.
    """
    with pytest.raises(SystemExit) as excinfo:
        cli._export_partition_lever(
            modes_raw="dpx",
            streams_per_partition=2,
            max_latency_ms=400.0,
            framework="custom",
            nodes=2,
        )
    assert excinfo.value.code == 2
    assert "one card" in capsys.readouterr().err


def test_launch_validates_the_card_the_session_will_actually_mutate(monkeypatch):
    """Not card 0, when the session manages another.

    Validating a different board than the apply path touches would defeat the
    reason this check happens at launch instead of at the mode change.
    """
    monkeypatch.setenv(pl.PARTITION_GPU_ENV, "3")
    asked: list[int] = []
    monkeypatch.setattr(gp, "supported_modes", lambda gpu_id: (asked.append(gpu_id), ("SPX", "DPX"))[1])
    monkeypatch.setattr(gp, "unsupported_modes", lambda modes, gpu_id=0: (asked.append(gpu_id), ())[1])
    monkeypatch.setattr(gp, "partition_count_conflicts", lambda gpu_id=0: (asked.append(gpu_id), ())[1])

    cli._export_partition_lever(
        modes_raw="dpx",
        streams_per_partition=2,
        max_latency_ms=400.0,
        framework="custom",
    )
    assert asked == [3, 3, 3]


class TestUnpartitionedRunOnASplitCard:
    """A failed restore must not turn later baselines into silent lies.

    ``partitioned`` logs a restore failure rather than raising, so it cannot mask
    the exception that caused the exit. The cost is that the card can be left
    split, and the runs requesting *no* mode are the ones with nothing to notice.
    """

    def test_a_split_card_refuses_a_run_that_asked_for_no_mode(self, tmp_path, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _FakeSmi("QPX"))
        monkeypatch.setenv(pl.PARTITION_MODES_ENV, "spx,dpx")
        rc, error, ran, _ = _run(tmp_path, monkeypatch)
        # Refused before the benchmark started: the number would have been filed
        # as the unpartitioned baseline.
        assert (rc, ran) == (2, False)
        assert "QPX" in str(error) and "no partition mode" in str(error)

    def test_an_unpartitioned_card_runs_as_before(self, tmp_path, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _FakeSmi("SPX"))
        monkeypatch.setenv(pl.PARTITION_MODES_ENV, "spx,dpx")
        rc, error, ran, seen = _run(tmp_path, monkeypatch)
        assert (rc, error, ran) == (0, None, True)
        assert seen == {}

    def test_the_check_is_silent_when_the_lever_is_off(self, tmp_path, monkeypatch):
        # A split card with no lever is the operator's own arrangement, and
        # nothing here has touched the hardware.
        monkeypatch.setattr(subprocess, "run", _FakeSmi("CPX"))
        rc, error, ran, _ = _run(tmp_path, monkeypatch)
        assert (rc, error, ran) == (0, None, True)
