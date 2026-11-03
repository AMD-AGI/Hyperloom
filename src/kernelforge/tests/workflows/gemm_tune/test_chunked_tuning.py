# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Chunked tuning: the deadline and the measured cost of each chunk decide how many shapes are tuned, and a chunk
killed at the deadline loses only its own shapes."""

from __future__ import annotations

import math
import os
import tempfile
import time
import types
from pathlib import Path

import pytest

import kernelforge.workflows.gemm_tune.tuners._aiter_dense_common as ac
import kernelforge.workflows.gemm_tune.tuners._dense_args as dense_args
from kernelforge.tests.workflows.gemm_tune._sglang_dense_bf16_run import _csv, _ctx, _prep
from kernelforge.workflows.gemm_tune.dense_shapes import compute_decode_m_values
from kernelforge.workflows.gemm_tune.tuners import _chunks
from kernelforge.workflows.gemm_tune.tuners import sglang_dense_bf16 as sd
from kernelforge.workflows.gemm_tune.tuners.base import TuneContext

#: A shape's cost on the 2026-10-26 image (5943 candidates), against the old image's fixed 93 s.
SHAPE_S = 110.0
#: The outer cap the failing runs had.
TIMEOUT_S = 1_799


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _budget(clock: _Clock, *, cards: int = 1, timeout_s: float = TIMEOUT_S) -> _chunks.ChunkBudget:
    return _chunks.ChunkBudget(clock.now + timeout_s, cards, 93, 120, clock)


class TestPacer:
    def test_first_chunk_is_one_shape_per_card(self):
        assert _chunks.ChunkPacer(_budget(_Clock(), cards=4)).size() == 4

    def test_next_chunk_is_sized_by_the_measured_wave(self):
        clock = _Clock()
        pacer = _chunks.ChunkPacer(_budget(clock, cards=2, timeout_s=1_000))
        clock.now = 200.0
        assert pacer.record(2, 200.0) == 200.0
        # (1000 - 200 - 120) // 200 = 3 waves of 2 cards.
        assert pacer.room() == pacer.size() == 6

    def test_a_chunk_never_holds_more_than_the_wave_cap(self):
        clock = _Clock()
        pacer = _chunks.ChunkPacer(_budget(clock, cards=1, timeout_s=100_000))
        pacer.record(1, 10.0)
        assert pacer.size() == _chunks.MAX_WAVES_PER_CHUNK < pacer.room()

    def test_no_room_for_a_wave_is_zero(self):
        clock = _Clock()
        pacer = _chunks.ChunkPacer(_budget(clock, timeout_s=300))
        clock.now = 100.0
        pacer.record(1, 100.0)
        assert pacer.room() == pacer.size() == 0

    def test_a_chunk_is_killed_before_the_reserve_is_spent(self):
        clock = _Clock()
        pacer = _chunks.ChunkPacer(_budget(clock, timeout_s=1_000))
        assert pacer.timeout_s() == 1_000 - 60


class TestUnits:
    def test_a_unit_too_large_is_passed_over_not_a_stop(self):
        pending = [["a", "b", "c"], ["d"], ["e"]]
        assert _chunks.take_units(pending, 2, 2, at_least_one=False) == [["d"], ["e"]]
        assert pending == [["a", "b", "c"]]

    def test_a_unit_larger_than_the_chunk_size_goes_alone_when_time_allows(self):
        # One card caps a chunk at four shapes, and a dispatch group's band alone holds five.
        band = [["m1", "m16", "m32", "m48", "m64", "m2048"], ["m4096"]]
        assert _chunks.take_units(band, 8, 4, at_least_one=False) == [["m1", "m16", "m32", "m48", "m64", "m2048"]]
        assert _chunks.take_units(band, 8, 4, at_least_one=False) == [["m4096"]]

    def test_every_band_group_is_reached_on_one_card(self):
        clock, ran = _Clock(), []

        def run(_index, rows, _timeout_s):
            clock.now += 10.0 * len(rows)
            ran.append(rows)
            return True

        units = [[f"group{g}-row{r}" for r in range(6)] for g in range(3)]
        tally = _chunks.run_in_chunks(units, _budget(clock, cards=1, timeout_s=100_000), run)
        assert tally.shapes_left == 0
        assert ran == [list(unit) for unit in units]

    def test_the_first_chunk_takes_a_unit_larger_than_its_room(self):
        pending = [["a", "b", "c"]]
        assert _chunks.take_units(pending, 1, 1, at_least_one=True) == [["a", "b", "c"]]
        assert _chunks.take_units([["a", "b", "c"]], 1, 1, at_least_one=False) == []

    def test_a_group_takes_its_whole_decode_band_with_its_first_row(self):
        band = compute_decode_m_values(64)
        rows = ["2048,4096,4096", "4096,4096,4096", "2048,5120,4096"]
        rows += [f"{m},4096,4096" for m in band] + [f"{m},5120,4096" for m in band]
        units = _chunks.decode_band_units(rows, "M,N,K", band)
        assert units == [
            [*(f"{m},4096,4096" for m in band), "2048,4096,4096"],
            ["4096,4096,4096"],
            [*(f"{m},5120,4096" for m in band), "2048,5120,4096"],
        ]


def _deadline_run(clock: _Clock, timeout_s: int, cards: int, writes):
    """A tuner process on ``clock``: a chunk costs SHAPE_S a wave of ``cards``, and one that cannot finish within its
    kill timeout is killed there having written nothing; ``writes`` writes a finished chunk's rows."""

    def run(cmd, **kwargs):
        shapes = Path(cmd[cmd.index("-i") + 1]).read_text(encoding="utf-8").splitlines()[1:]
        cost = math.ceil(len(shapes) / cards) * SHAPE_S
        if cost > kwargs["timeout_s"]:
            clock.now += kwargs["timeout_s"]
            return 124, "", "TimeoutExpired"
        clock.now += cost
        writes(cmd, shapes, kwargs)
        return 1, "", ""

    return run


class TestSglangDenseBf16:
    """18 shapes at 110 s against a 1799 s cap: one process for all of them was killed and kept nothing."""

    M_VALUES = [16 * (i + 1) for i in range(18)]

    def _run(self, tmp_path, monkeypatch, cards):
        _prep(tmp_path, monkeypatch, tuned_rows=[])
        monkeypatch.setattr(sd, "_compute_m_values", lambda conc, thorough=False: list(self.M_VALUES))
        clock = _Clock()
        monkeypatch.setattr(sd, "time", types.SimpleNamespace(monotonic=clock), raising=False)

        def writes(cmd, shapes, _kwargs):
            rows = [
                f"gfx950,256,{s.split(',')[0]},4096,4096,False,torch.bfloat16,torch.bfloat16,False,False,"
                f"hipblaslt,1,0,9.0,knl,0.0,700.0,3000.0"
                for s in shapes
            ]
            Path(cmd[cmd.index("-o") + 1]).write_text(_csv(rows), encoding="utf-8")
            Path(cmd[cmd.index("-o2") + 1]).write_text(_csv(rows), encoding="utf-8")

        monkeypatch.setattr(sd, "run_subprocess", _deadline_run(clock, TIMEOUT_S, cards, writes))
        result = sd.SglangDenseBf16Tuner(_ctx(tmp_path, timeout_s=TIMEOUT_S, mp=cards)).run()
        return result, clock

    def test_one_card_keeps_every_chunk_that_finished(self, tmp_path, monkeypatch):
        result, clock = self._run(tmp_path, monkeypatch, cards=1)
        # 1 + 4 + 4 + 4 + 2 shapes end at 1650 s; no third of a wave was left for another.
        assert result.status == "partial_output"
        assert result.total_shapes == 15 and result.expected_shapes == 18
        assert clock.now <= TIMEOUT_S - sd._TIMEOUT_RESERVE_S

    def test_four_cards_tune_all_of_them(self, tmp_path, monkeypatch):
        result, clock = self._run(tmp_path, monkeypatch, cards=4)
        assert result.status == "ok"
        assert result.total_shapes == 18
        assert clock.now == 5 * SHAPE_S

    def test_a_killed_chunk_loses_only_its_own_shapes(self, tmp_path, monkeypatch):
        _prep(tmp_path, monkeypatch, tuned_rows=[])
        monkeypatch.setattr(sd, "_compute_m_values", lambda conc, thorough=False: [1, 512, 2048, 8192])
        clock = _Clock()
        monkeypatch.setattr(sd, "time", types.SimpleNamespace(monotonic=clock), raising=False)
        calls = []

        def run(cmd, **kwargs):
            calls.append(kwargs["timeout_s"])
            if len(calls) > 1:
                # The second chunk's shapes are far slower than the first measured: killed at its timeout.
                clock.now += kwargs["timeout_s"]
                return 124, "", "TimeoutExpired"
            clock.now += SHAPE_S
            m = Path(cmd[cmd.index("-i") + 1]).read_text(encoding="utf-8").splitlines()[1].split(",")[0]
            row = f"gfx950,256,{m},4096,4096,False,torch.bfloat16,torch.bfloat16,False,False,hipblaslt,1,0,9.0,knl,0.0,700.0,3000.0"
            Path(cmd[cmd.index("-o") + 1]).write_text(_csv([row]), encoding="utf-8")
            Path(cmd[cmd.index("-o2") + 1]).write_text(_csv([row]), encoding="utf-8")
            return 1, "", ""

        monkeypatch.setattr(sd, "run_subprocess", run)
        result = sd.SglangDenseBf16Tuner(_ctx(tmp_path, timeout_s=TIMEOUT_S, mp=1)).run()

        assert len(calls) == 2
        assert result.status == "partial_output" and result.total_shapes == 1
        assert result.candidate is True


_DENSE_HDR = "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio"


def _dense_ctx(tmp_path, cards) -> TuneContext:
    return TuneContext(
        profile=types.SimpleNamespace(),
        framework="sglang",
        precision="fp8",
        quant_type="blockscale",
        gpu_type="mi355x",
        tp=1,
        conc=64,
        tokens=[64],
        mp=cards,
        output_dir=tmp_path,
        iters=1,
        warmup=0,
        min_improvement_pct=3.0,
        timeout_s=TIMEOUT_S,
        untuned_csv=tmp_path / "in.csv",
    )


class TestAiterDense:
    """The a8w8 family's single process had the same fate: killed at the cap, rc 124, a failed run."""

    @pytest.fixture
    def tuner(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FORGE_SPLITK_TRIAL", "0")
        root = tmp_path / "tmp"
        root.mkdir()
        monkeypatch.setenv("TMPDIR", str(root))
        monkeypatch.setattr(tempfile, "tempdir", None)
        # Prefill shapes, one dispatch group each: no decode band to add.
        rows = [f"8192,{4096 + 128 * i},5120" for i in range(18)]
        (tmp_path / "in.csv").write_text("M,N,K\n" + "\n".join(rows) + "\n")
        monkeypatch.setattr(ac, "find_tuner_script", lambda k: tmp_path / "script.py")
        monkeypatch.setattr(ac, "_resolve_input_csv", lambda ctx, wd, needs_q_dtype_w=False: tmp_path / "in.csv")
        monkeypatch.setattr(ac, "resolve_aiter_root", lambda: str(tmp_path))
        monkeypatch.setattr(ac._tr, "is_isolation_enabled", lambda: False)
        monkeypatch.setattr(ac._tr, "with_task_timeout", lambda cmd: cmd)
        monkeypatch.setattr(
            dense_args, "probe_script", lambda script: types.SimpleNamespace(supports=lambda flag: True)
        )
        clock = _Clock()
        monkeypatch.setattr(ac, "time", types.SimpleNamespace(time=clock), raising=False)

        def writes(cmd, shapes, kwargs):
            # aiter's compare candidate, under the temp root it was pinned to.
            out = Path(cmd[cmd.index("-o") + 1])
            compare = Path(kwargs["env_override"]["TMPDIR"]) / "aiter_compare"
            compare.mkdir(exist_ok=True)
            rows = [f"gfx950,256,{s},ck,8,0,200.0,knl,100,1000,0.0" for s in shapes]
            candidate = compare / f"{out.stem}.4242.candidate.csv"
            candidate.write_text(_DENSE_HDR + "\n" + "\n".join(rows) + "\n")
            # The kernel's file clock lags time.time() by up to a tick; a real chunk runs for minutes.
            later = time.time() + 5
            os.utime(candidate, (later, later))
            Path(cmd[cmd.index("-o2") + 1]).write_text(_DENSE_HDR + "\n" + "\n".join(rows) + "\n")

        def run(cards, fault=lambda index: None):
            calls = []
            chunk = _rc_zero(_deadline_run(clock, TIMEOUT_S, cards, writes))

            def faulting(cmd, **kwargs):
                calls.append(cmd)
                failure = fault(len(calls) - 1)
                if failure is not None:
                    clock.now += SHAPE_S
                    return failure
                return chunk(cmd, **kwargs)

            monkeypatch.setattr(ac, "run_subprocess", faulting)
            spec = ac.DenseTunerSpec("a8w8_blockscale", "a8w8_blockscale", "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE")
            return ac.run_aiter_dense_tuner(spec, _dense_ctx(tmp_path, cards), tmp_path)

        return run

    def test_one_card_keeps_every_chunk_that_finished(self, tuner):
        result = tuner(1)
        assert result.status == "partial_output"
        assert result.total_shapes == 15 and result.expected_shapes == 18
        assert "3 shape(s) not reached before the deadline" in result.error
        assert len(Path(result.artifact_path).read_text().splitlines()) == 1 + 15

    def test_a_failing_chunk_neither_stops_the_rest_nor_hides(self, tuner):
        # aiter exits 1 when a shape in its input has no valid candidate, and a GPU fault kills the process; either
        # costs that chunk's shapes only.
        result = tuner(1, fault=lambda index: (1, "", "HIP error: an illegal memory access") if index == 1 else None)
        assert result.status == "partial_output"
        assert result.expected_shapes == 18 and 1 < result.total_shapes < 18
        assert "exited 1" in result.error and "illegal memory access" in result.error

    def test_four_cards_tune_all_of_them(self, tuner):
        result = tuner(4)
        assert result.status == "ok" and result.error == ""
        assert result.total_shapes == result.expected_shapes == 18

    def test_every_chunk_failing_is_a_failed_run(self, tuner):
        result = tuner(4, fault=lambda index: (1, "", "HIP error: an illegal memory access"))
        assert result.status == "failed" and result.error_class == "subprocess_error"


def _rc_zero(run):
    """aiter's dense tuners exit 0 when they finish."""

    def wrapped(cmd, **kwargs):
        rc, out, err = run(cmd, **kwargs)
        return (0 if rc == 1 else rc), out, err

    return wrapped
