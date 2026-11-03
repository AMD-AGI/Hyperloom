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

#: A shape's cost on the 2026-10-26 image (5943 candidates), against the old image's fixed 93 s.
SHAPE_S = 110.0
#: The outer cap the failing runs had.
TIMEOUT_S = 1_799
_DENSE_HDR = "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio"
_HIP_FAULT = (1, "", "HIP error: an illegal memory access")
_KILLED = (124, "", "TimeoutExpired")


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _pacer(clock: _Clock, *, cards: int = 1, timeout_s: float = TIMEOUT_S) -> _chunks.ChunkPacer:
    return _chunks.ChunkPacer(_chunks.ChunkBudget(clock.now + timeout_s, cards, 93, 120, clock))


@pytest.mark.parametrize(
    "cards, timeout_s, now, wave, size, room",
    [
        pytest.param(4, TIMEOUT_S, 0.0, None, 4, 4, id="first-chunk-one-shape-per-card"),
        # (1000 - 200 - 120) // 200 = 3 waves of 2 cards.
        pytest.param(2, 1_000, 200.0, (2, 200.0), 6, 6, id="sized-by-the-measured-wave"),
        pytest.param(1, 100_000, 0.0, (1, 10.0), _chunks.MAX_WAVES_PER_CHUNK, 9_988, id="capped-at-max-waves"),
        pytest.param(1, 300, 100.0, (1, 100.0), 0, 0, id="no-room-for-a-wave"),
    ],
)
def test_pacer_sizes_chunks(cards, timeout_s, now, wave, size, room):
    clock = _Clock()
    pacer = _pacer(clock, cards=cards, timeout_s=timeout_s)
    clock.now = now
    if wave:
        assert pacer.record(*wave) == wave[1]
    assert (pacer.size(), pacer.room()) == (size, room)


def test_a_chunk_is_killed_before_the_reserve_is_spent():
    assert _pacer(_Clock(), timeout_s=1_000).timeout_s() == 1_000 - 60


_BAND = ["m1", "m16", "m32", "m48", "m64", "m2048"]


@pytest.mark.parametrize(
    "pending, room, size, at_least_one, taken, left",
    [
        pytest.param([["a", "b", "c"], ["d"], ["e"]], 2, 2, False, [["d"], ["e"]], [["a", "b", "c"]], id="passed-over"),
        # One card caps a chunk at four shapes, and a dispatch group's band alone holds five.
        pytest.param([_BAND, ["m4096"]], 8, 4, False, [_BAND], [["m4096"]], id="oversize-goes-alone"),
        pytest.param([["m4096"]], 8, 4, False, [["m4096"]], [], id="next-unit-follows"),
        pytest.param([["a", "b", "c"]], 1, 1, True, [["a", "b", "c"]], [], id="first-chunk-takes-oversize"),
        pytest.param([["a", "b", "c"]], 1, 1, False, [], [["a", "b", "c"]], id="later-chunk-waits"),
    ],
)
def test_take_units(pending, room, size, at_least_one, taken, left):
    pending = [list(unit) for unit in pending]
    assert _chunks.take_units(pending, room, size, at_least_one=at_least_one) == taken
    assert pending == left


def test_every_band_group_is_reached_on_one_card():
    clock, ran = _Clock(), []

    def run(_index, rows, _timeout_s):
        clock.now += 10.0 * len(rows)
        ran.append(rows)
        return True

    units = [[f"group{g}-row{r}" for r in range(6)] for g in range(3)]
    budget = _chunks.ChunkBudget(100_000, 1, 93, 120, clock)
    assert _chunks.run_in_chunks(units, budget, run).shapes_left == 0
    assert ran == units


def test_a_group_takes_its_whole_decode_band_with_its_first_row():
    band = compute_decode_m_values(64)
    rows = ["2048,4096,4096", "4096,4096,4096", "2048,5120,4096"]
    rows += [f"{m},4096,4096" for m in band] + [f"{m},5120,4096" for m in band]
    assert _chunks.decode_band_units(rows, "M,N,K", band) == [
        [*(f"{m},4096,4096" for m in band), "2048,4096,4096"],
        ["4096,4096,4096"],
        [*(f"{m},5120,4096" for m in band), "2048,5120,4096"],
    ]


def _arg(cmd, flag) -> Path:
    return Path(cmd[cmd.index(flag) + 1])


def _deadline_run(clock: _Clock, cards: int, writes, *, rc: int = 1, fault=lambda index: None):
    """A tuner process on ``clock``: a chunk costs SHAPE_S a wave of ``cards``; one that cannot finish within its
    kill timeout is killed there having written nothing. ``fault(index)`` overrides a chunk's outcome."""
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        shapes = _arg(cmd, "-i").read_text(encoding="utf-8").splitlines()[1:]
        cost = math.ceil(len(shapes) / cards) * SHAPE_S
        failure = fault(len(calls) - 1) or (_KILLED if cost > kwargs["timeout_s"] else None)
        if failure:
            clock.now += kwargs["timeout_s"] if failure == _KILLED else SHAPE_S
            return failure
        clock.now += cost
        writes(cmd, shapes, kwargs)
        return rc, "", ""

    run.calls = calls
    return run


def _bf16_writes(cmd, shapes, _kwargs):
    rows = [
        f"gfx950,256,{s.split(',')[0]},4096,4096,False,torch.bfloat16,torch.bfloat16,False,False,"
        f"hipblaslt,1,0,9.0,knl,0.0,700.0,3000.0"
        for s in shapes
    ]
    for flag in ("-o", "-o2"):
        _arg(cmd, flag).write_text(_csv(rows), encoding="utf-8")


def _run_bf16(tmp_path, monkeypatch, cards, m_values, **run_kwargs):
    _prep(tmp_path, monkeypatch, tuned_rows=[])
    monkeypatch.setattr(sd, "_compute_m_values", lambda conc, thorough=False: list(m_values))
    clock = _Clock()
    monkeypatch.setattr(sd, "time", types.SimpleNamespace(monotonic=clock), raising=False)
    run = _deadline_run(clock, cards, _bf16_writes, **run_kwargs)
    monkeypatch.setattr(sd, "run_subprocess", run)
    return sd.SglangDenseBf16Tuner(_ctx(tmp_path, timeout_s=TIMEOUT_S, mp=cards)).run(), clock, run.calls


@pytest.mark.parametrize(
    "cards, status, total, end_s",
    [
        # 1 + 4 + 4 + 4 + 2 shapes end at 1650 s; no third of a wave was left for another.
        pytest.param(1, "partial_output", 15, 15 * SHAPE_S, id="one-card-keeps-finished-chunks"),
        pytest.param(4, "ok", 18, 5 * SHAPE_S, id="four-cards-tune-all"),
    ],
)
def test_sglang_bf16_under_the_cap(tmp_path, monkeypatch, cards, status, total, end_s):
    """18 shapes at 110 s against a 1799 s cap: one process for all of them was killed and kept nothing."""
    result, clock, _ = _run_bf16(tmp_path, monkeypatch, cards, [16 * (i + 1) for i in range(18)])
    assert (result.status, result.total_shapes, result.expected_shapes) == (status, total, 18)
    assert clock.now == end_s <= TIMEOUT_S - sd._TIMEOUT_RESERVE_S


def test_sglang_bf16_killed_chunk_loses_only_its_own_shapes(tmp_path, monkeypatch):
    # The second chunk's shapes are far slower than the first measured: killed at its timeout.
    result, _, calls = _run_bf16(tmp_path, monkeypatch, 1, [1, 512, 2048, 8192], fault=lambda i: _KILLED if i else None)
    assert len(calls) == 2
    assert (result.status, result.total_shapes, result.candidate) == ("partial_output", 1, True)


@pytest.fixture
def aiter_dense(tmp_path, monkeypatch):
    """The a8w8 family's single process had the same fate: killed at the cap, rc 124, a failed run."""
    monkeypatch.setenv("FORGE_SPLITK_TRIAL", "0")
    (tmp_path / "tmp").mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setattr(tempfile, "tempdir", None)
    # Prefill shapes, one dispatch group each: no decode band to add.
    rows = [f"8192,{4096 + 128 * i},5120" for i in range(18)]
    (tmp_path / "in.csv").write_text("M,N,K\n" + "\n".join(rows) + "\n")
    monkeypatch.setattr(ac, "find_tuner_script", lambda k: tmp_path / "script.py")
    monkeypatch.setattr(ac, "_resolve_input_csv", lambda ctx, wd, needs_q_dtype_w=False: tmp_path / "in.csv")
    monkeypatch.setattr(ac, "resolve_aiter_root", lambda: str(tmp_path))
    monkeypatch.setattr(ac._tr, "is_isolation_enabled", lambda: False)
    monkeypatch.setattr(ac._tr, "with_task_timeout", lambda cmd: cmd)
    monkeypatch.setattr(dense_args, "probe_script", lambda script: types.SimpleNamespace(supports=lambda flag: True))
    clock = _Clock()
    monkeypatch.setattr(ac, "time", types.SimpleNamespace(time=clock), raising=False)

    def writes(cmd, shapes, kwargs):
        # aiter's compare candidate, under the temp root it was pinned to.
        compare = Path(kwargs["env_override"]["TMPDIR"]) / "aiter_compare"
        compare.mkdir(exist_ok=True)
        body = _DENSE_HDR + "\n" + "\n".join(f"gfx950,256,{s},ck,8,0,200.0,knl,100,1000,0.0" for s in shapes) + "\n"
        candidate = compare / f"{_arg(cmd, '-o').stem}.4242.candidate.csv"
        candidate.write_text(body)
        # The kernel's file clock lags time.time() by up to a tick; a real chunk runs for minutes.
        later = time.time() + 5
        os.utime(candidate, (later, later))
        _arg(cmd, "-o2").write_text(body)

    def run(cards, fault):
        # aiter's dense tuners exit 0 when they finish.
        monkeypatch.setattr(ac, "run_subprocess", _deadline_run(clock, cards, writes, rc=0, fault=fault))
        spec = ac.DenseTunerSpec("a8w8_blockscale", "a8w8_blockscale", "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE")
        csv = tmp_path / "in.csv"
        ctx = _ctx(tmp_path, precision="fp8", quant_type="blockscale", mp=cards, timeout_s=TIMEOUT_S, untuned_csv=csv)
        return ac.run_aiter_dense_tuner(spec, ctx, tmp_path)

    return run


@pytest.mark.parametrize(
    "cards, fault, status, total, expected, errors",
    [
        pytest.param(1, None, "partial_output", 15, 18, ["3 shape(s) not reached before the deadline"], id="one-card"),
        # aiter exits 1 when a shape in its input has no valid candidate, and a GPU fault kills the process; either
        # costs that chunk's shapes only.
        pytest.param(1, 1, "partial_output", 14, 18, ["exited 1", "illegal memory access"], id="one-chunk-fails"),
        pytest.param(4, None, "ok", 18, 18, [], id="four-cards"),
        pytest.param(4, "all", "failed", 0, None, ["illegal memory access"], id="every-chunk-fails"),
    ],
)
def test_aiter_dense_under_the_cap(aiter_dense, cards, fault, status, total, expected, errors):
    result = aiter_dense(cards, lambda i: _HIP_FAULT if fault in ("all", i) else None)
    assert (result.status, result.total_shapes) == (status, total)
    assert expected is None or result.expected_shapes == expected
    assert all(e in result.error for e in errors) and (errors or result.error == "")
    if status == "failed":
        assert result.error_class == "subprocess_error"
    elif total:
        assert len(Path(result.artifact_path).read_text().splitlines()) == 1 + total
