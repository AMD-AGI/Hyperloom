# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Decode rows ship the fastest eligible candidate as timed in a CUDA graph; the GPU timer is replaced by a table."""

from __future__ import annotations

import csv
import sys
import time
import types
from pathlib import Path

import pytest

import kernelforge.workflows.gemm_tune.aiter_graph_retime as retime
import kernelforge.workflows.gemm_tune.aiter_splitk_validate as validate
import kernelforge.workflows.gemm_tune.tuners._aiter_dense_common as ac
from kernelforge.workflows.gemm_tune.aiter_graph_retime import Kernel, RetimeError, RetimeTarget, Timing, make_retime_fn
from kernelforge.workflows.gemm_tune.tuners._graph_retime import retime_decode_rows
from kernelforge.workflows.gemm_tune.tuners._serve_safe import _SplitKLimit
from kernelforge.workflows.gemm_tune.tuners.base import TuneContext

_HDR = ["gfx", "cu_num", "M", "N", "K", "libtype", "kernelId", "splitK", "us", "kernelName", "tflops", "bw", "errRatio"]
_OP = "a8w8_blockscale_bpreshuffle"
_ASM = frozenset({"asm"})


def _row(name: str, libtype: str, us: float, *, splitk: int = 0, err: str = "0.0", m: int = 64) -> list[str]:
    return ["gfx950", "256", str(m), "7168", "5120", libtype, "1", str(splitk), str(us), name, "1", "1", err]


def _write(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows([_HDR, *rows])
    return path


def _names(path: Path) -> list[str]:
    with path.open(newline="") as fh:
        return [r["kernelName"] for r in csv.DictReader(fh)]


class _GraphTimes:
    """A timing run answering in-graph microseconds per kernel name; a name mapped to a string fails with it."""

    def __init__(self, times: dict[str, float | str]):
        self.times = times
        self.timed: list[tuple[int, str]] = []

    def __call__(self, rows):
        self.timed += [(int(r["M"]), r["kernelName"]) for r in rows]
        answers = [self.times[r["kernelName"]] for r in rows]
        return [Timing(error=a) if isinstance(a, str) else Timing(us=a) for a in answers]


def _retime(tmp_path, deployed, profile, times, *, ceiling=8, rel_norm=None):
    art = _write(tmp_path / "tuned.csv", deployed)
    prof = _write(tmp_path / "profile.csv", profile)
    timer = _GraphTimes(times)
    limit = _SplitKLimit(2, lambda m, n, k: ceiling, rel_norm)
    return art, timer, retime_decode_rows(art, prof, limit, _ASM, timer)


# qkv M=64: the tuner ranks ck first, in a graph asm splitK=4 is faster.
_CK = _row("ck_pick", "ck", 14.48)
_ASM_SK4 = _row("asm_sk4", "asm", 14.60, splitk=4)


def test_the_fastest_eligible_candidate_in_a_graph_replaces_the_tuners_pick(tmp_path):
    art, _, retimed = _retime(tmp_path, [_CK], [_CK, _ASM_SK4], {"ck_pick": 14.28, "asm_sk4": 13.54})

    assert _names(art) == ["asm_sk4"]
    (record,) = [r.to_dict() for r in retimed]
    assert (record["M"], record["N"], record["K"]) == (64, 7168, 5120)
    assert record["tuner_pick"] == {"libtype": "ck", "kernelName": "ck_pick", "splitK": 0, "graph_us": 14.28}
    assert record["shipped"] == {"libtype": "asm", "kernelName": "asm_sk4", "splitK": 4, "graph_us": 13.54}
    assert record["graph_speedup"] == round(14.28 / 13.54, 4)


def test_a_pick_already_fastest_in_a_graph_stays_and_is_not_reported(tmp_path):
    art, _, retimed = _retime(tmp_path, [_CK], [_CK, _ASM_SK4], {"ck_pick": 13.0, "asm_sk4": 13.54})

    assert _names(art) == ["ck_pick"] and retimed == []


@pytest.mark.parametrize(
    ("candidate", "rel_norm"),
    [
        (_row("ck_wrong", "ck", 9.0, err="0.05"), None),
        (_row("asm_sk6_wrong", "asm", 9.0, splitk=6, err="0.05"), lambda m, n, k, sk, name: 0.02),
        (_row("asm_sk6_unmeasured", "asm", 9.0, splitk=6, err="0.05"), None),
        (_row("asm_sk8_over_ceiling", "asm", 9.0, splitk=8), None),
        (_row("ck_splitk", "ck", 9.0, splitk=2), None),
    ],
    ids=["inaccurate", "rel_norm_fails", "rel_norm_unmeasured", "over_ceiling", "splitk_not_forwarded"],
)
def test_a_candidate_the_gates_would_not_ship_is_never_chosen_even_if_faster(tmp_path, candidate, rel_norm):
    name = candidate[_HDR.index("kernelName")]
    art, timer, retimed = _retime(
        tmp_path, [_CK], [_CK, candidate], {"ck_pick": 14.0, name: 1.0}, ceiling=6, rel_norm=rel_norm
    )

    assert _names(art) == ["ck_pick"] and retimed == []
    assert name not in [timed for _, timed in timer.timed]


def test_an_asm_splitk_winner_ships_with_its_relative_norm_figure(tmp_path):
    winner = _row("asm_sk6", "asm", 15.0, splitk=6, err="0.013")

    art, _, _ = _retime(
        tmp_path, [_CK], [_CK, winner], {"ck_pick": 14.0, "asm_sk6": 12.0}, rel_norm=lambda m, n, k, sk, name: 0.0041
    )

    with art.open(newline="") as fh:
        (row,) = list(csv.DictReader(fh))
    assert (row["kernelName"], float(row["relNormErr"])) == ("asm_sk6", 0.0041)


def test_only_the_top_candidates_by_tuner_time_are_timed_besides_the_pick(tmp_path):
    profile = [_row(f"ck{i}", "ck", 10.0 + i) for i in range(5)]
    pick = _row("asm_slow", "asm", 20.0, splitk=1)
    times = {f"ck{i}": 30.0 - i for i in range(5)} | {"asm_slow": 40.0}

    art, timer, _ = _retime(tmp_path, [pick], [*profile, pick], times)

    assert sorted(name for _, name in timer.timed) == ["asm_slow", "ck0", "ck1", "ck2"]
    assert _names(art) == ["ck2"]


def test_rows_above_the_graph_batch_range_are_untouched(tmp_path):
    prefill = [_row("ck_pick", "ck", 50.0, m=512), _row("asm_fast", "asm", 51.0, splitk=1, m=512)]

    art, timer, retimed = _retime(tmp_path, prefill[:1], prefill, {"ck_pick": 60.0, "asm_fast": 10.0})

    assert _names(art) == ["ck_pick"] and retimed == [] and timer.timed == []


def test_a_pick_that_cannot_be_timed_keeps_its_row_and_is_reported(tmp_path):
    before = _write(tmp_path / "before.csv", [_CK]).read_text()

    art, _, retimed = _retime(tmp_path, [_CK], [_CK, _ASM_SK4], {"ck_pick": "HIP error: no device", "asm_sk4": 13.0})

    assert art.read_text() == before
    (record,) = [r.to_dict() for r in retimed]
    assert record["shipped"] == record["tuner_pick"] and record["shipped"]["graph_us"] is None
    assert "HIP error: no device" in record["errors"][0] and "graph_speedup" not in record


def test_a_candidate_that_cannot_be_timed_is_skipped_and_reported(tmp_path):
    slower = _row("asm_sk2", "asm", 15.0, splitk=2)
    times = {"ck_pick": 14.28, "asm_sk4": "kernel not present in the compiled registry", "asm_sk2": 13.9}

    art, _, retimed = _retime(tmp_path, [_CK], [_CK, _ASM_SK4, slower], times)

    assert _names(art) == ["asm_sk2"]
    (record,) = [r.to_dict() for r in retimed]
    assert record["shipped"]["kernelName"] == "asm_sk2" and "compiled registry" in record["errors"][0]


def test_an_op_without_a_registered_retime_keeps_the_tuners_picks(tmp_path):
    assert make_retime_fn(RetimeTarget("a8w8_blockscale", "", time.time() + 600), tmp_path) is None
    art = _write(tmp_path / "tuned.csv", [_CK])
    prof = _write(tmp_path / "profile.csv", [_CK, _ASM_SK4])
    before = art.read_text()

    assert retime_decode_rows(art, prof, _SplitKLimit(8), _ASM, None) == []
    assert art.read_text() == before


def test_an_unknown_libtype_cannot_be_timed(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiter", types.SimpleNamespace())  # CI has no aiter
    with pytest.raises(RuntimeError, match="opus"):
        retime._BpreshuffleGraphTimer._call(Kernel("opus", 3, "opus_k", 0))


def test_the_timer_runs_every_libtype_through_its_production_op_by_kernel_name(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(
        gemm_a8w8_blockscale_bpreshuffle_asm=lambda *a, **kw: calls.append(("asm", kw)),
        gemm_a8w8_blockscale_bpreshuffle_ck=lambda *a, **kw: calls.append(("ck", kw)),
        gemm_a8w8_blockscale_bpreshuffle_cktile=lambda *a, **kw: calls.append(("cktile", kw)),
    )
    monkeypatch.setitem(sys.modules, "aiter", fake)
    operands = ("x", "w", "xs", "ws", "out")

    retime._BpreshuffleGraphTimer._call(Kernel("asm", 5, "asm_k", 4))(*operands)
    retime._BpreshuffleGraphTimer._call(Kernel("ck", 7, "ck_k", 0))(*operands)
    retime._BpreshuffleGraphTimer._call(Kernel("cktile", 9, "tile_k", 0))(*operands)

    assert calls == [
        ("asm", {"splitK": 4, "kernelName": "asm_k"}),
        ("ck", {"kernelName": "ck_k"}),
        ("cktile", {"kernelName": "tile_k"}),
    ]


def test_the_worker_records_a_kernel_that_cannot_dispatch_and_times_the_rest(monkeypatch):
    def timer(m, n, k, kernel):
        if kernel.kernel_name == "missing":
            raise RuntimeError("kernel 'missing' is not present in the compiled registry")
        return 12.5

    monkeypatch.setitem(retime._RETIME_OPS, _OP, retime._RetimeOp(lambda device: timer, retime._BPRESHUFFLE_MODULES))
    rows = [dict(zip(_HDR, _row("missing", "ck", 1.0))), dict(zip(_HDR, _row("asm_sk4", "asm", 1.0, splitk=4)))]

    assert retime._time_rows(_OP, "cuda", rows) == [
        Timing(error="kernel 'missing' is not present in the compiled registry"),
        Timing(us=12.5),
    ]


def _worker(tmp_path, monkeypatch, *, rc: int = 0, results=None, seconds_left: float = 600.0):
    """A worker run under ``tmp_path`` whose subprocess exits ``rc`` after writing ``results``."""
    jit = tmp_path / "aiter_jit"
    (jit / "build").mkdir(parents=True)
    for name in ("module_aiter_core.so", "module_gemm_a8w8_blockscale_bpreshuffle.so"):
        (jit / name).write_text("")
    monkeypatch.setattr(retime, "resolve_package_root", lambda: tmp_path)
    monkeypatch.setattr(retime, "resolve_jit_build_dir", lambda root: jit / "build")
    seen = {}

    def run(command, *, timeout_s, log_file, env_override):
        # The shared runner caps the aiter build's jobs to the container and kills its whole group on timeout.
        seen.update(command=command, env=env_override, timeout=timeout_s)
        if results is not None:
            Path(command[-1]).write_text(results)
        return rc, "", ""

    monkeypatch.setattr(retime, "run_subprocess", run)
    return make_retime_fn(RetimeTarget(_OP, "", time.time() + seconds_left), tmp_path / "run"), seen


def test_the_worker_rebuilds_the_production_modules_from_the_candidates(tmp_path, monkeypatch):
    time_fn, seen = _worker(tmp_path, monkeypatch, results='[{"us": 13.5, "error": ""}, {"us": null, "error": "x"}]')
    rows = [dict(zip(_HDR, _row("ck_pick", "ck", 14.0))), dict(zip(_HDR, _row("asm_sk4", "asm", 14.6, splitk=4)))]

    assert time_fn(rows) == [Timing(us=13.5), Timing(error="x")]
    private = Path(seen["env"]["AITER_JIT_DIR"])
    assert sorted(p.name for p in private.iterdir()) == ["module_aiter_core.so"]
    with open(seen["env"]["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"], newline="") as fh:
        build = list(csv.DictReader(fh))
    assert [(r["M"], r["kernelName"]) for r in build] == [("1", "ck_pick"), ("2", "asm_sk4")]


def test_the_worker_gets_what_is_left_of_the_tuners_budget_and_none_once_it_is_spent(tmp_path, monkeypatch):
    rows = [dict(zip(_HDR, _row("ck_pick", "ck", 14.0)))]
    time_fn, seen = _worker(tmp_path / "left", monkeypatch, results='[{"us": 13.5, "error": ""}]', seconds_left=600.0)
    time_fn(rows)
    assert 590 < seen["timeout"] <= 600

    time_fn, seen = _worker(tmp_path / "spent", monkeypatch, results="[]", seconds_left=-1.0)
    with pytest.raises(RetimeError, match="budget"):
        time_fn(rows)
    assert "command" not in seen


def test_a_failed_worker_raises_with_its_log(tmp_path, monkeypatch):
    time_fn, _ = _worker(tmp_path, monkeypatch, rc=1)

    with pytest.raises(RetimeError, match="exited 1"):
        time_fn([dict(zip(_HDR, _CK))])


def test_a_failed_timing_run_keeps_every_decode_row_and_reports_it(tmp_path):
    art = _write(tmp_path / "tuned.csv", [_CK, _row("asm_pick", "asm", 9.0, splitk=1, m=16)])
    prof = _write(tmp_path / "profile.csv", [_CK, _ASM_SK4])
    before = art.read_text()

    def failing(rows):
        raise RetimeError("timing worker exited -11")

    retimed = retime_decode_rows(art, prof, _SplitKLimit(8), _ASM, failing)

    assert art.read_text() == before
    assert [(r.shape[0], r.errors) for r in retimed] == [
        (16, ("timing worker exited -11",)),
        (64, ("timing worker exited -11",)),
    ]


def _ctx(tmp_path: Path) -> TuneContext:
    return TuneContext(
        profile=types.SimpleNamespace(),
        framework="sglang",
        precision="fp8",
        quant_type="blockscale",
        gpu_type="mi355x",
        tp=1,
        conc=64,
        tokens=[64],
        mp=1,
        output_dir=tmp_path,
        iters=1,
        warmup=0,
        min_improvement_pct=3.0,
        timeout_s=60,
        untuned_csv=tmp_path / "in.csv",
    )


def test_the_bpreshuffle_tuner_reports_the_retimed_row(tmp_path, monkeypatch):
    monkeypatch.delenv("FORGE_SPLITK_TRIAL", raising=False)
    monkeypatch.setitem(validate._TRIAL_OPS, _OP, lambda m, n, k, sk, device="cuda": True)
    timer = _GraphTimes({"ck_pick": 14.28, "asm_sk4": 13.54})
    monkeypatch.setattr(retime, "_WorkerRetime", lambda target, production, run_dir: timer)
    (tmp_path / "in.csv").write_text("M,N,K\n64,7168,5120\n")
    _write(tmp_path / "tuned_bpre.csv", [_CK])
    _write(tmp_path / "profile_bpre.csv", [_CK, _ASM_SK4])
    monkeypatch.setattr(ac, "find_tuner_script", lambda key: tmp_path / "script.py")
    monkeypatch.setattr(ac, "_resolve_input_csv", lambda ctx, wd, needs_q_dtype_w=False: tmp_path / "in.csv")
    monkeypatch.setattr(ac, "resolve_aiter_root", lambda: str(tmp_path))
    monkeypatch.setattr(ac._tr, "is_isolation_enabled", lambda: False)
    monkeypatch.setattr(ac, "run_subprocess", lambda cmd, **kw: (0, "", ""))
    monkeypatch.setattr(ac, "_find_latest_candidate", lambda name, t: None)
    spec = ac.DenseTunerSpec("bpre", _OP, "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE", ("--splitK",))

    result = ac.run_aiter_dense_tuner(spec, _ctx(tmp_path), tmp_path)

    assert _names(Path(result.env_value)) == ["asm_sk4"]
    (record,) = result.to_dict()["graph_retime"]
    assert (record["tuner_pick"]["kernelName"], record["shipped"]["kernelName"]) == ("ck_pick", "asm_sk4")
