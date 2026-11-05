# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""``--select power``: which tuners are asked for aiter's lowest-energy picks, and that the deployed table keeps them."""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import tempfile
import time
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

import kernelforge.workflows.gemm_tune.tune_robustness as robustness_mod
import kernelforge.workflows.gemm_tune.tuners._aiter_dense_common as ac
import kernelforge.workflows.gemm_tune.tuners._select as sel
import kernelforge.workflows.gemm_tune.tuners.fmoe_ck as fm
import kernelforge.workflows.gemm_tune.tuners.sglang_dense_bf16 as bf16
from kernelforge.cli.gemm_tune_command import gemm_tune
from kernelforge.workflows.gemm_tune import pipeline
from kernelforge.workflows.gemm_tune.candidates import is_candidate
from kernelforge.workflows.gemm_tune.script_probe import ScriptSurface
from kernelforge.workflows.gemm_tune.tune_robustness import MergedRows, compare_report_lines
from kernelforge.workflows.gemm_tune.tuners._chunks import MergedCsv
from kernelforge.workflows.gemm_tune.tuners._serve_safe import _cap_splitk_to_serve_safe, _SplitKLimit
from kernelforge.workflows.gemm_tune.tuners.base import TuneContext, TuneResult

_HDR = "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio,select"


def _aiter_stdout(shape, pre, post, speedup, status, section, improve, reason):
    """What aiter's --compare prints for one shape: its performance table, then its Compare Report."""
    return "\n".join(
        [
            "============= Tune Performance Comparison =============",
            f"{'Shape':<40} | {'Pre-E2E(us)':>13} | {'Post-E2E(us)':>14} | {'Speedup':>8} | {'Status':>8}",
            f"{shape:<40} | {pre:>13} | {post:>14} | {speedup:>8} | {status:>8}",
            "============= Compare Report =============",
            "Comparison is latency only (pre_us / post_us).",
            f"--- {section} (1 shapes) ---",
            f"{shape:<40} | {pre:>10} | {post:>10} | {improve:>9} | {reason:>18}",
            "",
        ]
    )


_SHAPE = "(64, 5120, 5120)"
# aiter's compare skipped the shape: the tuned kernel did not beat the shipped one on latency.
_LOST = _aiter_stdout(_SHAPE, "16.00", "18.00", "0.89x", "OK", "Skipped", "-12.50%", "< 3.0% improve")
# The production op computed a wrong result with the tuned kernel: not a latency verdict. aiter computes no
# improvement for a failed post-run.
_MISMATCH = _aiter_stdout(_SHAPE, "16.00", "14.00", "1.14x", "MISMATCH", "Skipped", "N/A", "post-mismatch")
# The production op could not run the tuned kernel, so aiter has no post timing.
_ERROR = _aiter_stdout(_SHAPE, "16.00", "N/A", "N/A", "ERROR", "Skipped", "N/A", "post-error")
_MISMATCH_NO_BASELINE = _aiter_stdout(_SHAPE, "N/A", "14.00", "N/A", "MISMATCH", "Skipped", "N/A", "post-mismatch")
_WON = _aiter_stdout(_SHAPE, "20.00", "18.00", "1.11x", "OK", "Would update", "10.00%", "UPDATE")
_NEW = _aiter_stdout(_SHAPE, "N/A", "18.00", "N/A", "OK", "Would update", "N/A", "NEW")


def _surface(script, *flags, probed=True):
    return ScriptSurface(str(script), frozenset(flags), probed, "" if probed else "no --help output")


@pytest.fixture
def accepts_select(monkeypatch):
    monkeypatch.setattr(sel, "probe_script", lambda script, **kw: _surface(script, "--select", "--mp"))


def _ctx(tmp_path, select):
    return TuneContext(
        profile=types.SimpleNamespace(),
        framework="vllm-aiter",
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
        select=select,
        untuned_csv=tmp_path / "in.csv",
    )


class TestSelectPlan:
    def test_power_reads_the_script_s_help_afresh(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(sel, "probe_script", lambda script, **kw: calls.append(kw) or _surface(script, "--select"))

        sel.select_plan(tmp_path / "t.py", "power")

        assert calls == [{"use_cache": False}]

    def test_latency_asks_for_nothing_and_probes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sel, "probe_script", lambda script, **kw: pytest.fail("a latency tune must not probe"))
        assert sel.select_plan(tmp_path / "t.py", "latency") == sel.SelectPlan(())

    def test_power_is_asked_for_when_the_script_lists_select(self, tmp_path, accepts_select):
        plan = sel.select_plan(tmp_path / "t.py", "power")
        assert plan.args == ("--select", "power")
        assert plan.note == ""

    def test_an_aiter_without_select_tunes_for_latency_and_says_why(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sel, "probe_script", lambda script, **kw: _surface(script, "--mp"))
        plan = sel.select_plan(tmp_path / "t.py", "power")
        assert plan.args == ()
        assert "predates power selection" in plan.note

    def test_an_unreadable_help_is_not_taken_to_accept_select(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sel, "probe_script", lambda script, **kw: _surface(script, probed=False))
        plan = sel.select_plan(tmp_path / "t.py", "power")
        assert plan.args == ()
        assert "could not be read" in plan.note


@pytest.mark.parametrize("table", ["", "missing.csv"])
def test_a_power_run_that_left_no_table_says_so(tmp_path, caplog, table):
    select, note = sel.selected(sel.SelectPlan(("--select", "power")), tmp_path / table if table else "")

    assert (select, note) == ("latency", "the tuner left no table")
    assert "power selection not applied: the tuner left no table" in caplog.text


def _dense(tmp_path, monkeypatch, select, *, rows=("power",), splitk=0, profile=(), report=_LOST, rc=0):
    """Run the a8w8 blockscale dense tuner against a fake aiter whose tuned table holds one shape, which lost on latency."""
    monkeypatch.setenv("FORGE_SPLITK_TRIAL", "0")
    (tmp_path / "in.csv").write_text("M,N,K\n64,5120,5120\n")
    body = [f"gfx950,256,64,5120,5120,ck,8,{splitk},18.0,knl,90,900,0.0,{stamp}" for stamp in rows]
    (tmp_path / "tuned_a8w8_blockscale.csv").write_text("\n".join([_HDR, *body]) + "\n")
    (tmp_path / "profile_a8w8_blockscale.csv").write_text("\n".join([_HDR, *body, *profile]) + "\n")
    commands, retimed = [], []
    monkeypatch.setattr(ac, "find_tuner_script", lambda key: tmp_path / "gemm_a8w8_blockscale_tune.py")
    monkeypatch.setattr(ac, "_resolve_input_csv", lambda ctx, wd, needs_q_dtype_w=False: tmp_path / "in.csv")
    monkeypatch.setattr(ac, "resolve_aiter_root", lambda: str(tmp_path))
    monkeypatch.setattr(ac._tr, "is_isolation_enabled", lambda: False)
    monkeypatch.setattr(ac._tr, "with_task_timeout", lambda cmd: cmd)
    monkeypatch.setattr(ac, "run_subprocess", lambda cmd, **kw: (commands.append(cmd), (rc, report, ""))[1])
    monkeypatch.setattr(ac, "_find_latest_candidate", lambda name, start: None)
    monkeypatch.setattr(ac, "retime_table", lambda *args: retimed.append(args) or [])
    result = ac.run_aiter_dense_tuner(
        ac.DenseTunerSpec(
            "a8w8_blockscale", "a8w8_blockscale", "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE", ("--libtype", "all")
        ),
        _ctx(tmp_path, select),
        tmp_path,
    )
    return result, commands, retimed, _rows(tmp_path / "tuned_a8w8_blockscale.csv")


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


class TestDenseTuner:
    def test_a_power_table_keeps_the_pick_that_lost_on_latency_and_goes_to_e2e(
        self, tmp_path, monkeypatch, accepts_select
    ):
        result, commands, retimed, deployed = _dense(tmp_path, monkeypatch, "power")

        assert commands[0][-3:] == ["--select", "power", "--shape_grouped"]
        assert [(row["M"], row["select"]) for row in deployed] == [("64", "power")]
        assert retimed == []
        assert (result.select, result.select_note) == ("power", "")
        assert result.candidate is True and is_candidate(result)

    def test_the_same_run_for_latency_drops_the_loser_and_re_times(self, tmp_path, monkeypatch, accepts_select):
        result, commands, retimed, deployed = _dense(tmp_path, monkeypatch, "latency", rows=("",))

        assert "--select" not in commands[0] and "--shape_grouped" not in commands[0]
        assert deployed == []
        assert len(retimed) == 1
        assert result.select == "latency"
        assert result.candidate is False

    def test_a_power_pick_production_cannot_serve_is_dropped_not_swapped_for_the_fastest(
        self, tmp_path, monkeypatch, accepts_select
    ):
        faster_servable = "gfx950,256,64,5120,5120,ck,7,2,15.0,fast,99,990,0.0,"
        result, _, _, deployed = _dense(tmp_path, monkeypatch, "power", splitk=3, profile=(faster_servable,))

        assert deployed == []
        assert (result.select, result.candidate) == ("latency", False)
        assert result.select_note == "every power row was dropped as unservable, inaccurate or unverified"

    @pytest.mark.parametrize("report", [_WON, _NEW])
    def test_a_shape_the_split_k_cap_dropped_from_a_power_table_reports_nothing(
        self, tmp_path, monkeypatch, accepts_select, report
    ):
        result, _, _, deployed = _dense(tmp_path, monkeypatch, "power", splitk=3, report=report)

        assert deployed == []
        assert (result.status, result.total_shapes, result.select) == ("empty_output", 0, "latency")
        assert is_candidate(result) is False

    @pytest.mark.parametrize("report", [_MISMATCH, _ERROR, _MISMATCH_NO_BASELINE])
    @pytest.mark.parametrize("select", ["power", "latency"])
    def test_a_pick_the_production_op_ran_wrongly_is_dropped_from_any_table(
        self, tmp_path, monkeypatch, accepts_select, select, report
    ):
        result, _, _, deployed = _dense(tmp_path, monkeypatch, select, rows=(select,), report=report)

        assert deployed == []
        assert result.candidate is False

    def test_an_unstamped_table_from_a_power_run_goes_through_the_latency_rules(
        self, tmp_path, monkeypatch, accepts_select
    ):
        result, commands, retimed, deployed = _dense(tmp_path, monkeypatch, "power", rows=("",))

        assert commands[0][-3:] == ["--select", "power", "--shape_grouped"]
        assert deployed == [] and len(retimed) == 1
        assert result.select == "latency"
        assert "no power row" in result.select_note
        assert result.candidate is False

    def test_an_aiter_without_select_runs_a_latency_tune(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sel, "probe_script", lambda script, **kw: _surface(script, "--mp"))
        result, commands, retimed, deployed = _dense(tmp_path, monkeypatch, "power", rows=("",))

        assert "--select" not in commands[0] and "--shape_grouped" not in commands[0]
        assert deployed == [] and len(retimed) == 1
        assert result.select == "latency" and "predates" in result.select_note


def _write(path, rows):
    header = _HDR.split(",")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _row(kernel_id, splitk, us, name):
    return [
        "gfx950",
        "256",
        "64",
        "5120",
        "17408",
        "ck",
        str(kernel_id),
        str(splitk),
        str(us),
        name,
        "1",
        "1",
        "0.0",
        "",
    ]


@pytest.mark.parametrize(("with_profile", "expected"), [(True, ["sk2"]), (False, [])])
def test_without_a_profile_an_unservable_split_k_row_is_dropped(tmp_path, with_profile, expected):
    artifact, profile = tmp_path / "artifact.csv", tmp_path / "profile.csv"
    _write(artifact, [_row(9, 3, 39.0, "sk3")])
    _write(profile, [_row(9, 3, 39.0, "sk3"), _row(8, 2, 39.6, "sk2")])

    changed, _ = _cap_splitk_to_serve_safe(
        artifact, profile if with_profile else None, _SplitKLimit(2), forwarding_libtypes=frozenset({"ck"})
    )

    assert changed == 1
    assert [row["kernelName"] for row in _rows(artifact)] == expected


# Rows of aiter's MoE compare table, as base_tuner prints them.
_MOE_SLOWER = _aiter_stdout(
    "(4, 4096, 512, 256, 6)", "18.00", "20.00", "0.90x", "OK", "Skipped", "-11.11%", "< 3.0% improve"
)
_MOE_SLOWER_512 = _aiter_stdout(
    "(512, 4096, 512, 256, 6)", "18.00", "20.00", "0.90x", "OK", "Skipped", "-11.11%", "< 3.0% improve"
)
_MOE_MISMATCH = _aiter_stdout(
    "(512, 4096, 512, 256, 6)", "18.00", "16.00", "1.12x", "MISMATCH", "Skipped", "N/A", "post-mismatch"
)
_MOE_ERROR = _aiter_stdout("(512, 4096, 512, 256, 6)", "18.00", "N/A", "N/A", "ERROR", "Skipped", "N/A", "post-error")


def _fmoe(tmp_path, monkeypatch, select, *report, rc=0, untuned=None):
    """Run the fMoE tuner against a fake aiter that prints ``report``, rows of its compare table."""
    from kernelforge.tests.workflows.gemm_tune.test_fmoe_ck import _moe_ctx, _write_runtime_csv

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(fm, "find_tuner_script", lambda _name: tmp_path / "gemm_moe_tune.py")
    monkeypatch.setattr(fm, "resolve_aiter_root", lambda: None)
    monkeypatch.setattr(fm._tr, "is_isolation_enabled", lambda: False)
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        compare_dir = Path(kwargs["env_override"]["TMPDIR"]) / "aiter_compare"
        compare_dir.mkdir(exist_ok=True)
        header, *rows = Path(cmd[cmd.index("-i") + 1]).read_text(encoding="utf-8").splitlines()
        stamp = "power" if "--select" in cmd else ""
        body = [f"{row},stage1,stage2,{stamp}" for row in rows]
        candidate = compare_dir / f"{Path(cmd[cmd.index('-o') + 1]).stem}.1.candidate.csv"
        candidate.write_text("\n".join([f"{header},kernelName1,kernelName2,select", *body]) + "\n", encoding="utf-8")
        os.utime(candidate, (time.time() + 1, time.time() + 1))
        return rc, "".join(report), ""

    monkeypatch.setattr(fm, "run_subprocess", run)
    untuned = untuned or _write_runtime_csv(tmp_path)
    tuner = fm.FmoeCKTuner(_moe_ctx(tmp_path, moe_untuned_csv=untuned, select=select))
    return tuner.run(), commands


@pytest.mark.parametrize(("select", "promoted"), [("power", True), ("latency", False)])
def test_an_fmoe_power_table_that_improved_no_shape_still_goes_to_e2e(
    tmp_path, monkeypatch, accepts_select, select, promoted
):
    result, commands = _fmoe(tmp_path, monkeypatch, select, _MOE_SLOWER, _MOE_SLOWER_512)

    assert ("--select" in commands[0]) is promoted
    assert result.status == "no_improvement" and result.artifact_path
    assert is_candidate(result) is promoted


@pytest.mark.parametrize("failed", [_MOE_MISMATCH, _MOE_ERROR])
def test_an_fmoe_power_table_with_a_shape_the_production_op_ran_wrongly_is_not_promoted(
    tmp_path, monkeypatch, accepts_select, failed
):
    result, _ = _fmoe(tmp_path, monkeypatch, "power", _MOE_SLOWER, failed)

    assert result.select == "power"
    assert is_candidate(result) is False


@pytest.mark.parametrize(("select", "promoted"), [("power", True), ("latency", False)])
def test_a_bf16_power_table_slower_than_torch_still_goes_to_e2e(
    tmp_path, monkeypatch, accepts_select, select, promoted
):
    import kernelforge.tests.workflows.gemm_tune._sglang_dense_bf16_run as harness

    def row(m, libtype, us):
        stamp = "power" if select == "power" else ""
        return (
            f"gfx950,256,{m},4096,4096,False,torch.bfloat16,torch.bfloat16,False,False,"
            f"{libtype},438410,0,{us},knl,0.0,750.0,3000.0,{stamp}"
        )

    monkeypatch.setattr(harness, "_HDR", harness._HDR + ",select")
    picks = [row(1, "hipblaslt", 12.0), row(512, "hipblaslt", 30.0)]
    captured = harness._prep(
        tmp_path, monkeypatch, tuned_rows=picks, profile_rows=[*picks, row(1, "torch", 10.0), row(512, "torch", 25.0)]
    )

    result = harness._run(tmp_path, select=select)

    assert ("--select" in captured["cmd"]) is promoted
    assert result.status == "no_improvement"
    assert is_candidate(result) is promoted


def test_the_bf16_tuner_passes_select_with_every_chunk(tmp_path, accepts_select):
    tuner = bf16.SglangDenseBf16Tuner.__new__(bf16.SglangDenseBf16Tuner)
    tuner.ctx = _ctx(tmp_path, "power")
    tuner._plan = sel.select_plan(tmp_path / "gemm_a16w16_tune.py", "power")

    tail = tuner._tail(tmp_path / "u.csv", tmp_path / "t.csv", tmp_path / "p.csv", 60)

    assert tail[-2:] == ["--select", "power"]


def test_a_tuner_without_power_selection_says_it_tuned_for_latency(tmp_path, monkeypatch):
    class _Tuner:
        def execute(self):
            return TuneResult(tuner_name="vllm_moe_triton", status="ok")

    monkeypatch.setattr(pipeline, "_create_tuner", lambda name, ctx: _Tuner())
    spec = types.SimpleNamespace(
        name="vllm_moe_triton", should_run=True, fallback=False, skip_reason="", token_hint=None
    )

    results, _ = pipeline._execute_tuners([spec], _ctx(tmp_path, "power"), global_deadline=float("inf"))

    assert results[0].select == "latency"
    assert results[0].select_note == "vllm_moe_triton has no power selection; tuned for latency"
    assert results[0].to_dict()["select_note"] == results[0].select_note


def test_gemm_tune_run_hands_select_to_every_tuner(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kernelforge.workflows.gemm_tune.aiter_preflight.collect", lambda: {"soft": [], "hard": [], "aligned": True}
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    seen = []

    class _Tuner:
        def execute(self):
            return TuneResult(tuner_name="a8w8", status="no_improvement", select="power")

    monkeypatch.setattr(pipeline, "_create_tuner", lambda name, ctx: seen.append(ctx.select) or _Tuner())
    args = ["run", "--model-path", str(model), "--framework", "vllm", "--precision", "fp8", "--gpu-type", "mi300x"]
    args += ["--tuner", "a8w8", "--skip-gpu-check", "--output-dir", str(tmp_path / "out"), "--select", "power"]

    result = CliRunner().invoke(gemm_tune, args)

    assert result.exit_code == 0, result.output
    assert seen == ["power"]
    assert json.loads((tmp_path / "out" / "result.json").read_text())["tuners_run"][0]["select"] == "power"


_LAT = "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio"
_POW = _LAT + ",watts,select,power_cap_w"


def _merged(*files):
    merged = MergedRows()
    for lines in files:
        merged.absorb(lines)
    return merged.header, merged.rows


class TestChunkMerge:
    """aiter adds the power columns only to a run that measured some row's watts."""

    def test_a_header_only_run_does_not_decide_the_columns(self):
        power_row = "gfx950,256,1,2,3,ck,8,0,18.0,k,1,1,0.0,400,power,1400"

        assert _merged([_LAT], [_POW, power_row]) == (_POW, [power_row])

    def test_runs_with_different_columns_are_merged_by_name(self):
        header, rows = _merged(
            [_LAT, "gfx950,256,1,2,3,ck,8,0,18.0,k,1,1,0.0"],
            [_POW, 'gfx950,256,4,5,6,ck,9,0,20.0,"a,b",1,1,0.0,400,power,1400'],
        )

        assert header == _POW
        parsed = list(csv.DictReader([header, *rows]))
        assert [(r["M"], r["select"], r["kernelName"]) for r in parsed] == [("1", "", "k"), ("4", "power", "a,b")]

    def test_runs_with_the_same_columns_keep_their_lines(self):
        lines = ["gfx950,256,1,2,3,ck,8,0,18.0,k,1,1,0.0", 'gfx950,256,4,5,6,ck,9,0,20.0,"a,b",1,1,0.0']

        assert _merged([_LAT, lines[0]], [_LAT, lines[1]]) == (_LAT, lines)

    def test_a_run_of_header_only_files_keeps_the_first_header(self):
        assert _merged([_LAT], [_POW]) == (_LAT, [])

    def test_the_chunk_merger_writes_what_it_merged(self, tmp_path):
        merged = MergedCsv(tmp_path / "merged.csv")
        merged.absorb([_LAT])
        merged.absorb([_POW, "gfx950,256,1,2,3,ck,8,0,18.0,k,1,1,0.0,400,power,1400"])

        assert [row["select"] for row in _rows(tmp_path / "merged.csv")] == ["power"]


_POWER_ROW_256 = "gfx950,256,256,5120,5120,ck,8,0,18.0,knl,90,900,0.0,400,power,1400"
_REPORT_256 = _aiter_stdout(
    "(256, 5120, 5120)", "16.00", "18.00", "0.89x", "OK", "Skipped", "-12.50%", "< 3.0% improve"
)


@pytest.mark.parametrize(
    "chunks",
    [
        # The first chunk measured no power (header-only latency columns, exit 1); the second wrote a power row.
        [(_LAT + "\n", "", 1), (f"{_POW}\n{_POWER_ROW_256}\n", _REPORT_256, 0)],
        # The first chunk wrote a verified power row but exited 1; the deadline killed the second.
        [(f"{_POW}\n{_POWER_ROW_256}\n", _REPORT_256, 1), (_LAT + "\n", "", 124)],
    ],
)
def test_a_dense_power_pick_verified_in_one_chunk_survives_the_others(tmp_path, monkeypatch, accepts_select, chunks):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FORGE_SPLITK_TRIAL", "0")
    (tmp_path / "in.csv").write_text("M,N,K\n1,4096,4096\n256,5120,5120\n")
    monkeypatch.setattr(ac, "find_tuner_script", lambda key: tmp_path / "gemm_a8w8_blockscale_tune.py")
    monkeypatch.setattr(ac, "_resolve_input_csv", lambda ctx, wd, needs_q_dtype_w=False: tmp_path / "in.csv")
    monkeypatch.setattr(ac, "_demand_input_csv", lambda *a, **k: None)
    monkeypatch.setattr(ac, "resolve_aiter_root", lambda: str(tmp_path))
    monkeypatch.setattr(ac._tr, "is_isolation_enabled", lambda: False)
    monkeypatch.setattr(ac, "retime_table", lambda *args: pytest.fail("a power table is not re-timed"))
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        compare_dir = Path(kwargs["env_override"]["TMPDIR"]) / "aiter_compare"
        compare_dir.mkdir(exist_ok=True)
        candidate = compare_dir / f"{Path(cmd[cmd.index('-o') + 1]).stem}.{len(calls)}.candidate.csv"
        table, stdout, rc = chunks[len(calls) - 1]
        candidate.write_text(table)
        os.utime(candidate, (time.time() + 1, time.time() + 1))
        return rc, stdout, ""

    monkeypatch.setattr(ac, "run_subprocess", run)
    spec = ac.DenseTunerSpec(
        "a8w8_blockscale", "a8w8_blockscale", "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE", ("--libtype", "all")
    )

    result = ac.run_aiter_dense_tuner(spec, dataclasses.replace(_ctx(tmp_path, "power"), timeout_s=100000), tmp_path)

    assert len(calls) == 2
    assert (result.select, result.candidate) == ("power", True)
    assert [(row["M"], row["select"]) for row in _rows(Path(result.artifact_path))] == [("256", "power")]


def test_a_bf16_power_table_the_accuracy_filter_emptied_says_so(tmp_path, monkeypatch, accepts_select):
    import kernelforge.tests.workflows.gemm_tune._sglang_dense_bf16_run as harness

    monkeypatch.setattr(harness, "_HDR", harness._HDR + ",select")
    wrong = [
        f"gfx950,256,{m},4096,4096,False,torch.bfloat16,torch.bfloat16,False,False,hipblaslt,438410,0,12.0,knl,0.5,"
        "750.0,3000.0,power"
        for m in (1, 512)
    ]
    harness._prep(tmp_path, monkeypatch, tuned_rows=wrong, profile_rows=wrong)

    result = harness._run(tmp_path, select="power")

    assert result.status == "empty_output"
    assert (result.select, result.select_note) == (
        "latency",
        "every power row was dropped as unservable, inaccurate or unverified",
    )


def test_a_bf16_power_run_that_measured_nothing_says_so(tmp_path, monkeypatch, accepts_select):
    import kernelforge.tests.workflows.gemm_tune._sglang_dense_bf16_run as harness

    harness._prep(tmp_path, monkeypatch, tuned_rows=[], profile_rows=[])

    result = harness._run(tmp_path, select="power")

    assert result.status == "empty_output"
    assert (result.select, result.select_note) == (
        "latency",
        "aiter measured package power for no shape, so the table has no power row",
    )


def test_an_fmoe_power_table_with_no_compared_shape_is_not_promoted(tmp_path, monkeypatch, accepts_select):
    result, _ = _fmoe(tmp_path, monkeypatch, "power")

    assert (result.status, result.select) == ("empty_output", "power")
    assert is_candidate(result) is False


class TestCompareReport:
    """aiter's performance table repeats each shape with its own status words; only the report decides a row."""

    def test_only_the_compare_report_is_read(self):
        lines = compare_report_lines(_MISMATCH_NO_BASELINE + _WON)

        assert [line for line in lines if "|" in line] == [
            line for line in (_MISMATCH_NO_BASELINE + _WON).splitlines() if "post-mismatch" in line or "UPDATE" in line
        ]

    def test_an_aiter_without_a_compare_report_is_read_whole(self):
        text = "(64, 5120, 5120) | 16.00 | 14.00 | 12.50% | UPDATE"

        assert compare_report_lines(text) == [text]

    def test_a_dense_shape_whose_post_run_failed_is_one_failed_result_with_no_speedup(self):
        (row,) = ac._parse_tuner_stdout(_MISMATCH_NO_BASELINE, "")

        assert (row["post_failed"], row["speedup"], row.get("is_new")) == (True, None, None)

    @pytest.mark.parametrize("report", [_MOE_MISMATCH, _MOE_ERROR])
    def test_an_fmoe_shape_whose_post_run_failed_is_one_failed_result_with_no_speedup(self, report):
        tuner = fm.FmoeCKTuner.__new__(fm.FmoeCKTuner)

        (row,) = tuner._parse_compare_output(report)

        assert (row["post_failed"], row["speedup"]) == (True, None)


def test_an_fmoe_shape_with_nothing_compared_is_not_a_result(tmp_path, monkeypatch, accepts_select):
    new = _aiter_stdout("(4, 4096, 512, 256, 6)", "N/A", "18.00", "N/A", "OK", "Would update", "N/A", "NEW")

    result, _ = _fmoe(tmp_path, monkeypatch, "latency", new)

    assert (result.status, result.total_shapes) == ("empty_output", 0)


class TestAiterExitsOneForAShapeWithNoPowerRow:
    """aiter exits 1 when a shape stays untuned, which a power tune does to every shape it measured no power for."""

    @pytest.mark.parametrize(
        ("select", "rows", "kept"), [("power", ("power",), True), ("power", ("",), False), ("latency", ("",), False)]
    )
    def test_a_dense_run(self, tmp_path, monkeypatch, accepts_select, select, rows, kept):
        result, _, _, _ = _dense(tmp_path, monkeypatch, select, rows=rows, rc=1)

        assert (result.status, is_candidate(result)) == (("partial_output", True) if kept else ("failed", False))

    @pytest.mark.parametrize(("select", "kept"), [("power", True), ("latency", False)])
    def test_an_fmoe_run(self, tmp_path, monkeypatch, accepts_select, select, kept):
        result, _ = _fmoe(tmp_path, monkeypatch, select, _MOE_SLOWER, _MOE_SLOWER_512, rc=1)

        assert (result.status, is_candidate(result)) == (("partial_output", True) if kept else ("failed", False))


def test_an_fmoe_power_table_with_a_row_the_report_never_verified_is_not_promoted(
    tmp_path, monkeypatch, accepts_select
):
    result, _ = _fmoe(tmp_path, monkeypatch, "power", _MOE_SLOWER, rc=1)

    assert result.select == "power"
    assert is_candidate(result) is False


class TestPowerRowsTheReportNeverVerified:
    """aiter writes a batch's rows before it runs the production op on them."""

    def test_a_dense_run_that_crashed_before_its_report_deploys_nothing(self, tmp_path, monkeypatch, accepts_select):
        result, _, _, deployed = _dense(tmp_path, monkeypatch, "power", report="", rc=134)

        assert deployed == []
        assert is_candidate(result) is False

    def test_a_dense_power_row_only_the_candidate_csv_shows_is_dropped(self, tmp_path, monkeypatch, accepts_select):
        (tmp_path / "candidate_a8w8_blockscale.csv").write_text(
            f"{_HDR}\ngfx950,256,64,5120,5120,ck,8,0,18.0,knl,90,900,0.0,power\n"
        )
        result, _, _, deployed = _dense(tmp_path, monkeypatch, "power", report="Successfully tuned 1 shapes\n")

        assert deployed == []
        assert is_candidate(result) is False


def _two_key_untuned(tmp_path):
    """Two MoE keys (inter_dim 512 and 1024) for tokens 4 and 512, as one round's untuned CSV holds them."""
    from kernelforge.tests.workflows.gemm_tune.test_fmoe_ck import _write_runtime_csv

    first = _write_runtime_csv(tmp_path).read_text(encoding="utf-8").splitlines()
    second = _write_runtime_csv(tmp_path, inter_dim="1024").read_text(encoding="utf-8").splitlines()
    path = tmp_path / "two_keys.csv"
    path.write_text("\n".join([*first, *second[1:]]) + "\n", encoding="utf-8")
    return path


def _moe_line(token, inter_dim, post, improve, reason):
    return _aiter_stdout(
        f"({token}, 4096, {inter_dim}, 256, 6)", "18.00", post, "N/A", "OK", "Skipped", improve, reason
    )


_TWO_KEYS_SLOWER = [_moe_line(t, i, "20.00", "-11.11%", "< 3.0% improve") for t in (4, 512) for i in (512, 1024)]


@pytest.mark.parametrize(
    ("report", "promoted"),
    [
        (_TWO_KEYS_SLOWER, True),
        (_TWO_KEYS_SLOWER[:3] + [_moe_line(512, 1024, "16.00", "N/A", "post-mismatch")], False),
        (_TWO_KEYS_SLOWER[:3], False),
    ],
)
def test_an_fmoe_power_table_with_two_keys_is_promoted_only_when_every_row_passed(
    tmp_path, monkeypatch, accepts_select, report, promoted
):
    result, _ = _fmoe(tmp_path, monkeypatch, "power", *report, untuned=_two_key_untuned(tmp_path))

    assert result.select == "power"
    assert is_candidate(result) is promoted


def _blocklist(tmp_path, power):
    ctx = types.SimpleNamespace(
        gpu_type="mi355x", quant_type="blockscale", tp=1, faulted_blocklist_path=tmp_path / "f.json"
    )
    return robustness_mod.fault_blocklist_for(ctx, "a8w8_blockscale", power=power)


class TestAPowerTuneKeepsItsOwnFaultRecord:
    """Grouped, a power tune loses a whole shape to one faulting candidate; latency tunes must not skip it for that."""

    def test_a_latency_tune_reads_the_record_it_always_did(self, tmp_path):
        assert _blocklist(tmp_path, False).key == "mi355x:blockscale:1:a8w8_blockscale"

    def test_a_shape_a_power_tune_lost_is_not_skipped_by_a_latency_tune(self, tmp_path):
        power = _blocklist(tmp_path, True)
        power.record("64,5120,5120", "hard_fault")
        power.save()

        assert {p: _blocklist(tmp_path, p).is_blocked("64,5120,5120") for p in (True, False)} == {
            True: True,
            False: False,
        }

    def test_a_power_tune_skips_the_shapes_a_latency_tune_recorded(self, tmp_path):
        latency = _blocklist(tmp_path, False)
        latency.record("64,5120,5120", "hard_fault")
        latency.save()

        assert _blocklist(tmp_path, True).is_blocked("64,5120,5120")

    @pytest.mark.parametrize(("flags", "power"), [(("--select", "--mp"), True), (("--mp",), False)])
    def test_a_power_request_records_under_the_selection_the_run_used(self, tmp_path, monkeypatch, flags, power):
        """A request whose aiter has no --select runs, and records, as a latency tune."""
        monkeypatch.setattr(sel, "probe_script", lambda script, **kw: _surface(script, *flags))
        keys = []

        def isolated(invocation, guard):
            keys.append(guard.blocklist.key)
            return 1, "", "", None

        monkeypatch.setattr(ac._tr, "is_isolation_enabled", lambda: True)
        monkeypatch.setattr(ac._tr, "run_isolated", isolated)
        monkeypatch.setattr(fm._tr, "is_isolation_enabled", lambda: True)
        monkeypatch.setattr(fm._tr, "run_isolated", isolated)
        monkeypatch.setattr(ac, "find_tuner_script", lambda key: tmp_path / "gemm_a8w8_blockscale_tune.py")
        monkeypatch.setattr(ac, "_resolve_input_csv", lambda ctx, wd, needs_q_dtype_w=False: tmp_path / "in.csv")
        (tmp_path / "in.csv").write_text("M,N,K\n64,5120,5120\n")
        spec = ac.DenseTunerSpec("a8w8_blockscale", "a8w8_blockscale", "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE", ())
        ac.run_aiter_dense_tuner(spec, _ctx(tmp_path, "power"), tmp_path)
        from kernelforge.tests.workflows.gemm_tune.test_fmoe_ck import _moe_ctx, _write_runtime_csv

        monkeypatch.setattr(fm, "find_tuner_script", lambda _name: tmp_path / "gemm_moe_tune.py")
        monkeypatch.setattr(fm, "resolve_aiter_root", lambda: None)
        fm.FmoeCKTuner(_moe_ctx(tmp_path, moe_untuned_csv=_write_runtime_csv(tmp_path), select="power")).run()

        assert [key.endswith(":power") for key in keys] == [power, power]


def test_a_power_table_emptied_of_its_split_k_row_is_not_a_candidate(tmp_path, monkeypatch, accepts_select):
    result, _, _, deployed = _dense(tmp_path, monkeypatch, "power", splitk=1, report=_ERROR)

    assert deployed == []
    assert (result.select, is_candidate(result)) == ("latency", False)
