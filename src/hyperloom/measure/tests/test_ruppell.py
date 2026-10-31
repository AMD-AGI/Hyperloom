# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ruppell in a profile test: when it is built, what the server's environment gains, and what the report keeps.

The tree is the stand-in of :mod:`.ruppell_fixtures`, staged and pinned for the test; the real tree went through the same
calls on CPU.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
import tempfile
from pathlib import Path

import pytest
import yaml

from hyperloom.measure import ruppell
from hyperloom.measure.ruppell_stage import stage
from hyperloom.measure.tests.ruppell_fixtures import COMMIT, DATA, archive

EARLY = ("python", "ruppell", "_rocm_early")


def _point(directory: Path, **benchmark: object) -> Path:
    config = {"framework": "sglang", "envs": {"TP": 2}, **benchmark}
    path = directory / "point.yaml"
    path.write_text(yaml.safe_dump({"benchmark": config}), encoding="utf-8")
    return path


@pytest.fixture
def host(tmp_path, monkeypatch) -> dict[str, str]:
    """A ROCm 7.2.0 image on an MI355X with the pinned tree staged: the environment ``prepare`` reads."""
    rocm = tmp_path / "rocm"
    (rocm / ".info").mkdir(parents=True)
    (rocm / ".info" / "version").write_text("7.2.0-43\n")
    (rocm / "bin").mkdir()
    enumerator = rocm / "bin" / "rocm_agent_enumerator"
    enumerator.write_text("#!/bin/sh\necho gfx000\necho gfx950\n")
    enumerator.chmod(0o755)
    _, digest = stage(tmp_path / "cache", COMMIT, archive())
    monkeypatch.setattr(ruppell, "RUPPELL_COMMIT", COMMIT)
    monkeypatch.setattr(ruppell, "RUPPELL_TREE_SHA256", digest)
    monkeypatch.setenv("FAKE_RUPPELL_DATA", str(DATA))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    return {**os.environ, "ROCM_PATH": str(rocm), "HYPERLOOM_RUNNER_CACHE": str(tmp_path / "cache")}


def _tracer(tmp_path: Path, host: dict[str, str]) -> ruppell.Tracer:
    tracer = ruppell.prepare([_point(tmp_path)], host)
    assert isinstance(tracer, ruppell.Tracer), tracer
    return tracer


# --- when it is built ---------------------------------------------------------------------------------------------


def test_a_staged_tree_is_copied_and_built_for_the_cards_architecture_in_the_tests_own_directory(tmp_path, host):
    tracer = _tracer(tmp_path, host)

    assert tracer.work.parent == tmp_path / "tmp" and tracer.source.parent == tracer.work
    built = dict(line.split("=", 1) for line in (tracer.build / "BUILD_ENV").read_text().splitlines())
    assert built["RUPPELL_ARCH"] == "gfx950", "the first real architecture, not the CPU agent gfx000"
    assert (built["RUPPELL_PLATFORM"], built["RUPPELL_BUILD"]) == ("rocm", str(tracer.build))
    assert built["ROCM_PATH"] == host["ROCM_PATH"]
    tracer.close()
    assert not tracer.work.exists()


@pytest.mark.parametrize(
    ("benchmark", "says"),
    [
        ({"framework": "vllm"}, "the framework is vllm, not SGLang"),
        ({"agentx": {"enabled": True}}, "an AgentX launch takes its environment elsewhere"),
    ],
)
def test_a_point_ruppell_cannot_trace_runs_untraced(tmp_path, host, benchmark, says):
    assert ruppell.prepare([_point(tmp_path, **benchmark)], host) == f"ruppell: skipped ({says})"


@pytest.mark.parametrize(("version", "shown"), [("10.0.0", "10.0.0"), ("7.0.1", "7.0.1"), (None, "unknown")])
def test_a_rocm_outside_ruppells_range_runs_untraced(tmp_path, host, version, shown):
    info = Path(host["ROCM_PATH"]) / ".info" / "version"
    if version is None:
        info.unlink()
    else:
        info.write_text(version)

    assert ruppell.prepare([_point(tmp_path)], host).startswith(
        f"ruppell: skipped (ROCm {shown} is outside 7.0.2 to 8.0.0"
    )


def test_a_tree_that_is_not_staged_or_not_the_pinned_one_runs_untraced(tmp_path, host, monkeypatch):
    assert "not staged" in ruppell.prepare([_point(tmp_path)], {**host, "HYPERLOOM_RUNNER_CACHE": str(tmp_path / "x")})
    monkeypatch.setattr(ruppell, "RUPPELL_TREE_SHA256", "0" * 64)
    assert "holds" in ruppell.prepare([_point(tmp_path)], host)
    assert list((tmp_path / "tmp").iterdir()) == [], "the copy is removed"


def test_a_host_with_no_gpu_architecture_runs_untraced(tmp_path, host):
    (Path(host["ROCM_PATH"]) / "bin" / "rocm_agent_enumerator").write_text("#!/bin/sh\necho gfx000\n")
    assert "no GPU architecture" in ruppell.prepare([_point(tmp_path)], host)


def test_a_build_that_fails_or_hangs_runs_untraced_with_its_last_line(tmp_path, host, monkeypatch):
    assert ruppell.prepare([_point(tmp_path)], {**host, "FAKE_BUILD_RC": "2"}) == (
        "ruppell: skipped (bash build.sh exited 2: hipcc: error: no ROCm)"
    )
    monkeypatch.setattr(ruppell, "_BUILD_TIMEOUT_SEC", 1.0)
    assert (
        ruppell.prepare([_point(tmp_path)], {**host, "FAKE_BUILD_SLEEP": "30"})
        == "ruppell: skipped (bash build.sh ran past 1s)"
    )
    assert list((tmp_path / "tmp").iterdir()) == []


# --- what the server's environment gains --------------------------------------------------------------------------


def test_the_server_gets_ruppell_runs_environment_with_the_points_own_values_kept(tmp_path, host):
    tracer = _tracer(tmp_path, host)
    point = _point(tmp_path, envs={"TP": 2, "DEBUG_CLR_GRAPH_PACKET_CAPTURE": "1", "PYTHONPATH": "/farm/site:/img"})

    (traced,) = tracer.traced_points([point], host)

    assert traced.parent == tracer.work and yaml.safe_load(point.read_text())["benchmark"]["envs"]["TP"] == 2
    envs = yaml.safe_load(traced.read_text())["benchmark"]["envs"]
    build = tracer.build
    assert envs["RUPPELL_TOOL_LIB"] == envs["ROCP_TOOL_LIBRARIES"] == str(build / "libruppell_tool.so")
    assert (envs["RUPPELL_OUT"], envs["RUPPELL_MARKER_LIB"]) == (
        str(tracer.records),
        str(build / "libruppell_marker.so"),
    )
    assert (envs["NCCL_PROFILER_PLUGIN"], envs["RUPPELL_NCCL_KERNEL_CH"]) == (
        str(build / "libruppell_nccl_profiler.so"),
        "0",
    )
    assert (envs["RUPPELL_SGLANG"], envs["RUPPELL_DETAIL"]) == ("1", "5/1000")
    assert envs["DEBUG_CLR_GRAPH_PACKET_CAPTURE"] == "1", "a value the point sets is the point's"
    assert envs["TP"] == 2
    assert "PYTHONDONTWRITEBYTECODE" not in envs, "only what ruppell run sets"
    assert json.loads((tracer.records / "run.json").read_text())["commit"] == COMMIT


def test_ruppells_hook_goes_first_and_its_package_after_the_farm(tmp_path, host):
    """ruppell's start-up hook runs the sitecustomize it shadows from the start of sys.path: anywhere but first it runs a
    farm hook twice. Its package goes last so it shadows nothing of the farm's or the image's."""
    tracer = _tracer(tmp_path, host)
    point = _point(tmp_path, envs={"PYTHONPATH": "/farm/site:/img"})

    (traced,) = tracer.traced_points([point], {**host, "PYTHONPATH": "/runner"})

    path = yaml.safe_load(traced.read_text())["benchmark"]["envs"]["PYTHONPATH"].split(os.pathsep)
    site = tracer.work / "site"
    assert path == [str(tracer.source.joinpath(*EARLY)), "/farm/site", "/img", str(tracer.source / "python"), str(site)]
    entry_points = (site / "ruppell_sglang-0.0.dist-info" / "entry_points.txt").read_text()
    assert entry_points == "[sglang.srt.plugins]\nruppell = ruppell.adapters.sglang:register\n"


def test_a_point_with_no_path_of_its_own_builds_on_the_inherited_one(tmp_path, host):
    tracer = _tracer(tmp_path, host)

    (traced,) = tracer.traced_points([_point(tmp_path)], {**host, "PYTHONPATH": "/runner:/img"})

    path = yaml.safe_load(traced.read_text())["benchmark"]["envs"]["PYTHONPATH"].split(os.pathsep)
    assert path[:3] == [str(tracer.source.joinpath(*EARLY)), "/runner", "/img"]


# --- what the report keeps ----------------------------------------------------------------------------------------


def _ran(tracer: ruppell.Tracer) -> None:
    tracer.records.mkdir(parents=True, exist_ok=True)
    (tracer.records / "run.json").write_text(json.dumps({"commit": COMMIT}))


def test_the_report_keeps_the_census_the_check_and_the_module_table_beside_the_profile(tmp_path, host):
    tracer = _tracer(tmp_path, host)
    _ran(tracer)

    note = tracer.report(tmp_path / "out", 1)

    assert note == "ruppell: ok, 12356 detail steps, check ok"
    kept = tmp_path / "out" / "ruppell"
    assert sorted(p.name for p in kept.iterdir()) == [
        "census.jsonl.gz", "census_scopes.json", "census_scopes.npz", "census_summary.json", "check.json",
        "module_shares.csv", "run.json", "ruppell.log",
    ]  # fmt: skip
    assert gzip.decompress((kept / "census.jsonl.gz").read_bytes()) == (DATA / "offline_census.jsonl").read_bytes()
    assert json.loads((kept / "check.json").read_text())["ok"] is True
    assert not (tracer.work / "trace.json").exists(), "the export's trace is not kept"
    log = (kept / "ruppell.log").read_text()
    assert "--expect-ranks 1" in log
    assert re.search(r"build\.sh\n# exit 0 after \d+\.\ds\n", log), "the build's wall time is in the log"
    assert not (tmp_path / "out" / "profile").exists(), "nothing of ruppell's lands among the traces"


def test_a_failed_check_and_incomplete_records_are_said(tmp_path, host, monkeypatch):
    tracer = _tracer(tmp_path, host)
    _ran(tracer)
    summary = json.loads((DATA / "w243_census_summary.json").read_text())
    summary["incomplete_processes"] = ["rank 0 (worker)"]
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    monkeypatch.setenv("FAKE_SUMMARY", str(tmp_path / "summary.json"))
    monkeypatch.setenv("FAKE_CHECK_OK", "0")

    assert tracer.report(tmp_path / "out", 1) == "ruppell: ok, 12356 detail steps, check failed, records incomplete"


@pytest.mark.parametrize(
    ("setup", "says"),
    [
        ("no_records", "the server left no records"),
        ("census_fails", "python -m ruppell census exited 3: census: no records"),
    ],
)
def test_a_report_that_fails_is_a_note_and_never_raises(tmp_path, host, monkeypatch, setup, says):
    tracer = _tracer(tmp_path, host)
    if setup == "census_fails":
        _ran(tracer)
        monkeypatch.setenv("FAKE_CENSUS_RC", "3")

    note = tracer.report(tmp_path / "out", 1)

    assert note.startswith("ruppell: report failed (") and says in note
    assert (tmp_path / "out" / "ruppell" / "ruppell.log").is_file(), "the log says what failed"


def test_module_shares_are_taken_per_step_kind_from_a_census_ruppell_wrote(tmp_path):
    """``offline_census.jsonl`` is ruppell 0ef9b0d's ``census`` of its own offline records (five decode steps)."""
    rows = ruppell.module_shares(DATA / "offline_census.jsonl")

    assert rows == [{"kind": "decode", "module": "x", "gpu_ms": 0.022, "share_percent": 100.0, "steps": 5}]


def test_module_shares_keep_decode_and_prefill_apart_and_leave_startup_steps_out(tmp_path):
    def line(kind: str | None, by_op: dict[str, int], phase: str = "engine") -> str:
        return json.dumps({"step": 1, "by_op_ns": by_op, "fields": {"phase": phase, "kind": kind}})

    census = tmp_path / "census.jsonl"
    census.write_text(
        "\n".join(
            [
                line("decode", {"RadixAttention": 3_000_000, "RowParallelLinear": 1_000_000}),
                line("decode", {"RadixAttention": 1_000_000, "unattributed": 1_000_000}),
                line("prefill", {"RowParallelLinear": 2_000_000}),
                line(None, {"warmup": 9_000_000}, phase="startup"),
            ]
        )
        + "\n"
    )

    ruppell.write_module_shares(ruppell.module_shares(census), tmp_path / "m.csv")

    with (tmp_path / "m.csv").open() as handle:
        table = [(r["kind"], r["module"], r["gpu_ms"], r["share_percent"], r["steps"]) for r in csv.DictReader(handle)]
    assert table == [
        ("decode", "RadixAttention", "4.0", "66.667", "2"),
        ("decode", "RowParallelLinear", "1.0", "16.667", "2"),
        ("decode", "unattributed", "1.0", "16.667", "2"),
        ("prefill", "RowParallelLinear", "2.0", "100.0", "1"),
    ]
