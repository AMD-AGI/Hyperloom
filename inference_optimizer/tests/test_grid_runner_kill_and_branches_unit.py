# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for ``_grid_runner`` process-reaping (``_kill_stale_servers``) and
the ``run_grid`` per-variant failure branches (yaml build error, magpie
timeout, server-dead / overtime sentinels, missing workspace, invalid
measurement)."""
from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from inference_optimizer.orchestrator.action_executors import _grid_runner as gr
from inference_optimizer.orchestrator.action_executors._grid_runner import (
    GridVariant,
    run_grid,
)


# ---------------------------------------------------------------------------
# _kill_stale_servers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _single_node(monkeypatch):
    """Default every test in this module to single-node mode."""
    from inference_optimizer.orchestrator.action_executors import _multi_node_env
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: False)


def test_kill_stale_servers_noop_in_multi_node(monkeypatch):
    from inference_optimizer.orchestrator.action_executors import _multi_node_env
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: True)
    slept: list = []
    monkeypatch.setattr("time.sleep", lambda *_a: slept.append(True))
    gr._kill_stale_servers()
    assert slept == []  # returned before reaching the sleep


def _proc_open_factory(cmdlines: dict[str, bytes], maps: dict[str, str]):
    """Build a fake ``open`` that serves /proc cmdline + maps from dicts."""
    real_open = open

    def _fake_open(path, mode="r", *args, **kwargs):
        p = str(path)
        if p.endswith("/cmdline"):
            data = cmdlines.get(p)
            if data is None:
                raise OSError("no cmdline")
            return io.BytesIO(data)
        if p.endswith("/maps"):
            data = maps.get(p)
            if data is None:
                raise OSError("no maps")
            return io.StringIO(data)
        return real_open(path, mode, *args, **kwargs)

    return _fake_open


def test_kill_stale_servers_kills_pattern_and_orphan_atom(monkeypatch):
    killpg_calls: list[int] = []
    kill_calls: list[int] = []
    removed: list[str] = []
    slept: list[int] = []

    monkeypatch.setattr(os, "getpid", lambda: 999)
    monkeypatch.setattr(os, "getpgrp", lambda: 100)
    monkeypatch.setattr(os, "listdir", lambda _p: ["1", "2", "999", "abc"])
    cmdlines = {
        "/proc/1/cmdline": b"vllm serve\x00--model\x00m\x00",  # _KILL_PATTERNS
        "/proc/2/cmdline": b"python\x00--multiprocessing-fork\x00",  # orphan atom
    }
    maps = {"/proc/2/maps": "7f00 r-xp /ATOM/atom/libfoo.so\n"}
    monkeypatch.setattr("builtins.open", _proc_open_factory(cmdlines, maps))

    pgid_map = {1: 50, 2: 60}
    monkeypatch.setattr(os, "getpgid", lambda pid: pgid_map[pid])
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append(pgid))
    monkeypatch.setattr(os, "kill", lambda pid, sig: kill_calls.append(pid))
    monkeypatch.setattr("glob.glob", lambda pat: [f"{pat}_seg"])
    monkeypatch.setattr(os, "remove", lambda f: removed.append(f))
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

    gr._kill_stale_servers()

    assert set(killpg_calls) == {50, 60}
    assert set(kill_calls) == {1, 2}
    assert removed  # /dev/shm segments cleared
    # Orphan atom worker killed -> long KFD-release pause.
    assert slept == [8]


def test_kill_stale_servers_swallows_proc_errors(monkeypatch):
    slept: list[int] = []

    def _raise_getpgrp():
        raise OSError("no pgrp")

    monkeypatch.setattr(os, "getpid", lambda: 999)
    monkeypatch.setattr(os, "getpgrp", _raise_getpgrp)  # my_pgid = -1
    monkeypatch.setattr(os, "listdir", lambda _p: ["3", "4"])
    cmdlines = {
        "/proc/3/cmdline": b"sglang.srt\x00",  # matches; getpgid raises below
        # pid 4 cmdline missing -> open raises OSError -> continue
    }
    monkeypatch.setattr("builtins.open", _proc_open_factory(cmdlines, {}))

    def _getpgid(_pid):
        raise OSError("gone")

    monkeypatch.setattr(os, "getpgid", _getpgid)

    def _kill(_pid, _sig):
        raise ProcessLookupError("already dead")

    monkeypatch.setattr(os, "kill", _kill)
    monkeypatch.setattr(os, "killpg", lambda *_a: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr("glob.glob", lambda pat: [f"{pat}_seg"])

    def _remove(_f):
        raise OSError("held")

    monkeypatch.setattr(os, "remove", _remove)
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

    gr._kill_stale_servers()  # all errors swallowed
    assert slept == [2]  # no atom worker reaped -> short pause


# ---------------------------------------------------------------------------
# run_grid failure branches
# ---------------------------------------------------------------------------
def _write_base_yaml(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/wekafs/models/Qwen-Qwen3-8B",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "benchmark_script": "sglang_mi300x.sh",
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        },
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


@pytest.mark.asyncio
async def test_run_grid_yaml_build_error_branch(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _boom(*_a, **_k):
        raise ValueError("bad yaml render")

    monkeypatch.setattr(gr, "_build_variant_yaml", _boom)
    results = await run_grid(
        base_yaml_path=base, base_extra_args="",
        grid=[GridVariant("vA")], output_root=tmp_path / "out",
        variant_timeout_sec=5,
    )
    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].error_class == "yaml_build_error"


@pytest.mark.asyncio
async def test_run_grid_magpie_timeout_branch(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _timeout(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="magpie", timeout=5)

    monkeypatch.setattr(gr, "_run_magpie", _timeout)
    results = await run_grid(
        base_yaml_path=base, base_extra_args="",
        grid=[GridVariant("vA")], output_root=tmp_path / "out",
        variant_timeout_sec=5,
    )
    assert results[0].status == "failed"
    assert results[0].error_class == "magpie_timeout"


@pytest.mark.asyncio
async def test_run_grid_server_dead_branch(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _dead(*_a, **_k):
        return gr.SERVER_DEAD_RETURNCODE, "", "engine crashed"

    monkeypatch.setattr(gr, "_run_magpie", _dead)
    results = await run_grid(
        base_yaml_path=base, base_extra_args="",
        grid=[GridVariant("vA")], output_root=tmp_path / "out",
        variant_timeout_sec=5,
    )
    assert results[0].status == "failed"
    assert results[0].error_class == "server_init_dead"
    assert results[0].returncode == gr.SERVER_DEAD_RETURNCODE


@pytest.mark.asyncio
async def test_run_grid_overtime_kill_branch(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _overtime(*_a, **_k):
        return gr.OVERTIME_KILL_RETURNCODE, "", ""

    monkeypatch.setattr(gr, "_run_magpie", _overtime)
    results = await run_grid(
        base_yaml_path=base, base_extra_args="",
        grid=[GridVariant("vA")], output_root=tmp_path / "out",
        variant_timeout_sec=5, soft_deadline_sec=1.0,
    )
    assert results[0].status == "failed"
    assert results[0].killed_overtime is True
    # No server.log written -> no estimate, but still no crash.
    assert results[0].estimated_output_throughput is None


@pytest.mark.asyncio
async def test_run_grid_overtime_kill_estimates_tput_from_server_log(
    tmp_path, monkeypatch,
):
    """A killed-overtime variant salvages a rough output tput from the engine's
    partial ``server.log`` decode-throughput logs (informational only)."""
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _overtime(*_a, output_dir, **_k):
        # Mimic the engine dumping periodic decode throughput before the soft
        # deadline reaper fires.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "server.log").write_text(
            "Decode batch. gen throughput (token/s): 100.0, #queue-req: 0\n"
            "Decode batch. gen throughput (token/s): 900.0, #queue-req: 0\n"
            "Decode batch. gen throughput (token/s): 1000.0, #queue-req: 0\n"
            "Decode batch. gen throughput (token/s): 1100.0, #queue-req: 0\n"
            "Decode batch. gen throughput (token/s): 1200.0, #queue-req: 0\n"
        )
        return gr.OVERTIME_KILL_RETURNCODE, "", ""

    monkeypatch.setattr(gr, "_run_magpie", _overtime)
    results = await run_grid(
        base_yaml_path=base, base_extra_args="",
        grid=[GridVariant("vA")], output_root=tmp_path / "out",
        variant_timeout_sec=5, soft_deadline_sec=1.0,
    )
    r = results[0]
    assert r.status == "failed"
    assert r.killed_overtime is True
    # Real measurement stays absent so winner selection is unaffected.
    assert r.output_throughput is None
    # warmup trim drops the 100.0 ramp -> mean(900,1000,1100,1200)=1050.0
    assert r.estimated_output_throughput == pytest.approx(1050.0)
    assert any(
        w.startswith("estimated_output_throughput_from_server_log:")
        for w in r.nonfatal_warnings
    )


@pytest.mark.asyncio
async def test_run_grid_no_workspace_branch_stops_on_failure(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _no_ws(*_a, **_k):
        return 1, "stdout", "boom stderr"  # nonzero, no benchmark_* dir created

    monkeypatch.setattr(gr, "_run_magpie", _no_ws)
    results = await run_grid(
        base_yaml_path=base, base_extra_args="",
        grid=[GridVariant("vA"), GridVariant("vB")], output_root=tmp_path / "out",
        variant_timeout_sec=5, keep_going_on_failure=False,
    )
    # First variant fails with no workspace; keep_going_on_failure=False -> stop.
    assert len(results) == 1
    assert results[0].error_class == "no_benchmark_workspace"


@pytest.mark.asyncio
async def test_run_grid_invalid_measurement_branch(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _empty_report(magpie_python, config_path, output_dir, **_k):
        ws = Path(output_dir) / "benchmark_sglang_20260101_000000"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "benchmark_report.json").write_text(
            '{"success": false, "framework": "sglang"}',
        )
        return 0, "ok", ""

    monkeypatch.setattr(gr, "_run_magpie", _empty_report)
    results = await run_grid(
        base_yaml_path=base, base_extra_args="",
        grid=[GridVariant("vA")], output_root=tmp_path / "out",
        variant_timeout_sec=5,
    )
    assert results[0].status == "failed"
    assert results[0].error_class in {
        "benchmark_report_invalid_metric", "benchmark_report_missing",
    }
