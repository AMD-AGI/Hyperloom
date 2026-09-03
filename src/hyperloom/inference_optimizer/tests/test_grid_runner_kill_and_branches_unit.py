# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for ``_grid_runner`` process-reaping (``_kill_stale_servers``) and
the ``run_grid`` per-variant failure branches (yaml build error, magpie
timeout, server-dead / overtime sentinels, missing workspace, invalid
measurement)."""

from __future__ import annotations

import io
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _grid_runner as gr
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    run_grid,
)


@pytest.fixture(autouse=True)
def _single_node(monkeypatch):
    """Default every test in this module to single-node mode."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: False)


def test_kill_stale_servers_noop_in_multi_node(monkeypatch):
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: True)
    slept: list = []
    monkeypatch.setattr("time.sleep", lambda *_a: slept.append(True))
    gr._kill_stale_servers()
    assert slept == []


def _proc_open_factory(
    cmdlines: dict[str, bytes],
    maps: dict[str, str],
    environs: dict[str, bytes] | None = None,
):
    """Build a fake ``open`` that serves /proc cmdline + maps + environ from dicts."""
    real_open = open
    environs = environs or {}

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
        if p.endswith("/environ"):
            data = environs.get(p)
            if data is None:
                raise OSError("no environ")
            return io.BytesIO(data)
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
        "/proc/2/cmdline": b"python\x00--multiprocessing-fork\x00",
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
    assert slept == [8]


def _clear_gpu_mask_envs(monkeypatch) -> None:
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)


# ─────────────────────────────────────────────────────────────────────────────
# GPU-scoped reaping (AMD-AGI/Hyperloom#1354)
# ─────────────────────────────────────────────────────────────────────────────


def test_kill_stale_servers_skips_candidate_outside_our_gpu_mask(monkeypatch):
    """When we have our own visible-GPU mask (an operator carved us a subset
    of the machine), a matching candidate whose own mask is disjoint from
    ours must be left alone -- it belongs to someone else's GPU allocation."""
    _clear_gpu_mask_envs(monkeypatch)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")

    killpg_calls: list[int] = []
    kill_calls: list[int] = []

    monkeypatch.setattr(os, "getpid", lambda: 999)
    monkeypatch.setattr(os, "getpgrp", lambda: 100)
    monkeypatch.setattr(os, "listdir", lambda _p: ["1"])
    cmdlines = {"/proc/1/cmdline": b"vllm serve\x00--model\x00m\x00"}
    environs = {"/proc/1/environ": b"ROCR_VISIBLE_DEVICES=0,1\x00PATH=/bin\x00"}
    monkeypatch.setattr("builtins.open", _proc_open_factory(cmdlines, {}, environs))
    monkeypatch.setattr(os, "getpgid", lambda pid: 50)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append(pgid))
    monkeypatch.setattr(os, "kill", lambda pid, sig: kill_calls.append(pid))
    monkeypatch.setattr("glob.glob", lambda pat: [])
    monkeypatch.setattr("time.sleep", lambda s: None)

    gr._kill_stale_servers()

    assert killpg_calls == []
    assert kill_calls == []


def test_kill_stale_servers_kills_candidate_overlapping_our_gpu_mask(monkeypatch):
    """A matching candidate whose mask overlaps ours (same GPU allocation)
    is reaped, same as with no mask at all."""
    _clear_gpu_mask_envs(monkeypatch)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")

    killpg_calls: list[int] = []
    kill_calls: list[int] = []

    monkeypatch.setattr(os, "getpid", lambda: 999)
    monkeypatch.setattr(os, "getpgrp", lambda: 100)
    monkeypatch.setattr(os, "listdir", lambda _p: ["1"])
    cmdlines = {"/proc/1/cmdline": b"vllm serve\x00--model\x00m\x00"}
    environs = {"/proc/1/environ": b"ROCR_VISIBLE_DEVICES=6,7\x00PATH=/bin\x00"}
    monkeypatch.setattr("builtins.open", _proc_open_factory(cmdlines, {}, environs))
    monkeypatch.setattr(os, "getpgid", lambda pid: 50)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append(pgid))
    monkeypatch.setattr(os, "kill", lambda pid, sig: kill_calls.append(pid))
    monkeypatch.setattr("glob.glob", lambda pat: [])
    monkeypatch.setattr("time.sleep", lambda s: None)

    gr._kill_stale_servers()

    assert killpg_calls == [50]
    assert kill_calls == [1]


def test_kill_stale_servers_skips_candidate_with_unreadable_environ(monkeypatch):
    """A candidate whose /proc/<pid>/environ cannot be read (permission,
    already exited) is skipped, not reaped -- unknown scope is never treated
    as safe to kill."""
    _clear_gpu_mask_envs(monkeypatch)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")

    kill_calls: list[int] = []

    monkeypatch.setattr(os, "getpid", lambda: 999)
    monkeypatch.setattr(os, "getpgrp", lambda: 100)
    monkeypatch.setattr(os, "listdir", lambda _p: ["1"])
    cmdlines = {"/proc/1/cmdline": b"vllm serve\x00--model\x00m\x00"}
    # No environ entry for pid 1 -> _proc_open_factory raises OSError on read.
    monkeypatch.setattr("builtins.open", _proc_open_factory(cmdlines, {}, {}))
    monkeypatch.setattr(os, "getpgid", lambda pid: 50)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(os, "kill", lambda pid, sig: kill_calls.append(pid))
    monkeypatch.setattr("glob.glob", lambda pat: [])
    monkeypatch.setattr("time.sleep", lambda s: None)

    gr._kill_stale_servers()

    assert kill_calls == []


def test_kill_stale_servers_skips_candidate_declaring_no_mask(monkeypatch):
    """A candidate with a readable but empty environ (no GPU-mask var set at
    all) is skipped -- its GPU scope is unknown, not "the whole machine"."""
    _clear_gpu_mask_envs(monkeypatch)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")

    kill_calls: list[int] = []

    monkeypatch.setattr(os, "getpid", lambda: 999)
    monkeypatch.setattr(os, "getpgrp", lambda: 100)
    monkeypatch.setattr(os, "listdir", lambda _p: ["1"])
    cmdlines = {"/proc/1/cmdline": b"vllm serve\x00--model\x00m\x00"}
    environs = {"/proc/1/environ": b"PATH=/bin\x00HOME=/root\x00"}
    monkeypatch.setattr("builtins.open", _proc_open_factory(cmdlines, {}, environs))
    monkeypatch.setattr(os, "getpgid", lambda pid: 50)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(os, "kill", lambda pid, sig: kill_calls.append(pid))
    monkeypatch.setattr("glob.glob", lambda pat: [])
    monkeypatch.setattr("time.sleep", lambda s: None)

    gr._kill_stale_servers()

    assert kill_calls == []


def test_kill_stale_servers_skips_shm_wipe_when_we_have_a_gpu_mask(monkeypatch):
    """The /dev/shm wipe carries no GPU/owner tag to scope by, so with a mask
    of our own it must not run at all: it could otherwise crash a correctly
    spared co-tenant's server by pulling its shared-memory segments out from
    under it, even though the per-pid reap above left it alone."""
    _clear_gpu_mask_envs(monkeypatch)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")

    glob_calls: list[str] = []
    removed: list[str] = []

    monkeypatch.setattr(os, "getpid", lambda: 999)
    monkeypatch.setattr(os, "getpgrp", lambda: 100)
    monkeypatch.setattr(os, "listdir", lambda _p: [])  # no candidates to reap
    monkeypatch.setattr("builtins.open", _proc_open_factory({}, {}, {}))
    monkeypatch.setattr("glob.glob", lambda pat: (glob_calls.append(pat), [f"{pat}_seg"])[1])
    monkeypatch.setattr(os, "remove", lambda f: removed.append(f))
    monkeypatch.setattr("time.sleep", lambda s: None)

    gr._kill_stale_servers()

    assert glob_calls == [], "the /dev/shm wipe must not even glob when we have a GPU mask"
    assert removed == []


def test_kill_stale_servers_reaps_everything_when_we_have_no_mask(monkeypatch):
    """With no mask on our own side (whole machine is ours, or nothing is
    scoping either side), every match is reaped regardless of the
    candidate's own mask -- this is the pre-existing, unscoped behavior."""
    _clear_gpu_mask_envs(monkeypatch)

    kill_calls: list[int] = []

    monkeypatch.setattr(os, "getpid", lambda: 999)
    monkeypatch.setattr(os, "getpgrp", lambda: 100)
    monkeypatch.setattr(os, "listdir", lambda _p: ["1"])
    cmdlines = {"/proc/1/cmdline": b"vllm serve\x00--model\x00m\x00"}
    # No environ served at all; must not matter since we have no mask.
    monkeypatch.setattr("builtins.open", _proc_open_factory(cmdlines, {}, {}))
    monkeypatch.setattr(os, "getpgid", lambda pid: 50)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(os, "kill", lambda pid, sig: kill_calls.append(pid))
    monkeypatch.setattr("glob.glob", lambda pat: [])
    monkeypatch.setattr("time.sleep", lambda s: None)

    gr._kill_stale_servers()

    assert kill_calls == [1]


def test_kill_stale_servers_skips_sleep_when_nothing_was_killed(monkeypatch):
    """The KFD-release pause must not be paid on the common case where the
    /proc scan finds nothing to reap. This function now runs at 4 call sites
    (was 1) instead of just before every Magpie launch, so paying an
    unconditional sleep here on the "GPU was already clean" case adds up
    fast (e.g. conc_sweep's own reap immediately followed by baseline's),
    per review on AMD-AGI/Hyperloom#1354."""
    _clear_gpu_mask_envs(monkeypatch)
    slept: list[int] = []

    monkeypatch.setattr(os, "getpid", lambda: 999)
    monkeypatch.setattr(os, "getpgrp", lambda: 100)
    monkeypatch.setattr(os, "listdir", lambda _p: ["1"])
    cmdlines = {"/proc/1/cmdline": b"python\x00-m\x00something_unrelated\x00"}
    monkeypatch.setattr("builtins.open", _proc_open_factory(cmdlines, {}, {}))
    monkeypatch.setattr("glob.glob", lambda pat: [])
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

    gr._kill_stale_servers()

    assert slept == []


def test_kill_stale_servers_swallows_proc_errors(monkeypatch):
    slept: list[int] = []

    def _raise_getpgrp():
        raise OSError("no pgrp")

    monkeypatch.setattr(os, "getpid", lambda: 999)
    monkeypatch.setattr(os, "getpgrp", _raise_getpgrp)  # my_pgid = -1
    monkeypatch.setattr(os, "listdir", lambda _p: ["3", "4"])
    cmdlines = {
        "/proc/3/cmdline": b"sglang.srt\x00",  # matches; getpgid raises below
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
    assert slept == [2]


def _write_base_yaml(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/path/models/Qwen-Qwen3-8B",
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
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
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
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
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
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
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
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
        variant_timeout_sec=5,
        soft_deadline_sec=1.0,
    )
    assert results[0].status == "failed"
    assert results[0].killed_overtime is True
    assert results[0].estimated_output_throughput is None


@pytest.mark.asyncio
async def test_run_grid_overtime_kill_estimates_tput_from_server_log(
    tmp_path,
    monkeypatch,
):
    """A killed-overtime variant salvages a rough output tput from the engine's
    partial ``server.log`` decode-throughput logs."""
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _overtime(*_a, output_dir, **_k):
        # Mimic the engine dumping periodic decode throughput before the reaper.
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
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
        variant_timeout_sec=5,
        soft_deadline_sec=1.0,
    )
    r = results[0]
    assert r.status == "failed"
    assert r.killed_overtime is True
    assert r.output_throughput is None
    # warmup trim drops the 100.0 ramp -> mean(900,1000,1100,1200)=1050.0
    assert r.estimated_output_throughput == pytest.approx(1050.0)
    assert any(w.startswith("estimated_output_throughput_from_server_log:") for w in r.nonfatal_warnings)


@pytest.mark.asyncio
async def test_run_grid_no_workspace_branch_stops_on_failure(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _no_ws(*_a, **_k):
        return 1, "stdout", "boom stderr"  # nonzero, no benchmark_* dir

    monkeypatch.setattr(gr, "_run_magpie", _no_ws)
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA"), GridVariant("vB")],
        output_root=tmp_path / "out",
        variant_timeout_sec=5,
        keep_going_on_failure=False,
    )
    assert len(results) == 1
    assert results[0].error_class == "no_benchmark_workspace"


@pytest.mark.asyncio
async def test_agentx_preflight_abort_keeps_its_own_error_class(tmp_path, monkeypatch):
    """An AgentX preflight abort must not be filed as a missing workspace.

    Of course no workspace exists -- Magpie never ran. But the generic class
    erases the one fact that decides what to do next: this is an environment
    gap, not a launch failure. Measured: with the cause gone, a missing pinned
    dependency read as a framework problem and opened an enablement round that
    burned the run's budget re-deriving an install this repository owns.
    """
    from hyperloom.orchestrator.actions.executors._subprocess_kill import (
        AGENTX_PREFLIGHT_ERROR_CLASS,
        AGENTX_PREFLIGHT_RETURNCODE,
    )

    base = tmp_path / "base.yaml"
    _write_base_yaml(base)
    diagnosis = "AgentX preflight failed: HYPERLOOM_AGENTX is on but aiperf was not found."

    monkeypatch.setattr(gr, "_run_magpie", lambda *_a, **_k: (AGENTX_PREFLIGHT_RETURNCODE, "", diagnosis))
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
        variant_timeout_sec=5,
        keep_going_on_failure=False,
    )
    assert len(results) == 1
    assert results[0].error_class == AGENTX_PREFLIGHT_ERROR_CLASS
    # The diagnosis itself has to survive too: it names the fix.
    assert "aiperf was not found" in (results[0].error or "")


@pytest.mark.asyncio
async def test_agentx_preflight_abort_abandons_the_rest_of_the_grid(tmp_path, monkeypatch):
    """The client is missing for the whole grid, not for one variant.

    The runtime repair has already run and memoized its outcome, so every
    remaining point fails identically -- and `keep_going_on_failure` would walk
    all of them to find that out. Nothing downstream stops it either: the
    writeback gate that halts a run is baseline-scoped.
    """
    from hyperloom.orchestrator.actions.executors._subprocess_kill import (
        AGENTX_PREFLIGHT_RETURNCODE,
    )

    base = tmp_path / "base.yaml"
    _write_base_yaml(base)
    monkeypatch.setattr(gr, "_run_magpie", lambda *_a, **_k: (AGENTX_PREFLIGHT_RETURNCODE, "", "aiperf was not found"))
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA"), GridVariant("vB"), GridVariant("vC")],
        output_root=tmp_path / "out",
        variant_timeout_sec=5,
        keep_going_on_failure=True,  # would otherwise walk every point
    )
    assert len(results) == 1, "the grid kept going after an environment abort"


@pytest.mark.asyncio
async def test_agentx_preflight_abort_never_reports_an_empty_error(tmp_path, monkeypatch):
    """A blank stderr must not become a blank `error`, the way the sibling
    branch's non-empty fallback already prevents."""
    from hyperloom.orchestrator.actions.executors._subprocess_kill import (
        AGENTX_PREFLIGHT_RETURNCODE,
    )

    base = tmp_path / "base.yaml"
    _write_base_yaml(base)
    monkeypatch.setattr(gr, "_run_magpie", lambda *_a, **_k: (AGENTX_PREFLIGHT_RETURNCODE, "", "   "))
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
        variant_timeout_sec=5,
        keep_going_on_failure=False,
    )
    assert (results[0].error or "").strip(), "an empty diagnosis reached the result"


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
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
        variant_timeout_sec=5,
    )
    assert results[0].status == "failed"
    assert results[0].error_class in {
        "benchmark_report_invalid_metric",
        "benchmark_report_missing",
    }


@pytest.mark.asyncio
async def test_run_grid_nonzero_rc_with_valid_measurement_fails(tmp_path, monkeypatch):
    """A parseable measurement must not launder a non-zero exit code into success."""
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _nonzero_but_valid(magpie_python, config_path, output_dir, **_k):
        ws = Path(output_dir) / "benchmark_sglang_20260101_000000"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "benchmark_report.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "framework": "sglang",
                    "throughput": {
                        "output_throughput": 1200.0,
                        "request_throughput": 120.0,
                        "completed_requests": 640,
                        "duration_seconds": 120.0,
                    },
                }
            )
        )
        return 1, "stdout tail", "server exited 1"

    monkeypatch.setattr(gr, "_run_magpie", _nonzero_but_valid)
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
        variant_timeout_sec=5,
    )
    r = results[0]
    assert r.status == "failed"
    assert r.error_class == "magpie_nonzero_after_valid_measurement"
    assert r.returncode == 1
    markers = list((tmp_path / "out").rglob("abort_reason.json"))
    assert len(markers) == 1
    assert json.loads(markers[0].read_text())["error_class"] == "magpie_nonzero_after_valid_measurement"


@pytest.mark.asyncio
async def test_server_dead_surfaces_log_excerpt(tmp_path, monkeypatch):
    """server_log_death_excerpt is used when a seeded server.log exists."""
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)
    out_root = tmp_path / "out"

    def _dead(magpie_python, config_path, output_dir, **_k):
        slog = Path(output_dir) / "server.log"
        slog.parent.mkdir(parents=True, exist_ok=True)
        slog.write_text(
            "Worker init started\n"
            "mla_gluon[bh16bn128] requires batch_size=1, got 512\n"
            "Engine core initialization failed.\n"
            "Traceback follows\n"
        )
        return gr.SERVER_DEAD_RETURNCODE, "", ""

    monkeypatch.setattr(gr, "_run_magpie", _dead)
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("fp8_kv")],
        output_root=out_root,
        variant_timeout_sec=5,
    )
    r = results[0]
    assert r.status == "failed"
    assert r.error_class == "server_init_dead"
    assert "mla_gluon" in (r.error or ""), "excerpt should mention mla_gluon"
    assert r.server_log_path is not None
    assert r.server_log_path.endswith("server.log")
