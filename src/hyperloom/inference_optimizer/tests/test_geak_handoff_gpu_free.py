# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK is handed its GPUs only once this session's leftover servers are gone and the cards hold no VRAM."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common.amd_smi import GpuVram
from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl
from hyperloom.orchestrator.actions.executors._gpu_pin import (
    _resolve_handoff_gpu_ids,
    _resolve_handoff_gpu_ids_space,
    _resolve_handoff_physical_gpus,
    resolve_gpu_pin,
)

IDLE = GpuVram(used_mib=300.0, total_mib=288000.0)
HELD = GpuVram(used_mib=281000.0, total_mib=288000.0)
HOLDER = "pgid=274559 python3 -m sglang.launch_server"


@pytest.fixture
def host(monkeypatch):
    """Script the VRAM readings; record the reap, the sleeps and the holder scan."""
    state = SimpleNamespace(readings=[], reaped=[], reap_calls=[], probes=0, sleeps=[])

    def _reap(session_dir):
        state.reap_calls.append(Path(session_dir))
        return list(state.reaped)

    def _probe():
        state.probes += 1
        return state.readings.pop(0) if len(state.readings) > 1 else state.readings[0]

    monkeypatch.setattr(sl, "reap_orphaned_servers", _reap)
    monkeypatch.setattr(sl.amd_smi, "gpu_vram_usage", _probe)
    monkeypatch.setattr(sl, "time", SimpleNamespace(monotonic=time.monotonic, sleep=state.sleeps.append))
    monkeypatch.setattr(sl, "_serving_process_groups", lambda: [HOLDER])
    return state


def test_reaps_first_then_hands_over_idle_cards(host, tmp_path):
    host.reaped = [274559]
    host.readings = [[IDLE] * 8]

    gpus = sl.free_gpus_for_handoff(tmp_path, list(range(8)), timeout_sec=60)

    assert gpus == sl.GpuHandoff(reaped=[274559], busy=[], holders=[], unverified="")
    assert host.reap_calls == [tmp_path]
    assert host.sleeps == []


def test_waits_for_a_reaped_server_vram_to_drain(host, tmp_path):
    host.readings = [[HELD] * 8, [HELD] * 8, [IDLE] * 8]

    gpus = sl.free_gpus_for_handoff(tmp_path, list(range(8)), timeout_sec=60)

    assert gpus.busy == []
    assert host.sleeps == [sl._GPU_FREE_POLL_SECONDS] * 2


def test_names_the_cards_still_held_at_the_deadline_and_who_holds_them(host, tmp_path):
    host.readings = [[HELD, IDLE]]

    gpus = sl.free_gpus_for_handoff(tmp_path, [0, 1], timeout_sec=0)

    assert gpus.busy == [(0, HELD.used_mib, HELD.total_mib)]
    assert gpus.holders == [HOLDER]
    assert gpus.unverified == ""


def test_only_the_named_cards_count(host, tmp_path):
    """A co-tenant on other cards of the node is not this hand-off's business."""
    host.readings = [[IDLE, IDLE, HELD, HELD]]

    assert sl.free_gpus_for_handoff(tmp_path, [0, 1], timeout_sec=0).busy == []


def test_unreadable_vram_still_reaps_but_verifies_nothing(host, tmp_path):
    host.readings = [None]

    gpus = sl.free_gpus_for_handoff(tmp_path, [0, 1], timeout_sec=60)

    assert (gpus.busy, gpus.unverified) == ([], "amd-smi could not report VRAM")
    assert host.reap_calls == [tmp_path]


def test_unresolved_cards_still_reap_but_verify_nothing(host, tmp_path):
    gpus = sl.free_gpus_for_handoff(tmp_path, None, timeout_sec=60)

    assert gpus.busy == []
    assert "resolved" in gpus.unverified
    assert host.reap_calls == [tmp_path]
    assert host.probes == 0


def test_a_card_amd_smi_does_not_report_is_unverified(host, tmp_path):
    host.readings = [[HELD, HELD]]

    gpus = sl.free_gpus_for_handoff(tmp_path, [0, 5], timeout_sec=0)

    assert gpus.busy == []
    assert "does not cover cards [0, 5]" in gpus.unverified


def _spawn_server_lookalike(*flags: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", "sglang.launch_server", *flags],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.time() + 5.0
    while "sglang.launch_server" not in sl._pid_cmdline(proc.pid):
        if time.time() >= deadline:
            raise AssertionError("lookalike never exec'd")
        time.sleep(0.02)
    return proc


def test_holders_name_each_serving_process_group_once_by_its_program(monkeypatch):
    monkeypatch.setattr(sl, "_HOLDERS_REPORTED", 10_000)
    server = _spawn_server_lookalike("--api-key", "hunter2", "--port", "30000")
    try:
        ours = [line for line in sl._serving_process_groups() if line.startswith(f"pgid={server.pid} ")]

        assert len(ours) == 1
        assert ours[0].endswith(" sglang.launch_server")
        assert "hunter2" not in ours[0]
    finally:
        server.kill()
        server.wait(timeout=5)


def test_holders_are_capped(monkeypatch):
    monkeypatch.setattr(sl, "_HOLDERS_REPORTED", 1)
    servers = [_spawn_server_lookalike() for _ in range(2)]
    try:
        assert len(sl._serving_process_groups()) == 1
    finally:
        for server in servers:
            server.kill()
            server.wait(timeout=5)


@pytest.mark.parametrize(
    ("environ", "tp", "cards"),
    [
        ({}, 8, [0, 1, 2, 3, 4, 5, 6, 7]),
        ({"HIP_VISIBLE_DEVICES": "4,5"}, 2, [4, 5]),
        ({"ROCR_VISIBLE_DEVICES": "4,5,6,7"}, 2, [4, 5]),
        ({"ROCR_VISIBLE_DEVICES": "4,5,6,7", "HIP_VISIBLE_DEVICES": "2,3"}, 2, [6, 7]),
        ({"ROCR_VISIBLE_DEVICES": "GPU-aa,GPU-bb"}, 2, None),
        ({"HIP_VISIBLE_DEVICES": "GPU-aa,GPU-bb"}, 2, None),
        ({"ROCR_VISIBLE_DEVICES": ""}, 2, None),
    ],
    ids=["unpinned", "hip_pin", "rocr_pin", "hip_inside_rocr", "rocr_uuid", "hip_uuid", "empty_mask"],
)
def test_physical_cards_follow_the_handoff_coordinates(environ, tp, cards):
    """The cards checked are the ones GEAK's servers land on, whichever coordinate system ``gpu_ids`` is in."""
    pin = resolve_gpu_pin(recipe_envs={}, environ=environ)
    gpu_ids = _resolve_handoff_gpu_ids(gpu_pin=pin, tp=tp)
    space = _resolve_handoff_gpu_ids_space(gpu_pin=pin)

    assert _resolve_handoff_physical_gpus(gpu_ids=gpu_ids, gpu_ids_space=space, gpu_pin=pin) == cards
