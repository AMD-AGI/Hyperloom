# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``_reconcile_crash_count`` in ``cli.py``."""

from __future__ import annotations

import json

from inference_optimizer.cli import _reconcile_crash_count
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.session_paths import reports_dir


def test_bumps_stale_state_json_up_to_live_value(tmp_path):
    # Disk recorded only 0; the live coordinator object saw 3 crashes.
    on_disk = SharedState(session_id="s", crash_count=0)
    on_disk.save(tmp_path)

    live = SharedState(session_id="s", crash_count=3)
    _reconcile_crash_count(live, tmp_path)

    reloaded = SharedState.load_or_init(tmp_path)
    assert reloaded.crash_count == 3


def test_patches_final_json_in_place(tmp_path):
    rdir = reports_dir(tmp_path)
    rdir.mkdir(parents=True, exist_ok=True)
    final_json = rdir / "final.json"
    final_json.write_text(
        json.dumps({"crash_count": 0, "stop_reason": "time_exhausted"}),
        encoding="utf-8",
    )
    SharedState(session_id="s", crash_count=0).save(tmp_path)

    live = SharedState(session_id="s", crash_count=2)
    _reconcile_crash_count(live, tmp_path)

    data = json.loads(final_json.read_text(encoding="utf-8"))
    assert data["crash_count"] == 2
    # Unrelated fields are preserved.
    assert data["stop_reason"] == "time_exhausted"


def test_never_lowers_a_higher_disk_count(tmp_path):
    # Disk somehow recorded more than memory — never regress it.
    SharedState(session_id="s", crash_count=5).save(tmp_path)

    live = SharedState(session_id="s", crash_count=2)
    _reconcile_crash_count(live, tmp_path)

    reloaded = SharedState.load_or_init(tmp_path)
    assert reloaded.crash_count == 5


def test_no_final_json_is_non_fatal(tmp_path):
    SharedState(session_id="s", crash_count=0).save(tmp_path)
    live = SharedState(session_id="s", crash_count=1)
    # No reports/final.json on disk — must not raise.
    _reconcile_crash_count(live, tmp_path)
    assert SharedState.load_or_init(tmp_path).crash_count == 1
