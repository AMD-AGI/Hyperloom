# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import time

from hyperloom.measure.runner.service import EXIT_CODES, Progress, RunnerFailure, wait_for_file, write_status


def test_status_is_written_atomically_with_its_fields(tmp_path):
    write_status(tmp_path, state="failed", error_class="spec_rejected", message="m")

    assert json.loads((tmp_path / "status.json").read_text()) == {
        "state": "failed",
        "error_class": "spec_rejected",
        "message": "m",
    }
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]


def test_wait_for_file_returns_true_when_it_appears_and_false_on_timeout(tmp_path):
    target = tmp_path / "READY"
    assert wait_for_file(target, 0.05, poll_sec=0.01) is False
    target.write_text("")
    assert wait_for_file(target, 0.05, poll_sec=0.01) is True


def test_failures_map_to_the_documented_exit_codes():
    assert {k: RunnerFailure(k, "m").exit_code for k in EXIT_CODES} == EXIT_CODES
    assert RunnerFailure("something_else", "m").exit_code == 1
    assert EXIT_CODES["preflight_mismatch"] == 4 and EXIT_CODES["node_fault"] == 5


def test_progress_records_phase_and_last_output_and_refreshes_in_the_background(tmp_path):
    with Progress(tmp_path, refresh_sec=0.02) as progress:
        progress.phase("running")
        before = json.loads((tmp_path / "progress.json").read_text())
        assert before["phase"] == "running" and before["last_output_unix"] is None
        progress.touch()
        time.sleep(0.15)
        after = json.loads((tmp_path / "progress.json").read_text())

    assert after["last_output_unix"] is not None
    assert after["updated_unix"] > before["updated_unix"]
