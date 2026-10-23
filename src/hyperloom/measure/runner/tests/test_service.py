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
        # Wait for the background refresh rather than a fixed sleep, which a loaded CI runner can outlast.
        deadline = time.monotonic() + 10
        after = before
        while after["last_output_unix"] is None and time.monotonic() < deadline:
            time.sleep(0.02)
            after = json.loads((tmp_path / "progress.json").read_text())

    assert after["last_output_unix"] is not None
    assert after["updated_unix"] > before["updated_unix"]


def test_progress_reports_the_last_line_of_the_newest_live_log_below_the_files_directory(tmp_path):
    files = tmp_path / "files"
    log = files / "round" / "server.log"
    log.parent.mkdir(parents=True)
    out = tmp_path
    with Progress(out, refresh_sec=3600, files_dir=files) as progress:
        assert "last_line" not in json.loads((out / "progress.json").read_text()), "no log is written yet"
        log.write_text("booting\n\x1b[32mdecode batch 8\x1b[0m\n")
        progress.phase("running")
        document = json.loads((out / "progress.json").read_text())
    assert document["last_line"] == "decode batch 8"
    assert 0.0 <= document["last_log_age_sec"] < 60.0
    assert document["phase"] == "running" and document["updated_unix"] > 0


def test_progress_without_a_files_directory_reports_no_line(tmp_path):
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "server.log").write_text("a line\n")
    with Progress(tmp_path, refresh_sec=3600) as progress:
        progress.phase("running")
        document = json.loads((tmp_path / "progress.json").read_text())
    assert "last_line" not in document and "last_log_age_sec" not in document


def test_two_writers_of_one_file_never_share_a_scratch_file(tmp_path, monkeypatch):
    """A second write that happens between the first's write and its rename must not take the first's scratch away."""
    import os

    from hyperloom.measure.runner import service

    target = tmp_path / "progress.json"
    real_replace = os.replace
    nested = []

    def replace_after_a_second_writer_ran(source, destination):
        if not nested:
            nested.append(True)
            service.write_json_atomic(target, {"writer": "second"})
        real_replace(source, destination)

    monkeypatch.setattr(service.os, "replace", replace_after_a_second_writer_ran)

    service.write_json_atomic(target, {"writer": "first"})

    assert json.loads(target.read_text()) == {"writer": "first"}
    assert [p.name for p in tmp_path.iterdir()] == ["progress.json"]


def test_a_reader_polling_while_writers_flush_always_gets_a_whole_document(tmp_path):
    import threading

    from hyperloom.measure.runner.service import write_json_atomic

    target = tmp_path / "progress.json"
    write_json_atomic(target, {"n": 0})
    stop = threading.Event()
    errors: list[Exception] = []

    def writer(tag: int) -> None:
        count = 0
        while not stop.is_set():
            count += 1
            try:
                write_json_atomic(target, {"n": count, "tag": tag, "pad": "x" * 2000})
            except Exception as exc:  # noqa: BLE001 - a failed write is as much a finding as a torn read
                errors.append(exc)
                return

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    try:
        for _ in range(1500):
            try:
                assert "n" in json.loads(target.read_text())
            except Exception as exc:  # noqa: BLE001 - every failure to read a whole document is the finding
                errors.append(exc)
    finally:
        stop.set()
        for thread in threads:
            thread.join()

    assert errors == []
    assert [p.name for p in tmp_path.iterdir()] == ["progress.json"]
