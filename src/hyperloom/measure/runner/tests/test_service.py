# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

from hyperloom.measure import failures
from hyperloom.measure.runner.service import (
    EXIT_CODES,
    Progress,
    RunnerFailure,
    join_notes,
    wait_for_file,
    write_status,
)


def test_status_is_written_atomically_with_its_fields(tmp_path):
    write_status(tmp_path, state="failed", error_class="spec_rejected", message="m")

    assert json.loads((tmp_path / "status.json").read_text()) == {
        "state": "failed",
        "error_class": "spec_rejected",
        "message": "m",
    }
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]


_RUNNER = Path(__file__).resolve().parents[1]
_RAISES = frozenset({"RunnerFailure", "OutputRejected"})


def _callee(call: ast.Call) -> str:
    return getattr(call.func, "id", None) or getattr(call.func, "attr", "")


def _class_expressions(node: ast.AST) -> list[ast.expr]:
    """The expressions an error class is written from: what a runner failure is raised with, what ``status.json`` gets,
    and what a local ``error_class`` is set to."""
    if isinstance(node, ast.Call) and _callee(node) in _RAISES and node.args:
        return [node.args[0]]
    if isinstance(node, ast.Call) and _callee(node) == "write_status":
        return [keyword.value for keyword in node.keywords if keyword.arg == "error_class"]
    if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "error_class" for target in node.targets):
        return [node.value]
    return []


def _strings(expr: ast.expr, constants: dict[str, str]) -> set[str]:
    if isinstance(expr, ast.IfExp):
        return _strings(expr.body, constants) | _strings(expr.orelse, constants)
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return {expr.value}
    if isinstance(expr, ast.Name):
        value = constants.get(expr.id, getattr(failures, expr.id, None))
        return {value} if isinstance(value, str) else set()
    return set()


def _written_error_classes() -> set[str]:
    written: set[str] = set()
    for path in _RUNNER.rglob("*.py"):
        if "tests" in path.relative_to(_RUNNER).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(tree):
            for expr in _class_expressions(node):
                written |= _strings(expr, constants)
    return written


def test_every_error_class_the_runner_writes_is_one_the_classifier_reads():
    """The classifier takes a class it does not list for a framework defect (``unknown_error_class``), so every class a
    runner module raises or writes to ``status.json`` is listed as infrastructure or as a program error."""
    written = _written_error_classes()

    assert {"no_spec", "spec_rejected", "unsafe_output", "output_too_large", "runner_error", "disk_full"} <= written
    assert written - failures.INFRA_ERROR_CLASSES - failures.PROGRAM_ERROR_CLASSES == set()


def test_notes_that_fit_are_joined_as_they_are():
    assert join_notes(["a", "b", "c"]) == "a; b; c" and join_notes([]) == ""


def test_notes_that_do_not_fit_are_cut_at_their_end_so_the_first_and_most_important_are_kept(tmp_path):
    notes = ["jit cache: the cache failed verification; ignored", *(f"note {n:03d} " + "x" * 100 for n in range(40))]

    message = join_notes(notes)

    assert len(message) <= 2000 and message.startswith(notes[0] + "; note 000 ")
    kept = message.count("note 0")
    assert 0 < kept < 40 and message.endswith(f"; {40 - kept} more notes left out")
    write_status(tmp_path, state="succeeded", message=message)
    assert json.loads((tmp_path / "status.json").read_text())["message"] == message, "and the status keeps all of it"


def test_a_first_note_that_is_longer_than_the_message_may_be_keeps_its_head_so_status_does_not_cut_its_start(tmp_path):
    message = join_notes(["z" * 2500, "second"])

    assert len(message) <= 2000 and message.startswith("z" * 1900) and message.endswith("; 1 more notes left out")
    write_status(tmp_path, state="succeeded", message=message)
    assert json.loads((tmp_path / "status.json").read_text())["message"] == message


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
