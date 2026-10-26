# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted ``events`` implementation: CloudEvents lines in rolling segments, one writer, best effort."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hyperloom.hosted import events as hosted_events
from hyperloom.hosted.events import EVENT_MAX_BYTES, SEGMENT_MAX_BYTES, HostedProgressEvents, segment_name
from hyperloom.hosted.host_info import Var, take_host_info
from hyperloom.plugs import ManifestSources, PlugContext, PlugHost, builtin_registry, resolve_manifest

_REQUIRED = ("specversion", "id", "source", "type", "time", "datacontenttype", "data")


def _lines(directory: Path) -> list[bytes]:
    out: list[bytes] = []
    for path in sorted(directory.glob("*.jsonl")):
        raw = path.read_bytes()
        assert raw == b"" or raw.endswith(b"\n"), f"{path.name} does not end with a newline"
        out += raw.splitlines()
    return out


def _events(directory: Path) -> list[dict]:
    return [json.loads(line) for line in _lines(directory)]


def _ids(directory: Path) -> list[int]:
    return [int(e["id"]) for e in _events(directory)]


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_every_event_carries_the_required_cloudevents_attributes(tmp_path):
    sink = HostedProgressEvents(tmp_path, "run-7")
    schema = json.loads(
        (Path(hosted_events.__file__).parents[1] / "contracts" / "hyperloom_events.v1.schema.json").read_text()
    )
    types = sorted(schema["events"])
    for event_type in types:
        sink.emit(event_type, {"k": "v"})
    sink.close()

    events = _events(tmp_path)
    assert [e["type"] for e in events] == types
    for event in events:
        assert all(name in event for name in _REQUIRED)
        assert event["specversion"] == "1.0"
        assert event["source"] == "hyperloom/run-7"
        assert event["datacontenttype"] == "application/json"
        assert event["id"].isdecimal()
        assert event["time"].endswith("Z")
    assert (tmp_path / "0000000001.jsonl").exists()


def test_ids_continue_across_a_restart_and_a_half_written_last_line_is_dropped(tmp_path):
    first = HostedProgressEvents(tmp_path, "r")
    for n in range(3):
        first.emit("hyperloom.warning", {"code": "c", "message": str(n)})
    first.close()
    segment = tmp_path / segment_name(1)
    with open(segment, "ab") as handle:  # a crash in the middle of event 4
        handle.write(b'{"specversion":"1.0","id":"4","sour')

    second = HostedProgressEvents(tmp_path, "r")
    second.emit("hyperloom.warning", {"code": "c", "message": "after"})
    second.close()

    assert _ids(tmp_path) == [1, 2, 3, 4]
    assert _events(tmp_path)[-1]["data"]["message"] == "after"


def test_a_restart_continues_from_the_previous_segment_when_the_last_one_is_empty(tmp_path):
    first = HostedProgressEvents(tmp_path, "r")
    first.emit("hyperloom.warning", {"code": "c", "message": "m"})
    first.close()
    (tmp_path / segment_name(2)).write_bytes(b"")

    second = HostedProgressEvents(tmp_path, "r")
    second.emit("hyperloom.warning", {"code": "c", "message": "m"})
    second.close()

    assert (tmp_path / segment_name(2)).read_bytes().count(b"\n") == 1
    assert _ids(tmp_path) == [1, 2]


def test_a_segment_rolls_before_it_passes_four_mib(tmp_path):
    sink = HostedProgressEvents(tmp_path, "r")
    blob = "x" * 12_000
    count = 400
    for _ in range(count):
        sink.emit("hyperloom.warning", {"code": "c", "message": blob})
    sink.close()

    segments = sorted(tmp_path.glob("*.jsonl"))
    assert [p.name for p in segments] == [segment_name(n) for n in range(1, len(segments) + 1)]
    assert len(segments) >= 2
    assert all(p.stat().st_size <= SEGMENT_MAX_BYTES for p in segments)
    assert _ids(tmp_path) == list(range(1, count + 1))


def test_threads_get_unique_increasing_ids_in_file_order(tmp_path):
    sink = HostedProgressEvents(tmp_path, "r")

    def work() -> None:
        for _ in range(150):
            sink.emit("hyperloom.warning", {"code": "c", "message": "m"})

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    sink.close()

    assert _ids(tmp_path) == list(range(1, 1201))


def test_an_oversized_event_is_sent_without_its_content_and_warns(tmp_path, caplog):
    sink = HostedProgressEvents(tmp_path, "r")
    with caplog.at_level(logging.WARNING, logger="hyperloom.hosted.events"):
        sink.emit("hyperloom.warning", {"code": "c", "message": "y" * (EVENT_MAX_BYTES * 2)})
    sink.emit("hyperloom.warning", {"code": "c", "message": "small"})
    sink.close()

    lines = _lines(tmp_path)
    assert all(len(line) + 1 <= EVENT_MAX_BYTES for line in lines)
    first = json.loads(lines[0])
    assert first["data"]["oversized"] is True and first["data"]["keys"] == ["code", "message"]
    assert [int(json.loads(line)["id"]) for line in lines] == [1, 2]
    assert any("over the" in r.getMessage() for r in caplog.records)


def test_fsync_happens_at_most_once_a_second_and_on_flush_and_close(tmp_path, monkeypatch):
    synced: list[int] = []
    real = os.fsync
    monkeypatch.setattr(hosted_events.os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    clock = _Clock()
    sink = HostedProgressEvents(tmp_path, "r", clock=clock)

    for _ in range(50):
        sink.emit("hyperloom.warning", {"code": "c", "message": "m"})
    assert synced == []
    clock.now += 1.0
    sink.emit("hyperloom.warning", {"code": "c", "message": "m"})
    assert len(synced) == 1
    for _ in range(50):
        clock.now += 0.01
        sink.emit("hyperloom.warning", {"code": "c", "message": "m"})
    assert len(synced) == 1
    sink.flush()
    assert len(synced) == 2
    sink.flush()
    assert len(synced) == 2
    sink.emit("hyperloom.warning", {"code": "c", "message": "m"})
    sink.close()
    assert len(synced) == 3


def test_a_failed_write_is_logged_not_raised_and_leaves_no_partial_line(tmp_path, monkeypatch, caplog):
    sink = HostedProgressEvents(tmp_path, "r")
    sink.emit("hyperloom.warning", {"code": "c", "message": "one"})
    real_write = os.write

    def half_write(fd, data):
        real_write(fd, bytes(data[: len(data) // 2]))
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(hosted_events.os, "write", half_write)
    with caplog.at_level(logging.ERROR, logger="hyperloom.hosted.events"):
        sink.emit("hyperloom.warning", {"code": "c", "message": "lost"})
    monkeypatch.setattr(hosted_events.os, "write", real_write)
    sink.emit("hyperloom.warning", {"code": "c", "message": "two"})
    sink.close()

    assert [e["data"]["message"] for e in _events(tmp_path)] == ["one", "two"]
    assert _ids(tmp_path) == [1, 2]
    assert any("could not write" in r.getMessage() for r in caplog.records)


def test_a_failed_segment_roll_is_retried_by_the_next_emit(tmp_path, monkeypatch):
    sink = HostedProgressEvents(tmp_path, "r", segment_max_bytes=1)  # every event after the first needs a new segment
    sink.emit("hyperloom.warning", {"code": "c", "message": "one"})
    real_open = os.open
    failures = [OSError(24, "Too many open files")]

    def flaky_open(path, *args, **kwargs):
        if failures:
            raise failures.pop()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(hosted_events.os, "open", flaky_open)
    sink.emit("hyperloom.warning", {"code": "c", "message": "lost"})
    sink.emit("hyperloom.warning", {"code": "c", "message": "two"})
    sink.close()

    assert [e["data"]["message"] for e in _events(tmp_path)] == ["one", "two"]
    assert _ids(tmp_path) == [1, 2]
    assert sorted(p.name for p in tmp_path.glob("*.jsonl")) == [segment_name(1), segment_name(2)]
    assert [len(p.read_bytes().splitlines()) for p in sorted(tmp_path.glob("*.jsonl"))] == [1, 1]


def test_emit_after_close_is_ignored(tmp_path):
    sink = HostedProgressEvents(tmp_path, "r")
    sink.close()
    sink.emit("hyperloom.warning", {"code": "c", "message": "m"})
    assert _lines(tmp_path) == []


def _host_env(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "state").mkdir()
    (tmp_path / "campaign.json").write_text("{}")
    for name, value in {
        Var.INPUT_FILE: str(tmp_path / "campaign.json"),
        Var.RUN_ID: "run-child",
        Var.RUN_CREATED_AT: "2026-10-01T12:30:00Z",
        Var.STATE_DIR: str(tmp_path / "state"),
        Var.EVENTS_DIR: str(tmp_path / "events"),
        Var.MCP_PORT: "8765",
        Var.MCP_TOKEN: "tok",
    }.items():
        monkeypatch.setenv(name, value)


def _hosted_host(tmp_path: Path, monkeypatch) -> PlugHost:
    _host_env(tmp_path, monkeypatch)
    info = take_host_info(os.environ)
    registry = builtin_registry()
    manifest = resolve_manifest(registry, ManifestSources(cli_plugs=("events=hosted",)))
    host = PlugHost(manifest, registry, session_dir=None, host_info=info)
    host.start()
    return host


def test_a_child_started_under_a_hosted_parent_has_no_events_directory_variable(tmp_path, monkeypatch):
    host = _hosted_host(tmp_path, monkeypatch)
    probe = "import os; print(repr(os.environ.get('HYPERLOOM_EVENTS_DIR')))"
    child = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=60)
    host.close()

    assert child.stdout.strip() == "None"
    assert Var.EVENTS_DIR not in os.environ


def test_the_directory_comes_from_the_host_inputs_not_from_the_environment(tmp_path, monkeypatch):
    _host_env(tmp_path, monkeypatch)
    info = take_host_info(os.environ)
    decoy = tmp_path / "decoy"
    monkeypatch.setenv(Var.EVENTS_DIR, str(decoy))
    registry = builtin_registry()
    host = PlugHost(
        resolve_manifest(registry, ManifestSources(cli_plugs=("events=hosted",))),
        registry,
        session_dir=None,
        host_info=info,
    )
    host.start()
    host.get("events").emit("hyperloom.warning", {"code": "c", "message": "m"})  # type: ignore[attr-defined]
    host.close()

    assert _ids(tmp_path / "events") == [1]
    assert not decoy.exists()


def test_the_hosted_implementation_refuses_to_start_without_host_inputs():
    with pytest.raises(ValueError, match="host inputs"):
        hosted_events.create(PlugContext("events", {}, None, None, lambda name: None))
