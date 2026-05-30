"""Tests for :class:`runtime.dead_letter.DeadLetter`."""

from __future__ import annotations

import json

import pytest

from hyperloom.agents.critic.runtime.dead_letter import DeadLetter
from hyperloom.agents.critic.runtime.errors import RuntimeAdapterError


def test_append_and_files_round_trip(tmp_path):
    dlq = DeadLetter(root=tmp_path)
    p = dlq.append(
        "upsert",
        {"scope": {}, "kind": "pitfall", "slug": "x"},
        attempts=2,
        last_error="503",
    )
    assert p.exists()
    files = dlq.files()
    assert files == [p]
    record = json.loads(p.read_text("utf-8").splitlines()[0])
    assert record["endpoint"] == "upsert"
    assert record["attempts"] == 2


def test_append_rejects_bad_endpoint(tmp_path):
    dlq = DeadLetter(root=tmp_path)
    with pytest.raises(RuntimeAdapterError):
        dlq.append("up/sert", {}, attempts=1, last_error="x")


def test_replay_succeeds_removes_lines(tmp_path):
    dlq = DeadLetter(root=tmp_path)
    dlq.append("upsert", {"k": 1}, attempts=1, last_error="x")
    dlq.append("upsert", {"k": 2}, attempts=1, last_error="x")

    seen = []

    def dispatcher(endpoint, payload):
        seen.append((endpoint, payload))

    summary = dlq.replay(dispatcher)
    assert summary.scanned == 2
    assert summary.succeeded == 2
    assert summary.failed == 0
    assert dlq.files() == []
    assert [s[0] for s in seen] == ["upsert", "upsert"]


def test_replay_keeps_failed_rows(tmp_path):
    dlq = DeadLetter(root=tmp_path)
    dlq.append("upsert", {"k": 1}, attempts=1, last_error="x")
    dlq.append("upsert", {"k": 2}, attempts=1, last_error="x")

    def dispatcher(endpoint, payload):
        if payload["k"] == 1:
            raise RuntimeError("boom")

    summary = dlq.replay(dispatcher)
    assert summary.scanned == 2
    assert summary.succeeded == 1
    assert summary.failed == 1
    files = dlq.files()
    # Surviving file should still hold one record.
    assert len(files) == 1
    survivors = [json.loads(line) for line in files[0].read_text("utf-8").splitlines() if line.strip()]
    assert len(survivors) == 1
    assert survivors[0]["payload"]["k"] == 1


def test_replay_empty_dir_returns_zero_summary(tmp_path):
    dlq = DeadLetter(root=tmp_path / "empty")
    summary = dlq.replay(lambda *a, **k: None)
    assert summary.to_dict() == {
        "scanned": 0,
        "succeeded": 0,
        "failed": 0,
        "failed_details": [],
    }


def test_replay_handles_corrupt_lines_as_failures(tmp_path):
    dlq = DeadLetter(root=tmp_path)
    dlq.append("upsert", {"k": 1}, attempts=1, last_error="x")
    # Append a bad line manually.
    p = dlq.files()[0]
    with p.open("a", encoding="utf-8") as fp:
        fp.write("{not json\n")

    summary = dlq.replay(lambda *a, **k: None)
    assert summary.scanned == 2
    assert summary.failed == 1
