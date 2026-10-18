# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The directory layer: segments merged across writers, and claim generations that one writer at a time wins."""

from __future__ import annotations

import json
import logging
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, ClaimLost, Role, Writer
from hyperloom_multibranch.contracts.notebook.records import Claim, ClaimKey, LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus, LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.local import claims as claims_module
from hyperloom_multibranch.notebook.local.claims import LeaseStore, LeaseTaken
from hyperloom_multibranch.notebook.local.notebook import SEGMENTS_DIR
from hyperloom_multibranch.notebook.local.segments import Entry, EntryKind, SegmentReader, SegmentWriter, make_writer_id
from hyperloom_multibranch.notebook.tests.factories import START, intent, request

KEY = ClaimKey("c0", intent("s1", "c0").fingerprint)


def _entry(writer: str, at: float) -> Callable[[int], Entry]:
    def build(seq: int) -> Entry:
        return Entry(
            f"{writer}:{seq}", seq, at, writer, Role.TOOLBOX, "toolbox", EntryKind.TOOL_RESULT, "s1", "", "x", {}
        )

    return build


def _claim(holder: str, expires_at: float = START + 60) -> Claim:
    return Claim(KEY, holder, intent(holder, "c0"), "a" * 64, 60.0, START, START, expires_at, ClaimStatus.HELD)


def _try_acquire(claims_dir: str, holder: str) -> bool:
    try:
        LeaseStore(Path(claims_dir)).acquire(_claim(holder), START)
    except LeaseTaken:
        return False
    return True


def test_readers_merge_every_writer_in_time_order(tmp_path: Path) -> None:
    first, second = SegmentWriter(tmp_path, "a.h.1.0000"), SegmentWriter(tmp_path, "b.h.2.0000")
    first.append(_entry(first.writer, START + 1))
    second.append(_entry(second.writer, START + 2))
    first.append(_entry(first.writer, START + 3))

    merged = SegmentReader(tmp_path).entries()

    assert [entry.entry_id for entry in merged] == ["a.h.1.0000:1", "b.h.2.0000:1", "a.h.1.0000:2"]


def test_a_line_still_being_written_is_skipped_until_it_is_whole(tmp_path: Path) -> None:
    writer = SegmentWriter(tmp_path, "a.h.1.0000")
    writer.append(_entry(writer.writer, START + 1))
    reader = SegmentReader(tmp_path)
    line = json.dumps(_entry(writer.writer, START + 2)(2).to_dict(), sort_keys=True)
    with writer.path.open("a", encoding="utf-8") as handle:
        handle.write(line[:40])

    assert len(reader.entries()) == 1

    with writer.path.open("a", encoding="utf-8") as handle:
        handle.write(line[40:] + "\n")
    assert [entry.seq for entry in reader.entries()] == [1, 2]


def test_a_reopened_writer_continues_its_sequence_and_refuses_another_writers_entry(tmp_path: Path) -> None:
    SegmentWriter(tmp_path, "a.h.1.0000").append(_entry("a.h.1.0000", START + 1))
    reopened = SegmentWriter(tmp_path, "a.h.1.0000")
    appended = reopened.append(_entry("a.h.1.0000", START + 2))

    assert appended.seq == 2
    with pytest.raises(ValueError):
        reopened.append(_entry("b.h.2.0000", START + 3))


def test_a_row_this_version_cannot_read_is_skipped_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    writer = SegmentWriter(tmp_path, "a.h.1.0000")
    writer.append(_entry(writer.writer, START + 1))
    row = {**_entry(writer.writer, START + 2)(2).to_dict(), "kind": "from_the_future"}
    with writer.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    with caplog.at_level(logging.WARNING):
        entries = SegmentReader(tmp_path).entries()

    assert [entry.seq for entry in entries] == [1]
    assert "from_the_future" in caplog.text


async def test_an_entry_the_notebook_cannot_decode_is_skipped_and_the_rest_still_reads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    notebook = LocalNotebook(tmp_path, Writer(Role.TOOLBOX, "toolbox"))
    await notebook.write_request(request("r1", intent("s1", "c0")))
    stray = SegmentWriter(tmp_path / SEGMENTS_DIR, "toolbox.old.1.0000")
    stray.append(
        lambda seq: replace(_entry(stray.writer, START)(seq), kind=EntryKind.REQUEST, detail={"request_id": 7})
    )

    with caplog.at_level(logging.WARNING):
        records = notebook._state().records()

    assert [record.request_id for record in records] == ["r1"]
    assert "undecodable request entry" in caplog.text


async def test_a_branch_head_without_a_recorded_value_has_an_empty_one(tmp_path: Path) -> None:
    scheduler = LocalNotebook(tmp_path, Writer(Role.SCHEDULER, "scheduler"))
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s1", from_commit="c0"))

    progress = await scheduler.progress()

    assert [(row.branch, dict(row.head_value)) for row in progress] == [(INTEGRATION_BRANCH, {}), ("s1", {})]


def test_writer_ids_are_unique_per_handle_and_name_the_role() -> None:
    first, second = make_writer_id(Role.GATE, "gate"), make_writer_id(Role.GATE, "gate")

    assert first != second
    assert first.startswith("gate.gate.")


def test_a_live_claim_blocks_the_next_generation_until_it_expires(tmp_path: Path) -> None:
    store = LeaseStore(tmp_path)
    held = store.acquire(_claim("s1"), START)

    with pytest.raises(LeaseTaken) as blocked:
        store.acquire(_claim("s2"), START)
    taken = store.acquire(_claim("s2", expires_at=START + 200), START + 100)

    assert blocked.value.claim.holder == "s1"
    assert (held.generation, taken.generation) == (1, 2)
    assert [claim.holder for claim in store.all_current()] == ["s2"]


def test_a_change_beaten_by_another_writer_is_reapplied_to_the_newer_generation(tmp_path: Path) -> None:
    store = LeaseStore(tmp_path)
    store.acquire(_claim("s1"), START)
    calls = []

    def renew(claim: Claim) -> Claim:
        calls.append(claim.generation)
        if len(calls) == 1:
            store.change(KEY, lambda current: replace(current, request_id="r1"))
        return replace(claim, expires_at=claim.expires_at + 60)

    changed = store.change(KEY, renew)

    assert calls == [1, 2]
    assert (changed.generation, changed.request_id) == (3, "r1")


def test_a_change_gives_up_when_it_never_wins_and_one_on_no_claim_is_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LeaseStore(tmp_path)

    with pytest.raises(ClaimLost, match="nobody claimed"):
        store.change(KEY, lambda claim: claim)
    store.acquire(_claim("s1"), START)
    monkeypatch.setattr(claims_module, "publish_once", lambda path, data: False)
    with pytest.raises(ClaimLost, match="kept changing"):
        store.change(KEY, lambda claim: claim)


def test_only_one_of_many_racing_processes_wins_a_claim(tmp_path: Path) -> None:
    holders = [f"s{index}" for index in range(6)]
    with ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context("spawn")) as pool:
        wins = list(pool.map(_try_acquire, [str(tmp_path)] * len(holders), holders))

    assert sum(wins) == 1
    current = LeaseStore(tmp_path).current(KEY)
    assert current is not None and current.holder == holders[wins.index(True)]
