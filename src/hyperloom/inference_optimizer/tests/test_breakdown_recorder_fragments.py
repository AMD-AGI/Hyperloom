# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fragment identity: which file a recorded item lands in, and the sequence number it carries."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.inference_optimizer.breakdown.recorder.recorder import Recorder


def test_item_filename_seq_matches_envelope_seq(tmp_path: Path) -> None:
    """A keyless item spends one sequence number, not two."""
    rec = Recorder(tmp_path, producer="coordinator")

    path = rec.record_item("measurements", {"value": 1})

    envelope = json.loads(path.read_text(encoding="utf-8"))
    filename_seq = int(path.stem.rsplit("-", 1)[-1])
    assert filename_seq == envelope["seq"]


def test_sequence_numbers_stay_monotonic_across_item_kinds(tmp_path: Path) -> None:
    """Sharing a number must not break the ordering assembler sorts on."""
    rec = Recorder(tmp_path, producer="coordinator")

    first = rec.record_item("measurements", {"value": 1})
    second = rec.record_item("measurements", {"value": 2}, key="keyed")
    third = rec.record_item("measurements", {"value": 3})

    seqs = [json.loads(p.read_text(encoding="utf-8"))["seq"] for p in (first, second, third)]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs), "sequence numbers must stay unique"


def test_keys_that_slugify_alike_get_distinct_fragments(tmp_path: Path) -> None:
    """``ck#1`` and ``ck-1`` are different kernels and need different files."""
    rec = Recorder(tmp_path, producer="coordinator")

    first = rec.record_item("measurements", {"value": 1}, key="ck#1")
    second = rec.record_item("measurements", {"value": 2}, key="ck-1")

    assert first != second, "sanitizing must not fold two keys onto one file"
    assert json.loads(first.read_text(encoding="utf-8"))["payload"]["value"] == 1
    assert json.loads(second.read_text(encoding="utf-8"))["payload"]["value"] == 2


def test_same_key_still_rewrites_one_fragment(tmp_path: Path) -> None:
    """Idempotence across retries is the whole point of a stable key."""
    rec = Recorder(tmp_path, producer="coordinator")

    first = rec.record_item("measurements", {"value": 1}, key="k1")
    second = rec.record_item("measurements", {"value": 2}, key="k1")

    assert first == second
    assert len(list(tmp_path.glob("measurements__*.json"))) == 1


def test_legacy_reuse_does_not_resurrect_the_collision(tmp_path: Path) -> None:
    """A legacy filename only belongs to a key that sanitizing left untouched."""
    rec = Recorder(tmp_path, producer="coordinator")
    legacy = tmp_path / "measurements__coordinator__a-b.json"
    legacy.write_text(
        json.dumps({"section": "measurements", "kind": "item", "seq": 1, "payload": {"who": "a/b"}}),
        encoding="utf-8",
    )

    written = rec.record_upsert_item("measurements", {"who": "a:b"}, key="a:b")

    assert written != legacy, "a sanitized key must not claim an ambiguous legacy file"
    assert json.loads(legacy.read_text(encoding="utf-8"))["payload"]["who"] == "a/b", "legacy left intact"
    assert json.loads(written.read_text(encoding="utf-8"))["payload"]["who"] == "a:b"


def test_fragment_written_under_the_old_name_keeps_that_name(tmp_path: Path) -> None:
    """A resumed session must update its fragment, not fork a second one."""
    rec = Recorder(tmp_path, producer="coordinator")
    legacy = tmp_path / "measurements__coordinator__k1.json"
    legacy.write_text(
        json.dumps({"section": "measurements", "kind": "item", "seq": 1, "payload": {"value": 1}}),
        encoding="utf-8",
    )

    written = rec.record_upsert_item("measurements", {"value": 2}, key="k1")

    assert written == legacy
    assert len(list(tmp_path.glob("measurements__*.json"))) == 1, "resume must not duplicate the fragment"
    assert json.loads(written.read_text(encoding="utf-8"))["payload"]["value"] == 2
