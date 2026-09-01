# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Extra unit tests for the candidate archive (loop/archive.py).

Complements test_archive_crash.py by covering the score helper, robust
read paths, and the prompt-digest layers (table capping, curated diffs,
truncation). Filesystem via tmp_path; no LLM / GPU."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kernelforge.loop.archive import CandidateArchive, CandidateRecord


# ── numeric helpers ────────────────────────────────────────────────────────────


def test_mean_case_speedup_delta_pct():
    assert CandidateArchive._mean_case_speedup_delta_pct(2.0, 1.0) == 100.0
    assert CandidateArchive._mean_case_speedup_delta_pct(None, 1.0) is None
    assert CandidateArchive._mean_case_speedup_delta_pct(2.0, None) is None


def test_fmt_num_handles_bad_values():
    assert CandidateArchive._fmt_num(1.2345, ".2f") == "1.23"
    assert CandidateArchive._fmt_num(None, ".2f") == "-"
    assert CandidateArchive._fmt_num("x", ".2f") == "-"
    assert CandidateArchive._fmt_num(1.0, ".2f", "x") == "1.00x"


def test_label_fallback():
    assert CandidateArchive._label("KEEP") == "KEEP*"
    assert CandidateArchive._label("UNKNOWN") == "UNKNOWN"
    assert CandidateArchive._label("") == "?"


# ── record + meta round-trip ───────────────────────────────────────────────────


def test_record_writes_all_files_and_meta(tmp_path):
    archive = CandidateArchive(str(tmp_path), kernel_file="flash.py")
    d = archive.record(
        CandidateRecord(
            iteration=1,
            commit_hash="abc",
            decision="KEEP",
            kept=True,
            validation_passed=True,
            wall_ms=1.0,
            mean_case_speedup=2.0,
            snr_db=40.0,
            baseline_wall_ms=2.0,
            best_wall_ms_before=2.0,
            best_mean_case_speedup_before=1.0,
            kernel_source="print(1)\n",
            change_diff="+ added\n",
            pmc_full="PMC SUMMARY",
            validation_text="validation ok",
            plan="vectorize",
        )
    )
    assert d is not None
    assert (d / "flash.py").read_text() == "print(1)\n"
    assert (d / "change.diff").exists()
    assert (d / "profile.txt").read_text() == "PMC SUMMARY"
    assert (d / "validation.txt").exists()
    meta = archive.load_meta(1)
    assert meta["decision"] == "KEEP"
    assert meta["mean_case_speedup"] == 2.0
    assert "speedup_vs_baseline" not in meta
    assert meta["files"]["kernel"] == "flash.py"


def test_default_kernel_basename(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    assert archive.kernel_basename == "kernel.py"


# ── robust read paths ──────────────────────────────────────────────────────────


def test_load_index_missing_is_empty(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    assert archive.load_index() == []


def test_read_index_file_skips_malformed_lines(tmp_path):
    # Under the resume design, candidate directories are the canonical source and
    # load_index() reconciles the on-disk index against them (orphaned index
    # entries without a complete dir are dropped). The malformed-line tolerance
    # now lives in the raw index reader, which skips non-JSON and blank lines.
    archive = CandidateArchive(str(tmp_path))
    archive.index_path.write_text('{"iter": 1}\nnot-json\n\n{"iter": 2}\n')
    entries = archive._read_index_file()
    assert [e["iter"] for e in entries] == [1, 2]


def test_load_meta_missing_is_empty(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    assert archive.load_meta(9) == {}


def test_read_candidate_file_missing_is_empty(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    assert archive.read_candidate_file(9, "change.diff") == ""


# ── digest ──────────────────────────────────────────────────────────────────────


def test_render_digest_empty_when_nothing_archived(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    assert archive.render_digest() == ""


def test_render_digest_basic_layers(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    archive.record(
        CandidateRecord(
            iteration=1,
            decision="KEEP",
            kept=True,
            wall_ms=1.0,
            mean_case_speedup=2.0,
            baseline_wall_ms=2.0,
            best_wall_ms_before=2.0,
            best_mean_case_speedup_before=1.0,
            change_diff="+ win\n",
            plan="vectorize",
        )
    )
    digest = archive.render_digest()
    assert "Solution archive" in digest
    assert "Trajectory (1 attempts" in digest
    assert "best mean case speedup=2.000000x" in digest
    assert "baseline=2.0000 ms" in digest
    assert "Notable prior solutions" in digest
    assert "2.0000x" in digest
    assert "+ win" in digest


def test_render_digest_diff_truncation(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    big_diff = "\n".join(f"+ line {i}" for i in range(200))
    archive.record(
        CandidateRecord(
            iteration=1,
            decision="KEEP",
            kept=True,
            wall_ms=1.0,
            baseline_wall_ms=2.0,
            best_wall_ms_before=2.0,
            change_diff=big_diff,
            plan="p",
        )
    )
    digest = archive.render_digest(max_diff_lines=10)
    assert "truncated to 10 lines" in digest


def test_render_digest_diff_unavailable(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    archive.record(
        CandidateRecord(
            iteration=1,
            decision="KEEP",
            kept=True,
            wall_ms=1.0,
            baseline_wall_ms=2.0,
            best_wall_ms_before=2.0,
            change_diff="",
            plan="no diff",
        )
    )
    digest = archive.render_digest()
    assert "diff unavailable" in digest


def test_render_digest_table_capping_keeps_and_recent(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    # One early KEEP + many REVERTs so the table must cap and keep the KEEP row.
    archive.record(
        CandidateRecord(
            iteration=1,
            decision="KEEP",
            kept=True,
            wall_ms=1.0,
            baseline_wall_ms=2.0,
            best_wall_ms_before=2.0,
            plan="the-keep",
        )
    )
    for i in range(2, 12):
        archive.record(
            CandidateRecord(
                iteration=i,
                decision="REVERT_PERF",
                kept=False,
                wall_ms=1.5,
                baseline_wall_ms=2.0,
                best_wall_ms_before=1.0,
                plan=f"revert-{i}",
            )
        )
    digest = archive.render_digest(max_table_rows=4)
    assert "older rows omitted" in digest
    assert "the-keep" in digest  # KEEP row is always retained
    assert "revert-11" in digest  # most recent retained


def test_select_for_diffs_prioritizes_keep_near_recent(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    index = [
        {"iter": 1, "decision": "KEEP", "wall_ms": 1.0},
        {"iter": 2, "decision": "REVERT_PERF", "wall_ms": 1.2},
        {"iter": 3, "decision": "REVERT_PERF", "wall_ms": 1.1},
        {"iter": 4, "decision": "REVERT_VALIDATION", "wall_ms": None},
        {"iter": 5, "decision": "REVERT_PERF", "wall_ms": 5.0},
    ]
    sel = archive._select_for_diffs(index, max_full_diffs=3, near_miss_count=2, recent_count=1)
    iters = [e["iter"] for e in sel]
    assert iters == sorted(iters)
    assert 1 in iters  # the KEEP
    assert len(iters) <= 3


# ── unusable archive root ───────────────────────────────────────────────────────


def test_unusable_root_degrades_instead_of_raising(tmp_path):
    # A file where forge_experiments/ should be: the archive must never take the
    # forge-loop down with it — it degrades, reports why, and reads as empty.
    (tmp_path / "forge_experiments").write_text("not a directory\n")

    archive = CandidateArchive(str(tmp_path))

    assert archive.degraded is True
    assert any("create" in err for err in archive.persistence_errors)
    # The change signature must still be computable (both components unknown),
    # otherwise every cache check would raise on a degraded archive.
    assert archive._fs_signature() == (None, None)
    assert archive.load_index() == []
    assert archive.max_iteration() == 0
    # A record on an unusable root fails closed rather than raising.
    assert archive.record(CandidateRecord(iteration=1, decision="KEEP")) is None


# ── metadata classification ─────────────────────────────────────────────────────


def _meta_payload(**overrides) -> dict:
    meta = {
        "archive_format": 2,
        "complete": True,
        "iteration": 1,
        "decision": "KEEP",
        "kept": True,
        "validation_passed": True,
        "files": {"kernel": "kernel.py"},
    }
    meta.update(overrides)
    return meta


@pytest.mark.parametrize(
    "payload",
    [
        [1, 2, 3],  # not a JSON object
        _meta_payload(files="kernel.py"),  # files must be a mapping
        _meta_payload(iteration=7),  # iteration must match dir
        _meta_payload(archive_format="2"),  # format must be an int
        _meta_payload(complete=False),  # format >= 2 needs marker
        {"iteration": 1, "decision": "KEEP", "kept": True},  # missing required keys
        _meta_payload(files={"kernel": "../escape.py"}),  # must stay inside the dir
        _meta_payload(files={"kernel": ""}),  # empty filename
        _meta_payload(files={"kernel": 7}),  # non-string filename
        _meta_payload(files={"kernel": "gone.py"}),  # referenced file missing
        _meta_payload(files={"kernel": "subdir"}),  # not a regular file
    ],
)
def test_incomplete_metadata_shapes_are_rejected(tmp_path, payload):
    archive = CandidateArchive(str(tmp_path))
    directory = archive._iter_dir(1)
    directory.mkdir()
    (directory / "kernel.py").write_text("kernel\n")
    (directory / "subdir").mkdir()
    (directory / "meta.json").write_text(json.dumps(payload))

    assert archive._inspect_complete_meta(directory, 1) == ("incomplete", None)
    assert archive._complete_meta(directory, 1) is None
    # A directory the archive cannot vouch for never reaches the index.
    assert archive.load_index() == []


def test_corrupt_metadata_is_quarantined_not_deleted(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    directory = archive._iter_dir(1)
    directory.mkdir()
    (directory / "kernel.py").write_text("salvageable kernel\n")
    (directory / "meta.json").write_text("{ truncated json")

    assert archive.load_meta(1) == {}
    assert not directory.exists()
    quarantined = list(archive.root.glob(".iter_001.incomplete-*"))
    assert len(quarantined) == 1
    # Quarantine preserves the bytes so a human can still recover the attempt.
    assert (quarantined[0] / "kernel.py").read_text() == "salvageable kernel\n"


def test_unreadable_metadata_is_preserved_and_reported(tmp_path, monkeypatch):
    archive = CandidateArchive(str(tmp_path))
    assert (
        archive.record(CandidateRecord(iteration=1, decision="KEEP", kept=True, kernel_source="healthy kernel\n"))
        is not None
    )
    meta_path = archive._iter_dir(1) / "meta.json"
    original_read_text = Path.read_text

    def transient_read(path, *args, **kwargs):
        if path == meta_path:
            raise OSError("simulated transient meta read failure")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", transient_read)

    # "unavailable" is not "corrupt": load_meta must give up empty-handed and
    # leave the candidate directory exactly where it is.
    assert archive.load_meta(1) == {}
    assert archive._iter_dir(1).is_dir()
    assert list(archive.root.glob(".iter_001.incomplete-*")) == []
    assert archive.degraded is True


def test_load_meta_on_unstattable_directory_degrades(tmp_path, monkeypatch):
    archive = CandidateArchive(str(tmp_path))
    directory = archive._iter_dir(3)
    directory.mkdir()  # no meta.json at all → "incomplete"
    original_stat = Path.stat

    def transient_stat(path, *args, **kwargs):
        if path == directory:
            raise OSError("simulated transient dir stat failure")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", transient_stat)

    assert archive.load_meta(3) == {}
    assert archive.degraded is True
    monkeypatch.undo()
    assert directory.is_dir()  # not quarantined on an uncertain stat


# ── reconcile / index repair ────────────────────────────────────────────────────


def test_reconcile_ignores_non_directory_iter_entries(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    archive.record(CandidateRecord(iteration=1, decision="KEEP", kept=True))
    stray = archive.root / "iter_005"
    stray.write_text("a file, not an iteration directory\n")
    archive._invalidate_cache()

    assert [entry["iter"] for entry in archive.load_index()] == [1]
    assert archive.max_iteration() == 1
    # A non-directory is skipped, not quarantined or removed.
    assert stray.is_file()


def test_unreadable_index_does_not_clobber_it(tmp_path, monkeypatch):
    archive = CandidateArchive(str(tmp_path))
    archive.record(CandidateRecord(iteration=1, decision="KEEP", kept=True))
    archive.record(CandidateRecord(iteration=2, decision="REVERT_PERF"))
    index_before = archive.index_path.read_bytes()
    original_read_text = Path.read_text

    def unreadable_index(path, *args, **kwargs):
        if path == archive.index_path:
            raise OSError("simulated index read failure")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable_index)
    archive._invalidate_cache()

    # meta.json is authoritative, so callers still get the full view; the index
    # we could not read must be left untouched rather than rewritten blind.
    assert [entry["iter"] for entry in archive.load_index()] == [1, 2]
    assert archive.degraded is True
    monkeypatch.undo()
    assert archive.index_path.read_bytes() == index_before


def test_unscannable_root_preserves_existing_index_entries(tmp_path, monkeypatch):
    archive = CandidateArchive(str(tmp_path))
    archive.record(CandidateRecord(iteration=1, decision="KEEP", kept=True))
    archive.record(CandidateRecord(iteration=2, decision="REVERT_PERF"))
    index_before = archive.index_path.read_bytes()
    original_iterdir = Path.iterdir

    def unlistable(path, *args, **kwargs):
        if path == archive.root:
            raise OSError("simulated directory scan failure")
        return original_iterdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "iterdir", unlistable)
    archive._invalidate_cache()

    # Nothing could be verified against meta.json, so every recorded line is
    # kept: an unscannable root must never look like "no attempts yet".
    assert [entry["iter"] for entry in archive.load_index()] == [1, 2]
    assert archive.max_iteration() == 2
    assert archive.degraded is True
    monkeypatch.undo()
    assert archive.index_path.read_bytes() == index_before


def test_index_rebuild_failure_still_returns_reconciled_view(tmp_path, monkeypatch):
    archive = CandidateArchive(str(tmp_path))
    archive.record(CandidateRecord(iteration=1, decision="KEEP", kept=True))
    archive.record(CandidateRecord(iteration=2, decision="REVERT_PERF"))
    truncated = archive.index_path.read_text().splitlines()[0] + "\n"
    archive.index_path.write_text(truncated)

    def failing_write(_entries):
        raise OSError("simulated index rebuild failure")

    monkeypatch.setattr(archive, "_write_index", failing_write)
    archive._invalidate_cache()

    assert [entry["iter"] for entry in archive.load_index()] == [1, 2]
    assert archive.degraded is True
    assert archive._index_cache is None  # a degraded view is never memoized
    assert archive.index_path.read_text() == truncated


def test_cache_add_entry_leaves_a_cold_cache_cold(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    archive.record(CandidateRecord(iteration=1, decision="KEEP", kept=True))
    archive._invalidate_cache()

    archive._cache_add_entry({"iter": 9, "decision": "KEEP", "dir": "iter_009"})

    # Folding into a cold cache must not conjure a one-entry cache out of thin
    # air — the next read has to reconcile from disk, which knows nothing of 9.
    assert archive._index_cache is None
    assert [entry["iter"] for entry in archive.load_index()] == [1]


# ── record collision handling ───────────────────────────────────────────────────


def test_record_replaces_stray_file_at_iteration_path(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    (archive.root / "iter_001").write_text("stray file squatting on iter_001\n")

    recorded = archive.record(CandidateRecord(iteration=1, decision="KEEP", kept=True, kernel_source="real kernel\n"))

    assert recorded == archive._iter_dir(1)
    assert archive.read_candidate_file(1, "kernel.py") == "real kernel\n"
    assert [entry["iter"] for entry in archive.load_index()] == [1]


def test_record_aborts_when_collision_cannot_be_quarantined(tmp_path, monkeypatch):
    archive = CandidateArchive(str(tmp_path))
    partial = archive._iter_dir(1)
    partial.mkdir()
    (partial / "kernel.py").write_text("partial kernel\n")
    original_replace = os.replace

    def refuse_quarantine(src, dst, *args, **kwargs):
        if str(src) == str(partial):
            raise OSError("simulated quarantine failure")
        return original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", refuse_quarantine)

    assert (
        archive.record(CandidateRecord(iteration=1, decision="KEEP", kept=True, kernel_source="replacement kernel\n"))
        is None
    )
    # Rather than write over ground it could not clear, record backs off and the
    # unreadable partial stays put for inspection.
    assert (partial / "kernel.py").read_text() == "partial kernel\n"
    assert archive.degraded is True
    assert list(archive.root.glob(".iter_001.tmp-*")) == []


def test_record_aborts_when_target_cannot_be_stated(tmp_path, monkeypatch):
    archive = CandidateArchive(str(tmp_path))
    target = archive._iter_dir(2)
    original_stat = Path.stat

    def transient_stat(path, *args, **kwargs):
        if path == target:
            raise OSError("simulated transient target stat failure")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", transient_stat)

    assert archive.record(CandidateRecord(iteration=2, decision="KEEP", kept=True, kernel_source="kernel\n")) is None
    monkeypatch.undo()
    assert not target.exists()
    assert archive.degraded is True
