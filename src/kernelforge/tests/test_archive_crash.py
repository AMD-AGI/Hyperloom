# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for how a CRASHed iteration is archived and surfaced to the next
agent prompt (loop/archive.py).

A crashed iteration is recorded like any other failed attempt, with a distinct
CRASH decision, so the next iteration's lineage digest shows a `crash` row (and,
when recent, the crashing diff) — letting the agent avoid repeating it. These
tests use only a temp dir; no LLM / GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelforge.loop.archive import CandidateArchive, CandidateRecord


def _seed(archive: CandidateArchive) -> None:
    # A kept baseline, then a recent crashing attempt.
    archive.record(
        CandidateRecord(
            iteration=1,
            commit_hash="aaaaaaa",
            decision="KEEP",
            kept=True,
            validation_passed=True,
            wall_ms=1.0,
            plan="vectorize global loads",
            kernel_source="def k():\n    return 0\n",
            change_diff="+ vectorized load\n",
            baseline_wall_ms=2.0,
            best_wall_ms_before=2.0,
        )
    )
    archive.record(
        CandidateRecord(
            iteration=2,
            commit_hash="bbbbbbb",
            decision="CRASH",
            kept=False,
            validation_passed=False,
            wall_ms=None,
            plan="risky shared-mem rewrite",
            kernel_source="def k():\n    raise RuntimeError\n",
            change_diff="+ CRASH_DIFF_MARKER risky shared-mem change\n",
            validation_text="iteration crashed: boom\nTraceback (most recent call last)\n",
            baseline_wall_ms=2.0,
            best_wall_ms_before=1.0,
        )
    )


def test_crash_decision_label():
    assert CandidateArchive._label("CRASH") == "crash"


def test_crash_appears_in_digest_with_diff(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    _seed(archive)

    digest = archive.render_digest()

    # Legend documents the crash outcome.
    assert "crash=raised an exception" in digest
    # The crashing attempt's plan shows in the trajectory, and — being recent —
    # its actual diff is inlined so the agent sees what blew up.
    assert "risky shared-mem rewrite" in digest
    assert "CRASH_DIFF_MARKER" in digest


def test_crash_recorded_on_disk(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    _seed(archive)

    index = archive.load_index()
    crash_entries = [e for e in index if e.get("decision") == "CRASH"]
    assert len(crash_entries) == 1
    assert crash_entries[0]["iter"] == 2

    # The crashing diff + failure text are persisted for later inspection.
    assert "CRASH_DIFF_MARKER" in archive.read_candidate_file(2, "change.diff")
    assert "iteration crashed" in archive.read_candidate_file(2, "validation.txt")
    assert archive.load_meta(2)["decision"] == "CRASH"


def test_archive_reconciles_next_iteration_from_index_and_directories(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    _seed(archive)
    (archive.root / "iter_007").mkdir()

    assert archive.max_iteration() == 2
    assert archive.reconcile_next_iteration(3) == 3
    assert archive.reconcile_next_iteration(12) == 12


def test_archive_refuses_to_overwrite_existing_iteration(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    original = CandidateRecord(
        iteration=1,
        decision="KEEP",
        kept=True,
        kernel_source="original kernel\n",
    )
    replacement = CandidateRecord(
        iteration=1,
        decision="REVERT_PERF",
        kernel_source="replacement kernel\n",
    )

    first_path = archive.record(original)
    second_path = archive.record(replacement)

    assert first_path is not None
    assert second_path is None
    assert archive.read_candidate_file(1, "kernel.py") == "original kernel\n"
    assert len(archive.load_index()) == 1


def test_record_failure_before_publish_leaves_no_final_directory(
    tmp_path,
    monkeypatch,
):
    archive = CandidateArchive(str(tmp_path))

    def fail_metadata_write(path, text):
        if path.name == "meta.json":
            raise OSError("simulated metadata write failure")
        path.write_text(text)

    monkeypatch.setattr(archive, "_write_text", fail_metadata_write, raising=False)

    recorded = archive.record(
        CandidateRecord(
            iteration=1,
            decision="KEEP",
            kept=True,
            kernel_source="candidate kernel\n",
            change_diff="candidate diff\n",
        )
    )

    assert recorded is None
    assert not archive._iter_dir(1).exists()
    assert list(archive.root.glob(".iter_001.tmp-*")) == []
    assert archive.load_index() == []


def test_record_replaces_legacy_partial_directory_and_repairs_index(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    partial = archive._iter_dir(1)
    partial.mkdir()
    (partial / "kernel.py").write_text("stale partial kernel\n")
    stale = {"iter": 1, "decision": "KEEP", "dir": "iter_001"}
    archive.index_path.write_text(json.dumps(stale) + "\n" + json.dumps(stale) + "\n")

    recorded = archive.record(
        CandidateRecord(
            iteration=1,
            decision="REVERT_PERF",
            kernel_source="replacement kernel\n",
        )
    )

    assert recorded == archive._iter_dir(1)
    assert archive.read_candidate_file(1, "kernel.py") == "replacement kernel\n"
    assert archive.load_meta(1)["decision"] == "REVERT_PERF"
    assert [entry["iter"] for entry in archive.load_index()] == [1]
    persisted = [json.loads(line) for line in archive.index_path.read_text().splitlines() if line.strip()]
    assert [entry["iter"] for entry in persisted] == [1]


def test_record_replaces_directory_with_malformed_completion_marker(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    partial = archive._iter_dir(1)
    partial.mkdir()
    (partial / "meta.json").write_text(
        json.dumps(
            {
                "archive_format": "invalid",
                "complete": True,
                "iteration": 1,
                "decision": "KEEP",
                "kept": True,
                "validation_passed": True,
                "files": {},
            }
        )
    )

    recorded = archive.record(
        CandidateRecord(
            iteration=1,
            decision="REVERT_VALIDATION",
            validation_text="replacement record\n",
        )
    )

    assert recorded == archive._iter_dir(1)
    assert archive.load_meta(1)["decision"] == "REVERT_VALIDATION"


def test_load_index_rebuilds_missing_entries_and_removes_duplicates(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    _seed(archive)
    first = archive.load_index()[0]
    archive.index_path.write_text(json.dumps(first) + "\n" + json.dumps(first) + "\n{malformed\n")

    rebuilt = archive.load_index()

    assert [entry["iter"] for entry in rebuilt] == [1, 2]
    persisted = [json.loads(line) for line in archive.index_path.read_text().splitlines() if line.strip()]
    assert [entry["iter"] for entry in persisted] == [1, 2]


@pytest.mark.parametrize("failure_kind", ["read", "stat"])
def test_transient_candidate_io_failure_preserves_directory_and_index(
    tmp_path,
    monkeypatch,
    failure_kind,
):
    archive = CandidateArchive(str(tmp_path))
    _seed(archive)
    candidate_dir = archive._iter_dir(1)
    index_before = archive.index_path.read_bytes()

    if failure_kind == "read":
        original_read_text = Path.read_text

        def transient_read(path, *args, **kwargs):
            if path == candidate_dir / "meta.json":
                raise OSError("simulated transient metadata read failure")
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", transient_read)
    else:
        original_stat = Path.stat

        def transient_stat(path, *args, **kwargs):
            if path == candidate_dir / "kernel.py":
                raise OSError("simulated transient candidate stat failure")
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", transient_stat)

    # The index is memoized, so a scan only touches disk on a cold/invalidated
    # read (resume, or after an on-disk change). Drop the warm cache so this
    # exercises the reconcile path the transient failure is about.
    archive._invalidate_cache()
    entries = archive.load_index()

    assert [entry["iter"] for entry in entries] == [1, 2]
    assert candidate_dir.is_dir()
    assert archive.index_path.read_bytes() == index_before
    assert archive.degraded is True
    monkeypatch.undo()
    archive._invalidate_cache()
    assert [entry["iter"] for entry in archive.load_index()] == [1, 2]
    assert candidate_dir.is_dir()


def test_transient_existing_meta_read_does_not_replace_healthy_candidate(
    tmp_path,
    monkeypatch,
):
    archive = CandidateArchive(str(tmp_path))
    original = CandidateRecord(
        iteration=1,
        decision="KEEP",
        kept=True,
        kernel_source="original kernel\n",
    )
    assert archive.record(original) == archive._iter_dir(1)
    meta_path = archive._iter_dir(1) / "meta.json"
    original_read_text = Path.read_text

    def transient_read(path, *args, **kwargs):
        if path == meta_path:
            raise OSError("simulated transient metadata read failure")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", transient_read)

    recorded = archive.record(
        CandidateRecord(
            iteration=1,
            decision="REVERT_PERF",
            kernel_source="replacement kernel\n",
        )
    )

    assert recorded is None
    assert archive._iter_dir(1).is_dir()
    assert (archive._iter_dir(1) / "kernel.py").read_text() == "original kernel\n"


def test_invalid_utf8_index_is_rebuilt_from_complete_metadata(tmp_path):
    archive = CandidateArchive(str(tmp_path))
    _seed(archive)
    archive.index_path.write_bytes(b"\xff\xfeinvalid index\n")

    rebuilt = archive.load_index()

    assert [entry["iter"] for entry in rebuilt] == [1, 2]
    persisted = [json.loads(line) for line in archive.index_path.read_text().splitlines() if line.strip()]
    assert [entry["iter"] for entry in persisted] == [1, 2]


def test_index_append_failure_is_surfaced_and_recoverable(tmp_path, monkeypatch):
    archive = CandidateArchive(str(tmp_path))

    def fail_append(_entry):
        raise OSError("simulated index append failure")

    monkeypatch.setattr(archive, "_append_index", fail_append)

    with pytest.raises(OSError, match="simulated index append failure"):
        archive.record(
            CandidateRecord(
                iteration=1,
                decision="KEEP",
                kept=True,
                kernel_source="durable candidate\n",
            )
        )

    assert archive.load_meta(1)["decision"] == "KEEP"
    recovered = CandidateArchive(str(tmp_path))
    assert [entry["iter"] for entry in recovered.load_index()] == [1]
