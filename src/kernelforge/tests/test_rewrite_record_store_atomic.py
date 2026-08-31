"""Atomicity and locking tests for local rewrite records."""

from __future__ import annotations

import json
import multiprocessing
import threading
from pathlib import Path

import pytest

from kernelforge.rewrite_by_flydsl import record_store

CANONICAL_ID = "kernel:flydsl:softmax:vllm:1.0:flydsl:mi355x"
SESSION_ID = "softmax-session"


def _artifact(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _session_dir(root: Path) -> Path:
    return root / record_store.canonical_relpath(CANONICAL_ID) / "sessions" / SESSION_ID


def _write(
    store: record_store.LocalRewriteRecords,
    source: Path,
    *,
    version: str,
) -> None:
    store.write(
        CANONICAL_ID,
        SESSION_ID,
        {"speedup": 2.0, "version": version},
        {"kernel.py": source},
    )


def _assert_old_session(root: Path, store: record_store.LocalRewriteRecords) -> None:
    session = _session_dir(root)
    assert (session / "files" / "kernel.py").read_bytes() == b"old"
    assert json.loads((session / record_store.KNOWLEDGE_FILENAME).read_text())["version"] == "old"
    assert store.read_bytes(CANONICAL_ID, SESSION_ID, "kernel.py") == b"old"
    assert not [
        path
        for path in session.parent.iterdir()
        if ".staging-" in path.name or ".backup-" in path.name or ".failed-" in path.name
    ]


@pytest.mark.parametrize("failure", ["copy", "json", "replace"])
def test_failed_session_write_preserves_the_complete_old_session(
    tmp_path,
    monkeypatch,
    failure,
):
    root = tmp_path / "records"
    store = record_store.LocalRewriteRecords(root)
    _write(store, _artifact(tmp_path, "old.py", b"old"), version="old")
    replacement = _artifact(tmp_path, "new.py", b"new")

    if failure == "copy":
        monkeypatch.setattr(
            record_store,
            "_copy_file_synced",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("copy failed")),
        )
    elif failure == "json":
        monkeypatch.setattr(
            record_store,
            "_write_json_synced",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("json failed")),
        )
    else:
        original_replace = record_store.os.replace

        def fail_staging_replace(source, destination):
            if ".staging-" in Path(source).name and Path(destination).name == SESSION_ID:
                raise OSError("replace failed")
            return original_replace(source, destination)

        monkeypatch.setattr(record_store.os, "replace", fail_staging_replace)

    with pytest.raises(OSError, match="failed"):
        _write(store, replacement, version="new")

    _assert_old_session(root, store)


@pytest.mark.parametrize("failure", ["json", "replace"])
def test_failed_champion_write_preserves_the_old_pointer(
    tmp_path,
    monkeypatch,
    failure,
):
    root = tmp_path / "records"
    store = record_store.LocalRewriteRecords(root)
    store.promote(CANONICAL_ID, "old-session", 2.0)
    if failure == "json":
        monkeypatch.setattr(
            record_store,
            "_write_json_synced",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("champion json failed")),
        )
    else:
        original_replace = record_store.os.replace

        def fail_champion_replace(source, destination):
            if Path(destination).name == record_store.CHAMPION_FILENAME:
                raise OSError("champion replace failed")
            return original_replace(source, destination)

        monkeypatch.setattr(record_store.os, "replace", fail_champion_replace)

    with pytest.raises(OSError, match="champion .* failed"):
        store.promote(CANONICAL_ID, "new-session", 3.0)

    identity_dir = root / record_store.canonical_relpath(CANONICAL_ID)
    assert store.champion_speedup(CANONICAL_ID) == 2.0
    assert json.loads((identity_dir / record_store.CHAMPION_FILENAME).read_text())["session_id"] == "old-session"
    assert not list(identity_dir.glob(f".{record_store.CHAMPION_FILENAME}.*"))


def _paused_writer(
    root: str,
    source: str,
    staged: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
    errors: multiprocessing.queues.Queue,
) -> None:
    original_copy = record_store._copy_file_synced

    def copy_then_pause(source_path, target_path):
        original_copy(source_path, target_path)
        staged.set()
        if not release.wait(10):
            raise TimeoutError("reader test did not release writer")

    record_store._copy_file_synced = copy_then_pause
    try:
        _write(
            record_store.LocalRewriteRecords(root),
            Path(source),
            version="new",
        )
    except Exception as error:  # pragma: no cover - surfaced through the parent
        errors.put(repr(error))


def test_cross_process_reader_observes_only_complete_sessions(tmp_path):
    root = tmp_path / "records"
    store = record_store.LocalRewriteRecords(root)
    _write(store, _artifact(tmp_path, "old.py", b"old"), version="old")
    replacement = _artifact(tmp_path, "new.py", b"new")
    context = multiprocessing.get_context("fork")
    staged = context.Event()
    release = context.Event()
    errors = context.Queue()
    writer = context.Process(
        target=_paused_writer,
        args=(str(root), str(replacement), staged, release, errors),
    )
    writer.start()
    assert staged.wait(10)

    observed: list[bytes] = []
    read_done = threading.Event()

    def read_during_write() -> None:
        observed.append(store.read_bytes(CANONICAL_ID, SESSION_ID, "kernel.py"))
        read_done.set()

    reader = threading.Thread(target=read_during_write)
    reader.start()
    assert not read_done.wait(0.2)
    release.set()
    writer.join(10)
    reader.join(10)

    assert writer.exitcode == 0
    assert errors.empty()
    assert read_done.is_set()
    assert observed == [b"new"]
    assert store.read_bytes(CANONICAL_ID, SESSION_ID, "kernel.py") == b"new"
    assert json.loads((_session_dir(root) / record_store.KNOWLEDGE_FILENAME).read_text())["version"] == "new"
    assert not [
        path
        for path in _session_dir(root).parent.iterdir()
        if ".staging-" in path.name or ".backup-" in path.name or ".failed-" in path.name
    ]


def test_materialize_rejects_symlinked_source_artifacts(tmp_path):
    root = tmp_path / "records"
    store = record_store.LocalRewriteRecords(root)
    source = _artifact(tmp_path, "source.py", b"content")
    _write(store, source, version="one")
    artifact = _session_dir(root) / "files" / "kernel.py"
    artifact.unlink()
    artifact.symlink_to(source)
    candidate = store.candidates(CANONICAL_ID, limit=1)[0]

    with pytest.raises(record_store.RewriteRecordError, match="regular file"):
        store.materialize(CANONICAL_ID, candidate, tmp_path / "bundles")


def test_local_top_n_ranks_all_sessions_not_only_twenty_recent(tmp_path):
    store = record_store.LocalRewriteRecords(tmp_path / "records")
    source = _artifact(tmp_path, "source.py", b"content")
    store.write(
        CANONICAL_ID,
        "old-best",
        {"speedup": 10.0, "value": {"tag": "old-best"}},
        {"kernel.py": source},
    )
    for index in range(21):
        store.write(
            CANONICAL_ID,
            f"recent-{index}",
            {"speedup": 1.0 + index / 100, "value": {"tag": f"recent-{index}"}},
            {"kernel.py": source},
        )

    ranked = store.candidates(CANONICAL_ID, limit=3)

    assert ranked[0].session_id == "old-best"
    assert ranked[0].speedup == 10.0
    assert len(ranked) == 3


def test_rewriting_a_record_keeps_the_measurement_a_consumer_recorded(tmp_path):
    """A replacing write must not hand the ranking back the claim that lost.

    Ranking trusts a measured value over any claim, and only a consumer that ran
    the candidate can produce one. Dropping it on rewrite would restore the
    inflated claim that the measurement exists to correct.
    """
    root = tmp_path / "records"
    store = record_store.LocalRewriteRecords(root)
    source = _artifact(tmp_path, "kernel.py", b"first")
    store.write(CANONICAL_ID, SESSION_ID, {"speedup": 9.0}, {"kernel.py": source})
    store.record_measured_speedup(CANONICAL_ID, SESSION_ID, 1.2)

    store.write(
        CANONICAL_ID,
        SESSION_ID,
        {"speedup": 9.0, "version": "second"},
        {"kernel.py": _artifact(tmp_path, "again.py", b"second")},
    )

    knowledge = json.loads((_session_dir(root) / record_store.KNOWLEDGE_FILENAME).read_text())
    assert knowledge[record_store.MEASURED_SPEEDUP_KEY] == 1.2
    # The rest of the record is still replaced, which is what replace means.
    assert knowledge["version"] == "second"
    assert store.candidates(CANONICAL_ID, limit=1)[0].measured_speedup == 1.2


def test_a_rewrite_that_measured_the_candidate_itself_wins(tmp_path):
    """Carrying the old value must not shadow a fresher one in the same write."""
    root = tmp_path / "records"
    store = record_store.LocalRewriteRecords(root)
    source = _artifact(tmp_path, "kernel.py", b"first")
    store.write(CANONICAL_ID, SESSION_ID, {"speedup": 9.0}, {"kernel.py": source})
    store.record_measured_speedup(CANONICAL_ID, SESSION_ID, 1.2)

    store.write(
        CANONICAL_ID,
        SESSION_ID,
        {"speedup": 9.0, record_store.MEASURED_SPEEDUP_KEY: 1.5},
        {"kernel.py": source},
    )

    knowledge = json.loads((_session_dir(root) / record_store.KNOWLEDGE_FILENAME).read_text())
    assert knowledge[record_store.MEASURED_SPEEDUP_KEY] == 1.5


def test_non_posix_local_store_fails_with_an_explicit_locking_error(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(record_store, "fcntl", None)
    store = record_store.LocalRewriteRecords(tmp_path / "records")

    with pytest.raises(record_store.RewriteRecordError, match="POSIX fcntl"):
        store.candidates(CANONICAL_ID, limit=1)
