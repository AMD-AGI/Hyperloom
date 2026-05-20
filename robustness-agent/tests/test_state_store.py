"""Tests for the disk-backed DetectorStateStore."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robustness_agent.state_store import (
    DetectorStateStore,
    DetectorStateView,
)


def test_empty_session_dir_starts_with_empty_state(tmp_path: Path) -> None:
    store = DetectorStateStore(session_dir=tmp_path)
    assert store.load_slot("gpu_leak") == {}
    assert store.snapshot() == {}
    # Nothing written yet → no file.
    assert not store.file_path.exists()


def test_save_then_flush_roundtrips(tmp_path: Path) -> None:
    store = DetectorStateStore(session_dir=tmp_path)
    store.save_slot("gpu_leak", {"consecutive_hits": 2})
    store.save_slot("progress", {"gain_history": [0.0, 0.1, 0.2]})
    assert not store.file_path.exists()  # not flushed yet
    store.flush_atomic()
    assert store.file_path.is_file()

    # Reload via a new store instance.
    again = DetectorStateStore(session_dir=tmp_path)
    assert again.load_slot("gpu_leak") == {"consecutive_hits": 2}
    assert again.load_slot("progress") == {
        "gain_history": [0.0, 0.1, 0.2],
    }
    assert again.load_slot("missing") == {}


def test_flush_is_noop_when_clean(tmp_path: Path) -> None:
    store = DetectorStateStore(session_dir=tmp_path)
    store.flush_atomic()
    assert not store.file_path.exists()


def test_malformed_json_recovers_to_empty(tmp_path: Path) -> None:
    target_dir = tmp_path / "agents" / "robustness"
    target_dir.mkdir(parents=True)
    (target_dir / "detector_state.json").write_text(
        "{not valid json", encoding="utf-8",
    )
    store = DetectorStateStore(session_dir=tmp_path)
    assert store.snapshot() == {}
    # Writing then flushing must overwrite the bad file with a valid one.
    store.save_slot("gpu_leak", {"consecutive_hits": 1})
    store.flush_atomic()
    parsed = json.loads(store.file_path.read_text(encoding="utf-8"))
    assert parsed == {"gpu_leak": {"consecutive_hits": 1}}


def test_non_dict_top_level_is_dropped(tmp_path: Path) -> None:
    target_dir = tmp_path / "agents" / "robustness"
    target_dir.mkdir(parents=True)
    (target_dir / "detector_state.json").write_text(
        json.dumps(["not", "a", "dict"]), encoding="utf-8",
    )
    store = DetectorStateStore(session_dir=tmp_path)
    assert store.snapshot() == {}


def test_non_dict_slot_values_are_dropped(tmp_path: Path) -> None:
    target_dir = tmp_path / "agents" / "robustness"
    target_dir.mkdir(parents=True)
    (target_dir / "detector_state.json").write_text(
        json.dumps({
            "gpu_leak": {"consecutive_hits": 1},
            "ray_pending": ["bad"],  # not a dict
            "progress": "also bad",
        }),
        encoding="utf-8",
    )
    store = DetectorStateStore(session_dir=tmp_path)
    snap = store.snapshot()
    assert snap == {"gpu_leak": {"consecutive_hits": 1}}


def test_atomic_write_replaces_in_place(tmp_path: Path) -> None:
    store = DetectorStateStore(session_dir=tmp_path)
    store.save_slot("gpu_leak", {"consecutive_hits": 1})
    store.flush_atomic()
    inode1 = store.file_path.stat().st_ino

    store.save_slot("gpu_leak", {"consecutive_hits": 2})
    store.flush_atomic()
    inode2 = store.file_path.stat().st_ino
    # os.replace produces a new inode (it's a rename), so this confirms
    # we're not append-writing into the live file.
    assert inode1 != inode2

    again = DetectorStateStore(session_dir=tmp_path)
    assert again.load_slot("gpu_leak") == {"consecutive_hits": 2}


def test_view_load_save_roundtrip(tmp_path: Path) -> None:
    store = DetectorStateStore(session_dir=tmp_path)
    view = store.view("gpu_leak")
    assert view.is_persistent is True
    assert view.slot_name == "gpu_leak"
    assert view.load() == {}

    view.save({"consecutive_hits": 3})
    store.flush_atomic()
    again = DetectorStateStore(session_dir=tmp_path).view("gpu_leak").load()
    assert again == {"consecutive_hits": 3}


def test_in_memory_view_is_noop(tmp_path: Path) -> None:
    view = DetectorStateView(store=None, slot="gpu_leak")
    assert view.is_persistent is False
    assert view.load() == {}
    view.save({"consecutive_hits": 99})  # noop
    assert view.load() == {}


def test_save_slot_rejects_non_dict(tmp_path: Path) -> None:
    store = DetectorStateStore(session_dir=tmp_path)
    with pytest.raises(TypeError):
        store.save_slot("gpu_leak", "not a dict")  # type: ignore[arg-type]


def test_load_slot_returns_copy(tmp_path: Path) -> None:
    store = DetectorStateStore(session_dir=tmp_path)
    store.save_slot("gpu_leak", {"consecutive_hits": 1})
    snap = store.load_slot("gpu_leak")
    snap["consecutive_hits"] = 99
    # Mutating the returned dict must not affect the store.
    assert store.load_slot("gpu_leak") == {"consecutive_hits": 1}
