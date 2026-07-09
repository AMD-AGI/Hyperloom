# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for lesser-covered helpers in :mod:`hyperloom.inference_optimizer.session.session_paths`.

Covers the per-turn workdir allocator + its stale-dir pruner, the ext-trace
shard path fallback, and the ``_runs_actions`` import-failure fallback.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.session import session_paths


def test_ext_trace_path_blank_component_falls_back_to_unknown(tmp_path):
    p = session_paths.ext_trace_path(tmp_path, "", 42)
    assert p.name == "unknown-42.jsonl"
    p2 = session_paths.ext_trace_path(tmp_path, "geak_v3", 7)
    assert p2.name == "geak_v3-7.jsonl"


def test_allocate_turn_workdir_creates_and_prunes(tmp_path):
    subdir = "critic-workdir"
    # Allocate several turns keeping only the newest 2.
    created = []
    for turn in range(5):
        wd = session_paths.allocate_turn_workdir(tmp_path, subdir, turn, keep=2)
        assert wd.is_dir()
        # Drop a file inside so the pruner has content to remove.
        (wd / "scratch.txt").write_text("x", encoding="utf-8")
        created.append(wd)

    root = tmp_path / subdir
    remaining = sorted(p.name for p in root.iterdir() if p.is_dir())
    # After the final allocate(keep=2) the pruner leaves the newest turns.
    assert len(remaining) <= 3  # keep=2 pruned before creating the newest dir
    # The most recent turn dir must exist.
    assert (root / "000004").is_dir()


def test_allocate_turn_workdir_prunes_nested_content(tmp_path):
    subdir = "robustness-workdir"
    for turn in range(4):
        wd = session_paths.allocate_turn_workdir(tmp_path, subdir, turn, keep=1)
        nested = wd / "a" / "b"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "deep.txt").write_text("y", encoding="utf-8")
    root = tmp_path / subdir
    dirs = [p for p in root.iterdir() if p.is_dir()]
    assert len(dirs) <= 2
    assert (root / "000003").is_dir()


def test_prune_old_workdirs_iterdir_error_is_swallowed(tmp_path):
    # A non-existent root -> iterdir raises OSError -> pruner returns quietly.
    missing = tmp_path / "does-not-exist"
    session_paths._prune_old_workdirs(missing, keep=2)  # must not raise


def test_prune_old_workdirs_noop_when_under_keep(tmp_path):
    root = tmp_path / "wd"
    root.mkdir()
    (root / "000000").mkdir()
    session_paths._prune_old_workdirs(root, keep=5)  # nothing to prune
    assert (root / "000000").is_dir()


def test_runs_actions_fallback_on_import_failure(monkeypatch):
    import hyperloom.inference_optimizer.session.session_paths as sp

    # Force the registry construction path to raise so the fallback is used.
    import hyperloom.orchestrator.actions.registry as ar

    class _Boom:
        def load(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(ar, "ActionRegistry", _Boom)
    sp._runs_actions.cache_clear()
    try:
        result = sp._runs_actions()
        assert result == sp._RUNS_ACTIONS_FALLBACK
    finally:
        sp._runs_actions.cache_clear()
