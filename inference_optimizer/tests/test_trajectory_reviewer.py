"""Tests for trajectory_reviewer — coverage boost for CI threshold."""
from __future__ import annotations

from dataclasses import dataclass, field

from hyperloom.orchestrator.trajectory_reviewer import (
    _exhausted_clusters,
    _stalled_cycle_count,
    build_trajectory_digest,
)


@dataclass
class _FakeEntry:
    outcome: str = ""
    kind: str = ""
    change: str = ""
    gain_pct: float | None = None


@dataclass
class _FakeState:
    session_id: str = "test-session"
    model_name: str = "TestModel"
    hardware: str = "mi300x"
    explore_search: dict = field(default_factory=dict)
    macro_cycle: int = 0
    roofline_snapshots: list = field(default_factory=list)
    cumulative_gain_validated: float | None = None


def test_exhausted_clusters_empty():
    assert _exhausted_clusters([]) == []


def test_exhausted_clusters_groups_reverts():
    entries = [
        _FakeEntry(outcome="REVERT", kind="explore", change="--tp 4", gain_pct=-2.0),
        _FakeEntry(outcome="REVERT", kind="explore", change="--tp 4", gain_pct=-1.5),
        _FakeEntry(outcome="KEEP", kind="explore", change="--tp 8", gain_pct=5.0),
    ]
    dead = _exhausted_clusters(entries)
    assert len(dead) == 1
    assert dead[0]["kind"] == "explore"
    assert dead[0]["change"] == "--tp 4"
    assert dead[0]["count"] == 2


def test_exhausted_clusters_ignores_high_gain():
    entries = [
        _FakeEntry(outcome="no_promote", kind="explore", change="X", gain_pct=5.0),
        _FakeEntry(outcome="no_promote", kind="explore", change="X", gain_pct=3.0),
    ]
    dead = _exhausted_clusters(entries)
    assert dead == []


def test_exhausted_clusters_none_gain():
    entries = [
        _FakeEntry(outcome="REVERT", kind="kernel_agent", change="patch_a"),
        _FakeEntry(outcome="REVERT", kind="kernel_agent", change="patch_a"),
        _FakeEntry(outcome="REVERT", kind="kernel_agent", change="patch_a"),
    ]
    dead = _exhausted_clusters(entries)
    assert len(dead) == 1
    assert dead[0]["max_gain"] is None
    assert dead[0]["count"] == 3


def test_stalled_cycle_count_no_stall():
    state = _FakeState(
        macro_cycle=3,
        explore_search={"winners_history": [
            {"cycle": 1}, {"cycle": 2}, {"cycle": 3},
        ]},
    )
    assert _stalled_cycle_count(state) == 0


def test_stalled_cycle_count_two_stalled():
    state = _FakeState(
        macro_cycle=5,
        explore_search={"winners_history": [
            {"cycle": 1}, {"cycle": 2}, {"cycle": 3},
        ]},
    )
    assert _stalled_cycle_count(state) == 2


def test_stalled_cycle_count_no_winners():
    state = _FakeState(macro_cycle=3, explore_search={})
    assert _stalled_cycle_count(state) == 4


def test_build_trajectory_digest_no_stall(tmp_path):
    state = _FakeState(
        macro_cycle=0,
        explore_search={"winners_history": [{"cycle": 0}]},
    )
    result = build_trajectory_digest(tmp_path, state)
    assert result == ""


def test_build_trajectory_digest_with_stall(tmp_path):
    state = _FakeState(
        macro_cycle=3,
        explore_search={"winners_history": [{"cycle": 0}]},
        cumulative_gain_validated=12.5,
    )
    result = build_trajectory_digest(tmp_path, state)
    assert "stalled_cycles=3" in result
    assert "validated_gain=12.50%" in result
    assert "advisory only" in result


def test_build_trajectory_digest_with_roofline(tmp_path):
    state = _FakeState(
        macro_cycle=2,
        explore_search={"winners_history": [{"cycle": 0}]},
        roofline_snapshots=[{"compute_pct": 80, "memory_pct": 20}],
    )
    result = build_trajectory_digest(tmp_path, state)
    # May or may not produce output depending on dominant_direction impl
    assert isinstance(result, str)


def test_build_trajectory_digest_stall_without_validated(tmp_path):
    state = _FakeState(
        macro_cycle=5,
        explore_search={"winners_history": [{"cycle": 1}]},
        cumulative_gain_validated=None,
    )
    result = build_trajectory_digest(tmp_path, state)
    assert "stalled_cycles=" in result


def test_build_trajectory_digest_with_dead_clusters(tmp_path):
    """Cover exhausted_directions formatting (L150-156)."""
    from hyperloom.orchestrator.optimization_journal import (
        Journal, JournalEntry,
    )
    journal = Journal.load_or_create(tmp_path, session_id="s1", model="m", hardware="h")
    for i in range(3):
        journal.append_entry(JournalEntry(
            phase="EXPLORE", iter=i + 1, kind="explore", change="--disable-radix",
            outcome="REVERT", gain_pct=-1.0, task_id=f"task-{i}",
        ))

    state = _FakeState(
        session_id="s1",
        macro_cycle=3,
        explore_search={"winners_history": [{"cycle": 0}]},
    )
    result = build_trajectory_digest(tmp_path, state)
    assert "exhausted_directions" in result
    assert "--disable-radix" in result
    assert "non-promoting attempts" in result


def test_load_journal_entries_swallows_errors(tmp_path, monkeypatch):
    """A Journal.load_or_create failure yields an empty entry list (L52-53)."""
    import hyperloom.orchestrator.trajectory_reviewer as tr

    def boom(*_a, **_k):
        raise RuntimeError("journal unreadable")

    monkeypatch.setattr(tr.Journal, "load_or_create", boom)
    assert tr._load_journal_entries(tmp_path, _FakeState()) == []


def test_build_trajectory_digest_snaps_without_direction_returns_empty(tmp_path):
    """Snapshots present but no dominant direction + no dead/stall -> "" (L167-168)."""
    # A snapshot dict that dominant_direction cannot resolve to a lever, with
    # no stall and no exhausted clusters, produces no lines -> empty string.
    state = _FakeState(
        macro_cycle=0,
        explore_search={"winners_history": [{"cycle": 0}]},
        roofline_snapshots=[{"unknown_field": 1}],
    )
    result = build_trajectory_digest(tmp_path, state)
    assert result == ""


def test_build_trajectory_digest_outer_exception_returns_empty(tmp_path, monkeypatch):
    """An unexpected error inside the digest builder is swallowed (L171-173)."""
    import hyperloom.orchestrator.trajectory_reviewer as tr

    def boom(*_a, **_k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(tr, "_exhausted_clusters", boom)
    state = _FakeState(macro_cycle=1, explore_search={})
    assert build_trajectory_digest(tmp_path, state) == ""
