"""Sanity tests for the 8 KB seeds shipped with PR-I."""

from __future__ import annotations

from pathlib import Path

from framework_agent.agent.kb_priors import read_priors


_KB_ROOT = (
    Path(__file__).resolve().parents[2] / "kb"
)


def test_8_seeds_load_through_kb_priors():
    """Read all priors; assert the 8 seed entry_ids are present."""
    entries = read_priors("sglang", kb_root=_KB_ROOT)
    ids = {e.entry_id for e in entries}
    expected = {
        "fw-perf-001", "fw-perf-002", "fw-perf-003",
        "fw-boundary-001", "fw-boundary-002", "fw-boundary-003",
        "fw-pitfall-001", "fw-pitfall-002",
    }
    assert expected.issubset(ids), (
        f"missing seeds: {expected - ids}; got {ids}"
    )


def test_perf_seeds_have_framework_attribute():
    """fw-perf-NNN must declare Framework: (target framework)."""
    entries = {e.entry_id: e for e in read_priors("sglang", kb_root=_KB_ROOT)}
    for eid in ("fw-perf-001", "fw-perf-002", "fw-perf-003"):
        e = entries[eid]
        assert e.target_framework in ("vllm", "sglang"), (
            f"{eid} target_framework={e.target_framework!r}"
        )


def test_boundary_seeds_are_cross_framework():
    """Boundary rules apply to both frameworks -> Framework: blank."""
    entries = {e.entry_id: e for e in read_priors("vllm", kb_root=_KB_ROOT)}
    for eid in ("fw-boundary-001", "fw-boundary-002", "fw-boundary-003"):
        assert entries[eid].target_framework == ""


def test_seeds_ranking_pitfall_first():
    """Pitfalls must surface before boundaries before perfs."""
    entries = read_priors("sglang", kb_root=_KB_ROOT)
    cats = [e.category for e in entries]
    # All pitfalls precede all boundaries.
    last_pitfall = max(
        (i for i, c in enumerate(cats) if c == "pitfall"), default=-1,
    )
    first_boundary = min(
        (i for i, c in enumerate(cats) if c == "boundary"), default=10**9,
    )
    assert last_pitfall < first_boundary
    # All boundaries precede all perfs.
    last_boundary = max(
        (i for i, c in enumerate(cats) if c == "boundary"), default=-1,
    )
    first_perf = min(
        (i for i, c in enumerate(cats) if c == "perf"), default=10**9,
    )
    assert last_boundary < first_perf
