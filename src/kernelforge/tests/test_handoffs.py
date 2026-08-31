"""Tests for lightweight immutable iteration handoffs."""

from __future__ import annotations

import pytest

from kernelforge.loop.handoffs import HandoffStore, IterationHandoff


def _handoff(iteration: int = 1) -> IterationHandoff:
    return IterationHandoff(
        iteration=iteration,
        analysis_commit="canonical-abc",
        canonical_verdict="REVERT_PERF",
        optimization_plan_path=(f"forge_experiments/orchestration/iter_{iteration:03d}/optimization_plan.md"),
        supervisor_ruling_path="forge_experiments/supervisor/latest.md",
        plan="Test vector loads.",
        lesson_path=f"forge_experiments/lessons/iter_{iteration:03d}.md",
        orchestration_artifacts=(f"forge_experiments/orchestration/iter_{iteration:03d}"),
        candidate_archive=(f"forge_experiments/candidates/iter_{iteration:03d}"),
    )


def test_handoff_store_writes_and_reads_latest(tmp_path):
    store = HandoffStore(str(tmp_path))

    first = store.write(_handoff(1))
    second = store.write(_handoff(2))

    assert first.is_file()
    assert second.is_file()
    latest = store.latest()
    assert latest is not None
    latest_path, payload = latest
    assert latest_path == second
    assert payload["iteration"] == 2
    assert payload["search_policy"]["mode"] == "EXPLOIT"
    assert payload["optimization_plan_path"] == ("forge_experiments/orchestration/iter_002/optimization_plan.md")
    assert payload["supervisor_ruling_path"] == "forge_experiments/supervisor/latest.md"


def test_handoff_store_is_idempotent_and_rejects_conflicts(tmp_path):
    store = HandoffStore(str(tmp_path))
    original = _handoff(1)

    first_path = store.write(original)
    second_path = store.write(original)
    assert first_path == second_path

    conflicting = IterationHandoff(
        **{
            **original.__dict__,
            "canonical_verdict": "KEEP",
        }
    )
    with pytest.raises(ValueError, match="conflicts"):
        store.write(conflicting)


def test_handoff_allows_direct_implementer_without_plan():
    handoff = IterationHandoff(
        iteration=1,
        analysis_commit="canonical-abc",
        canonical_verdict="REVERT_PERF",
    )

    payload = handoff.to_dict()

    assert payload["optimization_plan_path"] == ""


def test_handoff_store_rejects_noncurrent_shape(tmp_path):
    store = HandoffStore(str(tmp_path))
    store.path(1).write_text('{"schema_version": 1, "complete": true, "iteration": 1}')

    with pytest.raises(ValueError, match="unsupported handoff schema"):
        store.read(1)
