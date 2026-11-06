# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The store of lane candidates the canonical tree moved past before any iteration measured them."""

from __future__ import annotations

import json

from kernelforge.workflows.forge_loop.campaign.unmeasured_lanes import UnmeasuredLane, UnmeasuredLaneArchive


def _lane(**overrides) -> UnmeasuredLane:
    fields = {
        "round_iteration": 4,
        "lane_id": "2",
        "base_commit": "abc123",
        "dropped_iteration": 6,
        "canonical_commit": "def456",
        "plan": "Summary: m_512 launches 128 workgroups. Stage A in LDS.\n# Lane 2 plan",
        "summary": "m_512 launches 128 workgroups. Stage A in LDS.",
        "diff": "--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n",
        "lesson": "lane 2 staged A in LDS; its own sweep read m_512 67.8 -> 13.8 us",
        "lesson_reason": "",
    }
    return UnmeasuredLane(**{**fields, **overrides})


def test_a_filed_candidate_keeps_its_diff_plan_record_and_where_it_stood(tmp_path):
    lane = _lane()

    filed = UnmeasuredLaneArchive(tmp_path).record(lane)

    assert filed == tmp_path / "forge_experiments" / "unmeasured_lanes" / "round_004_lane_2"
    assert (filed / "change.diff").read_text() == lane.diff
    assert (filed / "plan.md").read_text() == lane.plan
    assert (filed / "lesson.md").read_text() == lane.lesson
    meta = json.loads((filed / "meta.json").read_text())
    assert meta == {
        "round_iteration": 4,
        "lane_id": "2",
        "base_commit": "abc123",
        "dropped_iteration": 6,
        "canonical_commit": "def456",
        "summary": lane.summary,
        "lesson_reason": "",
    }
    (row,) = [json.loads(line) for line in (filed.parent / "index.jsonl").read_text().splitlines()]
    assert row == {**meta, "dir": "round_004_lane_2", "has_lesson": True}
    # The staging directory is renamed into place, so nothing half-written is left beside it.
    assert sorted(path.name for path in filed.parent.iterdir()) == ["index.jsonl", "round_004_lane_2"]


def test_a_candidate_without_a_record_says_why(tmp_path):
    filed = UnmeasuredLaneArchive(tmp_path).record(_lane(lesson="", lesson_reason="provider cannot resume the session"))

    assert not (filed / "lesson.md").exists()
    assert json.loads((filed / "meta.json").read_text())["lesson_reason"] == "provider cannot resume the session"
    (row,) = [json.loads(line) for line in (filed.parent / "index.jsonl").read_text().splitlines()]
    assert row["has_lesson"] is False


def test_a_candidate_filed_before_a_restart_is_not_filed_twice(tmp_path):
    """A process that filed it and stopped before rewriting its queue drops the same candidate again on resume."""
    archive = UnmeasuredLaneArchive(tmp_path)
    first = archive.record(_lane())

    assert archive.record(_lane(dropped_iteration=7)) is None
    assert json.loads((first / "meta.json").read_text())["dropped_iteration"] == 6
    assert len((first.parent / "index.jsonl").read_text().splitlines()) == 1


def test_a_store_that_cannot_be_written_costs_the_record_and_not_the_iteration(tmp_path):
    experiments = tmp_path / "forge_experiments"
    experiments.mkdir()
    (experiments / "unmeasured_lanes").write_text("not a directory")

    assert UnmeasuredLaneArchive(tmp_path).record(_lane()) is None
    assert (experiments / "unmeasured_lanes").read_text() == "not a directory"
