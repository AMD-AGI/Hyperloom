# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fragment merge semantics in the breakdown assembler.

Producers write partial fragments from deep inside their own work, and a second
write of the same entity id merges into the first. What that merge keeps, and
what it silently replaces, decides whether a recorded fact survives to the
archive -- so these pin the merge rather than any one caller's output.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.breakdown.recorder import assembler as asm


# ---- _deep_merge ----


def test_a_partial_update_fills_gaps_without_dropping_prior_fields():
    merged = asm._deep_merge({"a": 1, "b": 2}, {"b": 2, "c": 3})
    assert merged == {"a": 1, "b": 2, "c": 3}


def test_nested_dicts_merge_key_by_key_rather_than_wholesale():
    """Whole-value replacement is what loses a sibling the update never mentioned."""
    merged = asm._deep_merge({"m": {"x": 1, "y": 2}}, {"m": {"y": 2, "z": 3}})
    assert merged["m"] == {"x": 1, "y": 2, "z": 3}


def test_filling_an_absent_field_is_not_a_conflict():
    conflicts: list[str] = []
    asm._deep_merge({"a": None}, {"a": 5}, conflicts=conflicts)
    asm._deep_merge({}, {"b": 5}, conflicts=conflicts)
    assert conflicts == []


def test_replacing_an_answer_with_a_different_one_is_reported():
    """Last-writer-wins is the behaviour; the point is that it is visible."""
    conflicts: list[str] = []
    merged = asm._deep_merge({"tput": 5081.01}, {"tput": 5100.76}, conflicts=conflicts)
    assert merged["tput"] == 5100.76
    assert conflicts == ["tput"]


def test_a_conflict_is_reported_at_its_full_path():
    conflicts: list[str] = []
    asm._deep_merge({"outer": {"inner": 1}}, {"outer": {"inner": 2}}, conflicts=conflicts)
    assert conflicts == ["outer.inner"]


def test_conflicts_are_only_collected_when_the_caller_asks():
    assert asm._deep_merge({"a": 1}, {"a": 2}) == {"a": 2}


def test_operation_start_time_keeps_the_earliest_partial_update():
    conflicts: list[str] = []
    merged = asm._deep_merge(
        {"started_at": "2026-08-27T14:00:00+00:00", "status": "running"},
        {"started_at": "2026-08-27T14:05:00+00:00", "status": "succeeded"},
        conflicts=conflicts,
    )

    assert merged["started_at"] == "2026-08-27T14:00:00+00:00"
    assert merged["status"] == "succeeded"
    assert "started_at" not in conflicts


def test_operation_start_time_with_period_in_stable_id_keeps_the_earliest_update():
    conflicts: list[str] = []
    operation_id = "op:geak_e2e_attempt:1000.0-1050.0:abc"
    merged = asm._merge_v4_entities(
        [
            {
                "operation_id": operation_id,
                "started_at": "2026-08-27T14:00:00+00:00",
                "status": "running",
            },
            {
                "operation_id": operation_id,
                "started_at": "2026-08-27T14:05:00+00:00",
                "status": "succeeded",
            },
        ],
        id_fields=("operation_id",),
        conflicts=conflicts,
    )

    assert merged[0]["started_at"] == "2026-08-27T14:00:00+00:00"
    assert merged[0]["status"] == "succeeded"
    assert f"{operation_id}.started_at" not in conflicts


# ---- _merge_lists ----


def test_rows_with_the_same_nested_id_merge_instead_of_duplicating():
    merged = asm._merge_lists(
        [{"attempt_id": "a1", "status": "running"}],
        [{"attempt_id": "a1", "gain_pct": 3.0}],
    )
    assert merged == [{"attempt_id": "a1", "status": "running", "gain_pct": 3.0}]


def test_a_new_id_appends_rather_than_overwriting():
    merged = asm._merge_lists(
        [{"attempt_id": "a1"}],
        [{"attempt_id": "a2"}],
    )
    assert [row["attempt_id"] for row in merged] == ["a1", "a2"]


def test_a_second_row_for_a_new_id_still_merges_onto_the_first():
    """The index has to learn ids the update itself introduced."""
    merged = asm._merge_lists(
        [],
        [{"attempt_id": "a1", "status": "running"}, {"attempt_id": "a1", "status": "done"}],
    )
    assert merged == [{"attempt_id": "a1", "status": "done"}]


def test_rows_without_a_known_id_are_appended_once():
    merged = asm._merge_lists([{"note": "x"}], [{"note": "x"}, {"note": "y"}])
    assert merged == [{"note": "x"}, {"note": "y"}]


def test_scalar_entries_deduplicate():
    assert asm._merge_lists(["a"], ["a", "b"]) == ["a", "b"]


def test_a_list_under_a_key_merges_by_id_through_deep_merge():
    merged = asm._deep_merge(
        {"attempts": [{"attempt_id": "a1", "status": "running"}]},
        {"attempts": [{"attempt_id": "a1", "status": "kept"}]},
    )
    assert merged["attempts"] == [{"attempt_id": "a1", "status": "kept"}]


def test_the_first_recognised_id_field_decides_identity():
    """A row carrying two id fields is keyed on the first in the registry order."""
    merged = asm._merge_lists(
        [{"attempt_id": "a1", "measurement_id": "m1", "n": 1}],
        [{"attempt_id": "a1", "measurement_id": "m2", "n": 2}],
    )
    assert len(merged) == 1
    assert merged[0]["n"] == 2


# ---- substream composition ----


def test_versions_fold_into_one_entry_per_tool():
    out = {"versions": [{"tool": "TraceLens", "v": "1"}, {"tool": "tracelens", "v": "2"}]}
    asm._compose_versions(out)
    assert out["versions"] == {"tracelens": {"tool": "tracelens", "v": "2"}}


def test_versions_ignore_rows_that_name_no_tool():
    out = {"versions": [{"v": "1"}, "junk"]}
    asm._compose_versions(out)
    assert out["versions"] == {}


def test_versions_are_left_alone_when_nothing_was_recorded():
    out = {}
    asm._compose_versions(out)
    assert "versions" not in out


def test_critic_and_robustness_substreams_fold_into_one_section():
    out = {
        "critic_iterations": [{"iteration": 1}],
        "robustness_signals": [{"name": "gain_plateau"}],
    }
    asm._compose_critic_robustness(out)

    assert "critic_iterations" not in out and "robustness_signals" not in out
    section = out["critic_robustness"]
    assert section["critic_iterations"] == [{"iteration": 1}]
    assert section["robustness_signals"] == [{"name": "gain_plateau"}]
    assert "kb_writes_summary" in section


def test_a_recorded_section_outranks_the_substreams():
    """A producer that wrote the whole section already said what it means."""
    out = {"critic_robustness": {"critic_iterations": ["kept"]}, "critic_iterations": [{"i": 1}]}
    asm._compose_critic_robustness(out)
    assert out["critic_robustness"] == {"critic_iterations": ["kept"]}
    assert "critic_iterations" not in out


def test_composing_critic_robustness_is_a_no_op_without_either_substream():
    out = {"other": 1}
    asm._compose_critic_robustness(out)
    assert out == {"other": 1}
