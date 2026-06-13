# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``orchestrator.optimization_journal``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.optimization_journal import (
    JOURNAL_FILENAME,
    Journal,
    JournalEntry,
    KIND_BACKEND,
    KIND_ENV,
    KIND_KERNEL_FILE,
    KIND_OTHER,
    KIND_PARAM,
    OUTCOME_KEEP,
    OUTCOME_REVERT,
    classify_change_kind,
    summarize_change,
)


# fixtures
@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    sd = tmp_path / "session-X"
    sd.mkdir()
    return sd


# Journal — construction
def test_load_or_create_mints_new_when_absent(session_dir: Path):
    j = Journal.load_or_create(
        session_dir, session_id="sid-1", model="m", hardware="mi300x",
        framework="sglang", baseline_throughput=600.0,
    )
    assert j.session_id == "sid-1"
    assert j.model == "m"
    assert j.hardware == "mi300x"
    assert j.framework == "sglang"
    assert j.baseline_throughput == 600.0
    assert j.entries == []
    expected = session_dir / "reports" / JOURNAL_FILENAME
    assert j.path == expected
    assert expected.parent.exists()
    assert not expected.exists()  # flushes only when something is appended


def test_load_or_create_round_trips_existing_file(session_dir: Path):
    j1 = Journal.load_or_create(
        session_dir, session_id="sid-1", model="m", hardware="mi300x",
        baseline_throughput=600.0,
    )
    j1.append_entry(JournalEntry(
        phase="EXPLORE", iter=1, kind=KIND_BACKEND,
        change="--attention-backend X", outcome=OUTCOME_KEEP, gain_pct=12.0,
    ))
    j1.finalize(final_throughput=900.0, total_gain_pct=50.0)

    j2 = Journal.load_or_create(
        session_dir, session_id="sid-1", model="m", hardware="mi300x",
        baseline_throughput=600.0,
    )
    assert j2.final_throughput == 900.0
    assert j2.total_gain_pct == 50.0
    assert len(j2.entries) == 1
    assert j2.entries[0].outcome == OUTCOME_KEEP
    assert j2.entries[0].gain_pct == 12.0


def test_load_or_create_keeps_existing_header_when_caller_passes_defaults(
    session_dir: Path,
):
    """Header fields from disk win over empty-string defaults on resume."""
    j1 = Journal.load_or_create(
        session_dir, session_id="sid-1", model="m", hardware="mi300x",
        baseline_throughput=700.0,
    )
    j1.append_entry(JournalEntry(
        phase="EXPLORE", iter=1, kind=KIND_PARAM,
        change="--max-num-batched-tokens 16384", outcome=OUTCOME_KEEP,
    ))
    j2 = Journal.load_or_create(
        session_dir, session_id="", model="", hardware="", baseline_throughput=0.0,
    )
    assert j2.session_id == "sid-1"
    assert j2.model == "m"
    assert j2.hardware == "mi300x"
    assert j2.baseline_throughput == 700.0


def test_load_or_create_recovers_from_corrupt_file(
    session_dir: Path,
):
    path = session_dir / "reports" / JOURNAL_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    j = Journal.load_or_create(
        session_dir, session_id="sid-x", model="m", hardware="h",
    )
    assert j.entries == []
    assert j.session_id == "sid-x"


# Journal — mutation + persistence
def test_append_entry_flushes_to_disk(session_dir: Path):
    j = Journal.load_or_create(
        session_dir, session_id="s", model="m", hardware="h",
        baseline_throughput=600.0,
    )
    j.append_entry(JournalEntry(
        phase="EXPLORE", iter=1, kind=KIND_BACKEND, change="X",
        outcome=OUTCOME_KEEP, gain_pct=10.0,
    ))
    blob = json.loads(j.path.read_text(encoding="utf-8"))
    assert blob["entries"][0]["change"] == "X"
    assert blob["entries"][0]["gain_pct"] == 10.0
    assert blob["entries"][0]["ts"]  # auto-stamped


def test_append_entry_dedupes_on_resume_replay(session_dir: Path):
    j = Journal.load_or_create(
        session_dir, session_id="s", model="m", hardware="h",
    )
    e = JournalEntry(
        phase="EXPLORE", iter=1, kind=KIND_BACKEND, change="X",
        outcome=OUTCOME_KEEP, gain_pct=10.0,
    )
    assert j.append_entry(e) is True
    assert j.append_entry(e) is False  # dedupe
    assert len(j.entries) == 1


def test_append_entry_dedupe_per_variant(session_dir: Path):
    """variant_name breaks the dedupe tie for two variants of the same round."""
    j = Journal.load_or_create(
        session_dir, session_id="s", model="m", hardware="h",
    )
    base_kwargs = dict(
        phase="EXPLORE", iter=2, kind=KIND_PARAM,
        change="--max-num-batched-tokens", outcome=OUTCOME_REVERT,
    )
    assert j.append_entry(JournalEntry(variant_name="v_8k", **base_kwargs))
    assert j.append_entry(JournalEntry(variant_name="v_16k", **base_kwargs))
    assert len(j.entries) == 2


def test_append_entry_dedupe_per_task_id(session_dir: Path):
    """``task_id`` breaks the dedupe tie for two same-kind tasks in one tick."""
    from inference_optimizer.orchestrator.optimization_journal import (
        KIND_KERNEL_FILE,
    )
    j = Journal.load_or_create(
        session_dir, session_id="s", model="m", hardware="h",
    )
    base_kwargs = dict(
        phase="KERNEL", iter=5, kind=KIND_KERNEL_FILE,
        change="kernel_opt", outcome=OUTCOME_KEEP,
    )
    assert j.append_entry(JournalEntry(task_id="task-1", **base_kwargs))
    assert j.append_entry(JournalEntry(task_id="task-2", **base_kwargs))
    assert len(j.entries) == 2
    # Re-appending an identical (task_id) row is treated as resume replay.
    assert not j.append_entry(JournalEntry(task_id="task-1", **base_kwargs))
    assert len(j.entries) == 2


def test_finalize_updates_only_summary_fields(session_dir: Path):
    j = Journal.load_or_create(
        session_dir, session_id="s", model="m", hardware="h",
        baseline_throughput=600.0,
    )
    j.append_entry(JournalEntry(
        phase="EXPLORE", iter=1, kind=KIND_BACKEND, change="X",
        outcome=OUTCOME_KEEP, gain_pct=10.0,
    ))
    j.finalize(final_throughput=900.0, total_gain_pct=50.0)
    blob = json.loads(j.path.read_text(encoding="utf-8"))
    assert blob["final_throughput"] == 900.0
    assert blob["total_gain_pct"] == 50.0
    assert blob["baseline_throughput"] == 600.0
    assert len(blob["entries"]) == 1


def test_finalize_with_partial_args_only_updates_given_fields(session_dir: Path):
    j = Journal.load_or_create(
        session_dir, session_id="s", model="m", hardware="h",
    )
    j.finalize(total_gain_pct=44.9)
    assert j.total_gain_pct == 44.9
    assert j.final_throughput is None
    j.finalize(final_throughput=875.0)
    assert j.total_gain_pct == 44.9
    assert j.final_throughput == 875.0


def test_update_baseline_ignores_non_positive(session_dir: Path):
    j = Journal.load_or_create(
        session_dir, session_id="s", model="m", hardware="h",
        baseline_throughput=600.0,
    )
    j.update_baseline(0.0)
    assert j.baseline_throughput == 600.0
    j.update_baseline(-5.0)
    assert j.baseline_throughput == 600.0
    j.update_baseline(700.0)
    assert j.baseline_throughput == 700.0


def test_to_dict_strips_none_in_entries(session_dir: Path):
    j = Journal.load_or_create(
        session_dir, session_id="s", model="m", hardware="h",
    )
    j.append_entry(JournalEntry(
        phase="EXPLORE", iter=1, kind=KIND_BACKEND, change="X",
        outcome=OUTCOME_KEEP, gain_pct=10.0,
    ))
    blob = json.loads(j.path.read_text(encoding="utf-8"))
    e = blob["entries"][0]
    # error_class / reason were None → stripped.
    assert "error_class" not in e
    assert "reason" not in e
    assert e["gain_pct"] == 10.0


# classify_change_kind / summarize_change vocab
def test_classify_change_kind_recognises_top_level_kinds():
    assert classify_change_kind("kernel_opt") == KIND_KERNEL_FILE
    assert classify_change_kind("integrate") == "integrate"
    assert classify_change_kind("baseline") == "baseline"
    assert classify_change_kind("profile") == "profile"
    assert classify_change_kind("anything_else") == KIND_OTHER


def test_classify_change_kind_explore_variant_dimensions():
    assert classify_change_kind("explore", {"extra_envs": {"X": "1"}}) == KIND_ENV
    assert classify_change_kind(
        "explore",
        {"extra_sglang_args": "--attention-backend FOO"},
    ) == KIND_BACKEND
    assert classify_change_kind(
        "explore",
        {"extra_sglang_args": "--max-num-batched-tokens 8192"},
    ) == KIND_PARAM
    assert classify_change_kind("explore", {}) == KIND_OTHER


def test_summarize_change_prefers_variant_args_and_envs():
    s = summarize_change(
        "explore",
        {
            "extra_sglang_args": "--attention-backend AITER",
            "extra_envs": {"K": "1"},
        },
    )
    assert "--attention-backend AITER" in s
    assert "K=1" in s


def test_summarize_change_falls_back_to_task_kind():
    assert summarize_change("baseline") == "baseline"
    assert summarize_change("") == "(unknown)"


def test_operation_kind_for_maps_kind_and_action():
    from inference_optimizer.orchestrator.optimization_journal import (
        operation_kind_for,
    )
    # Explore sub-kinds pass through.
    assert operation_kind_for("explore", "backend") == "backend"
    assert operation_kind_for("explore", "param") == "param"
    assert operation_kind_for("explore", "env") == "env"
    # Kernel kinds rename to the action labels dashboards filter on.
    assert operation_kind_for("kernel_opt", "kernel_file") == "kernel_opt"
    assert operation_kind_for("integrate", "integrate") == "kernel_integrate"
    # No / other kind falls back to the action.
    assert operation_kind_for("roofline", "") == "roofline"
    assert operation_kind_for("sweep", "other") == "sweep"
    assert operation_kind_for("", "") == "other"


def test_proposer_for_resolves_provenance():
    from inference_optimizer.orchestrator.optimization_journal import proposer_for
    assert proposer_for("specialist:serving_specialist") == "specialist:serving_specialist"
    assert proposer_for("llm_direct") == "orchestration"
    assert proposer_for("default_grid") == "grid"
    assert proposer_for("legacy:backends") == "orchestration"
    assert proposer_for("") == "orchestration"


def test_journal_entry_roundtrips_proposer_and_metrics():
    from inference_optimizer.orchestrator.optimization_journal import JournalEntry
    e = JournalEntry(
        phase="EXPLORE", iter=1, kind="backend", change="x", outcome="KEEP",
        provenance="specialist:serving_specialist", scope="domain",
        fingerprint="fp123", metrics={"runtime_sec": 42.0},
    )
    d = e.to_dict()
    assert d["provenance"] == "specialist:serving_specialist"
    assert d["scope"] == "domain"
    assert d["fingerprint"] == "fp123"
    assert d["metrics"] == {"runtime_sec": 42.0}
    back = JournalEntry.from_dict(d)
    assert back.provenance == "specialist:serving_specialist"
    assert back.metrics == {"runtime_sec": 42.0}
    # Empty metrics dict is stripped from the serialized form.
    assert "metrics" not in JournalEntry(
        phase="P", iter=0, kind="baseline", change="b", outcome="KEEP",
    ).to_dict()
