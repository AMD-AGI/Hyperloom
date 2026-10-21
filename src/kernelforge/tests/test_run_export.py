# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``kernelforge export-runs`` on plain forge-loop runs: the document a run's ``forge_experiments`` becomes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from hyperloom_kb.collect import collect
from hyperloom_kb.schema import TEXT_MAX_BYTES
from kernelforge.cli import main
from kernelforge.run_export.document import normalize

CAMPAIGN = "5f1c0a9e2b7d4c38a6e0f1b2c3d4e5f6"
WORKSPACE = "forge_loop_triton_softmax_1760500000"
GATE_TAG = "[gate edits=2 pass=True end=agent_stopped turns=17 mean_case_speedup=1.180000x raw_mean=0.0410ms]"
SESSION_RATIONALE = "Two passes over each row cost a second read. Fusing them halves the traffic."
ROUND_PLAN = """I profiled the kernel first; here is the plan.

# Fuse the row reduction

Small rows are bound by launch latency. This plan fuses the max and sum passes into one sweep.
It keeps the output layout unchanged. This fourth sentence is past the excerpt.
"""
LANE_PLAN = "# Lane two: vectorize loads\n\nUse 128-bit loads for the input rows. Keep the reduction as it is.\n"
HEADLINE = "Fuse the max and sum passes into one sweep"
MERGE_PLAN = f"stacked candidates 1+2: {HEADLINE} | Use 128-bit loads for the input rows"
SUMMARY_PLAN = """Summary: Small rows are bound by launch latency. Each row is read twice.
This plan fuses the max and sum passes into one sweep. The small cases should gain most.

# Fuse the row reduction

The body of the plan is not its summary.
"""
SUMMARY_LANE_PLAN = (
    "Summary: Loads are 32-bit. This lane widens them.\n\n# Lane two: vectorize loads\n\nUse 128-bit loads.\n"
)
CRITIC_FALLBACK_PLAN = """The Orchestration revision was unavailable. Treat the review below as binding.

## Critic review
Summary: The critic's own summary is not the plan's.

## Original draft
Summary: Small rows are bound by launch latency. This plan fuses the passes.

# Fuse the row reduction
Step one.
"""
SESSION_LESSON = (
    "One sweep saved a global read. Small rows gained most. More.\n"
    "HELD-FIXED: the output layout.\n"
    "NEGATIVES: a second pass over the row.\n"
)
LANE_LESSON = (
    "The loads were 32-bit. I widened them to 128-bit.\n\nHELD-FIXED: the reduction.\nNEGATIVES: scalar loads.\n"
)
KB_RECORD = "kernel:forge-loop:softmax:unknown:unknown:triton:mi300x/rewrite-softmax-92fe883d6dc2-e45ae0d5646e"
BASELINE_COMMIT = "ce146831869e122424b2a67038ae466565882a3d"
KEPT_COMMIT = "761247d9b335287cd89902de285dbee4c0e0d96b"
BASELINE_REPORT = "# Analysis report\n\nEach row is read twice; both passes are bound by memory bandwidth."
KEPT_REPORT = "# Analysis report\n\nThe fused sweep reads each row once; small rows are now bound by launch latency."


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_json(path: Path, value: Any) -> Path:
    return _write(path, json.dumps(value))


def _candidate(experiments: Path, iteration: int, **meta: Any) -> Path:
    """``candidates/iter_NNN`` as ``CandidateArchive.record`` writes it."""

    directory = experiments / "candidates" / f"iter_{iteration:03d}"
    _write(directory / "softmax_kernel.py", f"# candidate {iteration}\n")
    _write(directory / "change.diff", "--- a/softmax_kernel.py\n+++ b/softmax_kernel.py\n")
    record = {
        "complete": True,
        "iteration": iteration,
        "timestamp": "2026-10-15 10:12:00",
        "commit_hash": "",
        "kept": False,
        "validation_passed": True,
        "baseline_wall_ms": 0.052,
        "best_wall_ms_before": 0.052,
        "best_mean_case_speedup_before": 1.0,
        "files": {"kernel": "softmax_kernel.py", "diff": "change.diff", "profile": None, "validation": None},
        **meta,
    }
    _write_json(directory / "meta.json", record)
    return directory


def _plain_run(
    root: Path,
    name: str = WORKSPACE,
    campaign_id: str = CAMPAIGN,
    *,
    round_plan: str = ROUND_PLAN,
    lane_plan: str = LANE_PLAN,
) -> Path:
    """A workspace after a short forge-loop: a session that kept, a lane that reverted, and a merge."""

    experiments = root / name / "forge_experiments"
    config = {
        "operator_name": "_softmax_kernel",
        "framework": "unknown",
        "kernel_backend": "triton",
        "gpu_type": "mi300x",
        "kernel_path": "softmax_kernel.py",
    }
    _write_json(experiments / "campaign_config.json", config)
    _write_json(experiments / "run_state.json", {"campaign_id": campaign_id, "iteration": 3})
    events = [
        {"ts": "2026-10-15 09:58:00", "type": "iteration_started", "iter": 1},
        {"ts": "2026-10-15 10:00:00", "type": "iteration_started", "iter": 1},
        {"ts": "2026-10-15 10:04:30", "type": "iteration_result", "iter": 1, "decision": "KEEP"},
        {"ts": "2026-10-15 10:05:00", "type": "iteration_started", "iter": 2},
        {"ts": "2026-10-15 10:06:00", "type": "merge_attempt_staged", "iter": 2},
        {"ts": "2026-10-15 10:09:00", "type": "iteration_result", "iter": 2, "decision": "REVERT_PERF"},
    ]
    _write(experiments / "events.jsonl", "".join(json.dumps(event) + "\n" for event in events) + "{torn line\n")
    _write(experiments / "orchestration" / "iter_001" / "optimization_plan.md", round_plan)
    _write(experiments / "orchestration" / "iter_001" / "lane_002.md", lane_plan)
    plan_path = "forge_experiments/orchestration/iter_001/optimization_plan.md"
    _write_json(experiments / "handoffs" / "iter_001.json", {"optimization_plan_path": plan_path})
    _write(experiments / "lessons" / "iter_001.md", SESSION_LESSON)
    _write(experiments / "best" / "iter_001" / "softmax_kernel.py", "# best\n")
    _candidate(
        experiments,
        1,
        decision="KEEP",
        kept=True,
        commit_hash="abc1234",
        wall_ms=0.044,
        mean_case_speedup=1.18,
        plan=HEADLINE,
        rationale=f"{GATE_TAG} {SESSION_RATIONALE}",
    )
    _candidate(
        experiments,
        2,
        decision="REVERT_PERF",
        wall_ms=0.051,
        mean_case_speedup=1.02,
        best_wall_ms_before=0.044,
        best_mean_case_speedup_before=1.18,
        plan=lane_plan.strip(),
        rationale="lane 2 of a fan-out round",
    )
    _candidate(
        experiments,
        3,
        decision="REVERT_PERF",
        wall_ms=0.047,
        mean_case_speedup=1.1,
        best_wall_ms_before=0.044,
        best_mean_case_speedup_before=1.18,
        plan=MERGE_PLAN,
        rationale="stacked iterations 1 and 2; no Implementer session",
    )
    return experiments


def _loop_log(
    path: Path, workspace: str, implementer: str = "claude / Claude-Opus-5", kb_record: str = KB_RECORD
) -> Path:
    """A forge-loop stdout log of the fixture's three iterations; it names the kernel by the absolute path the run
    had, and the KB record written after the KEEP and rewritten after the loop."""

    text = (
        f"  Implementer: {implementer}\n"
        "Starting autonomous iteration loop\n"
        f"  Kernel: /home/someone/kernelforge-runs/{workspace}/softmax_kernel.py\n"
        "--- Iteration 1 (best mean case speedup: 1.000000x, remaining: 110 min) ---\n"
        "  [KEEP] mean case speedup=1.180000x — NEW BEST (+18.0% vs previous best)\n"
        f"  [kb] experience written: {kb_record} (speedup 1.180)\n"
        "--- Iteration 2 (best mean case speedup: 1.180000x, remaining: 90 min) ---\n"
        "--- Iteration 3 (best mean case speedup: 1.180000x, remaining: 70 min) ---\n"
        "BUDGET EXHAUSTED after 3 iterations in this session\n"
        f"  [kb] experience written: {kb_record} (speedup 1.180)\n"
    )
    return _write(path, text)


def _units(root: Path) -> dict[str, dict[str, Any]]:
    return {unit["unit_id"]: unit for unit in normalize([root])["runs"]}


def _meta(experiments: Path, iteration: int) -> dict[str, Any]:
    return json.loads((experiments / "candidates" / f"iter_{iteration:03d}" / "meta.json").read_text(encoding="utf-8"))


def _ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"name": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _generation(experiments: Path, commit: str, report: str) -> Path:
    """A published Analysis generation for ``commit``, ``analysis/<commit>/generation-001``, with its report."""

    directory = experiments / "analysis" / commit / "generation-001"
    _write(directory / "report.md", f"\n{report}\n")
    _write_json(experiments / "analysis" / commit / "published.json", {"generation_root": directory.name})
    return directory


def _made_elsewhere(relative: str) -> str:
    """The absolute path ``forge_experiments/<relative>`` had where the fixture's run was made."""

    return f"/home/someone/kernelforge-runs/{WORKSPACE}/forge_experiments/{relative}"


def _planned_on(experiments: Path, iteration: int, bundle: str) -> None:
    """The context round ``iteration`` planned with, naming ``bundle`` as its Analysis bundle after the report it
    summarizes."""

    refs = [{"kind": "analysis_summary", "path": f"{bundle}/report.md"}, {"kind": "analysis_bundle", "path": bundle}]
    _write_json(experiments / "orchestration" / f"iter_{iteration:03d}" / "context.json", {"evidence_refs": refs})


def _round_after_the_keep(experiments: Path) -> None:
    """Round 4, planned after iteration 1's KEEP: a single session's plan, its handoff, and the candidate it built."""

    _write(
        experiments / "orchestration" / "iter_004" / "optimization_plan.md", "# Widen the loads\n\nUse 128-bit loads.\n"
    )
    plan_path = "forge_experiments/orchestration/iter_004/optimization_plan.md"
    _write_json(experiments / "handoffs" / "iter_004.json", {"optimization_plan_path": plan_path})
    _candidate(
        experiments,
        4,
        decision="REVERT_PERF",
        wall_ms=0.046,
        best_wall_ms_before=0.044,
        plan="Widen the loads",
        rationale="Wider loads cut the instruction count.",
    )


def test_a_single_session_candidate_becomes_a_complete_unit(tmp_path: Path) -> None:
    experiments = _plain_run(tmp_path)
    _loop_log(tmp_path / "forge-loop.log", WORKSPACE)
    unit = _units(tmp_path)[f"{WORKSPACE}/iteration-1"]

    candidate = experiments / "candidates" / "iter_001"
    orchestration = experiments / "orchestration" / "iter_001"
    assert unit == {
        "unit_id": f"{WORKSPACE}/iteration-1",
        "run_id": CAMPAIGN,
        "seq": 1,
        "identity": {"operator": "_softmax_kernel", "kernel": "softmax", "backend": "triton", "gpu": "mi300x"},
        "provenance": {
            "llm_backend": "claude",
            "llm_model": "Claude-Opus-5",
            "kb_record": KB_RECORD,
            "candidate_origin": "session",
            "meta": _meta(experiments, 1),
        },
        "started_at": "2026-10-15T10:00:00",
        "completed_at": "2026-10-15T10:04:30",
        "best_wall_ms_before": 0.052,
        "decision": "KEEP",
        "candidate_wall_ms": 0.044,
        "plan_file": str((orchestration / "optimization_plan.md").resolve()),
        "plan": (
            "Small rows are bound by launch latency. This plan fuses the max and sum passes into one sweep. "
            "It keeps the output layout unchanged."
        ),
        "source_ms": 0.052,
        "source_code": str((candidate / "softmax_kernel.py").resolve()),
        "change_diff_file": str((candidate / "change.diff").resolve()),
        "reasoning": SESSION_LESSON.strip(),
    }


def test_a_lane_candidate_takes_its_lane_plan_from_the_round_that_planned_it(tmp_path: Path) -> None:
    experiments = _plain_run(tmp_path)
    unit = _units(tmp_path)[f"{WORKSPACE}/iteration-2"]

    lane_plan = experiments / "orchestration" / "iter_001" / "lane_002.md"
    assert unit["provenance"] == {"candidate_origin": "lane", "meta": _meta(experiments, 2)}
    assert unit["plan_file"] == str(lane_plan.resolve())
    assert unit["plan"] == "Use 128-bit loads for the input rows. Keep the reduction as it is."
    assert (unit["started_at"], unit["completed_at"]) == ("2026-10-15T10:05:00", "2026-10-15T10:09:00")
    assert (unit["best_wall_ms_before"], unit["candidate_wall_ms"], unit["source_ms"]) == (0.044, 0.051, 0.052)
    assert "reasoning" not in unit


def test_a_merge_is_named_by_its_stacked_plans_and_dated_by_its_archive(tmp_path: Path) -> None:
    experiments = _plain_run(tmp_path)
    unit = _units(tmp_path)[f"{WORKSPACE}/iteration-3"]

    assert unit["provenance"] == {"candidate_origin": "merge", "meta": _meta(experiments, 3)}
    assert unit["plan"] == MERGE_PLAN
    assert (unit["best_wall_ms_before"], unit["candidate_wall_ms"]) == (0.044, 0.047)
    assert (unit["started_at"], unit["completed_at"]) == ("2026-10-15T10:12:00", "2026-10-15T10:12:00")
    assert "reasoning" not in unit and "plan_file" not in unit


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        pytest.param({"pristine_baseline_wall_ms": 0.104}, 0.104, id="pristine-recorded"),
        pytest.param({}, 0.052, id="no-pristine"),
    ],
)
def test_source_ms_is_the_pristine_baseline_the_run_state_records(
    tmp_path: Path, state: dict[str, Any], expected: float
) -> None:
    """A run given --baseline-json archives the time of the kernel its search started from in each meta.json; its
    run state keeps the original kernel's time."""

    experiments = _plain_run(tmp_path)
    _write_json(experiments / "run_state.json", {"campaign_id": CAMPAIGN, "iteration": 3, **state})

    units = _units(tmp_path)

    assert {units[f"{WORKSPACE}/iteration-{seq}"]["source_ms"] for seq in (1, 2, 3)} == {expected}


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [
        pytest.param(
            {"operator_name": "FusedMoeKernel", "framework": "aiter"},
            {"operator": "FusedMoeKernel", "kernel": "fused_moe", "framework": "aiter"},
            id="operator-recorded",
        ),
        pytest.param(
            {"operator_name": "", "target_functions": ["softmax", "_softmax_kernel"]},
            {"kernel": "softmax"},
            id="no-operator",
        ),
    ],
)
def test_identity_names_the_operator_and_the_kernel_kernelforge_files_its_records_under(
    tmp_path: Path, recorded: dict[str, Any], expected: dict[str, str]
) -> None:
    """The kernel is the operator's name normalized, or with no operator recorded, the name Kernel Forge derives from
    the run's target functions; neither the backend's version nor the inputs' dtype is recorded, so neither is
    named."""

    experiments = _plain_run(tmp_path)
    config = json.loads((experiments / "campaign_config.json").read_text(encoding="utf-8"))
    _write_json(experiments / "campaign_config.json", {**config, **recorded})

    identities = [unit["identity"] for unit in _units(tmp_path).values()]

    assert identities == [{**expected, "backend": "triton", "gpu": "mi300x"}] * 3


def test_the_driver_hash_joins_identity_only_when_the_campaign_recorded_one(tmp_path: Path) -> None:
    experiments = _plain_run(tmp_path)
    assert "driver_sha256" not in _units(tmp_path)[f"{WORKSPACE}/iteration-1"]["identity"]

    digest = "ab" * 32
    config = json.loads((experiments / "campaign_config.json").read_text(encoding="utf-8"))
    _write_json(experiments / "campaign_config.json", {**config, "driver_sha256": digest})

    identities = [unit["identity"]["driver_sha256"] for unit in _units(tmp_path).values()]

    assert identities == [digest, digest, digest]


def test_a_candidate_without_an_archived_diff_links_no_diff_file(tmp_path: Path) -> None:
    experiments = _plain_run(tmp_path)
    (experiments / "candidates" / "iter_003" / "change.diff").unlink()

    units = _units(tmp_path)

    assert units[f"{WORKSPACE}/iteration-1"]["change_diff_file"].endswith("change.diff")
    assert "change_diff_file" not in units[f"{WORKSPACE}/iteration-3"]


def test_each_log_attaches_to_the_run_whose_kernel_it_announces(tmp_path: Path) -> None:
    _plain_run(tmp_path / "a", name="softmax-a", campaign_id="campaign-a")
    _plain_run(tmp_path / "b", name="softmax-b", campaign_id="campaign-b")
    _loop_log(
        tmp_path / "a" / "forge-loop.log", "softmax-a", implementer="claude / Claude-Sonnet-5", kb_record="record-a"
    )
    _loop_log(tmp_path / "b" / "forge-loop.log", "softmax-b", implementer="codex / gpt-5", kb_record="record-b")

    units = _units(tmp_path)

    first, second = units["softmax-a/iteration-1"], units["softmax-b/iteration-1"]
    assert (first["run_id"], first["provenance"]["llm_model"], first["provenance"]["kb_record"]) == (
        "campaign-a",
        "Claude-Sonnet-5",
        "record-a",
    )
    assert (second["run_id"], second["provenance"]["llm_backend"], second["provenance"]["kb_record"]) == (
        "campaign-b",
        "codex",
        "record-b",
    )


def test_runs_whose_workspaces_share_a_name_keep_their_own_logs(tmp_path: Path) -> None:
    """Both runs' workspaces are ``ws``, and their logs name the paths the run directories had in a container, so only
    the run directory tells their logs apart."""

    models = {"a": "Claude-Opus-5", "b": "Claude-Sonnet-5"}
    for run, model in models.items():
        _plain_run(tmp_path / run, name="ws", campaign_id=f"campaign-{run}")
        _loop_log(
            tmp_path / run / "forge-loop.log", f"{run}/ws", implementer=f"claude / {model}", kb_record=f"record-{run}"
        )

    units = {(unit["run_id"], unit["seq"]): unit for unit in normalize([tmp_path])["runs"]}

    kept = [units[(f"campaign-{run}", 1)]["provenance"] for run in models]
    assert [(provenance["llm_model"], provenance["kb_record"]) for provenance in kept] == [
        ("Claude-Opus-5", "record-a"),
        ("Claude-Sonnet-5", "record-b"),
    ]


def test_only_a_kept_candidate_carries_the_kb_record_its_iteration_wrote(tmp_path: Path) -> None:
    _plain_run(tmp_path)
    _loop_log(tmp_path / "forge-loop.log", WORKSPACE)

    units = _units(tmp_path)

    records = [units[f"{WORKSPACE}/iteration-{seq}"]["provenance"].get("kb_record") for seq in (1, 2, 3)]
    assert records == [KB_RECORD, None, None]


def test_lane_copies_are_not_runs_and_unreadable_archives_say_why(tmp_path: Path) -> None:
    experiments = _plain_run(tmp_path)
    lane_copy = tmp_path / "forge-lanes-x1y2" / "2" / "forge_experiments"
    _write_json(lane_copy / "campaign_config.json", {"operator_name": "softmax_kernel"})
    _write_json(lane_copy / "run_state.json", {"campaign_id": CAMPAIGN})
    _write(experiments / "candidates" / "iter_004" / "meta.json", "{not json")
    unnamed = tmp_path / "unnamed" / "forge_experiments"
    _write_json(unnamed / "campaign_config.json", {"operator_name": "softmax_kernel"})
    _write_json(unnamed / "run_state.json", {"iteration": 0})

    units = _units(tmp_path)

    assert not any(unit_id.startswith("2/") for unit_id in units)
    assert units[f"{WORKSPACE}/iteration-4"]["read_error"].endswith("meta.json is not a readable JSON object")
    assert units["unnamed/session"] == {
        "unit_id": "unnamed/session",
        "read_error": "run_state.json names no campaign_id",
    }


def test_summary_lines_give_the_plan_and_the_whole_lesson_is_the_reasoning(tmp_path: Path) -> None:
    experiments = _plain_run(tmp_path, round_plan=SUMMARY_PLAN, lane_plan=SUMMARY_LANE_PLAN)
    orchestration = experiments / "orchestration" / "iter_001"
    _write(experiments / "lessons" / "iter_001.md", SESSION_LESSON)
    _write(experiments / "lessons" / "iter_002.md", LANE_LESSON)
    _write(orchestration / "optimization_plan.reasoning.md", "Summary: This log is not the reasoning.\n")
    _write(orchestration / "lane_002.reasoning.md", "Summary: Neither is this one.\n")

    units = _units(tmp_path)

    session, lane, merge = (units[f"{WORKSPACE}/iteration-{seq}"] for seq in (1, 2, 3))
    assert session["plan"] == (
        "Small rows are bound by launch latency. Each row is read twice. "
        "This plan fuses the max and sum passes into one sweep. The small cases should gain most."
    )
    assert session["reasoning"] == SESSION_LESSON.strip()
    assert "lesson" not in session and "lesson_file" not in session
    assert session["plan_file"] == str((orchestration / "optimization_plan.md").resolve())
    assert (lane["plan"], lane["reasoning"]) == ("Loads are 32-bit. This lane widens them.", LANE_LESSON.strip())
    assert lane["plan_file"] == str((orchestration / "lane_002.md").resolve())
    assert "reasoning" not in merge


@pytest.mark.parametrize("lesson", [None, "\n"])
def test_a_plan_is_summarized_from_its_draft_and_a_session_without_a_lesson_falls_back(
    tmp_path: Path, lesson: str | None
) -> None:
    experiments = _plain_run(tmp_path, round_plan=CRITIC_FALLBACK_PLAN)
    lesson_path = experiments / "lessons" / "iter_001.md"
    if lesson is None:
        lesson_path.unlink()
    else:
        _write(lesson_path, lesson)
    _write(experiments / "orchestration" / "iter_001" / "optimization_plan.reasoning.md", "Summary: Ignored.\n")

    session = _units(tmp_path)[f"{WORKSPACE}/iteration-1"]

    assert session["plan"] == "Small rows are bound by launch latency. This plan fuses the passes."
    assert session["reasoning"] == SESSION_RATIONALE


def test_each_candidate_carries_the_report_of_the_analysis_its_round_planned_on(tmp_path: Path) -> None:
    experiments = _plain_run(tmp_path)
    _round_after_the_keep(experiments)
    _generation(experiments, BASELINE_COMMIT, BASELINE_REPORT)
    _generation(experiments, KEPT_COMMIT, KEPT_REPORT)
    _planned_on(experiments, 1, _made_elsewhere(f"analysis/{BASELINE_COMMIT}/generation-001"))
    _planned_on(experiments, 4, _made_elsewhere(f"analysis/{KEPT_COMMIT}/generation-001"))

    units = _units(tmp_path)

    reports = {
        seq: (unit.get("analysis_report"), unit["provenance"].get("analysis_generation"))
        for seq, unit in ((seq, units[f"{WORKSPACE}/iteration-{seq}"]) for seq in (1, 2, 3, 4))
    }
    assert reports == {
        1: (BASELINE_REPORT, f"analysis/{BASELINE_COMMIT}/generation-001"),
        2: (BASELINE_REPORT, f"analysis/{BASELINE_COMMIT}/generation-001"),
        3: (None, None),
        4: (KEPT_REPORT, f"analysis/{KEPT_COMMIT}/generation-001"),
    }


@pytest.mark.parametrize(
    ("report", "omitted"),
    [
        pytest.param("x" * TEXT_MAX_BYTES, None, id="at-the-limit"),
        pytest.param(
            "é" * (TEXT_MAX_BYTES // 2) + "x",
            f"report.md is {TEXT_MAX_BYTES + 1} bytes, over the {TEXT_MAX_BYTES} a text field holds",
            id="one-byte-over",
        ),
    ],
)
def test_a_report_a_text_field_cannot_hold_is_left_out_and_its_provenance_says_why(
    tmp_path: Path, report: str, omitted: str | None
) -> None:
    experiments = _plain_run(tmp_path)
    _generation(experiments, BASELINE_COMMIT, report)
    _planned_on(experiments, 1, _made_elsewhere(f"analysis/{BASELINE_COMMIT}/generation-001"))

    session = _units(tmp_path)[f"{WORKSPACE}/iteration-1"]

    assert session.get("analysis_report") == (report if omitted is None else None)
    assert session["provenance"]["analysis_generation"] == f"analysis/{BASELINE_COMMIT}/generation-001"
    assert session["provenance"].get("analysis_report_omitted") == omitted


@pytest.mark.parametrize(
    "bundle",
    [
        pytest.param(None, id="no-context"),
        pytest.param(_made_elsewhere(f"analysis/work/{BASELINE_COMMIT}"), id="analysis-still-at-work"),
        pytest.param(_made_elsewhere(f"analysis/{KEPT_COMMIT}/generation-001"), id="generation-not-in-this-run"),
    ],
)
def test_a_round_that_planned_on_no_published_generation_gives_no_report(tmp_path: Path, bundle: str | None) -> None:
    experiments = _plain_run(tmp_path)
    _generation(experiments, BASELINE_COMMIT, BASELINE_REPORT)
    if bundle is not None:
        _planned_on(experiments, 1, bundle)

    session = _units(tmp_path)[f"{WORKSPACE}/iteration-1"]

    assert "analysis_report" not in session
    assert "analysis_generation" not in session["provenance"]


def test_the_report_collects_as_text_and_no_analysis_file_is_linked(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    experiments = _plain_run(logs)
    generation = _generation(experiments, BASELINE_COMMIT, BASELINE_REPORT)
    _write(generation / "cases" / "M1024_N256-2bb066cf" / "profile" / "pmc_perf.csv", "Dispatch_ID,SQ_WAVES\n0,64\n")
    _planned_on(experiments, 1, _made_elsewhere(f"analysis/{BASELINE_COMMIT}/generation-001"))
    output = tmp_path / "runs.json"
    assert CliRunner().invoke(main, ["export-runs", str(logs), "-o", str(output)]).exit_code == 0

    [row] = collect("kernelforge-v1", output, dry_run=True).to_dict()["collected"]

    experience = row["experience"]
    assert experience["reflection"]["analysis_report"] == BASELINE_REPORT
    assert experience["provenance"]["extra"]["analysis_generation"] == f"analysis/{BASELINE_COMMIT}/generation-001"
    linked = [
        value["name"]
        for category in ("rationale", "change", "reflection")
        for value in experience[category].values()
        if isinstance(value, dict)
    ]
    assert linked == ["optimization_plan.md", "softmax_kernel.py", "change.diff"]
    assert "raw_logs" not in experience


def test_export_runs_writes_the_document(tmp_path: Path) -> None:
    _plain_run(tmp_path / "logs")
    output = tmp_path / "out" / "runs.json"

    result = CliRunner().invoke(main, ["export-runs", str(tmp_path / "logs"), "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert result.output == f"wrote 3 runs to {output}\n"
    assert json.loads(output.read_text(encoding="utf-8")) == normalize([tmp_path / "logs"])


def test_the_exported_document_collects_through_the_packaged_mapping(tmp_path: Path) -> None:
    _plain_run(tmp_path / "logs")
    _loop_log(tmp_path / "logs" / "forge-loop.log", WORKSPACE)
    output = tmp_path / "runs.json"
    assert CliRunner().invoke(main, ["export-runs", str(tmp_path / "logs"), "-o", str(output)]).exit_code == 0

    report = collect("kernelforge-v1", output, dry_run=True)

    [row] = report.collected
    assert (row.unit_id, row.experience.run_id, row.experience.seq) == (f"{WORKSPACE}/iteration-1", CAMPAIGN, 1)
    assert row.experience.objective == "kernel_time_ms@v1"
    assert row.experience.identity == {
        "operator": "_softmax_kernel",
        "kernel": "softmax",
        "backend": "triton",
        "gpu": "mi300x",
    }
    assert row.experience.baseline == {"best_wall_ms_before": 0.052, "source_ms": 0.052}
    assert row.experience.outcome == {"decision": "KEEP", "candidate_wall_ms": 0.044}
    assert row.experience.rationale["reasoning"] == SESSION_LESSON.strip()
    assert "lesson" not in row.experience.reflection
    assert row.experience.provenance.extra["llm_model"] == "Claude-Opus-5"
    assert row.experience.provenance.extra["meta"] == _meta(tmp_path / "logs" / WORKSPACE / "forge_experiments", 1)
    assert row.experience.raw_logs == ()
    assert {row.unit_id: row.reason.split(":")[0] for row in report.skipped} == {
        f"{WORKSPACE}/iteration-2": "No reasoning, which every Experience requires",
        f"{WORKSPACE}/iteration-3": "No reasoning, which every Experience requires",
    }


def test_a_run_collects_its_lanes_with_each_lesson_as_their_reasoning(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    experiments = _plain_run(logs, round_plan=SUMMARY_PLAN, lane_plan=SUMMARY_LANE_PLAN)
    orchestration = experiments / "orchestration" / "iter_001"
    _write(experiments / "lessons" / "iter_002.md", LANE_LESSON)
    output = tmp_path / "runs.json"
    assert CliRunner().invoke(main, ["export-runs", str(logs), "-o", str(output)]).exit_code == 0

    report = collect("kernelforge-v1", output, dry_run=True).to_dict()

    experiences = {row["unit_id"]: row["experience"] for row in report["collected"]}
    assert sorted(experiences) == [f"{WORKSPACE}/iteration-1", f"{WORKSPACE}/iteration-2"]
    session, lane = experiences[f"{WORKSPACE}/iteration-1"], experiences[f"{WORKSPACE}/iteration-2"]
    assert session["rationale"]["reasoning"] == SESSION_LESSON.strip()
    assert "lesson" not in session["reflection"]
    assert lane["rationale"] == {
        "reasoning": LANE_LESSON.strip(),
        "plan": "Loads are 32-bit. This lane widens them.",
        "plan_file": _ref(orchestration / "lane_002.md"),
    }
    assert lane["change"]["change_diff_file"]["name"] == "change.diff"
    assert (lane["baseline"], lane["outcome"]) == (
        {"best_wall_ms_before": 0.044, "source_ms": 0.052},
        {"decision": "REVERT_PERF", "candidate_wall_ms": 0.051},
    )
    origins = [experience["provenance"]["extra"]["candidate_origin"] for experience in (session, lane)]
    assert origins == ["session", "lane"]
    assert "raw_logs" not in session and "raw_logs" not in lane
    assert [(row["unit_id"], row["reason"].split(":")[0]) for row in report["skipped"]] == [
        (f"{WORKSPACE}/iteration-3", "No reasoning, which every Experience requires")
    ]


def test_export_runs_names_runs_by_their_workspace_from_a_relative_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plain_run(tmp_path)
    monkeypatch.chdir(tmp_path / WORKSPACE)

    result = CliRunner().invoke(main, ["export-runs", ".", "-o", str(tmp_path / "runs.json")])

    assert result.exit_code == 0, result.output
    runs = json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))["runs"]
    assert [run["unit_id"] for run in runs] == [f"{WORKSPACE}/iteration-{seq}" for seq in (1, 2, 3)]


@pytest.mark.parametrize("unreadable", ["plan", "log", "candidate record", "events log", "best bundle"])
def test_export_runs_stops_naming_a_path_it_cannot_read(tmp_path: Path, unreadable: str) -> None:
    """Kernel Forge creates its best bundles with mkdtemp, so a run copied out of a container holds directories its
    user cannot list as well as files it cannot read."""

    if os.geteuid() == 0:
        pytest.skip("root reads every file whatever its mode")
    logs = tmp_path / "logs"
    experiments = _plain_run(logs)
    paths = {
        "plan": experiments / "orchestration" / "iter_001" / "optimization_plan.md",
        "log": _loop_log(logs / "forge-loop.log", WORKSPACE),
        "candidate record": experiments / "candidates" / "iter_001" / "meta.json",
        "events log": experiments / "events.jsonl",
        "best bundle": experiments / "best" / "iter_001",
    }
    path = paths[unreadable]
    mode = path.stat().st_mode
    output = tmp_path / "runs.json"
    path.chmod(0)
    try:
        result = CliRunner().invoke(main, ["export-runs", str(logs), "-o", str(output)])
    finally:
        path.chmod(mode)

    assert result.exit_code == 1
    assert f"Error: cannot read {path}: Permission denied." in result.output
    assert "chown" in result.output
    assert not output.exists()


def test_export_runs_stops_on_a_log_two_runs_fit_equally_naming_both(tmp_path: Path) -> None:
    """The log names only the workspace ``ws``, which both runs have, so it cannot be given to either."""

    logs = tmp_path / "logs"
    for run in ("a", "b"):
        _plain_run(logs / run, name="ws", campaign_id=f"campaign-{run}")
    log = _loop_log(logs / "forge-loop.log", "ws")
    output = tmp_path / "runs.json"

    result = CliRunner().invoke(main, ["export-runs", str(logs), "-o", str(output)])

    assert result.exit_code == 1
    assert f"Error: {log} names" in result.output and "export them from separate log roots" in result.output
    assert all(str(logs / run / "ws") in result.output for run in ("a", "b"))
    assert not output.exists()


def test_export_runs_refuses_a_missing_root_or_output(tmp_path: Path) -> None:
    runner = CliRunner()

    missing_root = runner.invoke(main, ["export-runs", str(tmp_path / "absent"), "-o", str(tmp_path / "runs.json")])
    no_output = runner.invoke(main, ["export-runs", str(tmp_path)])
    no_root = runner.invoke(main, ["export-runs", "-o", str(tmp_path / "runs.json")])

    assert [missing_root.exit_code, no_output.exit_code, no_root.exit_code] == [2, 2, 2]
    assert "does not exist" in missing_root.output
    assert "--output" in no_output.output
    assert not (tmp_path / "runs.json").exists()
