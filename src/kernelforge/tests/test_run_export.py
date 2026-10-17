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
from kernelforge.cli import main
from kernelforge.orchestrator.summary_line import reasoning_log_section
from kernelforge.run_export.document import normalize
from kernelforge.run_export.forge_loop import AmbiguousOwnership

CAMPAIGN = "5f1c0a9e2b7d4c38a6e0f1b2c3d4e5f6"
WORKSPACE = "forge_loop_triton_softmax_1760500000"
GATE_TAG = "[gate edits=2 pass=True end=agent_stopped turns=17 mean_case_speedup=1.180000x raw_mean=0.0410ms]"
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
SESSION_LOG = "Summary: Each row was read twice. I fused the two passes into one sweep.\n\n## Evidence\nThe profile.\n"
LANE_LOG = "Summary: The loads were 32-bit. I widened them to 128-bit.\n\n## Measurements\nSmall rows gained.\n"
#: Where the run directory was mounted in the container the run was made in, which is where its transcripts name files.
CONTAINER_RUN = "/home/someone/kernelforge-runs/softmax-2h"
KB_RECORD = "kernel:forge-loop:softmax:unknown:unknown:triton:mi300x/rewrite-softmax-92fe883d6dc2-e45ae0d5646e"
PROJECTS = Path(".claude") / "projects"


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
        "operator_name": "softmax_kernel",
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
    _write(experiments / "lessons" / "iter_001.md", "One sweep saved a global read. Small rows gained most. More.\n")
    _write(experiments / "best" / "iter_001" / "softmax_kernel.py", "# best\n")
    _candidate(
        experiments,
        1,
        decision="KEEP",
        kept=True,
        commit_hash="abc1234",
        mean_case_speedup=1.18,
        plan=HEADLINE,
        rationale=f"{GATE_TAG} Two passes over each row cost a second read. Fusing them halves the traffic.",
    )
    _candidate(
        experiments,
        2,
        decision="REVERT_PERF",
        mean_case_speedup=1.02,
        plan=lane_plan.strip(),
        rationale="lane 2 of a fan-out round",
    )
    _candidate(
        experiments,
        3,
        decision="REVERT_PERF",
        mean_case_speedup=1.1,
        best_mean_case_speedup_before=1.18,
        plan=MERGE_PLAN,
        rationale="stacked iterations 1 and 2; no Implementer session",
    )
    return experiments


def _loop_log(path: Path, workspace: str, implementer: str = "claude / Claude-Opus-5") -> Path:
    """A forge-loop stdout log of the fixture's three iterations; it names the kernel by the absolute path the run
    had, and the KB record written after the KEEP and rewritten after the loop."""

    text = (
        f"  Implementer: {implementer}\n"
        "Starting autonomous iteration loop\n"
        f"  Kernel: /home/someone/kernelforge-runs/{workspace}/softmax_kernel.py\n"
        "--- Iteration 1 (best mean case speedup: 1.000000x, remaining: 110 min) ---\n"
        "  [KEEP] mean case speedup=1.180000x — NEW BEST (+18.0% vs previous best)\n"
        f"  [kb] experience written: {KB_RECORD} (speedup 1.180)\n"
        "--- Iteration 2 (best mean case speedup: 1.180000x, remaining: 90 min) ---\n"
        "--- Iteration 3 (best mean case speedup: 1.180000x, remaining: 70 min) ---\n"
        "BUDGET EXHAUSTED after 3 iterations in this session\n"
        f"  [kb] experience written: {KB_RECORD} (speedup 1.180)\n"
    )
    return _write(path, text)


def _container_log(plan: str, workspace: str = WORKSPACE, iteration: int = 1) -> str:
    """The reasoning log of ``orchestration/iter_NNN/<plan>.md`` as a session in the run's container named it."""

    return f"{CONTAINER_RUN}/{workspace}/forge_experiments/orchestration/iter_{iteration:03d}/{plan}.reasoning.md"


def _transcript(path: Path, prompt: str, *messages: str, started: str = "2026-10-15T10:01:00.000Z") -> Path:
    """A Claude Code transcript: the session's first prompt, then the agent's text messages."""

    records: list[dict[str, Any]] = [{"type": "user", "timestamp": started, "message": {"content": prompt}}]
    for number, text in enumerate(messages):
        records.append(
            {"type": "assistant", "message": {"id": f"msg-{number}", "content": [{"type": "text", "text": text}]}}
        )
    return _write(path, "".join(json.dumps(record) + "\n" for record in records))


def _implementer_prompt(log: str, experience: str = "(none)\n") -> str:
    """An Implementer's first prompt, which names its reasoning log as KernelForge's invocation section does, then
    the prior experience it is shown."""

    return f"## Target\nsoftmax_kernel.py\n{reasoning_log_section(log)}## Prior experience\n{experience}"


def _units(root: Path) -> dict[str, dict[str, Any]]:
    return {unit["unit_id"]: unit for unit in normalize([root])["runs"]}


def _resolved(*paths: Path) -> list[str]:
    return [str(path.resolve()) for path in paths]


def _ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"name": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def test_a_single_session_candidate_becomes_a_complete_unit(tmp_path: Path) -> None:
    experiments = _plain_run(tmp_path)
    log = _loop_log(tmp_path / "forge-loop.log", WORKSPACE)
    unit = _units(tmp_path)[f"{WORKSPACE}/iteration-1"]

    candidate = experiments / "candidates" / "iter_001"
    orchestration = experiments / "orchestration" / "iter_001"
    assert unit == {
        "unit_id": f"{WORKSPACE}/iteration-1",
        "kind": "iteration",
        "run_id": CAMPAIGN,
        "seq": 1,
        "identity": {"kernel": "softmax_kernel", "backend": "triton", "gpu": "mi300x"},
        "provenance": {
            "llm_backend": "claude",
            "llm_model": "Claude-Opus-5",
            "commit": "abc1234",
            "kb_record": KB_RECORD,
            "candidate_origin": "session",
        },
        "log_files": _resolved(
            log,
            experiments / "campaign_config.json",
            experiments / "run_state.json",
            candidate / "change.diff",
            candidate / "meta.json",
            candidate / "softmax_kernel.py",
            orchestration / "lane_002.md",
            orchestration / "optimization_plan.md",
            experiments / "best" / "iter_001" / "softmax_kernel.py",
            experiments / "handoffs" / "iter_001.json",
            experiments / "lessons" / "iter_001.md",
        ),
        "objective": "kernel_mean_case_speedup@v1",
        "started_at": "2026-10-15T10:00:00",
        "completed_at": "2026-10-15T10:04:30",
        "best_mean_case_speedup": 1.0,
        "decision": "KEEP",
        "mean_case_speedup": 1.18,
        "plan_file": str((orchestration / "optimization_plan.md").resolve()),
        "plan": (
            "Small rows are bound by launch latency. This plan fuses the max and sum passes into one sweep. "
            "It keeps the output layout unchanged."
        ),
        "source_ms": 0.052,
        "approach": HEADLINE,
        "source_code": str((candidate / "softmax_kernel.py").resolve()),
        "lesson_file": str((experiments / "lessons" / "iter_001.md").resolve()),
        "lesson": "One sweep saved a global read. Small rows gained most. More.",
        "reasoning": "Two passes over each row cost a second read. Fusing them halves the traffic.",
    }


def test_a_lane_candidate_takes_its_lane_plan_from_the_round_that_planned_it(tmp_path: Path) -> None:
    experiments = _plain_run(tmp_path)
    unit = _units(tmp_path)[f"{WORKSPACE}/iteration-2"]

    lane_plan = experiments / "orchestration" / "iter_001" / "lane_002.md"
    assert unit["provenance"] == {"candidate_origin": "lane"}
    assert unit["plan_file"] == str(lane_plan.resolve())
    assert unit["plan"] == "Use 128-bit loads for the input rows. Keep the reduction as it is."
    assert (unit["started_at"], unit["completed_at"]) == ("2026-10-15T10:05:00", "2026-10-15T10:09:00")
    assert str(lane_plan.resolve()) in unit["log_files"]
    assert "reasoning" not in unit and "approach" not in unit


def test_a_merge_is_named_by_its_stacked_plans_and_dated_by_its_archive(tmp_path: Path) -> None:
    _plain_run(tmp_path)
    unit = _units(tmp_path)[f"{WORKSPACE}/iteration-3"]

    assert unit["provenance"] == {"candidate_origin": "merge"}
    assert unit["approach"] == MERGE_PLAN
    assert unit["plan"] == MERGE_PLAN
    assert unit["best_mean_case_speedup"] == 1.18
    assert (unit["started_at"], unit["completed_at"]) == ("2026-10-15T10:12:00", "2026-10-15T10:12:00")
    assert "reasoning" not in unit and "plan_file" not in unit


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        pytest.param({"pristine_baseline_wall_ms": 0.104}, 0.104, id="pristine-recorded"),
        pytest.param({}, 0.052, id="no-pristine"),
    ],
)
def test_source_ms_is_the_pristine_baseline_every_score_divides_by(
    tmp_path: Path, state: dict[str, Any], expected: float
) -> None:
    """A run given --baseline-json archives the time of the kernel its search started from in each meta.json; its
    run state keeps the original kernel's time, which every score is relative to."""

    experiments = _plain_run(tmp_path)
    _write_json(experiments / "run_state.json", {"campaign_id": CAMPAIGN, "iteration": 3, **state})

    units = _units(tmp_path)

    assert {units[f"{WORKSPACE}/iteration-{seq}"]["source_ms"] for seq in (1, 2, 3)} == {expected}


def test_each_log_attaches_to_the_run_whose_kernel_it_announces(tmp_path: Path) -> None:
    _plain_run(tmp_path / "a", name="softmax-a", campaign_id="campaign-a")
    _plain_run(tmp_path / "b", name="softmax-b", campaign_id="campaign-b")
    log_a = _loop_log(tmp_path / "a" / "forge-loop.log", "softmax-a", implementer="claude / Claude-Sonnet-5")
    log_b = _loop_log(tmp_path / "b" / "forge-loop.log", "softmax-b", implementer="codex / gpt-5")

    units = _units(tmp_path)

    first, second = units["softmax-a/iteration-1"], units["softmax-b/iteration-1"]
    assert (first["run_id"], first["provenance"]["llm_model"]) == ("campaign-a", "Claude-Sonnet-5")
    assert (second["run_id"], second["provenance"]["llm_backend"]) == ("campaign-b", "codex")
    assert str(log_a.resolve()) in first["log_files"] and str(log_b.resolve()) not in first["log_files"]


def test_runs_whose_workspaces_share_a_name_keep_their_own_logs_and_transcripts(tmp_path: Path) -> None:
    """Both runs' workspaces are ``ws``, and their logs and transcripts name the paths the run directories had in a
    container, so only the run directory tells their evidence apart."""

    models = {"a": "Claude-Opus-5", "b": "Claude-Sonnet-5"}
    for run, model in models.items():
        _plain_run(tmp_path / run, name="ws", campaign_id=f"campaign-{run}")
        _loop_log(tmp_path / run / "forge-loop.log", f"{run}/ws", implementer=f"claude / {model}")
        _transcript(
            tmp_path / run / PROJECTS / f"-home-someone-kernelforge-runs-softmax-2h-{run}-ws" / "session.jsonl",
            _implementer_prompt(_container_log("optimization_plan", workspace=f"{run}/ws")),
            f"PLAN: Fuse the passes in run {run}",
        )

    units = {(unit["run_id"], unit["seq"]): unit for unit in normalize([tmp_path])["runs"]}

    for run, model in models.items():
        unit = units[(f"campaign-{run}", 1)]
        (other,) = set(models) - {run}
        assert (unit["provenance"]["llm_model"], unit["approach"]) == (model, f"Fuse the passes in run {run}")
        assert str((tmp_path / run / "forge-loop.log").resolve()) in unit["log_files"]
        assert not [path for path in unit["log_files"] if path.startswith(str((tmp_path / other).resolve()))]


@pytest.mark.parametrize("evidence", ["log", "transcript"])
def test_evidence_two_runs_fit_equally_stops_the_export_naming_both(tmp_path: Path, evidence: str) -> None:
    """The path names only the workspace ``ws``, which both runs have, so it cannot be given to either."""

    for run in ("a", "b"):
        _plain_run(tmp_path / run, name="ws", campaign_id=f"campaign-{run}")
    if evidence == "log":
        named = _loop_log(tmp_path / "forge-loop.log", "ws")
    else:
        named = _transcript(
            tmp_path / PROJECTS / "-work-ws" / "session.jsonl",
            _implementer_prompt("/work/ws/forge_experiments/orchestration/iter_001/optimization_plan.reasoning.md"),
            "PLAN: Fuse the passes",
        )

    with pytest.raises(AmbiguousOwnership) as caught:
        normalize([tmp_path])

    assert str(named) in str(caught.value)
    assert all(str(tmp_path / run / "ws") in str(caught.value) for run in ("a", "b"))


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
        "kind": "session",
        "read_error": "run_state.json names no campaign_id",
        "log_files": _resolved(unnamed / "campaign_config.json", unnamed / "run_state.json"),
    }


def test_summary_lines_give_the_plan_and_the_reasoning_and_each_reasoning_log_is_linked(tmp_path: Path) -> None:
    experiments = _plain_run(tmp_path, round_plan=SUMMARY_PLAN, lane_plan=SUMMARY_LANE_PLAN)
    orchestration = experiments / "orchestration" / "iter_001"
    session_log = _write(orchestration / "optimization_plan.reasoning.md", SESSION_LOG)
    lane_log = _write(orchestration / "lane_002.reasoning.md", LANE_LOG)

    units = _units(tmp_path)

    session, lane, merge = (units[f"{WORKSPACE}/iteration-{seq}"] for seq in (1, 2, 3))
    assert session["plan"] == (
        "Small rows are bound by launch latency. Each row is read twice. "
        "This plan fuses the max and sum passes into one sweep."
    )
    assert session["reasoning"] == "Each row was read twice. I fused the two passes into one sweep."
    assert session["reasoning_log"] == str(session_log.resolve())
    assert session["plan_file"] == str((orchestration / "optimization_plan.md").resolve())
    assert (lane["plan"], lane["reasoning"]) == (
        "Loads are 32-bit. This lane widens them.",
        "The loads were 32-bit. I widened them to 128-bit.",
    )
    assert [lane["plan_file"], lane["reasoning_log"]] == _resolved(orchestration / "lane_002.md", lane_log)
    assert {str(session_log.resolve()), str(lane_log.resolve())} <= set(session["log_files"]) & set(lane["log_files"])
    assert not {"reasoning", "reasoning_log", "reasoning_file"} & set(merge)


@pytest.mark.parametrize(
    "log",
    [pytest.param(None, id="no-log"), pytest.param("## Evidence\nThe profile showed two passes.\n", id="no-summary")],
)
def test_a_plan_is_summarized_from_its_draft_and_reasoning_without_a_summary_line_falls_back(
    tmp_path: Path, log: str | None
) -> None:
    experiments = _plain_run(tmp_path, round_plan=CRITIC_FALLBACK_PLAN)
    log_path = experiments / "orchestration" / "iter_001" / "optimization_plan.reasoning.md"
    if log is not None:
        _write(log_path, log)

    session = _units(tmp_path)[f"{WORKSPACE}/iteration-1"]

    assert session["plan"] == "Small rows are bound by launch latency. This plan fuses the passes."
    assert session["reasoning"] == "Two passes over each row cost a second read. Fusing them halves the traffic."
    assert session.get("reasoning_log") == (str(log_path.resolve()) if log is not None else None)


def test_transcripts_are_linked_by_the_reasoning_log_their_first_prompt_names(tmp_path: Path) -> None:
    _plain_run(tmp_path)
    lane_project = tmp_path / PROJECTS / "-home-someone-kernelforge-runs-softmax-2h-forge-lanes-ab12-2"
    lane = _transcript(
        lane_project / "lane.jsonl",
        _implementer_prompt(_container_log("lane_002")),
        "Loads were 32-bit. I widened them.\nPLAN: Widen the loads",
        "The wider loads held up.\nPLAN: Vectorize the row loads",
    )
    session = _transcript(
        tmp_path / PROJECTS / f"-home-someone-kernelforge-runs-softmax-2h-{WORKSPACE}" / "session.jsonl",
        _implementer_prompt(_container_log("optimization_plan")),
        f"Each row was read twice. I fused the passes.\nPLAN: {HEADLINE}",
    )
    other_run = _transcript(
        lane_project / "other-run.jsonl",
        _implementer_prompt(_container_log("lane_002", workspace="forge_loop_triton_softmax_1760599999")),
        "PLAN: Not this run's lane",
    )
    planner = _transcript(
        lane_project / "planner.jsonl", "Plan the round.", f"The lane will log to {_container_log('lane_002')}."
    )

    units = _units(tmp_path)

    session_unit, lane_unit, merge = (units[f"{WORKSPACE}/iteration-{seq}"] for seq in (1, 2, 3))
    assert lane_unit["approach"] == "Vectorize the row loads"
    assert lane_unit["reasoning"] == "Loads were 32-bit. I widened them. PLAN: Widen the loads"
    assert lane_unit["reasoning_file"] == (PROJECTS / lane_project.name / "lane.jsonl").as_posix()
    assert (session_unit["approach"], session_unit["reasoning"]) == (
        HEADLINE,
        "Each row was read twice. I fused the passes. PLAN: Fuse the max and sum passes into one sweep",
    )
    assert session_unit["reasoning_file"] == session.relative_to(tmp_path).as_posix()
    assert str(lane.resolve()) in lane_unit["log_files"] and str(session.resolve()) in session_unit["log_files"]
    unlinked = {str(path.resolve()) for path in (other_run, planner)}
    assert not unlinked & {path for unit in units.values() for path in unit["log_files"]}
    assert "reasoning_file" not in merge


def test_a_later_session_is_not_linked_by_the_earlier_log_its_prior_experience_names(tmp_path: Path) -> None:
    """A later session is shown an earlier iteration's lesson, whose record of the session's tool calls names the log
    that session kept."""

    _plain_run(tmp_path)
    project = tmp_path / PROJECTS / f"-home-someone-kernelforge-runs-softmax-2h-{WORKSPACE}"
    session = _transcript(
        project / "session.jsonl", _implementer_prompt(_container_log("optimization_plan")), f"PLAN: {HEADLINE}"
    )
    lesson = f"Recent provider progress (machine-captured):\n- tool: Edit {_container_log('optimization_plan')}\n"
    later = _transcript(
        project / "later.jsonl",
        _implementer_prompt(_container_log("optimization_plan", iteration=4), lesson),
        "PLAN: Split each row across two waves",
        started="2026-10-15T10:20:00.000Z",
    )

    unit = _units(tmp_path)[f"{WORKSPACE}/iteration-1"]

    assert (unit["approach"], unit["reasoning_file"]) == (HEADLINE, session.relative_to(tmp_path).as_posix())
    assert str(later.resolve()) not in unit["log_files"]


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
    assert row.experience.outcome == {"decision": "KEEP", "mean_case_speedup": 1.18}
    assert row.experience.provenance.extra["llm_model"] == "Claude-Opus-5"
    assert {row.unit_id: row.reason.split(":")[0] for row in report.skipped} == {
        f"{WORKSPACE}/iteration-2": "No reasoning, which every Experience requires",
        f"{WORKSPACE}/iteration-3": "No reasoning, which every Experience requires",
    }


def test_a_run_that_kept_reasoning_logs_collects_its_lanes_with_each_log_as_a_file(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    experiments = _plain_run(logs, round_plan=SUMMARY_PLAN, lane_plan=SUMMARY_LANE_PLAN)
    orchestration = experiments / "orchestration" / "iter_001"
    session_log = _write(orchestration / "optimization_plan.reasoning.md", SESSION_LOG)
    lane_log = _write(orchestration / "lane_002.reasoning.md", LANE_LOG)
    lane = _transcript(
        logs / PROJECTS / "-home-someone-kernelforge-runs-softmax-2h-forge-lanes-ab12-2" / "lane.jsonl",
        _implementer_prompt(_container_log("lane_002")),
        "PLAN: Vectorize the row loads",
    )
    output = tmp_path / "runs.json"
    assert CliRunner().invoke(main, ["export-runs", str(logs), "-o", str(output)]).exit_code == 0

    report = collect("kernelforge-v1", output, dry_run=True).to_dict()

    experiences = {row["unit_id"]: row["experience"] for row in report["collected"]}
    session, lane_experience = experiences[f"{WORKSPACE}/iteration-1"], experiences[f"{WORKSPACE}/iteration-2"]
    assert session["rationale"]["reasoning"] == "Each row was read twice. I fused the two passes into one sweep."
    assert session["rationale"]["reasoning_log"] == _ref(session_log)
    assert lane_experience["rationale"] == {
        "reasoning": "The loads were 32-bit. I widened them to 128-bit.",
        "plan": "Loads are 32-bit. This lane widens them.",
        "plan_file": _ref(orchestration / "lane_002.md"),
        "reasoning_file": lane.relative_to(logs).as_posix(),
        "reasoning_log": _ref(lane_log),
    }
    assert lane_experience["change"]["approach"] == "Vectorize the row loads"
    assert {_ref(lane_log)["sha256"], _ref(lane)["sha256"]} <= {ref["sha256"] for ref in lane_experience["raw_logs"]}
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


@pytest.mark.parametrize("unreadable", ["transcript", "plan", "log", "transcript directory", "best bundle"])
def test_export_runs_stops_naming_a_path_it_cannot_read(tmp_path: Path, unreadable: str) -> None:
    """Claude Code run as root keeps each project's transcripts in a directory only root can list, and Kernel Forge
    creates its best bundles with mkdtemp, so a run copied out of a container holds directories its user cannot list
    as well as files it cannot read."""

    if os.geteuid() == 0:
        pytest.skip("root reads every file whatever its mode")
    logs = tmp_path / "logs"
    experiments = _plain_run(logs)
    transcript = _transcript(logs / PROJECTS / "-work-session" / "session.jsonl", "Plan the round.")
    paths = {
        "transcript": transcript,
        "plan": experiments / "orchestration" / "iter_001" / "optimization_plan.md",
        "log": _loop_log(logs / "forge-loop.log", WORKSPACE),
        "transcript directory": transcript.parent,
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


def test_export_runs_stops_on_a_log_two_runs_fit_equally(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    for run in ("a", "b"):
        _plain_run(logs / run, name="ws", campaign_id=f"campaign-{run}")
    log = _loop_log(logs / "forge-loop.log", "ws")
    output = tmp_path / "runs.json"

    result = CliRunner().invoke(main, ["export-runs", str(logs), "-o", str(output)])

    assert result.exit_code == 1
    assert f"Error: {log} names" in result.output and "export them from separate log roots" in result.output
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
