# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``kernelforge export-runs`` on plain forge-loop runs: the document a run's ``forge_experiments`` becomes.

The fixtures write a run through the layout, log lines, event types, gate tag and critic fallback Kernel Forge's own
writers use, so a change at a writer reaches the exporter's tests; what a test expects the exporter to produce stays
literal.
"""

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
from kernelforge.workflows.forge_loop.campaign.experiments_layout import (
    CANDIDATE_META,
    ROUND_CONTEXT,
    ExperimentsLayout,
    lane_plan_name,
)
from kernelforge.workflows.forge_loop.iteration.fanout import LANE_ROOT_PREFIX, lane_rationale
from kernelforge.workflows.forge_loop.campaign.loop_log import (
    LOOP_BANNER,
    experiment_line,
    iteration_header,
    kb_write_line,
)
from kernelforge.workflows.forge_loop.campaign.run_state import ITERATION_RESULT, ITERATION_STARTED, KB_RECORD_WRITTEN
from kernelforge.workflows.forge_loop.roles.critic_fallback import render_critic_fallback
from kernelforge.workflows.forge_loop.export.document import normalize

CAMPAIGN = "5f1c0a9e2b7d4c38a6e0f1b2c3d4e5f6"
EXPERIMENT = "a1b2c3d4"
WORKSPACE = "forge_loop_triton_softmax_1760500000"
# The tag an Implementer session put in front of its rationale before it recorded only its RATIONALE line.
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
TOKEN_PLAN = "Summary: per token: one scale."
CRITIC_FALLBACK_PLAN = render_critic_fallback(
    draft_plan="""Summary: Small rows are bound by launch latency. This plan fuses the passes.

# Fuse the row reduction
Step one.
""",
    critic_review="Summary: The critic's own summary is not the plan's.",
    critic_verdict="REVISE",
)
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
EMPTY_RAW_LOG = {
    "name": "raw-logs-not-collected.log",
    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "bytes": 0,
}


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_json(path: Path, value: Any) -> Path:
    return _write(path, json.dumps(value))


def _candidate(layout: ExperimentsLayout, iteration: int, **meta: Any) -> Path:
    """``candidates/iter_NNN`` as ``CandidateArchive.record`` writes it."""

    directory = layout.candidate(iteration)
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
    _write_json(directory / CANDIDATE_META, record)
    return directory


def _plain_run(
    root: Path,
    name: str = WORKSPACE,
    campaign_id: str = CAMPAIGN,
    *,
    round_plan: str = ROUND_PLAN,
    lane_plan: str = LANE_PLAN,
    experiment: str = EXPERIMENT,
) -> ExperimentsLayout:
    """A workspace after a short forge-loop: a session that kept, a lane that reverted, and a merge."""

    layout = ExperimentsLayout.of(root / name)
    config = {
        "operator_name": "_softmax_kernel",
        "framework": "unknown",
        "kernel_backend": "triton",
        "gpu_type": "mi300x",
        "kernel_path": "softmax_kernel.py",
    }
    _write_json(layout.campaign_config, config)
    _write_json(layout.run_state, {"campaign_id": campaign_id, "iteration": 3})
    events = [
        {"ts": "2026-10-15 09:58:00", "type": ITERATION_STARTED, "iter": 1},
        {"ts": "2026-10-15 10:00:00", "type": ITERATION_STARTED, "iter": 1},
        {
            "ts": "2026-10-15 10:04:30",
            "type": ITERATION_RESULT,
            "iter": 1,
            "decision": "KEEP",
            "experiment_id": experiment,
        },
        {"ts": "2026-10-15 10:05:00", "type": ITERATION_STARTED, "iter": 2},
        {"ts": "2026-10-15 10:06:00", "type": "merge_attempt_staged", "iter": 2},
        {
            "ts": "2026-10-15 10:09:00",
            "type": ITERATION_RESULT,
            "iter": 2,
            "decision": "REVERT_PERF",
            "experiment_id": experiment,
        },
    ]
    _write(layout.events, "".join(json.dumps(event) + "\n" for event in events) + "{torn line\n")
    plan = _write(layout.round(1) / lane_plan_name(1), round_plan)
    _write(layout.round(1) / lane_plan_name(2), lane_plan)
    _write_json(layout.handoff(1), {"optimization_plan_path": str(plan.relative_to(layout.root.parent))})
    _write(layout.lesson(1), SESSION_LESSON)
    _write(layout.root / "best" / "iter_001" / "softmax_kernel.py", "# best\n")
    _candidate(
        layout,
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
        layout,
        2,
        decision="REVERT_PERF",
        wall_ms=0.051,
        mean_case_speedup=1.02,
        best_wall_ms_before=0.044,
        best_mean_case_speedup_before=1.18,
        plan=lane_plan.strip(),
        rationale=lane_rationale("2"),
    )
    _candidate(
        layout,
        3,
        decision="REVERT_PERF",
        wall_ms=0.047,
        mean_case_speedup=1.1,
        best_wall_ms_before=0.044,
        best_mean_case_speedup_before=1.18,
        plan=MERGE_PLAN,
        rationale="stacked iterations 1 and 2; no Implementer session",
    )
    return layout


def _loop_log(path: Path, experiment: str = EXPERIMENT, kb_record: str = KB_RECORD) -> Path:
    """A forge-loop stdout log of the fixture's three iterations. It names the session by the experiment id the run
    recorded, and the KB record written after the KEEP and rewritten after the loop."""

    written = kb_write_line({"written": True, "solution": kb_record, "speedup": 1.18})
    lines = (
        "  Agent profile: claude / Claude-Opus-5: available",
        LOOP_BANNER,
        experiment_line(experiment),
        iteration_header(1, 1.0, 110 * 60),
        "  [KEEP] mean case speedup=1.180000x — NEW BEST (+18.0% vs previous best)",
        written,
        iteration_header(2, 1.18, 90 * 60),
        iteration_header(3, 1.18, 70 * 60),
        "BUDGET EXHAUSTED after 3 iterations in this session",
        written,
    )
    return _write(path, "".join(f"{line}\n" for line in lines))


def _units(root: Path) -> dict[str, dict[str, Any]]:
    return {unit["unit_id"]: unit for unit in normalize([root])["runs"]}


def _meta(layout: ExperimentsLayout, iteration: int) -> dict[str, Any]:
    return json.loads((layout.candidate(iteration) / CANDIDATE_META).read_text(encoding="utf-8"))


_PUBLISHED_META = (
    "commit_hash",
    "decision",
    "kept",
    "validation_passed",
    "wall_ms",
    "mean_case_speedup",
    "snr_db",
    "vgpr",
    "baseline_wall_ms",
    "best_wall_ms_before",
    "best_mean_case_speedup_before",
    "delta_vs_best_pct",
    "session_end_reason",
    "turns",
    "agent_profile_id",
    "agent_provider",
    "agent_model",
    "blockage_epoch",
)


def _published_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {key: meta[key] for key in _PUBLISHED_META if key in meta and meta[key] is not None and meta[key] != ""}


def _ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"name": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _generation(layout: ExperimentsLayout, commit: str, report: str) -> Path:
    """A published Analysis generation for ``commit``, ``analysis/<commit>/generation-001``, with its report."""

    directory = layout.root / "analysis" / commit / "generation-001"
    _write(directory / "report.md", f"\n{report}\n")
    _write_json(layout.root / "analysis" / commit / "published.json", {"generation_root": directory.name})
    return directory


def _made_elsewhere(relative: str) -> str:
    """The absolute path ``forge_experiments/<relative>`` had where the fixture's run was made."""

    return str(ExperimentsLayout.of(f"/home/someone/kernelforge-runs/{WORKSPACE}").root / relative)


def _planned_on(layout: ExperimentsLayout, iteration: int, bundle: str) -> None:
    """The context round ``iteration`` planned with, naming ``bundle`` as its Analysis bundle after the report it
    summarizes."""

    refs = [{"kind": "analysis_summary", "path": f"{bundle}/report.md"}, {"kind": "analysis_bundle", "path": bundle}]
    _write_json(layout.round(iteration) / ROUND_CONTEXT, {"evidence_refs": refs})


def _round_after_the_keep(layout: ExperimentsLayout) -> None:
    """Round 4, planned after iteration 1's KEEP: a single session's plan, its handoff, and the candidate it built."""

    plan = _write(layout.round(4) / lane_plan_name(1), "# Widen the loads\n\nUse 128-bit loads.\n")
    _write_json(layout.handoff(4), {"optimization_plan_path": str(plan.relative_to(layout.root.parent))})
    _candidate(
        layout,
        4,
        decision="REVERT_PERF",
        wall_ms=0.046,
        best_wall_ms_before=0.044,
        plan="Widen the loads",
        rationale="Wider loads cut the instruction count.",
    )


def test_a_single_session_candidate_becomes_a_complete_unit(tmp_path: Path) -> None:
    layout = _plain_run(tmp_path)
    _loop_log(tmp_path / "forge-loop.log")
    unit = _units(tmp_path)[f"{WORKSPACE}/iteration-1"]

    candidate = layout.candidate(1)
    assert unit == {
        "unit_id": f"{WORKSPACE}/iteration-1",
        "run_id": CAMPAIGN,
        "seq": 1,
        "identity": {"operator": "_softmax_kernel", "kernel": "softmax", "backend": "triton", "gpu": "mi300x"},
        "provenance": {
            "kb_record": KB_RECORD,
            "candidate_origin": "session",
            "meta": _meta(layout, 1),
        },
        "started_at": "2026-10-15T10:00:00",
        "completed_at": "2026-10-15T10:04:30",
        "best_wall_ms_before": 0.052,
        "best_mean_case_speedup_before": 1.0,
        "decision": "KEEP",
        "candidate_wall_ms": 0.044,
        "mean_case_speedup": 1.18,
        "plan_file": str((layout.round(1) / lane_plan_name(1)).resolve()),
        "plan": (
            "Small rows are bound by launch latency. This plan fuses the max and sum passes into one sweep. "
            "It keeps the output layout unchanged."
        ),
        "source_ms": 0.052,
        "source_code": str((candidate / "softmax_kernel.py").resolve()),
        "change_diff_file": str((candidate / "change.diff").resolve()),
        # Archived before meta.json kept the declared headline and reasoning: the plan is the session's headline, and
        # its rationale without the gate tag is its reasoning.
        "summary": HEADLINE,
        "reasoning": SESSION_RATIONALE,
        "lesson": SESSION_LESSON.strip(),
    }


@pytest.mark.parametrize(
    ("seq", "declared", "expected"),
    [
        pytest.param(
            1,
            {"headline": "fuse the max and sum passes", "reasoning": "One sweep reads each row once."},
            {"summary": "fuse the max and sum passes", "reasoning": "One sweep reads each row once."},
            id="session",
        ),
        pytest.param(
            2,
            {
                "headline": "widen the loads",
                "reasoning": "128-bit loads cut the load count.",
                "rationale": lane_rationale("2", "128-bit loads cut the load count."),
            },
            {"summary": "widen the loads", "reasoning": "128-bit loads cut the load count."},
            id="lane",
        ),
        pytest.param(
            1, {"headline": "", "reasoning": "", "rationale": "no rationale provided"}, {}, id="session-declared-none"
        ),
        pytest.param(1, {"rationale": f"{GATE_TAG} no rationale provided"}, {"summary": HEADLINE}, id="archived-none"),
    ],
)
def test_a_candidate_takes_the_headline_and_the_reasoning_its_implementer_declared(
    tmp_path: Path, seq: int, declared: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Not its plan, which for a lane is the lane's whole plan, nor a rationale Kernel Forge wrote in its place."""

    layout = _plain_run(tmp_path)
    _write_json(layout.candidate(seq) / CANDIDATE_META, {**_meta(layout, seq), **declared})

    unit = _units(tmp_path)[f"{WORKSPACE}/iteration-{seq}"]

    assert {key: unit[key] for key in ("summary", "reasoning") if key in unit} == expected
    assert unit["provenance"]["candidate_origin"] == ("lane" if seq == 2 else "session")


@pytest.mark.parametrize(
    ("lesson", "omitted"),
    [
        pytest.param("x" * TEXT_MAX_BYTES, None, id="at-the-limit"),
        pytest.param(
            "é" * (TEXT_MAX_BYTES // 2) + "x",
            f"iter_001.md is {TEXT_MAX_BYTES + 1} bytes, over the {TEXT_MAX_BYTES} a text field holds",
            id="one-byte-over",
        ),
    ],
)
def test_a_lesson_a_text_field_cannot_hold_is_left_out_and_its_provenance_says_why(
    tmp_path: Path, lesson: str, omitted: str | None
) -> None:
    layout = _plain_run(tmp_path)
    _write(layout.lesson(1), lesson)

    session = _units(tmp_path)[f"{WORKSPACE}/iteration-1"]

    assert session.get("lesson") == (lesson if omitted is None else None)
    assert session["provenance"].get("lesson_omitted") == omitted
    assert session["reasoning"] == SESSION_RATIONALE


def test_a_lane_candidate_takes_its_lane_plan_from_the_round_that_planned_it(tmp_path: Path) -> None:
    layout = _plain_run(tmp_path)
    unit = _units(tmp_path)[f"{WORKSPACE}/iteration-2"]

    lane_plan = layout.round(1) / lane_plan_name(2)
    assert unit["provenance"] == {"candidate_origin": "lane", "meta": _meta(layout, 2)}
    assert unit["plan_file"] == str(lane_plan.resolve())
    assert unit["plan"] == "Use 128-bit loads for the input rows. Keep the reduction as it is."
    assert (unit["started_at"], unit["completed_at"]) == ("2026-10-15T10:05:00", "2026-10-15T10:09:00")
    assert (unit["best_wall_ms_before"], unit["candidate_wall_ms"], unit["source_ms"]) == (0.044, 0.051, 0.052)
    assert "reasoning" not in unit


def test_a_merge_is_named_by_its_stacked_plans_and_dated_by_its_archive(tmp_path: Path) -> None:
    layout = _plain_run(tmp_path)
    unit = _units(tmp_path)[f"{WORKSPACE}/iteration-3"]

    assert unit["provenance"] == {"candidate_origin": "merge", "meta": _meta(layout, 3)}
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

    layout = _plain_run(tmp_path)
    _write_json(layout.run_state, {"campaign_id": CAMPAIGN, "iteration": 3, **state})

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

    layout = _plain_run(tmp_path)
    config = json.loads(layout.campaign_config.read_text(encoding="utf-8"))
    _write_json(layout.campaign_config, {**config, **recorded})

    identities = [unit["identity"] for unit in _units(tmp_path).values()]

    assert identities == [{**expected, "backend": "triton", "gpu": "mi300x"}] * 3


def test_the_driver_hash_joins_identity_only_when_the_campaign_recorded_one(tmp_path: Path) -> None:
    layout = _plain_run(tmp_path)
    assert "driver_sha256" not in _units(tmp_path)[f"{WORKSPACE}/iteration-1"]["identity"]

    digest = "ab" * 32
    config = json.loads(layout.campaign_config.read_text(encoding="utf-8"))
    _write_json(layout.campaign_config, {**config, "driver_sha256": digest})

    identities = [unit["identity"]["driver_sha256"] for unit in _units(tmp_path).values()]

    assert identities == [digest, digest, digest]


def test_a_candidate_without_an_archived_diff_links_no_diff_file(tmp_path: Path) -> None:
    layout = _plain_run(tmp_path)
    (layout.candidate(3) / "change.diff").unlink()

    units = _units(tmp_path)

    assert units[f"{WORKSPACE}/iteration-1"]["change_diff_file"].endswith("change.diff")
    assert "change_diff_file" not in units[f"{WORKSPACE}/iteration-3"]


def test_each_log_attaches_to_the_run_whose_experiment_it_announces(tmp_path: Path) -> None:
    _plain_run(tmp_path / "a", name="softmax-a", campaign_id="campaign-a", experiment="exp-a")
    _plain_run(tmp_path / "b", name="softmax-b", campaign_id="campaign-b", experiment="exp-b")
    _loop_log(tmp_path / "a" / "forge-loop.log", "exp-a", kb_record="record-a")
    _loop_log(tmp_path / "b" / "forge-loop.log", "exp-b", kb_record="record-b")

    units = _units(tmp_path)

    first, second = units["softmax-a/iteration-1"], units["softmax-b/iteration-1"]
    assert (first["run_id"], first["provenance"]["kb_record"]) == ("campaign-a", "record-a")
    assert (second["run_id"], second["provenance"]["kb_record"]) == ("campaign-b", "record-b")


def test_runs_whose_workspaces_share_a_name_keep_their_own_logs(tmp_path: Path) -> None:
    """Both runs' workspaces are ``ws``. The experiment id each session printed tells their logs apart."""

    for run in ("a", "b"):
        _plain_run(tmp_path / run, name="ws", campaign_id=f"campaign-{run}", experiment=f"exp-{run}")
        _loop_log(tmp_path / run / "forge-loop.log", f"exp-{run}", kb_record=f"record-{run}")

    units = {(unit["run_id"], unit["seq"]): unit for unit in normalize([tmp_path])["runs"]}

    kept = [units[(f"campaign-{run}", 1)]["provenance"] for run in ("a", "b")]
    assert [provenance["kb_record"] for provenance in kept] == ["record-a", "record-b"]


def test_a_controller_task_keeps_the_kb_record_its_session_printed(tmp_path: Path) -> None:
    """The controller writes the log under tasks/<op> and moves forge_experiments to workspaces/<op>."""

    op = "op-42"
    root = tmp_path / "controller"
    layout = _plain_run(root / "workspaces", name=op)
    _loop_log(root / "tasks" / op / "forge-loop.stdout.log")

    unit = _units(root)[f"{op}/iteration-1"]

    assert layout.root.parent != (root / "tasks" / op)
    assert unit["provenance"]["kb_record"] == KB_RECORD
    dumped = json.dumps(unit)
    assert "forge-loop.log" not in dumped
    assert "forge-loop.stdout.log" not in dumped


def test_two_controller_tasks_on_one_kernel_file_each_keep_their_own_record(tmp_path: Path) -> None:
    """Two in-place controller tasks record one kernel_path. Each kept candidate takes its own session's record, and
    the export writes the document."""

    logs = tmp_path / "controller"
    output = tmp_path / "runs.json"
    for run in ("a", "b"):
        op = f"op-{run}"
        layout = _plain_run(logs / "workspaces", name=op, campaign_id=f"campaign-{run}", experiment=f"exp-{run}")
        config = json.loads(layout.campaign_config.read_text(encoding="utf-8"))
        _write_json(layout.campaign_config, {**config, "kernel_path": "src/softmax_kernel.py"})
        _loop_log(logs / "tasks" / op / "forge-loop.stdout.log", f"exp-{run}", kb_record=f"record-{run}")

    result = CliRunner().invoke(main, ["export-runs", str(logs), "-o", str(output)])

    assert result.exit_code == 0, result.output
    units = {
        unit["run_id"]: unit for unit in json.loads(output.read_text(encoding="utf-8"))["runs"] if unit["seq"] == 1
    }
    assert [units[f"campaign-{run}"]["provenance"]["kb_record"] for run in ("a", "b")] == ["record-a", "record-b"]


def test_only_a_kept_candidate_carries_the_kb_record_its_iteration_wrote(tmp_path: Path) -> None:
    _plain_run(tmp_path)
    _loop_log(tmp_path / "forge-loop.log")

    units = _units(tmp_path)

    records = [units[f"{WORKSPACE}/iteration-{seq}"]["provenance"].get("kb_record") for seq in (1, 2, 3)]
    assert records == [KB_RECORD, None, None]


def _unarchived(layout: ExperimentsLayout, iteration: int, **event: Any) -> None:
    """An iteration that archived no candidate: only its ``iteration_result`` event, and its round's handoff."""

    plan = _write(layout.round(iteration) / lane_plan_name(1), "# Inspect the loads\n\nRead the ISA first.\n")
    _write_json(layout.handoff(iteration), {"optimization_plan_path": str(plan.relative_to(layout.root.parent))})
    record = {"ts": "2026-10-15 10:20:00", "type": ITERATION_RESULT, "iter": iteration, "experiment_id": EXPERIMENT}
    record.update(best_after_ms=0.044, best_after_mean_case_speedup=1.18, session_end_reason="agent_stopped")
    with layout.events.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({**record, **event}) + "\n")


def test_an_iteration_whose_session_changed_nothing_is_a_unit_of_its_own(tmp_path: Path) -> None:
    """Its Implementer took part, so what it declared and the lesson after it are an Experience without a kernel."""

    layout = _plain_run(tmp_path)
    _unarchived(
        layout,
        5,
        decision="NO_CHANGES",
        plan="inspect only",
        reasoning="The loads are already 128-bit, so no change beats the best.",
        techniques={"listed": ["vectorized-loads"], "unlisted": []},
        rationale="The loads are already 128-bit, so no change beats the best.",
    )
    _write(layout.lesson(5), "Nothing to change: the loads are 128-bit.\n")

    unit = _units(tmp_path)[f"{WORKSPACE}/iteration-5"]

    assert (unit["seq"], unit["decision"], unit["summary"]) == (5, "NO_CHANGES", "inspect only")
    assert unit["reasoning"] == "The loads are already 128-bit, so no change beats the best."
    assert unit["lesson"] == "Nothing to change: the loads are 128-bit."
    assert (unit["techniques"], unit["techniques_declared"]) == (["vectorized-loads"], True)
    assert (unit["best_wall_ms_before"], unit["best_mean_case_speedup_before"]) == (0.044, 1.18)
    assert unit["plan_file"] == str((layout.round(5) / lane_plan_name(1)).resolve())
    assert "source_code" not in unit and "change_diff_file" not in unit and unit["candidate_wall_ms"] is None
    assert unit["provenance"]["candidate_origin"] == "session"


def test_an_iteration_that_changed_nothing_collects_as_an_experience_without_a_kernel(tmp_path: Path) -> None:
    layout = _plain_run(tmp_path / "logs")
    _unarchived(layout, 5, decision="NO_CHANGES", plan="inspect only", reasoning="The loads are already 128-bit.")
    output = tmp_path / "runs.json"
    assert CliRunner().invoke(main, ["export-runs", str(tmp_path / "logs"), "-o", str(output)]).exit_code == 0

    report = collect("kernelforge-v1", output, dry_run=True)

    [experience] = [row.experience for row in report.collected if row.unit_id == f"{WORKSPACE}/iteration-5"]
    assert experience.outcome == {"decision": "NO_CHANGES"}
    assert experience.change == {"summary": "inspect only"}
    assert experience.rationale["reasoning"] == "The loads are already 128-bit."
    assert experience.provenance.extra["meta"]["session_end_reason"] == "agent_stopped"


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(
            {"reasoning": "", "rationale": "planning failed; no Implementer session: TimeoutError"}, id="declared-none"
        ),
        pytest.param({"rationale": "planning failed; no Implementer session: TimeoutError"}, id="recorded-before"),
    ],
)
def test_an_iteration_no_agent_gave_an_account_of_is_not_collected(tmp_path: Path, event: dict[str, Any]) -> None:
    """A round that failed to plan ran no Implementer: Kernel Forge's own rationale is not reasoning."""

    layout = _plain_run(tmp_path / "logs")
    _unarchived(layout, 5, decision="ORCHESTRATION_ERROR", **event)
    output = tmp_path / "runs.json"
    assert CliRunner().invoke(main, ["export-runs", str(tmp_path / "logs"), "-o", str(output)]).exit_code == 0

    report = collect("kernelforge-v1", output, dry_run=True)

    skipped = {row.unit_id: row.reason.split(":")[0] for row in report.skipped}
    assert skipped[f"{WORKSPACE}/iteration-5"] == "No account by an agent"


def _record_written(layout: ExperimentsLayout, iteration: int, record: str, experiment: str = EXPERIMENT) -> None:
    """The event Kernel Forge appends once its own experience KB has filed the run under ``record``."""

    event = {"ts": "2026-10-15 10:04:40", "type": KB_RECORD_WRITTEN, "iter": iteration, "experiment_id": experiment}
    with layout.events.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({**event, "record": record}) + "\n")


def test_a_kept_candidate_takes_the_kb_record_from_the_runs_events_without_a_log(tmp_path: Path) -> None:
    """A run collected as it ends has no stdout log to read, and gives the unit a later export with the log gives."""

    layout = _plain_run(tmp_path)
    _record_written(layout, 1, KB_RECORD)
    _record_written(layout, 2, "a record for an iteration that kept nothing")
    _record_written(layout, 1, "the best written again later")

    alone = _units(tmp_path)
    _loop_log(tmp_path / "forge-loop.log")
    with_log = _units(tmp_path)

    records = [alone[f"{WORKSPACE}/iteration-{seq}"]["provenance"].get("kb_record") for seq in (1, 2, 3)]
    assert records == [KB_RECORD, None, None]
    assert alone == with_log


def test_the_runs_events_outrank_a_log_that_names_another_record(tmp_path: Path) -> None:
    layout = _plain_run(tmp_path)
    _record_written(layout, 1, KB_RECORD)
    _loop_log(tmp_path / "forge-loop.log", kb_record="a record the log alone names")

    assert _units(tmp_path)[f"{WORKSPACE}/iteration-1"]["provenance"]["kb_record"] == KB_RECORD


def test_lane_copies_are_not_runs_and_unreadable_archives_say_why(tmp_path: Path) -> None:
    layout = _plain_run(tmp_path)
    lane_copy = ExperimentsLayout.of(tmp_path / f"{LANE_ROOT_PREFIX}x1y2" / "2")
    _write_json(lane_copy.campaign_config, {"operator_name": "softmax_kernel"})
    _write_json(lane_copy.run_state, {"campaign_id": CAMPAIGN})
    _write(layout.candidate(4) / CANDIDATE_META, "{not json")
    unnamed = ExperimentsLayout.of(tmp_path / "unnamed")
    _write_json(unnamed.campaign_config, {"operator_name": "softmax_kernel"})
    _write_json(unnamed.run_state, {"iteration": 0})

    units = _units(tmp_path)

    assert not any(unit_id.startswith("2/") for unit_id in units)
    assert units[f"{WORKSPACE}/iteration-4"]["read_error"].endswith("meta.json is not a readable JSON object")
    assert units["unnamed/session"] == {
        "unit_id": "unnamed/session",
        "read_error": "run_state.json names no campaign_id",
    }


def test_summary_lines_give_the_plan_and_the_whole_lesson_is_the_reflection(tmp_path: Path) -> None:
    layout = _plain_run(tmp_path, round_plan=SUMMARY_PLAN, lane_plan=SUMMARY_LANE_PLAN)
    orchestration = layout.round(1)
    _write(layout.lesson(1), SESSION_LESSON)
    _write(layout.lesson(2), LANE_LESSON)
    _write(orchestration / "optimization_plan.reasoning.md", "Summary: This log is not the reasoning.\n")
    _write(orchestration / "lane_002.reasoning.md", "Summary: Neither is this one.\n")

    units = _units(tmp_path)

    session, lane, merge = (units[f"{WORKSPACE}/iteration-{seq}"] for seq in (1, 2, 3))
    assert session["plan"] == (
        "Small rows are bound by launch latency. Each row is read twice. "
        "This plan fuses the max and sum passes into one sweep. The small cases should gain most."
    )
    # The lesson is written knowing the outcome, so it is the reflection, and the reasoning stays the session's own.
    assert (session["reasoning"], session["lesson"]) == (SESSION_RATIONALE, SESSION_LESSON.strip())
    assert "lesson_file" not in session
    assert session["plan_file"] == str((orchestration / lane_plan_name(1)).resolve())
    assert (lane["plan"], lane["lesson"]) == ("Loads are 32-bit. This lane widens them.", LANE_LESSON.strip())
    assert "reasoning" not in lane
    assert lane["plan_file"] == str((orchestration / lane_plan_name(2)).resolve())
    assert "reasoning" not in merge and "lesson" not in merge


@pytest.mark.parametrize("lesson", [None, "\n"])
def test_a_plan_is_summarized_from_its_draft_and_a_session_without_a_lesson_falls_back(
    tmp_path: Path, lesson: str | None
) -> None:
    layout = _plain_run(tmp_path, round_plan=CRITIC_FALLBACK_PLAN)
    lesson_path = layout.lesson(1)
    if lesson is None:
        lesson_path.unlink()
    else:
        _write(lesson_path, lesson)
    _write(layout.round(1) / "optimization_plan.reasoning.md", "Summary: Ignored.\n")

    session = _units(tmp_path)[f"{WORKSPACE}/iteration-1"]

    assert session["plan"] == "Small rows are bound by launch latency. This plan fuses the passes."
    assert session["reasoning"] == SESSION_RATIONALE


def test_each_candidate_carries_the_report_of_the_analysis_its_round_planned_on(tmp_path: Path) -> None:
    layout = _plain_run(tmp_path)
    _round_after_the_keep(layout)
    _generation(layout, BASELINE_COMMIT, BASELINE_REPORT)
    _generation(layout, KEPT_COMMIT, KEPT_REPORT)
    _planned_on(layout, 1, _made_elsewhere(f"analysis/{BASELINE_COMMIT}/generation-001"))
    _planned_on(layout, 4, _made_elsewhere(f"analysis/{KEPT_COMMIT}/generation-001"))

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
    layout = _plain_run(tmp_path)
    _generation(layout, BASELINE_COMMIT, report)
    _planned_on(layout, 1, _made_elsewhere(f"analysis/{BASELINE_COMMIT}/generation-001"))

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
    layout = _plain_run(tmp_path)
    _generation(layout, BASELINE_COMMIT, BASELINE_REPORT)
    if bundle is not None:
        _planned_on(layout, 1, bundle)

    session = _units(tmp_path)[f"{WORKSPACE}/iteration-1"]

    assert "analysis_report" not in session
    assert "analysis_generation" not in session["provenance"]


def test_the_report_collects_as_text_and_no_analysis_file_is_linked(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    layout = _plain_run(logs)
    generation = _generation(layout, BASELINE_COMMIT, BASELINE_REPORT)
    _write(generation / "cases" / "M1024_N256-2bb066cf" / "profile" / "pmc_perf.csv", "Dispatch_ID,SQ_WAVES\n0,64\n")
    _planned_on(layout, 1, _made_elsewhere(f"analysis/{BASELINE_COMMIT}/generation-001"))
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
    assert experience["raw_logs"] == [EMPTY_RAW_LOG]


@pytest.mark.parametrize(
    "stale",
    [pytest.param(None, id="new-directory"), pytest.param("a log someone left\n", id="stale-raw-log")],
)
def test_export_runs_writes_the_document_and_an_empty_raw_log_beside_it(tmp_path: Path, stale: str | None) -> None:
    _plain_run(tmp_path / "logs")
    output = tmp_path / "out" / "runs.json"
    raw_log = output.parent / "raw-logs-not-collected.log"
    if stale is not None:
        _write(raw_log, stale)

    result = CliRunner().invoke(main, ["export-runs", str(tmp_path / "logs"), "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert result.output == f"wrote 3 runs to {output}\n"
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document == normalize([tmp_path / "logs"])
    assert document["raw_logs"] == [raw_log.name]
    assert raw_log.read_bytes() == b""


def test_the_exported_document_collects_through_the_packaged_mapping(tmp_path: Path) -> None:
    layout = _plain_run(tmp_path / "logs")
    _loop_log(tmp_path / "logs" / "forge-loop.log")
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
    assert row.experience.baseline == {
        "best_wall_ms_before": 0.052,
        "best_mean_case_speedup_before": 1.0,
        "source_ms": 0.052,
    }
    assert row.experience.outcome == {"decision": "KEEP", "candidate_wall_ms": 0.044, "mean_case_speedup": 1.18}
    assert row.experience.change["summary"] == HEADLINE
    assert row.experience.rationale["reasoning"] == SESSION_RATIONALE
    assert row.experience.reflection["lesson"] == SESSION_LESSON.strip()
    assert "llm_backend" not in row.experience.provenance.extra
    assert "llm_model" not in row.experience.provenance.extra
    assert row.experience.provenance.extra["meta"] == _published_meta(_meta(layout, 1))
    assert [ref.to_dict() for ref in row.experience.raw_logs] == [EMPTY_RAW_LOG]
    assert {row.unit_id: row.reason.split(":")[0] for row in report.skipped} == {
        f"{WORKSPACE}/iteration-2": "No account by an agent",
        f"{WORKSPACE}/iteration-3": "No account by an agent",
    }


def test_a_token_named_case_and_a_per_token_plan_still_collect(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    layout = _plain_run(logs)
    record = _meta(layout, 1)
    record["bench"] = {"case_times": {"tokens_128_e8": 1.2}}
    record["plan"] = TOKEN_PLAN
    record["rationale"] = '{"max_tokens": 4}'
    _write_json(layout.candidate(1) / CANDIDATE_META, record)
    _write(layout.round(1) / lane_plan_name(1), f"{TOKEN_PLAN}\n")
    output = tmp_path / "runs.json"
    assert CliRunner().invoke(main, ["export-runs", str(logs), "-o", str(output)]).exit_code == 0

    exported = next(run for run in json.loads(output.read_text(encoding="utf-8"))["runs"] if run["seq"] == 1)
    assert exported["provenance"]["meta"]["bench"]["case_times"]["tokens_128_e8"] == 1.2
    assert exported["provenance"]["meta"]["plan"] == TOKEN_PLAN

    report = collect("kernelforge-v1", output, dry_run=True)
    row = next(item for item in report.collected if item.unit_id == f"{WORKSPACE}/iteration-1")
    assert row.experience.rationale["plan"] == "per token: one scale."
    meta = row.experience.provenance.extra["meta"]
    assert meta == _published_meta(record)
    assert meta["mean_case_speedup"] == 1.18
    for name in ("plan", "rationale", "bench"):
        assert name not in meta


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        pytest.param(
            {"techniques": ["split-k"], "techniques_unlisted": ["atomic epilogue"]},
            {"techniques": ["split-k"], "techniques_unlisted": ["atomic epilogue"], "techniques_declared": True},
            id="named",
        ),
        pytest.param({"techniques": [], "techniques_unlisted": []}, {"techniques_declared": True}, id="declared-none"),
        pytest.param({}, {}, id="never-declared"),
    ],
)
def test_the_techniques_a_session_declared_reach_its_experience(
    tmp_path: Path, declared: dict[str, Any], expected: dict[str, Any]
) -> None:
    logs = tmp_path / "logs"
    layout = _plain_run(logs)
    _write_json(layout.candidate(1) / CANDIDATE_META, {**_meta(layout, 1), **declared})
    output = tmp_path / "runs.json"
    assert CliRunner().invoke(main, ["export-runs", str(logs), "-o", str(output)]).exit_code == 0

    report = collect("kernelforge-v1", output, dry_run=True).to_dict()

    [change] = [row["experience"]["change"] for row in report["collected"] if row["unit_id"].endswith("iteration-1")]
    assert {key: value for key, value in change.items() if key.startswith("techniques")} == expected


def test_a_run_collects_its_lanes_with_each_lesson_as_their_reflection(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    layout = _plain_run(logs, round_plan=SUMMARY_PLAN, lane_plan=SUMMARY_LANE_PLAN)
    _write(layout.lesson(2), LANE_LESSON)
    output = tmp_path / "runs.json"
    assert CliRunner().invoke(main, ["export-runs", str(logs), "-o", str(output)]).exit_code == 0

    report = collect("kernelforge-v1", output, dry_run=True).to_dict()

    experiences = {row["unit_id"]: row["experience"] for row in report["collected"]}
    assert sorted(experiences) == [f"{WORKSPACE}/iteration-1", f"{WORKSPACE}/iteration-2"]
    session, lane = experiences[f"{WORKSPACE}/iteration-1"], experiences[f"{WORKSPACE}/iteration-2"]
    assert session["rationale"]["reasoning"] == SESSION_RATIONALE
    assert session["reflection"]["lesson"] == SESSION_LESSON.strip()
    assert lane["rationale"] == {
        "plan": "Loads are 32-bit. This lane widens them.",
        "plan_file": _ref(layout.round(1) / lane_plan_name(2)),
    }
    assert lane["reflection"]["lesson"] == LANE_LESSON.strip()
    assert lane["change"]["change_diff_file"]["name"] == "change.diff"
    assert (lane["baseline"], lane["outcome"]) == (
        {"best_wall_ms_before": 0.044, "best_mean_case_speedup_before": 1.18, "source_ms": 0.052},
        {"decision": "REVERT_PERF", "candidate_wall_ms": 0.051, "mean_case_speedup": 1.02},
    )
    origins = [experience["provenance"]["extra"]["candidate_origin"] for experience in (session, lane)]
    assert origins == ["session", "lane"]
    assert session["raw_logs"] == lane["raw_logs"] == [EMPTY_RAW_LOG]
    assert [(row["unit_id"], row["reason"].split(":")[0]) for row in report["skipped"]] == [
        (f"{WORKSPACE}/iteration-3", "No account by an agent")
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
    layout = _plain_run(logs)
    paths = {
        "plan": layout.round(1) / lane_plan_name(1),
        "log": _loop_log(logs / "forge-loop.log"),
        "candidate record": layout.candidate(1) / CANDIDATE_META,
        "events log": layout.events,
        "best bundle": layout.root / "best" / "iter_001",
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


def test_a_log_from_a_run_that_was_not_exported_attaches_to_no_run(tmp_path: Path) -> None:
    """The log names the same kernel_path and workspace as the exported run, and announces a session that run did not
    record. The kernel line does not decide, so the candidate keeps no record and the export still writes."""

    logs = tmp_path / "logs"
    layout = _plain_run(logs, name="ws", campaign_id="campaign-a", experiment="exp-exported")
    config = json.loads(layout.campaign_config.read_text(encoding="utf-8"))
    _write_json(layout.campaign_config, {**config, "kernel_path": "k.py"})
    _write(
        logs / "orphan.log",
        "\n".join(
            [
                LOOP_BANNER,
                "  Kernel: /srv/other/ws/k.py",
                experiment_line("exp-other"),
                iteration_header(1, 1.0, 110 * 60),
                kb_write_line({"written": True, "solution": "record-other", "speedup": 1.18}),
                "",
            ]
        ),
    )
    output = tmp_path / "runs.json"

    result = CliRunner().invoke(main, ["export-runs", str(logs), "-o", str(output)])

    assert result.exit_code == 0, result.output
    kept = next(unit for unit in json.loads(output.read_text(encoding="utf-8"))["runs"] if unit["seq"] == 1)
    assert "kb_record" not in kept["provenance"]


def test_a_resumed_session_keeps_the_record_its_own_session_printed(tmp_path: Path) -> None:
    """An interrupted session and the session that resumed it both print a record under iteration 2. The candidate
    takes the record of the session its iteration_result names."""

    layout = _plain_run(tmp_path, experiment="exp-new")
    _write_json(layout.candidate(2) / CANDIDATE_META, {**_meta(layout, 2), "kept": True})
    sessions = (
        LOOP_BANNER,
        experiment_line("exp-old"),
        iteration_header(2, 1.0, 90 * 60),
        kb_write_line({"written": True, "solution": "record-old", "speedup": 1.02}),
        LOOP_BANNER,
        experiment_line("exp-new"),
        iteration_header(2, 1.18, 70 * 60),
        kb_write_line({"written": True, "solution": "record-new", "speedup": 1.18}),
    )
    _write(tmp_path / "forge-loop.log", "".join(f"{line}\n" for line in sessions))

    unit = _units(tmp_path)[f"{WORKSPACE}/iteration-2"]

    assert unit["provenance"]["kb_record"] == "record-new"


def test_two_logs_that_disagree_on_one_session_leave_the_record_unset(tmp_path: Path) -> None:
    _plain_run(tmp_path)
    _loop_log(tmp_path / "a.log", kb_record="record-a")
    _loop_log(tmp_path / "b.log", kb_record="record-b")

    unit = _units(tmp_path)[f"{WORKSPACE}/iteration-1"]

    assert "kb_record" not in unit["provenance"]


def test_export_runs_refuses_a_missing_root_or_output(tmp_path: Path) -> None:
    runner = CliRunner()

    missing_root = runner.invoke(main, ["export-runs", str(tmp_path / "absent"), "-o", str(tmp_path / "runs.json")])
    no_output = runner.invoke(main, ["export-runs", str(tmp_path)])
    no_root = runner.invoke(main, ["export-runs", "-o", str(tmp_path / "runs.json")])

    assert [missing_root.exit_code, no_output.exit_code, no_root.exit_code] == [2, 2, 2]
    assert "does not exist" in missing_root.output
    assert "--output" in no_output.output
    assert not (tmp_path / "runs.json").exists()
