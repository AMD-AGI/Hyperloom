# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The lesson an iteration records: who writes it, the scope of the negatives it carries, and how it reaches the next prompt."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.memory.lessons import SummaryOutcome
from kernelforge.workflows.forge_loop.campaign.run_state import RunState
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _lane_agent_factory,
    _lesson_result,
    _make_loop,
    _reduction_loop,
    _stub_measurement,
)


def _attach_lessons(loop, workspace):
    from kernelforge.workflows.forge_loop.memory.lessons import LessonStore

    loop.lessons = LessonStore(str(workspace))
    return loop.lessons


async def _fake_summarizer(_prompt: str) -> str:
    return "tried three tile shapes\n- [worse] BLOCK_N 128 | 0.94x"


def test_record_lesson_writes_a_carried_lane_lesson_without_asking(tmp_path, monkeypatch, capsys):
    """The lane already answered, in its round. An answer without a record is the machine fallback, naming why."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)
    asked = {"n": 0}

    async def counting(_prompt: str) -> str:
        asked["n"] += 1
        return "should not run"

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": counting,
            },
            lane_record=SummaryOutcome(text="lane 2 widened the loads"),
        )
    )
    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(iteration=4),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": counting,
                "findings": "compile error: invalid cast",
            },
            lane_record=SummaryOutcome(reason="session replied with no text"),
        )
    )

    assert asked["n"] == 0
    assert store.read(3).startswith("lane 2 widened the loads")
    assert store.read(4).startswith("(no agent summary)")
    assert "session replied with no text" in capsys.readouterr().out


async def test_a_lane_without_a_record_names_why_in_its_lesson(tmp_path, monkeypatch, capsys):
    """A provider that cannot resume the lane session leaves a reason, and the queue carries it into the lesson."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)

    await loop._fill_lane_queue(
        agent_factory=_lane_agent_factory({"tune prefill": ("return 1", "return 2")}),
        lane_plans=["tune prefill"],
    )
    queued = json.loads(loop._lane_queue_path().read_text(encoding="utf-8"))["candidates"]
    loop._lane_queue = []
    loop._restore_lane_queue()
    lane = loop._take_lane_candidate()
    await loop._record_lesson(
        result=_lesson_result(iteration=1),
        decision="REVERT_PERF",
        session_sink={"session_started": True, "plan": lane.plan},
        diff_summary=loop._diff_summary_from_diff(lane.diff),
        lane_record=lane.lesson,
    )

    assert [entry["lesson_reason"] for entry in queued] == ["provider cannot resume the session"]
    assert lane.lesson == SummaryOutcome(reason="provider cannot resume the session")
    assert "Summary unavailable: provider cannot resume the session" in store.read(1)
    assert "[lesson] no summary (provider cannot resume the session)" in capsys.readouterr().out


def test_record_lesson_writes_narrative_and_outcome(tmp_path, monkeypatch):
    """Both authors land in one document: the session's, then the loop's."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)
    loop.best_wall_ms = 1.2

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _fake_summarizer,
            },
        )
    )

    text = store.read(3)
    assert "BLOCK_N 128" in text
    assert "OUTCOME: REVERT_PERF" in text
    assert "wall 1.5000 ms vs best 1.2000 ms" in text
    assert "session ended: turn_cap" in text


def test_record_lesson_falls_back_to_gate_findings(tmp_path, monkeypatch):
    """A provider that cannot resume still leaves the gate's rejections behind."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(iteration=4),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": None,
                "plan": "x",
                "findings": "compile error: invalid cast\n---\ncorrect but not faster",
            },
            diff_summary="kernel.py | 3 +-",
        )
    )

    text = store.read(4)
    assert "invalid cast" in text
    assert "correct but not faster" in text
    assert "kernel.py | 3 +-" in text
    assert "OUTCOME: REVERT_PERF" in text


def test_record_lesson_outcome_only_records_machine_verdict(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(iteration=4),
            decision="REVERT_PERF",
            # Nothing to summarize and nothing observed: no narrative is possible.
            session_sink={
                "session_started": True,
                "summarize": None,
                "plan": "x",
            },
            diff_summary="",
        )
    )

    # An outcome-only document still says what the outcome was measured under.
    assert store.read(4).strip().startswith("SCOPE: measured on ")
    assert "OUTCOME:" in store.read(4)


_THREE_CASES = {"decode-t1": 1.0, "decode-t64": 2.0, "prefill-t16384": 3.0}


async def _scoped_summarizer(_prompt: str) -> str:
    return "swept split-K on decode-t1; every point slower\nHELD-FIXED: BLOCK_N=16, num_warps=8"


def test_record_lesson_scopes_the_negative_to_the_measured_suite(tmp_path, monkeypatch):
    """The whole suite was measured, and the session said what it pinned."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "sweep split-K",
            },
        )
    )

    scope = store.scope_of(3)
    assert scope.cases == ("decode-t1", "decode-t64", "prefill-t16384")
    assert scope.held_fixed == (("BLOCK_N", "16"), ("num_warps", "8"))
    assert scope.lane_restricted is False


async def _unreachable_summarizer(_prompt: str) -> str:
    return (
        "the transposed read is the only real fix and this build CANNOT emit "
        "it\n"
        "HELD-FIXED: BLOCK_N=16\n"
        "DISPROOF: untested — a build-only screen of the one instruction "
        "would settle it"
    )


def test_record_lesson_carries_an_undisproven_cannot_into_the_scope(tmp_path, monkeypatch, capsys):
    """The claim closes nothing until the experiment it names is actually run."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _unreachable_summarizer,
                "plan": "transpose the LDS read",
            },
        )
    )

    from kernelforge.workflows.forge_loop.memory.lessons import UNDISPROVEN_CLAIM

    assert store.scope_of(3).disproof == UNDISPROVEN_CLAIM
    assert "feasibility claim not disproved" in store.read(3)
    assert "without running the experiment" in capsys.readouterr().out
    assert "VALIDITY: RE-OPENABLE (undisproven feasibility claim)" in (
        store.render_for_prompt(current_cases=tuple(_THREE_CASES), kernel_source="BLOCK_N = 16\n")
    )


async def _refuted_summarizer(_prompt: str) -> str:
    return (
        "the transposed read is the only real fix and this build CANNOT emit "
        "it\n"
        "HELD-FIXED: BLOCK_N=16\n"
        "DISPROOF: falsified — a build-only screen shows gfx950 assembles "
        "ds_read_b64_tr_b16"
    )


def test_record_lesson_carries_a_refuted_cannot_as_an_open_direction(tmp_path, monkeypatch, capsys):
    """The session's own experiment killed its premise; the axis is reachable.

    Scoring this as an obligation discharged would leave the record IN SCOPE
    and still suppressing the direction the same line proved open.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _refuted_summarizer,
                "plan": "transpose the LDS read",
            },
        )
    )

    from kernelforge.workflows.forge_loop.memory.lessons import is_claim_disproved

    assert is_claim_disproved(store.scope_of(3).disproof)
    assert "feasibility claim disproved by a build-only screen" in store.read(3)
    assert "shown reachable" in capsys.readouterr().out
    rendered = store.render_for_prompt(current_cases=tuple(_THREE_CASES), kernel_source="BLOCK_N = 16\n")
    assert "VALIDITY: RE-OPEN (feasibility claim disproved)" in rendered
    assert "VALIDITY: IN SCOPE" not in rendered


def test_record_lesson_keeps_a_restricted_lanes_scope(tmp_path, monkeypatch):
    """A lane told to touch one case measured one case; the ban stays there."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "waves_per_eu on prefill-t16384 only",
            },
        )
    )

    scope = store.scope_of(3)
    assert scope.cases == ("prefill-t16384",)
    assert scope.lane_restricted is True


def test_record_lesson_says_when_no_premise_was_recorded(tmp_path, monkeypatch):
    """A session that pinned nothing it could name leaves a re-openable record."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _fake_summarizer,
                "plan": "sweep split-K",
            },
        )
    )

    assert store.scope_of(3).held_fixed == ()
    assert "held fixed (not recorded)" in store.read(3)
    assert "VALIDITY: RE-OPENABLE" in store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source="",
    )


def test_an_unreadable_kernel_is_not_reported_as_a_kernel_that_dropped_it(tmp_path, monkeypatch, capsys):
    """An I/O failure must not become a factual claim about the source.

    ``_read_source_file`` collapses an unreadable file to "", which scans as a
    kernel that assigns nothing: every pinned constant reported missing,
    every stored negative re-opened, and no way to tell that from a kernel that
    really did move on.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)
    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "sweep split-K",
            },
        )
    )
    Path(loop.ic.kernel_file).unlink()

    assert loop._read_kernel_source() == ""
    assert loop._kernel_source_for_scope() is None
    assert "kernel source unreadable" in capsys.readouterr().out

    rendered = store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source=loop._kernel_source_for_scope(),
    )
    assert "were not checked against the current kernel" in rendered
    assert "not assigned in the kernel source checked" not in rendered


def test_an_unreadable_kernel_reaches_the_prompt_as_unchecked(tmp_path, monkeypatch):
    """The loop's own prompt build must carry the distinction, not just the store."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=2, baseline_case_times=_THREE_CASES)
    _stub_measurement(monkeypatch, walls=[0.5, 0.9])

    kernel = Path(loop.ic.kernel_file)
    prompts: list[str] = []
    edits = {"n": 0}

    async def summarizer(_prompt: str) -> str:
        # The next prompt is built before the next session touches the file.
        kernel.unlink()
        return "swept split-K on decode-t1; every point slower\nHELD-FIXED: BLOCK_N=16"

    async def agent_fn(_kernel_path, experiment_history, session_sink):
        prompts.append(experiment_history)
        edits["n"] += 1
        kernel.write_text(f"def kernel():\n    return {edits['n'] + 1}\n")
        session_sink["plan"] = "sweep split-K"
        session_sink["end_reason"] = "candidate_submitted"
        session_sink["summarize"] = summarizer
        return "rationale"

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=agent_fn)))

    assert len(prompts) == 2
    assert "were not checked against the current kernel" in prompts[1]
    assert "not assigned in the kernel source checked" not in prompts[1]


async def _positive_summarizer(_prompt: str) -> str:
    # Complies with the summary prompt's contract: the record itself states
    # that no direction it covers measured worse.
    return "widened the tile on every case; 1.2x, nothing measured worse\nNEGATIVES: none"


def test_a_pin_that_moved_to_a_sibling_file_is_not_reported_as_dropped(tmp_path, monkeypatch):
    """Tile and dispatch constants live outside the anchor kernel.

    The implementer prompt says so itself. Checking only the anchor turns a
    constant that moved into a constant the task no longer assigns, which
    re-opens every negative measured under it.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)
    sibling = workspace / "tiles.py"
    sibling.write_text('CONFIG = {"BLOCK_N": 16}\nnum_warps = 8\n')
    loop.ic.source_files.append(str(sibling))

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "sweep split-K",
            },
        )
    )

    sources = loop._kernel_source_for_scope()
    assert sources is not None and len(sources) == 2
    rendered = store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source=sources,
    )
    assert "VALIDITY: IN SCOPE" in rendered
    assert "not assigned in the kernel source checked" not in rendered


def test_a_source_that_cannot_be_parsed_is_not_a_source_that_dropped_the_pin(tmp_path, monkeypatch):
    """A syntax error is "not checked", never "the constant is gone"."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "sweep split-K",
            },
        )
    )
    Path(loop.ic.kernel_file).write_text("def kernel(:\n    return 1\n")

    rendered = store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source=loop._kernel_source_for_scope(),
    )
    assert "could not be parsed" in rendered
    assert "not assigned in the kernel source checked" not in rendered


def test_an_iteration_that_measured_no_negative_is_not_reopenable(tmp_path, monkeypatch):
    """Nothing in the document was closed, so there is nothing to re-open.

    Only the summarizer writes HELD-FIXED:, and only for a direction that
    measured WORSE. Treating its absence as a re-opened premise labels every
    positive iteration RE-OPENABLE and degrades the store to "nothing is
    settled".
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(kept=True, mean_case_speedup=1.2),
            decision="KEEP",
            session_sink={
                "session_started": True,
                "summarize": _positive_summarizer,
                "plan": "widen the tile",
                "findings": "",
            },
        )
    )

    scope = store.scope_of(3)
    assert scope.carries_negative is False
    assert scope.held_fixed == ()
    rendered = store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source="",
    )
    assert "VALIDITY: IN SCOPE" in rendered
    assert "VALIDITY: RE-OPENABLE" not in rendered


def test_an_in_session_gate_rejection_is_a_measured_negative(tmp_path, monkeypatch):
    """A rejection the agent hit is a negative even when the loop kept nothing."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="NO_CHANGES",
            session_sink={
                "session_started": True,
                "summarize": _positive_summarizer,
                "plan": "widen the tile",
                "findings": "correct but not faster: 0.97x\n---\ncompile error",
            },
        )
    )

    assert store.scope_of(3).carries_negative is True
    assert "VALIDITY: RE-OPENABLE" in store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source="",
    )


async def _in_session_negative_summarizer(_prompt: str) -> str:
    """A KEEP session that measured four regressions on the way there."""
    return (
        "tried split-K=4 (0.91x), BLOCK_N=128 (0.94x) and two waves_per_eu "
        "settings, reverted all four inside the session, then submitted the "
        "tile widening that measured 1.2x\n"
        "NEGATIVES: split-K=4, BLOCK_N=128, waves_per_eu=2/4"
    )


async def _unmarked_summarizer(_prompt: str) -> str:
    """An older document, or a model that ignored the marker contract."""
    return "tried a few things and submitted the tile widening"


def test_a_kept_iteration_that_measured_negatives_in_session_is_reopenable(tmp_path, monkeypatch):
    """The loop kept a candidate; the record says four directions measured worse.

    Those four are invisible to the loop — they were reverted before the
    candidate it measured — so deciding from its own verdict would stamp this
    document "no measured negative" and, with no HELD-FIXED line, render it IN
    SCOPE. That promotes four negatives measured under unknown constants into a
    standing ban. The document's own marker is what the loop has to read.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(kept=True, mean_case_speedup=1.2),
            decision="KEEP",
            session_sink={
                "session_started": True,
                "summarize": _in_session_negative_summarizer,
                "plan": "widen the tile",
                # The in-session gate allowed on the first Stop, so it logged
                # nothing: findings cannot catch these either.
                "findings": "",
            },
        )
    )

    scope = store.scope_of(3)
    assert scope.carries_negative is True
    assert scope.held_fixed == ()

    rendered = store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source="",
    )
    assert "VALIDITY: RE-OPENABLE" in rendered
    assert "no measured negative" not in store.read(3)
    assert "the constants it was measured under were not recorded" in rendered


def test_a_document_without_the_marker_is_unknown_not_negative_free(tmp_path, monkeypatch):
    """No marker means nobody answered, and the note must not answer for it."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(kept=True, mean_case_speedup=1.2),
            decision="KEEP",
            session_sink={
                "session_started": True,
                "summarize": _unmarked_summarizer,
                "plan": "widen the tile",
                "findings": "",
            },
        )
    )

    assert store.scope_of(3).carries_negative is None
    rendered = store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source="",
    )
    assert "no measured negative" not in store.read(3)
    assert "whether anything measured worse was not recorded" in rendered
    assert "VALIDITY: RE-OPENABLE" in rendered


def test_a_record_that_never_answered_the_disproof_question_says_so(tmp_path, monkeypatch, capsys):
    """An unanswered question is the state no verdict fires on, so it is printed.

    Nothing reads the prose for "cannot", so a session that ignored the marker
    leaves any feasibility claim in its record resting on the citation rule
    alone. That is the one degradation an operator cannot see in the verdict.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(kept=True, mean_case_speedup=1.2),
            decision="KEEP",
            session_sink={
                "session_started": True,
                "summarize": _unmarked_summarizer,
                "plan": "widen the tile",
                "findings": "",
            },
        )
    )

    assert store.scope_of(3).disproof is None
    assert "recorded as unchecked" in capsys.readouterr().out


def test_the_loops_own_negative_overrides_a_record_that_claims_none(tmp_path, monkeypatch):
    """Machine truth beats prose: a REVERT is a negative whatever the text says."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _positive_summarizer,  # writes "NEGATIVES: none"
                "plan": "widen the tile",
                "findings": "",
            },
        )
    )

    assert store.scope_of(3).carries_negative is True


def test_a_crash_is_a_measured_negative(tmp_path, monkeypatch):
    """CRASH starts with no REVERT and leaves no speedup, so only a whitelist sees it.

    A session cut off before the Stop hook also leaves no findings, so nothing
    else in the loop's view would report this iteration as having gone wrong.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(crashed=True, validation_passed=False, mean_case_speedup=None),
            decision="CRASH",
            session_sink={
                "session_started": True,
                "summarize": _positive_summarizer,  # writes "NEGATIVES: none"
                "plan": "widen the tile",
                "findings": "",
            },
        )
    )

    assert store.scope_of(3).carries_negative is True
    assert "VALIDITY: RE-OPENABLE" in store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source="",
    )


def test_a_build_failure_is_a_measured_negative(tmp_path, monkeypatch):
    loop, _workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    assert loop._loop_measured_a_negative("BUILD_FAILED", _lesson_result(validation_passed=False), {}) is True
    assert loop._loop_measured_a_negative("KEEP", _lesson_result(), {}) is False
    assert loop._loop_measured_a_negative("NO_CHANGES", _lesson_result(mean_case_speedup=None), {}) is False


def test_a_machine_written_document_is_decided_by_the_loops_verdict_alone(tmp_path, monkeypatch):
    """The loop authored it, so the loop's view of it is complete.

    A fallback document contains what the loop observed and nothing else, so
    there is no unseen reverted direction for a marker to report — and no
    summarizer to write one.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(kept=True, mean_case_speedup=1.2),
            decision="KEEP",
            session_sink={
                "session_started": True,
                "summarize": None,
                "plan": "widen the tile",
                "findings": "",
            },
            diff_summary="kernel.py | 3 +-",
        )
    )

    assert "(no agent summary)" in store.read(3)
    assert store.scope_of(3).carries_negative is False


def test_one_unreadable_sibling_does_not_indict_the_constant_it_holds(tmp_path, monkeypatch, capsys):
    """Part of the declared source went unchecked; say that, do not guess.

    Dropping the unreadable file leaves the survivors looking like the whole
    declared set, so a constant living in the missing one reads as deleted.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)
    sibling = workspace / "tiles.py"
    sibling.write_text("BLOCK_N = 16\nnum_warps = 8\n")
    loop.ic.source_files.append(str(sibling))

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "sweep split-K",
            },
        )
    )
    sibling.unlink()

    sources = loop._kernel_source_for_scope()
    assert sources is not None and sources[1] is None
    assert "source unreadable" in capsys.readouterr().out

    rendered = store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source=sources,
    )
    assert "BLOCK_N was not checked" in rendered
    assert "not assigned in the kernel source checked" not in rendered


def test_an_unscored_case_is_not_recorded_as_a_case_that_was_measured(tmp_path, monkeypatch):
    """Scoring excluded the noisy case, so nothing was measured on it."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)
    loop._unscored_cases = {"decode-t64"}

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "sweep split-K",
            },
        )
    )

    scope = store.scope_of(3)
    assert scope.cases == ("decode-t1", "prefill-t16384")
    assert scope.lane_restricted is False


def test_an_unscored_case_does_not_reopen_a_stored_negative_in_the_prompt(tmp_path, monkeypatch):
    """The render path validates against the scored suite, not every case."""
    from kernelforge.workflows.forge_loop.memory.lessons import LessonScope, LessonStore

    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    loop.ic.preloop_baseline_unscored_cases = ["decode-t64"]
    _stub_measurement(monkeypatch, walls=[0.5])

    store = LessonStore(str(workspace))
    store.write(1, "swept split-K; every point slower")
    store.append_scope(
        1,
        LessonScope(
            cases=("decode-t1", "prefill-t16384"),
            held_fixed=(("BLOCK_N", "16"),),
            carries_negative=True,
        ),
    )

    prompts: list[str] = []

    async def agent_fn(_kernel_path, experiment_history, session_sink):
        prompts.append(experiment_history)
        session_sink["plan"] = "inspect only"
        return "No source change was needed."

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=agent_fn)))

    assert loop._scored_case_ids() == ["decode-t1", "prefill-t16384"]
    assert prompts and "swept split-K" in prompts[0]
    assert "not measured on decode-t64" not in prompts[0]


def test_a_scope_that_cannot_be_written_says_so(tmp_path, monkeypatch, capsys):
    """A failed append leaves an unscoped document; the operator hears about it."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)
    monkeypatch.setattr(store, "append_scope", lambda *_a, **_k: False)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _fake_summarizer,
                "plan": "sweep split-K",
            },
        )
    )

    out = capsys.readouterr().out
    assert "scope not recorded for iter 3" in out
    assert "no held-fixed constants recorded" not in out


def test_handoff_carries_the_scope_of_the_iterations_lesson(tmp_path, monkeypatch):
    """The planner reads handoffs; a refutation quoted from one needs its scope."""
    from kernelforge.workflows.forge_loop.iteration.handoffs import HandoffStore

    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)
    loop.run_state = RunState()
    loop.handoff_store = HandoffStore(str(workspace))

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "sweep split-K",
            },
        )
    )
    loop._record_iteration_handoff(
        iteration=3,
        decision="REVERT_PERF",
        optimization_plan_path="",
        session_sink={"plan": "sweep split-K"},
    )

    payload = json.loads(loop.handoff_store.path(3).read_text())
    assert payload["plan"].startswith("sweep split-K\nSCOPE: measured on ")
    assert "BLOCK_N=16" in payload["plan"]
    assert store.scope_of(3) is not None


def test_handoff_plan_is_unchanged_when_no_lesson_was_written(tmp_path, monkeypatch):
    from kernelforge.workflows.forge_loop.iteration.handoffs import HandoffStore

    loop, workspace = _make_loop(tmp_path, monkeypatch)
    _attach_lessons(loop, workspace)
    loop.run_state = RunState()
    loop.handoff_store = HandoffStore(str(workspace))

    loop._record_iteration_handoff(
        iteration=3,
        decision="NO_CHANGES",
        optimization_plan_path="",
        session_sink={"plan": "inspect only"},
    )

    payload = json.loads(loop.handoff_store.path(3).read_text())
    assert payload["plan"] == "inspect only"
    assert payload["lesson_path"] == ""


def test_record_lesson_skips_an_empty_iteration(tmp_path, monkeypatch):
    """No exploration recorded and no candidate measured -> no document."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(iteration=5),
            decision="NO_CHANGES",
            session_sink={"session_started": False, "summarize": None},
        )
    )

    assert store.existing_iterations() == []


def test_record_lesson_keeps_outcome_when_no_diff_summary_fails(tmp_path, monkeypatch):
    """A started, cut-off session must not disappear just because it has no diff."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)

    async def broken(_prompt: str) -> str:
        raise RuntimeError("resume handle expired")

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(
                iteration=10,
                session_end_reason="turn_cap",
                turns=40,
                wall_ms=None,
                snr_db=None,
            ),
            decision="NO_CHANGES",
            session_sink={
                "session_started": True,
                "summarize": broken,
                "findings": "",
                "progress_log": [],
            },
        )
    )

    text = store.read(10)
    assert text.strip() == (
        "SCOPE: measured on case | held fixed (not recorded) | "
        "no measured negative | whether a feasibility claim was disproved "
        "was not recorded\n\n"
        "OUTCOME: NO_CHANGES | session ended: turn_cap | turns 40 | "
        "summary unavailable: RuntimeError: resume handle expired"
    )


def test_record_lesson_uses_progress_when_no_diff_summary_is_unavailable(tmp_path, monkeypatch):
    """Machine-captured provider progress is the no-diff narrative fallback."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(
                iteration=11,
                session_end_reason="turn_cap",
                turns=40,
            ),
            decision="NO_CHANGES",
            session_sink={
                "session_started": True,
                "summarize": None,
                "plan": "evaluate wider vector loads",
                "findings": "",
                "progress_log": [
                    "tool: Read kernel.py",
                    "progress: not supported by codex backend",
                    "tool: Bash python driver.py --bench BLOCK_N=128",
                ],
            },
        )
    )

    text = store.read(11)
    assert text.startswith("(no agent summary) last observed:")
    assert "Summary unavailable: provider cannot resume the session" in text
    assert "Implementer turns: 40" in text
    assert "Final plan: evaluate wider vector loads" in text
    assert "tool: Read kernel.py" in text
    assert "tool: Bash python driver.py --bench BLOCK_N=128" in text
    assert "not supported by codex" not in text
    assert "OUTCOME: NO_CHANGES" in text


def test_record_lesson_without_an_agent_session(tmp_path, monkeypatch):
    """The baseline measurement path has nothing to summarize."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(iteration=6),
            decision="KEEP",
            session_sink={},
        )
    )

    assert store.existing_iterations() == []


def test_record_lesson_skips_the_summarizer_with_no_time_left(tmp_path, monkeypatch):
    """With no time to produce it, record the objective outcome alone."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)
    monkeypatch.setattr(loop, "_time_remaining", lambda: 0.0)

    started = {"n": 0}

    async def counting_summarizer(_prompt: str) -> str:
        started["n"] += 1
        return "should not run"

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(iteration=7),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": counting_summarizer,
            },
        )
    )

    assert started["n"] == 0
    assert "OUTCOME: REVERT_PERF" in store.read(7)


def test_record_lesson_still_summarizes_when_no_session_can_be_admitted(tmp_path, monkeypatch):
    """The last iteration of a session is the one whose record matters most.

    The loop stops admitting implementer sessions well before the clock runs out
    (``budget_reserve_sec``), but the campaign is resumable and its next
    session reads this document. Gating the summary on the session-admission
    reserve silently dropped that handoff record.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)
    # Below the session-admission reserve, far above what a summary needs.
    monkeypatch.setattr(loop, "_time_remaining", lambda: 600.0)
    assert loop._is_budget_exhausted() is True

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(iteration=9),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _fake_summarizer,
            },
        )
    )

    assert "BLOCK_N 128" in store.read(9)


def test_record_lesson_survives_a_failing_summarizer(tmp_path, monkeypatch, capsys):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)

    async def broken(_prompt: str) -> str:
        raise RuntimeError("session gone")

    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(iteration=8),
            decision="CRASH",
            session_sink={
                "session_started": True,
                "summarize": broken,
                "findings": "boom",
            },
        )
    )

    assert "OUTCOME: CRASH" in store.read(8)
    # The reason must reach the operator, not just log.debug: an outcome-only
    # run is otherwise indistinguishable from a provider that cannot resume.
    printed = capsys.readouterr().out
    assert "RuntimeError" in printed
    assert "session gone" in printed


def test_record_lesson_falls_back_when_summary_cannot_be_persisted(tmp_path, monkeypatch, capsys):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)
    real_write = store.write
    writes = {"count": 0}

    def fail_first_write(iteration, text):
        writes["count"] += 1
        if writes["count"] == 1:
            return None
        return real_write(iteration, text)

    monkeypatch.setattr(store, "write", fail_first_write)
    asyncio.run(
        loop._record_lesson(
            result=_lesson_result(iteration=12),
            decision="REVERT_VALIDATION",
            session_sink={
                "session_started": True,
                "summarize": _fake_summarizer,
                "findings": "compile error: invalid cast",
            },
            diff_summary="kernel.py | 2 +-",
        )
    )

    text = store.read(12)
    assert writes["count"] >= 2
    assert text.startswith("(no agent summary)")
    assert "compile error: invalid cast" in text
    assert "OUTCOME: REVERT_VALIDATION" in text
    assert "tried three tile shapes" not in text
    printed = capsys.readouterr().out
    assert "failed to persist lesson document" in printed
    assert "[lesson] recorded iter 12:" not in printed


def test_lesson_flows_from_one_iteration_into_the_next_prompt(tmp_path, monkeypatch):
    """Lock the whole chain the feature exists for.

    agent edits -> session_sink["summarize"] -> lessons/iter_NNN.md
    -> next iteration's prompt.

    Every hop has a unit test; this asserts they are actually connected, which
    is where a wiring regression would otherwise hide.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=2)
    # Two iterations: the first improves and is kept, the second does not.
    _stub_measurement(monkeypatch, walls=[0.5, 0.9])

    kernel = Path(loop.ic.kernel_file)
    prompts: list[str] = []
    edits = {"n": 0}

    async def summarizer(_prompt: str) -> str:
        return "vectorized the global load\n- [better] 128-bit loads | 1.8x"

    async def agent_fn(_kernel_path, experiment_history, session_sink):
        prompts.append(experiment_history)
        edits["n"] += 1
        kernel.write_text(f"def kernel():\n    return {edits['n'] + 1}\n")
        session_sink["plan"] = f"edit {edits['n']}"
        session_sink["end_reason"] = "candidate_submitted"
        session_sink["summarize"] = summarizer
        return "rationale"

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=agent_fn)))

    # The first iteration produced a document with both authors' halves.
    first = (workspace / "forge_experiments" / "lessons" / "iter_001.md").read_text()
    assert "vectorized the global load" in first
    assert "128-bit loads" in first
    assert "OUTCOME: KEEP" in first

    # ...and the second iteration's prompt carried it, plus the absolute path
    # of the directory holding the full history.
    assert len(prompts) == 2
    assert "vectorized the global load" not in prompts[0]  # nothing to inject yet
    assert "128-bit loads" in prompts[1]
    assert "Implementer session records from recent iterations" in prompts[1]
    lessons_dir = str((workspace / "forge_experiments" / "lessons").resolve())
    assert lessons_dir in prompts[1]


def test_prompt_omits_the_digest_once_lessons_exist(tmp_path, monkeypatch):
    """The digest is reserved for the supervisor once the header renders.

    Both carry the recent iterations, so inlining them together would spend the
    prompt budget saying the same thing twice.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=2)
    _stub_measurement(monkeypatch, walls=[0.5, 0.9])

    kernel = Path(loop.ic.kernel_file)
    prompts: list[str] = []
    edits = {"n": 0}

    async def summarizer(_prompt: str) -> str:
        return "Attempted one tile variant; measured 0.9x."

    async def agent_fn(_kernel_path, experiment_history, session_sink):
        prompts.append(experiment_history)
        edits["n"] += 1
        kernel.write_text(f"def kernel():\n    return {edits['n'] + 1}\n")
        session_sink["summarize"] = summarizer
        return "rationale"

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=agent_fn)))

    # The archive digest's own header would appear verbatim if it were inlined.
    assert "Solution archive — your lineage so far" not in prompts[1]
    assert "Long-Horizon Memory" in prompts[1]
    assert "Implementer session records from recent iterations" in prompts[1]
