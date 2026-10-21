# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One session's retrieval evaluated: pitfalls avoided, insights taken up, and how shown directions fared."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.records import Learning, LearningRecord, NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import LearningKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.learning_search import LEARNINGS_LOG
from hyperloom_multibranch.notebook.rsi.archive import TreeArchive
from hyperloom_multibranch.notebook.rsi.learning_eval import evaluate_learnings, names, read_session
from hyperloom_multibranch.notebook.rsi.lessons import LessonPool, LessonRevision, lesson_id_for, revision_id_for
from hyperloom_multibranch.notebook.rsi.similarity import task_conditions
from hyperloom_multibranch.notebook.tests.factories import FakeClock, profile
from hyperloom_multibranch.notebook.tests.sessions import GEMM, SPLIT_KV, scripted_session

CONDITIONS = task_conditions(profile())
CHUNKED = "chunked prefill makes the server exit during warmup on this image"

PITFALL = Learning("s2", LearningKind.PITFALL, "split kv regresses at this concurrency", "attention", "split kv")
INSIGHT = Learning("s1", LearningKind.INSIGHT, "the moe gemm wants a tuned kernel", "moe gemm", GEMM.mechanism)
VAGUE = Learning("s3", LearningKind.TOOLING, "the profiler wants record shapes on")


def _between(snapshot: NotebookSnapshot, first: str, then: str) -> float:
    at = {record.request_id: record.requested_at for record in snapshot.records}
    return (at[first] + at[then]) / 2


def _log(snapshot: NotebookSnapshot) -> list[dict[str, Any]]:
    """s2 shown the split kv pitfall after r2; s1 the gemm insight after r1, twice, once with a vague learning."""

    after_r1 = _between(snapshot, "r1", "r4")
    return [
        {"at": _between(snapshot, "r2", "r3"), "branch": "s2", "source": "claim", "here": ["s2:1"], "past": []},
        {"at": after_r1, "branch": "s1", "source": "brief", "here": ["s1:1", "s3:1"], "past": ["exp-1"]},
        {"at": after_r1 + 0.0001, "branch": "s1", "source": "tool", "here": ["s1:1"], "past": ["exp-1", "exp-2"]},
    ]


def test_each_learning_counts_once_at_its_first_showing_and_each_later_test_once(snapshot: NotebookSnapshot) -> None:
    learnt = dataclasses.replace(
        snapshot,
        learnings=(
            LearningRecord("s2:1", 1.0, PITFALL),
            LearningRecord("s1:1", 2.0, INSIGHT),
            LearningRecord("s3:1", 3.0, VAGUE),
        ),
    )

    found = evaluate_learnings(learnt, _log(snapshot))

    assert (found.retrievals, found.by_source) == (3, {"claim": 1, "brief": 1, "tool": 1})
    # s2 tested chunked prefill and fp8 kv after the pitfall, never split kv; s1's next test was the gemm again.
    assert (found.pitfalls_shown, found.pitfalls_avoided, found.insights_shown, found.insights_taken_up) == (1, 1, 1, 1)
    assert (found.tests_in_shown_directions, found.committed_in_shown_directions) == (1, 0)
    assert (found.tests_elsewhere, found.committed_elsewhere) == (2, 0)
    assert (found.past_shown, found.past_distinct) == (3, 2)
    assert "| pitfalls avoided | 100% (1/1) |" in found.render()


def test_a_brief_lead_counts_once_and_is_taken_up_by_a_later_test_making_its_change(
    snapshot: NotebookSnapshot,
) -> None:
    at = {record.request_id: record.requested_at for record in snapshot.records}
    before_r1, after_r2 = at["r1"] - 0.001, _between(snapshot, "r2", "r3")
    seqs = {"ref": "exp-seqs", "change": "--max-num-seqs 256"}
    log = [
        {"at": before_r1, "branch": "s1", "source": "brief", "here": [], "past": [], "leads": [seqs]},
        {"at": after_r2, "branch": "s1", "source": "brief", "here": [], "past": [], "leads": [seqs]},
        {
            "at": after_r2,
            "branch": "s2",
            "source": "brief",
            "here": [],
            "past": [],
            "leads": [
                {"ref": "exp-fp8", "change": "--kv-cache-dtype fp8"},
                {"ref": "inference:older", "change": "--enable-expert-parallel"},
                {"ref": "exp-unnamed", "change": ""},
            ],
        },
    ]

    found = evaluate_learnings(snapshot, log)

    # s1 committed --max-num-seqs 256 (r1); s2 made the fp8 change (r5), which failed the accuracy check.
    assert (found.leads_shown, found.leads_taken_up, found.leads_committed) == (4, 2, 1)
    assert "| brief leads taken up | 50% (2/4) |" in found.render()


def test_a_lead_counts_once_and_is_taken_up_by_a_later_test_making_every_change_it_names(
    snapshot: NotebookSnapshot,
) -> None:
    before_r1 = min(record.requested_at for record in snapshot.records) - 1.0
    after_r2 = _between(snapshot, "r2", "r3")
    fp8 = {"ref": "exp-fp8", "change": "--kv-cache-dtype fp8"}
    log = [
        {
            "at": before_r1,
            "branch": "s1",
            "source": "brief",
            "here": [],
            "past": [],
            "leads": [
                {"ref": "exp-seqs", "change": "--max-num-seqs 256"},
            ],
        },
        {
            "at": after_r2,
            "branch": "s2",
            "source": "brief",
            "here": [],
            "past": [],
            "leads": [
                fp8,
                {"ref": "exp-never", "change": "--speculative-config ngram"},
                {"ref": "inference:older", "change": ""},
            ],
        },
        {"at": after_r2 + 0.0001, "branch": "s2", "source": "brief", "here": [], "past": [], "leads": [fp8]},
    ]

    found = evaluate_learnings(snapshot, log)

    # r1 made exp-seqs' change and committed; r5 made exp-fp8's and failed the accuracy check.
    assert (found.leads_shown, found.leads_taken_up, found.leads_committed) == (4, 2, 1)
    assert "| brief leads taken up | 50% (2/4) |" in found.render()


def test_a_learning_names_a_direction_by_what_it_gives_of_it() -> None:
    target_only = dataclasses.replace(INSIGHT, mechanism="")

    assert names(INSIGHT, GEMM) and names(target_only, dataclasses.replace(GEMM, mechanism="another kernel"))
    assert not names(INSIGHT, SPLIT_KV) and not names(VAGUE, GEMM) and not names(INSIGHT, None)


async def test_a_session_is_read_off_its_notebook_folder_and_its_log(tmp_path: Path) -> None:
    clock, root = FakeClock(), tmp_path / "notebook"
    await scripted_session(root, clock)
    s2 = LocalNotebook(root, Writer(Role.SPECIALIST_RUNTIME, "s2", "s2"), clock=clock)
    shown = await s2.append_learning(PITFALL)
    log = tmp_path / LEARNINGS_LOG
    line = {"at": clock.now, "branch": "s2", "source": "tool", "here": [shown.entry_id], "past": []}
    log.write_text(json.dumps(line) + "\n" + "not json\n", encoding="utf-8")

    found = await read_session(root, log)

    assert (found.retrievals, found.pitfalls_shown, found.pitfalls_avoided, found.tests_elsewhere) == (1, 1, 1, 0)


async def test_pooled_lessons_shown_count_apart_and_one_written_again_unshown_is_a_retrieval_miss(
    tmp_path: Path,
) -> None:
    clock, root = FakeClock(), tmp_path / "notebook"
    await scripted_session(root, clock)
    s2 = LocalNotebook(root, Writer(Role.SPECIALIST_RUNTIME, "s2", "s2"), clock=clock)
    await s2.append_learning(
        Learning("s2", LearningKind.PITFALL, "the server exits during warmup with chunked prefill")
    )
    pool = LessonPool.beside(TreeArchive(tmp_path / "archive"))
    shown, missed = lesson_id_for("shown"), lesson_id_for("missed")
    for lesson_id, text in ((shown, "fp8 kv cache costs accuracy on this model"), (missed, CHUNKED)):
        pool.write(
            LessonRevision(revision_id_for(lesson_id), lesson_id, LearningKind.PITFALL, text, CONDITIONS, "x", 1.0, "p")
        )
    log = tmp_path / LEARNINGS_LOG
    line = {"at": clock.now, "branch": "s2", "source": "brief", "here": [], "past": [shown, shown, "exp-1"]}
    log.write_text(json.dumps(line) + "\n", encoding="utf-8")

    found = await read_session(root, log, tmp_path / "archive")
    unpooled = await read_session(root, log)

    assert (found.past_shown, found.lessons_shown, found.lessons_distinct) == (3, 2, 1)
    assert (found.learnings_written, found.rediscovered) == (1, 1)
    assert "| learnings that say a past lesson again, unshown | 100% (1/1) |" in found.render()
    assert unpooled.rediscovered == 0
