# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which server a long-context check asked, as the result records it, and the outcome a reader derives from the result.

An AgentX test with a golden layer checks on a real-acceptance server, not its measured one; ``LongContextResult.server``
says so, read from the outputs file the runner wrote and rederived from the verified bundle like the other bundle
fields. A record written before the field existed is a ``held`` check.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts._validate import MeasurementError
from hyperloom.contracts.long_context import (
    LONG_CONTEXT_COMPLETE,
    LONG_CONTEXT_HELD,
    LONG_CONTEXT_INCOMPLETE,
    LONG_CONTEXT_REAL_ACCEPTANCE,
    LongContextResult,
    LongContextRoot,
    refuses,
)
from hyperloom.measure.long_context.plan import LONG_CONTEXT_DIR, OUTPUTS_FILE
from hyperloom.measure.long_context.rounds import CheckServer, run_long_context
from hyperloom.measure.long_context.tests.support import PROMPTS, FakeServer, prompt_set, staged_plan
from hyperloom_multibranch.toolbox.long_context import first_difference, long_context_of, rederive
from hyperloom_multibranch.toolbox.tests.test_long_context import _Host


def _landed(tmp_path: Path, server: str, **fake: object):
    plan = staged_plan(tmp_path / "cache")
    with FakeServer(**fake) as answering:  # type: ignore[arg-type]
        asyncio.run(
            run_long_context(plan, prompt_set(), _Host(answering), tmp_path / "out", server=CheckServer(4096, server))
        )
    return tmp_path / "out", plan


def _outputs_path(directory: Path) -> Path:
    return directory / LONG_CONTEXT_DIR / OUTPUTS_FILE


def _edit(directory: Path, **changes: object) -> None:
    document = json.loads(_outputs_path(directory).read_text())
    for key, value in changes.items():
        if value is None:
            document.pop(key)
        else:
            document[key] = value
    _outputs_path(directory).write_text(json.dumps(document))


def _reading(directory: Path, plan) -> LongContextResult:  # type: ignore[no-untyped-def]
    return LongContextResult.from_document(json.loads(json.dumps(long_context_of(plan, directory).to_document())))


def test_a_check_on_the_real_acceptance_server_records_it_and_the_bundle_supports_it(tmp_path) -> None:
    directory, plan = _landed(tmp_path, LONG_CONTEXT_REAL_ACCEPTANCE)

    check = _reading(directory, plan)

    assert (check.status, check.server, check.n) == (LONG_CONTEXT_COMPLETE, LONG_CONTEXT_REAL_ACCEPTANCE, PROMPTS)
    assert check.to_document()["server"] == LONG_CONTEXT_REAL_ACCEPTANCE
    assert first_difference(check, rederive(check, directory)) is None


def test_a_check_on_the_held_server_writes_no_server_key_into_the_result(tmp_path) -> None:
    directory, plan = _landed(tmp_path, LONG_CONTEXT_HELD)

    check = long_context_of(plan, directory)

    assert check.server == LONG_CONTEXT_HELD and "server" not in check.to_document()
    assert json.loads(_outputs_path(directory).read_text())["server"] == LONG_CONTEXT_HELD


def test_outputs_and_results_written_before_the_field_read_as_held(tmp_path) -> None:
    directory, plan = _landed(tmp_path, LONG_CONTEXT_HELD)
    _edit(directory, server=None)

    check = long_context_of(plan, directory)

    assert (check.status, check.server) == (LONG_CONTEXT_COMPLETE, LONG_CONTEXT_HELD)
    assert first_difference(check, rederive(check, directory)) is None


def test_a_recorded_server_the_bundle_does_not_support_is_named(tmp_path) -> None:
    directory, plan = _landed(tmp_path, LONG_CONTEXT_HELD)
    recorded = replace(long_context_of(plan, directory), server=LONG_CONTEXT_REAL_ACCEPTANCE)

    message = first_difference(recorded, rederive(recorded, directory))

    assert message == "long_context.server: recorded 'real_acceptance', the bundle's outputs give 'held'"


def test_an_outputs_file_naming_another_server_is_unread_and_the_check_incomplete(tmp_path) -> None:
    directory, plan = _landed(tmp_path, LONG_CONTEXT_HELD)
    _edit(directory, server="simulated")

    check = long_context_of(plan, directory)

    assert check.status == LONG_CONTEXT_INCOMPLETE and "server" in check.reason and check.score is None


def test_a_result_naming_another_server_is_refused(tmp_path) -> None:
    directory, plan = _landed(tmp_path, LONG_CONTEXT_HELD)
    document = {**long_context_of(plan, directory).to_document(), "server": "simulated"}

    with pytest.raises(MeasurementError, match="server must be one of"):
        LongContextResult.from_document(document)


def test_a_reader_derives_incomplete_dropped_or_passed_from_the_status_and_the_bar(tmp_path) -> None:
    def outcome(check: LongContextResult) -> str:
        if check.status == LONG_CONTEXT_INCOMPLETE:
            return "incomplete"
        return "dropped" if refuses(check, check.root) else "passed"

    good_dir, plan = _landed(tmp_path / "good", LONG_CONTEXT_REAL_ACCEPTANCE)
    bad_dir, _ = _landed(tmp_path / "bad", LONG_CONTEXT_REAL_ACCEPTANCE, broken={"niah"})
    good, bad = long_context_of(plan, good_dir), long_context_of(plan, bad_dir)
    root = LongContextRoot(good.prompt_set_sha256, ("root-1",), PROMPTS, PROMPTS, 1.0, 0.8, 1.0)
    unanswered = replace(long_context_of(plan, good_dir), root=root)
    unanswered = LongContextResult.from_document(
        {**unanswered.to_document(), "status": LONG_CONTEXT_INCOMPLETE, "reason": "check server did not start: crash",
         "n": 0, "passed": 0, "score": None, "tasks": [], "prompts": []}
    )  # fmt: skip

    assert outcome(replace(good, root=root)) == "passed"
    assert outcome(replace(bad, root=root)) == "dropped"
    assert outcome(unanswered) == "incomplete"
