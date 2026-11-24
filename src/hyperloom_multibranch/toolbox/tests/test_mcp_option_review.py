# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``option_review`` on a running MCP service: a specialist records its verdicts on the options the launch leaves
unset, for its own branch, and reads what its review still leaves out; until it leaves none out, its tests that change
server arguments are refused."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom_multibranch.blackboard.tests.factories import FakeClock
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES
from hyperloom_multibranch.toolbox.mcp_tools import SpecialistToolOptions
from hyperloom_multibranch.toolbox.option_reviews import NOTE_CHARS, OptionReviews, ReviewStanding
from hyperloom_multibranch.toolbox.tests.test_mcp_tools import Rig, _running, seeded  # noqa: F401
from hyperloom_multibranch.toolbox.tests.test_option_reviews import reviewed_toolbox

NOTE = "read in the source"


@pytest.fixture
def reviews(tmp_path: Path) -> OptionReviews:
    """The reviews of a session whose launch leaves ``--threads``, ``--no-jinja`` and ``--keep`` unset."""
    directory = tmp_path / "MB-toolbox"
    directory.mkdir()
    return reviewed_toolbox(directory)


@pytest.fixture
def rig(seeded: tuple[Path, FakeClock], reviews: OptionReviews) -> Iterator[Rig]:
    yield from _running(seeded, SpecialistToolOptions(claim_ttl_seconds=60.0, option_reviews=reviews))


def _tool(rig: Rig, name: str) -> dict[str, object] | None:
    _, reply = rig.post(rig.token("s1"), {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    return next((tool for tool in reply["result"]["tools"] if tool["name"] == name), None)


def _entry(option: str, verdict: str = "no_effect_on_step", note: str | None = None) -> dict[str, str]:
    return {"option": option, "verdict": verdict, "note": f"{NOTE}: {option}" if note is None else note}


def test_a_review_is_recorded_for_the_branch_of_the_token_and_answers_what_it_leaves_out(
    rig: Rig, reviews: OptionReviews, tmp_path: Path
) -> None:
    first = json.loads(rig.ok(rig.token("s1"), "option_review", entries=[_entry("-t", "candidate"), _entry("--keep")]))

    assert first == {
        "total": 3,
        "covered": 2,
        "remaining": 1,
        "first_remaining": ["--no-jinja"],
        "complete": False,
        "candidates": ["--threads"],
    }
    last = json.loads(rig.ok(rig.token("s1"), "option_review", entries=[_entry("--jinja")]))
    assert (last["covered"], last["remaining"], last["first_remaining"], last["complete"]) == (3, 0, [], True)
    assert reviews.standing("s2").covered == 0
    kept = reviewed_toolbox(tmp_path / "MB-toolbox").standing("s1")
    assert kept == ReviewStanding(3, 3, (), ("--threads",)), "a review outlives the process that took it"
    assert json.loads(rig.ok(rig.token("s1"), "option_review", entries=[]))["covered"] == 3


def test_an_entry_that_cannot_be_taken_fails_the_call_naming_it_and_stores_nothing(
    rig: Rig, reviews: OptionReviews
) -> None:
    entries = [
        _entry("--threads"),
        _entry("--nope"),
        _entry("--ctx-size"),
        _entry("--old-size"),
        _entry("--no-jinja", note="  "),
        _entry("--keep", note="x" * (NOTE_CHARS + 1)),
    ]

    text = rig.failed(rig.token("s1"), "option_review", entries=entries)

    assert "--nope: the server lists no such option" in text
    assert "--ctx-size: the launch sets it" in text
    assert "--old-size: the server lists no such option" in text
    assert "--no-jinja: the note is empty" in text
    assert f"--keep: the note has {NOTE_CHARS + 1} characters, {NOTE_CHARS} at most" in text
    assert reviews.standing("s1").covered == 0


@pytest.mark.parametrize(
    "arguments",
    [
        {"entries": [_entry("--threads", "maybe")]},
        {"entries": [_entry("--threads")], "branch": "s2"},
    ],
    ids=["unknown-verdict", "names-a-branch"],
)
def test_a_review_the_schema_does_not_take_is_invalid_params(
    rig: Rig, reviews: OptionReviews, arguments: dict[str, object]
) -> None:
    _, reply = rig.call(rig.token("s1"), "option_review", **arguments)

    assert reply["error"]["code"] == -32602, reply
    assert reviews.standing("s1").covered == 0 and reviews.standing("s2").covered == 0


def test_the_review_tool_s_description_names_no_server_option(rig: Rig) -> None:
    tool = _tool(rig, "option_review")

    assert tool is not None
    described = json.dumps(tool)
    assert re.search(r"(?<![\w-])--?[A-Za-z]", described) is None, "a tool description names no server option"


def test_a_session_with_no_option_reviews_serves_no_review_tool(seeded: tuple[Path, FakeClock]) -> None:
    for plain in _running(seeded, SpecialistToolOptions(claim_ttl_seconds=60.0)):
        assert _tool(plain, "option_review") is None


SUBMIT = {"kind": "test", "purpose": "explore", "parent_commit": "c1"}
PATCH = {"component": "llamacpp", "add_patches": [{"path": "patch/llamacpp/0001.patch", "sha256": "a" * 64}]}
EVERY_UNSET = [_entry("--threads", "candidate"), _entry("--no-jinja"), _entry("--keep")]


def test_a_test_that_changes_server_arguments_waits_for_its_branchs_review_of_every_unset_option(rig: Rig) -> None:
    token, change = rig.token("s1"), {"config": {"extra_server_args": "--threads 8"}}

    refused = rig.failed(token, "submit_test", **SUBMIT, delta=change)

    assert refused == (
        "rejected (options_unreviewed): 3 option(s) the launch leaves unset have no verdict in your branch's option "
        "review (first: --threads, --no-jinja, --keep): record them with option_review, then submit again"
    )
    assert "options_unreviewed" in REJECTION_CODES and rig.toolbox.submitted == []
    rig.ok(token, "option_review", entries=EVERY_UNSET[:2])
    assert "1 option(s)" in rig.failed(token, "submit_test", **SUBMIT, delta=change)
    rig.ok(token, "option_review", entries=EVERY_UNSET[2:])
    rig.ok(token, "submit_test", **SUBMIT, delta=change)
    assert [request.branch_id for request in rig.toolbox.submitted] == ["s1"]
    assert "3 option(s)" in rig.failed(rig.token("s2"), "submit_test", **SUBMIT, delta=change)


def test_a_delta_that_only_removes_a_server_argument_waits_for_the_review(rig: Rig) -> None:
    change = {"config": {"remove_args": ["--ctx-size"]}}
    refused = rig.failed(rig.token("s1"), "submit_test", **SUBMIT, delta=change)

    assert refused.startswith("rejected (options_unreviewed): 3 option(s)") and rig.toolbox.submitted == []


@pytest.mark.parametrize(
    "request_",
    [
        {**SUBMIT, "delta": {"config": {"extra_envs": {"SERVER_THREADS": "8"}}}},
        {**SUBMIT, "delta": {"config": {"unset_envs": ["SERVER_THREADS"]}}},
        {**SUBMIT, "delta": {"components": [PATCH]}},
    ],
    ids=["sets-a-variable", "unsets-a-variable", "patches-the-source"],
)
def test_a_test_that_changes_no_server_argument_needs_no_review(rig: Rig, request_: dict[str, object]) -> None:
    rig.ok(rig.token("s1"), "submit_test", **request_)

    assert len(rig.toolbox.submitted) == 1
