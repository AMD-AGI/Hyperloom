# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guard for the ``ci-e2e`` concurrency group.

The workflow answers three triggers (``pull_request`` / ``issue_comment`` /
``workflow_dispatch``) and cancels in-progress runs that share its group. GitHub
resolves ``concurrency`` when the run is *created*, before the ``resolve`` job's
``if`` can decline the work, so a group keyed on the PR number alone lets any
comment on a PR cancel that PR's in-flight multi-hour GPU run and then skip
itself.

The fix keys non-``/retest`` comments to a group of their own, which only holds
while the group expression and ``resolve.if`` agree on what a retest comment is.
These tests pin that agreement; there is no way to unit-test the cancellation
itself short of running the workflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_RETEST_COMMAND = "/retest"


def _find_workflow() -> Path | None:
    """Locate ``ci-e2e.yml``; returns None when running from an installed wheel."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".github" / "workflows" / "ci-e2e.yml"
        if candidate.is_file():
            return candidate
    return None


_WORKFLOW = _find_workflow()

pytestmark = pytest.mark.skipif(
    _WORKFLOW is None,
    reason="ci-e2e guard needs the source checkout (.github/workflows/)",
)


@pytest.fixture(scope="module")
def workflow() -> dict:
    assert _WORKFLOW is not None
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def concurrency_group(workflow: dict) -> str:
    return " ".join(str(workflow["concurrency"]["group"]).split())


@pytest.fixture(scope="module")
def resolve_condition(workflow: dict) -> str:
    return " ".join(str(workflow["jobs"]["resolve"]["if"]).split())


def test_an_in_flight_run_is_still_preempted(workflow: dict) -> None:
    """The point of the group: a newer commit must not queue behind the old run."""
    assert workflow["concurrency"]["cancel-in-progress"] is True


def test_a_comment_that_cannot_start_a_run_cannot_cancel_one(concurrency_group: str) -> None:
    """Non-retest comments must land in a group of their own, keyed per comment."""
    assert "github.event_name == 'issue_comment'" in concurrency_group
    assert f"!contains(github.event.comment.body, '{_RETEST_COMMAND}')" in concurrency_group
    assert "github.event.comment.id" in concurrency_group


def test_the_group_and_the_gate_agree_on_what_a_retest_is(
    concurrency_group: str,
    resolve_condition: str,
) -> None:
    """Drift here silently restores the bug, in one direction or the other.

    A group looking for a command the gate no longer honours lets an ordinary
    comment cancel a run again; a gate honouring a command the group does not
    recognise stops a real retest from preempting the run it means to replace.
    """
    predicate = f"contains(github.event.comment.body, '{_RETEST_COMMAND}')"
    assert predicate in resolve_condition
    assert predicate in concurrency_group


def test_every_trigger_still_resolves_to_a_group(concurrency_group: str) -> None:
    """Each of the three triggers must contribute a key, or runs collide repo-wide."""
    for key in (
        "github.event.pull_request.number",  # pull_request
        "github.event.issue.number",  # issue_comment (/retest)
        "inputs.head_sha",  # workflow_dispatch (fork PR smoke)
        "inputs.head_ref",
        "github.ref",  # last-resort fallback
    ):
        assert key in concurrency_group
    # Dispatch used to share refs/heads/main and cancel-in-progress the
    # previous GPU run. head_sha must win over github.ref.
    assert concurrency_group.index("inputs.head_sha") < concurrency_group.index("github.ref")
