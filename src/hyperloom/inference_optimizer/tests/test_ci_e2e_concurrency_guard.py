# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guard for what starts a ``ci-e2e`` run and what can cancel one."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_RUN_LABEL = "run-e2e"


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
def triggers(workflow: dict) -> dict:
    # YAML 1.1 reads the bare key `on` as the boolean True.
    return workflow[True]


@pytest.fixture(scope="module")
def concurrency_group(workflow: dict) -> str:
    return " ".join(str(workflow["jobs"]["e2e"]["concurrency"]["group"]).split())


@pytest.fixture(scope="module")
def resolve_condition(workflow: dict) -> str:
    return " ".join(str(workflow["jobs"]["resolve"]["if"]).split())


def test_a_pr_runs_only_once_the_run_label_is_on(resolve_condition: str) -> None:
    """Adding the label starts a run; other PR events run only while it stays on."""
    assert f"github.event.label.name == '{_RUN_LABEL}'" in resolve_condition
    assert f"contains(github.event.pull_request.labels.*.name, '{_RUN_LABEL}')" in resolve_condition


def test_nothing_but_the_label_or_a_dispatch_can_start_a_run(triggers: dict) -> None:
    """A comment or a label being removed is not a way in."""
    assert set(triggers) == {"pull_request", "workflow_dispatch"}
    assert "unlabeled" not in triggers["pull_request"]["types"]


def test_an_in_flight_run_is_still_preempted(workflow: dict) -> None:
    """The point of the group: a newer commit must not queue behind the old run."""
    assert workflow["jobs"]["e2e"]["concurrency"]["cancel-in-progress"] is True


def test_an_event_that_starts_nothing_cannot_cancel_a_run(workflow: dict) -> None:
    """REGRESSION GUARD. A workflow-level group is joined by every event on the PR,
    adding an unrelated label included, before any job decides to run -- and with
    cancel-in-progress that cancels the multi-hour GPU run in flight. Only the job
    that runs after `resolve` may join the group."""
    assert "concurrency" not in workflow
    assert workflow["jobs"]["e2e"]["needs"] == "resolve"


def test_every_trigger_still_resolves_to_a_group(concurrency_group: str) -> None:
    """Each trigger must contribute a key, or runs collide repo-wide."""
    for key in (
        "needs.resolve.outputs.pr_number",  # pull_request
        "needs.resolve.outputs.head_sha",  # workflow_dispatch (fork PR smoke)
        "needs.resolve.outputs.head_ref",
        "github.ref",  # last-resort fallback
    ):
        assert key in concurrency_group
    # Dispatch used to share refs/heads/main and cancel-in-progress the previous GPU run. head_sha must win over
    # github.ref.
    assert concurrency_group.index("needs.resolve.outputs.head_sha") < concurrency_group.index("github.ref")
