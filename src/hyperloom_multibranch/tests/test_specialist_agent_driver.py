# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The agent drivers (:mod:`hyperloom_multibranch.specialist.agent_driver`): which driver a checkout gets, the
capabilities the session's planners ask of it, and how each makes a manifest and reads an agent's keeps and its exit.
The Arbor checkout is the adapter tests' fake."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.contracts.recipe import LaunchConfig
from hyperloom_multibranch.scheduling.agent_plans import arbor_agents
from hyperloom_multibranch.specialist.agent_driver import (
    ARBOR,
    DRIVERS,
    ONBOARDED,
    AgentCheckout,
    Capabilities,
    ConfigRefusal,
    Ended,
    EndKind,
    KeepPlace,
    SnapshotKeeps,
    WorkspaceKeeps,
    driver_for,
)
from hyperloom_multibranch.specialist.agent_manifest import EXIT_FILE, REFUSED_FILE, AgentManifest
from hyperloom_multibranch.specialist.arbor_adapter import BRANCH_RUN, MARKERS
from hyperloom_multibranch.tests.test_specialist_arbor_adapter import CHECKOUT_FILES, _committed, _spec, _write

SESSION = "run"
SNAPSHOTS = {"snapshots": {"session": SESSION}}
KEEP_FILE = {"keep": {"path": "run/state.json", "field": "keeps"}}


def _manifest(**fields: object) -> AgentManifest:
    """A manifest of an agent in its checkout, with ``fields`` over the ones every manifest has."""

    base = {"source": "/agents/a", "commit": "c0ffee", "config_file": "c.json", "config": {}, "launch": "run"}
    return AgentManifest.read({**base, **fields})


def _place(tmp_path: Path) -> KeepPlace:
    """Where a run reads the keeps of an agent checked out under ``tmp_path``; marks read no worktree."""

    agent = tmp_path / "agent"
    return KeepPlace(agent, agent / "src", SimpleNamespace(), "vllm", LaunchConfig(), {}, lambda *_: "")


def test_an_arbor_checkout_gets_arbor_s_driver_which_has_every_capability(tmp_path: Path) -> None:
    checkout = _write(tmp_path / "checkout", CHECKOUT_FILES)

    driver = driver_for(checkout)

    assert driver is ARBOR and driver.name == "arbor" and driver.config_from_spec
    assert driver.capabilities == Capabilities(plan_share=True, dossier_pass=True, review_host=True)


@pytest.mark.parametrize("missing", MARKERS)
def test_any_other_checkout_falls_back_to_the_onboarded_driver_which_plans_nothing(
    tmp_path: Path, missing: str
) -> None:
    lacking = _write(tmp_path / "lacking", CHECKOUT_FILES)
    (lacking / missing).unlink()
    other = _write(tmp_path / "other", {"README.md": "another agent\n"})

    for checkout in (lacking, other, tmp_path / "nowhere"):
        driver = driver_for(checkout)
        assert driver is ONBOARDED and driver.name == "onboarded" and not driver.config_from_spec
        assert driver.capabilities == Capabilities()


def test_the_drivers_are_asked_in_order_and_the_onboarded_one_takes_what_none_detects(tmp_path: Path) -> None:
    arbor = _write(tmp_path / "arbor", CHECKOUT_FILES)

    assert DRIVERS == (ARBOR, ONBOARDED)
    assert driver_for(arbor, (ONBOARDED, ARBOR)) is ONBOARDED
    assert driver_for(tmp_path / "nowhere", (ARBOR,)) is ONBOARDED


async def test_the_planners_take_the_agents_whose_driver_has_a_plan_share(tmp_path: Path) -> None:
    arbor = str(_committed(_write(tmp_path / "arbor", CHECKOUT_FILES)))
    other = str(_committed(_write(tmp_path / "other", {"README.md": "another agent\n"})))

    found = await arbor_agents((other, arbor, other), tmp_path / "work")

    assert list(found) == [1]
    fetched = sorted((tmp_path / "work" / "sources").iterdir())
    assert [driver_for(path).capabilities.plan_share for path in fetched] == [False, True]


async def test_arbor_s_manifest_follows_from_the_spec_and_a_refusal_of_it_is_final(tmp_path: Path) -> None:
    spec = _spec(tmp_path, workload={"kind": "agentx", "conc": 8})
    checkout = AgentCheckout("/agents/arbor", "c0ffee", _write(tmp_path / "checkout", CHECKOUT_FILES))

    async def asked() -> AgentManifest:
        raise AssertionError("Arbor is never onboarded by the model")

    with pytest.raises(ConfigRefusal, match="^Arbor would refuse the config written for it: an AgentX recipe's"):
        await ARBOR.manifest(spec, checkout, BRANCH_RUN, asked)
    manifest = _manifest(config={"component": {"host": "hyperloom-mb"}}, **SNAPSHOTS)
    gone = tmp_path / "gone"
    fitted = ARBOR.fit(manifest, replace(spec, dossier=str(gone)), checkout.path)
    assert fitted == (manifest, (f"the dossier pass's dossier {gone} is not there: started without a dossier",))


async def test_any_other_agent_s_manifest_is_the_onboarding_model_s_as_it_wrote_it(tmp_path: Path) -> None:
    spec, manifest = _spec(tmp_path), _manifest(**KEEP_FILE)
    checkout = AgentCheckout("/agents/other", "c0ffee", tmp_path / "checkout")

    async def asked() -> AgentManifest:
        return manifest

    assert await ONBOARDED.manifest(spec, checkout, BRANCH_RUN, asked) is manifest
    assert ONBOARDED.fit(manifest, spec, checkout.path) == (manifest, ())


def test_snapshots_are_read_for_arbor_and_for_any_agent_its_manifest_runs_as_a_component(tmp_path: Path) -> None:
    place = _place(tmp_path)
    snapshots, keep_file = _manifest(**SNAPSHOTS), _manifest(**KEEP_FILE)

    assert isinstance(ARBOR.keeps(snapshots, place), SnapshotKeeps)
    assert isinstance(ONBOARDED.keeps(snapshots, place), SnapshotKeeps)
    assert isinstance(ONBOARDED.keeps(keep_file, place), WorkspaceKeeps)
    assert ONBOARDED.keeps(_manifest(), place).mark() is None


def test_the_newest_snapshot_marks_the_keeps_and_a_latest_that_names_none_is_said(tmp_path: Path) -> None:
    place = _place(tmp_path)
    reader = ARBOR.keeps(_manifest(**SNAPSHOTS), place)
    outbox = place.checkout / SESSION / "keeps"

    assert (reader.mark(), reader.unread) == (None, "")
    outbox.mkdir(parents=True)
    (outbox / "LATEST").write_text("two")
    assert reader.mark() is None and reader.unread.startswith(f"the LATEST of {SESSION}/keeps names no keep")
    (outbox / "LATEST").write_text("2\n")
    assert (reader.mark(), reader.unread) == (2, "")


def test_a_keep_file_s_count_marks_the_keeps_and_one_being_written_changes_nothing(tmp_path: Path) -> None:
    place = _place(tmp_path)
    reader = ONBOARDED.keeps(_manifest(**KEEP_FILE), place)
    state = place.checkout / "run" / "state.json"

    assert reader.mark() is None and reader.unread.startswith("the keep file run/state.json cannot be read")
    unread = reader.unread
    state.parent.mkdir(parents=True)
    state.write_text('{"keeps": [')
    assert (reader.mark(), reader.unread) == (None, unread)
    state.write_text(json.dumps({"keeps": [{"decision": "KEEP"}] * 2}))
    assert (reader.mark(), reader.unread) == (2, "")


@pytest.mark.parametrize(
    ("returncode", "ended"),
    [
        (2, Ended(EndKind.REFUSED, "no deadline: component.deadline is None")),
        (0, Ended(EndKind.FINISHED, "deadline")),
        (1, Ended(EndKind.CRASHED)),
        (-15, Ended(EndKind.CRASHED)),
    ],
)
def test_an_agent_run_as_a_component_refuses_with_exit_2_and_says_why_it_finished(
    tmp_path: Path, returncode: int, ended: Ended
) -> None:
    session = tmp_path / SESSION
    session.mkdir()
    (session / REFUSED_FILE).write_text(json.dumps({"reasons": ["no deadline: component.deadline is None"]}))
    (session / EXIT_FILE).write_text(json.dumps({"reason": "deadline", "keeps": 0}))
    manifest = _manifest(**SNAPSHOTS)

    assert ARBOR.ended(returncode, manifest, tmp_path) == ONBOARDED.ended(returncode, manifest, tmp_path) == ended


def test_an_agent_with_no_snapshots_never_refuses_and_says_nothing_of_its_end(tmp_path: Path) -> None:
    manifest = _manifest(**KEEP_FILE)

    assert ONBOARDED.ended(2, manifest, tmp_path) == Ended(EndKind.CRASHED)
    assert ONBOARDED.ended(0, manifest, tmp_path) == Ended(EndKind.FINISHED)
    assert ARBOR.ended(0, _manifest(**SNAPSHOTS), tmp_path) == Ended(EndKind.FINISHED, "it wrote no exit reason")
