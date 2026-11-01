# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import json
import time
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.specialist import arbor_adapter
from hyperloom_multibranch.specialist.agent_onboarding import PROFILE_FILE
from hyperloom_multibranch.specialist.arbor_adapter import DOSSIER_PASS, MAPPED
from hyperloom_multibranch.specialist.dossier_pass import (
    DOSSIER_CHARGE,
    DOSSIER_HOLDER,
    DossierOutcome,
    DossierSetup,
    DossierStart,
    run_dossier_pass,
)
from hyperloom_multibranch.tests.branch_rig import branch_rig
from hyperloom_multibranch.tests.test_specialist_arbor_adapter import CHECKOUT_FILES, _committed, _spec, _write
from hyperloom_multibranch.tests.test_specialist_autonomous_session import ROOT, LocalSandbox, Toolbox

#: Arbor's entry point in a fake checkout, run as a dossier pass: what it does is fake-dossier.json's ``do``. ``mapped``
#: writes a dossier, the map and exits 0 with ``mapped``; ``nomap`` writes the dossier and ends with ``deadline``;
#: ``hang`` writes the dossier and waits to be stopped; ``crash`` dies after its start; ``refuse`` refuses its config
#: (exit 2). Every run keeps the config it was given in seen-config.json.
FAKE_DOSSIER = """
import json, pathlib, sys, time
config = json.loads(pathlib.Path(sys.argv[sys.argv.index("--config") + 1]).read_text())
pathlib.Path("seen-config.json").write_text(json.dumps(config))
fake = json.loads(pathlib.Path("fake-dossier.json").read_text())
do = fake["do"]
session = pathlib.Path(config["session_root"])
session.mkdir(parents=True, exist_ok=True)


def leave(reason):
    (session / "component_exit.json").write_text(json.dumps({"reason": reason}))
    sys.exit(0)


if do == "refuse":
    (session / "component_refused.json").write_text(json.dumps({"reasons": ["the pack has no dossier skill"]}))
    sys.exit(2)
if do == "crash":
    time.sleep(0.5)
    print("Traceback: the profiler died", flush=True)
    sys.exit(1)
dossier = (session if fake.get("in_session") else pathlib.Path(".")) / "docs/dossier/qwen3-8b-1k1k-c64"
dossier.mkdir(parents=True, exist_ok=True)
(dossier / "README.md").write_text("# Qwen3-8B 1k/1k C64\\n")
if do == "hang":
    time.sleep(600)
if do == "nomap":
    leave("deadline")
directions = [{"id": "m-attn", "kind": "increment", "title": "Attention", "status": "open"}]
(session / "directions.json").write_text(json.dumps({"directions": directions}))
leave("mapped")
"""
SHARES = OpShares("c0ffee", "prelude:p1", "", {"paged_attention_ll4mi_QKV_mfma4_kernel": 32.5, "Cijk_Alik": 27.3})


async def _tick(_seconds: float) -> None:
    await asyncio.sleep(0.05)


async def _pass(
    tmp_path: Path, do: str, *, cards: int = 1, seconds: float = 30.0, arbor: bool = True, in_session: bool = False
) -> tuple[DossierOutcome, Toolbox, Path, DossierSetup]:
    """The dossier pass of a fake Arbor doing ``do``, on a rig whose toolbox has ``cards`` free, given ``seconds``,
    writing its dossier in its session (``in_session``) or, by default, in its checkout."""

    rig = await branch_rig(tmp_path, ROOT, "a1")
    files = {
        **CHECKOUT_FILES,
        "arbor/__main__.py": FAKE_DOSSIER,
        "fake-dossier.json": json.dumps({"do": do, "in_session": in_session}),
        "docs/dossier/shipped/README.md": "# one of Arbor's own\n",
    }
    source = _committed(_write(tmp_path / "arbor-repo", files if arbor else {"README.md": "not Arbor\n"}))
    toolbox = Toolbox(rig.blackboard)
    toolbox.cards_left = cards
    sandbox = LocalSandbox(tmp_path / "logs")
    worktree = rig.base.worktree(DOSSIER_HOLDER, tmp_path / DOSSIER_HOLDER / "worktree")
    setup = DossierSetup(
        _spec(tmp_path, tp=1),
        rig.repo,
        toolbox,  # type: ignore[arg-type]
        lambda _lease: sandbox,
        worktree,
        tmp_path / DOSSIER_HOLDER / "session" / "agent",
        sleep=_tick,
        poll_seconds=0.05,
        start_seconds=0.2,
        exit_grace_seconds=0.3,
    )
    start = DossierStart(str(source), rig.root, time.time() + seconds, SHARES, "kernel shares")
    outcome = await run_dossier_pass(start, setup)
    return outcome, toolbox, setup.agent_dir / "src", setup


@pytest.fixture(autouse=True)
def _no_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arbor_adapter, "INSTALL", ())


async def test_the_pass_maps_on_its_own_lease_from_the_start_commit_and_gives_the_cards_back(tmp_path: Path) -> None:
    outcome, toolbox, checkout, setup = await _pass(tmp_path, "mapped")

    session = checkout / DOSSIER_PASS.session_dir
    assert (outcome.reason, outcome.map) == (MAPPED, session / "directions.json")
    assert outcome.dossier == checkout / "docs" / "dossier" / "qwen3-8b-1k1k-c64"  # not the one Arbor ships
    assert (toolbox.leases, toolbox.charges, toolbox.released) == ([(DOSSIER_HOLDER, 1)], [DOSSIER_CHARGE], ["L1"])
    config = json.loads((checkout / "seen-config.json").read_text())
    assert config["component"]["stop_at"] == "map" and "plan" not in config["component"]
    assert (config["workspace_root"], config["component"]["root"]) == (
        str(setup.worktree.path),
        setup.worktree.commit(),
    )
    assert config["server_flags"] == "--max-num-seqs 128"  # the start commit's launch
    profile = json.loads((session / PROFILE_FILE).read_text())
    assert (profile["source"], profile["shares"]) == ("kernel shares", dict(SHARES.shares))


async def test_a_pass_whose_agent_writes_the_dossier_in_its_session_is_found_there(tmp_path: Path) -> None:
    outcome, _, checkout, _ = await _pass(tmp_path, "mapped", in_session=True)

    session = checkout / DOSSIER_PASS.session_dir
    assert (outcome.reason, outcome.dossier) == (MAPPED, session / "docs" / "dossier" / "qwen3-8b-1k1k-c64")


@pytest.mark.parametrize(
    ("do", "said", "wrote"),
    [
        ("nomap", "it ended without a map: deadline", True),
        ("hang", "it set no map by its deadline and was stopped", True),
        ("crash", "it exited 1: Traceback: the profiler died", False),
        (
            "refuse",
            "it did not start: attempt 1: `python3 -m arbor run --config hyperloom-mb/dossier.json` refused",
            False,
        ),
    ],
)
async def test_a_pass_that_sets_no_map_says_why_keeps_its_dossier_and_gives_the_cards_back(
    tmp_path: Path, do: str, said: str, wrote: bool
) -> None:
    outcome, toolbox, checkout, _ = await _pass(tmp_path, do, seconds=0.6)

    assert outcome.map is None and outcome.reason.startswith(said)
    assert (outcome.dossier is not None) == wrote
    assert toolbox.released == ["L1"]


async def test_a_source_that_is_not_arbor_or_no_cards_to_lease_runs_no_pass(tmp_path: Path) -> None:
    other, toolbox, _, _ = await _pass(tmp_path / "other", "mapped", arbor=False)
    assert other.map is None and "this source is not Arbor" in other.reason
    assert toolbox.released == ["L1"]

    busy, toolbox, _, _ = await _pass(tmp_path / "busy", "mapped", cards=0)
    assert busy.map is None and busy.reason.startswith("no cards were leased for it")
    assert toolbox.released == []
