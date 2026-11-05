# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""MB's side of an autonomous branch's agent, one driver per optimizer: how MB knows the agent, starts it, reads its
keeps and its exit, and what it takes part in beyond its branch.

A driver (:class:`AgentDriver`) knows its optimizer's checkout when it sees one (``detect``), writes what the optimizer
reads into the branch's worktree before it starts (``prepare``), gives the manifest MB runs it by (``manifest``) and
fits that manifest once MB has set the keys of the component mode in it (``fit``), reads its keeps (``keeps``, a
:class:`KeepReader`) and says how its process ended (``ended``). The session's planners ask a driver's
:class:`Capabilities`, not which optimizer it drives: whether its agent takes a share of the session's directions,
runs the dossier pass that maps them, and takes the session's review of the map for its own.

:func:`driver_for` asks each of :data:`DRIVERS` in turn whether a checkout is its optimizer's. Arbor
(:class:`ArborDriver`) is :mod:`arbor_adapter` behind the protocol, and has every capability; any other agent is
onboarded by the model (:class:`OnboardedDriver`), the fallback, and has none.

An agent's keeps are read as its manifest says they show, whichever driver gave it: an agent run as a component
(Arbor's mode, which the onboarding model may give another agent too) writes a snapshot of each
(:class:`SnapshotKeeps`); any other has a keep file, or none, and its keep is its workspace
(:class:`WorkspaceKeeps`). How its process ended is read the same way (:func:`manifest_ended`): only an agent run as a
component refuses its config, with exit 2, and says why it finished.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Protocol

from hyperloom.contracts.recipe import LaunchConfig
from hyperloom_multibranch.specialist.agent_import import (
    ComponentStart,
    KeptChange,
    KeptChangeError,
    latest_snapshot,
    read_json,
    read_kept_change,
    read_snapshot,
    with_tables,
)
from hyperloom_multibranch.specialist.agent_manifest import WORKSPACE, AgentManifest, read_field
from hyperloom_multibranch.specialist.agent_spec import AgentSpec
from hyperloom_multibranch.specialist.arbor_adapter import (
    EXIT_REFUSED,
    ArborCheckout,
    ArborRefused,
    ArborRun,
    arbor_manifest,
    dossier_notes,
    fit_keys,
    is_arbor,
    refused_reasons,
)
from hyperloom_multibranch.specialist.start_tables import StartTable, from_start_tables
from hyperloom_multibranch.specialist.worktree import BranchWorktree

log = logging.getLogger(__name__)

#: The onboarding model's manifest for the checkout at hand, asked only when awaited: by a driver that has no manifest
#: of its own.
Asked = Callable[[], Awaitable[AgentManifest]]


@dataclass(frozen=True)
class Capabilities:
    """What an agent takes part in beyond its own branch, which the session's planners ask of its driver:
    ``plan_share``, a share of the session's directions, which it reads from its spec's ``plan``; ``dossier_pass``, the
    dossier pass that maps those directions on one lease before the share branches open, its checkout's command line
    generating and splitting a map; ``review_host``, the session's review of the map standing in for its own review
    gate (its spec's ``review``)."""

    plan_share: bool = False
    dossier_pass: bool = False
    review_host: bool = False


class EndKind(Enum):
    """How an agent's process ended: it finished its run, refused its config (it is never started again with it), or
    crashed (the runtime starts it again)."""

    FINISHED = "finished"
    REFUSED = "refused"
    CRASHED = "crashed"


@dataclass(frozen=True)
class Ended:
    """How an agent's process ended (``kind``), and ``why``, as the agent said it: the reasons it refused its config,
    or the reason it finished; ``""`` when it said nothing."""

    kind: EndKind
    why: str = ""


@dataclass(frozen=True)
class AgentCheckout:
    """An agent's ``source``, fetched at ``commit`` into ``path``."""

    source: str
    commit: str
    path: Path


@dataclass(frozen=True)
class KeepPlace:
    """Where a run reads its agent's keeps, and against what: ``agent_dir``, the agent's directory, and ``checkout``,
    its checkout in it, which its keep file and outbox are relative to, and a config source of its own; ``worktree``,
    the branch's workspace the agent edits, of which ``component`` is the framework; ``launch``, the start commit's
    launch the agent was started with; ``start_tables``, the start commit's kernel tables as they were laid out for it;
    and ``applied``, what patches make of a root's component, called with the root, a list it adds the files of the
    last patch outside the component to, and the patches (the session's, which refuses a change too large to keep)."""

    agent_dir: Path
    checkout: Path
    worktree: BranchWorktree
    component: str
    launch: LaunchConfig
    start_tables: Mapping[str, StartTable]
    applied: Callable[[str, list[str], Sequence[str]], str]


class ConfigRefusal(ValueError):
    """The config a driver wrote from the spec alone would be refused as it starts: another attempt would write the
    same, so the preflight ends with it."""


class KeepReader(Protocol):
    """How a run reads its agent's keeps: what its keep signal says of them now (:meth:`mark`), and one keep whole
    (:meth:`read`). ``unread`` says why the signal could not be read at the last :meth:`mark`, ``""`` when it could."""

    unread: str

    def mark(self) -> int | str | None:
        """What the agent's keep signal says of its keeps now: a count, a digest of its keep file, ``None`` when it has
        none or it cannot be read, which :attr:`unread` then says when it is not only being written."""

    def read(self, seq: int, root: str) -> KeptChange:
        """The agent's keep ``seq``, read against the workspace's commit ``root``; raise
        :class:`~hyperloom_multibranch.specialist.agent_import.KeptChangeError` or
        :class:`~hyperloom_multibranch.specialist.worktree.WorktreeError` when it cannot be."""


class AgentDriver(Protocol):
    """MB's side of one optimizer run as an autonomous branch's agent: ``name``, the optimizer's; ``capabilities``,
    what its agent takes part in beyond its branch; ``config_from_spec``, whether the driver writes the agent's config
    from the spec alone, so a config refused as the agent starts is refused at every attempt."""

    name: str
    capabilities: Capabilities
    config_from_spec: bool

    def detect(self, checkout: Path) -> bool:
        """Whether ``checkout``, an agent's fetched source, is this driver's optimizer."""

    def prepare(self, spec: AgentSpec, workspace: Path) -> None:
        """Write what the optimizer reads into ``workspace``, the branch's worktree, before its manifest is made."""

    async def manifest(self, spec: AgentSpec, checkout: AgentCheckout, run: ArborRun, asked: Asked) -> AgentManifest:
        """The manifest MB runs the agent in ``checkout`` by for ``spec``, for ``run`` when the agent is Arbor (a
        branch's or the dossier pass); ``asked`` is the onboarding model's. Raise :class:`ConfigRefusal` when the
        config written would be refused."""

    def fit(self, manifest: AgentManifest, spec: AgentSpec, checkout: Path) -> tuple[AgentManifest, tuple[str, ...]]:
        """``manifest`` as the agent in ``checkout`` runs it, once MB has set its component keys, and what the branch
        should note of how it was fitted."""

    def keeps(self, manifest: AgentManifest, place: KeepPlace) -> KeepReader:
        """The reader of the keeps of the agent ``manifest`` runs, at ``place``."""

    def ended(self, returncode: int | None, manifest: AgentManifest, checkout: Path) -> Ended:
        """How the agent ``manifest`` runs in ``checkout`` ended, its process having exited with ``returncode``."""


class SnapshotKeeps:
    """The keeps of an agent run as a component (Arbor's mode): its newest snapshot is its keep signal, and each keep
    is read whole from its snapshot, chained back to the root (:func:`~.agent_import.read_snapshot`), its kernel tables
    as given read as the start's."""

    def __init__(self, manifest: AgentManifest, place: KeepPlace) -> None:
        self._manifest = manifest
        self._place = place
        self.unread = ""

    def mark(self) -> int | None:
        """The keep the agent's newest snapshot holds; ``None`` before the first, or when ``LATEST`` cannot be read,
        which :attr:`unread` then says."""

        outbox = self._manifest.snapshots_outbox
        try:
            latest = latest_snapshot(self._place.checkout / outbox)
        except OSError as error:
            self.unread = f"the snapshot outbox {outbox} cannot be read ({error.strerror})"
            return None
        except ValueError as error:
            self.unread = f"the LATEST of {outbox} names no keep ({error})"
            return None
        self.unread = ""
        return latest

    def read(self, seq: int, root: str) -> KeptChange:
        """The agent's keep ``seq``, from its snapshot, with the files of its own change left out as outside the
        component named in its ``outside``."""

        place = self._place
        outside: list[str] = []
        copies = [path for table in place.start_tables.values() for path in table.defaults]
        kept = read_snapshot(
            place.checkout / self._manifest.snapshots_outbox,
            seq,
            root,
            partial(place.applied, root, outside),
            ComponentStart(place.launch, self._manifest.config, copies),
        )
        return replace(from_start_tables(kept, place.start_tables), outside=tuple(outside))


class WorkspaceKeeps:
    """The keeps of an agent that writes no snapshot: its keep file's counter, or its digest, is its keep signal (it may
    have none), and a keep is the workspace's state, with the launch its config source holds, or the start's."""

    def __init__(self, manifest: AgentManifest, place: KeepPlace) -> None:
        self._manifest = manifest
        self._place = place
        self.unread = ""

    def mark(self) -> int | str | None:
        """What the agent's keep file says of its keeps now; ``None`` when it has none, cannot be read mid-write, or
        holds no counter the manifest names, which :attr:`unread` then says."""

        keep = self._manifest.keep
        if keep is None:
            return None
        path = self._place.checkout / keep.path
        try:
            mark = keep.mark(path.read_bytes())
        except json.JSONDecodeError as error:
            log.debug("the keep file %s is being written: %s", path, error)
            return None
        except OSError as error:
            self.unread = f"the keep file {keep.path} cannot be read ({error.strerror})"
            return None
        except (ValueError, KeyError, IndexError) as error:
            self.unread = f"the keep file {keep.path} holds no keep count at {keep.counter!r} ({error!r})"
            return None
        self.unread = ""
        return mark

    def read(self, seq: int, root: str) -> KeptChange:
        """The agent's keep ``seq``, its state now; an agent that keeps no config of its own runs with the root's."""

        manifest, place = self._manifest, self._place
        workspace = place.worktree.path
        patch = without_file(place.worktree.diff_from(root, place.component), manifest)
        source = manifest.config_source
        started = KeptChange(seq, patch, place.launch.server_args, dict(place.launch.envs), self._summary())
        if source is None:
            return started
        where = workspace if source.root == WORKSPACE else place.checkout
        kept = read_kept_change(started, read_json(where / source.path), source)
        return with_tables(from_start_tables(kept, place.start_tables), (workspace, place.agent_dir))

    def _summary(self) -> str:
        manifest = self._manifest
        if manifest.keep is None or not manifest.summary_field:
            return ""
        try:
            state = read_json(self._place.checkout / manifest.keep.path)
            return str(read_field(state, manifest.summary_field))
        except (KeptChangeError, KeyError, IndexError, ValueError):
            return ""


class ArborDriver(AgentDriver):
    """Arbor, run as a component of MB (:mod:`arbor_adapter`): its checkout says what it is, and its manifest and
    config follow from the spec, never from the onboarding model. It takes a share of the directions as its
    ``component.plan``, runs the dossier pass and splits its map, and takes the session's review for its own."""

    name = "arbor"
    capabilities = Capabilities(plan_share=True, dossier_pass=True, review_host=True)
    config_from_spec = True

    def detect(self, checkout: Path) -> bool:
        return is_arbor(checkout)

    def prepare(self, spec: AgentSpec, workspace: Path) -> None:
        """Nothing: :func:`~.arbor_adapter.arbor_manifest` links the framework's built files into the workspace and
        carries the dossier pass's dossier into Arbor's checkout itself."""

    async def manifest(self, spec: AgentSpec, checkout: AgentCheckout, run: ArborRun, asked: Asked) -> AgentManifest:
        try:
            arbor = ArborCheckout(checkout.source, checkout.commit, checkout.path)
            return await asyncio.to_thread(arbor_manifest, spec, arbor, run)
        except ArborRefused as error:
            raise ConfigRefusal(f"Arbor would refuse the config written for it: {error}") from error

    def fit(self, manifest: AgentManifest, spec: AgentSpec, checkout: Path) -> tuple[AgentManifest, tuple[str, ...]]:
        """``manifest`` without an input the pinned Arbor lacks (:func:`~.arbor_adapter.fit_keys`), noted, and a note
        when the dossier pass's dossier is gone (:func:`~.arbor_adapter.dossier_notes`)."""

        config, notes = fit_keys(manifest.config, checkout)
        return replace(manifest, config=config), (*notes, *dossier_notes(checkout, spec.dossier))

    def keeps(self, manifest: AgentManifest, place: KeepPlace) -> KeepReader:
        """Arbor runs as a component: a snapshot of each keep."""

        return SnapshotKeeps(manifest, place)

    def ended(self, returncode: int | None, manifest: AgentManifest, checkout: Path) -> Ended:
        return manifest_ended(returncode, manifest, checkout)


class OnboardedDriver(AgentDriver):
    """Any other agent, the fallback: the onboarding model reads its checkout and writes its manifest, which another
    attempt may write otherwise. It takes no part in the session's planning."""

    name = "onboarded"
    capabilities = Capabilities()
    config_from_spec = False

    def detect(self, checkout: Path) -> bool:
        """Every checkout: the onboarding model reads any agent."""

        return True

    def prepare(self, spec: AgentSpec, workspace: Path) -> None:
        """Nothing: the onboarding model reads the agent as it comes."""

    async def manifest(self, spec: AgentSpec, checkout: AgentCheckout, run: ArborRun, asked: Asked) -> AgentManifest:
        """The onboarding model's."""

        return await asked()

    def fit(self, manifest: AgentManifest, spec: AgentSpec, checkout: Path) -> tuple[AgentManifest, tuple[str, ...]]:
        """``manifest`` as the model wrote it: an attempt it got wrong fails, and the next one is asked to correct it."""

        return manifest, ()

    def keeps(self, manifest: AgentManifest, place: KeepPlace) -> KeepReader:
        """As the model's manifest says they show: snapshots, for an agent it runs as a component, else the keep file
        and the workspace."""

        if manifest.snapshots is not None:
            return SnapshotKeeps(manifest, place)
        return WorkspaceKeeps(manifest, place)

    def ended(self, returncode: int | None, manifest: AgentManifest, checkout: Path) -> Ended:
        return manifest_ended(returncode, manifest, checkout)


ARBOR = ArborDriver()
ONBOARDED = OnboardedDriver()
#: The drivers :func:`driver_for` asks, in order; the last takes any checkout.
DRIVERS: tuple[AgentDriver, ...] = (ARBOR, ONBOARDED)


def driver_for(checkout: Path, drivers: Sequence[AgentDriver] = DRIVERS) -> AgentDriver:
    """The driver of the agent fetched into ``checkout``: the first of ``drivers`` that detects it there, else
    :data:`ONBOARDED`."""

    return next((driver for driver in drivers if driver.detect(checkout)), ONBOARDED)


def manifest_ended(returncode: int | None, manifest: AgentManifest, checkout: Path) -> Ended:
    """How the agent ``manifest`` runs in ``checkout`` ended with ``returncode``, as its manifest says it ends: one run
    as a component refuses its config with exit 2 (:data:`~.arbor_adapter.EXIT_REFUSED`), saying why in its refusal
    file, and says why it finished in its exit file (:func:`exit_reason`); any exit but 0 is a crash."""

    component = manifest.snapshots is not None
    if returncode == EXIT_REFUSED and component:
        return Ended(EndKind.REFUSED, refused_reasons(checkout / manifest.snapshots_refused))
    if returncode != 0:
        return Ended(EndKind.CRASHED)
    return Ended(EndKind.FINISHED, exit_reason(manifest, checkout) if component else "")


def exit_reason(manifest: AgentManifest, checkout: Path) -> str:
    """Why the agent ``manifest`` runs as a component in ``checkout`` says it ended, as the exit file it writes before
    an exit 0 has it."""

    path = checkout / manifest.snapshots_exit
    if not path.exists():
        return "it wrote no exit reason"
    try:
        document = read_json(path)
    except KeptChangeError as error:
        return f"its exit reason cannot be read: {error}"
    reason = document.get("reason") if isinstance(document, dict) else None
    return reason if isinstance(reason, str) and reason else f"its exit file names no reason: {document!r}"


def without_file(patch: str, manifest: AgentManifest) -> str:
    """``patch`` without the sections of the file the agent keeps its config in, which the config delta carries."""

    if manifest.config_source is None or manifest.config_source.root != WORKSPACE:
        return patch
    path = manifest.config_source.path
    sections = patch.split("diff --git ")
    kept = [section for section in sections[1:] if section.split("\n", 1)[0] != f"a/{path} b/{path}"]
    return "".join([sections[0], *(f"diff --git {section}" for section in kept)])


__all__ = [
    "ARBOR",
    "DRIVERS",
    "ONBOARDED",
    "AgentCheckout",
    "AgentDriver",
    "ArborDriver",
    "Asked",
    "Capabilities",
    "ConfigRefusal",
    "EndKind",
    "Ended",
    "KeepPlace",
    "KeepReader",
    "OnboardedDriver",
    "SnapshotKeeps",
    "WorkspaceKeeps",
    "driver_for",
    "exit_reason",
    "manifest_ended",
    "without_file",
]
