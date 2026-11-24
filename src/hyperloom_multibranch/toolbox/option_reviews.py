# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The options of the session's server, laid in the toolbox directory, and each branch's review of them.

:func:`lay_option_inventory` asks the base build the tests run for its options and writes them, with which of them the
base launch sets, as ``server_options.json`` (:mod:`hyperloom.measure.runner.native.server_options`) in the toolbox
directory. A session whose server is not a program built from its checkout has no such file.

:class:`OptionReviews` keeps, by branch, the verdict a specialist records on each option the launch leaves unset
(``option-reviews/<branch>.json`` in the same directory) and says what a branch's review still leaves out
(:class:`ReviewStanding`). A review is read against the inventory as it is now: a verdict on an option the server no
longer lists, or the launch now sets, counts for nothing. A note is its option's own: a review in which two options
share one is refused, since one answer pasted on every option gives each a verdict and looks at none.

Until a branch's review leaves none out, :meth:`OptionReviews.check` refuses its explore tests that change the server's
arguments, so no argument is tried before every one the launch could take was looked at. An explore test that sets
the environment variable the server reads an option from is refused whatever the review: an option is set as an
argument, where the review and every check of a launch's arguments read it. The check is the specialists': it sits
where they submit, since a branch another agent drives has no review to record.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from hyperloom.contracts.measurement import TestKind, TestRequest
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom.inference_optimizer.native_builds import OptionVerdict
from hyperloom.inference_optimizer.session.session_paths import fs_safe_id
from hyperloom.measure.farm import prepend_base_bins
from hyperloom.measure.runner.native.server_options import OptionInventory, server_inventory, write_inventory
from hyperloom.orchestrator.state.base_launch import server_base
from hyperloom_common.io import atomic_write_json
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from inference_suite import runner_owned

if TYPE_CHECKING:
    from inference_suite.contracts import BaseLaunch

__all__ = [
    "NOTE_CHARS",
    "OPTION_BY_ENV_CODE",
    "OPTIONS_UNREVIEWED_CODE",
    "OptionReviews",
    "ReviewEntry",
    "ReviewRefused",
    "ReviewStanding",
    "lay_option_inventory",
]

_REVIEWS_DIR = "option-reviews"
NOTE_CHARS = 240
OPTIONS_UNREVIEWED_CODE = "options_unreviewed"
OPTION_BY_ENV_CODE = "option_set_by_env"
_SCHEMA = 1
#: How many of the options still without a verdict a refusal names; its count says how many there are.
_FIRST_NAMED = 5


def lay_option_inventory(base: BaseLaunch, probe_env: Mapping[str, str], directory: Path) -> OptionInventory | None:
    """Write the options of the base server build ``probe_env`` names, read against the session's frozen ``base``
    (its arguments, and the flags the runner sets on every launch of it), into ``directory`` and return them;
    ``None``, and no file, without a base build or for a framework whose server is no native program.

    The program is asked in this process's environment under the launch's own and ``probe_env``, the runner variable
    naming the base builds' executables (which go first on ``PATH``, as in a runner) and whatever else a process of
    this machine needs to run one.

    Raises:
        FrameworkProbeError: The base build's server cannot print its ``--help``.
        FarmError: A base build ``probe_env`` names is not there.
    """
    if not probe_env:
        return None
    launch = server_base(base)
    env = {**os.environ, **launch.env, **probe_env}
    prepend_base_bins(env)
    owned = runner_owned(base).args
    inventory = server_inventory(base.framework, env, launch.args, cwd=directory, runner_owned=owned)
    if inventory is not None:
        write_inventory(inventory, directory)
    return inventory


@dataclass(frozen=True)
class ReviewEntry:
    """A specialist's verdict on one option, by any spelling the server's help lists for it, with the evidence in
    a ``note`` of one line."""

    option: str
    verdict: OptionVerdict
    note: str


class ReviewRefused(ValueError):
    """Entries of one call cannot be taken; the message names each with its reason, and nothing of the call is kept."""


@dataclass(frozen=True)
class ReviewStanding:
    """How far a branch's review covers the options the launch leaves unset: how many there are, how many have a
    verdict, the ones that have none and the ones the branch called candidates, both in the inventory's order."""

    total_unset: int
    covered: int
    remaining: tuple[str, ...]
    candidates: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.remaining


@dataclass(frozen=True)
class _Verdict:
    verdict: OptionVerdict
    note: str


def _one_line(note: str) -> str:
    return " ".join(note.split())


def _problem(inventory: OptionInventory, entry: ReviewEntry) -> str | None:
    """Why ``entry`` cannot be taken; ``None`` when it can."""
    option = inventory.option(entry.option)
    if option is None:
        return f"{entry.option}: the server lists no such option"
    if option.set_by_base:
        return f"{entry.option}: the launch sets it, so it is not one to review"
    note = _one_line(entry.note)
    if not note:
        return f"{entry.option}: the note is empty"
    if len(note) > NOTE_CHARS:
        return f"{entry.option}: the note has {len(note)} characters, {NOTE_CHARS} at most"
    return None


def _shared_notes(verdicts: Mapping[str, _Verdict]) -> list[str]:
    """Why the review ``verdicts`` cannot stand: each note two or more of its options share, in any case, with
    those options; the first few of them when there are many."""
    holders: dict[str, list[str]] = {}
    for name, kept in verdicts.items():
        holders.setdefault(kept.note.casefold(), []).append(name)
    shared = [names for names in holders.values() if len(names) > 1]
    problems = [
        f'the note "{verdicts[names[0]].note}" is given to {len(names)} options '
        f"({', '.join(names[:_FIRST_NAMED])}{' ...' if len(names) > _FIRST_NAMED else ''}): a note says what you "
        "read about its own option"
        for names in shared[:_FIRST_NAMED]
    ]
    if len(shared) > _FIRST_NAMED:
        problems.append(f"{len(shared) - _FIRST_NAMED} more notes are shared")
    return problems


def _changes_arguments(delta: RecipeDelta) -> bool:
    """Whether ``delta`` changes the server's arguments: anything of its config but the environment."""
    return replace(delta.config, envs={}, unset_envs=()) != ConfigDelta()


def _standing(inventory: OptionInventory, verdicts: Mapping[str, _Verdict]) -> ReviewStanding:
    unset = [option.name for option in inventory.unset]
    remaining = tuple(name for name in unset if name not in verdicts)
    candidates = tuple(name for name in unset if name in verdicts and verdicts[name].verdict is OptionVerdict.CANDIDATE)
    return ReviewStanding(len(unset), len(unset) - len(remaining), remaining, candidates)


class OptionReviews:
    """The branches' reviews, kept in ``directory`` (the toolbox's), of the session's option ``inventory``."""

    def __init__(self, directory: Path, inventory: OptionInventory) -> None:
        self._directory = directory
        self._inventory = inventory
        self._writing = threading.Lock()

    def record(self, branch: str, entries: Sequence[ReviewEntry]) -> ReviewStanding:
        """Add ``entries`` to ``branch``'s review, a later verdict on an option replacing its earlier one, and return
        where the review stands.

        Raises:
            ReviewRefused: An entry names no option of the server or one the launch sets, carries no verdict, or its
                note is empty or longer than a line of ``NOTE_CHARS`` characters; or the review would hold one note
                for two options.
        """
        inventory = self._inventory
        problems = [problem for entry in entries if (problem := _problem(inventory, entry)) is not None]
        if problems:
            raise ReviewRefused("; ".join(problems))
        with self._writing:
            verdicts = self._verdicts(branch)
            for entry in entries:
                option = inventory.option(entry.option)
                assert option is not None  # _problem refused the others
                verdicts[option.name] = _Verdict(entry.verdict, _one_line(entry.note))
            shared = _shared_notes(verdicts)
            if shared:
                raise ReviewRefused("; ".join(shared))
            document = {
                "schema": _SCHEMA,
                "reviews": {
                    name: {"verdict": kept.verdict.value, "note": kept.note} for name, kept in verdicts.items()
                },
            }
            atomic_write_json(self._path(branch), document)
        return _standing(inventory, verdicts)

    def standing(self, branch: str) -> ReviewStanding:
        """Where ``branch``'s review stands."""
        return _standing(self._inventory, self._verdicts(branch))

    def check(self, request: TestRequest) -> None:
        """Refuse ``request``, one a specialist submits, when it is a test that sets the environment variable the
        server reads an option from, or one that changes the server's arguments while its branch's review still
        leaves an unset option without a verdict.

        Raises:
            RequestRejected: As ``option_set_by_env``, naming each variable with its option; as
                ``options_unreviewed``, naming how many options are left and the first of them.
        """
        if request.kind is not TestKind.TEST:
            return
        by_env = {option.env: option.name for option in self._inventory.options if option.env}
        named = [f"{name} ({by_env[name]})" for name in sorted(request.delta.config.envs) if name in by_env]
        if named:
            raise RequestRejected(
                f"{', '.join(named)}: the server reads an option from each of these variables; set the option as a "
                "server argument, where its review and the checks of a launch's arguments read it",
                OPTION_BY_ENV_CODE,
            )
        if not _changes_arguments(request.delta):
            return
        remaining = self.standing(request.branch_id).remaining
        if remaining:
            first = ", ".join(remaining[:_FIRST_NAMED]) + (" ..." if len(remaining) > _FIRST_NAMED else "")
            raise RequestRejected(
                f"{len(remaining)} option(s) the launch leaves unset have no verdict in your branch's option review "
                f"(first: {first}): record them with option_review, then submit again",
                OPTIONS_UNREVIEWED_CODE,
            )

    def _path(self, branch: str) -> Path:
        return self._directory / _REVIEWS_DIR / f"{fs_safe_id(branch)}.json"

    def _verdicts(self, branch: str) -> dict[str, _Verdict]:
        try:
            document = json.loads(self._path(branch).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        return {
            name: _Verdict(OptionVerdict(kept["verdict"]), kept["note"]) for name, kept in document["reviews"].items()
        }
