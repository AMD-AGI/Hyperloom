# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the scheduler, the integrator and the final selection call on the other multi-branch modules.

Provisional, as :mod:`hyperloom_multibranch.contracts.notebook.peers` is: each protocol and object here gives way to
its owner's once that is merged, and lists only what the scheduling side calls or reads. The specialist lifecycle,
the rule gate and the experiment repository belong to the specialist-evolution module. The toolbox is the toolbox
module's own :mod:`hyperloom_multibranch.contracts.toolbox`. Every method is a coroutine.

The integrator tests a merge before it writes it: :meth:`ExperimentRepo.merge` proposes the change, the merge test
runs from the integration head with that change, and only an improvement is written with
:meth:`ExperimentRepo.commit_integration`. Nothing on the integration branch is ever reverted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hyperloom.contracts.recipe import PatchRef, Recipe, RecipeDelta, ValueChange
from hyperloom_multibranch.contracts.notebook.peers import Judgement
from hyperloom_multibranch.contracts.notebook.records import BaselineRecord

# MB_FAKE_PROVISIONAL_PORTS: every protocol and object in this module is provisional and gives way to its owner's
# type (hyperloom_multibranch.specialist, .rulegate, .repo) once that is merged.


@dataclass(frozen=True)
class ForkRequest:
    """The rule gate asking for a new branch to carry a correctness fix that regresses performance on its own branch."""

    from_branch: str
    parent_commit: str
    reason: str


@dataclass(frozen=True)
class PatchConflict:
    """A patch of ``component`` that does not apply where the merge puts it; ``detail`` is what applying it said."""

    component: str
    path: str
    detail: str


@dataclass(frozen=True)
class RepoMerge:
    """A candidate commit's change carried onto ``onto``.

    ``delta`` is the change from ``onto``'s recipe to the merged one, with the candidate's value taken wherever the
    two set a server argument or environment variable differently (listed in ``config_conflicts``). When the candidate
    moves a component to another upstream ref, ``onto``'s patches of that component are replayed on the new ref.
    ``patch_conflicts`` are the patches that did not apply; ``delta`` leaves them out.
    """

    onto: str
    delta: RecipeDelta
    config_conflicts: tuple[ValueChange, ...] = ()
    patch_conflicts: tuple[PatchConflict, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.patch_conflicts


class SpecialistLifecycle(Protocol):
    """Long-running specialists, one per branch; the runtime restarts a crashed one on the same branch by itself.

    Every process it starts goes through the session's process registry it was built with, so the session stop kills
    it even when :meth:`stop` hangs.
    """

    async def start(self, branch: str, start_commit: str) -> None: ...

    async def stop(self, branch: str) -> None:
        """Stop the branch's specialist; returns once no process of it is left."""

    async def set_k(self, branch: str, k: int) -> None:
        """How many tests the branch's specialist may have running at once."""


class RuleGate(Protocol):
    """The rule gate as the scheduling side uses it; fork requests reach the scheduler's admission by awaiting it."""

    async def baseline_test_count(self) -> int:
        """How many PRELUDE baseline tests the threshold needs."""

    async def establish_baseline(self, root_commit: str, request_ids: tuple[str, ...]) -> BaselineRecord:
        """Compute the baseline and thresholds from the measured tests and record them in the notebook."""

    async def compare(self, request_id: str) -> Judgement:
        """Judge a settled test against its parent commit's recorded value as a specialist candidate is judged.

        Runs the confirming retest and the accuracy rounds a commit needs, and returns the final judgement.
        """

    async def accuracy_passes(self, request_id: str) -> bool:
        """Whether an ``accuracy`` test clears the accuracy gate every commit clears."""


class ExperimentRepo(Protocol):
    async def init_root(self, recipe: Recipe) -> str:
        """Store the baseline recipe as the root commit, start the integration branch at it, and return the commit."""

    async def recipe_at(self, commit: str) -> Recipe: ...

    async def head(self, branch: str) -> str: ...

    async def merge(self, candidate: str, onto: str) -> RepoMerge:
        """Carry ``candidate``'s change, relative to its parent, onto ``onto``; writes nothing."""

    async def patch_text(self, commit: str, path: str) -> str:
        """The content of the patch file ``path`` in ``commit``'s recipe tree."""

    async def store_patch(self, component: str, text: str) -> PatchRef:
        """Store a patch file of ``component`` that a later delta adds."""

    async def commit_integration(self, onto: str, delta: RecipeDelta, merged_from: tuple[str, ...]) -> str:
        """Write ``delta`` as a merge of ``merged_from`` on the integration branch, whose head must be ``onto``."""

    async def checkout(self, commit: str, destination: Path) -> None:
        """Write the patch and kernel files ``commit``'s recipe names under ``destination``, at the paths it names them by."""


__all__ = [
    "ExperimentRepo",
    "ForkRequest",
    "PatchConflict",
    "RepoMerge",
    "RuleGate",
    "SpecialistLifecycle",
]
