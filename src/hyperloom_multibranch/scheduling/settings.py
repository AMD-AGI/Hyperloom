# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What one multi-branch run is given: the InferenceX or AgentX workload, the wall-clock and GPU-hour budgets and the
hard limits."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from hyperloom.common.perf_metric import GRADED_INTVTY_P50, GRADED_OUTPUT
from hyperloom.common.workload_defaults import DEFAULT_EP
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.scheduling.reference import ReferenceSource
from hyperloom_multibranch.scheduling.roster import RosterCaps

#: FINAL's sweep concs as multiples of an InferenceX workload's conc; the primary operating point is one of them.
SWEEP_FACTORS = (0.5, 1.0, 2.0)
#: How long the dossier pass that maps two or more Arbors' directions may run, unless the run says otherwise. Arbor
#: starts its first iteration only with an hour and its 3 min margin left (its estimate with none measured), and ends
#: the pass as soon as the map is set: on Ruby, two of its iterations wrote a Qwen3-8B dossier and map in 15-25 min.
DOSSIER_MINUTES = 90.0
#: The first port of the blocks the Arbor agents' servers take, unless the run names another. Sessions that share a
#: machine each name their own, since every agent serves on the host's network.
AGENT_PORT = 31100


def sweep_concs(conc: int) -> tuple[int, ...]:
    """The concs FINAL sweeps around ``conc``, ascending."""

    return tuple(sorted({max(round(conc * factor), 1) for factor in SWEEP_FACTORS}))


class KernelOptimizer(str, Enum):
    """The kernel optimizer a session's recipes are scoped to in the Recipe KB."""

    FORGE = "forge"
    GEAK = "geak"


class PlaybookMode(str, Enum):
    """Which scheduling playbook a session plays once the RSI archive is configured: the champion of the pool beside
    it, or, in a challenger session, the oldest playbook on trial, the champion while none is."""

    CHAMPION = "champion"
    CHALLENGER = "challenger"


@dataclass(frozen=True)
class InferenceXShape:
    """The synthetic requests an InferenceX workload sends: input and output sequence lengths."""

    isl: int
    osl: int


@dataclass(frozen=True)
class Workload:
    """The fixed workload, spelled as the single-lineage command line spells it, and the kernel optimizer its recipes
    are scoped to. ``inferencex`` is the request shape of an InferenceX workload and ``None`` for an AgentX trace
    replay, whose shape the trace fixes; ``conc`` is the primary operating point of either. ``ep`` is the
    expert-parallel size an AgentX workload is served at; an InferenceX one is served at :data:`DEFAULT_EP`.

    ``gpu_type`` is named, not detected: the toolbox writes it into every measurement spec, built before PRELUDE runs.
    """

    model: str
    framework: str
    tp: int
    conc: int
    precision: str
    gpu_type: str
    kernel_optimizer: KernelOptimizer
    inferencex: InferenceXShape | None
    ep: int = DEFAULT_EP

    @property
    def model_basename(self) -> str:
        return Path(self.model.rstrip("/")).name

    @property
    def kind(self) -> WorkloadKind:
        return WorkloadKind.AGENTX if self.inferencex is None else WorkloadKind.INFERENCEX

    @property
    def metric(self) -> str:
        """The metric the session's values are read in: the frontier policy, final selection, the dashboard and the
        final recipe's metric. AgentX is graded on P50 interactivity, the key its rule gate's baseline is written
        under."""

        return GRADED_INTVTY_P50 if self.inferencex is None else GRADED_OUTPUT

    @property
    def profiles_moved_head_on_merge(self) -> bool:
        """Whether the integration head a large merge moved is profiled at once, or only once a branch opens on it.
        An AgentX profile replays the trace window, as long as a test, on one of the integration branch's slots, where
        it would hold up the merge tests offered on the moved head."""

        return self.inferencex is not None

    @property
    def final_concs(self) -> tuple[int, ...]:
        """The concs FINAL measures a candidate at: the sweep around ``conc`` for InferenceX, and ``conc`` alone for
        AgentX, which defines no other operating point."""

        return (self.conc,) if self.inferencex is None else sweep_concs(self.conc)


@dataclass(frozen=True)
class ReusedPrelude:
    """The completed PRELUDE of an earlier session that a run roots on instead of starting a child of its own: that
    session's directory, the child session its root was read from, and the digest of the root recipe it committed."""

    session: str
    child_session: str
    root_recipe_digest: str


@dataclass(frozen=True)
class RunSettings:
    """``max_hours`` is the whole session's wall-clock budget, PRELUDE included, ``max_gpu_hours`` the GPU hours
    the session may use, tests' GPU time plus card time any lease holds, and ``max_llm_usd`` the dollars the session's
    recorded LLM calls, the specialists' and the judge's and patch resolver's, may cost; 0 means no limit for any of
    them. ``reuse_prelude`` is the earlier
    PRELUDE the run roots on, ``None`` when it runs a child of its own. ``user_directions`` are the directions the user
    wrote, each opening one branch where EXPLORE starts, ahead of the opening batch, within the branch cap; ``agents``
    the third-party agents the user named, each opening one autonomous branch the same way, and ``agent_directions``
    the directions map whose disjoint shares the Arbor agents among them work, one share each (``""`` for none; see
    :mod:`~hyperloom_multibranch.scheduling.agent_plans`), ``agent_pack`` the pack those Arbors run (``""`` for
    the adapter's default, Arbor's serving template), ``agent_launch`` the file of the launch their recipe's adds
    to (``""`` when the recipe's is the whole launch), ``agent_dossier_minutes`` how long the dossier pass that
    maps two or more Arbors' directions may run when no map is given (0: none, the map is generated from the root's
    shares; :mod:`~hyperloom_multibranch.specialist.dossier_pass`), and ``agent_port`` the first port of the blocks
    their servers take, one block per agent branch and one for the dossier pass. ``stop_after_map`` ends the session
    once those Arbors' directions are mapped, one Arbor's too, before EXPLORE, and leaves the map and its dossier for a
    person to review. ``references`` are the
    directories of reference material the run copies in before its first phase, each with the commit it was taken at.
    ``dev_leases`` is how many dev sandboxes the session lends its specialists at once; 0 lends none. ``playbook`` says
    which scheduling playbook the session plays."""

    workload: Workload
    max_hours: float
    max_gpu_hours: float
    max_llm_usd: float
    caps: RosterCaps
    judge_interval_seconds: float
    reuse_prelude: ReusedPrelude | None = None
    user_directions: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()
    references: tuple[ReferenceSource, ...] = ()
    dev_leases: int = 0
    playbook: PlaybookMode = PlaybookMode.CHAMPION
    agent_directions: str = ""
    agent_pack: str = ""
    agent_launch: str = ""
    agent_dossier_minutes: float = DOSSIER_MINUTES
    agent_port: int = AGENT_PORT
    stop_after_map: bool = False


__all__ = [
    "AGENT_PORT",
    "DOSSIER_MINUTES",
    "SWEEP_FACTORS",
    "InferenceXShape",
    "KernelOptimizer",
    "PlaybookMode",
    "ReusedPrelude",
    "RunSettings",
    "Workload",
    "sweep_concs",
]
