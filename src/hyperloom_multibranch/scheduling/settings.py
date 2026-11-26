# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What one multi-branch run is given: the InferenceX or AgentX workload, the wall-clock and GPU-hour budgets and the
hard limits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from hyperloom.common.agentx_early_stop import EarlyStopMode, early_stop_mode
from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.common.workload_defaults import DEFAULT_EP
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.rulegate.agentx import AgentXObjective
from hyperloom_multibranch.scheduling.reference import ReferenceSource
from hyperloom_multibranch.scheduling.roster import RosterCaps

#: FINAL's sweep concs as multiples of an InferenceX workload's conc; the primary operating point is one of them.
SWEEP_FACTORS = (0.5, 1.0, 2.0)
#: How long the dossier pass that maps two or more component agents' directions may run, unless the run says otherwise.
#: A component agent starts its first iteration only with an hour and its 3 min margin left (its estimate with none
#: measured), and ends the pass as soon as the map is set.
DOSSIER_MINUTES = 90.0
#: The least time each component agent's share branch must have to its cutoff for the shares to open, unless the run
#: says otherwise: with less, that first iteration never starts. An agent may measure the root's GSM8K before it, a few
#: minutes on a small model and up to 30 on a large one, so a share given just this may still run nothing.
SHARE_MINUTES = 63.0
#: The first port of the blocks the component agents' servers take, unless the run names another. Sessions that share
#: a machine each name their own, since every agent serves on the host's network.
AGENT_PORT = 31100
#: The axis an AgentX session is graded on unless the run names another.
DEFAULT_AGENTX_OBJECTIVE = AgentXObjective.P50_INTERACTIVITY
#: The K an AgentX specialist branch opens at when the judge names none, within ``max_k_per_branch``.
AGENTX_OPEN_K = 2
#: The longest boot budget a run may give its tests' servers: six hours, past any checkpoint load a test would wait for.
#: The round raises the serving framework's own ready timeout above the budget (``inference_suite.compose``), so the
#: whole range buys the boot time it promises.
MAX_BOOT_BUDGET_SEC = 6 * 3600
#: The conc sweep's tests in flight by default: a concurrency's baseline and optimized point side by side.
DEFAULT_CONC_SWEEP_PARALLEL = 2


def sweep_concs(conc: int) -> tuple[int, ...]:
    """The concs FINAL sweeps around ``conc``, ascending."""

    return tuple(sorted({max(round(conc * factor), 1) for factor in SWEEP_FACTORS}))


class KernelOptimizer(str, Enum):
    """The kernel optimizer a session's recipes are scoped to in the Recipe KB."""

    FORGE = "forge"
    GEAK = "geak"


class PlaybookMode(str, Enum):
    """Which scheduling playbook a session plays once the RSI archive is configured: the champion of the pool beside
    it, or, in a challenger session, the oldest playbook on trial, the champion while none is; ``auto`` makes it a
    challenger session when its id draws it (:func:`~hyperloom_multibranch.blackboard.rsi.playbook.drawn_challenger`)."""

    AUTO = "auto"
    CHAMPION = "champion"
    CHALLENGER = "challenger"


@dataclass(frozen=True)
class InferenceXShape:
    """The requests an InferenceX workload sends: random prompts of ``isl`` tokens answered to ``osl``, or, when
    ``dataset`` names one, the prompts of that ``DATASET_PATH`` file, as given when the session started, answered to at
    most ``osl``."""

    isl: int
    osl: int
    dataset: str = ""


@dataclass(frozen=True)
class Workload:
    """The fixed workload, spelled as the single-lineage command line spells it, and the kernel optimizer its recipes
    are scoped to. ``inferencex`` is the request shape of an InferenceX workload and ``None`` for an AgentX trace
    replay, whose shape the trace fixes; ``conc`` is the primary operating point of either. ``ep`` is the
    expert-parallel size either is served at.
    ``agentx_duration_sec`` is an AgentX workload's trace window, ``None`` for the measuring harness's default.
    ``agentx_objective`` is the axis an AgentX workload is graded on; an InferenceX one is graded on output throughput.
    ``boot_budget_sec`` is at least how long the server of a test may take to be ready, which may also lengthen how
    long an attempt of the test may run (:func:`~hyperloom_multibranch.scheduling.local_peers.attempt_timeout_sec`); it
    never shortens either; ``None`` for the defaults of both.

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
    agentx_duration_sec: int | None = None
    agentx_objective: AgentXObjective = DEFAULT_AGENTX_OBJECTIVE
    boot_budget_sec: int | None = None

    @property
    def model_basename(self) -> str:
        return Path(self.model.rstrip("/")).name

    @property
    def kind(self) -> WorkloadKind:
        return WorkloadKind.AGENTX if self.inferencex is None else WorkloadKind.INFERENCEX

    @property
    def metric(self) -> str:
        """The metric the session's values are read in: the frontier policy, final selection, the dashboard and the
        final recipe's metric. AgentX is graded on its objective's metric, the key its rule gate's baseline is written
        under."""

        return self.agentx_objective.metric if self.inferencex is None else GRADED_OUTPUT

    @property
    def profiles_moved_head_on_merge(self) -> bool:
        """Whether the integration head a large merge moved is profiled at once, or only once a branch opens on it.
        An AgentX profile replays the trace window, as long as a test, on one of the integration branch's slots, where
        it would hold up the merge tests offered on the moved head."""

        return self.inferencex is not None

    @property
    def prelude_awaits_root_analysis(self) -> bool:
        """Whether PRELUDE waits for the analysis of the root's profile, so the opening judge call reads the root's op
        shares, or only for its PROFILE test, the analysis going on once EXPLORE opened. An InferenceX analysis is
        short; an AgentX one, on the CPU alone, can run for tens of minutes while every card would wait."""

        return self.inferencex is not None

    @property
    def integration_k(self) -> int:
        """The K the integration branch opens at: 2 for InferenceX, whose merge tests and profiles are short; 1 for
        AgentX, whose merge test holds one of a few large slots for its whole trace window."""

        return 1 if self.inferencex is None else 2

    @property
    def open_k(self) -> int:
        """The K a specialist branch the judge opens runs at when the judge names none: :data:`AGENTX_OPEN_K` for
        AgentX, whose long tests leave a branch's one slot idle while its specialist thinks, so its branches together
        may hold more K than tests run at once and the queue holds the excess; 1 for InferenceX."""

        return AGENTX_OPEN_K if self.inferencex is None else 1

    @property
    def bundles_merges(self) -> bool:
        """Whether a merge round tests the commits it is offered together, in one merge test, before it tests them one
        by one. An AgentX merge test holds one of a few large slots for its whole trace window, so one test per commit
        is the integration branch's bottleneck; an InferenceX one is short, and attributes its gain to one commit."""

        return self.inferencex is None

    @property
    def final_concs(self) -> tuple[int, ...]:
        """The concs FINAL measures a candidate at: the sweep around ``conc`` for InferenceX, and ``conc`` alone for
        AgentX, which defines no other operating point."""

        return (self.conc,) if self.inferencex is None else sweep_concs(self.conc)


@dataclass(frozen=True)
class RunSettings:
    """``max_hours`` is the whole session's wall-clock budget, PRELUDE included, ``max_gpu_hours`` the GPU hours
    the session may use, tests' GPU time plus card time any lease holds, and ``max_llm_usd`` the dollars the session's
    recorded LLM calls, the specialists' and the judge's and patch resolver's, may cost; 0 means no limit for any of
    them. ``user_directions`` are the directions the user
    wrote, each opening one branch where EXPLORE starts, ahead of the opening batch, within the branch cap; ``agents``
    the third-party agents the user named, each opening one autonomous branch the same way, and ``agent_directions``
    the directions map whose disjoint shares the component agents among them work, one share each (``""`` for none; see
    :mod:`~hyperloom_multibranch.scheduling.agent_plans`), ``agent_pack`` the pack those component agents run (``""``
    for the adapter's default), ``agent_launch`` the file of the launch their recipe's adds
    to (``""`` for the session's frozen base), ``agent_dossier_minutes`` how long the dossier pass that maps
    two or more component agents' directions may run when no map is given (0: none, the map is generated from the root's
    shares; :mod:`~hyperloom_multibranch.specialist.dossier_pass`), ``agent_share_minutes`` the least time to their
    cutoff the share branches of a split map must have to open, and the least an agent branch whose lease is refused
    for want of cards must still have to run to keep waiting for them (0: they always open, and wait while any time is
    left), and ``agent_port`` the first port of the blocks
    their servers take, one block per agent branch, a mapping branch's included, and one for a pass before EXPLORE. ``stop_after_map`` ends the session
    once the component agents' directions are mapped, those of a single one too, before EXPLORE, and leaves the map and
    its dossier for a person to review; the run after the review is given that map as ``agent_directions`` and that
    dossier as ``agent_dossier``, which every agent branch gets in place of a dossier pass's (``""`` for none).
    ``review_gate`` has the component agents wait, once their map is made, until a person approves or rejects it
    (:mod:`.review`): the mapping branch waits in EXPLORE while the others go on, a given map pauses the session before
    EXPLORE. It tells each of them the host reviewed; off, nothing waits and each is started with its own gate off.
    ``references`` are the
    directories of reference material the run copies in before its first phase, each with the commit it was taken at.
    ``reference_script`` is the operator's launch file the session freezes as its base (``""`` to resolve it from the
    recipe sources; :mod:`~hyperloom_multibranch.scheduling.session_base`).
    ``dev_leases`` is how many dev sandboxes the session lends its specialists at once; 0 lends none. ``playbook`` says
    which scheduling playbook the session plays. ``inferencex_commit`` is the commit a fresh session's base launch reads
    InferenceX at (``""`` for the head of its ``main``). ``agentx_early_stop`` is how the AgentX checkpoint treats an
    explore that trails its parent (``toolbox.agentx_checkpoint``): recorded only by default; the manifest keeps it,
    so a resumed session judges as it began. ``make_before_break`` has a hosted session keep a finished test's
    sandbox until the next test's runs, so that the cluster does not give its node to another job between two tests,
    where the grant fits at least three tests at once; it runs one test fewer at once there. Off, every test the grant
    fits runs at once and each sandbox is closed when its test ends. A session measured on this machine ignores it.
    ``conc_sweep`` runs the concurrency sweep of the closed session after CLOSE
    (:class:`~hyperloom_multibranch.scheduling.final.ConcSweep`); ``conc_sweep_concs`` are the operator's points, in
    place of the base's InferenceX arm's; ``conc_sweep_parallel`` is how many of the sweep's tests run at once, within
    ``caps.max_parallel_tests``."""

    workload: Workload
    max_hours: float
    max_gpu_hours: float
    max_llm_usd: float
    caps: RosterCaps
    judge_interval_seconds: float
    user_directions: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()
    references: tuple[ReferenceSource, ...] = ()
    reference_script: str = ""
    dev_leases: int = 0
    playbook: PlaybookMode = PlaybookMode.AUTO
    agent_directions: str = ""
    agent_pack: str = ""
    agent_launch: str = ""
    agent_dossier_minutes: float = DOSSIER_MINUTES
    agent_share_minutes: float = SHARE_MINUTES
    agent_port: int = AGENT_PORT
    stop_after_map: bool = False
    agent_dossier: str = ""
    review_gate: bool = True
    inferencex_commit: str = ""
    agentx_early_stop: EarlyStopMode = EarlyStopMode.SHADOW
    make_before_break: bool = True
    conc_sweep: bool = False
    conc_sweep_concs: tuple[int, ...] = ()
    conc_sweep_parallel: int = DEFAULT_CONC_SWEEP_PARALLEL


def session_early_stop(environ: Mapping[str, str]) -> EarlyStopMode:
    """How a new session holds its explores to the AgentX checkpoint: the mode ``environ`` names, shadow when it names
    none (single-lineage's default is enforce); a resumed session keeps the mode its manifest recorded."""
    return early_stop_mode(environ, EarlyStopMode.SHADOW)


__all__ = [
    "AGENTX_OPEN_K",
    "AGENT_PORT",
    "DEFAULT_AGENTX_OBJECTIVE",
    "DEFAULT_CONC_SWEEP_PARALLEL",
    "DOSSIER_MINUTES",
    "MAX_BOOT_BUDGET_SEC",
    "SHARE_MINUTES",
    "SWEEP_FACTORS",
    "InferenceXShape",
    "KernelOptimizer",
    "PlaybookMode",
    "RunSettings",
    "Workload",
    "sweep_concs",
    "session_early_stop",
]
