# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rounds of one test of a point, on the one server the test holds.

A test of a point that warms up (:func:`warms_up`) is two rounds: a discarded bench that boots the test's server, then
the measured bench on that server, which the accuracy eval follows inside the same round. The eval never runs before
the measured bench, so the bench starts on a server no eval has loaded. A point that does not warm up is measured in
one round, on a server the test boots.

A test that must evaluate has no result without its score: its caller keeps neither a measured bench whose eval
failed nor the discarded bench of a test whose measured round did not run.

:class:`HeldTest` is the one place a round's purpose becomes how the round holds the server, which seed its load
draws and whether it evaluates. It gives every round the test's port, refuses a round out of that order before it
runs, and stops the server once, however the test ends.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from hyperloom.common.benchmark_config import read_benchmark_point
from hyperloom.common.ports import pick_server_port
from hyperloom.common.round_measurement import load_measured
from inference_suite import RoundResult, stop
from inference_suite.contracts import Lifecycle, LifecycleMode

from ..serving import _server_lifecycle
from ..workload._agentx_checkpoint import AgentxCheckpoint
from .rounds import SuiteRound, SuiteRoute, run_suite_round

log = logging.getLogger(__name__)


class RoundPurpose(Enum):
    """What a round of a held test is for."""

    #: The discarded bench that boots the test's server: the load on the warm-up seed, and no eval.
    WARMUP = "warmup"
    #: The measured bench, then the eval the point asks for.
    MEASURE = "measure"


@dataclass(frozen=True)
class HeldRound:
    """What a caller says about one round of a held test; the test adds how the round holds the server.

    The fields are :class:`~.rounds.SuiteRound`'s of the same names.
    """

    config_path: Path
    round_dir: Path
    deadline_unix: int | None = None
    env_drop: frozenset[str] = frozenset()
    checkpoint: AgentxCheckpoint | None = None


class HeldTestOrderError(RuntimeError):
    """A round was asked of a held test out of the order a test runs its rounds in; the round did not run."""


def can_hold(config_path: Path) -> bool:
    """Whether this host can hold the server of the point at ``config_path`` from one round of a test to the next."""
    return bool(_server_lifecycle.resolve_lifecycle_params(config_path)["eligible"])


def warms_up(config_path: Path) -> bool:
    """Whether a test of the point at ``config_path`` discards a first bench before it measures.

    It does when this host can hold the point's server (:func:`can_hold`) and the point is not measured once.
    """
    return can_hold(config_path) and not read_benchmark_point(config_path).measured_once


class HeldTest:
    """One test's server, held in ``slot`` from the round that boots it until :meth:`stop`.

    ``port`` is the port the server answers on; without one the test takes a free port at its first round.
    ``must_evaluate`` makes the measured round evaluate whatever the point's ``RUN_EVAL`` says; a point whose round
    runs no eval is then refused. Leaving the ``with`` block stops the server.
    """

    def __init__(self, slot: Path, *, port: int | None = None, must_evaluate: bool = False) -> None:
        self.slot = Path(slot)
        self.must_evaluate = must_evaluate
        self._port = port
        #: Whether the load of each round that ran measured.
        self._ran: dict[RoundPurpose, bool] = {}
        self._survivors: tuple[int, ...] | None = None

    def __enter__(self) -> HeldTest:
        return self

    def __exit__(self, *exc: object) -> None:
        if survivors := self.stop():
            log.error("held test: pids %s outlived SIGKILL; %s keeps their record", survivors, self.slot)

    @property
    def warmed(self) -> bool:
        """Whether the test's warm-up has run."""
        return RoundPurpose.WARMUP in self._ran

    def evaluates(self, purpose: RoundPurpose) -> bool | None:
        """Whether the ``purpose`` round runs the eval after its load; ``None`` leaves it to the point's ``RUN_EVAL``."""
        if purpose is RoundPurpose.WARMUP:
            return False
        return True if self.must_evaluate else None

    def run(
        self, purpose: RoundPurpose, route: SuiteRoute, round_: HeldRound, *, on_poll: Callable[[], None]
    ) -> RoundResult:
        """Run ``round_`` as the test's ``purpose`` round on ``route``, as :func:`~.rounds.run_suite_round` runs one.

        Raises:
            HeldTestOrderError: When the test's order has no such round now; nothing ran.
            SuiteRoundRefused: When the round is refused before it runs.
        """
        boots = self._boots(purpose, warms_up(round_.config_path))
        if self._port is None:
            self._port = pick_server_port()
        request = SuiteRound(
            config_path=round_.config_path,
            round_dir=round_.round_dir,
            lifecycle=Lifecycle(LifecycleMode.BOOT if boots else LifecycleMode.REUSE, str(self.slot)),
            warmup=purpose is RoundPurpose.WARMUP,
            deadline_unix=round_.deadline_unix,
            env_drop=round_.env_drop,
            checkpoint=round_.checkpoint,
            evaluates=self.evaluates(purpose),
            port=self._port,
        )
        ran = run_suite_round(route, request, on_poll=on_poll)
        self._ran[purpose] = load_measured(ran.report)
        return ran

    def _boots(self, purpose: RoundPurpose, warms: bool) -> bool:
        """Whether the ``purpose`` round boots the test's server, on a point that ``warms`` up or does not.

        The warm-up is the first round of a point that warms up, and boots. The measured round runs once: on the
        server a warm-up whose load measured left up, or, on a point that does not warm up, alone on one it boots.
        What a step that failed after the warm-up's load means is its caller's to rule on.
        """
        if self._survivors is not None:
            raise HeldTestOrderError(f"the test in {self.slot} has stopped its server")
        if purpose in self._ran:
            raise HeldTestOrderError(f"the test in {self.slot} has run its {purpose.value} round")
        if purpose is RoundPurpose.WARMUP:
            if not warms:
                raise HeldTestOrderError(f"the point of the test in {self.slot} is measured without a warm-up")
            if self._ran:
                raise HeldTestOrderError(f"the warm-up is the first round of the test in {self.slot}")
            return True
        if not warms:
            return True
        if RoundPurpose.WARMUP not in self._ran:
            raise HeldTestOrderError(f"the measured round of the test in {self.slot} follows its warm-up")
        if not self._ran[RoundPurpose.WARMUP]:
            raise HeldTestOrderError(f"the warm-up of the test in {self.slot} measured no load")
        return False

    def stop(self) -> tuple[int, ...]:
        """Stop the test's server, once; the pids that outlived it. A test that has stopped runs no further round."""
        if self._survivors is None:
            self._survivors = stop(self.slot).survivors
        return self._survivors


@dataclass(frozen=True)
class HeldPass:
    """One round of a held test, as the code between the test's owner and the round carries it."""

    test: HeldTest
    purpose: RoundPurpose

    @property
    def evaluates(self) -> bool | None:
        """Whether the round is told to evaluate (:meth:`HeldTest.evaluates`)."""
        return self.test.evaluates(self.purpose)

    @property
    def reuses_server(self) -> bool:
        """Whether the round runs on the server its test's warm-up left up."""
        return self.purpose is RoundPurpose.MEASURE and self.test.warmed

    def run(self, route: SuiteRoute, round_: HeldRound, *, on_poll: Callable[[], None]) -> RoundResult:
        """Run ``round_`` as this round of its test (:meth:`HeldTest.run`)."""
        return self.test.run(self.purpose, route, round_, on_poll=on_poll)


__all__ = [
    "HeldPass",
    "HeldRound",
    "HeldTest",
    "HeldTestOrderError",
    "RoundPurpose",
    "can_hold",
    "warms_up",
]
