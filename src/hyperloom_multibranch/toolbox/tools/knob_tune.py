# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``knob_tune`` tool: a sweep of a compile-time kernel knob of a component the toolbox builds from its pinned
checkout (:mod:`hyperloom.measure.runner.native.knob_sweep`), timed on a card leased for each round.

A specialist names the knob (``knob``, one the rule pack states), the keys to sweep (``axes``: the values of each axis;
an axis it does not name takes all of its values), the values to force (``values``; all of the knob's when omitted),
the weight shapes each key is timed at (``shapes``: ``[m, k]`` pairs, the model's), the copies of each weight in
one launch (``copies``, so the weights read are past the GPU's cache as a model's are) and the rounds. The builds are the
toolbox's native builds (:class:`~hyperloom.measure.runner.native.session.NativeBuildSession`), keyed by their patch and
run on the build CPUs with no card held. Each timed round leases one card from the pool the tests take theirs from, as
``kernel_tune``'s job does, and gives it back at the round's end, so a sweep never holds a card through its builds and
a lease that would leave fewer cards than one test needs is refused (``lease_leaves_no_measurement``). The sweep runs
in a thread a cancelled job does not end, so a cancelled job's sweep starts no further build or benchmark and leases
no further card.

The answer is the sweep's document: the ranked table of every key and value against the build of the pinned commit,
the noise of the measurement, the value each key is given where one beat the default beyond the noise and matched the
CPU backend, and that patch (a git patch over the pinned commit). Commit it on the branch and submit a test: the
microbenchmark is advice, and only a test says whether the model gets faster.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from hyperloom.common.outside_facts import kernel_knobs
from hyperloom.common.outside_facts.kernel_knob import KnobSpecError
from hyperloom.measure.runner.native.backend_ops import check_cases, time_cases
from hyperloom.measure.runner.native.knob_sweep import (
    DEFAULT_COPIES,
    SweepError,
    SweepHooks,
    SweepPlan,
    SweepTiming,
    plan_sweep,
    run_sweep,
)
from hyperloom.measure.runner.native.session import NativeBuildSession
from hyperloom.measure.runner.native.session_builds import SessionNativeBuilds
from hyperloom.orchestrator.actions.cancel_channel import CancelScope, current_cancel_scope
from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseKind
from hyperloom_multibranch.contracts.toolbox import LeaseCharge, RequestRejected
from hyperloom_multibranch.toolbox.branch_leases import BranchLeases
from hyperloom_multibranch.toolbox.tools.base import InvalidRequest, ToolFailure, ToolOutput, ToolRequest
from hyperloom_multibranch.toolbox.tools.kernel_tune import HOLDER_PREFIX, card_env

__all__ = [
    "KNOB_TUNE_TOOL",
    "KnobTuneTool",
    "check_params",
    "knob_tune_tool",
]

KNOB_TUNE_TOOL = "knob_tune"
_KNOB_PARAM = "knob"
_AXES_PARAM = "axes"
_VALUES_PARAM = "values"
_SHAPES_PARAM = "shapes"
_ROUNDS_PARAM = "rounds"
_COPIES_PARAM = "copies"
_PARAMS = frozenset({_KNOB_PARAM, _AXES_PARAM, _VALUES_PARAM, _SHAPES_PARAM, _ROUNDS_PARAM, _COPIES_PARAM})
_DEFAULT_ROUNDS = 5
_CARDS = 1
_PARALLEL_BUILDS = 2
_RUN_TIMEOUT_SEC = 300
_T = TypeVar("_T")


def _strings(value: Any, what: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise InvalidRequest(f"{what} is a list")
    return [str(item) for item in value]


def _shapes(value: Any) -> list[tuple[int, int]]:
    shapes = []
    for shape in value if isinstance(value, Sequence) and not isinstance(value, str) else [None]:
        if (
            not isinstance(shape, Sequence)
            or len(shape) != 2
            or not all(isinstance(size, int) and not isinstance(size, bool) for size in shape)
        ):
            raise InvalidRequest(f"{_SHAPES_PARAM} is a list of [m, k] weight shapes of whole numbers")
        shapes.append((shape[0], shape[1]))
    return shapes


def check_params(params: Mapping[str, Any]) -> SweepPlan:
    """The sweep the caller's parameters ask for; raises :class:`InvalidRequest` for one no sweep can run."""
    if unknown := sorted(set(params) - _PARAMS):
        raise InvalidRequest(f"{KNOB_TUNE_TOOL} takes {sorted(_PARAMS)}, not {unknown}")
    axes = params.get(_AXES_PARAM) or {}
    rounds = params.get(_ROUNDS_PARAM, _DEFAULT_ROUNDS)
    copies = params.get(_COPIES_PARAM, DEFAULT_COPIES)
    if not isinstance(axes, Mapping):
        raise InvalidRequest(f"{_AXES_PARAM} maps an axis to the values of it to sweep")
    if not all(isinstance(number, int) and not isinstance(number, bool) for number in (rounds, copies)):
        raise InvalidRequest(f"{_ROUNDS_PARAM} and {_COPIES_PARAM} are whole numbers")
    try:
        return plan_sweep(
            str(params.get(_KNOB_PARAM, "")),
            {str(axis): _strings(held, f"{_AXES_PARAM}.{axis}") for axis, held in axes.items()},
            _strings(params.get(_VALUES_PARAM) or [], _VALUES_PARAM),
            SweepTiming(_shapes(params.get(_SHAPES_PARAM)), rounds, copies),
        )
    except KnobSpecError as refused:
        raise InvalidRequest(str(refused)) from None


def _unless_cancelled(scope: CancelScope | None, step: Callable[..., _T], *args: Any) -> _T:
    """``step(*args)``, one build or benchmark of a sweep, unless its job was cancelled before the step started."""
    if scope is not None and scope.cancelled:
        raise SweepError("the job was cancelled")
    return step(*args)


class KnobTuneTool:
    """``knob_tune`` over the native builds of each component the toolbox pins (``builds``, by component), timed on a
    card leased from ``leases`` for each round."""

    name = KNOB_TUNE_TOOL

    def __init__(self, leases: BranchLeases, builds: Mapping[str, SessionNativeBuilds]) -> None:
        self._leases = leases
        self._builds = builds

    def validate(self, request: ToolRequest) -> None:
        """Refuse a request no sweep here can run (:class:`InvalidRequest`), or a lease that would leave fewer cards than
        one test needs (:class:`RequestRejected`)."""
        self._builds_of(check_params(request.params))
        self._leases.check(_CARDS)

    def _builds_of(self, plan: SweepPlan) -> SessionNativeBuilds:
        builds = self._builds.get(plan.knob.component)
        if builds is None:
            raise InvalidRequest(f"{plan.name} is a knob of {plan.knob.component}, which this toolbox does not build")
        gfx = builds.session.planner.site.machine.gfx_arch
        if gfx not in plan.knob.archs:
            raise InvalidRequest(f"{plan.name} sets a branch compiled for {list(plan.knob.archs)}, not for {gfx}")
        return builds

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        plan = check_params(request.params)
        builds = self._builds_of(plan)
        step = partial(partial, _unless_cancelled, current_cancel_scope())
        hold = partial(self._held, asyncio.get_running_loop(), request.branch, workdir)
        try:
            result = await asyncio.to_thread(
                run_sweep,
                plan,
                builds.read_upstream,
                workdir,
                SweepHooks(
                    build=step(builds.build_patch),
                    hold=step(hold),
                    time=step(partial(time_cases, timeout=_RUN_TIMEOUT_SEC)),
                    check=step(partial(check_cases, timeout=_RUN_TIMEOUT_SEC)),
                ),
                parallel=_PARALLEL_BUILDS,
            )
        except (SweepError, KnobSpecError) as failed:
            raise ToolFailure(str(failed)) from None
        winners = result.document["winners"]
        summary = (
            f"{plan.name}: {len(winners)} of {len(plan.keys)} keys beat the default beyond the noise"
            if winners
            else f"{plan.name}: no value beat the default beyond the noise on any of {len(plan.keys)} keys"
        )
        return ToolOutput(result=result.document, summary=f"{summary}; microbenchmark times are advice only")

    @contextlib.contextmanager
    def _held(self, loop: asyncio.AbstractEventLoop, branch: str, workdir: Path) -> Iterator[Mapping[str, str]]:
        """One card for one round, leased by ``tool:<job id>`` and charged to ``branch``, given back at its end."""
        charge = LeaseCharge(LeaseKind.TOOL_JOB, branch, KNOB_TUNE_TOOL)
        leasing = self._leases.lease(f"{HOLDER_PREFIX}{workdir.name}", _CARDS, workdir, charge=charge)
        try:
            lease = asyncio.run_coroutine_threadsafe(leasing, loop).result()
        except (RequestRejected, SandboxUnavailable) as exc:
            raise SweepError(f"no card for a round of the sweep: {exc}") from None
        try:
            yield card_env(lease.gpu_ids)
        finally:
            self._leases.release(lease.lease_id)


def knob_tune_tool(leases: BranchLeases, session: NativeBuildSession, root: Path) -> KnobTuneTool | None:
    """The tool over ``session``'s builds of each component the session pins a commit of that has a knob of a branch
    compiled for the session's GPU architecture, laid below ``root``; ``None`` when there is none."""
    site = session.planner.site
    gfx = site.machine.gfx_arch
    components = sorted({knob.component for knob in kernel_knobs().values() if gfx in knob.archs} & set(site.commits))
    if not components:
        return None
    builds = {
        component: SessionNativeBuilds(
            root / component, component, site.sources / component / site.commits[component], session
        )
        for component in components
    }
    return KnobTuneTool(leases, builds)
