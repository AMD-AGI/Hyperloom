# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Eval points: the functions of a brain an evaluator may score on their own.

A brain marks a function with :func:`eval_point`. The function takes ``(state, ask, ports)`` and answers with a
contract type, exactly like :meth:`Brain.propose` and :meth:`Brain.review`, which are the points ``main.propose`` and
``main.review``. A unit inside the brain (a specialist, say) is a point of its own, so an evaluator can replay
recorded calls against it directly, or watch it while the main points run.

The decorator does nothing unless an :class:`Observer` is installed with :func:`observing`; an evaluator installs
one to record or score each call, and may answer a call in the unit's place to hold what is upstream of it fixed.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import re
import time
import typing
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterator, Protocol, TypeVar

from hyperloom.brain_api.brain import CONTRACT_VERSION, BrainContractError
from hyperloom.brain_api.calls import CallSite, Proposal, Review
from hyperloom.brain_api.ports import Ports
from hyperloom.brain_api.record import CALL_TYPES, MAIN_POINTS, BrainCall, valid_point_name
from hyperloom.brain_api.state import SearchState

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])
_VERSION = re.compile(r"^\d+\.\d+$")


@dataclass(frozen=True)
class EvalPoint:
    """A scorable function: its name, the version of what its answers mean, and the kind of call it is."""

    name: str
    version: str
    call_site: CallSite


class Observer(Protocol):
    """What an evaluator installs around a brain while it runs."""

    async def replace(self, point: EvalPoint, state: SearchState, ask: Any) -> Proposal | Review | None:
        """An answer to use instead of calling the point, or ``None`` to call it."""
        ...

    async def observe(self, call: BrainCall) -> None:
        """A point was called; ``call`` holds its answer or its error."""
        ...


_OBSERVER: ContextVar[tuple[Observer, str] | None] = ContextVar("hyperloom_brain_observer", default=None)


@contextlib.contextmanager
def observing(observer: Observer, *, brain: str) -> Iterator[None]:
    """Install ``observer`` for every point the current task calls; ``brain`` names the brain in each record."""
    token = _OBSERVER.set((observer, brain))
    try:
        yield
    finally:
        _OBSERVER.reset(token)


def _call_site_of(fn: Callable[..., Any], name: str) -> CallSite:
    hints = typing.get_type_hints(fn)
    pair = (hints.get("ask"), hints.get("return"))
    for site, types in CALL_TYPES.items():
        if pair == types:
            return site
    raise TypeError(f"eval point {name!r} must take ask: X and return Y for a contract pair, got {pair}")


def eval_point(name: str, *, version: str = "0.1") -> Callable[[F], F]:
    """Mark an ``async (self, state, ask, ports)`` method as the eval point ``name``.

    ``version`` is the version of what the point's answers mean; bump it when that changes, so scores of the old
    and the new meaning are never compared.
    """
    if not valid_point_name(name):
        raise ValueError(f"eval point name {name!r} is not dotted lower_snake_case, e.g. 'specialist.kv.propose'")
    if not _VERSION.match(version):
        raise ValueError(f"eval point {name!r} version {version!r} is not major.minor")

    def wrap(fn: F) -> F:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"eval point {name!r} must be an async function")
        point = EvalPoint(name, version, _call_site_of(fn, name))
        answer_type = CALL_TYPES[point.call_site][1]

        @functools.wraps(fn)
        async def inner(self: Any, state: SearchState, ask: Any, ports: Ports) -> Any:
            installed = _OBSERVER.get()
            if installed is None:
                return await fn(self, state, ask, ports)
            observer, brain = installed
            replaced = await observer.replace(point, state, ask)
            if replaced is not None:
                if not isinstance(replaced, answer_type):
                    raise TypeError(f"observer answered {name!r} with a {type(replaced).__name__}")
                return replaced
            record = functools.partial(
                BrainCall,
                point.call_site,
                CONTRACT_VERSION,
                brain,
                state,
                ask,
                point=name,
                point_version=version,
                started_unix=time.time(),
            )
            started = time.monotonic()
            try:
                answer = await fn(self, state, ask, ports)
            except Exception as exc:
                await observer.observe(
                    record(error=f"{type(exc).__name__}: {exc}", duration_sec=time.monotonic() - started)
                )
                raise
            await observer.observe(record(answer, duration_sec=time.monotonic() - started))
            return answer

        inner.__eval_point__ = point  # type: ignore[attr-defined]
        return inner  # type: ignore[return-value]

    return wrap


def point_of(fn: Callable[..., Any] | None) -> EvalPoint | None:
    """The eval point ``fn`` is marked as, if any."""
    return getattr(fn, "__eval_point__", None)


def points_of(brain: object) -> dict[str, Callable[..., Awaitable[Any]]]:
    """Every eval point of ``brain`` by name: ``main.propose``, ``main.review``, and what ``eval_points()`` lists.

    Raises:
        BrainContractError: if a main method is not its main point, or a listed point is unmarked or misnamed.
    """
    points: dict[str, Callable[..., Awaitable[Any]]] = {}
    for site, name in MAIN_POINTS.items():
        fn = getattr(brain, site.value, None)
        point = point_of(fn)
        if point is None or point.name != name:
            raise BrainContractError(f"{type(brain).__name__}.{site.value} must be marked @eval_point({name!r})")
        points[name] = fn  # type: ignore[assignment]
    listed = getattr(brain, "eval_points", None)
    for name, fn in (listed() if callable(listed) else {}).items():
        point = point_of(fn)
        if point is None or point.name != name:
            raise BrainContractError(f"eval_points() lists {name!r} for a function marked {point}")
        if points.setdefault(name, fn) != fn:
            raise BrainContractError(f"eval point {name!r} is listed twice")
    return points
