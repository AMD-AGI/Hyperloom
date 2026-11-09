# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The brain the body calls, and the check that a brain honours the contract it is loaded against."""

from __future__ import annotations

import inspect
from typing import Mapping, Protocol, runtime_checkable

from hyperloom.brain_api.calls import CallSite, Proposal, ProposeAsk, Review, ReviewAsk
from hyperloom.brain_api.ports import Ports
from hyperloom.brain_api.state import SearchState

#: ``major.minor``. Before 1.0 every minor may break, so a brain must match it exactly; from 1.0 on a brain must
#: match the major, and a newer minor only adds.
CONTRACT_VERSION = "0.1"


class BrainContractError(RuntimeError):
    """A brain that does not honour the contract it was loaded against."""


@runtime_checkable
class Brain(Protocol):
    """Proposes candidates and reviews them. Each call reads one :class:`SearchState` and keeps nothing between
    calls, so any call can be replayed on its own."""

    name: str
    contract_version: str

    async def propose(self, state: SearchState, ask: ProposeAsk, ports: Ports) -> Proposal:
        """Candidates that answer ``ask``, best first."""
        ...

    async def review(self, state: SearchState, ask: ReviewAsk, ports: Ports) -> Review:
        """One verdict per candidate in ``ask``."""
        ...


#: Call site -> the brain method the body calls there.
CALL_SITE_METHODS: Mapping[CallSite, str] = {
    CallSite.PROPOSE: "propose",
    CallSite.REVIEW: "review",
}


def _parse(version: object) -> tuple[int, int] | None:
    if not isinstance(version, str):
        return None
    major, dot, minor = version.partition(".")
    if not (dot and major.isdigit() and minor.isdigit()):
        return None
    return int(major), int(minor)


def compatible(brain_version: str, body_version: str = CONTRACT_VERSION) -> bool:
    """Whether a brain built for ``brain_version`` may run on a body speaking ``body_version``."""
    brain, body = _parse(brain_version), _parse(body_version)
    if brain is None or body is None:
        return False
    if body[0] == 0:
        return brain == body
    return brain[0] == body[0] and brain[1] <= body[1]


def validate_brain(brain: object) -> Brain:
    """Return ``brain`` if it honours this contract.

    Raises:
        BrainContractError: if it names no brain, targets an incompatible contract, or lacks a call site.
    """
    name = getattr(brain, "name", None)
    if not (isinstance(name, str) and name):
        raise BrainContractError(f"{type(brain).__name__} has no name")
    version = getattr(brain, "contract_version", None)
    if _parse(version) is None:
        raise BrainContractError(f"brain {name!r} contract_version {version!r} is not major.minor")
    if not compatible(str(version)):
        raise BrainContractError(f"brain {name!r} targets contract {version}, the body speaks {CONTRACT_VERSION}")
    missing = [m for m in CALL_SITE_METHODS.values() if not inspect.iscoroutinefunction(getattr(brain, m, None))]
    if missing:
        raise BrainContractError(f"brain {name!r} has no async {', '.join(missing)}")
    return brain  # type: ignore[return-value]
