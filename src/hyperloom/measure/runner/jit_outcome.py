# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What aiter's JIT says of a test's end beyond what the server's own logs say, for a test that patches aiter's sources.

Two things only the runner can tell, because only it knows which modules the patch touched:

* a server that failed while a touched module was being built, with the compiler's error in its log, failed *because of the
  candidate's change*: that is the candidate's ``compile_error``, with the first errors as the reason. The same error in a
  module the patch did not touch is not the candidate's, and the test keeps the verdict the logs gave it;
* a test that ended well but ran a touched module from a stale copy (:meth:`JitCacheSession.stale_modules`) measured old
  code. Its numbers are not the candidate's, so it ends as an infrastructure failure, which is a ``measurement_failed`` that
  no negative result is written for.
"""

from __future__ import annotations

from typing import Any

from hyperloom.contracts.measurement import FailureClass
from hyperloom.measure.jit.builds import BuildLog
from hyperloom.measure.jit.session import JitCacheSession
from hyperloom.measure.returncodes import HOSTED_INFRA_FAILURE_RC

__all__ = ["COMPILE_ERROR", "STALE_MODULE", "refine_outcome"]

COMPILE_ERROR = "compile_error"
STALE_MODULE = "stale_aiter_module"
_SHOWN_MODULES = 5
#: The failure classes whose cause the logs did not settle: a compile error the logs name settles it.
_OPEN_CLASSES = frozenset({FailureClass.UNCLEAR.value, FailureClass.CANDIDATE.value})


def refine_outcome(document: dict[str, Any], session: JitCacheSession | None, builds: BuildLog) -> dict[str, Any]:
    """``document`` (the outcome the runner reports) with the verdict the touched modules decide, or as it was."""
    if session is None or not session.touched:
        return document
    failure_class = document["failure_class"]
    if failure_class is None and document["returncode"] == 0:
        stale = session.stale_modules(builds)
        if stale:
            return _failed(document, FailureClass.INFRA, f"{STALE_MODULE}: {_named(stale)} ran as the image built it")
    elif failure_class in _OPEN_CLASSES:
        broken = sorted(module for module in builds.failed if module in session.touched)
        if broken:
            reason = f"{COMPILE_ERROR}: {broken[0]}: {builds.failed[broken[0]]}"
            return _failed(document, FailureClass.CANDIDATE, reason)
    return document


def _named(modules: list[str]) -> str:
    more = len(modules) - _SHOWN_MODULES
    return ", ".join(modules[:_SHOWN_MODULES]) + (f" and {more} more" if more > 0 else "")


def _failed(document: dict[str, Any], failure_class: FailureClass, reason: str) -> dict[str, Any]:
    refined = {**document, "error_class": None, "failure_class": failure_class.value, "failure_reason": reason}
    if refined["returncode"] == 0:
        refined["returncode"] = HOSTED_INFRA_FAILURE_RC
    return refined
