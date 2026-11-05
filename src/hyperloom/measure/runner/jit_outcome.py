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

And one thing it can tell for every test that runs aiter: a server that stopped because a tuned table names a kernel the
module it loaded lacks (``... is not present in the compiled registry``). aiter builds a GEMM module's kernels from the
tuned table and loads a module it finds without building it, so the verdict depends on where the module came from. One
this test built (its log names the build) or the cache gave it (under this test's key, which covers the tables) was
generated from the table the test runs, and that table names a kernel its own build did not make: the candidate's
failure. Any other module was built elsewhere, from another table (the image's, or one a test on the node left), so
nothing was learned about the candidate: an infrastructure failure, retried in a fresh sandbox away from the node.
"""

from __future__ import annotations

from typing import Any

from hyperloom.contracts.measurement import FailureClass
from hyperloom.measure.jit.builds import BuildLog
from hyperloom.measure.jit.session import JitCacheSession
from hyperloom.measure.returncodes import HOSTED_INFRA_FAILURE_RC

__all__ = ["COMPILE_ERROR", "KERNEL_NOT_BUILT", "STALE_JIT_MODULE", "STALE_MODULE", "failed_outcome", "refine_outcome"]

COMPILE_ERROR = "compile_error"
STALE_MODULE = "stale_aiter_module"
STALE_JIT_MODULE = "stale_jit_module"
KERNEL_NOT_BUILT = "kernel_not_built"
_SHOWN_MODULES = 5
#: The failure classes whose cause the logs did not settle: a compile error the logs name settles it.
_OPEN_CLASSES = frozenset({FailureClass.UNCLEAR.value, FailureClass.CANDIDATE.value})


def refine_outcome(document: dict[str, Any], session: JitCacheSession | None, builds: BuildLog) -> dict[str, Any]:
    """``document`` (the outcome the runner reports) with the verdict a missing kernel or the touched modules decide, or
    as it was."""
    missing = _missing_kernel_outcome(document, session, builds)
    if missing is not None:
        return missing
    if session is None or not session.touched:
        return document
    failure_class = document["failure_class"]
    if failure_class is None and document["returncode"] == 0:
        stale = session.stale_modules(builds)
        if stale:
            return failed_outcome(
                document, FailureClass.INFRA, f"{STALE_MODULE}: {_named(stale)} ran as the image built it"
            )
    elif failure_class in _OPEN_CLASSES:
        broken = sorted(module for module in builds.failed if module in session.touched)
        if broken:
            reason = f"{COMPILE_ERROR}: {broken[0]}: {builds.failed[broken[0]]}"
            return failed_outcome(document, FailureClass.CANDIDATE, reason)
    return document


def _missing_kernel_outcome(
    document: dict[str, Any], session: JitCacheSession | None, builds: BuildLog
) -> dict[str, Any] | None:
    """The verdict on a failure whose log says a module lacks a kernel its tuned table names; ``None`` for any other."""
    if document["failure_class"] not in _OPEN_CLASSES or not builds.missing_kernels:
        return None
    foreign = sorted(module for module in builds.missing_kernels if not _built_here(module, session, builds))
    if foreign:
        module = foreign[0]
        return failed_outcome(
            document,
            FailureClass.INFRA,
            f"{STALE_JIT_MODULE}: {module} lacks kernel '{builds.missing_kernels[module]}' and was not built by this "
            "test from the table it runs",
        )
    module = min(builds.missing_kernels)
    return failed_outcome(
        document,
        FailureClass.CANDIDATE,
        f"{KERNEL_NOT_BUILT}: {module} was built by this test from its tuned table, which names kernel "
        f"'{builds.missing_kernels[module]}' that the build did not make",
    )


def _built_here(module: str, session: JitCacheSession | None, builds: BuildLog) -> bool:
    """Whether ``module`` was generated from this test's tables: built in this test, or given by the cache."""
    built = module in builds.started and module not in builds.failed
    return built or (session is not None and session.gave(module))


def _named(modules: list[str]) -> str:
    more = len(modules) - _SHOWN_MODULES
    return ", ".join(modules[:_SHOWN_MODULES]) + (f" and {more} more" if more > 0 else "")


def failed_outcome(document: dict[str, Any], failure_class: FailureClass, reason: str) -> dict[str, Any]:
    """``document`` ended as a ``failure_class`` failure for ``reason``; a run that exited 0 gets the infra exit code."""
    refined = {**document, "error_class": None, "failure_class": failure_class.value, "failure_reason": reason}
    if refined["returncode"] == 0:
        refined["returncode"] = HOSTED_INFRA_FAILURE_RC
    return refined
