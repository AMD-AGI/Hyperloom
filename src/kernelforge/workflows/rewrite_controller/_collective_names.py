# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Does this operator's name suggest it needs more than one rank?

Used to refuse a task that declares several ranks for an operator that reads
like ordinary single-GPU work: without it, a rank count typed onto the wrong
task costs a whole campaign budget before anything notices.

Deliberately permissive. Whether the ranks were really needed is settled later
by watching the run -- the probe reports how many ranks launched, which devices
they bound and how they reduced -- so letting a doubtful name through costs
little, while refusing a real collective because it is named unusually costs
the operator the whole lane. Kernel names are not written to a convention:
they arrive camel-cased, mangled, abbreviated (``custom_ar``, ``ag_gemm``) and
sometimes named for their role rather than their collective
(``EpDispatchCombineOp``), so a phrase is matched against the name with its
separators removed and the short forms are matched as whole words.

That same reasoning is why there is no approximate matching here. A misspelled
or truncated spelling escaping the check costs what any doubtful name costs --
nothing the run does not settle -- which does not buy a similarity scan with
two thresholds to tune.
"""

from __future__ import annotations

import re

#: Collective vocabulary, compared against the name with its separators
#: removed, so ``all_reduce``, ``allReduce`` and ``allreduce`` are one entry.
_PHRASES: tuple[str, ...] = (
    "allreduce",
    "allgather",
    "allgatherv",
    "reducescatter",
    "alltoall",
    "broadcast",
    "bcast",
    "gather",
    "scatter",
    "sendrecv",
    "collective",
    "nccl",
    "rccl",
    "ncclx",
    "epdispatch",
    "dispatchcombine",
    "epcombine",
    "p2p",
)

#: Short forms that are only a signal as a whole word. ``ar`` inside "arange"
#: means nothing; ``custom_ar`` means an all-reduce.
_ABBREVIATIONS: frozenset[str] = frozenset({"ar", "ag", "rs", "a2a", "ep", "comm", "dist"})

#: A stated rank count, required of a multi-rank operator name by
#: :func:`carries_parallelism_suffix`.
#:
#: Not evidence of a collective, and deliberately not consulted by
#: :func:`looks_like_multi_rank_operator`. A suffix says which rank count a
#: recipe was tuned for, not that ranks are needed to compute the answer --
#: ``fused_moe_tp8`` is a tensor-parallel shard of ordinary single-GPU work.
#: Counting it as evidence would make the collective test vacuous, since the
#: suffix is separately mandatory.
_PARALLEL_SUFFIX = re.compile(r"^(?:tp|ep|dp|pp|cp|sp)\d+$")

_NORMALISE_DELIMS = re.compile(r"[\W]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# Itanium mangling prefixes each identifier with its length ("5aiter",
# "33reduce_scatter_..."), gluing a digit onto the token it precedes.
_DIGIT_LETTER_BOUNDARY = re.compile(r"(?<=\d)(?=[a-z])")


def normalise_kernel_name(name: str) -> str:
    """Lowercase and underscore-delimit a name so token tests are stable."""
    if not name:
        return ""
    text = _CAMEL_BOUNDARY.sub("_", str(name))
    text = _NORMALISE_DELIMS.sub("_", text)
    text = _DIGIT_LETTER_BOUNDARY.sub("_", text.lower())
    return text.strip("_")


def carries_parallelism_suffix(*names: str) -> bool:
    """Whether any supplied name states the rank count it was tuned for.

    Required rather than encouraged: ``world_size`` is not part of the identity
    that keys the experience store, so this suffix is the only thing separating
    one rank count's recipe from another's.
    """
    for name in names:
        tokens = normalise_kernel_name(name).split("_")
        if any(_PARALLEL_SUFFIX.match(token) for token in tokens):
            return True
    return False


def looks_like_multi_rank_operator(*names: str) -> bool:
    """Whether any supplied name reads like an operator that spans ranks.

    Several names are accepted because a task carries more than one spelling of
    itself -- the published ``operator_name`` and the normalized identity -- and
    either may be the one that says what the operator does.
    """
    for name in names:
        normalised = normalise_kernel_name(name)
        if not normalised:
            continue
        tokens = set(normalised.split("_"))
        if tokens & _ABBREVIATIONS:
            return True
        squashed = normalised.replace("_", "")
        if any(phrase in squashed for phrase in _PHRASES):
            return True
    return False


__all__ = [
    "carries_parallelism_suffix",
    "looks_like_multi_rank_operator",
    "normalise_kernel_name",
]
