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
(``EpDispatchCombineOp``). Matching is therefore fuzzy on purpose.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

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

#: A parallelism suffix is itself evidence of multi-rank intent, and the
#: analyst prompt already requires one on a multi-rank operator name.
_PARALLEL_SUFFIX = re.compile(r"^(?:tp|ep|dp|pp|cp|sp)\d+$")

#: Below this length a fuzzy comparison stops discriminating, so short entries
#: are matched literally or as whole words and never approximately.
_FUZZY_MIN_LEN = 6

#: Tuned to accept a truncated or misspelled spelling (``allreduc``,
#: ``algather``) while still rejecting an unrelated word of the same length.
_FUZZY_RATIO = 0.85

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


def _fuzzy_contains(haystack: str, phrase: str) -> bool:
    """Whether some window of ``haystack`` reads closely enough like ``phrase``."""
    if len(phrase) < _FUZZY_MIN_LEN or len(haystack) < _FUZZY_MIN_LEN:
        return False
    matcher = SequenceMatcher(None, phrase, "", autojunk=False)
    # Windows one character either side of the phrase absorb a dropped or an
    # extra character, which is what most odd spellings amount to.
    for width in (len(phrase) - 1, len(phrase), len(phrase) + 1):
        if width < _FUZZY_MIN_LEN or width > len(haystack):
            continue
        for start in range(len(haystack) - width + 1):
            matcher.set_seq2(haystack[start : start + width])
            if matcher.ratio() >= _FUZZY_RATIO:
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
        if tokens & _ABBREVIATIONS or any(_PARALLEL_SUFFIX.match(token) for token in tokens):
            return True
        squashed = normalised.replace("_", "")
        if any(phrase in squashed for phrase in _PHRASES):
            return True
        if any(_fuzzy_contains(squashed, phrase) for phrase in _PHRASES):
            return True
    return False


__all__ = [
    "looks_like_multi_rank_operator",
    "normalise_kernel_name",
]
