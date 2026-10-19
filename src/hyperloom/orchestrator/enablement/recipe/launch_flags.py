# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Requested launch flags: how a requested server argv is read.

A leaf of the recipe package: both the replay-sufficiency verdict and the
launch-evidence projection read the requested argv, and neither may import the
other (``sufficiency`` reaches ``projections`` through ``steps``).
"""

from __future__ import annotations

import shlex

#: The requested spellings of each parallelism axis ``observed_model_binding``
#: reports, so the requested side is compared against the binding rather than
#: against a launch line the extractor strips these from.
PARALLELISM_AXES: tuple[tuple[str, frozenset[str]], ...] = (
    ("tp", frozenset({"tensor_parallel_size", "tp_size", "tp"})),
    ("dp", frozenset({"data_parallel_size", "dp_size"})),
    ("pp", frozenset({"pipeline_parallel_size", "pp_size"})),
)


def normalize_flag(name: str) -> str:
    return str(name or "").lstrip("-").replace("-", "_")


def requested_flags(argv: str) -> dict[str, str]:
    """Parse an argv string into ``{normalized_flag: value}``.

    JSON operands keep their inner quotes, as the launch passes them: a POSIX
    split turns ``{"mode":3}`` into ``{mode:3}``, which is no longer JSON and
    cannot be compared with what the engine recorded.
    """
    from hyperloom.inference_optimizer.grid_server_args import tokenize_server_args_preserving_json

    tokenized = tokenize_server_args_preserving_json(str(argv or ""))
    if tokenized is not None:
        tokens = tokenized[1]
    else:
        try:
            tokens = shlex.split(str(argv or ""))
        except ValueError:
            return {}
    flags: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("-"):
            index += 1
            continue
        if "=" in token:
            name, _, value = token.partition("=")
            flags[normalize_flag(name)] = value
        else:
            value = ""
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
                value = tokens[index + 1]
                index += 1
            flags[normalize_flag(token)] = value
        index += 1
    return flags


def requested_setting_names(argv: str) -> frozenset[str]:
    """The setting names a requested argv sets, ``--no-X`` naming ``X`` as well.

    The parallelism widths are always included: they are compared on every
    launch, through the binding, whether or not the argv spells them.
    """
    names = set(requested_flags(argv))
    names |= {name[3:] for name in names if name.startswith("no_")}
    return frozenset(names.union(*(spellings for _axis, spellings in PARALLELISM_AXES)))
