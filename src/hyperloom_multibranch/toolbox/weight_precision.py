# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The model's weight precision is the session's: what a launch's flags say about it, and how two launches differ.

A change of the weights' precision, down or up, changes what is being optimized: an FP8-weight launch of a BF16 session
measures another model, whatever it gains. The precision of intermediates (the KV cache, attention compute, collectives,
the activations of weights the checkpoint already holds in FP8) may change, with the weights as they are and accuracy
passing.

The flags read here set the weights' format, their quantization at load time, their dtype, or the checkpoint itself.
Only launch flags are read: a source patch that casts weights at load is not.

The comparison is Arbor's (``arbor/measure/precision.py``): ``--flag=value`` reads as ``--flag value``; a flag's values
are the words up to the next flag, and a word before any flag is a positional, not compared; a word that reads as a
number (``-1``) is a value, not a flag; a JSON object or array compares in canonical form (sorted keys, no spaces) and
any other value as its text; a flag given without a value reads as on; a flag added, removed or given another value is
a change. The spellings of one setting are one flag, and a value it is given twice counts once, under one spelling or
two. A flag the launch removes from the resolved launch (``LaunchConfig.remove_args``) has the value
:data:`~hyperloom.contracts.recipe.REMOVED`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from hyperloom.contracts.recipe import REMOVED, LaunchConfig
from hyperloom.inference_optimizer.grid_server_args import tokenize_server_args_preserving_json

#: The launch flags that set the served weights' precision, by the name a change is reported under, each with every
#: spelling the frameworks accept for it: vLLM's and SGLang's quantization method, its config and its parameter file,
#: ATOM's online quantization, the weights' dtype (``--dtype`` sets it in both frameworks), and the checkpoint (vLLM's
#: ``--model``, SGLang's ``--model-path``), since another checkpoint may hold another precision.
WEIGHT_PRECISION_FLAGS: Mapping[str, tuple[str, ...]] = {
    "--quantization": ("--quantization", "-q"),
    "--quantization-config": ("--quantization-config",),
    "--quantization-param-path": ("--quantization-param-path",),
    "--online-quant-config": ("--online-quant-config", "--online_quant_config"),
    "--dtype": ("--dtype",),
    "the checkpoint": ("--model", "--model-path"),
}

_NOT_SET = "(not set)"
_ON = "(on)"


def weight_precision_changes(before: LaunchConfig, after: LaunchConfig) -> list[str]:
    """Each weight-precision setting ``after`` changes from ``before``, as ``"flag: old -> new"`` (``(not set)`` for
    a setting a launch does not make); ``[]`` when the weights' precision is as it was."""
    old, new = _settings(before), _settings(after)
    return [
        f"{name}: {_shown(old.get(name))} -> {_shown(new.get(name))}"
        for name in sorted({*old, *new})
        if old.get(name) != new.get(name)
    ]


def _settings(launch: LaunchConfig) -> dict[str, frozenset[tuple[str, ...]]]:
    """The weight-precision settings ``launch`` makes, by name: every value its flags give the setting."""
    found: dict[str, set[tuple[str, ...]]] = {}
    removed = [(flag, (REMOVED,)) for flag in launch.remove_args]
    for flag, values in [*_flags(launch.server_args), *removed]:
        name = next((name for name, spellings in WEIGHT_PRECISION_FLAGS.items() if flag in spellings), None)
        if name is not None:
            found.setdefault(name, set()).add(values)
    return {name: frozenset(values) for name, values in found.items()}


def _flags(server_args: str) -> list[tuple[str, tuple[str, ...]]]:
    """``server_args`` as (flag, values) pairs, in the order given; the words before the first flag are left out."""
    parsed = tokenize_server_args_preserving_json(server_args)
    words = parsed[1] if parsed is not None else server_args.split()
    pairs: list[tuple[str, list[str]]] = []
    for word in words:
        if word.startswith("-") and len(word) > 1 and not _is_number(word):
            flag, assigned, value = word.partition("=")
            pairs.append((flag, [value] if assigned else []))
        elif pairs:
            pairs[-1][1].append(word)
    return [(flag, tuple(_canonical(value) for value in values)) for flag, values in pairs]


def _canonical(value: str) -> str:
    """``value`` as compared: a JSON object or array in canonical form, any other value as given."""
    stripped = value.strip()
    if stripped.startswith(("{", "[")):
        try:
            return json.dumps(json.loads(stripped), sort_keys=True, separators=(",", ":"))
        except ValueError:
            return value
    return value


def _is_number(word: str) -> bool:
    try:
        float(word)
    except ValueError:
        return False
    return True


def _shown(values: frozenset[tuple[str, ...]] | None) -> str:
    if values is None:
        return _NOT_SET
    return " | ".join(sorted(" ".join(value) or _ON for value in values))


__all__ = ["WEIGHT_PRECISION_FLAGS", "weight_precision_changes"]
