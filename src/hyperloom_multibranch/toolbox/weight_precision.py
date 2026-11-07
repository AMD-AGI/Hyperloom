# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The model's weight precision is the session's: what a launch's flags say about it, and how two launches differ.

A change of the weights' precision, down or up, changes what is being optimized: an FP8-weight launch of a BF16 session
measures another model, whatever it gains. The precision of intermediates (the KV cache, attention compute, collectives,
the activations of weights the checkpoint already holds in FP8) may change, with the weights as they are and accuracy
passing.

The flags read here set the weights' format, their quantization at load time, their dtype, the checkpoint itself or its
revision, or carry a config file that can set any of them; and two flags with other uses, whose JSON value overrides
the model's config, count in what they set under a dtype or quantization key. Only launch flags are read: a source
patch that casts weights at load is not, nor what a config file holds.

A launch word is first read as the frameworks' argument parsers read it (:func:`_flag`): ``--flag=value`` is ``--flag
value``; a long flag's name reads ``_`` as ``-`` and ends at its first ``.``, and ``--flag.key value`` is a value of
``--flag`` that keeps its key (vLLM merges it into ``--flag``'s JSON); ``-qVALUE`` is ``-q VALUE``; and a long flag that
is no spelling of a setting but begins one or more (``--dty``) is each setting it begins, as argparse takes a prefix of
an option for the option, unless the word is an option of the frameworks' servers itself (:data:`_OWN_OPTIONS`).

The comparison is then Arbor's (``arbor/measure/precision.py``): a flag's values are the words up to the next flag, and
a word before any flag is a positional, not compared; a word that reads as a number (``-1``) is a value, not a flag; a
JSON object or array compares in canonical form (sorted keys, no spaces) and any other value as its text; a flag given
without a value reads as on; a flag added, removed or given another value is a change. The spellings of one setting are
one flag, and a value it is given twice counts once, under one spelling or two. An override flag's value is what it
sets under the keys (:func:`_weight_keys`), and a value that sets none of them is not one. A flag the launch removes
from the resolved launch (``LaunchConfig.remove_args``) has the value :data:`~hyperloom.contracts.recipe.REMOVED`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from hyperloom.contracts.recipe import REMOVED, LaunchConfig
from hyperloom.inference_optimizer.grid_server_args import tokenize_server_args_preserving_json

#: The launch flags that set the served weights' precision, by the name a change is reported under, each with its
#: spellings: vLLM's and SGLang's quantization method, its config and its parameter file, ATOM's online quantization,
#: the weights' dtype (``--dtype`` sets it in both frameworks), the checkpoint (vLLM's ``--model``, SGLang's
#: ``--model-path``) and its revision, since another checkpoint may hold another precision, a config file (vLLM's and
#: SGLang's ``--config``), which can carry any of these flags and is not read here, ModelOpt's and torchao's
#: quantization, and SGLang's quantize-and-serve. A launch word counts as one of these spellings when the frameworks'
#: parsers read it as that spelling: see :func:`_flag` and :func:`_names`.
WEIGHT_PRECISION_FLAGS: Mapping[str, tuple[str, ...]] = {
    "--quantization": ("--quantization", "-q"),
    "--quantization-config": ("--quantization-config",),
    "--quantization-param-path": ("--quantization-param-path",),
    "--online-quant-config": ("--online-quant-config", "--online_quant_config"),
    "--dtype": ("--dtype",),
    "the checkpoint": ("--model", "--model-path"),
    "the checkpoint's revision": ("--revision",),
    "--config": ("--config",),
    "--modelopt-quant": ("--modelopt-quant",),
    "--torchao-config": ("--torchao-config",),
    "--quantize-and-serve": ("--quantize-and-serve",),
}

#: The keys of a model's config that set its weights' dtype or quantization.
WEIGHT_PRECISION_KEYS = frozenset({"torch_dtype", "dtype", "quantization_config"})

#: Flags with other uses (the context length, say) whose JSON value overrides the model's config: vLLM's
#: ``--hf-overrides`` and SGLang's ``--json-model-override-args``. They count only in what they set under one of
#: :data:`WEIGHT_PRECISION_KEYS`, at any depth of the value, by the name a change of that is reported under.
WEIGHT_PRECISION_OVERRIDE_FLAGS: Mapping[str, str] = {
    flag: f"{flag} ({', '.join(sorted(WEIGHT_PRECISION_KEYS))})"
    for flag in ("--hf-overrides", "--json-model-override-args")
}

#: The one-letter spellings (``-q``), whose value argparse also reads attached to them.
_SHORT_SPELLINGS = frozenset(
    spelling for spellings in WEIGHT_PRECISION_FLAGS.values() for spelling in spellings if len(spelling) == 2
)

_OVERRIDE_SETTINGS = frozenset(WEIGHT_PRECISION_OVERRIDE_FLAGS.values())

#: The options of the frameworks' servers that are a strict prefix of a spelling here: argparse reads each as itself,
#: never as a spelling it begins. SGLang's diffusion server (``sglang serve`` of a diffusion model) takes ``--mode`` for
#: its ``--performance-mode``, which begins ``--model``. On their main branches of 2026-11-07 no other long option of
#: vLLM's ``serve`` or of SGLang's servers begins a spelling here without being one.
_OWN_OPTIONS = frozenset({"--mode"})

_NOT_SET = "(not set)"
_ON = "(on)"
_EMPTY = "''"


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
    """The weight-precision settings ``launch`` makes, by name: every value its flags give the setting; an override
    flag's value as what it sets under the keys, and none when it sets none of them."""
    found: dict[str, set[tuple[str, ...]]] = {}
    removed = [(_flag(flag)[0], (REMOVED,)) for flag in launch.remove_args]
    for flag, values in [*_flags(launch.server_args), *removed]:
        for name in _names(flag):
            if name not in _OVERRIDE_SETTINGS:
                found.setdefault(name, set()).add(values)
            elif keys := _weight_keys(values):
                found.setdefault(name, set()).add(keys)
    return {name: frozenset(values) for name, values in found.items()}


def _flags(server_args: str) -> list[tuple[str, tuple[str, ...]]]:
    """``server_args`` as (flag, values) pairs, in the order given; the words before the first flag are left out."""
    parsed = tokenize_server_args_preserving_json(server_args)
    words = parsed[1] if parsed is not None else server_args.split()
    pairs: list[tuple[str, list[str]]] = []
    for word in words:
        if word.startswith("-") and len(word) > 1 and not _is_number(word):
            pairs.append(_flag(word))
        elif pairs:
            pairs[-1][1].append(word)
    return [(flag, tuple(_canonical(value) for value in values)) for flag, values in pairs]


def _flag(word: str) -> tuple[str, list[str]]:
    """The option a flag word names, as the frameworks' parsers read it, with the values the word carries itself.

    argparse reads a value attached to a one-letter option as its value (``-qfp8`` and ``-q=fp8`` are ``-q fp8``).
    vLLM reads a long flag's name with ``_`` as ``-`` up to its first ``.``, and merges ``--flag.key value`` into
    ``--flag``'s JSON value: that is a value of ``--flag`` here, led by its key (``.key``), so a key added, removed or
    given another value changes ``--flag``.
    """
    if word[:2] in _SHORT_SPELLINGS and len(word) > 2:
        return word[:2], [word[2:].removeprefix("=")]
    name, assigned, value = word.partition("=")
    values = [value] if assigned else []
    if name.startswith("--"):
        name, dotted, key = name.partition(".")
        name = name.replace("_", "-")
        if dotted:
            values.insert(0, f".{key}")
    return name, values


#: Each setting by its spellings as :func:`_flag` reads them, the override flags' included.
_SETTING_OF_SPELLING = {
    **{_flag(spelling)[0]: name for name, spellings in WEIGHT_PRECISION_FLAGS.items() for spelling in spellings},
    **WEIGHT_PRECISION_OVERRIDE_FLAGS,
}


def _names(flag: str) -> set[str]:
    """The settings ``flag`` makes: the one whose spelling it is, else, for a long flag, each setting it begins a
    spelling of (argparse takes a prefix of one option for that option and refuses one that several options begin, so
    which of them the prefix names depends on the options the framework has). An option that a spelling only begins
    (``--config-format``, ``--model-impl``) is none of them, nor is one that only begins a spelling (``--mode``)."""
    if flag in _SETTING_OF_SPELLING:
        return {_SETTING_OF_SPELLING[flag]}
    if not flag.startswith("--") or flag == "--" or flag in _OWN_OPTIONS:
        return set()
    return {name for spelling, name in _SETTING_OF_SPELLING.items() if spelling.startswith(flag)}


def _weight_keys(values: tuple[str, ...]) -> tuple[str, ...]:
    """What an override flag's ``values`` set under :data:`WEIGHT_PRECISION_KEYS`: ``"path=value"`` for each place in
    the value, at any depth, whose key is one of them (the value under it in canonical JSON, as a whole), sorted; ``()``
    when it sets none of them. Two launches set the same thing under the keys when they hold the same values at the
    same paths, whatever else their values set.

    ``--flag.a.b value`` is the value vLLM merges it into, ``{"a": {"b": value}}`` (``value`` read as JSON, else as its
    text; a key ending in ``+`` appends the value's comma-separated items), so a dotted path that ends at a key or
    passes through one sets it. A value that is not JSON counts, by its text, when the text names a key: what it sets
    cannot be read. A removal (:data:`~hyperloom.contracts.recipe.REMOVED`) is a value of its own.
    """
    if values == (REMOVED,):
        return values
    path: list[str] = []
    words = values
    if values and values[0].startswith("."):
        path, words = values[0][1:].split("."), values[1:]
    text = " ".join(words)
    value: object
    if path and path[-1].endswith("+"):
        path[-1] = path[-1].removesuffix("+")
        value, parsed = text.split(","), True
    else:
        try:
            value, parsed = json.loads(text), True
        except ValueError:
            value, parsed = text, False
    for key in reversed(path):
        value = {key: value}
    found: list[str] = []
    _find_keys(value, "", found)
    if not parsed and any(key in text for key in WEIGHT_PRECISION_KEYS):
        found.append(f"(unparsed) {' '.join(values)}")
    return tuple(sorted(found))


def _find_keys(node: object, path: str, found: list[str]) -> None:
    """Append to ``found`` each place under ``node`` whose key is a weight-precision key, with its value; below such a
    key nothing more is looked for, since its value counts whole."""
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key in WEIGHT_PRECISION_KEYS:
                found.append(f"{here}={json.dumps(value, sort_keys=True, separators=(',', ':'))}")
            else:
                _find_keys(value, here, found)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _find_keys(value, f"{path}[{index}]", found)


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
    return " | ".join(sorted(" ".join(word or _EMPTY for word in value) or _ON for value in values))


__all__ = [
    "WEIGHT_PRECISION_FLAGS",
    "WEIGHT_PRECISION_KEYS",
    "WEIGHT_PRECISION_OVERRIDE_FLAGS",
    "weight_precision_changes",
]
