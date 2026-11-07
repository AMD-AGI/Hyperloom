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
``--flag`` that keeps its key (vLLM gives it as a JSON value of ``--flag``); ``-qVALUE`` is ``-q VALUE``; and a long flag that
is no spelling of a setting but begins one or more (``--dty``) is each setting it begins, as argparse takes a prefix of
an option for the option, unless the word is an option of the frameworks' servers itself (:data:`_OWN_OPTIONS`).

The comparison is then Arbor's (``arbor/measure/precision.py``): a flag's values are the words up to the next flag, and
a word before any flag is a positional, not compared; a word that reads as a number (``-1``) is a value, not a flag; a
JSON object or array compares in canonical form (sorted keys, no spaces) and any other value as its text; a flag given
without a value reads as on; a flag added, removed or given another value is a change. The spellings of one setting are
one flag, and a value it is given twice counts once, under one spelling or two. An override flag's values are what its
occurrences set under the keys (:func:`_override_values`): a launch none of whose occurrences sets a key does not make
the setting, and in one where some occurrence does, an occurrence that sets none is a value too, since the parser keeps
one occurrence. A flag the launch removes from the resolved launch (``LaunchConfig.remove_args``) has the value
:data:`~hyperloom.contracts.recipe.REMOVED`.
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
#: The value of an override flag's occurrence that sets none of :data:`WEIGHT_PRECISION_KEYS`.
_NO_KEY = ("(no key)",)


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
    """The weight-precision settings ``launch`` makes, by name: every value its flags give the setting, an override
    flag's as :func:`_override_values` reads its occurrences."""
    found: dict[str, set[tuple[str, ...]]] = {}
    overrides: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    removed = [(_flag(flag)[0], (REMOVED,)) for flag in launch.remove_args]
    for flag, values in [*_flags(launch.server_args), *removed]:
        for name in _names(flag):
            if name in _OVERRIDE_SETTINGS:
                overrides.setdefault(name, []).append((flag, values))
            else:
                found.setdefault(name, set()).add(values)
    settings = {name: frozenset(values) for name, values in found.items()}
    for name, occurrences in overrides.items():
        if override := _override_values(occurrences):
            settings[name] = override
    return settings


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
    vLLM reads a long flag's name with ``_`` as ``-`` up to its first ``.``, and gives ``--flag.key value`` to
    ``--flag`` as the JSON value ``{"key": value}``, merged with the flag's other dotted occurrences: that is a value of
    ``--flag`` here, led by its key (``.key``), so a key added, removed or given another value changes ``--flag``.
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


def _override_values(occurrences: list[tuple[str, tuple[str, ...]]]) -> frozenset[tuple[str, ...]]:
    """The values of an override flag that ``occurrences``, its (flag, values) pairs in one launch, give it: what each
    occurrence sets under :data:`WEIGHT_PRECISION_KEYS`, vLLM's dotted ones of one spelling as the one JSON value it
    merges them into; none when no occurrence sets a key.

    argparse keeps the last occurrence's value, and vLLM gives its merged dotted ones after all the others, so an
    occurrence that sets no key can drop what another sets: ``--hf-overrides '{"torch_dtype": "bfloat16"}'
    --hf_overrides '{"max_position_embeddings": 4096}'`` serves no dtype override. Which occurrence a framework keeps is
    not read here: when some occurrence sets a key, one that sets none is a value too (:data:`_NO_KEY`). A removal
    (:data:`~hyperloom.contracts.recipe.REMOVED`) is a value of its own.
    """
    values: set[tuple[str, ...]] = set()
    dotted: dict[str, dict[str, object]] = {}
    for flag, words in occurrences:
        if words == (REMOVED,):
            values.add(words)
        elif words and words[0].startswith("."):
            _merge(dotted.setdefault(flag, {}), _dotted_value(words))
        else:
            values.add(_plain_value(words))
    values.update(_set_under_keys(value) for value in dotted.values())
    return frozenset(values) if values - {_NO_KEY} else frozenset()


def _plain_value(words: tuple[str, ...]) -> tuple[str, ...]:
    """What ``--flag value`` sets under the keys; a value that is not JSON counts by its text when the text names a key,
    since what it sets cannot be read."""
    text = " ".join(words)
    try:
        value = json.loads(text)
    except ValueError:
        return (f"(unparsed) {text}",) if any(key in text for key in WEIGHT_PRECISION_KEYS) else _NO_KEY
    return _set_under_keys(value)


def _dotted_value(words: tuple[str, ...]) -> dict[str, object]:
    """The JSON value vLLM makes of ``--flag.a.b value``, ``words`` being ``(".a.b", "value")``: ``{"a": {"b": value}}``,
    with ``value`` read as JSON, else as its text, and a last key ending in ``+`` taking its comma-separated items as a
    list. A dotted path that ends at a key or passes through one sets it."""
    *path, last = words[0][1:].split(".")
    text = " ".join(words[1:])
    value: object
    if last.endswith("+"):
        last, value = last.removesuffix("+"), text.split(",")
    else:
        try:
            value = json.loads(text)
        except ValueError:
            value = text
    nested: dict[str, object] = {last: value}
    for key in reversed(path):
        nested = {key: nested}
    return nested


def _merge(into: dict[str, object], value: dict[str, object]) -> None:
    """Merge ``value`` into ``into`` as vLLM merges dotted occurrences: an object key by key, a list appended to a list,
    anything else replacing what was there."""
    for key, item in value.items():
        current = into.get(key)
        if isinstance(item, dict) and isinstance(current, dict):
            _merge(current, item)
        elif isinstance(item, list) and isinstance(current, list):
            into[key] = [*current, *item]
        else:
            into[key] = item


def _set_under_keys(value: object) -> tuple[str, ...]:
    """``"path=value"`` for each place in ``value``, at any depth, whose key is one of :data:`WEIGHT_PRECISION_KEYS`
    (the value under it in canonical JSON, as a whole), sorted; :data:`_NO_KEY` when there is none. Two values set the
    same thing under the keys when they hold the same values at the same paths, whatever else they set."""
    found: list[str] = []
    _find_keys(value, "", found)
    return tuple(sorted(found)) or _NO_KEY


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
