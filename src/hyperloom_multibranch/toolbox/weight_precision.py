# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The model's weight precision is the session's: what a launch's flags say about it, and how two launches differ.

A change of the weights' precision, down or up, changes what is being optimized: an FP8-weight launch of a BF16 session
measures another model, whatever it gains. The precision of intermediates (the KV cache, attention compute, collectives,
the activations of weights the checkpoint already holds in FP8) may change, with the weights as they are and accuracy
passing.

The flags read here set the weights' format, their quantization at load time, their dtype, the checkpoint itself or its
revision, or name a file that can set any of them (a config file of flags, or the model's own config), or how the
checkpoint is loaded (its load format, of which a change only among :data:`SPEED_ONLY_LOAD_FORMATS` is no change, and
the loader's extra config, whose change is none where both launches load in one of them), or ModelOpt's checkpoint
paths; and two flags with other uses, whose JSON value overrides the model's config, count in what they set under a
dtype or quantization key. Only launch flags are read: a source patch that casts weights at load is not, nor what a
file holds.

A launch word is first read as the frameworks' argument parsers read it (:func:`_flag`): ``--flag=value`` is ``--flag
value``; a long flag's name reads ``_`` as ``-`` and ends at its first ``.``, and ``--flag.key value`` is a value of
``--flag`` that keeps its key (vLLM gives it as a JSON value of ``--flag``), whose value is the next word whatever it is
(:func:`_flags`); ``-qVALUE`` is ``-q VALUE``; and a long flag that is no spelling of a setting but begins one or more
(``--dty``) is each setting it begins, as argparse takes a prefix of an option for the option, unless the word is an
option of the frameworks' servers itself (:data:`_OWN_OPTIONS`).

The comparison is then Arbor's (``arbor/measure/precision.py``): a flag's values are the words up to the next flag, and
a word before any flag is a positional, not compared; a word that reads as a number (``-1``) is a value, not a flag; a
JSON object or array compares in canonical form (sorted keys, no spaces) and any other value as its text; a flag given
without a value reads as on; a flag added, removed or given another value is a change. The spellings of one setting are
one flag, and a value it is given twice counts once, under one spelling or two. An override flag's values are what its
occurrences set under the keys (:func:`_override_setting`): a launch none of whose occurrences sets a key does not make
the setting, and in one where some occurrence does, an occurrence that sets none is a value too. A flag the launch
removes from the resolved launch (``LaunchConfig.remove_args``) has the value
:data:`~hyperloom.contracts.recipe.REMOVED`.

Beyond Arbor's, a setting a launch gives two or more distinct values also compares by the value that takes effect
(:func:`_in_order`): the server keeps one, so two launches that give the same values in another order, or that drop
one occurrence and give its value again, can serve another precision.

vLLM reads a word ``-O<rest>`` other than ``-O`` itself as ``--optimization-level <rest>`` before anything else, and so
does the comparison. A word that begins with ``-`` and holds a ``.`` but no ``=`` (``--flag.key``, ``-0.5``) followed
by a word that begins with ``-`` is read in two ways: vLLM takes it as a dotted flag whose value is that next word,
argparse in the other frameworks reads that next word as a flag. The comparison reads such a pair as before (a dotted
flag takes the next word, a number is a value), and :func:`ambiguous_words` names each one a launch adds or changes
(``-O3.0`` among them), which the toolbox refuses instead of reading it either way.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import islice
from typing import NamedTuple

from hyperloom.contracts.recipe import REMOVED, LaunchConfig
from hyperloom.inference_optimizer.grid_server_args import tokenize_server_args_preserving_json

#: The launch flags that set the served weights' precision, by the name a change is reported under, each with its
#: spellings: vLLM's and SGLang's quantization method, its config and its parameter file, ATOM's online quantization,
#: the weights' dtype (``--dtype`` in both frameworks), the checkpoint (vLLM's ``--model``, SGLang's ``--model-path``)
#: and its revision, since another checkpoint may hold another precision, the files that can set any of them (none is
#: read here: vLLM's and SGLang's ``--config`` of flags, and the model config vLLM's ``--hf-config-path`` and SGLang's
#: ``--decrypted-config-file`` read instead of the checkpoint's), ModelOpt's and torchao's quantization, and SGLang's
#: quantize-and-serve, ModelOpt's checkpoint paths (a restore takes the quantizer state from its file, and any of them
#: sends SGLang's online ``modelopt_fp4`` to ModelOpt's calibrating loader), and the load format and its loader's extra
#: config (:data:`SPEED_ONLY_LOAD_FORMATS`). A launch word counts as one of these spellings when the frameworks'
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
    "--hf-config-path": ("--hf-config-path",),
    "--decrypted-config-file": ("--decrypted-config-file",),
    "--modelopt-checkpoint-restore-path": ("--modelopt-checkpoint-restore-path",),
    "--modelopt-checkpoint-save-path": ("--modelopt-checkpoint-save-path",),
    "--modelopt-export-path": ("--modelopt-export-path",),
    "--load-format": ("--load-format",),
    "--model-loader-extra-config": ("--model-loader-extra-config",),
}

#: The values of vLLM's and SGLang's ``--load-format`` (lower-cased, as both read it; absent is ``auto``) among which a
#: change is none: on both main branches of 2026-11-07 each is the default loader's, which copies the checkpoint's own
#: tensors as stored and differs only in the files it picks (``*.safetensors`` else ``*.bin``, ``*.safetensors``,
#: ``*.pt``, ``*.bin`` through numpy). Any other value may quantize what it loads (SGLang's ``flash_rl`` serves a BF16
#: checkpoint in FP8) or load no checkpoint (``dummy``), so a change to, from or between them is a change;
#: ``runai_streamer`` among them, since SGLang does not route it to ModelOpt's loader, which quantizes the weights.
SPEED_ONLY_LOAD_FORMATS = frozenset({"auto", "pt", "safetensors", "npcache", "fastsafetensors"})

_LOAD_FORMAT = "--load-format"

#: The settings whose change is none when both launches load in :data:`SPEED_ONLY_LOAD_FORMATS`. Their default loaders
#: refuse an extra config with keys other than their threading ones, so there a change of the extra config is none;
#: under any other format a key can set what is quantized (SGLang's ``bitsandbytes`` reads
#: ``qlora_adapter_name_or_path``), and a change of it is a change.
_SPEED_ONLY_WHERE_LOADED_SO = frozenset({_LOAD_FORMAT, "--model-loader-extra-config"})

#: The keys of a model's config that set its weights' dtype or quantization: vLLM and SGLang read
#: ``compression_config`` (compressed-tensors' key) as the quantization config when ``quantization_config`` is absent.
WEIGHT_PRECISION_KEYS = frozenset({"torch_dtype", "dtype", "quantization_config", "compression_config"})

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

#: The option vLLM reads a word ``-O<rest>`` as, ``<rest>`` its value.
_OPTIMIZATION_LEVEL = "--optimization-level"

_NOT_SET = "(not set)"
_ON = "(on)"
_EMPTY = "''"
#: The value of an override flag's occurrence that sets none of :data:`WEIGHT_PRECISION_KEYS`.
_NO_KEY = ("(no key)",)


class _Occurrence(NamedTuple):
    """A flag of a launch as :func:`_flag` reads it, with the values it gives; ``dotted`` when it is vLLM's
    ``--flag.key value`` (or ``-f.key value``), its key (``.key``) leading the values."""

    flag: str
    values: tuple[str, ...]
    dotted: bool = False


@dataclass(frozen=True)
class _Setting:
    """What a launch gives one weight-precision setting: each distinct value of its occurrences and, when there are two
    or more, the one that takes effect (:func:`_in_order`), which another order of the same values can change."""

    values: frozenset[tuple[str, ...]]
    in_effect: tuple[str, ...] | None


def weight_precision_changes(before: LaunchConfig, after: LaunchConfig) -> list[str]:
    """Each weight-precision setting ``after`` changes from ``before``, as ``"flag: old -> new"`` (``(not set)`` for
    a setting a launch does not make); ``[]`` when the weights' precision is as it was."""
    old, new = _settings(before), _settings(after)
    return [
        f"{name}: {_shown(old.get(name))} -> {_shown(new.get(name))}"
        for name in sorted({*old, *new})
        if old.get(name) != new.get(name) and not (name in _SPEED_ONLY_WHERE_LOADED_SO and _speed_only(old, new))
    ]


def _speed_only(*launches: Mapping[str, _Setting]) -> bool:
    """Whether each of ``launches`` gives its load format only values of :data:`SPEED_ONLY_LOAD_FORMATS`, or none."""
    return all(
        len(value) == 1 and value[0].lower() in SPEED_ONLY_LOAD_FORMATS
        for launch in launches
        if _LOAD_FORMAT in launch
        for value in launch[_LOAD_FORMAT].values
    )


def ambiguous_words(before: LaunchConfig, after: LaunchConfig) -> list[str]:
    """Each pair of words of ``after``'s launch that the frameworks read in two ways and that ``before``'s does not give
    as often, as ``"word next"``, in the order ``after`` gives them; ``[]`` when ``after`` adds or changes none.

    A pair is a word that begins with ``-`` and holds a ``.`` but no ``=`` (a dotted flag written without its value,
    ``-O3.0``, a number such as ``-0.5``) and the word after it, which begins with ``-``. vLLM takes a dotted flag's
    value to be the next word, whatever it is (``--hf-overrides.foo -qfp8`` sets ``foo`` to ``-qfp8``), and so a
    number's too, where argparse in the other frameworks reads the number as a value and the next word as a flag. A
    pair counts as a whole and each time it is given: ``after`` adds one where it gives a word of ``before`` before
    another word, or gives a pair of ``before`` once more.
    """
    added = Counter(_ambiguous_pairs(after.server_args)) - Counter(_ambiguous_pairs(before.server_args))
    return [" ".join(pair) for pair in added]


def _ambiguous_pairs(server_args: str) -> Iterator[tuple[str, str]]:
    words = _words(server_args)
    for word, after in zip(words, words[1:]):
        if word.startswith("-") and "." in word and "=" not in word and after.startswith("-"):
            yield word, after


def _settings(launch: LaunchConfig) -> dict[str, _Setting]:
    """The weight-precision settings ``launch`` makes, by name, each read from its occurrences in the order the server
    is given them: a removal from the resolved launch first, since the launch's own words come after the resolved
    launch's, then the launch's own; an override flag's as :func:`_override_setting` reads them."""
    occurrences: dict[str, list[_Occurrence]] = {}
    removed = [_Occurrence(_flag(flag).flag, (REMOVED,)) for flag in launch.remove_args]
    for occurrence in [*removed, *_flags(launch.server_args)]:
        for name in _names(occurrence.flag):
            occurrences.setdefault(name, []).append(occurrence)
    settings = {
        name: _override_setting(found) if name in _OVERRIDE_SETTINGS else _flag_setting(found)
        for name, found in occurrences.items()
    }
    return {name: setting for name, setting in settings.items() if setting is not None}


def _flags(server_args: str) -> list[_Occurrence]:
    """``server_args`` as its flags with their values, in the order given, the dotted ones after the others; the words
    before the first flag are left out.

    vLLM takes each dotted flag word (a flag with a ``.`` before any ``=``) out of the launch before argparse reads the
    rest, and one written without ``=`` takes the next word with it as its value, whatever that word is:
    ``--hf-overrides.foo -qfp8`` sets ``foo`` to ``-qfp8`` and quantizes nothing. A word after that value is read with
    the flag before the dotted one, as argparse reads what is left.
    """
    words = _vllm_words(_words(server_args))
    found: list[tuple[_Occurrence, list[str]]] = []
    dotted: list[_Occurrence] = []
    for word in words:
        if not (word.startswith("-") and len(word) > 1 and not _is_number(word)):
            if found:
                found[-1][1].append(word)
            continue
        occurrence = _flag(word)
        if not occurrence.dotted:
            found.append((occurrence, []))
            continue
        taken = () if "=" in word else tuple(islice(words, 1))
        dotted.append(occurrence._replace(values=(*occurrence.values, *taken)))
    plain = [occurrence._replace(values=(*occurrence.values, *more)) for occurrence, more in found]
    return [occurrence._replace(values=tuple(map(_canonical, occurrence.values))) for occurrence in [*plain, *dotted]]


def _words(server_args: str) -> list[str]:
    """``server_args`` as its words, a JSON value one word."""
    parsed = tokenize_server_args_preserving_json(server_args)
    return parsed[1] if parsed is not None else server_args.split()


def _vllm_words(words: list[str]) -> Iterator[str]:
    """``words`` with each ``-O<rest>`` other than ``-O`` itself as ``--optimization-level <rest>`` (``-O=<rest>``
    too), as vLLM reads them before anything else."""
    for word in words:
        if word.startswith("-O") and word != "-O":
            yield _OPTIMIZATION_LEVEL
            yield word[3:] if word[2] == "=" else word[2:]
        else:
            yield word


def _flag(word: str) -> _Occurrence:
    """The option a flag word names, as the frameworks' parsers read it, with the values the word carries itself.

    argparse reads a value attached to a one-letter option as its value (``-qfp8`` and ``-q=fp8`` are ``-q fp8``).
    vLLM reads a long flag's name with ``_`` as ``-`` up to its first ``.``, and gives ``--flag.key value`` to
    ``--flag`` as the JSON value ``{"key": value}``, merged with the flag's other dotted occurrences: that is a value of
    ``--flag`` here, led by its key (``.key``), so a key added, removed or given another value changes ``--flag``. It
    reads a one-dash word with a ``.`` before any ``=`` so too (``-cc.custom_ops+ x``), its option the word's name up
    to the ``.`` as argparse reads that.
    """
    name, assigned, value = word.partition("=")
    values = (value,) if assigned else ()
    if "." in name:
        name, _, key = name.partition(".")
        return _Occurrence(_flag(name).flag, (f".{key}", *values), dotted=True)
    if word[:2] in _SHORT_SPELLINGS and len(word) > 2:
        return _Occurrence(word[:2], (word[2:].removeprefix("="),))
    return _Occurrence(name.replace("_", "-") if name.startswith("--") else name, values)


#: Each setting by its spellings as :func:`_flag` reads them, the override flags' included.
_SETTING_OF_SPELLING = {
    **{_flag(spelling).flag: name for name, spellings in WEIGHT_PRECISION_FLAGS.items() for spelling in spellings},
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


def _flag_setting(occurrences: list[_Occurrence]) -> _Setting:
    """The setting a flag's ``occurrences`` in one launch make: the values each gives, a dotted one's led by its key,
    and when they differ the one in effect, a spelling's dotted ones as the JSON value vLLM merges them into."""
    values = frozenset(occurrence.values for occurrence in occurrences)
    if len(values) < 2:
        return _Setting(values, None)
    last = _in_order(occurrences)[-1]
    return _Setting(values, (_canonical_json(last),) if isinstance(last, dict) else last)


def _override_setting(occurrences: list[_Occurrence]) -> _Setting | None:
    """The setting an override flag's ``occurrences`` in one launch make: what each sets under
    :data:`WEIGHT_PRECISION_KEYS`, vLLM's dotted ones of one spelling as the one JSON value it merges them into, and
    when these differ the one in effect; ``None`` when no occurrence sets a key.

    argparse keeps the last occurrence's value, and vLLM gives its merged dotted ones after all the others, so an
    occurrence that sets no key can drop what another sets: ``--hf-overrides '{"torch_dtype": "bfloat16"}'
    --hf_overrides '{"max_position_embeddings": 4096}'`` serves no dtype override. When some occurrence sets a key, one
    that sets none is a value too (:data:`_NO_KEY`), whichever the server keeps. A removal
    (:data:`~hyperloom.contracts.recipe.REMOVED`) is a value of its own.
    """
    in_order = [
        _set_under_keys(value) if isinstance(value, dict) else _plain_value(value) for value in _in_order(occurrences)
    ]
    values = frozenset(in_order)
    if not values - {_NO_KEY}:
        return None
    return _Setting(values, in_order[-1] if len(values) > 1 else None)


def _in_order(occurrences: list[_Occurrence]) -> list[tuple[str, ...] | dict[str, object]]:
    """The values ``occurrences`` give one setting, in the order argparse reads them, whose last takes effect: each
    occurrence that is not dotted in turn, then each spelling's dotted ones as the one JSON value vLLM merges them
    into (:func:`_merge`), in the order the spellings first come, as vLLM gives them after all the others."""
    plain: list[tuple[str, ...] | dict[str, object]] = []
    dotted: dict[str, dict[str, object]] = {}
    for occurrence in occurrences:
        if occurrence.dotted:
            _merge(dotted.setdefault(occurrence.flag, {}), _dotted_value(occurrence.values))
        else:
            plain.append(occurrence.values)
    return [*plain, *dotted.values()]


def _plain_value(words: tuple[str, ...]) -> tuple[str, ...]:
    """What ``--flag value`` sets under the keys; a value that is not JSON counts by its text when the text names a key,
    since what it sets cannot be read. A removal is a value of its own."""
    if words == (REMOVED,):
        return words
    text = " ".join(words)
    try:
        value = json.loads(text)
    except ValueError:
        return (f"(unparsed) {text}",) if any(key in text for key in WEIGHT_PRECISION_KEYS) else _NO_KEY
    return _set_under_keys(value)


def _dotted_value(words: tuple[str, ...]) -> dict[str, object]:
    """The JSON value vLLM makes of ``--flag.a.b value``, ``words`` being ``(".a.b", "value")``:
    ``{"a": {"b": value}}``, with ``value`` read as JSON, else as its text, and a last key ending in ``+`` taking its
    comma-separated items as a list. A dotted path that ends at a key or passes through one sets it."""
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
                found.append(f"{here}={_canonical_json(value)}")
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
            return _canonical_json(json.loads(stripped))
        except ValueError:
            return value
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _is_number(word: str) -> bool:
    try:
        float(word)
    except ValueError:
        return False
    return True


def _shown(setting: _Setting | None) -> str:
    if setting is None:
        return _NOT_SET
    shown = " | ".join(sorted(_shown_value(value) for value in setting.values))
    return shown if setting.in_effect is None else f"{shown} ({_shown_value(setting.in_effect)} in effect)"


def _shown_value(value: tuple[str, ...]) -> str:
    return " ".join(word or _EMPTY for word in value) or _ON


__all__ = [
    "SPEED_ONLY_LOAD_FORMATS",
    "WEIGHT_PRECISION_FLAGS",
    "WEIGHT_PRECISION_KEYS",
    "WEIGHT_PRECISION_OVERRIDE_FLAGS",
    "ambiguous_words",
    "weight_precision_changes",
]
