# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Server-argument composition helpers.

Shared by the explore / integrate_patch / grid executors and the workload-env
builder: merge/remove/replace semantics for ``EXTRA_*_ARGS``, last-wins dedupe
for vLLM/atom single-value flags, shell-safety validation, JSON-valued flag
compaction/repair, the sglang watchdog / context-length / attention-backend /
MoE-runner injections, and ``apply_runtime_benchmark_overrides`` for the
materialized Magpie YAML. Nothing here runs a benchmark — the run/parse/
winner-selection loop lives in :mod:`._grid_runner`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
from typing import Any

from hyperloom.common.coerce import optional_positive_int, to_str_list

# Single definition of "is this an option name rather than a value". Two copies
# drifted apart once already; the fingerprint module is a leaf, so importing it
# here adds no cycle.
from ._canonical_fingerprint import _is_flag as _is_flag_token


log = logging.getLogger(__name__)

_UNSAFE_SERVER_ARG_CHARS_RE = re.compile(r"[;&|`$<>\r\n]")


def validate_server_args_shell_safe(server_args: str | None) -> str:
    """Reject server-arg strings that would be shell control syntax.

    Magpie benchmark scripts expand ``EXTRA_*_ARGS`` through shell wrappers, so
    this is the final sink-side guard against LLM/payload content escaping from
    argv-like flags into shell control operators.
    """
    args = str(server_args or "").strip()
    if not args:
        return ""
    if _UNSAFE_SERVER_ARG_CHARS_RE.search(args):
        raise ValueError("extra_server_args contains shell control characters")
    try:
        tokens = shlex.split(args)
    except ValueError as exc:
        raise ValueError(f"extra_server_args is not shell-tokenizable: {exc}") from exc
    expect_value = False
    for token in tokens:
        if token.startswith("-"):
            expect_value = "=" not in token
            continue
        if expect_value:
            expect_value = False
            continue
        raise ValueError("extra_server_args must be argv-like flags, not bare positional arguments")
    return args


def server_args_env_name(framework: str | None) -> str:
    """Return the Magpie env var used to append backend server args.

    Resolution is exact (registry-keyed) with a substring fallback so a
    framework string carrying a version suffix (e.g. ``"vllm@0.21"``) still
    maps correctly. Unknown names fall back to the default framework's env.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        str: The ``EXTRA_*_ARGS`` env name for the framework (e.g.
        ``"EXTRA_XDIT_ARGS"`` for xDiT, ``"EXTRA_SGLANG_ARGS"`` default).
    """
    from hyperloom.inference_optimizer import framework_registry

    name = str(framework or "").strip().lower()
    if framework_registry.is_supported(name):
        return framework_registry.extra_args_env(name)
    # Substring fallback for version-suffixed names.
    for fw in framework_registry.names():
        if fw in name:
            return framework_registry.extra_args_env(fw)
    return framework_registry.extra_args_env(framework_registry.DEFAULT_FRAMEWORK)


def merge_server_args(*parts: str | None) -> str:
    """Merge server arg strings preserving left-to-right override semantics.

    Only removes empty chunks; does NOT de-duplicate option names, because
    repeated flags are how later args override base args (e.g. ``--block-size
    1`` then ``--block-size 256``).

    Args:
        *parts (str | None): Server-arg chunks to merge, in override order;
            empty/``None`` chunks are dropped.

    Returns:
        str: The space-joined non-empty chunks.
    """
    return " ".join(str(p).strip() for p in parts if str(p or "").strip())


_WHITESPACE_RUN_RE = re.compile(r"(\s+)")


def _split_keeping_separators(text: str) -> list[str]:
    """Split into alternating word / whitespace pieces; ``"".join`` is exact.

    Whitespace is *the* argv separator on this path and nothing else is. Magpie
    expands ``EXTRA_*_ARGS`` through a shell wrapper without quoting it, so the
    shell word-splits on whitespace and performs no quote removal: a quote byte
    reaches the server as part of the value. Splitting the same way is therefore
    not an approximation of the sink, it IS the sink — which is why ``shlex``,
    POSIX or not, was the wrong model here. It disagreed with the transport
    about where the tokens are, and every disagreement showed up as a removal
    that deleted the wrong bytes.

    Keeping the separator pieces is what makes reassembly byte-exact, so an
    untouched neighbour is returned with its own whitespace — including the
    interior of a JSON string value, which a ``" ".join`` silently collapsed
    (``{"chat_template":"a  b"}`` reached the server as ``"a b"``).
    """
    return [piece for piece in _WHITESPACE_RUN_RE.split(text) if piece]


def _words_of(text: str) -> list[str]:
    """The non-whitespace pieces of ``text`` after JSON normalization."""
    return [p for p in _split_keeping_separators(_reserialize_json_blobs(str(text or "").strip())) if not p.isspace()]


def _logical_value(word: str) -> str:
    """The bytes a comparison should use, with one matched quote pair removed.

    Removal specs are written by hand and by an LLM, and both quote habitually:
    ``--foo "bar"`` and ``--foo bar`` name the same configuration. Comparing raw
    bytes made the quoted spelling a silent no-op on either side. Only a matched
    outer pair is stripped, so a stray edge quote (``'my`` from a
    whitespace-bearing quoted value) keeps its bytes and still fails to match
    anything but itself.
    """
    if len(word) >= 2 and word[0] == word[-1] and word[0] in "\"'":
        return word[1:-1]
    return word


def _plain_span_end(words: list[str], start: int) -> int:
    """First index at or after ``start`` that names an option."""
    end = start
    while end < len(words) and not _is_flag_token(words[end]):
        end += 1
    return end


def _span_end(words: list[str], flag_index: int) -> int:
    """Index just past the operands of the option at ``flag_index``.

    An option owns every following word up to the next option name, with one
    exception. A JSON value split by its own interior whitespace can produce a
    fragment spelled exactly like an option: ``{"t":"Answer --now please"}``
    splits at ``--now``. Ending the span there left the rest of the blob behind,
    so removing the flag handed ``--now please"}`` to the launcher and
    :func:`validate_server_args_shell_safe` refused the whole string — one the
    previous shlex-based removal took out cleanly.

    So the span may run past an option name while the JSON scan is unbalanced,
    and only as far as the word that balances it. When nothing balances it the
    plain scan is what stands: an operand carrying a stray ``}`` (``--foo a}``)
    must not let the span run to the end of the string, which is how removing
    one flag once deleted the entire tail of the configuration.

    Fragments are undeliverable either way — :func:`_transport_unsafe_tokens`
    reports them — but that is a statement about the value, not a licence to
    rewrite the flags around it.
    """
    plain_end = _plain_span_end(words, flag_index + 1)
    depth, in_string, escaped = 0, False, False
    for index in range(flag_index + 1, plain_end):
        depth, in_string, escaped = _json_scan(words[index], depth, in_string, escaped)
    if depth == 0 and not in_string:
        return plain_end
    end = plain_end
    while end < len(words):
        depth, in_string, escaped = _json_scan(words[end], depth, in_string, escaped)
        end += 1
        if depth == 0 and not in_string:
            return _plain_span_end(words, end)
    return plain_end


def _words_inside_json(words: list[str]) -> set[int]:
    """Indices of words that sit inside an unterminated JSON blob.

    A word is only an option name when it is argv. The interior whitespace that
    fragments a JSON value can leave a fragment spelled like an option —
    including one on :data:`_BENCHMARK_HARNESS_FLAG_DENYLIST` — and matching it
    there cut the blob in half on the path every compose reaches. A word the
    scan meets while a brace or a string is still open is part of a value.

    Symmetric with :func:`_span_end`, including its fallback: a stray closing
    brace (``--foo a}``) drives the running depth negative, and carrying that
    forward would mark every later word as nested and make the flags after it
    unremovable. A depth below zero is not an open blob, so it clamps to closed.
    """
    inside: set[int] = set()
    depth, in_string, escaped = 0, False, False
    for index, word in enumerate(words):
        if depth > 0 or in_string:
            inside.add(index)
        depth, in_string, escaped = _json_scan(word, depth, in_string, escaped)
        depth = max(depth, 0)
    return inside


# Payloads already reported as undeliverable, so the warning below fires once
# per distinct payload per process. ``remove_server_args`` is reached by every
# compose of every variant of every cycle, so a single legitimate quoted operand
# (``--tool-call-parser 'my parser'``) used to print a multi-line WARNING
# hundreds of times across one session. Keyed on the owning option names plus
# how many tokens were undeliverable — deliberately not on the values, which is
# the whole point of not echoing them. Two payloads that differ only in a value
# under the same option therefore report once; the message names the option so
# an operator can find both.
#
# Insertion-ordered so a full cache evicts its OLDEST entry. Clearing the whole
# cache instead re-armed every option already reported, which on a long session
# is the repetition this exists to stop.
_UNSAFE_TRANSPORT_WARNED: dict[str, None] = {}

_UNSAFE_TRANSPORT_WARNED_CAP = 256


def _unsafe_token_owners(tokens: list[str], unsafe: list[str]) -> list[str]:
    """Option names the undeliverable tokens belong to.

    Reported instead of the tokens themselves. An operand can carry a
    credential — the reason :func:`remove_server_args` does not echo the args
    string — and so can a removal spec, which is written by the same LLM /
    operator path and reaches this module the same way. An option name is not a
    secret and is the part an operator needs in order to find the value.

    A token with no owner is dropped rather than reported as itself. ``unsafe``
    is always a subset of ``tokens``, so this cannot happen today; falling back
    to the token would put the value in the log the moment it could, which is
    the one thing this function exists to prevent.

    A fragment inside a JSON value can be spelled like an option, and taking it
    as one would name the following tokens after a byte of somebody's value —
    the same leak by a different route. :func:`_words_inside_json` is what tells
    the two apart.
    """
    unsafe_set = set(unsafe)
    nested = _words_inside_json(tokens)
    owners: list[str] = []
    current = "<positional>"
    for index, token in enumerate(tokens):
        if index not in nested and _is_flag_token(token):
            current = token.partition("=")[0]
        if token in unsafe_set:
            owners.append(current)
    return sorted(set(owners))


def _warn_undeliverable_tokens(tokens: list[str], unsafe: list[str], remove_count: int) -> None:
    """Report undeliverable tokens once per distinct payload per process."""
    owners = _unsafe_token_owners(tokens, unsafe)
    names = ", ".join(owners)
    key = f"{len(unsafe)}|{'|'.join(owners)}"
    if key in _UNSAFE_TRANSPORT_WARNED:
        log.debug(
            "Server args still carry %d token(s) the unquoted EXTRA_*_ARGS transport "
            "cannot represent, under %s; already reported this process.",
            len(unsafe),
            names,
        )
        return
    while len(_UNSAFE_TRANSPORT_WARNED) >= _UNSAFE_TRANSPORT_WARNED_CAP:
        _UNSAFE_TRANSPORT_WARNED.pop(next(iter(_UNSAFE_TRANSPORT_WARNED)))
    _UNSAFE_TRANSPORT_WARNED[key] = None
    log.warning(
        "Server args carry %d token(s) the unquoted EXTRA_*_ARGS transport cannot "
        "represent, under %s; applying the %d removal spec(s) to the rest and passing "
        "those tokens through verbatim. The value reaches the server with its "
        "quote/whitespace bytes intact, which is almost certainly not what was "
        "intended. Values and removal specs are withheld because both are "
        "author-supplied and can carry credentials. Reported once per payload.",
        len(unsafe),
        names,
        remove_count,
    )


def _warn_operand_list_widened(flag: str, operand_count: int) -> None:
    """Report a removal that took more operands than the spec named."""
    log.warning(
        "Removal spec names one operand of %s but the served args give it "
        "%d; removing the whole operand list, since a flag's operands "
        "cannot be split without leaving bare argv words. Attribute any "
        "resulting gain or regression to all of them, not just the one "
        "named.",
        flag,
        operand_count,
    )


def _parse_remove_specs(removes: list[str]) -> tuple[set[str], set[tuple[str, str]]]:
    """Split removal specs into bare-flag names and (flag, operand) pairs.

    Both keys a pair spec registers are compared on :func:`_logical_value`, so
    the spelling of the quoting on either side does not decide whether a removal
    happens. A multi-operand spec registers its whole operand list AND its first
    operand, which is what lets ``--foo bar`` name ``--foo bar baz``.
    """
    remove_flags: set[str] = set()
    remove_pairs: set[tuple[str, str]] = set()
    for spec in removes:
        words = _words_of(spec)
        nested = _words_inside_json(words)
        i = 0
        while i < len(words):
            word = words[i]
            if i in nested or not _is_flag_token(word):
                i += 1
                continue
            if "=" in word:
                flag, _, value = word.partition("=")
                remove_pairs.add((flag, _logical_value(value)))
                i += 1
                continue
            end = _span_end(words, i)
            if end > i + 1:
                operands = words[i + 1 : end]
                remove_pairs.add((word, " ".join(_logical_value(o) for o in operands)))
                remove_pairs.add((word, _logical_value(operands[0])))
                i = end
            else:
                remove_flags.add(word)
                i += 1
    return remove_flags, remove_pairs


def remove_server_args(server_args: str | None, remove_args: Any) -> str:
    """Remove flag specs from a server-arg string.

    ``remove_args`` entries are flag-oriented. ``"--foo"`` removes ``--foo`` and
    its following operands when any are present; ``"--foo=bar"`` removes that
    token shape, plus any operand that follows it; ``"--foo bar"`` removes
    ``--foo`` when ``bar`` is its first (or only) operand.

    SEMANTIC NOTE — a pair spec removes the WHOLE operand list, not just the
    operand it names: ``remove_args: ["--cuda-graph-bs 1"]`` against
    ``--cuda-graph-bs 1 2 4`` removes all three operands. A flag's operands are
    a unit; deleting ``1`` alone leaves ``2 4`` as bare argv words, which
    :func:`validate_server_args_shell_safe` rejects outright and which the paths
    that skip it hand to the server as positionals. When a spec names one
    operand and the flag carries several, this logs a WARNING rather than
    widening the removal silently, because the gain or regression then belongs
    to a different knob than the one the spec named.

    Matching is quote-insensitive on both sides (:func:`_logical_value`) but
    reassembly is byte-exact for everything retained: words and the whitespace
    between them come back unchanged, so removing one flag cannot rewrite
    another's value — the failure that shipped a mangled
    ``--online_quant_config`` to the server. Both sides are split on whitespace
    only, which is exactly what the unquoted ``EXTRA_*_ARGS`` transport does
    (see :func:`_split_keeping_separators`); there is no separate untokenizable
    path, because there is nothing left for a quote to unbalance. A fragment of
    a JSON value is never read as an option name, however it happens to be
    spelled (:func:`_words_inside_json`), so a value can neither be cut in half
    nor be removed as though it were a flag.

    A word the transport cannot carry is passed through rather than abandoning
    the whole removal, and is reported once per distinct payload per process
    (see :func:`_warn_undeliverable_tokens`; this is a per-compose sink, so an
    unconditional warning repeated itself for a whole session) — this is a
    launch-path sink (``_workload_envs`` writes the result straight into
    ``EXTRA_*_ARGS``) and ``strip_benchmark_harness_flags`` rides on it, so a
    skipped removal serves a benchmark-only flag and misattributes the gain.
    """
    args = str(server_args or "").strip()
    removes = to_str_list(remove_args)
    if not args or not removes:
        return args
    pieces = _split_keeping_separators(_reserialize_json_blobs(args))
    word_at = [i for i, piece in enumerate(pieces) if not piece.isspace()]
    words = [pieces[i] for i in word_at]
    if not words:
        return args
    unsafe = _transport_unsafe_tokens(words)
    if unsafe:
        _warn_undeliverable_tokens(words, unsafe, len(removes))

    nested = _words_inside_json(words)
    remove_flags, remove_pairs = _parse_remove_specs(removes)

    dropped: set[int] = set()

    def drop(first: int, last: int) -> None:
        """Drop words ``[first, last)`` and one flanking separator each."""
        for k in range(first, last):
            at = word_at[k]
            dropped.add(at)
            # Take the separator BEFORE the word so the neighbours it sat
            # between end up adjacent exactly once; at the head of the string
            # there is none, so take the one after instead.
            if at - 1 >= 0 and pieces[at - 1].isspace():
                dropped.add(at - 1)
            elif at + 1 < len(pieces) and pieces[at + 1].isspace():
                dropped.add(at + 1)

    i = 0
    while i < len(words):
        word = words[i]
        if i in nested or not _is_flag_token(word):
            i += 1
            continue
        end = _span_end(words, i)
        operands = words[i + 1 : end]
        if "=" in word:
            # ``--flag=v`` still owns any operand that follows it. argparse reads
            # the extras as positionals rather than as the flag's list, so this
            # shape is malformed at the source -- but dropping only the
            # ``--flag=v`` word left them behind as bare argv words, which is the
            # one outcome this function promises never to produce.
            flag, _, value = word.partition("=")
            if flag in remove_flags or (flag, _logical_value(value)) in remove_pairs:
                if operands:
                    _warn_operand_list_widened(flag, len(operands) + 1)
                drop(i, end)
            i = end
            continue
        if word in remove_flags:
            drop(i, end)
            i = end
            continue
        if operands:
            whole = (word, " ".join(_logical_value(o) for o in operands))
            first = (word, _logical_value(operands[0]))
            if whole in remove_pairs or first in remove_pairs:
                if len(operands) > 1 and whole not in remove_pairs:
                    _warn_operand_list_widened(word, len(operands))
                drop(i, end)
                i = end
                continue
        i += 1

    return "".join(piece for i, piece in enumerate(pieces) if i not in dropped).strip()


# Serving-ineligible harness flags. Enroll here; compose_server_args strips them
# from what a grid launches and _lift_to_current_best from what a KEEP persists.
_BENCHMARK_HARNESS_FLAG_DENYLIST: tuple[str, ...] = ("--no-enable-prefix-caching",)


def strip_benchmark_harness_flags(server_args: str | None) -> str:
    """Drop every :data:`_BENCHMARK_HARNESS_FLAG_DENYLIST` entry from ``server_args``."""
    return remove_server_args(server_args, _BENCHMARK_HARNESS_FLAG_DENYLIST)


def compose_server_args(
    *,
    inherited_args: str | None = "",
    base_extra_args: str | None = "",
    variant_extra_args: str | None = "",
    remove_args: Any = None,
    args_mode: str = "append",
) -> str:
    """Compose inherited/base/variant args with optional remove/replace semantics.

    Always applies :func:`strip_benchmark_harness_flags` to the result.
    """
    mode = str(args_mode or "append").strip().lower()
    if mode == "replace":
        pruned_base = remove_server_args(base_extra_args, remove_args)
        pruned_variant = remove_server_args(variant_extra_args, remove_args)
        composed = merge_server_args(pruned_base, pruned_variant)
    else:
        combined_base = merge_server_args(inherited_args, base_extra_args)
        pruned = remove_server_args(combined_base, remove_args)
        composed = merge_server_args(pruned, variant_extra_args)
    return strip_benchmark_harness_flags(composed)


# A JSON "bareword": an identifier-like token that appears where a double-quoted
# JSON key or string value should be (letters/digits/underscore plus the ``.``,
# ``/``, ``-`` common in model ids and paths). Numbers, ``true``/``false``/
# ``null`` are handled separately so they stay unquoted.
_JSON_BAREWORD = r"[A-Za-z_][A-Za-z0-9_./-]*"
_UNQUOTED_KEY_RE = re.compile(r"([{,]\s*)(" + _JSON_BAREWORD + r")(\s*:)")
_UNQUOTED_VALUE_RE = re.compile(r"([:\[,]\s*)(" + _JSON_BAREWORD + r")")


def _repair_unquoted_json(blob: str) -> str | None:
    """Best-effort repair of a JSON blob whose double quotes were stripped.

    A shlex round-trip (``shlex.split`` then space-join without re-quoting)
    strips the inner double quotes of a JSON-valued server arg, turning a
    stored-valid ``{"method":"ngram"}`` into ``{method:ngram}`` — which vLLM's
    ``json.loads`` rejects at boot. Re-quote bare object keys and bare string
    values, then VALIDATE by parsing: return the compact valid-JSON string, or
    ``None`` when it still does not parse (caller keeps the blob verbatim).

    This is a narrowly scoped recovery heuristic for known JSON-valued server
    flags after shlex damage, not a general parser for JSON-like syntax.
    """

    def _quote_value(m: "re.Match[str]") -> str:
        prefix, word = m.group(1), m.group(2)
        if word in ("true", "false", "null"):
            return m.group(0)  # JSON literals stay unquoted
        return f'{prefix}"{word}"'

    # Keys first (so a re-quoted key is not re-matched as a value), then values.
    candidate = _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', blob)
    candidate = _UNQUOTED_VALUE_RE.sub(_quote_value, candidate)
    try:
        return json.dumps(json.loads(candidate), separators=(",", ":"))
    except Exception:
        return None


def compact_json_server_args(
    server_args: str | None,
    framework: str | None,
) -> str:
    """Normalize JSON-valued server args for unquoted Magpie expansion.

    Magpie's scripts expand ``$EXTRA_*_ARGS`` UNQUOTED, so shell quote wrappers
    stored by a prior ``shlex.join`` become literal argv bytes and separator
    spaces inside JSON values are word-split. Re-serialising each valid JSON
    object/array removes both hazards for every framework, including sglang
    (the default when ``framework`` is missing).

    JSON string values that themselves contain spaces cannot survive unquoted
    expansion and are left intact rather than corrupted; callers must reject
    those values before launch.

    Empty strings and strings with no ``{``/``[`` are returned unchanged. Any
    blob that does not parse (and cannot be safely repaired) retains its complete
    original substring, including balanced shell wrappers.
    """
    args = str(server_args or "").strip()
    if not args or ("{" not in args and "[" not in args):
        return args
    return _reserialize_json_blobs(args)


def _reserialize_json_blobs(args: str) -> str:
    """Normalize every JSON object/array while preserving invalid substrings.

    Framework-agnostic core shared by :func:`compact_json_server_args`,
    :func:`remove_server_args`, and the GBrain recipe sanitizer. Each balanced
    ``{...}``/``[...]`` blob is re-serialised with compact separators; a blob
    whose inner double quotes were stripped by an earlier shlex round-trip is
    repaired via :func:`_repair_unquoted_json`. Directly-adjacent shell single
    quotes are removed only when parsing or repair succeeds. Otherwise the full
    original substring is retained byte-for-byte.
    """
    if "{" not in args and "[" not in args:
        return args
    out: list[str] = []
    i = 0
    n = len(args)
    while i < n:
        ch = args[i]
        if ch in "{[":
            # A prior shlex.join can wrap a JSON token in shell single quotes.
            # These args are later expanded from an environment variable
            # without eval, so the wrappers become literal argv characters.
            # Strip only directly-adjacent wrappers around the balanced blob.
            single_quote_wrapped = i > 0 and args[i - 1] == "'" and out and out[-1] == "'"
            # Walk to the balanced close, honouring quoted strings.
            depth = 0
            in_str = False
            esc = False
            j = i
            while j < n:
                c = args[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif c in "{[":
                    depth += 1
                elif c in "}]":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            single_quote_wrapped = bool(single_quote_wrapped and j < n and args[j] == "'")
            blob = args[i:j]
            rendered: str | None = None
            try:
                rendered = json.dumps(json.loads(blob), separators=(",", ":"))
            except Exception:
                # A prior shlex round-trip can strip the JSON double quotes,
                # leaving an unquoted-bareword object (``{"m":"ngram"}`` ->
                # ``{m:ngram}``) that vLLM's json.loads rejects at boot. Try to
                # re-quote bare keys/values.
                rendered = _repair_unquoted_json(blob)
            if rendered is None:
                # Keep both wrappers when the content is not valid/repairable.
                # The opening quote is already in ``out``; leave the closing
                # quote for the next loop iteration.
                out.append(blob)
                i = j
            else:
                if single_quote_wrapped:
                    out.pop()
                out.append(rendered)
                i = j + 1 if single_quote_wrapped else j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# Whitespace-bearing / JSON-valued flags are NOT enumerated here on purpose.
# A by-name catalogue only ever protects the frameworks someone remembered to
# enroll: the vLLM-shaped list this module used to carry had no reader left, yet
# its docstring still promised dedup-time protection, so atom's
# ``--online_quant_config`` looked covered while nothing guarded it. Value shape
# is the property that actually matters, and
# :func:`tokenize_server_args_preserving_json` decides it by parsing, which is
# framework-agnostic and fails closed.

_MULTI_VALUE_FLAGS = (
    "--cuda-graph-bs",
    "--cuda-graph-max-bs",
)

# vLLM / atom argparse-style single-value options safe to collapse last-wins.
# vLLM hard-errors on a duplicate / conflicting flag (e.g.
# ``--attention-backend``); collapsing to last-wins keeps the variant override.
_VLLM_SINGLE_VALUE_FLAGS = frozenset(
    {
        "--attention-backend",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--block-size",
        "--kv-cache-dtype",
        "--quantization",
        "--dtype",
        "--swap-space",
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
    }
)


def tokenize_server_args_preserving_json(
    server_args: str | None,
) -> tuple[str, list[str]] | None:
    """Tokenize server args without stripping JSON's inner double quotes.

    Valid JSON blobs are normalized first, then ``shlex``'s non-POSIX mode keeps
    their quote bytes intact. The unquoted ``EXTRA_*_ARGS`` transport cannot
    represent an argv token containing whitespace; those inputs (including JSON
    strings with spaces and quoted non-JSON values) return ``None`` so callers
    can fail closed rather than silently changing token boundaries.

    Returns:
        ``(normalized_text, tokens)`` when every token is transport-safe;
        otherwise ``None``.
    """
    normalized = _reserialize_json_blobs(str(server_args or "").strip())
    if not normalized:
        return "", []
    try:
        tokens = shlex.split(normalized, posix=False)
    except ValueError:
        return None
    if _transport_unsafe_tokens(tokens):
        return None
    return normalized, tokens


def _json_scan(text: str, depth: int, in_string: bool, escaped: bool) -> tuple[int, bool, bool]:
    """Advance a JSON brace/string scan through ``text``.

    The state has to be carried across tokens, not restarted per token: a
    fragment that ends mid-string (``{"k":"v``) leaves the scan inside a string,
    and restarting would read the next fragment's closing quote as an opening
    one, hiding the ``}`` that balances the blob.
    """
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
    return depth, in_string, escaped


def _json_token_depth(token: str) -> int:
    """Net ``{[``/``]}`` nesting depth of ``token``, ignoring braces in strings.

    A balanced JSON value must survive as one token, so a non-zero depth at a
    token boundary means embedded whitespace split it.
    """
    return _json_scan(token, 0, False, False)[0]


def _transport_unsafe_tokens(tokens: list[str]) -> list[str]:
    """Tokens the unquoted ``EXTRA_*_ARGS`` transport cannot carry verbatim.

    Magpie expands the env through a shell wrapper without quoting it, so the
    shell word-splits on whitespace but performs no quote removal: a token with
    embedded whitespace loses its boundary, and a quote byte at either edge
    reaches the server as part of the value.

    ``remove_server_args`` passes whitespace-split words, which can never carry
    embedded whitespace themselves — there a JSON value that did shows up as
    several fragments, each with nonzero brace depth, which is the second check.
    The whitespace check stays for any caller tokenizing some other way.
    """
    unsafe: list[str] = []
    for token in tokens:
        if (
            any(ch.isspace() for ch in token)
            or _json_token_depth(token) != 0
            or token.startswith(("'", '"'))
            or token.endswith(("'", '"'))
        ):
            unsafe.append(token)
    return unsafe


def dedup_vllm_server_args(
    server_args: str | None,
    framework: str | None,
) -> str:
    """Collapse repeated vLLM/atom single-value flags to last-wins.

    vLLM crashes on duplicated single-value flags; sglang tolerates repeats, so
    this is scoped to the vllm/atom framework envs and is a no-op for sglang.
    Only the flags in :data:`_VLLM_SINGLE_VALUE_FLAGS` are touched; every other
    token is preserved verbatim and in order. For each affected flag the LAST
    occurrence wins, matching the override intent of :func:`merge_server_args`.

    JSON values are treated as opaque, quote-preserving tokens while unrelated
    duplicated flags are still collapsed. Returns ``server_args`` unchanged
    when the framework is sglang, the string is empty, carries a multi-value
    flag, or cannot be represented by Magpie's unquoted argv transport.

    Args:
        server_args (str | None): The server-arg string to dedupe.
        framework (str | None): Framework name; matched case-insensitively
            (sglang is a no-op).

    Returns:
        str: The deduped server-arg string, or the input unchanged when no
        dedupe applies.
    """
    args = str(server_args or "").strip()
    if not args:
        return args
    if server_args_env_name(framework) == "EXTRA_SGLANG_ARGS":
        return args
    if any(f in args for f in _MULTI_VALUE_FLAGS):
        return args
    parsed = tokenize_server_args_preserving_json(args)
    if parsed is None:
        return args
    normalized, tokens = parsed
    # Collect the token span of every recognized single-value flag.
    spans: list[tuple[str, int, int]] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        name = tok.split("=", 1)[0] if tok.startswith("--") else None
        if name in _VLLM_SINGLE_VALUE_FLAGS:
            if "=" in tok:  # ``--flag=value`` is self-contained.
                spans.append((name, i, i))
                i += 1
            elif i + 1 < n and not tokens[i + 1].startswith("-"):  # ``--flag value``
                spans.append((name, i, i + 1))
                i += 2
            else:  # Bare ``--flag`` with no value.
                spans.append((name, i, i))
                i += 1
        else:
            i += 1
    drop: set[int] = set()
    by_name: dict[str, list[tuple[str, int, int]]] = {}
    for span in spans:
        by_name.setdefault(span[0], []).append(span)
    for occurrences in by_name.values():
        # Keep only the last occurrence; drop the token span of the earlier ones.
        for _name, start, end in occurrences[:-1]:
            drop.update(range(start, end + 1))
    if not drop:
        return normalized
    kept = [tok for idx, tok in enumerate(tokens) if idx not in drop]
    return " ".join(kept)


def _shell_safe_dedupe(args: str) -> str:
    """Last-wins dedupe for single-token-valued flags only.

    Collapses repeated ``--flag value`` (or ``--flag=value``) pairs whose value
    is a single whitespace-free token, keeping the last occurrence. JSON values
    remain opaque tokens; actual whitespace-bearing argv values fail closed.

    Args:
        args (str): The server-arg string to dedupe.

    Returns:
        str: The last-wins deduped string, or the input unchanged when it cannot
        be represented by the unquoted argv transport.
    """
    if not args.strip():
        return ""
    if any(f in args for f in _MULTI_VALUE_FLAGS):
        return args
    parsed = tokenize_server_args_preserving_json(args)
    if parsed is None:
        return args
    normalized, tokens = parsed
    pairs: dict[str, list[str]] = {}
    order: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            if "=" in t:  # Normalize so it dedupes against ``--flag value``.
                flag, _, val = t.partition("=")
                pair = [flag, val]
                i += 1
            else:
                flag = t
                i += 1
                if i < len(tokens) and not tokens[i].startswith("--"):
                    pair = [flag, tokens[i]]
                    i += 1
                else:
                    pair = [flag]
            if flag not in pairs:
                order.append(flag)
            pairs[flag] = pair
        else:
            key = f"__pos_{len(order)}__"
            order.append(key)
            pairs[key] = [t]
            i += 1
    out: list[str] = []
    for k in order:
        out.extend(pairs[k])
    rendered = " ".join(out)
    return rendered if rendered != normalized else normalized


# sglang scheduler watchdog timeout injection: the first request's JIT compile
# can exceed sglang's default watchdog, firing SIGQUIT mid-warmup. Inject a
# longer timeout unless the user already pinned one.
DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC = 1800

SGLANG_WATCHDOG_TIMEOUT_ENV = "SGLANG_WATCHDOG_TIMEOUT"

_SGLANG_WATCHDOG_FLAG = "--watchdog-timeout"

# Matches space- or equals-separated form without false-matching a longer flag.
_SGLANG_WATCHDOG_RE = re.compile(r"--watchdog-timeout(?:[=\s]|$)")


def resolve_sglang_watchdog_timeout() -> int:
    """Resolve the sglang scheduler watchdog timeout in seconds.

    Reads ``$SGLANG_WATCHDOG_TIMEOUT`` (integer seconds) and falls back to
    :data:`DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC` when the env var is unset,
    empty, non-integer, or non-positive. A malformed value logs a warning
    and uses the default rather than crashing the YAML materialization.

    Returns:
        int: The resolved watchdog timeout in seconds.
    """
    raw = os.environ.get(SGLANG_WATCHDOG_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC
    try:
        val = int(raw)
    except ValueError:
        log.warning(
            "%s=%r is not an integer; using default %ds.",
            SGLANG_WATCHDOG_TIMEOUT_ENV,
            raw,
            DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC,
        )
        return DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC
    if val <= 0:
        log.warning(
            "%s=%d is not positive; using default %ds.",
            SGLANG_WATCHDOG_TIMEOUT_ENV,
            val,
            DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC,
        )
        return DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC
    return val


def inject_sglang_watchdog_timeout(
    server_args: str | None,
    framework: str | None,
) -> str:
    """Append ``--watchdog-timeout <N>`` to ``server_args`` for sglang runs.

    Returns ``server_args`` unchanged when the framework is not sglang
    (empty/unknown is treated as sglang) or the flag is already present.
    Otherwise appends the value from :func:`resolve_sglang_watchdog_timeout`;
    no other flag is touched.

    Args:
        server_args (str | None): The server-arg string to augment.
        framework (str | None): Framework name; empty/unknown treated as sglang.

    Returns:
        str: ``server_args`` with ``--watchdog-timeout`` appended, or unchanged
        for non-sglang frameworks or when the flag is already present.
    """
    args = str(server_args or "").strip()
    if server_args_env_name(framework) != "EXTRA_SGLANG_ARGS":
        return args
    if _SGLANG_WATCHDOG_RE.search(args):
        return args
    timeout = resolve_sglang_watchdog_timeout()
    return merge_server_args(args, f"{_SGLANG_WATCHDOG_FLAG} {timeout}")


# sglang ``--context-length`` cap injection: sglang sizes ``max_total_tokens``
# off the model's ``max_position_embeddings``, so a huge native window balloons
# the aiter workspace_buffer past GPU memory. Cap to ISL+OSL+headroom (floored,
# clamped to the native window) unless the flag is already pinned.
DEFAULT_SGLANG_CONTEXT_HEADROOM_TOKENS = 2048

DEFAULT_SGLANG_CONTEXT_FLOOR_TOKENS = 8192

SGLANG_CONTEXT_HEADROOM_ENV = "SGLANG_CONTEXT_HEADROOM_TOKENS"

SGLANG_CONTEXT_FLOOR_ENV = "SGLANG_CONTEXT_FLOOR_TOKENS"

_SGLANG_CONTEXT_LENGTH_FLAG = "--context-length"

# Matches space- or equals-separated form without false-matching a longer flag.
_SGLANG_CONTEXT_LENGTH_RE = re.compile(r"--context-length(?:[=\s]|$)")

_SGLANG_ATTN_BACKEND_FLAG = "--attention-backend"

_SGLANG_ATTN_BACKEND_RE = re.compile(r"--attention-backend(?:[=\s]|$)")

_SGLANG_DUAL_CHUNK_BACKEND = "dual_chunk_flash_attn"


def _resolve_nonneg_int_env(name: str, default: int) -> int:
    """Read a non-negative integer env override, else return ``default``.

    A blank/non-integer/negative value logs a warning and falls back to the
    default rather than crashing the YAML materialization.

    Args:
        name (str): Environment variable name to read.
        default (int): Fallback value when unset/invalid.

    Returns:
        int: The parsed non-negative integer, or ``default``.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        log.warning(
            "%s=%r is not an integer; using default %d.",
            name,
            raw,
            default,
        )
        return default
    if val < 0:
        log.warning(
            "%s=%d is negative; using default %d.",
            name,
            val,
            default,
        )
        return default
    return val


def resolve_sglang_context_cap(isl: int, osl: int) -> int:
    """Resolve the sglang ``--context-length`` cap for an ISL+OSL workload.

    Returns ``max(isl + osl + headroom, floor)`` (headroom / floor are
    operator-tunable via ``$SGLANG_CONTEXT_HEADROOM_TOKENS`` /
    ``$SGLANG_CONTEXT_FLOOR_TOKENS``). Caller clamps to the model's native
    window before injecting.

    Args:
        isl (int): Input sequence length.
        osl (int): Output sequence length.

    Returns:
        int: ``max(isl + osl + headroom, floor)``.
    """
    headroom = _resolve_nonneg_int_env(
        SGLANG_CONTEXT_HEADROOM_ENV,
        DEFAULT_SGLANG_CONTEXT_HEADROOM_TOKENS,
    )
    floor = _resolve_nonneg_int_env(
        SGLANG_CONTEXT_FLOOR_ENV,
        DEFAULT_SGLANG_CONTEXT_FLOOR_TOKENS,
    )
    return max(int(isl) + int(osl) + headroom, floor)


def validate_warm_replay_context_length(
    server_args: str | None,
    framework: str | None,
    isl: int,
    osl: int,
    max_model_len: int | str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Validate a replayed SGLang context window without changing its config.

    Exact Recipe identities do not include workload shape. A champion may
    therefore carry ``--context-length`` from a shorter run even though the
    current ISL+OSL no longer fits. Compatible or absent pins are preserved
    byte-for-byte; incompatible pins fail the replay preflight.
    """
    args = str(server_args or "")
    if server_args_env_name(framework) != "EXTRA_SGLANG_ARGS":
        return args, {"status": "not_sglang"}
    required = max(0, int(isl or 0)) + max(0, int(osl or 0))
    if required <= 0:
        return args, {"status": "target_shape_unknown"}
    max_len = optional_positive_int(max_model_len)
    if max_len is not None and max_len < required:
        raise ValueError(f"target workload exceeds MAX_MODEL_LEN: isl+osl={required} > max_model_len={max_len}")
    parsed = tokenize_server_args_preserving_json(args)
    if parsed is None:
        raise ValueError("warm replay server args are not safely tokenizable")
    _normalized, tokens = parsed
    values: list[int] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == _SGLANG_CONTEXT_LENGTH_FLAG:
            if index + 1 >= len(tokens):
                raise ValueError("--context-length is missing its value")
            raw_value = tokens[index + 1]
            index += 2
        elif token.startswith(f"{_SGLANG_CONTEXT_LENGTH_FLAG}="):
            raw_value = token.split("=", 1)[1]
            index += 1
        else:
            index += 1
            continue
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"--context-length must be an integer, got {raw_value!r}") from exc
        if value <= 0:
            raise ValueError("--context-length must be positive")
        values.append(value)
    if not values:
        return args, {
            "status": "context_length_absent",
            "required_context_length": required,
        }
    effective = values[-1]
    if effective >= required:
        return args, {
            "status": "compatible",
            "effective_context_length": effective,
            "required_context_length": required,
        }
    raise ValueError(
        "warm replay context length is incompatible with target workload: "
        f"context_length={effective} < isl+osl={required}"
    )


def inject_sglang_context_length(
    server_args: str | None,
    framework: str | None,
    model_path: str | None,
    isl: int,
    osl: int,
    max_model_len: int | str | None = None,
) -> str:
    """Append ``--context-length <N>`` to ``server_args`` for sglang runs.

    Returns ``server_args`` unchanged when the framework is not sglang
    (empty/unknown treated as sglang), the flag is already present, or the
    model's ``max_position_embeddings`` cannot be read. Otherwise appends
    ``min(max_pos, max_model_len, cap)`` from :func:`resolve_sglang_context_cap`;
    only this flag is added.

    sglang sizes its window off ``--context-length`` (it does not honor
    ``--max-model-len``), so without this clamp a workload cap above the run's
    explicit ``--max-model-len`` would inject a self-contradictory config. The
    ``max_model_len`` ceiling is applied only when it is a positive value.

    Under ``HYPERLOOM_AGENTX`` the ISL/OSL cap is skipped entirely: the AgentX
    corpus carries its own lengths and ISL/OSL are placeholders, so the window
    is ``min(max_pos, max_model_len)``.

    Args:
        server_args (str | None): The server-arg string to augment.
        framework (str | None): Framework name; empty/unknown treated as sglang.
        model_path (str | None): Model path used to read
            ``max_position_embeddings``.
        isl (int): Input sequence length.
        osl (int): Output sequence length.
        max_model_len (int | str | None): The run's explicit ``MAX_MODEL_LEN``
            ceiling; clamps ``--context-length`` so it never exceeds it. Ignored
            when unset, non-positive, or non-integer.

    Returns:
        str: ``server_args`` with ``--context-length`` appended, or unchanged
        for non-sglang frameworks, when the flag is present, or when the model
        window cannot be read.
    """
    args = str(server_args or "").strip()
    if server_args_env_name(framework) != "EXTRA_SGLANG_ARGS":
        return args
    if _SGLANG_CONTEXT_LENGTH_RE.search(args):
        return args
    from hyperloom.inference_optimizer.cli.model_gate import _load_model_max_position_embeddings

    max_pos = _load_model_max_position_embeddings(str(model_path or ""))
    if not max_pos:
        return args
    # AgentX replays a fixed trace corpus, so ISL/OSL are placeholders here and
    # the ISL+OSL+headroom ceiling (8192 at the 1024/1024 defaults) would pin
    # sglang's window two orders of magnitude below what the corpus needs --
    # every oversized trace then 4xxs. Corpus length is a property of the
    # workload, not of a synthetic shape, so the cap does not apply; the model's
    # own window, clamped by an explicit MAX_MODEL_LEN, is the only ceiling.
    # Imported lazily: _workload_envs imports this module.
    from ._workload_envs import agentx_enabled

    if agentx_enabled():
        context_length = int(max_pos)
    else:
        context_length = min(int(max_pos), resolve_sglang_context_cap(isl, osl))
    max_model_len_int = optional_positive_int(max_model_len)
    if max_model_len_int is not None:
        context_length = min(context_length, max_model_len_int)
    return merge_server_args(
        args,
        f"{_SGLANG_CONTEXT_LENGTH_FLAG} {context_length}",
    )


def _resolve_dual_chunk_backend(gpu_type: str | None = None) -> str:
    """Pick the dual-chunk attention backend for the current hardware.

    ``dual_chunk_flash_attn`` is the only backend sglang accepts when the
    model declares ``dual_chunk_attention_config``. It requires sm90+; on
    AMD/ROCm the preflight gate blocks these models before they reach here.
    Override via ``$HYPERLOOM_DUAL_CHUNK_BACKEND``.

    Args:
        gpu_type (str | None): Caller-known GPU type; accepted for parity but
            the canonical backend is returned regardless.

    Returns:
        str: ``$HYPERLOOM_DUAL_CHUNK_BACKEND`` when set, else
        ``dual_chunk_flash_attn``.
    """
    override = os.environ.get("HYPERLOOM_DUAL_CHUNK_BACKEND", "").strip()
    if override:
        return override
    return _SGLANG_DUAL_CHUNK_BACKEND


def inject_sglang_attention_backend(
    server_args: str | None,
    framework: str | None,
    model_path: str | None,
    gpu_type: str | None = None,
) -> str:
    """Append an ``--attention-backend`` for dual-chunk sglang models.

    Models that declare ``dual_chunk_attention_config`` make sglang hard-reject
    its default aiter backend. The backend is picked by
    :func:`_resolve_dual_chunk_backend`; ``gpu_type`` (when known by the caller)
    takes precedence over runtime autodetect.

    Returns ``server_args`` unchanged when: framework is not sglang, an
    ``--attention-backend`` is already pinned (operator wins), or the model
    config has no dual-chunk block (fail-safe: inject nothing).

    Args:
        server_args (str | None): The server-arg string to augment.
        framework (str | None): Framework name; empty/unknown treated as sglang.
        model_path (str | None): Model path checked for a dual-chunk config.
        gpu_type (str | None): Caller-known GPU type; takes precedence over
            autodetect.

    Returns:
        str: ``server_args`` with ``--attention-backend`` appended, or unchanged
        for non-sglang frameworks, when already pinned, or for non-dual-chunk
        models.
    """
    args = str(server_args or "").strip()
    if server_args_env_name(framework) != "EXTRA_SGLANG_ARGS":
        return args
    if _SGLANG_ATTN_BACKEND_RE.search(args):
        return args
    from hyperloom.inference_optimizer.cli.model_gate import _model_has_dual_chunk_attention

    if not _model_has_dual_chunk_attention(str(model_path or "")):
        return args
    backend = _resolve_dual_chunk_backend(gpu_type)
    if backend != _SGLANG_DUAL_CHUNK_BACKEND:
        log.info(
            "dual-chunk model on AMD/ROCm: injecting --attention-backend %s (dual_chunk_flash_attn needs sm90+).",
            backend,
        )
    return merge_server_args(
        args,
        f"{_SGLANG_ATTN_BACKEND_FLAG} {backend}",
    )


# sglang MoE runner backend injection: sglang's default routes MoE models
# through aiter's CK 2-stage fused-MoE kernel, whose JIT build is broken in some
# ROCm images. Inject the ROCm-capable ``triton`` backend for MoE models on AMD
# unless the operator already pinned one. Override via
# ``$HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND``.
HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND_ENV = "HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND"

DEFAULT_SGLANG_AMD_MOE_RUNNER_BACKEND = "triton"

_SGLANG_MOE_RUNNER_BACKEND_FLAG = "--moe-runner-backend"

# Matches space- or equals-separated form without false-matching a longer flag.
_SGLANG_MOE_RUNNER_BACKEND_RE = re.compile(r"--moe-runner-backend(?:[=\s]|$)")

# sglang MoE schemes whose ``create_moe_runner`` only builds a runner for the
# aiter backend (the others fall through to a bare ``pass``, so the first
# forward pass dies on a missing ``runner``). Two are selected online through
# ``--quantization`` rather than the checkpoint's own config: quark_int4fp8_moe
# always, and mxfp4 only when the checkpoint is NOT mxfp4-serialized (sglang
# then routes to its dynamic-quant MoE method, which is aiter-only too).
_AITER_ONLY_ONLINE_QUANT_METHODS = frozenset({"quark_int4fp8_moe"})

_AITER_ONLY_UNLESS_SERIALIZED_QUANT_METHOD = "mxfp4"

_SGLANG_QUANTIZATION_RE = re.compile(r"--quantization[=\s]+(\S+)")


def _online_quant_requires_aiter_moe_runner(server_args: str, model_path: str) -> bool:
    """Whether ``--quantization`` selects an aiter-only MoE scheme.

    Args:
        server_args (str): The server-arg string to read ``--quantization`` from.
        model_path (str): Model path, used to tell a serialized mxfp4
            checkpoint (which gets the backend-flexible method) from an online
            dynamic-quant one.

    Returns:
        bool: ``True`` when the selected scheme only has an aiter MoE runner.
    """
    match = _SGLANG_QUANTIZATION_RE.search(server_args or "")
    if not match:
        return False
    quantization = match.group(1).strip().strip("\"'").lower()
    if quantization in _AITER_ONLY_ONLINE_QUANT_METHODS:
        return True
    if quantization != _AITER_ONLY_UNLESS_SERIALIZED_QUANT_METHOD:
        return False
    from hyperloom.inference_optimizer.cli.model_gate import _model_declared_quant_method

    # Mirrors sglang: is_checkpoint_mxfp4_serialized = "mxfp4" in quant_method.
    return "mxfp4" not in _model_declared_quant_method(model_path)


def moe_runner_requires_aiter(server_args: str | None, model_path: str | None) -> bool:
    """Whether this model + server args resolve to an aiter-only MoE scheme.

    Folds the two ways sglang can land on such a scheme: the checkpoint's own
    Quark MX-FP4 config, and an online ``--quantization`` selection.

    Args:
        server_args (str | None): Server args, read for ``--quantization``.
        model_path (str | None): Model path whose ``config.json`` is inspected.

    Returns:
        bool: ``True`` when only the aiter MoE runner can serve this model.
    """
    from hyperloom.inference_optimizer.cli.model_gate import _model_moe_runner_requires_aiter

    path = str(model_path or "")
    return _model_moe_runner_requires_aiter(path) or _online_quant_requires_aiter_moe_runner(
        str(server_args or ""),
        path,
    )


def inject_sglang_moe_runner_backend(
    server_args: str | None,
    framework: str | None,
    model_path: str | None,
    gpu_type: str | None = None,
) -> str:
    """Append a ``--moe-runner-backend`` for MoE sglang models on AMD/ROCm.

    Returns ``server_args`` unchanged when: framework is not sglang, a
    ``--moe-runner-backend`` is already pinned (operator wins), the GPU is not
    an AMD/ROCm runner, the model is not Mixture-of-Experts (fail-safe: inject
    nothing), or the checkpoint carries a quant scheme only the aiter runner
    implements. Otherwise appends the backend from
    ``$HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND`` (default ``triton``); only this
    flag is added.

    Args:
        server_args (str | None): The server-arg string to augment.
        framework (str | None): Framework name; empty/unknown treated as sglang.
        model_path (str | None): Model path checked for Mixture-of-Experts.
        gpu_type (str | None): Caller-known GPU type; used to gate AMD/ROCm.

    Returns:
        str: ``server_args`` with ``--moe-runner-backend`` appended, or unchanged
        for non-sglang frameworks, when already pinned, off AMD/ROCm, for
        non-MoE models, or for aiter-only MoE quant schemes.
    """
    args = str(server_args or "").strip()
    if server_args_env_name(framework) != "EXTRA_SGLANG_ARGS":
        return args
    if _SGLANG_MOE_RUNNER_BACKEND_RE.search(args):
        return args
    from hyperloom.inference_optimizer.cli.model_gate import _model_is_moe
    from hyperloom.inference_optimizer.gpu_types import _resolve_amd_gpu_type

    if not _resolve_amd_gpu_type(gpu_type):
        return args
    if not _model_is_moe(str(model_path or "")):
        return args
    # An aiter-only MoE scheme has no triton runner: injecting one crashes the
    # server on the first forward pass. Let sglang resolve the backend itself.
    if moe_runner_requires_aiter(args, str(model_path or "")):
        log.info(
            "MoE model with an aiter-only quant scheme: skipping "
            "--moe-runner-backend injection (the triton runner has no "
            "implementation for it)."
        )
        return args
    backend = (
        os.environ.get(HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND_ENV, "").strip() or DEFAULT_SGLANG_AMD_MOE_RUNNER_BACKEND
    )
    log.info(
        "MoE model on AMD/ROCm: injecting --moe-runner-backend %s (aiter CK "
        "2-stage fused-MoE JIT build is broken in this image).",
        backend,
    )
    return merge_server_args(
        args,
        f"{_SGLANG_MOE_RUNNER_BACKEND_FLAG} {backend}",
    )


def apply_runtime_benchmark_overrides(
    bench: dict[str, Any],
    *,
    model_path: str | None = None,
    gpu_type: str | None = None,
    benchmark_script: str | None = None,
) -> dict[str, Any]:
    """Apply runtime env/CLI overrides to a Magpie benchmark YAML.

    Single shared path for baseline/profile and grid executors.
    ``benchmark_script`` (must be pre-sanitized via :func:`sanitize_script_name`)
    force-selects a specific Magpie script, applied AFTER the
    ``gpu_type``-derived generic script so the operator pick wins.

    Args:
        bench (dict[str, Any]): The Magpie ``benchmark`` config to mutate.
        model_path (str | None): Overrides ``benchmark.model`` when set.
        gpu_type (str | None): Pins ``runner_type`` and the generic
            ``{framework}_{gpu_type}.sh`` script.
        benchmark_script (str | None): Pre-sanitized script name that
            force-selects a Magpie script (applied last).

    Returns:
        dict[str, Any]: The mutated ``benchmark["envs"]`` mapping.
    """
    if model_path:
        bench["model"] = str(model_path)

    precision = os.environ.get("PRECISION", "").strip()
    if precision:
        bench["precision"] = precision

    if gpu_type:
        bench["runner_type"] = str(gpu_type)
        # Force-pin the generic ``{framework}_{gpu_type}.sh`` so Magpie's
        # resolver doesn't fall through to InferenceX native scripts that
        # ignore ``EXTRA_*_ARGS``.
        framework = str(bench.get("framework") or "").lower()
        if framework:
            bench["benchmark_script"] = f"{framework}_{gpu_type}.sh"
        else:
            bench.pop("benchmark_script", None)

    if benchmark_script:
        bench["benchmark_script"] = str(benchmark_script)

    # AgentX switch on the shared rebuild path: without this, the gpu_type block
    # above re-pins the synthetic {framework}_{gpu_type}.sh and silently reverts
    # a materialize-time AgentX swap (grid/baseline/profile executors rebuild via
    # this function). No-op when HYPERLOOM_AGENTX is off. Lazy import avoids a
    # module-load cycle with _workload_envs.
    from ._workload_envs import apply_agentx_switch, apply_scriptable_runtime_defaults

    apply_agentx_switch(bench, model_path)

    envs = bench.setdefault("envs", {})
    # Same hazard as the AgentX swap above: the gpu_type block re-pins the bare
    # {framework}_{gpu_type}.sh over the bundled absolute path the materialize
    # path resolved, so grid variants must re-apply the scriptable defaults.
    apply_scriptable_runtime_defaults(
        bench,
        envs,
        gpu_type=gpu_type,
        explicit_benchmark_script=bool(benchmark_script),
    )
    for env_key in ("ISL", "OSL", "MAX_MODEL_LEN", "TP", "CONC"):
        val = os.environ.get(env_key, "").strip()
        if not val:
            continue
        # TP yaml-explicit wins: a stale state.tp must not downgrade a
        # YAML-pinned TP.
        if env_key == "TP":
            yaml_tp = envs.get("TP")
            if yaml_tp not in (None, 0, "", "0"):
                continue
        envs[env_key] = int(val)

    explicit_rocr = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    if explicit_rocr:
        envs["ROCR_VISIBLE_DEVICES"] = explicit_rocr
    else:
        tp_val = int(envs.get("TP", 1) or 1)
        existing_rocr = str(envs.get("ROCR_VISIBLE_DEVICES", "")).strip()
        existing_count = len([x for x in existing_rocr.split(",") if x.strip()]) if existing_rocr else 0
        if tp_val > 1 and existing_count < tp_val:
            envs["ROCR_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(tp_val))

    return envs
