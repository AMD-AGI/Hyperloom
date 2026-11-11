# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Server argv as text: JSON-preserving tokenization, flag lookup, JSON normalization and removal."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Sequence
from typing import Any

from hyperloom_common.coerce import to_str_list


def _unwrap_one_pair(s: str) -> str:
    """Strip one balanced pair of outer shell quotes from *s* when safe to do so."""
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        inner = s[1:-1]
        if inner and not inner.startswith(("{", "[")) and s[0] not in inner and not any(ch.isspace() for ch in inner):
            return inner
    return s


def _unwrap_shell_quotes(token: str) -> str:
    """Drop one balanced pair of shell quotes from a token produced by shlex."""
    if token.startswith("-") and "=" in token:
        flag, _, value = token.partition("=")
        unwrapped = _unwrap_one_pair(value)
        return f"{flag}={unwrapped}" if unwrapped != value else token
    return _unwrap_one_pair(token)


def split_args_preserving_json(text: str) -> list[str] | None:
    """Tokenize a server-arg string WITHOUT stripping JSON's inner double quotes."""
    try:
        tokens = shlex.split(reserialize_json_blobs(text), posix=False)
    except ValueError:
        return None
    return [_unwrap_shell_quotes(tok) for tok in tokens]


def parse_removal_specs(remove_args: Any) -> list[tuple[str, str | None]]:
    """Parse removal specs into ``(flag, value)`` pairs, in the order they are written.

    One entry may name several options (``"--a --b"``), and a removal may be scoped to a value
    (``"--a 8"``, ``"--a=8"``); a bare flag parses as ``(flag, None)`` and drops every copy of it.
    """
    parsed: list[tuple[str, str | None]] = []
    for spec in to_str_list(remove_args):
        spec_tokens = split_args_preserving_json(spec)
        if spec_tokens is None:
            spec_tokens = spec.split()
        i = 0
        while i < len(spec_tokens):
            tok = spec_tokens[i]
            if not tok.startswith("--"):
                i += 1
                continue
            if "=" in tok:
                flag, _, value = tok.partition("=")
                parsed.append((flag, value))
                i += 1
            elif i + 1 < len(spec_tokens) and not spec_tokens[i + 1].startswith("--"):
                parsed.append((tok, spec_tokens[i + 1]))
                i += 2
            else:
                parsed.append((tok, None))
                i += 1
    return parsed


def removal_option_names(remove_args: Any) -> list[str]:
    """The bare long option names a removal spec names, in order, each one once.

    Only for a consumer whose contract is option names, such as an AgentX point's ``launch_overrides.remove_args``:
    the value a spec is scoped to is dropped here, so this must not stand in for a removal composed against
    args we build ourselves -- ``remove_server_args`` keeps the scope there.
    """
    names: list[str] = []
    for flag, _value in parse_removal_specs(remove_args):
        if flag not in names:
            names.append(flag)
    return names


def _removed_token_count(tokens: list[str], index: int, removals: set[tuple[str, str | None]]) -> int:
    """Count the tokens at *index* that *removals* deletes; 0 when the token survives."""
    tok = tokens[index]
    if not tok.startswith("--"):
        return 0
    flag, sep, value = tok.partition("=")
    drops_flag = (flag, None) in removals
    if sep and (drops_flag or (flag, value) in removals):
        return 1
    if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
        if drops_flag or (flag, tokens[index + 1]) in removals:
            return 2
    return 1 if drops_flag else 0


def remove_server_args(server_args: str | None, remove_args: Any) -> str:
    """Remove flag specs from a server-arg string."""
    # Normalized up front as well as inside the tokenizer, so the nothing-to-remove early return below hands back the
    # same shape a caller with a non-empty denylist would get.
    args = reserialize_json_blobs(str(server_args or "").strip())
    removes = to_str_list(remove_args)
    if not args or not removes:
        return args
    # Non-POSIX split plus the wrapper strip: a plain operand written as ``--tool-call-parser 'kimi_k3'`` -- or
    # ``--tool-call-parser='kimi_k3'`` -- must not keep its quotes, because EXTRA_*_ARGS travel unquoted and
    # they would reach argv literally. _unwrap_shell_quotes only touches whitespace-free content, and on a
    # leading-dash token only the value side of the first ``=``, so a JSON blob (starts with ``{``/``[``) is never
    # affected and token boundaries cannot shift.
    tokens = split_args_preserving_json(args)
    if tokens is None:
        return args

    removals: set[tuple[str, str | None]] = set(parse_removal_specs(removes))

    out: list[str] = []
    i = 0
    while i < len(tokens):
        dropped = _removed_token_count(tokens, i, removals)
        if dropped:
            i += dropped
            continue
        out.append(tokens[i])
        i += 1
    # No re-serialisation on the way out: the tokens are already JSON-compacted and the non-POSIX split kept each one
    # byte-for-byte, so re-joining the survivors cannot corrupt a sibling flag.
    return " ".join(out)


# A JSON "bareword": an identifier-like token that appears where a double-quoted JSON key or string value should be
# (letters/digits/underscore plus the ``.``, ``/``, ``-`` common in model ids and paths). It may start with ``/`` so an
# absolute path value (a spec-decode draft ``model``) is quoted too; a bare number never matches and stays numeric.
_JSON_BAREWORD = r"[+-]?[A-Za-z_/][A-Za-z0-9_./-]*"
_UNQUOTED_KEY_RE = re.compile(r"([{,]\s*)(" + _JSON_BAREWORD + r")(\s*:)")
_UNQUOTED_VALUE_RE = re.compile(r"([:\[,]\s*)(" + _JSON_BAREWORD + r")")


def _repair_unquoted_json(blob: str) -> str | None:
    """Best-effort repair of a JSON blob whose double quotes were stripped."""

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
    except ValueError:
        return None


def compact_json_server_args(server_args: str | None) -> str:
    """Normalize JSON-valued server args for unquoted EXTRA_*_ARGS transport."""
    args = str(server_args or "").strip()
    if not args or ("{" not in args and "[" not in args):
        return args
    return reserialize_json_blobs(args)


def _json_blob_end(args: str, start: int) -> int:
    """Offset just past the close balancing the bracket at *start*, or ``len(args)`` when it never balances."""
    depth = 0
    in_str = False
    esc = False
    j = start
    n = len(args)
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
                return j + 1
        j += 1
    return n


def reserialize_json_blobs(args: str) -> str:
    """Normalize every JSON object/array while preserving invalid substrings."""
    if "{" not in args and "[" not in args:
        return args
    out: list[str] = []
    i = 0
    n = len(args)
    while i < n:
        if args[i] not in "{[":
            out.append(args[i])
            i += 1
            continue
        j = _json_blob_end(args, i)
        # A prior shlex.join can wrap a JSON token in shell single quotes.
        single_quote_wrapped = bool(
            i > 0 and args[i - 1] == "'" and out and out[-1] == "'" and j < n and args[j] == "'"
        )
        blob = args[i:j]
        try:
            rendered: str | None = json.dumps(json.loads(blob), separators=(",", ":"))
        except ValueError:
            # A prior shlex round-trip can strip the JSON double quotes, leaving an unquoted-bareword object
            # (``{"m":"ngram"}`` -> ``{m:ngram}``) that vLLM's json.loads rejects at boot.
            rendered = _repair_unquoted_json(blob)
        if rendered is None:
            # Keep both wrappers when the content is not valid/repairable.
            out.append(blob)
            i = j
            continue
        if single_quote_wrapped:
            out.pop()
        out.append(rendered)
        i = j + 1 if single_quote_wrapped else j
    return "".join(out)


# Canonical flag names (as ``flag_occurrences`` spells them) for the launch settings read in more than one place.
SPECULATIVE_CONFIG_FLAG = "speculative-config"
SPECULATIVE_CONFIG_KEY_PREFIX = f"{SPECULATIVE_CONFIG_FLAG}."


def tokenize_server_args_preserving_json(
    server_args: str | None,
) -> tuple[str, list[str]] | None:
    """Tokenize server args without stripping JSON's inner double quotes."""
    normalized = reserialize_json_blobs(str(server_args or "").strip())
    if not normalized:
        return "", []
    try:
        tokens = shlex.split(normalized, posix=False)
    except ValueError:
        return None
    for token in tokens:
        # A balanced JSON value must remain one token.
        depth = 0
        in_string = False
        escaped = False
        for char in token:
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
        if depth != 0:
            return None
        if any(ch.isspace() for ch in token):
            return None
        # ``shlex.split(..., posix=False)`` can fracture a quoted operand with whitespace into edge-quoted pieces
        # (``"my`` / ``parser"``).
        if token.startswith(("'", '"')) or token.endswith(("'", '"')):
            return None
    return normalized, tokens


def is_flag(token: str) -> bool:
    """``token`` names a flag rather than carrying a value (``-tp`` does, ``-1`` does not)."""
    return token.startswith("-") and not token[1:2].isdigit() and token.strip("-") != ""


def canonical_flag_name(flag: str) -> str:
    """``flag`` with no leading dashes, lowercase and underscores as dashes: one key per argparse destination."""
    return flag.lstrip("-").lower().replace("_", "-")


def flag_occurrences(server_args: str | None) -> list[tuple[str, str]]:
    """Every flag a composed launch spells, in order, as ``(name, value)``; ``""`` for a bare flag.

    Names come back canonical -- no leading dashes, lowercase, underscores as dashes -- so the spellings the engines
    accept for one argparse destination meet in one key.
    """
    split = tokenize_server_args_preserving_json(server_args)
    return argv_flag_occurrences(split[1] if split is not None else str(server_args or "").split())


def argv_flag_occurrences(tokens: Sequence[str]) -> list[tuple[str, str]]:
    """:func:`flag_occurrences` of a launch already split into its argv."""
    occurrences: list[tuple[str, str]] = []
    for index, token in enumerate(tokens):
        if not is_flag(token):
            continue
        name, assigned, value = token.partition("=")
        if not assigned and index + 1 < len(tokens) and not is_flag(tokens[index + 1]):
            value = tokens[index + 1]
        occurrences.append((canonical_flag_name(name), value))
    return occurrences


def last_alias_value(occurrences: list[tuple[str, str]], aliases: frozenset[str]) -> str | None:
    """The value of the last flag spelling one argparse destination -- the one the engine keeps."""
    return next((value for name, value in reversed(occurrences) if name in aliases), None)


def json_object(text: str) -> dict[str, Any]:
    """``text`` read as a JSON object, or ``{}`` when it is neither."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def speculative_config(occurrences: list[tuple[str, str]]) -> dict[str, Any]:
    """The speculative config a launch spells, both spellings merged the way vLLM merges them.

    It arrives as a JSON object under ``--speculative-config`` and as ``--speculative-config.<key>`` flags, and the
    per-key flags win over the object.
    """
    config: dict[str, Any] = {}
    for flag, value in dict(occurrences).items():
        if flag == SPECULATIVE_CONFIG_FLAG:
            config = {**json_object(value), **config}
        elif flag.startswith(SPECULATIVE_CONFIG_KEY_PREFIX):
            config[flag.removeprefix(SPECULATIVE_CONFIG_KEY_PREFIX).replace("-", "_")] = value
    return config


def without_speculative_config(argv: Sequence[str]) -> list[str]:
    """``argv`` with every ``--speculative-config`` and ``--speculative-config.<key>`` flag and its value dropped."""
    kept: list[str] = []
    dropping = False
    for token in argv:
        if is_flag(token):
            name = canonical_flag_name(token.partition("=")[0])
            dropping = name == SPECULATIVE_CONFIG_FLAG or name.startswith(SPECULATIVE_CONFIG_KEY_PREFIX)
        if not dropping:
            kept.append(token)
    return kept


def normalize_argv(argv: Sequence[str]) -> tuple[str, ...]:
    """``argv`` as a launch identity reads it: ``--flag=value`` split in two and JSON values compacted.

    Order is kept: of two aliases of one flag, or a switch and its negation, the engine applies the last, so
    reordering would give launches that behave differently the same identity.
    """
    normalized: list[str] = []
    for token in argv:
        if is_flag(token) and "=" in token:
            flag, _, value = token.partition("=")
            normalized.extend((flag, reserialize_json_blobs(value)))
        else:
            normalized.append(reserialize_json_blobs(token))
    return tuple(normalized)


__all__ = [
    "SPECULATIVE_CONFIG_FLAG",
    "SPECULATIVE_CONFIG_KEY_PREFIX",
    "argv_flag_occurrences",
    "canonical_flag_name",
    "compact_json_server_args",
    "flag_occurrences",
    "is_flag",
    "json_object",
    "last_alias_value",
    "normalize_argv",
    "parse_removal_specs",
    "removal_option_names",
    "remove_server_args",
    "reserialize_json_blobs",
    "speculative_config",
    "split_args_preserving_json",
    "tokenize_server_args_preserving_json",
    "without_speculative_config",
]
