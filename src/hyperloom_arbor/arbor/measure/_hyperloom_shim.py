# Copied from AMD-AGI/Hyperloom 68b491e49:src/hyperloom/contracts/recipe.py (REMOVED), src/hyperloom/inference_optimizer/grid_server_args.py (tokenize_server_args_preserving_json and what it reaches) by scripts/sync_weight_precision.py; do not edit it here.
# Hyperloom's toolbox owns it; only its imports are Arbor's.
"""What Hyperloom's weight-precision module imports from Hyperloom, for Arbor's copy of it.

``REMOVED`` and the tokenizer are copied; ``LaunchConfig`` holds only the two fields the module reads.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass

REMOVED = '<removed>'


@dataclass(frozen=True)
class LaunchConfig:
    """A launch as the module reads it: its server arguments, and the long options it removes."""

    server_args: str = ""
    remove_args: tuple[str, ...] = ()


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


def _reserialize_json_blobs(args: str) -> str:
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


def tokenize_server_args_preserving_json(
    server_args: str | None,
) -> tuple[str, list[str]] | None:
    """Tokenize server args without stripping JSON's inner double quotes."""
    normalized = _reserialize_json_blobs(str(server_args or "").strip())
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
