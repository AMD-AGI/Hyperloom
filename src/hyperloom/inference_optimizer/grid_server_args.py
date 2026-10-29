# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Server-argument composition helpers."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
from dataclasses import dataclass
from typing import Any

from hyperloom.common.coerce import optional_positive_int, to_str_list
from hyperloom.inference_optimizer.framework_registry import server_args_env_name


log = logging.getLogger(__name__)

_UNSAFE_SERVER_ARG_CHARS_RE = re.compile(r"[;&|`$<>\r\n]")
#: What the 2nd..nth value of a not-yet-whitelisted multi-value flag may look
#: like: a batch size, a length, a ratio. Deliberately not a general token.
_NUMERIC_VALUE_RE = re.compile(r"^\d+(?:[.,]\d+)*$")


def validate_server_args_shell_safe(server_args: str | None) -> str:
    """Reject server-arg strings that would be shell control syntax."""
    args = str(server_args or "").strip()
    if not args:
        return ""
    if _UNSAFE_SERVER_ARG_CHARS_RE.search(args):
        raise ValueError("extra_server_args contains shell control characters")
    try:
        tokens = shlex.split(args)
    except ValueError as exc:
        raise ValueError(f"extra_server_args is not shell-tokenizable: {exc}") from exc
    state = "positional"
    for token in tokens:
        if token.startswith("-"):
            if "=" in token:
                state = "positional"
            elif token in _MULTI_VALUE_FLAGS:
                state = "many"
            else:
                state = "any"
            continue
        if state == "many" and _NUMERIC_VALUE_RE.match(token):
            continue
        if state == "any":
            state = "numeric"
            continue
        if state == "numeric" and _NUMERIC_VALUE_RE.match(token):
            continue
        raise ValueError("extra_server_args must be argv-like flags, not bare positional arguments")
    return args


def merge_server_args(*parts: str | None) -> str:
    """Merge server arg strings preserving left-to-right override semantics."""
    return " ".join(str(p).strip() for p in parts if str(p or "").strip())


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


def _split_args_preserving_json(text: str) -> list[str] | None:
    """Tokenize a server-arg string WITHOUT stripping JSON's inner double quotes."""
    try:
        tokens = shlex.split(_reserialize_json_blobs(text), posix=False)
    except ValueError:
        return None
    return [_unwrap_shell_quotes(tok) for tok in tokens]


def _parse_removal_specs(remove_args: Any) -> list[tuple[str, str | None]]:
    """Parse removal specs into ``(flag, value)`` pairs, in the order they are written.

    One entry may name several options (``"--a --b"``), and a removal may be scoped to a value
    (``"--a 8"``, ``"--a=8"``); a bare flag parses as ``(flag, None)`` and drops every copy of it.
    """
    parsed: list[tuple[str, str | None]] = []
    for spec in to_str_list(remove_args):
        spec_tokens = _split_args_preserving_json(spec)
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

    Only for a consumer whose contract is option names, such as Magpie's ``launch_overrides.remove_args``:
    the value a spec is scoped to is dropped here, so this must not stand in for a removal composed against
    args we build ourselves -- ``remove_server_args`` keeps the scope there.
    """
    names: list[str] = []
    for flag, _value in _parse_removal_specs(remove_args):
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
    args = _reserialize_json_blobs(str(server_args or "").strip())
    removes = to_str_list(remove_args)
    if not args or not removes:
        return args
    # Non-POSIX split plus the wrapper strip: a plain operand written as ``--tool-call-parser 'kimi_k3'`` -- or
    # ``--tool-call-parser='kimi_k3'`` -- must not keep its quotes, because Magpie expands EXTRA_*_ARGS unquoted and
    # they would reach argv literally. _unwrap_shell_quotes only touches whitespace-free content, and on a
    # leading-dash token only the value side of the first ``=``, so a JSON blob (starts with ``{``/``[``) is never
    # affected and token boundaries cannot shift.
    tokens = _split_args_preserving_json(args)
    if tokens is None:
        return args

    removals: set[tuple[str, str | None]] = set(_parse_removal_specs(removes))

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


# Serving-ineligible harness flags: legal to benchmark with, wrong to hand back as a serving recipe. Stripped where a
# recipe is persisted, never on the launch path: the baseline keeps the operator's flags, so a variant must too.
_BENCHMARK_HARNESS_FLAG_DENYLIST: tuple[str, ...] = ("--no-enable-prefix-caching",)


def strip_benchmark_harness_flags(server_args: str | None) -> str:
    """Drop every :data:`_BENCHMARK_HARNESS_FLAG_DENYLIST` entry from ``server_args``."""
    return remove_server_args(server_args, _BENCHMARK_HARNESS_FLAG_DENYLIST)


def normalize_server_args(server_args: str | None) -> str:
    """Compact JSON values and drop one layer of shell wrappers, removing nothing.

    Magpie expands ``EXTRA_*_ARGS`` unquoted, so a ``--quantization 'fp8'`` wrapper would reach argv literally.
    """
    args = _reserialize_json_blobs(str(server_args or "").strip())
    if not args:
        return ""
    tokens = _split_args_preserving_json(args)
    return args if tokens is None else " ".join(tokens)


def compose_server_args(
    *,
    inherited_args: str | None = "",
    base_extra_args: str | None = "",
    variant_extra_args: str | None = "",
    remove_args: Any = None,
    args_mode: str = "append",
) -> str:
    """Compose a delta or a complete snapshot with its final removals.

    In append mode the variant is a new assignment, after inherited removals.
    In replace mode both inputs are snapshots, so removals apply to both.
    """
    mode = str(args_mode or "append").strip().lower()
    if mode == "replace":
        raw = merge_server_args(base_extra_args, variant_extra_args)
        pruned_base = remove_server_args(base_extra_args, remove_args)
        pruned_variant = remove_server_args(variant_extra_args, remove_args)
        composed = merge_server_args(pruned_base, pruned_variant)
    else:
        combined_base = merge_server_args(inherited_args, base_extra_args)
        raw = merge_server_args(combined_base, variant_extra_args)
        pruned = remove_server_args(combined_base, remove_args)
        composed = merge_server_args(pruned, variant_extra_args)
    result = normalize_server_args(composed)
    # Compare against the RAW inputs, not against ``composed``.
    _warn_on_damaged_json_values(raw, result)
    return result


def _json_flag_values(args: str) -> dict[str, list[str]]:
    """Map each :data:`SPACE_VALUE_FLAGS` occurrence to its raw value token."""
    found: dict[str, list[str]] = {}
    for flag in SPACE_VALUE_FLAGS:
        start = 0
        while True:
            i = args.find(flag + " ", start)
            if i < 0:
                break
            value = args[i + len(flag) :].strip().split(" ", 1)[0]
            if value[:1] in ("{", "["):
                found.setdefault(flag, []).append(value)
            start = i + len(flag)
    return found


def _warn_on_damaged_json_values(before: str, after: str) -> None:
    """Log loudly when composition turned a parseable JSON flag value unparseable."""
    was = _json_flag_values(before)
    now = _json_flag_values(after)
    for flag, values in now.items():
        healthy_before = any(_parses_as_json(v) for v in was.get(flag, []))
        if healthy_before and not any(_parses_as_json(v) for v in values):
            log.error(
                "server-arg composition CORRUPTED %s: its value parsed as JSON "
                "before composition and does not after. The launched server will "
                "abort at argv parse. Damaged value: %s",
                flag,
                values[0][:200] if values else "<missing>",
            )


def _parses_as_json(value: str) -> bool:
    try:
        json.loads(value)
    except ValueError:
        return False
    return True


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
    """Normalize JSON-valued server args for unquoted Magpie expansion."""
    args = str(server_args or "").strip()
    if not args or ("{" not in args and "[" not in args):
        return args
    return _reserialize_json_blobs(args)


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


# Flags whose values may be JSON or otherwise space-bearing.
SPACE_VALUE_FLAGS = (
    "--json-model-override-args",
    "--override-generation-config",
    "--tool-call-parser",
    # JSON-object-valued flags: after ``compact_json_server_args`` these are a single space-free shell word, but their
    # value still contains inner double quotes (``{"cudagraph_mode":"PIECEWISE"}``).
    "--compilation-config",
    "--speculative-config",
    "--hf-overrides",
    "--kv-transfer-config",
)

_MULTI_VALUE_FLAGS = (
    "--cuda-graph-bs",
    "--cuda-graph-max-bs",
)

# vLLM / atom argparse-style single-value options safe to collapse last-wins. vLLM hard-errors on a duplicate /
# conflicting flag (e.g. ``--attention-backend``); collapsing to last-wins keeps the variant override.
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


def server_arg_value(args: str | None, flag: str) -> str:
    """Value of the last ``flag`` occurrence (space or ``=`` form), else ``""``.

    Whitespace-split, so a quoted or malformed token elsewhere in ``args`` does not void the lookup the way it voids
    :func:`tokenize_server_args_preserving_json`.
    """
    toks = str(args or "").replace("=", " ").split()
    value = ""
    for i, tok in enumerate(toks):
        if tok == flag and i + 1 < len(toks):
            value = toks[i + 1]
    return value


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


def dedup_vllm_server_args(
    server_args: str | None,
    framework: str | None,
) -> str:
    """Collapse repeated vLLM/atom single-value flags to last-wins."""
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


# sglang scheduler watchdog timeout injection: the first request's JIT compile can exceed sglang's default watchdog,
# firing SIGQUIT mid-warmup.
DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC = 1800

SGLANG_WATCHDOG_TIMEOUT_ENV = "SGLANG_WATCHDOG_TIMEOUT"

SGLANG_WATCHDOG_FLAG = "--watchdog-timeout"

# Matches space- or equals-separated form without false-matching a longer flag.
_SGLANG_WATCHDOG_RE = re.compile(r"--watchdog-timeout(?:[=\s]|$)")


def resolve_sglang_watchdog_timeout() -> int:
    """Resolve the sglang scheduler watchdog timeout in seconds."""
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
    """Append ``--watchdog-timeout <N>`` to ``server_args`` for sglang runs."""
    args = str(server_args or "").strip()
    if server_args_env_name(framework) != "EXTRA_SGLANG_ARGS":
        return args
    if _SGLANG_WATCHDOG_RE.search(args):
        return args
    timeout = resolve_sglang_watchdog_timeout()
    return merge_server_args(args, f"{SGLANG_WATCHDOG_FLAG} {timeout}")


# sglang ``--context-length`` cap injection: sglang sizes ``max_total_tokens`` off the model's
# ``max_position_embeddings``, so a huge native window balloons the aiter workspace_buffer past GPU memory.
DEFAULT_SGLANG_CONTEXT_HEADROOM_TOKENS = 2048

DEFAULT_SGLANG_CONTEXT_FLOOR_TOKENS = 8192

SGLANG_CONTEXT_HEADROOM_ENV = "SGLANG_CONTEXT_HEADROOM_TOKENS"

SGLANG_CONTEXT_FLOOR_ENV = "SGLANG_CONTEXT_FLOOR_TOKENS"

_SGLANG_CONTEXT_LENGTH_FLAG = "--context-length"

# Matches space- or equals-separated form without false-matching a longer flag.
_SGLANG_CONTEXT_LENGTH_RE = re.compile(r"--context-length(?:[=\s]|$)")

#: The flags the launch guards add when a recipe leaves them out (``inject_sglang_context_length`` and
#: ``inject_sglang_watchdog_timeout``), sized for the run they launch; a recipe that keeps them carries that run's
#: settings into every later one.
INJECTED_LAUNCH_FLAGS = (_SGLANG_CONTEXT_LENGTH_FLAG, SGLANG_WATCHDOG_FLAG)

_SGLANG_ATTN_BACKEND_FLAG = "--attention-backend"

_SGLANG_ATTN_BACKEND_RE = re.compile(r"--attention-backend(?:[=\s]|$)")

_SGLANG_DUAL_CHUNK_BACKEND = "dual_chunk_flash_attn"


def _resolve_nonneg_int_env(name: str, default: int) -> int:
    """Read a non-negative integer env override, else return ``default``."""
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
    """Resolve the sglang ``--context-length`` cap for an ISL+OSL workload."""
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
) -> dict[str, Any]:
    """Validate a replayed SGLang context window and return its compatibility evidence."""
    args = str(server_args or "")
    if server_args_env_name(framework) != "EXTRA_SGLANG_ARGS":
        return {"status": "not_sglang"}
    required = max(0, int(isl or 0)) + max(0, int(osl or 0))
    if required <= 0:
        return {"status": "target_shape_unknown"}
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
        return {
            "status": "context_length_absent",
            "required_context_length": required,
        }
    effective = values[-1]
    if effective >= required:
        return {
            "status": "compatible",
            "effective_context_length": effective,
            "required_context_length": required,
        }
    raise ValueError(
        "warm replay context length is incompatible with target workload: "
        f"context_length={effective} < isl+osl={required}"
    )


@dataclass(frozen=True)
class WorkloadTokenWindow:
    """Tokens a workload needs from the server: ISL, OSL and the run's MAX_MODEL_LEN cap."""

    isl: int
    osl: int
    max_model_len: int | str | None = None


def inject_sglang_context_length(
    server_args: str | None,
    framework: str | None,
    model_path: str | None,
    window: WorkloadTokenWindow,
) -> str:
    """Append ``--context-length <N>`` to ``server_args`` for sglang runs."""
    args = str(server_args or "").strip()
    if server_args_env_name(framework) != "EXTRA_SGLANG_ARGS":
        return args
    if _SGLANG_CONTEXT_LENGTH_RE.search(args):
        return args
    from hyperloom.inference_optimizer.model_config_utils import _load_model_max_position_embeddings

    max_pos = _load_model_max_position_embeddings(str(model_path or ""))
    if not max_pos:
        return args
    # AgentX replays a fixed trace corpus, so ISL/OSL are placeholders here and the ISL+OSL+headroom ceiling (8192 at
    # the 1024/1024 defaults) would pin sglang's window two orders of magnitude below what the corpus needs -- every
    # oversized trace then 4xxs.
    from hyperloom.common.perf_metric import agentx_enabled

    if agentx_enabled():
        context_length = int(max_pos)
    else:
        context_length = min(int(max_pos), resolve_sglang_context_cap(window.isl, window.osl))
    max_model_len_int = optional_positive_int(window.max_model_len)
    if max_model_len_int is not None:
        context_length = min(context_length, max_model_len_int)
    return merge_server_args(
        args,
        f"{_SGLANG_CONTEXT_LENGTH_FLAG} {context_length}",
    )


def inject_sglang_attention_backend(
    server_args: str | None,
    framework: str | None,
    model_path: str | None,
) -> str:
    """Append an ``--attention-backend`` for dual-chunk sglang models."""
    args = str(server_args or "").strip()
    if server_args_env_name(framework) != "EXTRA_SGLANG_ARGS":
        return args
    if _SGLANG_ATTN_BACKEND_RE.search(args):
        return args
    from hyperloom.inference_optimizer.model_config_utils import _model_has_dual_chunk_attention

    if not _model_has_dual_chunk_attention(str(model_path or "")):
        return args
    backend = os.environ.get("HYPERLOOM_DUAL_CHUNK_BACKEND", "").strip() or _SGLANG_DUAL_CHUNK_BACKEND
    if backend != _SGLANG_DUAL_CHUNK_BACKEND:
        log.info("dual-chunk model: injecting --attention-backend %s from HYPERLOOM_DUAL_CHUNK_BACKEND.", backend)
    return merge_server_args(
        args,
        f"{_SGLANG_ATTN_BACKEND_FLAG} {backend}",
    )


# sglang MoE runner backend: Hyperloom pins none for the baseline or grid
# variants, the fmoe_ck tuner's validation bench aside (``phases/kernel.py``
# serves it with ``--moe-runner-backend aiter``). sglang's own
# ``--moe-runner-backend auto`` follows ``SGLANG_USE_AITER`` (aiter when the
# harness pre-shuffles MoE weights for it, triton otherwise) on current
# sglang/ROCm images, verified end-to-end on a real MoE checkpoint.
# ``moe_runner_requires_aiter`` below strips an *inherited*
# ``--moe-runner-backend`` that would crash an aiter-only quant scheme (grid
# variants, baseline retries).
#
# Matches space- or equals-separated form without false-matching a longer flag.
SGLANG_MOE_RUNNER_BACKEND_RE = re.compile(r"--moe-runner-backend(?:[=\s]|$)")

# Same flag with the value that follows it, so a removal takes the pair and not just the name.
# The value is optional so a bare, value-less flag (an operator typo, or a flag left dangling at
# end-of-string) is stripped too rather than surviving into a retry that is supposed to launch
# without it.
_SGLANG_MOE_RUNNER_BACKEND_PAIR_RE = re.compile(r"(?:^|\s)--moe-runner-backend(?:(?:[=\s]+)(?!--)\S+)?")


def remove_moe_runner_backend_arg(args: str) -> str:
    """Remove any existing SGLang MoE runner backend flag from an args string."""
    return " ".join(_SGLANG_MOE_RUNNER_BACKEND_PAIR_RE.sub(" ", str(args or "")).split())


# sglang MoE schemes whose ``create_moe_runner`` only builds a runner for the aiter backend (the others fall through
# to a bare ``pass``, so the first forward pass dies on a missing ``runner``).
_AITER_ONLY_ONLINE_QUANT_METHODS = frozenset({"quark_int4fp8_moe"})

_AITER_ONLY_UNLESS_SERIALIZED_QUANT_METHOD = "mxfp4"

_SGLANG_QUANTIZATION_RE = re.compile(r"--quantization[=\s]+(\S+)")


def _online_quant_requires_aiter_moe_runner(server_args: str, model_path: str) -> bool:
    """Whether ``--quantization`` selects an aiter-only MoE scheme."""
    match = _SGLANG_QUANTIZATION_RE.search(server_args or "")
    if not match:
        return False
    quantization = match.group(1).strip().strip("\"'").lower()
    if quantization in _AITER_ONLY_ONLINE_QUANT_METHODS:
        return True
    if quantization != _AITER_ONLY_UNLESS_SERIALIZED_QUANT_METHOD:
        return False
    from hyperloom.inference_optimizer.model_config_utils import _model_declared_quant_method

    # Mirrors sglang: is_checkpoint_mxfp4_serialized = "mxfp4" in quant_method.
    return "mxfp4" not in _model_declared_quant_method(model_path)


def moe_runner_requires_aiter(server_args: str | None, model_path: str | None) -> bool:
    """Whether this model + server args resolve to an aiter-only MoE scheme."""
    from hyperloom.inference_optimizer.model_config_utils import _model_moe_runner_requires_aiter

    path = str(model_path or "")
    return _model_moe_runner_requires_aiter(path) or _online_quant_requires_aiter_moe_runner(
        str(server_args or ""),
        path,
    )


def dedupe_extra_server_args(args_str: str) -> str:
    """Collapse repeated ``--flag value`` pairs into a unique launch string."""
    if not args_str:
        return ""
    parsed = tokenize_server_args_preserving_json(args_str)
    if parsed is None:
        return args_str
    normalized, tokens = parsed
    pair_by_flag: dict[str, list[str]] = {}
    order: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            if "=" in t:
                flag, _, value = t.partition("=")
                values = [value] if value else []
                i += 1
            else:
                flag = t
                i += 1
                values = []
                if flag in _MULTI_VALUE_FLAGS:
                    while i < len(tokens) and not tokens[i].startswith("--"):
                        values.append(tokens[i])
                        i += 1
                elif i < len(tokens) and not tokens[i].startswith("--"):
                    values = [tokens[i]]
                    i += 1
            pair = [flag, *values] if values else [flag]
            if flag not in pair_by_flag:
                order.append(flag)
            pair_by_flag[flag] = pair
        else:
            # Stray positional token; preserve as-is.
            key = f"__positional_{len(order)}__"
            order.append(key)
            pair_by_flag[key] = [t]
            i += 1
    out: list[str] = []
    for k in order:
        out.extend(pair_by_flag[k])
    rendered = " ".join(out)
    return rendered if rendered != normalized else normalized


def merge_cumulative_extra_server_args(
    base_args: str,
    candidate_args: str,
    full_args: str,
) -> str:
    """Build cumulative launch args for a KEEP without double-stacking."""
    base = str(base_args or "").strip()
    candidate = str(candidate_args or "").strip()
    full = str(full_args or "").strip()
    if full and full != candidate:
        merged = full
    elif candidate and base:
        if candidate == base or candidate.startswith(f"{base} ") or base in candidate.split():
            merged = candidate
        else:
            merged = f"{base} {candidate}".strip()
    else:
        merged = candidate or full or base
    return dedupe_extra_server_args(merged)


def _json_with_field_set(blob: str, field: str, value: str) -> str | None:
    """``blob`` with ``field`` set to ``value``, compact; ``None`` when it is not a JSON object."""
    try:
        config = json.loads(blob)
    except json.JSONDecodeError:
        repaired = _repair_unquoted_json(blob)
        config = json.loads(repaired) if repaired is not None else None
    if not isinstance(config, dict):
        return None
    if config.get(field) == value:
        return blob
    config[field] = value
    return json.dumps(config, separators=(",", ":"))


def _with_dotted_config_merged(args: str, flag: str) -> str:
    """``args`` with vLLM's cudagraph mode forced off, keeping every other configured field.

    vLLM collects every dotted ``--compilation-config.<field>`` into one dict and appends that
    dict to argv LAST, so a dotted arg replaces a JSON-form ``--compilation-config`` outright --
    every field, in either order, with no warning. Appending is therefore only correct when
    there is no JSON form to lose; otherwise the field is merged into the existing object here.
    """
    option, off_value = flag.split()
    config_option, field = option.split(".", 1)
    tokens = _split_args_preserving_json(args)
    if tokens is None:
        # Unsplittable args hold no object we could identify to merge into.
        return f"{args} {flag}".strip()
    if "--enforce-eager" in tokens:
        return args

    def _value_of(idx: int) -> tuple[str, int | None]:
        """The option's value and the index holding it (``None`` for the ``--flag=value`` form)."""
        token = tokens[idx]
        if "=" in token:
            return token.split("=", 1)[1], None
        return (tokens[idx + 1], idx + 1) if idx + 1 < len(tokens) else ("", None)

    dotted_mode_idx: int | None = None
    json_idx: int | None = None
    has_other_dotted = False
    for idx, token in enumerate(tokens):
        name = token.split("=", 1)[0]
        if name == option:
            dotted_mode_idx = idx
        elif name.startswith(f"{config_option}."):
            has_other_dotted = True
        elif name == config_option:
            json_idx = idx

    if dotted_mode_idx is not None:
        value, value_idx = _value_of(dotted_mode_idx)
        if value == off_value:
            return args
        if value_idx is None:
            tokens[dotted_mode_idx] = f"{option}={off_value}"
        else:
            tokens[value_idx] = off_value
        return " ".join(tokens)
    if json_idx is None or has_other_dotted:
        # Dotted fields merge with each other, and the operator's own dotted arg is already
        # costing them any JSON form, so appending loses nothing that was not lost already.
        return f"{args} {flag}".strip()

    value, value_idx = _value_of(json_idx)
    merged = _json_with_field_set(value, field, off_value)
    if merged is None:
        log.warning(
            "%s does not hold a JSON object, so capture is disabled by appending %s; "
            "vLLM resolves the pair by discarding the existing value",
            config_option,
            flag,
        )
        return f"{args} {flag}".strip()
    if merged == value:
        return args
    if value_idx is None:
        tokens[json_idx] = f"{config_option}={merged}"
    else:
        tokens[value_idx] = merged
    return " ".join(tokens)


def with_capture_disabled(args: str, flag: str) -> str:
    """``args`` with ``flag`` applied once (idempotent).

    A dotted option (``--config.field value``) is merged into any existing ``--config`` rather
    than appended, because the server resolves the pair by discarding that one; see
    :func:`_with_dotted_config_merged`. A plain flag is appended unless already present, by
    whole token so ``--disable-cuda-graph-extra`` is not mistaken for ``--disable-cuda-graph``.
    """
    if "." in flag.split()[0]:
        return _with_dotted_config_merged(args or "", flag)
    if flag in (args or "").split():
        return args or ""
    return f"{args} {flag}".strip()
