# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Server-argument composition helpers."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
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

    remove_flags: set[str] = set()
    remove_pairs: set[tuple[str, str | None]] = set()
    for spec in removes:
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
                remove_pairs.add((flag, value))
                i += 1
            elif i + 1 < len(spec_tokens) and not spec_tokens[i + 1].startswith("--"):
                remove_pairs.add((tok, spec_tokens[i + 1]))
                i += 2
            else:
                remove_flags.add(tok)
                i += 1

    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        flag = tok.split("=", 1)[0] if tok.startswith("--") else ""
        if flag and "=" in tok:
            _flag, _, value = tok.partition("=")
            if _flag in remove_flags or (_flag, value) in remove_pairs:
                i += 1
                continue
        if flag and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            value = tokens[i + 1]
            if flag in remove_flags or (flag, value) in remove_pairs:
                i += 2
                continue
        if flag and flag in remove_flags:
            i += 1
            continue
        out.append(tok)
        i += 1
    # No re-serialisation on the way out: the tokens are already JSON-compacted and the non-POSIX split kept each one
    # byte-for-byte, so re-joining the survivors cannot corrupt a sibling flag.
    return " ".join(out)


# Serving-ineligible harness flags.
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
    result = strip_benchmark_harness_flags(composed)
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
    try:
        was = _json_flag_values(before)
        now = _json_flag_values(after)
    except Exception:  # never let a diagnostic break composition
        return
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
    except Exception:
        return False
    return True


# A JSON "bareword": an identifier-like token that appears where a double-quoted JSON key or string value should be
# (letters/digits/underscore plus the ``.``, ``/``, ``-`` common in model ids and paths).
_JSON_BAREWORD = r"[+-]?[A-Za-z_][A-Za-z0-9_./-]*"
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
    except Exception:
        return None


def compact_json_server_args(
    server_args: str | None,
    framework: str | None,
) -> str:
    """Normalize JSON-valued server args for unquoted Magpie expansion."""
    args = str(server_args or "").strip()
    if not args or ("{" not in args and "[" not in args):
        return args
    return _reserialize_json_blobs(args)


def _reserialize_json_blobs(args: str) -> str:
    """Normalize every JSON object/array while preserving invalid substrings."""
    if "{" not in args and "[" not in args:
        return args
    out: list[str] = []
    i = 0
    n = len(args)
    while i < n:
        ch = args[i]
        if ch in "{[":
            # A prior shlex.join can wrap a JSON token in shell single quotes.
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
                # A prior shlex round-trip can strip the JSON double quotes, leaving an unquoted-bareword object
                # (``{"m":"ngram"}`` -> ``{m:ngram}``) that vLLM's json.loads rejects at boot.
                rendered = _repair_unquoted_json(blob)
            if rendered is None:
                # Keep both wrappers when the content is not valid/repairable.
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
# Compatibility export used by ``_grid_runner`` and out-of-tree tests.
_SPACE_VALUE_FLAGS = SPACE_VALUE_FLAGS

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


def _shell_safe_dedupe(args: str) -> str:
    """Last-wins dedupe for single-token-valued flags only."""
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


# sglang scheduler watchdog timeout injection: the first request's JIT compile can exceed sglang's default watchdog,
# firing SIGQUIT mid-warmup.
DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC = 1800

SGLANG_WATCHDOG_TIMEOUT_ENV = "SGLANG_WATCHDOG_TIMEOUT"

_SGLANG_WATCHDOG_FLAG = "--watchdog-timeout"

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
    return merge_server_args(args, f"{_SGLANG_WATCHDOG_FLAG} {timeout}")


# sglang ``--context-length`` cap injection: sglang sizes ``max_total_tokens`` off the model's
# ``max_position_embeddings``, so a huge native window balloons the aiter workspace_buffer past GPU memory.
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
) -> tuple[str, dict[str, Any]]:
    """Validate a replayed SGLang context window without changing its config."""
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
    """Append ``--context-length <N>`` to ``server_args`` for sglang runs."""
    args = str(server_args or "").strip()
    if server_args_env_name(framework) != "EXTRA_SGLANG_ARGS":
        return args
    if _SGLANG_CONTEXT_LENGTH_RE.search(args):
        return args
    from hyperloom.inference_optimizer.cli.model_gate import _load_model_max_position_embeddings

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
        context_length = min(int(max_pos), resolve_sglang_context_cap(isl, osl))
    max_model_len_int = optional_positive_int(max_model_len)
    if max_model_len_int is not None:
        context_length = min(context_length, max_model_len_int)
    return merge_server_args(
        args,
        f"{_SGLANG_CONTEXT_LENGTH_FLAG} {context_length}",
    )


def _resolve_dual_chunk_backend(gpu_type: str | None = None) -> str:
    """Pick the dual-chunk attention backend for the current hardware."""
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
    """Append an ``--attention-backend`` for dual-chunk sglang models."""
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


# sglang MoE runner backend: Hyperloom no longer forces a backend. sglang's own
# ``--moe-runner-backend auto`` correctly follows ``SGLANG_USE_AITER`` (aiter
# when the harness pre-shuffles MoE weights for it, triton otherwise) without
# crashing on current sglang/ROCm images; verified end-to-end on a real MoE
# checkpoint before this override was removed. ``moe_runner_requires_aiter``
# below is still used to strip an *inherited* ``--moe-runner-backend`` that
# would crash an aiter-only quant scheme (grid variants, baseline retries).
_SGLANG_MOE_RUNNER_BACKEND_FLAG = "--moe-runner-backend"

# Matches space- or equals-separated form without false-matching a longer flag.
_SGLANG_MOE_RUNNER_BACKEND_RE = re.compile(r"--moe-runner-backend(?:[=\s]|$)")

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
    from hyperloom.inference_optimizer.cli.model_gate import _model_declared_quant_method

    # Mirrors sglang: is_checkpoint_mxfp4_serialized = "mxfp4" in quant_method.
    return "mxfp4" not in _model_declared_quant_method(model_path)


def moe_runner_requires_aiter(server_args: str | None, model_path: str | None) -> bool:
    """Whether this model + server args resolve to an aiter-only MoE scheme."""
    from hyperloom.inference_optimizer.cli.model_gate import _model_moe_runner_requires_aiter

    path = str(model_path or "")
    return _model_moe_runner_requires_aiter(path) or _online_quant_requires_aiter_moe_runner(
        str(server_args or ""),
        path,
    )
