# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared helper for the ``explore`` executor's grid runs.

Takes a base Magpie YAML + a list of (name, extra_server_args, extra_envs)
variants, runs Magpie once per variant, parses ``benchmark_report.json``,
returns the winners.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
from typing import Any

from hyperloom.common.coerce import optional_positive_int


log = logging.getLogger(__name__)

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
    # Substring fallback for version-suffixed names; check longest/specific
    # names first so an overlapping substring cannot match the wrong branch.
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


def _coerce_str_list(value: Any) -> list[str]:
    """Normalize optional string/list controls to non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def remove_server_args(server_args: str | None, remove_args: Any) -> str:
    """Remove flag specs from a server-arg string.

    ``remove_args`` entries are flag-oriented. ``"--foo"`` removes ``--foo`` and
    its following value when one is present; ``"--foo=bar"`` removes that exact
    token shape; ``"--foo bar"`` removes the exact flag/value pair. Unknown /
    unparseable inputs are left untouched rather than guessed.
    """
    args = str(server_args or "").strip()
    removes = _coerce_str_list(remove_args)
    if not args or not removes:
        return args
    try:
        tokens = shlex.split(args)
    except ValueError:
        return args

    remove_flags: set[str] = set()
    remove_pairs: set[tuple[str, str | None]] = set()
    for spec in removes:
        try:
            spec_tokens = shlex.split(spec)
        except ValueError:
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
    return " ".join(out)


def compose_server_args(
    *,
    inherited_args: str | None = "",
    base_extra_args: str | None = "",
    variant_extra_args: str | None = "",
    remove_args: Any = None,
    args_mode: str = "append",
) -> str:
    """Compose inherited/base/variant args with optional remove/replace semantics."""
    mode = str(args_mode or "append").strip().lower()
    if mode == "replace":
        pruned_base = remove_server_args(base_extra_args, remove_args)
        pruned_variant = remove_server_args(variant_extra_args, remove_args)
        return merge_server_args(pruned_base, pruned_variant)
    combined_base = merge_server_args(inherited_args, base_extra_args)
    pruned = remove_server_args(combined_base, remove_args)
    return merge_server_args(pruned, variant_extra_args)

def compact_json_server_args(
    server_args: str | None,
    framework: str | None,
) -> str:
    """Strip separator spaces inside JSON-valued vLLM/atom server args.

    Magpie's scripts expand ``vllm serve ... $EXTRA_VLLM_ARGS`` UNQUOTED, so any
    token with an embedded space is word-split by the shell before vLLM sees it.
    LLM/specialist explore variants routinely pass JSON-valued flags with the
    conventional separator spaces, e.g. ``--compilation-config '{"full_cuda_
    graph": true}'`` or ``--speculative-config '{"method": "eagle"}'``; the
    space after ``:`` (or ``,``) splits the value into two shell words and the
    server dies at boot with an argparse/JSON error (observed: ``Value
    {"method": cannot be converted``). Re-serialising each JSON object/array
    with compact separators removes the SEPARATOR spaces only — ``json.dumps``
    deliberately preserves spaces *inside* string values. This mirrors the
    deliberately space-free ``--hf-overrides`` injection in ``_workload_envs``
    and is the same round-trip-safe contract the dedup helpers rely on.

    LIMITATION — separator spaces only: the value becomes a single shell word
    only when the JSON has no space *inside* a string value. A value like
    ``{"model":"draft model name"}`` still contains a literal space, and under
    Magpie's unquoted ``$EXTRA_VLLM_ARGS`` expansion that space word-splits no
    matter what — there is no way to keep a space-bearing token as one word via
    unquoted expansion. Such flags (string values holding spaces — a free-text
    name, a path with spaces) are NOT supported: they are left intact rather
    than corrupted (stripping the inner space would change the value), and the
    server will still fail to boot. Callers must avoid space-bearing JSON string
    values for vLLM/atom. See
    ``test_compact_json_server_args_internal_space_unsupported``.

    No-op for sglang (its Magpie path differs and tolerates the quoting), for
    empty strings, and for strings with no ``{``/``[``. Any blob that does not
    parse as JSON is left verbatim (vLLM reports it rather than us mangling it).
    """
    args = str(server_args or "").strip()
    if not args or ("{" not in args and "[" not in args):
        return args
    if server_args_env_name(framework) == "EXTRA_SGLANG_ARGS":
        return args
    out: list[str] = []
    i = 0
    n = len(args)
    while i < n:
        ch = args[i]
        if ch in "{[":
            # Walk to the balanced close, honouring quoted strings so braces
            # inside a JSON string value do not throw off the depth count.
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
            blob = args[i:j]
            try:
                out.append(json.dumps(json.loads(blob), separators=(",", ":")))
            except Exception:
                out.append(blob)
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out)

# Flags whose value can contain spaces / JSON; never tokenize-dedupe these
# because the downstream Magpie scripts expand $EXTRA_*_ARGS unquoted and the
# value would be word-split anyway. If any is present, the dedup helpers leave
# the WHOLE arg string untouched (the only behaviour that round-trips safely).
_SPACE_VALUE_FLAGS = (
    "--json-model-override-args",
    "--override-generation-config",
    "--tool-call-parser",
    # JSON-object-valued flags: after ``compact_json_server_args`` these are a
    # single space-free shell word, but their value still contains inner double
    # quotes (``{"cudagraph_mode":"PIECEWISE"}``). ``dedup_vllm_server_args``
    # tokenizes with ``shlex.split`` (which STRIPS those quotes) and rejoins
    # without re-quoting, corrupting the JSON to ``{cudagraph_mode:PIECEWISE}``
    # -> vLLM boot fails with ``Invalid JSON``. Listing them here makes both
    # dedup helpers leave the whole arg string untouched (round-trip safe), the
    # same contract already relied on for the flags above.
    "--compilation-config",
    "--speculative-config",
    "--hf-overrides",
    "--kv-transfer-config",
)

_MULTI_VALUE_FLAGS = (
    "--cuda-graph-bs",
    "--cuda-graph-max-bs",
)

# vLLM / atom expose argparse-style single-value options that are safe to
# collapse last-wins. Unlike sglang (where a repeated flag is harmless
# last-wins), vLLM hard-errors on a duplicate / conflicting
# ``--attention-backend`` — the crash propagates through ``EngineCoreClient``
# -> ``wait_for_engine_startup`` -> RuntimeError, then NCCL ``Broken pipe`` on
# every rank. The duplication arises when
# the operator's YAML ``EXTRA_VLLM_ARGS`` already pins a flag and a sweep /
# kernel variant appends the same flag via ``extra_server_args``; ``merge_
# server_args`` keeps BOTH tokens by design. This set lists the single-value
# flags safe to collapse to last-wins so the variant override survives.
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

def dedup_vllm_server_args(
    server_args: str | None,
    framework: str | None,
) -> str:
    """Collapse repeated vLLM/atom single-value flags to last-wins (#520).

    vLLM v0.21.0 crashes ``EngineCoreProc`` on a duplicated
    ``--attention-backend`` (and conflicting copies of other single-value
    flags); sglang tolerates repeats. So this is scoped to the vllm/atom
    framework envs and is a no-op for sglang. Only the flags in
    :data:`_VLLM_SINGLE_VALUE_FLAGS` are touched; every other token (unknown
    flags, store-true switches, positional values) is preserved verbatim and
    in order. For each affected flag the LAST occurrence wins, matching the
    left-to-right override intent of :func:`merge_server_args` (so a
    sweep/kernel ``extra_args`` value overrides the YAML base).

    Returns ``server_args`` unchanged when the framework is sglang, the string
    is empty, it carries a space/JSON-valued flag (see
    :data:`_SPACE_VALUE_FLAGS` / :data:`_MULTI_VALUE_FLAGS`), or it cannot be
    shell-parsed.

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
    # Never tokenize a string carrying a space/JSON-valued (or multi-value)
    # flag: the unquoted ``$EXTRA_*_ARGS`` expansion in Magpie's scripts cannot
    # round-trip a value with embedded spaces, so re-joining ``shlex`` tokens
    # would corrupt it (e.g. ``--override-generation-config '{"temperature":
    # 0.7}'`` would be split into separate words). Mirrors the guard in
    # :func:`_shell_safe_dedupe`; leave the whole string untouched.
    if any(f in args for f in _SPACE_VALUE_FLAGS + _MULTI_VALUE_FLAGS):
        return args
    try:
        tokens = shlex.split(args)
    except ValueError:
        # Unparseable: leave it to vLLM to report rather than mangling it.
        return args
    # Collect the token span of every recognized single-value flag.
    spans: list[tuple[str, int, int]] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        name = tok.split("=", 1)[0] if tok.startswith("--") else None
        if name in _VLLM_SINGLE_VALUE_FLAGS:
            if "=" in tok:
                # ``--flag=value`` is self-contained.
                spans.append((name, i, i))
                i += 1
            elif i + 1 < n and not tokens[i + 1].startswith("-"):
                # ``--flag value`` consumes the following token.
                spans.append((name, i, i + 1))
                i += 2
            else:
                # Bare ``--flag`` with no value: treat as a single token.
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
        return args
    kept = [tok for idx, tok in enumerate(tokens) if idx not in drop]
    return " ".join(kept)

def _shell_safe_dedupe(args: str) -> str:
    """Last-wins dedupe for single-token-valued flags only.

    Targeted and conservative: collapses repeated ``--flag value`` (or
    ``--flag=value``) pairs whose value is a single whitespace-free token,
    keeping the last occurrence. If the string contains a flag known to carry
    a space/JSON value (which the unquoted ``$EXTRA_*_ARGS`` expansion in the
    Magpie scripts cannot safely round-trip anyway), the string is returned
    unchanged to avoid mangling it.

    Args:
        args (str): The server-arg string to dedupe.

    Returns:
        str: The last-wins deduped string, or the input unchanged when it
        carries a space/JSON-valued flag.
    """
    if not args.strip():
        return ""
    if any(f in args for f in _SPACE_VALUE_FLAGS + _MULTI_VALUE_FLAGS):
        return args
    tokens = args.split()
    pairs: dict[str, list[str]] = {}
    order: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            if "=" in t:
                # Normalize ``--flag=value`` so it dedupes against ``--flag value``.
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
    return " ".join(out)

# sglang scheduler watchdog timeout injection: on MI300X with aiter, the first
# request's ``mha_batch_prefill`` JIT compile can exceed sglang's 300s default
# watchdog, firing SIGQUIT mid-warmup -> baseline_failed. Inject a longer
# timeout via ``EXTRA_SGLANG_ARGS`` unless the user already pinned one.
DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC = 1800

SGLANG_WATCHDOG_TIMEOUT_ENV = "SGLANG_WATCHDOG_TIMEOUT"

_SGLANG_WATCHDOG_FLAG = "--watchdog-timeout"

# Matches space- or equals-separated form so a user-pinned value suppresses
# injection without false-matching a longer flag.
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
# off the model's ``max_position_embeddings``, so a huge native window (e.g.
# Mistral-Nemo's 1024000) balloons the aiter workspace_buffer past GPU memory
# -> HIP OOM -> baseline_failed. vllm already caps via ``--max-model-len``, so
# this fixes the sglang-only asymmetry: cap to ISL+OSL+headroom (floored,
# clamped to the native window) unless the flag is already pinned.
DEFAULT_SGLANG_CONTEXT_HEADROOM_TOKENS = 2048

DEFAULT_SGLANG_CONTEXT_FLOOR_TOKENS = 8192

SGLANG_CONTEXT_HEADROOM_ENV = "SGLANG_CONTEXT_HEADROOM_TOKENS"

SGLANG_CONTEXT_FLOOR_ENV = "SGLANG_CONTEXT_FLOOR_TOKENS"

_SGLANG_CONTEXT_LENGTH_FLAG = "--context-length"

# Matches space- or equals-separated form so an operator-pinned value
# suppresses injection without false-matching a longer flag.
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
    explicit ``--max-model-len`` would inject a self-contradictory config (#697:
    ``--context-length 84048`` while ``--max-model-len 82000``). The
    ``max_model_len`` ceiling is applied only when it is a positive value;
    an unset / non-positive value preserves the prior ``min(max_pos, cap)``
    behaviour.

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
    cap = resolve_sglang_context_cap(isl, osl)
    context_length = min(int(max_pos), cap)
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
    model declares ``dual_chunk_attention_config``. It requires sm90+
    (NVIDIA Hopper); on AMD/ROCm the preflight gate
    (``_detect_incompatible_model_config``) blocks these models before
    they reach this point. If a session somehow arrives here on AMD
    (e.g. operator override), return the canonical backend and let sglang
    raise the clear error rather than silently injecting triton which
    sglang also rejects.  Override via ``$HYPERLOOM_DUAL_CHUNK_BACKEND``.

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

    Models that declare ``dual_chunk_attention_config`` (Qwen 1M) make
    sglang hard-reject its default aiter backend with ``ValueError: Dual
    chunk attention is enabled, but attention backend is set to aiter.``.
    On NVIDIA sm90+ the fix is ``dual_chunk_flash_attn``; on AMD/ROCm that
    kernel is unsupported (``sm90 and above``), so we inject ``triton``
    instead (see :func:`_resolve_dual_chunk_backend`). ``gpu_type`` (when
    known by the caller) takes precedence over runtime autodetect.

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

# sglang MoE runner backend injection: on MI300X/MI355X with aiter, sglang's
# default ``--moe-runner-backend auto`` routes Mixture-of-Experts models
# through aiter's CK 2-stage fused-MoE kernel. Its first-request JIT build
# (``module_moe_ck2stages_*``) is broken in some ROCm images — ``thrust`` pulls
# in a missing ``<cub/detail/detect_cuda_runtime.cuh>`` so hipcc fails to
# compile, and the killed build leaves a stale lock that makes the next
# attempts hang on "waiting for baton release" until sglang's 600s warmup
# read-timeout fires -> baseline_failed. ``triton`` is the ROCm-capable
# fused-MoE backend sglang itself falls back to (same "aiter CK kernel doesn't
# support all GEMM dimensions" reason), so inject it for MoE models on AMD
# unless the operator already pinned a backend. Override via
# ``$HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND``.
HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND_ENV = "HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND"

DEFAULT_SGLANG_AMD_MOE_RUNNER_BACKEND = "triton"

_SGLANG_MOE_RUNNER_BACKEND_FLAG = "--moe-runner-backend"

# Matches space- or equals-separated form so a user-pinned value suppresses
# injection without false-matching a longer flag.
_SGLANG_MOE_RUNNER_BACKEND_RE = re.compile(r"--moe-runner-backend(?:[=\s]|$)")

def inject_sglang_moe_runner_backend(
    server_args: str | None,
    framework: str | None,
    model_path: str | None,
    gpu_type: str | None = None,
) -> str:
    """Append a ``--moe-runner-backend`` for MoE sglang models on AMD/ROCm.

    Returns ``server_args`` unchanged when: framework is not sglang, a
    ``--moe-runner-backend`` is already pinned (operator wins), the GPU is not
    an AMD/ROCm runner, or the model is not Mixture-of-Experts (fail-safe:
    inject nothing). Otherwise appends the backend from
    ``$HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND`` (default ``triton``); only this
    flag is added.

    Args:
        server_args (str | None): The server-arg string to augment.
        framework (str | None): Framework name; empty/unknown treated as sglang.
        model_path (str | None): Model path checked for Mixture-of-Experts.
        gpu_type (str | None): Caller-known GPU type; used to gate AMD/ROCm.

    Returns:
        str: ``server_args`` with ``--moe-runner-backend`` appended, or unchanged
        for non-sglang frameworks, when already pinned, off AMD/ROCm, or for
        non-MoE models.
    """
    args = str(server_args or "").strip()
    if server_args_env_name(framework) != "EXTRA_SGLANG_ARGS":
        return args
    if _SGLANG_MOE_RUNNER_BACKEND_RE.search(args):
        return args
    from hyperloom.inference_optimizer.cli.model_gate import _model_is_moe, _resolve_amd_gpu_type

    if not _resolve_amd_gpu_type(gpu_type):
        return args
    if not _model_is_moe(str(model_path or "")):
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

    Single shared path for baseline/profile and grid executors so
    backends/params/sweep no longer fall back to shipped YAML defaults.
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
        # hardcode ``--result-dir /workspace/`` and ignore ``EXTRA_*_ARGS``.
        # See ``design/magpie-generic-script-and-user-data-path.md``.
        framework = str(bench.get("framework") or "").lower()
        if framework:
            bench["benchmark_script"] = f"{framework}_{gpu_type}.sh"
        else:
            bench.pop("benchmark_script", None)

    if benchmark_script:
        bench["benchmark_script"] = str(benchmark_script)

    envs = bench.setdefault("envs", {})
    for env_key in ("ISL", "OSL", "MAX_MODEL_LEN", "TP", "CONC"):
        val = os.environ.get(env_key, "").strip()
        if not val:
            continue
        # TP yaml-explicit wins: a stale state.tp re-exported on resume must
        # not downgrade a YAML-pinned TP.
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
