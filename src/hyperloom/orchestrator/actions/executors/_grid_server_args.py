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
from hyperloom.inference_optimizer.framework_registry import server_args_env_name


log = logging.getLogger(__name__)

_UNSAFE_SERVER_ARG_CHARS_RE = re.compile(r"[;&|`$<>\r\n]")
#: What the 2nd..nth value of a not-yet-whitelisted multi-value flag may look
#: like: a batch size, a length, a ratio. Deliberately not a general token.
_NUMERIC_VALUE_RE = re.compile(r"^\d+(?:[.,]\d+)*$")


def validate_server_args_shell_safe(server_args: str | None) -> str:
    """Reject server-arg strings that would be shell control syntax.

    Magpie benchmark scripts expand ``EXTRA_*_ARGS`` through shell wrappers, so
    this is the final sink-side guard against LLM/payload content escaping from
    argv-like flags into shell control operators.

    A flag may be followed by more than one value token (argparse ``nargs="+"``
    semantics). ``--cuda-graph-bs 1 2 4 8`` is a real sglang invocation and is
    already recognized as multi-valued by :data:`_MULTI_VALUE_FLAGS`; requiring
    exactly one value here rejected it at the sink while the explore side let it
    through.

    The relaxation is scoped rather than blanket, because "any flag anywhere
    earlier permits any bare token afterwards" stops rejecting anything at all:

    * ``--flag=value`` already carries its value, so a bare token after it is
      unambiguously positional.
    * a flag in :data:`_MULTI_VALUE_FLAGS` takes an unlimited value list -- but
      every entry in that whitelist is a list of batch sizes, so the list is
      still digits-only. Letting the whitelist waive the token shape as well
      would readmit ``--cuda-graph-bs 1 2 run.sh``.
    * any other flag takes one arbitrary value; further tokens are accepted only
      while they still look like list elements (digits), never as bare words.
      That covers an ``nargs="+"`` flag not yet on the whitelist -- those carry
      batch sizes or lengths -- without readmitting ``--foo bar some_script.sh``.

    Shell control characters are blocked separately above, so this remains a
    secondary "this looks like argv" guard.
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


def _unwrap_one_pair(s: str) -> str:
    """Strip one balanced pair of outer shell quotes from *s* when safe to do so.

    JSON blobs (inner content starts with ``{`` or ``[``) are never touched
    because they contain double quotes that must survive into the downstream arg.
    """
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        inner = s[1:-1]
        if inner and not inner.startswith(("{", "[")) and s[0] not in inner and not any(ch.isspace() for ch in inner):
            return inner
    return s


def _unwrap_shell_quotes(token: str) -> str:
    """Drop one balanced pair of shell quotes from a token produced by shlex.

    ``shlex.split(..., posix=False)`` keeps quote bytes in the token it returns,
    which is exactly what protects a JSON value's inner double quotes. But a
    plain operand written as ``--tool-call-parser 'kimi'`` or
    ``--tool-call-parser='kimi'`` must not keep its wrappers, because Magpie
    expands ``EXTRA_*_ARGS`` unquoted and the wrappers would reach argv
    literally.

    For a leading-dash token the unwrap is applied to the right-hand side of
    the first ``=`` only, so token boundaries never shift and the flag name is
    never altered. A JSON value (starting with ``{`` or ``[``) is left verbatim
    in both positions.
    """
    if token.startswith("-") and "=" in token:
        flag, _, value = token.partition("=")
        unwrapped = _unwrap_one_pair(value)
        return f"{flag}={unwrapped}" if unwrapped != value else token
    return _unwrap_one_pair(token)


def _split_args_preserving_json(text: str) -> list[str] | None:
    """Tokenize a server-arg string WITHOUT stripping JSON's inner double quotes.

    ``shlex.split(text)`` defaults to ``posix=True``, which consumes every quote
    byte. Splitting a JSON-valued flag that way and space-joining the result
    turns a stored-valid ``--compilation-config {"mode":3}`` into
    ``{mode:3}``, and vLLM then aborts at argv parse with
    ``Invalid JSON: key must be a string``. Valid blobs are compacted first (so
    a blob is a single whitespace-free word) and then split in non-POSIX mode,
    which preserves the quote bytes verbatim — a lossless round trip.

    Returns ``None`` when the string is not tokenizable at all, so callers can
    leave the input untouched rather than guess.

    Deliberately not :func:`tokenize_server_args_preserving_json`, which shares
    the same tokenizing core but a different contract: it returns the tokens
    raw for a caller that inspects them, and fails closed when a blob was split
    by embedded whitespace. Rewriting a string means the wrappers must come off
    (see :func:`_unwrap_shell_quotes`), and a removal must degrade to "leave it
    alone" rather than reject the whole string — ``strip_benchmark_harness_flags``
    routes every composed variant through here.
    """
    try:
        tokens = shlex.split(_reserialize_json_blobs(text), posix=False)
    except ValueError:
        return None
    return [_unwrap_shell_quotes(tok) for tok in tokens]


def remove_server_args(server_args: str | None, remove_args: Any) -> str:
    """Remove flag specs from a server-arg string.

    ``remove_args`` entries are flag-oriented. ``"--foo"`` removes ``--foo`` and
    its following value when one is present; ``"--foo=bar"`` removes that exact
    token shape; ``"--foo bar"`` removes the exact flag/value pair. Unknown /
    unparseable inputs are left untouched rather than guessed.

    Tokenization is quote-preserving (:func:`_split_args_preserving_json`), so
    every flag this function does NOT remove survives byte-for-byte, JSON values
    included. This matters far beyond explicit removals:
    :func:`strip_benchmark_harness_flags` routes EVERY composed variant through
    here with a non-empty denylist, so a lossy round trip would corrupt a
    sibling ``--compilation-config`` even for a variant that removes nothing.
    """
    # Normalized up front as well as inside the tokenizer, so the
    # nothing-to-remove early return below hands back the same shape a caller
    # with a non-empty denylist would get. Invalid substrings are preserved
    # byte-for-byte, and the pass is idempotent, so the second application
    # inside :func:`_split_args_preserving_json` is a no-op.
    args = _reserialize_json_blobs(str(server_args or "").strip())
    removes = to_str_list(remove_args)
    if not args or not removes:
        return args
    # Non-POSIX split plus the wrapper strip: a plain operand written as
    # ``--tool-call-parser 'kimi_k3'`` -- or ``--tool-call-parser='kimi_k3'`` --
    # must not keep its quotes, because Magpie expands EXTRA_*_ARGS unquoted and
    # they would reach argv literally. _unwrap_shell_quotes only touches
    # whitespace-free content, and on a leading-dash token only the value side of
    # the first ``=``, so a JSON blob (starts with ``{``/``[``) is never affected
    # and token boundaries cannot shift.
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
    # No re-serialisation on the way out: the tokens are already JSON-compacted
    # and the non-POSIX split kept each one byte-for-byte, so re-joining the
    # survivors cannot corrupt a sibling flag. After-the-fact re-quoting was
    # never a workable alternative -- _repair_unquoted_json has to guess where
    # the quotes went, and a value like ``["+fused_rms_norm_gated"]`` (whose
    # ``+`` the heuristic cannot reconstruct) is unrecoverable once damaged.
    return " ".join(out)


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
    # Compare against the RAW inputs, not against ``composed``. The tripwire
    # exists to catch a lossy round trip inside ``remove_server_args`` -- and
    # ``composed`` is already that function's output, so damage done there makes
    # the "before" side unparseable too, ``healthy_before`` False, and the
    # tripwire silent on exactly the failure it was written for. The one or two
    # earlier removal calls are inside the window now. Flags the removal specs
    # deliberately dropped are not reported: the loop walks what survived.
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
    """Log loudly when composition turned a parseable JSON flag value unparseable.

    This is a regression tripwire, not a repair. The composer damaged
    ``--compilation-config`` for an entire optimization session by shlex
    round-tripping it lossily: the value stayed a single shell word, so nothing
    downstream looked wrong, and the only symptom was every variant server dying
    at argv parse ~18s in while the baseline (which never routes through
    :func:`compose_server_args`) ran clean for 4206s. The damage was silent
    because ``_repair_unquoted_json`` "succeeded" on the sibling
    ``--speculative-config`` and merely returned ``None`` for the one blob it
    could not reconstruct. Emitting a loud, greppable line here converts that
    class of failure from a multi-round mystery into one log grep.
    """
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


# A JSON "bareword": an identifier-like token that appears where a double-quoted
# JSON key or string value should be (letters/digits/underscore plus the ``.``,
# ``/``, ``-`` common in model ids and paths). An optional leading ``+``/``-``
# sign covers vLLM's custom-op toggles (``custom_ops:["+fused_rms_norm_gated"]``)
# — without it the repair silently failed on exactly those values. A sign is
# only accepted when a letter/underscore follows, so numbers (``-1``) and
# ``true``/``false``/``null`` are still handled separately and stay unquoted.
_JSON_BAREWORD = r"[+-]?[A-Za-z_][A-Za-z0-9_./-]*"
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

    It is NOT the fix for the composer: :func:`remove_server_args` no longer
    damages JSON in the first place (it tokenizes quote-preservingly). This
    remains only as a recovery layer for strings that were already persisted in
    damaged form by the earlier lossy round trip, or that arrive damaged from
    another producer. Never rely on it for newly composed args — a blob is only
    repairable when every stripped-quote value happens to be re-quotable, which
    is not decidable in general.
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


# Flags whose values may be JSON or otherwise space-bearing. Kept public so the
# coordinator and launch paths share one catalogue; JSON presence must NOT make
# dedup abandon the entire arg string. Actual whitespace-bearing argv tokens are
# detected by :func:`tokenize_server_args_preserving_json` and fail closed.
SPACE_VALUE_FLAGS = (
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
# Compatibility export used by ``_grid_runner`` and out-of-tree tests.
_SPACE_VALUE_FLAGS = SPACE_VALUE_FLAGS

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
    for token in tokens:
        # A balanced JSON value must remain one token. Non-zero depth at a token
        # boundary means an embedded whitespace split it.
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
        # ``shlex.split(..., posix=False)`` can fracture a quoted operand with
        # whitespace into edge-quoted pieces (``"my`` / ``parser"``). Reject
        # any such edge, not only a token carrying both wrappers.
        if token.startswith(("'", '"')) or token.endswith(("'", '"')):
            return None
    return normalized, tokens


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
    conc: Any = None,
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

    apply_agentx_switch(bench, model_path, conc=conc)

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
