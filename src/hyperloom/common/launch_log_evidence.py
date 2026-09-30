# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bounded, dependency-free server-launch log evidence helpers."""

from __future__ import annotations

import ast
import hashlib
import logging
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


log = logging.getLogger(__name__)

#: Launch flags that are RUN-/TOPOLOGY-specific (host, device set, model path,
#: parallelism, ports, seeds); stripped from forwarded server-launch flags.
_RUN_SPECIFIC_LAUNCH_FLAGS: frozenset[str] = frozenset(
    {
        "--model-path",
        # vLLM's own spelling. Without it the projected argv keeps a run-local
        # model operand the SGLang spelling has always had stripped.
        "--model",
        "--tokenizer",
        "--tokenizer-path",
        "--served-model-name",
        "--host",
        "--port",
        "--nccl-port",
        "--dist-init-addr",
        "--base-gpu-id",
        "--gpu-id-step",
        "--node-rank",
        "--nnodes",
        "--tensor-parallel-size",
        "--tp-size",
        "--tp",
        "--data-parallel-size",
        "--dp-size",
        "--pipeline-parallel-size",
        "--pp-size",
        "--random-seed",
        "--download-dir",
        "--pid",
    }
)

#: Profiling-only flags are not part of a clean throughput baseline.
_PROFILING_LAUNCH_FLAGS: frozenset[str] = frozenset(
    {
        "--enable-profile-cuda-graph",
        "--enable-shape-discovery-for-cuda-graph-profile",
        "--enable-profile",
        "--enable-torch-compile-debug-mode",
        "--debug-cuda-graph",
    }
)

#: Spellings of one knob, folded to the name SGLang reports: it accepts
#: ``--tensor-parallel-size`` but reports ``tp_size``.
_SGLANG_FLAG_ALIASES: dict[str, str] = {
    "--tp": "--tp-size",
    "--tensor-parallel-size": "--tp-size",
    "--dp": "--dp-size",
    "--data-parallel-size": "--dp-size",
    "--pipeline-parallel-size": "--pp-size",
}

#: Per-backend marker for the start of a captured launch argv.
_LAUNCH_ARGV_MARKERS: dict[str, str] = {
    "sglang": "launch_server",
    "vllm": "vllm",
}


def split_launch_flags(argv_tail: str) -> str:
    """Remove run-specific and profiling flags from a captured launch argv."""
    try:
        tokens = shlex.split(argv_tail)
    except ValueError:
        tokens = argv_tail.split()
    kept: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        flag = token.split("=", 1)[0]
        # ``vllm serve <model>`` carries the model as a positional, so the
        # flag list above cannot reach it. Left in, it would travel as an
        # observed launch flag -- a host model path in the durable record, and
        # a term the requested side can never match.
        if token == "serve":
            index += 1
            if index < len(tokens) and not tokens[index].startswith("-"):
                index += 1
            continue
        if flag in _RUN_SPECIFIC_LAUNCH_FLAGS or flag in _PROFILING_LAUNCH_FLAGS:
            if "=" not in token and index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
                index += 2
            else:
                index += 1
            continue
        kept.append(token)
        index += 1
    return " ".join(kept)


def launch_flag_setting_name(flag: str, framework: str) -> str:
    """The setting key a launch flag sets under ``framework``.

    Under SGLang ``--dp`` and ``--dp-size`` both set ``dp_size``; an engine with
    no record folds nothing, which is the spelling the flag already carries.
    """
    spec = _LAUNCH_RECORDS.get(str(framework or "").strip().lower())
    aliases = spec.flag_aliases if spec else {}
    return aliases.get(flag, flag).lstrip("-").replace("-", "_")


def launch_argv_from_log(path: str, framework: str) -> str:
    """Extract and normalize the engine launch argv from one benchmark log."""
    marker = _LAUNCH_ARGV_MARKERS.get(str(framework or "").strip().lower())
    if not marker:
        return ""
    pattern = re.compile(r"(?:-m\s+\S*" + re.escape(marker) + r"\S*|" + re.escape(marker) + r")\b(.*)$")
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if marker not in line:
                    continue
                match = pattern.search(line)
                tail = (match.group(1) if match else "").strip()
                if not tail:
                    start = line.find("--")
                    tail = line[start:].strip() if start >= 0 else ""
                # The gate says "this line IS the launch command", and it says
                # it by naming the model served. Keyed on ``--model-path``
                # alone it was a gate only SGLang could pass: vLLM writes
                # ``--model <m>`` or ``serve <m>``, so every vLLM session
                # produced empty observed flags, every requested flag was read
                # as absent, and the verdict was insufficient by construction
                # rather than by evidence.
                if not _names_a_model(tail):
                    continue
                flags = split_launch_flags(tail)
                if flags:
                    return flags
    except OSError:
        return ""
    return ""


#: Every spelling of the model operand the supported launchers emit. SGLang
#: writes ``--model-path``; vLLM is launched either as ``-m ...api_server
#: --model <model>`` or as the bare console script ``vllm serve <model>``, so a
#: read keyed on ``--model-path`` alone is one vLLM cannot satisfy.
_MODEL_OPERAND_FLAGS: tuple[str, ...] = ("--model-path", "--model")
_TOKENIZER_OPERAND_FLAGS: tuple[str, ...] = ("--tokenizer-path", "--tokenizer")
_SERVED_NAME_FLAGS: tuple[str, ...] = ("--served-model-name",)
_TP_FLAGS: tuple[str, ...] = ("--tensor-parallel-size", "--tp-size", "--tp")
_DP_FLAGS: tuple[str, ...] = ("--data-parallel-size", "--dp-size")
_PP_FLAGS: tuple[str, ...] = ("--pipeline-parallel-size", "--pp-size")


def _operand_for(tokens: list[str], flags: tuple[str, ...]) -> str:
    """Return the operand of the first of ``flags`` present in ``tokens``."""
    for index, token in enumerate(tokens):
        name, separator, attached = token.partition("=")
        if name not in flags:
            continue
        if separator:
            return attached
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
            return tokens[index + 1]
    return ""


def _serve_subcommand_operand(tokens: list[str]) -> str:
    """Return the positional model operand of a ``serve`` subcommand."""
    for index, token in enumerate(tokens):
        if token != "serve":
            continue
        for candidate in tokens[index + 1 :]:
            if not candidate.startswith("-"):
                return candidate
        return ""
    return ""


def _names_a_model(tail: str) -> bool:
    """Whether a candidate launch tail names the model the server will serve.

    Exact over tokens rather than a substring test: ``--model`` is a prefix of
    ``--model-len`` and of ``--model-loader-extra-config``, neither of which
    names a model.
    """
    try:
        tokens = shlex.split(tail)
    except ValueError:
        tokens = tail.split()
    return bool(_operand_for(tokens, _MODEL_OPERAND_FLAGS) or _serve_subcommand_operand(tokens))


def _digest(value: str) -> str:
    """Digest a model operand: it compares, while the path could not travel."""
    text = str(value or "").strip()
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}" if text else ""


def _binding_from_tokens(tokens: list[str]) -> dict[str, Any]:
    model = _operand_for(tokens, _MODEL_OPERAND_FLAGS) or _serve_subcommand_operand(tokens)
    if not model:
        return {}
    return {
        "model_digest": _digest(model),
        "tokenizer_digest": _digest(_operand_for(tokens, _TOKENIZER_OPERAND_FLAGS)),
        "served_model_digest": _digest(_operand_for(tokens, _SERVED_NAME_FLAGS)),
        "tp": _operand_for(tokens, _TP_FLAGS),
        "dp": _operand_for(tokens, _DP_FLAGS),
        "pp": _operand_for(tokens, _PP_FLAGS),
    }


def observed_model_binding_from_log(path: str, framework: str) -> dict[str, Any]:
    """Read the model and parallelism the server was actually launched with.

    Read from the raw launch line, before :func:`split_launch_flags` removes
    those operands as run-local: without it the evidence carries only the
    *requested* model, and a server that resolved a different one yields
    evidence that is non-empty and wrong.

    Returns:
        ``{model_digest, tokenizer_digest, served_model_digest, tp, dp, pp}``,
        or ``{}`` for a framework whose launch line this reader cannot match.
    """
    marker = _LAUNCH_ARGV_MARKERS.get(str(framework or "").strip().lower())
    if not marker:
        return {}
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if marker not in line:
                    continue
                start = line.find("--")
                serve_at = line.find(" serve ")
                if start < 0 and serve_at < 0:
                    continue
                tail = line[min(x for x in (start, serve_at) if x >= 0) :].strip()
                try:
                    tokens = shlex.split(tail)
                except ValueError:
                    tokens = tail.split()
                binding = _binding_from_tokens(tokens)
                if binding:
                    return binding
    except OSError:
        return {}
    return {}


_SGLANG_SERVER_ARGS_LOG_RE = re.compile(r"\bserver_args\s*=\s*(?:ServerArgs\s*\(|\{)")
#: Shared scan caps: a launch record is read from the head of a log, bounded.
_SERVER_ARGS_MAX_CHARS = 512 * 1024
_SERVER_ARGS_MAX_LINES = 2048
_SERVER_ARGS_MAX_FIELDS = 2048
_SGLANG_OBSERVED_IDENTITY_FIELDS = frozenset(
    {
        "model_path",
        "tokenizer_path",
        "served_model_name",
        "tp_size",
        "dp_size",
        "mem_fraction_static",
        "context_length",
        "chunked_prefill_size",
        "quantization",
        "dtype",
        "kv_cache_dtype",
        "attention_backend",
        "prefill_attention_backend",
        "decode_attention_backend",
        "disable_radix_cache",
        "trust_remote_code",
    }
)


#: vLLM never echoes an argv line. It prints the RESOLVED argument dict under
#: this header instead, which is the authoritative record of what the server was
#: launched with -- more so than a command line, because it is what the parser
#: produced rather than what was typed. A reader that only looks for a command
#: line therefore finds nothing in ANY vLLM log, successful or failed, and every
#: requested setting is judged unconfirmed for want of an observed side.
_VLLM_NON_DEFAULT_ARGS_RE = re.compile(r"non[-_]default args:\s*\{", re.IGNORECASE)

#: The vLLM spellings of the settings the decision compares. Names differ from
#: SGLang's for the same knob, so the allowlist is per engine even though the
#: reader is not.
_VLLM_OBSERVED_IDENTITY_FIELDS: frozenset[str] = frozenset(
    {
        "model",
        "tokenizer",
        "served_model_name",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "max_model_len",
        "gpu_memory_utilization",
        "quantization",
        "dtype",
        "kv_cache_dtype",
        "block_size",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "enable_expert_parallel",
        "enforce_eager",
        "trust_remote_code",
        "swap_space",
        "cpu_offload_gb",
    }
)

#: vLLM's flag names are the keys it reports, so nothing needs folding.
_VLLM_FLAG_ALIASES: dict[str, str] = {}


@dataclass(frozen=True)
class _LaunchRecord:
    """Where one engine records the settings it resolved, and how it names them.

    Supporting an engine is a row here: nothing about reading a record is
    engine-specific except this data. An engine that records nothing -- atom,
    xdit, a custom workload -- has no row, so the readers return nothing and the
    launch is simply unobserved.
    """

    #: Header the record follows. Matched outside quoted runs only.
    marker: re.Pattern[str]
    #: Keys the identity read keeps; the config read keeps the whole record.
    identity_fields: frozenset[str]
    #: Launch-flag spellings folded to the names this engine reports.
    flag_aliases: Mapping[str, str]
    #: Whether an unreadable value in an allowlisted field voids the record.
    #: SGLang's is all literals, so one that is not means the record is not the
    #: one we think; vLLM routinely prints object reprs, where skipping the key
    #: is the only way to read the rest.
    identity_rejects_unreadable_values: bool


_LAUNCH_RECORDS: dict[str, _LaunchRecord] = {
    "sglang": _LaunchRecord(
        marker=_SGLANG_SERVER_ARGS_LOG_RE,
        identity_fields=_SGLANG_OBSERVED_IDENTITY_FIELDS,
        flag_aliases=_SGLANG_FLAG_ALIASES,
        identity_rejects_unreadable_values=True,
    ),
    "vllm": _LaunchRecord(
        marker=_VLLM_NON_DEFAULT_ARGS_RE,
        identity_fields=_VLLM_OBSERVED_IDENTITY_FIELDS,
        flag_aliases=_VLLM_FLAG_ALIASES,
        identity_rejects_unreadable_values=False,
    ),
}


def _is_inside_string_literal(text: str, index: int) -> bool:
    """Whether ``text[index]`` sits inside a quoted run earlier on the line.

    A log line may QUOTE the marker while carrying no launch record at all --
    ``WARNING ignored user text: "non-default args: {...}"`` is attacker- or
    user-supplied text echoed into the log. Treating that as an observed launch
    hands the decision an identity the server never ran with.
    """
    quote = ""
    escaped = False
    for char in text[:index]:
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in "'\"":
            quote = char
    return bool(quote)


def _balanced_record_payload(text: str, marker: re.Pattern[str]) -> str:
    """Return the balanced expression belonging to a real launch record.

    Anchored at the delimiter the MARKER matched, not at the first one on the
    line: a line may carry an unrelated dict before the record
    (``context={...} non-default args: {...}``), and starting at the first brace
    reads the unrelated one and silently ignores the actual record.

    A quoted marker is skipped. A log line may QUOTE the marker while carrying
    no launch record at all -- ``WARNING ignored user text: "server_args={...}"``
    is attacker- or user-supplied text echoed into the log -- and treating that
    as an observed launch hands the decision an identity the server never ran
    with. Braces inside string literals are likewise not structure: a model path
    may legally contain ``}``, so quoting is tracked while balancing.

    An ``(`` opener is returned as an inert ``_ServerArgs(...)`` call, which
    parses as keywords; SGLang prints that form, vLLM never does.
    """
    for match in marker.finditer(text):
        if _is_inside_string_literal(text, match.start()):
            continue
        payload = _balance_from(text, match.end() - 1)
        if payload:
            return payload
        # Unbalanced so far: the record continues on the next line.
        return ""
    return ""


def _balance_from(text: str, start: int) -> str:
    """The balanced delimited run beginning at ``text[start]``, or empty when incomplete."""
    closing: list[str] = []
    delimiters = {"(": ")", "[": "]", "{": "}"}
    quote = ""
    escaped = False
    for index, char in enumerate(text[start:], start):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in ("'", '"'):
            quote = char
        elif char in delimiters:
            closing.append(delimiters[char])
        elif char in ")]}":
            if not closing or char != closing.pop():
                return ""
            if not closing:
                expression = text[start : index + 1]
                return f"_ServerArgs{expression}" if text[start] == "(" else expression
    return ""


def _safe_server_args_value(node: ast.AST) -> Any:
    """Evaluate a literal ServerArgs value without executing log content."""
    return _bounded_server_args_value(ast.literal_eval(node))


def _bounded_server_args_value(value: Any, *, depth: int = 0) -> Any:
    """Return a JSON-safe bounded ServerArgs value."""
    if depth > 4:
        raise ValueError("ServerArgs value nesting exceeds cap")
    if isinstance(value, str):
        if len(value) > 4096:
            raise ValueError("ServerArgs string exceeds cap")
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            raise ValueError("ServerArgs list exceeds cap")
        return [_bounded_server_args_value(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 64 or not all(isinstance(key, str) for key in value):
            raise ValueError("ServerArgs dict exceeds cap")
        return {key: _bounded_server_args_value(item, depth=depth + 1) for key, item in sorted(value.items())}
    raise ValueError("ServerArgs value is not JSON-safe")


def observed_server_identity_from_log(path: str, framework: str) -> dict[str, Any]:
    """Parse the allowlisted identity from ``framework``'s launch record."""
    spec = _LAUNCH_RECORDS.get(str(framework or "").strip().lower())
    if spec is None:
        return {}
    return _server_args_from_log(path, spec, keep=spec.identity_fields)


def observed_server_config_from_log(path: str, framework: str) -> dict[str, Any]:
    """Parse every setting in ``framework``'s launch record.

    The whole record :func:`observed_server_identity_from_log` allowlists, which
    for an engine that echoes no argv is the only account of what it ran.

    How complete that account is differs by engine, and the difference matters
    to a caller weighing a setting the record does not mention. SGLang's
    ``server_args`` is its entire resolved configuration, so an absent setting
    is genuinely unknown. vLLM records only what differs from its defaults, so a
    setting left at its default is absent rather than reported -- a restatement
    of a default value reads as unknown, not as already active. An engine with
    no record at all yields nothing, leaving the launch unobserved rather than
    misread.
    """
    spec = _LAUNCH_RECORDS.get(str(framework or "").strip().lower())
    if spec is None:
        return {}
    return _server_args_from_log(path, spec, keep=None)


def observed_sglang_server_identity_from_log(path: str) -> dict[str, Any]:
    """Parse allowlisted identity from a capped SGLang ``server_args`` record."""
    return observed_server_identity_from_log(path, "sglang")


def observed_vllm_server_identity_from_log(path: str) -> dict[str, Any]:
    """Parse allowlisted identity from vLLM's ``non-default args: {...}`` record."""
    return observed_server_identity_from_log(path, "vllm")


def _server_args_from_log(path: str, spec: _LaunchRecord, *, keep: frozenset[str] | None) -> dict[str, Any]:
    """Parse a capped launch record, limited to ``keep`` when given."""
    chunks: list[str] = []
    remaining = _SERVER_ARGS_MAX_CHARS
    scanned_lines = 0
    parsed = ""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for _ in range(_SERVER_ARGS_MAX_LINES):
                line = handle.readline(remaining)
                if not line:
                    break
                scanned_lines += 1
                remaining -= len(line)
                if chunks or spec.marker.search(line):
                    chunks.append(line)
                    parsed = _balanced_record_payload("".join(chunks), spec.marker)
                    if parsed:
                        break
                if remaining <= 0:
                    break
    except OSError:
        return {}
    text = "".join(chunks)
    content = parsed or _balanced_record_payload(text, spec.marker)
    if not content or len(content) > _SERVER_ARGS_MAX_CHARS:
        if scanned_lines >= _SERVER_ARGS_MAX_LINES or remaining <= 0:
            log.debug(
                "launch record unavailable after scanning bounded log %s (lines=%d chars_remaining=%d)",
                path,
                scanned_lines,
                remaining,
            )
        return {}
    try:
        record = ast.parse(content, mode="eval").body
        fields: list[tuple[str, ast.expr]] = []
        if isinstance(record, ast.Dict):
            if len(record.keys) > _SERVER_ARGS_MAX_FIELDS:
                return {}
            for key, value in zip(record.keys, record.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    return {}
                fields.append((key.value, value))
        elif isinstance(record, ast.Call):
            if len(record.keywords) > _SERVER_ARGS_MAX_FIELDS:
                return {}
            for keyword in record.keywords:
                if keyword.arg is None:
                    return {}
                fields.append((keyword.arg, keyword.value))
        else:
            return {}
        values: dict[str, Any] = {}
        for name, value in fields:
            if keep is not None and name not in keep:
                continue
            try:
                values[name] = _safe_server_args_value(value)
            except (ValueError, TypeError):
                # An object repr among 500 settings is not worth losing the rest
                # of the record over, so the config read skips the key. An
                # allowlisted one is only voided where the engine's record is
                # meant to be all literals.
                if keep is not None and spec.identity_rejects_unreadable_values:
                    raise
                continue
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return {}
    return {key: values[key] for key in sorted(values)}
