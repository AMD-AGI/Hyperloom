# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bounded, dependency-free server-launch log evidence helpers."""

from __future__ import annotations

import ast
import logging
import re
import shlex
from typing import Any


log = logging.getLogger(__name__)

#: Launch flags that are RUN-/TOPOLOGY-specific (host, device set, model path,
#: parallelism, ports, seeds); stripped from forwarded server-launch flags.
_RUN_SPECIFIC_LAUNCH_FLAGS: frozenset[str] = frozenset(
    {
        "--model-path",
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

#: Per-backend marker for the start of a captured launch argv.
_LAUNCH_ARGV_MARKERS: dict[str, str] = {
    "sglang": "launch_server",
    "vllm": "vllm",
}


def _split_launch_flags(argv_tail: str) -> str:
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
        if flag in _RUN_SPECIFIC_LAUNCH_FLAGS or flag in _PROFILING_LAUNCH_FLAGS:
            if "=" not in token and index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
                index += 2
            else:
                index += 1
            continue
        kept.append(token)
        index += 1
    return " ".join(kept)


def _launch_argv_from_log(path: str, marker: str) -> str:
    """Extract and normalize the engine launch argv from one benchmark log."""
    pattern = re.compile(r"(?:-m\s+\S*" + re.escape(marker) + r"\S*|" + re.escape(marker) + r")\b(.*)$")
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if marker not in line or "--model-path" not in line:
                    continue
                match = pattern.search(line)
                tail = (match.group(1) if match else "").strip()
                if not tail:
                    start = line.find("--")
                    tail = line[start:].strip() if start >= 0 else ""
                flags = _split_launch_flags(tail)
                if flags:
                    return flags
    except OSError:
        return ""
    return ""


_SGLANG_SERVER_ARGS_LOG_RE = re.compile(r"\bserver_args\s*=\s*ServerArgs\s*\(")
_SGLANG_SERVER_ARGS_MAX_CHARS = 512 * 1024
_SGLANG_SERVER_ARGS_MAX_LINES = 2048
_SGLANG_SERVER_ARGS_MAX_FIELDS = 2048
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


def _extract_balanced_server_args(text: str) -> str:
    """Return the balanced ``ServerArgs(...)`` argument text."""
    match = _SGLANG_SERVER_ARGS_LOG_RE.search(text)
    if match is None:
        return ""
    start = match.end() - 1
    depth = 0
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
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
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


def _observed_sglang_server_identity_from_log(path: str) -> dict[str, Any]:
    """Parse a capped archived SGLang ``server_args=ServerArgs(...)`` record."""
    chunks: list[str] = []
    remaining = _SGLANG_SERVER_ARGS_MAX_CHARS
    scanned_lines = 0
    parsed = ""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for _ in range(_SGLANG_SERVER_ARGS_MAX_LINES):
                line = handle.readline()
                if not line:
                    break
                scanned_lines += 1
                if len(line) > remaining:
                    line = line[:remaining]
                remaining -= len(line)
                if chunks or _SGLANG_SERVER_ARGS_LOG_RE.search(line):
                    chunks.append(line)
                    parsed = _extract_balanced_server_args("".join(chunks))
                    if parsed:
                        break
                if remaining <= 0:
                    break
    except OSError:
        return {}
    text = "".join(chunks)
    content = parsed or _extract_balanced_server_args(text)
    if not content or len(content) > _SGLANG_SERVER_ARGS_MAX_CHARS:
        if scanned_lines >= _SGLANG_SERVER_ARGS_MAX_LINES or remaining <= 0:
            log.debug(
                "sglang observed identity unavailable after scanning bounded log %s (lines=%d chars_remaining=%d)",
                path,
                scanned_lines,
                remaining,
            )
        return {}
    try:
        call = ast.parse(f"_ServerArgs({content})", mode="eval").body
        if not isinstance(call, ast.Call) or len(call.keywords) > _SGLANG_SERVER_ARGS_MAX_FIELDS:
            return {}
        values: dict[str, Any] = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                return {}
            if keyword.arg not in _SGLANG_OBSERVED_IDENTITY_FIELDS:
                continue
            values[keyword.arg] = _safe_server_args_value(keyword.value)
    except (SyntaxError, ValueError, TypeError):
        return {}
    return {key: values[key] for key in sorted(values)}
