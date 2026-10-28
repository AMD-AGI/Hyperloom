# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate explicit server limits before runtime setup."""

from __future__ import annotations

import argparse
import os
import re
import shlex

from ..grid_server_args import compact_json_server_args
from ..model_config_utils import _load_model_max_position_embeddings


def _explicit_limits(server_args: str) -> dict[str, str]:
    tokens = shlex.split(compact_json_server_args(server_args))
    values = {}
    for index, token in enumerate(tokens):
        flag, separator, value = token.partition("=")
        if flag in {"--gpu-memory-utilization", "--max-model-len"}:
            values[flag] = value if separator else tokens[index + 1] if index + 1 < len(tokens) else ""
    return values


def _vllm_context_length(raw: str) -> int | None:
    # vLLM accepts automatic sizing and decimal/binary unit suffixes on its raw CLI.
    value = raw.strip()
    if value.lower() == "auto" or value == "-1":
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmMgGtT])", value)
    if match:
        number, suffix = match.groups()
        power = "kmgt".index(suffix.lower()) + 1
        if suffix.isupper():
            return int(number) * 1024**power
        return int(float(number) * 1000**power)
    return int(value)


def validate_server_args(args: argparse.Namespace | None, model_path: str) -> None:
    """Validate explicit limits against locally available model context."""
    native = _load_model_max_position_embeddings(model_path)
    explicit = getattr(args, "max_model_len", None)
    context = explicit if explicit is not None else os.environ.get("MAX_MODEL_LEN", "").strip() or None
    checks = [("--max-model-len" if explicit is not None else "$MAX_MODEL_LEN", context, int, native)]
    framework = str(getattr(args, "framework", None) or os.environ.get("FRAMEWORK", "")).strip().lower()
    if framework == "vllm":
        values = _explicit_limits(str(getattr(args, "server_args", None) or ""))
        checks.extend(
            [
                ("vllm --gpu-memory-utilization", values.get("--gpu-memory-utilization"), float, 1),
                ("vllm --max-model-len", values.get("--max-model-len"), _vllm_context_length, native),
            ]
        )
    for flag, raw, parse, upper in checks:
        if raw is None:
            continue
        bound = f"0 < value <= {upper}" if upper is not None else "value > 0"
        message = f"{flag} must satisfy {bound}; got {raw!r}"
        try:
            value = parse(raw)
        except ValueError as exc:
            raise ValueError(message) from exc
        if value is not None and not (value > 0 and (upper is None or value <= upper)):
            raise ValueError(message)
