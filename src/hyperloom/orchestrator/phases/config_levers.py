# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The config lever one FRAMEWORK proposal carries: its name, server args, env overrides and removal controls."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from hyperloom_common.coerce import to_mapping
from hyperloom.inference_optimizer.grid_server_args import tokenize_server_args_preserving_json

from ..actions.executors import controls_of, normalize_proposal

log = logging.getLogger(__name__)


def _transportable_server_args(args: Any) -> tuple[str, str]:
    """A lever's ``extra_args`` as the argv string Magpie transports, or ``""`` and why it cannot be."""
    tokens = [str(a) for a in args if str(a).strip()] if isinstance(args, (list, tuple)) else []
    if any(ch.isspace() for token in tokens for ch in token):
        return "", "has a whitespace-bearing argv token"
    parsed_args = tokenize_server_args_preserving_json(args if isinstance(args, str) else " ".join(tokens))
    if parsed_args is None:
        if isinstance(args, str):
            return "", "has server args unsupported by Magpie's unquoted argv transport"
        return "", "has unparsable server args"
    return parsed_args[0], ""


def config_lever_from_proposal(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The name, server args, env overrides and removal controls one lever proposal carries.

    Args Magpie cannot transport are dropped, and the removal controls with them: they may be coupled to those args.
    """
    extra_envs = {str(k).strip(): str(v) for k, v in to_mapping(entry.get("extra_envs")).items() if str(k).strip()}
    extra_server_args, fault = _transportable_server_args(entry.get("extra_args"))
    if fault:
        log.warning(
            "FRAMEWORK config lever %r %s; dropping the args%s",
            entry.get("name"),
            fault,
            " while preserving its environment overrides" if extra_envs else "",
        )
        if not extra_envs:
            return {}
    controls = {} if fault else controls_of(normalize_proposal(entry))
    if not (extra_server_args or extra_envs or controls):
        return {}
    return {
        "name": str(entry.get("name") or ""),
        "extra_server_args": extra_server_args,
        "extra_envs": extra_envs,
        **controls,
    }
