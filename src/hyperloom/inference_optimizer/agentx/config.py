# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Translate Hyperloom workload controls into Magpie's native AgentX configuration."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from hyperloom.common.coerce import to_float

_CANONICAL = "canonical"
_FAST = "fast"


def agentx_setting(envs: Mapping[str, Any], name: str) -> str:
    """Resolve one AgentX setting from persisted YAML, then the process."""
    return str(envs.get(name) or os.environ.get(name) or "").strip()


def native_agentx_config(envs: Mapping[str, Any], existing: Any = None) -> str | dict[str, Any]:
    """Build Magpie's native AgentX value without erasing YAML-native pins."""
    existing_config = dict(existing) if isinstance(existing, Mapping) else {}
    # Hyperloom changes CONC per baseline/grid/sweep round. Magpie gives an
    # explicit agentx.concurrency precedence over benchmark.envs.CONC, so
    # carrying a source-YAML value here would make every materialized rung run
    # the same load while being labelled with a different concurrency.
    existing_config.pop("concurrency", None)
    # ``resolved`` describes the previous point.  Magpie will recompute it for
    # this round after Hyperloom settles CONC; carrying it forward makes a
    # topology-changing input look pre-resolved when it is not.
    previous_resolved = existing_config.pop("resolved", None)
    if not existing_config.get("selector") and isinstance(previous_resolved, Mapping):
        # A previously unique point can become ambiguous when only CONC
        # changes (for example GLM's TP4/HiCache and TP8/no-offload arms both
        # contain CONC=2). Preserve the accepted arm, not its old concurrency,
        # as an explicit Magpie selector before asking it to resolve the rung.
        stable_selector_keys = (
            "tp",
            "pp",
            "pcp-size",
            "ep",
            "dcp-size",
            "dp-attn",
            "spec-decoding",
            "kv-offloading",
            "kv-offload-backend",
        )
        derived_selector = {key: previous_resolved[key] for key in stable_selector_keys if key in previous_resolved}
        if derived_selector:
            existing_config["selector"] = derived_selector
    mode = agentx_setting(envs, "AGENTX_MODE").lower() or str(existing_config.get("mode") or _CANONICAL).lower()
    if mode not in {_CANONICAL, _FAST}:
        raise ValueError("AGENTX_MODE must be 'canonical' or 'fast'")
    recipe = agentx_setting(envs, "AGENTX_RECIPE") or str(existing_config.get("recipe") or "").strip()
    config_file = agentx_setting(envs, "AGENTX_CONFIG_FILE") or str(existing_config.get("config_file") or "").strip()
    selector_raw = agentx_setting(envs, "AGENTX_SELECTOR")
    threshold_raw = agentx_setting(envs, "AGENTX_FAILED_REQUEST_THRESHOLD")

    existing_selector = existing_config.get("selector")
    selector: dict[str, Any] = dict(existing_selector) if isinstance(existing_selector, Mapping) else {}
    if selector_raw:
        try:
            parsed = json.loads(selector_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"AGENTX_SELECTOR must be a JSON object: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("AGENTX_SELECTOR must be a JSON object")
        selector = parsed

    if (
        not existing_config
        and mode == _CANONICAL
        and not recipe
        and not config_file
        and not selector
        and not threshold_raw
    ):
        # This is the public, zero-boilerplate Magpie switch the integration is
        # designed around. Expand to an object only for an explicit override.
        return "enable"

    config: dict[str, Any] = {**existing_config, "enabled": True, "mode": mode}
    if recipe:
        config["recipe"] = recipe
    if config_file:
        config["config_file"] = config_file
    if selector:
        config["selector"] = selector
    if threshold_raw:
        try:
            threshold = float(threshold_raw)
        except ValueError as exc:
            raise ValueError("AGENTX_FAILED_REQUEST_THRESHOLD must be a number between 0 and 1") from exc
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("AGENTX_FAILED_REQUEST_THRESHOLD must be between 0 and 1")
        config["failed_request_threshold"] = threshold
    elif "failed_request_threshold" in config:
        configured_threshold = to_float(config["failed_request_threshold"])
        if configured_threshold is None or not 0.0 <= configured_threshold <= 1.0:
            raise ValueError("benchmark.agentx.failed_request_threshold must be between 0 and 1")
        config["failed_request_threshold"] = configured_threshold
    return config
