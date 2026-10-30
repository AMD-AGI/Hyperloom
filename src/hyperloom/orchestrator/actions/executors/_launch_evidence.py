# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared, bounded launch-evidence construction for benchmark measurements."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from hyperloom.common.launch_log_evidence import (
    engine_adjusted_settings_from_log,
    launch_argv_from_log,
    observed_model_binding_from_log,
    observed_server_config_from_log,
    observed_server_identity_from_log,
    operand_digest,
)
from hyperloom.inference_optimizer.framework_registry import server_args_env_name

log = logging.getLogger(__name__)


def _binding_from_vllm_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Derive the binding from a vLLM ``non-default args`` parse.

    Separate from the SGLang one because the field names differ; sharing a table
    would silently read ``{}`` for one of the two frameworks.

    The record lists only non-default settings, so an absent width is vLLM's
    default of 1 rather than unknown: ``--data-parallel-size 1`` never appears
    in it, and reading that absence as unobserved fails every such launch.
    """
    model = str(identity.get("model") or "")
    if not model:
        return {}
    return {
        "model_digest": operand_digest(model),
        "tokenizer_digest": operand_digest(str(identity.get("tokenizer") or "")),
        "served_model_digest": operand_digest(str(identity.get("served_model_name") or "")),
        "tp": str(identity.get("tensor_parallel_size", 1)),
        "dp": str(identity.get("data_parallel_size", 1)),
        "pp": str(identity.get("pipeline_parallel_size", 1)),
    }


def _binding_from_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Derive the binding from an SGLang identity parse, which has no argv line."""
    model = str(identity.get("model_path") or "")
    if not model:
        return {}
    return {
        "model_digest": operand_digest(model),
        "tokenizer_digest": operand_digest(str(identity.get("tokenizer_path") or "")),
        "served_model_digest": operand_digest(str(identity.get("served_model_name") or "")),
        "tp": str(identity.get("tp_size") or ""),
        "dp": str(identity.get("dp_size") or ""),
        "pp": "",
    }


class _ConfigRecipe(NamedTuple):
    """The materialized config's bytes and its ``benchmark`` block."""

    raw: bytes
    benchmark: dict[str, Any]


def _read_config_recipe(config_path: Path) -> _ConfigRecipe:
    """Read the materialized config; one that was read but will not parse still counts as materialized."""
    raw = b""
    benchmark: dict[str, Any] = {}
    try:
        raw = config_path.read_bytes()
        parsed = yaml.safe_load(raw.decode("utf-8")) or {}
        if isinstance(parsed, dict):
            raw_benchmark = parsed.get("benchmark")
            benchmark = raw_benchmark if isinstance(raw_benchmark, dict) else {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        log.debug("launch evidence could not read materialized config %s", config_path, exc_info=True)
    return _ConfigRecipe(raw, benchmark)


def _observed_launch_from_log(actual_server_log: str | None, framework: str) -> dict[str, Any]:
    """The evidence document's observed half: what the server's own log says the engine resolved."""
    flags = ""
    identity: dict[str, Any] = {}
    config: dict[str, Any] = {}
    adjusted: dict[str, dict[str, str]] = {}
    binding: dict[str, Any] = {}
    if actual_server_log:
        config = observed_server_config_from_log(actual_server_log, framework)
        adjusted = engine_adjusted_settings_from_log(actual_server_log, framework)
        flags = launch_argv_from_log(actual_server_log, framework)
        # Read from the raw launch line, which still carries the operands
        # ``split_launch_flags`` strips: without it the evidence records only
        # the *requested* model and cannot detect a server that resolved a
        # different one.
        binding = observed_model_binding_from_log(actual_server_log, framework)
        if not flags:
            # Neither engine echoes an argv line reliably -- vLLM never does,
            # in no log, successful or failed -- so without the record every
            # requested setting would be judged unconfirmed. The resolved
            # settings the engine DOES print are the observed side.
            identity = observed_server_identity_from_log(actual_server_log, framework)
            if not binding:
                # The two engines name the model differently, so the binding
                # is read per engine even though the record is not.
                if framework == "vllm":
                    binding = _binding_from_vllm_identity(identity)
                else:
                    binding = _binding_from_identity(identity)
    return {
        "observed_server_launch_flags": flags,
        # What the engine resolved, the only account of a launch neither engine
        # echoes as argv. Complete for SGLang; non-defaults only for vLLM.
        "observed_server_config": config,
        # Settings the engine rewrote after parsing them: the config above holds
        # the resolved value, which is not what a flag would pass.
        "engine_adjusted_settings": adjusted,
        "observed_server_identity": identity,
        "observed_model_binding": binding,
    }


def _warm_reuse_record(actual_server_log: str | None, slot: Path, caller_reused_ready_server: bool) -> dict[str, Any]:
    """Whether this measurement ran against an already-ready server, and where that server came from."""
    actual_path = Path(actual_server_log) if actual_server_log else None
    from_warmup = bool(actual_path and actual_path.is_relative_to(slot / "warmup_round"))
    if from_warmup:
        provenance = "warmup_round"
    elif caller_reused_ready_server:
        provenance = "caller_ready_server"
    else:
        provenance = "fresh_or_unobserved"
    return {
        "reused_ready_server": bool(caller_reused_ready_server or from_warmup),
        "provenance": provenance,
        "source_server_log_path": actual_server_log or "",
    }


@dataclass(frozen=True)
class RequestedLaunch:
    """The launch a measurement requested: its materialized config and any caller overrides of it."""

    config_path: Path
    framework: str
    server_args: str | None = None
    server_env: dict[str, str] | None = None
    model_path: str | None = None


def build_launch_evidence(
    requested: RequestedLaunch,
    *,
    actual_server_log: str | None,
    slot: Path,
    caller_reused_ready_server: bool = False,
) -> dict[str, Any]:
    """Build declared and observed evidence for one measured server launch."""
    recipe = _read_config_recipe(requested.config_path)
    benchmark = recipe.benchmark

    resolved_framework = str(benchmark.get("framework") or requested.framework or "sglang").strip().lower()
    args_env = server_args_env_name(resolved_framework)
    envs = benchmark.get("envs") if isinstance(benchmark.get("envs"), dict) else {}
    declared_env = {str(key): str(value) for key, value in envs.items() if str(key) != args_env}
    declared_args = str(envs.get(args_env) or "").strip()
    requested_env = requested.server_env if requested.server_env is not None else declared_env
    requested_args = str(requested.server_args).strip() if requested.server_args is not None else declared_args
    requested_model = str(requested.model_path if requested.model_path is not None else benchmark.get("model") or "")

    return {
        "schema_version": 1,
        "materialized_config_path": str(requested.config_path) if recipe.raw else "",
        "recipe_digest": f"sha256:{hashlib.sha256(recipe.raw).hexdigest()}" if recipe.raw else "",
        "framework": resolved_framework,
        "model_path": requested_model,
        "requested_server_args": requested_args,
        "requested_server_env": requested_env,
        "actual_server_log_path": actual_server_log or "",
        **_observed_launch_from_log(actual_server_log, resolved_framework),
        "requested_model_digest": operand_digest(requested_model),
        "warm_reuse": _warm_reuse_record(actual_server_log, slot, caller_reused_ready_server),
    }


def persist_launch_evidence(evidence: dict[str, Any], *, slot: Path) -> str:
    """Persist evidence in its owning slot and return its path."""
    try:
        slot.mkdir(parents=True, exist_ok=True)
        path = slot / "launch_evidence.json"
        path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
        return str(path)
    except OSError:
        log.warning("launch evidence could not persist in %s", slot, exc_info=True)
        return ""
