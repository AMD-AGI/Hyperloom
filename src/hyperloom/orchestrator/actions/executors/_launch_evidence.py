# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared, bounded launch-evidence construction for benchmark measurements."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.launch_log_evidence import (
    launch_argv_from_log,
    observed_sglang_server_identity_from_log,
)
from hyperloom.inference_optimizer.framework_registry import server_args_env_name

log = logging.getLogger(__name__)


def build_launch_evidence(
    *,
    config_path: Path,
    actual_server_log: str | None,
    framework: str,
    slot: Path,
    caller_reused_ready_server: bool = False,
    requested_server_args: str | None = None,
    requested_server_env: dict[str, str] | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Build declared and observed evidence for one measured server launch."""
    raw_config = b""
    benchmark: dict[str, Any] = {}
    try:
        raw_config = config_path.read_bytes()
        parsed = yaml.safe_load(raw_config.decode("utf-8")) or {}
        if isinstance(parsed, dict):
            raw_benchmark = parsed.get("benchmark")
            benchmark = raw_benchmark if isinstance(raw_benchmark, dict) else {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        log.debug("launch evidence could not read materialized config %s", config_path, exc_info=True)

    resolved_framework = str(benchmark.get("framework") or framework or "sglang").strip().lower()
    args_env = server_args_env_name(resolved_framework)
    envs = benchmark.get("envs") if isinstance(benchmark.get("envs"), dict) else {}
    declared_env = {str(key): str(value) for key, value in envs.items() if str(key) != args_env}
    declared_args = str(envs.get(args_env) or "").strip()
    requested_env = requested_server_env if requested_server_env is not None else declared_env
    requested_args = str(requested_server_args).strip() if requested_server_args is not None else declared_args
    recipe_digest = f"sha256:{hashlib.sha256(raw_config).hexdigest()}" if raw_config else ""

    observed_flags = ""
    observed_server_identity: dict[str, Any] = {}
    if actual_server_log:
        try:
            observed_flags = launch_argv_from_log(actual_server_log, resolved_framework)
            if not observed_flags and resolved_framework == "sglang":
                observed_server_identity = observed_sglang_server_identity_from_log(actual_server_log)
        except Exception:  # noqa: BLE001 - evidence collection must not alter a measurement
            log.debug("launch evidence could not inspect server log %s", actual_server_log, exc_info=True)

    warmup_root = slot / "warmup_round"
    actual_path = Path(actual_server_log) if actual_server_log else None
    reused_from_warmup = bool(actual_path and actual_path.is_relative_to(warmup_root))
    reused = bool(caller_reused_ready_server or reused_from_warmup)
    return {
        "schema_version": 1,
        "materialized_config_path": str(config_path) if raw_config else "",
        "recipe_digest": recipe_digest,
        "framework": resolved_framework,
        "model_path": str(model_path if model_path is not None else benchmark.get("model") or ""),
        "requested_server_args": requested_args,
        "requested_server_flags": requested_args,
        "requested_server_env": requested_env,
        "actual_server_log_path": actual_server_log or "",
        "observed_server_launch_flags": observed_flags,
        "observed_server_identity": observed_server_identity,
        "warm_reuse": {
            "reused_ready_server": reused,
            "provenance": (
                "warmup_round"
                if reused_from_warmup
                else ("caller_ready_server" if caller_reused_ready_server else "fresh_or_unobserved")
            ),
            "source_server_log_path": actual_server_log or "",
        },
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
