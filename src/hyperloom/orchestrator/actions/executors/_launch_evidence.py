# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared, bounded launch-evidence construction for benchmark measurements."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.launch_log_evidence import (
    launch_argv_from_log,
    observed_sglang_server_identity_from_log,
    split_launch_flags,
)
from hyperloom.inference_optimizer.framework_registry import server_args_env_name

log = logging.getLogger(__name__)

#: Framework-agnostic snapshot of the baseline server's launch, written beside
#: the run's other artifacts and copied forward on import.
LAUNCH_CONFIG_FILENAME = "launch_config.json"
_LAUNCH_CONFIG_SCHEMA_VERSION = 1

# Only env keys with a server-affecting prefix participate in the launch
# snapshot; The set spans frameworks so the same generic capture serves SGLang, vLLM, and Atom.
_SERVER_ENV_PREFIXES: tuple[str, ...] = (
    "SGLANG_",
    "SGL_",
    "VLLM_",
    "ATOM_",
    "AITER_",
    "USE_ROCM_",
    "ROCM_",
    "HIP_",
    "HSA_",
    "NCCL_",
    "RCCL_",
    "TORCH_",
    "PYTORCH_",
)


def server_env_subset(env: Mapping[str, str]) -> dict[str, str]:
    """Keep only server-affecting env keys, upper-cased."""
    out: dict[str, str] = {}
    for key, value in (env or {}).items():
        norm = str(key).strip().upper()
        if norm and norm.startswith(_SERVER_ENV_PREFIXES):
            out[norm] = str(value)
    return out


def _flags_from_cmdline_tokens(tokens: list[str]) -> str:
    """Strip the interpreter/entrypoint prefix, keep the launch flags."""
    for index, token in enumerate(tokens):
        if token.startswith("--"):
            return split_launch_flags(" ".join(tokens[index:]))
    return ""


def capture_launch_config_via_proc(pid: int, framework: str) -> tuple[str, dict[str, str]] | None:
    """Snapshot ``(launch_flags, server_env)`` from a live server process.

    Reads ``/proc/<pid>/cmdline`` and ``/proc/<pid>/environ`` directly, so it is
    framework-agnostic and captures env the recipe script exported internally.
    Returns ``None`` when the process is gone or unreadable.
    """
    try:
        cmdline_raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        environ_raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    tokens = [t for t in cmdline_raw.decode("utf-8", errors="replace").split("\x00") if t]
    flags = _flags_from_cmdline_tokens(tokens)
    env: dict[str, str] = {}
    for entry in environ_raw.decode("utf-8", errors="replace").split("\x00"):
        if "=" not in entry:
            continue
        key, _, value = entry.partition("=")
        env[key] = value
    return flags, server_env_subset(env)


def write_launch_config(
    out_dir: str | Path,
    *,
    framework: str,
    launch_flags: str,
    env: Mapping[str, str],
    source: str,
) -> str:
    """Persist a ``launch_config.json`` snapshot; return its path (or ``\"\"``)."""
    payload = {
        "schema_version": _LAUNCH_CONFIG_SCHEMA_VERSION,
        "framework": str(framework or "").strip().lower(),
        "launch_flags": str(launch_flags or "").strip(),
        "env": dict(sorted(server_env_subset(env).items())),
        "source": str(source or ""),
    }
    try:
        path = Path(out_dir) / LAUNCH_CONFIG_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return str(path)
    except OSError:
        log.debug("could not persist launch_config.json in %s", out_dir, exc_info=True)
        return ""


def read_launch_config(search_dirs: list[str]) -> tuple[str, dict[str, str], str] | None:
    """Return ``(launch_flags, env, framework)`` from the first found snapshot."""
    for base in search_dirs:
        if not base:
            continue
        candidate = Path(base) / LAUNCH_CONFIG_FILENAME
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        env_raw = payload.get("env")
        env = {str(k): str(v) for k, v in env_raw.items()} if isinstance(env_raw, Mapping) else {}
        return (
            str(payload.get("launch_flags") or "").strip(),
            env,
            str(payload.get("framework") or "").strip().lower(),
        )
    return None


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
