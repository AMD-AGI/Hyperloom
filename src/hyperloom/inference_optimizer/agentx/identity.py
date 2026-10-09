# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Separate a native replay workload from the implementation being optimized."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

# Frozen protocol options for persisted InferenceX version-1 launch contracts.
# Managed sessions instead use their audited Magpie installation to validate
# argv semantics, including profiler flags and future protected options.
_PROTOCOL_ARGS = frozenset(
    {
        "--model",
        "--model-path",
        "--max-model-len",
        "--context-length",
        "--served-model-name",
        "--host",
        "--port",
        "--tp",
        "--tp-size",
        "--tensor-parallel-size",
        "--pp",
        "--pp-size",
        "--pipeline-parallel-size",
        "--ep",
        "--ep-size",
        "--expert-parallel-size",
        "--dp",
        "--dp-size",
        "--data-parallel-size",
        "--data-parallel-rank",
        "--data-parallel-start-rank",
        "--data-parallel-size-local",
        "--data-parallel-address",
        "--data-parallel-rpc-port",
        "--enable-dp-attention",
        "--enable-expert-parallel",
        "--decode-context-parallel-size",
        "--prefill-context-parallel-size",
        "--dcp-size",
        "--pcp-size",
        "--nnodes",
        "--node-rank",
        "--dist-init-addr",
    }
)


def _argument_groups(argv: list[str]) -> tuple[list[str], list[list[str]]]:
    prefix: list[str] = []
    groups: list[list[str]] = []
    for token in argv:
        if token.startswith("--"):
            if not re.fullmatch(r"--[A-Za-z0-9][A-Za-z0-9_-]*", token.partition("=")[0]):
                raise ValueError("invalid long option")
            groups.append([token])
        elif groups:
            if "=" in groups[-1][0] or (
                token.startswith("-") and not re.fullmatch(r"-\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", token)
            ):
                raise ValueError("ambiguous option operand")
            groups[-1].append(token)
        else:
            prefix.append(token)
    return prefix, groups


def _expected_launch_argv(base: list[str], overrides: Mapping[str, Any]) -> list[str]:
    effective = list(base)
    appended = overrides.get("append_args", [])
    removed = overrides.get("remove_args", [])
    replace = overrides.get("replace_args", False)
    if not all(
        isinstance(values, list) and all(isinstance(value, str) for value in values) for values in (appended, removed)
    ):
        raise ValueError("invalid argument override")
    if type(replace) is not bool or len(set(removed)) != len(removed):
        raise ValueError("invalid argument controls")
    if appended or removed or replace:
        prefix, groups = _argument_groups(base)
        extra_prefix, extra = _argument_groups(appended)
        if extra_prefix or any(group[0].partition("=")[0] in _PROTOCOL_ARGS for group in extra):
            raise ValueError("protocol argument override")
        for name in removed:
            matches = [group for group in groups if group[0].partition("=")[0] == name]
            if name in _PROTOCOL_ARGS or len(matches) != 1:
                raise ValueError("ambiguous or protected argument removal")
            groups.remove(matches[0])
        if replace:
            groups = [group for group in groups if group[0].partition("=")[0] in _PROTOCOL_ARGS]
        effective = prefix + [token for group in groups + extra for token in group]
    if overrides.get("executable") is not None:
        effective[0] = overrides["executable"]
    return effective


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value using the upstream launch evidence representation."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def has_launch_contract(benchmark: Mapping[str, Any]) -> bool:
    """Identify the persisted optimization contract; old native sessions lack it."""
    agentx = benchmark.get("agentx")
    overrides = agentx.get("launch_overrides") if isinstance(agentx, Mapping) else None
    return isinstance(overrides, Mapping) and type(overrides.get("version")) is int and overrides["version"] == 1


def native_workload_fingerprint(benchmark: Mapping[str, Any], static_fingerprint: str) -> str:
    """Bind the replay and model while allowing explicit server implementation changes.

    Arbitrary runtime environment changes belong in launch_overrides, whose
    upstream validator rejects replay/model/topology controls. The remaining
    benchmark environment stays part of the fixed session workload.
    """
    payload = dict(benchmark)
    payload.pop("workload_spec", None)
    agentx = dict(payload.get("agentx") or {})
    agentx.pop("launch_overrides", None)
    payload["agentx"] = agentx
    return canonical_sha256({"static_execution_fingerprint": static_fingerprint, "benchmark": payload})


def _validate_managed_receipt(
    benchmark: Mapping[str, Any], evidence: Mapping[str, Any], workspace: Path | None
) -> list[str]:
    """Delegate managed argv and diagnostic instrumentation to audited Magpie."""
    profile = ((benchmark.get("profiler") or {}).get("torch_profiler") or {}).get("enabled") is True
    if profile and workspace is None:
        return ["server_launch_profile_workspace_missing"]
    failure = "server_launch_profile_verification_failed" if profile else "server_launch_managed_verification_failed"
    from hyperloom.common.env_safety import scrub_benchmark_process_env
    from hyperloom.orchestrator.actions.executors.benchmark_backend import resolve_benchmark_interpreter

    from .native import _MAGPIE_SOURCE_IDENTITY_CODE

    code = (
        _MAGPIE_SOURCE_IDENTITY_CODE
        + """
import sys
import Magpie
from Magpie.modes.benchmark import BenchmarkConfig
from Magpie.modes.benchmark.agentx_launch import apply_launch_args, read_launch_evidence, _validate_golden_launch
package = Path(Magpie.__file__).resolve().parent
commit, _ = _resolve_magpie_source_identity(package)
_validate_magpie_execution_tree(package, commit)
payload = json.load(sys.stdin)
benchmark, evidence = payload["benchmark"], payload["evidence"]
if payload["profile"]:
    config = BenchmarkConfig.from_dict(benchmark)
    actual = read_launch_evidence(config, Path(payload["workspace"]))
    if actual != evidence:
        raise ValueError("profile receipt does not match the report")
else:
    expected = apply_launch_args(evidence["base_argv"], benchmark["agentx"]["launch_overrides"], benchmark["framework"])
    if evidence["effective_argv"] != expected:
        raise ValueError("managed effective argv does not match the candidate")
    _validate_golden_launch(benchmark["agentx"]["resolved"]["server-launch-spec"], expected)
"""
    )
    import os

    try:
        checked = subprocess.run(
            [resolve_benchmark_interpreter(), "-c", code],
            input=json.dumps(
                {
                    "benchmark": dict(benchmark),
                    "evidence": dict(evidence),
                    "workspace": str(workspace),
                    "profile": profile,
                }
            ),
            capture_output=True,
            text=True,
            timeout=120,
            env=scrub_benchmark_process_env(dict(os.environ)),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return [failure]
    return [] if checked.returncode == 0 else [failure]


def validate_server_launch(benchmark: Mapping[str, Any], evidence: Any, *, workspace: Path | None = None) -> list[str]:
    """Validate the observed upstream server launch against the submitted candidate."""
    if not has_launch_contract(benchmark):
        return []
    if not isinstance(evidence, dict):
        return ["server_launch_missing"]
    errors: list[str] = []
    if (
        type(evidence.get("version")) is not int
        or evidence.get("version") != 1
        or evidence.get("framework") != benchmark.get("framework")
    ):
        errors.append("server_launch_identity_mismatch")
    unsigned = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    if evidence.get("evidence_sha256") != canonical_sha256(unsigned):
        errors.append("server_launch_evidence_hash_mismatch")
    overrides = benchmark["agentx"]["launch_overrides"]
    resolved = benchmark["agentx"].get("resolved", {})
    spec = resolved.get("server-launch-spec") if isinstance(resolved, Mapping) else None
    if evidence.get("overrides_sha256") != canonical_sha256(overrides):
        errors.append("server_launch_candidate_mismatch")
    for key in ("base_argv", "effective_argv"):
        args = evidence.get(key)
        if not isinstance(args, list) or not args or any(not isinstance(arg, str) or "\0" in arg for arg in args):
            errors.append(f"server_launch_{key}_invalid")
    if not isinstance(spec, Mapping) and not any(code.endswith("_argv_invalid") for code in errors):
        try:
            expected_argv = _expected_launch_argv(evidence["base_argv"], overrides)
        except ValueError:
            errors.append("server_launch_arguments_invalid")
        else:
            if evidence["effective_argv"] != expected_argv:
                errors.append("server_launch_arguments_mismatch")
    executable = overrides.get("executable")
    if executable and (
        not isinstance(evidence.get("effective_argv"), list)
        or not evidence["effective_argv"]
        or evidence["effective_argv"][0] != executable
    ):
        errors.append("server_launch_executable_mismatch")
    observed_env = evidence.get("effective_env")
    expected_env = {**overrides.get("env", {}), **dict.fromkeys(overrides.get("unset_env", []))}
    if not isinstance(observed_env, dict):
        errors.append("server_launch_environment_missing")
    elif observed_env != expected_env:
        errors.append("server_launch_environment_mismatch")
    base_env = evidence.get("base_env")
    if (
        not isinstance(base_env, dict)
        or base_env.keys() != expected_env.keys()
        or any(value is not None and not isinstance(value, str) for value in base_env.values())
    ):
        errors.append("server_launch_base_environment_invalid")
    resolved_executable = evidence.get("resolved_executable")
    if not isinstance(resolved_executable, str) or not PurePosixPath(resolved_executable).is_absolute():
        errors.append("server_launch_resolved_executable_invalid")
    runtime_environment = evidence.get("runtime_environment")
    if not isinstance(runtime_environment, dict) or any(
        not isinstance(name, str) or not isinstance(value, str) for name, value in runtime_environment.items()
    ):
        errors.append("server_launch_runtime_environment_invalid")
    if evidence.get("source_files", {}) != overrides.get("source_files", {}):
        errors.append("server_launch_source_mismatch")
    if evidence.get("absent_source_files", []) != overrides.get("absent_source_files", []):
        errors.append("server_launch_absent_source_mismatch")
    if isinstance(spec, Mapping):
        if evidence.get("owner") != "magpie":
            errors.append("server_launch_owner_mismatch")
        if evidence.get("recipe_source_files") != spec.get("source_files"):
            errors.append("server_launch_recipe_sources_mismatch")
        profiler = benchmark.get("profiler") or {}
        torch = profiler.get("torch_profiler") or {}
        if not any(code.endswith("_argv_invalid") for code in errors):
            errors.extend(_validate_managed_receipt(benchmark, evidence, workspace))
        if torch.get("enabled") is not True:
            if evidence.get("server_spec_sha256") != canonical_sha256(spec):
                errors.append("server_launch_spec_mismatch")
            if evidence.get("base_argv") != spec.get("argv"):
                errors.append("server_launch_recipe_argv_mismatch")
            if evidence.get("torch_profiler") is not None:
                errors.append("server_launch_unexpected_profile")
    return errors


def verified_workload_fingerprint(benchmark: Mapping[str, Any]) -> str | None:
    """Recompute the saved workload identity when interpreting benchmark artifacts."""
    workload = benchmark.get("workload_spec")
    execution = workload.get("execution") if isinstance(workload, Mapping) else None
    if not isinstance(execution, Mapping):
        return None
    static = execution.get("static_execution_fingerprint")
    if not isinstance(static, str) or not re.fullmatch(r"[a-f0-9]{64}", static):
        return None
    fingerprint = native_workload_fingerprint(benchmark, static)
    return fingerprint if fingerprint == execution.get("workload_fingerprint") else None
