# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Execution-boundary preparation for AgentX runs."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Mapping, MutableMapping

import yaml

# aiperf capability preflight is memoized per resolved binary: the probe shells out with a timeout and its result
# cannot change within a run, so a multi-point grid must not re-probe every round.
_PREFLIGHTED_BINS: dict[str, bool] = {}


def _profile_compatibility_checkout(
    bench: Mapping[str, object],
    *,
    config_path: str | Path,
    explicit_inferencex_path: str,
    env: Mapping[str, str],
) -> Path:
    """Validate the one non-native leg allowed in a pinned AgentX session.

    ``ProfileExecutor`` deliberately replaces native Magpie AgentX with the
    phase-gated ``aiperf_client.sh`` compatibility harness.  That exception
    must be narrow: it is diagnostic-only, runs from the disposable checkout
    beside the materialized profile config, and stays at the session's pinned
    InferenceX commit.
    """
    if str(bench.get("benchmark_script") or "") != "aiperf_client.sh":
        raise ValueError(
            "Pinned native AgentX session requires benchmark.agentx to remain "
            "enabled outside the verified profiler compatibility path"
        )
    raw_workload = bench.get("workload_spec")
    workload = raw_workload if isinstance(raw_workload, Mapping) else {}
    if str(workload.get("harness") or "") != "hyperloom-profiler-compat":
        raise ValueError("Pinned AgentX profiler compatibility config is missing its diagnostic harness identity")
    raw_envs = bench.get("envs")
    bench_envs = raw_envs if isinstance(raw_envs, Mapping) else {}
    raw_profiler = bench.get("profiler")
    profiler = raw_profiler if isinstance(raw_profiler, Mapping) else {}
    raw_torch = profiler.get("torch_profiler")
    torch_profiler = raw_torch if isinstance(raw_torch, Mapping) else {}

    def enabled(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
                "enable",
                "enabled",
            }
        return bool(value)

    if not (enabled(bench_envs.get("PROFILE")) or enabled(torch_profiler.get("enabled"))):
        raise ValueError("Pinned AgentX profiler compatibility config has no active torch profile marker")

    required_pins = (
        "HYPERLOOM_AGENTX_EXPECTED_RECIPE_FINGERPRINT",
        "HYPERLOOM_AGENTX_EXPECTED_EXECUTION_FINGERPRINT",
        "HYPERLOOM_AGENTX_GPU_COUNT",
        "MAGPIE_REF",
        "INFERENCEX_REF",
    )
    missing = [name for name in required_pins if not str(env.get(name) or "").strip()]
    if missing:
        raise ValueError(
            "Pinned AgentX profiler compatibility path is missing session authority: " + ", ".join(missing)
        )
    expected_ref = str(env.get("INFERENCEX_REF") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", expected_ref):
        raise ValueError("Pinned AgentX profiler compatibility path requires a full 40-character INFERENCEX_REF")

    configured = str(bench.get("inferencex_path") or "").strip()
    if not configured or not explicit_inferencex_path:
        raise ValueError("Pinned AgentX profiler compatibility path requires an explicit isolated InferenceX checkout")
    from .native import validate_native_checkout_path

    configured_path = Path(configured).expanduser()
    explicit_path = Path(explicit_inferencex_path).expanduser()
    if configured_path.is_symlink() or explicit_path.is_symlink():
        raise ValueError("Pinned AgentX profiler compatibility checkout must not be a symbolic link")
    try:
        resolved = validate_native_checkout_path(configured).resolve(strict=True)
        explicit_resolved = validate_native_checkout_path(explicit_inferencex_path).resolve(strict=True)
        config_parent = Path(config_path).expanduser().resolve(strict=True).parent
    except (OSError, RuntimeError) as exc:
        raise ValueError("Pinned AgentX profiler compatibility checkout is not resolvable") from exc
    if (
        resolved != explicit_resolved
        or resolved.name != ".agentx-profile-inferencex"
        or resolved.parent != config_parent
    ):
        raise ValueError(
            "Pinned AgentX profiler compatibility checkout is not the isolated tree beside its materialized config"
        )
    try:
        top_proc = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        head_proc = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Cannot verify the pinned AgentX profiler compatibility checkout") from exc
    try:
        top = Path((top_proc.stdout or "").strip()).resolve(strict=True)
    except (OSError, RuntimeError):
        top = Path()
    head = (head_proc.stdout or "").strip().lower()
    if top_proc.returncode != 0 or head_proc.returncode != 0 or top != resolved or head != expected_ref:
        raise ValueError(
            "Pinned AgentX profiler compatibility checkout does not match the session's isolated InferenceX revision"
        )
    return resolved


def _scrub_native_agentx_ambient_env(env: MutableMapping[str, str]) -> None:
    """Remove replay controls that Magpie's local runner would inherit.

    Magpie deliberately starts its local launcher from ``os.environ.copy()``.
    Native AgentX replay controls must instead come from the resolved YAML or
    the pinned InferenceX launcher; otherwise an unrelated login-shell export
    can change the workload without changing its recipe fingerprint.
    """
    from .native import scrub_native_agentx_ambient_env

    scrub_native_agentx_ambient_env(env)


def maybe_prepare_agentx(
    *,
    env: MutableMapping[str, str],
    inferencex_path: str,
    config_path: str | Path,
    allow_profile_compat: bool = False,
) -> bool:
    """Prepare either native Magpie AgentX or the profiler compatibility client."""
    expected_recipe = str(env.get("HYPERLOOM_AGENTX_EXPECTED_RECIPE_FINGERPRINT") or "").strip()
    expected_execution = str(env.get("HYPERLOOM_AGENTX_EXPECTED_EXECUTION_FINGERPRINT") or "").strip()
    expected_materialized_execution = str(
        env.get("HYPERLOOM_AGENTX_EXPECTED_MATERIALIZED_EXECUTION_FINGERPRINT") or ""
    ).strip()
    expected_gpu_count = str(env.get("HYPERLOOM_AGENTX_GPU_COUNT") or "").strip()
    # These pins are written only after the CLI has selected and accepted one
    # native recipe.  From that point on they are the authority at the process
    # boundary: editing the materialized YAML must never turn a native run into
    # a generic Magpie run or replace the accepted identity with self-asserted
    # YAML metadata.
    pinned_native_session = any((expected_recipe, expected_execution, expected_gpu_count))
    try:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        bench = config.get("benchmark", {}) if isinstance(config, dict) else {}
        bench = bench if isinstance(bench, dict) else {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError, TypeError, ValueError) as exc:
        if pinned_native_session:
            raise ValueError("Pinned native AgentX config is unreadable at the launch boundary") from exc
        bench = {}
    from .native import (
        native_agentx_enabled,
        native_execution_identity,
        resolve_native_launcher,
    )

    native = native_agentx_enabled(bench.get("agentx"))
    explicit_path = str(inferencex_path or "").strip()
    configured_path = str(bench.get("inferencex_path") or "").strip()
    runtime_path = str(env.get("INFERENCEX_PATH") or "").strip()
    effective_inferencex_path = (
        explicit_path or runtime_path or configured_path if native else explicit_path or configured_path or runtime_path
    )
    pinned_profile_compat = False
    if pinned_native_session and not native:
        if not allow_profile_compat:
            raise ValueError("Pinned native AgentX session requires benchmark.agentx to remain enabled")
        _profile_compatibility_checkout(
            bench,
            config_path=config_path,
            explicit_inferencex_path=explicit_path,
            env=env,
        )
        pinned_profile_compat = True
    if native:
        if not effective_inferencex_path:
            raise ValueError(
                "Native AgentX requires an InferenceX checkout via benchmark.inferencex_path or INFERENCEX_PATH"
            )
        resolve_native_launcher(
            inferencex_path=effective_inferencex_path,
            benchmark_script=str(bench.get("benchmark_script") or ""),
        )
        # Normalize the exact environment Magpie will inherit before identity
        # validation.  Its local runner copies this mapping and then overlays
        # benchmark.envs, so hashing the pre-scrub login shell would either
        # miss the real child value or report false drift for blocked hooks.
        _scrub_native_agentx_ambient_env(env)
        raw_envs = bench.get("envs")
        bench_envs = raw_envs if isinstance(raw_envs, dict) else {}
        local_model_path = str(bench_envs.get("MODEL_PATH") or "").strip()
        if local_model_path:
            env["MODEL_PATH"] = local_model_path
        else:
            env.pop("MODEL_PATH", None)
        raw_workload = bench.get("workload_spec")
        workload = raw_workload if isinstance(raw_workload, dict) else {}
        raw_execution = workload.get("execution")
        execution = raw_execution if isinstance(raw_execution, dict) else {}
        yaml_execution = str(execution.get("static_execution_fingerprint") or "").strip()
        if pinned_native_session:
            missing_pins = [
                name
                for name, value in (
                    ("HYPERLOOM_AGENTX_EXPECTED_RECIPE_FINGERPRINT", expected_recipe),
                    ("HYPERLOOM_AGENTX_EXPECTED_EXECUTION_FINGERPRINT", expected_execution),
                    (
                        "HYPERLOOM_AGENTX_EXPECTED_MATERIALIZED_EXECUTION_FINGERPRINT",
                        expected_materialized_execution,
                    ),
                    ("HYPERLOOM_AGENTX_GPU_COUNT", expected_gpu_count),
                    ("MAGPIE_REF", str(env.get("MAGPIE_REF") or "").strip()),
                    ("INFERENCEX_REF", str(env.get("INFERENCEX_REF") or "").strip()),
                )
                if not value
            ]
            if missing_pins:
                raise ValueError("Pinned native AgentX session is missing launch authority: " + ", ".join(missing_pins))
            if not yaml_execution:
                raise ValueError("Pinned native AgentX config is missing workload_spec.execution")
            if yaml_execution != expected_execution:
                raise ValueError(
                    "Native AgentX YAML execution fingerprint differs from the "
                    f"session pin: {yaml_execution!r} != {expected_execution!r}"
                )
            yaml_materialized_execution = str(execution.get("execution_fingerprint") or "").strip()
            if yaml_materialized_execution != expected_materialized_execution:
                raise ValueError(
                    "Native AgentX YAML materialized execution fingerprint "
                    "differs from the session pin: "
                    f"{yaml_materialized_execution!r} != "
                    f"{expected_materialized_execution!r}"
                )

        if not pinned_native_session:
            # Standalone callers may use this helper as a lightweight native
            # launcher/path validator before the Hyperloom CLI has finalized a
            # session.  There is no accepted identity to compare yet; the
            # strict re-resolution below is reserved for a pinned session.
            return True

        # Re-resolve at the last possible boundary.  The recipe fingerprint is
        # intentionally concurrency-independent, so compare the complete
        # persisted topology and recipe identity as well as both fingerprints.
        from .native import preview_native_recipe

        preview = preview_native_recipe(
            dict(bench),
            inferencex_path=effective_inferencex_path,
        )
        topology = preview.get("topology")
        topology = topology if isinstance(topology, dict) else {}
        entry = preview.get("entry")
        entry = entry if isinstance(entry, dict) else {}
        resolved_benchmark = preview.get("benchmark")
        resolved_benchmark = resolved_benchmark if isinstance(resolved_benchmark, dict) else {}
        config_file = str(preview.get("config_file") or "")
        current_recipe = {
            "name": str(preview.get("recipe") or ""),
            "config_file": config_file,
            "recipe_fingerprint": str(topology.get("recipe_fingerprint") or ""),
            "image": str(entry.get("image") or resolved_benchmark.get("docker_image") or ""),
            "runner": str(entry.get("runner") or ""),
            "model": str(entry.get("model") or ""),
            "model_prefix": str(entry.get("model-prefix") or ""),
            "framework": str(entry.get("framework") or ""),
            "precision": str(entry.get("precision") or ""),
            "concurrency": int(topology.get("conc") or 0),
            "duration_seconds": int(topology.get("duration_seconds") or 0),
            "launcher": str(resolved_benchmark.get("benchmark_script") or ""),
        }
        current_topology = {
            key: topology.get(key)
            for key in (
                "tp",
                "pp",
                "pcp_size",
                "ep",
                "gpu_count",
                "conc",
                "duration_seconds",
                "recipe_fingerprint",
            )
        }
        if pinned_native_session:
            if current_recipe["recipe_fingerprint"] != expected_recipe:
                raise ValueError(
                    "Native AgentX recipe changed after materialization: "
                    f"expected {expected_recipe!r}, resolved "
                    f"{current_recipe['recipe_fingerprint']!r}"
                )
            try:
                pinned_gpu_count = int(expected_gpu_count)
            except ValueError as exc:
                raise ValueError("HYPERLOOM_AGENTX_GPU_COUNT must be a positive integer") from exc
            if pinned_gpu_count <= 0 or current_topology["gpu_count"] != pinned_gpu_count:
                raise ValueError(
                    "Native AgentX physical topology changed after materialization: "
                    f"expected {pinned_gpu_count} GPUs, resolved "
                    f"{current_topology['gpu_count']!r}"
                )
            saved_topology = workload.get("resolved_topology")
            saved_recipe = workload.get("recipe")
            if not isinstance(saved_topology, dict) or any(
                saved_topology.get(key) != value for key, value in current_topology.items()
            ):
                raise ValueError("Native AgentX resolved topology changed after materialization")
            if not isinstance(saved_recipe, dict) or any(
                saved_recipe.get(key) != value for key, value in current_recipe.items()
            ):
                raise ValueError("Native AgentX recipe identity changed after materialization")

        current_execution = native_execution_identity(
            inferencex_path=effective_inferencex_path,
            benchmark_script=str(resolved_benchmark.get("benchmark_script") or ""),
            config_file=config_file,
            resolved_benchmark=resolved_benchmark,
            expected_ref=str(env.get("INFERENCEX_REF") or ""),
            magpie_execution=preview.get("magpie_execution"),
            expected_magpie_ref=str(env.get("MAGPIE_REF") or ""),
            launch_env=env,
        )
        if expected_execution and current_execution["static_execution_fingerprint"] != expected_execution:
            raise ValueError(
                "Native AgentX execution inputs changed after materialization: "
                f"expected {expected_execution!r}, resolved "
                f"{current_execution['static_execution_fingerprint']!r}"
            )
        if (
            expected_materialized_execution
            and current_execution["execution_fingerprint"] != expected_materialized_execution
        ):
            raise ValueError(
                "Native AgentX resolved BenchmarkConfig changed after "
                "materialization: expected execution fingerprint "
                f"{expected_materialized_execution!r}, resolved "
                f"{current_execution['execution_fingerprint']!r}"
            )
        if pinned_native_session and execution != current_execution:
            raise ValueError("Native AgentX persisted execution identity changed after materialization")
        return True

    if str(bench.get("benchmark_script") or "") != "aiperf_client.sh":
        return False
    if pinned_native_session and not pinned_profile_compat:
        raise ValueError("Pinned native AgentX session cannot use an unverified generic client")
    if not effective_inferencex_path:
        raise ValueError("AgentX requires an InferenceX checkout via benchmark.inferencex_path or INFERENCEX_PATH")

    from .deploy import deploy_agentx_assets
    from .preflight import resolve_aiperf_bin

    # Deploy BEFORE preflight so the client is in place regardless of preflight memoization state.
    deploy_agentx_assets(Path(effective_inferencex_path) / "benchmarks")
    aiperf_bin = resolve_aiperf_bin(env)
    raw_bench_envs = bench.get("envs")
    bench_envs = raw_bench_envs if isinstance(raw_bench_envs, dict) else {}
    raw_profiler = bench.get("profiler")
    profiler = raw_profiler if isinstance(raw_profiler, dict) else {}
    raw_torch_profiler = profiler.get("torch_profiler")
    torch_profiler = raw_torch_profiler if isinstance(raw_torch_profiler, dict) else {}
    require_progress_api = str(bench_envs.get("PROFILE") or "") == "1" or bool(torch_profiler.get("enabled"))
    preflight_key = aiperf_bin or ""
    previous_check = _PREFLIGHTED_BINS.get(preflight_key)
    if previous_check is None or (require_progress_api and not previous_check):
        # A missing or stale client is installed here rather than reported, so the memoized key is the binary that
        # actually passed -- which the repair may have only just put on PATH.
        aiperf_bin = _preflight_or_repair(aiperf_bin, env=env, require_progress_api=require_progress_api)
        _PREFLIGHTED_BINS[aiperf_bin or ""] = require_progress_api or bool(previous_check)
    return True


def _preflight_or_repair(
    aiperf_bin: str | None,
    *,
    env: Mapping[str, str],
    require_progress_api: bool = False,
) -> str | None:
    """Capability-check aiperf, installing the pinned build once if it is absent."""
    from .preflight import AgentXPreflightError, check_aiperf_capability, resolve_aiperf_bin

    try:
        check_aiperf_capability(aiperf_bin, env=env, require_progress_api=require_progress_api)
        return aiperf_bin
    except AgentXPreflightError as exc:
        if not getattr(exc, "repairable", False):
            raise
        # An operator override is not a supply gap, and installing cannot close it: ``ensure_aiperf`` returns 0
        # without doing anything when AIPERF_BIN is set, and ``resolve_aiperf_bin`` would hand back that same binary
        # afterwards.
        override = (env.get("AIPERF_BIN") or "").strip()
        if override:
            raise AgentXPreflightError(
                f"{exc} AIPERF_BIN is set to {override!r}, so this is the build being "
                f"checked and no install can replace it. Point AIPERF_BIN at a pinned "
                f"build, or unset it and let install.sh supply one.",
                repairable=False,
            ) from exc
        from .repair import ensure_aiperf_installed

        repair_error = ensure_aiperf_installed(env=env)
        if repair_error is not None:
            raise AgentXPreflightError(
                f"{exc} Automatic repair was attempted and failed: {repair_error}",
                repairable=False,
            ) from exc

    # Re-resolve: the install is what put aiperf on PATH, so the pre-repair lookup (possibly None) says nothing about
    # what is there now.
    repaired_bin = resolve_aiperf_bin(env)
    try:
        check_aiperf_capability(repaired_bin, env=env, require_progress_api=require_progress_api)
    except AgentXPreflightError as exc:
        # The install reported success and the build is still unusable, so this is no longer a supply gap this process
        # can close.
        raise AgentXPreflightError(
            f"{exc} The pinned build was installed during this run and the check still fails.",
            repairable=False,
        ) from exc
    return repaired_bin
