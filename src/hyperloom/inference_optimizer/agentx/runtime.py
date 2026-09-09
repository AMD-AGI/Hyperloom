# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Execution-boundary preparation for AgentX runs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

# aiperf capability preflight is memoized per resolved binary: the probe shells out with a timeout and its result
# cannot change within a run, so a multi-point grid must not re-probe every round.
_PREFLIGHTED_BINS: dict[str, bool] = {}


def maybe_prepare_agentx(
    *,
    env: Mapping[str, str],
    inferencex_path: str,
    config_path: str | Path,
) -> bool:
    """Deploy the AgentX client + capability-preflight aiperf for a run."""
    try:
        bench = (yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}).get("benchmark", {}) or {}
    except Exception:  # noqa: BLE001 — config unreadable: let Magpie surface it
        bench = {}
    if str(bench.get("benchmark_script") or "") != "aiperf_client.sh":
        return False

    from .deploy import deploy_agentx_assets
    from .preflight import resolve_aiperf_bin

    # Deploy BEFORE preflight so the client is in place regardless of preflight memoization state.
    deploy_agentx_assets(Path(inferencex_path) / "benchmarks")
    aiperf_bin = resolve_aiperf_bin(env)
    bench_envs = bench.get("envs") if isinstance(bench.get("envs"), dict) else {}
    profiler = bench.get("profiler") if isinstance(bench.get("profiler"), dict) else {}
    torch_profiler = profiler.get("torch_profiler") if isinstance(profiler.get("torch_profiler"), dict) else {}
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
