# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Execution-boundary preparation for AgentX runs.

Factored out of ``_grid_runner._run_magpie`` so the deploy + capability-preflight
logic is directly unit-testable (the in-place hook self-disables under pytest and
was otherwise uncoverable). The caller is responsible for the OFF-path gate
(``agentx_enabled``) so this module — and the ``agentx`` package — is imported
only when AgentX is actually on (A2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

# aiperf capability preflight is memoized per resolved binary: the probe shells
# out with a timeout and its result cannot change within a run, so a multi-point
# grid must not re-probe every round.
_PREFLIGHTED_BINS: set[tuple[str, bool]] = set()


def maybe_prepare_agentx(
    *,
    env: Mapping[str, str],
    inferencex_path: str,
    config_path: str | Path,
) -> bool:
    """Deploy the AgentX client + capability-preflight aiperf for a run.

    Only acts when the materialized config's ``benchmark_script`` is the AgentX
    client; otherwise a no-op returning False. Deploys every call (idempotent +
    cheap, survives Magpie re-resolving InferenceX); preflight is memoized per
    resolved binary.

    Args:
        env: The child-process environment (used to resolve aiperf + its PATH).
        inferencex_path: Resolved InferenceX checkout (its ``benchmarks/`` dir
            receives the assets).
        config_path: The materialized Magpie YAML for this round.

    Returns:
        True if AgentX assets were prepared, False if the resolved script is not
        the AgentX client.

    Raises:
        AgentXPreflightError: If aiperf is missing or not AgentX-capable.
    """
    try:
        bench = (yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}).get("benchmark", {}) or {}
    except Exception:  # noqa: BLE001 — config unreadable: let Magpie surface it
        bench = {}
    if str(bench.get("benchmark_script") or "") != "aiperf_client.sh":
        return False

    from .deploy import deploy_agentx_assets
    from .preflight import check_aiperf_capability, resolve_aiperf_bin

    # Deploy BEFORE preflight so the client is in place regardless of preflight
    # memoization state.
    deploy_agentx_assets(Path(inferencex_path) / "benchmarks")
    aiperf_bin = resolve_aiperf_bin(env)
    bench_envs = bench.get("envs") if isinstance(bench.get("envs"), dict) else {}
    profiler = bench.get("profiler") if isinstance(bench.get("profiler"), dict) else {}
    torch_profiler = (
        profiler.get("torch_profiler") if isinstance(profiler.get("torch_profiler"), dict) else {}
    )
    require_progress_api = str(bench_envs.get("PROFILE") or "") == "1" or bool(torch_profiler.get("enabled"))
    preflight_key = (aiperf_bin or "", require_progress_api)
    if preflight_key not in _PREFLIGHTED_BINS:
        check_aiperf_capability(
            aiperf_bin,
            require_progress_api=require_progress_api,
        )  # raises if missing/incapable
        _PREFLIGHTED_BINS.add(preflight_key)
    return True
