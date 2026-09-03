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
_PREFLIGHTED_BINS: dict[str, bool] = {}


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
        AgentXPreflightError: If aiperf is missing or not AgentX-capable and the
            packaged installer could not supply it.
    """
    try:
        bench = (yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}).get("benchmark", {}) or {}
    except Exception:  # noqa: BLE001 — config unreadable: let Magpie surface it
        bench = {}
    if str(bench.get("benchmark_script") or "") != "aiperf_client.sh":
        return False

    from .deploy import deploy_agentx_assets
    from .preflight import resolve_aiperf_bin

    # Deploy BEFORE preflight so the client is in place regardless of preflight
    # memoization state.
    deploy_agentx_assets(Path(inferencex_path) / "benchmarks")
    aiperf_bin = resolve_aiperf_bin(env)
    bench_envs = bench.get("envs") if isinstance(bench.get("envs"), dict) else {}
    profiler = bench.get("profiler") if isinstance(bench.get("profiler"), dict) else {}
    torch_profiler = profiler.get("torch_profiler") if isinstance(profiler.get("torch_profiler"), dict) else {}
    require_progress_api = str(bench_envs.get("PROFILE") or "") == "1" or bool(torch_profiler.get("enabled"))
    preflight_key = aiperf_bin or ""
    previous_check = _PREFLIGHTED_BINS.get(preflight_key)
    if previous_check is None or (require_progress_api and not previous_check):
        # A missing or stale client is installed here rather than reported, so
        # the memoized key is the binary that actually passed -- which the repair
        # may have only just put on PATH.
        aiperf_bin = _preflight_or_repair(aiperf_bin, env=env, require_progress_api=require_progress_api)
        _PREFLIGHTED_BINS[aiperf_bin or ""] = require_progress_api or bool(previous_check)
    return True


def _preflight_or_repair(
    aiperf_bin: str | None,
    *,
    env: Mapping[str, str],
    require_progress_api: bool = False,
) -> str | None:
    """Capability-check aiperf, installing the pinned build once if it is absent.

    aiperf is a dependency AgentX declares for itself and whose install this
    repository already owns (``install.sh::ensure_aiperf``, pinned by
    ``AIPERF_REF``). The preflight was the first place that knew it was missing,
    and it only said so -- to an operator who is not on this path. Downstream
    that read as an ordinary benchmark failure and opened an enablement round, so
    a supply problem was handed to a specialist as if it were a framework bug.

    Repairing here keeps the runtime flag as the single source of truth for "this
    box needs aiperf". Only a ``repairable`` verdict triggers an install: an
    operator pinning a corpus the scenario does not admit is not fixed by
    reinstalling the same build, and spending minutes on pip before saying so
    would only delay the diagnosis.

    Args:
        aiperf_bin: The binary resolved from ``env`` (None/empty means absent).
        env: Environment the benchmark subprocess will run with.

    Returns:
        The binary that passed the check -- re-resolved after a repair, since the
        install is what puts it on ``PATH``.

    Raises:
        AgentXPreflightError: If the build is unusable and could not be repaired.
            A post-repair failure is marked non-repairable so no caller retries
            an install that has already been tried and did not help.
    """
    from .preflight import AgentXPreflightError, check_aiperf_capability, resolve_aiperf_bin

    try:
        check_aiperf_capability(aiperf_bin, env=env, require_progress_api=require_progress_api)
        return aiperf_bin
    except AgentXPreflightError as exc:
        if not getattr(exc, "repairable", False):
            raise
        # An operator override is not a supply gap, and installing cannot close
        # it: ``ensure_aiperf`` returns 0 without doing anything when AIPERF_BIN
        # is set, and ``resolve_aiperf_bin`` would hand back that same binary
        # afterwards. Repairing here would report a successful install that
        # never happened and steer the reader away from the one thing that
        # fixes it.
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

    # Re-resolve: the install is what put aiperf on PATH, so the pre-repair
    # lookup (possibly None) says nothing about what is there now.
    repaired_bin = resolve_aiperf_bin(env)
    try:
        check_aiperf_capability(repaired_bin, env=env, require_progress_api=require_progress_api)
    except AgentXPreflightError as exc:
        # The install reported success and the build is still unusable, so this
        # is no longer a supply gap this process can close. Re-raise it as
        # non-repairable: the repair result is memoized as a success, so a later
        # round that trusted ``repairable`` would re-enter this branch, get that
        # memoized success back, and arrive here again having done nothing.
        raise AgentXPreflightError(
            f"{exc} The pinned build was installed during this run and the check still fails.",
            repairable=False,
        ) from exc
    return repaired_bin
