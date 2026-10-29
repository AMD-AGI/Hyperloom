# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Materialize and prepare the workload through its owning benchmark contract."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx

from ._accuracy_gate import materialized_run_eval_disabled
from ._subprocess_kill import BENCHMARK_TIMEOUT_ENV, default_benchmark_timeout_sec, resolve_benchmark_timeouts


@dataclass(frozen=True)
class PreparationFailure:
    """A workload dependency that must be fixed before spawning the benchmark."""

    returncode: int
    error: str


def composed_benchmark_timeout_sec(config_path: Path) -> float:
    """The hard cap the rounds of one materialized config are granted, sized to the env that config launches with.

    The workload a round has to get through is named by the composed env, not by the
    variant spec: the pin that sizes it -- a sweep's ``CONC``, an agentic flow -- is as
    often inherited from the base YAML or from the recipe's base layer as it is set by
    the variant, and a variant may equally clear one the base pinned. Sizing the grant
    from the variant's own deltas alone would grant the time some other, usually
    smaller, run needs, so it is read back off the config the round is launched from.

    Only the size comes from that env. The hard cap's override is the operator's grant,
    read and validated once against the process env; a config's value for that name is
    the workload's own and is never parsed as a grant here. A round whose config clears
    the override runs on the size its workload needs. A config that names a workload pin
    this cannot read raises, as launching that round would: the caller owns the blast
    radius of one unusable config, and the grid runner ends that variant on it.
    """
    bench = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("benchmark") or {}
    envs = dict(os.environ)
    envs.update({str(k): str(v) for k, v in (bench.get("envs") or {}).items()})
    if is_magpie_agentx(bench):
        # AgentX keeps the variant's deltas in its launch overrides and applies them over the
        # inherited env when it starts; the same order here, so the cap is sized to the env the
        # round runs under rather than to the one it inherited.
        launch = (bench.get("agentx") or {}).get("launch_overrides") or {}
        for name in launch.get("unset_env") or []:
            envs.pop(str(name), None)
        envs.update({str(k): str(v) for k, v in (launch.get("env") or {}).items()})
    if BENCHMARK_TIMEOUT_ENV in envs and BENCHMARK_TIMEOUT_ENV in os.environ:
        return resolve_benchmark_timeouts()[1]
    return default_benchmark_timeout_sec(envs)


def prepare_benchmark_runtime(config_path: Path, env: dict[str, str]) -> PreparationFailure | None:
    """Prepare synthetic or MLPerf dependencies; native AgentX is owned by Magpie."""
    from hyperloom.inference_optimizer import inferencex_checkout

    from ._subprocess_kill import (
        AGENTX_PREFLIGHT_RETURNCODE,
        EVAL_PROBE_UNPATCHABLE_RETURNCODE,
        INFERENCEX_CHECKOUT_INVALID_RETURNCODE,
    )
    from ._inferencex_patcher import (
        ensure_benchmark_lib_eval_start_patched,
        ensure_benchmark_serving_seed_patched,
        ensure_eval_probe_patched,
        ensure_eval_unbound_outputs_patched,
        eval_probe_targets_exist,
    )

    log = logging.getLogger(__name__)
    from .magpie_agentx import native_config

    if native_config(config_path):
        return None
    # Pin Magpie's InferenceX resolution to ``$INFERENCEX_PATH`` (its highest-precedence rung) so it loads the patched
    # checkout, not a stale copy.
    inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
    if inferencex_path:
        invalid = inferencex_checkout.unusable_reason(inferencex_path)
        if invalid:
            log.error("grid_runner: %s; failing this benchmark", invalid)
            return PreparationFailure(INFERENCEX_CHECKOUT_INVALID_RETURNCODE, invalid)
        env["MAGPIE_INFERENCEX_PATH"] = inferencex_path
        # Baseline patches its own checkout, but explore / sweep never pass through that hook: re-assert here so a
        # resumed session or a re-cloned checkout still emits the eval-start marker.
        ensure_benchmark_lib_eval_start_patched(Path(inferencex_path))

    # The generation bounds + pathology probe are asserted whether or not ``$INFERENCEX_PATH`` is set: unset falls
    # back to the same env discovery the baseline arm uses ($MAGPIE_PATH/InferenceX).
    probe_root = Path(inferencex_path) if inferencex_path else None
    # Best-effort: without it a warmup round pre-fills the measured round's prefix cache, which inflates the number
    # but never breaks the round.
    ensure_benchmark_serving_seed_patched(probe_root)
    # Best-effort, unlike the probe below: a missing guard only costs the reason a failed round reports.
    ensure_eval_unbound_outputs_patched(probe_root)
    if not ensure_eval_probe_patched(probe_root) and not materialized_run_eval_disabled(config_path):
        eval_bounds_msg = (
            "eval generation bounds + pathology probe are not installed "
            "(utils/evals/patches/lm_eval_sitecustomize.py, inferencex="
            f"{inferencex_path or '<unset>'}, INFERENCEX_PATH="
            f"{os.environ.get('INFERENCEX_PATH', '') or '<unset>'}); this "
            "variant runs eval, so it would be scored against a baseline that "
            "truncated at a different point"
        )
        if eval_probe_targets_exist(probe_root):
            log.error("grid_runner: %s; failing this benchmark", eval_bounds_msg)
            return PreparationFailure(EVAL_PROBE_UNPATCHABLE_RETURNCODE, eval_bounds_msg)
        log.warning("grid_runner: %s", eval_bounds_msg)
    from ._workload_envs import prepare_agentx_runtime

    _agx_err = prepare_agentx_runtime(env=env, inferencex_path=inferencex_path, config_path=config_path)
    if _agx_err:
        log.error("%s; failing this benchmark", _agx_err)
        return PreparationFailure(AGENTX_PREFLIGHT_RETURNCODE, _agx_err)
    return None


def apply_benchmark_round_seed(config_path: Path, env: dict[str, str], *, warmup: bool) -> None:
    """Reseed a discarded warmup round's client so it cannot pre-fill the measured round's prefix cache.

    Native AgentX replays its own workload through Magpie and is left exactly as it was.

    Args:
        config_path: The round's materialized Magpie YAML.
        env: The benchmark subprocess env, edited in place.
        warmup: Whether the round's result is discarded.
    """
    from ._inferencex_patcher import apply_round_seed
    from .magpie_agentx import native_config

    apply_round_seed(env, warmup=warmup and not native_config(config_path))
