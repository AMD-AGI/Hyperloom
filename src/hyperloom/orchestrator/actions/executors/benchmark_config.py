# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Materialize and prepare the workload through its owning benchmark contract."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ._accuracy_gate import materialized_run_eval_disabled


@dataclass(frozen=True)
class PreparationFailure:
    """A workload dependency that must be fixed before spawning the benchmark."""

    returncode: int
    error: str


def prepare_benchmark_runtime(config_path: Path, env: dict[str, str]) -> PreparationFailure | None:
    """Prepare synthetic or MLPerf dependencies; native AgentX is owned by Magpie."""
    from ._subprocess_kill import AGENTX_PREFLIGHT_RETURNCODE, EVAL_PROBE_UNPATCHABLE_RETURNCODE
    from ._inferencex_patcher import (
        ensure_benchmark_lib_eval_start_patched,
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
        env["MAGPIE_INFERENCEX_PATH"] = inferencex_path
        # Baseline patches its own checkout, but explore / sweep never pass through that hook: re-assert here so a
        # resumed session or a re-cloned checkout still emits the eval-start marker.
        ensure_benchmark_lib_eval_start_patched(Path(inferencex_path))

    # The generation bounds + pathology probe are asserted whether or not ``$INFERENCEX_PATH`` is set: unset falls
    # back to the same env discovery the baseline arm uses ($MAGPIE_PATH/InferenceX).
    probe_root = Path(inferencex_path) if inferencex_path else None
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
