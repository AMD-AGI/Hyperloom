# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runtime benchmark overrides shared by grid, baseline and profile launches."""

from __future__ import annotations

import os
from typing import Any

from hyperloom.common.agentx_workload import is_agentx_client_script, pin_mlperf_round_concurrency
from hyperloom.inference_optimizer.session.paths import session_dir
from hyperloom.orchestrator.state.base_launch import session_cards

from ._workload_envs import (
    apply_agentx_switch,
    apply_scriptable_runtime_defaults,
)
from ._gpu_pin import pin_rocr_visible_devices


def _apply_sizing_env_overrides(envs: dict[str, Any]) -> None:
    """Pull the round's shape (ISL/OSL/MAX_MODEL_LEN/TP/CONC) out of the ambient env into ``envs``."""
    for env_key in ("ISL", "OSL", "MAX_MODEL_LEN", "TP", "CONC"):
        val = os.environ.get(env_key, "").strip()
        if not val:
            continue
        # TP yaml-explicit wins: a stale state.tp must not downgrade a YAML-pinned TP.
        if env_key == "TP" and envs.get("TP") not in (None, 0, "", "0"):
            continue
        envs[env_key] = int(val)


def apply_runtime_benchmark_overrides(
    bench: dict[str, Any],
    *,
    model_path: str | None = None,
    gpu_type: str | None = None,
    benchmark_script: str | None = None,
    conc: Any = None,
) -> dict[str, Any]:
    """Apply runtime env/CLI overrides to a point YAML."""
    agentx_mode = True if is_agentx_client_script(str(bench.get("benchmark_script") or "")) else None

    if model_path:
        bench["model"] = str(model_path)

    precision = os.environ.get("PRECISION", "").strip()
    if precision:
        bench["precision"] = precision

    if benchmark_script:
        bench["benchmark_script"] = str(benchmark_script)

    apply_agentx_switch(bench, model_path, conc=conc, active=agentx_mode)

    envs: dict[str, Any] = bench.setdefault("envs", {})
    apply_scriptable_runtime_defaults(
        bench,
        envs,
        gpu_type=gpu_type,
        explicit_benchmark_script=bool(benchmark_script),
    )
    _apply_sizing_env_overrides(envs)
    pin_mlperf_round_concurrency(envs)
    pin_rocr_visible_devices(envs, cards=session_cards(session_dir(), int(envs.get("TP", 1) or 1)))

    return envs
