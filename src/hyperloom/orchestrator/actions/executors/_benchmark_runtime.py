# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runtime benchmark overrides shared by grid, baseline and profile launches."""

from __future__ import annotations

import os
from typing import Any

from ._workload_envs import apply_agentx_switch, apply_scriptable_runtime_defaults


def apply_runtime_benchmark_overrides(
    bench: dict[str, Any],
    *,
    model_path: str | None = None,
    gpu_type: str | None = None,
    benchmark_script: str | None = None,
    conc: Any = None,
    agentx_mode: bool | None = None,
) -> dict[str, Any]:
    """Apply runtime env/CLI overrides to a Magpie benchmark YAML."""
    if agentx_mode is None and str(bench.get("benchmark_script") or "") == "aiperf_client.sh":
        agentx_mode = True

    if model_path:
        bench["model"] = str(model_path)

    precision = os.environ.get("PRECISION", "").strip()
    if precision:
        bench["precision"] = precision

    if gpu_type:
        bench["runner_type"] = str(gpu_type)
        # Force-pin the generic ``{framework}_{gpu_type}.sh`` so Magpie's resolver doesn't fall through to InferenceX
        # native scripts that ignore ``EXTRA_*_ARGS``.
        framework = str(bench.get("framework") or "").lower()
        if framework:
            bench["benchmark_script"] = f"{framework}_{gpu_type}.sh"
        else:
            bench.pop("benchmark_script", None)

    if benchmark_script:
        bench["benchmark_script"] = str(benchmark_script)

    # AgentX switch on the shared rebuild path: without this, the gpu_type block above re-pins the synthetic
    # {framework}_{gpu_type}.sh and silently reverts a materialize-time AgentX swap (grid/baseline/profile executors
    # rebuild via this function).
    apply_agentx_switch(bench, model_path, conc=conc, active=agentx_mode)

    envs: dict[str, Any] = bench.setdefault("envs", {})
    # Same hazard as the AgentX swap above: the gpu_type block re-pins the bare {framework}_{gpu_type}.sh over the
    # bundled absolute path the materialize path resolved, so grid variants must re-apply the scriptable defaults.
    apply_scriptable_runtime_defaults(
        bench,
        envs,
        gpu_type=gpu_type,
        explicit_benchmark_script=bool(benchmark_script),
    )
    for env_key in ("ISL", "OSL", "MAX_MODEL_LEN", "TP", "CONC"):
        val = os.environ.get(env_key, "").strip()
        if not val:
            continue
        # TP yaml-explicit wins: a stale state.tp must not downgrade a YAML-pinned TP.
        if env_key == "TP":
            yaml_tp = envs.get("TP")
            if yaml_tp not in (None, 0, "", "0"):
                continue
        envs[env_key] = int(val)

    explicit_rocr = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    if explicit_rocr:
        envs["ROCR_VISIBLE_DEVICES"] = explicit_rocr
    else:
        tp_val = int(envs.get("TP", 1) or 1)
        existing_rocr = str(envs.get("ROCR_VISIBLE_DEVICES", "")).strip()
        existing_count = len([x for x in existing_rocr.split(",") if x.strip()]) if existing_rocr else 0
        if tp_val > 1 and existing_count < tp_val:
            envs["ROCR_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(tp_val))

    return envs
