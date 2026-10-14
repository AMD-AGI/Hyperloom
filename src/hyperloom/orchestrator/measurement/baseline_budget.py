# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stop the run when the baseline itself breaks a session budget.

Every candidate is graded against the baseline, so a baseline over the latency or power budget means nothing that
follows can clear it. Stopping costs one baseline; continuing spends the whole ``--max-hours`` refusing every candidate
to learn something already knowable.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from hyperloom.common.perf_metric import latency_veto_reason, power_veto_reason

log = logging.getLogger(__name__)

#: What the run report says for each stop this module decides; merged into the report's stop-reason table.
BASELINE_BUDGET_STOP_EXPLANATIONS: dict[str, str] = {
    "baseline_over_latency_budget": (
        "The baseline's own mean end-to-end latency exceeded --max-latency-ms, or the baseline reported no "
        "end-to-end latency at all, so the run stopped before optimizing. The budget refuses any KEEP over the "
        "ceiling or without a measured latency, and the reference the run is measured against already fails it — no "
        "candidate built on it could have been promoted, so continuing would have spent the whole time budget "
        "refusing every winner in turn. If the baseline reported no latency, make the workload's entrypoint write "
        "e2el_mean_ms; otherwise either the ceiling is lower than this workload's floor on this hardware, or the "
        "baseline configuration itself is the thing to fix. Relaunch with a ceiling the baseline can meet, or "
        "without one, to see what the search finds."
    ),
    "baseline_over_power_budget": (
        "The baseline's serving GPUs drew more than --max-power-w in total, a card drew more than its "
        "--max-per-gpu-power-w limit, or the baseline round reported no per-GPU power, so the run stopped before "
        "optimizing: every candidate is measured against that reference, so none could have been promoted. If the "
        "round reported no power, read its gpu_power.json: unread_gpus names cards that may have been serving but "
        "could not be read, an empty serving_gpus means no card held a model during the measured phase, and "
        "query_errors means amd-smi could not be read. Otherwise lower the cards' power cap (amd-smi set "
        "--power-cap) so the baseline fits, or relaunch with a budget the baseline can meet."
    ),
}


def _per_gpu_watts(by_gpu: Mapping[str, float]) -> str:
    return ", ".join(f"GPU {gpu} {watts:.0f} W" for gpu, watts in sorted(by_gpu.items()))


def baseline_budget_stop_reason(state: Any, result: Mapping[str, Any]) -> str:
    """The stop reason when the baseline *result* breaks a budget, logged; ``""`` when it fits every one."""
    latency_budget = float(getattr(state, "latency_budget_ms", 0.0) or 0.0)
    latency_veto = latency_veto_reason(result.get("e2el_mean_ms"), latency_budget)
    if latency_veto:
        e2el = result.get("e2el_mean_ms")
        log.error(
            "baseline does not satisfy --max-latency-ms (%s): budget %.1f ms, baseline %s. "
            "No candidate can clear a ceiling the reference already breaks; stopping.",
            latency_veto,
            latency_budget,
            f"{float(e2el):.1f} ms" if isinstance(e2el, (int, float)) else "reported no end-to-end latency",
        )
        return "baseline_over_latency_budget"
    power_budget = float(getattr(state, "power_budget_w", 0.0) or 0.0)
    per_gpu_budget = dict(getattr(state, "power_budget_per_gpu_w", None) or {})
    by_gpu = result.get("gpu_power_by_gpu_w")
    power_veto = power_veto_reason(by_gpu, total_budget_w=power_budget, per_gpu_budget_w=per_gpu_budget)
    if not power_veto:
        return ""
    measured = (
        f"{sum(by_gpu.values()):.0f} W total ({_per_gpu_watts(by_gpu)})"
        if isinstance(by_gpu, Mapping) and by_gpu
        else "reported no per-GPU power"
    )
    log.error(
        "baseline does not satisfy the power budget (%s): --max-power-w %s, --max-per-gpu-power-w %s; "
        "baseline %s. No candidate can clear a budget the reference already breaks; stopping.",
        power_veto,
        f"{power_budget:.0f} W" if power_budget > 0 else "unset",
        _per_gpu_watts(per_gpu_budget) or "unset",
        measured,
    )
    return "baseline_over_power_budget"
