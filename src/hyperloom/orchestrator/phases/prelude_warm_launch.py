# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The recipe config a PRELUDE warm replay launches with, the launches it skips, and the task it opens."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hyperloom.common.perf_metric import agentx_active
from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import (
    SKIP_CONTEXT_COPY_DRAFT,
    SKIP_RESTATES_BASELINE,
)

from ..actions.executors._grid_runner import GridRecipe, GridVariant
from ..actions.executors._spec_decode_context_copy import context_copy_refusal
from ..actions.executors.explore import filter_context_copy_drafts, restates_the_baseline_launch
from ..state.task_registry import TaskSpec

log = logging.getLogger(__name__)


def suppress_recipe_config(config: dict[str, Any]) -> None:
    """Blank a low-confidence recipe's own config so only its kernel section is replayed."""
    config["extra_server_args"] = ""
    config["extra_envs"] = {}
    config["config_source"] = ""
    config["config_tier"] = "suppressed_low_confidence"
    config["donor_expected_gain"] = 0.0


def warm_replay_task_spec(params: dict[str, Any], *, lanes: list[str] | None, ttl: int) -> TaskSpec:
    """The single PRELUDE warm-replay task ``params`` opens."""
    return TaskSpec(
        kind="replay_warm_recipe",
        params=params,
        idempotency_key="warm-replay-prelude",
        requires_lanes=lanes,
        lease_ttl_sec=ttl,
        dispatch_class="coordinator",
    )


def baseline_context_copy_refusal(state: Any, args: str, envs: Mapping[str, Any]) -> str:
    """Why ``args`` and ``envs`` laid onto the baseline's materialized config are refused on the synthetic benchmark,
    or ``""`` when they are not.

    Judged as explore judges a variant, so a draft the baseline already launches with is not refused. An AgentX session
    refuses nothing; without a baseline config the launch is judged against an empty base.
    """
    if agentx_active(benchmark_mode=getattr(state, "benchmark_mode", "")):
        return ""
    variant_envs = {str(k): str(v) for k, v in envs.items()}
    config_path = str(getattr(state, "baseline_config_path", "") or "")
    if not config_path:
        return context_copy_refusal("", {}, str(args or ""), variant_envs)
    _, refused = filter_context_copy_drafts(
        [GridVariant("warm_replay", str(args or ""), variant_envs)], recipe=GridRecipe(base_yaml_path=Path(config_path))
    )
    return refused[0][1] if refused else ""


class _WarmLaunchSkipMixin:
    """The launches a PRELUDE warm replay is skipped for before it opens a task, recorded as the phase's skips."""

    def _skip_restated_replay(
        self,
        args: str,
        envs: Mapping[str, Any],
        code_changes: Sequence[Any],
        identity: Mapping[str, Any],
    ) -> bool:
        """Skip a replay whose launch would be the baseline's own; True when it did.

        Patches and kernel items change the code, which no launch comparison sees,
        so a replay carrying any of them always runs.
        """
        if code_changes:
            return False
        if not restates_the_baseline_launch(self.shared_state, args, envs):
            return False
        log.info("PRELUDE: warm replay skipped: %r %s only restate the baseline launch", args, sorted(envs))
        self._skip_warm_replay(
            code=SKIP_RESTATES_BASELINE,
            outcome={"status": "skipped", "reason": SKIP_RESTATES_BASELINE, **identity},
            details={"extra_server_args": args, "extra_envs": sorted(envs)},
        )
        return True

    def _skip_context_copy_replay(
        self,
        args: str,
        envs: Mapping[str, Any],
        overlay: Mapping[str, Any],
        identity: Mapping[str, Any],
    ) -> bool:
        """Skip a replay whose launch introduces or changes a context-copying draft on the synthetic benchmark.

        The replay would lift the draft into ``current_best`` as explore and integrate would not. True when it did.
        """
        refusal = baseline_context_copy_refusal(self.shared_state, args, envs)
        if not refusal:
            return False
        log.warning("PRELUDE: warm replay skipped: %s", refusal)
        rollback = self._unwind_warm_kernel_half(overlay)
        self._skip_warm_replay(
            code=SKIP_CONTEXT_COPY_DRAFT,
            outcome={
                "status": "skipped" if rollback.get("ok") else "rollback_failed",
                "reason": f"{SKIP_CONTEXT_COPY_DRAFT}: {refusal}",
                "rollback": rollback,
                **identity,
            },
            details={"extra_server_args": args, "extra_envs": sorted(envs)},
        )
        return True
