# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ask the predictor once per decision point and queue its answer.

Stepped from the FRAMEWORK_AGENT entry hook and on every tick. The request runs
in a worker thread so the tick loop never waits on it; the tick that finds it
finished files the answer into ``specialist_rounds``, where it joins the
untested-proposal queue the Coordinator benches from, beside the specialists'.

A decision point is ``c{macro_cycle}-s{stack_depth}-r{roofline_count}``. A KEEP,
a new macro-cycle and a new roofline each change what the answer is conditioned
on. A key counts as asked once its request goes out, so it is never asked twice
(``SharedState.predictor_asked_keys``), not even after a failure. An answer that
arrives after the macro-cycle moved on is filed under the cycle it was asked in,
where the queue no longer looks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from hyperloom.orchestrator.phases.machine_state import (
    PHASE_FRAMEWORK_AGENT,
    phase_budget_remaining_seconds,
    phase_cap_seconds,
    phase_cumulative_seconds,
)
from hyperloom.orchestrator.predictor import config as predictor_config
from hyperloom.orchestrator.predictor.client import Prediction, predict
from hyperloom.orchestrator.predictor.mandate import patch_mandate
from hyperloom.orchestrator.predictor.payload import build_request
from hyperloom.orchestrator.predictor.rows import QUEUE_DOMAIN, QUEUE_PRIORITY, proposal_rows
from hyperloom.orchestrator.predictor.sidecars import load_sidecars
from hyperloom.orchestrator.predictor.source_sites import load_source_sites

log = logging.getLogger(__name__)

#: One explore variant on a large model, server restart plus benchmark. An
#: answer is worth asking for only while the phase can still bench one variant.
MIN_VARIANT_SEC = 600.0

#: Decision points remembered; an evicted key risks one repeat request at a depth long since passed.
MAX_ASKED_KEYS = 200


def decision_point_key(state: Any) -> str:
    """``c{macro_cycle}-s{stack_depth}-r{roofline_snapshot_count}``."""
    snapshots = state.roofline_snapshots if isinstance(state.roofline_snapshots, list) else []
    return f"c{int(state.macro_cycle or 0)}-s{len(state.optimization_stack or [])}-r{len(snapshots)}"


def _headroom_sec(state: Any) -> float | None:
    """Seconds the phase can still spend: the tighter of this entry's budget and the cap across entries."""
    limits: list[float] = []
    remaining = phase_budget_remaining_seconds(state)
    if remaining is not None:
        limits.append(float(remaining))
    cap = phase_cap_seconds(state)
    if cap is not None:
        limits.append(float(cap) - float(phase_cumulative_seconds(state)))
    return min(limits) if limits else None


def _skip_reason(state: Any, conf: predictor_config.PredictorConfig) -> str | None:
    """Why not to ask at this tick, or ``None`` to ask."""
    if not conf.enabled:
        return "disabled"
    if str(state.phase or "").strip().upper() != PHASE_FRAMEWORK_AGENT or state.framework_agent_phase_done:
        return "not in an open FRAMEWORK_AGENT phase"
    if str(state.framework or "").strip().lower() not in predictor_config.SUPPORTED_FRAMEWORKS:
        return f"no flag catalogue for framework {state.framework!r}"
    headroom = _headroom_sec(state)
    if headroom is not None and headroom < MIN_VARIANT_SEC:
        return f"phase headroom {headroom:.0f}s is below one variant"
    if decision_point_key(state) in state.predictor_asked_keys:
        return "decision point already asked"
    return None


def _note_asked(state: Any, key: str) -> None:
    keys = state.predictor_asked_keys
    if key not in keys:
        keys.append(key)
        del keys[:-MAX_ASKED_KEYS]


class PredictorPump:
    """The request one session has in flight, if any."""

    def __init__(self) -> None:
        self._inflight: asyncio.Task[Prediction] | None = None
        self._key = ""
        self._cycle = 0
        self._conf = predictor_config.PredictorConfig()
        self._started = 0.0

    async def step(self, state: Any) -> None:
        """File an answer that has arrived, or ask when this decision point warrants it."""
        if self._inflight is not None:
            if self._inflight.done():
                task, self._inflight = self._inflight, None
                self._file(state, task.result())
            return
        conf = predictor_config.load()
        reason = _skip_reason(state, conf)
        if reason is not None:
            log.debug("predictor: not asking (%s)", reason)
            return
        self._key, self._conf, self._started = decision_point_key(state), conf, time.monotonic()
        self._cycle = int(state.macro_cycle or 0)
        # Before anything can raise: a request that fails is not retried at the same decision point.
        _note_asked(state, self._key)
        trace = state.last_trace_analyze if isinstance(state.last_trace_analyze, dict) else {}
        report_path = trace.get("analysis_md_path")
        sites = await asyncio.to_thread(load_source_sites, report_path)
        sidecars = await asyncio.to_thread(load_sidecars, report_path)
        request = build_request(state, session_id=str(state.session_id or ""), sites=sites, sidecars=sidecars)
        self._inflight = asyncio.create_task(
            asyncio.to_thread(predict, request, endpoint=conf.endpoint, timeout_sec=conf.timeout_sec)
        )
        log.info("predictor: asked %s at decision point %s (mode %s)", conf.endpoint, self._key, conf.mode)

    def _file(self, state: Any, answer: Prediction) -> None:
        key, latency_ms = self._key, int((time.monotonic() - self._started) * 1000)
        if not self._conf.enqueues:
            log.info(
                "predictor (shadow): key=%s parsed=%s latency=%dms actions=%r",
                key,
                answer.parsed,
                latency_ms,
                [(a.server_args, a.envs, a.source_change[:200]) for a in answer.actions],
            )
            return
        if not answer.parsed:
            log.info("predictor: no answer at %s after %dms (%s)", key, latency_ms, answer.error)
            return
        rows = proposal_rows(answer, key=key, state=state)
        mandate = patch_mandate(answer, key=key)
        if not rows and mandate is None:
            log.info("predictor: nothing new to queue at %s (%d proposals)", key, len(answer.config_actions))
            return
        state.record_specialist_round(
            {
                "round_id": key,
                "task_id": key,
                "cycle": self._cycle,
                "domain": QUEUE_DOMAIN,
                "priority": QUEUE_PRIORITY,
                "proposal_set": rows,
                "predict_meta": {
                    "latency_ms": latency_ms,
                    "samples": answer.meta.get("samples"),
                    "prompt_chars": answer.meta.get("prompt_chars"),
                    "actions_returned": len(answer.actions),
                },
                **(mandate or {}),
            }
        )
        log.info(
            "predictor: queued %d row(s) at %s after %dms%s: %r",
            len(rows),
            key,
            latency_ms,
            " with a source-change mandate" if mandate else "",
            [(row["extra_args"], row["extra_envs"], row["votes"]) for row in rows],
        )
