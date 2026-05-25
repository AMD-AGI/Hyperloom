"""N28 — ValidateStackExecutor reads SharedState from the ACTIVE session_dir,
not from a module-import-time captured one.

Empirical bug case (May 2026 SOLAR-10.7B TP=1 session 20260521T062837Z):

* 07:11:10  params executor promotes ``decode_steps_16`` to
            ``SharedState.optimization_stack`` (+0.74% gain over baseline)
            via ``_lift_to_current_best`` -> ``optimization_stack.append``
            -> ``Coordinator._promote_to_shared_state`` -> ``state.save()``.
            On-disk ``state.json`` at the per-session subdir now has
            ``stack_len=1``.
* 07:13:36  LLM proposes ``validate_stack`` to measure cumulative_gain
            of the new stack. Coordinator queues + dispatches the
            ``validate_stack`` task.
* 07:13:36  ValidateStackExecutor.__call__ runs. It tries to load the
            stack via ``SharedState.load_or_init(self.session_dir)``.
            **BUG**: ``self.session_dir`` was captured at module-import
            time (in ``validate_stack_executor = ValidateStackExecutor()``
            at the bottom of validate_stack.py) via
            ``_resolve_session_dir()`` -> ``paths.session_dir()`` ->
            ``$USER_DATA_PATH`` = workspace root
            (``/wekafs/xiaofei/sessions``) BEFORE cli.py's
            ``make_session_dir(model_name=...)`` created the per-session
            subdir and pinned ``INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR``.
            So the executor looks for state.json at the workspace root
            (where it doesn't exist) instead of at the per-session
            subdir.
* Result:   load_or_init returns a blank SharedState ->
            ``stack_len=0`` warning fires -> validate_stack degrades to
            a pure baseline re-run -> reports validated_gain=0.33%
            (baseline noise), NOT the +0.74% the operator wanted.

Fix (N28): resolve the active session_dir LAZILY at every ``__call__``,
using the same ``ctx.extra["session_dir"]`` channel
``SubAgentRunner`` already populates for every executor invocation
(see sub_agent_runner.py:127 and the analogous code in
roofline.py:444, target_analysis.py:95, report.py:403,
session_breakdown.py:87 -- N28 brings validate_stack into
alignment with the same convention).

Tests pinned here:

* ``ctx.extra["session_dir"]`` overrides whatever ``self.session_dir``
  was captured at construction time.
* ``self.session_dir`` is used when ctx.extra doesn't carry one
  (direct-instantiation back-compat for tests).
* ``_resolve_session_dir()`` is the last-resort fallback (executor
  built with ``session_dir=None`` AND ctx.extra empty).
* Bug-reproducer: simulate the SOLAR case where
  ``self.session_dir`` points at workspace root but
  ``ctx.extra["session_dir"]`` is the per-session subdir that has
  the populated state.json -> the executor MUST read from the
  ctx.extra path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator.action_executors.validate_stack import (
    ValidateStackExecutor,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task


def _write_state_with_stack(
    session_dir: Path, *, variant_name: str, tput: float,
) -> None:
    """Persist a SharedState with a single optimization_stack entry --
    mimics the post-promote state.json the SOLAR session had at
    07:11:10 on the per-session subdir."""
    session_dir.mkdir(parents=True, exist_ok=True)
    state = SharedState()
    state.baseline_tput = 3175.87
    state.optimization_stack = [{
        "action": "params",
        "variant_name": variant_name,
        "candidate_extra_sglang_args": "--num-continuous-decode-steps 16",
        "extra_envs": {},
        "tput": tput,
        "ts": "2026-05-21T07:11:10+00:00",
    }]
    state.gain_per_stack_entry = [0.7410]
    state.save(session_dir)


def _ctx_with_session_dir(session_dir: Path | str) -> RunnerContext:
    task = Task(
        task_id="t-validate-stack-1",
        kind="validate_stack",
        state="running",
        params={},  # SOLAR-style: LLM passed no explicit stack/params override
        idempotency_key="vs-1",
        requires_lanes=["benchmark_lane"],
    )
    return RunnerContext(
        task=task,
        lease=None,
        extra={"session_dir": str(session_dir)},
    )


def _ctx_without_session_dir() -> RunnerContext:
    task = Task(
        task_id="t-validate-stack-2",
        kind="validate_stack",
        state="running",
        params={},
        idempotency_key="vs-2",
        requires_lanes=["benchmark_lane"],
    )
    return RunnerContext(task=task, lease=None, extra={})


@pytest.fixture
def stub_super_call(monkeypatch):
    """Stub the heavy ``BaselineExecutor.__call__`` (Magpie subprocess)
    so we only exercise validate_stack's stack-resolution + warning
    logic. Returns a minimal succeeded result; the test asserts on the
    enrichment fields validate_stack adds on top."""
    async def fake_super_call(self, ctx):
        return {
            "status": "succeeded",
            "output_throughput": 3175.87,  # baseline-noise floor
            "ttft_mean_ms": 1700.0,
            "e2el_mean_ms": 20500.0,
            "workspace": "/tmp/fake",
        }
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.action_executors.baseline.BaselineExecutor.__call__",
        fake_super_call,
    )
    return fake_super_call


@pytest.fixture
def stub_resolve_config(monkeypatch, tmp_path):
    """Make ``_resolve_default_config`` return a valid file path so we
    don't trip the early ``missing_config`` failure path."""
    cfg = tmp_path / "baseline.yaml"
    cfg.write_text("framework: sglang\n", encoding="utf-8")
    monkeypatch.setattr(
        ValidateStackExecutor, "_resolve_default_config", lambda self: cfg,
    )
    return cfg


# ---------------------------------------------------------------------------
# 1. BUG REPRODUCER: ctx.extra carries per-session subdir with the
#    populated stack, self.session_dir points elsewhere (workspace root).
#    Pre-N28 behaviour read self.session_dir -> empty stack -> bogus
#    warning. Post-N28 must read ctx.extra -> see the stack.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_n28_solar_repro_uses_ctx_session_dir_over_self(
    tmp_path, stub_super_call, stub_resolve_config, caplog,
):
    """The exact SOLAR-10.7B failure: per-session subdir has the
    populated stack, but self.session_dir is pinned to workspace root."""
    workspace_root = tmp_path / "workspace_root"
    workspace_root.mkdir()  # no state.json here -- empty SharedState load
    per_session = tmp_path / "workspace_root" / "MyModel" / "20260521T071110Z"
    _write_state_with_stack(
        per_session, variant_name="decode_steps_16", tput=3199.40,
    )

    # Executor captured workspace_root at module-import time (the bug).
    ex = ValidateStackExecutor(session_dir=workspace_root)
    # Coordinator injects the CORRECT path via ctx.extra (the fix).
    ctx = _ctx_with_session_dir(per_session)

    import logging
    caplog.set_level(logging.WARNING)
    result = await ex(ctx)

    # The executor MUST have read the per-session subdir's state.json,
    # found stack_len=1, and applied --num-continuous-decode-steps 16.
    assert result["validated_stack_len"] == 1
    assert "--num-continuous-decode-steps 16" in result["applied_args"]
    # Bogus "stack_len=0" warning MUST NOT fire.
    bogus_warnings = [
        rec for rec in caplog.records
        if "optimization_stack is empty" in rec.getMessage()
    ]
    assert not bogus_warnings, (
        f"Pre-N28 bug: bogus stack_len=0 warning fired despite the "
        f"per-session subdir having stack_len=1. Warnings: "
        f"{[r.getMessage() for r in bogus_warnings]}"
    )


# ---------------------------------------------------------------------------
# 2. Resolution-priority contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ctx_session_dir_wins_over_self_session_dir(
    tmp_path, stub_super_call, stub_resolve_config,
):
    """When BOTH ctx.extra['session_dir'] and self.session_dir are
    set, ctx wins (it's the runtime-active dir, self.session_dir is
    construction-time stale)."""
    self_dir = tmp_path / "stale_self_dir"
    ctx_dir = tmp_path / "ctx_dir"
    _write_state_with_stack(
        self_dir, variant_name="stale_winner", tput=1000.0,
    )
    _write_state_with_stack(
        ctx_dir, variant_name="ctx_winner", tput=2000.0,
    )

    ex = ValidateStackExecutor(session_dir=self_dir)
    result = await ex(_ctx_with_session_dir(ctx_dir))

    # Stack came from ctx_dir (variant_name="ctx_winner"), not self_dir.
    applied = result["applied_entries"]
    assert applied
    assert applied[0]["variant_name"] == "ctx_winner"


@pytest.mark.asyncio
async def test_self_session_dir_used_when_ctx_empty(
    tmp_path, stub_super_call, stub_resolve_config,
):
    """When ctx.extra carries no session_dir (direct-instantiation
    tests / standalone CLI invocations), fall back to self.session_dir."""
    self_dir = tmp_path / "self_dir"
    _write_state_with_stack(
        self_dir, variant_name="self_winner", tput=3199.40,
    )

    ex = ValidateStackExecutor(session_dir=self_dir)
    result = await ex(_ctx_without_session_dir())

    assert result["validated_stack_len"] == 1
    applied = result["applied_entries"]
    assert applied[0]["variant_name"] == "self_winner"


@pytest.mark.asyncio
async def test_explicit_stack_param_still_wins_over_both(
    tmp_path, stub_super_call, stub_resolve_config,
):
    """task.params['stack'] is the highest-priority override and
    bypasses any disk read -- this contract was already in place,
    N28 must not regress it."""
    self_dir = tmp_path / "self_dir"
    ctx_dir = tmp_path / "ctx_dir"
    _write_state_with_stack(self_dir, variant_name="self_w", tput=1000.0)
    _write_state_with_stack(ctx_dir, variant_name="ctx_w", tput=2000.0)

    ex = ValidateStackExecutor(session_dir=self_dir)
    # Inject explicit stack into task.params (test/Coordinator override).
    task = Task(
        task_id="t-explicit",
        kind="validate_stack",
        state="running",
        params={
            "stack": [{
                "action": "backends",
                "variant_name": "explicit_w",
                "candidate_extra_sglang_args": "--attention-backend aiter",
                "tput": 4000.0,
            }],
        },
        idempotency_key="vs-explicit",
        requires_lanes=["benchmark_lane"],
    )
    ctx = RunnerContext(
        task=task, lease=None, extra={"session_dir": str(ctx_dir)},
    )
    result = await ex(ctx)
    applied = result["applied_entries"]
    assert applied
    assert applied[0]["variant_name"] == "explicit_w"


@pytest.mark.asyncio
async def test_warns_only_when_truly_empty(
    tmp_path, stub_super_call, stub_resolve_config, caplog,
):
    """The legitimate empty-stack warning should still fire when the
    resolved session_dir genuinely has no stack -- N28 doesn't
    suppress the diagnostic, only the false-positive case from the
    SOLAR bug."""
    truly_empty = tmp_path / "empty_session"
    truly_empty.mkdir()  # no state.json
    ex = ValidateStackExecutor(session_dir=truly_empty)

    import logging
    caplog.set_level(logging.WARNING)
    result = await ex(_ctx_with_session_dir(truly_empty))

    assert result["validated_stack_len"] == 0
    bogus_warnings = [
        rec for rec in caplog.records
        if "optimization_stack is empty" in rec.getMessage()
    ]
    assert bogus_warnings, (
        "Genuine empty-stack case must still surface the diagnostic "
        "warning -- N28 only fixes the FALSE-positive case where the "
        "stack is non-empty but the executor was reading the wrong dir."
    )
