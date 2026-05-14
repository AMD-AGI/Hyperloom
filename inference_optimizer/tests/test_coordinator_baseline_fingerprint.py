"""Baseline-params fingerprint capture in the Coordinator audit trail.

``Coordinator._baseline_params_fingerprint`` projects the eight task.params
fields that determine baseline behavior end-to-end (script choice / leak
path / config / model / GPU / accuracy gate). Both the success path
(``_promote_to_shared_state``) and the failure path
(``_handle_unpromotable_result``) stamp this fingerprint on the audit
entry's ``extras`` so the prompt's FAILURE RECOVERY block + the
self-loop denial helper can detect "same params failed twice".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import (
    Coordinator,
    _BASELINE_FINGERPRINT_KEYS,
    _baseline_params_fingerprint,
)
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.task_registry import Task
from inference_optimizer.paths import make_session_dir


def _heartbeat() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _mute_action_scoring(coordinator: Coordinator) -> None:
    coordinator.shared_state.action_scores = {}


def _mk_baseline_task(params: dict, *, task_id: str = "t-fp-1") -> Task:
    return Task(
        task_id=task_id,
        kind="baseline",
        state="queued",
        params=params,
        idempotency_key=f"idem-{task_id}",
    )


# ---------------------------------------------------------------------------
# Pure-function fingerprint tests
# ---------------------------------------------------------------------------
def test_fingerprint_keys_covers_recovery_surface():
    """Every key the prompt references must be in the projection."""
    expected = {
        "benchmark_script", "result_dir", "extra_sglang_args",
        "extra_envs", "model_path", "gpu_type", "config_path",
        "disable_run_eval",
    }
    assert set(_BASELINE_FINGERPRINT_KEYS) == expected


def test_fingerprint_normalizes_extra_envs_order():
    """Dict ordering MUST NOT affect equality (we store a sorted list)."""
    fp1 = _baseline_params_fingerprint({"extra_envs": {"A": "1", "B": "2"}})
    fp2 = _baseline_params_fingerprint({"extra_envs": {"B": "2", "A": "1"}})
    assert fp1 == fp2
    assert fp1["extra_envs"] == [["A", "1"], ["B", "2"]]


def test_fingerprint_missing_keys_become_none_or_empty():
    """Absent scalar keys → ``None``; absent dict (``extra_envs``) → ``[]``.

    Either way the fingerprint comparison stays well-defined (``None ==
    None``, ``[] == []``), which is the only invariant the self-loop
    helper needs.
    """
    fp = _baseline_params_fingerprint({"benchmark_script": "sglang_mi300x.sh"})
    assert fp["benchmark_script"] == "sglang_mi300x.sh"
    assert fp["result_dir"] is None
    assert fp["extra_sglang_args"] is None
    assert fp["extra_envs"] == []
    assert fp["model_path"] is None
    # Two fingerprints whose only difference is "absent vs empty extra_envs"
    # must still compare equal.
    fp_with_empty = _baseline_params_fingerprint({
        "benchmark_script": "sglang_mi300x.sh",
        "extra_envs": {},
    })
    assert fp == fp_with_empty


def test_fingerprint_stringifies_scalar_values():
    fp = _baseline_params_fingerprint({
        "benchmark_script": "sglang_mi300x.sh",
        "model_path": "/wekafs/models/DeepSeek-R1",
        "gpu_type": "mi300x",
    })
    assert all(isinstance(v, str) for k, v in fp.items() if v is not None and k != "extra_envs")


def test_fingerprint_different_overrides_produce_different_fingerprints():
    import json
    a = _baseline_params_fingerprint({"benchmark_script": "sglang_mi300x.sh"})
    b = _baseline_params_fingerprint({"benchmark_script": "dsr1_fp8_mi300x.sh"})
    c = _baseline_params_fingerprint({"result_dir": "/workspace"})
    d = _baseline_params_fingerprint({"extra_sglang_args": "--mem-fraction-static 0.9"})
    # Compare via canonical JSON since fingerprint values contain lists
    # (``extra_envs``) and aren't directly hashable.
    encoded = {json.dumps(x, sort_keys=True) for x in (a, b, c, d)}
    assert len(encoded) == 4


# ---------------------------------------------------------------------------
# Success path: fingerprint lands in audit_extras
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_promote_baseline_records_fingerprint(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task = _mk_baseline_task({
            "benchmark_script": "sglang_mi300x.sh",
            "model_path": "/wekafs/models/DeepSeek-R1",
            "gpu_type": "mi300x",
        })
        result = {
            "output_throughput": 1500.0,
            "materialized_config": "/tmp/baseline.with_envs.yaml",
            "workspace": "/runs/baseline/t-fp-1",
        }
        await c._promote_to_shared_state("baseline", result, task=task)
        last = c.shared_state.last_baseline
        assert last["status"] == "succeeded"
        fp = last["extras"]["fingerprint"]
        assert fp["benchmark_script"] == "sglang_mi300x.sh"
        assert fp["model_path"] == "/wekafs/models/DeepSeek-R1"
        assert fp["gpu_type"] == "mi300x"
    finally:
        await c.stop()


# ---------------------------------------------------------------------------
# Failure path: fingerprint lands in audit_extras for baseline only
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_handle_unpromotable_baseline_records_fingerprint(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task = _mk_baseline_task({
            "benchmark_script": "dsr1_fp8_mi300x.sh",  # the leaky one
        })
        result = {
            "status": "failed",
            "error_class": "no_report",
            "error": "benchmark_report.json missing",
        }
        await c._handle_unpromotable_result(task, result)
        attempt = c.shared_state.baseline_attempts[-1]
        assert attempt["status"] == "failed"
        fp = attempt["extras"]["fingerprint"]
        assert fp["benchmark_script"] == "dsr1_fp8_mi300x.sh"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_non_baseline_omits_fingerprint(session_dir):
    """The fingerprint surface is baseline-only today; other kinds
    intentionally don't carry one (they'd need their own key set)."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task = Task(
            task_id="t-be-fail",
            kind="backends",
            state="queued",
            params={"benchmark_script": "sglang_mi300x.sh"},
            idempotency_key="idem-be",
        )
        await c._handle_unpromotable_result(task, {"status": "failed"})
        attempt = c.shared_state.backends_attempts[-1]
        # We DO record an attempt for backends (it's in _AUDIT_ACTIONS).
        assert attempt["status"] == "failed"
        # But the fingerprint key isn't there — only baseline is wired.
        assert "fingerprint" not in attempt["extras"]
    finally:
        await c.stop()
