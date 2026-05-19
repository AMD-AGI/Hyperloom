"""Roofline-v2 C4b: sub-agent ``RooflineExecutor`` tests.

Pins the contract C4c (Coordinator integration) and C5 (prompt
renderer) build on top of:

* **Happy path** — well-formed JSON from the sub-agent backend lands
  in the result with snapshot_id / analyzed_at_iso / paths populated
  by the executor (not by the analyzer LLM, which is forbidden from
  overriding them).
* **Idempotency (D2)** — re-proposing against the same snapshot
  short-circuits with ``idempotency_hit=True`` and **does not** spend
  another sub-agent LLM token.
* **No cached analysis** — empty / missing ``analysis_md_text``
  produces the canonical fallback with
  ``error="no_cached_analysis_md"`` (and a future sequence_denial in
  C4c will prevent this from happening in practice — but the executor
  is robust either way).
* **Backend failure paths** — ``BackendError``, ``asyncio.TimeoutError``,
  and any unexpected ``Exception`` raised by ``backend.run()`` all
  produce the same fallback schema with an informative ``error``
  field. The optimisation loop must never see a failed task.
* **JSON parsing robustness** — bare JSON, fenced JSON
  (```json ... ```), JSON with surrounding whitespace, JSON embedded
  in stray prose: all parse. Truly malformed text → fallback with
  ``error="json_parse_failed"`` and raw_text preserved for forensics.
* **analysis.md truncation** — inputs exceeding
  ``ROOFLINE_ANALYSIS_MD_MAX_BYTES`` are truncated UTF-8-safely at
  the last newline before the cap, with a tail marker so the
  analyzer can mention truncation.
* **Field protection** — the analyzer cannot override snapshot_id /
  based_on_analysis_md / analyzed_at_gain_pct via its JSON output;
  the executor's contextual fields always win.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors import (
    ROOFLINE_ANALYSIS_MD_MAX_BYTES,
    RooflineExecutor,
    make_roofline_executor,
)
from inference_optimizer.orchestrator.action_executors.roofline import (
    _compose_analyzer_user_prompt,
    _parse_analyzer_json,
    _truncate_analysis_md,
)
from inference_optimizer.orchestrator.backends.base import (
    BackendError,
    BackendTurnResult,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
def _ctx() -> RunnerContext:
    task = Task(
        task_id="t-roofline-1", kind="roofline", state="running",
        params={}, idempotency_key="roofline:t-1",
    )
    return RunnerContext(task=task, lease=None, extra={})


def _state_with_cached_analysis(*, analysis_md: str = "FAKE_REPORT",
                                 snapshot_id: int = 3,
                                 gain: float = 2.5) -> SharedState:
    state = SharedState()
    state.last_select_kernels = {
        "analysis_md_text": analysis_md,
        "analysis_md_path": "/sessions/abc/select_kernels/analysis.md",
        "roofline_snapshot_id": snapshot_id,
    }
    state.cumulative_gain_validated = gain
    return state


@dataclass
class _StubBackend:
    """Minimal Backend protocol stub for the sub-agent under test.

    Captures every ``run()`` call so tests can assert prompt content,
    and lets tests script a single response (raw_text) or an
    exception to raise instead.
    """

    raw_text: str = ""
    raise_exc: Exception | None = None
    raise_after_delay: float = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "tools": tools,
            "max_turns": max_turns,
        })
        if self.raise_after_delay > 0:
            await asyncio.sleep(self.raise_after_delay)
        if self.raise_exc is not None:
            raise self.raise_exc
        return BackendTurnResult(intents=[], raw_text=self.raw_text)


def _well_formed_json_response() -> str:
    return json.dumps({
        "primary_bottleneck": "comm",
        "bottleneck_distribution": {
            "comm": 0.45, "compute": 0.30, "memory": 0.15,
            "latency": 0.05, "idle": 0.05,
        },
        "suggested_prunes": [
            {"family": "kernel_opt",
             "reason": "compute saturated 91.2%, no reusable_native",
             "confidence": "high"},
        ],
        "suggested_next_actions": [
            {"kind": "params",
             "rationale": "try enable_two_batch_overlap",
             "priority": "high"},
        ],
        "reprofile_recommended": False,
        "reprofile_reason": "",
    })


def _make_executor(state: SharedState, backend: _StubBackend,
                   *, sp: str = "ANALYZER_SP",
                   timeout_sec: float = 60.0) -> RooflineExecutor:
    return RooflineExecutor(
        shared_state=state,
        backend_factory=lambda: backend,
        analyzer_system_prompt_loader=lambda: sp,
        timeout_sec=timeout_sec,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_happy_path_parses_well_formed_response():
    state = _state_with_cached_analysis(analysis_md="my analysis here",
                                          snapshot_id=4, gain=3.2)
    backend = _StubBackend(raw_text=_well_formed_json_response())
    executor = _make_executor(state, backend)

    result = await executor(_ctx())

    assert result["status"] == "succeeded"
    assert result["degraded"] is False
    assert result["snapshot_id"] == 4
    assert result["analyzed_at_gain_pct"] == 3.2
    assert result["based_on_analysis_md"] == \
        "/sessions/abc/select_kernels/analysis.md"
    assert result["primary_bottleneck"] == "comm"
    assert result["bottleneck_distribution"]["comm"] == 0.45
    assert len(result["suggested_prunes"]) == 1
    assert result["suggested_prunes"][0]["family"] == "kernel_opt"
    assert result["raw_llm_response"] == _well_formed_json_response()
    assert "analyzed_at_iso" in result and result["analyzed_at_iso"]

    # The sub-agent received the correct prompt construction.
    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call["system_prompt"] == "ANALYZER_SP"
    assert call["tools"] is None
    assert call["max_turns"] == 1
    assert "cumulative_gain_validated_pct: 3.200" in call["prompt"]
    assert "my analysis here" in call["prompt"]


@pytest.mark.asyncio
async def test_happy_path_result_round_trips_through_recorder():
    """C2's ``record_roofline_analysis`` must accept the executor
    output without dropping any documented field."""
    state = _state_with_cached_analysis(snapshot_id=5)
    backend = _StubBackend(raw_text=_well_formed_json_response())
    executor = _make_executor(state, backend)
    result = await executor(_ctx())

    state.record_roofline_analysis(result)
    cached = state.last_roofline_analysis
    assert cached["snapshot_id"] == 5
    assert cached["primary_bottleneck"] == "comm"
    assert cached["bottleneck_distribution"]["comm"] == 0.45
    assert len(cached["suggested_prunes"]) == 1
    assert cached["suggested_prunes"][0]["confidence"] == "high"
    assert cached["raw_llm_response"]


# ---------------------------------------------------------------------------
# Idempotency (D2)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_idempotency_short_circuits_same_snapshot():
    """Re-proposing roofline against the same snapshot must NOT
    spend another sub-agent token."""
    state = _state_with_cached_analysis(snapshot_id=7)
    state.last_roofline_analysis = {
        "snapshot_id": 7,
        "primary_bottleneck": "compute",
        "suggested_prunes": [],
        "suggested_next_actions": [],
        "raw_llm_response": "prev cached",
    }
    backend = _StubBackend(raw_text=_well_formed_json_response())
    executor = _make_executor(state, backend)

    result = await executor(_ctx())

    assert result["idempotency_hit"] is True
    assert result["snapshot_id"] == 7
    assert result["primary_bottleneck"] == "compute"
    assert result["status"] == "succeeded"
    assert backend.calls == []  # NO sub-agent call


@pytest.mark.asyncio
async def test_idempotency_does_not_trigger_on_new_snapshot():
    state = _state_with_cached_analysis(snapshot_id=8)
    state.last_roofline_analysis = {"snapshot_id": 7,
                                     "primary_bottleneck": "memory"}
    backend = _StubBackend(raw_text=_well_formed_json_response())
    executor = _make_executor(state, backend)

    result = await executor(_ctx())

    assert "idempotency_hit" not in result or result.get("idempotency_hit") is False
    assert result["snapshot_id"] == 8
    assert result["primary_bottleneck"] == "comm"  # from fresh analysis
    assert len(backend.calls) == 1


# ---------------------------------------------------------------------------
# No cached analysis → fallback
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_cached_analysis_md_returns_fallback():
    state = SharedState()  # no last_select_kernels populated
    backend = _StubBackend(raw_text=_well_formed_json_response())
    executor = _make_executor(state, backend)

    result = await executor(_ctx())

    assert result["status"] == "succeeded"
    assert result["degraded"] is True
    assert result["primary_bottleneck"] == "unknown"
    assert result["error"] == "no_cached_analysis_md"
    assert backend.calls == []


@pytest.mark.asyncio
async def test_empty_analysis_md_text_returns_fallback():
    state = _state_with_cached_analysis(analysis_md="", snapshot_id=3)
    backend = _StubBackend(raw_text=_well_formed_json_response())
    executor = _make_executor(state, backend)

    result = await executor(_ctx())

    assert result["degraded"] is True
    assert result["error"] == "no_cached_analysis_md"
    assert backend.calls == []


@pytest.mark.asyncio
async def test_zero_snapshot_id_returns_fallback():
    state = SharedState()
    state.last_select_kernels = {
        "analysis_md_text": "report",
        "analysis_md_path": "/p",
        "roofline_snapshot_id": 0,  # zero is the "uninitialised" sentinel
    }
    backend = _StubBackend(raw_text=_well_formed_json_response())
    executor = _make_executor(state, backend)
    result = await executor(_ctx())
    assert result["error"] == "no_cached_analysis_md"


# ---------------------------------------------------------------------------
# Backend failure paths
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_backend_error_returns_fallback_with_diagnostic():
    state = _state_with_cached_analysis(snapshot_id=3)
    backend = _StubBackend(raise_exc=BackendError("rate_limit_429"))
    executor = _make_executor(state, backend)

    result = await executor(_ctx())

    assert result["status"] == "succeeded"  # never bubble failure to scheduler
    assert result["degraded"] is True
    assert "backend_error" in result["error"]
    assert "rate_limit_429" in result["error"]
    assert result["snapshot_id"] == 3


@pytest.mark.asyncio
async def test_timeout_returns_fallback():
    state = _state_with_cached_analysis(snapshot_id=3)
    backend = _StubBackend(raise_after_delay=10.0,
                           raise_exc=BackendError("late"))
    executor = _make_executor(state, backend, timeout_sec=0.05)

    result = await executor(_ctx())

    assert result["status"] == "succeeded"
    assert result["degraded"] is True
    assert "sub_agent_timeout" in result["error"]


@pytest.mark.asyncio
async def test_unexpected_exception_returns_fallback():
    state = _state_with_cached_analysis(snapshot_id=3)
    backend = _StubBackend(raise_exc=RuntimeError("unexpected"))
    executor = _make_executor(state, backend)

    result = await executor(_ctx())

    assert result["status"] == "succeeded"
    assert result["degraded"] is True
    assert "backend_unexpected" in result["error"]


@pytest.mark.asyncio
async def test_backend_factory_failure_returns_fallback():
    state = _state_with_cached_analysis(snapshot_id=3)

    def _broken_factory():
        raise RuntimeError("anthropic_sdk_missing")

    executor = RooflineExecutor(
        shared_state=state,
        backend_factory=_broken_factory,
        analyzer_system_prompt_loader=lambda: "SP",
    )
    result = await executor(_ctx())

    assert result["degraded"] is True
    assert "backend_factory_failed" in result["error"]


@pytest.mark.asyncio
async def test_system_prompt_load_failure_returns_fallback():
    state = _state_with_cached_analysis(snapshot_id=3)
    backend = _StubBackend(raw_text=_well_formed_json_response())

    def _broken_sp():
        raise OSError("permission denied")

    executor = RooflineExecutor(
        shared_state=state,
        backend_factory=lambda: backend,
        analyzer_system_prompt_loader=_broken_sp,
    )
    result = await executor(_ctx())
    assert result["degraded"] is True
    assert "analyzer_sp_load_failed" in result["error"]
    assert backend.calls == []


# ---------------------------------------------------------------------------
# JSON parsing robustness
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_malformed_json_returns_fallback_with_raw_text():
    state = _state_with_cached_analysis(snapshot_id=3)
    backend = _StubBackend(raw_text="this is not json at all")
    executor = _make_executor(state, backend)

    result = await executor(_ctx())

    assert result["status"] == "succeeded"
    assert result["degraded"] is True
    assert result["error"] == "json_parse_failed"
    assert result["raw_llm_response"] == "this is not json at all"
    assert result["primary_bottleneck"] == "unknown"


@pytest.mark.asyncio
async def test_fenced_json_response_parses():
    """Sub-agent may wrap JSON in a ```json fence despite system prompt
    saying not to — the parser strips it."""
    state = _state_with_cached_analysis(snapshot_id=3)
    payload = _well_formed_json_response()
    backend = _StubBackend(raw_text=f"```json\n{payload}\n```")
    executor = _make_executor(state, backend)

    result = await executor(_ctx())

    assert result["degraded"] is False
    assert result["primary_bottleneck"] == "comm"


@pytest.mark.asyncio
async def test_json_embedded_in_prose_extracted():
    """Sub-agent may surround JSON with whitespace / a one-line preamble
    — the regex fallback grabs the JSON object."""
    state = _state_with_cached_analysis(snapshot_id=3)
    payload = _well_formed_json_response()
    backend = _StubBackend(raw_text=f"Analysis:\n\n{payload}\n\nDone.")
    executor = _make_executor(state, backend)
    result = await executor(_ctx())
    assert result["degraded"] is False
    assert result["primary_bottleneck"] == "comm"


# ---------------------------------------------------------------------------
# Field protection (analyzer cannot override executor-owned fields)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_analyzer_cannot_override_snapshot_id_or_paths():
    state = _state_with_cached_analysis(snapshot_id=42, gain=3.2)
    # Analyzer tries to claim a different snapshot / path / gain
    response = json.dumps({
        "primary_bottleneck": "comm",
        "bottleneck_distribution": {},
        "suggested_prunes": [],
        "suggested_next_actions": [],
        "reprofile_recommended": False,
        "reprofile_reason": "",
        "snapshot_id": 999,
        "based_on_analysis_md": "/evil/path",
        "analyzed_at_gain_pct": 99.9,
        "status": "failed",  # try to fake a status override
    })
    backend = _StubBackend(raw_text=response)
    executor = _make_executor(state, backend)
    result = await executor(_ctx())

    # Executor wins for context fields
    assert result["snapshot_id"] == 42
    assert result["based_on_analysis_md"] == \
        "/sessions/abc/select_kernels/analysis.md"
    assert result["analyzed_at_gain_pct"] == 3.2
    assert result["status"] == "succeeded"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------
def test_truncation_preserves_leading_sections_and_appends_marker():
    big = ("line\n" * 100000)  # ~600 KB
    truncated = _truncate_analysis_md(big, max_bytes=10_000)
    assert len(truncated.encode("utf-8")) <= 10_000 + 100  # +marker
    assert truncated.startswith("line\n")
    assert "truncated" in truncated


def test_truncation_passthrough_for_small_inputs():
    small = "hello world\n"
    assert _truncate_analysis_md(small, max_bytes=10_000) == small


def test_truncation_default_max_bytes_constant():
    """The default cap is consciously set to ~200KB worst-case
    Hyperloom-produced analysis.md."""
    assert ROOFLINE_ANALYSIS_MD_MAX_BYTES == 200 * 1024


# ---------------------------------------------------------------------------
# JSON parser unit tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected_key", [
    ('{"k": 1}', "k"),
    ('  {"k": 2}\n', "k"),
    ('```json\n{"k": 3}\n```', "k"),
    ('```\n{"k": 4}\n```', "k"),
    ('Some preamble {"k": 5} trailing', "k"),
])
def test_parser_accepts_various_wrappings(raw, expected_key):
    parsed = _parse_analyzer_json(raw)
    assert isinstance(parsed, dict)
    assert expected_key in parsed


@pytest.mark.parametrize("raw", [
    "",
    "completely non-json text",
    "{not really json}",
    "[1, 2, 3]",  # list, not dict
])
def test_parser_rejects_non_dict_or_unparseable(raw):
    assert _parse_analyzer_json(raw) is None


# ---------------------------------------------------------------------------
# User prompt composition
# ---------------------------------------------------------------------------
def test_user_prompt_includes_all_four_sections():
    prompt = _compose_analyzer_user_prompt(
        analysis_md="REPORT",
        cumulative_gain_pct=4.7,
        optimization_stack=[{"kind": "params", "gain_pct": 2.0}],
        pruned_families=["kernel_opt", "deep_kernel_analysis"],
    )
    assert "cumulative_gain_validated_pct: 4.700" in prompt
    assert '"kind": "params"' in prompt
    assert "deep_kernel_analysis" in prompt
    assert "kernel_opt" in prompt
    assert "analysis_md: |\nREPORT\n" in prompt


def test_user_prompt_handles_empty_state():
    prompt = _compose_analyzer_user_prompt(
        analysis_md="",
        cumulative_gain_pct=0.0,
        optimization_stack=None,
        pruned_families=None,
    )
    assert "cumulative_gain_validated_pct: 0.000" in prompt
    assert "optimization_stack: []" in prompt
    assert "pruned_families: []" in prompt


# ---------------------------------------------------------------------------
# Factory signature contract
# ---------------------------------------------------------------------------
def test_make_roofline_executor_factory_signature():
    """``cli._register_executors`` calls this factory with exactly
    ``shared_state=``; the optional ``backend_factory=`` is the test
    seam."""
    state = SharedState()
    exe = make_roofline_executor(shared_state=state)
    assert isinstance(exe, RooflineExecutor)
    assert exe.shared_state is state

    custom = _StubBackend()
    exe2 = make_roofline_executor(
        shared_state=state,
        backend_factory=lambda: custom,
    )
    assert exe2.shared_state is state
