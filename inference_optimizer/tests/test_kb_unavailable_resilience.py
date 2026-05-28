"""KB-unavailable resilience tests.

Operator hard requirement (May 2026): **if KB is not available, do
not affect the main logic**. Every KB call site must degrade
gracefully — never propagate transport / business errors back into
the optimizer's main loop, never abort the launch, never lose
correctness on local-only paths (journal, stack push, replay
decisions).

This module exercises the soft-degrade contract at each KB touch
point that could break the main flow:

* T0 anchor — even if T0 fails mid-flight, client is disabled and
  cli returns normally.
* _fact_write_hook — KB disabled / None → silent no-op (journal
  still writes).
* _record_fact_per_task / _record_fact_per_variant — KB read
  failure → write without source_session_ids merge (no clobber);
  KB write failure → falls through to NDJSON enqueue, never raises.
* cortex_finalize_recipe_and_journal — KB None / disabled → skip
  silently; KB read failure → write without sessions=.
* _maybe_enqueue_warm_replay / _inject_warm_recipe_history /
  _promote_warm_replay — do not call KB at all (read SharedState
  in-memory), so KB outages cannot affect them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from inference_optimizer.cortex_kb_client import (
    CortexKBClient,
    CortexKBError,
)
from inference_optimizer.orchestrator.coordinator import Coordinator


KB_URL = "http://kb-down.local"


# ===========================================================================
# 1. KB client itself — soft-degrade on transport error
# ===========================================================================
def test_disabled_client_returns_empty_for_all_reads(tmp_path):
    """``client.enabled=False`` (set when KB is unreachable) returns
    empty / no-op for every read so callers never observe an error."""
    client = CortexKBClient(
        session_dir=tmp_path / "session", kb_url=KB_URL, enabled=False,
    )
    assert client.lessons(model="M", hardware="H") == []
    assert client.pitfalls(model="M", hardware="H") == []
    assert client.read_recipe_exact(model="M", hardware="H") == {}
    assert client.read_lesson_exact(statement="X") == {}
    assert client.read_pitfall_exact(description="Y") == {}
    point, tier, conf = client.find_recipe_with_fallback(workload="M", hw="H")
    assert point == {}
    assert tier == "miss"
    assert conf == 0.0


def test_disabled_client_skips_writes_silently(tmp_path):
    """Disabled client returns ``status=skip_disabled`` for writes
    without touching the network or the NDJSON queue."""
    client = CortexKBClient(
        session_dir=tmp_path / "session", kb_url=KB_URL, enabled=False,
    )
    out = client.propose_lesson(statement="X", measured_impact="y")
    assert out["status"] == "skip_disabled"
    # NDJSON queue stays empty.
    assert not client.pending_path.exists() or client.pending_path.stat().st_size == 0


def test_write_falls_back_to_ndjson_when_kb_returns_500(tmp_path):
    """KB DOWN mid-session: propose_lesson catches CortexKBError and
    enqueues to NDJSON. Caller sees ``status=queued``, never an
    exception. The NDJSON pending file grows with the queued envelope."""
    client = CortexKBClient(session_dir=tmp_path / "session", kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/propose").mock(
            return_value=httpx.Response(500, json={"detail": "boom"}),
        )
        out = client.propose_lesson(
            statement="X", measured_impact="y",
            applicable_models=["M"], applicable_hardware=["H"],
        )
    assert out["status"] == "queued"
    # Pending file should have one line.
    pending = client.pending_path.read_text(encoding="utf-8")
    assert pending.strip(), "NDJSON pending should carry the queued write"


# ===========================================================================
# 2. Coordinator-side soft-degrade
# ===========================================================================
@dataclass
class _MinimalState:
    """Minimal SharedState surface for the coordinator KB hooks."""
    framework: str = "sglang"
    model_name: str = "DeepSeek-R1"
    gpu_type: str = "MI300X"
    baseline_tput: float = 600.0
    tick: int = 0
    phase: str = "EXPLORE"
    optimization_stack: list = field(default_factory=list)
    warm_start_recipe: dict = field(default_factory=dict)
    warm_replay_attempted: bool = False
    warm_replay_outcome: dict = field(default_factory=dict)
    warm_history_injected: bool = False
    explore_search: dict = field(default_factory=dict)
    stack_fingerprint_meta: dict = field(default_factory=dict)
    baseline_workload_extra: dict = field(default_factory=dict)
    cortex_session_id: str = "session-X"
    model_class: str = "moe_mla"
    precision: str = "fp8"
    tp: int = 8
    ep: int = 0
    conc: int = 64
    isl: int = 1024
    osl: int = 1024
    max_model_len: int = 0

    def save(self, *a, **kw):
        pass


class _Task:
    def __init__(self, task_id="t-1", kind="kernel_opt"):
        self.task_id = task_id
        self.kind = kind


def _coord_with_kb(kb, tmp_path):
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = _MinimalState()
    coord.cortex_kb = kb
    coord._fact_writes_enabled = True
    coord._journal = None
    return coord


def test_fact_write_hook_silent_when_kb_disabled(tmp_path):
    """When ``client.enabled=False`` (or client is None), the fact-
    write hook MUST be a silent no-op — no journal append, no KB
    call, no exception."""
    class _DisabledKB:
        enabled = False

    coord = _coord_with_kb(_DisabledKB(), tmp_path)
    # No exception expected.
    coord._record_fact_per_task(
        task=_Task(),
        source_session_id="session-X",
        result_dict={"gain_pct": 7.0, "output_throughput": 750.0},
        kept=True,
    )
    # No KB call attempted (would have hit propose_lesson / read_lesson).


def test_fact_write_hook_silent_when_kb_none(tmp_path):
    """``cortex_kb=None`` (legacy / SDK without KB) is the same as
    disabled — silent no-op."""
    coord = _coord_with_kb(None, tmp_path)
    coord._record_fact_per_task(
        task=_Task(),
        source_session_id="session-X",
        result_dict={"gain_pct": 7.0, "output_throughput": 750.0},
        kept=True,
    )


def test_fact_write_hook_recovers_when_kb_read_throws(tmp_path):
    """KB read raises but write is queued via NDJSON — caller never
    observes the exception. The lesson is still proposed (just without
    the source_session_ids merge fields)."""
    class _ReadFailingKB:
        enabled = True

        def __init__(self):
            self.lesson_calls = []

        def read_lesson_exact(self, **kw):
            raise CortexKBError("transport down", category="transport")

        def read_pitfall_exact(self, **kw):
            raise CortexKBError("transport down", category="transport")

        def propose_lesson(self, **kw):
            self.lesson_calls.append(kw)
            # Simulate the NDJSON-fallback path: write succeeded as
            # queued.
            return {"status": "queued"}

        def propose_pitfall(self, **kw):
            return {"status": "queued"}

    kb = _ReadFailingKB()
    coord = _coord_with_kb(kb, tmp_path)
    coord._record_fact_per_task(
        task=_Task(),
        source_session_id="session-X",
        result_dict={"gain_pct": 7.0, "output_throughput": 750.0},
        kept=True,
    )
    # Lesson was still proposed.
    assert len(kb.lesson_calls) == 1
    # source_session_ids fields were skipped (read failed).
    extra = kb.lesson_calls[0]["extra_attrs"]
    assert "source_session_ids" not in extra
    assert "validated_count" not in extra


@pytest.mark.asyncio
async def test_fact_write_hook_recovers_when_kb_write_throws(tmp_path):
    """KB write raises (e.g. unexpected non-CortexKBError, OSError on
    NDJSON file). The hook MUST swallow and continue. We test the
    outer ``except Exception`` guard in ``_fact_write_hook`` by making
    propose raise an unrecoverable exception."""
    class _WriteFailingKB:
        enabled = True

        def read_lesson_exact(self, **kw):
            return {}

        def read_pitfall_exact(self, **kw):
            return {}

        def propose_lesson(self, **kw):
            raise OSError("disk full")

        def propose_pitfall(self, **kw):
            raise OSError("disk full")

    coord = _coord_with_kb(_WriteFailingKB(), tmp_path)

    class _Result:
        result = {"gain_pct": 7.0, "output_throughput": 750.0}

    # No exception expected — _fact_write_hook swallows.
    await coord._fact_write_hook(
        task=_Task(), result=_Result(), kept=True,
    )


# ===========================================================================
# 3. CLOSE-time finalize is silent when KB is down
# ===========================================================================
def test_cortex_finalize_silent_when_kb_none(tmp_path):
    coord = _coord_with_kb(None, tmp_path)
    # Should not raise.
    coord.cortex_finalize_recipe_and_journal()


def test_cortex_finalize_silent_when_kb_disabled(tmp_path):
    class _DisabledKB:
        enabled = False

    coord = _coord_with_kb(_DisabledKB(), tmp_path)
    # Should not raise (journal still finalises; KB writes skipped).
    coord.cortex_finalize_recipe_and_journal()


# ===========================================================================
# 4. warm_replay path is KB-independent
# ===========================================================================
@pytest.mark.asyncio
async def test_inject_warm_recipe_history_works_without_kb_calls(tmp_path):
    """The injection helper reads SharedState only — no KB call. If
    KB is down it still works because warm_start_recipe was already
    populated by T0 (or empty if T0 also degraded)."""
    coord = _coord_with_kb(None, tmp_path)
    coord.shared_state.warm_start_recipe = {
        "tier": "T1_exact",
        "recipe": {
            "attrs": {
                "what_failed": [{
                    "name": "x",
                    "extra_sglang_args": "--bad-flag",
                    "extra_envs": {},
                    "gain_pct": -10.0,
                }],
            },
        },
    }
    added = coord._inject_warm_recipe_history_into_ledger()
    assert added == 1


@pytest.mark.asyncio
async def test_warm_replay_promote_works_without_kb_calls(tmp_path):
    """``_promote_warm_replay`` decides reproduced / drift / failed
    from result + SharedState. No KB call → KB outage cannot block
    the warm-replay decision."""
    coord = _coord_with_kb(None, tmp_path)
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "expected_gain_pct": 25.0,
    }
    task = _Task()
    task.params = {
        "extra_sglang_args": "--attention-backend AITER",
        "extra_envs": {},
    }
    result = {"status": "succeeded", "output_throughput": 738.0}
    # Should not raise.
    coord._promote_warm_replay(result, task=task)
    # And it should have made a decision.
    assert coord.shared_state.warm_replay_outcome["status"] == "reproduced"


# ===========================================================================
# 5. CLI _bootstrap_cortex_kb soft-degrades on mid-flight T0 failure
# ===========================================================================
def test_t0_runtime_failure_does_not_sys_exit(tmp_path, monkeypatch):
    """If T0 anchor catches a CortexKBError mid-flight (between IR-3
    probe and T0 call), the cli MUST disable the client and continue
    rather than ``sys.exit(2)``. Tested by monkey-patching
    ``run_t0_anchor`` to raise."""
    import argparse
    from inference_optimizer import cli as cli_mod

    args = argparse.Namespace(
        cortex_enabled=True,
        cortex_kb_url=KB_URL,
        kb_degraded_reason=None,
        framework="sglang",
        model_class="moe_mla",
    )

    def _raise(*a, **kw):
        raise CortexKBError("simulated T0 failure", category="transport")

    monkeypatch.setattr(cli_mod, "run_t0_anchor", _raise)

    # Provide a minimal session_dir with a manifest the bootstrap reads.
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    manifest = {
        "session_id": "s1",
        "model_name": "DeepSeek-R1",
        "model_path": "/path/to/DeepSeek-R1",
        "framework": "sglang",
        "gpu_type": "MI300X",
        "stack_fingerprint": {},
        "image": "",
    }
    # _bootstrap_cortex_kb signature: (args, *, session_dir, manifest, resume=False)
    client = cli_mod._bootstrap_cortex_kb(
        args, session_dir=session_dir, manifest=manifest, resume=False,
    )
    # Client returned (no sys.exit), with enabled flipped off.
    assert client is not None
    assert client.enabled is False
    assert args.cortex_enabled is False
    assert args.kb_degraded_reason == "t0_runtime_fail"
