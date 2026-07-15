# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end reactor tests plus L1/L2 finalizer integration. The subprocess-transport JSON-IO contract is exercised in test_runtime_cli.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.agents.robustness.decision.action_ladder import (
    ActionLadder,
    ActionLadderConfig,
    Finding,
)
from hyperloom.agents.robustness.decision.policy_aware import PolicyAware
from hyperloom.agents.robustness.role.envelope import IntentType
from hyperloom.agents.robustness.role.findings import FindingSink, FindingSinkConfig
from hyperloom.agents.robustness.role.postmortem import (
    PostmortemFinalizer,
    PostmortemFinalizerConfig,
    finalize_session,
)
from hyperloom.agents.robustness.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from hyperloom.agents.robustness.role.reactor import Reactor, ReactorComponents
from hyperloom.agents.robustness.signals import Classifier
from hyperloom.agents.robustness.signals.crash import CrashConfig
from hyperloom.agents.robustness.sources.base import (
    DegradeRouter,
    SourceData,
    SourceUnavailable,
)


class _FakeSource:
    """Stub source that returns a fixed snapshot or raises a fixed exception."""

    def __init__(self, name: str, snapshot: SourceData | Exception):
        self.name = name
        self._snapshot = snapshot
        self.calls = 0

    async def fetch(self, ctx) -> SourceData:
        self.calls += 1
        if isinstance(self._snapshot, BaseException):
            raise self._snapshot
        return self._snapshot


def _build_reactor(
    *,
    primary: _FakeSource,
    fallback: _FakeSource | None = None,
    tmp_path: Path,
    classifier: Classifier | None = None,
    cooldown_ticks: int = 0,
) -> tuple[Reactor, FindingSink]:
    fb = fallback or _FakeSource("fb", SourceData(coordinator_events=[], sources_used=["fb"]))
    router = DegradeRouter(primary, fb, fail_threshold=2, recheck_interval_s=0.0)
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="sess-1"))
    components = ReactorComponents(
        router=router,
        classifier=classifier or Classifier(configs={"crash": CrashConfig(medium_threshold=2)}),
        ladder=ActionLadder(config=ActionLadderConfig(cooldown_ticks=cooldown_ticks)),
        policy=PolicyAware(),
        sink=sink,
    )
    return Reactor(components), sink


def _build_reactor_with_finalizer(
    *,
    tmp_path: Path,
    session_id: str = "sess-1",
) -> tuple[Reactor, PostmortemFinalizer]:
    primary = _FakeSource("primary", SourceData(sources_used=["primary"]))
    fallback = _FakeSource("fb", SourceData(sources_used=["fb"]))
    router = DegradeRouter(
        primary,
        fallback,
        fail_threshold=2,
        recheck_interval_s=0.0,
    )
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id=session_id))
    finalizer = PostmortemFinalizer(
        session_dir=tmp_path,
        session_id=session_id,
    )
    components = ReactorComponents(
        router=router,
        classifier=Classifier(configs={"crash": CrashConfig(medium_threshold=2)}),
        ladder=ActionLadder(config=ActionLadderConfig(cooldown_ticks=0)),
        policy=PolicyAware(),
        sink=sink,
        finalizer=finalizer,
    )
    return Reactor(components), finalizer


def _ctx(
    crash_count: int = 0,
    *,
    session_id: str = "sess-1",
    now_unix: float = 1.0,
    stop_reason: str = "",
) -> ReactorContext:
    return ReactorContext(
        tick_index=0,
        shared_state=SharedStateSnapshot(
            session_id=session_id,
            crash_count=crash_count,
            stop_reason=stop_reason,
        ),
        inbox=[],
        now_unix=now_unix,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reactor_emits_heartbeat_when_no_symptoms(tmp_path: Path):
    primary = _FakeSource("server", SourceData(sources_used=["server"]))
    reactor, sink = _build_reactor(primary=primary, tmp_path=tmp_path)
    intents = await reactor.tick(_ctx())
    assert len(intents) == 1
    assert intents[0].type is IntentType.SEND_MESSAGE
    assert intents[0].payload["topic"] == "heartbeat"
    assert reactor.tick_index == 1
    assert not sink.file_path.exists()


@pytest.mark.asyncio
async def test_reactor_emits_alert_for_crash_count_and_persists_finding(tmp_path: Path):
    primary = _FakeSource("server", SourceData(sources_used=["server"]))
    reactor, sink = _build_reactor(primary=primary, tmp_path=tmp_path)
    intents = await reactor.tick(_ctx(crash_count=2))
    assert any(i.type is IntentType.ALERT for i in intents)
    assert sink.file_path.exists()
    rows = sink.file_path.read_text().splitlines()
    assert rows
    row = json.loads(rows[0])
    assert row["symptom_name"] == "crash_count_rising"
    assert row["intents"][0]["intent_type"] == "alert"


# ---------------------------------------------------------------------------
# DegradeRouter integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reactor_falls_back_to_secondary_when_primary_fails(tmp_path: Path):
    primary = _FakeSource("server", SourceUnavailable("down"))
    fallback = _FakeSource(
        "local",
        SourceData(
            session_pods=[{"pod": {"namespace": "ns", "name": "p"}, "phase": "Failed"}],
            sources_used=["local"],
        ),
    )
    reactor, _ = _build_reactor(primary=primary, fallback=fallback, tmp_path=tmp_path)
    intents = await reactor.tick(_ctx())
    assert any(i.type is IntentType.ALERT for i in intents)
    assert any(i.payload.get("severity") == "high" for i in intents if i.type is IntentType.ALERT)


# ---------------------------------------------------------------------------
# Policy filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reactor_drops_invalid_intents_emitted_by_extra_evaluator(tmp_path: Path):
    from hyperloom.agents.robustness.role.envelope import Intent, IntentType

    class CustomLadder(ActionLadder):
        def _intents_for(self, sym):  # type: ignore[override]
            base = super()._intents_for(sym)
            base.append(Intent(type=IntentType.ALERT, payload={"summary": "no severity"}))
            return base

    primary = _FakeSource("server", SourceData(sources_used=["server"]))
    fallback = _FakeSource("local", SourceData(sources_used=["local"]))
    router = DegradeRouter(primary, fallback, fail_threshold=2, recheck_interval_s=0.0)
    components = ReactorComponents(
        router=router,
        classifier=Classifier(configs={"crash": CrashConfig(medium_threshold=2)}),
        ladder=CustomLadder(),
        policy=PolicyAware(),
        sink=None,
    )
    reactor = Reactor(components)
    intents = await reactor.tick(_ctx(crash_count=2))
    assert all(i.payload.get("severity") for i in intents if i.type is IntentType.ALERT)


# ---------------------------------------------------------------------------
# Finalizer integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reactor_does_not_finalize_while_stop_reason_empty(
    tmp_path: Path,
):
    reactor, finalizer = _build_reactor_with_finalizer(tmp_path=tmp_path)
    await reactor.tick(_ctx(stop_reason=""))
    assert finalizer.is_finalized() is False
    assert not (tmp_path / "reports" / "robustness_postmortem.md").exists()


@pytest.mark.asyncio
async def test_reactor_finalizes_on_stop_reason_transition(tmp_path: Path):
    reactor, finalizer = _build_reactor_with_finalizer(tmp_path=tmp_path)
    await reactor.tick(_ctx(stop_reason=""))
    await reactor.tick(_ctx(stop_reason="budget_exhausted"))
    assert finalizer.is_finalized() is True
    md = (tmp_path / "reports" / "robustness_postmortem.md").read_text(
        encoding="utf-8",
    )
    assert "budget_exhausted" in md


@pytest.mark.asyncio
async def test_reactor_finalize_fires_only_once(tmp_path: Path):
    reactor, finalizer = _build_reactor_with_finalizer(tmp_path=tmp_path)
    await reactor.tick(_ctx(stop_reason="budget_exhausted"))
    md_path = tmp_path / "reports" / "robustness_postmortem.md"
    md_path.write_text("MUTATED\n", encoding="utf-8")
    await reactor.tick(_ctx(stop_reason="budget_exhausted"))
    await reactor.tick(_ctx(stop_reason="other_reason"))
    assert md_path.read_text(encoding="utf-8") == "MUTATED\n"


@pytest.mark.asyncio
async def test_reactor_finalize_optional_when_disabled(tmp_path: Path):
    primary = _FakeSource("primary", SourceData(sources_used=["primary"]))
    fallback = _FakeSource("fb", SourceData(sources_used=["fb"]))
    router = DegradeRouter(
        primary,
        fallback,
        fail_threshold=2,
        recheck_interval_s=0.0,
    )
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="sess-1"))
    components = ReactorComponents(
        router=router,
        classifier=Classifier(configs={"crash": CrashConfig(medium_threshold=2)}),
        ladder=ActionLadder(config=ActionLadderConfig(cooldown_ticks=0)),
        policy=PolicyAware(),
        sink=sink,
        finalizer=None,
    )
    reactor = Reactor(components)
    await reactor.tick(_ctx(stop_reason="end"))
    assert not (tmp_path / "reports" / "robustness_postmortem.md").exists()


# ---------------------------------------------------------------------------
# FindingSink unit tests (folded in from test_findings_sink.py)
# ---------------------------------------------------------------------------


def _finding(**overrides) -> Finding:
    base = dict(
        tick_index=1,
        timestamp_unix=1.0,
        symptom_name="x",
        severity="medium",
        summary="s",
        intents=[{"intent_type": "alert", "payload": {"severity": "medium", "summary": "s"}}],
        evidence={"k": 1},
        rca_text="",
    )
    base.update(overrides)
    return Finding(**base)


@pytest.mark.asyncio
async def test_sink_appends_jsonl_rows(tmp_path: Path):
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="sess-1"))
    written = await sink.append_many([_finding(tick_index=1), _finding(tick_index=2)])
    assert written == 2
    path = sink.file_path
    assert path.exists()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert [r["tick_index"] for r in rows] == [1, 2]
    assert all(r["intents"][0]["intent_type"] == "alert" for r in rows)


@pytest.mark.asyncio
async def test_sink_appends_across_calls(tmp_path: Path):
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="sess-2"))
    await sink.append_many([_finding(tick_index=1)])
    await sink.append_many([_finding(tick_index=2)])
    rows = sink.file_path.read_text().splitlines()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_sink_creates_subdirectories_when_missing(tmp_path: Path):
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="abc"))
    assert not sink.file_path.exists()
    await sink.append_many([_finding()])
    assert sink.file_path.exists()
    assert sink.file_path.parent.name == "findings"


@pytest.mark.asyncio
async def test_sink_is_resilient_to_io_errors(tmp_path: Path, caplog):
    # Block creation by pre-creating a file where the parent dir would go
    blocker = tmp_path / "blocked"
    blocker.write_text("dummy")
    cfg2 = FindingSinkConfig(session_dir=blocker / "x", session_id="sess")
    sink2 = FindingSink(cfg2)
    with caplog.at_level("WARNING"):
        written = await sink2.append_many([_finding()])
    assert written == 1
    assert any("findings sink" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_sink_no_op_on_empty_iterable(tmp_path: Path):
    sink = FindingSink(FindingSinkConfig(session_dir=tmp_path, session_id="sess"))
    assert await sink.append_many([]) == 0
    assert not sink.file_path.exists()


# ---------------------------------------------------------------------------
# PostmortemFinalizer unit tests (folded in from test_finalize_postmortem.py)
# ---------------------------------------------------------------------------


def _write_finding(path: Path, **overrides) -> None:
    base = dict(
        tick_index=overrides.pop("tick_index", 1),
        timestamp_unix=overrides.pop("timestamp_unix", 1.0),
        symptom_name=overrides.pop("symptom_name", "x"),
        severity=overrides.pop("severity", "low"),
        summary=overrides.pop("summary", "s"),
        intents=overrides.pop("intents", []),
        evidence=overrides.pop("evidence", {}),
        rca_text=overrides.pop("rca_text", ""),
    )
    base.update(overrides)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(base) + "\n")


def _make_run_result(
    session_dir: Path,
    action: str,
    task_id: str,
    payload: dict,
) -> Path:
    task_dir = session_dir / "runs" / action / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "result.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return task_dir


def test_finalize_creates_postmortem_and_decision_trace(tmp_path: Path) -> None:
    findings_dir = tmp_path / "agents" / "robustness" / "findings"
    findings_dir.mkdir(parents=True)
    findings_path = findings_dir / "sess-1.jsonl"
    _write_finding(
        findings_path,
        tick_index=2,
        symptom_name="gpu_leak_persistent",
        severity="high",
        summary="ROCm KFD leak repeated",
        evidence={"used_mb": 71000},
        rca_text="reboot mitigated last 3 times",
        intents=[
            {
                "intent_type": "alert",
                "payload": {"severity": "high"},
            }
        ],
    )
    _write_finding(
        findings_path,
        tick_index=5,
        symptom_name="quota_low_hit",
        severity="medium",
    )

    _make_run_result(
        tmp_path,
        "integrate",
        "kernel_42",
        {
            "decision": "KEEP",
            "gain_pct": 12.5,
            "kernel_id": "fa_v2",
        },
    )
    _make_run_result(
        tmp_path,
        "integrate",
        "kernel_99",
        {
            "decision": "REVERT",
            "gain_pct": -3.0,
            "error_class": "regression",
        },
    )
    _make_run_result(
        tmp_path,
        "baseline",
        "task_0",
        {
            "decision": None,
            "output_throughput": 123.4,
            "status": "ok",
        },
    )

    finalizer = PostmortemFinalizer(
        session_dir=tmp_path,
        session_id="sess-1",
    )
    assert finalizer.finalize(stop_reason="budget_exhausted") is True
    assert finalizer.is_finalized() is True

    md_path = tmp_path / "reports" / "robustness_postmortem.md"
    trace_path = tmp_path / "reports" / "decision_trace.json"
    assert md_path.is_file()
    assert trace_path.is_file()

    md = md_path.read_text(encoding="utf-8")
    assert "Robustness postmortem" in md
    assert "budget_exhausted" in md
    assert "gpu_leak_persistent" in md
    assert "HIGH=1" in md
    # Decision-trace summary table renders per-action counts.
    assert "`integrate`" in md
    assert "`baseline`" in md

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["total_tasks"] == 3
    integrate = trace["tasks_by_action"]["integrate"]
    assert len(integrate) == 2
    decisions = {t["decision"] for t in integrate}
    assert decisions == {"KEEP", "REVERT"}
    assert any(t.get("gain_pct") == 12.5 for t in integrate)


def test_finalize_idempotent_via_marker(tmp_path: Path) -> None:
    finalizer = PostmortemFinalizer(
        session_dir=tmp_path,
        session_id="sess-x",
    )
    assert finalizer.finalize(stop_reason="r1") is True
    # Second invocation must not overwrite.
    md_path = tmp_path / "reports" / "robustness_postmortem.md"
    first_body = md_path.read_text(encoding="utf-8")
    md_path.write_text("MUTATED\n", encoding="utf-8")
    assert finalizer.finalize(stop_reason="r2") is False
    assert md_path.read_text(encoding="utf-8") == "MUTATED\n"
    assert first_body  # sanity: first body was non-empty


def test_finalize_no_findings_no_runs(tmp_path: Path) -> None:
    assert (
        finalize_session(
            tmp_path,
            session_id="empty",
            stop_reason="manual_close",
        )
        is True
    )
    md = (tmp_path / "reports" / "robustness_postmortem.md").read_text(
        encoding="utf-8",
    )
    assert "No HIGH-severity finding" in md
    assert "No ``runs/" in md or 'No "runs/' in md or "result.json" in md
    trace = json.loads((tmp_path / "reports" / "decision_trace.json").read_text(encoding="utf-8"))
    assert trace["total_tasks"] == 0
    assert trace["tasks_by_action"] == {}


def test_finalize_skips_malformed_jsonl_lines(tmp_path: Path) -> None:
    findings_dir = tmp_path / "agents" / "robustness" / "findings"
    findings_dir.mkdir(parents=True)
    findings_path = findings_dir / "sess-1.jsonl"
    findings_path.write_text(
        "{not json}\n"
        + json.dumps(
            {
                "tick_index": 1,
                "timestamp_unix": 1.0,
                "symptom_name": "x",
                "severity": "high",
                "summary": "s",
                "intents": [],
                "evidence": {},
                "rca_text": "",
            }
        )
        + "\n"
        + "\n",
        encoding="utf-8",
    )
    finalizer = PostmortemFinalizer(
        session_dir=tmp_path,
        session_id="sess-1",
    )
    assert finalizer.finalize(stop_reason="r") is True
    md = (tmp_path / "reports" / "robustness_postmortem.md").read_text(
        encoding="utf-8",
    )
    assert "HIGH=1" in md


def test_finalize_picks_first_high_severity_as_flashpoint(
    tmp_path: Path,
) -> None:
    findings_dir = tmp_path / "agents" / "robustness" / "findings"
    findings_dir.mkdir(parents=True)
    findings_path = findings_dir / "sess-1.jsonl"
    _write_finding(
        findings_path,
        tick_index=10,
        symptom_name="late_high",
        severity="high",
        summary="later high",
    )
    _write_finding(
        findings_path,
        tick_index=3,
        symptom_name="early_high",
        severity="high",
        summary="earlier high",
    )
    _write_finding(
        findings_path,
        tick_index=1,
        severity="medium",
    )
    finalizer = PostmortemFinalizer(
        session_dir=tmp_path,
        session_id="sess-1",
    )
    assert finalizer.finalize(stop_reason="r") is True
    md = (tmp_path / "reports" / "robustness_postmortem.md").read_text(
        encoding="utf-8",
    )
    assert "early_high" in md
    # Flashpoint section appears before catalogue.
    flash_pos = md.index("Flashpoint")
    catalogue_pos = md.index("Findings catalogue")
    assert flash_pos < catalogue_pos


def test_finalize_caps_tasks_per_action(tmp_path: Path) -> None:
    import os

    for i in range(8):
        task_dir = _make_run_result(
            tmp_path,
            "sweep",
            f"task_{i:03d}",
            {"decision": "KEEP" if i % 2 == 0 else "REVERT", "gain_pct": float(i)},
        )
        # Deterministic mtime: i=7 newest, i=0 oldest.
        ts = 1_700_000_000 + i * 10
        os.utime(task_dir, (ts, ts))
    cfg = PostmortemFinalizerConfig(max_tasks_per_action=3)
    finalizer = PostmortemFinalizer(
        session_dir=tmp_path,
        session_id="sess-1",
        config=cfg,
    )
    assert finalizer.finalize(stop_reason="r") is True
    trace = json.loads((tmp_path / "reports" / "decision_trace.json").read_text(encoding="utf-8"))
    sweep_tasks = trace["tasks_by_action"]["sweep"]
    assert len(sweep_tasks) == 3
    task_ids = [t["task_id"] for t in sweep_tasks]
    assert task_ids == ["task_007", "task_006", "task_005"]
