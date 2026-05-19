"""Reactor + L1/L2 finalizer integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from robustness_agent.decision.action_ladder import (
    ActionLadder,
    ActionLadderConfig,
)
from robustness_agent.decision.policy_aware import PolicyAware
from robustness_agent.finalize.postmortem import PostmortemFinalizer
from robustness_agent.findings.sink import FindingSink, FindingSinkConfig
from robustness_agent.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.role.reactor import Reactor, ReactorComponents
from robustness_agent.signals import Classifier
from robustness_agent.signals.crash import CrashConfig
from robustness_agent.sources.base import DegradeRouter, SourceData


class _FakeSource:
    def __init__(self, name: str, snapshot: SourceData) -> None:
        self.name = name
        self._snapshot = snapshot

    async def fetch(self, ctx) -> SourceData:
        return self._snapshot


def _build_reactor(
    *, tmp_path: Path, session_id: str = "sess-1",
) -> tuple[Reactor, PostmortemFinalizer]:
    primary = _FakeSource("primary", SourceData(sources_used=["primary"]))
    fallback = _FakeSource("fb", SourceData(sources_used=["fb"]))
    router = DegradeRouter(
        primary, fallback, fail_threshold=2, recheck_interval_s=0.0,
    )
    sink = FindingSink(
        FindingSinkConfig(session_dir=tmp_path, session_id=session_id)
    )
    finalizer = PostmortemFinalizer(
        session_dir=tmp_path, session_id=session_id,
    )
    components = ReactorComponents(
        router=router,
        classifier=Classifier(crash_config=CrashConfig(medium_threshold=2)),
        ladder=ActionLadder(config=ActionLadderConfig(cooldown_ticks=0)),
        policy=PolicyAware(),
        sink=sink,
        finalizer=finalizer,
    )
    return Reactor(components), finalizer


def _ctx(
    *, stop_reason: str = "", session_id: str = "sess-1", now_unix: float = 1.0,
) -> ReactorContext:
    return ReactorContext(
        tick_index=0,
        shared_state=SharedStateSnapshot(
            session_id=session_id, stop_reason=stop_reason,
        ),
        inbox=[],
        now_unix=now_unix,
    )


@pytest.mark.asyncio
async def test_reactor_does_not_finalize_while_stop_reason_empty(
    tmp_path: Path,
):
    reactor, finalizer = _build_reactor(tmp_path=tmp_path)
    await reactor.tick(_ctx(stop_reason=""))
    assert finalizer.is_finalized() is False
    assert not (tmp_path / "reports" / "robustness_postmortem.md").exists()


@pytest.mark.asyncio
async def test_reactor_finalizes_on_stop_reason_transition(tmp_path: Path):
    reactor, finalizer = _build_reactor(tmp_path=tmp_path)
    await reactor.tick(_ctx(stop_reason=""))
    await reactor.tick(_ctx(stop_reason="budget_exhausted"))
    assert finalizer.is_finalized() is True
    md = (tmp_path / "reports" / "robustness_postmortem.md").read_text(
        encoding="utf-8",
    )
    assert "budget_exhausted" in md


@pytest.mark.asyncio
async def test_reactor_finalize_fires_only_once(tmp_path: Path):
    reactor, finalizer = _build_reactor(tmp_path=tmp_path)
    await reactor.tick(_ctx(stop_reason="budget_exhausted"))
    md_path = tmp_path / "reports" / "robustness_postmortem.md"
    md_path.write_text("MUTATED\n", encoding="utf-8")
    # Subsequent ticks must not re-trigger.
    await reactor.tick(_ctx(stop_reason="budget_exhausted"))
    await reactor.tick(_ctx(stop_reason="other_reason"))
    assert md_path.read_text(encoding="utf-8") == "MUTATED\n"


@pytest.mark.asyncio
async def test_reactor_finalize_optional_when_disabled(tmp_path: Path):
    primary = _FakeSource("primary", SourceData(sources_used=["primary"]))
    fallback = _FakeSource("fb", SourceData(sources_used=["fb"]))
    router = DegradeRouter(
        primary, fallback, fail_threshold=2, recheck_interval_s=0.0,
    )
    sink = FindingSink(
        FindingSinkConfig(session_dir=tmp_path, session_id="sess-1")
    )
    # No finalizer wired — the reactor must still process ticks fine
    # with stop_reason set.
    components = ReactorComponents(
        router=router,
        classifier=Classifier(crash_config=CrashConfig(medium_threshold=2)),
        ladder=ActionLadder(config=ActionLadderConfig(cooldown_ticks=0)),
        policy=PolicyAware(),
        sink=sink,
        finalizer=None,
    )
    reactor = Reactor(components)
    await reactor.tick(_ctx(stop_reason="end"))
    assert not (tmp_path / "reports" / "robustness_postmortem.md").exists()
