"""E1 / E2 / E4 / E5 critic-health signal tests."""

from __future__ import annotations

from robustness_agent.role.prompt_inputs import (
    InboxItem,
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.signals import SymptomSeverity
from robustness_agent.signals.critic_health import (
    CriticHealthConfig,
    evaluate_critic_health_signals,
)
from robustness_agent.sources.base import SourceData


def _ctx(*, inbox=None) -> ReactorContext:
    return ReactorContext(
        tick_index=1,
        shared_state=SharedStateSnapshot(session_id="sess-1"),
        inbox=list(inbox or []),
        now_unix=1.0,
    )


# ---------------------------------------------------------------------------
# E1 — critic_kb_outage
# ---------------------------------------------------------------------------

def test_e1_kb_outage_fires_after_streak():
    critic_health = {
        "workdir_root": "/p/critic-workdir",
        "workdir_count": 4,
        "recent_judges": [
            # newest first
            {"turn_dir": "000004", "kb_read_skipped_reason": "kb_unreachable"},
            {"turn_dir": "000003", "kb_read_skipped_reason": "kb_unreachable"},
            {"turn_dir": "000002", "kb_read_skipped_reason": "kb_unreachable"},
            {"turn_dir": "000001", "kb_read_skipped_reason": None},
        ],
    }
    data = SourceData(local_critic_health=critic_health)
    out = evaluate_critic_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "critic_kb_outage")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["consecutive_judges"] == 3


def test_e1_kb_outage_silent_below_threshold():
    critic_health = {
        "workdir_root": "/p",
        "workdir_count": 2,
        "recent_judges": [
            {"turn_dir": "000002", "kb_read_skipped_reason": "kb_unreachable"},
            {"turn_dir": "000001", "kb_read_skipped_reason": "kb_unreachable"},
        ],
    }
    data = SourceData(local_critic_health=critic_health)
    out = evaluate_critic_health_signals(_ctx(), data)
    assert all(s.name != "critic_kb_outage" for s in out)


def test_e1_kb_outage_streak_resets_on_clean_judge():
    critic_health = {
        "recent_judges": [
            {"turn_dir": "5", "kb_read_skipped_reason": "kb_unreachable"},
            {"turn_dir": "4", "kb_read_skipped_reason": None},  # reset
            {"turn_dir": "3", "kb_read_skipped_reason": "kb_unreachable"},
            {"turn_dir": "2", "kb_read_skipped_reason": "kb_unreachable"},
            {"turn_dir": "1", "kb_read_skipped_reason": "kb_unreachable"},
        ],
    }
    data = SourceData(local_critic_health=critic_health)
    out = evaluate_critic_health_signals(_ctx(), data)
    # Streak is only 1 (the newest), below threshold.
    assert all(s.name != "critic_kb_outage" for s in out)


# ---------------------------------------------------------------------------
# E2 — critic_unavailable_streak
# ---------------------------------------------------------------------------

def test_e2_critic_unavailable_streak_fires():
    coord_events = [
        {"topic": "review_verdict", "agent": "critic",
         "payload": {"verdict": "needs_review", "source": "critic_unavailable",
                     "target_proposal_msg_id": "p1"}},
        {"topic": "review_verdict", "agent": "critic",
         "payload": {"verdict": "needs_review", "source": "critic_unavailable",
                     "target_proposal_msg_id": "p2"}},
        {"topic": "review_verdict", "agent": "critic",
         "payload": {"verdict": "needs_review", "source": "critic_unavailable",
                     "target_proposal_msg_id": "p3"}},
    ]
    data = SourceData(coordinator_events=coord_events)
    out = evaluate_critic_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "critic_unavailable_streak")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["consecutive_verdicts"] == 3
    assert "p3" in sym.evidence["sample_targets"]


def test_e2_critic_unavailable_streak_resets_on_real_critic_verdict():
    coord_events = [
        {"topic": "review_verdict", "agent": "critic",
         "payload": {"source": "critic_unavailable", "target_proposal_msg_id": "p1"}},
        {"topic": "review_verdict", "agent": "critic",
         "payload": {"source": "critic", "target_proposal_msg_id": "p2"}},
        {"topic": "review_verdict", "agent": "critic",
         "payload": {"source": "critic_unavailable", "target_proposal_msg_id": "p3"}},
    ]
    data = SourceData(coordinator_events=coord_events)
    out = evaluate_critic_health_signals(_ctx(), data)
    # Streak from the newest is 1 (only p3) — silent.
    assert all(s.name != "critic_unavailable_streak" for s in out)


def test_e2_critic_unavailable_streak_reads_inbox_too():
    inbox = [
        InboxItem(seq=1, msg_id="m1", from_agent="critic",
                  topic="review_verdict",
                  payload={"source": "critic_unavailable",
                           "target_proposal_msg_id": "p1"}),
        InboxItem(seq=2, msg_id="m2", from_agent="critic",
                  topic="review_verdict",
                  payload={"source": "critic_unavailable",
                           "target_proposal_msg_id": "p2"}),
        InboxItem(seq=3, msg_id="m3", from_agent="critic",
                  topic="review_verdict",
                  payload={"source": "critic_unavailable",
                           "target_proposal_msg_id": "p3"}),
    ]
    out = evaluate_critic_health_signals(_ctx(inbox=inbox), SourceData())
    sym = next(s for s in out if s.name == "critic_unavailable_streak")
    assert sym.evidence["consecutive_verdicts"] == 3


# ---------------------------------------------------------------------------
# E4 — critic_prune_stuck
# ---------------------------------------------------------------------------

def test_e4_critic_prune_stuck_fires_at_double_keep_count():
    data = SourceData(local_critic_health={
        "workdir_root": "/p", "workdir_count": 150, "recent_judges": [],
    })
    out = evaluate_critic_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "critic_prune_stuck")
    assert sym.severity is SymptomSeverity.MEDIUM
    assert sym.evidence["workdir_count"] == 150


def test_e4_critic_prune_stuck_silent_below_threshold():
    data = SourceData(local_critic_health={
        "workdir_root": "/p", "workdir_count": 60, "recent_judges": [],
    })
    out = evaluate_critic_health_signals(_ctx(), data)
    assert all(s.name != "critic_prune_stuck" for s in out)


# ---------------------------------------------------------------------------
# E5 — critic_runtime_stuck
# ---------------------------------------------------------------------------

def test_e5_critic_runtime_stuck_fires_on_timeout_marker():
    data = SourceData(local_log_errors=[
        {"pattern": r"runtime\.cli .* timed out after \d+s",
         "line": "ERROR: runtime.cli commit-review timed out after 30s"},
    ])
    out = evaluate_critic_health_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "critic_runtime_stuck")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["hit_count"] == 1


def test_e5_silent_without_runtime_cli_substring():
    data = SourceData(local_log_errors=[
        {"pattern": "some_other_pattern",
         "line": "ERROR: something timed out"},  # no ``runtime.cli``
    ])
    out = evaluate_critic_health_signals(_ctx(), data)
    assert all(s.name != "critic_runtime_stuck" for s in out)


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------

def test_custom_thresholds_apply():
    cfg = CriticHealthConfig(
        min_outage_judges=1, min_unavailable_verdicts=1,
        max_workdir_count=10, min_runtime_stuck_hits=1,
    )
    data = SourceData(local_critic_health={
        "workdir_root": "/p", "workdir_count": 11,
        "recent_judges": [
            {"turn_dir": "1", "kb_read_skipped_reason": "kb_unreachable"},
        ],
    })
    out = evaluate_critic_health_signals(_ctx(), data, config=cfg)
    names = {s.name for s in out}
    assert "critic_kb_outage" in names
    assert "critic_prune_stuck" in names
