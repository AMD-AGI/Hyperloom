# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the dashboard shows, read from the files a campaign already writes."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kernelforge.loop.direction_ledger import DIRECTIONS_FILE, DirectionLedger, direction_standings, keep_gains
from kernelforge.loop.early_stop import EarlyStopPolicy, early_stop_standing, window_gain
from kernelforge.loop.operator_inbox import OperatorInbox

DASHBOARD_JSON = "dashboard.json"
DASHBOARD_HTML = "dashboard.html"
_RECENT_EVENTS = 15
_TEXT_LIMIT = 6000
_RECENT_ROUNDS = 30
# How much of each round's own words the page carries: the plan, the implementer's account, the failure.
_ROUND_TEXT_LIMITS = {"plan": 2000, "rationale": 4000, "validation": 1500}
_VALIDATION_FAILURES = frozenset(
    {"REVERT_VALIDATION", "REVERT_VALIDATION_TIMEOUT", "REVERT_VALIDATION_ERROR", "BUILD_FAILED", "CRASH"}
)
# Events worth a line in the recent-events panel; iteration results have their own panels.
_NOTABLE_EVENTS = frozenset(
    {
        "session_started",
        "run_terminated",
        "analysis_result",
        "supervisor_ruling",
        "relay_profile_switched",
        "search_policy_decision",
        "merge_attempt_started",
        "merge_attempt_result",
        "operator_message_consumed",
        "direction_added",
        "direction_audit",
        "null_control",
        "incumbent_tree_reset",
    }
)


@dataclass(frozen=True)
class LiveStatus:
    """What only the running loop knows: its state, its clock, and how its scores are anchored."""

    status: str
    termination_reason: str = ""
    deadline_unix: float | None = None
    time_remaining_sec: float | None = None
    anchor_label: str = ""
    serving_speedup: float | None = None
    usage_path: str = ""
    early_stop: dict | None = None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_text(path: Path, limit: int = _TEXT_LIMIT) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text if len(text) <= limit else text[:limit] + "\n…"


def _read_events(path: Path) -> list[dict]:
    events: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return events
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _iteration_row(event: dict) -> dict:
    return {
        "iter": event.get("iter"),
        "ts": event.get("ts", ""),
        "decision": str(event.get("decision") or ""),
        "score": event.get("mean_case_speedup"),
        "best_after": event.get("best_after_mean_case_speedup"),
        "plan": str(event.get("plan") or ""),
        "model": str(event.get("agent_model") or ""),
        "session_end_reason": str(event.get("session_end_reason") or ""),
        "lane_id": event.get("lane_id"),
        "direction_id": event.get("direction_id"),
        "keep_bar": event.get("keep_bar"),
    }


def _event_summary(event: dict) -> dict:
    detail = {key: value for key, value in event.items() if key not in {"type", "iter", "ts"}}
    text = ", ".join(f"{key}={value}" for key, value in detail.items() if value not in (None, "", [], {}))
    return {"type": event.get("type", ""), "iter": event.get("iter"), "ts": event.get("ts", ""), "detail": text[:300]}


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "\n…"


def _round(campaign_root: Path, event: dict) -> dict:
    """One iteration as a reviewer reads it: what was planned, what the implementer says it did, and the verdict."""
    iteration = int(event.get("iter") or 0)
    folder = campaign_root / "candidates" / f"iter_{iteration:03d}"
    meta = _read_json(folder / "meta.json") or {}
    decision = str(event.get("decision") or "")
    validation = _read_text(folder / "validation.txt") if decision in _VALIDATION_FAILURES else ""
    return _iteration_row(event) | {
        "best_before": meta.get("best_mean_case_speedup_before"),
        "delta_vs_best_pct": meta.get("delta_vs_best_pct"),
        "turns": event.get("turns"),
        "plan": _clip(str(meta.get("plan") or event.get("plan") or ""), _ROUND_TEXT_LIMITS["plan"]),
        "rationale": _clip(str(meta.get("rationale") or ""), _ROUND_TEXT_LIMITS["rationale"]),
        "validation": _clip(validation, _ROUND_TEXT_LIMITS["validation"]),
    }


def _natural(case_id: str) -> list:
    """``m=7`` before ``m=14``: digits compare as numbers."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", case_id)]


def _cases(state: dict) -> list[dict]:
    anchor = state.get("baseline_case_times") or {}
    best = state.get("best_case_times") or {}
    unscored = set(state.get("unscored_cases") or [])
    return [
        {
            "case_id": case_id,
            "anchor_ms": anchor[case_id],
            "best_ms": best.get(case_id),
            "unscored": case_id in unscored,
        }
        for case_id in sorted(anchor, key=_natural)
    ]


def _campaign(state: dict) -> dict:
    keys = (
        "campaign_id",
        "git_branch",
        "session_status",
        "session_index",
        "iteration",
        "search_mode",
        "termination_reason",
        "search_start_mean_case_speedup",
    )
    view = {key: state.get(key) for key in keys}
    view["cumulative"] = state.get("cumulative") or {}
    view["best"] = state.get("best") or {}
    view["stall"] = state.get("stall") or {}
    return view


def _inbox(campaign_root: Path) -> dict:
    inbox = OperatorInbox(campaign_root.parent)
    try:
        pending = [path.read_text(encoding="utf-8", errors="replace").strip() for path in inbox.pending()]
        return {"pending": pending, "active": list(inbox.active_guidance())}
    except OSError:
        return {"pending": [], "active": []}


def _directions(campaign_root: Path, state: dict, events: list[dict], live_block: dict) -> dict:
    """The ledger's standings and, while early stop is on, where each of its criteria stands."""
    if not (campaign_root / DIRECTIONS_FILE).is_file():
        return {}
    policy = EarlyStopPolicy(**(live_block.get("early_stop") or {"enabled": True}))
    ledger = DirectionLedger(campaign_root)
    results = [event for event in events if event.get("type") == "iteration_result"]
    gains = keep_gains(results, state.get("search_start_mean_case_speedup"))
    standings = direction_standings(ledger.directions(), results, exhaust_after=policy.exhaust_after, gains=gains)
    queue = _read_json(campaign_root / "orchestration" / "lane_queue.json")
    standing = early_stop_standing(
        policy,
        standings,
        gain=window_gain(results, gains, policy.window_rounds),
        audits=ledger.audits(),
        lane_queue_empty=not (queue.get("candidates") if isinstance(queue, dict) else queue),
        inbox_pending=bool(OperatorInbox(campaign_root.parent).pending()),
    )
    return {
        "ledger": [s.to_view() | {"source": s.direction.source} for s in standings],
        "audits": ledger.audits(),
        "policy": {
            "min_directions": policy.min_directions,
            "window_rounds": policy.window_rounds,
            "min_gain": policy.min_gain,
            "exhaust_after": policy.exhaust_after,
        },
        "standing": {
            "terminal": standing.terminal,
            "window_gain": standing.window_gain,
            "criteria_met": standing.criteria_met,
            "audit_due": standing.audit_due,
            "unmet": list(standing.reasons),
        },
    }


def collect_snapshot(campaign_root: Path, live: LiveStatus | None = None) -> dict:
    """Everything one render needs, as plain JSON. Without ``live`` the previous render's live block is reused."""
    campaign_root = Path(campaign_root)
    state = _read_json(campaign_root / "run_state.json") or {}
    events = _read_events(campaign_root / "events.jsonl")
    if live is not None:
        live_block = asdict(live)
    else:
        previous = _read_json(campaign_root / DASHBOARD_JSON) or {}
        live_block = previous.get("live") or {}
    usage_path = Path(live_block.get("usage_path") or campaign_root / "llm_usage.json")
    notable = [event for event in events if event.get("type") in _NOTABLE_EVENTS]
    results = [event for event in events if event.get("type") == "iteration_result"]
    return {
        "rendered_at": time.time(),
        "campaign_root": str(campaign_root),
        "live": live_block,
        "campaign": _campaign(state),
        "iterations": [_iteration_row(event) for event in results],
        "rounds": [_round(campaign_root, event) for event in reversed(results[-_RECENT_ROUNDS:])],
        "cases": _cases(state),
        "usage": _read_json(usage_path) or {},
        "supervisor": _read_text(campaign_root / "supervisor" / "latest.md"),
        "events": [_event_summary(event) for event in notable[-_RECENT_EVENTS:]],
        "inbox": _inbox(campaign_root),
        "directions": _directions(campaign_root, state, events, live_block),
        "noise_floor": _read_json(campaign_root / "noise_floor.json") or {},
        "ablation": _read_json(campaign_root / "ablation.json") or {},
    }
