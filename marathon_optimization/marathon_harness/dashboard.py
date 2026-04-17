"""Dashboard — live terminal status with ALL stack actions, score history,
branch tracking, KM/WD status, events, findings.  Writes dashboard.txt for remote cat.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ipc

log = logging.getLogger(__name__)

REFRESH_INTERVAL_S = 30


class Dashboard:
    """Observability dashboard — prints to terminal + writes dashboard.txt."""

    def __init__(self, state: Any, session_dir: str):
        self.state = state
        self.session_dir = session_dir
        self.km_current: dict[str, Any] = {}
        self.wd_status_text: str = "idle"
        self.wd_active_investigations: int = 0
        self.wd_patterns: dict[str, int] = {}
        self._last_lines: list[str] = []
        self._cached_events: list[dict[str, Any]] = []
        self._cached_findings: list[dict[str, Any]] = []
        self._events_cursor: str = ""
        self._findings_cursor: str = ""

    # ------------------------------------------------------------------
    # Update hooks (called by orchestrator / KM / WD)
    # ------------------------------------------------------------------

    def log_branch(self, branch_type: str, action: dict[str, Any]) -> None:
        """Record add/remove/complete in branch_log."""
        self.state.branch_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": branch_type,
            "action": {
                "id": action.get("id", "?"),
                "action": action.get("action", "?"),
                "score": action.get("score"),
                "description": action.get("description", "")[:80],
            },
        })
        # Cap branch log to last 500 entries
        if len(self.state.branch_log) > 500:
            self.state.branch_log = self.state.branch_log[-500:]

    def log_score_snapshot(self, action_stack: list[dict[str, Any]]) -> None:
        """Take a score snapshot for history."""
        self.state.score_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scores": {a.get("id", "?"): a.get("score", 0) for a in action_stack},
        })
        # Cap to last 100 snapshots
        if len(self.state.score_history) > 100:
            self.state.score_history = self.state.score_history[-100:]

    def update_km(self, kernel: str, round_num: int, backend: str, last_outcome: str) -> None:
        self.km_current = {
            "kernel": kernel,
            "round": round_num,
            "backend": backend,
            "last_outcome": last_outcome,
            "updated_at": time.time(),
        }

    def update_wd(self, status: str, active_investigations: int, patterns: dict[str, int]) -> None:
        self.wd_status_text = status
        self.wd_active_investigations = active_investigations
        self.wd_patterns = patterns

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, shutdown: asyncio.Event) -> None:
        while not shutdown.is_set():
            try:
                self._render()
                self._write_file()
            except Exception as exc:
                log.debug("Dashboard render error: %s", exc)
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=REFRESH_INTERVAL_S)
                break
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render(self) -> None:
        s = self.state
        lines: list[str] = []
        w = 90

        lines.append("=" * w)
        lines.append(f"MARATHON HARNESS — {s.model_name or '?'} on {s.gpu_count}x {s.gpu_type}")
        lines.append(f"Phase: {s.phase} | Tier: {s.current_time_tier}")
        elapsed_h = (time.time() - s.start_time) / 3600
        lines.append(f"Elapsed: {elapsed_h:.1f}h / 24h | Cost: ${s.total_llm_cost_usd:.2f} | LLM calls: {s.total_llm_calls}")
        if s.baseline_tput_per_gpu > 0:
            best = s.best_tput_per_gpu or s.current_tput_per_gpu
            gain_pct = (best - s.baseline_tput_per_gpu) / s.baseline_tput_per_gpu * 100 if s.baseline_tput_per_gpu > 0 else 0
            lines.append(
                f"Throughput: {s.baseline_tput_per_gpu:.1f} → {best:.1f} tok/s/GPU "
                f"(+{gain_pct:.1f}%)"
            )
        if s.target_tput_per_gpu > 0:
            lines.append(f"Target: {s.target_tput_per_gpu:.1f} | Gap: {s.target_gap_pct:.1f}%")
        lines.append("-" * w)

        # FULL ACTION STACK (ALL actions)
        sorted_stack = sorted(s.action_stack, key=lambda a: -a.get("score", 0))
        lines.append(f"ACTION STACK ({len(sorted_stack)} actions):")
        if not sorted_stack:
            lines.append("  (empty)")
        for i, action in enumerate(sorted_stack):
            marker = "►" if i == 0 else " "
            score = action.get("score", 0)
            prev = self._prev_score(action.get("id", ""))
            delta = ""
            if prev is not None and abs(prev - score) > 0.01:
                delta = f" (was {prev:.1f})"
            desc = action.get("description", "")[:55]
            lines.append(
                f"  {marker} [{score:.1f}{delta}] {action.get('id', '?')}: "
                f"{action.get('action', '?')} — {desc}"
            )
        lines.append("-" * w)

        # BRANCH LOG (last 20)
        lines.append("BRANCH LOG (last 20):")
        for entry in s.branch_log[-20:]:
            ts = entry["timestamp"][-8:] if "T" in entry.get("timestamp", "") else "??:??:??"
            typ = entry.get("type", "?")
            act = entry.get("action", {})
            symbol = {"add": "+", "remove": "-", "complete": "✓"}.get(typ, "?")
            score_str = f"[{act['score']:.1f}]" if act.get("score") is not None else ""
            lines.append(f"  {ts} {symbol} {score_str} {act.get('id', '?')}: {act.get('action', '')}")
        if not s.branch_log:
            lines.append("  (no activity yet)")
        lines.append("-" * w)

        # SCORE EVOLUTION
        if s.score_history:
            recent = s.score_history[-5:]
            all_ids: set[str] = set()
            for snap in recent:
                all_ids.update(snap.get("scores", {}).keys())
            if all_ids:
                lines.append(f"SCORE EVOLUTION (last {len(recent)} snapshots):")
                for aid in sorted(all_ids):
                    scores_str = " → ".join(
                        str(snap.get("scores", {}).get(aid, "-")) for snap in recent
                    )
                    lines.append(f"  {aid}: {scores_str}")
                lines.append("-" * w)

        # COMPLETED ACTIONS
        lines.append(f"COMPLETED ({len(s.completed_actions)} total, last 10):")
        for ca in s.completed_actions[-10:]:
            result = ca.get("result", {})
            gain = result.get("gain_pct", None)
            status = result.get("status", "?")
            if gain is not None and gain != 0:
                sign = "+" if gain > 0 else ""
                gain_str = f"{sign}{gain:.1f}%"
            elif result.get("needs_benchmark"):
                gain_str = "pending bench"
            else:
                gain_str = "no e2e"
            lines.append(f"  {ca.get('id', '?')}: {status} ({gain_str})")
        if not s.completed_actions:
            lines.append("  (none yet)")
        lines.append("-" * w)

        # KERNEL MANAGER
        km = self.km_current
        if km and time.time() - km.get("updated_at", 0) < 300:
            lines.append(
                f"KERNEL MANAGER: Round {km.get('round', '?')}/5 on {km.get('kernel', '?')} "
                f"via {km.get('backend', '?')}"
            )
            lines.append(f"  Last outcome: {km.get('last_outcome', '?')}")
        else:
            lines.append("KERNEL MANAGER: idle")
        lines.append(
            f"  Queue: ? pending | Completed: {s.kernel_manager_merges_completed} | "
            f"Kept: {s.kernel_manager_merges_kept}"
        )
        lines.append("-" * w)

        # WATCHDOG
        lines.append(f"WATCHDOG: {self.wd_status_text}")
        lines.append(f"  Events: {s.events_written} total | Findings consumed: {s.watchdog_findings_consumed}")
        lines.append(f"  Active investigations: {self.wd_active_investigations}/2")
        if s.watchdog_hw_blocked_kernels:
            lines.append(f"  HW-blocked: {s.watchdog_hw_blocked_kernels}")
        if self.wd_patterns:
            top_patterns = sorted(self.wd_patterns.items(), key=lambda x: -x[1])[:5]
            lines.append(f"  Top patterns: {dict(top_patterns)}")
        lines.append("-" * w)

        # KERNEL UNIQUENESS
        if s.discovered_kernels:
            lines.append(f"KERNELS: {s.unique_kernel_count} unique / {s.kernel_attempt_count} attempts")
            recent_kernels = sorted(
                s.discovered_kernels.values(),
                key=lambda k: k.get("discovered_at", 0), reverse=True,
            )[:5]
            for k in recent_kernels:
                lines.append(f"  {k.get('kernel_name', '?')} — attempts: {k.get('attempt_count', 0)}")
            lines.append("-" * w)

        # RECENT EVENTS (cursor-based incremental read)
        new_events = ipc.read_new_events(self.session_dir, after_id=self._events_cursor)
        if new_events:
            self._cached_events.extend(new_events)
            self._cached_events = self._cached_events[-100:]
            self._events_cursor = new_events[-1].get("id", self._events_cursor)
        lines.append(f"RECENT EVENTS (last 10 of {len(self._cached_events)}):")
        for evt in self._cached_events[-10:]:
            ts = evt.get("timestamp", "")[-8:] if "T" in evt.get("timestamp", "") else "?"
            lines.append(
                f"  [{ts}] {evt.get('type', '?')} {evt.get('kernel_name', '')} "
                f"(sev:{evt.get('severity', '?')})"
            )
        if not self._cached_events:
            lines.append("  (none)")
        lines.append("-" * w)

        # RECENT FINDINGS (cursor-based incremental read)
        new_findings = ipc.read_new_findings(
            self.session_dir, after_event_id=self._findings_cursor)
        if new_findings:
            self._cached_findings.extend(new_findings)
            self._cached_findings = self._cached_findings[-50:]
            self._findings_cursor = new_findings[-1].get("event_id", self._findings_cursor)
        lines.append(f"RECENT FINDINGS (last 5 of {len(self._cached_findings)}):")
        for fnd in self._cached_findings[-5:]:
            lines.append(
                f"  {fnd.get('event_id', '?')}: {fnd.get('classification', '?')} — "
                f"{(fnd.get('root_cause') or '')[:60]}"
            )
            guidance = fnd.get("actionable_guidance", {})
            lines.append(f"    resubmit={fnd.get('resubmit')} approach={guidance.get('approach', '?')}")
        if not self._cached_findings:
            lines.append("  (none)")

        lines.append("=" * w)

        self._last_lines = lines
        print("\033[2J\033[H" + "\n".join(lines), flush=True)

    def _write_file(self) -> None:
        try:
            p = Path(self.session_dir) / "dashboard.txt"
            p.write_text("\n".join(self._last_lines))
        except OSError:
            pass

    def _prev_score(self, action_id: str) -> float | None:
        if len(self.state.score_history) < 2:
            return None
        prev = self.state.score_history[-2]
        return prev.get("scores", {}).get(action_id)
