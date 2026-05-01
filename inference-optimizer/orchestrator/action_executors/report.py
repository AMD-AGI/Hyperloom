"""Real ``report`` ActionExecutor — DESIGN v0.6 §16 report action.

Reads the session's SharedState + bus event log and produces:

* ``$SESSION_DIR/report/final.json`` — machine-readable summary (the
  same shape Hyperloom dashboards consume)
* ``$SESSION_DIR/report/final.md``   — human-readable Markdown summary

Returned dict surfaces both paths so the bus event has actionable
references. The generated files are intentionally compact: stop_reason,
baseline + best, cumulative gain, action timeline (counts per kind),
and a top-N highlight list of decisions / verdicts.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..message_bus import MessageBus
from ..shared_state import SharedState
from ...paths import db_path_for
from ...storage.connection import SqliteConnection


log = logging.getLogger(__name__)


def _build_summary_dict(state: SharedState, ev_counts: dict[str, int],
                        highlights: list[dict]) -> dict[str, Any]:
    return {
        "session_id":       state.session_id,
        "model_name":       state.model_name,
        "model_path":       state.model_path,
        "model_class":      state.model_class,
        "stop_reason":      state.stop_reason,
        "baseline_tput":    state.baseline_tput,
        "baseline_accuracy": state.baseline_accuracy,
        "current_best":     state.current_best,
        "cumulative_gain":  state.cumulative_gain,
        "crash_count":      state.crash_count,
        "pruned_families":  state.pruned_families,
        "max_minutes":      state.max_minutes,
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "event_counts_by_topic": ev_counts,
        "highlights": highlights,
    }


def _format_md(summary: dict[str, Any]) -> str:
    cb = summary.get("current_best") or {}
    cb_tput = cb.get("tput") if isinstance(cb, dict) else None
    lines: list[str] = []
    lines.append(f"# Inference Optimizer Report — {summary['session_id']}")
    lines.append("")
    lines.append(f"- **Model**: {summary['model_name']}  (`{summary['model_path']}`)")
    lines.append(f"- **Stop reason**: `{summary['stop_reason']}`")
    lines.append(f"- **Budget**: {summary['max_minutes']} minutes")
    lines.append(f"- **Generated**: {summary['report_generated_at']}")
    lines.append("")
    lines.append("## Throughput")
    lines.append("")
    lines.append(f"- baseline_tput   : `{summary['baseline_tput']:.1f}` tok/s/GPU")
    if cb_tput is not None:
        lines.append(f"- current_best   : `{cb_tput:.1f}` tok/s/GPU "
                      f"(action=`{cb.get('action','?')}`)")
    lines.append(f"- cumulative_gain: `{summary['cumulative_gain']:.2f}%`")
    if cb.get("ttft_mean_ms") is not None:
        lines.append(f"- ttft_mean      : `{cb.get('ttft_mean_ms'):.1f}` ms")
    if cb.get("e2el_mean_ms") is not None:
        lines.append(f"- e2el_mean      : `{cb.get('e2el_mean_ms'):.1f}` ms")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- crash_count    : {summary['crash_count']}")
    lines.append(f"- pruned_families: {summary['pruned_families'] or '(none)'}")
    lines.append("")
    lines.append("## Event counts")
    lines.append("")
    if not summary.get("event_counts_by_topic"):
        lines.append("- (no events recorded)")
    else:
        for topic, n in sorted(summary["event_counts_by_topic"].items(),
                                key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{topic}`: {n}")
    lines.append("")
    lines.append("## Highlights")
    lines.append("")
    if not summary["highlights"]:
        lines.append("(no highlight events captured)")
    else:
        for h in summary["highlights"][:50]:
            lines.append(
                f"- `{h.get('topic','?')}` from `{h.get('from_agent','?')}`: "
                f"{h.get('summary','')}"
            )
    lines.append("")
    return "\n".join(lines)


def _highlight(payload: dict, topic: str, from_agent: str) -> dict[str, Any]:
    """Pick the most useful 1-line summary out of an event's payload."""
    summary = ""
    if topic == "proposal":
        summary = f"action_name={payload.get('action_name')}"
    elif topic == "review_verdict":
        summary = (f"verdict={payload.get('verdict')} "
                   f"reason={(payload.get('reasoning') or '')[:60]}")
    elif topic == "decision":
        summary = (f"kind={payload.get('kind')} "
                   f"action={payload.get('action_name')} "
                   f"task={(payload.get('task_id') or '')[:8]}")
    elif topic == "delegated_result":
        out_tput = (payload.get("result") or {}).get("output_throughput")
        decision = (payload.get("result") or {}).get("decision")
        summary = (f"kind={payload.get('kind')} state={payload.get('state')} "
                   f"tput={out_tput} decision={decision}")
    elif topic == "response":
        summary = (f"kind={payload.get('kind')} "
                   f"status={payload.get('status')}")
    elif topic == "alert":
        summary = (f"sev={payload.get('severity')} {payload.get('summary','')}")
    else:
        summary = json.dumps({k: v for k, v in payload.items()
                                if isinstance(v, (str, int, float, bool))})[:80]
    return {"topic": topic, "from_agent": from_agent, "summary": summary,
            "payload": payload}


# ---------------------------------------------------------------------------
class ReportExecutor:
    """ActionExecutor for the ``report`` action.

    Honours ``ctx.task.params``:
        output_dir:        write final.{md,json} here (default
                           ``$SESSION_DIR/report``)
        highlight_topics:  list of topics to surface in ``highlights``
                           (default: proposal / review_verdict / decision /
                            delegated_result / response / alert)
        max_highlights:    cap the highlights list (default 50)
    """

    DEFAULT_HIGHLIGHT_TOPICS = (
        "proposal", "review_verdict", "decision",
        "delegated_result", "response", "alert",
    )

    def __init__(self, *, max_highlights: int = 50):
        self.max_highlights = int(max_highlights)

    async def __call__(self, ctx) -> dict[str, Any]:
        # Resolve session dir from db path (executor doesn't get session_dir
        # directly — but the BaselineExecutor pattern means we infer from
        # task.params or parent process env). For the CLI flow, the
        # session_dir is the parent of `storage/conductor.db`.
        session_dir = self._resolve_session_dir(ctx)
        if session_dir is None:
            return {"status": "failed",
                    "error": "report_executor: could not resolve session_dir"}

        params = ctx.task.params or {}
        output_dir = Path(params.get("output_dir") or (session_dir / "report"))
        output_dir.mkdir(parents=True, exist_ok=True)
        max_highlights = int(params.get("max_highlights", self.max_highlights))
        highlight_topics = (
            params.get("highlight_topics") or self.DEFAULT_HIGHLIGHT_TOPICS
        )

        state = SharedState.load_or_init(session_dir)

        # Pull bus stats. We open a fresh read-only-ish connection (the
        # Conductor's connection is in the same process; SQLite WAL lets
        # us share without contention).
        db = SqliteConnection(db_path_for(session_dir))
        try:
            bus = MessageBus(db)
            ev_rows = await bus.tail(n=10_000)
            ev_counts = Counter(m.topic for m in ev_rows)
            highlights: list[dict] = []
            for m in ev_rows:
                if m.topic in highlight_topics:
                    highlights.append(_highlight(m.payload or {}, m.topic, m.from_agent))
        finally:
            db.close()

        highlights = highlights[:max_highlights]
        summary = _build_summary_dict(state, dict(ev_counts), highlights)

        json_path = output_dir / "final.json"
        md_path = output_dir / "final.md"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        md_path.write_text(_format_md(summary), encoding="utf-8")

        log.info(
            "report_executor: wrote %s and %s (cumulative_gain=%.2f%%)",
            md_path, json_path, state.cumulative_gain,
        )
        return {
            "status":      "succeeded",
            "session_id":  state.session_id,
            "json_path":   str(json_path),
            "md_path":     str(md_path),
            "summary":     summary,
        }

    def _resolve_session_dir(self, ctx) -> Path | None:
        """Best-effort session_dir resolution.

        Strategy (in order):
        1. ``ctx.extra['session_dir']``     — Conductor injects this for in-process runs
        2. ``task.params['session_dir']``   — explicit wins (e.g. tests)
        3. ``$INFERENCE_OPTIMIZER_SESSION_DIR`` env var — the CLI sets this
           on every ``optimize`` invocation so report tasks dispatched from
           an LLM proposal find the right session even when params is empty.
        4. ``$INFERENCE_OPTIMIZER_SESSION_ROOT`` + most-recent heuristic — last resort
        5. None → executor returns failed status with an error
        """
        extra = getattr(ctx, "extra", None) or {}
        if extra.get("session_dir"):
            return Path(extra["session_dir"])
        params = ctx.task.params or {}
        if params.get("session_dir"):
            return Path(params["session_dir"])
        import os as _os
        explicit = _os.environ.get("INFERENCE_OPTIMIZER_SESSION_DIR")
        if explicit and Path(explicit).exists():
            return Path(explicit)
        root = _os.environ.get("INFERENCE_OPTIMIZER_SESSION_ROOT")
        if root and Path(root).exists():
            sessions = sorted(Path(root).iterdir(), key=lambda p: p.stat().st_mtime,
                              reverse=True)
            for s in sessions:
                if (s / "state.json").exists():
                    return s
        return None


report_executor = ReportExecutor()


__all__ = ["ReportExecutor", "report_executor"]
