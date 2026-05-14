"""Real ``report`` ActionRunner — DESIGN v0.6 §16 report action.

Reads the session's SharedState + bus event log and produces:

* ``$SESSION_DIR/reports/final.json`` — machine-readable summary (the
  same shape Hyperloom dashboards consume)
* ``$SESSION_DIR/reports/final.md``   — human-readable Markdown summary

Returned dict surfaces both paths so the bus event has actionable
references. The generated files are intentionally compact: stop_reason,
baseline + best, cumulative gain, action timeline (counts per kind),
and a top-N highlight list of decisions / verdicts.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..message_bus import MessageBus
from ..shared_state import SharedState
from ...paths import db_path_for
from ...storage.connection import SqliteConnection


log = logging.getLogger(__name__)


def _build_summary_dict(
    state: SharedState,
    ev_counts: dict[str, int],
    highlights: list[dict],
    *,
    external_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "session_id":       state.session_id,
        "model_name":       state.model_name,
        "model_path":       state.model_path,
        "model_class":      state.model_class,
        "stop_reason":      state.stop_reason,
        "baseline_tput":    state.baseline_tput,
        "baseline_accuracy": state.baseline_accuracy,
        "current_best":     state.current_best,
        "cumulative_gain":  state.cumulative_gain,
        # Phase 3 — separate the per-round-sum gain (kept as
        # ``cumulative_gain`` for back-compat) from the validated
        # cumulative gain, which is what the run actually delivered.
        "cumulative_gain_validated":          state.cumulative_gain_validated,
        "cumulative_gain_validated_ts":       state.cumulative_gain_validated_ts,
        "cumulative_gain_validated_stack_len": state.cumulative_gain_validated_stack_len,
        "optimization_stack_len":             len(state.optimization_stack or []),
        "crash_count":      state.crash_count,
        "pruned_families":  state.pruned_families,
        "max_minutes":      state.max_minutes,
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "event_counts_by_topic": ev_counts,
        "highlights": highlights,
    }
    if external_baseline:
        summary["external_baseline"] = external_baseline
    return summary


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
    lines.append(f"- baseline_tput        : `{summary['baseline_tput']:.1f}` tok/s/GPU")
    if cb_tput is not None:
        lines.append(f"- current_best        : `{cb_tput:.1f}` tok/s/GPU "
                      f"(action=`{cb.get('action','?')}`)")
    # Per-round sum — useful for *seeing* what each step contributed,
    # but doesn't reflect what's actually deliverable end-to-end.
    lines.append(
        f"- cumulative_gain     : `{summary['cumulative_gain']:.2f}%`"
        f"  *(per-round sum — informational only)*"
    )
    # Validated gain — the only honest number. We always print it so
    # the report can never silently quote the (often inflated) raw sum.
    val_gain = summary.get("cumulative_gain_validated", 0.0) or 0.0
    val_ts = summary.get("cumulative_gain_validated_ts") or ""
    val_len = summary.get("cumulative_gain_validated_stack_len", 0) or 0
    stack_len = summary.get("optimization_stack_len", 0) or 0
    if val_ts:
        stale = " ⚠ stack changed since validation" if stack_len > val_len else ""
        lines.append(
            f"- cumulative_gain_val : `{val_gain:.2f}%` "
            f"(validated_at_stack_len={val_len}, ts={val_ts}){stale}"
        )
    else:
        lines.append(
            f"- cumulative_gain_val : `0.00%` "
            f"⚠ never validated — no `validate_stack` action ran in this session"
        )
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

    ext = summary.get("external_baseline")
    if ext:
        lines.extend(_format_external_baseline_section(ext))

    return "\n".join(lines)


def _format_external_baseline_section(ext: dict[str, Any]) -> list[str]:
    """Render the advisory external-baseline section (report-only).

    ``ext`` is the dict materialised by
    :func:`_load_external_baseline` from
    ``$SESSION_DIR/target_analysis/target_baseline.json``. We render
    facts only — no derived gap percentage, no "should reach" wording —
    so this section never reads as an implicit KPI the next run is
    expected to hit. The whole feature is opt-in via
    ``--compare-against-gpu``.
    """
    lines: list[str] = []
    lines.append("## External baseline (InferenceX, advisory)")
    lines.append("")
    status = str(ext.get("status") or "unknown")
    q = ext.get("query") or {}
    lines.append(
        "- Query: "
        f"model=`{q.get('model') or '(unset)'}`  "
        f"gpu=`{q.get('gpu') or '(unset)'}`  "
        f"framework=`{q.get('framework') or '(any)'}`  "
        f"precision=`{q.get('precision') or '(any)'}`  "
        f"ISL/OSL=`{q.get('isl') or '(any)'}/{q.get('osl') or '(any)'}`"
    )
    lines.append(f"- Fetched at: {ext.get('fetched_at') or '(unknown)'}")
    lines.append(f"- Status: `{status}` (rows matched: {ext.get('row_count', 0)})")
    warning = ext.get("warning") or ""
    if warning:
        lines.append(f"- Warning: {warning}")

    best = ext.get("best")
    if status == "ok" and isinstance(best, dict):
        lines.append("")
        lines.append(
            f"- Reference best per-GPU throughput: "
            f"**{float(best.get('tput_per_gpu', 0.0)):.1f}** tok/s/GPU "
            f"at concurrency {best.get('conc')}, decode TP {best.get('decode_tp')}"
        )
        ttft = float(best.get("mean_ttft_ms") or 0.0)
        tpot = float(best.get("mean_tpot_ms") or 0.0)
        e2el = float(best.get("mean_e2el_ms") or 0.0)
        if ttft:
            lines.append(f"- Reference mean TTFT: {ttft:.1f} ms")
        if tpot:
            lines.append(f"- Reference mean TPOT: {tpot:.3f} ms")
        if e2el:
            lines.append(f"- Reference mean E2E latency: {e2el:.1f} ms")
        if best.get("date"):
            lines.append(f"- Reference run date: {best.get('date')}")
    else:
        lines.append("")
        lines.append(
            "- No reference best available — orchestrator was not affected by "
            "this section."
        )

    lines.append("")
    lines.append(
        "> Advisory only. This block does not feed Objective, scoring, or any "
        "agent prompt; it is shown here purely for post-mortem comparison."
    )
    lines.append("")
    return lines


def _load_external_baseline(session_dir: Path) -> dict[str, Any] | None:
    """Best-effort load of ``target_analysis/target_baseline.json``.

    Returns the parsed dict on success or ``None`` if the file does not
    exist / is unreadable. Errors are swallowed: a corrupt baseline JSON
    must never break report generation.
    """
    try:
        from ...session_paths import target_baseline_json
        path = target_baseline_json(session_dir)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "report_executor: failed to load external baseline from %s: %s",
            session_dir, exc,
        )
        return None


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
    """ActionRunner for the ``report`` action.

    Honours ``ctx.task.params``:
        output_dir:        write final.{md,json} here (default
                           ``$SESSION_DIR/reports``)
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
        # Resolve session dir from db path (runner doesn't get session_dir
        # directly — but the BaselineExecutor pattern means we infer from
        # task.params or parent process env). For the CLI flow, the
        # session_dir is the parent of `storage/coordinator.db`.
        session_dir = self._resolve_session_dir(ctx)
        if session_dir is None:
            return {"status": "failed",
                    "error": "report_executor: could not resolve session_dir"}

        params = ctx.task.params or {}
        from ...session_paths import reports_dir
        output_dir = Path(params.get("output_dir") or reports_dir(session_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        max_highlights = int(params.get("max_highlights", self.max_highlights))
        highlight_topics = (
            params.get("highlight_topics") or self.DEFAULT_HIGHLIGHT_TOPICS
        )

        state = SharedState.load_or_init(session_dir)

        # Pull bus stats. We open a fresh read-only-ish connection (the
        # Coordinator's connection is in the same process; SQLite WAL lets
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
        external_baseline = _load_external_baseline(session_dir)
        summary = _build_summary_dict(
            state,
            dict(ev_counts),
            highlights,
            external_baseline=external_baseline,
        )

        json_path = output_dir / "final.json"
        md_path = output_dir / "final.md"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        md_path.write_text(_format_md(summary), encoding="utf-8")

        log.info(
            "report_executor: wrote %s and %s "
            "(cumulative_gain=%.2f%% per_round_sum / %.2f%% validated)",
            md_path, json_path,
            state.cumulative_gain, state.cumulative_gain_validated,
        )
        publish_result = self._maybe_publish_results(session_dir, state)
        return {
            "status":      "succeeded",
            "session_id":  state.session_id,
            "json_path":   str(json_path),
            "md_path":     str(md_path),
            "summary":     summary,
            "publish_result": publish_result,
        }

    def _resolve_session_dir(self, ctx) -> Path | None:
        """Best-effort session_dir resolution.

        Strategy (in order):
        1. ``ctx.extra['session_dir']``     — Coordinator injects this for in-process runs
        2. ``task.params['session_dir']``   — explicit wins (e.g. tests)
        3. :func:`paths.session_dir`        — honours ``$USER_DATA_PATH``
           and otherwise returns ``/workspace/hyperloom``. Returns the
           path only if it exists and contains ``state.json``.
        4. None → runner returns failed status with an error
        """
        extra = getattr(ctx, "extra", None) or {}
        if extra.get("session_dir"):
            return Path(extra["session_dir"])
        params = ctx.task.params or {}
        if params.get("session_dir"):
            return Path(params["session_dir"])
        from ...paths import session_dir as _sd
        candidate = _sd()
        if candidate.exists() and (candidate / "state.json").exists():
            return candidate
        return None

    def _maybe_publish_results(self, session_dir: Path, state: SharedState) -> dict[str, Any]:
        """Best-effort publish hook for code-driven optimizer runs.

        Prompt/skill-driven Web runs use actions/report.md directly. This hook
        covers runs that execute the Python ReportExecutor. It is opt-in unless
        the results service URL is explicitly configured.
        """
        service_url = os.environ.get("HYPERLOOM_RESULTS_SERVICE_URL", "")
        auto_publish = os.environ.get("HYPERLOOM_RESULTS_AUTO_PUBLISH", "").lower()
        if not service_url and auto_publish not in {"1", "true", "yes"}:
            return {"enabled": False, "reason": "HYPERLOOM_RESULTS_SERVICE_URL not set"}

        repo_root = Path(__file__).resolve().parents[3]
        helper = repo_root / "ci" / "publish_artifacts.py"
        if not helper.exists():
            return {"enabled": False, "reason": f"{helper} not found"}

        cmd = [
            "python3",
            str(helper),
            "--task-dir",
            str(session_dir),
            "--out-dir",
            str(session_dir / "normalized"),
            "--model",
            state.model_name or "unknown",
            "--display-name",
            state.session_id or "hyperloom-report",
        ]
        if service_url:
            cmd.extend(["--url", service_url])

        try:
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            return {
                "enabled": True,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
        except Exception as e:
            log.warning("report_executor: result publish failed: %s", e)
            return {"enabled": True, "error": str(e)}


report_executor = ReportExecutor()


__all__ = ["ReportExecutor", "report_executor"]
