# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The dashboard as one self-contained HTML page: inline CSS, no script, nothing fetched, every value escaped."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

from hyperloom_multibranch.notebook.rsi.stranded import short_reason

_STYLE = """
body { font: 14px/1.45 system-ui, sans-serif; margin: 24px; color: #1f2328; }
h1 { font-size: 20px; margin-bottom: 4px; }
h2 { font-size: 16px; margin: 28px 0 8px; }
p.meta, p.empty { color: #59636e; }
table { border-collapse: collapse; }
th, td { border: 1px solid #d0d7de; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f6f8fa; }
td.stopped { color: #59636e; }
"""
_BRANCH = "branch"
_FROM = "from"
_NODE = "node"
_REQUEST = "request"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(item) for item in value)
    return str(value)


def _pct(value: float | None) -> str:
    return "" if value is None else f"{value:+.2f}%"


def _unpriced(tokens: Mapping[str, int]) -> str:
    counts = ", ".join(f"{count:,} {kind.replace('_', ' ')}" for kind, count in tokens.items() if count)
    return f", plus unpriced calls of {counts} tokens" if counts else ""


def _table(title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return f'<section><h2>{escape(title)}</h2><p class="empty">none</p></section>'
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(_text(cell))}</td>" for cell in row) + "</tr>" for row in rows)
    return (
        f"<section><h2>{escape(title)}</h2><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></section>"
    )


def _branches(rows: Sequence[Mapping[str, Any]]) -> str:
    def frontier(row: Mapping[str, Any]) -> str:
        score = row["frontier"]
        return _pct(score["expected_gain_pct"]) if score else ""

    return _table(
        "Branches",
        (_BRANCH, "state", "start", "head", "tests", "commits", "GPU hours", "LLM $", "gain", "expected gain"),
        [
            (
                row["branch"],
                "running" if row["running"] else "stopped",
                row["start_commit"],
                row["head_commit"],
                row["tests"],
                row["commits"],
                row["gpu_hours"],
                row["llm_usd"],
                _pct(row["gain_pct"]),
                frontier(row),
            )
            for row in rows
        ],
    )


def _stranded(rows: Sequence[Mapping[str, Any]]) -> str:
    def commits(row: Mapping[str, Any]) -> str:
        return "; ".join(
            f"{item['commit_id'][:8]}: {short_reason(item['reason']) or item['outcome']}" for item in row["commits"]
        )

    return _table(
        "Ahead of integration, merges failing",
        (_BRANCH, "state", "lead", "head", "integration", "bar", "commits not merged"),
        [
            (
                row["branch"],
                "running" if row["running"] else "stopped",
                _pct(row["lead_pct"]),
                row["head_value"],
                row["integration_value"],
                _pct(row["bar_pct"]),
                commits(row),
            )
            for row in rows
        ],
    )


def _in_flight(in_flight: Mapping[str, Any]) -> str:
    tests = _table(
        "Tests waiting for a result",
        (_REQUEST, "kind", _BRANCH, _FROM, "purpose", "waiting (s)"),
        [
            (row["request_id"], row["kind"], row["branch"], row["parent_commit"], row["purpose"], row["waiting_s"])
            for row in in_flight["tests"]
        ],
    )
    claims = _table(
        "Claims held",
        (_BRANCH, _FROM, "target", "mechanism", _REQUEST, "expires in (s)"),
        [
            (
                row["branch"],
                row["parent_commit"],
                row["target"],
                row["mechanism"],
                row["request_id"],
                row["expires_in_s"],
            )
            for row in in_flight["claims"]
        ],
    )
    return tests + claims


def _exploration(dashboard: Mapping[str, Any]) -> str:
    directions = _table(
        "Directions",
        ("target", "mechanism", "layer", "states tried", "best gain", "on the heads of", "claimed by"),
        [
            (
                row["target"],
                row["mechanism"],
                row["layer"],
                len(row["tried_on"]),
                _pct(row["best_gain_pct"]),
                row["committed_on"],
                row["claimed_by"],
            )
            for row in dashboard["directions"]
        ],
    )
    hypotheses = _table(
        "Hypotheses",
        (_BRANCH, "status", "hypothesis", "evidence", "updated"),
        [
            (row["branch"], row["status"], row["statement"], row["evidence"], row["updated_at"])
            for row in dashboard["hypotheses"]
        ],
    )
    return directions + hypotheses


def _lineage(dashboard: Mapping[str, Any]) -> str:
    commits = _table(
        "Commits",
        ("commit", "parent", _BRANCH, "origin", "merged from"),
        [
            (row["commit_id"], row["parent_commit"], row["branch"], row["origin"], row["merged_from"])
            for row in dashboard["lineage"]["commits"]
        ],
    )
    splits = _table(
        "Splits",
        (_BRANCH, "split from", "at"),
        [(row["branch"], row["from_branch"], row["start_commit"]) for row in dashboard["splits"]],
    )
    return commits + splits


def _infrastructure(dashboard: Mapping[str, Any]) -> str:
    sandboxes = _table(
        "Sandboxes",
        (_NODE, "GPU", "backend", "image", "tests", "GPU hours", "failed attempts"),
        [
            (
                row["node"],
                row["gpu_model"],
                row["backend"],
                row["image_digest"],
                row["tests"],
                row["gpu_seconds"] / 3600.0,
                row["failed_attempts"],
            )
            for row in dashboard["sandboxes"]
        ],
    )
    changes = _table(
        "Environment changes",
        (_NODE, _REQUEST, "at", "changed"),
        [
            (
                row["node"],
                row["request_id"],
                row["at"],
                "; ".join(f"{name} {before} → {after}" for name, (before, after) in row["changed"].items()),
            )
            for row in dashboard["environment_changes"]
        ],
    )
    quarantined = _table(
        "Quarantined",
        (_NODE, "GPUs", "reason"),
        [(row["node"], row["gpu_ids"], row["reason"]) for row in dashboard["quarantined"]],
    )
    queue = dashboard.get("queue")
    queued = _table("Toolbox queue", ("item", "value"), list(queue.items())) if queue is not None else ""
    return sandboxes + changes + quarantined + queued


def render_html(dashboard: Mapping[str, Any]) -> str:
    """The page of a dashboard :func:`build_dashboard` built."""

    session = dashboard.get("session") or {}
    title = f"Session {session.get('session_id') or 'without a profile'}"
    task = ", ".join(
        f"{key} {session[key]}"
        for key in ("model", "hardware", "framework", "tp", "conc", "isl", "osl")
        if session.get(key)
    )
    spend = dashboard["spend"]
    spent = (
        f"spent {_text(spend['gpu_hours'])} GPU hours and ${_text(spend['llm_usd'])} of LLM calls"
        f"{_unpriced(spend['llm_unpriced_tokens'])}"
    )
    body = "".join(
        (
            f"<h1>{escape(title)}</h1>",
            f'<p class="meta">{escape(task)}; {spent}; generated {escape(str(dashboard["generated_at_utc"]))}</p>',
            _branches(dashboard["branches"]),
            _stranded(dashboard.get("stranded") or []),
            _in_flight(dashboard["in_flight"]),
            _exploration(dashboard),
            _lineage(dashboard),
            _infrastructure(dashboard),
        )
    )
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head><body>{body}</body></html>\n"
    )


__all__ = ["render_html"]
