#!/usr/bin/env python3
"""Send Hyperloom CI summary to Teams / Power Automate webhook.

Power Automate accepts large JSON bodies (256 KB) but the downstream Teams
card renderer fails when a single Adaptive Card contains too many Table rows.
Keep cards small: 10 model rows per card, each rendered as a bordered
Adaptive Card Table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
import urllib3


def _fmt_num(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):+.2f}%"
    except (TypeError, ValueError):
        return str(v)


def _short_model(model: str | None) -> str:
    return (model or "?").split("/")[-1]


def _params(row: dict[str, Any]) -> str:
    value = row.get("params_b")
    if isinstance(value, (int, float)):
        if value >= 100:
            return f"{value:.0f}B"
        return f"{value:.1f}B"
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb](?:\b|[-_])", row.get("model") or "")
    return f"{match.group(1)}B" if match else "-"


def _gain_prefix(gain: Any) -> str:
    if gain is None:
        return ""
    try:
        g = float(gain)
    except (TypeError, ValueError):
        return ""
    if g >= 20:
        return "*** "
    if g >= 5:
        return "** "
    if g >= 1:
        return "* "
    if g > 0:
        return "+ "
    if g == 0:
        return "0 "
    return ""


def _gain_color(gain: Any) -> str:
    if gain is None:
        return "Default"
    try:
        g = float(gain)
    except (TypeError, ValueError):
        return "Default"
    if g >= 5:
        return "Good"
    if g > 0:
        return "Accent"
    if g == 0:
        return "Warning"
    return "Attention"


def _text(text: Any, weight: str | None = None, color: str | None = None) -> dict[str, Any]:
    block = {"type": "TextBlock", "text": str(text), "wrap": True, "size": "Small"}
    if weight:
        block["weight"] = weight
    if color:
        block["color"] = color
    return block


def _cell(text: Any, weight: str | None = None, color: str | None = None) -> dict[str, Any]:
    return {"type": "TableCell", "items": [_text(text, weight, color)]}


def _sort_key(row: dict[str, Any]) -> tuple[int, float]:
    gain = row.get("gain_pct")
    if gain is None:
        return (1, 0.0)
    try:
        return (0, -float(gain))
    except (TypeError, ValueError):
        return (1, 0.0)


def _build_payload(
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    part: int,
    total_parts: int,
    dashboard_url: str,
) -> dict[str, Any]:
    n = len(all_rows)
    succeeded = sum(1 for r in all_rows if r.get("final_status") == "Succeeded")
    with_metrics = sum(
        1 for r in all_rows
        if r.get("baseline_tok_per_gpu") and r.get("optimized_tok_per_gpu")
    )
    with_gain = sum(
        1 for r in all_rows
        if (r.get("gain_pct") or 0) > 0
        and r.get("baseline_tok_per_gpu")
        and r.get("optimized_tok_per_gpu")
    )
    run_id = os.environ.get("GITHUB_RUN_ID", "?")
    repo = os.environ.get("GITHUB_REPOSITORY", "AMD-AGI/Hyperloom")
    run_url = f"https://github.com/{repo}/actions/runs/{run_id}"

    body: list[dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": f"**Hyperloom optimize-submit [{run_id}] part {part}/{total_parts}**",
            "size": "Medium",
            "weight": "Bolder",
            "wrap": True,
        },
        {
            "type": "TextBlock",
            "text": (
                f"{succeeded}/{n} SaFE succeeded · {with_metrics} with metrics · "
                f"{with_gain} with positive gain"
            ),
            "wrap": True,
            "isSubtle": True,
        },
        {
            "type": "TextBlock",
            "text": f"[GHA]({run_url})" + (f" · [Dashboard]({dashboard_url})" if dashboard_url else ""),
            "wrap": True,
        },
    ]

    table_rows = [{
        "type": "TableRow",
        "style": "accent",
        "cells": [
            _cell("Model", "Bolder"),
            _cell("Stack", "Bolder"),
            _cell("Params", "Bolder"),
            _cell("Baseline tok/s/GPU", "Bolder"),
            _cell("Optimized tok/s/GPU", "Bolder"),
            _cell("Gain", "Bolder"),
        ],
    }]
    for row in rows:
        gain = row.get("gain_pct")
        gain_text = "-" if gain is None else (_gain_prefix(gain) + ("0%" if gain == 0 else _fmt_pct(gain))).strip()
        stack = f"{row.get('framework') or '-'} {row.get('precision') or '-'} TP={row.get('tp') or '-'}"
        table_rows.append({
            "type": "TableRow",
            "cells": [
                _cell(_short_model(row.get("model"))),
                _cell(stack),
                _cell(_params(row)),
                _cell(_fmt_num(row.get("baseline_tok_per_gpu"))),
                _cell(_fmt_num(row.get("optimized_tok_per_gpu"))),
                _cell(gain_text, color=_gain_color(gain)),
            ],
        })

    body.append({
        "type": "Table",
        "firstRowAsHeader": True,
        "showGridLines": True,
        "columns": [{"width": 6}, {"width": 3}, {"width": 2}, {"width": 3}, {"width": 3}, {"width": 3}],
        "rows": table_rows,
    })

    # Footer: per-model perf-leaderboard publish count, on the LAST card only
    # (avoid spam-repeating the same number across every chunk for big batches).
    # Counts are computed by the workflow's "Count perf-leaderboard publish
    # status" step (walks task-artifacts-merged/**/perf_publish_marker.txt)
    # and injected via the PERF_PUBLISH_OK / PERF_PUBLISH_TOTAL env vars.
    if part == total_parts:
        try:
            perf_ok    = int(os.environ.get("PERF_PUBLISH_OK")    or 0)
            perf_total = int(os.environ.get("PERF_PUBLISH_TOTAL") or 0)
        except ValueError:
            perf_ok = perf_total = 0
        if perf_total > 0:
            failed = perf_total - perf_ok
            tail = (
                f"perf-leaderboard publish: **{perf_ok}/{perf_total}** sent"
                + (f" · {failed} failed" if failed else "")
            )
            body.append({
                "type": "TextBlock",
                "text": tail,
                "wrap": True,
                "isSubtle": True,
                "spacing": "Small",
            })

    return {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "type": "AdaptiveCard",
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "version": "1.5",
                "msteams": {"width": "Full"},
                "body": body,
            },
        }],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="ci_summary.json")
    parser.add_argument("--url", default=os.environ.get("WEBHOOK_URL", ""))
    parser.add_argument("--dashboard-url", default=os.environ.get("DASHBOARD_URL", ""))
    parser.add_argument("--rows-per-card", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    if not args.url:
        print("WEBHOOK_URL is not set; skipping webhook")
        return 0
    payload = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    rows = sorted(payload.get("rows") or payload.get("models") or [], key=_sort_key)
    if not rows:
        print("summary has no rows; skipping webhook")
        return 0

    urllib3.disable_warnings()
    chunks = [rows[i:i + args.rows_per_card] for i in range(0, len(rows), args.rows_per_card)]
    for idx, chunk in enumerate(chunks, 1):
        body = _build_payload(chunk, rows, idx, len(chunks), args.dashboard_url)
        body_bytes = len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
        print(f"Webhook part {idx}/{len(chunks)}: {len(chunk)} rows, {body_bytes} bytes")
        response = requests.post(args.url, json=body, timeout=args.timeout, verify=False)
        print(f"Webhook part {idx}: HTTP {response.status_code}")
        if response.status_code >= 300:
            print(response.text[:500])
            return 1
        if idx < len(chunks):
            time.sleep(3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
