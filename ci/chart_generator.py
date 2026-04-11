#!/usr/bin/env python3
"""Generate CI result charts and send via webhook."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("chart-generator")


def generate_chart(summary: dict | str, output_path: str) -> str:
    """Generate a grouped bar chart comparing CI results vs InferenceX.

    Args:
        summary: Either a dict (ci_summary.json contents) or a path to the file.
        output_path: Where to save the PNG.

    Returns:
        The output_path on success.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if isinstance(summary, (str, Path)):
        with open(summary) as f:
            summary = json.load(f)

    models = [m for m in summary.get("models", []) if m.get("status") == "completed"]
    if not models:
        log.warning("No completed models to chart")
        return output_path

    labels = []
    baseline_vals = []
    optimized_vals = []
    inferencex_vals = []

    for m in models:
        name = m.get("model", "?")
        if len(name) > 20:
            name = name.split("-")[-1] if "-" in name else name[:20]
        prec = m.get("precision", "")
        labels.append(f"{name}\n({prec})")

        baseline_vals.append(m.get("baseline_tok_per_gpu") or 0)
        optimized_vals.append(m.get("optimized_tok_per_gpu") or 0)
        inferencex_vals.append(m.get("inferenceX_tok_per_gpu") or 0)

    x = np.arange(len(labels))
    n_bars = 3
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.5), 5))

    bars_bl = ax.bar(x - width, baseline_vals, width, label="CI Baseline", color="#4A90D9")
    bars_opt = ax.bar(x, optimized_vals, width, label="CI Optimized", color="#2ECC71")
    bars_ifx = ax.bar(x + width, inferencex_vals, width, label="InferenceX", color="#95A5A6",
                      edgecolor="#7F8C8D", linewidth=1.2, linestyle="--")

    def _annotate(bars):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.0f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

    _annotate(bars_bl)
    _annotate(bars_opt)
    _annotate(bars_ifx)

    trigger = summary.get("trigger", "manual")
    ts = summary.get("timestamp", "")
    date_str = ts[:10] if ts else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ax.set_title(f"Hyperloom CI — {trigger} — {date_str}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Output tok/s/GPU", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Chart saved to %s", output_path)
    return output_path


def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def send_chart_webhook(webhook_url: str, image_path: str, summary: dict):
    """Send chart as base64 image in a Teams Adaptive Card."""
    import requests

    b64 = _image_to_base64(image_path)
    data_uri = f"data:image/png;base64,{b64}"

    trigger = summary.get("trigger", "manual")
    stats = summary.get("stats", {})
    completed = stats.get("completed", 0)
    total = stats.get("total", 0)
    avg_gain = stats.get("avg_gain_pct")
    gain_text = f"+{avg_gain:.1f}%" if avg_gain is not None else "N/A"

    card = {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {
                        "type": "TextBlock",
                        "weight": "Bolder",
                        "size": "Large",
                        "text": f"Hyperloom CI Results — {trigger}",
                    },
                    {
                        "type": "TextBlock",
                        "isSubtle": True,
                        "spacing": "None",
                        "text": f"{completed}/{total} models completed | Avg gain: {gain_text}",
                    },
                    {
                        "type": "Image",
                        "url": data_uri,
                        "size": "stretch",
                        "altText": "CI Results Chart",
                    },
                ],
            },
        }],
    }

    try:
        resp = requests.post(webhook_url, json=card, timeout=30)
        if resp.status_code < 300:
            log.info("Chart webhook sent successfully")
        else:
            log.warning("Chart webhook returned %d: %s", resp.status_code, resp.text[:200])
            _send_fallback(webhook_url, summary)
    except Exception as e:
        log.warning("Chart webhook failed: %s", e)
        _send_fallback(webhook_url, summary)


def _send_fallback(webhook_url: str, summary: dict):
    """Plain text fallback if image webhook fails."""
    import requests

    models = summary.get("models", [])
    lines = ["Hyperloom CI Results:"]
    for m in models:
        bl = m.get("baseline_tok_per_gpu")
        opt = m.get("optimized_tok_per_gpu")
        ifx = m.get("inferenceX_tok_per_gpu")
        gain = m.get("gain_pct")
        lines.append(
            f"  {m.get('model','?')} ({m.get('precision','')}): "
            f"BL={bl:.0f} OPT={opt:.0f} IFX={ifx:.0f} "
            f"Gain={gain:+.1f}%" if all(v is not None for v in [bl, opt, ifx, gain])
            else f"  {m.get('model','?')}: {m.get('status','?')}"
        )
    try:
        requests.post(webhook_url, json={"text": "\n".join(lines)}, timeout=10)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Generate CI chart and optionally send webhook")
    parser.add_argument("summary", help="Path to ci_summary.json")
    parser.add_argument("-o", "--output", default="ci_chart.png", help="Output PNG path")
    parser.add_argument("--webhook", default=None, help="Webhook URL (or env var name)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    with open(args.summary) as f:
        summary = json.load(f)

    generate_chart(summary, args.output)

    webhook = args.webhook
    if webhook and not webhook.startswith("http"):
        webhook = os.environ.get(webhook)
    if webhook:
        send_chart_webhook(webhook, args.output, summary)


if __name__ == "__main__":
    main()
