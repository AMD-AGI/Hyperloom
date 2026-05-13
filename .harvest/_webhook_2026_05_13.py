"""Build + send the 2026-05-13 batch summary to the Teams / PowerAutomate webhook.

The destination is a Power Automate Workflows webhook (the Microsoft
replacement for the legacy Teams Incoming Webhooks they retired in Jan 2025).
Workflows REQUIRE an Adaptive Card or Message Card payload — plain
``{"text": "..."}`` is silently accepted (HTTP 202) but the flow run fails
with "The message must include either an adaptive card or message card
formatted payload".

We emit an Adaptive Card v1.5 with a real ``Table`` body so Teams renders
proper columns instead of one wall of monospaced text.
"""
import argparse
import os
import sys

# Same column order as build_summary.render_markdown:
#   # | Model | Frm | Prec | TP | Params | Baseline tok/s/GPU | Optimized tok/s/GPU | Gain | InfX | vs InfX
ENTRIES = [
    # (model_name, frm, prec, tp, params_b, baseline, optimized, gain, infx, vs_infx, final_status)
    ("Qwen3-Coder-Next",                   "vllm",   "FP8",  4, 79.7,  4047.11, 4178.16,  3.24, None, None, "Succeeded"),
    ("Qwen2.5-Coder-32B-Instruct",         "sglang", "FP8",  1, 32.8,  2016.17, 2035.69,  0.97, None, None, "Succeeded"),
    ("Mistral-7B-v0.1",                    "sglang", "FP8",  1,  7.2,  4753.18, 4758.11,  0.10, None, None, "Succeeded"),
    ("NVIDIA-Nemotron-3-Nano-30B-A3B-BF16","vllm",   "FP8",  1, 30.0,   None,    None,    None, None, None, "Failed"),
]


def fmt_num(v):
    if v is None:
        return "—"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def fmt_pct(p):
    return f"{p:+.2f}%" if p is not None else "—"


def gain_medal(g):
    """Same medal scheme as build_summary.gain_medal."""
    if g is None:
        return ""
    if g >= 50:
        return "🥇🥇🥇🥇"
    if g >= 20:
        return "🥇🥇🥇"
    if g >= 5:
        return "🥇🥇"
    if g >= 1:
        return "🥇"
    if g > 0:
        return "🟢"
    if g == 0:
        return "➖"
    return ""


def status_icon(final_status, baseline, optimized):
    if final_status == "Succeeded" and baseline and optimized:
        return "✅"
    if final_status == "Failed":
        return "❌"
    return "🟡"


def _cell(text, weight=None, color=None):
    """Single TableCell with one TextBlock inside."""
    tb = {"type": "TextBlock", "text": str(text), "wrap": True}
    if weight:
        tb["weight"] = weight
    if color:
        tb["color"] = color
    return {"type": "TableCell", "items": [tb]}


def _gain_color(g):
    if g is None:
        return "Default"
    if g >= 5:
        return "Good"
    if g > 0:
        return "Accent"
    if g == 0:
        return "Warning"
    return "Attention"


def build():
    """Return the Adaptive Card payload (dict) ready to POST as ``json=``."""
    n = len(ENTRIES)
    succeeded = sum(1 for e in ENTRIES if e[10] == "Succeeded")
    with_gain = sum(1 for e in ENTRIES if (e[7] or 0) > 0 and e[5] and e[6])
    beat_infx = 0

    rows = sorted(
        ENTRIES,
        key=lambda r: (0, -(r[7] if r[7] is not None else 0)) if r[7] is not None else (1, 0.0),
    )

    batch_tag = "2026-05-13"
    gha_run1 = "https://github.com/AMD-AGI/Hyperloom/actions/runs/25789927636"
    gha_run2 = "https://github.com/AMD-AGI/Hyperloom/actions/runs/25795816178"
    dashboard = "http://core42.example-internal-host.invalid/hyperloom-results/dashboard"

    # Header rows (TextBlock so we can use bold + markdown links)
    body_blocks = [
        {
            "type": "TextBlock",
            "text": f"**Hyperloom optimize-submit [{batch_tag}] done**",
            "size": "Medium",
            "weight": "Bolder",
            "wrap": True,
        },
        {
            "type": "TextBlock",
            "text": f"{succeeded}/{n} succeeded · {with_gain} with gain · "
                    f"{beat_infx} compare against the baseline",
            "wrap": True,
            "isSubtle": True,
        },
        {
            "type": "TextBlock",
            "text": (
                f"GHA: [{gha_run1.split('/')[-1]}]({gha_run1}) + "
                f"[{gha_run2.split('/')[-1]}]({gha_run2})  ·  "
                f"[Dashboard]({dashboard})"
            ),
            "wrap": True,
        },
    ]

    # Build the Adaptive Card Table.
    #
    # Card width: set msteams.width=Full on the AdaptiveCard so Teams
    # gives us the conversation's full width (~1200px+) instead of the
    # compact-default ~640px that wraps every numeric cell into
    # vertical stacks. Drop the # column since the row order is the
    # gain sort order anyway and the index adds no info.
    columns = [
        {"width": 6},   # Model      ~30%
        {"width": 3},   # Stack      ~15%
        {"width": 2},   # Params     ~10%
        {"width": 3},   # Baseline   ~15%
        {"width": 3},   # Optimized  ~15%
        {"width": 3},   # Gain       ~15%
    ]
    # total ratio = 20
    table_rows = [
        {
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
        }
    ]
    for idx, (model, frm, prec, tp, params_b, bl, op, gp, _ifx, _vs, fs) in enumerate(rows, 1):
        icon = status_icon(fs, bl, op)
        params_s = f"{params_b:.1f}B" if params_b else "—"
        bl_s = fmt_num(bl)
        op_s = fmt_num(op)
        medal = gain_medal(gp)
        if gp is None:
            gain_text = "—"
        elif gp == 0:
            gain_text = f"{medal} 0%".strip()
        else:
            gain_text = f"{medal} {fmt_pct(gp)}".strip()
        stack_s = f"{frm} {prec} TP={tp}"
        table_rows.append({
            "type": "TableRow",
            "cells": [
                _cell(f"{model} {icon}"),
                _cell(stack_s),
                _cell(params_s),
                _cell(bl_s),
                _cell(op_s),
                _cell(gain_text, color=_gain_color(gp)),
            ],
        })

    body_blocks.append({
        "type": "Table",
        "firstRowAsHeader": True,
        "showGridLines": True,
        "columns": columns,
        "rows": table_rows,
    })

    card = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.5",
        # msteams.width=Full asks Teams to render the card at the full
        # conversation width (~1200px desktop) instead of the compact
        # ~640px default. Without this the numeric columns wrap into
        # vertical stacks like "Pa\nra\nms" and "4047.\n11" which is
        # unreadable.
        "msteams": {"width": "Full"},
        "body": body_blocks,
    }
    return {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": card,
        }],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--url", default=os.environ.get("WEBHOOK_URL", ""))
    ap.add_argument("--timeout", type=int, default=15)
    args = ap.parse_args()

    import json as _json
    payload = build()
    payload_json = _json.dumps(payload, indent=2)
    print("─" * 80)
    print(payload_json[:2500])
    if len(payload_json) > 2500:
        print(f"... (truncated, full payload {len(payload_json)} chars)")
    print("─" * 80)
    print(f"Payload size: {len(payload_json.encode('utf-8'))} bytes "
          f"(Power Automate Workflows limit 256 KB)")

    if args.dry_run:
        return 0
    if not args.send:
        print("\nNo --send flag. Pass --send + ensure $WEBHOOK_URL is set.")
        return 0
    if not args.url:
        print("ERROR: --send requires --url or $WEBHOOK_URL", file=sys.stderr)
        return 2

    import requests
    import urllib3
    urllib3.disable_warnings()
    # NOTE: Power Automate Workflows webhook expects the Adaptive Card
    # wrapper {"type": "message", "attachments": [...]} — NOT {"text": ...}.
    resp = requests.post(args.url, json=payload, timeout=args.timeout, verify=False)
    print(f"\nWebhook response: HTTP {resp.status_code}")
    if resp.text:
        print(f"Body: {resp.text[:500]}")
    return 0 if resp.status_code < 300 else 1


if __name__ == "__main__":
    sys.exit(main())
