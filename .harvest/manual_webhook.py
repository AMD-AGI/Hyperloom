"""Send the 2026-05-12 batch=10 results to the Teams webhook.

Usage:
  python .harvest/manual_webhook.py --dry-run                # print body, no HTTP
  WEBHOOK_URL=$(gh secret view WEBHOOK_URL ...)  # populate the same env
  python .harvest/manual_webhook.py --send                   # POST to $WEBHOOK_URL

Important: we deliberately do NOT hardcode any URL here. The single source of
truth for the destination is the same WEBHOOK_URL value the CI workflow uses
(see optimize-submit.yml::Send webhook step). The script reads it from the
$WEBHOOK_URL environment variable; pass --url ONLY to override for one-off
testing (and only when you really know which channel that URL belongs to).

Body shape mirrors what the workflow sends: header + the rendered
ci_summary.md table with medal emojis, sorted by gain% descending.
"""

from __future__ import annotations

import argparse
import json
import sys

RUN_URL = "https://github.com/AMD-AGI/Hyperloom/actions/runs/25749785697"

# 6 with detailed numbers + 1 succeeded but user did not paste data + 3 fails.
# Order: gain% desc; failures last.
ENTRIES = [
    # rank, model, fmwk, prec, tp, params_b, baseline, optimized, gain%, note, status
    (1, "Qwen2.5-Coder-14B-Instruct-AWQ", "vllm",   "INT4", 1, 14.8,  559.06, 1971.41, 252.5,  "GEN tok/s; total 1209.29→4264.32 +252.7%; peak 5341.15 @ CONC=256", "ok"),
    (2, "Qwen3-14B-AWQ",                  "vllm",   "INT4", 1, 14.8, 1420.89, 1552.48,   9.26, "peak 4821.10 @ ISL=512 OSL=512 CONC=256",                              "ok"),
    (3, "DeepSeek-V2-Lite-Chat",          "sglang", "FP8",  1, 15.7, 5434.27, 5697.57,   4.85, "peak 11984.04 @ CONC=256",                                             "ok"),
    (4, "DeepSeek-Coder-V2-Lite-Instruct","sglang", "FP8",  1, 15.7, 4267.08, 4464.96,   4.64, "peak 10969.47 @ CONC=256",                                             "ok"),
    (5, "Qwen2.5-Coder-14B-Instruct",     "sglang", "FP8",  1, 14.8, 2818.36, 2840.95,   0.80, "full Phase 0-10; 7 kernels optimized; peak 4119 @ CONC=256",           "ok"),
    (6, "Mistral-7B-v0.1",                "sglang", "FP8",  1,  7.2, 4717.80, 4751.69,   0.72, "validated cumulative gain +0.30%; --decode-attention-backend aiter",   "ok"),
    (7, "Qwen2.5-Coder-32B-Instruct-AWQ", "vllm",   "INT4", 1, 32.8, None,    None,      None, "GHA marked Succeeded but data not yet pasted in chat",                 "pending"),
    (8, "Qwen/Qwen3-Coder-Next",          "—",      "—",    None, 159.0, None, None,    None, "159GB — SaFE model download Failed (server reported \"Unknown error during download\")", "download_fail"),
    (9, "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "—", "—", None, 63.0, None, None, None, "63GB — SaFE model download Failed",                                     "download_fail"),
    (10, "Qwen/Qwen2.5-Coder-32B-Instruct","—",     "—",    None, 65.0, None, None,     None, "65GB — SaFE model download Failed",                                     "download_fail"),
]


def _gain_emoji(g: float | None) -> str:
    if g is None:
        return "—"
    medals = ""
    if g >= 100:
        medals = "🥇🥇🥇🥇🥇 "
    elif g >= 30:
        medals = "🥇🥇🥇🥇 "
    elif g >= 10:
        medals = "🥇🥇🥇 "
    elif g >= 5:
        medals = "🥇🥇 "
    elif g >= 1:
        medals = "🥇 "
    elif g > 0:
        medals = "🟢 "
    elif g == 0:
        return "➖ 0%"
    return f"{medals}+{g:.2f}%" if g >= 0 else f"{g:.2f}%"


def _status_icon(status: str) -> str:
    return {
        "ok": "✅",
        "pending": "🟡",
        "download_fail": "❌",
    }.get(status, "?")


def build_body() -> str:
    n_ok = sum(1 for e in ENTRIES if e[10] == "ok")
    n_pending = sum(1 for e in ENTRIES if e[10] == "pending")
    n_fail = sum(1 for e in ENTRIES if e[10] == "download_fail")

    lines: list[str] = []
    lines.append(f"**Hyperloom optimize-submit batch=10 [run 25749785697] done**  ")
    lines.append(
        f"{n_ok}/10 succeeded · {n_pending} pending data · {n_fail} download-fail "
        f"(SaFE model-register Failed at phase 0)  "
    )
    lines.append(f"GHA: {RUN_URL}  ")
    lines.append(
        "Webhook context: previous batch=10 was 0/10 (skill abort + missing report); "
        "this run added the V2 SKILL.md prompt prefix, 8h wait_ready timeout, and the SSE-settle "
        "post-poll fix. Net delta: **+7 succeeded** vs last batch."
    )
    lines.append("")
    lines.append("| # | Model | Frm | Prec | TP | Params | Baseline tok/s/GPU | Optimized tok/s/GPU | Gain | Notes |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---|---|")

    for rank, model, frm, prec, tp, params_b, bl, op, gp, note, status in ENTRIES:
        icon = _status_icon(status)
        bls = f"{bl:.2f}" if bl is not None else "—"
        ops = f"{op:.2f}" if op is not None else "—"
        gain_s = _gain_emoji(gp)
        tp_s = str(tp) if tp is not None else "—"
        pb_s = f"{params_b:.1f}B" if params_b else "—"
        lines.append(
            f"| {rank} | {model} {icon} | {frm} | {prec} | {tp_s} | {pb_s} | {bls} | {ops} | {gain_s} | {note} |"
        )

    lines.append("")
    lines.append("Legend: ✅ full Phase 0-10 + ci_metrics.json · 🟡 GHA succeeded but agent data not in chat yet · ❌ SaFE model download Failed (not a skill issue)")
    lines.append("")
    lines.append("Known follow-ups:")
    lines.append(
        "1. luochen results-service `/api/import` returns 500 — postgres pod is in liveness-probe crashloop "
        "(restart count 8, `pg_isready ... timed out after 1s`). 26-record payload is staged locally at "
        "`.harvest/manual_publish_payload.json` and will be republished once luochen restores the service."
    )
    lines.append(
        "2. The 3 model-download failures consistently come back as \"Unknown error during download\" within seconds, "
        "so the wait_ready 8h timeout we added is not the bottleneck — SaFE download Job itself is failing on these repos."
    )
    return "\n".join(lines)


def main() -> int:
    import os
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--send", action="store_true")
    ap.add_argument(
        "--url",
        default=os.environ.get("WEBHOOK_URL", ""),
        help="Webhook URL. Default: $WEBHOOK_URL (same env CI uses).",
    )
    ap.add_argument("--write", default="")
    ap.add_argument("--timeout", type=int, default=15)
    args = ap.parse_args()

    body = build_body()
    print("─" * 80)
    print(body)
    print("─" * 80)
    print(f"Body length: {len(body.encode('utf-8'))} bytes")

    if args.write:
        from pathlib import Path
        Path(args.write).write_text(body, encoding="utf-8")
        print(f"Wrote body to {args.write}")

    if args.dry_run:
        return 0
    if not args.send:
        print("\nNo --send flag, not posting.")
        return 0
    if not args.url:
        print("ERROR: --send requires --url", file=sys.stderr)
        return 2

    import requests, urllib3
    urllib3.disable_warnings()
    resp = requests.post(args.url, json={"text": body}, timeout=args.timeout, verify=False)
    print(f"\nWebhook response: HTTP {resp.status_code}")
    if resp.status_code >= 300:
        print(f"Body: {resp.text[:500]}")
    else:
        print("✓ Webhook accepted")
    return 0 if resp.status_code < 300 else 1


if __name__ == "__main__":
    sys.exit(main())
