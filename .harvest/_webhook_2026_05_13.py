"""Build + send the 2026-05-13 batch summary to the Teams webhook."""
import argparse
import os
import sys

ENTRIES = [
    # rank, model, frm, prec, tp, params, baseline, optimized, gain, peak, peak_conc, note, status, task_id, sid
    ("Qwen3-Coder-Next", "vllm", "FP8", 4, 79.7, 4047.11, 4178.16, 3.24, None, None,
     "comm-bound (16.1% compute / 46.9% comm); best=--max-num-batched-tokens 32768",
     "ok", "opt-457df9ae", "ffad2e62"),
    ("Qwen2.5-Coder-32B-Instruct", "sglang", "FP8", 1, 32.8, 2016.17, 2035.69, 0.97, None, None,
     "91.4% vendor kernel; 5 Triton kernels 0% E2E; KV-FP8 -88% HARMFUL",
     "ok", "opt-6ec0eb01", "d15821fc"),
    ("Mistral-7B-v0.1", "sglang", "FP8", 1, 7.2, 4753.18, 4758.11, 0.10, None, None,
     "92.4% vendor kernel; --decode-attention-backend aiter +0.10% (noise)",
     "ok", "opt-7984731a", "33ffbfac"),
    ("NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "vllm", "FP8", 1, 30.0, None, None, None, None, None,
     "abort: NemotronH hybrid Mamba2+Attn+MoE incompatible with vLLM 0.19 + ROCm Triton MLIR",
     "abort", "opt-26572b61", "0e721f40"),
]


def medal(g):
    if g is None:
        return "—"
    if g >= 100:
        m = "🥇🥇🥇🥇🥇 "
    elif g >= 30:
        m = "🥇🥇🥇🥇 "
    elif g >= 10:
        m = "🥇🥇🥇 "
    elif g >= 5:
        m = "🥇🥇 "
    elif g >= 1:
        m = "🥇 "
    elif g > 0:
        m = "🟢 "
    elif g == 0:
        return "➖ 0%"
    else:
        return f"{g:.2f}%"
    return f"{m}+{g:.2f}%"


def icon(s):
    return {"ok": "✅", "abort": "❌"}.get(s, "?")


def build():
    n_ok = sum(1 for e in ENTRIES if e[12] == "ok")
    n_abort = sum(1 for e in ENTRIES if e[12] == "abort")
    lines = []
    lines.append(
        "**Hyperloom optimize-submit batch 2026-05-13 done** (run "
        "[25789927636](https://github.com/AMD-AGI/Hyperloom/actions/runs/25789927636) + "
        "[25795816178](https://github.com/AMD-AGI/Hyperloom/actions/runs/25795816178))  "
    )
    lines.append(
        f"{n_ok}/{len(ENTRIES)} succeeded · {n_abort} aborted (model-stack incompatibility)  "
    )
    lines.append(
        "Note: CI auto-publish hit the known luochen `submitted_at` HTTP-500 bug — "
        "data POSTed manually via .harvest/manual_publish.py (31 records on dashboard now). "
        "Webhook content was empty because artifact_count=0 (data persisted only to sandbox "
        "ephemeral, then GC'd before SaFE could upload). Fixed: prompt prefix now writes "
        "to /hyperloom/users persist dir; publish_results strips submitted_at."
    )
    lines.append("")
    lines.append(
        "| # | Model | Frm | Prec | TP | Params | Baseline tok/s/GPU | Optimized | Gain | Notes |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---|---|")
    for i, (model, frm, prec, tp, params, bl, op, gp, _, _, note, status, _, _) in enumerate(ENTRIES, 1):
        bls = f"{bl:.2f}" if bl is not None else "—"
        ops = f"{op:.2f}" if op is not None else "—"
        gs = medal(gp)
        pb = f"{params:.1f}B"
        lines.append(
            f"| {i} | {model} {icon(status)} | {frm} | {prec} | {tp} | {pb} | "
            f"{bls} | {ops} | {gs} | {note} |"
        )
    lines.append("")
    lines.append(
        "Legend: ✅ full Phase 0-10 + ci_metrics  ·  ❌ abort with partial-report-on-abort  ·  "
        "Real data verified against SaFE artifact API or live-cat'd from sandbox before GC."
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--url", default=os.environ.get("WEBHOOK_URL", ""))
    ap.add_argument("--timeout", type=int, default=15)
    args = ap.parse_args()

    body = build()
    print("─" * 80)
    print(body)
    print("─" * 80)
    print(f"Body length: {len(body.encode('utf-8'))} bytes")

    if args.dry_run:
        return 0
    if not args.send:
        print("\nNo --send flag, not posting. Pass --send + ensure $WEBHOOK_URL or --url is set.")
        return 0
    if not args.url:
        print("ERROR: --send requires --url or $WEBHOOK_URL", file=sys.stderr)
        return 2

    import requests
    import urllib3
    urllib3.disable_warnings()
    resp = requests.post(args.url, json={"text": body}, timeout=args.timeout, verify=False)
    print(f"\nWebhook response: HTTP {resp.status_code}")
    if resp.status_code >= 300:
        print(f"Body: {resp.text[:500]}")
    return 0 if resp.status_code < 300 else 1


if __name__ == "__main__":
    sys.exit(main())
