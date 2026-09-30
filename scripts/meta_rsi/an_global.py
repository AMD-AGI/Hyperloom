"""Global token picture across all fetched runs: shares by component, model, phase, outcome."""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(os.environ.get("PULSE_ROUND_DIR", "/wekafs/csl/Hyperloom-Sessions/meta_rsi/pulse15d")) / "analysis"
BUNDLES = Path(os.environ.get("PULSE_BUNDLES", "/root/pulse15d/02_bundles"))

# USD per 1M tokens: (input, cache_write, cache_read, output). Assumed list prices, reported as parameters.
PRICES = {
    "opus": (5.0, 6.25, 0.50, 25.0),
    "sonnet": (3.0, 3.75, 0.30, 15.0),
    "gpt": (1.25, 1.25, 0.125, 10.0),
    "gemini": (1.25, 1.25, 0.31, 10.0),
    "glm": (0.0, 0.0, 0.0, 0.0),
}


def price_family(model: str | None, fallback: str = "opus") -> str:
    m = (model or "").lower()
    for fam in ("opus", "sonnet", "gpt", "gemini", "glm"):
        if fam in m:
            return fam
    return fallback


def cost(row: dict, fam: str) -> float:
    p = PRICES[fam]
    return (
        (row.get("input_tokens") or 0) * p[0]
        + (row.get("cache_creation_input_tokens") or 0) * p[1]
        + (row.get("cache_read_input_tokens") or 0) * p[2]
        + (row.get("output_tokens") or 0) * p[3]
    ) / 1e6


def weighted(row: dict) -> float:
    return (
        (row.get("input_tokens") or 0)
        + 1.25 * (row.get("cache_creation_input_tokens") or 0)
        + 0.1 * (row.get("cache_read_input_tokens") or 0)
        + 5 * (row.get("output_tokens") or 0)
    )


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    runs = [json.loads(l) for l in open(OUT / "runs_ledger.jsonl")]
    comp_w, comp_cost, comp_calls = Counter(), Counter(), Counter()
    comp_raw = defaultdict(Counter)
    comp_model = defaultdict(Counter)
    comp_phase = defaultdict(Counter)
    stop_cost, stop_runs = Counter(), Counter()
    per_run_cost, per_hour_cost = [], []
    orch_ctx, orch_out, orch_write, orch_rounds = [], [], [], []
    run_rows = []
    for r in runs:
        # session model for rows without one (specialists run on the session's Claude model)
        orch_models = r.get("models", {}).get("orchestration", {})
        sess_model = max(orch_models, key=orch_models.get) if orch_models else "claude"
        total_cost = 0.0
        for comp, tok in r.get("tokens", {}).items():
            models = r.get("models", {}).get(comp, {})
            model = max(models, key=models.get) if models else None
            if model in (None, "None"):
                model = sess_model
            fam = price_family(model)
            c = cost(tok, fam)
            comp_cost[comp] += c
            comp_w[comp] += tok.get("weighted", 0)
            comp_calls[comp] += r.get("calls", {}).get(comp, 0)
            for f in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"):
                comp_raw[comp][f] += tok.get(f, 0)
            comp_model[comp][model] += c
            for ph, w in r.get("phase_weighted", {}).get(comp, {}).items():
                comp_phase[comp][ph] += w
            total_cost += c
        stop = str(r.get("stop_reason"))
        stop_cost[stop] += total_cost
        stop_runs[stop] += 1
        per_run_cost.append(total_cost)
        hrs = (r.get("elapsed_min") or 0) / 60
        if hrs >= 0.25:
            per_hour_cost.append(total_cost / hrs)
        run_rows.append(
            (total_cost, r["name"], r.get("model_name"), r.get("stage_reached"), stop, r.get("elapsed_min"))
        )

    # orchestration per-call shape, straight from the ledgers
    for r in runs:
        ledger = BUNDLES.glob(f"*/{r['name']}/{r['run_dir']}/reports/trace/llm_calls.jsonl")
        for path in ledger:
            first_ctx = None
            for line in path.open(errors="ignore"):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("component") != "orchestration" or row.get("status") not in (None, "ok"):
                    continue
                ctx = (
                    (row.get("input_tokens") or 0)
                    + (row.get("cache_creation_input_tokens") or 0)
                    + (row.get("cache_read_input_tokens") or 0)
                )
                if not ctx:
                    continue
                orch_ctx.append(ctx)
                orch_out.append(row.get("output_tokens") or 0)
                orch_write.append(row.get("cache_creation_input_tokens") or 0)
                if first_ctx is None:
                    first_ctx = (row.get("cache_creation_input_tokens") or 0) + (row.get("input_tokens") or 0)
                if first_ctx:
                    orch_rounds.append(ctx / first_ctx)

    total_cost = sum(comp_cost.values())
    total_w = sum(comp_w.values())
    lines = []
    lines.append(
        f"runs={len(runs)} archives={len({r['name'] for r in runs})} total_cost_usd={total_cost:,.0f} total_weighted={total_w / 1e9:.2f}B"
    )
    lines.append("\n# component share (cost USD, share, weighted, calls)")
    for comp, c in comp_cost.most_common():
        lines.append(
            f"{comp:14s} ${c:>10,.0f} {100 * c / total_cost:5.1f}%  w={comp_w[comp] / 1e9:6.2f}B calls={comp_calls[comp]:>8,d}  "
            f"raw in={comp_raw[comp]['input_tokens'] / 1e6:,.0f}M cw={comp_raw[comp]['cache_creation_input_tokens'] / 1e6:,.0f}M "
            f"cr={comp_raw[comp]['cache_read_input_tokens'] / 1e6:,.0f}M out={comp_raw[comp]['output_tokens'] / 1e6:,.0f}M"
        )
    lines.append("\n# cost by component x model")
    for comp in comp_cost:
        lines.append(f"{comp:14s} " + ", ".join(f"{m}=${c:,.0f}" for m, c in comp_model[comp].most_common(6)))
    lines.append("\n# weighted share by component x phase (top)")
    for comp in comp_cost:
        tot = sum(comp_phase[comp].values()) or 1
        lines.append(
            f"{comp:14s} " + ", ".join(f"{p}={100 * w / tot:.0f}%" for p, w in comp_phase[comp].most_common(7))
        )
    lines.append("\n# cost by stop_reason (runs, USD, share)")
    for stop, c in stop_cost.most_common(15):
        lines.append(f"{stop:40s} runs={stop_runs[stop]:5d} ${c:>10,.0f} {100 * c / total_cost:5.1f}%")
    lines.append("\n# per-run cost USD p50/p90/p99/max, per-hour p50/p90")
    lines.append(
        f"run: {pct(per_run_cost, 0.5):.1f} / {pct(per_run_cost, 0.9):.1f} / {pct(per_run_cost, 0.99):.1f} / {max(per_run_cost):.1f}; "
        f"hour: {pct(per_hour_cost, 0.5):.1f} / {pct(per_hour_cost, 0.9):.1f}"
    )
    lines.append(
        "\n# orchestration per call: context p50/p90, output p50/p90, cache write p50/p90, context/first-call-prefix p50/p90"
    )
    lines.append(
        f"ctx {pct(orch_ctx, 0.5):,} / {pct(orch_ctx, 0.9):,}; out {pct(orch_out, 0.5):,} / {pct(orch_out, 0.9):,}; "
        f"write {pct(orch_write, 0.5):,} / {pct(orch_write, 0.9):,}; rounds~ {pct(orch_rounds, 0.5):.1f} / {pct(orch_rounds, 0.9):.1f}  (n={len(orch_ctx):,})"
    )
    lines.append("\n# top 12 runs by cost")
    for c, name, model, stage, stop, el in sorted(run_rows, reverse=True)[:12]:
        lines.append(f"${c:8,.0f} {name[:60]:60s} stage={stage} stop={stop} min={el}")
    text = "\n".join(lines)
    (OUT / "global_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
