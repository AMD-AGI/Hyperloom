# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Join run ledgers with the Pulse index and split the token picture into recent and earlier runs.

A run is recent when its session started on or after ``PULSE_RECENT_SINCE``; earlier runs ran
code that has changed since, so findings use the recent ones.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from an_global import cost, pct, price_family
from round_env import recent_since, round_dir

ERAS = ("recent", "earlier")


def load_index(root: Path) -> dict:
    idx = {}
    for line in open(root / "00_enum_global/index_rows.jsonl"):
        r = json.loads(line)
        name = r.get("session_id")
        if not name:
            continue
        prev = idx.get(name)
        if prev is None or (r.get("session_started_at") or "") > (prev.get("session_started_at") or ""):
            idx[name] = r
    return idx


def run_cost(r: dict) -> tuple[float, dict]:
    orch_models = r.get("models", {}).get("orchestration", {})
    sess_model = max(orch_models, key=orch_models.get) if orch_models else "claude"
    per = {}
    for comp, tok in r.get("tokens", {}).items():
        models = r.get("models", {}).get(comp, {})
        model = max(models, key=models.get) if models else None
        if model in (None, "None"):
            model = sess_model
        per[comp] = cost(tok, price_family(model))
    return sum(per.values()), per


def main() -> None:
    root, since = round_dir(), recent_since()
    out = root / "analysis"
    idx = load_index(root)
    runs = [json.loads(l) for l in open(out / "runs_ledger.jsonl")]
    joined = []
    for r in runs:
        ix = idx.get(r["name"], {})
        started = (ix.get("session_started_at") or r.get("created") or "")[:10]
        total, per = run_cost(r)
        joined.append(
            {
                **r,
                "started": started,
                "era": "recent" if started >= since else "earlier",
                "ix_stop": ix.get("stop_reason"),
                "ix_duration_h": (ix.get("duration_seconds") or 0) / 3600,
                "ix_gain": ix.get("gain"),
                "ix_model": ix.get("model_name"),
                "ix_forge": ix.get("forge_enabled"),
                "cluster": ix.get("cluster"),
                "family": ix.get("dispatch_family"),
                "cost": total,
                "cost_by_comp": per,
            }
        )
    with open(out / "runs_joined.jsonl", "w") as fh:
        for j in joined:
            fh.write(json.dumps(j) + "\n")

    lines = []
    for era in ERAS:
        rs = [j for j in joined if j["era"] == era]
        if not rs:
            continue
        tot = sum(j["cost"] for j in rs) or 1
        comp = Counter()
        phase = defaultdict(Counter)
        for j in rs:
            for c, v in j["cost_by_comp"].items():
                comp[c] += v
            for c, ph in j.get("phase_weighted", {}).items():
                for p, w in ph.items():
                    phase[c][p] += w
        lines.append(
            f"\n## era={era} runs={len(rs)} cost=${tot:,.0f} per-run p50=${pct([j['cost'] for j in rs], 0.5):.1f} p90=${pct([j['cost'] for j in rs], 0.9):.1f}"
        )
        lines.append("components: " + ", ".join(f"{c}={100 * v / tot:.1f}%" for c, v in comp.most_common()))
        for c in ("orchestration", "specialist"):
            t = sum(phase[c].values()) or 1
            lines.append(f"  {c} by phase: " + ", ".join(f"{p}={100 * w / t:.0f}%" for p, w in phase[c].most_common(8)))
        stops = Counter()
        stop_cost = Counter()
        for j in rs:
            stops[str(j["ix_stop"])] += 1
            stop_cost[str(j["ix_stop"])] += j["cost"]
        lines.append(
            "  stop_reason (runs, cost share): "
            + ", ".join(f"{s}={n}/{100 * stop_cost[s] / tot:.0f}%" for s, n in stops.most_common(10))
        )
        models = Counter()
        for j in rs:
            for m, n in j.get("models", {}).get("orchestration", {}).items():
                models[m] += 1
        lines.append("  orchestration models (row counts): " + ", ".join(f"{m}={n}" for m, n in models.most_common(6)))

    # scenario candidates: recent runs grouped by model/framework/gpu
    cand = defaultdict(list)
    for j in joined:
        if j["era"] != "recent":
            continue
        key = (j.get("ix_model") or j.get("model_name"), j.get("framework"), j.get("gpu"))
        cand[key].append(j)
    lines.append(
        "\n## recent scenario groups (model, framework, gpu): runs, cost p50, llm calls p50, duration p50 h, stops"
    )
    for key, rs in sorted(cand.items(), key=lambda kv: -len(kv[1])):
        calls = [sum(j.get("calls", {}).values()) for j in rs]
        lines.append(
            f"{str(key):70s} n={len(rs):3d} cost_p50=${pct([j['cost'] for j in rs], 0.5):6.1f} calls_p50={pct(calls, 0.5):5d} "
            f"dur_p50={pct([j['ix_duration_h'] for j in rs], 0.5):4.1f}h stops={dict(Counter(str(j['ix_stop']) for j in rs).most_common(4))}"
        )
    text = "\n".join(lines)
    (out / "era_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
