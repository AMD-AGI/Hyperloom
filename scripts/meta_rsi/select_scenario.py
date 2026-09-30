"""Pick the A/B validation scenario by a fixed score over historical runs.

Hard filters: the runs are other users' (PULSE_EXCLUDE_ROOTS, default the operator's own
/wekafs/csl) and started on or after PULSE_RECENT_SINCE; the model has at least
PULSE_MIN_MODEL_B billion parameters (default 30) and is on local disk; the precision runs on
MI325X (no MXFP4); the run used the synthetic benchmark (the agentx client and corpus are not
installed here); and the scenario has at least two such runs that lasted an hour or more.
Score (0..1): 0.35 LLM activity per hour (normalized to the best candidate), 0.30 phase
coverage, 0.25 reliability (share of runs that reached CLOSE without a baseline failure),
0.10 duration fit (median duration / 6 h, capped at 1).
"""

import json
import os
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ANALYSIS = Path(os.environ.get("PULSE_ROUND_DIR", "/wekafs/csl/Hyperloom-Sessions/meta_rsi/pulse15d")) / "analysis"
MODELS = Path(os.environ.get("PULSE_MODELS_DIR", "/wekafs/models"))
PHASES = ("PRELUDE", "ENABLEMENT", "FRAMEWORK_AGENT", "KERNEL_AGENT", "SWEEP")
MAX_HOURS = 6.0


def local_model_dir(name: str) -> str:
    if not name:
        return ""
    base = name.split("/")[-1]
    for cand in os.listdir(MODELS):
        if cand == base or cand.endswith("-" + base) or cand.split("-", 1)[-1] == base:
            return str(MODELS / cand)
    return ""


def phases_touched(run: dict) -> set:
    seen = set()
    for comp in ("orchestration", "specialist"):
        for ph, w in (run.get("phase_weighted") or {}).get(comp, {}).items():
            if w > 0 and ph in PHASES:
                seen.add(ph)
    return seen


_SIZE = re.compile(r"(\d+(?:\.\d+)?)[bB](?![a-zA-Z])")
_LARGE_FAMILIES = ("deepseek-v4", "deepseek-r1", "minimax-m", "glm-5", "kimi-k")


def model_billions(name: str) -> float:
    """Parameter count in billions from the model name; unsized large families count as large."""
    sizes = [float(s) for s in _SIZE.findall(name)]
    if sizes:
        return max(sizes)
    return 1000.0 if any(f in name.lower() for f in _LARGE_FAMILIES) else 0.0


def run_model(run: dict) -> str:
    """The run's model directory name: from its breakdown, else the path segment above the run."""
    name = str(run.get("model_path") or run.get("model_name") or "").rstrip("/").split("/")[-1]
    if name:
        return name
    parts = [p for p in Path(run["path"]).parts[:-1] if not p.startswith("_")]
    return parts[-1] if parts else ""


def load_candidates() -> dict:
    since = os.environ.get("PULSE_RECENT_SINCE", "2026-09-15")
    exclude = [p for p in os.environ.get("PULSE_EXCLUDE_ROOTS", "/wekafs/csl").split(":") if p]
    min_b = float(os.environ.get("PULSE_MIN_MODEL_B", "30"))
    groups = defaultdict(list)
    for line in open(ANALYSIS / "weka_sep_runs.jsonl"):
        r = json.loads(line)
        if since and str(r.get("created") or r.get("first_ts") or "")[:10] < since:
            continue
        if any(r["path"].startswith(root) for root in exclude):
            continue
        model = run_model(r)
        if model_billions(model) < min_b:
            continue
        r["source"] = "weka-mi325x"
        r["hours"] = (r.get("elapsed_min") or 0) / 60
        groups[(model, r.get("framework"), r.get("tp"))].append(r)
    return groups


def main() -> None:
    groups = load_candidates()
    rows = []
    for (model, fw, tp), runs in groups.items():
        runs = [r for r in runs if r.get("framework") and r["hours"] >= 1.0]
        if len(runs) < 2 or not fw:
            continue
        mdir = local_model_dir(model)
        if not mdir or "mxfp4" in model.lower():
            continue
        activity = st.median(
            sum((r.get("calls") or {}).get(c, 0) for c in ("orchestration", "specialist")) / r["hours"] for r in runs
        )
        coverage = st.median(len(phases_touched(r)) / len(PHASES) for r in runs)
        reliable = sum(
            1 for r in runs if r.get("stage_reached") == "close" and "baseline" not in str(r.get("stop_reason"))
        ) / len(runs)
        fit = min(1.0, st.median(r["hours"] for r in runs) / MAX_HOURS)
        cost_h = st.median(r["cost"] / r["hours"] for r in runs)
        rows.append(
            {
                "model": model,
                "framework": fw,
                "tp": tp,
                "model_dir": mdir,
                "runs": len(runs),
                "activity_per_h": round(activity, 1),
                "coverage": round(coverage, 2),
                "reliability": round(reliable, 2),
                "fit": round(fit, 2),
                "cost_per_h": round(cost_h, 2),
                "examples": [r["path"] for r in runs][:4],
            }
        )
    best_act = max((r["activity_per_h"] for r in rows), default=1) or 1
    for r in rows:
        r["score"] = round(
            0.35 * r["activity_per_h"] / best_act + 0.30 * r["coverage"] + 0.25 * r["reliability"] + 0.10 * r["fit"], 3
        )
    rows.sort(key=lambda r: -r["score"])
    (ANALYSIS / "scenario_scores.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(
            f"{r['score']:.3f} {r['model']:40s} {r['framework']:7s} tp={r['tp']} runs={r['runs']} act/h={r['activity_per_h']:6.1f} "
            f"cov={r['coverage']:.2f} rel={r['reliability']:.2f} fit={r['fit']:.2f} $/h={r['cost_per_h']:.2f}"
        )
    if not rows:
        print("no candidate passed the filters; fall back to the arm A scenario", file=sys.stderr)


if __name__ == "__main__":
    main()
