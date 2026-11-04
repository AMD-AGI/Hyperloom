"""Roofline progress tracker — table + ASCII bar chart over campaign history."""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Optional

ROOFLINE_TARGET = 0.85
_GFX950_MAF_FLOPS_FP4 = 10.0e15 * 0.5   # 50% of vendor peak (MAF fraction)
_GFX950_MAF_BW        = 8.0e12  * 0.8   # 80% of peak HBM


def load_keeps(lineage_path: str) -> list[dict]:
    path = Path(lineage_path)
    if not path.exists(): return []
    keeps = []
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("decision") in ("KEEP", "STOP"):
            keeps.append(entry)
    keeps.sort(key=lambda e: e.get("ts", 0))
    return keeps


def render_table(keeps: list[dict], baseline_us: Optional[float] = None) -> str:
    if not keeps: return "No KEEPs yet.\n"
    lines = [f"{'Gen':>4}  {'Tier':>10}  {'µs':>9}  {'Δ% base':>8}  {'Roofline':>9}  Label",
             "-" * 72]
    first_us = keeps[0].get("weakest_us", 0) or (baseline_us or 0)
    for k in keeps:
        gen = k.get("generation", "?"); tier = k.get("tier", "?")
        us  = k.get("weakest_us", 0); rf   = k.get("roofline_frac")
        label = k.get("reason", k.get("label", ""))[:40]
        ref = baseline_us or first_us or us
        delta = ((ref - us) / ref * 100.0) if ref and us else 0.0
        rf_str = f"{rf*100:.1f}%" if rf else "—"
        lines.append(f"{gen:>4}  {tier:>10}  {us:>9.3f}  {delta:>+7.1f}%  {rf_str:>9}  {label}")
    best_k = min(keeps, key=lambda k: k.get("weakest_us", 9e9))
    best_rf = best_k.get("roofline_frac", 0) or 0
    best_us_val = best_k.get("weakest_us", 0)
    ref = baseline_us or (keeps[0].get("weakest_us") or best_us_val)
    total_gain = ((ref - best_us_val) / ref * 100.0) if ref and best_us_val else 0.0
    lines.append("-" * 72)
    lines.append(f"BEST: {best_us_val:.3f}µs  total={total_gain:+.1f}%  "
                 f"roofline={best_rf*100:.1f}%  target={ROOFLINE_TARGET*100:.0f}%  "
                 f"remaining={max(0, ROOFLINE_TARGET - best_rf)*100:.1f}pp")
    return "\n".join(lines)


def render_ascii_chart(keeps: list[dict], width: int = 60) -> str:
    if not keeps: return ""
    lines = [f"\nRoofline progress (target: {ROOFLINE_TARGET*100:.0f}% MAF)\n"]
    target_col = int(ROOFLINE_TARGET * width)
    for k in keeps:
        rf = k.get("roofline_frac") or 0; gen = k.get("generation", "?"); us = k.get("weakest_us", 0)
        filled = int(rf * width)
        bar = list("█" * filled + "░" * (width - filled))
        if target_col < len(bar): bar[target_col] = "|"
        lines.append(f"  gen{gen:>3} [{''.join(bar)}] {rf*100:5.1f}%  {us:.1f}µs")
    lines.append(f"\n         {'':>{target_col}}^ {ROOFLINE_TARGET*100:.0f}% target")
    return "\n".join(lines)


def render_summary(keeps: list[dict], baseline_us: Optional[float] = None,
                   shape_label: str = "") -> str:
    header = f"=== Roofline Trajectory{': ' + shape_label if shape_label else ''} ==="
    return f"{header}\n\n{render_table(keeps, baseline_us)}\n{render_ascii_chart(keeps)}\n"


def update_tracker(run_dir: str, new_entry: dict) -> str:
    traj_path = Path(run_dir) / "roofline_trajectory.jsonl"
    with open(traj_path, "a") as f:
        f.write(json.dumps(new_entry) + "\n")
    keeps = load_keeps(str(Path(run_dir) / "lineage.jsonl"))
    return render_summary(keeps)


def load_trajectory(run_dir: str) -> str:
    keeps = load_keeps(str(Path(run_dir) / "lineage.jsonl"))
    return render_summary(keeps) if keeps else "No KEEPs yet.\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--baseline-us", type=float, default=None)
    parser.add_argument("--shape", default="")
    args = parser.parse_args()
    print(render_summary(load_keeps(args.lineage), args.baseline_us, args.shape))


if __name__ == "__main__":
    main()
