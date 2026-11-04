"""Roofline progress tracker — records every KEEP and renders trajectory.

Reads from campaign lineage.jsonl and renders a text table + ASCII chart:
  baseline → ... → current → target (85% MAF)

Usage:
    python3 -m arbor.policy.evolve_asm.roofline_tracker \
        --lineage experiments/.../lineage.jsonl --baseline-us 91.4
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Optional

ROOFLINE_TARGET = 0.85


def load_keeps(lineage_path: str) -> list[dict]:
    p = Path(lineage_path)
    if not p.exists():
        return []
    keeps = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get('decision') in ('KEEP', 'STOP', 'BASELINE'):
            keeps.append(e)
    keeps.sort(key=lambda e: e.get('ts', 0))
    return keeps


def render_table(keeps: list[dict], baseline_us: Optional[float] = None) -> str:
    if not keeps:
        return "No KEEPs yet.\n"
    ref = baseline_us or keeps[0].get('weakest_us') or 1.0
    lines = [f"{'Gen':>4}  {'Tier':>10}  {'µs':>9}  {'Δ%':>7}  {'Roofline':>9}  Label"]
    lines.append('-' * 68)
    for k in keeps:
        gen = k.get('generation', '?')
        tier = k.get('tier', '?')
        us = k.get('weakest_us') or k.get('metric', 0)
        rf = k.get('roofline_frac')
        label = (k.get('reason') or k.get('label') or '')[:36]
        delta = ((ref - us) / ref * 100.0) if ref and us else 0.0
        rf_s = f"{rf*100:.1f}%" if rf else '—'
        lines.append(f"{gen:>4}  {tier:>10}  {us:>9.3f}  {delta:>+6.1f}%  {rf_s:>9}  {label}")
    best = min(keeps, key=lambda k: k.get('weakest_us') or k.get('metric') or 9e9)
    best_us = best.get('weakest_us') or best.get('metric') or 0
    best_rf = best.get('roofline_frac') or 0
    gain = ((ref - best_us) / ref * 100.0) if ref and best_us else 0.0
    lines.append('-' * 68)
    lines.append(f"BEST: {best_us:.3f}µs  gain={gain:+.1f}%  "
                 f"roofline={best_rf*100:.1f}%  target={ROOFLINE_TARGET*100:.0f}%  "
                 f"remaining={max(0,ROOFLINE_TARGET-best_rf)*100:.1f}pp")
    return '\n'.join(lines)


def render_ascii_chart(keeps: list[dict], width: int = 60) -> str:
    if not keeps:
        return ''
    target_col = int(ROOFLINE_TARGET * width)
    lines = [f"\nRoofline progress (target: {ROOFLINE_TARGET*100:.0f}% MAF)\n"]
    for k in keeps:
        rf = k.get('roofline_frac') or 0
        gen = k.get('generation', '?')
        us = k.get('weakest_us') or k.get('metric', 0)
        filled = int(rf * width)
        bar = list('█' * filled + '░' * (width - filled))
        if target_col < len(bar):
            bar[target_col] = '|'
        lines.append(f"  gen{gen:>3} [{''.join(bar)}] {rf*100:5.1f}%  {us:.1f}µs")
    lines.append(f"\n         {'':{target_col}}^ {ROOFLINE_TARGET*100:.0f}% target")
    return '\n'.join(lines)


def render_summary(keeps: list[dict], baseline_us: Optional[float] = None,
                   label: str = '') -> str:
    header = f"=== Roofline Trajectory{': ' + label if label else ''} ==="
    return f"{header}\n\n{render_table(keeps, baseline_us)}\n{render_ascii_chart(keeps)}\n"


def update_tracker(run_dir: str, entry: dict) -> str:
    """Append entry to lineage, return current summary string."""
    path = Path(run_dir) / 'lineage.jsonl'
    with open(path, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    keeps = load_keeps(str(path))
    return render_summary(keeps)


def load_trajectory(run_dir: str) -> str:
    keeps = load_keeps(str(Path(run_dir) / 'lineage.jsonl'))
    return render_summary(keeps) if keeps else "No KEEPs yet.\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--lineage', required=True)
    p.add_argument('--baseline-us', type=float, default=None)
    p.add_argument('--shape', default='')
    args = p.parse_args()
    keeps = load_keeps(args.lineage)
    print(render_summary(keeps, args.baseline_us, args.shape))

if __name__ == '__main__':
    main()
