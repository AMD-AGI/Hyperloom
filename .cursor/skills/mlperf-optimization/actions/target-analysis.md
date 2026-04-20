# Action: Target Analysis

## Overview

Analyzes external performance targets to quantify the optimization gap and adjust
DFS action priorities accordingly.

## Inputs
- `$TARGET_DIR` or target time-to-train numbers (optional)
- Baseline ms/iter and time_to_train from current run

## KB Query

```
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B target comparison" --top-k 5 --compact
```

## Procedure

### Step 1: Determine target source

Targets can come from:
1. **Prior optimization run:** Path to a results directory with `results.tsv`
2. **MLPerf submission results:** Known time-to-train from published submissions
3. **User-provided numbers:** Direct time-to-train or ms/iter target
4. **Hardware comparison:** e.g., "match NVIDIA B200 MLPerf training time"

### Step 2: Parse target results

```python
if TARGET_DIR:
    import csv
    with open(f"{TARGET_DIR}/results.tsv") as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
    best = min(
        [r for r in rows if r["status"] in ("keep", "baseline")],
        key=lambda r: float(r["ms_per_iter"])
    )
    target_ms_per_iter = float(best["ms_per_iter"])
```

### Step 3: Compute gap and urgency

```python
gap_pct = (baseline_ms_per_iter - target_ms_per_iter) / target_ms_per_iter * 100
target_gap_multiplier = 1 + min(gap_pct, 100) / 100
```

### Step 4: Update heuristic priors

```python
if target_techniques:
    for tech in target_techniques:
        if "fusion" in tech.lower():
            priors["fusion-flags"] *= 1.5
        if "gbs" in tech.lower() or "batch" in tech.lower():
            priors["config-selection"] *= 1.5
```

## Outputs
- `target_ms_per_iter` or `target_time_to_train`
- `target_gap_pct`: how far behind we are
- `target_gap_multiplier`: urgency multiplier
- Updated heuristic priors

## Heuristic Update

- Large gap (>30%): multiply all DFS scores by target_gap_multiplier
- Target techniques identified: boost corresponding action scores by 1.5x

## Failure Handling

- If TARGET_DIR not found: skip target analysis, use default gap multiplier of 1.0
