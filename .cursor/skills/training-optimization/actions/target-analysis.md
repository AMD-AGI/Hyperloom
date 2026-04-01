# Action: Target Analysis

## Inputs
- `$TARGET_DIR` or target ms/iter numbers (optional)
- Baseline ms/iter from current run

## Procedure

### Step 1: Determine target source

Targets can come from:
1. **Prior optimization run:** Path to a results directory with `results.tsv`
2. **CI/CD baseline:** Known ms/iter from nightly CI runs
3. **User-provided numbers:** Direct ms/iter target
4. **Hardware comparison:** e.g., "match NVIDIA A100/H100 training speed"

### Step 2: Parse target results

```python
if TARGET_DIR:
    import csv
    with open(f"{TARGET_DIR}/results.tsv") as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
    # Find the best (lowest) ms/iter from kept attempts
    best = min(
        [r for r in rows if r["status"] == "keep"],
        key=lambda r: float(r["ms_per_iter"])
    )
    target_ms_per_iter = float(best["ms_per_iter"])
```

### Step 3: Compute gap and urgency

```python
gap_pct = (baseline_ms_per_iter - target_ms_per_iter) / target_ms_per_iter * 100
target_gap_multiplier = 1 + min(gap_pct, 100) / 100

# Extract techniques used in target
if TARGET_DIR:
    kept_actions = [r for r in rows if r["status"] == "keep"]
    target_techniques = [r["description"] for r in kept_actions]
```

### Step 4: Identify transferable techniques

For each technique in the target:
- **Config overrides:** Directly applicable if same model/framework
- **Code patches:** May need adaptation for different Primus version
- **Kernel optimizations:** Hardware-specific, may not transfer across GPU types
- **Parallelism configs:** Transfer if same GPU count

### Step 5: Update heuristic priors

```python
# If target used fusion flags successfully, boost fusion-flags score
for tech in target_techniques:
    if "fusion" in tech.lower() or "permute" in tech.lower():
        priors["fusion-flags"] *= 1.5
    if "parallelism" in tech.lower() or "tp=" in tech.lower():
        priors["parallelism"] *= 1.5
```

## Outputs
- `target_ms_per_iter`: target reference
- `target_gap_pct`: how far behind we are
- `target_gap_multiplier`: urgency multiplier for heuristic
- `target_techniques`: list of techniques used in target (for KB seeding)
- Updated heuristic priors

## Failure Handling
- If TARGET_DIR not found: skip target analysis, use default gap multiplier of 1.0
- If no kept attempts in target: use baseline from target instead
