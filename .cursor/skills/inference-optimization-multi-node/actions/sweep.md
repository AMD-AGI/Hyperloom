# Action: Parameter Sweep

Full ISL/OSL/CONC sweep with the optimized version to map the Pareto frontier.

## Inputs
- Final optimized server config (backends + params + kernel patches)
- `$SCRIPTS_DIR/run_sweep.sh`

## Procedure

**Remote mode:** Run sweeps inside the existing RayJob via Ray Dashboard REST (`POST /api/jobs/`). Do not create extra SaFE workloads for sweep. See [`../modes/REMOTE.md`](../modes/REMOTE.md) "Sweep" section for all options.

### Using the sweep script

```bash
export MODEL="$MODEL_PATH" TP=$TP INFERENCEX_PATH="$INFERENCEX_PATH"
export CONC_VALUES="4 16 64"
export ISL_OSL_CONFIGS="1024:1024 8192:1024 1024:8192"
export RESULT_DIR="/wekafs/inference-optimization/results/sweep_$(date +%Y-%m-%d-%H-%M)"

bash "$SCRIPTS_DIR/run_sweep.sh"
```

**Sweep script features:**
- Single server launch for ALL ISL/OSL configs (request-level params, not server params)
- Default CONC: 3 values (`4 16 64`). Override with `CONC_VALUES="4 8 16 32 64"`
- Adaptive num_prompts: OSL≤1024 → CONC×5, OSL≤4096 → CONC×3, OSL>4096 → CONC×2
- Smart ordering: configs sorted by estimated cost, short first
- Auto-skip extreme combos: `num_prompts × OSL > MAX_OUTPUT_TOKENS` (default 2M)
- Progress: `[N/total +elapsed]` and total wall time

## Accuracy Validation
N/A — sweep uses the same optimized binary, no new changes to validate.

## Outputs
- `results.tsv` with all configs
- Per-config: (CONC, ISL, OSL, output_tput, tput_per_gpu, TPOT, TTFT)
- Pareto frontier identification

## Heuristic Update
N/A — sweep is a measurement action, not an optimization action.

## Failure Handling
- Individual config times out: skip and log, continue sweep
- Server crashes mid-sweep: restart with same config, resume from failed point
