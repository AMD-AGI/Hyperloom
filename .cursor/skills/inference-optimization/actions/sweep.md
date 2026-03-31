# Action: Parameter Sweep

Full ISL/OSL/CONC sweep with the optimized version to map the Pareto frontier.

## Inputs
- Final optimized server config (backends + params + kernel patches)
- `$SCRIPTS_DIR/run_sweep.sh`

## Procedure

### Using the sweep script

```bash
export MODEL="$MODEL_PATH" TP=$TP INFERENCEX_PATH="$INFERENCEX_PATH"
export CONC_VALUES="4 16 64"
export ISL_OSL_CONFIGS="1024:1024 8192:1024 1024:8192"
export RESULT_DIR="/shared_nfs/inference-optimization/results/sweep_$(date +%Y-%m-%d-%H-%M)"

bash "$SCRIPTS_DIR/run_sweep.sh"
```

**Sweep script features:**
- Single server launch for ALL ISL/OSL configs (request-level params, not server params)
- Default CONC: 3 values (`4 16 64`). Override with `CONC_VALUES="4 8 16 32 64"`
- Adaptive num_prompts: OSL≤1024 → CONC×5, OSL≤4096 → CONC×3, OSL>4096 → CONC×2
- Smart ordering: configs sorted by estimated cost, short first
- Auto-skip extreme combos: `num_prompts × OSL > MAX_OUTPUT_TOKENS` (default 2M)
- Progress: `[N/total +elapsed]` and total wall time

### SaFE MCP parallel sweep (faster)

For maximum speed, create one SaFE workload per config:
```
Tool: workload_create
Args: {
    "display_name": "sweep-<model>-<isl><osl>-c<conc>",
    "workspace_id": "control-plane-prod",
    ...
}
```

15 configs × 15 nodes = all parallel, ~10 min total vs ~75 min serial.

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
