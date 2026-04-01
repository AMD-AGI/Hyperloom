# Action: Parameter Sweep

Full ISL/OSL/CONC sweep with the optimized version to map the Pareto frontier.

## Inputs
- Final optimized server config (backends + params + kernel patches)
- `$SCRIPTS_DIR/run_sweep.sh`

## Procedure

> **[CLAW MODE]** Serial sweep on RayJob via `exec_on_gpu`:
> ```bash
> exec_on_gpu "export MODEL='$MODEL' TP=$TP INFERENCEX_PATH='$INFERENCEX_PATH' \
>   CONC_VALUES='4 16 64' \
>   ISL_OSL_CONFIGS='1024:1024 8192:1024 1024:8192' \
>   RESULT_DIR='$RESULT_DIR/sweep' \
>   SGLANG_EXTRA_ARGS='$TUNED_SERVER_ARGS' && \
>   bash $SCRIPTS_DIR/run_sweep.sh"
> ```

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
    "workspace_id": "GEAK_WORKSPACE",
    ...
}
```

15 configs × 15 nodes = all parallel, ~10 min total vs ~75 min serial.

### [CLAW] Option C: Parallel sweep via Ray submit

Submit each config as a separate Ray task (uses the existing RayJob cluster, no extra workload creation):

```python
import ray

ray.init(address=RAY_HEAD_ADDRESS)

@ray.remote(num_gpus=8)
def run_sweep_config(model, tp, conc, isl, osl, result_dir, extra_args):
    import subprocess, os
    env = {
        "MODEL": model, "TP": str(tp), "CONC": str(conc),
        "ISL": str(isl), "OSL": str(osl),
        "RESULT_DIR": result_dir,
        "SGLANG_EXTRA_ARGS": extra_args,
    }
    cmd = f"bash {SCRIPTS_DIR}/run_baseline.sh"
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, env={**os.environ, **env})

configs = [(64, 1024, 1024), (16, 1024, 1024), (4, 1024, 1024),
           (64, 8192, 1024), (16, 8192, 1024)]
futures = [run_sweep_config.remote(MODEL, TP, c, i, o,
    f"/shared_nfs/inference-optimization/results/sweep_{TIMESTAMP}/c{c}_i{i}_o{o}",
    TUNED_SERVER_ARGS) for c, i, o in configs]
results = ray.get(futures)
```

**Note:** Requires enough GPUs in the Ray cluster to run configs in parallel. With a single RayJob (8 GPU), configs run sequentially.

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
