# integrate — adopt GEAK candidate kernels

**Family**: `deep_kernel` · **Cost**: ~12‑25 min · **Risk**: 15% accuracy

Mandatory follow-up to a successful `kernel_opt` (IR-3). Patches the
workspace with the winning kernel(s), rebuilds the extension, and
re‑runs `scripts/run_baseline.sh` to confirm the gain.

Lane discipline (acquires three):

1. `workspace_mutation` — rebuild
2. `server_lifecycle` — kill+launch
3. `benchmark_lane` — measure

Outputs:

- `update_state` `current_tput=...` if accuracy_gate KEEPs
- artifact `results/integrate/<ts>/metrics.json`

If `accuracy_gate.compare_to_baseline → REVERT`, this action MUST roll
the workspace back to the prior commit before exiting.
