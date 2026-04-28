# Action: `integrate` (STUB)

> Family: **deep_kernel** · guided + marathon · accuracy_risk=0.15.

After kernel-opt KEEP, integrate the winning kernel into the framework
workspace, run `scripts/run_baseline.sh` to validate, and if accuracy
passes, KEEP and snapshot.

## Output schema

```json
{
  "integrated_kernel_id": "...",
  "post_integrate_tput": 5430.0,
  "delta_vs_pre_integrate_pct": 4.1,
  "accuracy_verdict": "keep|revert",
  "patch_files": ["..."]
}
```

## TODO (IMPL-CHECKLIST §4.32 / IR-3 / IR-6)

- [ ] Enforce `process_management.enforce_run_baseline_sh("integrate")`
- [ ] `patch_inductor.py --target-file <f>` ; for tile changes also pass `--best-config`
- [ ] On revert: rollback patches, write event `integrate_reverted`
