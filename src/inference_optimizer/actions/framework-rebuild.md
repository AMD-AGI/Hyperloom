# Action: `framework-rebuild` (STUB)

> Family: **long** · marathon-only · accuracy_risk=0.15.

Rebuild sglang / vllm / inductor with a candidate patch (e.g. fused
op, custom dispatcher hint). Highest cost action in the catalog.

## TODO (IMPL-CHECKLIST §4.36 / §4.42)

- [ ] applicable_when: kernel_dispatch_shows_aiter_dominance + cumulative_gain_plateau (DESIGN §12.2 example)
- [ ] Required pre-flight: deep-kernel-analysis must have completed
- [ ] Wire to `scripts/run_baseline.sh` + accuracy_gate after rebuild
