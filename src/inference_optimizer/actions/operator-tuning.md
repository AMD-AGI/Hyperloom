# Action: `operator-tuning` (STUB)

> Family: **deep_kernel** · marathon-only · accuracy_risk=0.10.

Autotune operator libraries (e.g. AITER block sizes, TKW templates).
Heavier than `kernel-opt` because it walks a larger search space.

## TODO (IMPL-CHECKLIST §4.34)

- [ ] Define search-space schema
- [ ] Per-operator KEEP/REVERT log
- [ ] IR-7: never modify GEAK config
