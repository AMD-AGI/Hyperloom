# Action: `vendor-kernel-config` (STUB)

> Family: **deep_kernel** · marathon-only · accuracy_risk=0.10.

Vendor-supplied kernel libraries (rocBLAS, hipBLASLt, AITER) often expose
config knobs (heuristic policy, library version pin). This action
discovers the right combination through guided trial.

## TODO (IMPL-CHECKLIST §4.35)

- [ ] Inventory vendor knobs in scope (rocBLAS env vars, hipBLASLt heuristic)
- [ ] Per-knob accuracy gate (some knobs change numerics)
