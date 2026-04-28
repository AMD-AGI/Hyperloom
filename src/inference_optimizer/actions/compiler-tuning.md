# Action: `compiler-tuning` (STUB)

> Family: **long** · marathon-only · accuracy_risk=0.05.

Inductor / triton / CK compile-flag tuning. Parameter space includes
`max_autotune`, `epilogue_fusion`, register-budget knobs.

## TODO (IMPL-CHECKLIST §4.38 / IR-6)

- [ ] Always pass `--target-file` to `patch_inductor.py`
- [ ] On block-size or warp-count change, ALSO pass `--best-config`
