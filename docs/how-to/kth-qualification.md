# Optional Kernel Trust Harness qualification

Hyperloom can call Kernel Trust Harness (KTH) after a controller patch is
applied and before KEEP is recorded. AutoSpec, planning, detectors, verdicts,
and attestation stay in KTH. Hyperloom only builds a candidate envelope and
enforces the KEEP gate.

## Default

KTH is **off**. Existing integration, exploratory GEAK benchmarks, and KEEP
rules are unchanged.

## Enable

```bash
export HYPERLOOM_KTH_ENABLE=1
export HYPERLOOM_KTH_QUALIFY_EXECUTABLE=kth-qualify   # optional
export HYPERLOOM_KTH_TIMEOUT_S=300                    # optional
export HYPERLOOM_KTH_EXPECTED_SHA=<kth-revision>      # optional pin
```

`HYPERLOOM_KTH_ENABLE=1` also selects the adaptive envelope request
(`schema_version` `2.0.0`). Publications that already carry a
`kth_qualification.plan_id` continue to use the pinned-plan request unless
adaptive mode is enabled.

## KEEP invariant

`Blocked` and `Inconclusive` cannot become Hyperloom `KEEP`. Provider failure,
timeout, malformed attestation, or binding mismatch fail closed (`needs_review`)
and revert the patch. `Eligible` may continue through the existing performance
validator.

KTH does not add a second performance search or ranking loop.
