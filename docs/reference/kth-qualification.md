# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Independent Kernel Trust Harness qualification before performance evaluation."""

Hyperloom can attach a fail-closed KTH provider after a Controller patch is
applied and before the performance validator runs. The host, not the
publication, maps a covered operation onto a KTH plan. Missing plan metadata
cannot skip qualification for a covered operation. A publication cannot select
a plan the host did not map, cannot supply a command or adapter, and cannot
weaken the evidence contract.

## Behavior

| Outcome | Hyperloom action |
| --- | --- |
| Operation is not in the host map | Existing apply → measure flow is unchanged |
| `Eligible for performance evaluation` | Performance validator runs once |
| `Blocked` | Benchmark is skipped; the patch is reverted or quarantined |
| `Inconclusive` | Benchmark is skipped; the patch is reverted or quarantined |
| Missing executable, timeout, malformed attestation, digest mismatch, incomplete evidence, provider failure | Fail closed (`needs_review`); benchmark is skipped |

Eligible is permission to measure. It is not a KEEP and is not a claim of
universal correctness.

## Host configuration

Set these variables when an operation family must be qualified. Leave them
unset to keep KTH disabled.

```bash
export HYPERLOOM_KTH_QUALIFY_EXECUTABLE=/path/to/kth-qualify
export HYPERLOOM_KTH_ROOT=/path/to/kernel-trust-harness
export HYPERLOOM_KTH_EXPECTED_SHA="$(git -C "$HYPERLOOM_KTH_ROOT" rev-parse HEAD)"
export HYPERLOOM_KTH_OPERATION_PLANS='{"rms_norm":"host/rms-norm-v1"}'
export HYPERLOOM_KTH_KERNEL_PATH_PLANS='{"kernels/rms_norm.py":"host/rms-norm-v1"}'
export HYPERLOOM_KTH_ALLOWED_PATHS='kernels/rms_norm.py,csrc/include/'
export HYPERLOOM_KTH_TIMEOUT_S=300
```

The host-pinned KTH revision is computed from `HYPERLOOM_KTH_ROOT` or taken
from `HYPERLOOM_KTH_EXPECTED_SHA`. The child process hash is compared with
that pin; a self-reported revision is not trusted on its own.

## Evidence

Hyperloom persists the request, streamed logs, attestation, verdict, subject
digest, detector, patch digest, and pre/post tree identities under the session
`kth_qualification/` directory. Performance numbers are recorded only if the
validator actually ran.
