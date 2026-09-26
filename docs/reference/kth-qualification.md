---
myst:
    html_meta:
        "description": "Optional Kernel Trust Harness (KTH) qualification gate for Controller patches: where it runs, what Eligible, Blocked and Inconclusive do, how to configure and pin KTH, and where evidence is recorded."
        "keywords": "Hyperloom, Kernel Trust Harness, KTH, qualification, attestation, KEEP, Controller patch, GEAK, AutoSpec, fail closed"
---

# Kernel Trust Harness qualification

Hyperloom can require an independent [Kernel Trust Harness
(KTH)](https://github.com/AMD-AI/kernel-trust-harness) qualification for every
Controller patch before the patch is benchmarked. The gate is off by default;
with it off, Controller integration behaves exactly as it does without KTH.

## Who decides what

| Component | Owns | Does not own |
|---|---|---|
| GEAK / KernelForge Controller | Generating candidates, exploratory checks and micro-benchmarks, the publication | Final qualification |
| Hyperloom | Applying the patch, building the request from the exact applied candidate, validating the attestation, enforcing the verdict, KEEP/REVERT | AutoSpec, operation specifications, oracle or detector selection, thresholds, verdict policy |
| KTH | AutoSpec, OperationSpec resolution, obligation planning, detector and oracle execution, verdict, attestation | Performance search, KEEP |

GEAK observations travel to KTH as evidence labelled with their producer
(`geak_harness`, trust `hypothesized`); they cannot override a KTH verdict.

## Where qualification runs

`integrate_controller_patches` qualifies each publication after its patch is
applied to the integration worktree and before the performance validator runs:

```text
apply (strict, three-way or reconstructed merge)
  → read the exact applied diff against the integration HEAD
  → kth-qualify --request request.json --out attestation.json
  → validate the attestation against what was sent
  → Eligible only: performance validator (E2E benchmark) → KEEP decision
  → before the KEEP commit: the worktree must still carry the qualified bytes
```

The candidate KTH sees is the diff the KEEP would commit, not the published
`change.patch`: a patch rebuilt against earlier KEEPs is qualified on the bytes
it actually lands, against the HEAD it lands on. If the worktree changes
between qualification and commit, the candidate is reverted.

## Verdicts

| Outcome | Integration status | Benchmarked | KEEP |
|---|---|---|---|
| `Eligible for performance evaluation`, validated | continues to the E2E validator | yes, once | only if E2E keeps it |
| `Blocked` | `reverted_kth_blocked` | no | no |
| `Inconclusive` | `reverted_kth_inconclusive` | no | no |
| Provider failure, timeout, missing executable, malformed or incomplete attestation, unsupported schema, binding or digest mismatch, replayed request, candidate changed after qualification, publication trying to steer qualification | `reverted_kth_failed` | no | no |

`Eligible` means only that the candidate may be measured; it is not a
correctness proof. `Blocked` means KTH found an independent violation.
`Inconclusive` means KTH could not establish enough evidence — for example
AutoSpec could not resolve a trusted specification — so the candidate has not
earned a benchmark either. Every non-eligible candidate is reverted through the
same path as an E2E failure and counts toward `reverted_count`.

## What Hyperloom checks

Every attestation must answer the request Hyperloom just sent (a fresh
`request_id` per attempt, in a fresh artifacts directory), use the request's
schema version, carry one of the three verdicts with its matching exit code
(`0`, `2`, `3`), and name a full KTH revision. Beyond that:

- **Reviewed plan** (`schema_version` `1.0.0`): the plan, candidate identity
  and execution binding must match what was sent, the `kth-subject-v1` digest
  is recomputed from the base commit, patch bytes, kernel path and plan, and an
  `Eligible` needs complete mandatory-oracle coverage with executed cases, a
  plan whose paths cover every changed file, and an artifact digest when the
  plan is source-bound.
- **Adaptive envelope** (`schema_version` `2.0.0`): the envelope digest and
  every binding field Hyperloom supplied must match, the `kth-subject-v2`
  digest is recomputed from the binding, and an `Eligible` needs a fully
  verified specification, no AutoSpec uncertainty, and no uncovered or
  unavailable obligation.

These are structural checks on the evidence KTH returns, not a second verdict
policy. KTH's `execution_mode` (`real` or `simulated`; KTH labels CPU runs
`simulated`) is recorded but not judged: whether it suffices is KTH's policy.
An adaptive request carries only the envelope; Hyperloom never asserts that a
reviewed specification applies to a candidate, so operations with a reviewed
specification go through a reviewed plan instead.

A publication whose `publication.json` carries a key that could steer
qualification — a plan, verdict, threshold, tolerance, detector, oracle,
policy, reference, adapter, import or command — is rejected before KTH is
called. Hyperloom never forwards the Controller manifest wholesale; only
scalar exploratory observations (`mean_case_speedup`, `best_wall_ms`,
`iteration`, `correctness_passed`) reach KTH, labelled as `geak_harness`
evidence. A GEAK `correctness_passed` does not change what KTH decides.

## Enable

Install KTH where Hyperloom can execute it, then:

```bash
export HYPERLOOM_KTH_ENABLE=1
export HYPERLOOM_KTH_QUALIFY_EXECUTABLE=/opt/kth/bin/kth-qualify
export HYPERLOOM_KTH_EXPECTED_SHA="$(git -C /opt/kth/src rev-parse HEAD)"
export HYPERLOOM_KTH_PLANS='{"<repo-relative kernel path>": "<KTH host plan ID>"}'
```

`HYPERLOOM_KTH_PLANS` maps a publication's `kernel_path` to a plan registered
in KTH's host-owned registry; the plan IDs are KTH's, and Hyperloom holds no
knowledge of what they check. A kernel with no mapped plan is sent to KTH's
adaptive path as a `KernelCandidateEnvelope`. See
[Environment variables](environment-variables.md#kernel-trust-harness-qualification)
for every setting. An invalid setting stops Controller integration rather than
silently disabling the gate.

## Pin the KTH revision

Each attestation names the KTH revision that produced it; Hyperloom rejects one
without a full hexadecimal revision. Set `HYPERLOOM_KTH_EXPECTED_SHA` to the
revision you deployed and any other revision fails closed. KTH reports the Git
revision of its own checkout, or `KTH_SHA` from its environment when it is
installed without one — set `KTH_SHA` in that case, or every attestation
fails closed.

## Evidence

Each attempt writes `<session>/kth_qualification/<request_id>/`:

| File | Contents |
|---|---|
| `request.json` | The request sent to KTH, including the patch |
| `envelope.json` | The candidate envelope (adaptive requests only) |
| `attestation.json` | KTH's attestation, unmodified |
| `stdout.log`, `stderr.log` | KTH's output, capped at 1 MiB each, with recognizable credentials masked |
| `result.json` | Hyperloom's validated outcome: status, reason, verdict, subject and patch digests, revision, KTH's execution mode, repair feedback, and whether the candidate was admitted to performance evaluation |

The same outcome is recorded under `kth` in the Controller integration result
(`integration/results/NNNN.json`). KTH runs with control-plane credentials
removed from its environment.

## Limits

The gate covers Controller patch integration. What KTH can qualify is KTH's to
state; see its [capability states](https://github.com/AMD-AI/kernel-trust-harness/blob/49154fb80508bb37d008f87f8bfae8759209742f/docs/kth_core_architecture.md#capability-states)
and [limitations](https://github.com/AMD-AI/kernel-trust-harness/blob/49154fb80508bb37d008f87f8bfae8759209742f/LIMITATIONS.md).
In particular, the adaptive path returns `Inconclusive` for any artifact KTH
has no host-registered adapter for, so an unmapped kernel cannot be kept while
the gate is on. The subprocess contract is
[`docs/hyperloom_provider_contract.md`](https://github.com/AMD-AI/kernel-trust-harness/blob/49154fb80508bb37d008f87f8bfae8759209742f/docs/hyperloom_provider_contract.md).
