# Integration Failure Semantics

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Purpose

Define one shared vocabulary for how each integration fails, so the main package consumes failures as task result / policy denial / state warning / report warning instead of each integration inventing its own field.

## Shared Failure Vocabulary

- hard fail: the end-to-end optimization cannot continue.
- soft degrade: the capability is missing but the main loop continues.
- needs manual review: outcome uncertain; a human must decide.
- skip: disabled by operator config or unsupported environment.

## Per-Integration Failure Mode

| Integration | Failure mode | How the main package consumes it |
|---|---|---|
| Kernel request | soft degrade / needs manual review | handler returns a result dict with status; coordinator records kernel attempt + may REVERT |
| Kernel shape/path gate | hard reject of that request | PolicyGate / handler validation denies the request; loop continues |
| Critic | soft degrade | missing context -> `needs_review` / `critic_unavailable` verdict; loop continues |
| Robustness | soft degrade / skip | invalid envelope rejected; multi-node without server -> skip via mock |
| Framework discovery | soft degrade | discover failure increments a retry counter; phase marks done and advances to EXPLORE |
| Framework apply/bench | soft degrade | apply/bench failure -> REVERT; candidate skipped |
| Benchmark (Magpie) | soft degrade / hard fail | parse failure -> failed task result; baseline failure can gate the run |
| InferenceX target | skip | `no_target_gpu_configured` marker; report renders uniformly |
| KB write | soft degrade | best-effort; never aborts the run |
| Multi-node provisioning | hard fail | topology gate / provision error exits before the loop starts |

## Rule

Integrations must surface failure through one of:

- a task result (state + decision),
- a policy denial (with rule + hint),
- a state warning,
- a report/breakdown warning.

Integrations must not invent bespoke per-integration failure fields that the main package has to special-case.

## Cleanup Guidance

- Keep all live subprocess/JSON/artifact bridges and their contract tests.
- Remove only documentation that presents an unimplemented bridge as live.
- Before deleting any seemingly-unused bridge, confirm there is no runtime wiring via string/subprocess/artifact path.
