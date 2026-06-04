# Step 07 Execution Status

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Design Intent Understood

`07-clarify-integration-boundaries.MD` asks to give each peripheral integration one clear entry, keep only live bridges, remove unimplemented/retired wording, and unify failure semantics.

## Findings

- Each integration already has a single live entry (see the boundary map). The framework integration is already consistently described as `fa phase-discover` in the runtime code and SKILL.md.
- `framework-agent/README.md` was already corrected in Step 02 (live = `fa phase-discover`; `fa agent` = future).
- Remaining stale wording in the main package: `inference_optimizer/scripts/install.sh` still called the framework path a "bandit arm".

## Implemented In This Step

### Boundary + failure-semantics artifacts

- `01-integration-boundary-map.MD`: single live entry per integration + peripheral owner table.
- `02-failure-semantics.MD`: shared failure vocabulary + per-integration failure consumption table.

### Corrected stale framework wording in install.sh

`install.sh` now describes the `fa` CLI as serving the Coordinator-owned FRAMEWORK_PR phase (`fa phase-discover`) instead of a "bandit arm". Two comment blocks updated; no behavior change.

### Removed legacy bare Codex Critic mode

Removed the `codex_bare` Critic backend selection path.

Updated:

- CLI backend validation and `_build_backends`.
- `--critic-codex-bare` and hidden `--critic-real` flags removed.
- Critic backend documentation now lists only `agent` and `mock`.
- Preflight / operator messages now point at `--critic-mock` as the
  offline bypass.

Kept:

- `CodexBackend` class itself, because kernel and backend unit tests still
  use the generic Codex backend.

## Verification

- Lint: no errors.
- Tests: framework PR, kernel handler, critic/robustness envelope, and protocol-layer tests pass.

## Not Done Yet (decision-gated or higher-risk)

- Robustness legacy mode removal: needs the same legacy-mode decision from Step 01.
- Old Cortex v1 graph KB removal: tracked under Step 06.
- Consolidating duplicated benchmark parameter docstrings: lower priority, can fold into Step 09 comment compression.

## Next Safe Follow-Ups

- Resolve the robustness legacy mode decision, then delete the unused paths and their docs.
- Keep one contract test per live bridge; drop tests that assert subprocess temp-file layout unless it is a real contract.
