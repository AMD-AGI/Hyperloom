# Initial Dead-Code Candidates

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Purpose

This document lists initial candidates for Step 02.

These are not all final deletion decisions. They are candidates identified during Step 01 by comparing the design intent against current code structure, runtime wiring, tests, and docs.

Deletion rule:

- Delete immediately only when the candidate clearly does not protect a live contract.
- If it may protect old sessions, dashboards, operators, or subprocess contracts, move it to compatibility decision instead of deleting immediately.

## High-Confidence Candidates

### Multi-node backup files

Observed candidates:

- `inference_optimizer/multi_node/cli.py.bak.1778853677`
- `inference_optimizer/multi_node/scripts/bootstrap.sh.bak.1778853855`

Why candidate:

- Backup artifacts are not live runtime contracts.
- They should not be imported or packaged as active implementation.

Recommended action:

- Delete in Step 02 after confirming no packaging rule includes them.

### Stale coverage omit entries

Observed candidate category:

- Coverage omit entries that reference deleted or retired modules.

Why candidate:

- Coverage config should describe current source, not historical files.
- Stale omit entries increase noise during cleanup.

Recommended action:

- Verify each omitted file exists.
- Remove entries for nonexistent files.
- Keep omit entries for live subprocess-heavy files only if still intentional.

### Stale documentation references to removed actions

Observed candidate category:

- Docs or prompts that describe removed actions as active runtime behavior.

Known retired concepts:

- `setup`
- `classify`
- `backends`
- `params`
- `validate_stack`
- `select_kernels`

Why candidate:

- Prompt or docs that mention retired positive paths can cause LLM agents and maintainers to follow non-live workflows.

Recommended action:

- Remove or rewrite as short migration notes.
- Do not keep long historical explanations in live docs or prompts.

### Unimplemented future feature descriptions

Observed candidate category:

- Framework-agent 5th-role or `fa agent` descriptions that are not wired into current `inference_optimizer` runtime.

Why candidate:

- Future feature descriptions are not live contracts.
- Presenting them as implemented increases engineering confusion.

Recommended action:

- Rewrite as future/non-live note, or delete from operational docs.

## Medium-Confidence Candidates

### Historical examples under `inference_optimizer/examples`

Observed candidates:

- P0/P1/P2 demo scripts under `inference_optimizer/examples`.

Why candidate:

- They appear historical and are not the production CLI entry.
- They may lock old architecture concepts.

Decision needed:

- Are these user-facing docs or disposable historical demos?

Recommended action:

- If not user-facing, delete.
- If retained, move to archived examples and remove from cleanup-critical path.

### Old Cortex v1 graph KB residuals

Observed candidate category:

- Old Cortex graph-style constants or methods retained for grep stability or legacy naming.

Why candidate:

- Current design appears centered on RecipeKB / recipe snapshot.
- Old graph lifecycle adds conceptual complexity.

Decision needed:

- Is old Cortex v1 graph KB still reachable in production?

Recommended action:

- If no, delete from live runtime.
- If migration is still needed, isolate it as an importer or compatibility tool.

### Robustness legacy mode surfaces

Observed candidate category:

- Legacy robustness runtime paths and docs.

Why candidate:

- Current optimizer integration uses the reactor/subprocess tick path.
- Legacy mode increases maintenance and test surface.

Decision needed:

- Do any operators still launch robustness legacy mode?

Recommended action:

- If no, delete or archive.
- Keep only the envelope/reactor contract needed by `inference_optimizer`.

## Compatibility Candidates, Not Immediate Deletions

### Old session breakdown readers

Why not immediate:

- They may be needed by dashboards or offline analysis of archived sessions.

Recommended action:

- Keep until downstream consumers confirm old session support can be dropped.
- Separate archived reader logic from live producer logic.

### Old state migrations

Why not immediate:

- They may be needed for resume.

Recommended action:

- Keep until old session resume support is explicitly dropped.
- Stop adding new historical branches.

### `extra_sglang_args` read alias

Why not immediate:

- Old sub-agent payloads and session artifacts may still contain it.

Recommended action:

- Keep read-only for now.
- Continue blocking new writer sites.
- Remove after compatibility decision.

### Legacy kernel-agent artifact layout readers

Why not immediate:

- Breakdown may still need to collect old kernel-agent runs.

Recommended action:

- Keep if archived reporting is retained.
- Delete if old breakdown support is dropped.

## Retired Feature Positive Paths To Search And Remove

Search targets for Step 02:

- positive execution of `setup`;
- positive execution of `classify`;
- positive execution of old `backends` action;
- positive execution of old `params` action;
- positive execution of `validate_stack`;
- positive execution of `select_kernels`;
- imports or references to old scoring module as live behavior;
- prompt instructions that encourage these actions.

Keep only:

- migration readers,
- archived breakdown aliases,
- explicit rejection tests,
- short upgrade notes.

## Do Not Delete Yet

Do not delete these until decisions are made:

- old session resume migration;
- old session breakdown reader;
- payload alias helper;
- robustness envelope mirror;
- critic/robustness subprocess bridge;
- kernel handler safety gates;
- framework PR apply/revert flow;
- session path helpers;
- downstream breakdown schema.

## Step 02 Starting Point

The safest first deletion batch should be:

1. backup files;
2. stale coverage omit entries for nonexistent modules;
3. stale docs that claim removed actions are live;
4. unimplemented future-feature wording;
5. old examples, only after confirming they are not user-facing.

This order cuts noise without risking live runtime contracts.
