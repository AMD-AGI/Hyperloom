# Step 01 Artifacts: Baseline And Live Contracts

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Purpose

This directory is the execution output for `01-establish-baseline-and-contracts.MD`.

The design intent of Step 01 is not to delete code immediately. It is to make the cleanup boundary explicit before deletion starts:

- What end-to-end behavior must survive?
- Which runtime contracts already have code-level enforcement?
- Which cleanup prerequisites are not yet implemented?
- Which compatibility decisions still need product-level confirmation?
- Which tests should become the minimum safety net?
- Which dead-code candidates are safe enough to enter the next step?

## Files

- `01-implementation-status.MD`: What Step 01 expects, what is already implemented, and what is missing.
- `02-live-contracts.MD`: The protected live runtime contracts that future cleanup must not break.
- `03-compatibility-decisions.MD`: Compatibility areas that need explicit keep/drop decisions before code is cut.
- `04-e2e-smoke-scenarios.MD`: End-to-end smoke scenarios used to judge whether cleanup preserved behavior.
- `05-test-baseline-and-tiering.MD`: Test tiers and which current tests belong to each tier conceptually.
- `06-dead-code-initial-candidates.MD`: Initial candidates for Step 02, with why they are candidates and what must be confirmed.

## How To Use This Directory

Before deleting or merging a code path, check:

- Is it listed as a protected live contract?
- Is it only a compatibility path awaiting decision?
- Is it in the initial dead-code candidate list?
- Does any smoke scenario depend on it?
- Is there a contract test that should remain after the cleanup?

If the answer is unclear, do not delete the path yet. Either add it to the compatibility decision list or ask for a product decision.
