<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Copilot code-review instructions (Hyperloom)

Advisory review. Flag only what static gates can't.

## What to review for

- **Correctness / semantic edge cases**: null/empty/boundary inputs, off-by-one,
  unhandled `None`, silent truncation, a branch that returns the wrong default.
- **Duplication / parallel routes**: a second implementation of something that already
  exists. Point to the existing one and ask to extend it.
- **Failure-hiding error handling**: a `try`/fallback that swallows an error and returns
  a default or `None`, so the caller can't tell success from silent failure. This includes
  a new broad `except Exception` / bare `except` that hides the failure rather than catching
  a specific, expected error (ruff's `E`/`F`/`W` and `pylint --errors-only` catch neither).
- **Concurrency**: missing/incorrect `await`, races on shared state, unawaited tasks,
  blocking calls on the event loop.
- **Unused abstraction**: a flag, strategy, or generic helper added for a single caller,
  or a parameter always passed the same value.
- **Contract & cache invariants**: a change that silently alters an external contract
  or breaks a documented rule.
- **PR focus**: the PR addresses one aspect. If it bundles unrelated changes, say so.
- **Fix-around instead of fix-upstream**: a local workaround for what is really a
  GEAK/Magpie/TraceLens/framework defect.
- **Debt growth**: a new feature flag / env toggle used to route around a design
  problem, or a new suppression without a stated reason.

## What NOT to flag

- Formatting, import order, naming, line length → ruff.
- Cyclomatic complexity, unused variables → ruff/pylint.
- Known-vuln patterns, injection, secrets → CodeQL / gitleaks / bandit.

## How to comment

- **Frame structurally.** Say what shape the code should have and why rather than flagging
 an isolated line. The best comment makes the implementation efficient.
- **Argue from cost and clarity.** Justify each note by what it buys: less duplication,
  one source of truth, etc.
- **Be specific not tedious.** Anchor to a line or the existing code being duplicated,
  but skip style/taste nits a linter would catch.
- **Prefer deletion and reuse.** When you see a leaner form — reuse an existing helper,
  drop a redundant layer, fold a flag away — propose it directly.
- **Architectural correctness.** Does the change respect the system's boundaries and
  control flow? Flag a diff that bypasses an established pipeline, reaches around the owning layer,
  moves a responsibility to the wrong module, or reintroduces a retired/forbidden construct.
