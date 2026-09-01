<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Copilot code-review instructions (Hyperloom)

Advisory review. Flag **only** what static gates can't. Ruff (hard gate on E/F/W),
CodeQL (security-and-quality), bandit/pylint/mypy already cover style, broad-except,
complexity, unused names, and suppressions — **do not** duplicate them. If a finding
would be caught by ruff or CodeQL, stay silent.

## What to review for

- **Semantic edge cases**: null/empty/boundary inputs, off-by-one, unhandled `None`,
  silent truncation, a branch that returns the wrong default.
- **Concurrency**: incorrect `async`/`await`, missing `await`, shared-state races,
  blocking calls on the event loop, unawaited tasks.
- **PR focus**: the diff does one thing. If it bundles unrelated changes, say so.
- **Fix-around instead of fix-upstream**: a local workaround for what is really a
  GEAK/Magpie/TraceLens/framework defect. Ask for the upstream ticket.
- **Debt growth**: a *new* feature flag / env toggle used to route around a design
  problem, or a new suppression without a stated reason.

## What NOT to flag

- Formatting, import order, naming, line length → ruff.
- Broad `except`, cyclomatic complexity, unused variables → ruff/pylint.
- Known-vuln patterns, injection, secrets → CodeQL / gitleaks / bandit.

## How to comment

Be specific and terse: name the exact line and the concrete failure input. Prefer one
high-signal comment over several speculative ones. If nothing meets the bar above,
leave no review comments.

Architecture-specific invariants live in path-scoped files under
[`.github/instructions/`](instructions/) and fire only on the relevant files.
