<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Copilot code-review instructions (Hyperloom)

Advisory review. Flag **only** what static gates can't. Ruff,
CodeQL, bandit/pylint/mypy already cover style — do not duplicate them. If a finding
would be caught by ruff or CodeQL, stay silent.

## What to review for

- **PR focus**: the diff does one thing. If it bundles unrelated changes, say so.
- **Fix-around instead of fix-upstream**: a local workaround for what is really a
  GEAK/Magpie/TraceLens/framework defect. Ask for the upstream ticket.
- **Debt growth**: a new feature flag / env toggle used to route around a design
  problem, or a new suppression without a stated reason.


## What NOT to flag

- Formatting, import order, naming, line length → ruff.
- Broad `except`, cyclomatic complexity, unused variables → ruff/pylint.
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
  The authoritative invariants live in `AGENTs.md` and the path-scoped instruction
  files — treat a violation of them as a correctness bug, not a style preference.
- If nothing rises to a real design or correctness concern, leave no comments.

Architecture-specific invariants live in path-scoped files under
[`.github/instructions/`](instructions/) and fire only on the relevant files.
