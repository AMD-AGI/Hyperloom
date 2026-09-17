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
- **Size and complexity**: a function past ~60 lines, cyclomatic complexity past 10, or a
  module pushed past ~800 lines. No linter covers this (Ruff runs `E`/`F`/`W`; Pylint is
  `--errors-only`), so review is the only place it surfaces. Name the seam to split on,
  not just the line count — and treat an existing long function the change merely edits as
  out of scope.
- **Cohesion and coupling**: a module that has acquired a second job, a responsibility
  moved to the wrong layer, a dependency pointing back up the layers, a new import cycle,
  or a caller reaching around the layer that owns a thing. A diff that bypasses an
  established pipeline or reintroduces a retired construct belongs here too — the
  architecture doc named in `AGENTS.md` is the authority on where a boundary lives.
- **Failure-hiding error handling**: a `try`/fallback that swallows an error and returns
  a default or `None`, so the caller can't tell success from silent failure. This includes
  a new broad `except Exception` that hides the failure rather than catching a specific,
  expected error — neither ruff's `E`/`F`/`W` nor `pylint --errors-only` flags it.
- **Concurrency**: missing/incorrect `await`, races on shared state, unawaited tasks,
  blocking calls on the event loop.
- **Unused abstraction**: a flag, strategy, or generic helper added for a single caller,
  or a parameter always passed the same value.
- **Contract & cache invariants**: a change that silently alters an external contract
  or breaks a documented rule.
- **Test strategy**: exported behaviour — a CLI flag, public function, persisted schema,
  artifact layout — landing without a test that pins the contract and its failure modes.
  The inverse too: per-function unit tests bolted onto internal plumbing that only threads
  a flow together, where an end-to-end test is what the change actually needs.
- **PR focus**: the PR addresses one aspect. If it bundles unrelated changes, say so.
- **Fix-around instead of fix-upstream**: a local workaround for what is really a
  GEAK/Magpie/TraceLens/framework defect.
- **Debt growth**: a new feature flag / env toggle used to route around a design
  problem, or a new suppression without a stated reason.
- **Missing changelog entry**: an observable change — a behaviour, interface, default,
  flag, or artifact — with no `CHANGELOG.md` entry under `[Unreleased]`, and no note in
  the description saying why the change is unobservable.

## What NOT to flag

- Bare `except:`, formatting, import order, naming, line length → ruff.
- Unused variables, unreachable code → ruff/pylint. (Cyclomatic complexity is *not*
  covered by either — review it under **Size and complexity** above.)
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
