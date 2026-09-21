<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# AGENTS.md

The authoring contract for anyone (human or AI) changing this repo: **how to make a
change land cleanly.**

Hyperloom is an autonomous agentic system that optimizes end-to-end LLM inference on AMD
GPUs (host code and GPU kernels). A Python Coordinator drives an iterative
Think → Decide → Implement → Benchmark loop, delegating to LLM roles and programmatic
handlers and integrating external components.

## Where things are documented

| Topic | Source of truth |
|---|---|
| Optimization loop, phase chain, orchestration model | [`docs/conceptual/optimization-loop.md`](docs/conceptual/optimization-loop.md) |
| Profiling → TraceLens → kernel value chain | [`docs/reference/kernel-execution-path.md`](docs/reference/kernel-execution-path.md) |
| External components (Magpie, TraceLens, GEAK, IntelliKit, AgentKernelArena) | [`docs/components/`](docs/components/) |
| Agent instructions / runtime behavior | [`src/hyperloom/inference_optimizer/SKILL.md`](src/hyperloom/inference_optimizer/SKILL.md) |
| Workflow & PR process | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Style, module layout, tests, shell/YAML, commit hygiene, REUSE, local setup | [`docs/contributing/style-guide.md`](docs/contributing/style-guide.md) |
| PR checklist / AI review prompt | [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md), [`.github/copilot-instructions.md`](.github/copilot-instructions.md) |

Ships two packages: `src/hyperloom/` (the optimizer) and `src/kernelforge/` (the Forge
kernel backend). Both are exercised in CI.

## Common commands

```bash
pip install -e .[test]            # dev + test extras
pip install -e ".[test,ci]"       # mirror the coverage CI job locally

pytest                            # full suite (testpaths: src/**/tests + scripts/tests)
pytest src/hyperloom/inference_optimizer/tests/test_prompt_builder.py -k subset

ruff check .                      # lint (E, F, W; line-length 120, E501/E741 ignored)
ruff format --check .
mypy src/hyperloom                # advisory (runs as a non-gating CI job)
```

## Authoring rules of engagement

The rules for changing this code. Where a rule names a boundary, the linked architecture
doc is the authority on where that boundary lives.

- **One concern per change.** A PR fixes one issue or adds one capability. If you must
  bundle, say why in the description. Don't ride unrelated refactors in on a fix.
- **Diff budget.** Prefer reviewable diffs. A large diff is a signal to split, not to push
  harder. Cleanup in an unrelated file is a separate PR.
- **Size is a design signal.** A function that keeps growing, a branch tree you have to
  scroll, a module that collects everything — that is the design telling you a boundary is
  missing, and the answer is the split, not a bigger screen. New or rewritten code that
  crosses a trigger in the style guide § *Size and complexity* — the authority on the
  numbers, how to measure them, and when a long unit is fine as it stands — needs a reason
  in the PR description or a split. Editing a unit that was already over is not a demand to
  repay its debt; adding branches or a second responsibility to it is. These are review
  triggers, not gates: no linter measures them today and the tree carries a backlog above
  all three. Maintainability, readability, extensibility, and reliability are what the
  thresholds stand in for; when a threshold and one of those disagree, say so and keep the
  clearer code.
- **Review feedback is a hypothesis.** A comment can be wrong, or right about the symptom
  and wrong about the fix. Before acting on one, ask what you would build if this code did
  not exist yet, and whether the mechanism under discussion should exist at all. Answering
  every finding with one more guard is how a codebase grows; deleting the mechanism, or
  explaining why the current shape is right, is often the better answer. Converge on the
  implementation that is correct, not one that is merely defensible.
- **Verify the blast radius, not the tree.** While iterating, run the tests covering what
  you touched — a full suite run is slow enough that it becomes a reason to skip verifying
  at all. Widen the selection when a change crosses a boundary, not by default. The full
  local run belongs at the end, before you open a PR; see the style guide's local
  development checklist.
- **Test the contract, not the plumbing.** What you export — a CLI flag, a public
  function, a persisted schema, an artifact layout — is pinned by tests that state the
  contract and its failure modes, because someone outside this repo depends on it
  holding; a unit test or a CLI/filesystem one, whichever pins it more directly.
  Internal functions that only thread a business flow together do not each need
  one: per-function coverage there buys tests that assert the current implementation and
  break on the next refactor. Prefer covering those flows through their entry point, and
  unit-test an internal helper when it carries real logic of its own. When you replace a
  test, carry its contract and failure-mode assertions across and keep them running in the
  default CI selection. The coverage gate is a floor CI enforces, not the target.
- **Every change states its observable effect in its own PR description.** Anything an
  operator can observe — a behaviour, an interface, a default, a flag, an artifact — is
  spelled out in the PR that changes it. Not a follow-up: the release cut aggregates
  those descriptions into the GitHub release and cannot reconstruct them from commit
  subjects. Write it for someone who will never read the diff: what they will now see,
  and what the old behaviour cost them. Refactors with nothing observable, and test- or
  docs-only changes, are exempt — say which in the PR description rather than leaving
  the omission to be guessed at.
- **Fix upstream, not around it.** When the root cause is inside a component (GEAK, Magpie,
  TraceLens, IntelliKit) or a framework, fix it there and pin the fix — don't paper over it
  with a local workaround.
- **New framework or platform?** Work bring-up through the owning components before
  integration code lands — see [`CONTRIBUTING.md`](CONTRIBUTING.md) § *Proposing a new
  framework or platform*.
- **Don't grow the debt.** No new broad `except Exception` / bare `except` — catch the
  specific error or let it raise. No new feature flag or env toggle to route around a
  design problem; a flag is a decision deferred.
- **Trust the caller.** Validate at the system boundary, then trust internal callers. The
  agents driving this system are capable, so redundant re-checks, layered fallbacks, and
  belt-and-braces defaults buy nothing — they hide the failure they were added to survive
  and bury the real path.
- **Finish the replacement.** When a new path supersedes an old one, migrate the callers
  and delete what it replaced — the superseded implementation, the switch that chose
  between them, the configuration that fed it, the fixtures that only ever described the
  old shape — carrying the regression assertions across to the new path. Dead code goes;
  version control is the archive, and commented-out blocks and `# removed …` tombstones
  rot and mislead. A wrapper, an alias, dual-format parsing or a fallback route kept only
  to preserve an old *internal* shape is the replacement left unfinished. Compatibility is
  owed to identified consumers — supported public APIs, CLI behaviour, persisted data,
  plugins, anything deployed separately — so establish that boundary before deleting: a
  file's location in this repo and a text search that found nothing are not evidence that
  an entry point is unused. If the migration has to be staged, name the consumer still on
  the old path and the condition that retires it, and keep the adapter narrow enough that
  it does not become the new general entry point.
- **Comment below the local average.** Python explains most of itself; prefer a clearer
  name or a smaller function over a sentence about it. A comment earns its place only by
  saying what the code cannot — an invariant, a constraint from outside the file, why the
  slower path is the correct one. Never narrate the change itself: no step or plan
  numbering, no "previously this did X", nothing addressed to the reviewer. Module
  docstrings are a separate requirement; see the style guide.
- **Clean design.** One boundary rule per concern, owned by one module. A module should
  read as one job: cohesive inside, a minimal typed interface outward, and dependencies
  pointing one way down the layers — no cycles, and no reaching around the layer that owns
  a thing to touch what is behind it. Derive over hardcode — a single computed source
  beats duplicated constants, and duplicated state or a duplicated decision is the same
  problem: name the owner it derives from, and state the invariant any cache or replica
  left standing has to hold. A second copy of a behaviour is a bug you will later fix
  once and miss elsewhere; extend the existing one, or lift the shared part out.
- **Simplify by removing a mechanism.** A refactor that ends with the same moving parts in
  new positions has not simplified anything. The win is one mechanism fewer — a duplicated
  behaviour, a second source of state or configuration, an obsolete decision path, a
  forwarding layer with no job of its own — and it is judged across the whole
  responsibility and its callers, including the helpers, interfaces and parameter plumbing
  the change itself introduces. Extract a helper when it names a complete job or carries a
  rule that is genuinely the same in both places; inlining a thin wrapper or merging two
  equivalent entry points is the same kind of win, not the opposite of one. Caller count
  settles nothing on its own: a single-caller helper can earn its name, and code that only
  looks alike but answers to different contracts should stay apart rather than become one
  function with modes.
- **Leave nothing behind.** Working notes, audit trails, and analysis write-ups are
  byproducts of doing the work, not deliverables — don't commit them, least of all at the
  repo root, unless they were asked for. The change is the artifact.
- **The repo is English.** Code, identifiers, comments, docstrings, commit messages, and
  docs are English regardless of the language the work was discussed in. If a change would
  land anything else, flag it rather than committing it quietly.
