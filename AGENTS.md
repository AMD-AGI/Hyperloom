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
- **Size budget.** Prefer reviewable diffs. A large diff is a signal to split, not to push
  harder. Cleanup in an unrelated file is a separate PR.
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
- **Delete, don't comment out.** Dead code goes; version control is the archive.
  Commented-out blocks and `# removed …` tombstones rot and mislead.
- **Comment below the local average.** Python explains most of itself; prefer a clearer
  name or a smaller function over a sentence about it. A comment earns its place only by
  saying what the code cannot — an invariant, a constraint from outside the file, why the
  slower path is the correct one. Never narrate the change itself: no step or plan
  numbering, no "previously this did X", nothing addressed to the reviewer. Module
  docstrings are a separate requirement; see the style guide.
- **Clean design.** One boundary rule per concern, owned by one module. Derive over
  hardcode — a single computed source beats duplicated constants. Minimal typed
  interfaces.
- **Leave nothing behind.** Working notes, audit trails, and analysis write-ups are
  byproducts of doing the work, not deliverables — don't commit them, least of all at the
  repo root, unless they were asked for. The change is the artifact.
- **The repo is English.** Code, identifiers, comments, docstrings, commit messages, and
  docs are English regardless of the language the work was discussed in. If a change would
  land anything else, flag it rather than committing it quietly.
