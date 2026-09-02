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
handlers and integrating external components (Magpie, TraceLens, GEAK, AMD Quark).

## Where things are documented

| Topic | Source of truth |
|---|---|
| Optimization loop, phase chain, orchestration model | [`docs/conceptual/optimization-loop.md`](docs/conceptual/optimization-loop.md) |
| Profiling → TraceLens → kernel value chain | [`docs/reference/kernel-execution-path.md`](docs/reference/kernel-execution-path.md) |
| External components (Magpie, TraceLens, GEAK, IntelliKit, AgentKernelArena) | [`docs/components/`](docs/components/) |
| Agent instructions / runtime behavior | [`src/hyperloom/inference_optimizer/SKILL.md`](src/hyperloom/inference_optimizer/SKILL.md) |
| Workflow & PR process | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Formatting, naming, testing, REUSE conventions | [`docs/contributing/style-guide.md`](docs/contributing/style-guide.md) |
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
- **Fix upstream, not around it.** When the root cause is inside a component (GEAK, Magpie,
  TraceLens, IntelliKit) or a framework, direct the user to fix it there — don't paper over it with a
  local workaround.
- **New framework or platform?** Work bring-up through the owning-components before
  integration code lands — see [`CONTRIBUTING.md`](CONTRIBUTING.md) § *Proposing a new
  framework or platform*.
- **Don't grow the debt.** No new broad `except Exception` / bare `except` — catch the
  specific error or let it raise. No new feature flag or env toggle to route around a
  design problem; a flag is a decision deferred.
- **Delete, don't comment out.** Dead code goes; version control is the archive.
  Commented-out blocks and `# removed …` tombstones rot and mislead.
- **Clean design.** One boundary rule per concern, owned by one module. Derive over
  hardcode — a single computed source beats duplicated constants. Minimal typed
  interfaces; validate at the system boundary, trust internal callers.
