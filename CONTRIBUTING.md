# Contributing to Hyperloom

Thank you for helping improve Hyperloom. This guide covers the expected workflow, local setup, and quality checks.

## Pull Request workflow
- Create a feature branch off `main`.
- Keep changes focused and include context in the PR description (problem, approach, test coverage).
- Ensure merge requirements and applicable GitHub checks pass before requesting review (see [CI and documentation-only changes](#ci-and-documentation-only-changes)).
- Avoid committing generated artifacts; keep diffs minimal.

## CI and documentation-only changes

This repository treats **documentation-only** pushes and pull requests the same way across automation: when **every** changed file in that event matches **only** the [canonical paths-ignore list](#canonical-paths-ignore-list) below, matching GitHub Actions workflows **do not** start (no workflow run is created for that event).

### What is skipped today on GitHub Actions

| Check (concept) | Implemented in Actions | Skipped for doc-only events? |
|-----------------|------------------------|------------------------------|
| **Pytest** (full suite with coverage reporting) | [`.github/workflows/tests-coverage.yml`](.github/workflows/tests-coverage.yml) (invokes `pytest` with flags from `pyproject.toml` via `ci/coverage_summary.py`) | Yes (`paths-ignore` on `push` / `pull_request` for all branches) |
| **CodeQL** | [`.github/workflows/codeql.yml`](.github/workflows/codeql.yml) | Yes on **PR and push** when doc-only (`paths-ignore`); **no** — the **weekly schedule** on the default branch still runs a full analysis |
| **Ruff** (lint + format check) | [`.github/workflows/lint.yml`](.github/workflows/lint.yml) (`ruff check` / `ruff format --check`; steps use `continue-on-error: true` until backlog is cleared) | Yes (same `paths-ignore` as tests / CodeQL) |
| **Pylint** (errors-only) | [`.github/workflows/lint.yml`](.github/workflows/lint.yml) (`pylint --errors-only` on `inference_optimizer`, `robustness_agent`, `framework_agent`, critic `runtime`, `quantization_agent`; advisory `continue-on-error`) | Yes (same `paths-ignore`) |
| **Mypy** | Local / optional tooling only here | N/A |

If you add standalone workflows for **pytest**, **ruff**, **pylint**, or similar, copy the **same** `paths-ignore` blocks as in `tests-coverage.yml` / `codeql.yml` so documentation-only PRs stay consistent and cheap.

### Canonical paths-ignore list

Use this list (or keep it in sync) for any workflow that should skip on documentation-only changes:

- `**/*.md`
- `docs/**`
- `LICENSE*`
- `COPYRIGHT`
- `CODEOWNERS`
- `.gitattributes`

If **any** changed file falls **outside** these patterns (for example `.py`, `pyproject.toml`, or `.github/workflows/*.yml`), the workflows that declare this list run as usual.

These GitHub jobs are optional from a default merge-policy perspective; skipping them on doc-only PRs saves runner time. You should still run **local** `pytest`, **ruff**, and **mypy** when your edits are not purely cosmetic (for example, Markdown that embeds commands, code blocks, or configuration snippets).

## Development setup
- Python 3.10+.
- Create and activate a virtual environment.
- Install dependencies (including test extras):  
  `pip install -e .[test]`
- To mirror the coverage CI job locally (pytest-cov + same flags as Actions):  
  `pip install -e ".[test,ci]"` then run `pytest` with the arguments printed by `python3 ci/coverage_summary.py --pytest-ci-args` (they match the `[tool.hyperloom.tests_coverage]` table in `pyproject.toml`).
- If you plan to run lint/type checks, install tools:  
  `pip install ruff mypy`

## Testing
- Run the full test suite from the repo root:  
  `pytest`
- To target a directory or file:  
  `pytest inference_optimizer/tests`  
  `pytest inference_optimizer/tests/test_prompt_builder.py -k subset`

### Coverage (source of truth)

**Authoritative UT coverage** for this repository comes only from the GitHub Actions workflow [`.github/workflows/tests-coverage.yml`](.github/workflows/tests-coverage.yml). **Policy lives in `pyproject.toml`**: `[tool.coverage.run]` / `[tool.coverage.report]` (measured trees and `fail_under`), and `[tool.hyperloom.tests_coverage]` (full CI `pytest` argv: marker filter + pytest-cov flags, plus the documented name of the optional relax variable for `fail_under`). The workflow runs `ci/coverage_summary.py` so the Summary table tracks **`[tool.coverage.run].source`** without duplicating that list in YAML. Measured code includes **`ci/`** (alongside `OOB/`, `quantization_agent/`, etc.). The workflow installs **`OOB/`** (`pip install -e OOB/.`) so `agent_mcp_server` tests run; mirror that locally when working on [`inference_optimizer/tests/test_oob_units.py`](inference_optimizer/tests/test_oob_units.py).

Setting the repository variable **`COVERAGE_RELAX_FAIL_UNDER`** to `1` / `true` / `yes` / `on` disables `fail_under` enforcement in CI (pytest uses `--cov-fail-under=0` and the strict coverage gate is skipped). Default is strict when the variable is unset.

The default pytest **`testpaths`** include **`quantization_agent/tests`** so quantization driver code is exercised in CI, not only via inference_optimizer tests.

Open the workflow run, then the **Summary** tab on the *Tests with Coverage* job for per-tree line coverage. **Combined line coverage across all configured source trees must meet `fail_under`** (90% today, `[tool.coverage.report]` in `pyproject.toml`); the job fails if the threshold is not met.

Do not treat ad hoc local `pytest --cov=...` invocations or any other workflow as the canonical headline metric unless that workflow is explicitly documented here. If you add a second CI job that prints coverage, keep it non-authoritative or remove it to avoid conflicting percentages.

## Linting and formatting
- Ruff:  
  `ruff check .`
- Type checks (mypy):  
  `mypy inference_optimizer kernel-agent robustness-agent`
  - Adjust paths if you change package locations.
- CI runs **Pylint** with **`--errors-only`** (fatal/error severity only, not style) on several first-party packages from [`.github/workflows/lint.yml`](.github/workflows/lint.yml) (advisory `continue-on-error` today). Root **`[tool.pylint.main]`** in `pyproject.toml` holds minimal defaults (e.g. `jobs`); tighten or add message disables there as the backlog shrinks.

## Before opening a PR
- [ ] Tests pass (`pytest`) when you changed executable code or behavior-affecting config.
- [ ] Lint clean (`ruff check .`) when you changed Python sources.
- [ ] Type checks clean (`mypy ...`) when you changed typed packages.
- [ ] No unwanted files (build artifacts, large logs, credentials).

## Security
- Do not include secrets in code or logs.
- Report vulnerabilities privately as described in `SECURITY.md`.
