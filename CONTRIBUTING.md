# Contributing to Hyperloom

Thank you for helping improve Hyperloom. This guide covers the expected workflow, local setup, and quality checks.

## Pull Request workflow
- Create a feature branch off `main`.
- Keep changes focused and include context in the PR description (problem, approach, test coverage).
- Ensure merge requirements and applicable GitHub checks pass before requesting review (see [CI and documentation-only changes](#ci-and-documentation-only-changes)).
- Avoid committing generated artifacts; keep diffs minimal.

## Proposing a new framework or platform

Hyperloom includes external components (GEAK, Magpie, TraceLens, IntelliKit, frameworks). Building for a new framework or platform must work bring-up through the owning-components before integration code lands, and open a tracking issue for it. Owners are named in [`.github/CODEOWNERS`](.github/CODEOWNERS).

## CI and documentation-only changes

This repository treats **documentation-only** pushes and pull requests the same way across automation: when **every** changed file in that event matches **only** the [canonical paths-ignore list](#canonical-paths-ignore-list) below, matching GitHub Actions workflows **do not** start (no workflow run is created for that event).

### What is skipped today on GitHub Actions

| Check (concept) | Implemented in Actions | Skipped for doc-only events? |
|-----------------|------------------------|------------------------------|
| **Pytest** (full suite with coverage reporting) | [`.github/workflows/tests-coverage.yml`](.github/workflows/tests-coverage.yml) (reads ``[tool.hyperloom.tests_coverage]`` / coverage config from ``pyproject.toml`` via inline Python) | Yes (`paths-ignore` on `push` / `pull_request` for all branches) |
| **CodeQL** | [`.github/workflows/codeql.yml`](.github/workflows/codeql.yml) | Yes on **PR and push** when doc-only (`paths-ignore`); **no** — the **weekly schedule** on the default branch still runs a full analysis |
| **Ruff** (lint + format check) | [`.github/workflows/lint.yml`](.github/workflows/lint.yml) (`ruff check` / `ruff format --check`; hard gate) | Yes (same `paths-ignore` as tests / CodeQL) |
| **Pylint** (errors-only) | [`.github/workflows/lint.yml`](.github/workflows/lint.yml) (`pylint --errors-only` on `hyperloom.inference_optimizer`, `hyperloom.orchestrator`, `hyperloom.agents.robustness`, `hyperloom.agents.framework`, `hyperloom.agents.critic.runtime`, and `hyperloom.agents.quantization`; advisory `continue-on-error`) | Yes (same `paths-ignore`) |
| **Mypy** | [`.github/workflows/lint.yml`](.github/workflows/lint.yml) (advisory `continue-on-error`; config in `[tool.mypy]`) | Yes (same `paths-ignore`) |

If you add standalone workflows for **pytest**, **ruff**, **pylint**, or similar, copy the **same** `paths-ignore` blocks as in `tests-coverage.yml` / `codeql.yml` so documentation-only PRs stay consistent and cheap.

### Canonical paths-ignore list

Use this list (or keep it in sync) for any workflow that should skip on documentation-only changes:

- `**/*.md`
- `docs/**`
- `LICENSE*`
- `COPYRIGHT`
- `**/CODEOWNERS`
- `.gitattributes`

If **any** changed file falls **outside** these patterns (for example `.py`, `pyproject.toml`, or `.github/workflows/*.yml`), the workflows that declare this list run as usual.

These GitHub jobs are optional from a default merge-policy perspective; skipping them on doc-only PRs saves runner time. You should still run **local** `pytest`, **ruff**, and **mypy** when your edits are not purely cosmetic (for example, Markdown that embeds commands, code blocks, or configuration snippets).

## Coding style

See **[docs/contributing/style-guide.md](docs/contributing/style-guide.md)** for Python, shell, YAML, testing, and REUSE conventions.

## Development setup
- Python 3.10+.
- Create and activate a virtual environment.
- Install dependencies (including test extras):  
  `pip install -e .[test]`
- To mirror the coverage CI job locally:  
  `pip install -e ".[test,ci]"` then run `pytest` with the same arguments as in the `[tool.hyperloom.tests_coverage]` table in `pyproject.toml` (see ``tests-coverage.yml`` for the exact list: marker filter + ``--cov`` flags).
- If you plan to run lint/type checks, install tools:  
  `pip install ruff mypy`  
  Or install the bundled dev extra: `pip install -e ".[test,dev]"`

## Pre-commit (recommended)

Hyperloom ships a [`.pre-commit-config.yaml`](.pre-commit-config.yaml) that mirrors most static-analysis gates before code reaches CI.

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files   # optional baseline after clone
```

### Hooks

| Hook | Purpose |
|------|---------|
| **pre-commit-hooks** | Trailing whitespace, EOF, merge conflicts, large files, private keys |
| **ruff** | Python lint (`E`/`F`/`W`), same as `lint.yml` |
| **bandit** | Security lint on production Python (tests excluded) |
| **shellcheck** | Shell script analysis |
| **yamllint** / **actionlint** | YAML and GitHub Actions workflow lint |
| **reuse** | REUSE/SPDX compliance |
| **gitleaks** | Secret scan (mirrors `secret-scan.yml`) |
| **codespell** | Typos in docs and comments |

**Intentionally not in pre-commit** (too slow or environment-specific): full **pytest**/coverage, **Pylint**, **CodeQL**, and E2E pytest markers.

## Testing
- Run the full test suite from the repo root:  
  `pytest`
- To target a directory or file:  
  `pytest src/hyperloom/inference_optimizer/tests`  
  `pytest src/hyperloom/inference_optimizer/tests/test_prompt_builder.py -k subset`

### Coverage (source of truth)

**Authoritative UT coverage** for this repository comes only from the GitHub Actions workflow [`.github/workflows/tests-coverage.yml`](.github/workflows/tests-coverage.yml). **Policy lives in `pyproject.toml`**: `[tool.coverage.run]` / `[tool.coverage.report]` (measured trees and report options), and `[tool.hyperloom.tests_coverage]` (full CI `pytest` argv: marker filter + pytest-cov flags). The workflow writes the job Summary from the same ``source`` list. CI **enforces** a minimum line coverage of **90%** (`[tool.coverage.report].fail_under` in `pyproject.toml`) via a dedicated "Enforce coverage fail_under" step in `tests-coverage.yml`; set the `COVERAGE_RELAX_FAIL_UNDER` repository variable to `1`/`true`/`yes`/`on` to skip enforcement.

The default pytest **`testpaths`** include **`src/hyperloom/agents/quantization/tests`** so quantization driver code is exercised in CI, not only via inference_optimizer tests.

Open the workflow run, then the **Summary** tab on the *Tests with Coverage* job for per-tree line coverage (informational only).

Do not treat ad hoc local `pytest --cov=...` invocations or any other workflow as the canonical headline metric unless that workflow is explicitly documented here. If you add a second CI job that prints coverage, keep it non-authoritative or remove it to avoid conflicting percentages.

## Linting and formatting
- Ruff:  
  `ruff check .`
- Type checks (mypy):  
  `mypy src/hyperloom`  
  Configuration lives in `[tool.mypy]` in `pyproject.toml`. CI runs mypy as an **advisory** job until the type-check backlog is reduced.
- CI runs **Pylint** with **`--errors-only`** (fatal/error severity only, not style) on several first-party packages from [`.github/workflows/lint.yml`](.github/workflows/lint.yml) (advisory `continue-on-error` today). Root **`[tool.pylint.main]`** in `pyproject.toml` holds minimal defaults (e.g. `jobs`); tighten or add message disables there as the backlog shrinks.

## Before opening a PR
- [ ] Tests pass (`pytest`) when you changed executable code or behavior-affecting config.
- [ ] Lint clean (`ruff check .`) when you changed Python sources.
- [ ] Type checks clean (`mypy ...`) when you changed typed packages.
- [ ] No unwanted files (build artifacts, large logs, credentials).

## Security
- Do not include secrets in code or logs.
- Report vulnerabilities privately as described in `SECURITY.md`.
