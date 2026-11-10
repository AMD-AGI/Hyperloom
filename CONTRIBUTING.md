# Contributing to Hyperloom

Thank you for helping improve Hyperloom. This guide covers the expected workflow, local setup, and quality checks. The authoring rules that a change is reviewed against live in [`AGENTS.md`](AGENTS.md).

## Pull Request workflow
- Create a feature branch off `main`.
- Keep changes focused and include context in the PR description (problem, approach, test coverage).
- Ensure merge requirements and applicable GitHub checks pass before requesting review (see [CI and documentation-only changes](#ci-and-documentation-only-changes)).
- Describe anything an operator can observe — a behaviour, an interface, a default, a flag, an artifact — in the PR description; the release cut aggregates those into the [GitHub release](https://github.com/AMD-AGI/Hyperloom/releases). Refactors with nothing observable, and test- or docs-only changes, are exempt; say which in the PR description. See [`AGENTS.md`](AGENTS.md) § *Authoring rules of engagement*.
- Avoid committing generated artifacts.

## Proposing a new framework or platform

Hyperloom includes external components (GEAK, Magpie, TraceLens, IntelliKit, frameworks). Work bring-up for a new framework or platform through the owning components before integration code lands, and open a tracking issue for it. Owners are named in [`.github/CODEOWNERS`](.github/CODEOWNERS).

## CI and documentation-only changes

This repository treats **documentation-only** pushes and pull requests the same way across automation: an event is documentation-only when **every** changed file matches the [canonical documentation-only list](#canonical-documentation-only-list) below. Workflows whose checks the `main` ruleset requires still start on such an event and skip only their expensive jobs, so their checks report; CodeQL does not start at all.

### What is skipped today on GitHub Actions

| Check (concept) | Implemented in Actions | On doc-only events |
|-----------------|------------------------|--------------------|
| **Pytest** (full suite with coverage reporting) | [`.github/workflows/tests-coverage.yml`](.github/workflows/tests-coverage.yml) (reads ``[tool.hyperloom.tests_coverage]`` / coverage config from ``pyproject.toml`` via inline Python) | Starts; the test shards do not run and the required `coverage (Python 3.x)` checks report success without measuring |
| **CodeQL** | [`.github/workflows/codeql.yml`](.github/workflows/codeql.yml) | Does not start on **PR and push** (`paths-ignore`); the **weekly schedule** on the default branch still runs a full analysis |
| **Ruff** (lint + format check) | [`.github/workflows/lint.yml`](.github/workflows/lint.yml) (`ruff check` / `ruff format --check` / `check_broad_suppress.py`; hard gate) | Runs (required check `ruff`) |
| **Pylint** (errors-only) | [`.github/workflows/lint.yml`](.github/workflows/lint.yml) (`pylint --errors-only` on `hyperloom.inference_optimizer`, `hyperloom.orchestrator`, `hyperloom.agents.framework`, `hyperloom.agents.critic.runtime`, and `hyperloom.agents.quantization`; advisory `continue-on-error`) | Skipped (job condition) |
| **Mypy** | [`.github/workflows/lint.yml`](.github/workflows/lint.yml) (advisory `continue-on-error`; config in `[tool.mypy]`) | Skipped (job condition) |

The decision is made once per event by [`.github/workflows/code-changes.yml`](.github/workflows/code-changes.yml), which runs [`scripts/ci_code_changed.py`](scripts/ci_code_changed.py) and exposes `code=true|false` to the calling workflow. A workflow whose check is required on `main` must not use `paths-ignore` (a workflow that does not start leaves its required check pending forever); call `code-changes.yml` and condition its jobs instead. Other workflows may copy the `paths-ignore` block from `codeql.yml`.

### Canonical documentation-only list

`scripts/ci_code_changed.py` (`DOCS_ONLY_PATTERNS`) holds this list and matches it as GitHub path filters do (`*` within one directory, `**` across them; other glob syntax is refused). `codeql.yml`'s `paths-ignore` blocks repeat it, and a test keeps them equal:

- `**/*.md`
- `docs/**`
- `LICENSE*`
- `COPYRIGHT`
- `**/CODEOWNERS`
- `.gitattributes`

If **any** changed file falls **outside** these patterns (for example `.py`, `pyproject.toml`, or `.github/workflows/*.yml`), every check runs as usual.

Skipping the test shards on doc-only changes saves runner time. You should still run **local** `pytest`, **ruff**, and **mypy** when your edits are not purely cosmetic (for example, Markdown that embeds commands, code blocks, or configuration snippets).

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
- [ ] PR description states the observable effect, or says why the change is unobservable.
- [ ] Tests pass (`pytest`) when you changed executable code or behavior-affecting config.
- [ ] Lint clean (`ruff check .`) when you changed Python sources.
- [ ] Type checks clean (`mypy ...`) when you changed typed packages.
- [ ] No unwanted files (build artifacts, large logs, credentials).

## Security
- Do not include secrets in code or logs.
- Report vulnerabilities privately as described in `SECURITY.md`.
