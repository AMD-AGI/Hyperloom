---
myst:
    html_meta:
        "description": "Coding conventions for contributors to Hyperloom. Covers Python formatting, linting, type checking, shell scripts, YAML, documentation, and REUSE licensing requirements."
        "keywords": "Hyperloom, contributing, coding style, Python, Ruff, mypy, Bandit, pytest, REUSE, license, AMD GPU, ROCm"
---

# Hyperloom coding style guide

This document describes the conventions contributors should follow when changing
[Hyperloom](https://github.com/AMD-AGI/Hyperloom). It complements
[CONTRIBUTING.md](../../CONTRIBUTING.md) (workflow and checks) and the machine-readable
settings in `pyproject.toml`.

When this guide and tooling disagree, **tooling wins** — update the guide if you change
`pyproject.toml`, `.pre-commit-config.yaml`, or CI workflows.

## Principles

1. **Minimal, focused diffs** — one logical change per pull request; avoid drive-by refactors.
2. **Correctness before cleverness** — prefer readable code with tests over micro-optimizations.
3. **Automate what you can** — run `pre-commit` locally; let CI enforce the rest.
4. **No secrets in the tree** — credentials belong in environment variables or secret stores.
5. **License hygiene** — every file must satisfy [REUSE](https://reuse.software/) (see below).

## Python

### Language and layout

| Setting | Value | Source |
|---------|-------|--------|
| Minimum Python | 3.10 | `requires-python` in `pyproject.toml` |
| Target version | 3.10 (`py310`) | `[tool.ruff] target-version` |
| Line length | 120 | `[tool.ruff] line-length` |
| Package layout | `src/hyperloom/...` | setuptools `where = ["src"]` |

### Formatting and lint (Ruff)

Ruff is the single formatter and linter for Python.

- **Format:** `ruff format .` (Black-compatible; 120 columns).
- **Lint:** `ruff check .` — rules `E`, `F`, `W` (pycodestyle errors, Pyflakes, warnings).
- **Ignored globally:** `E501` (line length — owned by the formatter), `E741` (single-letter names in math/parsing helpers).

Run both before opening a PR that touches Python:

```bash
ruff check .
ruff format --check .   # or `ruff format .` to apply
```

**Do not** add `# noqa` or per-file ignores unless there is a documented reason (import cycles, test patterns). Existing per-file ignores live in `[tool.ruff.lint.per-file-ignores]` — extend that table instead of inline suppressions.

**Future rules** (`B`, `I`, `UP`, `SIM`, `RUF`) are commented in `pyproject.toml` and will be enabled once the backlog is zero. New code should already follow import sorting and common bugbear patterns even before those rules are turned on.

### Module structure

Follow patterns in existing packages (for example, `hyperloom.orchestrator`):

```python
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One-line module summary."""

from __future__ import annotations

import stdlib...
import third_party...
from hyperloom... import local...
```

- **`from __future__ import annotations`** — use in new modules for forward references and cleaner hints.
- **Module docstring** — required for public modules under `src/hyperloom/`.
- **Imports** — stdlib, then third party, then `hyperloom.*`, separated by blank lines. Some modules intentionally import after constants (see per-file `E402` ignores); do not reorder those without understanding the cycle.
- **Constants** — `UPPER_SNAKE_CASE` at module level; prefix private constants with `_`.
- **Types** — use modern syntax (`str | None`, `list[str]`, `collections.abc` for parameters). Prefer typed public APIs; `Any` only at boundaries (JSON, subprocess, LLM payloads).

### Type checking (mypy)

mypy is **recommended locally**, not yet a CI gate:

```bash
pip install mypy
mypy src/hyperloom
```

Guidelines:

- Add type hints to new public functions and dataclass fields.
- Use `TYPE_CHECKING` blocks for import-only types.
- Do not silence mypy with broad `# type: ignore` — narrow the ignore or fix the type.

When mypy is promoted to CI, configuration will live in `pyproject.toml` under `[tool.mypy]`.

### Security (Bandit)

Bandit scans production code (`src/hyperloom`, `scripts/`). Tests are excluded.

- `B101` (assert) is skipped repo-wide — asserts are allowed in tests and invariants.
- Fix medium-and-higher findings before merge; do not add new `nosec` comments without a security review comment in the PR.

### Pylint

CI runs `pylint --errors-only` on core packages (fatal/error severity only). Fix new error-level issues in touched modules; style/convention messages are intentionally out of scope.

### Tests (pytest)

| Convention | Detail |
|------------|--------|
| Location | `**/tests/` next to the code under test; operator scripts use `scripts/tests/` |
| Discovery | `[tool.pytest.ini_options] testpaths` in `pyproject.toml` |
| Async | `asyncio_mode = auto` |
| Markers | Register new markers in `pyproject.toml`; use `@pytest.mark.<name>` |

**E2E markers** (skipped in CI by default):

- `critic_agent_e2e`, `robustness_agent_e2e`, `targeted_build_e2e`

**Coverage:** CI enforces **90% line coverage** on measured trees (`[tool.coverage.report] fail_under`). CLI drivers, subprocess wrappers, and hardware-only paths are omitted from the denominator — see `[tool.coverage.run] omit`. Add unit tests for logic you introduce; do not chase coverage on omitted paths.

**Naming:** `test_<behavior>.py`, functions `test_<scenario>`, classes `Test<Component>`.

## Shell scripts

Shell scripts live under `scripts/`, `src/hyperloom/**/assets/`, and agent tool directories.

- Target **bash** unless the shebang says otherwise.
- **Quote variable expansions** — most ShellCheck findings are `SC2086` (unquoted `$var`).
- Use `set -euo pipefail` in new scripts when safe (existing scripts may omit it for compatibility — match neighbors).
- Run **ShellCheck** locally: pre-commit includes `shellcheck-py`.

## YAML and GitHub Actions

- Workflow files must include REUSE SPDX headers (see below).
- **yamllint** uses the `relaxed` preset; line-length is disabled to avoid churn.
- **actionlint** validates `.github/workflows/` — pin action versions (`@v7`), avoid `${{ }}` injection pitfalls.

When adding a workflow that should skip on documentation-only changes, copy the **canonical `paths-ignore` list** from `CONTRIBUTING.md`.

## Markdown and documentation

- User-facing docs: `docs/` (Sphinx / Read the Docs).
- Agent skills and operator references may live beside code (`SKILL.md`, `references/`).
- Use MyST/Sphinx conventions for new `docs/` pages; CI builds with `sphinx-build -b html docs docs/_build/html`.
- Link to ROCm docs where appropriate: [Hyperloom on ROCm](https://rocm.docs.amd.com/projects/hyperloom/en/latest/index.html).

## Licensing (REUSE)

Every committed file must have clear copyright and license metadata:

1. **Preferred:** SPDX header at the top of the file:

   ```text
   # SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
   # SPDX-License-Identifier: MIT
   ```

   Use `#` for Python/shell/YAML, `<!-- -->` for Markdown/HTML as appropriate.

2. **Fallback:** aggregate annotation in `REUSE.toml` for file types that cannot carry headers (some binary/config patterns).

Run locally:

```bash
pip install reuse
reuse lint
```

CI enforces this via the **REUSE Compliance** workflow.

## Commit and pull request hygiene

- Branch from `main`; keep commits logically grouped.
- PR description: problem, approach, test evidence.
- **Do not commit:** virtualenvs, `.coverage`, build artifacts, large logs, credentials, local `.env`.
- **Changelog:** user-visible changes should note `CHANGELOG.md` when maintainers expect a release note.

## Local development checklist

```bash
python -m venv .venv && source .venv/bin/activate   # or Windows equivalent
pip install -e ".[test,ci]"
pip install pre-commit ruff mypy reuse
pre-commit install
pre-commit run --all-files   # first-time baseline

pytest -m "not critic_agent_e2e and not robustness_agent_e2e and not targeted_build_e2e"
ruff check . && ruff format --check .
mypy src/hyperloom
reuse lint
```

## Related configuration files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Ruff, Bandit, pytest, coverage, packaging |
| `.pre-commit-config.yaml` | Local hooks mirroring static analysis |
| `.gitleaks.toml` | Secret-scan allowlists |
| `REUSE.toml` | Default license annotation |
| `.github/workflows/lint.yml` | Ruff, Bandit, Pylint (CI) |
| `.github/workflows/tests-coverage.yml` | Pytest + coverage gate |
| `.github/workflows/secret-scan.yml` | Gitleaks |
| `.github/workflows/reuse-lint.yml` | REUSE |
| `.github/workflows/codeql.yml` | CodeQL security analysis |
