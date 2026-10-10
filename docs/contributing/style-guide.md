---
myst:
    html_meta:
        "description": "Coding conventions for contributors to Hyperloom. Covers Python formatting, linting, type checking, shell scripts, YAML, documentation, and REUSE licensing requirements."
        "keywords": "Hyperloom, contributing, coding style, Python, Ruff, mypy, Bandit, pytest, REUSE, license, AMD GPU, ROCm"
---

# Hyperloom coding style guide

This document describes the conventions contributors should follow when changing
[Hyperloom](https://github.com/AMD-AGI/Hyperloom). It complements
[AGENTS.md](../../AGENTS.md) (authoring rules), [CONTRIBUTING.md](../../CONTRIBUTING.md)
(workflow and checks), and the machine-readable settings in `pyproject.toml`.

When this guide and tooling disagree, **tooling wins** — update the guide if you change
`pyproject.toml`, `.pre-commit-config.yaml`, or CI workflows.

## Principles

1. **Correctness before cleverness** — prefer readable code with tests over micro-optimizations.
2. **Automate what you can** — run `pre-commit` locally; let CI enforce the rest.
3. **No secrets in the tree** — credentials belong in environment variables or secret stores.
4. **License hygiene** — every file must satisfy [REUSE](https://reuse.software/) (see below).
5. **Shape is reviewable** — maintainability, readability, extensibility, and reliability are review criteria, not afterthoughts; see [Size and complexity](#size-and-complexity).

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

### Size and complexity

Two layers. The **gate** is the `code-metrics` CI job (`scripts/code_metrics.py`): it measures industry-standard white-box metrics at their tools' default thresholds and fails the PR when the code gets worse. The **review triggers** below it are the softer numbers a reviewer asks about.

| Gated dimension | Fails when | Source of the threshold | Tool |
|-----------------|-----------|-------------------------|------|
| Cyclomatic complexity | > 10 | McCabe (1976); mccabe / ruff `C901` default | ruff `C901` |
| Cognitive complexity | > 15 | SonarSource rule S3776 default | complexipy |
| Statements, branches, returns per function | > 50, > 12, > 6 | pylint `R0915`, `R0912`, `R0911` defaults | ruff `PLR0915`/`PLR0912`/`PLR0911` |
| Arguments, positional arguments | > 5, > 5 | pylint `R0913`, `R0917` defaults (`self`/`cls` not counted) | ruff `PLR0913`/`PLR0917` |
| Local variables, nested block depth | > 15, > 5 | pylint `R0914`, `R1702` defaults | ruff `PLR0914`/`PLR1702` |
| Public methods per class | > 20 | pylint `R0904` default | ruff `PLR0904` |
| Module length | > 1000 lines | pylint `C0302` default | the script |
| Maintainability index | < 10 | radon rank C ("extremely low") | radon |
| Duplicated code | any clone of >= 100 tokens and >= 10 lines | SonarSource CPD defaults | jscpd |
| Dead code | any finding at >= 80% confidence | vulture's recommended CI setting | vulture |

The thresholds, their sources and the exact tool versions live in `pyproject.toml` under `[tool.hyperloom.code_metrics]`. The scope is all of `src` and `scripts`, minus the same vendored and shipped-example trees as Ruff's `extend-exclude`; files under a `tests/` directory are exempt from everything but duplication. Suppression comments (`# noqa` for the Ruff-measured dimensions, complexipy's ignore marker, `jscpd:ignore-start`) do not hide a unit from the gate: Ruff and complexipy run with their ignore switches, and jscpd reads copies with its markers defused. Vulture alone honours `# noqa`, because that marker belongs to Ruff: `# noqa: F401` is the sanctioned form for a side-effect import or a re-export, and Ruff's `RUF100` flags one that suppresses nothing.

Units that were already over a threshold when the gate landed are recorded with their value in `scripts/code_metrics_baseline.json`, keyed by file and qualified name (`Class.method`), so moving code inside a file does not disturb them. The baseline only goes down:

- a unit over a threshold that is not in the baseline fails the PR — new code meets the limits;
- a baselined unit that got worse than its recorded value fails — **do not grow the backlog**: adding branches or lines to a unit, or lines to a module, that is already over is a failure, not a judgement call;
- a baselined unit that improved, dropped under the limit or was deleted fails as *out of date* until `python scripts/code_metrics.py --update-baseline` is run and the baseline committed in the same PR (the command can only lower or remove entries);
- relative to the base branch, the PR's baseline may only lose entries or lower values, and the config may not loosen (no raised limit, no new exclusion, no file the base's scope measured left unmeasured). Adding a unit to the baseline is not a way to pass.

CI judges a PR with the base branch's copy of the gate scripts, so a PR that edits `scripts/code_metrics*.py` does not grade itself; the report lists every edit to the gate's scripts, workflows or tool pins under *Gate implementation changed* for a reviewer. A gate change that the base's copy cannot run (a new config key, say) lands in two steps: first teach the scripts to accept it, then use it.

A unit that only moved to another file keeps its baseline entry when its qualified name (a module: its file name) is unique among the moved units and its value is no worse; a renamed unit, or one moved to another class, does not, and has to meet the limits. The report — new, worse and out-of-date units with links to the lines — is on the job summary and in one sticky PR comment. Run the gate locally with the install line in the script's docstring; `--base-ref origin/main` adds the base-branch check.

Editing a unit that was already over is not a demand to repay its debt — the gate only asks that it not get worse. Extracting a helper while you are in there is in scope, and the gate rewards it with an out-of-date entry to tighten; a standalone rewrite of an unrelated module is a separate PR (see [`AGENTS.md`](../../AGENTS.md) § *One concern per change*).

**Review triggers.** These are not gated; they are the point at which a reviewer asks for a split or for the reason the shape is right.

| Unit | Trigger | Where the number comes from |
|------|---------|-----------------------------|
| Function length | ~60 lines | Just above the tree's 90th percentile |
| Module length | ~800 lines | Roughly the tree's 90th percentile |

Neither number identifies a problem on its own. A long function can be one prompt template with a complexity of 1, and a short one can carry a dozen field comparisons that still need semantic review. Crossing a trigger asks the reviewer to look for a responsibility boundary, not to assume there is one — and "this is a single template" is an accepted answer. Split when it improves ownership, data flow, or testability. A function counts from its `def` line through its last line, decorators excluded and blank, comment and docstring lines included; a nested or `async` function is measured on its own. Module length is physical lines. Tests are exempt from the triggers — a table-driven test that gains a case per behaviour is doing its job — though not from the duplication and boundary rules.

Passing a trigger is not a merge blocker — it means the PR description says why, or the change splits. Passing the [Complexity ceiling](#complexity-ceiling) below is; inside its scope the `code-metrics` gate is stricter still.

Structure the split along the boundaries the code already has — one job per module, cohesive inside, dependencies pointing one way down the layers. A split that only moves lines to a second file, leaving the two halves reaching into each other, trades one long file for a cycle.

#### Complexity ceiling

Cyclomatic complexity above 20 is the one number here a review blocks on. It covers two cases: a function the change adds, and a function whose complexity the change raises — from at or below 20 to above it, or higher still when it was already above. A unit that stood above 20 before the change and that the change does not make worse is backlog, not this PR's debt.

The ceiling sits at twice the trigger because the trigger asks a question the ceiling has stopped accepting answers to: at 10, "this is one dispatch table" settles it; at 20 the unit carries more branches than a reviewer can keep in their head while reading it, and a paragraph in the description does not make it reviewable. Split it, or keep the new branches out of it.

Measure both sides — the head tree, and the same file at the merge base, since the verdict is a comparison:

```bash
ruff check --select C901 --config "lint.mccabe.max-complexity=20" --force-exclude \
  --output-format concise <changed .py files under src/, tests excluded>

git show "$(git merge-base origin/main HEAD)":<path> > /tmp/base.py
ruff check --select C901 --config "lint.mccabe.max-complexity=20" --isolated \
  --output-format concise /tmp/base.py
```

The tree carried 124 units above 20 when the ceiling was introduced (2026-10-10). The ceiling is what stops that number growing; it is not a demand to pay the 124 down.

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

#### Dependency direction

Imports between packages point one way, and CI enforces it: the `import-linter (architecture contracts)` job in `lint.yml` (and the `import-linter` pre-commit hook) runs `lint-imports` against the contracts in [`.importlinter`](../../.importlinter). A PR that adds an import breaking one of them fails, and the output names the contract and the import chain.

| Contract | Rule |
|----------|------|
| `hl-top-layers` | `hyperloom.cli` > `orchestrator` > `agents` > `inference_optimizer` > `common`: a package imports only from layers below it |
| `kf-only-common` | `kernelforge` imports nothing from `hyperloom` except `hyperloom.common` |
| `common-is-leaf` | `hyperloom.common` imports nothing else from the project, `kernelforge` included |
| `agents-independent` | `hyperloom.agents.*` packages do not import each other |
| `kf-backends-independent` | `kernelforge.kernel_backends.*` packages do not import each other |
| `kf-entry-protected` | `hyperloom.agents`, `inference_optimizer` and `common` do not import `kernelforge` directly; the orchestrator and `hyperloom.cli` are the way in |
| `acyclic` | no import cycle between sibling packages anywhere under `hyperloom` or `kernelforge` |

Imports inside functions count; imports under `if TYPE_CHECKING:` and test code do not. A new package under `hyperloom/agents/` or `kernelforge/kernel_backends/` is added to its independence list in the same PR.

Two contracts carry the violations that existed when the gate landed, as exact `importer -> imported` lines under `ignore_imports`: `hl-top-layers` lists the upward imports of `hyperloom.inference_optimizer.cli` (an entry point that sits inside the core layer), and `acyclic` lists the imports that break today's cycles. These baselines only shrink. The contracts fail on an `ignore_imports` line that no longer matches an import, so a PR that removes one of those imports deletes its line in the same commit. Do not add a line to get a new import through: move the code to the layer it belongs in, or invert the dependency (pass a callable or a protocol down instead of importing up). The baseline is per module pair, so another import between two modules already listed is not caught; do not lean on that.

Run it locally without installing the project:

```bash
pip install "import-linter==2.15"
PYTHONPATH=src lint-imports --no-cache
```

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

CI runs `pylint --errors-only` on core packages (fatal/error severity only). Fix new error-level issues in touched modules; convention, refactor, and style messages are intentionally out of scope here; the size and complexity rules among them (`R0912`, `R0915`, ...) are gated by the `code-metrics` job instead — see [Size and complexity](#size-and-complexity).

### Tests (pytest)

| Convention | Detail |
|------------|--------|
| Location | `**/tests/` next to the code under test; operator scripts use `scripts/tests/` |
| Discovery | `[tool.pytest.ini_options] testpaths` in `pyproject.toml` |
| Async | `asyncio_mode = auto` |
| Markers | Register new markers in `pyproject.toml`; use `@pytest.mark.<name>` |

**E2E markers** (skipped in CI by default):

- `critic_agent_e2e`, `targeted_build_e2e`

**What to test:** Pin the **exported surface** — CLI flags, public functions, persisted schemas, artifact layouts — with tests that state the contract *and* its failure modes; those are what callers outside this repo depend on. Pick the boundary by what is being protected: a focused unit test for deterministic logic, or a CLI/filesystem/serialization integration test where that pins the contract more directly. Internal functions that only thread a business flow together do not each need one: per-function tests there assert the current implementation and break on the next refactor. Prefer covering those flows through their entry point, and unit-test an internal helper when it carries real logic of its own.

**When you replace a test,** carry its contract and failure-mode assertions across, and keep them in the default CI selection — the `*_e2e` markers above are excluded from it, so they supplement that baseline rather than stand in for it. The 90% line-coverage floor cannot show that a specific assertion survived.

**Coverage:** CI enforces **90% line coverage** on measured trees (`[tool.coverage.report] fail_under`). CLI drivers, subprocess wrappers, and hardware-only paths are omitted from the denominator — see `[tool.coverage.run] omit`. Cover the logic you introduce as described under *What to test*; do not chase coverage on omitted paths, and do not pad internal plumbing with per-function tests to move the number.

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

CI enforces this through the **REUSE Compliance** workflow.

## Commit and pull request hygiene

- Branch from `main`; keep commits logically grouped.
- PR description: problem, approach, test evidence.
- **Do not commit:** virtualenvs, `.coverage`, build artifacts, large logs, credentials, local `.env`.
- **Observable effect:** required. Anything an operator can observe is described in the PR that changes it, and the release cut aggregates those into the GitHub release — see [`AGENTS.md`](../../AGENTS.md) § *Authoring rules of engagement* for what counts and what is exempt.

## Local development checklist

```bash
python -m venv .venv && source .venv/bin/activate   # or Windows equivalent
pip install -e ".[test,ci]"
pip install pre-commit ruff mypy reuse
pre-commit install
pre-commit run --all-files   # first-time baseline

pytest -m "not critic_agent_e2e and not targeted_build_e2e"
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
| `.importlinter` | Architecture contracts (import-linter) |
| `.github/workflows/lint.yml` | Ruff, import-linter, Bandit, Pylint (CI) |
| `.github/workflows/tests-coverage.yml` | Pytest + coverage gate |
| `.github/workflows/secret-scan.yml` | Gitleaks |
| `.github/workflows/reuse-lint.yml` | REUSE |
| `.github/workflows/codeql.yml` | CodeQL security analysis |
