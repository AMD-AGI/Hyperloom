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
- **Lint:** `ruff check .` — base rules `E`, `F`, `W` (pycodestyle errors, Pyflakes, warnings), `BLE` (no blind `except Exception`), `RUF100` (no unused `# noqa`), plus the principle rules below.
- **Ruff version:** CI pins `ruff==0.16.2` (`lint.yml`) and the pre-commit hook uses the same `rev`; bump both together.
- **Ignored globally:** `E501` (line length — owned by the formatter), `E741` (single-letter names in math/parsing helpers).

Run both before opening a PR that touches Python:

```bash
ruff check .
ruff format --check .   # or `ruff format .` to apply
```

**Do not** add `# noqa` or per-file ignores unless there is a documented reason (import cycles, test patterns). Existing per-file ignores live in `[tool.ruff.lint.per-file-ignores]` — extend that table instead of inline suppressions.

**Principle rules** are hard gates (`extend-select` in `pyproject.toml`): the tree is at zero and any new finding fails CI.

| Rules | What they hold |
|-------|----------------|
| `G`, `LOG` | Logging takes `%`-style arguments, not f-strings; correct logger API (`exc_info` only where there is an exception, no root-logger calls in new code) |
| `TD001`, `TD003`–`TD007`, `FIX001`, `FIX003`, `FIX004` | A `TODO` is allowed but must link an issue and be well-formed; no `FIXME`/`XXX`/`HACK` markers |
| `PGH` | No blanket `# noqa` or `# type: ignore` — name the code (`# type: ignore[import-not-found]`) |
| `RSE`, `RET`, `PIE` | `raise X` not `raise X()`; explicit, consistent returns with no `else` after `return`; no redundant `pass`, `range(0, n)` or `lambda: []` wrappers |
| `ERA` | No commented-out code. The detector is a heuristic: a prose comment that happens to parse as Python (`# Subcommand: verify`) trips it — reword the comment |
| `DTZ` | `datetime` values carry a timezone |
| `A` | Do not shadow builtins (`aiter`, AMD's kernel library, is allowed by name) |
| `C4` | Comprehension and collection-call idioms (`{}` over `dict()`, no `[x for x in y]`) |

Where existing code needs the old shape for a behavioural reason — persisted naive timestamps, keyword names that mirror a stdlib signature, root-logger calls in an unconfigured CLI — the exception is a per-file ignore with that reason next to it, not a `# noqa`.

**Not yet enabled:** `I` (import sorting) and `UP` (pyupgrade) are clean mechanical rewrites but touch hundreds of lines that open PRs also touch, so turning them on waits for a team decision on when to land the rewrite. `B`, `SIM` and the rest of `RUF` carry a backlog and belong behind a ratchet rather than a hard gate. New code should already follow import sorting and common bugbear patterns.

### Size and complexity

Two layers. The **gate** is the `code-metrics` CI job (`scripts/code_metrics.py`): it measures the white-box metrics below on every PR and fails the PR when a file the PR touches gets worse. The **review triggers** under it are the softer numbers a reviewer asks about.

| Gated dimension | Fails when | Why this number | Tool |
|-----------------|-----------|-----------------|------|
| Cyclomatic complexity | > 20 | The [Complexity ceiling](#complexity-ceiling): twice McCabe's 10, the point where a function has more paths than a reviewer can hold while reading it | ruff `C901` |
| Cognitive complexity | > 30 | Twice SonarSource's S3776 default of 15, the same step as the ceiling; it weights nesting, so past 30 a function has to be re-read rather than read | complexipy |
| Function length | > 80 lines | A function past one screen; counted as below, `def` line through the last line | the script |
| Nested block depth | > 5 | pylint `R1702` default: past five levels the innermost line depends on six conditions at once | ruff `PLR1702` |
| Module length | > 1200 lines (over 800 is a report-only warning) | Half again the 800-line review trigger; at that size a module almost always holds more than one job | the script |
| Duplicated code | any clone of >= 100 tokens and >= 10 lines | SonarSource CPD defaults | jscpd |
| Dead code | any finding at >= 80% confidence | vulture's recommended CI setting | vulture |

The thresholds, the reason for each and the exact tool versions live in `pyproject.toml` under `[tool.hyperloom.code_metrics]`. The scope is all of `src` and `scripts`, minus the same vendored and shipped-example trees as Ruff's `extend-exclude`; files under a `tests/` directory are exempt from everything but duplication. Suppression comments (`# noqa` for the Ruff-measured dimensions, complexipy's ignore marker, `jscpd:ignore-start`) do not hide a unit from the gate: Ruff and complexipy run with their ignore switches, and jscpd reads copies with its markers defused. Vulture alone honours `# noqa`, because that marker belongs to Ruff: `# noqa: F401` is the sanctioned form for a side-effect import or a re-export, and Ruff's `RUF100` flags one that suppresses nothing.

Units that were already over a threshold when the gate landed are recorded with their value in `scripts/code_metrics_baseline.txt`, one sorted line per unit (`<metric> <path>::<Class.method> <value>`, or `<metric> <path> <value>` for a module), so moving code inside a file does not disturb them and a change to the baseline is one line per unit in the diff.

**Only the files a PR touches are judged** (the diff against the merge base; for a push to `main`, against the commit it replaced). A file the PR does not touch never fails it, whatever is found there; the report lists such findings as information. In a touched file:

- a unit over a threshold that is not in the baseline fails — new code meets the limits;
- a baselined unit that got worse than its recorded value fails — **do not grow the backlog**: adding branches or lines to a unit, or lines to a module, that is already over is a failure, not a judgement call;
- in both cases the same file at the merge base is measured too, and a unit the base already had at that value or worse is backlog, not this PR's debt — the comparison the [Complexity ceiling](#complexity-ceiling) defines;
- a baselined unit that improved, dropped under the limit or was deleted fails as *out of date* until `python scripts/code_metrics.py --update-baseline` is run and the baseline committed in the same PR (the command can only lower or remove entries).

Across the whole file, not only touched entries: relative to the base branch the PR's baseline may only lose entries or lower values, and the config may not loosen (no raised limit, no new exclusion, no file the base's scope measured left unmeasured). Adding a unit to the baseline is not a way to pass.

**Override: the `baseline-raise` label.** A PR that has to land a new or worse unit, or grow the baseline, carries the `baseline-raise` label, and its description says why. The gate reads the label from the GitHub API when it runs (adding or removing the label re-runs the job); new, worse and baseline-growth findings are then *waived* — still listed in full in the job log and the report, no longer failing. It waives nothing else: an out-of-date entry, a loosened config and the checks below still fail.

CI judges a PR with the base branch's copy of the gate scripts, so a PR that edits `scripts/code_metrics*.py` does not grade itself; the report lists every edit to the gate's scripts, workflows or tool pins under *Gate implementation changed* for a reviewer. A gate change that the base's copy cannot run (a new config key, say) lands in two steps: first teach the scripts to accept it, then use it.

A unit that only moved to another file keeps its baseline entry when its qualified name (a module: its file name) is unique among the moved units and its value is no worse; a renamed unit, or one moved to another class, does not, and has to meet the limits. The report — new, worse and out-of-date units with links to the lines — is on the job summary and in one sticky PR comment. Run the gate locally with the install line in the script's docstring; `--base-ref origin/main` judges the files changed since the merge base and adds the base-branch checks.

Editing a unit that was already over is not a demand to repay its debt — the gate only asks that it not get worse. Extracting a helper while you are in there is in scope, and the gate rewards it with an out-of-date entry to tighten; a standalone rewrite of an unrelated module is a separate PR (see [`AGENTS.md`](../../AGENTS.md) § *One concern per change*).

**Checks without a baseline.** The same job runs four checks that have no backlog to carry; each fails the PR on its own:

| Check | Fails when | Scope |
|-------|-----------|-------|
| Comments | an added run of more than 8 consecutive full-line `#` comments; an added comment that points at a PR or issue (`#1234`, `PR 1234`, `GH-1234`, `issue #12`, a `/pull/` link) or narrates an incident (`after the outage`, a dated `incident`, `postmortem`). A comment starting with `TODO` and the line after it may link an issue, as Ruff's `TD003` asks. A comment the PR only moves, re-indents or carries through a rename is not added | added lines of `.py` files, tests included; docstrings are not comments |
| English only | a CJK character (CJK ideographs, CJK symbols and punctuation, halfwidth and fullwidth forms) | every tracked text file, and on a PR its title, body and every commit message. Test data that needs a multi-byte character uses a non-CJK one (the euro sign is three bytes in UTF-8) |
| Production code does not import test code | a module under `src/` or `scripts/` that is not itself a test imports a `tests` package or a `test_*` module (a module merely named `test` is not test code) | the whole tree. Move what both sides need into a non-test module |
| Repeated literals | the PR takes a string or bytes value (3+ characters) or a number (other than 0, 1, -1 and 2) from fewer than 3 occurrences in a module to 3 or more | the production modules the PR touches, counted before and after (a renamed module against the file it came from; a literal that only moved between touched modules is not added); dict keys, `x["key"]` subscripts, the key of `.get`/`.pop`/`.setdefault` and of an `in` test, keyword names, docstrings, f-string text, annotations and `__all__` do not count. Name the value once as a module-level constant |

The comment and literal checks are diff-only on purpose: the history in a comment is cheap to keep out and expensive to strip later, and a whole-tree literal count would flag most modules in the tree for values nobody is changing.

**Review triggers.** These are not gated; they are the point at which a reviewer asks for a split or for the reason the shape is right. One of them has a ceiling above it that a reviewer blocks on; see [Complexity ceiling](#complexity-ceiling).

| Unit | Trigger | Where the number comes from |
|------|---------|-----------------------------|
| Function length | ~60 lines | Just above the tree's 90th percentile |
| Cyclomatic complexity | 10 | McCabe default; measurable on demand with `ruff check --select C901` |
| Module length | ~800 lines | Roughly the tree's 90th percentile |

Neither number identifies a problem on its own. A long function can be one prompt template with a complexity of 1, and a short one can carry a dozen field comparisons that still need semantic review. Crossing a trigger asks the reviewer to look for a responsibility boundary, not to assume there is one — and "this is a single template" is an accepted answer. Split when it improves ownership, data flow, or testability.

**How the lines are counted:** a function spans its `def` line through its last line, decorators excluded and blank, comment and docstring lines included; a nested or `async` function is measured on its own, not folded into its parent. Module length is physical lines. Tests are exempt from the size triggers — a table-driven test that gains a case per behaviour is doing its job — though not from the duplication and boundary rules.

Measure rather than argue:

```bash
ruff check --select C901 --config "lint.mccabe.max-complexity=10" src/hyperloom src/kernelforge
```

Passing a trigger is not a merge blocker — it means the PR description says why, or the change splits. Passing the ceiling below is, and so is the `code-metrics` gate above.

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

CI now measures the ceiling: the `code-metrics` job (see [Size and complexity](#size-and-complexity)) applies exactly these two cases to every function in the files a PR touches under `src/` and `scripts/`, comparing the head with the merge base, and the units already above 20 are its baseline. The review rule stands; the job makes the measurement for it.

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

CI runs `pylint --errors-only` on core packages (fatal/error severity only). Fix new error-level issues in touched modules; convention, refactor, and style messages are intentionally out of scope here; the size and complexity limits CI does hold are the `code-metrics` job's — see [Size and complexity](#size-and-complexity).

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
