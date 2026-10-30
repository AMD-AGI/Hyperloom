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
| `B006`, `B012` | No mutable default argument (`def f(x=[])`); no `return`, `break` or `continue` inside `finally`, which discards the exception in flight |

Where existing code needs the old shape for a behavioural reason — persisted naive timestamps, keyword names that mirror a stdlib signature, root-logger calls in an unconfigured CLI — the exception is a per-file ignore with that reason next to it, not a `# noqa`.

**Not yet enabled:** `I` (import sorting) and `UP` (pyupgrade) are clean mechanical rewrites but touch hundreds of lines that open PRs also touch, so turning them on waits for a team decision on when to land the rewrite. The rest of `B`, `SIM` and the rest of `RUF` carry a backlog and belong behind a ratchet rather than a hard gate. New code should already follow import sorting and common bugbear patterns.

### Size and complexity

Two layers. The **gate** is the `code-metrics` CI job (`scripts/code_metrics.py`): it measures the white-box metrics below on every PR and fails the PR when a file the PR touches gets worse. The **review triggers** under it are the softer numbers a reviewer asks about.

| Gated dimension | Fails when | Why this number | Tool |
|-----------------|-----------|-----------------|------|
| Cyclomatic complexity | > 20 | The [Complexity ceiling](#complexity-ceiling): twice McCabe's 10, the point where a function has more paths than a reviewer can hold while reading it. Every `and` / `or`, ternary, comprehension and `assert` counts as a branch, so it reads higher than ruff `C901` | radon `cc` |
| Cognitive complexity | > 30 | Twice SonarSource's S3776 default of 15, the same step as the ceiling; it weights nesting, so past 30 a function has to be re-read rather than read | complexipy |
| Function length | > 50 lines | Lines of code in the body, counted as below: past 50 a function rarely does one thing | the script |
| Nested block depth | > 4 | Past four levels the innermost line depends on five conditions at once | ruff `PLR1702` |
| Parameter count | > 5 | pylint `R0913` default: past five, the call site stops reading as a sentence and the values travel together anyway — pass the object they come from, or name them in a dataclass (A8). A first parameter named `self`/`cls` does not count; every other named parameter does, keyword-only and `_`-named ones included, `@overload`/`@override` signatures are counted like any other, and `*args` and `**kwargs` count one each, so it reads higher than ruff `PLR0913` | the script |
| Module length | > 1200 lines (over 800 is a report-only warning) | Half again the 800-line review trigger; at that size a module almost always holds more than one job | the script |
| Duplicated code | any clone of >= 100 tokens and >= 10 lines | SonarSource CPD defaults | jscpd |
| Dead code | any finding at >= 80% confidence | vulture's recommended CI setting | vulture |
| Directory size | > 50 entries | Past 50 a reader can no longer scan the listing for the module they need; group the modules into subpackages by responsibility. The entries are the production modules directly in the directory and its subdirectories that hold one; a directory is judged when the change touches a file under it | the script |

The thresholds, the reason for each and the exact tool versions live in `pyproject.toml` under `[tool.hyperloom.code_metrics]`. The scope is all of `src` and `scripts`, minus the same vendored and shipped-example trees as Ruff's `extend-exclude`; files under a `tests/` directory are exempt from everything but duplication. Suppression comments (`# noqa` for the Ruff-measured dimensions, complexipy's ignore marker, `jscpd:ignore-start`) do not hide a unit from the gate: Ruff and complexipy run with their ignore switches, and jscpd reads copies with its markers defused. Vulture alone honours `# noqa`, because that marker belongs to Ruff: `# noqa: F401` is the sanctioned form for a side-effect import or a re-export, and Ruff's `RUF100` flags one that suppresses nothing.

Nothing is recorded in the repository: the reference is the base itself. In the same run the gate measures the tree at the merge base (for a push to `main`, the commit it replaced) with the same tools and this tree's config, and compares each unit with it, so the units already over a threshold on `main` are its backlog and a unit a merged PR improved is the next PR's limit.

**Only the files a PR touches are judged** (the diff against the merge base). A file the PR does not touch never fails it, whatever is found there; the report lists such findings as information. In a touched file:

- a unit over a threshold that the base did not have over it fails — new code meets the limits;
- a unit worse than at the base fails — **do not grow the backlog**: adding branches or lines to a unit, or lines to a module, that is already over is a failure, not a judgement call;
- a unit the base already had at that value or worse passes — backlog, not this PR's debt, the comparison the [Complexity ceiling](#complexity-ceiling) defines; one the PR improved is listed, and there is nothing to update.

The config may not loosen relative to the base branch (no raised limit, no new exclusion, no file the base's scope measured left unmeasured).

**Override: the `baseline-raise` label.** A PR that has to land a new or worse unit, or keep a failing design-check finding, carries the `baseline-raise` label, and its description says why. The gate reads the label from the GitHub API when it runs (adding or removing the label re-runs the job; the push to `main` reads the label of the PR whose merge it is); new and worse findings and the failing design checks are then *waived* — still listed in full in the job log and the report, no longer failing. It waives nothing else: a loosened config and the checks without a baseline still fail.

CI judges a PR with the base branch's copy of the gate scripts, so a PR that edits `scripts/code_metrics*.py` does not grade itself; the report lists every edit to the gate's scripts, workflows or tool pins under *Gate implementation changed* for a reviewer. A gate change that the base's copy cannot run (a new config key, say) lands in two steps: first teach the scripts to accept it, then use it.

A unit that only moved to another file is compared with the unit it left at the base when its qualified name (a module: its file name) is unique among the moved units, its value is no worse and the unit it left is no longer over there; a renamed unit, a copy, or one moved to another class is not, and has to meet the limits. The report — new, worse and improved units with links to the lines — is on the job summary and in one sticky PR comment; the job summary of each push to `main` also lists every unit over a threshold, which is where the backlog is read. Run the gate locally with the install line in the script's docstring: without arguments it lists every unit over a threshold and fails none; `--base-ref origin/main` judges the files changed since the merge base as CI does.

Editing a unit that was already over is not a demand to repay its debt — the gate only asks that it not get worse. Extracting a helper while you are in there is in scope, and the report lists the improvement; a standalone rewrite of an unrelated module is a separate PR (see [`AGENTS.md`](../../AGENTS.md) § *One concern per change*).

**Checks without a baseline.** The same job runs six checks that have no backlog to carry, and the design checks below. Each of the six fails the PR on its own, and `baseline-raise` does not waive them:

| Check | Fails when | Scope |
|-------|-----------|-------|
| Comments | an added run of more than 8 consecutive full-line `#` comments; an added comment that points at a PR or issue (`#1234`, `PR 1234`, `GH-1234`, `issue #12`, a `/pull/` link) or narrates an incident (`after the outage`, a dated `incident`, `postmortem`). A comment starting with `TODO` and the line after it may link an issue, as Ruff's `TD003` asks. A comment the PR only moves, re-indents or carries through a rename is not added | added lines of `.py` files, tests included; docstrings are not comments |
| Broad catch reason | an added `# noqa: BLE001` with no reason after the code on the same line (`except Exception:  # noqa: BLE001 - outermost CLI boundary, logged`, or `- isolation boundary: one sweep point`). A line the PR only moves or re-indents is not added | added lines of `.py` files, tests included |
| Broad catch records the error | an added broad `except` (bare, `Exception`, `BaseException`) that neither re-raises -- directly, through a `raise*` helper, or by keeping the error for a later `raise` -- nor records it: a logger call, `print`, `traceback`, `warnings` or a `*log*` helper. Whether the handler sits at the boundary that owns the failure is review-pr rule S8; a handler that logs with `exc_info` needs no `# noqa: BLE001`, and names its boundary in a plain comment on the `except` line. An `except` whose body the PR does not touch and whose code its own file lost (only its comment changed) is not added; the same line removed in another file exempts nothing | added lines of production `.py` files |
| English only | a CJK character (CJK ideographs, CJK symbols and punctuation, halfwidth and fullwidth forms) | every tracked text file, and on a PR its title, body and every commit message. Test data that needs a multi-byte character uses a non-CJK one (the euro sign is three bytes in UTF-8) |
| Production code does not import test code | a module under `src/` or `scripts/` that is not itself a test imports a `tests` package or a `test_*` module (a module merely named `test` is not test code) | the whole tree. Move what both sides need into a non-test module |
| Repeated literals | the PR takes a string or bytes value (3+ characters) or a number (other than 0, 1, -1 and 2) from fewer than 3 occurrences in a module to 3 or more | the production modules the PR touches, counted before and after (a renamed module against the file it came from; a literal that only moved between touched modules is not added); dict keys, `x["key"]` subscripts, the key of `.get`/`.pop`/`.setdefault` and of an `in` test, keyword names, docstrings, f-string text, annotations and `__all__` do not count. Name the value once as a module-level constant |

The comment, broad-catch and literal checks are diff-only on purpose: the history in a comment is cheap to keep out and expensive to strip later, a whole-tree broad-catch check would fail every PR on catches written before it, and a whole-tree literal count would flag most modules in the tree for values nobody is changing.

**Design checks.** The same job also runs diff-only checks for the code-shape review rules A1 and A8-A12 (`scripts/code_metrics_design.py`). Each has its own section in the report. A1, A8 and the A12 forwarder fail the PR (the `baseline-raise` label waives them, with the reason in the description); A9, A10, A11 and the A12 single-implementation abstraction are advisory, as the A rules rate them, and are listed without failing. They judge only what the PR adds in the production modules it touches; tests (a `tests/` package, a `test_*.py` module, `conftest.py`) are exempt from all of them. A line is *added* when the diff adds it and the PR did not remove the same text elsewhere (a move, a re-indent, a renamed file); a function or class is *new* when its `def`/`class` line is added and the same file at the merge base (for a renamed file, the file it came from) had no unit of that qualified name, so editing an existing function never makes it new.

| Check | Fails when | Exempt |
|-------|-----------|--------|
| A1 Private name of another package | an added import, or attribute access through an imported name, of a private name (`_x`, not a dunder) or of a module under a private package, owned by another package unit. The units come from `.importlinter`: the root packages and every module a contract names as a layer or an independent module (`hyperloom.cli`, `hyperloom.orchestrator`, `hyperloom.agents.critic`, `kernelforge.kernel_backends.hip`, ...); a module belongs to the longest unit that prefixes it, and `scripts/` is a unit of its own. The units are read from the merge base's contracts file, so a PR cannot redraw the units it is judged by, and moving `import-contracts` counts as loosening | same-unit use; dunders; third-party packages. Fix: make the name public in its owner, or move the caller there |
| A8 Flag argument | a new function whose first statement is `if <flag>:` (or `if not <flag>:`, or a comparison of a two-value `Literal` parameter) with a branch ending in `return` or `raise`, where the flag is a `bool` (annotated, or defaulting to `True`/`False`) or two-value `Literal` parameter | `main`, click/typer commands (`@*.command`, `.group`, `.option`, ...), functions taking an `argparse.Namespace`. Fix: one function per value |
| A9 Tuple return | a new function annotated to return `tuple[...]` (also `typing.Tuple`, and inside `Optional`/`Union`/`\|`) of more than `tuple-return-max-elements` (2) positions, or returning a tuple display that long | `tuple[X, ...]`; dunders (`__reduce__`), whose shape the protocol fixes. Fix: a dataclass or `NamedTuple` |
| A10 Raw vocabulary comparison | an added `==`/`!=` against, `in` over a tuple/list/set display of, or `case` pattern of, a string that is the value of a member of exactly one `Enum` under `vocabulary-roots` (`src`), in a module that names that enum (imports or defines it), when the other side reads as a member: its `.value`, or a name whose words end like the enum's (`status` or `run_status` for `RunStatus`) | a value two enums share, or one compared in a module that never names its enum: that is another vocabulary using the same word; a comparison with raw input (`raw`, `v`, `str(x)`): that is the conversion to the member. Fix: compare with the member; convert outside input once, where it enters |
| A11 Hidden global write | an added `global` statement, or store into `globals()` | none (`nonlocal` is a closure, not module state). Fix: pass the value in and return the result, or keep the state on an object the caller owns |
| A12 Pure forwarder | a new function (not a dunder, no decorator but `staticmethod`/`classmethod`) whose body is one `return` of a call passing exactly its own parameters (receiver aside) through unchanged | a call of a builtin (by its bare name), of a class (a factory), of `.get` (a table lookup) or of a method of a constant (`", ".join`); a call of one of the parameters; a decorated function (the decorator caches, registers or wraps); a method overriding one its base class (under `vocabulary-roots`) defines. Fix: call the target directly |
| A12 Single-implementation abstraction | a new ABC or `Protocol` with exactly one implementation under `vocabulary-roots` (a subclass; for a protocol also a class defining all its methods), in the same package unit | an interface a lower unit declares for a higher one to implement. Fix: use the implementation until a second one exists |

A10 is limited to the enum's own vocabulary on purpose: matched against every enum value in `src`, almost every hit in the tree was a dict field that uses the same word for another vocabulary (a `status` of `"failed"` or `"unknown"` read from a tool's JSON), not a bypassed enum; and the place that converts outside input to the member has to compare the raw string, so only a subject that already reads as the member is judged.

**Review triggers.** These are not gated; they are the point at which a reviewer asks for a split or for the reason the shape is right. One of them has a ceiling above it that a reviewer blocks on; see [Complexity ceiling](#complexity-ceiling).

| Unit | Trigger | Where the number comes from |
|------|---------|-----------------------------|
| Cyclomatic complexity | 10 | McCabe default; measurable on demand with `radon cc` |
| Module length | ~800 lines | Roughly the tree's 90th percentile |

Neither number identifies a problem on its own. A long function can be one prompt template with a complexity of 1, and a short one can carry a dozen field comparisons that still need semantic review. Crossing a trigger asks the reviewer to look for a responsibility boundary, not to assume there is one — and "this is a single template" is an accepted answer. Split when it improves ownership, data flow, or testability.

**How the lines are counted:** a function counts the lines of code in its body, from its first statement through its last line; the `def` lines, decorators, and blank, comment and docstring lines do not count; a nested or `async` function is measured on its own, not folded into its parent. Module length is physical lines. Tests are exempt from the size triggers — a table-driven test that gains a case per behaviour is doing its job — though not from the duplication and boundary rules.

Measure rather than argue:

```bash
radon cc -s -n C src/hyperloom src/kernelforge
```

Passing a trigger is not a merge blocker — it means the PR description says why, or the change splits. Passing the ceiling below is, and so is the `code-metrics` gate above.

Structure the split along the boundaries the code already has — one job per module, cohesive inside, dependencies pointing one way down the layers. A split that only moves lines to a second file, leaving the two halves reaching into each other, trades one long file for a cycle.

#### Complexity ceiling

Cyclomatic complexity above 20 is the one number here a review blocks on. It covers two cases: a function the change adds, and a function whose complexity the change raises — from at or below 20 to above it, or higher still when it was already above. A unit that stood above 20 before the change and that the change does not make worse is backlog, not this PR's debt.

The ceiling sits at twice the trigger because the trigger asks a question the ceiling has stopped accepting answers to: at 10, "this is one dispatch table" settles it; at 20 the unit carries more branches than a reviewer can keep in their head while reading it, and a paragraph in the description does not make it reviewable. Split it, or keep the new branches out of it.

Measure both sides — the head tree, and the same file at the merge base, since the verdict is a comparison:

```bash
radon cc -s -n D <changed .py files under src/, tests excluded>

git show "$(git merge-base origin/main HEAD)":<path> > /tmp/base.py
radon cc -s -n D /tmp/base.py
```

The tree carried 124 units above 20 when the ceiling was introduced (2026-10-10). The ceiling is what stops that number growing; it is not a demand to pay the 124 down.

CI now measures the ceiling: the `code-metrics` job (see [Size and complexity](#size-and-complexity)) applies exactly these two cases to every function in the files a PR touches under `src/` and `scripts/`, comparing the head with the merge base, so the units already above 20 there are its backlog. The `baseline-raise` label waives that CI check; review then accepts the override only when the PR description gives the reason the function stays above 20.

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
| `kf-evaluation-foundation` | `kernelforge.evaluation` imports no kernelforge product or entry package |
| `kf-tools-foundation` | `kernelforge.utils` imports no other kernelforge package |
| `kf-mcp-servers-light` | the stdio MCP server modules (`knowledge.pr_knowledge.mcp_server`, `workflows.forge_loop.roles.specialist_probe_server`) do not reach `agent_backends`, `cli` or the forge-loop beyond its roles, so a server starts without the agent runtime or the iteration loop |
| `common-is-leaf` | `hyperloom.common` imports nothing else from the project, `kernelforge` included |
| `agents-independent` | `hyperloom.agents.*` packages do not import each other |
| `kf-backends-independent` | `kernelforge.kernel_backends.*` packages do not import each other |
| `kf-entry-protected` | `hyperloom.agents`, `inference_optimizer` and `common` do not import `kernelforge` directly; the orchestrator and `hyperloom.cli` are the way in |
| `kf-public-interfaces` | `hyperloom` imports a kernelforge package only through what its `__init__` exports (or `kernelforge.workflows.forge_loop.api` for the loop), never one of its modules |
| `acyclic` | no import cycle between sibling packages anywhere under `hyperloom` or `kernelforge` |

Imports inside functions count; imports under `if TYPE_CHECKING:` and test code do not. A new package under `hyperloom/agents/` or `kernelforge/kernel_backends/` is added to its independence list in the same PR, a new stdio MCP server module is added to the `kf-mcp-servers-light` sources, and a new top-level kernelforge package or workflow under `kernelforge/workflows/` is added to the `kf-public-interfaces` list.

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

- `targeted_build_e2e`

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

When adding a workflow that should skip on documentation-only changes, follow `CONTRIBUTING.md` (*CI and documentation-only changes*): a workflow whose check the `main` ruleset requires calls `code-changes.yml` and conditions its jobs; any other workflow may use the canonical `paths-ignore` list.

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

pytest -m "not targeted_build_e2e"
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
