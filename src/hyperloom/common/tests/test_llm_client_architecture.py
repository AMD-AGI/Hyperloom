# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Architecture guard: LLM provider access has exactly one sanctioned owner.

Hyperloom must never make scattered, ad-hoc LLM API calls. Every LLM
interaction goes through one of two sanctioned paths:

* **agentic work** -- an agent runtime (Claude Agent SDK / Codex SDK), reached
  through ``hyperloom.common.codex_session`` and the ``orchestrator/roles/``
  backends built on top of it;
* **single-shot inference** -- ``hyperloom.common.llm_config``, which is also
  the only module allowed to construct a provider SDK client
  (``get_openai_client`` / ``get_async_openai_client`` and their
  ``get_anthropic_client`` counterparts) and the only module allowed to speak a
  raw completion API.

This module parses every first-party Python source file and fails when a module
outside :data:`_ALLOWLISTED_OWNERS` imports or constructs a provider SDK
client, calls a bare completion endpoint, or hand-rolls the provider HTTP
protocol.

Known violations
----------------
Retiring the pre-existing call sites was staged across several changes, so each
surviving violation was pinned in :data:`_KNOWN_VIOLATIONS` as a
``(repo-relative path, rule code) -> occurrence count`` map. The scan result is
compared to that map for **exact equality**, which turns it into a ratchet:

* a violation whose ``(path, rule)`` is not pinned fails the test, so no new
  ad-hoc client can land;
* pinning more occurrences than the tree actually contains fails the test too,
  so migrating a call site without shrinking the map fails and forces the entry
  to be dropped.

Every call site has now been migrated, so the map is empty and the scan must
find nothing. It can only ever shrink: never add an entry to unblock new code
-- route the new code through a sanctioned path instead.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

_RULES: dict[str, str] = {
    "LLM001": "imports a provider SDK client module/class",
    "LLM002": "constructs a provider SDK client",
    "LLM003": "calls a bare provider completion API",
    "LLM004": "hand-rolls an Anthropic Messages HTTP call",
    "LLM005": "hand-rolls an OpenAI chat-completions HTTP call",
}

_GUIDANCE = (
    "Sanctioned LLM paths:\n"
    "  * single-shot inference -> hyperloom.common.llm_config: build the client with\n"
    "    get_openai_client() / get_async_openai_client() and issue the call with\n"
    "    stream_chat_completion_text() / astream_chat_completion_text().\n"
    "  * agentic work -> an agent runtime: hyperloom.common.codex_session (Codex SDK)\n"
    "    or the Claude Agent SDK backends in hyperloom/orchestrator/roles/.\n"
    "Provider client construction lives in hyperloom/common/llm_config.py and nowhere else."
)

# Top-level packages of provider SDKs. ``claude_agent_sdk`` is deliberately
# absent: it is the sanctioned agent runtime, not a raw provider client.
_PROVIDER_SDK_MODULES = frozenset({"openai", "anthropic"})

_PROVIDER_CLIENT_CLASSES = frozenset(
    {
        "OpenAI",
        "AsyncOpenAI",
        "AzureOpenAI",
        "AsyncAzureOpenAI",
        "Anthropic",
        "AsyncAnthropic",
    }
)

# Endpoint path fragments that identify a hand-rolled provider HTTP call. The
# workload-probe endpoint ``/v1/completions`` is intentionally not listed: those
# probes talk to the inference server under optimization, not to an LLM
# provider, so they are not LLM interactions.
_HTTP_ENDPOINT_RULES: tuple[tuple[str, str], ...] = (
    ("/v1/messages", "LLM004"),
    ("/chat/completions", "LLM005"),
)

# ---------------------------------------------------------------------------
# Scan surface
# ---------------------------------------------------------------------------

# First-party source trees. A new top-level tree of Hyperloom code has to be
# added here, otherwise the guard silently stops covering it.
_SCAN_ROOTS: tuple[str, ...] = (
    "src/hyperloom",
    "scripts",
    "docs",
    "examples",
    "OOB",  # optional component; not always present in a clone (see CLAUDE.md)
)

# Directory names that never hold first-party sources: build output, caches,
# and third-party checkouts an operator may materialize inside the repo (the
# dependency checkout cache defaults to ``$REPO_ROOT/.cache``). ``ruff``'s
# ``extend-exclude`` skips the vendored trees for the same reason.
_PRUNED_DIR_NAMES = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
        "_audit_artifacts",
        "_bypass_repo_scan_fixture",  # ephemeral kernel-resolver test fixture
        "TraceLens-internal",
        "InferenceX",
    }
)

_TEST_DIR_NAMES = frozenset({"tests", "test", "testing"})

# The only modules allowed to own provider access, as repo-relative POSIX paths.
# Test files are allowlisted separately by :func:`_is_test_file`, because they
# have to be able to build fakes and monkeypatch the real SDK symbols.
_ALLOWLISTED_OWNERS = frozenset(
    {
        # Single-shot inference + the sole home of client construction.
        "src/hyperloom/common/llm_config.py",
        # Codex agent runtime session wrapper.
        "src/hyperloom/common/codex_session.py",
    }
)

# ---------------------------------------------------------------------------
# Known violations -- see the module docstring. Empty, and shrink only.
# ---------------------------------------------------------------------------

_KNOWN_VIOLATIONS: dict[tuple[str, str], int] = {}

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Violation:
    """One offending construct: where it is, which rule it broke, and what it is."""

    path: str
    line: int
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.code} {_RULES[self.code]} -- {self.detail}"


def _called_name(func: ast.expr) -> str:
    """Rightmost identifier of a call target (``a.b.C`` -> ``C``)."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Node ids of every docstring constant, which must not count as code."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def _has_http_post(tree: ast.AST) -> bool:
    """True when the module issues an HTTP POST (``httpx.post`` / ``client.post``).

    Deliberately shallow: it only gates the endpoint-literal rules so that a
    module merely naming a provider route (docs, tables) is not reported.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _called_name(node.func) == "post":
            return True
    return False


def _scan_source(text: str, path: str) -> list[_Violation]:
    """Return every architecture violation in one module's source."""
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:  # first-party sources must always parse
        raise AssertionError(f"{path}: unparseable, LLM architecture guard cannot cover it: {exc}") from exc

    found: list[_Violation] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _PROVIDER_SDK_MODULES:
                names = ", ".join(alias.name for alias in node.names)
                found.append(_Violation(path, node.lineno, "LLM001", f"from {node.module} import {names}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _PROVIDER_SDK_MODULES:
                    found.append(_Violation(path, node.lineno, "LLM001", f"import {alias.name}"))
        elif isinstance(node, ast.Call):
            name = _called_name(node.func)
            if name in _PROVIDER_CLIENT_CLASSES:
                found.append(_Violation(path, node.lineno, "LLM002", f"{name}(...)"))
            elif name == "create" and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if isinstance(owner, ast.Attribute) and owner.attr == "responses":
                    found.append(_Violation(path, node.lineno, "LLM003", ".responses.create(...)"))
                elif (
                    isinstance(owner, ast.Attribute)
                    and owner.attr == "completions"
                    and isinstance(owner.value, ast.Attribute)
                    and owner.value.attr == "chat"
                ):
                    found.append(_Violation(path, node.lineno, "LLM003", ".chat.completions.create(...)"))

    if _has_http_post(tree):
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings:
                continue
            for fragment, code in _HTTP_ENDPOINT_RULES:
                if fragment in node.value:
                    found.append(_Violation(path, node.lineno, code, f"POST to {fragment!r}"))

    return sorted(found, key=lambda v: (v.line, v.code, v.detail))


# ---------------------------------------------------------------------------
# Tree walk
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path | None:
    """Walk up for the pyproject.toml; returns None when installed from a wheel."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    return None


_REPO_ROOT = _find_repo_root()

pytestmark = pytest.mark.skipif(
    _REPO_ROOT is None,
    reason="LLM architecture guard needs the source checkout (pyproject.toml + src/)",
)


def _is_pruned(relative: Path) -> bool:
    return any(part in _PRUNED_DIR_NAMES or part.endswith(".egg-info") for part in relative.parts)


def _is_test_file(relative: Path) -> bool:
    if any(part in _TEST_DIR_NAMES for part in relative.parts):
        return True
    name = relative.name
    return name == "conftest.py" or name.startswith("test_") or name.endswith("_test.py")


def _scan_tree() -> list[_Violation]:
    """Scan every first-party source file outside the allowlist."""
    assert _REPO_ROOT is not None
    found: list[_Violation] = []
    for root_name in _SCAN_ROOTS:
        root = _REPO_ROOT / root_name
        if not root.is_dir():
            continue
        try:
            paths = sorted(root.rglob("*.py"))
        except OSError:
            # A parallel test can tear down a fixture directory while this walk
            # is in flight; skip the root rather than fail the architecture guard.
            continue
        for path in paths:
            relative = path.relative_to(_REPO_ROOT)
            posix = relative.as_posix()
            if _is_pruned(relative) or _is_test_file(relative) or posix in _ALLOWLISTED_OWNERS:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            found.extend(_scan_source(text, posix))
    return found


def _ratchet_problems(
    violations: list[_Violation],
    pinned_counts: dict[tuple[str, str], int],
) -> list[str]:
    """Compare a scan against the pin map, both directions.

    Returns one human-readable problem line per drifting ``(path, rule)``: an
    unpinned or under-pinned occurrence is new debt, an over-pinned one is a
    migrated violation whose pin has to go. Empty means the pin map is exact.
    """
    by_key: dict[tuple[str, str], list[_Violation]] = {}
    for violation in violations:
        by_key.setdefault((violation.path, violation.code), []).append(violation)

    problems: list[str] = []
    for key in sorted(set(by_key) | set(pinned_counts)):
        sites = by_key.get(key, [])
        actual = len(sites)
        pinned = pinned_counts.get(key, 0)
        if actual == pinned:
            continue
        if actual > pinned:
            listing = "\n".join(f"      {site}" for site in sites[pinned:])
            problems.append(f"  NEW unsanctioned LLM access ({actual} found, {pinned} pinned):\n{listing}")
        else:
            problems.append(
                f"  STALE pin {key[0]} {key[1]} ({pinned} pinned, {actual} found): "
                "the violation was migrated -- shrink _KNOWN_VIOLATIONS to match."
            )
    return problems


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_no_unsanctioned_llm_provider_access() -> None:
    """Every LLM call site is either sanctioned or a pinned, shrinking violation."""
    problems = _ratchet_problems(_scan_tree(), _KNOWN_VIOLATIONS)
    if problems:
        pytest.fail(
            "LLM provider access is centralized; this scan found drift:\n"
            + "\n".join(problems)
            + "\n\n"
            + _GUIDANCE
            + "\n\nA NEW finding must be fixed, never pinned. A STALE pin must be removed:\n"
            f"see the 'Known violations' section of {Path(__file__).name}.",
            pytrace=False,
        )


def test_single_shot_owner_exists() -> None:
    """The allowlist must not rot: the single-shot inference owner has to exist.

    ``codex_session.py`` is allowlisted ahead of its arrival and so is not
    asserted here; every other entry must be a real module, otherwise the
    allowlist is protecting a path nobody uses.
    """
    assert _REPO_ROOT is not None
    owner = _REPO_ROOT / "src/hyperloom/common/llm_config.py"
    assert owner.is_file(), f"sanctioned single-shot inference owner is missing: {owner}"


def test_pinned_violations_are_scannable_paths() -> None:
    """Every pinned path is a file the scan actually visits (guards typos)."""
    assert _REPO_ROOT is not None
    for path, code in sorted(_KNOWN_VIOLATIONS):
        assert code in _RULES, f"_KNOWN_VIOLATIONS pins unknown rule {code!r} for {path}"
        relative = Path(path)
        assert (_REPO_ROOT / relative).is_file(), f"_KNOWN_VIOLATIONS pins a missing file: {path}"
        assert any(path.startswith(f"{root}/") for root in _SCAN_ROOTS), (
            f"_KNOWN_VIOLATIONS pins {path}, which is outside _SCAN_ROOTS"
        )
        assert not _is_test_file(relative), f"_KNOWN_VIOLATIONS pins a test file: {path}"
        assert path not in _ALLOWLISTED_OWNERS, f"_KNOWN_VIOLATIONS pins an allowlisted owner: {path}"


# ---------------------------------------------------------------------------
# Ratchet self-tests -- the pin map has to bite in both directions.
# ---------------------------------------------------------------------------

_PROBE = _Violation("src/hyperloom/probe.py", 3, "LLM002", "OpenAI(...)")


def test_ratchet_accepts_an_exact_pin() -> None:
    assert _ratchet_problems([_PROBE], {("src/hyperloom/probe.py", "LLM002"): 1}) == []


def test_ratchet_rejects_an_unpinned_violation() -> None:
    (problem,) = _ratchet_problems([_PROBE], {})
    assert "NEW" in problem
    assert "src/hyperloom/probe.py:3" in problem


def test_ratchet_rejects_an_extra_occurrence_in_a_pinned_file() -> None:
    extra = _Violation("src/hyperloom/probe.py", 9, "LLM002", "OpenAI(...)")
    (problem,) = _ratchet_problems([_PROBE, extra], {("src/hyperloom/probe.py", "LLM002"): 1})
    assert "NEW" in problem
    assert "src/hyperloom/probe.py:9" in problem


def test_ratchet_rejects_a_pin_that_outlived_its_violation() -> None:
    (problem,) = _ratchet_problems([], {("src/hyperloom/probe.py", "LLM002"): 1})
    assert "STALE" in problem


def test_ratchet_rejects_a_partially_migrated_pin() -> None:
    """Fixing one of two occurrences still forces the count down."""
    (problem,) = _ratchet_problems([_PROBE], {("src/hyperloom/probe.py", "LLM002"): 2})
    assert "STALE" in problem
    assert "2 pinned, 1 found" in problem


# ---------------------------------------------------------------------------
# Detector self-tests -- a guard that cannot detect anything passes vacuously.
# ---------------------------------------------------------------------------


def _codes(source: str) -> list[str]:
    return [v.code for v in _scan_source(source, "probe.py")]


def test_detects_sdk_import_forms() -> None:
    assert _codes("from openai import OpenAI\n") == ["LLM001"]
    assert _codes("from openai import AsyncOpenAI  # type: ignore\n") == ["LLM001"]
    assert _codes("import openai\n") == ["LLM001"]
    assert _codes("import openai.types as t\n") == ["LLM001"]
    assert _codes("from anthropic import AsyncAnthropic\n") == ["LLM001"]


def test_detects_client_construction() -> None:
    assert _codes("c = OpenAI(api_key='k')\n") == ["LLM002"]
    assert _codes("c = openai.AsyncOpenAI(**kwargs)\n") == ["LLM002"]
    assert _codes("c = AsyncAzureOpenAI()\n") == ["LLM002"]


def test_detects_bare_completion_calls() -> None:
    assert _codes("client.chat.completions.create(model='m')\n") == ["LLM003"]
    assert _codes("await self._client.chat.completions.create(**kw)\n") == ["LLM003"]
    assert _codes("self._client.responses.create(**params)\n") == ["LLM003"]


def test_detects_hand_rolled_provider_http() -> None:
    assert _codes("httpx.post(base + '/v1/messages', json=body)\n") == ["LLM004"]
    assert _codes("url = base + '/v1/messages'\nhttpx.post(url, json=body)\n") == ["LLM004"]
    assert _codes("await client.post('/chat/completions', json=payload)\n") == ["LLM005"]
    assert _codes("httpx.post(f'{base}/v1/messages', json=body)\n") == ["LLM004"]


def test_ignores_sanctioned_and_unrelated_code() -> None:
    # The sanctioned agent runtime is not a raw provider client.
    assert _codes("from claude_agent_sdk import query\n") == []
    # llm_config's streaming helpers are the sanctioned single-shot call path.
    assert _codes("from hyperloom.common import llm_config\nllm_config.get_openai_client()\n") == []
    assert _codes("text, _ = llm_config.stream_chat_completion_text(client, model='m')\n") == []
    # Workload probes POST to the inference server under optimization.
    assert _codes("client.post(front + '/v1/completions', json=body)\n") == []
    # A route named in prose must not trip the endpoint rules.
    assert _codes('"""Appends /v1/messages to base_url."""\nhttpx.post(url)\n') == []
    assert _codes("def f():\n    '''POST /chat/completions.'''\n    httpx.post(url)\n") == []
    # Endpoint literals without any POST in the module are documentation.
    assert _codes("ROUTE = '/v1/messages'\n") == []


def test_reports_actionable_location() -> None:
    source = "import os\n\n\nfrom openai import AsyncOpenAI\n"
    (violation,) = _scan_source(source, "src/hyperloom/x.py")
    assert violation.line == 4
    assert violation.path == "src/hyperloom/x.py"
    assert "AsyncOpenAI" in str(violation)
