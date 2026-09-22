# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Architecture guard: duplication patterns collapsed by lean/zgong/bs-3 must not regrow.

Rules:
  DUP001  Module-level ``_now_iso = now_iso`` alias (bare, no partial).
  DUP002  Inline boolean true-token set literal ``{"1", "true", "yes", "on"}``
          or its false-token mirror outside the canonical owners.
  DUP003  Private ``_to_float`` / ``_to_int`` / ``_float_or_none`` / ``_int_or_none``
          helper that coerces and returns ``None`` on failure, outside the canonical owner.

Ratchet semantics (per-file counts):
  - count > pin  →  NEW violation, CI fails
  - count < pin  →  STALE pin, please shrink _KNOWN_VIOLATIONS
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Repo root discovery
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    return None


_REPO_ROOT = _find_repo_root()

pytestmark = pytest.mark.skipif(
    _REPO_ROOT is None,
    reason="dedup guard needs the source checkout (pyproject.toml + src/)",
)

_SCAN_ROOTS: tuple[str, ...] = ("src/hyperloom", "src/kernelforge")

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
    }
)

_TEST_DIR_NAMES = frozenset({"tests", "test"})


def _is_pruned(relative: Path) -> bool:
    return any(part in _PRUNED_DIR_NAMES or part.endswith(".egg-info") for part in relative.parts)


def _is_test_file(relative: Path) -> bool:
    if any(part in _TEST_DIR_NAMES for part in relative.parts):
        return True
    name = relative.name
    return name == "conftest.py" or name.startswith("test_") or name.endswith("_test.py")


# ---------------------------------------------------------------------------
# DUP001 — bare now_iso alias
# ---------------------------------------------------------------------------

# Module-level _now_iso = now_iso (no functools.partial) outside common/timeutil.py.
_DUP001_ALLOWLIST = frozenset(
    {
        "src/hyperloom/common/timeutil.py",
    }
)

# Legitimate aliases with a precision argument (partial) that were intentionally left.
_DUP001_PARTIAL_ALLOWLIST = frozenset(
    {
        "src/hyperloom/orchestrator/actions/executors/explore.py",
        "src/hyperloom/orchestrator/actions/executors/integrate_patch.py",
        "src/hyperloom/orchestrator/actions/executors/roofline.py",
        "src/hyperloom/orchestrator/state/orchestration_memory.py",
    }
)


def _scan_dup001(text: str, path: str) -> int:
    """Count bare ``_now_iso = now_iso`` module-level assignments."""
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError:
        return 0
    count = 0
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if not (isinstance(t, ast.Name) and "_now_iso" in t.id):
                continue
            # bare alias: RHS is a plain Name (not a Call like functools.partial)
            if isinstance(node.value, ast.Name):
                count += 1
    return count


# ---------------------------------------------------------------------------
# DUP002 — inline boolean token sets
# ---------------------------------------------------------------------------

# Files that are the canonical owners or have documented exemptions.
_DUP002_ALLOWLIST = frozenset(
    {
        "src/hyperloom/common/env.py",
        "src/hyperloom/orchestrator/trace/trace_env.py",  # shim re-exports env_flag, keeps no literals
        "src/hyperloom/inference_optimizer/multi_node/scripts/_server_flag_denylist.py",
        "src/hyperloom/inference_optimizer/multi_node/scripts/_sglang_shape_gate.py",
    }
)

# Current allowable counts per file (pin-to-zero means "this file must stay clean").
# Files not listed here must have zero occurrences.
_DUP002_KNOWN: dict[str, int] = {
    "src/hyperloom/agents/kernel/tools/_io_utils.py": 1,
    "src/hyperloom/agents/kernel/tools/bypass_trace_analysis.py": 3,
    "src/hyperloom/agents/kernel/tools/tracelens_analysis.py": 1,
    "src/hyperloom/common/pr_monitor_urls.py": 1,
    "src/hyperloom/inference_optimizer/assets/host_probe/hl_host_probe.py": 1,
    "src/hyperloom/inference_optimizer/breakdown/collectors/_common.py": 2,
    "src/hyperloom/inference_optimizer/breakdown/session_package.py": 1,
    "src/hyperloom/inference_optimizer/cli/__init__.py": 2,
    "src/hyperloom/inference_optimizer/cli/quantization.py": 1,
    "src/hyperloom/inference_optimizer/multi_node/cli.py": 1,
    "src/hyperloom/inference_optimizer/multi_node/commands/infera.py": 1,
    "src/hyperloom/orchestrator/actions/executors/_accuracy_gate.py": 2,
    "src/hyperloom/orchestrator/actions/executors/_framework_rewrite_evidence.py": 2,
    "src/hyperloom/orchestrator/actions/executors/_grid_runner.py": 1,
    "src/hyperloom/orchestrator/actions/executors/_inferencex_patcher.py": 2,
    "src/hyperloom/orchestrator/actions/executors/_multi_node_env.py": 1,
    "src/hyperloom/orchestrator/actions/executors/_multi_node_server_lifecycle.py": 3,
    "src/hyperloom/orchestrator/actions/executors/_ray_backend.py": 5,
    "src/hyperloom/orchestrator/actions/executors/_subprocess_kill.py": 1,
    "src/hyperloom/orchestrator/actions/executors/_workload_envs.py": 1,
    "src/hyperloom/orchestrator/actions/executors/report.py": 1,
    "src/hyperloom/orchestrator/actions/executors/roofline.py": 1,
    "src/hyperloom/orchestrator/enablement/recipe/credentials.py": 2,
    "src/hyperloom/orchestrator/gpu_lanes.py": 1,
    "src/hyperloom/orchestrator/kernel/gemm_shape_coverage.py": 1,
    "src/hyperloom/orchestrator/kernel/request_handlers.py": 1,
    "src/hyperloom/orchestrator/loop/coordinator.py": 3,
    "src/hyperloom/orchestrator/loop/coordinator_helpers.py": 2,
    "src/hyperloom/orchestrator/loop/dispatcher.py": 2,
    "src/hyperloom/orchestrator/phases/explore.py": 1,
    "src/hyperloom/orchestrator/phases/kernel.py": 2,
    "src/hyperloom/orchestrator/policy/gate.py": 1,
    "src/hyperloom/orchestrator/specialists/profile.py": 2,
    "src/kernelforge/cli.py": 1,
    "src/kernelforge/config.py": 1,
    "src/kernelforge/gemm_tune/tune_robustness.py": 1,
}

_BOOL_TOKEN_PATTERN = re.compile(
    r'"1".*?"true".*?"yes".*?"on"'
    r'|"0".*?"false".*?"no".*?"off"'
    r'|{[^}]*"1"[^}]*"true"[^}]*}'
    r'|{[^}]*"0"[^}]*"false"[^}]*}',
    re.DOTALL,
)


def _scan_dup002(text: str) -> int:
    return len(_BOOL_TOKEN_PATTERN.findall(text))


# ---------------------------------------------------------------------------
# DUP003 — private numeric coercer helpers
# ---------------------------------------------------------------------------

_DUP003_ALLOWLIST = frozenset(
    {
        "src/hyperloom/common/coerce.py",
        "src/hyperloom/orchestrator/bus/resource_lock.py",  # _lease_iso — timestamp, not numeric coercer
    }
)

# Per-file allowable counts.
_DUP003_KNOWN: dict[str, int] = {
    "src/hyperloom/common/provenance.py": 1,  # existing pre-branch
    "src/hyperloom/inference_optimizer/baseline_comparison/inferencex_client.py": 1,  # existing pre-branch
    "src/hyperloom/inference_optimizer/breakdown/collectors/_common.py": 2,  # comma-stripping wrapper
    "src/hyperloom/orchestrator/actions/executors/_gpu_metrics.py": 1,  # defensive: int/float only
}

_COERCER_PATTERN = re.compile(
    r"def\s+_(?:to_float|to_int|float_or_none|int_or_none|_to_float|_to_int)\s*\(",
)


def _scan_dup003(text: str) -> int:
    return len(_COERCER_PATTERN.findall(text))


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def _scan_all() -> dict[str, dict[str, int]]:
    """Return {rule: {posix_path: count}} for every scanned file."""
    assert _REPO_ROOT is not None
    result: dict[str, dict[str, int]] = {"DUP001": {}, "DUP002": {}, "DUP003": {}}

    for root_name in _SCAN_ROOTS:
        root = _REPO_ROOT / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(_REPO_ROOT)
            posix = relative.as_posix()
            if _is_pruned(relative) or _is_test_file(relative):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue

            # DUP001
            if posix not in _DUP001_ALLOWLIST and posix not in _DUP001_PARTIAL_ALLOWLIST:
                n = _scan_dup001(text, posix)
                if n:
                    result["DUP001"][posix] = n

            # DUP002
            if posix not in _DUP002_ALLOWLIST:
                n = _scan_dup002(text)
                if n:
                    result["DUP002"][posix] = n

            # DUP003
            if posix not in _DUP003_ALLOWLIST:
                n = _scan_dup003(text)
                if n:
                    result["DUP003"][posix] = n

    return result


def _ratchet_problems(
    actual: dict[str, int],
    known: dict[str, int],
    rule: str,
) -> list[str]:
    problems: list[str] = []
    for path in sorted(set(actual) | set(known)):
        a = actual.get(path, 0)
        k = known.get(path, 0)
        if a == k:
            continue
        if a > k:
            problems.append(f"  NEW {rule} in {path} ({a} found, {k} allowed)")
        else:
            problems.append(
                f"  STALE {rule} pin for {path} ({k} pinned, {a} found): shrink _DUP002_KNOWN / _DUP003_KNOWN to match"
            )
    return problems


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_bare_now_iso_aliases() -> None:
    """No module outside common/timeutil.py should define a bare ``_now_iso = now_iso`` alias."""
    counts = _scan_all()["DUP001"]
    if counts:
        listing = "\n".join(f"  {p}:{n}" for p, n in sorted(counts.items()))
        pytest.fail(
            "DUP001: bare _now_iso alias found (use 'from hyperloom.common.timeutil import now_iso' directly):\n"
            + listing,
            pytrace=False,
        )


def test_no_new_inline_bool_token_sets() -> None:
    """Inline true/false token sets must not grow beyond the pinned allowance."""
    counts = _scan_all()["DUP002"]
    problems = _ratchet_problems(counts, _DUP002_KNOWN, "DUP002")
    if problems:
        pytest.fail(
            "DUP002: inline boolean token set (use env_flag / env_bool from common/env.py):\n" + "\n".join(problems),
            pytrace=False,
        )


def test_no_new_private_numeric_coercers() -> None:
    """Private _to_float/_to_int helpers must not grow beyond the pinned allowance."""
    counts = _scan_all()["DUP003"]
    problems = _ratchet_problems(counts, _DUP003_KNOWN, "DUP003")
    if problems:
        pytest.fail(
            "DUP003: private numeric coercer (use to_float/to_int from common/coerce.py):\n" + "\n".join(problems),
            pytrace=False,
        )
