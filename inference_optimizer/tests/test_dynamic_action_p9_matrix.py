# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Meta-tests pinning the dynamic_action test-suite thresholds.

Verifies the per-layer count gates (unit / integration / invariant),
the mocked sub-agent fixture surface, and the importability of the
specialist regression modules. If these tests fail, fix by adding
tests — never by relaxing the thresholds.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


# ===========================================================================
# Test inventory — walk the dynamic_action_* test modules and count
# ``def test_`` / ``async def test_`` declarations.
# ===========================================================================
_TESTS_DIR = Path(__file__).parent
_DYNAMIC_ACTION_TEST_GLOB = "test_dynamic_action_*.py"


def _count_tests_in_file(path: Path) -> int:
    """Count ``def test_`` / ``async def test_`` declarations,
    including class methods."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return 0
    count = 0

    def _walk(node: ast.AST) -> None:
        nonlocal count
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Parametrised tests count once each; runtime multiplication
            # is pytest's concern.
            if node.name.startswith("test_"):
                count += 1
        for child in ast.iter_child_nodes(node):
            _walk(child)

    _walk(tree)
    return count


def _collect_dynamic_action_test_files() -> dict[str, int]:
    out: dict[str, int] = {}
    for path in sorted(_TESTS_DIR.glob(_DYNAMIC_ACTION_TEST_GLOB)):
        out[path.name] = _count_tests_in_file(path)
    return out


# Bucketing for each dynamic_action_* test file:
#   * unit        — single-module behaviour at a narrow API boundary
#   * integration — multi-module flows (Coordinator hooks, runner,
#                    end-to-end pipeline)
#   * invariant   — red-line block-merge tests
_FILE_BUCKET: dict[str, str] = {
    "test_dynamic_action_dispatch.py":             "unit",
    "test_dynamic_action_seed_kit.py":             "unit",
    "test_dynamic_action_tools.py":                "unit",
    "test_dynamic_action_proposal.py":             "unit",
    "test_dynamic_action_critic.py":               "unit",
    "test_dynamic_action_summary.py":              "unit",
    "test_dynamic_action_orchestration_prompt.py": "unit",
    "test_dynamic_action_resume.py":               "unit",
    "test_dynamic_action_runner.py":               "integration",
    "test_dynamic_action_e2e.py":                  "integration",
    "test_dynamic_action_invariants.py":           "invariant",
    "test_dynamic_action_p9_matrix.py":            "unit",
    "test_dynamic_action_history.py":              "unit",
    "test_dynamic_action_cli.py":                  "unit",
}


def _bucket_counts() -> dict[str, int]:
    counts: dict[str, int] = {"unit": 0, "integration": 0, "invariant": 0}
    for name, n in _collect_dynamic_action_test_files().items():
        bucket = _FILE_BUCKET.get(name)
        if bucket is None:
            # New ``test_dynamic_action_*`` files must declare a bucket
            # so the matrix audit cannot be bypassed.
            raise AssertionError(
                f"new test file {name!r} is not classified in "
                f"_FILE_BUCKET; add it to unit / integration / invariant"
            )
        counts[bucket] += n
    return counts


# ===========================================================================
# Per-layer minimum counts.
# ===========================================================================
def test_unit_layer_at_least_eighty():
    counts = _bucket_counts()
    assert counts["unit"] >= 80, (
        f"unit-layer minimum is 80; got {counts['unit']}. "
        f"per-file breakdown: { _collect_dynamic_action_test_files() }"
    )


def test_integration_layer_at_least_fifteen():
    counts = _bucket_counts()
    assert counts["integration"] >= 15, (
        f"integration-layer minimum is 15; got {counts['integration']}"
    )


def test_invariant_layer_at_least_twenty_five():
    counts = _bucket_counts()
    assert counts["invariant"] >= 25, (
        f"invariant-layer minimum is 25; got {counts['invariant']}"
    )


def test_total_at_least_one_twenty():
    counts = _bucket_counts()
    total = sum(counts.values())
    assert total >= 120, (
        f"total minimum is 120; got {total} ({counts})"
    )


# ===========================================================================
# Every dynamic_action_* test module imports successfully so CI fails
# fast rather than mid-collection.
# ===========================================================================
@pytest.mark.parametrize(
    "module_name",
    sorted(_FILE_BUCKET.keys()),
)
def test_dynamic_action_test_module_imports(module_name: str):
    """Import every test file so a broken public import fails here
    rather than during collection of an individual test."""
    stem = module_name.replace(".py", "")
    full = f"inference_optimizer.tests.{stem}"
    importlib.import_module(full)


# ===========================================================================
# Mocked sub-agent fixture surface — required building blocks remain
# importable.
# ===========================================================================
def test_mocked_subagent_surface_mockbackend_present():
    """``MockBackend`` / ``MockTurn`` / ``ScriptedPlan`` remain
    importable; new tests should compose them rather than inventing
    fresh mocks."""
    from inference_optimizer.orchestrator.backends.mock_backend import (
        MockBackend, MockTurn, ScriptedPlan,
    )
    plan = ScriptedPlan(turns=[MockTurn(raw_text="hello")])
    backend = MockBackend(plan)
    assert backend.remaining_turns == 1


@pytest.mark.asyncio
async def test_mocked_subagent_runs_in_runner_loop():
    """Smoke test: a ``MockBackend`` drives :class:`DynamicActionRunner`
    end-to-end without a live LLM."""
    import json
    import tempfile
    from pathlib import Path
    from dataclasses import dataclass, field

    from inference_optimizer.orchestrator.dynamic_action_proposal import (
        DynamicRunnerTerminalState,
    )
    from inference_optimizer.orchestrator.dynamic_action_runner import (
        DynamicActionRunner,
    )
    from inference_optimizer.orchestrator.backends.mock_backend import (
        MockBackend, MockTurn, ScriptedPlan,
    )
    from inference_optimizer.orchestrator.sub_agent_runner import (
        RunnerContext,
    )
    from inference_optimizer.session_paths import (
        dynamic_action_artifact_dir,
    )

    sd = Path(tempfile.mkdtemp())
    dyn_id = "dyn-0-1"
    art = dynamic_action_artifact_dir(sd, dyn_id)
    art.mkdir(parents=True, exist_ok=True)
    (art / "spec.json").write_text(json.dumps({
        "dyn_id": dyn_id,
        "payload": {
            "motivation_gap_text": "m",
            "scope_domains": ["serving_specialist", "kernel_switch_specialist"],
            "side_effects_declared": ["framework_source"],
            "budget_hint": "medium",
        },
    }), encoding="utf-8")
    (art / "seed_kit.json").write_text(json.dumps({
        "motivation_gap_text": "m",
        "roofline_summary": "",
        "profile_keyslices": [],
        "kept_patches": [],
        "reverted_patches": [],
        "kb_pitfalls": [],
        "source_root_hints": [],
    }), encoding="utf-8")

    proposal_text = (
        "thinking\n```json\n"
        + json.dumps({
            "tool": "emit_proposal",
            "args": {
                "name": "p9_mock",
                "provenance": "dynamic",
                "patch_text": (
                    "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-o\n+n\n"
                ),
                "scope_domains": [
                    "serving_specialist", "kernel_switch_specialist",
                ],
                "cross_domain_rationale": (
                    "serving_specialist coupled with "
                    "kernel_switch_specialist; risk regression"
                ),
                "expected_qualitative_argument": "qualitative",
            },
        })
        + "\n```"
    )
    backend = MockBackend(ScriptedPlan(turns=[MockTurn(raw_text=proposal_text)]))
    runner = DynamicActionRunner(backend, framework_source_roots=())

    @dataclass
    class _StubTask:
        task_id: str = "t-mock"
        kind: str = "dynamic_action"
        params: dict = field(default_factory=lambda: {"dyn_id": "dyn-0-1"})

    ctx = RunnerContext(
        task=_StubTask(), lease=None, extra={"session_dir": str(sd)},
    )
    result = await runner.run(ctx)
    assert result.terminal_state == DynamicRunnerTerminalState.COMPLETED


# ===========================================================================
# Invariant naming convention — every test in the invariants file
# must embed ``inv`` in its name.
# ===========================================================================
def test_invariants_use_inv_naming_prefix():
    path = _TESTS_DIR / "test_dynamic_action_invariants.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.name.startswith("test_")
                and "inv" not in node.name
            ):
                bad.append(node.name)
    assert not bad, (
        f"invariant tests should embed ``inv`` in their name; "
        f"offenders: {bad}"
    )


# ===========================================================================
# Specialist regression suites must keep importing cleanly.
# ===========================================================================
@pytest.mark.parametrize("regression_module", [
    "test_specialist_lifecycle",
    "test_specialist_integration",
    "test_specialist_concurrent_dispatch",
    "test_specialist_runner_truncation",
])
def test_specialist_regression_modules_importable(regression_module: str):
    importlib.import_module(
        f"inference_optimizer.tests.{regression_module}",
    )


def test_invariants_file_exists():
    path = _TESTS_DIR / "test_dynamic_action_invariants.py"
    assert path.is_file()


# ===========================================================================
# Invariants file must declare all eight red-line classes.
# ===========================================================================
def test_invariants_cover_all_eight_red_lines():
    path = _TESTS_DIR / "test_dynamic_action_invariants.py"
    body = path.read_text(encoding="utf-8")
    for tag in (
        "TestInvariant_1_MicroBench",
        "TestInvariant_2_SharedStateProtection",
        "TestInvariant_3_ProvenanceLiteral",
        "TestInvariant_4_KernelOwnedDenial",
        "TestInvariant_5_NoServerNoMagpie",
        "TestInvariant_6_NoSelfMetric",
        "TestInvariant_7_IntegratePatchOnly",
        "TestInvariant_8_CrossDynIsolation",
    ):
        assert tag in body, (
            f"invariant class {tag!r} missing"
        )


# ===========================================================================
# ``MockBackend`` is network-free across repeated calls.
# ===========================================================================
@pytest.mark.asyncio
async def test_mocked_subagent_no_network():
    from inference_optimizer.orchestrator.backends.mock_backend import (
        MockBackend, MockTurn, ScriptedPlan,
    )
    plan = ScriptedPlan(turns=[
        MockTurn(raw_text="first"),
        MockTurn(raw_text="second"),
    ])
    backend = MockBackend(plan)
    r1 = await backend.run(prompt="x", system_prompt="y", tools=None)
    r2 = await backend.run(prompt="x", system_prompt="y", tools=None)
    assert r1.raw_text == "first"
    assert r2.raw_text == "second"
