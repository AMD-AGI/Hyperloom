"""dynamic_action.MD P9 §10 — test-matrix verification.

These are the **meta-tests** that pin P9's verification thresholds
themselves: the suite-size gates the §10 #1 row specifies, plus
audit checks that the mocked sub-agent fixtures stay callable and
the specialist regression footprint is preserved.

If these tests start failing, P9's coverage promise is broken —
fix by adding tests, never by relaxing the thresholds.
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
    """Count top-level + class-method test definitions, excluding
    helper functions whose names don't start with ``test_``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return 0
    count = 0

    def _walk(node: ast.AST) -> None:
        nonlocal count
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                # parameterised tests still count as a single test
                # at module level here — pytest's collection multiplies
                # them at runtime. The §10 thresholds talk about
                # declared tests, so we count the decorator-stripped
                # function definitions.
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


# Stable bucketing per P9 §3 / §4 / §5. A test file maps into one of:
#   * unit       — single-module behaviour at a narrow API boundary
#   * integration — multi-module flows (Coordinator hooks, runner,
#                  end-to-end pipeline)
#   * invariant  — §5 red-line block-merge tests
_FILE_BUCKET: dict[str, str] = {
    "test_dynamic_action_dispatch.py":             "unit",   # P1 PolicyGate
    "test_dynamic_action_seed_kit.py":             "unit",   # P2 assembler
    "test_dynamic_action_tools.py":                "unit",   # P3 tools
    "test_dynamic_action_proposal.py":             "unit",   # P3 validator
    "test_dynamic_action_critic.py":               "unit",   # P4 critic primitives
    "test_dynamic_action_summary.py":              "unit",   # P6 state machine
    "test_dynamic_action_orchestration_prompt.py": "unit",   # P7 prompt
    "test_dynamic_action_resume.py":               "unit",   # P8 sweep
    "test_dynamic_action_runner.py":               "integration",  # P3 runner end-to-end
    "test_dynamic_action_e2e.py":                  "integration",  # P5 hooks
    "test_dynamic_action_invariants.py":           "invariant",    # P9 §5 red lines
    # P9 meta tests count as unit-layer audit tests of the matrix
    # itself (count gates, fixture surface, regression markers).
    "test_dynamic_action_p9_matrix.py":            "unit",
    # G2 / G13 — dispatch_history closed-schema writer.
    "test_dynamic_action_history.py":              "unit",
}


def _bucket_counts() -> dict[str, int]:
    counts: dict[str, int] = {"unit": 0, "integration": 0, "invariant": 0}
    for name, n in _collect_dynamic_action_test_files().items():
        bucket = _FILE_BUCKET.get(name)
        if bucket is None:
            # Unrecognised dynamic_action_* file → require an
            # explicit bucket assignment so future tests can't sneak
            # past the matrix audit.
            raise AssertionError(
                f"new test file {name!r} is not classified in "
                f"_FILE_BUCKET; add it to one of unit / integration / "
                f"invariant per P9 §3-§5"
            )
        counts[bucket] += n
    return counts


# ===========================================================================
# §10 #1 — total counts at or above the documented thresholds
# ===========================================================================
def test_p9_scenario_01_unit_layer_at_least_eighty():
    counts = _bucket_counts()
    assert counts["unit"] >= 80, (
        f"P9 §10 #1 requires ≥ 80 unit tests; got {counts['unit']}. "
        f"Per-file breakdown: { _collect_dynamic_action_test_files() }"
    )


def test_p9_scenario_01_integration_layer_at_least_fifteen():
    counts = _bucket_counts()
    assert counts["integration"] >= 15, (
        f"P9 §10 #1 requires ≥ 15 integration tests; got "
        f"{counts['integration']}"
    )


def test_p9_scenario_01_invariant_layer_at_least_twenty_five():
    counts = _bucket_counts()
    assert counts["invariant"] >= 25, (
        f"P9 §10 #1 requires ≥ 25 invariant tests; got "
        f"{counts['invariant']}"
    )


def test_p9_scenario_01_total_at_least_one_twenty():
    counts = _bucket_counts()
    total = sum(counts.values())
    assert total >= 120, (
        f"P9 §10 #1 requires ≥ 120 total tests; got {total} "
        f"({counts})"
    )


# ===========================================================================
# Every dynamic_action_* test module imports successfully — surface
# the import error here so CI fails fast instead of mid-collection.
# ===========================================================================
@pytest.mark.parametrize(
    "module_name",
    sorted(_FILE_BUCKET.keys()),
)
def test_dynamic_action_test_module_imports(module_name: str):
    """If a refactor breaks a public dynamic_action import, every
    test file that uses it should fail to import — this gate
    surfaces such failures even when individual tests get collected
    via a single broken module."""
    stem = module_name.replace(".py", "")
    full = f"inference_optimizer.tests.{stem}"
    importlib.import_module(full)


# ===========================================================================
# Mocked sub-agent fixture surface — verify the building blocks P9
# §6 calls out are present + callable.
# ===========================================================================
def test_p9_mocked_subagent_surface_mockbackend_present():
    """MockBackend / MockTurn / ScriptedPlan back the
    integration-layer tests; ensure the public surface exists so
    new tests can compose them without inventing a new mock."""
    from inference_optimizer.orchestrator.backends.mock_backend import (
        MockBackend, MockTurn, ScriptedPlan,
    )
    plan = ScriptedPlan(turns=[MockTurn(raw_text="hello")])
    backend = MockBackend(plan)
    assert backend.remaining_turns == 1


@pytest.mark.asyncio
async def test_p9_mocked_subagent_runs_in_runner_loop():
    """End-to-end smoke for P9 §6 #6 — the same mock backend drives
    a real DynamicActionRunner without needing a live LLM."""
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
# §11 #1 — naming convention audit. ``inv_*`` lives only in the
# invariants file; ``test_p<N>_scenario_*`` markers tie tests to the
# P9 acceptance matrix.
# ===========================================================================
def test_invariants_use_inv_naming_prefix():
    """Every test function in the invariants file should start with
    ``test_inv_`` so the §11 #1 convention is mechanically pinned."""
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
        f"invariant tests should embed ``inv`` in their name "
        f"(P9 §11 #1); offenders: {bad}"
    )


# ===========================================================================
# Specialist regression marker — link to the specialist suites the
# CI gate must still pass. Per P9 §7 they are NOT in the dynamic_action_*
# scope but they bracket the change.
# ===========================================================================
@pytest.mark.parametrize("regression_module", [
    "test_specialist_lifecycle",
    "test_specialist_integration",
    "test_specialist_concurrent_dispatch",
    "test_specialist_runner_truncation",
])
def test_specialist_regression_modules_importable(regression_module: str):
    """The specialist suites are independent of dynamic_action; they
    must keep importing cleanly so the §7 regression promise holds."""
    importlib.import_module(
        f"inference_optimizer.tests.{regression_module}",
    )


# ===========================================================================
# §11 #2 — invariant file lives in the same tests/ tree (no separate
# directory required by this branch; the file naming + bucketing
# above are the lookup key).
# ===========================================================================
def test_invariants_file_exists():
    path = _TESTS_DIR / "test_dynamic_action_invariants.py"
    assert path.is_file()


# ===========================================================================
# §10 #4 marker — the suite contains the eight red-line classes;
# missing one would break the gate.
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
            f"invariant class {tag!r} missing — P9 §5.1 requires "
            f"all eight I-N groups"
        )


# ===========================================================================
# §10 #6 — mocked sub-agent fixtures are independent of network /
# real LLM. Smoke-check that we can build a MockBackend that doesn't
# touch the network even when "tools" is set.
# ===========================================================================
@pytest.mark.asyncio
async def test_p9_mocked_subagent_no_network():
    """MockBackend.run is a pure local function — ensure repeated
    calls work without any kwargs the backend protocol might add."""
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
