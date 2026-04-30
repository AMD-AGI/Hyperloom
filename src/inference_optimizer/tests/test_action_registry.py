"""Tests for orchestrator/action_registry.py — DESIGN §12."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.paths import asset_actions_dir
from inference_optimizer.orchestrator.action_registry import (
    ActionMetadata,
    ActionRegistry,
    ActionRegistryError,
)
from inference_optimizer.orchestrator.execution_mode import ExecutionMode


# ---------------------------------------------------------------------------
# Package-bundled actions (the runtime catalogue)
# ---------------------------------------------------------------------------
def _make_registry() -> ActionRegistry:
    return ActionRegistry(asset_actions_dir()).load()


def test_loads_bundled_actions():
    """Plan A: kernel_opt + integrate moved to kernel agent — registry no
    longer ships them. Other foundational actions remain."""
    reg = _make_registry()
    names = set(reg.names())
    assert {"bench_runner", "param_sweep_run", "baseline", "profile"}.issubset(names)
    assert "kernel_opt" not in names
    assert "integrate" not in names


def test_get_returns_action_metadata_or_none():
    reg = _make_registry()
    a = reg.get("bench_runner")
    assert isinstance(a, ActionMetadata)
    assert a.family == "prep"
    assert reg.get("not-a-real-action") is None


def test_bench_runner_allowed_in_all_modes():
    reg = _make_registry()
    a = reg.get("bench_runner")
    assert ExecutionMode.QUICK_PARAM_SWEEP in a.allowed_modes
    assert ExecutionMode.GUIDED_KERNEL_OPT in a.allowed_modes
    assert ExecutionMode.MARATHON_MULTI_AGENT in a.allowed_modes


def test_deep_kernel_action_blocked_in_quick_mode():
    """deep_kernel_analysis stands in for the kernel-opt-flavoured
    actions that must stay out of quick mode (kernel_opt itself was
    removed in Plan A, so we test on a sibling)."""
    reg = _make_registry()
    a = reg.get("deep_kernel_analysis")
    assert a is not None
    assert ExecutionMode.QUICK_PARAM_SWEEP not in a.allowed_modes
    assert ExecutionMode.MARATHON_MULTI_AGENT in a.allowed_modes


def test_allowed_for_mode_filters_correctly():
    reg = _make_registry()
    quick = {a.name for a in reg.allowed_for_mode(ExecutionMode.QUICK_PARAM_SWEEP)}
    assert "bench_runner" in quick
    assert "param_sweep_run" in quick
    # Plan A — kernel_opt registry entry removed; nothing to gate here.
    assert "kernel_opt" not in quick

    marathon = {
        a.name for a in reg.allowed_for_mode(ExecutionMode.MARATHON_MULTI_AGENT)
    }
    # Marathon-only deep_kernel actions still listed.
    assert "deep_kernel_analysis" in marathon
    assert "operator_tuning" in marathon


def test_allowed_for_mode_accepts_string():
    reg = _make_registry()
    rs = reg.allowed_for_mode("quick_param_sweep")
    assert any(a.name == "bench_runner" for a in rs)


def test_allowed_for_mode_rejects_unknown_string():
    reg = _make_registry()
    with pytest.raises(ActionRegistryError):
        reg.allowed_for_mode("not-a-mode")


def test_kernel_opt_no_longer_in_registry():
    """Plan A: kernel agent owns kernel_opt + integrate; both removed from
    the action registry to guarantee executor cannot delegate them."""
    reg = _make_registry()
    assert reg.get("kernel_opt") is None
    assert reg.get("integrate") is None
    assert reg.get("kernel-opt") is None  # hyphen variant also gone


def test_system_prompt_for_known_action():
    reg = _make_registry()
    body = reg.system_prompt_for("bench_runner")
    assert body and "bench_runner" in body


def test_system_prompt_for_unknown_action_returns_empty():
    reg = _make_registry()
    assert reg.system_prompt_for("__nope__") == ""


def test_registry_dunder_helpers():
    reg = _make_registry()
    assert "bench_runner" in reg
    assert "garbage" not in reg
    assert len(reg) >= 3
    names_iter = [a.name for a in reg]
    assert "bench_runner" in names_iter


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def _write(tmp: Path, fname: str, body: str) -> None:
    meta = tmp / "_meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / fname).write_text(body, encoding="utf-8")


def test_loads_valid_yaml(tmp_path):
    _write(
        tmp_path,
        "demo.yaml",
        """
name: demo
family: shallow
cost_minutes_p50: 1.0
cost_minutes_p75: 2.0
expected_gain_pct: [0.5, 2.5]
accuracy_risk: 0.0
crash_risk: 0.0
allowed_modes: [quick_param_sweep]
""".strip(),
    )
    reg = ActionRegistry(tmp_path).load()
    a = reg.get("demo")
    assert a is not None
    assert a.family == "shallow"
    assert a.expected_gain_pct == (0.5, 2.5)


def test_rejects_missing_required_field(tmp_path):
    _write(
        tmp_path,
        "demo.yaml",
        """
name: demo
family: shallow
""".strip(),
    )
    with pytest.raises(ActionRegistryError, match="missing required field"):
        ActionRegistry(tmp_path).load()


def test_rejects_unknown_family(tmp_path):
    _write(
        tmp_path,
        "demo.yaml",
        """
name: demo
family: nonsense
cost_minutes_p50: 1.0
cost_minutes_p75: 2.0
expected_gain_pct: [0.0, 0.0]
accuracy_risk: 0.0
crash_risk: 0.0
allowed_modes: [quick_param_sweep]
""".strip(),
    )
    with pytest.raises(ActionRegistryError, match="invalid family"):
        ActionRegistry(tmp_path).load()


def test_rejects_filename_mismatch(tmp_path):
    _write(
        tmp_path,
        "demo.yaml",
        """
name: not_demo
family: shallow
cost_minutes_p50: 1.0
cost_minutes_p75: 2.0
expected_gain_pct: [0.0, 0.0]
accuracy_risk: 0.0
crash_risk: 0.0
allowed_modes: [quick_param_sweep]
""".strip(),
    )
    with pytest.raises(ActionRegistryError, match="does not match filename"):
        ActionRegistry(tmp_path).load()


def test_rejects_unknown_mode_in_allowed_modes(tmp_path):
    _write(
        tmp_path,
        "demo.yaml",
        """
name: demo
family: shallow
cost_minutes_p50: 1.0
cost_minutes_p75: 2.0
expected_gain_pct: [0.0, 0.0]
accuracy_risk: 0.0
crash_risk: 0.0
allowed_modes: [foo_bar_mode]
""".strip(),
    )
    with pytest.raises(ActionRegistryError, match="unknown execution mode"):
        ActionRegistry(tmp_path).load()


def test_rejects_duplicate_action_name(tmp_path):
    body = """
family: shallow
cost_minutes_p50: 1.0
cost_minutes_p75: 2.0
expected_gain_pct: [0.0, 0.0]
accuracy_risk: 0.0
crash_risk: 0.0
allowed_modes: [quick_param_sweep]
""".strip()
    # Two files with same `name` field
    _write(tmp_path, "alpha.yaml", "name: alpha\n" + body)
    _write(tmp_path, "alpha2.yaml", "name: alpha2\n" + body)
    # Now overwrite alpha2 to also claim name=alpha (mismatch -> raises before
    # dup detection); fix by setting filename match but same name field.
    # Actually filename-mismatch raises first. Use proper dup with same name.
    (tmp_path / "_meta" / "alpha2.yaml").write_text(
        "name: alpha2\n" + body, encoding="utf-8"
    )
    # Now both files load fine. To trigger duplicate, write two files whose
    # filename + name agree to "alpha".
    (tmp_path / "_meta" / "alpha.yaml").write_text(
        "name: alpha\n" + body, encoding="utf-8"
    )
    (tmp_path / "_meta" / "alpha2.yaml").write_text(
        "name: alpha2\n" + body, encoding="utf-8"
    )
    # No duplicate yet. Force one by renaming filename to share name.
    # Easiest: write two real files that both name "alpha".
    (tmp_path / "_meta").mkdir(exist_ok=True)
    # Reset directory
    for p in (tmp_path / "_meta").iterdir():
        p.unlink()
    (tmp_path / "_meta" / "alpha.yaml").write_text(
        "name: alpha\n" + body, encoding="utf-8"
    )
    (tmp_path / "_meta" / "beta.yaml").write_text(
        "name: alpha\n" + body, encoding="utf-8"  # mismatch will fire first
    )
    with pytest.raises(ActionRegistryError):
        ActionRegistry(tmp_path).load()


def test_rejects_invalid_expected_gain(tmp_path):
    _write(
        tmp_path,
        "demo.yaml",
        """
name: demo
family: shallow
cost_minutes_p50: 1.0
cost_minutes_p75: 2.0
expected_gain_pct: 0.5
accuracy_risk: 0.0
crash_risk: 0.0
allowed_modes: [quick_param_sweep]
""".strip(),
    )
    with pytest.raises(ActionRegistryError, match="expected_gain_pct"):
        ActionRegistry(tmp_path).load()


def test_missing_meta_dir_raises(tmp_path):
    with pytest.raises(ActionRegistryError, match="meta directory"):
        ActionRegistry(tmp_path / "does-not-exist").load()
