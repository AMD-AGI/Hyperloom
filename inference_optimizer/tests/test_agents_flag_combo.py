"""Cover the four agent combinations selected by --no-kernel / --no-framework.

The flags are independent toggles:

  --no-kernel    : strips kernel-owned arms + profile + pmc_roofline
  --no-framework : strips the framework_pr arm

Together they yield 4 modes:
  both (default), kernel-only, framework-only, neither (pure params search).

We verify the contract at three layers:

* ``default_enabled_actions`` returns the right action set per combination
* ``_register_executors`` registers / skips ``framework_pr`` accordingly
* ``SharedState.framework_enabled`` round-trips through state.json so resume
  honours the persisted toggle even when CLI flags are absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from inference_optimizer.cli import _register_executors
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
    FULL_ENABLED_ACTIONS,
    NO_KERNEL_ENABLED_ACTIONS,
    default_enabled_actions,
)


# ---------------------------------------------------------------------------
# default_enabled_actions: 4 combinations
# ---------------------------------------------------------------------------
def test_default_enabled_both_default():
    """no flags → full kernel + framework set."""
    enabled = default_enabled_actions(no_kernel=False, no_framework=False)
    assert enabled == FULL_ENABLED_ACTIONS
    assert "framework_pr" in enabled
    assert "kernel_opt" in enabled


def test_default_enabled_kernel_only():
    """--no-framework only → kernel arms kept, framework_pr stripped."""
    enabled = default_enabled_actions(no_kernel=False, no_framework=True)
    assert "framework_pr" not in enabled
    assert "kernel_opt" in enabled
    assert "deep_kernel_analysis" in enabled
    # All other FULL entries (minus framework_pr) must be preserved
    assert set(enabled) == set(FULL_ENABLED_ACTIONS) - {"framework_pr"}


def test_default_enabled_framework_only():
    """--no-kernel only → kernel arms stripped, framework_pr kept (legacy)."""
    enabled = default_enabled_actions(no_kernel=True, no_framework=False)
    assert enabled == NO_KERNEL_ENABLED_ACTIONS
    assert "framework_pr" in enabled
    assert "kernel_opt" not in enabled


def test_default_enabled_neither():
    """--no-kernel + --no-framework → pure parameter-search."""
    enabled = default_enabled_actions(no_kernel=True, no_framework=True)
    assert "framework_pr" not in enabled
    assert "kernel_opt" not in enabled
    assert "baseline" in enabled
    assert "params" in enabled
    assert "backends" in enabled
    assert "validate_stack" in enabled
    assert set(enabled) == set(NO_KERNEL_ENABLED_ACTIONS) - {"framework_pr"}


# ---------------------------------------------------------------------------
# _register_executors: which kinds get wired in each combination
# ---------------------------------------------------------------------------
def _make_fake_coordinator() -> tuple[MagicMock, dict[str, object]]:
    """Return ``(coord, registered)`` capturing every ``register_executor`` call."""
    registered: dict[str, object] = {}
    coord = MagicMock()
    coord.sub = MagicMock()
    coord.sub.register_executor.side_effect = (
        lambda kind, fn: registered.__setitem__(kind, fn)
    )
    return coord, registered


def test_register_executors_both_default():
    coord, registered = _make_fake_coordinator()
    _register_executors(coord, no_kernel=False, no_framework=False)
    assert "framework_pr" in registered
    assert "kernel_opt" in registered
    assert "profile" in registered
    assert "baseline" in registered


def test_register_executors_no_framework():
    coord, registered = _make_fake_coordinator()
    _register_executors(coord, no_kernel=False, no_framework=True)
    assert "framework_pr" not in registered
    assert "kernel_opt" in registered
    assert "profile" in registered


def test_register_executors_no_kernel():
    coord, registered = _make_fake_coordinator()
    _register_executors(coord, no_kernel=True, no_framework=False)
    assert "framework_pr" in registered
    assert "kernel_opt" not in registered
    assert "profile" not in registered


def test_register_executors_neither():
    coord, registered = _make_fake_coordinator()
    _register_executors(coord, no_kernel=True, no_framework=True)
    assert "framework_pr" not in registered
    assert "kernel_opt" not in registered
    assert "profile" not in registered
    assert "baseline" in registered
    assert "params" in registered


# ---------------------------------------------------------------------------
# SharedState persistence: framework_enabled must round-trip + default True
# ---------------------------------------------------------------------------
def test_shared_state_default_framework_enabled_is_true():
    s = SharedState()
    assert s.framework_enabled is True


def test_shared_state_framework_enabled_roundtrips(tmp_path: Path):
    s = SharedState(session_id="x", framework_enabled=False, kernel_enabled=False)
    s.save(tmp_path)
    raw = json.loads((tmp_path / "state.json").read_text())
    assert raw["framework_enabled"] is False
    assert raw["kernel_enabled"] is False
    reloaded = SharedState.load_or_init(tmp_path)
    assert reloaded.framework_enabled is False
    assert reloaded.kernel_enabled is False


def test_shared_state_legacy_state_json_defaults_framework_enabled_true(
    tmp_path: Path,
):
    """A state.json from an older build (no framework_enabled key) must load
    with framework_enabled=True so resumed runs keep the fa arm active."""
    legacy = {
        "session_id": "legacy",
        "model_name": "m",
        "kernel_enabled": True,
    }
    (tmp_path / "state.json").write_text(json.dumps(legacy))
    reloaded = SharedState.load_or_init(tmp_path)
    assert reloaded.framework_enabled is True
    assert reloaded.kernel_enabled is True


# ---------------------------------------------------------------------------
# Defensive: the four combinations together must cover every entry once
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("no_kernel", [False, True])
@pytest.mark.parametrize("no_framework", [False, True])
def test_all_combinations_yield_disjoint_or_subset_relationships(
    no_kernel: bool, no_framework: bool,
):
    enabled = set(default_enabled_actions(
        no_kernel=no_kernel, no_framework=no_framework,
    ))
    assert "baseline" in enabled
    assert "report" in enabled
    if no_kernel:
        assert "kernel_opt" not in enabled
        assert "profile" not in enabled
    if no_framework:
        assert "framework_pr" not in enabled
