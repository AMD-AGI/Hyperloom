# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ActionRegistry + ActionMetadata + PolicyGate integration tests."""

from __future__ import annotations


import pytest
import yaml

from hyperloom.orchestrator.actions.registry import (
    ActionMetadata,
    ActionRegistry,
    ActionRegistryError,
    VALID_FAMILIES,
)
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.policy.gate import PolicyDenied, PolicyGate


# ActionMetadata schema
def _good_payload() -> dict:
    return {
        "name": "baseline",
        "family": "prep",
        "cost_minutes_p50": 5.0,
        "cost_minutes_p75": 10.0,
        "expected_gain_pct": [0.0, 0.0],
        "accuracy_risk": 0.0,
        "crash_risk": 0.05,
        "requires_lanes": ["server_lifecycle", "benchmark_lane"],
        "allowed_tools": ["emit_intent", "Read", "Bash"],
    }


def test_metadata_from_yaml_dict_minimal_ok():
    meta = ActionMetadata.from_yaml_dict(_good_payload(), expected_name="baseline")
    assert meta.name == "baseline"
    assert meta.family == "prep"
    assert meta.requires_lanes == ("server_lifecycle", "benchmark_lane")
    assert meta.allowed_tools == ("emit_intent", "Read", "Bash")
    assert meta.preferred_backend == "claude"


def test_metadata_unknown_family_rejected():
    bad = _good_payload()
    bad["family"] = "weird_family"
    with pytest.raises(ActionRegistryError, match="family"):
        ActionMetadata.from_yaml_dict(bad, expected_name="baseline")


def test_metadata_filename_mismatch_rejected():
    with pytest.raises(ActionRegistryError, match="does not match"):
        ActionMetadata.from_yaml_dict(_good_payload(), expected_name="other_name")


def test_metadata_accuracy_risk_out_of_range_rejected():
    bad = _good_payload()
    bad["accuracy_risk"] = 1.5
    with pytest.raises(ActionRegistryError, match="accuracy_risk"):
        ActionMetadata.from_yaml_dict(bad, expected_name="baseline")


def test_metadata_expected_gain_pct_must_be_pair():
    bad = _good_payload()
    bad["expected_gain_pct"] = [5.0]
    with pytest.raises(ActionRegistryError, match="expected_gain_pct"):
        ActionMetadata.from_yaml_dict(bad, expected_name="baseline")


def test_metadata_no_allowed_modes_field():
    """allowed_modes is not accepted because full mode is the only supported mode."""
    payload = _good_payload()
    payload["allowed_modes"] = ["quick"]  # extra fields silently ignored
    meta = ActionMetadata.from_yaml_dict(payload, expected_name="baseline")
    assert not hasattr(meta, "allowed_modes")


def test_valid_families_v06():
    assert VALID_FAMILIES == frozenset(
        {
            "prep",
            "analysis",
            "shallow",
            "deep_kernel",
            "long",
            "creative",
            "resilience",
        }
    )


# ActionRegistry — loads shipped actions
@pytest.fixture
def registry() -> ActionRegistry:
    return ActionRegistry().load()


def test_registry_loads_p1_1_actions(registry):
    names = {meta.name for meta in registry.all()}
    expected = {"baseline", "target_analysis", "explore", "report"}
    assert expected.issubset(names)


def test_registry_baseline_metadata(registry):
    baseline = registry.get("baseline")
    assert baseline is not None
    assert baseline.family == "prep"
    assert "server_lifecycle" in baseline.requires_lanes
    assert "benchmark_lane" in baseline.requires_lanes
    assert "Bash" in baseline.allowed_tools
    assert baseline.accuracy_risk == 0.0
    assert baseline.lease_ttl_sec == 4200


def test_registry_get_unknown_returns_none(registry):
    assert registry.get("does_not_exist") is None


def test_registry_load_missing_dir_raises(tmp_path):
    bogus = ActionRegistry(actions_dir=tmp_path / "nope")
    with pytest.raises(ActionRegistryError, match="not found"):
        bogus.load()


def test_registry_validates_filename_matches_yaml_name(tmp_path):
    """Mismatched filename ↔ yaml name field should fail loud."""
    actions_dir = tmp_path / "actions"
    meta_dir = actions_dir / "_meta"
    meta_dir.mkdir(parents=True)
    bad_path = meta_dir / "bench_runner.yaml"
    bad_path.write_text(
        yaml.safe_dump(
            {
                "name": "wrong_name",
                "family": "prep",
                "cost_minutes_p50": 1.0,
                "cost_minutes_p75": 2.0,
                "expected_gain_pct": [0.0, 0.0],
                "accuracy_risk": 0.0,
                "crash_risk": 0.0,
            }
        )
    )
    reg = ActionRegistry(actions_dir=actions_dir)
    with pytest.raises(ActionRegistryError, match="does not match"):
        reg.load()


# PolicyGate × ActionRegistry integration
@pytest.fixture
def gate_with_registry(registry) -> PolicyGate:
    return PolicyGate(role_registry=default_role_registry(), action_registry=registry)


def test_gate_delegate_known_action_ok(gate_with_registry):
    gate_with_registry.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.DELEGATE,
            payload={"action_name": "baseline"},
        ),
    )


def test_gate_delegate_unknown_action_rejected(gate_with_registry):
    with pytest.raises(PolicyDenied) as exc:
        gate_with_registry.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.DELEGATE,
                payload={"action_name": "make_coffee"},
            ),
        )
    assert exc.value.rule == "unknown_action"


def test_gate_delegate_kernel_owned_still_kernel_owned_check(gate_with_registry):
    """Even with registry wired, kernel_agent-owned actions get the kernel_owned reject."""
    with pytest.raises(PolicyDenied) as exc:
        gate_with_registry.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.DELEGATE,
                payload={"action_name": "kernel_opt"},
            ),
        )
    assert exc.value.rule == "kernel_owned_by_kernel_agent"


def test_gate_propose_action_unknown_rejected(gate_with_registry):
    with pytest.raises(PolicyDenied) as exc:
        gate_with_registry.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={"action_name": "bogus_action", "predicted_gain_pct": 5.0},
            ),
        )
    assert exc.value.rule == "unknown_action"


def test_gate_propose_action_kernel_owned_rejected(gate_with_registry):
    """Kernel-owned actions are REQUEST-only: propose_action is gated exactly
    like delegate (symmetry), so a proposal cannot bypass the kernel REQUEST
    handler by materializing as a kind=<kernel action> task."""
    with pytest.raises(PolicyDenied) as exc:
        gate_with_registry.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={"action_name": "kernel_opt", "predicted_gain_pct": 12.0},
            ),
        )
    assert exc.value.rule == "kernel_owned_by_kernel_agent"


def test_gate_delegate_unknown_action_no_registry_passes():
    """Without a registry wired, PolicyGate falls back to permissive."""
    g = PolicyGate(role_registry=default_role_registry(), action_registry=None)
    g.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.DELEGATE,
            payload={"action_name": "anything_goes"},
        ),
    )
