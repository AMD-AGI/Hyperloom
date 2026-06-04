"""Contract test: keep envelope tables aligned with inference_optimizer.

Skipped when ``inference_optimizer`` is not importable (the agent
package keeps it as a soft dependency). CI environments that run
against the upstream repo should add the inference_optimizer path to
``PYTHONPATH`` so the test executes.

Locations probed automatically:

* anything already on ``sys.path``
* ``~/lss/Hyperloom`` (the standard local checkout used in dev)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _try_import_inference_optimizer():
    candidate_roots = [
        Path.home() / "lss" / "Hyperloom",
        Path.home() / "Hyperloom-rs-build" / "Hyperloom",
    ]
    for root in candidate_roots:
        if (root / "inference_optimizer" / "orchestrator").is_dir():
            sys.path.insert(0, str(root))
            break
    try:
        import inference_optimizer.protocol.intent as upstream_ip
        import inference_optimizer.orchestrator.policy as upstream_policy
        import inference_optimizer.orchestrator.agent_role as upstream_role
    except ImportError:
        return None
    return upstream_ip, upstream_policy, upstream_role


_UPSTREAM = _try_import_inference_optimizer()
pytestmark = pytest.mark.skipif(
    _UPSTREAM is None,
    reason="inference_optimizer not importable; contract check skipped",
)


def test_intent_type_values_match_upstream():
    from robustness_agent.role.envelope import IntentType

    upstream_ip, _, _ = _UPSTREAM  # type: ignore[misc]
    assert {t.value for t in IntentType} == {
        t.value for t in upstream_ip.IntentType
    }


def test_payload_required_matches_upstream():
    from robustness_agent.role.envelope import IntentType, PAYLOAD_REQUIRED

    upstream_ip, _, _ = _UPSTREAM  # type: ignore[misc]
    upstream_table = upstream_ip._PAYLOAD_REQUIRED  # noqa: SLF001
    for it in IntentType:
        local = PAYLOAD_REQUIRED[it]
        upstream = upstream_table[upstream_ip.IntentType(it.value)]
        assert local == upstream, f"{it} drift: local={local} upstream={upstream}"


def test_robustness_only_intents_match_upstream():
    from robustness_agent.role.envelope import ROBUSTNESS_ONLY_INTENTS

    _, upstream_policy, _ = _UPSTREAM  # type: ignore[misc]
    upstream_set = {t.value for t in upstream_policy.ROBUSTNESS_ONLY_INTENTS}
    upstream_set.add("kill_task")  # upstream lists kill_task separately as KILL_TASK_SOURCE_ALLOWLIST
    local_set = {t.value for t in ROBUSTNESS_ONLY_INTENTS}
    assert local_set == upstream_set


def test_kill_task_scope_matches_upstream():
    from robustness_agent.role.envelope import KILL_TASK_ALLOWED_SCOPES

    _, upstream_policy, _ = _UPSTREAM  # type: ignore[misc]
    assert KILL_TASK_ALLOWED_SCOPES == upstream_policy.KILL_TASK_ALLOWED_SCOPES


def test_core_state_fields_match_upstream():
    from robustness_agent.role.envelope import CORE_STATE_FIELDS

    _, upstream_policy, _ = _UPSTREAM  # type: ignore[misc]
    assert CORE_STATE_FIELDS == upstream_policy.CORE_STATE_FIELDS
