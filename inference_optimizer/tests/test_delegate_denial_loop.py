"""Delegate idempotency + policy-denial ladder tests."""

from __future__ import annotations

import re

import pytest

from inference_optimizer.orchestrator.backends import MockBackend, ScriptedPlan
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.paths import make_session_dir


def _silent_coordinator(session_dir) -> Coordinator:
    silent = ScriptedPlan(turns=[])
    return Coordinator(
        session_dir,
        backends={
            "orchestration": MockBackend(silent, name="o"),
            "kernel": MockBackend(silent, name="k"),
            "critic": MockBackend(silent, name="c"),
            "robustness": MockBackend(silent, name="r"),
        },
    )


def _delegate(
    *,
    action: str = "long_running",
    params: dict | None = None,
    key: str | None = "dup-key-1",
) -> Intent:
    payload: dict = {"action_name": action, "params": params or {"x": 1}}
    if key is not None:
        payload["idempotency_key"] = key
    return Intent(type=IntentType.DELEGATE, payload=payload)


@pytest.fixture
def session_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


@pytest.mark.asyncio
async def test_delegate_terminal_collision_appends_retry_suffix(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        first, _ = await c.tasks.create_or_return_existing(
            kind="long_running",
            params={"x": 1},
            idempotency_key="dup-key-1",
        )
        await c.tasks.transition(first.task_id, "running", evidence={})
        await c.tasks.transition(first.task_id, "succeeded", evidence={})

        await c._handle_delegate("orchestration", _delegate(key="dup-key-1"))
        queued = await c.tasks.by_state("queued")
        keys = {t.idempotency_key for t in queued}
        assert "dup-key-1-retry1" in keys
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_running_collision_denies_without_new_task(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        await c.tasks.create_or_return_existing(
            kind="long_running",
            params={"x": 1},
            idempotency_key="dup-key-run",
        )
        before = len(await c.tasks.by_state("queued"))
        await c._handle_delegate("orchestration", _delegate(key="dup-key-run"))
        after = len(await c.tasks.by_state("queued"))
        assert after == before
        obs = await c.bus.tail(topic="observation")
        denied = [
            m for m in obs
            if m.payload.get("kind") == "policy_denied"
            and m.payload.get("rule") == "duplicate_idempotency_key_running"
        ]
        assert denied
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_fallback_key_uses_tick_and_content_fingerprint(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.tick = 42
        await c._handle_delegate(
            "orchestration",
            _delegate(key=None, params={"grid": [{"name": "a"}]}),
        )
        queued = await c.tasks.by_state("queued")
        assert len(queued) == 1
        key = queued[0].idempotency_key
        assert re.match(
            r"^orchestration:long_running:t42:[0-9a-f]{10}$",
            key,
        )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_policy_denial_streak_records_streak_at_two(session_dir):
    """v0.8 §3.9 — the v0.6 scoreboard's ``locked_reason`` was retired.
    The denial streak is still tracked via
    :attr:`SharedState.policy_denial_streak` (a pure fact); the LLM
    sees it as a count, not a priority lock."""
    c = _silent_coordinator(session_dir)
    try:
        from inference_optimizer.orchestrator.policy import PolicyDenied

        intent = _delegate(action="backends", key="k1")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        await c._record_policy_denied(
            "orchestration", intent, pd, action_name="backends",
        )
        await c._record_policy_denied(
            "orchestration", intent, pd, action_name="backends",
        )
        streak = c.shared_state.policy_denial_streak.get(
            "backends:duplicate_idempotency_key", 0,
        )
        assert streak >= 2
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_policy_denial_streak_prunes_family_at_five(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        from inference_optimizer.orchestrator.policy import PolicyDenied
        intent = _delegate(action="params", key="k1")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        for _ in range(5):
            await c._record_policy_denied(
                "orchestration", intent, pd, action_name="params",
            )
        assert "params" in c.shared_state.pruned_families
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_policy_denial_streak_sets_stop_reason_at_ten(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        from inference_optimizer.orchestrator.policy import PolicyDenied
        intent = _delegate(action="backends", key="k1")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        for _ in range(10):
            await c._record_policy_denied(
                "orchestration", intent, pd, action_name="backends",
            )
        assert c.shared_state.stop_reason == "policy_loop"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_successful_delegate_resets_policy_denial_streak(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        from inference_optimizer.orchestrator.policy import PolicyDenied
        intent = _delegate(key="k-reset")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        await c._record_policy_denied(
            "orchestration", intent, pd, action_name="long_running",
        )
        assert c.shared_state.policy_denial_streak.get(
            "long_running:duplicate_idempotency_key"
        ) == 1
        await c._handle_delegate("orchestration", _delegate(key="fresh-key"))
        assert not any(
            k.startswith("long_running:")
            for k in c.shared_state.policy_denial_streak
        )
    finally:
        await c.stop()


# ============================================================================
# Merged from test_v08_action_deprecated_rule.py
# ============================================================================

"""v0.8 M3 + KB_gaps/Gap-10 — ``action_deprecated`` rule tests.

KB_design §3.4 / §3.15 §2.3 merged the v0.6 ``backends`` / ``params``
/ ``validate_stack`` actions into a single ``explore``. KB_design
§3.13 M3 §PR7 closes the loop by adding a PolicyGate
``action_deprecated`` rule that denies the legacy names at the intent
boundary with a structured replacement hint.

This file exercises the rule against the three intent channels
PolicyGate guards (delegate / propose_action / request) and verifies
the supporting infrastructure (cli / phase_state / prompt_builder)
is consistent with the closure:

* :data:`policy.DEPRECATED_ACTION_NAMES` matches the §3.15 §2.3
  retirement list.
* :data:`policy.DEPRECATED_ACTION_REPLACEMENTS` maps each deprecated
  name to a non-empty replacement string.
* :func:`PolicyGate.validate_intent` raises ``PolicyDenied`` with
  ``rule='action_deprecated'`` for both ``delegate`` and
  ``propose_action`` carrying a deprecated ``action_name``, and the
  denial hint references the replacement action ``explore``.
* The rule fires *before* the kernel-owned / unknown-action /
  phase-incompatible checks (Inv-11.3 orthogonality).
* :data:`phase_state.PHASE_ALLOWED_ACTIONS[PHASE_EXPLORE]` contains
  only ``{explore, specialist, recover}`` — the legacy names are
  gone from the allowlist (PR 5.1).
* :data:`cli._REAL_EXECUTORS_FULL` does not register the legacy
  executors (PR 5.2).
* :data:`prompt_builder.FULL_ENABLED_ACTIONS` /
  :data:`prompt_builder.NO_KERNEL_ENABLED_ACTIONS` no longer contain
  the legacy names (PR 5.3).
"""


import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.intent_parser import (
    Intent, IntentType,
)
from inference_optimizer.orchestrator.phase_state import (
    PHASE_ALLOWED_ACTIONS,
    PHASE_EXPLORE,
)
from inference_optimizer.orchestrator.policy import (
    DEPRECATED_ACTION_NAMES,
    DEPRECATED_ACTION_REPLACEMENTS,
    PolicyDenied,
    PolicyGate,
)
from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
    FULL_ENABLED_ACTIONS,
    NO_KERNEL_ENABLED_ACTIONS,
)


# ===========================================================================
# 1. Constant surface
# ===========================================================================
def test_deprecated_action_names_matches_design_v0_6_set():
    """KB_design §3.15 §2.3 lists the v0.6 actions that v0.8 retired
    in favour of ``explore``. The PolicyGate constant MUST mirror that
    set 1-1 — adding or removing an entry here is a design change."""
    assert DEPRECATED_ACTION_NAMES == frozenset({
        "backends", "params", "validate_stack",
    })


def test_deprecated_action_replacements_cover_every_name():
    """Every deprecated name MUST carry a non-empty replacement string
    so the policy_denial hint always points the LLM at a concrete
    next action. ``explore`` is the canonical target for all three."""
    for name in DEPRECATED_ACTION_NAMES:
        assert name in DEPRECATED_ACTION_REPLACEMENTS, (
            f"missing replacement for deprecated action {name!r}"
        )
        replacement = DEPRECATED_ACTION_REPLACEMENTS[name]
        assert replacement.strip()
        assert "explore" in replacement


# ===========================================================================
# 2. PolicyGate denial paths (delegate / propose_action / request)
# ===========================================================================
@pytest.fixture
def gate() -> PolicyGate:
    """Plain gate without ActionRegistry / shared_state — Gap-10 only
    exercises the deprecation rule which doesn't depend on either."""
    return PolicyGate(role_registry=default_role_registry())


@pytest.mark.parametrize("action_name", sorted(DEPRECATED_ACTION_NAMES))
def test_delegate_with_deprecated_action_name_is_denied(gate, action_name):
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": action_name,
            "predicted_gain_pct": 1.0,
            "params": {"grid": []},
        },
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "action_deprecated"
    assert action_name in str(exc.value)
    assert "explore" in (exc.value.hint or "")


@pytest.mark.parametrize("action_name", sorted(DEPRECATED_ACTION_NAMES))
def test_propose_action_with_deprecated_action_name_is_denied(gate, action_name):
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": action_name,
            "predicted_gain_pct": 1.0,
        },
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "action_deprecated"
    assert action_name in str(exc.value)
    assert "explore" in (exc.value.hint or "")


def test_request_with_deprecated_kind_is_denied(gate):
    """Defense in depth: a REQUEST whose ``kind`` happens to collide
    with a deprecated action name (no current production kind does,
    but operator extensions might) is denied with the same rule."""
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel",
            "kind": "backends",
        },
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "action_deprecated"


def test_explore_action_is_not_denied(gate):
    """Sanity check: the canonical replacement action MUST pass the
    deprecation gate (it then proceeds to the phase / role checks
    further down the chain, which we don't exercise here)."""
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "explore",
            "predicted_gain_pct": 1.0,
        },
    )
    # No exception — explore passes the deprecation gate. (The
    # downstream unknown-action gate only fires when an
    # ActionRegistry is wired; we deliberately don't wire one so we
    # don't shadow the rule under test.)
    gate.validate_intent("orchestration", intent)


# ===========================================================================
# 2b. Coordinator-internal analysis actions are never LLM-proposable
# ===========================================================================
# ``roofline`` and ``profile`` are both Coordinator-enqueued (PRELUDE
# bootstrap + watermark refresh). Mode is selected by the operator via
# ``--enable-roofline`` / ``--no-enable-roofline``; the LLM never
# picks. PolicyGate denies any propose/delegate/request that names
# either action so the orchestration loop cannot sneak a manual
# analysis past the auto-enqueue dedup.
_INTERNAL_ANALYSIS_ACTIONS = ("roofline", "profile")


@pytest.mark.parametrize("action_name", _INTERNAL_ANALYSIS_ACTIONS)
def test_delegate_with_analysis_action_is_denied(gate, action_name):
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": action_name,
            "predicted_gain_pct": 1.0,
            "params": {},
        },
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "analysis_action_not_llm_proposable"
    assert action_name in str(exc.value)
    assert "--enable-roofline" in (exc.value.hint or "")


@pytest.mark.parametrize("action_name", _INTERNAL_ANALYSIS_ACTIONS)
def test_propose_action_with_analysis_action_is_denied(gate, action_name):
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": action_name,
            "predicted_gain_pct": 1.0,
        },
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "analysis_action_not_llm_proposable"
    assert action_name in str(exc.value)


@pytest.mark.parametrize("action_name", _INTERNAL_ANALYSIS_ACTIONS)
def test_request_with_analysis_kind_is_denied(gate, action_name):
    """A REQUEST whose ``kind`` names roofline/profile is denied with
    the same rule — the Coordinator-internal enqueue bypasses
    PolicyGate, but any LLM-routed REQUEST must not."""
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel",
            "kind": action_name,
        },
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "analysis_action_not_llm_proposable"


# ===========================================================================
# 3. Rule precedence (Inv-11.3 orthogonality)
# ===========================================================================
def test_deprecation_fires_before_kernel_owned_check(gate):
    """If an action name is *both* deprecated and kernel-owned
    (hypothetical — none collide today), the deprecation rule must
    win. We exercise this by mocking the kernel-owned set transiently
    — the production constants disjointness is already guaranteed by
    other tests."""
    from inference_optimizer.orchestrator import policy as policy_mod
    original = policy_mod.KERNEL_OWNED_ACTIONS
    try:
        policy_mod.KERNEL_OWNED_ACTIONS = frozenset(
            list(original) + ["backends"],
        )
        intent = Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "backends",
                "params": {"grid": []},
            },
        )
        with pytest.raises(PolicyDenied) as exc:
            gate.validate_intent("orchestration", intent)
        # Deprecation wins — NOT kernel_owned_by_kernel_agent.
        assert exc.value.rule == "action_deprecated"
    finally:
        policy_mod.KERNEL_OWNED_ACTIONS = original


# ===========================================================================
# 4. Supporting infrastructure parity
# ===========================================================================
def test_phase_explore_allowlist_drops_legacy_actions():
    """The EXPLORE allowlist contains only the canonical action set:
    merged grid runner, specialist dispatch, integrate_patch,
    assess_remaining_gaps (IR-7 self-stop wrapper), dynamic_action
    (dynamic_action.MD P1 supplementary cross-domain channel), the
    auto-managed analysis kinds (``roofline`` and ``profile``, both
    Coordinator-enqueued on watermark crossings; mode picked by
    ``--enable-roofline``), and ``recover``. PolicyGate's
    ``analysis_action_not_llm_proposable`` rule keeps the LLM from
    delegating either analysis kind directly.
    """
    assert PHASE_ALLOWED_ACTIONS[PHASE_EXPLORE] == frozenset({
        "explore", "specialist", "integrate_patch",
        "assess_remaining_gaps", "dynamic_action",
        "roofline", "profile", "recover",
    })


def test_full_enabled_actions_drops_legacy_grid_actions():
    """KB_gaps/Gap-10 PR 5.3 — the per-tick action catalogue presented
    to Orchestration MUST NOT advertise the deprecated names."""
    for name in DEPRECATED_ACTION_NAMES:
        assert name not in FULL_ENABLED_ACTIONS
        assert name not in NO_KERNEL_ENABLED_ACTIONS


def test_full_enabled_actions_still_contains_explore():
    """Sanity: the replacement action stays enabled. Same for
    ``sweep`` / ``recover`` / ``baseline`` (canonical retentions)."""
    assert "explore" in FULL_ENABLED_ACTIONS
    assert "sweep" in FULL_ENABLED_ACTIONS
    assert "recover" in FULL_ENABLED_ACTIONS
    assert "baseline" in FULL_ENABLED_ACTIONS


def test_cli_real_executors_drops_legacy_registrations():
    """KB_gaps/Gap-10 PR 5.2 — cli ``_REAL_EXECUTORS_FULL`` no longer
    binds an executor for the deprecated names. A v0.6 resume that
    still has a queued ``backends`` task will surface as
    ``no_executor`` (intended; PolicyGate would reject a fresh
    delegate anyway)."""
    from inference_optimizer import cli as cli_mod
    for name in DEPRECATED_ACTION_NAMES:
        assert name not in cli_mod._REAL_EXECUTORS_FULL


def test_cli_real_executors_still_contains_explore_and_sweep():
    """Sanity: the canonical EXPLORE-phase executors stay registered."""
    from inference_optimizer import cli as cli_mod
    assert "explore" in cli_mod._REAL_EXECUTORS_FULL
    assert "sweep" in cli_mod._REAL_EXECUTORS_FULL
    assert "baseline" in cli_mod._REAL_EXECUTORS_FULL


# ===========================================================================
# 5. KB_gaps/Dead-C — validate_stack dead-path residue
# ===========================================================================
def test_dead_c_sequence_actions_drops_validate_stack(tmp_path, monkeypatch):
    """KB_gaps/Dead-C — the ``sequence_actions`` allow-list inside
    ``_sequence_denial_for_action`` no longer enumerates the deprecated
    ``backends`` / ``params`` / ``validate_stack`` names. They short-
    circuit to ``None`` (PolicyGate ``action_deprecated`` already
    denied them upstream)."""
    from inference_optimizer.orchestrator.backends import (
        MockBackend, MockCriticBackend, MockKernelBackend,
        MockRobustnessBackend, ScriptedPlan,
    )
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.paths import make_session_dir

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    silent = ScriptedPlan(turns=[], default_intent=Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    ))
    coord = Coordinator(
        make_session_dir(),
        backends={
            "orchestration": MockBackend(silent, name="orch"),
            "kernel": MockKernelBackend(),
            "critic": MockCriticBackend(),
            "robustness": MockRobustnessBackend(),
        },
    )
    for legacy in ("backends", "params", "validate_stack"):
        assert coord._sequence_denial_for_action(legacy) is None, (
            f"legacy action {legacy!r} should short-circuit out of "
            "_sequence_denial_for_action (denied earlier at PolicyGate)"
        )


def test_dead_c_mission_summary_tag_points_at_explore():
    """KB_gaps/Dead-C — the mission-summary ``stack changed`` warning
    must NOT name the retired ``validate_stack`` action; it points the
    LLM at ``explore`` (which inlines the rebench)."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    s = SharedState(
        baseline_tput=100.0,
        optimization_stack=[{"action": "integrate", "kernel_id": "k1"}],
    )
    text = s.to_mission_summary()
    assert "stack changed" in text
    assert "RUN `explore`" in text
    assert "validate_stack" not in text


def test_dead_c_robustness_md_prune_branch_family_list():
    """KB_gaps/Dead-C — the Robustness prompt's ``prune_branch`` family
    enumeration drops the retired ``validate_stack`` family (and the
    legacy ``backends`` / ``params`` aliases) and keeps the canonical
    ``explore`` family."""
    from inference_optimizer.paths import asset_system_prompts_dir

    fragment = (asset_system_prompts_dir() / "robustness.md").read_text(
        encoding="utf-8"
    )
    prune_lines = [ln for ln in fragment.splitlines() if "prune_branch" in ln]
    assert prune_lines, "prune_branch row missing from robustness.md"
    row = prune_lines[0]
    for retired in ("validate_stack", "backends", "params"):
        assert retired not in row, (
            f"prune_branch family list still advertises retired {retired!r}"
        )
    assert "explore" in row
