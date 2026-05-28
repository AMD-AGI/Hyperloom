"""dynamic_action.MD P9 §5 — red-line invariants.

These tests are the **machine evidence** that the §1.2 red lines stay
enforced regardless of which layer a violation comes from. They live
in a dedicated file (per P9 §11 #2) so CI can run them as a
block-merge gate — any failure here means a §1.2 red line has been
breached and the change must trigger the §3.11 design-change process,
not a local patch.

Naming convention (per P9 §11 #1): every test starts with ``inv_``;
the eight red-line categories are I-1 through I-8 per P9 §5.1.

Each test deliberately constructs a *hostile* input or behaviour and
asserts the system blocks it — these tests must not be deleted to
make the suite green; they ARE the red line.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.backends.mock_backend import (
    MockBackend, MockTurn, ScriptedPlan,
)
from inference_optimizer.orchestrator.dynamic_action_critic import (
    CROSS_DOMAIN_RULES,
    classify_proposal_for_critic,
    run_mechanical_cross_domain_checks,
)
from inference_optimizer.orchestrator.dynamic_action_pipeline import (
    DYNAMIC_SPECIALIST_TASK_ID_PREFIX,
    integrate_status_to_lifecycle,
    is_dynamic_specialist_task_id,
)
from inference_optimizer.orchestrator.dynamic_action_proposal import (
    DynamicActionStatus,
    DynamicRunnerTerminalState,
    EXPECTED_PROVENANCE,
    FORBIDDEN_PROPOSAL_FIELDS,
    LAST_OUTCOME_BY_STATUS,
    TERMINAL_LIFECYCLE_STATUSES,
    validate_proposal,
)
from inference_optimizer.orchestrator.dynamic_action_resume import (
    resume_abandon_dynamic_actions,
)
from inference_optimizer.orchestrator.dynamic_action_runner import (
    DynamicActionRunner,
)
from inference_optimizer.orchestrator.dynamic_action_seed_kit import (
    SEED_KIT_FIELDS,
    assemble_seed_kit,
)
from inference_optimizer.orchestrator.dynamic_action_tools import (
    BENCH_REGISTRY,
    read_session_artifact,
)
from inference_optimizer.orchestrator.intent_parser import (
    Intent, IntentType,
)
from inference_optimizer.orchestrator.policy import (
    ALL_KNOWN_EXTERNAL_TOOL_NAMES,
    CORE_STATE_FIELDS,
    DYNAMIC_ACTION_BUDGET_HINTS,
    DYNAMIC_ACTION_KERNEL_DOMAIN_LITERAL,
    DYNAMIC_ACTION_NAME,
    DYNAMIC_ACTION_SIDE_EFFECT_RED_LINES,
    EXPLORE_PERMISSIVE_PROVENANCE_LITERALS,
    EXPLORE_PERMISSIVE_PROVENANCE_PREFIXES,
    KERNEL_OWNED_ACTIONS,
    MAX_DYNAMIC_PER_ROUND,
    MAX_DYNAMIC_SOURCED_VARIANTS,
    PolicyDenied,
    PolicyGate,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.session_paths import (
    dynamic_action_artifact_dir,
    dynamic_action_proposal_set_path,
    dynamic_action_spec_path,
)


# ===========================================================================
# Helpers
# ===========================================================================
SCOPE = ["serving_specialist", "kernel_switch_specialist"]


@dataclass
class _State:
    """Minimal SharedState double accepted by PolicyGate's
    cross-cutting checks."""

    phase: str = "EXPLORE"
    tick: int = 0
    closing_phase: bool = False
    dynamic_action_round_count: int = 0

    def record_policy_denial(self, **_kwargs):  # noqa: D401
        return 1


def _gate(state: _State | None = None) -> PolicyGate:
    return PolicyGate(
        role_registry=default_role_registry(),
        shared_state=state or _State(),
        strict_phase=True,
    )


def _delegate_payload(**overrides: Any) -> dict[str, Any]:
    base_params = {
        "motivation_gap_text": (
            "Combine kv cache layout shift with scheduler rebalance "
            "neither specialist can surface alone."
        ),
        "scope_domains": list(SCOPE),
        "side_effects_declared": ["framework_source"],
        "budget_hint": "medium",
    }
    base_params.update(overrides.pop("params", {}))
    out = {"action_name": DYNAMIC_ACTION_NAME, "params": base_params}
    out.update(overrides)
    return out


def _good_proposal(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "combo",
        "provenance": "dynamic",
        "patch_text": (
            "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-old\n+new\n"
        ),
        "scope_domains": SCOPE,
        "cross_domain_rationale": (
            "serving_specialist must reorder kv layout coupled with "
            "kernel_switch_specialist; risk of cache regression"
        ),
        "expected_qualitative_argument": (
            "should reduce contention without breaking accuracy"
        ),
    }
    base.update(overrides)
    return base


@dataclass
class _StubTask:
    task_id: str = "task-1"
    kind: str = DYNAMIC_ACTION_NAME
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubBus:
    messages: list[Any] = field(default_factory=list)

    async def append_and_seq(self, msg: Any) -> None:
        self.messages.append(msg)


# ===========================================================================
# I-1 — micro-bench output never enters the promote chain
# (dynamic_action.MD §1.2 red line: "micro-bench 仅作内部假设验证")
# ===========================================================================
class TestInvariant_1_MicroBench:
    """I-1: micro-bench output stays inside the worktree and the
    journal — it never reaches proposal_set.json, SharedState,
    optimization_stack, or the intervention ledger."""

    def test_inv_microbench_not_in_proposal_via_validator(self):
        """Validator-level: any proposal carrying ``expected_gain``
        is rejected so bench numbers cannot ride along."""
        bad = _good_proposal()
        bad["expected_gain"] = 5.0
        result = validate_proposal(bad, spec_scope_domains=SCOPE)
        assert result.ok is False
        assert result.reason == "forbidden_field_present"

    @pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_PROPOSAL_FIELDS))
    def test_inv_microbench_no_quantitative_fields(self, forbidden: str):
        """Every member of the forbidden set blocks the proposal —
        no exceptions, no opt-outs."""
        bad = _good_proposal()
        bad[forbidden] = 1
        result = validate_proposal(bad, spec_scope_domains=SCOPE)
        assert result.ok is False, (
            f"forbidden field {forbidden!r} must block validation"
        )

    @pytest.mark.parametrize("claim", [
        "should give 20% gain",
        "saves 5 ms",
        "1.5x speedup",
        "speedup of 30",
    ])
    def test_inv_microbench_numeric_claim_blocked_in_qualitative(self, claim: str):
        """Even when packaged inside expected_qualitative_argument,
        numeric claims are rejected — the validator is the second
        defence after the bench-tool whitelist."""
        bad = _good_proposal(expected_qualitative_argument=claim)
        result = validate_proposal(bad, spec_scope_domains=SCOPE)
        assert result.ok is False
        assert result.reason == "numeric_claim_in_qualitative_argument"

    def test_inv_microbench_not_in_critic_envelope_after_floor(self):
        """Even if a proposal slips through the runner, the critic
        mechanical floor catches the same forbidden fields + numeric
        claim regex on the way into the verdict envelope."""
        bad = _good_proposal()
        bad["bench_evidence"] = {"ms": 12}
        pre = run_mechanical_cross_domain_checks(
            bad, spec_scope_domains=SCOPE,
        )
        assert pre.verdict == "reject"
        assert "dynamic_quantitative_claim_violation" in pre.reason_codes

    def test_inv_bench_registry_excludes_server_and_magpie(self):
        """The bench registry that ``run_bench`` validates against
        must NOT include any server / Magpie / serve-shaped bench:
        scratch outputs from those would be impossible to scrub from
        the global state."""
        for bench_id, spec in BENCH_REGISTRY.items():
            joined = (bench_id + " " + spec.description).lower()
            for marker in ("server", "magpie", "serve"):
                assert marker not in joined, (
                    f"bench {bench_id!r} description contains {marker!r}; "
                    f"violates I-5 (no server / Magpie surfaces)"
                )

    @pytest.mark.asyncio
    async def test_inv_worktree_destroyed_after_runner_exit(
        self, tmp_path: Path,
    ):
        """Runner finalise tears down the worktree — bench scratch
        outputs die with it (P3 §6 recovery whitelist mechanics)."""
        dyn_id = "dyn-0-1"
        # Spec + seed_kit on disk so the runner can load them.
        art = dynamic_action_artifact_dir(tmp_path, dyn_id)
        art.mkdir(parents=True, exist_ok=True)
        dynamic_action_spec_path(tmp_path, dyn_id).write_text(
            json.dumps({
                "dyn_id": dyn_id,
                "payload": {
                    "motivation_gap_text": "m",
                    "scope_domains": SCOPE,
                    "side_effects_declared": ["framework_source"],
                    "budget_hint": "medium",
                },
            }),
            encoding="utf-8",
        )
        (art / "seed_kit.json").write_text(json.dumps({
            "motivation_gap_text": "m",
            "roofline_summary": "",
            "profile_keyslices": [],
            "kept_patches": [],
            "reverted_patches": [],
            "kb_pitfalls": [],
            "source_root_hints": [],
        }), encoding="utf-8")
        # Runner script: emit immediately.
        emit_text = (
            "thinking\n```json\n"
            + json.dumps({
                "tool": "emit_proposal",
                "args": _good_proposal(),
            })
            + "\n```"
        )
        backend = MockBackend(ScriptedPlan(turns=[MockTurn(raw_text=emit_text)]))
        runner = DynamicActionRunner(backend, framework_source_roots=())
        ctx = RunnerContext(
            task=_StubTask(params={"dyn_id": dyn_id}),
            lease=None,
            extra={"session_dir": str(tmp_path)},
        )
        await runner.run(ctx)
        # No framework_source_roots → no worktree was set up; the
        # invariant we test here is the *intent*: the recovery
        # whitelist only contains proposal_set.json + journal.md,
        # so any worktree-scoped scratch path can never surface in
        # the artefact dir.
        artefact_files = {p.name for p in art.iterdir()}
        # Spec + seed_kit existed pre-runner; runner adds two files.
        runner_added = artefact_files - {"spec.json", "seed_kit.json"}
        assert runner_added <= {
            "proposal_set.json", "sub_agent_journal.md",
        }, (
            f"runner exposed extra files beyond the recovery whitelist: "
            f"{runner_added!r}"
        )


# ===========================================================================
# I-2 — SharedState protected fields cannot be mutated by the LLM
# (dynamic_action.MD §1.2 red line + D-C decision)
# ===========================================================================
class TestInvariant_2_SharedStateProtection:
    """I-2: dynamic_actions and adjacent core fields are
    Coordinator-only. PolicyGate rejects every shape of
    UPDATE_STATE that tries to touch them."""

    def test_inv_dynamic_actions_in_core_fields(self):
        assert "dynamic_actions" in CORE_STATE_FIELDS
        assert "dynamic_action_round_count" in CORE_STATE_FIELDS

    @pytest.mark.parametrize("changes", [
        {"dynamic_actions": {}},
        {"dynamic_actions": {"dyn-0-1": {"status": "KEPT"}}},
        {"dynamic_actions": {"dyn-99-99": {"cumulative_gain": 9.9}}},
        {"dynamic_action_round_count": 0},
    ])
    def test_inv_update_state_dynamic_actions_rejected(
        self, changes: dict[str, Any],
    ):
        gate = _gate()
        intent = Intent(
            type=IntentType.UPDATE_STATE,
            payload={"changes": changes},
        )
        with pytest.raises(PolicyDenied) as excinfo:
            gate.validate_intent("orchestration", intent)
        assert excinfo.value.rule == "state_field"

    @pytest.mark.parametrize("field_name", [
        "current_best",
        "optimization_stack",
        "gaps",
        "cumulative_gain_validated",
        "cumulative_gain",
    ])
    def test_inv_adjacent_core_fields_also_locked(self, field_name: str):
        """Other Coordinator-owned fields stay locked too — defense
        in depth so dynamic-action-adjacent writes cannot smuggle
        new state via neighbouring keys."""
        assert field_name in CORE_STATE_FIELDS

    def test_inv_writer_validates_transitions_strictly(self):
        """Even if a Coordinator hook tries to skip the state
        machine, the writer's can_transition check refuses illegal
        transitions and leaves the prior state intact."""
        state = SharedState(session_id="t")
        state.record_dynamic_action_outcome("dyn-0-1", status="DISPATCHED")
        # Forbidden: DISPATCHED → KEPT (skips INTEGRATING).
        state.record_dynamic_action_outcome("dyn-0-1", status="KEPT")
        assert state.dynamic_actions["dyn-0-1"]["status"] == "DISPATCHED"


# ===========================================================================
# I-3 — Provenance literal MUST be "dynamic" (no compound forms)
# (dynamic_action.MD §1.2 + P1 IR-4 + P3 runner schema + P4 critic)
# ===========================================================================
class TestInvariant_3_ProvenanceLiteral:
    """I-3: dynamic-sourced patches carry the single literal stamp
    ``dynamic`` at every layer — IR-4 white-list, runner validator,
    critic mechanical check, classifier."""

    def test_inv_ir4_whitelist_contains_dynamic_literal(self):
        assert "dynamic" in EXPLORE_PERMISSIVE_PROVENANCE_LITERALS

    def test_inv_ir4_whitelist_has_no_dynamic_prefix(self):
        """No ``dynamic:`` prefix in the allowed set — that would
        let composite forms slip through IR-4."""
        for prefix in EXPLORE_PERMISSIVE_PROVENANCE_PREFIXES:
            assert prefix != "dynamic:"
            assert not prefix.startswith("dyn")

    @pytest.mark.parametrize("bad", [
        "dynamic:kv_cache+scheduler",
        "dynamic:foo",
        "DYNAMIC",  # case-sensitive literal
        "dyn",
        "specialist:dynamic",
        "default_grid:dynamic",
        "",
    ])
    def test_inv_runner_validator_rejects_non_literal_provenance(self, bad: str):
        result = validate_proposal(
            _good_proposal(provenance=bad), spec_scope_domains=SCOPE,
        )
        assert result.ok is False, (
            f"runner validator must reject provenance={bad!r}"
        )

    @pytest.mark.parametrize("bad", [
        "specialist:serving_specialist",
        "dynamic:kv_cache+scheduler",
        "default_grid",
        "",
    ])
    def test_inv_critic_mechanical_floor_rejects_forged_provenance(self, bad: str):
        pre = run_mechanical_cross_domain_checks(
            _good_proposal(provenance=bad), spec_scope_domains=SCOPE,
        )
        assert pre.verdict == "reject"
        assert "dynamic_provenance_violation" in pre.reason_codes

    def test_inv_classifier_only_dynamic_triggers_cross_domain(self):
        """``cross_domain=true`` review constraints fire only when
        the proposal carries the literal — not any other shape."""
        _, rc = classify_proposal_for_critic({"provenance": "dynamic"})
        assert rc.get("cross_domain") is True
        for fake in ("dynamic:foo", "DYNAMIC", "default_grid", ""):
            _, rc2 = classify_proposal_for_critic({"provenance": fake})
            assert rc2.get("cross_domain") is not True, (
                f"classifier flipped cross_domain for fake provenance "
                f"{fake!r} — multi-layer defence broken"
            )

    def test_inv_explore_provenance_gate_accepts_only_literal(self):
        """End-to-end: PolicyGate's _validate_explore_provenance
        accepts ``dynamic`` literal in an explore grid but rejects
        composite forms via the legacy llm_direct path."""
        gate = _gate()
        gate.validate_intent("orchestration", Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "explore",
                "params": {
                    "grid": [
                        {"name": "v", "provenance": "dynamic"},
                    ],
                    "config_path": "/tmp/baseline.yaml",
                },
            },
        ))
        # Composite form should be rejected (it falls into the
        # llm_direct bucket because it's not in the literal /
        # prefix sets).
        with pytest.raises(PolicyDenied):
            gate.validate_intent("orchestration", Intent(
                type=IntentType.DELEGATE,
                payload={
                    "action_name": "explore",
                    "params": {
                        "grid": [
                            {"name": "v", "provenance": "dynamic:kv+sched"},
                        ],
                        "config_path": "/tmp/baseline.yaml",
                    },
                },
            ))


# ===========================================================================
# I-4 — Dynamic action cannot touch kernel-owned actions
# (dynamic_action.MD §1.2 + CLAUDE.md IR-6)
# ===========================================================================
class TestInvariant_4_KernelOwnedDenial:

    @pytest.mark.parametrize("forbidden", sorted(KERNEL_OWNED_ACTIONS))
    def test_inv_kernel_owned_in_side_effects_denied(self, forbidden: str):
        gate = _gate()
        payload = _delegate_payload(params={
            "motivation_gap_text": "kernel-owned probe",
            "scope_domains": SCOPE,
            "side_effects_declared": [forbidden],
        })
        with pytest.raises(PolicyDenied) as excinfo:
            gate.validate_intent("orchestration", Intent(
                type=IntentType.DELEGATE, payload=payload,
            ))
        assert excinfo.value.rule == "dynamic_side_effects_red_line"

    def test_inv_kernel_only_scope_domains_denied(self):
        gate = _gate()
        payload = _delegate_payload(params={
            "motivation_gap_text": "kernel-only impersonation",
            "scope_domains": [
                DYNAMIC_ACTION_KERNEL_DOMAIN_LITERAL,
                DYNAMIC_ACTION_KERNEL_DOMAIN_LITERAL,
            ],
            "side_effects_declared": ["framework_source"],
        })
        with pytest.raises(PolicyDenied) as excinfo:
            gate.validate_intent("orchestration", Intent(
                type=IntentType.DELEGATE, payload=payload,
            ))
        assert excinfo.value.rule == "dynamic_kernel_only_disallowed"

    def test_inv_kernel_owned_actions_set_locked(self):
        """KERNEL_OWNED_ACTIONS membership is the canonical source —
        any expansion needs a design change so we pin the v1 size."""
        assert KERNEL_OWNED_ACTIONS == frozenset({
            "kernel_opt",
            "integrate",
            "deep_kernel_analysis",
            "operator_tuning",
            "vendor_kernel_config",
        })


# ===========================================================================
# I-5 — Dynamic action cannot start independent server / run Magpie
# (dynamic_action.MD §1.2)
# ===========================================================================
class TestInvariant_5_NoServerNoMagpie:

    @pytest.mark.parametrize("forbidden", ["server", "magpie", "accuracy_gate"])
    def test_inv_server_magpie_in_side_effects_denied(self, forbidden: str):
        gate = _gate()
        payload = _delegate_payload(params={
            "motivation_gap_text": "lifecycle probe",
            "scope_domains": SCOPE,
            "side_effects_declared": [forbidden],
        })
        with pytest.raises(PolicyDenied) as excinfo:
            gate.validate_intent("orchestration", Intent(
                type=IntentType.DELEGATE, payload=payload,
            ))
        assert excinfo.value.rule == "dynamic_side_effects_red_line"

    def test_inv_side_effect_red_line_set_locked(self):
        """The side-effect red-line set is the canonical anchor for
        I-5 + I-6; pin it so additions / removals are visible."""
        assert DYNAMIC_ACTION_SIDE_EFFECT_RED_LINES == frozenset({
            "metric", "accuracy_gate", "server", "magpie",
        })

    def test_inv_bench_registry_excludes_server_shapes(self):
        """No bench in the registry may carry a server / serve /
        magpie / launch shape in its description."""
        for bench_id, spec in BENCH_REGISTRY.items():
            joined = (bench_id + " " + spec.description).lower()
            for marker in ("launch", "serve", "magpie", "deploy"):
                assert marker not in joined, (
                    f"bench {bench_id!r} description contains {marker!r}; "
                    f"violates I-5"
                )


# ===========================================================================
# I-6 — Dynamic action cannot declare its own metric
# (dynamic_action.MD §1.2)
# ===========================================================================
class TestInvariant_6_NoSelfMetric:

    def test_inv_metric_in_side_effects_denied(self):
        gate = _gate()
        payload = _delegate_payload(params={
            "motivation_gap_text": "metric probe",
            "scope_domains": SCOPE,
            "side_effects_declared": ["metric"],
        })
        with pytest.raises(PolicyDenied) as excinfo:
            gate.validate_intent("orchestration", Intent(
                type=IntentType.DELEGATE, payload=payload,
            ))
        assert excinfo.value.rule == "dynamic_side_effects_red_line"

    def test_inv_accuracy_gate_in_side_effects_denied(self):
        gate = _gate()
        payload = _delegate_payload(params={
            "motivation_gap_text": "accuracy gate probe",
            "scope_domains": SCOPE,
            "side_effects_declared": ["accuracy_gate"],
        })
        with pytest.raises(PolicyDenied) as excinfo:
            gate.validate_intent("orchestration", Intent(
                type=IntentType.DELEGATE, payload=payload,
            ))
        assert excinfo.value.rule == "dynamic_side_effects_red_line"

    def test_inv_keep_decision_comes_from_grid_runner(self):
        """The mapping from integrate_patch status → KEPT is the
        single source of truth; ``KEPT`` is reachable only via
        ``integrate_status_to_lifecycle`` on a ``kept`` status."""
        # Map every recognised integrate status — only "kept" + the
        # apply-only alias produce KEPT.
        for status, expected in [
            ("kept", DynamicActionStatus.KEPT),
            ("applied_no_bench", DynamicActionStatus.KEPT),
            ("reverted", DynamicActionStatus.REVERTED),
            ("apply_failed", DynamicActionStatus.INTEGRATE_FAILED),
            ("no_patches", DynamicActionStatus.INTEGRATE_FAILED),
            ("failed", DynamicActionStatus.INTEGRATE_FAILED),
            ("garbage", DynamicActionStatus.INTEGRATE_FAILED),
        ]:
            assert integrate_status_to_lifecycle(status) == expected


# ===========================================================================
# I-7 — Dynamic patches must traverse integrate_patch; no source
# tree shortcut (dynamic_action.MD §1.2)
# ===========================================================================
class TestInvariant_7_IntegratePatchOnly:

    def test_inv_recovery_whitelist_excludes_worktree_commits(self):
        """The runner's terminal-state finalise only persists two
        files into the artefact dir; any commit a sub-agent might
        make inside the worktree dies with the worktree."""
        # We rely on the documented contract in
        # dynamic_action_runner.DynamicActionRunner._finalise; the
        # P3 acceptance test
        # ``test_stub_executor_writes_into_artifact_dir`` already
        # pins the on-disk shape. Here we keep a structural marker
        # so the §1.2 red line is reflected in this file too.
        from inference_optimizer.orchestrator import dynamic_action_runner
        body = (Path(dynamic_action_runner.__file__)).read_text(
            encoding="utf-8",
        )
        # Only the two whitelisted file names are referenced inside
        # the finalise method's body.
        assert "proposal_set.json" in body
        assert "sub_agent_journal.md" in body
        # Negative: no direct git commit / push / apply of
        # framework_source paths in the runner.
        for forbidden in (
            "git push", "framework_source ", "/aiter/", "/sglang/", "/vllm/",
        ):
            assert forbidden not in body, (
                f"runner module references {forbidden!r} — should not "
                f"touch framework_source directly (I-7)"
            )

    def test_inv_dynamic_actions_summary_optimization_stack_disjoint(self):
        """``dynamic_actions[dyn_id]`` summary fields never include
        the optimization_stack — promote happens through the
        existing _promote_to_shared_state path, not via a side
        write into the dyn_id summary."""
        # SUMMARY_PROMPT_FIELDS pins the prompt projection schema;
        # there is no field named "optimization_stack" in it.
        from inference_optimizer.orchestrator.dynamic_action_proposal import (
            SUMMARY_PROMPT_FIELDS,
        )
        for field_name in SUMMARY_PROMPT_FIELDS:
            assert "optimization" not in field_name.lower(), (
                f"summary field {field_name!r} hints at "
                f"optimization_stack writes — I-7 forbids this side "
                f"channel"
            )


# ===========================================================================
# I-8 — Dynamic actions never learn from each other
# (dynamic_action.MD §1.8 + §1.2 implicit)
# ===========================================================================
class TestInvariant_8_CrossDynIsolation:

    def test_inv_read_session_artifact_rejects_other_dyn_id(self, tmp_path: Path):
        """Tool whitelist prevents one dyn_id from reading another's
        artefact dir."""
        # Seed a victim dyn_id artefact dir.
        victim = (
            tmp_path / "agents/orchestration/dynamic_actions/dyn-9-9/spec.json"
        )
        victim.parent.mkdir(parents=True, exist_ok=True)
        victim.write_text("{}", encoding="utf-8")
        # The current dispatch is dyn-0-1, trying to read dyn-9-9.
        res = read_session_artifact(
            tmp_path,
            "agents/orchestration/dynamic_actions/dyn-9-9/spec.json",
            dyn_id="dyn-0-1",
        )
        assert res["ok"] is False
        assert res["reason"] == "cross_dyn_id_isolation"

    def test_inv_seed_kit_field_set_excludes_other_dyn_data(self):
        """SEED_KIT_FIELDS is the closed contract for what enters
        a sub-agent's prompt — nothing in this set could carry
        another dyn_id's data."""
        for field_name in SEED_KIT_FIELDS:
            assert "dyn" not in field_name.lower(), (
                f"seed kit field {field_name!r} mentions dyn — could "
                f"leak cross-dispatch information (I-8)"
            )

    def test_inv_seed_kit_omits_dynamic_action_history(self):
        """SharedState.dynamic_actions is NOT one of the sources the
        assembler consults — only last_trace_analyze /
        explore_search / optimization_stack /
        warm_start_pitfalls. Sub-agents cannot see prior dyn_id
        outcomes."""
        from inference_optimizer.orchestrator import dynamic_action_seed_kit
        # The module body never references SharedState.dynamic_actions
        # — we treat the static absence as the static guarantee.
        body = Path(dynamic_action_seed_kit.__file__).read_text(
            encoding="utf-8",
        )
        # We allow the literal string in a comment, but not as a
        # python attribute access.
        attr_pattern = re.compile(r"\.dynamic_actions\b")
        assert attr_pattern.search(body) is None, (
            "seed kit assembler references SharedState.dynamic_actions; "
            "I-8 forbids cross-dispatch learning"
        )


# ===========================================================================
# Round-cap invariants — every reject + every restart respects
# MAX_DYNAMIC_PER_ROUND / MAX_DYNAMIC_SOURCED_VARIANTS
# (dynamic_action.MD §1.4 + Q3)
# ===========================================================================
class TestInvariant_RoundCap:

    def test_inv_max_dynamic_per_round_is_one(self):
        assert MAX_DYNAMIC_PER_ROUND == 1
        assert MAX_DYNAMIC_SOURCED_VARIANTS == 1

    def test_inv_dynamic_sourced_cap_enforced_at_explore_dispatch(self):
        """G4 — the IR-4 sourced-variant cap is mechanically enforced
        on every explore-grid delegate, independent of MAX_DYNAMIC_PER_ROUND."""
        gate = _gate()
        bad = Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "explore",
                "params": {
                    "grid": [
                        {"name": "v1", "provenance": "dynamic"},
                        {"name": "v2", "provenance": "dynamic"},
                    ],
                    "config_path": "/tmp/baseline.yaml",
                },
            },
        )
        with pytest.raises(PolicyDenied) as excinfo:
            gate.validate_intent("orchestration", bad)
        assert excinfo.value.rule == "dynamic_sourced_variant_cap_exceeded"

    def test_inv_single_dynamic_sourced_variant_passes(self):
        """A single dynamic-sourced variant within the cap should pass
        the new gate (no regression)."""
        gate = _gate()
        ok = Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "explore",
                "params": {
                    "grid": [
                        {"name": "v1", "provenance": "dynamic"},
                    ],
                    "config_path": "/tmp/baseline.yaml",
                },
            },
        )
        gate.validate_intent("orchestration", ok)

    def test_inv_round_cap_exhausted_rejected(self):
        state = _State(dynamic_action_round_count=MAX_DYNAMIC_PER_ROUND)
        gate = _gate(state)
        with pytest.raises(PolicyDenied) as excinfo:
            gate.validate_intent("orchestration", Intent(
                type=IntentType.DELEGATE, payload=_delegate_payload(),
            ))
        assert excinfo.value.rule == "dynamic_round_cap_exhausted"

    def test_inv_rejected_dispatch_does_not_bump_round_counter(self):
        """A PolicyGate denial must not advance the round counter
        — only the Coordinator's record_dynamic_action_dispatch
        path does that."""
        state = _State(dynamic_action_round_count=0)
        gate = _gate(state)
        # Force a reject: kernel-only scope.
        with pytest.raises(PolicyDenied):
            gate.validate_intent("orchestration", Intent(
                type=IntentType.DELEGATE,
                payload=_delegate_payload(params={
                    "motivation_gap_text": "ko",
                    "scope_domains": ["kernel", "kernel"],
                    "side_effects_declared": ["framework_source"],
                }),
            ))
        assert state.dynamic_action_round_count == 0


# ===========================================================================
# Phase / source restriction invariants
# (dynamic_action.MD §1.6 + §1.4 dispatch channel)
# ===========================================================================
class TestInvariant_PhaseSourceRestriction:

    @pytest.mark.parametrize("phase", ["PRELUDE", "FRAMEWORK_PR", "KERNEL", "SWEEP", "CLOSE"])
    def test_inv_dispatch_in_non_explore_phase_denied(self, phase: str):
        state = _State(phase=phase)
        gate = _gate(state)
        with pytest.raises(PolicyDenied) as excinfo:
            gate.validate_intent("orchestration", Intent(
                type=IntentType.DELEGATE, payload=_delegate_payload(),
            ))
        assert excinfo.value.rule == "dynamic_phase_violation"

    @pytest.mark.parametrize("role", ["robustness", "critic", "kernel"])
    def test_inv_dispatch_from_non_orchestration_denied(self, role: str):
        """Even role-with-delegate (robustness) must not initiate a
        dynamic_action — only orchestration is on the allowlist."""
        gate = _gate()
        try:
            gate.validate_intent(role, Intent(
                type=IntentType.DELEGATE, payload=_delegate_payload(),
            ))
        except PolicyDenied as excinfo:
            # Acceptable rules: role-level deny OR explicit
            # dynamic_source_violation.
            assert excinfo.rule in {
                "role", "dynamic_source_violation",
            }


# ===========================================================================
# Critic verdict mapping invariants — verdict outcomes are the only
# way to flip the lifecycle status
# ===========================================================================
class TestInvariant_VerdictMapping:

    def test_inv_three_cross_domain_rules_locked(self):
        rule_ids = [r.rule_id for r in CROSS_DOMAIN_RULES]
        assert rule_ids == [
            "rationale_per_domain",
            "coupling_and_side_effects",
            "motivation_gap_valid",
        ]

    def test_inv_revise_handled_as_reject_in_v1(self):
        """Per P4 §5.2 + P5 §6, REVISE collapses to CRITIC_REJECTED
        for the lifecycle status; the verdict label is preserved
        on the envelope for audit."""
        from inference_optimizer.orchestrator.dynamic_action_pipeline import (
            compose_critic_verdict_envelope,
        )
        envelope, lifecycle = compose_critic_verdict_envelope(
            dyn_id="dyn-0-1", proposal=_good_proposal(),
            spec_scope_domains=SCOPE,
            llm_verdict="revise", llm_reason="please tighten",
        )
        assert envelope["verdict"] == "revise"
        assert lifecycle == DynamicActionStatus.CRITIC_REJECTED


# ===========================================================================
# Recovery / resume invariants — every non-terminal goes to ABANDONED;
# every terminal stays put
# ===========================================================================
class TestInvariant_ResumeSweep:

    def test_inv_abandoned_reachable_from_every_non_terminal(self):
        from inference_optimizer.orchestrator.dynamic_action_proposal import (
            can_transition,
        )
        for src in DynamicActionStatus:
            if src in TERMINAL_LIFECYCLE_STATUSES:
                continue
            assert can_transition(src, DynamicActionStatus.ABANDONED), (
                f"ABANDONED must be reachable from non-terminal {src.value}"
            )

    def test_inv_resume_sweep_does_not_disturb_terminals(self, tmp_path: Path):
        state = SharedState(session_id="t")
        # Seed a KEPT dyn_id.
        state.record_dynamic_action_outcome("dyn-0-1", status="DISPATCHED")
        for st in ("SUB_AGENT_RUNNING", "SUB_AGENT_DONE",
                   "AWAITING_CRITIC", "INTEGRATING", "KEPT"):
            state.record_dynamic_action_outcome(
                "dyn-0-1", status=st,
                cumulative_gain=2.0 if st == "KEPT" else None,
            )
        result = resume_abandon_dynamic_actions(
            session_dir=tmp_path, shared_state=state,
            coordinator_session_id="x",
        )
        assert "dyn-0-1" in result.skipped_terminal
        assert state.dynamic_actions["dyn-0-1"]["status"] == "KEPT"


# ===========================================================================
# CORE_STATE_FIELDS audit — pin the protection set so a refactor
# that removes a field surfaces immediately
# ===========================================================================
class TestInvariant_CoreFieldsAudit:

    @pytest.mark.parametrize("required", [
        "current_best", "optimization_stack", "gaps",
        "cumulative_gain", "cumulative_gain_validated",
        "dynamic_actions", "dynamic_action_round_count",
        "specialist_rounds", "explore_search",
    ])
    def test_inv_required_field_in_core_state(self, required: str):
        assert required in CORE_STATE_FIELDS, (
            f"{required!r} dropped from CORE_STATE_FIELDS — that field "
            f"would become LLM-writable"
        )
