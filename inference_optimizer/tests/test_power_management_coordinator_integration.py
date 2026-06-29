"""Integration tests for ``power_management`` wiring across Coordinator paths.

These deliberately avoid spinning up an asyncio event loop or a real LLM
backend. They walk the public surfaces that the Coordinator + SubAgentRunner
touch when a `power_management` proposal flows through the system:

1. ``actions/_meta`` metadata loads and is in the right family/phase
2. ``cli._REAL_EXECUTORS_FULL`` registers the executor
3. ``prompt_builder`` exposes the action in both kernel + no-kernel modes
   and emits the grid-injection hint
4. ``SubAgentRunner.register_executor`` accepts the executor and routes
   a dry-run task to it end-to-end
5. ``session_paths.runs_dir`` resolves a workspace without raising on
   the new ``power_management`` action name
"""

from __future__ import annotations

import asyncio

import pytest

from inference_optimizer.cli import _REAL_EXECUTORS_FULL
from inference_optimizer.orchestrator.action_executors import (
    power_management_executor,
)
from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
    FULL_ENABLED_ACTIONS,
    GRID_INJECTABLE_ACTIONS,
    NO_KERNEL_ENABLED_ACTIONS,
    _format_grid_injection_hint,
)
from inference_optimizer.orchestrator.task_registry import Task
from inference_optimizer.session_paths import runs_dir


# ---------------------------------------------------------------------------
# Registry / metadata
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def registry() -> ActionRegistry:
    return ActionRegistry().load()


class TestRegistryWiring:
    def test_metadata_loads(self, registry):
        meta = registry.get("power_management")
        assert meta is not None
        assert meta.family == "shallow"
        assert meta.pipeline_phase == "explore"

    def test_required_lanes_include_server_and_bench(self, registry):
        meta = registry.get("power_management")
        assert {"server_lifecycle", "benchmark_lane"} <= set(meta.requires_lanes)

    def test_description_under_200_chars(self, registry):
        meta = registry.get("power_management")
        assert len(meta.description) < 200


# ---------------------------------------------------------------------------
# CLI executor table
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase allowlist (v0.8)
# ---------------------------------------------------------------------------
class TestPhaseAllowlist:
    def test_allowed_in_kernel_not_sweep_or_explore(self):
        from inference_optimizer.orchestrator import phase_state

        assert phase_state.is_action_allowed_in_phase(
            "power_management", phase_state.PHASE_KERNEL_AGENT,
        )
        assert not phase_state.is_action_allowed_in_phase(
            "power_management", phase_state.PHASE_SWEEP,
        )
        assert not phase_state.is_action_allowed_in_phase(
            "power_management", phase_state.PHASE_EXPLORE,
        )

class TestCliRegistration:
    def test_real_executors_full_includes_power_management(self):
        assert "power_management" in _REAL_EXECUTORS_FULL
        assert _REAL_EXECUTORS_FULL["power_management"] is power_management_executor


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
class TestPromptBuilder:
    def test_in_full_enabled(self):
        assert "power_management" in FULL_ENABLED_ACTIONS

    def test_not_in_no_kernel_enabled(self):
        # --no-kernel runs never enter KERNEL (where PM now lives), so
        # power_management is not offered in that action set.
        assert "power_management" not in NO_KERNEL_ENABLED_ACTIONS

    def test_in_grid_injectable_set(self):
        assert "power_management" in GRID_INJECTABLE_ACTIONS

    def test_grid_hint_mentions_rocm_smi_and_grid_shape(self):
        hint = _format_grid_injection_hint("power_management")
        assert hint is not None
        assert "rocm-smi" in hint
        assert "power_cap_w" in hint
        assert "perflevel" in hint


# ---------------------------------------------------------------------------
# Session paths
# ---------------------------------------------------------------------------
class TestSessionPaths:
    def test_runs_dir_resolves_for_power_management(self, tmp_path):
        sd = tmp_path / "session"
        sd.mkdir()
        out = runs_dir(sd, "power_management", "pm-task-1")
        assert out == sd / "runs" / "power_management" / "pm-task-1"

    def test_runs_dir_still_rejects_garbage_action(self, tmp_path):
        sd = tmp_path / "session"
        sd.mkdir()
        with pytest.raises(ValueError):
            runs_dir(sd, "garbage_action_xyz", "task-1")


# ---------------------------------------------------------------------------
# End-to-end: the executor registered in _REAL_EXECUTORS_FULL is callable
# through the canonical RunnerContext that SubAgentRunner constructs
# ---------------------------------------------------------------------------
class TestRegisteredExecutorIsCallable:
    def test_registered_callable_runs_dry_run_to_completion(self, tmp_path):
        # Pull the executor through the exact same dict the Coordinator
        # walks at startup, NOT the module-level handle, so a regression
        # in the cli wiring would surface as a hard failure here.
        fn = _REAL_EXECUTORS_FULL["power_management"]

        task = Task(
            task_id="pm-int-1",
            kind="power_management",
            state="running",
            params={
                "dry_run": True,
                "grid": [
                    {"name": "low", "power_cap_w": 200},
                    {"name": "high", "perflevel": "high"},
                ],
                "output_dir": str(tmp_path / "ws"),
            },
            idempotency_key="pm-int-1",
            requires_lanes=["server_lifecycle", "benchmark_lane"],
        )
        ctx = RunnerContext(task=task, lease=None, extra={})
        result = asyncio.run(fn(ctx))

        assert result["status"] == "succeeded"
        assert result["dry_run"] is True
        assert result["grid_size"] == 2
        names = {v["name"] for v in result["resolved_grid"]}
        assert names == {"low", "high"}


# ---------------------------------------------------------------------------
# SharedState wiring: audit set, ledger fields, apply/record helpers
# ---------------------------------------------------------------------------
class TestSharedStateWiring:
    """Pin the SharedState plumbing so power_management gets the same
    cross-call surfaces as explore:

    * ``_AUDIT_ACTIONS`` membership → ``power_management_attempts``
      auto-records via ``record_action_attempt``.
    * ``_KEY_METRIC_MAP`` entry → audit rows carry ``best_gain_pct``
      as the key metric for prompt rendering.
    * ``power_management_search`` + ``host_state_applied`` fields
      default to the right empty shapes on a fresh state.
    * ``apply_power_management_search_update`` preserves Coordinator-
      owned ``accepted`` rows across executor-driven updates.
    * ``record_power_management_accepted`` dedupes by fingerprint
      and removes matching entries from ``rejected``.
    """

    def test_audit_actions_includes_power_management(self):
        from inference_optimizer.orchestrator.shared_state import _AUDIT_ACTIONS
        assert "power_management" in _AUDIT_ACTIONS

    def test_key_metric_map_uses_best_gain_pct(self):
        from inference_optimizer.orchestrator.shared_state import _KEY_METRIC_MAP
        assert _KEY_METRIC_MAP["power_management"] == (
            "best_gain_pct", "gain_pct",
        )

    def test_fresh_state_has_empty_pm_ledger_and_no_host_state(self):
        from inference_optimizer.orchestrator.shared_state import SharedState
        ss = SharedState()
        assert ss.power_management_search == {}
        assert ss.host_state_applied is None
        assert ss.last_power_management == {}
        assert ss.power_management_attempts == []

    def test_record_action_attempt_appends_to_pm_attempts(self):
        from inference_optimizer.orchestrator.shared_state import SharedState
        ss = SharedState()
        entry = ss.record_action_attempt(
            action="power_management",
            task_id="t1",
            status="succeeded",
            decision="promoted",
            result={"best_gain_pct": 4.2, "workspace": "/tmp/ws"},
            extras={"bound_kind": "memory"},
        )
        assert entry is not None
        # The audit row lands in both the snapshot AND the rolling list.
        assert ss.last_power_management["task_id"] == "t1"
        assert ss.last_power_management["decision"] == "promoted"
        assert len(ss.power_management_attempts) == 1
        assert ss.power_management_attempts[0]["key_metric"] == 4.2
        assert ss.power_management_attempts[0]["key_metric_kind"] == "gain_pct"
        assert ss.power_management_attempts[0]["extras"]["bound_kind"] == "memory"

    def test_apply_search_update_preserves_accepted(self):
        # Executor updates only carry tested/rejected/last_round. The
        # Coordinator-owned ``accepted`` list must not be overwritten.
        from inference_optimizer.orchestrator.shared_state import SharedState
        ss = SharedState()
        ss.power_management_search = {
            "schema_version": 1,
            "accepted": [{
                "name": "cap_300", "fingerprint": "abc",
                "power_settings": {"power_cap_w": 300, "devices": []},
            }],
            "rejected": [], "tested": {}, "name_index": {},
            "cursor": 0,
        }
        ss.apply_power_management_search_update({
            "schema_version": 1,
            "tested": {"def": {"fingerprint": "def"}},
            "rejected": [{"fingerprint": "def"}],
            "name_index": {"new_var": "def"},
            "cursor": 1,
            # NOTE: no ``accepted`` field in the executor update.
        })
        assert ss.power_management_search["accepted"][0]["name"] == "cap_300"
        assert "def" in ss.power_management_search["tested"]

    def test_record_accepted_dedupes_by_fingerprint_and_clears_rejected(self):
        # A variant previously rejected that later wins should NOT
        # appear in both buckets — record_accepted removes it from
        # rejected so the dedup-set logic in the executor stays
        # consistent.
        from inference_optimizer.orchestrator.shared_state import SharedState
        from inference_optimizer.orchestrator.action_executors.power_management import (
            power_variant_fingerprint,
        )
        ss = SharedState()
        settings = {
            "power_cap_w": 300, "perflevel": "high",
            "sclk_idx": None, "mclk_idx": None, "pcie_idx": None,
            "perf_deterministic_mhz": None, "fan_pct": None, "devices": [],
        }
        fp = power_variant_fingerprint(settings)
        ss.power_management_search = {
            "schema_version": 1,
            "accepted": [],
            "rejected": [{
                "name": "cap_300_high", "fingerprint": fp,
                "power_settings": settings, "reason": "not_keep",
            }],
            "tested": {}, "name_index": {}, "cursor": 0,
        }
        ss.record_power_management_accepted({
            "name": "cap_300_high",
            "power_settings": settings,
            "gain_pct": 4.2,
        })
        accepted = ss.power_management_search["accepted"]
        rejected = ss.power_management_search["rejected"]
        assert len(accepted) == 1
        assert accepted[0]["fingerprint"] == fp
        assert accepted[0]["gain_pct"] == 4.2
        assert rejected == []  # the matching reject was cleared.

    def test_pm_search_ledger_migration_normalizes_missing_keys(self):
        # A legacy state.json with a partial pm_search dict (e.g. only
        # ``accepted``) must round-trip through ``from_dict`` with all
        # defaults filled in. Empty dicts collapse to {} so the
        # executor's "no prior ledger" code path stays unconditional.
        from inference_optimizer.orchestrator.shared_state import SharedState
        out = SharedState._normalize_pm_search_ledger({
            "accepted": [{"name": "x", "fingerprint": "a"}],
        })
        assert out["schema_version"] == 1
        assert out["accepted"] == [{"name": "x", "fingerprint": "a"}]
        assert out["rejected"] == []
        assert out["tested"] == {}
        assert out["name_index"] == {}
        assert out["cursor"] == 0
        assert out["last_round"] == {}

    def test_pm_search_normalize_empty_returns_empty(self):
        from inference_optimizer.orchestrator.shared_state import SharedState
        assert SharedState._normalize_pm_search_ledger({}) == {}
        assert SharedState._normalize_pm_search_ledger(None) == {}
        assert SharedState._normalize_pm_search_ledger("bogus") == {}


# ---------------------------------------------------------------------------
# Prompt-summary rendering for the power_management surfaces
# ---------------------------------------------------------------------------
class TestPromptSummaryRendering:
    """``SharedState.to_prompt_summary`` must surface
    ``last_power_management``, ``power_management_search``, and
    ``host_state_applied`` so the Orchestration LLM sees the same
    cross-call signal it does for explore. Without these
    lines the LLM cannot tell whether a prior power_management round
    won, which winner is currently applied, or which fingerprints
    are already in the dedup ledger.
    """

    def test_to_prompt_summary_renders_pm_attempt_line(self):
        from inference_optimizer.orchestrator.shared_state import SharedState
        ss = SharedState()
        ss.record_action_attempt(
            action="power_management",
            task_id="pm-prompt-1",
            status="succeeded",
            decision="promoted",
            result={"best_gain_pct": 3.7, "workspace": "/tmp/ws"},
            extras={"bound_kind": "memory"},
        )
        summary = ss.to_prompt_summary()
        assert "last_power_management=" in summary
        # Pulled from _format_attempt — decision + key_metric must be
        # visible so the LLM can rank the action against its peers.
        assert "decision=promoted" in summary
        assert "gain_pct=3.70" in summary

    def test_to_prompt_summary_renders_pm_search_block(self):
        from inference_optimizer.orchestrator.shared_state import SharedState
        ss = SharedState()
        ss.power_management_search = {
            "schema_version": 1,
            "accepted": [{
                "name": "cap_80_high",
                "fingerprint": "abcd",
                "power_settings": {
                    "power_cap_w": 300, "perflevel": "high",
                    "sclk_idx": None, "mclk_idx": None, "pcie_idx": None,
                    "perf_deterministic_mhz": None, "fan_pct": None,
                    "devices": [],
                },
                "gain_pct": 4.5, "tput": 105.0,
            }],
            "rejected": [{
                "name": "cap_60_only",
                "fingerprint": "deef",
                "power_settings": {
                    "power_cap_w": 240,
                    "perflevel": None, "sclk_idx": None, "mclk_idx": None,
                    "pcie_idx": None, "perf_deterministic_mhz": None,
                    "fan_pct": None, "devices": [],
                },
                "gain_pct": 0.4,
            }],
            "tested": {"abcd": {}, "deef": {}},
            "name_index": {"cap_80_high": "abcd", "cap_60_only": "deef"},
            "cursor": 2,
        }
        summary = ss.to_prompt_summary()
        assert "power_management_search=" in summary
        # Counts head-line.
        assert "cursor=2" in summary
        assert "accepted=1" in summary
        assert "rejected=1" in summary
        # Per-row content — gain%, knob string, name truncated to 28 cols.
        assert "cap_80_high" in summary
        assert "+4.50%" in summary
        assert "power_cap_w=300" in summary
        assert "perflevel=high" in summary

    def test_to_prompt_summary_renders_host_state_applied_block(self):
        from inference_optimizer.orchestrator.shared_state import SharedState
        ss = SharedState()
        ss.host_state_applied = {
            "variant_name": "cap_80_high",
            "smi_commands": [
                "rocm-smi --setpoweroverdrive 300 --autorespond yes",
                "rocm-smi --setperflevel high --autorespond yes",
            ],
            "gain_pct": 4.5,
            "ts": "2026-05-26T11:30:00+00:00",
        }
        summary = ss.to_prompt_summary()
        assert "host_state_applied=" in summary
        assert "variant=cap_80_high" in summary
        assert "+4.50%" in summary
        assert "--setpoweroverdrive 300" in summary
        assert "--setperflevel high" in summary

    def test_to_prompt_summary_empty_pm_surfaces_collapse_to_none(self):
        from inference_optimizer.orchestrator.shared_state import SharedState
        ss = SharedState()
        summary = ss.to_prompt_summary()
        # Empty defaults must render as ``(none)`` — same convention as
        # last_baseline / last_profile / etc. — so a prompt diff
        # doesn't grow on a fresh run.
        assert "last_power_management=(none)" in summary
        assert "power_management_search=(none)" in summary
        assert "host_state_applied=(none)" in summary


# ---------------------------------------------------------------------------
# Final report rendering of host_state_applied
# ---------------------------------------------------------------------------
class TestReportHostStateApplied:
    """The final report's ``## GPU power state applied`` section is the
    operator-facing companion to ``current_best.extra_server_args``:
    server flags on one side, rocm-smi setters on the other. Both are
    required to bring a fresh box back to the run's end state past a
    reboot, so the section must surface the verbatim shell commands
    plus the device subset + gain.
    """

    def test_summary_includes_host_state_applied_when_present(self):
        from inference_optimizer.orchestrator.shared_state import SharedState
        from inference_optimizer.orchestrator.action_executors.report import (
            _build_summary_dict,
        )
        state = SharedState(session_id="rep-1", baseline_tput=1000.0)
        state.host_state_applied = {
            "variant_name": "cap_80_high",
            "smi_commands": [
                "rocm-smi --setpoweroverdrive 300 --autorespond yes",
                "rocm-smi --setperflevel high --autorespond yes",
            ],
            "device_ids": [],
            "probed_range_w": [200, 400],
            "top_sclk_mhz": 1900,
            "gain_pct": 4.5,
            "ts": "2026-05-26T11:30:00+00:00",
        }
        summary = _build_summary_dict(state, {}, [])
        assert summary["host_state_applied"]["variant_name"] == "cap_80_high"

    def test_summary_omits_host_state_applied_when_none(self):
        from inference_optimizer.orchestrator.shared_state import SharedState
        from inference_optimizer.orchestrator.action_executors.report import (
            _build_summary_dict,
        )
        state = SharedState(session_id="rep-2", baseline_tput=1000.0)
        assert state.host_state_applied is None
        summary = _build_summary_dict(state, {}, [])
        # Empty / missing host_state_applied must not leave a stray
        # key in the summary — the report renderer keys off
        # ``summary.get("host_state_applied")`` and an empty dict
        # would produce an empty section.
        assert "host_state_applied" not in summary

    def test_format_md_renders_smi_commands_block(self):
        from inference_optimizer.orchestrator.shared_state import SharedState
        from inference_optimizer.orchestrator.action_executors.report import (
            _build_summary_dict, _format_md,
        )
        state = SharedState(
            session_id="rep-3",
            baseline_tput=1000.0,
            current_best={"action": "explore", "tput": 1100.0,
                          "variant_name": "aiter"},
        )
        state.host_state_applied = {
            "variant_name": "cap_80_high",
            "smi_commands": [
                "rocm-smi --setpoweroverdrive 300 --autorespond yes",
                "rocm-smi --setperflevel high --autorespond yes",
            ],
            "device_ids": [0, 1],
            "probed_range_w": [200, 400],
            "top_sclk_mhz": 1900,
            "gain_pct": 4.5,
            "ts": "2026-05-26T11:30:00+00:00",
        }
        summary = _build_summary_dict(state, {}, [])
        md = _format_md(summary)
        assert "## GPU power state applied" in md
        assert "cap_80_high" in md
        assert "+4.50%" in md
        assert "[0, 1]" in md
        assert "[200 W, 400 W]" in md
        assert "1900 MHz" in md
        # Re-apply block is fenced so copy-paste is trivial.
        assert "```bash" in md
        assert "rocm-smi --setpoweroverdrive 300 --autorespond yes" in md
        assert "rocm-smi --setperflevel high --autorespond yes" in md

    def test_format_md_omits_section_when_no_host_state(self):
        from inference_optimizer.orchestrator.shared_state import SharedState
        from inference_optimizer.orchestrator.action_executors.report import (
            _build_summary_dict, _format_md,
        )
        state = SharedState(session_id="rep-4", baseline_tput=1000.0)
        summary = _build_summary_dict(state, {}, [])
        md = _format_md(summary)
        # No power_management round won this session → section
        # entirely absent (NOT rendered as an empty "(none)" block,
        # because a fresh operator with no power state to recreate
        # shouldn't see a section that suggests one).
        assert "## GPU power state applied" not in md
