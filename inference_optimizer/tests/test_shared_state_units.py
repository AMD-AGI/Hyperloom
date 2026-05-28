"""Focused unit tests for ``SharedState`` helpers that lacked direct coverage.

Existing tests exercise SharedState through the full Coordinator + executor
round-trip, but a handful of pure-data helpers (search ledger migration,
policy-denial bookkeeping, kernel-patch identity resolution, prune family
mutators) only have integration coverage and miss specific edge branches.
This module fills in the gap with small targeted tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.shared_state import SharedState


# ---------------------------------------------------------------------------
# _migrate_search_ledger
# ---------------------------------------------------------------------------

class TestMigrateSearchLedger:
    def test_empty_ledger_returns_empty(self):
        assert SharedState._migrate_search_ledger(None, schema_target=1) == {}
        assert SharedState._migrate_search_ledger({}, schema_target=1) == {}

    def test_non_dict_returns_empty(self):
        assert SharedState._migrate_search_ledger(["not", "a", "dict"], schema_target=1) == {}

    def test_already_fingerprinted_entries_pass_through(self):
        fp = "0123456789abcdef"
        ledger = {
            "tested": {
                fp: {
                    "name": "variant_a",
                    "extra_server_args": "--max-num-seqs 128",
                    "extra_envs": {},
                },
            },
            "schema_version": 2,
        }
        out = SharedState._migrate_search_ledger(ledger, schema_target=2)
        # Fingerprint-keyed entry stayed in place, with fingerprint set.
        assert fp in out["tested"]
        assert out["tested"][fp]["fingerprint"] == fp
        # Name index is rebuilt from the entry.
        assert out["name_index"]["variant_a"] == fp

    def test_legacy_name_keyed_entries_are_rekeyed(self):
        # Display-name keyed ledger: migration computes the fingerprint from
        # the stored args/envs and re-keys under it.
        ledger = {
            "tested": {
                "max_num_seqs_128": {
                    "extra_server_args": "--max-num-seqs 128",
                    "extra_envs": {"FOO": "bar"},
                },
            },
        }
        out = SharedState._migrate_search_ledger(ledger, schema_target=2)
        # Display name no longer in `tested`; some 16-char hex key is.
        assert "max_num_seqs_128" not in out["tested"]
        keys = list(out["tested"].keys())
        assert len(keys) == 1
        fp = keys[0]
        assert len(fp) == 16
        entry = out["tested"][fp]
        assert entry["name"] == "max_num_seqs_128"
        assert entry["fingerprint"] == fp
        assert out["name_index"]["max_num_seqs_128"] == fp

    def test_non_dict_tested_entries_dropped(self):
        ledger = {
            "tested": {
                "good": {"extra_server_args": "--a 1", "extra_envs": {}},
                "bad": "not-a-dict",
                42: {"extra_server_args": "--b 2"},
            },
        }
        out = SharedState._migrate_search_ledger(ledger, schema_target=1)
        # bad entry dropped; good + numeric-keyed entry retained.
        for value in out["tested"].values():
            assert isinstance(value, dict)

    def test_defaults_filled_in_when_missing(self):
        out = SharedState._migrate_search_ledger({"tested": {}}, schema_target=1)
        assert out["accepted"] == []
        assert out["rejected"] == []
        assert out["tested"] == {}
        assert out["name_index"] == {}
        assert out["schema_version"] >= 1


# ---------------------------------------------------------------------------
# pruned families + policy denial book-keeping
# ---------------------------------------------------------------------------

class TestPolicyDenialAndPruned:
    def test_add_pruned_family_is_idempotent(self):
        s = SharedState()
        assert s.add_pruned_family("kernel_opt") is True
        assert s.is_pruned("kernel_opt") is True
        # Second add returns False.
        assert s.add_pruned_family("kernel_opt") is False
        # prune_family is an alias.
        assert s.prune_family("kernel_opt") is False

    def test_record_policy_denial_tracks_streak(self):
        s = SharedState()
        first = s.record_policy_denial(
            action_name="kernel_opt", rule="cooldown",
            hint="hint", intent_type="propose_action", tick=1,
        )
        second = s.record_policy_denial(
            action_name="kernel_opt", rule="cooldown",
            hint="hint", intent_type="propose_action", tick=2,
        )
        assert first == 1 and second == 2
        # Resetting drops the matching streak rows.
        s.reset_policy_denial_streak("kernel_opt")
        again = s.record_policy_denial(
            action_name="kernel_opt", rule="cooldown",
            hint="hint", intent_type="propose_action", tick=3,
        )
        assert again == 1

    def test_record_policy_denial_keeps_intent_keys_when_present(self):
        s = SharedState()
        s.record_policy_denial(
            action_name="profile",
            rule="missing-prereq",
            hint="needs baseline",
            intent_type="propose_action",
            tick=1,
            intent_payload={"action": "profile", "params": {}},
        )
        assert s.policy_denial_history[-1]["intent_payload_keys"] == ["action", "params"]

    def test_reset_policy_denial_streak_ignores_blank_name(self):
        s = SharedState()
        s.policy_denial_streak["foo:bar"] = 3
        s.reset_policy_denial_streak("")
        assert s.policy_denial_streak == {"foo:bar": 3}

    def test_policy_denial_history_caps_to_50(self):
        s = SharedState()
        for tick in range(60):
            s.record_policy_denial(
                action_name="x", rule="r",
                hint="h", intent_type="propose_action", tick=tick,
            )
        assert len(s.policy_denial_history) == 50

    def test_policy_denial_summary_returns_empty_when_history_empty(self):
        assert SharedState().to_policy_denial_summary() == ""

    def test_policy_denial_summary_includes_recent_rows(self):
        s = SharedState()
        for tick in range(4):
            s.record_policy_denial(
                action_name=f"a{tick}", rule="rule",
                hint=f"hint-{tick}", intent_type="propose_action", tick=tick,
            )
        summary = s.to_policy_denial_summary(top_k=2)
        # Newest two rows surface in the summary.
        assert "a2" in summary
        assert "a3" in summary


# ---------------------------------------------------------------------------
# apply_changes
# ---------------------------------------------------------------------------

class TestApplyChanges:
    def test_empty_changes_returns_empty(self):
        assert SharedState().apply_changes({}, allow_core=True) == {}

    def test_unknown_keys_are_skipped(self):
        s = SharedState()
        applied = s.apply_changes({"unknown_field": 1}, allow_core=True)
        assert applied == {}

    def test_known_field_set(self):
        s = SharedState()
        applied = s.apply_changes({"model_name": "foo"}, allow_core=True)
        assert applied == {"model_name": "foo"}
        assert s.model_name == "foo"


# ---------------------------------------------------------------------------
# kernel-patch identity helpers
# ---------------------------------------------------------------------------

class TestKernelPatchIdentity:
    def test_resolves_explicit_payload(self):
        s = SharedState()
        kid, patch, target, args = s._resolve_kernel_patch_identity({
            "kernel_id": "k1",
            "patch_path": "/tmp/k1.py",
            "target_file": "/srv/k1.py",
            "extra_server_args": " --foo 1 ",
        })
        assert (kid, patch, target, args) == (
            "k1", "/tmp/k1.py", "/srv/k1.py", "--foo 1",
        )

    def test_falls_back_to_last_kernel_opt_patch(self):
        s = SharedState()
        s.last_kernel_opt = {"kernel_id": "k1", "best_artifact_path": "/srv/best.py"}
        kid, patch, target, args = s._resolve_kernel_patch_identity({
            "kernel_id": "k1",
        })
        assert patch == "/srv/best.py"
        assert kid == "k1"
        assert target == ""
        assert args == ""

    def test_kernel_patch_key_empty_when_payload_incomplete(self):
        s = SharedState()
        assert s.kernel_patch_key(None) == ""
        assert s.kernel_patch_key({"kernel_id": "k1"}) == ""

    def test_kernel_patch_key_concatenates_fields(self):
        s = SharedState()
        key = s.kernel_patch_key({
            "kernel_id": "k1",
            "patch_path": "/srv/k1.py",
            "extra_server_args": "--a 1",
        })
        assert key == "k1|/srv/k1.py|--a 1"

    def test_find_rejected_kernel_patch_lookup(self):
        s = SharedState()
        s.rejected_kernel_patches.append({
            "key": "k1|/srv/k1.py|--a 1",
            "reason": "no_e2e_gain",
        })
        hit = s.find_rejected_kernel_patch({
            "kernel_id": "k1",
            "patch_path": "/srv/k1.py",
            "extra_server_args": "--a 1",
        })
        assert hit and hit["reason"] == "no_e2e_gain"

    def test_find_rejected_kernel_patch_missing_returns_none(self):
        assert SharedState().find_rejected_kernel_patch({"kernel_id": "x"}) is None


# ---------------------------------------------------------------------------
# load_or_init / save round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_load_or_init_returns_default_when_missing(self, tmp_path):
        s = SharedState.load_or_init(tmp_path)
        assert s.session_id == ""

    def test_save_then_load_round_trip(self, tmp_path):
        s = SharedState(session_id="abc", baseline_tput=42.0)
        s.save(tmp_path)
        loaded = SharedState.load_or_init(tmp_path)
        assert loaded.session_id == "abc"
        assert loaded.baseline_tput == 42.0

    def test_save_atomic_path_uses_state_json(self, tmp_path):
        s = SharedState()
        s.save(tmp_path)
        assert (tmp_path / "state.json").is_file()

    def test_from_dict_drops_unknown_keys(self):
        raw = {"session_id": "abc", "unknown_field": "ignored"}
        s = SharedState.from_dict(raw)
        assert s.session_id == "abc"
        assert not hasattr(s, "unknown_field")

