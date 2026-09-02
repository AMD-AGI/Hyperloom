# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The predictor's own attribution bucket, on both sides of the record.

The point of the bucket is measurement: without it a predicted variant is
graded as ``explore`` and a predicted patch as ``framework_agent``, so the
question the whole integration exists to answer -- is this proposer worth its
benchmark cycles -- has no answer in ``session_breakdown.json``.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.breakdown import agent_ownership as ao
from hyperloom.inference_optimizer.breakdown.collectors import optimizations as collector
from hyperloom.inference_optimizer.breakdown.recorder.instrument import _resolve_agent
from hyperloom.orchestrator.actions.executors.explore import _round_provenance
from hyperloom.orchestrator.predictor.pump import PROVENANCE


class TestProvenanceOwnership:
    def test_known_provenance_names_an_agent(self):
        assert ao.agent_from_provenance(PROVENANCE) == "primatune"

    def test_lookup_is_case_and_space_insensitive(self):
        assert ao.agent_from_provenance("  PrimaTune ") == "primatune"

    def test_free_form_labels_stay_audit_only(self):
        """``provenance`` has no allowlist; only a closed set earns a bucket."""
        for label in ("llm_direct", "default_grid", "specialist:moe", "", None):
            assert ao.agent_from_provenance(label) == ""


class TestResolveAgent:
    def test_predicted_config_is_not_credited_to_explore(self):
        """It runs as an explore variant, which is the machinery, not the author."""
        assert _resolve_agent("explore", result={"provenance": PROVENANCE}) == "primatune"
        # Without the stamp the same action still belongs to explore.
        assert _resolve_agent("explore", result={}) == "explore"

    def test_predicted_patch_is_not_credited_to_framework_agent(self):
        """integrate_patch would otherwise resolve through patch_author."""
        result = {
            "provenance": PROVENANCE,
            "lever_kind": ao.LEVER_SOURCE_PATCH,
            "source_phase": "FRAMEWORK_AGENT",
        }
        assert _resolve_agent("integrate_patch", result=result) == "primatune"
        # The same patch without the stamp keeps the old owner.
        assert _resolve_agent("integrate_patch", result={"source_phase": "FRAMEWORK_AGENT"}) == "framework_agent"

    def test_existing_buckets_are_undisturbed(self):
        assert _resolve_agent("replay_warm_recipe", result={}) == "warm_replay"
        assert _resolve_agent("baseline", result={}) == "coordinator"
        assert _resolve_agent("kernel_opt", result={}) == "kernel_agent"
        assert _resolve_agent("critic", result={}) == "critic"
        assert _resolve_agent("nothing_known", result={}, phase="") == ao.UNATTRIBUTED


class TestClosedSets:
    def test_collector_pre_creates_the_bucket(self):
        """A leaderboard should show a zero, not omit the row."""
        assert "primatune" in collector._SOURCES
        assert collector._empty_summary()["primatune"] == {"keeps": 0, "total_gain_pct": 0.0}

    def test_lever_kinds_stay_closed(self):
        """The lever is what changed, not who proposed it; no new value here."""
        assert "primatune" not in ao.LEVER_KINDS
        # The patch channel stamps an existing lever rather than inventing one.
        assert ao.patch_lever_kind({"lever_kind": ao.LEVER_SOURCE_PATCH}) == ao.LEVER_SOURCE_PATCH
        # An invented lever is rejected, which is why provenance carries this.
        assert ao.patch_lever_kind({"lever_kind": "primatune"}) == ""


class TestRoundProvenance:
    def test_agreeing_grid_reports_its_proposer(self):
        rows = [
            {"variant_name": "a", "outcome": "KEEP", "provenance": PROVENANCE},
            {"variant_name": "b", "outcome": "REVERT", "provenance": PROVENANCE},
        ]
        assert _round_provenance(rows) == PROVENANCE

    def test_mixed_grid_reports_nothing(self):
        """A round with two proposers has no single owner to name."""
        rows = [
            {"variant_name": "a", "outcome": "KEEP", "provenance": PROVENANCE},
            {"variant_name": "b", "outcome": "KEEP", "provenance": "llm_direct"},
        ]
        assert _round_provenance(rows) == ""

    def test_deduplicated_variants_do_not_break_agreement(self):
        """They were never measured and carry no provenance to agree with."""
        rows = [
            {"variant_name": "a", "outcome": "KEEP", "provenance": PROVENANCE},
            {"variant_name": "b", "outcome": "SKIPPED_DEDUP", "provenance": ""},
        ]
        assert _round_provenance(rows) == PROVENANCE

    def test_unstamped_round_reports_nothing(self):
        assert _round_provenance([{"variant_name": "a", "outcome": "KEEP"}]) == ""
        assert _round_provenance([]) == ""
        assert _round_provenance(None) == ""
