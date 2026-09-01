# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the framework-rewrite switch manifest and its lever plumbing.

Covers three layers:

* manifest parsing and dependency reasoning (``_framework_switch_manifest``);
* the SharedState lever ledger and its attribution recorder;
* the explore-side seeding that turns registered levers into variants, in both
  directions (additive when the levers are dormant, leave-one-out when they are
  already on).

The dependency edges get the most attention. They are what lets an enabler be
benched together with the rewrite it unlocks, and the failure mode when they are
wrong is silent: the enabler measures flat, gets rejected, and every rewrite that
needed it is then measured with a permanently cold cache — so the loss is not one
lever's gain but the ceiling of the whole bundle.
"""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.common.env_safety import GPU_MASK_ENV_NAMES
from hyperloom.orchestrator.actions.executors import _framework_switch_manifest as manifest


def _entry(switch: str, **kwargs: Any) -> dict[str, Any]:
    """Build a manifest entry.

    Args:
        switch: Switch name.
        **kwargs: Additional manifest fields.

    Returns:
        The entry dict.
    """
    return {"switch": switch, **kwargs}


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_empty_input_is_not_an_error():
    """No manifest is the normal case for a non-rewrite patch."""
    assert manifest.parse_manifest(None) == ([], [])
    assert manifest.parse_manifest([]) == ([], [])
    assert manifest.parse_manifest({}) == ([], [])


def test_a_scalar_manifest_is_rejected_with_a_reason():
    """A malformed manifest reports why rather than degrading to silence."""
    switches, problems = manifest.parse_manifest("HL_X=1")
    assert switches == []
    assert problems and "list or dict" in problems[0]


def test_entries_are_normalised_and_uppercased():
    """Switch names are env vars, so they are canonicalised to upper case."""
    switches, problems = manifest.parse_manifest(
        [_entry("hl_geom_cache", category="memoize_invariant", target="t.py:f")]
    )
    assert problems == []
    assert switches[0]["switch"] == "HL_GEOM_CACHE"
    assert switches[0]["value"] == manifest.DEFAULT_SWITCH_VALUE
    assert switches[0]["category"] == "memoize_invariant"
    assert switches[0]["target"] == "t.py:f"


def test_dict_form_is_accepted():
    """A manifest keyed by switch name parses the same as the list form."""
    switches, _ = manifest.parse_manifest({"HL_A": {"category": "fuse_collectives"}})
    assert [s["switch"] for s in switches] == ["HL_A"]
    assert switches[0]["category"] == "fuse_collectives"


def test_an_explicit_value_is_kept():
    """A rewrite needing a mode rather than a boolean can say so."""
    switches, _ = manifest.parse_manifest([_entry("HL_ATTN", value="aiter")])
    assert switches[0]["value"] == "aiter"


def test_an_invalid_env_name_is_dropped():
    """A name that is not a legal env var could never be set."""
    switches, problems = manifest.parse_manifest([_entry("not a var")])
    assert switches == []
    assert any("valid environment variable" in p for p in problems)


@pytest.mark.parametrize("reserved", sorted(manifest.FORBIDDEN_SWITCHES)[:6])
def test_a_reserved_benchmark_variable_is_dropped(reserved):
    """Claiming e.g. PATH or TP would retarget the benchmark, not toggle a path."""
    switches, problems = manifest.parse_manifest([_entry(reserved)])
    assert switches == []
    assert any("reserved benchmark variable" in p for p in problems)


@pytest.mark.parametrize("mask", sorted(GPU_MASK_ENV_NAMES))
def test_a_gpu_mask_switch_is_dropped(mask):
    """Claiming a visibility mask would select the hardware, not toggle a path."""
    switches, problems = manifest.parse_manifest([_entry(mask)])
    assert switches == []
    assert any("reserved benchmark variable" in p for p in problems)


def test_a_credential_shaped_switch_is_dropped():
    """A manifest is held to the same boundary as a reference recipe."""
    switches, problems = manifest.parse_manifest([_entry("MY_TOKEN")])
    assert switches == []
    assert any("credential-shaped name" in p for p in problems)


def test_a_switch_colliding_with_benchmark_config_is_dropped():
    """A switch already set by the config would be toggled by unrelated config.

    That is worse than not having the lever: the rewrite would appear to be under
    the orchestrator's control while actually following the benchmark's own env.
    """
    switches, problems = manifest.parse_manifest(
        [_entry("HL_CACHE")],
        reserved_env={"HL_CACHE"},
    )
    assert switches == []
    assert any("already set by the benchmark configuration" in p for p in problems)


def test_a_duplicate_switch_is_dropped_once():
    """Two entries for one switch would register the lever twice."""
    switches, problems = manifest.parse_manifest([_entry("HL_A"), _entry("hl_a")])
    assert [s["switch"] for s in switches] == ["HL_A"]
    assert any("duplicate" in p for p in problems)


def test_an_unrecognised_category_is_kept_with_a_warning():
    """An unknown category is still information; dropping the lever would lose more."""
    switches, problems = manifest.parse_manifest([_entry("HL_A", category="something_new")])
    assert switches[0]["category"] == "something_new"
    assert any("unrecognised category" in p for p in problems)


def test_an_oversized_manifest_is_truncated():
    """A manifest claiming dozens of switches would consume the whole bench budget."""
    switches, problems = manifest.parse_manifest([_entry(f"HL_S{i}") for i in range(manifest.MAX_SWITCHES + 5)])
    assert len(switches) == manifest.MAX_SWITCHES
    assert any("keeping the first" in p for p in problems)


def test_a_dangling_dependency_reference_is_dropped():
    """An edge to a switch that is not in the manifest cannot be honoured.

    Keeping it would make the dependency closure silently incomplete, which is
    exactly the case that causes an enabler to be benched alone.
    """
    switches, problems = manifest.parse_manifest([_entry("HL_A", depends_on=["HL_MISSING"])])
    assert switches[0]["depends_on"] == []
    assert any("not in this manifest" in p for p in problems)


def test_a_self_reference_is_dropped():
    """A switch cannot depend on itself; the closure would be meaningless."""
    switches, problems = manifest.parse_manifest([_entry("HL_A", depends_on=["HL_A"])])
    assert switches[0]["depends_on"] == []
    assert any("self-reference" in p for p in problems)


def test_one_sided_edges_are_mirrored():
    """Declaring one direction states a real relationship; honour both halves.

    A specialist that writes only ``enables`` on the enabler would otherwise leave
    the dependent with an empty ``depends_on``, and the dependent's bundle would
    then omit the enabler it needs.
    """
    switches, _ = manifest.parse_manifest([_entry("HL_HOIST", enables=["HL_CACHE"]), _entry("HL_CACHE")])
    by_name = {s["switch"]: s for s in switches}
    assert by_name["HL_CACHE"]["depends_on"] == ["HL_HOIST"]
    assert by_name["HL_HOIST"]["enables"] == ["HL_CACHE"]


def test_a_dependency_cycle_is_broken_and_reported():
    """A cycle makes the closure unbounded, so the back edge is dropped."""
    switches, problems = manifest.parse_manifest(
        [_entry("HL_A", depends_on=["HL_B"]), _entry("HL_B", depends_on=["HL_A"])]
    )
    assert any("cyclic dependency" in p for p in problems)
    for entry in switches:
        assert manifest.dependency_closure(entry["switch"], switches) <= {"HL_A", "HL_B"}


def test_a_switch_with_dependents_is_flagged_as_an_enabler():
    """The enabler flag is derived, so a specialist cannot forget to set it."""
    switches, _ = manifest.parse_manifest([_entry("HL_HOIST", enables=["HL_CACHE"]), _entry("HL_CACHE")])
    by_name = {s["switch"]: s for s in switches}
    assert by_name["HL_HOIST"]["enabler"] is True
    assert by_name["HL_CACHE"]["enabler"] is False


# --------------------------------------------------------------------------
# dependency reasoning
# --------------------------------------------------------------------------


def _chain() -> list[dict[str, Any]]:
    """Return a parsed three-switch chain: HOIST -> CACHE -> DERIVED."""
    switches, problems = manifest.parse_manifest(
        [
            _entry("HL_HOIST", category="hoist_loop_invariant", enables=["HL_CACHE"]),
            _entry("HL_CACHE", category="memoize_invariant", enables=["HL_DERIVED"]),
            _entry("HL_DERIVED", category="memoize_invariant"),
            _entry("HL_STANDALONE", category="fuse_collectives"),
        ]
    )
    assert problems == []
    return switches


def test_switch_env_turns_everything_on():
    """The patch is inert until its switches are set, so benching needs them all."""
    assert manifest.switch_env(_chain()) == {
        "HL_HOIST": "1",
        "HL_CACHE": "1",
        "HL_DERIVED": "1",
        "HL_STANDALONE": "1",
    }


def test_switch_env_can_be_restricted():
    """A bundle turns on only its own closure."""
    assert manifest.switch_env(_chain(), only={"HL_HOIST", "HL_CACHE"}) == {
        "HL_HOIST": "1",
        "HL_CACHE": "1",
    }


def test_dependency_closure_is_transitive():
    """A rewrite two links down the chain needs both links above it."""
    assert manifest.dependency_closure("HL_DERIVED", _chain()) == {"HL_DERIVED", "HL_CACHE", "HL_HOIST"}
    assert manifest.dependency_closure("HL_STANDALONE", _chain()) == {"HL_STANDALONE"}


def test_dependents_closure_is_transitive():
    """Turning an enabler off has to turn off everything below it."""
    assert manifest.dependents_closure("HL_HOIST", _chain()) == {"HL_HOIST", "HL_CACHE", "HL_DERIVED"}
    assert manifest.dependents_closure("HL_DERIVED", _chain()) == {"HL_DERIVED"}


def test_closure_of_an_unknown_switch_is_itself():
    """Callers never have to special-case a name the manifest does not carry."""
    assert manifest.dependency_closure("HL_ABSENT", _chain()) == {"HL_ABSENT"}


# --------------------------------------------------------------------------
# additive variants
# --------------------------------------------------------------------------


def test_additive_variants_bundle_each_lever_with_its_dependencies():
    """An enabler is never measured without the rewrite it unlocks.

    This is the load-bearing case: benched alone the hoist saves only an
    allocation, so a 1% threshold rejects it — and the caches downstream then
    never hit, which loses the whole bundle rather than one lever.
    """
    variants = manifest.additive_variants(_chain())
    by_name = {v["name"]: v for v in variants}
    assert set(by_name["fwlever_hl_derived"]["extra_envs"]) == {"HL_DERIVED", "HL_CACHE", "HL_HOIST"}
    assert set(by_name["fwlever_hl_cache"]["extra_envs"]) == {"HL_CACHE", "HL_HOIST"}
    assert set(by_name["fwlever_hl_hoist"]["extra_envs"]) == {"HL_HOIST"}
    assert set(by_name["fwlever_hl_standalone"]["extra_envs"]) == {"HL_STANDALONE"}


def test_additive_variants_are_ordered_smallest_bundle_first():
    """Single-lever attribution lands before the combinations consume the budget."""
    sizes = [len(v["framework_lever_bundle"]) for v in manifest.additive_variants(_chain())]
    assert sizes == sorted(sizes)


def test_additive_variants_name_their_dependencies_in_the_note():
    """The note has to explain why a bundle carries more than one switch."""
    variants = {v["name"]: v for v in manifest.additive_variants(_chain())}
    assert "with its dependencies (HL_HOIST)" in variants["fwlever_hl_cache"]["note"]


def test_additive_variants_include_the_authored_full_stack():
    """The author's own combination is a hypothesis worth one leg."""
    names = [v["name"] for v in manifest.additive_variants(_chain())]
    assert "fwlever_all" in names
    full = next(v for v in manifest.additive_variants(_chain()) if v["name"] == "fwlever_all")
    assert len(full["extra_envs"]) == 4
    # A combination test attributes to no single lever.
    assert full["framework_lever"] == ""


def test_additive_variants_deduplicate_identical_bundles():
    """Two levers with the same closure would bench the same configuration twice."""
    switches, _ = manifest.parse_manifest([_entry("HL_A"), _entry("HL_B", depends_on=["HL_A"])])
    variants = manifest.additive_variants(switches)
    bundles = [frozenset(v["framework_lever_bundle"]) for v in variants]
    assert len(bundles) == len(set(bundles))


def test_a_single_lever_needs_no_full_stack_variant():
    """With one switch the bundle and the full stack are the same experiment."""
    switches, _ = manifest.parse_manifest([_entry("HL_ONLY")])
    assert [v["name"] for v in manifest.additive_variants(switches)] == ["fwlever_hl_only"]


# --------------------------------------------------------------------------
# leave-one-out variants
# --------------------------------------------------------------------------


def test_leave_one_out_removes_a_lever_with_its_dependents():
    """Leaving a dependent on without its enabler measures a broken config."""
    variants = {v["name"]: v for v in manifest.leave_one_out_variants(_chain())}
    assert set(variants["fwlever_drop_hl_cache"]["unset_envs"]) == {"HL_CACHE", "HL_DERIVED"}
    assert set(variants["fwlever_drop_hl_derived"]["unset_envs"]) == {"HL_DERIVED"}
    assert set(variants["fwlever_drop_hl_standalone"]["unset_envs"]) == {"HL_STANDALONE"}


def test_leave_one_out_skips_a_removal_that_empties_the_stack():
    """Removing everything reproduces the pre-patch baseline, already measured."""
    names = [v["name"] for v in manifest.leave_one_out_variants(_chain())]
    # Dropping the root enabler would take HL_CACHE and HL_DERIVED with it, but
    # HL_STANDALONE survives, so it is a real experiment and is kept.
    assert "fwlever_drop_hl_hoist" in names
    chain_only, _ = manifest.parse_manifest([_entry("HL_HOIST", enables=["HL_CACHE"]), _entry("HL_CACHE")])
    assert "fwlever_drop_hl_hoist" not in [v["name"] for v in manifest.leave_one_out_variants(chain_only)]


def test_leave_one_out_needs_at_least_two_levers():
    """With one lever, removing it is just the baseline."""
    switches, _ = manifest.parse_manifest([_entry("HL_ONLY")])
    assert manifest.leave_one_out_variants(switches) == []


def test_leave_one_out_variants_carry_no_extra_envs():
    """A removal variant must not also add anything, or the delta is unreadable."""
    for variant in manifest.leave_one_out_variants(_chain()):
        assert "extra_envs" not in variant
        assert variant["unset_envs"]


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def test_summary_is_empty_for_nothing():
    """A non-rewrite patch logs no manifest block."""
    assert manifest.summarize([], []) == ""


def test_summary_lists_switches_and_problems():
    """The log line carries both what was accepted and what was dropped."""
    switches, _ = manifest.parse_manifest([_entry("HL_A", category="memoize_invariant", target="a.py:f")])
    text = manifest.summarize(switches, ["dropped HL_B: bad name"])
    assert "HL_A" in text
    assert "category=memoize_invariant" in text
    assert "! dropped HL_B" in text


# --------------------------------------------------------------------------
# SharedState lever ledger
# --------------------------------------------------------------------------


def _state():
    """Return a fresh SharedState."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    return SharedState()


def test_levers_are_registered_with_their_state():
    """A registered lever records whether it is currently on."""
    state = _state()
    switches = _chain()
    assert state.record_authored_framework_levers(
        switches,
        default_on=False,
        specialist_task_id="t-1",
        stack_delta_pct=-0.4,
    )
    rows = state.authored_framework_levers
    assert [r["switch"] for r in rows] == ["HL_HOIST", "HL_CACHE", "HL_DERIVED", "HL_STANDALONE"]
    assert all(r["default_on"] is False for r in rows)
    assert rows[0]["specialist_task_id"] == "t-1"
    assert rows[0]["stack_delta_pct"] == pytest.approx(-0.4)
    assert rows[0]["enabler"] is True
    assert rows[0]["attributed_gain_pct"] is None


def test_re_registering_a_lever_does_not_duplicate_it():
    """A rewrite re-authored in a later round keeps one lever row."""
    state = _state()
    state.record_authored_framework_levers(_chain(), default_on=False)
    state.record_authored_framework_levers(_chain(), default_on=True)
    assert len(state.authored_framework_levers) == 4
    assert all(r["default_on"] is True for r in state.authored_framework_levers)


def test_re_registering_preserves_a_measured_attribution():
    """A measurement outranks a re-registration; losing it would re-bench the lever."""
    state = _state()
    state.record_authored_framework_levers(_chain(), default_on=False)
    state.record_framework_lever_attribution("HL_CACHE", gain_pct=7.5, source="additive")
    state.record_authored_framework_levers(_chain(), default_on=True)
    row = next(r for r in state.authored_framework_levers if r["switch"] == "HL_CACHE")
    assert row["attributed_gain_pct"] == pytest.approx(7.5)
    assert row["attribution_source"] == "additive"


def test_registering_nothing_is_a_noop():
    """A non-rewrite patch leaves the ledger untouched."""
    state = _state()
    assert state.record_authored_framework_levers([], default_on=True) is False
    assert state.authored_framework_levers == []


def test_attribution_of_an_unknown_switch_is_ignored():
    """A stale attribution cannot invent a lever row."""
    state = _state()
    assert state.record_framework_lever_attribution("HL_GHOST", gain_pct=1.0, source="additive") is False


def _entry_parsed(switch: str) -> dict[str, Any]:
    """Return a single parsed manifest entry for ``switch``."""
    switches, _ = manifest.parse_manifest([_entry(switch)])
    return switches[0]


# --------------------------------------------------------------------------
# explore seeding
# --------------------------------------------------------------------------


def test_dormant_levers_seed_additive_variants():
    """Code kept inert is turned on one dependency-closed bundle at a time."""
    from hyperloom.orchestrator.actions.executors.explore import framework_lever_grid

    state = _state()
    state.record_authored_framework_levers(_chain(), default_on=False)
    payload = framework_lever_grid(state)
    assert payload
    assert all(v["framework_lever_source"] == "additive" for v in payload)
    assert all("extra_envs" in v for v in payload)


def test_active_levers_seed_leave_one_out_variants():
    """A stack that already won is attributed by removing one lever at a time."""
    from hyperloom.orchestrator.actions.executors.explore import framework_lever_grid

    state = _state()
    state.record_authored_framework_levers(_chain(), default_on=True)
    payload = framework_lever_grid(state)
    assert payload
    assert all(v["framework_lever_source"] == "leave_one_out" for v in payload)
    assert all("unset_envs" in v for v in payload)


def test_already_attributed_levers_are_not_re_benched():
    """A lever with a number spends no further legs."""
    from hyperloom.orchestrator.actions.executors.explore import framework_lever_grid

    state = _state()
    state.record_authored_framework_levers(_chain(), default_on=False)
    for row in state.authored_framework_levers:
        state.record_framework_lever_attribution(row["switch"], gain_pct=1.0, source="additive")
    assert framework_lever_grid(state) == []


def test_no_levers_means_no_seeding():
    """The seeder is inert for every framework that has no authored rewrites."""
    from hyperloom.orchestrator.actions.executors.explore import framework_lever_grid

    assert framework_lever_grid(None) == []
    assert framework_lever_grid(_state()) == []


def test_lever_variants_survive_the_payload_parser():
    """The lever metadata has to reach the GridVariant, or attribution is lost."""
    from hyperloom.orchestrator.actions.executors.explore import (
        _grid_variants_from_payload,
        framework_lever_grid,
    )

    state = _state()
    state.record_authored_framework_levers(_chain(), default_on=True)
    payload = framework_lever_grid(state)
    variants = _grid_variants_from_payload(payload)
    assert len(variants) == len(payload)
    assert all(getattr(v, "framework_lever_source", "") == "leave_one_out" for v in variants)
    assert all(v.unset_envs for v in variants)


# --------------------------------------------------------------------------
# attribution sign convention
# --------------------------------------------------------------------------


def test_additive_attribution_takes_the_measured_gain_directly():
    """Switching a lever on measures its contribution as-is."""
    from hyperloom.orchestrator.actions.executors.explore import _framework_lever_attributions

    seeds = [{"name": "fwlever_hl_a", "framework_lever": "HL_A", "framework_lever_source": "additive"}]
    outcomes = [{"variant_name": "fwlever_hl_a", "outcome": "KEEP", "metrics": {"gain_pct": 6.2}}]
    assert _framework_lever_attributions(outcomes, seeds) == [
        {
            "switch": "HL_A",
            "gain_pct": 6.2,
            "source": "additive",
            "variant_name": "fwlever_hl_a",
            "outcome": "KEEP",
        }
    ]


def test_leave_one_out_attribution_negates_the_measured_gain():
    """Removing a lever measures the negative of its contribution.

    A stack that drops 8% without a lever means the lever was worth about 8%.
    Getting this sign wrong would invert every verdict in the report.
    """
    from hyperloom.orchestrator.actions.executors.explore import _framework_lever_attributions

    seeds = [
        {
            "name": "fwlever_drop_hl_a",
            "framework_lever": "HL_A",
            "framework_lever_source": "leave_one_out",
        }
    ]
    outcomes = [{"variant_name": "fwlever_drop_hl_a", "outcome": "REVERT", "metrics": {"gain_pct": -8.0}}]
    assert _framework_lever_attributions(outcomes, seeds)[0]["gain_pct"] == pytest.approx(8.0)


def test_a_combination_variant_attributes_to_nothing():
    """The full-stack leg is a combination test, not one rewrite's number."""
    from hyperloom.orchestrator.actions.executors.explore import _framework_lever_attributions

    seeds = [{"name": "fwlever_all", "framework_lever": "", "framework_lever_source": "additive"}]
    outcomes = [{"variant_name": "fwlever_all", "outcome": "KEEP", "metrics": {"gain_pct": 12.0}}]
    assert _framework_lever_attributions(outcomes, seeds) == []


def test_an_unmeasured_variant_attributes_to_nothing():
    """A failed or killed leg produces no number, so it records none."""
    from hyperloom.orchestrator.actions.executors.explore import _framework_lever_attributions

    seeds = [{"name": "fwlever_hl_a", "framework_lever": "HL_A", "framework_lever_source": "additive"}]
    outcomes = [{"variant_name": "fwlever_hl_a", "outcome": "FAILED", "metrics": {}}]
    assert _framework_lever_attributions(outcomes, seeds) == []


def test_non_lever_variants_are_ignored():
    """An ordinary config variant in the same round is not a lever attribution."""
    from hyperloom.orchestrator.actions.executors.explore import _framework_lever_attributions

    seeds = [{"name": "fwlever_hl_a", "framework_lever": "HL_A", "framework_lever_source": "additive"}]
    outcomes = [{"variant_name": "myfw_hw_queues_2", "outcome": "KEEP", "metrics": {"gain_pct": 3.0}}]
    assert _framework_lever_attributions(outcomes, seeds) == []


# --------------------------------------------------------------------------
# two-tier verdict in integrate_patch
# --------------------------------------------------------------------------


_REWRITE_PATCH = """\
diff --git a/src.py b/src.py
index 0000000..1111111 100644
--- a/src.py
+++ b/src.py
@@ -1,2 +1,2 @@
 def f():
-    return 1
+    return 2
"""

# A patch that gates its rewrite on an environment switch but does not declare it.
# This is the shape a real specialist delivered: four env-gated patches and no
# ``framework_switches`` key in the done payload at all.
_ENV_GATED_PATCH_WITHOUT_MANIFEST = """\
diff --git a/src.py b/src.py
index 0000000..1111111 100644
--- a/src.py
+++ b/src.py
@@ -1,2 +1,3 @@
 def f():
-    return 1
+    import os
+    return 2 if os.environ.get("HL_UNDECLARED_CACHE", "") == "1" else 1
"""


# Default manifest for the integrate_patch runs below: a hoist enabler plus the
# cache it unlocks. Named so a test can pass ``switches=[]`` to mean "no manifest
# at all", which is a materially different case from "the default one".
_DEFAULT_MANIFEST: list[dict[str, Any]] = [
    {"switch": "HL_HOIST", "category": "hoist_loop_invariant", "enables": ["HL_CACHE"]},
    {"switch": "HL_CACHE", "category": "memoize_invariant"},
]


@pytest.fixture(autouse=True)
def _allowlist_tmp_framework_roots(monkeypatch, tmp_path):
    """Let integrate_patch treat the test's temp checkout as framework source."""
    from .conftest import patch_integrate_patch_allowlist

    patch_integrate_patch_allowlist(monkeypatch, tmp_path)


async def _run_rewrite_integrate(
    tmp_path,
    monkeypatch,
    *,
    delta_pct: float,
    accuracy_pass: bool | None = True,
    switches: "list[dict[str, Any]] | None" = _DEFAULT_MANIFEST,
    parity_delta_pct: float = 0.0,
    parity_accuracy_pass: bool | None = True,
    parity_tput_missing: bool = False,
    patch_body: str | None = None,
    extra_params: "dict[str, Any] | None" = None,
):
    """Run integrate_patch on a switch-gated rewrite patch with a faked bench.

    Args:
        tmp_path: Test temp dir.
        monkeypatch: Pytest monkeypatch.
        delta_pct: Throughput delta the switches-on leg should imply.
        accuracy_pass: Accuracy verdict the switches-on leg reports.
        switches: The manifest the specialist delivered.
        parity_delta_pct: Throughput delta the switch-off parity leg should imply.
            0.0 models a correctly inert patch.
        parity_accuracy_pass: Accuracy verdict the parity leg reports.

    Returns:
        ``(result, repo, benched_envs, legs)`` where ``benched_envs`` is the env
        the switches-on leg ran with and ``legs`` records every bench invocation.
    """
    import json as _json

    from .conftest import init_git_repo
    from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
    from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
    from hyperloom.orchestrator.state.task_registry import Task

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    init_git_repo(repo)
    workspace = session_dir / "runs" / "specialist" / "t-spec-rw"
    (workspace / "worktree" / "patches").mkdir(parents=True)
    (workspace / "worktree" / "patches" / "001_rewrite.patch").write_text(
        patch_body or _REWRITE_PATCH, encoding="utf-8"
    )
    (workspace / "specialist_done.json").write_text(
        _json.dumps(
            {
                "gap_canonical_id": "gap.rewrite",
                "domain": "framework_rewrite_specialist",
                "proposal_set": [],
                "patches_written": ["patches/001_rewrite.patch"],
                "empty": False,
                "summary": "switch-gated rewrites",
                manifest.MANIFEST_KEY: list(switches or []),
            }
        ),
        encoding="utf-8",
    )

    base_tput = 100.0
    executor = IntegratePatchExecutor(session_dir=session_dir)
    benched_envs: dict[str, str] = {}
    legs: list[dict[str, Any]] = []

    async def _fake_bench(**kwargs):
        legs.append(dict(kwargs))
        is_parity = bool(kwargs.get("unset_envs"))
        if is_parity:
            if parity_tput_missing:
                # The leg ran but no throughput came back: a measurement failure,
                # which says nothing either way about whether the patch is inert.
                return ({}, {"accuracy_pass": None, "timed_out": False})
            # A well-behaved default-off patch reproduces the base exactly.
            return (
                {"output_throughput": base_tput * (1.0 + parity_delta_pct / 100.0)},
                {"accuracy_pass": parity_accuracy_pass, "timed_out": False},
            )
        benched_envs.update(kwargs.get("extra_envs_applied") or {})
        return (
            {"output_throughput": base_tput * (1.0 + delta_pct / 100.0)},
            {"accuracy_pass": accuracy_pass, "timed_out": False},
        )

    async def _noop_kb(**_kwargs):
        return None

    monkeypatch.setattr(executor, "_bench_patch", _fake_bench)
    monkeypatch.setattr(executor, "_maybe_write_framework_kb_record", _noop_kb)

    task_params: dict[str, Any] = {
        "specialist_task_id": "t-spec-rw",
        "framework_source_root": str(repo),
        "base_tput": base_tput,
        "accuracy_baseline": 1.0,
    }
    if extra_params:
        task_params.update(extra_params)
    task = Task(
        task_id="t-int-rw",
        kind="integrate_patch",
        state="queued",
        params=task_params,
        idempotency_key="t-int-rw",
        requires_lanes=tuple(),
    )
    result = await executor(RunnerContext(task=task, lease=None, extra={}))
    return result, repo, benched_envs, legs


@pytest.mark.asyncio
async def test_a_rewrite_patch_is_benched_with_its_switches_on(tmp_path, monkeypatch):
    """An applied rewrite patch is inert, so benching it as-is would measure nothing."""
    _, _, benched_envs, _ = await _run_rewrite_integrate(tmp_path, monkeypatch, delta_pct=8.0)
    assert benched_envs == {"HL_HOIST": "1", "HL_CACHE": "1"}


@pytest.mark.asyncio
async def test_a_winning_bundle_keeps_and_registers_levers_as_on(tmp_path, monkeypatch):
    """Clearing the gate puts the switches in the running config, ready for removal tests."""
    result, repo, _, _ = await _run_rewrite_integrate(tmp_path, monkeypatch, delta_pct=8.0)
    assert result["status"] == "kept"
    assert result["framework_lever_outcome"] == "default_on"
    assert {row["switch"] for row in result["framework_levers"]} == {"HL_HOIST", "HL_CACHE"}
    assert result["extra_envs_applied"] == {"HL_HOIST": "1", "HL_CACHE": "1"}
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_an_unprofitable_bundle_is_kept_inert_with_levers_registered(tmp_path, monkeypatch):
    """A bundle that misses the threshold keeps its code dormant instead of reverting.

    The rewrites are default-off, so keeping them costs nothing at runtime, and
    reverting would throw away the ones that do pay together with the one that does
    not — including any enabler, whose entire purpose is to make another rewrite
    profitable rather than to be profitable itself.
    """
    result, repo, _, _ = await _run_rewrite_integrate(tmp_path, monkeypatch, delta_pct=0.2)
    assert result["status"] == "kept_inert"
    assert result["framework_lever_outcome"] == "registered_off"
    assert len(result["patches_applied"]) == 1
    # Nothing may enter the running configuration.
    assert result["extra_envs_applied"] == {}
    assert result["config_changes_applied"] == {}
    # The code is still on disk, just dormant.
    assert (repo / "src.py").read_text().endswith("return 2\n")
    assert "enabler" in result["reason"]


@pytest.mark.asyncio
async def test_an_incorrect_switched_rewrite_is_kept_inert_and_flagged(tmp_path, monkeypatch):
    """A correctness failure on a bundle names the bundle, not a switch.

    This used to revert, on the reasoning that dormant-but-wrong code would be
    turned on later by explore. That risk is covered elsewhere: every lever variant
    explore benches goes through the same quality gate (``is_valid_measurement``
    rejects a scriptable measurement whose gate failed), so a broken switch is
    caught the moment it is the one being measured — which is also the only way to
    learn *which* switch it is.

    What reverting actually cost was the rest of the bundle. A live four-switch
    patch hit +65.5% and was discarded whole on the gate, taking three switches
    that were never implicated with it. So the code stays inert and flagged, and
    explore bisects it.
    """
    result, repo, _, _ = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=0.2,
        accuracy_pass=False,
    )
    assert result["status"] == "kept_inert"
    assert result.get("quality_unverified") is True
    # Applied but dormant: nothing may enter current_best off this verdict.
    assert result["config_changes_applied"] == {}
    assert result["extra_envs_applied"] == {}
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_a_patch_without_a_manifest_still_reverts_on_a_miss(tmp_path, monkeypatch):
    """The inert-keep path is opt-in via the manifest; ordinary patches are unchanged.

    Without switches, a kept patch would be live, so keeping an unprofitable one
    would silently degrade the configuration.
    """
    result, repo, _, _ = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=0.2,
        switches=[],
    )
    assert result["status"] == "reverted"
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_inert_keep_flows_into_the_lever_ledger(tmp_path, monkeypatch):
    """The writeback registers inert levers without lifting current_best.

    Both halves matter: without registration the dormant code is unreachable, and
    with a current_best lift the run would claim a gain it did not measure.
    """
    from hyperloom.orchestrator.loop.writeback import WritebackCollaborator

    result, _, _, _ = await _run_rewrite_integrate(tmp_path, monkeypatch, delta_pct=0.2)
    state = _state()
    state.baseline_tput = 100.0
    collaborator = WritebackCollaborator.__new__(WritebackCollaborator)
    collaborator.shared_state = state

    class _Outcome:
        changed = False
        audit_decision = None
        audit_extras: dict[str, Any] = {}

    outcome = _Outcome()
    await collaborator._promote_integrate_patch(result, None, outcome)
    assert outcome.audit_decision == "kept_inert"
    assert [row["switch"] for row in state.authored_framework_levers] == ["HL_HOIST", "HL_CACHE"]
    assert all(row["default_on"] is False for row in state.authored_framework_levers)
    assert not state.current_best


# --------------------------------------------------------------------------
# switch-off parity
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_manifest_without_a_patch_registers_no_levers(tmp_path, monkeypatch):
    """Switches gating code that was never delivered must not become levers.

    Registering them would leave the ledger pointing at absent code, and a later
    explore round would set switches that do nothing while reporting the result as
    that rewrite's contribution.
    """
    import json as _json

    from .conftest import init_git_repo
    from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
    from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
    from hyperloom.orchestrator.state.task_registry import Task

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    init_git_repo(repo)
    workspace = session_dir / "runs" / "specialist" / "t-spec-nopatch"
    workspace.mkdir(parents=True)
    (workspace / "specialist_done.json").write_text(
        _json.dumps(
            {
                "proposal_set": [],
                "patches_written": [],
                manifest.MANIFEST_KEY: _DEFAULT_MANIFEST,
            }
        ),
        encoding="utf-8",
    )
    result = await IntegratePatchExecutor(session_dir=session_dir)(
        RunnerContext(
            task=Task(
                task_id="t-int-nopatch",
                kind="integrate_patch",
                state="queued",
                params={"specialist_task_id": "t-spec-nopatch", "framework_source_root": str(repo)},
                idempotency_key="t-int-nopatch",
                requires_lanes=tuple(),
            ),
            lease=None,
            extra={},
        )
    )
    assert result["status"] == "no_patches"
    assert not result.get("framework_levers")


@pytest.mark.asyncio
async def test_parity_leg_runs_with_every_switch_removed(tmp_path, monkeypatch):
    """The parity leg must guarantee the switches are absent, not merely unset.

    Unsetting is not enough on its own: an earlier accepted rewrite can have put
    a switch into the base configuration, and the leg would then silently measure
    the switched-on path while claiming to measure parity.
    """
    _, _, _, legs = await _run_rewrite_integrate(tmp_path, monkeypatch, delta_pct=8.0)
    parity_legs = [leg for leg in legs if leg.get("unset_envs")]
    assert len(parity_legs) == 1
    assert set(parity_legs[0]["unset_envs"]) == {"HL_HOIST", "HL_CACHE"}
    assert parity_legs[0]["extra_envs_applied"] == {}
    # A distinct variant name, or the second leg would collide with the first.
    assert parity_legs[0]["variant_suffix"] == "-parity"


@pytest.mark.asyncio
async def test_a_clean_parity_leg_is_recorded_on_the_keep(tmp_path, monkeypatch):
    """A KEEP carries its own evidence that the patch is inert when disabled."""
    result, _, _, _ = await _run_rewrite_integrate(tmp_path, monkeypatch, delta_pct=8.0)
    assert result["status"] == "kept"
    parity = result["switch_off_parity"]
    assert parity["ran"] is True
    assert parity["ok"] is True
    assert parity["delta_pct"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_a_patch_that_is_not_inert_when_disabled_is_reverted(tmp_path, monkeypatch):
    """A rewrite that changes throughput with its switches off breaks the contract.

    Everything downstream assumes a disabled rewrite is a no-op: keeping inert
    code, comparing levers against a shared base, and stacking several rewrite
    patches in one session all rely on it. A patch that quietly takes effect while
    "off" would corrupt every later measurement, and the switches-on bench alone
    cannot detect it — which is why this costs a dedicated leg.
    """
    result, repo, _, _ = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=8.0,
        parity_delta_pct=9.0,
    )
    assert result["status"] == "reverted"
    assert result["error_class"] == "switch_off_parity_failed"
    assert "not a default-off rewrite" in result["reason"]
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_an_env_gated_patch_without_a_manifest_is_rejected(tmp_path, monkeypatch):
    """A gate the manifest does not declare disables every guarantee, silently.

    Observed on a live session: the specialist delivered four patches, each gating
    its rewrite on ``os.environ.get("MYFW_...")``, and no ``framework_switches``
    key at all. With an empty manifest the whole scheme quietly stands down --
    nothing is turned on for the measurement, no parity leg runs, no lever is
    registered -- and the patch is benched as an ordinary diff. That run then
    measured +1.4% *and* moved the output (ssim 0.4527 under a 0.4740 floor, lpips
    31% over), which a genuinely default-off patch cannot do. The parity leg exists
    precisely to catch that, and it never ran.

    The failure mode is the mirror of the manifest-without-a-patch case already
    handled here, so it gets the same treatment: refuse the deliverable instead of
    falling back to the unguarded path.
    """
    result, repo, _, legs = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=8.0,
        switches=[],
        patch_body=_ENV_GATED_PATCH_WITHOUT_MANIFEST,
    )
    assert result["status"] == "reverted"
    assert result["error_class"] == "framework_switch_gates_undeclared"
    assert "HL_UNDECLARED_CACHE" in result["reason"]
    # Refused before spending a benchmark leg on it.
    assert not legs, "an undeclared gate must be caught before benching"
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_a_plain_patch_without_env_gates_still_needs_no_manifest(tmp_path, monkeypatch):
    """The check must not turn every ordinary framework patch into a rejection.

    Most framework work is a straight edit with no switch at all; only a patch that
    reads an undeclared environment switch is contradicting its own manifest.
    """
    result, _, _, legs = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=8.0,
        switches=[],
    )
    assert result.get("error_class") != "framework_switch_gates_undeclared"
    assert legs, "a plain patch must still be benched"


@pytest.mark.asyncio
async def test_a_parity_leg_that_produced_no_measurement_is_not_called_a_parity_violation(tmp_path, monkeypatch):
    """ "We could not measure it" and "the patch is not inert" are different findings.

    On a live session a parity leg whose report was read too early came back with no
    throughput, and the verdict said the patch "is not actually inert" — for a patch
    worth +4.7% whose parity leg had in fact measured 0.3510 against a 0.3492 base,
    0.5% apart. Conflating the two is worse than losing the run: the KB record teaches
    every later session that this rewrite breaks when disabled, which is a lesson
    drawn from a filesystem race.

    The patch is still reverted — an unverified patch must not be left on disk to skew
    later measurements — but under its own outcome, and the reason must not assert
    something the measurement cannot support.
    """
    result, repo, _, _ = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=8.0,
        parity_tput_missing=True,
    )
    assert result["status"] == "reverted"
    assert result["error_class"] == "switch_off_parity_inconclusive"
    assert "not actually inert" not in result["reason"]
    assert "could not be measured" in result["reason"]
    assert (repo / "src.py").read_text().endswith("return 1\n")


def test_both_parity_outcomes_are_writable_to_the_framework_kb():
    """A verdict the KB rejects is a verdict that never reaches the next session.

    The first live run raised ``ValueError: outcome='reverted_switch_off_parity' must
    be one of [...]`` — the new verdict was never added to the allowed set, so the
    record was dropped and the lesson lost.
    """
    from hyperloom.orchestrator.knowledge import kb_writeback

    assert kb_writeback.OUTCOME_REVERTED_SWITCH_OFF_PARITY in kb_writeback.ALLOWED_OUTCOMES
    assert kb_writeback.OUTCOME_REVERTED_PARITY_INCONCLUSIVE in kb_writeback.ALLOWED_OUTCOMES
    # The two must stay distinct: one is a property of the patch, the other of the run.
    assert kb_writeback.OUTCOME_REVERTED_SWITCH_OFF_PARITY != kb_writeback.OUTCOME_REVERTED_PARITY_INCONCLUSIVE


@pytest.mark.asyncio
async def test_parity_noise_inside_the_band_is_tolerated(tmp_path, monkeypatch):
    """Run-to-run variance must not be mistaken for a behavioural change."""
    result, _, _, _ = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=8.0,
        parity_delta_pct=1.0,
    )
    assert result["status"] == "kept"


@pytest.mark.asyncio
async def test_a_parity_leg_that_fails_correctness_reverts(tmp_path, monkeypatch):
    """Different output with the switches off is a behavioural change too."""
    result, repo, _, _ = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=8.0,
        parity_accuracy_pass=False,
    )
    assert result["status"] == "reverted"
    assert result["error_class"] == "switch_off_parity_failed"
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_parity_guards_the_inert_keep_too(tmp_path, monkeypatch):
    """An inert KEEP leaves code on disk, so it needs the same guarantee.

    Without this, unprofitable-but-not-inert code would stay in the tree and skew
    the base of every subsequent measurement in the session.
    """
    result, repo, _, _ = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=0.2,
        parity_delta_pct=-5.0,
    )
    assert result["status"] == "reverted"
    assert result["error_class"] == "switch_off_parity_failed"
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_a_correctness_failure_still_spends_a_parity_leg_on_a_switched_bundle(tmp_path, monkeypatch):
    """A quality regression condemns one switch, not the whole bundle.

    Measured on a live session: a four-switch patch cached the SP seqlen
    rendezvous and reached 0.577 fps against a 0.3487 base -- +65.5%, three timed
    runs 0.3% apart -- and was reverted whole because the quality gate failed. The
    switches are benched together, so "the output moved" localises to the bundle,
    not to a switch; at least one is broken and the rest may be exactly the win
    the evidence pointed at.

    Default-off code costs nothing to keep, and per-lever attribution in explore
    is what separates the good switches from the bad one. But keeping it is only
    safe if the tree really is unchanged with every switch unset, which is what
    the parity leg measures — so on a switched bundle it is now worth its leg even
    when the gate failed.
    """
    result, repo, _, legs = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=8.0,
        accuracy_pass=False,
    )
    assert [leg for leg in legs if leg.get("unset_envs")], "parity must run to decide if the code can stay"
    assert result["status"] == "kept_inert"
    assert result.get("quality_unverified") is True
    # The code stays on disk with the switches off, for explore to bisect.
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_a_correctness_failure_without_switches_still_reverts(tmp_path, monkeypatch):
    """An unswitched patch has nothing to bisect, so a quality regression reverts it.

    Without a manifest the code is live the moment it is applied: there is no
    "off" state to fall back to, so a moved output means the patch must go.
    """
    result, repo, _, legs = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=8.0,
        accuracy_pass=False,
        switches=[],
    )
    assert result["status"] == "reverted"
    assert not [leg for leg in legs if leg.get("unset_envs")]
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_a_bundle_that_is_not_inert_still_reverts_despite_the_bisect_path(tmp_path, monkeypatch):
    """Keeping code is only safe when 'off' is genuinely off.

    A bundle that fails both the quality gate and parity is not a bisect
    candidate: with the switches unset it already changes the tree's behaviour, so
    leaving it would skew every later measurement in the session.
    """
    result, repo, _, _ = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=8.0,
        accuracy_pass=False,
        parity_delta_pct=9.0,
    )
    assert result["status"] == "reverted"
    assert result["error_class"] == "switch_off_parity_failed"
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_a_patch_without_switches_spends_no_parity_leg(tmp_path, monkeypatch):
    """Parity only means something for a default-off patch."""
    _, _, _, legs = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=8.0,
        switches=[],
    )
    assert not [leg for leg in legs if leg.get("unset_envs")]


@pytest.mark.asyncio
async def test_a_raising_parity_leg_does_not_read_as_a_pass(tmp_path, monkeypatch):
    """A probe that cannot run has proved nothing, so it must not wave the patch through."""
    from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor

    executor = IntegratePatchExecutor(session_dir=tmp_path)

    async def _raising_bench(**_kwargs):
        raise RuntimeError("bench harness exploded")

    monkeypatch.setattr(executor, "_bench_patch", _raising_bench)
    verdict = await executor._switch_off_parity(
        params={},
        output_root=tmp_path,
        specialist_task_id="t",
        switch_manifest=_chain(),
        base_tput=100.0,
    )
    assert verdict["ran"] is True
    assert verdict["ok"] is False
    assert "raised" in verdict["reason"]


@pytest.mark.asyncio
async def test_parity_without_a_base_is_reported_as_not_run(tmp_path):
    """No base to compare against means the check was skipped, not that it passed."""
    from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor

    verdict = await IntegratePatchExecutor(session_dir=tmp_path)._switch_off_parity(
        params={},
        output_root=tmp_path,
        specialist_task_id="t",
        switch_manifest=_chain(),
        base_tput=0.0,
    )
    assert verdict["ran"] is False
    assert "no positive base throughput" in verdict["reason"]


@pytest.mark.asyncio
async def test_parity_can_be_switched_off_explicitly(tmp_path):
    """An operator debugging the harness can take the extra leg out."""
    from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor

    verdict = await IntegratePatchExecutor(session_dir=tmp_path)._switch_off_parity(
        params={"enable_switch_off_parity": False},
        output_root=tmp_path,
        specialist_task_id="t",
        switch_manifest=_chain(),
        base_tput=100.0,
    )
    assert verdict["ran"] is False
    assert verdict["ok"] is True


@pytest.mark.asyncio
async def test_env_gated_patch_proceeds_to_bench_when_the_proposal_arms_it(tmp_path, monkeypatch):
    """An enablement round may arm its gate through the proposal instead of the manifest.

    Only the manifest feeds ``switch_env``, so an enablement fix that sets the gate
    in its proposal is self-consistent even with no manifest: the env is on for the
    bench. Refusing it would force env-shaped fixes to be rewritten as source
    patches. The gate stays recorded as an auditable problem.
    """
    result_en, _, _, legs_en = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=8.0,
        switches=[],
        patch_body=_ENV_GATED_PATCH_WITHOUT_MANIFEST,
        extra_params={"enablement": True, "extra_envs": {"HL_UNDECLARED_CACHE": "1"}},
    )
    assert result_en.get("error_class") != "framework_switch_gates_undeclared", (
        "an enablement gate armed by the proposal must not be refused"
    )
    assert legs_en, "enablement round must have attempted a bench"
    problems = result_en.get("framework_switch_problems") or []
    assert any("undeclared environment switch" in p for p in problems), (
        f"the demoted gate must stay auditable in the result, got {problems!r}"
    )


@pytest.mark.asyncio
async def test_env_gated_patch_refused_when_nothing_arms_it(tmp_path, monkeypatch):
    """An enablement gate armed by neither manifest nor proposal benches inert.

    ``switch_env`` only turns on manifest entries, so letting this through spends a
    leg reproducing the same failure and feeds the stall streak.
    """
    result_en, _, _, legs_en = await _run_rewrite_integrate(
        tmp_path,
        monkeypatch,
        delta_pct=8.0,
        switches=[],
        patch_body=_ENV_GATED_PATCH_WITHOUT_MANIFEST,
        extra_params={"enablement": True},
    )
    assert result_en["status"] == "reverted"
    assert result_en["error_class"] == "framework_switch_gates_undeclared"
    assert "HL_UNDECLARED_CACHE" in result_en["reason"]
    assert not legs_en, "a gate nothing turns on must not spend a bench leg"
