# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``recipe_steps``: the ordered enablement replay contract (R1a)."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.inference_optimizer.breakdown.collectors.sessions import (
    _build_attempt_summary,
    collect_enablement,
)
from hyperloom.orchestrator.enablement.recipe.steps import build_recipe_steps

SPEC_TASK = "spec-final"


def _steps(**enablement):
    return build_recipe_steps(enablement, attempt_summary=_build_attempt_summary)


def _kinds(steps):
    return [s["kind"] for s in steps]


def _attempt(task_id="b1", **kw):
    row = {
        "ok": True,
        "attempt_root": f"/s/enablement/builds/{task_id}",
        "installed_versions": {"aiter_ref": "v0.1.0", "arch": "gfx950"},
        "build_probes": [],
        "failure_class": "ok",
    }
    row.update(kw)
    return row


def _sentinel(task_id="b1", probe_task_id=SPEC_TASK, **kw):
    return {"task_id": task_id, "probe_task_id": probe_task_id, "routed": True, **kw}


def _linked_build_state(**kw):
    state = {
        "build_manifest": [_attempt(), _sentinel()],
        "last_specialist_task_id": SPEC_TASK,
    }
    state.update(kw)
    return state


# ---- AC1: order fidelity ---------------------------------------------------


def test_recipe_steps_order_matches_execution_order():
    steps = _steps(
        setup_commands=["pip install a", "pip install b"],
        kept_patches=["/p/1.patch", "/p/2.patch"],
        framework_root="/fr",
        **_linked_build_state(),
    )
    assert _kinds(steps) == ["setup", "setup", "build", "patch", "patch"]
    assert [s["cmd"] for s in steps if s["kind"] == "setup"] == ["pip install a", "pip install b"]
    assert [s["path"] for s in steps if s["kind"] == "patch"] == ["/p/1.patch", "/p/2.patch"]


def test_recipe_steps_setup_before_build_before_patch():
    steps = _steps(
        setup_commands=["pip install a"],
        kept_patches=["/p/1.patch"],
        **_linked_build_state(),
    )
    index = {kind: [i for i, s in enumerate(steps) if s["kind"] == kind] for kind in ("setup", "build", "patch")}
    assert max(index["setup"]) < min(index["build"]) < min(index["patch"])


def test_recipe_steps_build_is_not_last_when_patches_exist():
    steps = _steps(kept_patches=["/p/1.patch"], **_linked_build_state())
    build_at = _kinds(steps).index("build")
    assert any(s["kind"] == "patch" for s in steps[build_at + 1 :])


def test_recipe_steps_patch_order_follows_kept_patches():
    """kept_patches is the source of truth, not the kept_rounds row order."""
    steps = _steps(
        kept_patches=["/p/first.patch", "/p/second.patch"],
        kept_rounds=[
            {"patches": ["/p/second.patch"], "artifacts": []},
            {"patches": ["/p/first.patch"], "artifacts": []},
        ],
    )
    assert [s["path"] for s in steps] == ["/p/first.patch", "/p/second.patch"]


# ---- AC2: field traceability ----------------------------------------------


def test_recipe_steps_fields_trace_to_state():
    steps = _steps(setup_commands=["pip install a"], kept_patches=["/p/1.patch"], framework_root="/fr")
    setup, patch = steps
    assert set(setup) == {"kind", "cmd", "occurrence", "credential_class", "input_identity", "unresolved_inputs"}
    assert set(patch) == {"kind", "path", "root", "root_id"}
    assert patch["path"] == "/p/1.patch" and patch["root"] == "/fr"


def test_a_step_from_state_predating_the_ledger_claims_no_input_identity():
    """No identity was ever taken, which is not the same as none being needed."""
    setup = _steps(setup_commands=["pip install ./local.whl"])[0]
    assert setup["input_identity"] is None and setup["unresolved_inputs"] is None


def test_recipe_steps_patch_root_records_framework_root_verbatim():
    for root in ("/sgl-workspace/sglang", ""):
        steps = _steps(kept_patches=["/p/1.patch"], framework_root=root)
        assert steps[0]["root"] == root


def test_recipe_steps_setup_projects_applied_not_skipped():
    """A sanitised, never-executed command must never become a step (AT8)."""
    steps = _steps(setup_commands=["pip install real"])
    assert [s["cmd"] for s in steps] == ["pip install real"]


def test_recipe_steps_patch_paths_equal_kept_patches():
    """The emitted path sequence is exactly kept_patches, in list order (AT9)."""
    kept = ["/p/a.patch", "/p/b.patch", "/p/c.patch"]
    steps = _steps(kept_patches=kept, kept_rounds=[{"patches": list(reversed(kept)), "artifacts": []}])
    assert [s["path"] for s in steps] == kept


def test_recipe_steps_reuses_build_attempt_summary():
    entry = _attempt(action={"component": "aiter", "max_jobs": 4})
    steps = _steps(build_manifest=[entry, _sentinel()], last_specialist_task_id=SPEC_TASK)
    summary = _build_attempt_summary(entry)
    build = steps[0]
    assert build["ref"] == summary["ref"]
    assert build["gpu_arch"] == summary["gpu_arch"]
    assert build["component"] == summary["component"]
    assert build["max_jobs"] == summary["max_jobs"]


# ---- D0: populated vs null on the build step -------------------------------


def test_recipe_steps_build_nulls_absent_component_and_max_jobs():
    """A production-shaped entry carries no ``action``; the keys stay, null."""
    steps = _steps(**_linked_build_state())
    build = steps[0]
    assert build["component"] is None and build["max_jobs"] is None
    assert build["ref"] == "v0.1.0" and build["gpu_arch"] == "gfx950"
    for key in ("component", "ref", "gpu_arch", "max_jobs"):
        assert key in build


def test_the_build_element_declares_its_contract_keys_and_no_others():
    """Shape is fixed by the contract: a key is never omitted, only null."""
    build = _steps(**_linked_build_state())[0]
    assert set(build) == {
        "kind",
        "component",
        "ref",
        "gpu_arch",
        "max_jobs",
        "build_task_id",
        "build_driver",
        "build_inputs",
    }


def test_recipe_steps_build_populates_component_when_present():
    steps = _steps(
        build_manifest=[_attempt(action={"component": "vllm", "max_jobs": 12}), _sentinel()],
        last_specialist_task_id=SPEC_TASK,
    )
    assert steps[0]["component"] == "vllm" and steps[0]["max_jobs"] == 12


# ---- D1: which rounds contribute -------------------------------------------


def test_recipe_steps_includes_advanced_round():
    steps = _steps(
        kept_patches=["/p/advanced.patch"],
        kept_rounds=[{"patches": ["/p/advanced.patch"], "artifacts": []}],
    )
    assert [s["path"] for s in steps] == ["/p/advanced.patch"]


def test_recipe_steps_excludes_contentless_round():
    """A launch-only probe round contributes no step and shifts no order."""
    steps = _steps(
        kept_patches=["/p/a.patch", "/p/b.patch"],
        kept_rounds=[
            {"patches": ["/p/a.patch"], "artifacts": []},
            {"patches": [], "artifacts": []},
            {"patches": ["/p/b.patch"], "artifacts": []},
        ],
    )
    assert [s["path"] for s in steps] == ["/p/a.patch", "/p/b.patch"]


def test_recipe_steps_setup_only_round_still_emits_setup_step():
    """Setup is not D1-gated: an empty kept_rounds row never drops a command."""
    steps = _steps(setup_commands=["pip install lone"], kept_rounds=[{"patches": [], "artifacts": []}])
    assert _kinds(steps) == ["setup"]
    assert steps[0]["cmd"] == "pip install lone"


def test_recipe_steps_artifact_selection_includes_advanced_round():
    """An advanced round's artifact is in the replayed set with no patch of its own."""
    artifact = {"target": "/fr/srt/a.py", "rel_target": "srt/a.py"}
    out = collect_enablement(
        Path("/tmp/sess"),
        {
            "enablement_attempts": 1,
            "enablement_kept_rounds": [{"patches": [], "artifacts": [artifact]}],
            "enablement_kept_artifacts": [artifact],
        },
        [],
    )
    assert [a["target"] for a in out["kept_artifacts"]] == ["/fr/srt/a.py"]


def test_recipe_steps_artifact_selection_excludes_superseded_target():
    """Last-wins per target: the earlier entry is not part of the replayed set."""
    early = {"target": "/fr/srt/a.py", "rel_target": "srt/a.py", "kind": "early"}
    late = {"target": "/fr/srt/a.py", "rel_target": "srt/a.py", "kind": "late"}
    out = collect_enablement(
        Path("/tmp/sess"),
        {
            "enablement_attempts": 1,
            "enablement_kept_rounds": [
                {"patches": ["/p/1.patch"], "artifacts": [early]},
                {"patches": [], "artifacts": [late]},
            ],
            "enablement_kept_artifacts": [late],
            "enablement_kept_patches": ["/p/1.patch"],
        },
        [],
    )
    assert [a["kind"] for a in out["kept_artifacts"]] == ["late"]


def test_a_kept_artifact_carries_the_root_its_own_resolution_returned():
    """An artifact installed outside the framework tree is bound to its own root,
    which is the only thing that says where the recipe must restore it."""
    artifact = {"target": "/pkg/aiter/x.py", "rel_target": "aiter/x.py", "root": "/pkg"}
    out = collect_enablement(
        Path("/tmp/sess"),
        {
            "enablement_attempts": 1,
            "enablement_kept_artifacts": [artifact],
            "enablement_framework_root": "/fr",
            "enablement_roots": [
                {"id": "r1", "path": "/fr"},
                {"id": "r2", "path": "/pkg"},
            ],
        },
        [],
    )
    assert [a["root_id"] for a in out["kept_artifacts"]] == ["r2"]


def test_a_kept_artifact_bound_to_no_recorded_root_carries_a_null_root_id():
    out = collect_enablement(
        Path("/tmp/sess"),
        {
            "enablement_attempts": 1,
            "enablement_kept_artifacts": [{"target": "/x/a.py", "rel_target": "a.py", "root": "/x"}],
            "enablement_roots": [{"id": "r1", "path": "/fr"}],
        },
        [],
    )
    assert out["kept_artifacts"][0]["root_id"] is None


# ---- D2: whether historical build attempts appear --------------------------


def test_recipe_steps_emits_build_when_probe_is_final_round():
    steps = _steps(**_linked_build_state())
    assert _kinds(steps) == ["build"]


def test_recipe_steps_omits_build_when_probe_link_absent():
    """A bare sentinel and a superseded probe each link no build."""
    bare = _steps(
        build_manifest=[_attempt(), {"task_id": "b1", "routed": True}],
        last_specialist_task_id=SPEC_TASK,
    )
    superseded = _steps(
        build_manifest=[_attempt(), _sentinel(probe_task_id="an-earlier-probe")],
        last_specialist_task_id=SPEC_TASK,
    )
    assert bare == [] and superseded == []


def test_recipe_steps_emits_at_most_one_build():
    steps = _steps(
        build_manifest=[
            _attempt("b1"),
            _attempt("b2"),
            _sentinel("b1"),
            _sentinel("b2"),
        ],
        last_specialist_task_id=SPEC_TASK,
    )
    assert _kinds(steps) == ["build"]


def test_recipe_steps_build_join_is_by_identity_under_interleaving():
    """Concurrent completions interleave; the join is an equality, not a position.

    R1b replaces R1a's nearest-preceding-row rule: with B's row physically
    nearer the matching sentinel, the step still binds A's row, which the
    positional rule could not do.
    """
    manifest = [
        _attempt("bA", installed_versions={"aiter_ref": "vA", "arch": "gfx950"}),
        _attempt("bB", installed_versions={"aiter_ref": "vB", "arch": "gfx942"}),
        _sentinel("bA"),
    ]
    steps = _steps(build_manifest=manifest, last_specialist_task_id=SPEC_TASK)
    assert steps[0]["ref"] == "vA" and steps[0]["build_task_id"] == "bA"
    assert _steps(build_manifest=manifest, last_specialist_task_id=SPEC_TASK) == steps


def test_recipe_steps_build_step_stands_with_no_joinable_row():
    """An unjoined step is emitted and reported, never silently dropped."""
    steps = _steps(build_manifest=[_sentinel("bZ")], last_specialist_task_id=SPEC_TASK)
    assert _kinds(steps) == ["build"]
    assert steps[0]["ref"] is None and steps[0]["build_inputs"] is None


# ---- AC4: declarativeness --------------------------------------------------


#: A template placeholder or a line break is what separates a value from an
#: instruction; a declarative step carries neither.
_NON_DECLARATIVE = ("{{", "${", "%(", "\n")


def _leaves(value):
    """Yield every scalar reachable from ``value``, whatever nests it."""
    if isinstance(value, dict):
        for nested in value.values():
            yield from _leaves(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _leaves(nested)
    else:
        yield value


def test_recipe_steps_is_pure_declarative_data():
    steps = _steps(
        setup_commands=["pip install a"],
        kept_patches=["/p/1.patch"],
        build_manifest=[_attempt(build_inputs={"component": "aiter", "env_keys": ["K"]}), _sentinel()],
        last_specialist_task_id=SPEC_TASK,
        framework_root="/fr",
    )
    assert json.loads(json.dumps(steps)) == steps
    assert any(isinstance(step.get("build_inputs"), dict) for step in steps)
    for step in steps:
        assert step["kind"] in ("setup", "build", "patch")
        for key, value in step.items():
            for leaf in _leaves(value):
                assert leaf is None or isinstance(leaf, (str, int)), key
                assert not isinstance(leaf, str) or not any(m in leaf for m in _NON_DECLARATIVE), key


# ---- AC5: non-regression ---------------------------------------------------


def test_recipe_steps_absent_for_empty_enablement():
    assert collect_enablement(Path("/tmp"), {}, []) == {}
    engaged = collect_enablement(Path("/tmp"), {"enablement_attempts": 1}, [])
    assert "recipe_steps" not in engaged


def test_enablement_state_roundtrip_ignores_recipe_steps():
    from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound

    loaded = EnablementRound.from_dict({"recipe_steps": [{"kind": "setup"}], "setup_commands": ["pip install a"]})
    assert loaded.setup_commands == ["pip install a"]
    assert not hasattr(loaded, "recipe_steps")


def test_remote_recipe_unaffected_by_recipe_steps():
    from hyperloom.orchestrator.knowledge.remote_recipe.values import (
        RECIPE_SECTIONS,
        build_publishable_recipe_config,
    )

    emitted = {"recipe_steps", "replay_sufficiency", "roots", "source_snapshots", "launch_evidence"}
    assert emitted.isdisjoint(set(RECIPE_SECTIONS))

    base = {"extra_envs": {"A": "1"}, "extra_server_args": "--x 1"}
    assert build_publishable_recipe_config(dict(base)) == build_publishable_recipe_config(
        {**base, "recipe_steps": [{"kind": "setup", "cmd": "pip install a"}]}
    )
