# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``recipe_steps``: the ordered enablement replay contract (R1a)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.enablement_section import collect_enablement
from hyperloom.orchestrator.enablement.recipe.attempts import build_attempt_summary as _build_attempt_summary
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


def test_recipe_steps_fields_trace_to_state():
    steps = _steps(setup_commands=["pip install a"], kept_patches=["/p/1.patch"], framework_root="/fr")
    setup, patch = steps
    assert set(setup) == {"kind", "cmd", "cmd_sanitized", "occurrence", "credential_class"}
    assert set(patch) == {"kind", "path", "root", "root_id", "targets"}
    assert patch["path"] == "/p/1.patch" and patch["root"] == "/fr"


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


@pytest.mark.parametrize(
    "cmd, secret",
    [
        ("pip install --index-url https://user:hunter2@pypi.internal/simple pkg", "hunter2"),
        ("git clone https://oauth2:ghp_AAAAAAAAAAAAAAAAAAAA@github.example/x.git", "ghp_AAAAAAAAAAAAAAAAAAAA"),
        ("HF_TOKEN=hf_BBBBBBBBBBBBBBBBBBBB pip install pkg", "hf_BBBBBBBBBBBBBBBBBBBB"),
    ],
)
def test_a_credentialed_setup_command_is_never_emitted_verbatim(cmd, secret):
    """``cmd`` lands in session_breakdown.json, which is written out and shipped.

    The ledger has always sanitised its own copy; the step reconstructed the raw
    text beside it and carried the credential into the artifact. Withholding it
    costs no consumer anything: a step with a ``credential_class`` already
    raises ``credential_required``, so the replay is refused whether or not the
    verbatim command travels with it.
    """
    digest = hashlib.sha256(cmd.encode("utf-8")).hexdigest()
    steps = _steps(
        setup_commands=[cmd],
        setup_executions=[
            {"seq": 1, "cmd_digest": digest, "outcome": "applied", "round_task_id": "t-1"},
        ],
    )
    setup = [s for s in steps if s["kind"] == "setup"]
    assert setup, "an applied setup row must project a step"
    blob = json.dumps(steps)
    assert secret not in blob, f"the recipe carried the credential: {blob}"
    assert setup[0]["credential_class"]
    assert setup[0]["cmd_sanitized"] is True


def test_a_clean_setup_command_stays_verbatim_so_it_can_be_replayed():
    """Sanitising one that carries nothing would cost the replay its command."""
    cmd = "pip install --no-build-isolation ninja"
    digest = hashlib.sha256(cmd.encode("utf-8")).hexdigest()
    steps = _steps(
        setup_commands=[cmd],
        setup_executions=[
            {"seq": 1, "cmd_digest": digest, "outcome": "applied", "round_task_id": "t-1"},
        ],
    )
    setup = [s for s in steps if s["kind"] == "setup"]
    assert setup[0]["cmd"] == cmd
    assert setup[0]["credential_class"] is None
    assert setup[0]["cmd_sanitized"] is False


def test_each_patch_step_names_the_root_it_was_resolved_against():
    """A stack spanning two trees must bind each patch to its own.

    ``root`` used to carry the round's framework root while ``root_id`` carried
    the patch's own -- agreeing on a single-root stack and contradicting each
    other on exactly the multi-root one this projection exists to describe. A
    consumer reading ``root`` would apply the patch to the wrong tree, and
    nothing said so, because the two fields disagreed silently.
    """
    steps = [
        s
        for s in _steps(
            framework_root="/srv/vllm",
            kept_patches=["/p/a.patch", "/p/b.patch"],
            patch_roots={"/p/a.patch": "/srv/vllm", "/p/b.patch": "/srv/aiter"},
            roots=[
                {"id": "r-vllm", "path": "/srv/vllm"},
                {"id": "r-aiter", "path": "/srv/aiter"},
            ],
        )
        if s["kind"] == "patch"
    ]

    assert [s["root"] for s in steps] == ["/srv/vllm", "/srv/aiter"]
    # The pair must agree: a step whose root and root_id name different trees is
    # a step no replay can act on.
    assert [s["root_id"] for s in steps] == ["r-vllm", "r-aiter"]


def test_recipe_steps_reuses_build_attempt_summary():
    entry = _attempt(action={"component": "aiter", "max_jobs": 4})
    steps = _steps(build_manifest=[entry, _sentinel()], last_specialist_task_id=SPEC_TASK)
    summary = _build_attempt_summary(entry)
    build = steps[0]
    assert build["ref"] == summary["ref"]
    assert build["gpu_arch"] == summary["gpu_arch"]
    assert build["component"] == summary["component"]
    assert build["max_jobs"] == summary["max_jobs"]


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


def test_recipe_steps_absent_for_empty_enablement():
    assert collect_enablement(Path("/tmp"), {}, []) == {}
    engaged = collect_enablement(Path("/tmp"), {"enablement_attempts": 1}, [])
    assert "recipe_steps" not in engaged


def test_enablement_state_roundtrip_ignores_recipe_steps():
    """The emitted key is a projection, so carrying it changes nothing loaded."""
    from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound

    persisted = {"setup_commands": ["pip install a"]}
    loaded = EnablementRound.from_dict({"recipe_steps": [{"kind": "setup"}], **persisted})
    assert loaded.setup_commands == ["pip install a"]
    assert not hasattr(loaded, "recipe_steps")
    assert loaded == EnablementRound.from_dict(persisted)


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


# --------------------------------------------------------------------------
# The lane-status keys the export carries, and what each is read off.
# --------------------------------------------------------------------------


def test_a_trigger_observation_is_not_evidence_that_a_round_ran():
    """``launch_observation_path`` is written by the three trigger paths --
    the failed eval and the two failed boots -- all of which run BEFORE any
    round is dispatched. Reading it as dispatch evidence reports a lane that
    never opened a round as one that did."""
    state = {"enablement": {"launch_observation_path": "/s/reports/bringup/round-abc-000.json"}}
    section = collect_enablement(Path("/tmp"), state, [])
    assert section == {} or section["dispatched"] is False


def test_the_specialist_a_round_settled_onto_is_dispatch_evidence():
    state = {"enablement": {"last_specialist_task_id": "spec-1"}}
    assert collect_enablement(Path("/tmp"), state, [])["dispatched"] is True


def test_a_setup_row_a_round_stamped_its_id_onto_is_dispatch_evidence():
    """The post-rework case the task-id fields alone cannot see: a round that
    ran setup and was killed before it settled."""
    state = {"enablement": {"setup_executions": [{"seq": 1, "round_task_id": "spec-1"}]}}
    assert collect_enablement(Path("/tmp"), state, [])["dispatched"] is True


def test_an_unattributed_setup_row_is_not_dispatch_evidence():
    state = {"enablement": {"setup_executions": [{"seq": 1, "round_task_id": ""}]}}
    section = collect_enablement(Path("/tmp"), state, [])
    assert section["dispatched"] is False


def test_the_accepted_config_path_reaches_the_export(tmp_path):
    """A consumer replaying needs the config the accepted bench launched with;
    the section is where it is named."""
    state = {"enablement": {"last_specialist_task_id": "spec-1", "accepted_config_path": str(tmp_path / "a.yaml")}}
    assert collect_enablement(tmp_path, state, [])["accepted_config_path"] == "a.yaml"


def test_a_setting_script_that_is_a_directory_is_named_nowhere(tmp_path):
    """``is_file()``, not ``exists()``: a directory is not a script a consumer
    can source, and naming it offers a replay input that cannot be replayed."""
    (tmp_path / "reports" / "enablement" / "enablement_setting.sh").mkdir(parents=True)
    state = {"enablement": {"last_specialist_task_id": "spec-1"}}
    assert "setting_script" not in collect_enablement(tmp_path, state, [])


def test_a_setting_script_that_is_a_file_is_named(tmp_path):
    script = tmp_path / "reports" / "enablement" / "enablement_setting.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    state = {"enablement": {"last_specialist_task_id": "spec-1"}}
    section = collect_enablement(tmp_path, state, [])
    assert section["setting_script"] == "reports/enablement/enablement_setting.sh"


def _collected_codes(value, *, present=True):
    """Run real recipe collection and return its replay_sufficiency codes."""
    state = {"enablement_attempts": 1, "enablement_kept_rounds": [{"patches": [], "artifacts": []}]}
    if present:
        state["enablement_build_extensions_not_carried"] = value
    out = collect_enablement(Path("/tmp/sess"), state, [])
    return [r["code"] for r in (out.get("replay_sufficiency") or {}).get("reasons") or []]


def test_collection_carries_a_named_gap_into_the_decision():
    """The observation has to reach the emitted section to mean anything.

    It is persisted onto durable state at the KEEP, but the section the rules
    read is built separately -- and its builder drops falsy values, so a
    tri-state projected through it loses both of its meaningful states.
    """
    assert "build_extensions_not_carried" in _collected_codes(["_moe_C.abi3.so"])


def test_collection_carries_an_unverifiable_scan_into_the_decision():
    assert "build_carry_unverified" in _collected_codes(None)


def test_collection_of_a_clean_scan_names_neither():
    codes = _collected_codes([])

    assert "build_extensions_not_carried" not in codes
    assert "build_carry_unverified" not in codes


def test_a_session_predating_the_observation_names_neither():
    """An older session never recorded it; absence is not an unreadable build."""
    codes = _collected_codes(None, present=False)

    assert "build_extensions_not_carried" not in codes
    assert "build_carry_unverified" not in codes


def _collected_lever_codes(value, *, present=True):
    state = {"enablement_attempts": 1, "enablement_kept_rounds": [{"patches": [], "artifacts": []}]}
    if present:
        state["enablement_levers_without_readers"] = value
    out = collect_enablement(Path("/tmp/sess"), state, [])
    return [r["code"] for r in (out.get("replay_sufficiency") or {}).get("reasons") or []]


def test_collection_carries_a_dangling_lever_into_the_decision():
    """The observation is useless unless it reaches the section the rules read."""
    assert "lever_has_no_reader" in _collected_lever_codes(["VLLM_HL_MQA_LOGITS_HEAD_CHUNK"])


def test_collection_carries_an_unscannable_lever_check():
    assert "levers_unverified" in _collected_lever_codes(None)


def test_collection_of_fully_read_levers_names_neither():
    codes = _collected_lever_codes([])

    assert "lever_has_no_reader" not in codes
    assert "levers_unverified" not in codes


def test_a_session_predating_the_lever_scan_names_neither():
    codes = _collected_lever_codes(None, present=False)

    assert "lever_has_no_reader" not in codes
    assert "levers_unverified" not in codes
