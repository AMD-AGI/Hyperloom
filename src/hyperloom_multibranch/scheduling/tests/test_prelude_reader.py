# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The prelude reader's contract: which main-flow files and keys it reads, and what it refuses."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest

import hyperloom
from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.contracts.recipe import PatchRef
from hyperloom_multibranch.scheduling.tests.child_sessions import (
    AGENTX_BENCHMARK,
    AGENTX_RUN_PERF,
    AGENTX_RUN_SECONDS,
    BASE_SHA,
    BASELINE_TPUT,
    COMPLETED_STOP_REASON,
    REMOTE_URL,
    ROOFLINE_PEAK,
    child_with_patch,
    framework_checkout,
    head_of,
    state_document,
    with_agentx_baseline_run,
    with_roofline_profile,
    write_child_session,
)

from hyperloom_multibranch.scheduling.prelude_reader import (
    MB_LATEST_STATE_SCHEMA_VERSION,
    ChildBaselineRun,
    MB_TRACE_FILE_GLOB,
    ChildProfile,
    PreludeArtifacts,
    PreludeReadError,
    UpstreamPin,
    framework_pin,
    read_child_session,
    read_prelude,
    root_recipe,
)
from hyperloom_multibranch.toolbox.analysis import TRACE_GLOB

#: The image the toolbox measures on, which the root recipe carries.
IMAGE = "sha256:" + "1" * 64


def test_pinned_schema_version_matches_the_main_flow() -> None:
    source = Path(hyperloom.__file__).parent / "orchestrator" / "state" / "shared_state.py"
    match = re.search(r"^LATEST_STATE_SCHEMA_VERSION: int = (\d+)$", source.read_text(), re.MULTILINE)
    assert match is not None
    assert int(match.group(1)) == MB_LATEST_STATE_SCHEMA_VERSION


def test_a_profile_holds_the_trace_the_toolbox_accepts() -> None:
    assert MB_TRACE_FILE_GLOB == TRACE_GLOB


def test_child_session_comes_from_launch_info(tmp_path: Path) -> None:
    launch = tmp_path / "launch.json"
    assert read_child_session(launch) is None
    launch.write_text(json.dumps({"session_dir": "/data/m/20260101-abc"}))
    assert read_child_session(launch) == Path("/data/m/20260101-abc")
    launch.write_text(json.dumps({"pid": 1}))
    with pytest.raises(PreludeReadError, match="session_dir"):
        read_child_session(launch)


def test_completed_child_session_is_read(tmp_path: Path) -> None:
    session = child_with_patch(tmp_path)

    artifacts = read_prelude(session)

    assert artifacts.succeeded
    assert artifacts.baseline_tput == BASELINE_TPUT
    assert artifacts.launch.server_args == "--mem-fraction-static 0.8"
    assert dict(artifacts.launch.envs) == {"FOO": "1"}
    assert [kept.patch_file for kept in artifacts.kept_patches] == [session / "enablement" / "1.patch"]
    assert artifacts.foreign_patches == ()
    assert artifacts.framework_root == tmp_path / "framework"
    assert artifacts.roofline_ceiling == {GRADED_OUTPUT: ROOFLINE_PEAK}
    assert artifacts.framework_version == "0.4.1"
    assert artifacts.unexpressed == ("setup_commands",)
    assert artifacts.setup_script == session / "reports" / "enablement" / "enablement_setting.sh"


def test_baseline_without_enablement_is_read(tmp_path: Path) -> None:
    session = write_child_session(
        tmp_path / "child",
        {"schema_version": MB_LATEST_STATE_SCHEMA_VERSION, "stop_reason": COMPLETED_STOP_REASON, "baseline_tput": 9.0},
    )
    artifacts = read_prelude(session)
    assert artifacts.succeeded
    assert artifacts.kept_patches == ()
    assert artifacts.framework_root is None
    assert artifacts.roofline_ceiling == {}
    assert artifacts.setup_script is None
    assert artifacts.profile is None
    assert framework_pin(artifacts) is None


def test_the_childs_latest_roofline_profile_is_located_beside_the_point_config_its_run_measured(
    tmp_path: Path,
) -> None:
    session = child_with_patch(tmp_path)
    earlier = with_roofline_profile(session, "a" * 32)
    latest = with_roofline_profile(session, "b" * 32)
    os.utime(earlier, (1.0, 1.0))

    profile = read_prelude(session).profile

    assert profile == ChildProfile("b" * 32, latest, latest.parents[1] / "baseline_config.with_envs.yaml")


def test_a_later_roofline_run_whose_capture_failed_does_not_hide_the_earlier_profile(tmp_path: Path) -> None:
    session = child_with_patch(tmp_path)
    profiled = with_roofline_profile(session, "a" * 32)
    os.utime(profiled, (1.0, 1.0))
    failed = with_roofline_profile(session, "b" * 32)
    shutil.rmtree(failed)
    failed.mkdir()
    dangling = session / "runs" / "roofline" / ("c" * 32) / "benchmark_vllm_20261015_064331" / "torch_trace"
    dangling.parent.mkdir(parents=True)
    dangling.symlink_to(tmp_path / "gone")

    assert read_prelude(session).profile == ChildProfile(
        "a" * 32, profiled, profiled.parents[1] / "baseline_config.with_envs.yaml"
    )


def test_a_child_whose_roofline_captures_all_failed_left_no_profile(tmp_path: Path) -> None:
    session = child_with_patch(tmp_path)
    failed = with_roofline_profile(session)
    shutil.rmtree(failed)
    failed.mkdir()

    assert read_prelude(session).profile is None


def test_without_a_roofline_snapshot_the_baseline_arm_ceiling_is_read(tmp_path: Path) -> None:
    state = {
        "schema_version": MB_LATEST_STATE_SCHEMA_VERSION,
        "stop_reason": COMPLETED_STOP_REASON,
        "baseline_tput": BASELINE_TPUT,
        "roofline_snapshots": [{"theoretical_peak_tok_per_sec": 0.0}],
        "baseline_roofline_ceiling": {"theoretical_peak_tok_per_sec": ROOFLINE_PEAK, "ceiling_arm": "baseline"},
    }

    artifacts = read_prelude(write_child_session(tmp_path / "child", state))

    assert artifacts.roofline_ceiling == {GRADED_OUTPUT: ROOFLINE_PEAK}


@pytest.mark.parametrize(
    ("stop_reason", "baseline_tput"),
    [("enablement_attempts_exhausted", 0.0), ("time_exhausted_during_prelude", 0.0), (COMPLETED_STOP_REASON, 0.0)],
)
def test_child_without_a_baseline_did_not_succeed(tmp_path: Path, stop_reason: str, baseline_tput: float) -> None:
    session = write_child_session(
        tmp_path / "child",
        {
            "schema_version": MB_LATEST_STATE_SCHEMA_VERSION,
            "stop_reason": stop_reason,
            "baseline_tput": baseline_tput,
        },
    )
    assert not read_prelude(session).succeeded


def test_other_schema_version_or_missing_state_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PreludeReadError, match="cannot read"):
        read_prelude(tmp_path)
    write_child_session(tmp_path, {"schema_version": MB_LATEST_STATE_SCHEMA_VERSION + 1})
    with pytest.raises(PreludeReadError, match="schema version"):
        read_prelude(tmp_path)


def test_missing_kept_patch_file_is_refused(tmp_path: Path) -> None:
    framework = framework_checkout(tmp_path)
    write_child_session(tmp_path / "child", state_document(framework, tmp_path / "gone.patch"))
    with pytest.raises(PreludeReadError, match="does not exist"):
        read_prelude(tmp_path / "child")


def _with_enablement(session: Path, **changes: object) -> PreludeArtifacts:
    state = json.loads((session / "state.json").read_text())
    state["enablement"].update(changes)
    return read_prelude(write_child_session(session, state))


def test_framework_is_pinned_at_the_base_recorded_for_any_path_in_its_checkout(tmp_path: Path) -> None:
    session = child_with_patch(tmp_path)
    framework = tmp_path / "framework"
    (framework / "python").mkdir()
    assert framework_pin(read_prelude(session)) == UpstreamPin(REMOTE_URL, BASE_SHA)
    for spelling in (f"{framework}/", str(framework / "python")):
        artifacts = _with_enablement(session, base_sha_by_root={spelling: BASE_SHA})
        assert framework_pin(artifacts) == UpstreamPin(REMOTE_URL, BASE_SHA)


def test_framework_without_kept_patches_or_a_recorded_base_is_pinned_at_head(tmp_path: Path) -> None:
    artifacts = _with_enablement(child_with_patch(tmp_path), kept_patches=[], base_sha_by_root={})
    assert framework_pin(artifacts) == UpstreamPin(REMOTE_URL, head_of(tmp_path / "framework"))


def test_kept_patches_without_a_recorded_base_are_refused(tmp_path: Path) -> None:
    artifacts = _with_enablement(child_with_patch(tmp_path), base_sha_by_root={"/elsewhere": BASE_SHA})
    with pytest.raises(PreludeReadError, match="recorded no commit"):
        framework_pin(artifacts)


def test_framework_without_remote_cannot_be_pinned(tmp_path: Path) -> None:
    framework = framework_checkout(tmp_path, remote="")
    session = write_child_session(tmp_path / "child", state_document(framework, tmp_path / "unused.patch"))
    artifacts = _with_enablement(session, kept_patches=[])
    with pytest.raises(PreludeReadError, match="no git remote"):
        framework_pin(artifacts)


def test_root_recipe_pins_the_framework_with_the_stored_patches(tmp_path: Path) -> None:
    artifacts = read_prelude(child_with_patch(tmp_path))
    patch = PatchRef("patch/sglang/0001.patch", "c" * 64)
    recipe = root_recipe(artifacts, "sglang", UpstreamPin(REMOTE_URL, BASE_SHA), (patch,), IMAGE)
    assert recipe.base_image_digest == IMAGE
    assert recipe.launch == artifacts.launch
    component = recipe.component("sglang")
    assert component is not None
    assert (component.upstream_repo, component.upstream_ref, component.patches) == (REMOTE_URL, BASE_SHA, (patch,))


@pytest.mark.parametrize(
    "pin", [UpstreamPin(REMOTE_URL, "abc1234"), UpstreamPin("", BASE_SHA)], ids=["short-ref", "no-url"]
)
def test_a_pin_the_recipe_refuses_is_a_read_error(tmp_path: Path, pin: UpstreamPin) -> None:
    artifacts = read_prelude(child_with_patch(tmp_path))
    with pytest.raises(PreludeReadError, match="not a valid recipe"):
        root_recipe(artifacts, "sglang", pin, (PatchRef("patch/sglang/0001.patch", "c" * 64),), IMAGE)


def test_root_recipe_without_framework_checkout_has_no_component(tmp_path: Path) -> None:
    session = write_child_session(
        tmp_path / "child",
        {"schema_version": MB_LATEST_STATE_SCHEMA_VERSION, "stop_reason": COMPLETED_STOP_REASON, "baseline_tput": 9.0},
    )
    assert root_recipe(read_prelude(session), "sglang", None, (), IMAGE).components == ()


def _patched_elsewhere(session: Path, tree: Path) -> Path:
    """``session`` with its one kept patch applying to ``tree`` instead of the framework checkout."""

    state = json.loads((session / "state.json").read_text())
    patch_file = state["enablement"]["kept_patches"][0]
    state["enablement"]["patch_roots"] = {patch_file: str(tree)}
    return write_child_session(session, state)


@pytest.mark.parametrize("tree", ["site-packages", "other-checkout"])
def test_a_patch_off_the_framework_checkout_marks_the_root_instead_of_failing(tmp_path: Path, tree: str) -> None:
    root = tmp_path / tree
    if tree == "other-checkout":
        framework_checkout(root)
        root = root / "framework"
    else:
        root.mkdir()
    session = _patched_elsewhere(child_with_patch(tmp_path), root)

    artifacts = read_prelude(session)
    recipe = root_recipe(artifacts, "sglang", framework_pin(artifacts), (), IMAGE)

    assert artifacts.kept_patches == () and [patch.root for patch in artifacts.foreign_patches] == [root]
    assert recipe.unexpressed == (f"patch:{root}/1.patch", "setup_commands")
    assert not recipe.complete
    component = recipe.component("sglang")
    assert component is not None and component.patches == ()


def test_a_framework_root_that_is_no_git_checkout_pins_nothing_and_its_patches_mark_the_root(tmp_path: Path) -> None:
    session = child_with_patch(tmp_path)
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    artifacts = _with_enablement(session, framework_root=str(site_packages), patch_roots={}, base_sha_by_root={})

    assert artifacts.framework_root is None and framework_pin(artifacts) is None
    recipe = root_recipe(artifacts, "sglang", None, (), IMAGE)
    assert recipe.components == ()
    assert recipe.unexpressed == (f"patch:{site_packages}/1.patch", "setup_commands")


def test_harness_flags_are_stripped_and_unexpressed_launch_controls_are_named(tmp_path: Path) -> None:
    accepted = {
        "extra_server_args": "--mem-fraction-static 0.8 --no-enable-prefix-caching",
        "remove_args": ["--foo"],
        "unset_envs": ["BAR"],
        "args_mode": "replace",
    }
    session = write_child_session(
        tmp_path / "child",
        {
            "schema_version": MB_LATEST_STATE_SCHEMA_VERSION,
            "stop_reason": COMPLETED_STOP_REASON,
            "baseline_tput": BASELINE_TPUT,
            "enablement": {"accepted_config": accepted},
        },
    )

    artifacts = read_prelude(session)

    assert artifacts.launch.server_args == "--mem-fraction-static 0.8"
    assert artifacts.unexpressed == ("remove_args", "unset_envs", "args_mode")
    assert root_recipe(artifacts, "sglang", None, (), IMAGE).unexpressed == ("args_mode", "remove_args", "unset_envs")


def _with_state(session: Path, **fields: object) -> Path:
    state = json.loads((session / "state.json").read_text())
    return write_child_session(session, {**state, **fields})


def test_the_baseline_run_is_the_one_the_childs_anchor_was_promoted_from(tmp_path: Path) -> None:
    child = child_with_patch(tmp_path)
    run = with_agentx_baseline_run(child)

    artifacts = read_prelude(child)

    assert artifacts.baseline_run == ChildBaselineRun(
        run_dir=run,
        benchmark_dir=run / AGENTX_BENCHMARK,
        point_config=run / "baseline_config.with_envs.yaml",
        wall_seconds=AGENTX_RUN_SECONDS,
        p50=AGENTX_RUN_PERF["e2e_norm_intvty_p50"],
    )
    assert artifacts.baseline_run_refusal == ""


def test_after_a_successful_enablement_only_its_revalidation_measured_the_root(tmp_path: Path) -> None:
    child = child_with_patch(tmp_path)
    revalidated = with_agentx_baseline_run(child, extras={"baseline_tput": BASELINE_TPUT, "is_revalidation": True})
    state = json.loads((child / "state.json").read_text())
    _with_state(child, enablement={**state["enablement"], "succeeded": True})

    run = read_prelude(child).baseline_run

    assert run is not None and run.run_dir == revalidated


@pytest.mark.parametrize(
    ("anchor", "state", "refusal"),
    [
        ({"status": "failed"}, {}, "promoted no baseline that succeeded"),
        ({"extras": {"baseline_tput": BASELINE_TPUT / 2}}, {}, "not the anchor's baseline_tput"),
        ({}, {"baseline_measure_round_dropped": True}, "cold start"),
        ({}, {"baseline_runtime_sec": None}, "no baseline_runtime_sec"),
    ],
)
def test_a_baseline_run_that_cannot_stand_for_the_root_is_refused_with_its_reason(
    tmp_path: Path, anchor: dict[str, object], state: dict[str, object], refusal: str
) -> None:
    child = child_with_patch(tmp_path)
    with_agentx_baseline_run(child, **anchor)
    _with_state(child, **state)

    artifacts = read_prelude(child)

    assert artifacts.baseline_run is None and refusal in artifacts.baseline_run_refusal


def test_an_anchor_promoted_before_the_enablement_succeeded_is_refused(tmp_path: Path) -> None:
    child = child_with_patch(tmp_path)
    with_agentx_baseline_run(child)
    state = json.loads((child / "state.json").read_text())
    _with_state(child, enablement={**state["enablement"], "succeeded": True})

    artifacts = read_prelude(child)

    assert artifacts.baseline_run is None and "after the anchor was promoted" in artifacts.baseline_run_refusal


def test_a_boot_origin_anchor_that_closed_the_enablement_lane_measured_the_root(tmp_path: Path) -> None:
    child = child_with_patch(tmp_path)
    extras = {"baseline_tput": BASELINE_TPUT, "is_revalidation": False, "enablement_succeeded": True}
    anchored = with_agentx_baseline_run(child, extras=extras)
    state = json.loads((child / "state.json").read_text())
    _with_state(child, enablement={**state["enablement"], "succeeded": True})

    run = read_prelude(child).baseline_run

    assert run is not None and run.run_dir == anchored


def test_a_child_without_a_baseline_run_is_still_read(tmp_path: Path) -> None:
    artifacts = read_prelude(child_with_patch(tmp_path))

    assert artifacts.succeeded and artifacts.baseline_run is None
    assert artifacts.baseline_run_refusal == "the child promoted no baseline that succeeded"
