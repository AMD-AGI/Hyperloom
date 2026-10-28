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
from hyperloom.contracts.recipe import LaunchConfig
from hyperloom_multibranch.scheduling.tests.child_sessions import (
    AGENTX_BENCHMARK,
    AGENTX_RUN_PERF,
    AGENTX_RUN_SECONDS,
    BASELINE_ERROR_CLASS,
    BASELINE_TPUT,
    COMPLETED_STOP_REASON,
    REMOTE_URL,
    ROOFLINE_PEAK,
    child_with_patch,
    failed_child,
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
COMMIT = "ab" * 20


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
    artifacts = read_prelude(child_with_patch(tmp_path))

    assert artifacts.framework_root == tmp_path / "framework"
    assert artifacts.roofline_ceiling == {GRADED_OUTPUT: ROOFLINE_PEAK}
    assert artifacts.framework_version == "0.4.1"


def test_a_child_that_named_no_framework_checkout_is_read(tmp_path: Path) -> None:
    session = write_child_session(
        tmp_path / "child",
        {"schema_version": MB_LATEST_STATE_SCHEMA_VERSION, "stop_reason": COMPLETED_STOP_REASON, "baseline_tput": 9.0},
    )
    artifacts = read_prelude(session)
    assert artifacts.framework_root is None
    assert artifacts.roofline_ceiling == {}
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


@pytest.mark.parametrize("stop_reason", ["baseline_arg_error", "time_exhausted_during_prelude", COMPLETED_STOP_REASON])
def test_a_child_without_a_baseline_is_refused_naming_how_it_ended(tmp_path: Path, stop_reason: str) -> None:
    session = write_child_session(
        tmp_path / "child",
        {"schema_version": MB_LATEST_STATE_SCHEMA_VERSION, "stop_reason": stop_reason, "baseline_tput": 0.0},
    )
    with pytest.raises(PreludeReadError, match=f"stop_reason={stop_reason!r} and baseline_tput=0.0$"):
        read_prelude(session)


def test_a_child_whose_baseline_failed_is_refused_naming_its_last_failure(tmp_path: Path) -> None:
    with pytest.raises(PreludeReadError) as refused:
        read_prelude(failed_child(tmp_path))

    assert str(refused.value) == (
        "the child ended with stop_reason='baseline_failed' and baseline_tput=0.0; "
        f"its last baseline attempt failed with {BASELINE_ERROR_CLASS!r}"
    )


def test_other_schema_version_or_missing_state_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PreludeReadError, match="cannot read"):
        read_prelude(tmp_path)
    write_child_session(tmp_path, {"schema_version": MB_LATEST_STATE_SCHEMA_VERSION + 1})
    with pytest.raises(PreludeReadError, match="schema version"):
        read_prelude(tmp_path)


def test_the_framework_is_pinned_at_the_head_of_its_checkout(tmp_path: Path) -> None:
    artifacts = read_prelude(child_with_patch(tmp_path))
    assert framework_pin(artifacts) == UpstreamPin(REMOTE_URL, head_of(tmp_path / "framework"))


def test_framework_without_remote_cannot_be_pinned(tmp_path: Path) -> None:
    session = write_child_session(tmp_path / "child", state_document(framework_checkout(tmp_path, remote="")))
    with pytest.raises(PreludeReadError, match="no git remote"):
        framework_pin(read_prelude(session))


def test_a_framework_path_that_is_no_git_checkout_pins_nothing(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    artifacts = read_prelude(write_child_session(tmp_path / "child", state_document(site_packages)))

    assert artifacts.framework_root is None and framework_pin(artifacts) is None


def test_root_recipe_pins_the_framework_without_a_patch_on_the_main_flows_launch() -> None:
    recipe = root_recipe("sglang", UpstreamPin(REMOTE_URL, COMMIT), IMAGE)
    assert (recipe.base_image_digest, recipe.launch, recipe.unexpressed) == (IMAGE, LaunchConfig(), ())
    component = recipe.component("sglang")
    assert component is not None
    assert (component.upstream_repo, component.upstream_ref, component.patches) == (REMOTE_URL, COMMIT, ())


@pytest.mark.parametrize(
    "pin", [UpstreamPin(REMOTE_URL, "abc1234"), UpstreamPin("", COMMIT)], ids=["short-ref", "no-url"]
)
def test_a_pin_the_recipe_refuses_is_a_read_error(pin: UpstreamPin) -> None:
    with pytest.raises(PreludeReadError, match="not a valid recipe"):
        root_recipe("sglang", pin, IMAGE)


def test_root_recipe_without_framework_checkout_has_no_component() -> None:
    assert root_recipe("sglang", None, IMAGE).components == ()


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


def test_a_child_without_a_baseline_run_is_still_read(tmp_path: Path) -> None:
    artifacts = read_prelude(child_with_patch(tmp_path))

    assert artifacts.baseline_run is None
    assert artifacts.baseline_run_refusal == "the child promoted no baseline that succeeded"
