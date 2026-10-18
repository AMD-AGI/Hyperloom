# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The prelude reader's contract: which main-flow files and keys it reads, and what it refuses."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import hyperloom
from hyperloom.contracts.recipe import PatchRef
from hyperloom_multibranch.scheduling.tests.child_sessions import (
    BASE_SHA,
    BASELINE_TPUT,
    COMPLETED_STOP_REASON,
    REMOTE_URL,
    ROOFLINE_PEAK,
    child_with_patch,
    framework_checkout,
    head_of,
    state_document,
    write_child_session,
)
from hyperloom_multibranch.scheduling.settings import PRIMARY_METRIC
from hyperloom_multibranch.scheduling.prelude_reader import (
    MB_LATEST_STATE_SCHEMA_VERSION,
    PreludeArtifacts,
    PreludeReadError,
    UpstreamPin,
    framework_pin,
    read_child_session,
    read_prelude,
    root_recipe,
)

#: The image the toolbox measures on, which the root recipe carries.
IMAGE = "sha256:" + "1" * 64


def test_pinned_schema_version_matches_the_main_flow() -> None:
    source = Path(hyperloom.__file__).parent / "orchestrator" / "state" / "shared_state.py"
    match = re.search(r"^LATEST_STATE_SCHEMA_VERSION: int = (\d+)$", source.read_text(), re.MULTILINE)
    assert match is not None
    assert int(match.group(1)) == MB_LATEST_STATE_SCHEMA_VERSION


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
    assert artifacts.roofline_ceiling == {PRIMARY_METRIC: ROOFLINE_PEAK}
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
    assert framework_pin(artifacts) is None


def test_without_a_roofline_snapshot_the_baseline_arm_ceiling_is_read(tmp_path: Path) -> None:
    state = {
        "schema_version": MB_LATEST_STATE_SCHEMA_VERSION,
        "stop_reason": COMPLETED_STOP_REASON,
        "baseline_tput": BASELINE_TPUT,
        "roofline_snapshots": [{"theoretical_peak_tok_per_sec": 0.0}],
        "baseline_roofline_ceiling": {"theoretical_peak_tok_per_sec": ROOFLINE_PEAK, "ceiling_arm": "baseline"},
    }

    artifacts = read_prelude(write_child_session(tmp_path / "child", state))

    assert artifacts.roofline_ceiling == {PRIMARY_METRIC: ROOFLINE_PEAK}


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
