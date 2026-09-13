# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-root identity, base_sha capture point, and content capture at the KEEP."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.breakdown.collectors.sessions import collect_enablement
from hyperloom.orchestrator.actions.executors._patch_snapshot import (
    _git_commit_kept,
    patch_declared_ops,
    replayed_stack_ops,
)
from hyperloom.orchestrator.actions.executors.integrate_patch import (
    IntegratePatchExecutor,
    KeepStackStateUnavailable,
    _git_head_sha,
)
from hyperloom.orchestrator.enablement.lane import _rearm_on_kept
from hyperloom.orchestrator.enablement.recipe.keep_records import (
    accepted_stack_artifacts,
    build_root_records,
    capture_root_snapshots,
    classify_root,
    collect_contributions,
    declared_targets,
)
from hyperloom.orchestrator.enablement.recipe.keep_probe import (
    _probe_env,
    keep_assertion_packages,
    probe_environment_closure,
    resolve_keep_interpreter,
)
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound

BUILD_TASK = "tb-1"
PROBE_TASK = "probe-9"
BASE_TEXT = "value = 1\n"
PATCHED_TEXT = "value = 2\n"
TARGET = "srt/module.py"


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "framework"
    (root / "srt").mkdir(parents=True)
    (root / TARGET).write_text(BASE_TEXT, encoding="utf-8")
    _git(root.parent, "init", "-q", str(root))
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


def test_base_sha_is_the_tree_the_patches_apply_to(repo: Path):
    """The recorded sha must predate the KEEP commit for that root.

    Reading HEAD after the KEEP commit records a tree that already contains the
    patch, so a replay applying the recorded patch step to the recorded base
    would apply it a second time.
    """
    recorded = _git_head_sha(repo)

    (repo / TARGET).write_text(PATCHED_TEXT, encoding="utf-8")
    ok, _note = _git_commit_kept(repo, "hyperloom KEEP", [TARGET])
    assert ok

    post_keep = _git_head_sha(repo)
    assert recorded != post_keep
    assert recorded == _git(repo, "rev-parse", "HEAD~1")
    # The patch applies exactly once against the recorded base: that tree still
    # holds the pre-patch content.
    assert _git(repo, "show", f"{recorded}:{TARGET}") == BASE_TEXT.strip()
    assert _git(repo, "show", f"{post_keep}:{TARGET}") == PATCHED_TEXT.strip()


def test_a_kept_patch_applies_exactly_once_to_the_recorded_base(repo: Path, tmp_path: Path):
    """The recorded base is the tree an R1a patch step replays against.

    Applying it to the post-KEEP commit is the failure the capture point exists
    to prevent: that tree already holds the change.
    """
    patch = tmp_path / "1.patch"
    patch.write_text(
        f"--- a/{TARGET}\n+++ b/{TARGET}\n@@ -1 +1 @@\n-{BASE_TEXT.strip()}\n+{PATCHED_TEXT.strip()}\n",
        encoding="utf-8",
    )
    recorded = _git_head_sha(repo)

    (repo / TARGET).write_text(PATCHED_TEXT, encoding="utf-8")
    ok, _note = _git_commit_kept(repo, "hyperloom KEEP", [TARGET])
    assert ok

    records = build_root_records(
        contributions={str(repo): {"patch_apply"}},
        base_sha_by_root={str(repo): recorded},
        git_roots=[str(repo)],
        session_framework_root=str(repo),
    )
    replay = tmp_path / "replay"
    _git(repo.parent, "clone", "-q", str(repo), str(replay))
    _git(replay, "checkout", "-q", records[0]["base_sha"])

    _git(replay, "apply", str(patch))
    assert (replay / TARGET).read_text(encoding="utf-8") == PATCHED_TEXT

    # A second application has nothing left to change, which is what proves the
    # recorded base was not already patched.
    second = subprocess.run(
        ["git", "-C", str(replay), "apply", "--check", str(patch)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode != 0


def test_a_patch_replayed_against_the_post_keep_commit_is_a_second_application(repo: Path, tmp_path: Path):
    patch = tmp_path / "1.patch"
    patch.write_text(
        f"--- a/{TARGET}\n+++ b/{TARGET}\n@@ -1 +1 @@\n-{BASE_TEXT.strip()}\n+{PATCHED_TEXT.strip()}\n",
        encoding="utf-8",
    )
    (repo / TARGET).write_text(PATCHED_TEXT, encoding="utf-8")
    _git_commit_kept(repo, "hyperloom KEEP", [TARGET])

    refused = subprocess.run(
        ["git", "-C", str(repo), "apply", "--check", str(patch)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0


def test_root_record_carries_the_pre_keep_base_sha(repo: Path):
    recorded = _git_head_sha(repo)
    (repo / TARGET).write_text(PATCHED_TEXT, encoding="utf-8")
    _git_commit_kept(repo, "hyperloom KEEP", [TARGET])

    records = build_root_records(
        contributions={str(repo): {"patch_apply"}},
        base_sha_by_root={str(repo): recorded},
        git_roots=[str(repo)],
        session_framework_root=str(repo),
    )
    assert records[0]["base_sha"] == recorded
    assert records[0]["base_sha"] != _git_head_sha(repo)


def test_non_git_root_has_a_null_base_sha_beside_its_is_git_flag():
    records = build_root_records(
        contributions={"/plain/tree": {"artifact_install"}},
        base_sha_by_root={},
        git_roots=[],
        session_framework_root="/fr",
    )
    assert records[0]["is_git"] is False and records[0]["base_sha"] == ""


def test_root_kinds_and_replay_anchors():
    assert classify_root("/fr", session_framework_root="/fr") == (
        "framework_checkout",
        {"anchor": "framework_root", "rel": ""},
    )
    kind, target = classify_root("/opt/venv/lib/python3.10/site-packages/aiter", session_framework_root="/fr")
    assert kind == "site_packages" and target == {"anchor": "site_packages", "rel": "aiter"}
    assert classify_root("/elsewhere", session_framework_root="/fr")[1]["anchor"] == "unmappable"


def test_contributions_split_inputs_from_output_targets():
    contributions = collect_contributions(
        framework_root="/fr",
        patch_roots={"/p/1.patch": "/fr"},
        artifacts=[{"target": "/pkg/a.py", "rel_target": "a.py", "root": "/pkg"}],
    )
    assert contributions["/fr"] == {"patch_apply"}
    assert contributions["/pkg"] == {"artifact_install"}


def test_artifact_only_contributions_do_not_fabricate_a_patch_binding():
    contributions = collect_contributions(
        framework_root="/fr",
        patch_roots={},
        artifacts=[{"target": "/pkg/a.py", "rel_target": "a.py", "root": "/pkg"}],
    )
    assert contributions == {"/pkg": {"artifact_install"}}


def test_keep_records_project_launch_evidence_before_returning_it(repo: Path, tmp_path: Path, monkeypatch):
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    monkeypatch.setattr(executor, "_probe_keep_environment", lambda *_args, **_kwargs: ({}, {}))
    evidence = {
        "framework": "sglang",
        "requested_server_args": "--tp 2",
        "requested_server_env": {"HF_TOKEN": "secret", "SAFE_SWITCH": "1"},
        "materialized_config_path": "/host/session/config.yaml",
        "actual_server_log_path": "/host/session/server.log",
    }
    out = executor._enablement_keep_records(
        SimpleNamespace(_ip_base_sha_by_root={}, _ip_shared_state=SimpleNamespace(enablement=None)),
        params={},
        specialist_task_id=PROBE_TASK,
        framework_root=repo,
        applied=[],
        applied_artifacts=[],
        done_payload={},
        provision_result=None,
        bench_result={"launch_evidence": evidence},
    )
    durable = out["enablement_launch_evidence"]
    assert durable["requested_server_env_keys"] == ["SAFE_SWITCH"]
    assert "requested_server_env" not in durable
    assert "materialized_config_path" not in durable
    assert "actual_server_log_path" not in durable
    assert "secret" not in str(durable)


@pytest.mark.parametrize("argv_key", ["requested_server_args", "observed_server_launch_flags"])
def test_keep_sanitizer_refusal_reaches_activation_verdict_and_resets(
    repo: Path, tmp_path: Path, monkeypatch, argv_key
):
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    monkeypatch.setattr(executor, "_probe_keep_environment", lambda *_args, **_kwargs: ({}, {}))
    state = SimpleNamespace(enablement=EnablementRound(origin="eval"))
    for argv, refused in (("--flag 'unterminated", True), ("", False)):
        evidence = {
            "framework": "sglang",
            "requested_model_digest": "sha256:model",
            "observed_model_binding": {"model_digest": "sha256:model"},
            "requested_server_args": "",
            "observed_server_launch_flags": "",
            argv_key: argv,
        }
        result = executor._enablement_keep_records(
            SimpleNamespace(_ip_base_sha_by_root={}, _ip_shared_state=state),
            params={},
            specialist_task_id=PROBE_TASK,
            framework_root=repo,
            applied=[],
            applied_artifacts=[],
            done_payload={},
            provision_result=None,
            bench_result={"launch_evidence": evidence},
        )
        if refused:
            assert argv_key not in result["enablement_launch_evidence"]
        _rearm_on_kept(state, result)
        state.enablement = EnablementRound.from_dict(asdict(state.enablement))
        collected = collect_enablement(executor.session_dir, {"enablement": asdict(state.enablement)}, [])
        decision = collected["replay_sufficiency"]
        activation_reasons = [reason for reason in decision["reasons"] if reason["code"] == "activation_incomplete"]
        expected = {"code": "activation_incomplete", "scope": "observed_server_launch_flags", "blocks": "both"}
        assert activation_reasons == ([expected] if refused else [])
        if refused:
            assert decision["status"] == "insufficient"


def test_declared_targets_separate_upserts_from_deletions():
    targets = declared_targets(
        framework_root="/fr",
        upserted=["srt/a.py"],
        deleted=["srt/gone.py"],
        artifacts=[{"rel_target": "srt/art.py", "root": "/fr"}],
    )
    assert targets["/fr"] == {"srt/a.py": "upsert", "srt/gone.py": "delete", "srt/art.py": "upsert"}


def test_snapshot_capture_is_portable_and_records_declared_ops(repo: Path, tmp_path: Path):
    (repo / TARGET).write_text(PATCHED_TEXT, encoding="utf-8")
    records = build_root_records(
        contributions={str(repo): {"patch_apply"}},
        base_sha_by_root={str(repo): "a" * 40},
        git_roots=[str(repo)],
        session_framework_root=str(repo),
    )
    session_dir = tmp_path / "session"
    manifests = capture_root_snapshots(
        records=records,
        targets={str(repo): {TARGET: "upsert", "srt/gone.py": "delete"}},
        dest_root=session_dir / "optimization_stack" / "enablement",
        session_dir=session_dir,
    )
    manifest = manifests[0]
    assert manifest["complete"] is True
    assert {f["rel"]: f["op"] for f in manifest["files"]} == {TARGET: "upsert", "srt/gone.py": "delete"}
    assert "framework_root" not in manifest and "snapshot_dir" not in manifest
    assert manifest["snapshot_ref"] == f"optimization_stack/enablement/{records[0]['id']}"
    assert (session_dir / manifest["snapshot_ref"] / "files" / TARGET).read_text() == PATCHED_TEXT


def test_a_later_capture_of_one_root_leaves_no_earlier_target_behind(repo: Path, tmp_path: Path):
    """The overlay is the declared set, and one root captures into one directory."""
    (repo / "srt" / "b.py").write_text("value = 3\n", encoding="utf-8")
    records = build_root_records(
        contributions={str(repo): {"patch_apply"}},
        base_sha_by_root={str(repo): "a" * 40},
        git_roots=[str(repo)],
        session_framework_root=str(repo),
    )
    session_dir = tmp_path / "session"
    dest_root = session_dir / "optimization_stack" / "enablement"
    capture_root_snapshots(
        records=records, targets={str(repo): {TARGET: "upsert"}}, dest_root=dest_root, session_dir=session_dir
    )
    manifests = capture_root_snapshots(
        records=records,
        targets={str(repo): {"srt/b.py": "upsert"}},
        dest_root=dest_root,
        session_dir=session_dir,
    )
    overlay = session_dir / manifests[0]["snapshot_ref"] / "files"
    assert (overlay / "srt/b.py").is_file()
    assert not (overlay / TARGET).exists()


def test_a_root_absent_from_this_keep_leaves_no_packageable_overlay(repo: Path, tmp_path: Path):
    """Packaging selects the overlay by path, so a stale directory would ship."""
    session_dir = tmp_path / "session"
    dest_root = session_dir / "optimization_stack" / "enablement"
    other = tmp_path / "other"
    (other / "srt").mkdir(parents=True)
    (other / TARGET).write_text("value = 9\n", encoding="utf-8")

    def _capture(root: Path):
        records = build_root_records(
            contributions={str(root): {"patch_apply"}},
            base_sha_by_root={},
            git_roots=[],
            session_framework_root=str(repo),
        )
        return records, capture_root_snapshots(
            records=records,
            targets={str(root): {TARGET: "upsert"}},
            dest_root=dest_root,
            session_dir=session_dir,
        )

    stale_records, _stale = _capture(other)
    stale_dir = dest_root / stale_records[0]["id"]
    assert (stale_dir / "files" / TARGET).is_file()

    _records, manifests = _capture(repo)
    assert not stale_dir.exists()
    assert (session_dir / manifests[0]["snapshot_ref"] / "files" / TARGET).is_file()


def test_a_record_declaring_no_target_keeps_no_earlier_overlay(repo: Path, tmp_path: Path):
    session_dir = tmp_path / "session"
    dest_root = session_dir / "optimization_stack" / "enablement"
    records = build_root_records(
        contributions={str(repo): {"patch_apply"}},
        base_sha_by_root={},
        git_roots=[],
        session_framework_root=str(repo),
    )
    capture_root_snapshots(
        records=records, targets={str(repo): {TARGET: "upsert"}}, dest_root=dest_root, session_dir=session_dir
    )
    assert capture_root_snapshots(records=records, targets={}, dest_root=dest_root, session_dir=session_dir) == []
    assert not (dest_root / records[0]["id"]).exists()


def test_the_captured_stack_is_the_accepted_one_not_this_rounds_installs():
    inherited = [{"target": "/fr/srt/base.py", "rel_target": "srt/base.py", "root": "/fr"}]
    applied = [{"target": "/fr/srt/new.py", "rel_target": "srt/new.py", "root": "/fr"}]
    stack = accepted_stack_artifacts(inherited=inherited, applied=applied)
    assert declared_targets(framework_root="/fr", upserted=[], deleted=[], artifacts=stack)["/fr"] == {
        "srt/base.py": "upsert",
        "srt/new.py": "upsert",
    }


def test_this_rounds_install_supersedes_the_inherited_record_at_one_target():
    inherited = [{"target": "/fr/srt/a.py", "rel_target": "srt/a.py", "root": "/fr", "source": "old"}]
    applied = [{"target": "/fr/srt/a.py", "rel_target": "srt/a.py", "root": "/fr", "source": "new"}]
    stack = accepted_stack_artifacts(inherited=inherited, applied=applied)
    assert [a["source"] for a in stack] == ["new"]


def test_undeclared_absent_target_is_recorded_missing_and_incomplete(repo: Path, tmp_path: Path):
    records = build_root_records(
        contributions={str(repo): {"patch_apply"}},
        base_sha_by_root={},
        git_roots=[str(repo)],
        session_framework_root=str(repo),
    )
    manifests = capture_root_snapshots(
        records=records,
        targets={str(repo): {"srt/never.py": "upsert"}},
        dest_root=tmp_path / "dest",
        session_dir=tmp_path,
    )
    assert manifests[0]["complete"] is False
    assert manifests[0]["files"][0]["op"] == "missing"


def test_keep_interpreter_prefers_the_override_then_the_bypass_backend():
    assert (
        resolve_keep_interpreter({"runtime_python_exe": "/a/py", "framework_python": "/b/py"}, backend_name="bypass")
        == "/a/py"
    )
    assert resolve_keep_interpreter({"framework_python": "/b/py"}, backend_name="magpie") == "/b/py"
    assert resolve_keep_interpreter({}, backend_name="bypass", bypass_interpreter="/c/py") == "/c/py"
    # Under any other backend the launching interpreter is not resolvable, and
    # naming a plausible one would reproduce the defect this closes.
    assert resolve_keep_interpreter({}, backend_name="magpie", bypass_interpreter="/c/py") == ""


def _installed_dist(root: Path, name: str, version: str) -> str:
    """Materialize an importable distribution and return its ``sys.path`` entry."""
    info = root / f"site-{version}" / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
    return str(info.parent)


def _build_manifest(installed_versions: dict[str, str]) -> list[dict]:
    """The single row routing leaves for an executed build: attempt and sentinel."""
    return [
        {
            "ok": True,
            "task_id": BUILD_TASK,
            "probe_task_id": PROBE_TASK,
            "attempt_root": f"/s/enablement/builds/{BUILD_TASK}",
            "installed_versions": installed_versions,
        }
    ]


def test_the_graded_override_decides_what_the_probe_observes(tmp_path: Path):
    """A source build reaches its packages only through the override's prefixes.

    A bare-environment probe would name a distribution set the graded server
    never imported.
    """
    override = {"pythonpath_prefixes": [_installed_dist(tmp_path, "overlaid", "3.1")]}
    closure, assertions = probe_environment_closure(sys.executable, override=override, packages=("overlaid",))
    bare_closure, bare_assertions = probe_environment_closure(sys.executable, override=None, packages=("overlaid",))
    assert closure["distributions"]["overlaid"] == "3.1" and assertions == {"overlaid": "3.1"}
    assert "overlaid" not in bare_closure["distributions"] and bare_assertions == {}


def test_a_build_reached_keep_asserts_the_linked_attempts_versions():
    packages = keep_assertion_packages(
        provision_versions=None,
        build_manifest=_build_manifest({"torch": "2.6", "aiter_sha": "abc1234"}),
        specialist_task_id=PROBE_TASK,
    )
    assert packages == ("torch", "aiter_sha")


def test_a_provisioned_keep_asserts_its_own_map():
    packages = keep_assertion_packages(
        provision_versions={"sglang": "0.4"},
        build_manifest=_build_manifest({"torch": "2.6"}),
        specialist_task_id=PROBE_TASK,
    )
    assert packages == ("sglang",)


def test_neither_provisioning_nor_a_linked_build_names_a_package():
    """A build whose probe is another round's is not this KEEP's version source."""
    assert (
        keep_assertion_packages(
            provision_versions=None,
            build_manifest=_build_manifest({"torch": "2.6"}),
            specialist_task_id="some-other-round",
        )
        == ()
    )
    assert keep_assertion_packages(provision_versions=None, build_manifest=[], specialist_task_id=PROBE_TASK) == ()


def test_a_provisioning_stage_that_installed_nothing_borrows_no_build_names():
    """An empty provisioning map is a stage that named nothing, not an absent one.

    The build's names belong to a KEEP that reached its runtime through the
    build; a round that provisioned its own asserts what that stage installed.
    """
    packages = keep_assertion_packages(
        provision_versions={},
        build_manifest=_build_manifest({"torch": "2.6"}),
        specialist_task_id=PROBE_TASK,
    )
    assert packages == ()


def test_a_build_without_provisioning_observes_versions_at_the_keep(tmp_path: Path):
    """The build map names the packages; only the KEEP probe supplies the versions.

    Sourcing the names from ``provision_result`` alone left the launch-only probe
    path -- the principal path carrying version assertions -- with an empty set,
    which reads as never observed. The recorded version is the one the probe
    finds, so a build-time map carried forward is visible as a stale value.
    """
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    ctx = SimpleNamespace(
        _ip_shared_state=SimpleNamespace(enablement=SimpleNamespace(build_manifest=_build_manifest({"demo": "1.0"})))
    )
    params = {
        "runtime_override": {
            "runtime_python_exe": sys.executable,
            "pythonpath_prefixes": [_installed_dist(tmp_path, "demo", "2.0")],
        }
    }
    closure, assertions = executor._probe_keep_environment(
        ctx, params, specialist_task_id=PROBE_TASK, provision_result=None
    )
    assert assertions == {"demo": "2.0"}
    assert closure["distributions"]["demo"] == "2.0"


def test_a_keep_whose_build_is_another_rounds_observes_nothing(tmp_path: Path):
    """Fail closed: no provisioning and no linked build is no version source."""
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    ctx = SimpleNamespace(
        _ip_shared_state=SimpleNamespace(enablement=SimpleNamespace(build_manifest=_build_manifest({"demo": "1.0"})))
    )
    params = {
        "runtime_override": {
            "runtime_python_exe": sys.executable,
            "pythonpath_prefixes": [_installed_dist(tmp_path, "demo", "2.0")],
        }
    }
    _closure, assertions = executor._probe_keep_environment(
        ctx, params, specialist_task_id="some-other-round", provision_result=None
    )
    assert assertions == {}


def test_an_override_naming_no_interpreter_off_the_bypass_path_observes_nothing(
    tmp_path: Path,
    monkeypatch,
):
    """An AITER runtime names no interpreter, and only bypass can say which ran.

    Under any other backend both probes report nothing rather than an
    environment the graded server never ran in.
    """
    from hyperloom.orchestrator.actions.executors import benchmark_backend

    monkeypatch.setattr(benchmark_backend, "resolve_backend_name", lambda: "magpie")
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    ctx = SimpleNamespace(
        _ip_shared_state=SimpleNamespace(enablement=SimpleNamespace(build_manifest=_build_manifest({"demo": "1.0"})))
    )
    params = {"runtime_override": {"pythonpath_prefixes": [_installed_dist(tmp_path, "demo", "2.0")]}}
    assert executor._probe_keep_environment(ctx, params, specialist_task_id=PROBE_TASK, provision_result=None) == (
        {},
        {},
    )


def test_the_same_override_under_the_bypass_backend_does_observe(tmp_path: Path, monkeypatch):
    """The counterpart: the backend, not the override, is what decides."""
    from hyperloom.orchestrator.actions.executors import benchmark_backend

    monkeypatch.setattr(benchmark_backend, "resolve_backend_name", lambda: "bypass")
    monkeypatch.setattr(benchmark_backend, "resolve_benchmark_interpreter", lambda: sys.executable)
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    ctx = SimpleNamespace(
        _ip_shared_state=SimpleNamespace(enablement=SimpleNamespace(build_manifest=_build_manifest({"demo": "1.0"})))
    )
    params = {"runtime_override": {"pythonpath_prefixes": [_installed_dist(tmp_path, "demo", "2.0")]}}
    closure, assertions = executor._probe_keep_environment(
        ctx, params, specialist_task_id=PROBE_TASK, provision_result=None
    )
    assert assertions == {"demo": "2.0"} and closure["distributions"]["demo"] == "2.0"


def test_a_keep_with_no_usable_runtime_observes_nothing(tmp_path: Path):
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    ctx = SimpleNamespace(_ip_shared_state=SimpleNamespace(enablement=SimpleNamespace(build_manifest=[])))
    assert executor._probe_keep_environment(ctx, {}, specialist_task_id=PROBE_TASK, provision_result=None) == ({}, {})


def test_a_round_spanning_two_roots_names_each_tree_on_its_own_terms(repo: Path, tmp_path: Path):
    """Each contributing tree carries its own binding, git flag and base commit."""
    second = tmp_path / "artifacts_root"
    (second / "lib").mkdir(parents=True)
    (second / "lib" / "a.so").write_bytes(b"\x00artifact")
    _git(second.parent, "init", "-q", str(second))
    _git(second, "config", "user.email", "t@example.com")
    _git(second, "config", "user.name", "t")
    _git(second, "add", "-A")
    _git(second, "commit", "-qm", "artifact base")

    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    ctx = SimpleNamespace(
        _ip_base_sha_by_root={str(repo): _git_head_sha(repo), str(second): _git_head_sha(second)},
        _ip_shared_state=SimpleNamespace(enablement=None),
    )
    out = executor._enablement_keep_records(
        ctx,
        params={},
        specialist_task_id=PROBE_TASK,
        framework_root=repo,
        # In the accepted stack, not merely recorded beside it: a root binding
        # is admitted for the patches this integration took.
        applied=[Path("/p/1.patch")],
        applied_artifacts=[{"target": str(second / "lib/a.so"), "rel_target": "lib/a.so", "root": str(second)}],
        done_payload={"patch_roots": {"/p/1.patch": str(repo)}},
        provision_result=None,
        bench_result={},
    )
    records = {r["path"]: r for r in out["enablement_roots"]}
    assert set(records) == {str(repo), str(second)}
    assert records[str(repo)]["contributions"] == ["patch_apply"]
    assert records[str(second)]["contributions"] == ["artifact_install"]
    assert all(r["is_git"] for r in records.values())
    assert records[str(second)]["base_sha"] == _git(second, "rev-parse", "HEAD")
    assert records[str(second)]["base_sha"] != records[str(repo)]["base_sha"]


def test_an_inherited_artifact_is_captured_by_the_keep_that_launched_it(repo: Path, tmp_path: Path):
    """The lane replaces these records with the latest KEEP's, so a round that
    captured only its own installs would drop an earlier round's payload.

    The inheritance arrives on the durable round state, not as a dispatch
    parameter: #1409 ended the "re-apply every prior round before each boot"
    model and deleted ``enablement_base_artifacts``. ``kept_artifacts`` is what
    the earlier round's rearm stacked, and it is the only thing left that says
    an earlier round contributed anything.
    """
    inherited_rel = "srt/inherited.py"
    (repo / inherited_rel).write_text("value = 7\n", encoding="utf-8")
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    ctx = SimpleNamespace(
        _ip_base_sha_by_root={str(repo): _git_head_sha(repo)},
        _ip_shared_state=SimpleNamespace(
            enablement=EnablementRound(
                kept_artifacts=[
                    {"target": str(repo / inherited_rel), "rel_target": inherited_rel, "root": str(repo)}
                ],
            )
        ),
    )
    out = executor._enablement_keep_records(
        ctx,
        params={},
        specialist_task_id=PROBE_TASK,
        framework_root=repo,
        applied=[],
        applied_artifacts=[{"target": str(repo / TARGET), "rel_target": TARGET, "root": str(repo)}],
        done_payload={},
        provision_result=None,
        bench_result={},
    )
    root_id = out["enablement_roots"][0]["id"]
    assert out["enablement_accepted_stack_targets"][root_id] == {TARGET: "upsert", inherited_rel: "upsert"}
    snapshot = out["enablement_source_snapshots"][0]
    assert {f["rel"] for f in snapshot["files"]} == {TARGET, inherited_rel}
    overlay = executor.session_dir / snapshot["snapshot_ref"] / "files"
    assert (overlay / inherited_rel).read_text(encoding="utf-8") == "value = 7\n"


def test_a_keep_that_cannot_see_the_round_state_captures_no_stack_at_all(repo: Path, tmp_path: Path):
    """No durable state means no view of the inherited stack, which must refuse.

    Pins the producer of :class:`KeepStackStateUnavailable`: delete the raise in
    ``_enablement_keep_records`` and this goes green on a silently single-round
    capture, which is the fail-open the whole replay contract exists to deny.
    """
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    ctx = SimpleNamespace(_ip_base_sha_by_root={str(repo): _git_head_sha(repo)})
    with pytest.raises(KeepStackStateUnavailable):
        executor._enablement_keep_records(
            ctx,
            params={},
            specialist_task_id=PROBE_TASK,
            framework_root=repo,
            applied=[],
            applied_artifacts=[{"target": str(repo / TARGET), "rel_target": TARGET, "root": str(repo)}],
            done_payload={},
            provision_result=None,
            bench_result={},
        )


def test_a_non_git_contributing_root_carries_no_base_commit(repo: Path, tmp_path: Path):
    plain = tmp_path / "plain_root"
    (plain / "lib").mkdir(parents=True)
    (plain / "lib" / "a.so").write_bytes(b"\x00artifact")

    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    ctx = SimpleNamespace(
        _ip_base_sha_by_root={str(repo): _git_head_sha(repo)},
        _ip_shared_state=SimpleNamespace(enablement=None),
    )
    out = executor._enablement_keep_records(
        ctx,
        params={},
        specialist_task_id=PROBE_TASK,
        framework_root=repo,
        applied=[Path("/p/1.patch")],
        applied_artifacts=[{"target": str(plain / "lib/a.so"), "rel_target": "lib/a.so", "root": str(plain)}],
        done_payload={"patch_roots": {"/p/1.patch": str(repo)}},
        provision_result=None,
        bench_result={},
    )
    record = next(r for r in out["enablement_roots"] if r["path"] == str(plain))
    assert record["is_git"] is False and record["base_sha"] == ""


def test_a_provisioned_keep_that_installed_nothing_observes_nothing(tmp_path: Path):
    """The caller must distinguish an absent provisioning result from an empty one."""
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    ctx = SimpleNamespace(
        _ip_shared_state=SimpleNamespace(enablement=SimpleNamespace(build_manifest=_build_manifest({"demo": "1.0"})))
    )
    override = {
        "runtime_python_exe": sys.executable,
        "pythonpath_prefixes": [_installed_dist(tmp_path, "demo", "2.0")],
    }
    provisioned = SimpleNamespace(
        ok=True,
        installed_versions={},
        runtime=SimpleNamespace(to_runtime_override=lambda: dict(override)),
    )
    closure, assertions = executor._probe_keep_environment(
        ctx, {}, specialist_task_id=PROBE_TASK, provision_result=provisioned
    )
    assert assertions == {}
    assert closure["distributions"]["demo"] == "2.0"


def test_a_provisioned_keep_reports_the_version_the_setup_replay_left(tmp_path: Path):
    """Provisioning names the package; the version is the probe's observation.

    The provisioning stage runs before the setup replay, so carrying its map
    forward would report the version a later install had already replaced.
    """
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    ctx = SimpleNamespace(_ip_shared_state=SimpleNamespace(enablement=SimpleNamespace(build_manifest=[])))
    override = {
        "runtime_python_exe": sys.executable,
        "pythonpath_prefixes": [_installed_dist(tmp_path, "demo", "2.0")],
    }
    provisioned = SimpleNamespace(
        ok=True,
        installed_versions={"demo": "1.0"},
        runtime=SimpleNamespace(to_runtime_override=lambda: dict(override)),
    )
    _closure, assertions = executor._probe_keep_environment(
        ctx, {}, specialist_task_id=PROBE_TASK, provision_result=provisioned
    )
    assert assertions == {"demo": "2.0"}


def test_the_probe_environment_carries_the_prefixes_and_the_runtime_env(tmp_path: Path):
    """An AITER runtime names no interpreter: a prefix list and a runtime_env
    are the whole of what makes its build importable."""
    prefix = _installed_dist(tmp_path, "overlaid", "3.1")
    env = _probe_env({"pythonpath_prefixes": [prefix], "runtime_env": {"AITER_REBUILD": "1"}})
    assert env["PYTHONPATH"].split(":")[0] == prefix
    assert env["AITER_REBUILD"] == "1"


def test_an_interpreter_the_probe_cannot_run_observes_nothing(tmp_path: Path):
    """Fail closed on the invocation too, not only on an unresolved interpreter."""
    assert probe_environment_closure(str(tmp_path / "absent-python"), override=None, packages=("demo",)) == ({}, {})
    silent = tmp_path / "silent-python"
    silent.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    silent.chmod(0o755)
    assert probe_environment_closure(str(silent), override=None, packages=("demo",)) == ({}, {})


# --------------------------------------------------------------------------
# The declared operation of a patch, and where it is read from.
# --------------------------------------------------------------------------


def _patch(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_a_deletion_is_declared_by_its_dev_null_post_image(repo: Path, tmp_path: Path):
    patch = _patch(
        tmp_path,
        "del.patch",
        f"--- a/{TARGET}\n+++ /dev/null\n@@ -1 +0,0 @@\n-{BASE_TEXT}",
    )
    assert patch_declared_ops(repo, [patch]) == {TARGET: "delete"}


def test_a_creation_is_an_upsert_even_when_the_tree_does_not_hold_it(repo: Path, tmp_path: Path):
    """The stripped-inputs case, stated as a unit.

    A KEEP reached without its mutation inputs boots against the base tree, so
    the created file is absent at capture time. A tree probe reads that absence
    as a deletion and declares a target the snapshot then satisfies trivially;
    the diff says the file must exist, and that is what a replay must produce.
    """
    rel = "srt/created.py"
    assert not (repo / rel).exists()
    patch = _patch(tmp_path, "new.patch", f"--- /dev/null\n+++ b/{rel}\n@@ -0,0 +1 @@\n+x = 1\n")
    assert patch_declared_ops(repo, [patch]) == {rel: "upsert"}


def test_a_later_patch_recreating_a_deleted_file_wins(repo: Path, tmp_path: Path):
    """Order, not set-merge: the accumulated upsert/delete lists put every
    deletion last, so the recreation loses and the replay removes a file the
    accepted stack requires."""
    first = _patch(tmp_path, "1.patch", f"--- a/{TARGET}\n+++ /dev/null\n@@ -1 +0,0 @@\n-{BASE_TEXT}")
    second = _patch(tmp_path, "2.patch", f"--- /dev/null\n+++ b/{TARGET}\n@@ -0,0 +1 @@\n+{PATCHED_TEXT}")
    assert patch_declared_ops(repo, [first, second]) == {TARGET: "upsert"}
    # ...and the other way round, so this is an ordering rule and not a
    # preference for one operation.
    assert patch_declared_ops(repo, [second, first]) == {TARGET: "delete"}


def test_a_rename_declares_both_ends(repo: Path, tmp_path: Path):
    moved = "srt/moved.py"
    patch = _patch(tmp_path, "mv.patch", f"--- a/{TARGET}\n+++ b/{moved}\n@@ -1 +1 @@\n-{BASE_TEXT}+{BASE_TEXT}")
    assert patch_declared_ops(repo, [patch]) == {TARGET: "delete", moved: "upsert"}


def test_a_plain_modify_declares_no_deletion_of_the_file_it_writes(repo: Path, tmp_path: Path):
    patch = _patch(tmp_path, "mod.patch", f"--- a/{TARGET}\n+++ b/{TARGET}\n@@ -1 +1 @@\n-{BASE_TEXT}+{PATCHED_TEXT}")
    assert patch_declared_ops(repo, [patch]) == {TARGET: "upsert"}


def test_the_keep_records_what_each_kept_patch_declares(repo: Path, tmp_path: Path):
    """Pins the producer of ``enablement_patch_targets``.

    The decision cross-checks each patch step against the snapshot of the files
    that step declares and cannot read the diffs itself, so this mapping is the
    only thing standing between a multi-round recipe and an unverified replay.
    """
    created = "srt/round_one.py"
    first = _patch(tmp_path, "1.patch", f"--- /dev/null\n+++ b/{created}\n@@ -0,0 +1 @@\n+x = 1\n")
    second = _patch(tmp_path, "2.patch", f"--- a/{TARGET}\n+++ b/{TARGET}\n@@ -1 +1 @@\n-{BASE_TEXT}+{PATCHED_TEXT}")
    base_sha = _git_head_sha(repo)
    # Really applied and committed, round by round: the capture proves a patch
    # is present by un-applying it, so a fixture that only writes the end state
    # would be certifying something no patch produced.
    for patch, message in ((first, "round one"), (second, "round two")):
        _git(repo, "apply", str(patch))
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", message)
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    ctx = SimpleNamespace(
        _ip_base_sha_by_root={str(repo): base_sha},
        # The first round is durable state; only the second is this round's.
        _ip_shared_state=SimpleNamespace(
            enablement=EnablementRound(kept_patches=[str(first)], patch_roots={str(first): str(repo)})
        ),
    )
    out = executor._enablement_keep_records(
        ctx,
        params={},
        specialist_task_id=PROBE_TASK,
        framework_root=repo,
        applied=[second],
        applied_artifacts=[],
        done_payload={"patch_roots": {str(second): str(repo)}},
        provision_result=None,
        bench_result={},
    )
    assert out["enablement_patch_targets"] == {
        str(first): {created: "upsert"},
        str(second): {TARGET: "upsert"},
    }
    root_id = out["enablement_roots"][0]["id"]
    # Both rounds' files, not just this round's: the capture is over the stack.
    assert out["enablement_accepted_stack_targets"][root_id] == {created: "upsert", TARGET: "upsert"}


def test_the_base_sha_reported_is_this_roots_own(repo: Path, tmp_path: Path):
    """An earlier round's scalar names the tree THAT round applied to.

    Preferring it outright hands a different root's sha to this one, and the
    snapshot is then captured against a base commit this tree never had.
    """
    other = "/some/other/framework/root"
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    ctx = SimpleNamespace(
        _ip_base_sha_by_root={str(repo): _git_head_sha(repo)},
        _ip_shared_state=SimpleNamespace(
            enablement=EnablementRound(
                base_sha="b" * 40,
                roots=[{"id": "other", "path": other, "base_sha": "b" * 40}],
            )
        ),
    )
    out = executor._enablement_keep_records(
        ctx,
        params={},
        specialist_task_id=PROBE_TASK,
        framework_root=repo,
        applied=[],
        applied_artifacts=[{"target": str(repo / TARGET), "rel_target": TARGET, "root": str(repo)}],
        done_payload={},
        provision_result=None,
        bench_result={},
    )
    assert out["enablement_base_sha"] == _git_head_sha(repo)
    assert out["enablement_base_sha"] != "b" * 40


def test_a_recorded_root_for_a_patch_outside_the_stack_is_ignored(repo: Path, tmp_path: Path):
    """The sibling rule to ``_sole_patch_root``'s selected-set check.

    A ``done_payload`` entry for a patch this integration did not take cannot
    attest anything about the accepted stack. Admitted, it adds a root record
    and a declared-target set for a tree no round wrote, and the capture is then
    judged against files that were never part of the stack.
    """
    from hyperloom.orchestrator.actions.executors.integrate_patch import _accepted_patch_roots

    unrelated = tmp_path / "aiter"
    applied = tmp_path / "2.patch"
    roots = _accepted_patch_roots(
        EnablementRound(),
        done_payload={
            "patch_roots": {
                str(applied): str(repo),
                "unselected-harvest.patch": str(unrelated),
            }
        },
        applied=[applied],
        framework_root=str(repo),
    )
    assert roots == {str(applied): str(repo)}


# --------------------------------------------------------------------------
# The base a multi-round stack replays onto.
# --------------------------------------------------------------------------


def test_an_advanced_round_carries_its_base_to_the_keep_that_follows(repo: Path, tmp_path: Path):
    """ADVANCED commits and stacks a patch while recording no per-root identity.

    Every KEEP is committed too, so by the next round HEAD already contains the
    advanced round's patch. If the KEEP reports THAT head as the recipe's base,
    the recipe names a tree its own first patch step has already been applied
    to, and a consumer replaying it applies that patch a second time -- while
    the verdict certifies the recipe as sufficient.

    Pins the durable ``base_sha_by_root`` write: drop it from
    ``_note_pre_mutation_head`` and this goes red.
    """
    from hyperloom.orchestrator.actions.executors.integrate_patch import _note_pre_mutation_head

    true_base = _git_head_sha(repo)
    state = SimpleNamespace(enablement=EnablementRound(framework_root=str(repo)))

    # Round one: ADVANCED. It captures the head, mutates, and commits.
    _note_pre_mutation_head(SimpleNamespace(_ip_shared_state=state), repo, enablement=True)
    (repo / TARGET).write_text(PATCHED_TEXT, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "advanced round")
    state.enablement.kept_patches = ["/p/1.patch"]
    state.enablement.patch_roots = {"/p/1.patch": str(repo)}
    assert state.enablement.base_sha_by_root == {str(repo): true_base}
    assert _git_head_sha(repo) != true_base, "the advanced round must really have moved HEAD"

    # Round two: a FRESH context, as the next integrate_patch invocation gets.
    ctx = SimpleNamespace(_ip_shared_state=state)
    _note_pre_mutation_head(ctx, repo, enablement=True)
    out = executor_keep_records(tmp_path, ctx, repo)
    assert out["enablement_base_sha"] == true_base
    assert out["enablement_base_sha"] != _git_head_sha(repo)


def executor_keep_records(tmp_path: Path, ctx, repo: Path):
    """Run the KEEP capture for a round that installed one artifact."""
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    return executor._enablement_keep_records(
        ctx,
        params={},
        specialist_task_id=PROBE_TASK,
        framework_root=repo,
        applied=[],
        applied_artifacts=[{"target": str(repo / TARGET), "rel_target": TARGET, "root": str(repo)}],
        done_payload={},
        provision_result=None,
        bench_result={},
    )


def test_an_ordinary_patch_round_does_not_seed_the_enablement_base(repo: Path):
    """The head before an unrelated patch is not the tree the enablement stack
    applies to, so a non-enablement round must not claim the base."""
    from hyperloom.orchestrator.actions.executors.integrate_patch import _note_pre_mutation_head

    state = SimpleNamespace(enablement=EnablementRound())
    _note_pre_mutation_head(SimpleNamespace(_ip_shared_state=state), repo, enablement=False)
    assert state.enablement.base_sha_by_root == {}


def test_the_recorded_base_is_never_replaced_by_a_later_reading(repo: Path):
    from hyperloom.orchestrator.actions.executors.integrate_patch import _note_pre_mutation_head

    state = SimpleNamespace(enablement=EnablementRound())
    _note_pre_mutation_head(SimpleNamespace(_ip_shared_state=state), repo, enablement=True)
    first = dict(state.enablement.base_sha_by_root)
    (repo / TARGET).write_text("moved on\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "later")
    _note_pre_mutation_head(SimpleNamespace(_ip_shared_state=state), repo, enablement=True)
    assert state.enablement.base_sha_by_root == first


# --------------------------------------------------------------------------
# Proving a patch is what the captured tree contains, rather than believing
# its headers.
# --------------------------------------------------------------------------


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)


def test_a_modification_that_was_never_applied_declares_nothing(repo: Path, tmp_path: Path):
    """The stripped-mutation-input case, for a MODIFIED file.

    Header classification calls an ordinary modification ``upsert``, and the
    capture calls any existing file an upsert, so a KEEP that booted without its
    mutation inputs ships the BASE bytes under a declaration saying they were
    changed -- and the recipe's patch step and its snapshot then describe
    different results with nothing objecting.
    """
    base_sha = _git_head_sha(repo)
    patch = _patch(tmp_path, "mod.patch", f"--- a/{TARGET}\n+++ b/{TARGET}\n@@ -1 +1 @@\n-{BASE_TEXT}+{PATCHED_TEXT}")
    # The headers still say what the patch WOULD do...
    assert patch_declared_ops(repo, [patch]) == {TARGET: "upsert"}
    # ...but the replay produces bytes the captured tree does not have.
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) is None
    assert (repo / TARGET).read_text(encoding="utf-8") == BASE_TEXT

    _git(repo, "apply", str(patch))
    _commit_all(repo, "applied")
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) == {str(patch): {TARGET: "upsert"}}


def test_an_unrecorded_edit_outside_the_patch_breaks_the_replay(repo: Path, tmp_path: Path):
    """Reverse-applying a patch only inspects its own hunks, so an edit made
    outside them passed while base + patch no longer reproduced the captured
    bytes. Replaying forward compares the whole declared file."""
    base_sha = _git_head_sha(repo)
    patch = _patch(tmp_path, "mod.patch", f"--- a/{TARGET}\n+++ b/{TARGET}\n@@ -1 +1 @@\n-{BASE_TEXT}+{PATCHED_TEXT}")
    _git(repo, "apply", str(patch))
    _commit_all(repo, "applied")
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) == {str(patch): {TARGET: "upsert"}}

    (repo / TARGET).write_text(PATCHED_TEXT + "an unrecorded line\n", encoding="utf-8")
    _commit_all(repo, "an edit no patch declares")
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) is None


def test_a_never_applied_patch_cannot_borrow_a_matching_block_elsewhere(repo: Path, tmp_path: Path):
    """Git finds a postimage with an offset, so reverse-applying a patch that
    was never applied succeeded against a similar block further down the file.
    The forward replay meets the exact preimage instead."""
    twin = "srt/twin.py"
    (repo / twin).write_text("header\nv=old\ntail\n" + "".join(f"x{i}\n" for i in range(10)) + "header\nv=new\ntail\n", encoding="utf-8")
    _commit_all(repo, "a file with two similar blocks")
    base_sha = _git_head_sha(repo)
    patch = _patch(tmp_path, "twin.patch", f"--- a/{twin}\n+++ b/{twin}\n@@ -1,3 +1,3 @@\n header\n-v=old\n+v=new\n tail\n")
    # Never applied: the tree still holds the base.
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) is None


def test_a_content_and_mode_patch_must_agree_on_the_mode(repo: Path, tmp_path: Path):
    """Git treats a mode disagreement as a warning, not a failed apply, and the
    snapshot copies whatever mode it finds -- so a recipe could restore a
    script without its execute bit and still certify."""
    script = "srt/run.sh"
    (repo / script).write_text("#!/bin/sh\necho one\n", encoding="utf-8")
    _commit_all(repo, "add the script")
    base_sha = _git_head_sha(repo)
    patch = _patch(
        tmp_path,
        "mode.patch",
        f"diff --git a/{script} b/{script}\nold mode 100644\nnew mode 100755\n"
        f"--- a/{script}\n+++ b/{script}\n@@ -1,2 +1,2 @@\n #!/bin/sh\n-echo one\n+echo two\n",
    )
    _git(repo, "apply", str(patch))
    _commit_all(repo, "content and mode")
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) == {str(patch): {script: "upsert"}}

    # Same bytes, mode put back: the replay would produce an executable file.
    (repo / script).chmod(0o644)
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) is None


def test_overlapping_rounds_on_one_file_both_verify(repo: Path, tmp_path: Path):
    """Each patch's preimage is the tree its predecessors left behind. Checked
    independently against the FINAL tree, the earlier patch of two edits to one
    file could never reverse, and a valid stack was refused."""
    base_sha = _git_head_sha(repo)
    first = _patch(tmp_path, "a1.patch", f"--- a/{TARGET}\n+++ b/{TARGET}\n@@ -1 +1 @@\n-{BASE_TEXT}+middle\n")
    second = _patch(tmp_path, "a2.patch", f"--- a/{TARGET}\n+++ b/{TARGET}\n@@ -1 +1 @@\n-middle\n+{PATCHED_TEXT}")
    for patch, msg in ((first, "r1"), (second, "r2")):
        _git(repo, "apply", str(patch))
        _commit_all(repo, msg)
    assert replayed_stack_ops(repo, [first, second], base_sha=base_sha) == {
        str(first): {TARGET: "upsert"},
        str(second): {TARGET: "upsert"},
    }


def test_a_round_editing_a_file_an_earlier_round_created_verifies(repo: Path, tmp_path: Path):
    """The preimage of a later patch can be a file that does not exist at the
    stack's base at all, so checking every patch against the base's inventory
    refused a legitimate stack."""
    base_sha = _git_head_sha(repo)
    created = "srt/made.py"
    first = _patch(tmp_path, "c1.patch", f"--- /dev/null\n+++ b/{created}\n@@ -0,0 +1 @@\n+one\n")
    second = _patch(tmp_path, "c2.patch", f"--- a/{created}\n+++ b/{created}\n@@ -1 +1 @@\n-one\n+two\n")
    for patch, msg in ((first, "create"), (second, "edit")):
        _git(repo, "apply", str(patch))
        _commit_all(repo, msg)
    assert replayed_stack_ops(repo, [first, second], base_sha=base_sha) == {
        str(first): {created: "upsert"},
        str(second): {created: "upsert"},
    }


def test_a_deletion_whose_twin_also_exists_at_base_resolves_to_the_right_path(repo: Path, tmp_path: Path):
    """Both candidate strip levels named a path that existed at base, so a
    base-inventory check could not disambiguate and the wrong path was
    recorded as deleted."""
    (repo / "srt" / "gone").write_text("inner\n", encoding="utf-8")
    (repo / "gone").write_text("outer\n", encoding="utf-8")
    _commit_all(repo, "two candidates")
    base_sha = _git_head_sha(repo)
    patch = _patch(tmp_path, "d.patch", "--- srt/gone\n+++ /dev/null\n@@ -1 +0,0 @@\n-inner\n")
    _git(repo, "apply", "-p0", str(patch))
    _commit_all(repo, "deleted the inner one")
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) == {str(patch): {"srt/gone": "delete"}}


def test_a_mixed_diff_declares_its_metadata_only_block_too(repo: Path, tmp_path: Path):
    """A pure rename, a mode-only change and a binary block carry no
    ``---``/``+++`` pair, so a header parse declared a non-empty but PARTIAL
    map and the accepted stack omitted exactly the same files -- nothing
    downstream could see the omission.

    The inventory is now git's own, so the mode-only block is declared rather
    than missed, and the patch is certified on ALL of its targets instead of
    being refused for a gap in the reader."""
    (repo / "srt" / "exec.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    _commit_all(repo, "add the script whose mode the patch changes")
    base_sha = _git_head_sha(repo)
    mixed = _patch(
        tmp_path,
        "mixed.patch",
        f"diff --git a/{TARGET} b/{TARGET}\n"
        f"--- a/{TARGET}\n+++ b/{TARGET}\n@@ -1 +1 @@\n-{BASE_TEXT}+{PATCHED_TEXT}"
        "diff --git a/srt/exec.sh b/srt/exec.sh\nold mode 100644\nnew mode 100755\n",
    )
    _git(repo, "apply", str(mixed))
    _commit_all(repo, "mixed applied")
    # The header parse still sees only the text half...
    assert patch_declared_ops(repo, [mixed]) == {TARGET: "upsert"}
    # ...while git reports both entries the apply actually touched.
    assert replayed_stack_ops(repo, [mixed], base_sha=base_sha) == {
        str(mixed): {TARGET: "upsert", "srt/exec.sh": "upsert"}
    }


def test_a_p0_deletion_is_not_recorded_against_a_path_that_never_existed(repo: Path, tmp_path: Path):
    """The strip level was guessed from paths that currently exist, and a
    deletion has erased exactly that evidence -- so a ``-p0`` deletion resolved
    at ``-p1`` and the capture emitted a complete tombstone for a path the tree
    never held, with both declaration maps agreeing with the wrong snapshot."""
    (repo / "srt" / "gone.py").write_text("gone\n", encoding="utf-8")
    _commit_all(repo, "add the file the patch deletes")
    base_sha = _git_head_sha(repo)
    patch = _patch(tmp_path, "del.patch", "--- srt/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-gone\n")
    _git(repo, "apply", "-p0", str(patch))
    _commit_all(repo, "deleted")

    # The header-only reader falls back to -p1 and names a path that was never
    # in the tree.
    assert patch_declared_ops(repo, [patch]) == {"gone.py": "delete"}
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) == {str(patch): {"srt/gone.py": "delete"}}


def test_a_patch_that_cannot_be_verified_leaves_the_recipe_refused(repo: Path, tmp_path: Path):
    """End to end: an unapplied patch reaches the decision as an undeclared
    step, not as a satisfied one."""
    from hyperloom.inference_optimizer.breakdown.collectors.sessions import collect_enablement

    base_sha = _git_head_sha(repo)
    patch = _patch(tmp_path, "never.patch", f"--- a/{TARGET}\n+++ b/{TARGET}\n@@ -1 +1 @@\n-{BASE_TEXT}+{PATCHED_TEXT}")
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    state = SimpleNamespace(enablement=EnablementRound(framework_root=str(repo)))
    ctx = SimpleNamespace(_ip_base_sha_by_root={str(repo): base_sha}, _ip_shared_state=state)
    out = executor._enablement_keep_records(
        ctx,
        params={},
        specialist_task_id=PROBE_TASK,
        framework_root=repo,
        applied=[patch],
        applied_artifacts=[],
        done_payload={"patch_roots": {str(patch): str(repo)}},
        provision_result=None,
        bench_result={},
    )
    assert out["enablement_patch_targets"] == {}
    state.enablement.kept_patches = [str(patch)]
    state.enablement.patch_roots = {str(patch): str(repo)}
    for key, value in out.items():
        setattr(state.enablement, key.removeprefix("enablement_"), value)
    section = collect_enablement(executor.session_dir, {"enablement": asdict(state.enablement)}, [])
    codes = [r["code"] for r in section["replay_sufficiency"]["reasons"]]
    assert section["replay_sufficiency"]["status"] == "insufficient"
    assert "patch_targets_unknown" in codes


def test_an_advanced_round_binds_its_patch_to_the_tree_it_applied_to(repo: Path, tmp_path: Path):
    """A stack whose rounds used different trees must keep each binding.

    An ADVANCED round never reaches the KEEP capture that writes the durable
    ``patch_roots``, so without this its patch is re-bound to the FINAL round's
    framework root. Since the capture now PROVES a patch against the tree it is
    bound to, the mis-binding does not certify anything -- it refuses the whole
    recipe, which for a legitimate multi-root stack is a false refusal.
    """
    from hyperloom.orchestrator.enablement.lane import _rearm_on_advanced

    other = tmp_path / "aiter"
    state = SimpleNamespace(enablement=EnablementRound(), baseline_failure_streak=1, baseline_total_failures=0)
    _rearm_on_advanced(
        state,
        {
            "status": "advanced",
            "advanced": True,
            "patches_applied": ["/p/1.patch"],
            "enablement_patch_roots": {"/p/1.patch": str(other)},
        },
    )
    assert state.enablement.patch_roots == {"/p/1.patch": str(other)}

    # A later round on a different tree must not re-point it.
    _rearm_on_advanced(
        state,
        {
            "status": "advanced",
            "advanced": True,
            "patches_applied": ["/p/2.patch"],
            "enablement_patch_roots": {"/p/1.patch": str(repo), "/p/2.patch": str(repo)},
        },
    )
    assert state.enablement.patch_roots == {"/p/1.patch": str(other), "/p/2.patch": str(repo)}


def test_the_base_reading_is_saved_before_the_mutation_that_invalidates_it(repo: Path, tmp_path: Path):
    """The reading is only correct BEFORE the mutation, and the mutation is the
    next thing that happens. Left for the rearm, a round that commits its patch
    and then dies resumes with the entry absent and HEAD already moved -- the
    exact state this map exists to prevent."""
    from hyperloom.orchestrator.actions.executors.integrate_patch import _note_pre_mutation_head

    saved: list[Path] = []
    state = SimpleNamespace(
        enablement=EnablementRound(),
        save=lambda session_dir, *a, **k: saved.append(Path(session_dir)),
    )
    session = tmp_path / "session"
    _note_pre_mutation_head(
        SimpleNamespace(_ip_shared_state=state), repo, enablement=True, session_dir=session
    )
    assert state.enablement.base_sha_by_root == {str(repo): _git_head_sha(repo)}
    assert saved == [session], "the reading must reach disk before the round mutates the tree"


def test_a_failing_save_does_not_stop_the_round(repo: Path, tmp_path: Path):
    """A record that cannot be written is still on the in-memory state, and the
    rearm saves again; a failed write must not cost the round."""
    from hyperloom.orchestrator.actions.executors.integrate_patch import _note_pre_mutation_head

    def _boom(*_a, **_k):
        raise OSError("disk full")

    state = SimpleNamespace(enablement=EnablementRound(), save=_boom)
    _note_pre_mutation_head(
        SimpleNamespace(_ip_shared_state=state), repo, enablement=True, session_dir=tmp_path
    )
    assert state.enablement.base_sha_by_root == {str(repo): _git_head_sha(repo)}
def test_a_hunk_body_cannot_pass_itself_off_as_a_file_header(repo: Path, tmp_path: Path):
    """A removed line beginning ``-- `` followed by an added line beginning
    ``++ `` reads as another file header to a text parse -- enough to make a
    block count agree while a mode-only block goes undeclared. The inventory is
    git's, so the real entries are the ones reported."""
    (repo / "srt" / "f").write_text("-- a/fake\n", encoding="utf-8")
    (repo / "srt" / "fake").write_text("untouched\n", encoding="utf-8")
    (repo / "srt" / "script").write_text("#!/bin/sh\n", encoding="utf-8")
    _commit_all(repo, "decoy base")
    base_sha = _git_head_sha(repo)
    (repo / "srt" / "f").write_text("++ b/fake\n", encoding="utf-8")
    (repo / "srt" / "script").chmod(0o755)
    _commit_all(repo, "content that looks like a header, plus a mode change")
    patch = _patch(tmp_path, "decoy.patch", _git(repo, "diff", "HEAD~1", "HEAD") + "\n")

    ops = replayed_stack_ops(repo, [patch], base_sha=base_sha)
    assert ops is not None
    declared = ops[str(patch)]
    # The decoy is NOT declared; the mode-only block IS.
    assert "srt/fake" not in declared
    assert declared == {"srt/f": "upsert", "srt/script": "upsert"}


def test_a_stack_touching_a_symlink_is_refused(repo: Path, tmp_path: Path):
    """``shutil.copy2`` follows a link and writes a REGULAR FILE, so a recipe
    certified over one restores the wrong kind of entry. Until the snapshot can
    represent a link, refusing is the fail-closed answer."""
    base_sha = _git_head_sha(repo)
    _git(repo, "-c", "core.symlinks=true", "--version")
    (repo / "srt" / "link").symlink_to("module.py")
    _commit_all(repo, "add a symlink")
    patch = _patch(tmp_path, "link.patch", _git(repo, "diff", "HEAD~1", "HEAD") + "\n")
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) is None


def test_a_deletion_is_not_satisfied_by_a_populated_directory(repo: Path, tmp_path: Path):
    """Treating "neither is a regular file" as agreement let a directory stand
    in for a deleted path. A deletion has to be actual absence."""
    victim = repo / "srt" / "victim.py"
    victim.write_text("bye\n", encoding="utf-8")
    _commit_all(repo, "add the file the patch deletes")
    base_sha = _git_head_sha(repo)
    victim.unlink()
    _commit_all(repo, "deleted")
    patch = _patch(tmp_path, "del.patch", _git(repo, "diff", "HEAD~1", "HEAD") + "\n")
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) == {str(patch): {"srt/victim.py": "delete"}}

    # A directory left at the deleted path is not a reproduced deletion.
    victim.mkdir()
    (victim / "live.py").write_text("still here\n", encoding="utf-8")
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) is None


def test_export_attributes_cannot_move_the_base_the_replay_compares_against(repo: Path, tmp_path: Path):
    """``git archive`` honours ``export-subst``, so it can hand the replay bytes
    a checkout of that commit would never produce -- accepting an unrecorded
    substitution as if the patch had made it. A real checkout cannot."""
    (repo / ".gitattributes").write_text("srt/tpl.py export-subst\n", encoding="utf-8")
    (repo / "srt" / "tpl.py").write_text("SHA = $Format:%H$\nvalue = 1\n", encoding="utf-8")
    _commit_all(repo, "a template tracked with export-subst")
    base_sha = _git_head_sha(repo)
    (repo / "srt" / "tpl.py").write_text("SHA = $Format:%H$\nvalue = 2\n", encoding="utf-8")
    _commit_all(repo, "the accepted change, away from the placeholder")
    patch = _patch(tmp_path, "tpl.patch", _git(repo, "diff", "HEAD~1", "HEAD") + "\n")
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) == {str(patch): {"srt/tpl.py": "upsert"}}

    # An unrecorded substitution no patch made must NOT verify.
    (repo / "srt" / "tpl.py").write_text(f"SHA = {base_sha}\nvalue = 2\n", encoding="utf-8")
    _commit_all(repo, "an unrecorded substitution")
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) is None

def test_a_patch_on_a_root_with_no_base_commit_declares_nothing(tmp_path: Path):
    """A tree with no identity has no preimage to replay from.

    Three successive attempts to certify such a root from the patch text alone
    each left a different hole -- a hunk body posing as a file header, an
    ambiguous strip level resolved to the wrong path, git's abbreviated
    ``dir/{old => new}`` rename summary inventing a source, and a declared
    symlink certified into a regular file. A patch step bound to a tree that
    names no base cannot be certified, so it declares nothing and the decision
    refuses it.
    """
    plain = tmp_path / "site-packages" / "pkg"
    plain.mkdir(parents=True)
    (plain / "mod.py").write_text(PATCHED_TEXT, encoding="utf-8")
    patch = _patch(tmp_path, "ng.patch", f"--- a/mod.py\n+++ b/mod.py\n@@ -1 +1 @@\n-{BASE_TEXT}+{PATCHED_TEXT}")

    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    state = SimpleNamespace(enablement=EnablementRound(framework_root=str(plain)))
    out = executor._enablement_keep_records(
        SimpleNamespace(_ip_base_sha_by_root={}, _ip_shared_state=state),
        params={},
        specialist_task_id=PROBE_TASK,
        framework_root=plain,
        applied=[patch],
        applied_artifacts=[],
        done_payload={"patch_roots": {str(patch): str(plain)}},
        provision_result=None,
        bench_result={},
    )
    assert out["enablement_patch_targets"] == {}


def test_an_artifact_on_a_root_with_no_base_commit_is_unaffected(tmp_path: Path):
    """Narrowing what a PATCH step may claim does not touch artifacts, which
    are judged against their own captured payload."""
    plain = tmp_path / "plain_root"
    (plain / "lib").mkdir(parents=True)
    (plain / "lib" / "a.so").write_bytes(b"\x00artifact")
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    state = SimpleNamespace(enablement=EnablementRound())
    out = executor._enablement_keep_records(
        SimpleNamespace(_ip_base_sha_by_root={}, _ip_shared_state=state),
        params={},
        specialist_task_id=PROBE_TASK,
        framework_root=plain,
        applied=[],
        applied_artifacts=[{"target": str(plain / "lib/a.so"), "rel_target": "lib/a.so", "root": str(plain)}],
        done_payload={},
        provision_result=None,
        bench_result={},
    )
    record = next(r for r in out["enablement_roots"] if r["path"] == str(plain))
    assert record["contributions"] == ["artifact_install"]
    root_id = record["id"]
    assert out["enablement_accepted_stack_targets"][root_id] == {"lib/a.so": "upsert"}


def test_a_round_deleting_what_a_later_round_recreates_replays(repo: Path, tmp_path: Path):
    """The comparison is over the FOLDED end state. Walking each patch's own
    operation against the final tree required the deleted file to be absent,
    and refused a stack a later round legitimately recreated."""
    victim = repo / "srt" / "v.py"
    victim.write_text("one\n", encoding="utf-8")
    _commit_all(repo, "add the file")
    base_sha = _git_head_sha(repo)
    victim.unlink()
    _commit_all(repo, "round one deletes it")
    first = _patch(tmp_path, "d1.patch", _git(repo, "diff", "HEAD~1", "HEAD") + "\n")
    victim.write_text("two\n", encoding="utf-8")
    _commit_all(repo, "round two recreates it")
    second = _patch(tmp_path, "d2.patch", _git(repo, "diff", "HEAD~1", "HEAD") + "\n")

    assert replayed_stack_ops(repo, [first, second], base_sha=base_sha) == {
        str(first): {"srt/v.py": "delete"},
        str(second): {"srt/v.py": "upsert"},
    }


def test_a_patch_created_file_the_base_gitignores_is_still_inventoried(repo: Path, tmp_path: Path):
    """``git add -A`` skips a newly created file the base's .gitignore matches,
    so an unforced stage drops it from the replay commit -- moving the
    completeness gap from the diff parser to the index."""
    (repo / ".gitignore").write_text("srt/generated.out\n", encoding="utf-8")
    _commit_all(repo, "ignore the generated file")
    base_sha = _git_head_sha(repo)
    patch = _patch(
        tmp_path,
        "gen.patch",
        f"--- a/{TARGET}\n+++ b/{TARGET}\n@@ -1 +1 @@\n-{BASE_TEXT}+{PATCHED_TEXT}"
        "--- /dev/null\n+++ b/srt/generated.out\n@@ -0,0 +1 @@\n+generated\n",
    )
    _git(repo, "apply", str(patch))
    _git(repo, "add", "-A", "--force")
    _git(repo, "commit", "-qm", "applied incl. the ignored file")

    ops = replayed_stack_ops(repo, [patch], base_sha=base_sha)
    assert ops == {str(patch): {TARGET: "upsert", "srt/generated.out": "upsert"}}


def test_an_apply_root_below_the_repository_top_level_replays(repo: Path, tmp_path: Path):
    """``git clone`` of a directory inside a worktree clones nothing, and the
    executor's own resolver accepts such a root."""
    inner = repo / "srt"
    base_sha = _git_head_sha(repo)
    patch = _patch(tmp_path, "inner.patch", f"--- a/module.py\n+++ b/module.py\n@@ -1 +1 @@\n-{BASE_TEXT}+{PATCHED_TEXT}")
    subprocess.run(["git", "-C", str(inner), "apply", str(patch)], check=True)
    _commit_all(repo, "applied under the subdirectory root")

    assert replayed_stack_ops(inner, [patch], base_sha=base_sha) == {str(patch): {"module.py": "upsert"}}


def test_the_execute_bit_compared_is_the_one_git_records(repo: Path, tmp_path: Path):
    """``mode & 0o111`` asks whether ANY execute bit is set, which accepts 0645
    -- a mode git classifies as non-executable and whose owner cannot run it.

    Restored after being deleted by accident while the non-git tests were being
    replaced. Its guarantee has nothing to do with that contract change, and
    the surviving content-plus-mode test compares 0755 against 0644, which the
    old incorrect ``0o111`` comparison would also have passed.
    """
    script = repo / "srt" / "run.sh"
    script.write_text("#!/bin/sh\necho one\n", encoding="utf-8")
    _commit_all(repo, "add the script")
    base_sha = _git_head_sha(repo)
    script.write_text("#!/bin/sh\necho two\n", encoding="utf-8")
    script.chmod(0o755)
    _commit_all(repo, "content and mode")
    patch = _patch(tmp_path, "x.patch", _git(repo, "diff", "HEAD~1", "HEAD") + "\n")
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) == {str(patch): {"srt/run.sh": "upsert"}}

    # 0645 has a group execute bit but not the owner one git records as 100755.
    script.chmod(0o645)
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) is None


def test_a_change_staging_would_normalize_away_is_still_inventoried(repo: Path, tmp_path: Path):
    """``git add`` applies eol normalization and clean filters, so a patch whose
    whole effect is a line ending was staged back to the byte-identical blob it
    started as: the replay commit showed no change, the inventory omitted the
    file, and nothing compared or shipped it. Forcing ignored files closed one
    way staging discards evidence; this is the other."""
    (repo / ".gitattributes").write_text("srt/crlf.txt text\n", encoding="utf-8")
    (repo / "srt" / "crlf.txt").write_bytes(b"old\n")
    (repo / "srt" / "other.txt").write_bytes(b"old\n")
    _commit_all(repo, "a file tracked as text")
    base_sha = _git_head_sha(repo)
    patch = _patch(
        tmp_path,
        "crlf.patch",
        "--- a/srt/crlf.txt\n+++ b/srt/crlf.txt\n@@ -1 +1 @@\n-old\n+old\r\n"
        "--- a/srt/other.txt\n+++ b/srt/other.txt\n@@ -1 +1 @@\n-old\n+new\n",
    )
    _git(repo, "apply", str(patch))
    _git(repo, "-c", "core.autocrlf=false", "add", "-A", "--force")
    _git(repo, "commit", "-qm", "applied")

    ops = replayed_stack_ops(repo, [patch], base_sha=base_sha)
    assert ops is not None
    # Both entries, not just the one normalization left visible.
    assert ops[str(patch)] == {"srt/crlf.txt": "upsert", "srt/other.txt": "upsert"}

    # And the negative control: with the file in the comparison set, content
    # no patch produced is refused.
    (repo / "srt" / "crlf.txt").write_bytes(b"not produced by any patch\n")
    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) is None


def test_an_apply_root_absent_at_base_is_created_for_the_replay(repo: Path, tmp_path: Path):
    """Git records no empty directory, so the apply root may not exist at base
    -- an untracked directory the first accepted patch populates."""
    base_sha = _git_head_sha(repo)
    fresh = repo / "srt" / "fresh"
    fresh.mkdir()
    patch = _patch(tmp_path, "fresh.patch", "--- /dev/null\n+++ b/made.py\n@@ -0,0 +1 @@\n+x = 1\n")
    subprocess.run(["git", "-C", str(fresh), "apply", str(patch)], check=True)
    _commit_all(repo, "populate a directory the base does not have")

    assert replayed_stack_ops(fresh, [patch], base_sha=base_sha) == {str(patch): {"made.py": "upsert"}}


def test_a_root_a_deletion_empties_is_recreated_for_the_next_round(repo: Path, tmp_path: Path):
    """A deletion that takes the last file under the apply root removes the
    directory, and the next step's git invocations would run with a missing
    working directory."""
    root = repo / "srt" / "solo"
    root.mkdir()
    (root / "only.py").write_text("one\n", encoding="utf-8")
    _commit_all(repo, "a root with a single file")
    base_sha = _git_head_sha(repo)

    first = _patch(tmp_path, "s1.patch", "--- a/only.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-one\n")
    subprocess.run(["git", "-C", str(root), "apply", str(first)], check=True)
    _commit_all(repo, "round one empties the root")
    assert not root.exists(), "git should have removed the now-empty directory"

    root.mkdir()
    second = _patch(tmp_path, "s2.patch", "--- /dev/null\n+++ b/only.py\n@@ -0,0 +1 @@\n+two\n")
    subprocess.run(["git", "-C", str(root), "apply", str(second)], check=True)
    _commit_all(repo, "round two recreates it")

    assert replayed_stack_ops(root, [first, second], base_sha=base_sha) == {
        str(first): {"only.py": "delete"},
        str(second): {"only.py": "upsert"},
    }


def test_a_tree_whose_attributes_transform_content_is_refused(repo: Path, tmp_path: Path):
    """``working-tree-encoding`` converts independently of ``text``, so it is not
    implied by ``-text``. Left enabled during staging it converted a re-encoded
    working file back to the blob it came from, and the change vanished from
    the inventory silently -- with the file outside the comparison set, content
    no patch produced verified too.

    The replay now disables every transformation at BOTH ends, so a tree whose
    attributes rewrite working-tree content no longer matches its own replay
    and the stack is refused. That is a real narrowing, in the same explicit
    category as symlinks and roots with no base: the capture would otherwise
    have to claim that bytes produced by an attribute were produced by a patch.
    What it is not is silent -- the omission is now a refusal.
    """
    rel = "srt/enc.txt"
    (repo / rel).write_bytes("old\n".encode("utf-16-le"))
    (repo / ".gitattributes").write_text(f"{rel} working-tree-encoding=UTF-16LE\n", encoding="utf-8")
    _commit_all(repo, "a file with a declared working-tree encoding")
    base_sha = _git_head_sha(repo)

    (repo / rel).write_bytes("new\n".encode("utf-16-le"))
    _commit_all(repo, "re-encode its content")
    patch = _patch(tmp_path, "enc.patch", _git(repo, "diff", "HEAD~1", "HEAD") + "\n")

    assert replayed_stack_ops(repo, [patch], base_sha=base_sha) is None


def test_a_smudge_filter_cannot_attribute_its_output_to_the_first_patch(
    repo: Path, tmp_path: Path, monkeypatch
):
    """A ``smudge`` filter rewrites the file on the way OUT of the object
    database. Disabling transformations only AFTER checkout cannot undo that,
    and the first replay commit then records the smudged bytes as an effect of
    its own patch -- content the recorded base and patch do not produce for a
    consumer without that filter configured.

    The filter is supplied through ``GIT_CONFIG_GLOBAL`` because that is where
    the hazard lives: host configuration the clone inherits and an outside
    consumer does not have. A repository-level filter would not be inherited by
    the clone at all, so a test using one proves nothing.
    """
    filtered, plain = "srt/filtered.txt", "srt/plain.txt"
    (repo / filtered).write_text("base\n", encoding="utf-8")
    (repo / plain).write_text("old\n", encoding="utf-8")
    (repo / ".gitattributes").write_text(f"{filtered} filter=replaytest\n", encoding="utf-8")
    _commit_all(repo, "a file declaring a smudge filter")
    base_sha = _git_head_sha(repo)

    (repo / plain).write_text("new\n", encoding="utf-8")
    _commit_all(repo, "the accepted patch touches only the plain file")
    patch = _patch(tmp_path, "sm.patch", _git(repo, "diff", "HEAD~1", "HEAD") + "\n")

    global_config = tmp_path / "gitconfig"
    global_config.write_text(
        "[filter \"replaytest\"]\n\tsmudge = sed s/base/unrecorded/\n", encoding="utf-8"
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))

    ops = replayed_stack_ops(repo, [patch], base_sha=base_sha)
    assert ops is not None, "the stack itself is valid and must still verify"
    # The smudged file is not an effect of the patch and must not be claimed.
    assert ops[str(patch)] == {plain: "upsert"}
