# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.knowledge.remote_recipe import (
    HyperloomRemoteKB,
    KBStoreError,
    RemoteRecipeClient,
    RemoteRecipeConfigurationError,
    build_remote_knowledge,
    convert_v1_recipe_to_knowledge,
    envelope_to_v1_recipe,
    has_new_keep,
    read_remote_champion,
    read_remote_recipe,
    write_final_remote_recipe,
)
from hyperloom.orchestrator.knowledge.remote_recipe._vendor import kb_store_client
from hyperloom.orchestrator.knowledge.remote_recipe.client import (
    _champion,
    _validate_download_listing,
    _verify_downloaded_files,
)
from hyperloom.orchestrator.knowledge.remote_recipe.models import (
    MAX_FILE_BYTES,
    MAX_PATH_BYTES,
    Artifact,
    KnowledgeBundle,
    RemoteRecipeValidationError,
)
from hyperloom.orchestrator.knowledge.recipe_kb_t0 import run_t0_anchor
from hyperloom.orchestrator.loop.writeback import WritebackCollaborator

_DOWNLOAD_BYTES = b"verified artifact"
_DOWNLOAD_SHA256 = hashlib.sha256(_DOWNLOAD_BYTES).hexdigest()


def _state(tmp_path: Path) -> SimpleNamespace:
    explore_patch = tmp_path / "explore.patch"
    explore_patch.write_text("explore", encoding="utf-8")
    framework_patch = tmp_path / "framework.patch"
    framework_patch.write_text("framework", encoding="utf-8")
    tuned = tmp_path / "tuned.csv"
    tuned.write_text("M,N,K\n1,2,3\n", encoding="utf-8")
    fusion = tmp_path / "fusion.patch"
    fusion.write_text("fusion", encoding="utf-8")
    rewrite = tmp_path / "rewrite.cu"
    rewrite.write_text("// optimized", encoding="utf-8")
    source = tmp_path / "source.cu"
    source.write_text("// source", encoding="utf-8")
    return SimpleNamespace(
        session_id="session-1",
        recipe_kb_session_id="session-1",
        baseline_tput=100.0,
        current_best={
            "tput": 130.0,
            "extra_server_args": "--page-size 32",
            "extra_envs": {"FINAL": "1"},
        },
        cumulative_gain_validated=30.0,
        cumulative_gain=30.0,
        optimization_stack=[
            {
                "action": "explore",
                "source_phase": "EXPLORE",
                "extra_server_args": "--page-size 32",
                "extra_envs": {"EXPLORE": "1"},
                "patch_path": str(explore_patch),
                "tput": 110.0,
            },
            {
                "action": "integrate_patch",
                "source_phase": "FRAMEWORK_AGENT",
                "extra_server_args": "--page-size 32 --enable-foo",
                "extra_envs": {"FRAMEWORK": "1"},
                "patch_path": str(framework_patch),
                "tput": 120.0,
            },
            {
                "action": "gemm_tuning",
                "source_phase": "KERNEL_AGENT",
                "tuned_file": str(tuned),
                "tput": 130.0,
            },
            {
                "action": "fusion",
                "source_phase": "KERNEL_AGENT",
                "patch_path": str(fusion),
                "target_file": str(source),
                "tput": 130.0,
            },
            {
                "action": "integrate",
                "source_phase": "KERNEL_AGENT",
                "integration_id": "integration-rmsnorm-1",
                "task_group_key": "group-rmsnorm",
                "kernel_id": "rmsnorm",
                "patch_path": str(rewrite),
                "target_file": str(source),
                "gain_pct": 4.25,
                "tput": 130.0,
            },
        ],
        gemm_tuning_attempts=[
            {"status": "failed", "decision": "REVERT", "error": "must not be persisted"}
        ],
        last_gemm_tuning={"status": "ok", "decision": "KEEP", "tuned_file": str(tuned)},
        last_fusion={
            "status": "ok",
            "decision": "KEEP",
            "patch": str(fusion),
            "source_file": str(source),
            "kernel_speedup": 1.2,
        },
        last_fusion_integrate={"decision": "KEEP", "new_tput": 130.0, "patch_path": str(fusion)},
        kernel_opt_task_attempts={
            "rmsnorm": {
                "last_decision": "KEEP",
                "task_group_key": "group-rmsnorm",
                "current_kernel_id": "rmsnorm",
                "kernel_name": "rmsnorm",
                "last_micro_speedup": 1.5,
                "last_artifact_path": str(rewrite),
                "last_source_file": str(source),
            }
        },
        kernel_opt_attempts={},
        last_action_failures=[{"action": "explore", "reason": "OOM"}],
        gaps=[{"description": "attention remains bound"}],
        warm_start_lessons=[{"statement": "use page size 32"}],
        warm_start_pitfalls=[{"description": "page size 1 regresses"}],
    )


def test_build_remote_knowledge_partitions_origins_and_files(tmp_path: Path) -> None:
    bundle = build_remote_knowledge(_state(tmp_path), tmp_path / "files")

    assert bundle.knowledge["optimized_throughput"] == 130.0
    assert bundle.knowledge["validated_e2e_gain"] == 30.0
    value = bundle.knowledge["value"]
    assert value["explore"]["extra_envs"] == {"EXPLORE": "1"}
    assert value["framework"]["extra_envs"] == {"FRAMEWORK": "1"}
    assert "config" not in value["explore"]
    assert "phase" not in value["framework"]
    assert value["explore"]["patches"][0].startswith("explore/patches/")
    assert value["framework"]["patches"][0].startswith("framework/patches/")
    assert not any(item.path.startswith("files/") for item in bundle.artifacts)
    assert isinstance(value["kernel"]["gemm"], dict)
    assert len(value["kernel"]["gemm"]["optimizations"]) == 1
    assert "must not be persisted" not in json.dumps(value["kernel"]["gemm"])
    assert isinstance(value["kernel"]["fusion"], dict)
    assert len(value["kernel"]["fusion"]["items"]) == 1
    assert isinstance(value["kernel"]["rewrite"], dict)
    rewrite = value["kernel"]["rewrite"]["items"][0]
    assert rewrite["id"]
    assert rewrite["id"] == "integration-rmsnorm-1"
    assert rewrite["kernel_name"] == "rmsnorm"
    assert rewrite["speedup"] == 1.5
    assert rewrite["e2e_gain_pct"] == 4.25
    assert rewrite["optimized_throughput"] == 130.0
    assert rewrite["experience_document"].endswith(".md")
    assert rewrite["patch"].startswith("kernel/rewrite/patches/")
    assert rewrite["source_files"][0].startswith("kernel/rewrite/source/")
    assert (tmp_path / "files" / rewrite["experience_document"]).is_file()
    serialized = json.dumps(bundle.knowledge)
    assert "object_id" not in serialized
    assert "bucket" not in serialized
    assert '"files": [' not in serialized


def test_empty_phase_sections_do_not_copy_cumulative_current_best(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.optimization_stack = [
        row for row in state.optimization_stack if row.get("source_phase") == "EXPLORE"
    ]
    bundle = build_remote_knowledge(state, tmp_path / "files-empty-framework")
    value = bundle.knowledge["value"]
    assert value["explore"]["extra_server_args"] == "--page-size 32"
    assert value["framework"]["extra_server_args"] == ""
    assert value["framework"]["extra_envs"] == {}


def test_geak_e2e_winner_serializes_config_and_artifacts(tmp_path: Path) -> None:
    state = _state(tmp_path)
    overlay = tmp_path / "geak-final" / "overlay"
    overlay.mkdir(parents=True)
    (overlay / "optimized_kernel.py").write_text("# optimized\n", encoding="utf-8")
    report = tmp_path / "geak-report.json"
    report.write_text('{"status":"ok"}\n', encoding="utf-8")
    launch = tmp_path / "launch.sh"
    launch.write_text("#!/bin/sh\n", encoding="utf-8")
    bench = tmp_path / "bench.sh"
    bench.write_text("#!/bin/sh\n", encoding="utf-8")
    state.optimization_stack = [
        {
            "action": "geak_e2e",
            "source_phase": "KERNEL_AGENT",
            "variant_name": "geak_e2e",
            "tput": 135.0,
            "candidate_extra_server_args": "--geak-fast",
            "extra_envs": {"GEAK_OPT": "1"},
            "final_overlay": str(overlay.parent),
            "accepted_kernels": ["attention"],
            "accepted_heads": ["decode"],
            "report_path": str(report),
            "ts": "2026-08-08T00:00:00+00:00",
        }
    ]
    state.current_best = {
        "action": "geak_e2e",
        "tput": 135.0,
        "geak_launch_script": str(launch),
        "geak_bench_script": str(bench),
        "final_overlay": str(overlay.parent),
        "geak_alignment": {"hot_speedup": 1.35},
    }
    state.geak_result = {"status": "ok", "throughput_speedup": 1.34}

    bundle = build_remote_knowledge(state, tmp_path / "files-geak")
    geak = bundle.knowledge["value"]["kernel"]["geak"]["items"][0]
    assert geak["optimized_throughput"] == 135.0
    assert geak["extra_envs"] == {"GEAK_OPT": "1"}
    assert geak["report"].startswith("kernel/geak/reports/")
    assert geak["launch_script"].startswith("kernel/geak/launch/")
    assert geak["bench_script"].startswith("kernel/geak/launch/")
    assert geak["overlay_files"] == [
        "kernel/geak/overlay/optimized_kernel.py"
    ]
    for rel in (
        geak["report"],
        geak["launch_script"],
        geak["bench_script"],
        *geak["overlay_files"],
    ):
        assert (tmp_path / "files-geak" / rel).is_file()


def test_micro_keep_without_integrate_stack_is_not_written(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.optimization_stack = [
        row for row in state.optimization_stack if row.get("action") != "integrate"
    ]
    bundle = build_remote_knowledge(state, tmp_path / "files-no-integrate")
    assert bundle.knowledge["value"]["kernel"]["rewrite"]["items"] == []


def test_micro_keep_with_e2e_revert_but_no_integrate_stack_is_not_written(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    state.optimization_stack = [
        row for row in state.optimization_stack if row.get("action") != "integrate"
    ]
    state.kernel_integrate_attempts = {
        "rmsnorm": {
            "kernel_id": "rmsnorm",
            "integration_id": "integration-rmsnorm-1",
            "last_decision": "REVERT",
            "best_gain_pct": -2.0,
        }
    }
    bundle = build_remote_knowledge(state, tmp_path / "files-revert")
    assert bundle.knowledge["value"]["kernel"]["rewrite"]["items"] == []


def test_integrate_stack_is_authoritative_for_rewrite_files(tmp_path: Path) -> None:
    state = _state(tmp_path)
    unintegrated_patch = tmp_path / "micro-only.cu"
    unintegrated_patch.write_text("// micro only", encoding="utf-8")
    unintegrated_source = tmp_path / "micro-source.cu"
    unintegrated_source.write_text("// micro source", encoding="utf-8")
    attempt = state.kernel_opt_task_attempts["rmsnorm"]
    attempt["last_artifact_path"] = str(unintegrated_patch)
    attempt["last_source_file"] = str(unintegrated_source)
    bundle = build_remote_knowledge(state, tmp_path / "files-integrated")
    rewrite = bundle.knowledge["value"]["kernel"]["rewrite"]["items"][0]
    assert rewrite["patch"].endswith("/rewrite.cu")
    assert rewrite["source_files"][0].endswith("/source.cu")
    assert "micro-only.cu" not in json.dumps(rewrite)
    assert rewrite["speedup"] == 1.5


def test_integrated_rewrite_missing_artifact_fails_closed(tmp_path: Path) -> None:
    state = _state(tmp_path)
    integrate = next(
        row for row in state.optimization_stack if row.get("action") == "integrate"
    )
    integrate["patch_path"] = str(tmp_path / "missing-rewrite.cu")
    state.kernel_opt_task_attempts["rmsnorm"]["last_artifact_path"] = ""
    with pytest.raises(RemoteRecipeValidationError, match="kernel/rewrite"):
        build_remote_knowledge(state, tmp_path / "files-missing-rewrite")


def test_accepted_gemm_missing_tuned_file_fails_closed(tmp_path: Path) -> None:
    state = _state(tmp_path)
    gemm = next(
        row for row in state.optimization_stack if row.get("action") == "gemm_tuning"
    )
    gemm["tuned_file"] = str(tmp_path / "missing-tuned.csv")
    state.last_gemm_tuning["tuned_file"] = gemm["tuned_file"]
    with pytest.raises(RemoteRecipeValidationError, match="kernel/gemm"):
        build_remote_knowledge(state, tmp_path / "files-missing-gemm")


def test_accepted_fusion_missing_patch_or_target_fails_closed(tmp_path: Path) -> None:
    state = _state(tmp_path)
    fusion = next(
        row for row in state.optimization_stack if row.get("action") == "fusion"
    )
    fusion["patch_path"] = str(tmp_path / "missing-fusion.patch")
    state.last_fusion["patch"] = fusion["patch_path"]
    with pytest.raises(RemoteRecipeValidationError, match="kernel/fusion"):
        build_remote_knowledge(state, tmp_path / "files-missing-fusion")


def test_bundle_rejects_path_mismatch_and_prefix(tmp_path: Path) -> None:
    source = tmp_path / "a"
    source.write_text("x", encoding="utf-8")
    bundle = KnowledgeBundle({"value": {}}, [Artifact("explore/a", source)])
    with pytest.raises(RemoteRecipeValidationError, match="absent from knowledge"):
        bundle.validate([])
    bad = KnowledgeBundle({"value": {}}, [Artifact("files/explore/a", source)])
    with pytest.raises(RemoteRecipeValidationError, match="files/ prefix"):
        bad.validate(["files/explore/a"])


def test_remote_client_internal_validation_error_paths(tmp_path: Path) -> None:
    assert _champion(None) == ("", 0.0, {})
    assert _champion({"champion": {"value": "not-a-number"}})[1] == 0.0
    assert _champion({"champion": {"optimized_throughput": 12.0}})[1] == 12.0
    with pytest.raises(RemoteRecipeValidationError, match="finite"):
        _champion({"champion": {"value": float("inf")}})

    with pytest.raises(RemoteRecipeValidationError, match="listing must be"):
        _validate_download_listing([])
    with pytest.raises(RemoteRecipeValidationError, match="files must be"):
        _validate_download_listing({"files": {"bad": 1}})
    with pytest.raises(RemoteRecipeValidationError, match="entry 0"):
        _validate_download_listing({"files": ["not-an-object"]})

    digest = hashlib.sha256(b"x").hexdigest()
    duplicate = {"path": "artifact", "size": 1, "sha256": digest}
    with pytest.raises(RemoteRecipeValidationError, match="duplicate"):
        _validate_download_listing({"files": [duplicate, duplicate]})

    not_a_directory = tmp_path / "not-a-directory"
    not_a_directory.write_text("x", encoding="utf-8")
    with pytest.raises(RemoteRecipeValidationError, match="not a directory"):
        _verify_downloaded_files(not_a_directory, {})

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "files-link"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(RemoteRecipeValidationError, match="symlink"):
        _verify_downloaded_files(symlink, {})

    class _ReadStore:
        def __init__(self, rollup, envelope=None) -> None:
            self.rollup = rollup
            self.envelope = envelope

        def get_rollup(self, _canonical_id):
            return self.rollup

        def get_session(self, _canonical_id, _session_id):
            return self.envelope

    identity = "inference:m:h:f:mt:a:v:p"
    assert RemoteRecipeClient(_ReadStore({})).read(identity, tmp_path / "miss") is None  # type: ignore[arg-type]
    assert (
        RemoteRecipeClient(
            _ReadStore({"champion": {"session_id": "missing", "value": 1.0}})
        ).read(identity, tmp_path / "missing-session")  # type: ignore[arg-type]
        is None
    )


def test_artifact_rejects_symlink_and_oversized_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("x", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(RemoteRecipeValidationError, match="symlink"):
        Artifact("explore/link", link).validate()
    oversized = tmp_path / "oversized"
    with oversized.open("wb") as handle:
        handle.truncate(MAX_FILE_BYTES + 1)
    with pytest.raises(RemoteRecipeValidationError, match="limit"):
        Artifact("explore/oversized", oversized).validate()


def test_path_limit_and_strict_json_validation(tmp_path: Path) -> None:
    source = tmp_path / "source-json"
    source.write_text("x", encoding="utf-8")
    with pytest.raises(RemoteRecipeValidationError, match="1024-byte"):
        Artifact("x" * (MAX_PATH_BYTES + 1), source).validate()
    with pytest.raises(RemoteRecipeValidationError, match="strict JSON"):
        KnowledgeBundle({"bad": float("nan")}).validate([])


def test_warm_replay_and_non_keep_actions_do_not_enable_write(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.optimization_stack = [
        {"action": "replay_warm_recipe"},
        {"action": "profile"},
        {"action": "roofline"},
        {"action": "conc_sweep"},
    ]
    state.kernel_opt_task_attempts = {}
    assert has_new_keep(state) is False


@pytest.mark.parametrize(
    ("url", "token", "match"),
    [
        ("", "", "KB_STORE_URL and KB_STORE_TOKEN"),
        ("https://kb.example", "", "KB_STORE_TOKEN"),
        ("", "token", "KB_STORE_URL"),
    ],
)
def test_facade_from_env_requires_complete_configuration(
    monkeypatch,
    url: str,
    token: str,
    match: str,
) -> None:
    monkeypatch.setenv("KB_STORE_URL", url)
    monkeypatch.setenv("KB_STORE_TOKEN", token)
    with pytest.raises(RemoteRecipeConfigurationError, match=match):
        HyperloomRemoteKB.from_env()


def test_facade_from_env_returns_configured_facade(monkeypatch) -> None:
    monkeypatch.setenv("KB_STORE_URL", "https://kb.example")
    monkeypatch.setenv("KB_STORE_TOKEN", "token")
    remote = HyperloomRemoteKB.from_env()
    assert isinstance(remote, HyperloomRemoteKB)
    assert isinstance(remote._client, RemoteRecipeClient)


@pytest.mark.parametrize(
    ("recipe_session_id", "state_session_id", "expected"),
    [
        ("recipe-session", "state-session", "recipe-session"),
        ("", "state-session", "state-session"),
        ("  ", "state-session", "state-session"),
    ],
)
def test_facade_delegates_read_and_write_with_session_fallback(
    tmp_path: Path,
    monkeypatch,
    recipe_session_id: str,
    state_session_id: str,
    expected: str,
) -> None:
    from hyperloom.orchestrator.knowledge import remote_recipe

    client = object()
    facade = HyperloomRemoteKB(client)  # type: ignore[arg-type]
    state = SimpleNamespace(
        recipe_kb_session_id=recipe_session_id,
        session_id=state_session_id,
    )
    read_result = {"canonical_id": "inference:m:h:f:mt:a:v:p"}
    write_result = SimpleNamespace(status="written", session_id=expected)
    calls: list[tuple] = []

    def _read(identity, destination, *, client):
        calls.append(("read", identity, destination, client))
        return read_result

    def _write(state_arg, identity, session_id, *, client):
        calls.append(("write", state_arg, identity, session_id, client))
        return write_result

    monkeypatch.setattr(remote_recipe, "read_remote_recipe", _read)
    monkeypatch.setattr(remote_recipe, "write_final_remote_recipe", _write)

    identity = "inference:m:h:f:mt:a:v:p"
    destination = tmp_path / "download"
    actual_read_result = facade.read(identity, destination)
    actual_write_result = facade.write(identity, state)
    assert actual_read_result is read_result
    assert actual_write_result is write_result
    assert calls == [
        ("read", identity, destination, client),
        ("write", state, identity, expected, client),
    ]


def test_facade_write_explicit_session_overrides_state(monkeypatch) -> None:
    from hyperloom.orchestrator.knowledge import remote_recipe

    facade = HyperloomRemoteKB(object())  # type: ignore[arg-type]
    state = SimpleNamespace(
        recipe_kb_session_id="recipe-session",
        session_id="state-session",
    )
    seen: list[str] = []
    expected_result = SimpleNamespace(status="written")

    def _write(_state, _identity, session_id, *, client):
        seen.append(session_id)
        return expected_result

    monkeypatch.setattr(remote_recipe, "write_final_remote_recipe", _write)
    actual_result = facade.write(
        "inference:m:h:f:mt:a:v:p",
        state,
        "explicit-session",
    )
    assert actual_result is expected_result
    assert seen == ["explicit-session"]


def test_degraded_kb_skips_remote_close_writer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _Journal:
        def finalize(self, **kwargs) -> None:
            pass

    coordinator = SimpleNamespace(
        shared_state=SimpleNamespace(current_best={"tput": 10.0}),
        session_dir=tmp_path,
        recipe_kb=None,
        knowledge_plane=SimpleNamespace(kb_disabled=True),
        _ensure_journal=lambda: _Journal(),
    )
    from hyperloom.orchestrator.knowledge import remote_recipe

    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.setattr(
        remote_recipe.HyperloomRemoteKB,
        "from_env",
        classmethod(
            lambda cls: (_ for _ in ()).throw(
                AssertionError("degraded CLOSE constructed HyperloomRemoteKB")
            )
        ),
    )

    WritebackCollaborator(coordinator).finalize_recipe_and_journal()


def test_local_close_ignores_ambient_kb_store(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _Journal:
        def finalize(self, **kwargs) -> None:
            pass

    coordinator = SimpleNamespace(
        shared_state=SimpleNamespace(current_best={}),
        session_dir=tmp_path,
        recipe_kb=None,
        knowledge_plane=None,
        _ensure_journal=lambda: _Journal(),
        _workload_canonical_id=lambda: "inference:m:h:f:mt:a:v:p",
    )
    calls: list[tuple] = []
    from hyperloom.orchestrator.knowledge import remote_recipe

    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "local")
    monkeypatch.setenv("KB_STORE_URL", "https://kb.example")
    monkeypatch.setenv("KB_STORE_TOKEN", "ambient-token")
    monkeypatch.setattr(
        remote_recipe.HyperloomRemoteKB,
        "from_env",
        classmethod(
            lambda cls: (_ for _ in ()).throw(
                AssertionError("local CLOSE constructed HyperloomRemoteKB")
            )
        ),
    )
    WritebackCollaborator(coordinator).finalize_recipe_and_journal()
    assert calls == []


def test_remote_close_writes_new_kb_once_and_skips_legacy_finalize(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _Journal:
        def finalize(self, **kwargs) -> None:
            pass

    class _LegacyRecipe:
        def get_authoritative_recipe(self, **kwargs):
            raise AssertionError("remote CLOSE read legacy RecipeKB")

        def put_recipe(self, **kwargs):
            raise AssertionError("remote CLOSE wrote legacy RecipeKB")

    coordinator = SimpleNamespace(
        shared_state=SimpleNamespace(
            current_best={"tput": 10.0},
        ),
        session_dir=tmp_path,
        recipe_kb=_LegacyRecipe(),
        knowledge_plane=None,
        _ensure_journal=lambda: _Journal(),
        _workload_canonical_id=lambda: "inference:m:h:f:mt:a:v:p",
    )
    calls: list[tuple] = []
    from hyperloom.orchestrator.knowledge import remote_recipe

    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.setenv("KB_STORE_URL", "https://kb.example")
    monkeypatch.setenv("KB_STORE_TOKEN", "token")

    class _Facade:
        def write(self, identity, state, session_id=None):
            calls.append((identity, state, session_id))
            return SimpleNamespace(
                status="written",
                reason="",
                session_id=session_id,
                optimized_throughput=10.0,
            )

    monkeypatch.setattr(
        remote_recipe.HyperloomRemoteKB,
        "from_env",
        classmethod(lambda cls: _Facade()),
    )
    WritebackCollaborator(coordinator).finalize_recipe_and_journal()
    assert calls == [
        (
            "inference:m:h:f:mt:a:v:p",
            coordinator.shared_state,
            tmp_path.name,
        )
    ]
    from hyperloom.inference_optimizer.session.session_paths import (
        recipe_snapshot_audit_jsonl,
    )

    audit_rows = [
        json.loads(line)
        for line in recipe_snapshot_audit_jsonl(tmp_path)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert audit_rows[-1]["status"] == "written"
    assert audit_rows[-1]["generator"] == "close"
    assert audit_rows[-1]["result"]["canonical_id"] == (
        "inference:m:h:f:mt:a:v:p"
    )


def test_remote_close_transport_failure_is_nonfatal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _Journal:
        def finalize(self, **kwargs) -> None:
            pass

    coordinator = SimpleNamespace(
        shared_state=SimpleNamespace(current_best={"tput": 10.0}),
        session_dir=tmp_path,
        recipe_kb=None,
        knowledge_plane=None,
        _ensure_journal=lambda: _Journal(),
        _workload_canonical_id=lambda: "inference:m:h:f:mt:a:v:p",
    )
    from hyperloom.orchestrator.knowledge import remote_recipe

    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.setenv("KB_STORE_URL", "https://kb.example")
    monkeypatch.setenv("KB_STORE_TOKEN", "token")
    monkeypatch.setattr(
        remote_recipe.HyperloomRemoteKB,
        "from_env",
        classmethod(
            lambda cls: (_ for _ in ()).throw(
                OSError("transport down")
            )
        ),
    )
    WritebackCollaborator(coordinator).finalize_recipe_and_journal()
    from hyperloom.inference_optimizer.session.session_paths import (
        recipe_snapshot_audit_jsonl,
    )

    row = json.loads(
        recipe_snapshot_audit_jsonl(tmp_path).read_text(encoding="utf-8")
    )
    assert row["status"] == "error"
    assert row["error"]["type"] == "OSError"


class _FakeStore:
    def __init__(
        self,
        *,
        champion: float = 0.0,
        conflict: bool = False,
        metric: str | None = None,
    ) -> None:
        self.champion = champion
        self.conflict = conflict
        self.metric = metric
        self.calls: list[tuple] = []
        self.files_listing: dict = {
            "files": [
                {
                    "path": "kernel/rewrite/verified.bin",
                    "size": len(_DOWNLOAD_BYTES),
                    "sha256": _DOWNLOAD_SHA256,
                    "_content": _DOWNLOAD_BYTES,
                }
            ]
        }
        self.skip_download_paths: set[str] = set()
        self.extra_download_file = False
        self.symlink_download_file = False
        self.envelope = {
            "schema_version": 2,
            "record_id": "record-1",
            "revision": 7,
            "canonical_id": "inference:m:h:f:mt:a:v:p",
            "session_id": "champion-session",
            "knowledge": {
                "optimized_throughput": 125.0,
                "validated_e2e_gain": 25.0,
                "value": {
                    "explore": {
                        "config": {
                            "extra_server_args": "--page-size 32",
                            "extra_envs": {"A": "1"},
                        }
                    }
                },
                "lessons": [{"statement": "x"}],
            },
        }

    def get_rollup(self, canonical_id):
        self.calls.append(("get_rollup", canonical_id))
        champion = {"session_id": "champion-session", "value": self.champion}
        if self.metric is not None:
            champion["metric"] = self.metric
        return {"champion": champion}

    def put_dir(self, canonical_id, session_id, files_dir):
        self.calls.append(("put_dir", canonical_id, session_id, Path(files_dir)))
        root = Path(files_dir)
        return {
            path.relative_to(root).as_posix(): f"kb://{path.relative_to(root).as_posix()}"
            for path in root.rglob("*")
            if path.is_file()
        }

    def put_knowledge(self, canonical_id, session_id, knowledge, *, mode):
        self.calls.append(("put_knowledge", canonical_id, session_id, mode))

    def set_champion(self, canonical_id, session_id, *, metric, value):
        self.calls.append(("set_champion", canonical_id, session_id, metric, value))
        if self.conflict:
            self.conflict = False
            self.champion = value - 1
            raise KBStoreError("POST champion -> HTTP 409: write_conflict")

    def get_session(self, canonical_id, session_id):
        self.calls.append(("get_session", canonical_id, session_id))
        return self.envelope

    def list_session_files(self, canonical_id, session_id):
        self.calls.append(("list_session_files", canonical_id, session_id))
        return self.files_listing

    def download_session(self, canonical_id, session_id, destination, *, include_values):
        self.calls.append(("download_session", canonical_id, session_id, include_values))
        files_root = Path(destination) / "files"
        files_root.mkdir(parents=True)
        for entry in self.files_listing.get("files") or []:
            relative = str(entry.get("path") or "")
            if relative in self.skip_download_paths:
                continue
            target = files_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            content = entry.get("_content", _DOWNLOAD_BYTES)
            if isinstance(content, str):
                content = content.encode()
            target.write_bytes(content)
        if self.extra_download_file:
            (files_root / "extra.bin").write_bytes(b"extra")
        if self.symlink_download_file:
            link = files_root / "download-link"
            link.symlink_to(Path(destination) / "outside")
        (Path(destination) / "values.json").write_text("stale", encoding="utf-8")
        (Path(destination) / "manifest.json").write_text("stale", encoding="utf-8")
        return []


def test_write_order_replace_metric_and_409_retry(tmp_path: Path) -> None:
    store = _FakeStore(champion=100.0, conflict=True)
    client = RemoteRecipeClient(store)  # type: ignore[arg-type]
    result = write_final_remote_recipe(
        _state(tmp_path),
        "inference:m:h:f:mt:a:v:p",
        "session-1",
        client=client,
    )
    names = [call[0] for call in store.calls]
    assert names[:4] == ["get_rollup", "put_dir", "put_knowledge", "set_champion"]
    assert names[-2:] == ["get_rollup", "set_champion"]
    assert store.calls[2][-1] == "replace"
    assert store.calls[3][-2:] == ("optimized_throughput", 130.0)
    assert result.status == "written"


def test_weaker_session_skips_before_upload(tmp_path: Path) -> None:
    store = _FakeStore(champion=140.0)
    result = write_final_remote_recipe(
        _state(tmp_path),
        "inference:m:h:f:mt:a:v:p",
        "session-1",
        client=RemoteRecipeClient(store),  # type: ignore[arg-type]
    )
    assert result.reason == "not_better_than_champion"
    assert [call[0] for call in store.calls] == ["get_rollup"]


def test_non_throughput_champion_metric_is_rejected(tmp_path: Path) -> None:
    store = _FakeStore(champion=1.0, metric="latency_ms")
    with pytest.raises(RemoteRecipeValidationError, match="latency_ms"):
        write_final_remote_recipe(
            _state(tmp_path),
            "inference:m:h:f:mt:a:v:p",
            "session-1",
            client=RemoteRecipeClient(store),  # type: ignore[arg-type]
        )
    assert [call[0] for call in store.calls] == ["get_rollup"]


def test_nonfinite_champion_value_is_rejected(tmp_path: Path) -> None:
    store = _FakeStore(champion=float("nan"))
    with pytest.raises(RemoteRecipeValidationError, match="finite"):
        write_final_remote_recipe(
            _state(tmp_path),
            "inference:m:h:f:mt:a:v:p",
            "session-1",
            client=RemoteRecipeClient(store),  # type: ignore[arg-type]
        )


def test_put_dir_refs_must_cover_every_uploaded_file(tmp_path: Path) -> None:
    store = _FakeStore(champion=100.0)
    store.put_dir = lambda *args, **kwargs: {}  # type: ignore[method-assign]
    with pytest.raises(RemoteRecipeValidationError, match="refs mismatch"):
        write_final_remote_recipe(
            _state(tmp_path),
            "inference:m:h:f:mt:a:v:p",
            "session-1",
            client=RemoteRecipeClient(store),  # type: ignore[arg-type]
        )
    assert not any(call[0] == "put_knowledge" for call in store.calls)


def test_read_sequence_writes_flat_recipe_and_files_only(tmp_path: Path) -> None:
    store = _FakeStore(champion=125.0)
    destination = tmp_path / "bundle"
    stale_files = destination / "files"
    stale_files.mkdir(parents=True)
    (stale_files / "old.patch").write_text("stale", encoding="utf-8")
    (destination / "recipe.json").write_text("stale", encoding="utf-8")
    (destination / "unrelated.txt").write_text("stale", encoding="utf-8")
    stale_dir = destination / "old-top-level"
    stale_dir.mkdir()
    (stale_dir / "old").write_text("stale", encoding="utf-8")
    row = read_remote_recipe(
        "inference:m:h:f:mt:a:v:p",
        destination,
        client=RemoteRecipeClient(store),  # type: ignore[arg-type]
    )
    assert [call[0] for call in store.calls] == [
        "get_rollup",
        "get_session",
        "list_session_files",
        "download_session",
    ]
    assert store.calls[-1][-1] is False
    saved = json.loads((tmp_path / "bundle" / "recipe.json").read_text(encoding="utf-8"))
    assert saved == row
    assert "knowledge" not in saved
    assert saved["schema_version"] == 2
    assert saved["record_id"] == "record-1"
    assert saved["revision"] == 7
    assert saved["version"] == 7
    assert saved["champion"]["session_id"] == "champion-session"
    assert saved["optimized_throughput"] == 125.0
    assert not (tmp_path / "bundle" / "values.json").exists()
    assert not (tmp_path / "bundle" / "manifest.json").exists()
    assert (tmp_path / "bundle" / "files").is_dir()
    assert not (tmp_path / "bundle" / "files" / "old.patch").exists()
    assert (
        tmp_path / "bundle" / "files" / "kernel" / "rewrite" / "verified.bin"
    ).read_bytes() == _DOWNLOAD_BYTES
    assert {path.name for path in destination.iterdir()} == {"recipe.json", "files"}


def test_read_rejects_existing_files_symlink(tmp_path: Path) -> None:
    store = _FakeStore(champion=125.0)
    destination = tmp_path / "bundle-link"
    target = tmp_path / "outside"
    target.mkdir()
    destination.mkdir()
    (destination / "files").symlink_to(target, target_is_directory=True)
    with pytest.raises(RemoteRecipeValidationError, match="symlink"):
        read_remote_recipe(
            "inference:m:h:f:mt:a:v:p",
            destination,
            client=RemoteRecipeClient(store),  # type: ignore[arg-type]
        )
    assert (destination / "files").is_symlink()


def test_read_rejects_destination_symlink(tmp_path: Path) -> None:
    store = _FakeStore(champion=125.0)
    target = tmp_path / "destination-target"
    target.mkdir()
    destination = tmp_path / "destination-link"
    destination.symlink_to(target, target_is_directory=True)
    with pytest.raises(RemoteRecipeValidationError, match="symlink destination"):
        read_remote_recipe(
            "inference:m:h:f:mt:a:v:p",
            destination,
            client=RemoteRecipeClient(store),  # type: ignore[arg-type]
        )
    assert destination.is_symlink()


def test_shared_store_reuses_one_rlock() -> None:
    store = _FakeStore()
    first = RemoteRecipeClient(store)  # type: ignore[arg-type]
    second = RemoteRecipeClient(store)  # type: ignore[arg-type]
    assert first._store_lock is second._store_lock
    assert first._store_lock is store._hyperloom_remote_recipe_lock


def test_read_champion_metric_must_be_throughput(tmp_path: Path) -> None:
    store = _FakeStore(champion=125.0, metric="latency_ms")
    with pytest.raises(RemoteRecipeValidationError, match="latency_ms"):
        read_remote_recipe(
            "inference:m:h:f:mt:a:v:p",
            tmp_path / "wrong-read-metric",
            client=RemoteRecipeClient(store),  # type: ignore[arg-type]
        )
    assert [call[0] for call in store.calls] == ["get_rollup"]


@pytest.mark.parametrize(
    "mode,match",
    [
        ("sha", "sha256 mismatch"),
        ("size", "size mismatch"),
        ("missing", "artifact set mismatch"),
        ("extra", "artifact set mismatch"),
        ("symlink", "symlink"),
    ],
)
def test_read_verifies_downloaded_manifest_before_recipe(
    tmp_path: Path,
    mode: str,
    match: str,
) -> None:
    store = _FakeStore(champion=125.0)
    entry = store.files_listing["files"][0]
    if mode == "sha":
        entry["sha256"] = "0" * 64
    elif mode == "size":
        entry["size"] = len(_DOWNLOAD_BYTES) + 1
    elif mode == "missing":
        store.skip_download_paths.add(entry["path"])
    elif mode == "extra":
        store.extra_download_file = True
    elif mode == "symlink":
        store.symlink_download_file = True
    destination = tmp_path / f"verify-{mode}"
    with pytest.raises(RemoteRecipeValidationError, match=match):
        read_remote_recipe(
            "inference:m:h:f:mt:a:v:p",
            destination,
            client=RemoteRecipeClient(store),  # type: ignore[arg-type]
        )
    assert not (destination / "recipe.json").exists()


def test_read_rejects_listing_without_required_size(tmp_path: Path) -> None:
    store = _FakeStore(champion=125.0)
    store.files_listing = {
        "files": [
            {
                "path": "kernel/rewrite/source.cu",
                "sha256": _DOWNLOAD_SHA256,
                "_content": _DOWNLOAD_BYTES,
                "download_url": "unused",
            }
        ]
    }
    with pytest.raises(RemoteRecipeValidationError, match="size is required"):
        read_remote_recipe(
            "inference:m:h:f:mt:a:v:p",
            tmp_path / "missing-size",
            client=RemoteRecipeClient(store),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "listing,match",
    [
        (
            {"files": [{"path": "../escape", "size": 1, "sha256": _DOWNLOAD_SHA256}]},
            "invalid artifact path",
        ),
        (
            {"files": [{"path": "a", "size": "1", "sha256": _DOWNLOAD_SHA256}]},
            "size must be an integer",
        ),
        (
            {"files": [{"path": "a", "size": -1, "sha256": _DOWNLOAD_SHA256}]},
            "outside",
        ),
        (
            {
                "files": [
                    {
                        "path": "a",
                        "size": MAX_FILE_BYTES + 1,
                        "sha256": _DOWNLOAD_SHA256,
                    }
                ]
            },
            "outside",
        ),
        (
            {
                "files": [
                    {"path": f"f-{index}", "sha256": _DOWNLOAD_SHA256}
                    for index in range(513)
                ]
            },
            "file count",
        ),
        ({"files": [{"path": "a", "size": 1}]}, "sha256"),
        (
            {"files": [{"path": "a", "sha256": _DOWNLOAD_SHA256}]},
            "size is required",
        ),
        (
            {"files": [{"path": "a", "size": 1, "sha256": "A" * 64}]},
            "sha256",
        ),
    ],
)
def test_read_validates_listing_before_cleanup(
    tmp_path: Path,
    listing: dict,
    match: str,
) -> None:
    store = _FakeStore(champion=125.0)
    store.files_listing = listing
    destination = tmp_path / "invalid-listing"
    destination.mkdir()
    stale = destination / "recipe.json"
    stale.write_text("preserve-on-validation-failure", encoding="utf-8")
    with pytest.raises(RemoteRecipeValidationError, match=match):
        read_remote_recipe(
            "inference:m:h:f:mt:a:v:p",
            destination,
            client=RemoteRecipeClient(store),  # type: ignore[arg-type]
        )
    assert stale.read_text(encoding="utf-8") == "preserve-on-validation-failure"
    assert not any(call[0] == "download_session" for call in store.calls)


@pytest.mark.parametrize(
    "mode,match",
    [
        ("not_object", "must be an object"),
        ("schema", "schema_version"),
        ("canonical_id", "canonical_id"),
        ("session_id", "session_id"),
        ("record_id", "record_id"),
        ("version", "revision or version"),
        ("knowledge", "knowledge"),
    ],
)
def test_bad_envelope_does_not_clean_destination(
    tmp_path: Path,
    mode: str,
    match: str,
) -> None:
    store = _FakeStore(champion=125.0)
    if mode == "not_object":
        store.envelope = []  # type: ignore[assignment]
    elif mode == "schema":
        store.envelope["schema_version"] = 1
    elif mode == "canonical_id":
        store.envelope["canonical_id"] = "inference:wrong"
    elif mode == "session_id":
        store.envelope["session_id"] = "wrong-session"
    elif mode == "record_id":
        store.envelope["record_id"] = ""
    elif mode == "version":
        store.envelope.pop("revision")
    elif mode == "knowledge":
        store.envelope["knowledge"] = []
    destination = tmp_path / f"bad-envelope-{mode}"
    destination.mkdir()
    sentinel = destination / "keep-me"
    sentinel.write_text("unchanged", encoding="utf-8")
    with pytest.raises(RemoteRecipeValidationError, match=match):
        read_remote_recipe(
            "inference:m:h:f:mt:a:v:p",
            destination,
            client=RemoteRecipeClient(store),  # type: ignore[arg-type]
        )
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert not any(call[0] == "list_session_files" for call in store.calls)


def test_nonfinite_write_throughput_skips_without_remote_calls(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.current_best["tput"] = float("inf")
    store = _FakeStore(champion=0.0)
    result = write_final_remote_recipe(
        state,
        "inference:m:h:f:mt:a:v:p",
        "session-1",
        client=RemoteRecipeClient(store),  # type: ignore[arg-type]
    )
    assert result.status == "skipped"
    assert result.reason == "nonfinite_optimized_throughput"
    assert store.calls == []


def test_nonfinite_built_metrics_are_normalized(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.current_best["tput"] = float("nan")
    state.cumulative_gain_validated = float("inf")
    state.cumulative_gain = float("-inf")
    bundle = build_remote_knowledge(state, tmp_path / "finite-knowledge")
    assert bundle.knowledge["optimized_throughput"] == 0.0
    assert bundle.knowledge["validated_e2e_gain"] == 0.0


def test_v1_conversion_helpers_round_trip_reader_fields() -> None:
    knowledge = convert_v1_recipe_to_knowledge(
        {
            "best_config": {"args": "--foo", "envs": {"A": "1"}},
            "best_throughput": 10.0,
            "validated_gain_pct": 2.0,
            "lessons": [{"statement": "keep foo"}],
        }
    )
    row = envelope_to_v1_recipe(
        {
            "schema_version": 2,
            "canonical_id": "inference:m:h:f:mt:a:v:p",
            "session_id": "s",
            "knowledge": knowledge,
        }
    )
    assert row["best_config"] == {"extra_server_args": "--foo", "extra_envs": {"A": "1"}}
    assert row["best_throughput"] == 10.0
    assert row["validated_gain_pct"] == 2.0
    assert row["lessons"] == [{"statement": "keep foo"}]


def test_v1_projection_accepts_flat_recipe_document() -> None:
    row = envelope_to_v1_recipe(
        {
            "schema_version": 2,
            "canonical_id": "inference:m:h:f:mt:a:v:p",
            "session_id": "s",
            "optimized_throughput": 10.0,
            "validated_e2e_gain": 2.0,
            "value": {
                "explore": {
                    "extra_server_args": "--foo",
                    "extra_envs": {"A": "1"},
                    "patches": [],
                    "artifacts": [],
                }
            },
        }
    )
    assert row["best_config"]["extra_server_args"] == "--foo"
    assert row["best_config"]["extra_envs"] == {"A": "1"}


def test_v1_projection_merges_explore_and_framework_config() -> None:
    row = envelope_to_v1_recipe(
        {
            "schema_version": 2,
            "canonical_id": "inference:m:h:f:mt:a:v:p",
            "session_id": "s",
            "optimized_throughput": 10.0,
            "value": {
                "explore": {
                    "extra_server_args": "--page-size 32",
                    "extra_envs": {"EXPLORE": "1", "SHARED": "explore"},
                },
                "framework": {
                    "extra_server_args": "--enable-foo",
                    "extra_envs": {"FRAMEWORK": "1", "SHARED": "framework"},
                },
            },
        }
    )
    config = row["best_config"]
    assert config["extra_server_args"] == "--page-size 32 --enable-foo"
    assert config["extra_envs"] == {
        "EXPLORE": "1",
        "FRAMEWORK": "1",
        "SHARED": "framework",
    }


def test_read_is_not_wired_into_t0_and_close_write_remains() -> None:
    t0_source = inspect.getsource(run_t0_anchor)
    close_source = inspect.getsource(WritebackCollaborator.finalize_recipe_and_journal)
    assert "read_remote" not in t0_source
    assert "remote_recipe" not in t0_source
    assert "HyperloomRemoteKB.from_env().write" in close_source
    assert "write_final_remote_recipe" not in close_source


def test_vendored_sdk_matches_upstream_git_blob() -> None:
    path = Path(kb_store_client.__file__)
    content = path.read_bytes()
    digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    assert digest == "cb7849422030798860ef5985cfc047392b60f061"


def test_read_alias_is_standalone_api() -> None:
    assert read_remote_champion is read_remote_recipe
