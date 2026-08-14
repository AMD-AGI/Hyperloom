# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from hyperloom.orchestrator.knowledge import remote_recipe
from hyperloom.orchestrator.knowledge.remote_recipe import (
    HyperloomRemoteKB,
    KBStoreError,
    RemoteRecipeClient,
    RemoteRecipeConfigurationError,
    RemoteWarmRecipeAdapter,
    build_remote_knowledge,
    convert_v1_recipe_to_knowledge,
    envelope_to_v1_recipe,
    has_new_keep,
    read_remote_champion,
    read_remote_recipe,
    write_final_remote_recipe,
    write_kernel_agent_kb,
)
from hyperloom.orchestrator.knowledge.remote_recipe._vendor import kb_store_client
from hyperloom.orchestrator.knowledge.remote_recipe.client import (
    _champion,
    _validate_download_listing,
    _validate_session_envelope,
    _verify_downloaded_files,
)
from hyperloom.orchestrator.knowledge.remote_recipe.models import (
    MAX_FILE_BYTES,
    MAX_PATH_BYTES,
    Artifact,
    KnowledgeBundle,
    RemoteRecipeValidationError,
    RemoteWriteResult,
    extract_knowledge_artifact_refs,
)
from hyperloom.orchestrator.knowledge.remote_recipe.sanitize import (
    sanitize_publish_env_mapping,
    sanitize_publish_server_args,
    sanitize_shared_knowledge,
)
from hyperloom.orchestrator.knowledge.remote_recipe.values import (
    KERNEL_AGENT_SESSION_ID,
    _Files,
)
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
                "extra_envs": {"VLLM_EXPLORE_TEST": "1"},
                "patch_path": str(explore_patch),
                "tput": 110.0,
            },
            {
                "action": "integrate_patch",
                "source_phase": "FRAMEWORK_AGENT",
                "extra_server_args": "--page-size 32 --enable-foo",
                "extra_envs": {"VLLM_FRAMEWORK_TEST": "1"},
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
    assert bundle.knowledge["knowledge_schema_version"] == 2
    assert bundle.knowledge["validated_e2e_gain"] == 30.0
    value = bundle.knowledge["value"]
    assert value["explore"]["extra_envs"] == {"VLLM_EXPLORE_TEST": "1"}
    assert value["framework"]["extra_envs"] == {"VLLM_FRAMEWORK_TEST": "1"}
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


def test_publish_sanitizer_allows_only_safe_replay_envs_and_args() -> None:
    envs = sanitize_publish_env_mapping(
        {
            "VLLM_ROCM_USE_AITER": "1",
            "SGLANG_SAFE_TOGGLE": "true",
            "OPENAI_API_KEY": "openai-secret",
            "VLLM_API_TOKEN": "vllm-secret",
            "UNKNOWN_TOGGLE": "1",
            "HIP_VISIBLE_DEVICES": "0,1",
            "VLLM_CONFIG_PATH": "/workspace/session/config.json",
            "VLLM_HEADER": "Bearer secret-value",
        }
    )
    assert envs == {
        "VLLM_ROCM_USE_AITER": "1",
        "SGLANG_SAFE_TOGGLE": "true",
    }

    assert (
        sanitize_publish_server_args(
            "--page-size 32 "
            "--api-key secret "
            "--download-dir /shared_nfs/private/cache "
            "--auth-token=secret "
            "--attention-backend ROCM_AITER"
        )
        == "--page-size 32 --attention-backend ROCM_AITER"
    )


def test_shared_knowledge_sanitizer_scrubs_nested_columns_and_paths() -> None:
    sanitized = sanitize_shared_knowledge(
        {
            "value": {
                "explore": {
                    "extra_envs": {
                        "VLLM_ROCM_USE_AITER": "1",
                        "HF_TOKEN": "secret",
                    },
                    "extra_server_args": "--page-size 32 --password hidden",
                },
                "kernel": {
                    "rewrite": {
                        "workspace": "/workspace/hyperloom/session",
                        "api_token": "secret",
                        "source_file": "kernel/rewrite/source/kernel.py",
                        "note": (
                            "failed at /home/operator/session/log.txt "
                            "with TOKEN=secret"
                        ),
                    }
                },
            }
        }
    )

    explore = sanitized["value"]["explore"]
    assert explore == {
        "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
        "extra_server_args": "--page-size 32",
    }
    rewrite = sanitized["value"]["kernel"]["rewrite"]
    assert "workspace" not in rewrite
    assert "api_token" not in rewrite
    assert rewrite["source_file"] == "kernel/rewrite/source/kernel.py"
    assert "[LOCAL_PATH]" in rewrite["note"]
    assert "secret" not in rewrite["note"]


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


def test_geak_results_are_excluded_from_remote_knowledge(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.optimization_stack = [
        {
            "action": "geak_e2e",
            "source_phase": "KERNEL_AGENT",
            "variant_name": "geak_e2e",
            "tput": 135.0,
            "candidate_extra_server_args": "--geak-fast",
            "extra_envs": {"GEAK_OPT": "1"},
            "accepted_kernels": ["attention"],
            "accepted_heads": ["decode"],
            "ts": "2026-08-08T00:00:00+00:00",
        }
    ]
    state.geak_result = {"status": "ok", "throughput_speedup": 1.34}

    bundle = build_remote_knowledge(state, tmp_path / "files-geak")
    kernel = bundle.knowledge["value"]["kernel"]
    assert set(kernel) == {"gemm", "fusion", "rewrite"}
    assert all(not artifact.path.startswith("kernel/geak/") for artifact in bundle.artifacts)


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
        bundle.validate()
    bad = KnowledgeBundle({"value": {}}, [Artifact("files/explore/a", source)])
    with pytest.raises(RemoteRecipeValidationError, match="files/ prefix"):
        bad.validate()
    missing = KnowledgeBundle(
        {"value": {"explore": {"patches": ["explore/missing.patch"]}}}
    )
    with pytest.raises(RemoteRecipeValidationError, match="missing artifacts"):
        missing.validate()


def test_mixed_slash_free_text_is_not_treated_as_an_artifact_ref(
    tmp_path: Path,
) -> None:
    source = tmp_path / "accepted.patch"
    source.write_text("diff", encoding="utf-8")
    files = _Files(tmp_path / "bundle-files")
    ref = files.add(source, category="explore", kind="patches")
    knowledge = {
        "value": {
            "explore": {
                "patches": [ref],
                "patch": "see notes at a/b\\c",
            }
        }
    }

    files.prune_superseded(knowledge)
    assert [artifact.path for artifact in files.artifacts] == [ref]
    KnowledgeBundle(knowledge, files.artifacts).validate()


def test_remote_client_internal_validation_error_paths(tmp_path: Path) -> None:
    assert _champion(None) == ("", 0.0, {})
    assert _champion({"sessions": [], "champion": None}) == ("", 0.0, {})
    valid = {
        "champion": {
            "session_id": "best",
            "metric": "optimized_throughput",
            "value": 12.0,
        }
    }
    assert _champion(valid, validate_metric=True)[1] == 12.0
    with pytest.raises(RemoteRecipeValidationError, match="missing champion"):
        _champion({})
    with pytest.raises(RemoteRecipeValidationError, match="sessions but no champion"):
        _champion({"sessions": [{"session_id": "candidate"}], "champion": None})
    with pytest.raises(RemoteRecipeValidationError, match="must be numeric"):
        _champion(
            {
                "champion": {
                    "session_id": "best",
                    "metric": "optimized_throughput",
                    "value": "not-a-number",
                }
            },
            validate_metric=True,
        )
    with pytest.raises(RemoteRecipeValidationError, match="finite"):
        _champion(
            {
                "champion": {
                    "session_id": "best",
                    "metric": "optimized_throughput",
                    "value": float("inf"),
                }
            },
            validate_metric=True,
        )

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

    with pytest.raises(RemoteRecipeValidationError, match="requested session"):
        _validate_session_envelope(
            {
                "schema_version": 2,
                "canonical_id": "inference:m:h:f:mt:a:v:p",
                "session_id": "actual-session",
                "record_id": "record",
                "revision": 1,
                "knowledge": {},
            },
            canonical_id="inference:m:h:f:mt:a:v:p",
            session_id="expected-session",
        )

    class _ReadStore:
        def __init__(self, envelope=None) -> None:
            self.envelope = envelope

        def get_best_record(self, _canonical_id):
            return self.envelope

    identity = "inference:m:h:f:mt:a:v:p"
    assert RemoteRecipeClient(_ReadStore()).read(identity, tmp_path / "miss") is None  # type: ignore[arg-type]


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
        KnowledgeBundle({"bad": float("nan")}).validate()


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

    outcome = WritebackCollaborator(coordinator).finalize_recipe_and_journal()
    assert outcome == {
        "status": "skipped",
        "reason": "degraded_kb",
        "backend": "disabled",
    }


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
    outcome = WritebackCollaborator(coordinator).finalize_recipe_and_journal()
    assert outcome == {
        "status": "skipped",
        "reason": "no_recipe_backend",
        "backend": "local",
    }
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
    outcome = WritebackCollaborator(coordinator).finalize_recipe_and_journal()
    assert outcome == {
        "status": "written",
        "reason": "",
        "backend": "kb-store",
        "canonical_id": "inference:m:h:f:mt:a:v:p",
        "session_id": tmp_path.name,
    }
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
    outcome = WritebackCollaborator(coordinator).finalize_recipe_and_journal()
    assert outcome == {
        "status": "error",
        "reason": "OSError",
        "backend": "kb-store",
        "canonical_id": "inference:m:h:f:mt:a:v:p",
        "session_id": tmp_path.name,
    }
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
        metric: str | None = "optimized_throughput",
    ) -> None:
        self.champion = champion
        self.champion_session = "champion-session"
        self.conflict = conflict
        self.metric = metric
        self.calls: list[tuple] = []
        self.published_knowledge: dict | None = None
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
        champion = {"session_id": self.champion_session, "value": self.champion}
        if self.metric is not None:
            champion["metric"] = self.metric
        return {"champion": champion}

    def get_best_record(self, canonical_id):
        self.calls.append(("get_best_record", canonical_id))
        return self.envelope

    def put_dir(self, canonical_id, session_id, files_dir):
        self.calls.append(("put_dir", canonical_id, session_id, Path(files_dir)))
        root = Path(files_dir)
        return {
            path.relative_to(root).as_posix(): f"kb://{path.relative_to(root).as_posix()}"
            for path in root.rglob("*")
            if path.is_file()
        }

    def put_knowledge(self, canonical_id, knowledge, *, session_id="", mode="merge"):
        self.calls.append(("put_knowledge", canonical_id, session_id, mode))
        self.published_knowledge = json.loads(json.dumps(knowledge))

    def set_champion(self, canonical_id, session_id, *, metric, value):
        self.calls.append(("set_champion", canonical_id, session_id, metric, value))
        if self.conflict:
            self.conflict = False
            raise KBStoreError("POST champion -> HTTP 409: write_conflict")
        # The store promotes whatever it is told: measured against the real
        # service, it accepts an equal or even a lower value.
        self.champion = value
        self.champion_session = session_id

    def list_session_files(self, canonical_id, session_id, *, kind=""):
        self.calls.append(("list_session_files", canonical_id, session_id, kind))
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


def test_write_boundary_sanitizes_directly_constructed_bundle(tmp_path: Path) -> None:
    store = _FakeStore()
    bundle = KnowledgeBundle(
        {
            "optimized_throughput": 130.0,
            "value": {
                "explore": {
                    "extra_server_args": "--page-size 32 --api-key secret",
                    "extra_envs": {
                        "VLLM_ROCM_USE_AITER": "1",
                        "OPENAI_API_KEY": "secret",
                    },
                    "workspace": "/workspace/private/session",
                }
            },
        }
    )

    result = RemoteRecipeClient(store).write_if_better(  # type: ignore[arg-type]
        "inference:m:h:f:mt:a:v:p",
        "session-1",
        bundle,
        optimized_throughput=130.0,
        files_dir=tmp_path,
    )

    assert result.status == "written"
    assert store.published_knowledge is not None
    explore = store.published_knowledge["value"]["explore"]
    assert explore == {
        "extra_server_args": "--page-size 32",
        "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
    }


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


@pytest.mark.parametrize(
    "rollup,match",
    [
        ({}, "missing champion"),
        (
            {"sessions": [{"session_id": "candidate"}], "champion": None},
            "sessions but no champion",
        ),
        ({"sessions": [], "champion": "bad"}, "must be an object"),
    ],
)
def test_malformed_rollup_fails_closed_before_write(
    tmp_path: Path,
    rollup: dict,
    match: str,
) -> None:
    class _MalformedRollupStore(_FakeStore):
        def get_rollup(self, canonical_id):
            self.calls.append(("get_rollup", canonical_id))
            return rollup

    store = _MalformedRollupStore()
    with pytest.raises(RemoteRecipeValidationError, match=match):
        write_final_remote_recipe(
            _state(tmp_path),
            "inference:m:h:f:mt:a:v:p",
            "session-1",
            client=RemoteRecipeClient(store),  # type: ignore[arg-type]
        )
    assert [call[0] for call in store.calls] == ["get_rollup"]


def test_absent_rollup_is_treated_as_first_write(tmp_path: Path) -> None:
    class _FirstWriteStore(_FakeStore):
        def get_rollup(self, canonical_id):
            self.calls.append(("get_rollup", canonical_id))
            return None

    store = _FirstWriteStore()
    result = write_final_remote_recipe(
        _state(tmp_path),
        "inference:m:h:f:mt:a:v:p",
        "session-1",
        client=RemoteRecipeClient(store),  # type: ignore[arg-type]
    )
    assert result.status == "written"
    assert [call[0] for call in store.calls] == [
        "get_rollup",
        "put_dir",
        "put_knowledge",
        "set_champion",
    ]


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
        "get_best_record",
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
    assert saved["session_id"] == "champion-session"
    assert saved["optimized_throughput"] == 125.0
    assert not (tmp_path / "bundle" / "values.json").exists()
    assert not (tmp_path / "bundle" / "manifest.json").exists()
    assert (tmp_path / "bundle" / "files").is_dir()
    assert not (tmp_path / "bundle" / "files" / "old.patch").exists()
    assert (
        tmp_path / "bundle" / "files" / "kernel" / "rewrite" / "verified.bin"
    ).read_bytes() == _DOWNLOAD_BYTES
    assert {path.name for path in destination.iterdir()} == {"recipe.json", "files"}


def test_read_pins_manifest_against_download_relisting(tmp_path: Path) -> None:
    class _RelistingStore(_FakeStore):
        def download_session(
            self,
            canonical_id,
            session_id,
            destination,
            *,
            include_values,
        ):
            assert self.list_session_files(canonical_id, session_id) is self.files_listing
            self.list_session_files("other:identity", session_id, kind="patch")
            return super().download_session(
                canonical_id,
                session_id,
                destination,
                include_values=include_values,
            )

    store = _RelistingStore()
    row = read_remote_recipe(
        "inference:m:h:f:mt:a:v:p",
        tmp_path / "pinned-manifest",
        client=RemoteRecipeClient(store),  # type: ignore[arg-type]
    )

    assert row is not None
    assert ("list_session_files", "other:identity", "champion-session", "patch") in store.calls


def test_read_rejects_non_json_knowledge_before_destination_cleanup(
    tmp_path: Path,
) -> None:
    store = _FakeStore()
    store.envelope["knowledge"]["not_json"] = object()
    destination = tmp_path / "strict-json"
    destination.mkdir()
    sentinel = destination / "keep-me"
    sentinel.write_text("unchanged", encoding="utf-8")

    with pytest.raises(RemoteRecipeValidationError, match="strict JSON"):
        read_remote_recipe(
            "inference:m:h:f:mt:a:v:p",
            destination,
            client=RemoteRecipeClient(store),  # type: ignore[arg-type]
        )

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert [call[0] for call in store.calls] == ["get_best_record"]


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


def test_read_does_not_consult_rollup_champion_metric(tmp_path: Path) -> None:
    store = _FakeStore(champion=125.0, metric="latency_ms")
    row = read_remote_recipe(
        "inference:m:h:f:mt:a:v:p",
        tmp_path / "direct-best-record",
        client=RemoteRecipeClient(store),  # type: ignore[arg-type]
    )
    assert row is not None
    assert [call[0] for call in store.calls] == [
        "get_best_record",
        "list_session_files",
        "download_session",
    ]


def test_read_passes_through_the_services_selection_reason(tmp_path: Path) -> None:
    store = _FakeStore()
    store.envelope["selected_by"] = {
        "reason": "champion",
        "metric": "optimized_throughput",
        "value": 125.0,
        "promoted_at": "2026-08-08T00:00:00Z",
    }
    row = read_remote_recipe(
        "inference:m:h:f:mt:a:v:p",
        tmp_path / "direct-best-record",
        client=RemoteRecipeClient(store),  # type: ignore[arg-type]
    )
    assert row is not None
    assert row["selected_by"] == store.envelope["selected_by"]


def _sections(tmp_path: Path):
    return kb_store_client.KnowledgeSections(tmp_path / "draft")


def test_an_agent_staged_section_overrides_what_the_stack_scrape_guessed(
    tmp_path: Path,
) -> None:
    sections = _sections(tmp_path)
    sections.write("framework", {"extra_server_args": "--authored-by-the-agent"})
    bundle = build_remote_knowledge(
        _state(tmp_path), tmp_path / "files", sections=sections
    )
    framework = bundle.knowledge["value"]["framework"]
    assert framework["extra_server_args"] == "--authored-by-the-agent"
    # Keys the agent did not stage still come from the scrape.
    assert framework["extra_envs"] == {"VLLM_FRAMEWORK_TEST": "1"}
    assert bundle.knowledge["provenance"]["staged_sections"] == ["framework"]


def test_agent_staged_sections_are_sanitized_at_publish_boundary(
    tmp_path: Path,
) -> None:
    sections = _sections(tmp_path)
    source = tmp_path / "kernel.py"
    source.write_text("pass\n", encoding="utf-8")
    sections.write(
        "kernel",
        {
            "rewrite": {
                "extra_envs": {
                    "SGLANG_SAFE_TOGGLE": "1",
                    "ANTHROPIC_API_KEY": "secret",
                },
                "workspace": "/home/operator/private/session",
                "access_token": "secret",
                "source_file": "kernel/rewrite/kernel.py",
            }
        },
        files=[source],
        kind="rewrite",
    )

    bundle = build_remote_knowledge(
        _state(tmp_path),
        tmp_path / "files",
        sections=sections,
    )

    rewrite = bundle.knowledge["value"]["kernel"]["rewrite"]
    assert rewrite["extra_envs"] == {"SGLANG_SAFE_TOGGLE": "1"}
    assert rewrite["source_file"] == "kernel/rewrite/kernel.py"
    assert "workspace" not in rewrite
    assert "access_token" not in rewrite


def test_a_section_nobody_staged_is_left_to_the_stack_scrape(tmp_path: Path) -> None:
    sections = _sections(tmp_path)
    sections.write("framework", {"extra_server_args": "--authored"})
    bundle = build_remote_knowledge(
        _state(tmp_path), tmp_path / "files", sections=sections
    )
    explore = bundle.knowledge["value"]["explore"]
    assert explore["extra_envs"] == {"VLLM_EXPLORE_TEST": "1"}
    assert bundle.knowledge["provenance"]["staged_sections"] == ["framework"]


def test_a_staged_file_is_published_under_its_own_section(tmp_path: Path) -> None:
    patch = tmp_path / "authored.patch"
    patch.write_text("authored", encoding="utf-8")
    sections = _sections(tmp_path)
    sections.write(
        "framework",
        {
            "note": "one",
            "patches": ["framework/patches/authored.patch"],
        },
        files=[patch],
        kind="patches",
    )
    bundle = build_remote_knowledge(
        _state(tmp_path), tmp_path / "files", sections=sections
    )
    published = {artifact.path for artifact in bundle.artifacts}
    assert "framework/patches/authored.patch" in published
    assert (tmp_path / "files" / "framework" / "patches" / "authored.patch").is_file()


def test_agent_column_free_text_is_not_treated_as_an_artifact_ref(
    tmp_path: Path,
) -> None:
    sections = _sections(tmp_path)
    sections.write(
        "kernel",
        {
            "gemm": {
                "optimizations": [{"id": "g1"}],
                "report": "tuned 3 shapes, 1.4x on the hot GEMM",
                "patch": "inlined the epilogue",
            }
        },
    )

    bundle = build_remote_knowledge(
        _state(tmp_path),
        tmp_path / "files-free-text",
        sections=sections,
    )

    gemm = bundle.knowledge["value"]["kernel"]["gemm"]
    assert gemm["report"] == "tuned 3 shapes, 1.4x on the hot GEMM"
    assert gemm["patch"] == "inlined the epilogue"


def test_orphaned_staged_file_falls_back_to_stack_scrape(tmp_path: Path) -> None:
    patch = tmp_path / "orphan.patch"
    patch.write_text("orphan", encoding="utf-8")
    sections = _sections(tmp_path)
    sections.write("framework", {"note": "no ref"}, files=[patch], kind="patches")

    bundle = build_remote_knowledge(
        _state(tmp_path),
        tmp_path / "files-orphan",
        sections=sections,
    )

    framework = bundle.knowledge["value"]["framework"]
    assert framework["extra_server_args"] == "--page-size 32 --enable-foo"
    assert "framework" not in bundle.knowledge["provenance"]["staged_sections"]
    assert "framework/patches/orphan.patch" not in {
        artifact.path for artifact in bundle.artifacts
    }


def test_conflicting_staged_artifact_falls_back_to_stack_scrape(tmp_path: Path) -> None:
    staged_patch = tmp_path / "staged" / "explore.patch"
    staged_patch.parent.mkdir()
    staged_patch.write_text("different", encoding="utf-8")
    sections = _sections(tmp_path)
    sections.write(
        "explore",
        {"patches": ["explore/patches/explore.patch"]},
        files=[staged_patch],
        kind="patches",
    )

    bundle = build_remote_knowledge(
        _state(tmp_path),
        tmp_path / "files-conflict",
        sections=sections,
    )

    assert "explore" not in bundle.knowledge["provenance"]["staged_sections"]
    assert bundle.knowledge["value"]["explore"]["patches"] == [
        "explore/patches/explore.patch"
    ]


def test_the_kernel_column_keeps_the_sub_columns_the_agent_left_alone(
    tmp_path: Path,
) -> None:
    sections = _sections(tmp_path)
    sections.write("kernel", {"gemm": {"authored": True}})
    bundle = build_remote_knowledge(
        _state(tmp_path), tmp_path / "files", sections=sections
    )
    kernel = bundle.knowledge["value"]["kernel"]
    assert kernel["gemm"] == {"authored": True}
    assert "rewrite" in kernel and "fusion" in kernel


def test_no_draft_at_all_publishes_exactly_what_it_used_to(tmp_path: Path) -> None:
    without = build_remote_knowledge(_state(tmp_path), tmp_path / "a")
    with_empty = build_remote_knowledge(
        _state(tmp_path), tmp_path / "b", sections=_sections(tmp_path)
    )
    assert without.knowledge["value"] == with_empty.knowledge["value"]
    assert without.knowledge["provenance"]["staged_sections"] == []


def test_a_record_without_a_selection_reason_still_reads(tmp_path: Path) -> None:
    store = _FakeStore()
    row = read_remote_recipe(
        "inference:m:h:f:mt:a:v:p",
        tmp_path / "direct-best-record",
        client=RemoteRecipeClient(store),  # type: ignore[arg-type]
    )
    assert row is not None
    assert row["selected_by"] == {}


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
        store.envelope["session_id"] = ""
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
            "best_config": {"args": "--foo", "envs": {"VLLM_LEGACY_TEST": "1"}},
            "best_throughput": 10.0,
            "validated_gain_pct": 2.0,
            "lessons": [{"statement": "keep foo"}],
        }
    )
    assert knowledge["knowledge_schema_version"] == 1
    assert knowledge["value"]["legacy_recipe"]["lessons"] == [
        {"statement": "keep foo"}
    ]
    row = envelope_to_v1_recipe(
        {
            "schema_version": 2,
            "canonical_id": "inference:m:h:f:mt:a:v:p",
            "session_id": "s",
            "knowledge": knowledge,
        }
    )
    assert row["best_config"] == {
        "extra_server_args": "--foo",
        "extra_envs": {"VLLM_LEGACY_TEST": "1"},
    }
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


def test_v2_warm_projection_replays_only_explore_config() -> None:
    row = envelope_to_v1_recipe(
        {
            "schema_version": 2,
            "knowledge_schema_version": 2,
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
    assert config["extra_server_args"] == "--page-size 32"
    assert config["extra_envs"] == {
        "EXPLORE": "1",
        "SHARED": "explore",
    }
    assert row["prs_tested"] == []


def test_remote_warm_adapter_projects_and_caches_explore_config(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, Path]] = []

    class _Remote:
        def read(self, identity: str, destination: Path):
            calls.append((identity, destination))
            return {
                "schema_version": 2,
                "knowledge_schema_version": 2,
                "canonical_id": identity,
                "session_id": "champion",
                "optimized_throughput": 10.0,
                "validated_e2e_gain": 2.0,
                "value": {
                    "explore": {
                        "extra_server_args": "--page-size 32",
                        "extra_envs": {"A": "1"},
                        "patches": ["ignored.patch"],
                    },
                    "framework": {
                        "extra_server_args": "--framework-flag",
                        "extra_envs": {"FRAMEWORK": "1"},
                    },
                },
            }

    identity = "inference:m:h:f:mt:a:v:p"
    destination = tmp_path / "remote-recipe"
    adapter = RemoteWarmRecipeAdapter(_Remote(), destination)  # type: ignore[arg-type]

    first = adapter.get_authoritative_recipe(canonical_id=identity)
    second = adapter.get_recipe(canonical_id=identity)

    assert first == second
    assert first["best_config"] == {
        "extra_server_args": "--page-size 32",
        "extra_envs": {"A": "1"},
    }
    assert first["prs_tested"] == []
    assert calls == [(identity, destination)]
    assert adapter.search(label_match={}, limit=5) == []


def test_direct_v1_envelope_replays_legacy_explore_config(tmp_path: Path) -> None:
    store = _FakeStore()
    store.envelope["artifacts"] = {"file_count": 0, "files": []}
    store.envelope["knowledge"] = {
        "knowledge_schema_version": 1,
        "optimized_throughput": 111.0,
        "value": {
            "legacy_recipe": {
                "best_config": {
                    "args": "--legacy-page-size 64",
                    "envs": {"LEGACY_EXPLORE": "1"},
                },
                "best_throughput": 111.0,
                "prs_tested": [{"number": 123, "patch": "must-not-replay"}],
            }
        },
    }
    adapter = RemoteWarmRecipeAdapter(
        RemoteRecipeClient(store),  # type: ignore[arg-type]
        tmp_path / "direct-v1",
    )

    row = adapter.get_authoritative_recipe(
        canonical_id="inference:m:h:f:mt:a:v:p"
    )

    assert row is not None
    assert row["best_config"] == {
        "extra_server_args": "--legacy-page-size 64",
        "extra_envs": {"LEGACY_EXPLORE": "1"},
    }
    assert row["prs_tested"] == []
    assert [call[0] for call in store.calls] == ["get_best_record"]
    assert (tmp_path / "direct-v1" / "files").is_dir()


def test_remote_read_is_wired_into_cli_t0_and_close_write_remains() -> None:
    bootstrap_source = inspect.getsource(
        __import__(
            "hyperloom.inference_optimizer.cli.kb",
            fromlist=["_bootstrap_recipe_kb"],
        )._bootstrap_recipe_kb
    )
    close_source = inspect.getsource(WritebackCollaborator.finalize_recipe_and_journal)
    assert "RemoteWarmRecipeAdapter" in bootstrap_source
    assert "HyperloomRemoteKB.from_env().write" in close_source
    assert "write_final_remote_recipe" not in close_source


def test_vendored_sdk_matches_upstream_git_blob() -> None:
    path = Path(kb_store_client.__file__)
    content = path.read_bytes()
    digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    assert digest == "22d08109fe680cd15c4053ba0ccc5891222eb6cf"


def test_read_alias_is_standalone_api() -> None:
    assert read_remote_champion is read_remote_recipe


# --------------------------------------------------------------- kernel-agent KB


def _kernel_state(tmp_path: Path, *, gain: float = 12.5):
    """A session whose only accepted work is one GEMM tuning KEEP."""
    tuned = tmp_path / "tuned.csv"
    tuned.write_text("M,N,K\n16,512,7168\n", encoding="utf-8")
    return SimpleNamespace(
        optimization_stack=[
            {
                "action": "gemm_tuning",
                "tuned_file": str(tuned),
                "gain_pct": gain,
                "tput": 6638.7,
                "variant_name": "forge_a8w8_blockscale",
            }
        ],
        last_gemm_tuning={"decision": "KEEP", "e2e_gain_pct": gain},
        kernel_opt_task_attempts={},
        current_best={"tput": 6638.7},
        cumulative_gain_validated=84.9,
        session_id="s1",
        recipe_kb_session_id="s1",
    )


def _kernel_client(store):
    """A client for an identity the kernel-agent KB holds nothing for yet.

    The fake store's envelope belongs to a recipe identity, and a read that
    fails is no longer treated as an empty record, so say "nothing published"
    the way the store does: an absent best record.
    """
    client = RemoteRecipeClient.__new__(RemoteRecipeClient)
    client.store = store
    client.read = lambda cid, dest: None  # type: ignore[method-assign]
    return client


def test_kernel_agent_write_publishes_under_the_kernel_metric(tmp_path):
    store = _FakeStore(champion=0.0, metric="kernel_gain_pct")

    result = write_kernel_agent_kb(
        _kernel_state(tmp_path),
        "kernel:hyperloom-m:h:vllm:mt:a:0.1:fp8",
        client=_kernel_client(store),
    )

    assert result.status == "written"
    champion = [c for c in store.calls if c[0] == "set_champion"][0]
    # Graded on kernel gain, under a metric name that is not serving throughput.
    assert champion[3] == "kernel_gain_pct"
    assert champion[4] == 12.5
    assert store.published_knowledge["kernel_gain_pct"] == 12.5


def test_kernel_agent_write_skips_when_it_improves_nothing(tmp_path):
    """Already published, and better: republishing would only churn the record."""
    store = _FakeStore(champion=30.0, metric="kernel_gain_pct")
    client = _kernel_client(store)
    published = {
        "value": {
            "gemm": {
                "optimizations": [
                    {"variant_name": "forge_a8w8_blockscale", "e2e_gain_pct": 30.0}
                ]
            },
            "fusion": {"items": []},
            "rewrite": {"items": []},
        }
    }
    client.read = lambda cid, dest: published  # type: ignore[method-assign]

    result = write_kernel_agent_kb(
        _kernel_state(tmp_path, gain=12.5), "kernel:hyperloom-m:h", client=client
    )

    assert result.status == "skipped"
    assert result.reason == "not_better_than_published"
    assert not [c for c in store.calls if c[0] == "put_knowledge"]


def test_kernel_agent_write_keeps_another_column_it_did_not_touch(tmp_path):
    """A GEMM-only session must not erase a rewrite an earlier run learned."""
    store = _FakeStore(champion=5.0, metric="kernel_gain_pct")
    client = _kernel_client(store)
    published_dir = tmp_path / "published"
    carried = published_dir / "files" / "kernel" / "rewrite" / "k1.py"
    carried.parent.mkdir(parents=True, exist_ok=True)
    carried.write_text("print('earlier rewrite')", encoding="utf-8")
    published = {
        "value": {
            "gemm": {"optimizations": []},
            "fusion": {"items": []},
            "rewrite": {
                "items": [
                    {
                        "kernel_name": "k1",
                        "e2e_gain_pct": 5.0,
                        "patch": "kernel/rewrite/k1.py",
                        "source_files": ["kernel/rewrite/k1.py"],
                    }
                ]
            },
        }
    }

    def _read(cid, dest):
        shutil.copytree(published_dir, dest, dirs_exist_ok=True)
        return published

    client.read = _read  # type: ignore[method-assign]
    # The upload directory is temporary, so record what it held at upload time.
    uploaded_paths: list[str] = []
    inner_put_dir = store.put_dir

    def _put_dir(canonical_id, session_id, files_dir):
        uploaded_paths.extend(
            path.relative_to(files_dir).as_posix()
            for path in sorted(Path(files_dir).rglob("*"))
            if path.is_file()
        )
        return inner_put_dir(canonical_id, session_id, files_dir)

    store.put_dir = _put_dir  # type: ignore[method-assign]

    result = write_kernel_agent_kb(
        _kernel_state(tmp_path, gain=12.5), "kernel:hyperloom-m:h", client=client
    )

    assert result.status == "written"
    value = store.published_knowledge["value"]
    # This session's GEMM lands ...
    assert len(value["gemm"]["optimizations"]) == 1
    # ... and the rewrite it never touched survives, with its artifact re-uploaded.
    assert [item["kernel_name"] for item in value["rewrite"]["items"]] == ["k1"]
    assert "kernel/rewrite/k1.py" in uploaded_paths


def test_kernel_agent_write_skips_a_session_without_kernel_work(tmp_path):
    store = _FakeStore()
    state = SimpleNamespace(
        optimization_stack=[{"action": "replay_warm_recipe", "tput": 100.0}],
        last_gemm_tuning={},
        kernel_opt_task_attempts={},
        current_best={"tput": 100.0},
        cumulative_gain_validated=10.0,
        session_id="s1",
        recipe_kb_session_id="s1",
    )

    result = write_kernel_agent_kb(
        state, "kernel:hyperloom-m:h", client=_kernel_client(store)
    )

    assert result.status == "skipped"
    assert result.reason == "no_kernel_optimization"
    assert store.calls == []


def test_kernel_agent_write_runs_even_when_the_recipe_write_failed(tmp_path):
    """The kernel record is independent: a recipe transport failure must not
    take the kernel agent's knowledge down with it."""
    coordinator = SimpleNamespace(
        shared_state=_kernel_state(tmp_path),
        session_dir=tmp_path,
        recipe_kb=None,
        knowledge_plane=None,
        _ensure_journal=lambda: SimpleNamespace(finalize=lambda **_k: None),
        _workload_canonical_id=lambda: "inference:m:h:vllm:mt:a:0.1:fp8",
    )
    written: list[tuple] = []

    def _fake_write(state, kernel_cid):
        written.append(kernel_cid)
        return RemoteWriteResult("written", "", kernel_cid, "sid", 12.5)

    with mock.patch.dict(
        os.environ,
        {
            "KNOWLEDGE_STORE_MODE": "remote",
            "KB_STORE_URL": "https://kb.example",
            "KB_STORE_TOKEN": "token",
        },
    ), mock.patch.object(
        remote_recipe.HyperloomRemoteKB,
        "from_env",
        classmethod(lambda cls: (_ for _ in ()).throw(OSError("transport down"))),
    ), mock.patch.object(remote_recipe, "write_kernel_agent_kb", _fake_write):
        WritebackCollaborator(coordinator).finalize_recipe_and_journal()

    assert written == ["kernel:hyperloom-m:h:vllm:mt:a:0.1:fp8"]


def _published_kernel_client(store, published: dict, published_dir: Path):
    """A client whose read() serves ``published`` plus its downloaded files."""
    client = RemoteRecipeClient.__new__(RemoteRecipeClient)
    client.store = store

    def _read(cid, dest):
        shutil.copytree(published_dir, dest, dirs_exist_ok=True)
        return published

    client.read = _read  # type: ignore[method-assign]
    return client


def _capture_uploads(store) -> dict[str, bytes]:
    """Record what each write actually uploads; the staging dir is temporary."""
    uploaded: dict[str, bytes] = {}
    inner_put_dir = store.put_dir

    def _put_dir(canonical_id, session_id, files_dir):
        uploaded.clear()
        uploaded.update(
            {
                path.relative_to(files_dir).as_posix(): path.read_bytes()
                for path in sorted(Path(files_dir).rglob("*"))
                if path.is_file()
            }
        )
        return inner_put_dir(canonical_id, session_id, files_dir)

    store.put_dir = _put_dir  # type: ignore[method-assign]
    return uploaded


def test_kernel_agent_write_carries_every_ref_an_inherited_record_declares(tmp_path):
    """A real GEMM record names its report too, not just the tuned table.

    Carrying only a hand-listed subset of ref keys leaves the rest dangling and
    the whole write dies in validation, silently, at the caller's except.
    """
    store = _FakeStore(champion=40.0, metric="kernel_gain_pct")
    published_dir = tmp_path / "published"
    files = published_dir / "files" / "kernel" / "gemm" / "artifacts"
    files.mkdir(parents=True, exist_ok=True)
    (files / "other.csv").write_text("OLD-TUNED-TABLE\n", encoding="utf-8")
    (files / "report.md").write_text("# earlier report\n", encoding="utf-8")
    published = {
        "value": {
            "gemm": {
                "optimizations": [
                    {
                        "variant_name": "other_variant",
                        "e2e_gain_pct": 40.0,
                        "tuned_file": "kernel/gemm/artifacts/other.csv",
                        "final_report_path": "kernel/gemm/artifacts/report.md",
                    }
                ]
            },
            "fusion": {"items": []},
            "rewrite": {"items": []},
        }
    }
    client = _published_kernel_client(store, published, published_dir)
    uploaded = _capture_uploads(store)

    result = write_kernel_agent_kb(
        _kernel_state(tmp_path, gain=12.5), "kernel:hyperloom-m:h", client=client
    )

    assert result.status == "written"
    assert "kernel/gemm/artifacts/report.md" in uploaded


def test_kernel_agent_write_uploads_nothing_the_merged_record_dropped(tmp_path):
    """This session stages before the merge rules on it; a losing record's
    files would otherwise be uploaded with nothing referencing them."""
    store = _FakeStore(champion=40.0, metric="kernel_gain_pct")
    published_dir = tmp_path / "published"
    gemm_files = published_dir / "files" / "kernel" / "gemm" / "artifacts"
    gemm_files.mkdir(parents=True, exist_ok=True)
    (gemm_files / "won.csv").write_text("WINNING-TABLE\n", encoding="utf-8")
    published = {
        "value": {
            # Same variant as the session's, and better: the session's GEMM loses.
            "gemm": {
                "optimizations": [
                    {
                        "variant_name": "forge_a8w8_blockscale",
                        "e2e_gain_pct": 40.0,
                        "tuned_file": "kernel/gemm/artifacts/won.csv",
                    }
                ]
            },
            "fusion": {"items": []},
            "rewrite": {"items": []},
        }
    }
    client = _published_kernel_client(store, published, published_dir)
    uploaded = _capture_uploads(store)
    state = _kernel_state(tmp_path, gain=12.5)
    # A new rewrite keeps the write alive past the "nothing improved" early exit.
    rewrite_patch = tmp_path / "k2.patch"
    rewrite_patch.write_text("--- a/k2.py\n+++ b/k2.py\n", encoding="utf-8")
    rewrite_source = tmp_path / "k2.py"
    rewrite_source.write_text("print('new rewrite')", encoding="utf-8")
    state.optimization_stack.append(
        {
            "action": "integrate",
            "decision": "KEEP",
            "kernel_id": "k2",
            "kernel_name": "k2",
            "gain_pct": 3.0,
            "tput": 6700.0,
            "patch_path": str(rewrite_patch),
            "target_file": str(rewrite_source),
        }
    )

    result = write_kernel_agent_kb(state, "kernel:hyperloom-m:h", client=client)

    assert result.status == "written"
    referenced = extract_knowledge_artifact_refs(store.published_knowledge)
    assert set(uploaded) <= referenced
    # The beaten record's tuned table is not carried along as an orphan.
    assert "kernel/gemm/artifacts/tuned.csv" not in uploaded


def test_kernel_agent_write_gives_a_colliding_inherited_file_its_own_ref(tmp_path):
    """Refs are ``category/kind/<basename>``, so ``tuned.csv`` collides across
    sessions. Resolving an inherited record to this session's bytes would make
    PRELUDE replay the wrong table under someone else's measured gain."""
    store = _FakeStore(champion=40.0, metric="kernel_gain_pct")
    published_dir = tmp_path / "published"
    files = published_dir / "files" / "kernel" / "gemm" / "artifacts"
    files.mkdir(parents=True, exist_ok=True)
    (files / "tuned.csv").write_text("OLD-TUNED-TABLE\n", encoding="utf-8")
    published = {
        "value": {
            "gemm": {
                "optimizations": [
                    {
                        "variant_name": "other_variant",
                        "e2e_gain_pct": 40.0,
                        "tuned_file": "kernel/gemm/artifacts/tuned.csv",
                    }
                ]
            },
            "fusion": {"items": []},
            "rewrite": {"items": []},
        }
    }
    client = _published_kernel_client(store, published, published_dir)
    uploaded = _capture_uploads(store)

    result = write_kernel_agent_kb(
        _kernel_state(tmp_path, gain=12.5), "kernel:hyperloom-m:h", client=client
    )

    assert result.status == "written"
    records = {
        item["variant_name"]: item
        for item in store.published_knowledge["value"]["gemm"]["optimizations"]
    }
    inherited_ref = records["other_variant"]["tuned_file"]
    session_ref = records["forge_a8w8_blockscale"]["tuned_file"]
    assert inherited_ref != session_ref
    # Each record still resolves to the bytes it was measured on.
    assert uploaded[inherited_ref] == b"OLD-TUNED-TABLE\n"
    assert uploaded[session_ref] == b"M,N,K\n16,512,7168\n"


def test_kernel_agent_write_accumulates_under_one_session(tmp_path):
    """Readers resolve an identity through its champion. An accumulating record
    that adds a kernel without raising the best gain scores no higher than the
    incumbent, so parking it on a per-run session would hide it forever."""
    store = _FakeStore(champion=0.0, metric="kernel_gain_pct")

    write_kernel_agent_kb(
        _kernel_state(tmp_path), "kernel:hyperloom-m:h", client=_kernel_client(store)
    )

    sessions = {call[2] for call in store.calls if call[0] == "put_knowledge"}
    assert sessions == {KERNEL_AGENT_SESSION_ID}


def test_kernel_agent_write_accepts_a_refused_promotion_it_already_holds(tmp_path):
    """Re-publishing the accumulated record at an unchanged score need not move
    the champion: it already points at the session being written."""
    store = _FakeStore(champion=12.5, conflict=True, metric="kernel_gain_pct")
    store.champion_session = KERNEL_AGENT_SESSION_ID

    result = write_kernel_agent_kb(
        _kernel_state(tmp_path), "kernel:hyperloom-m:h", client=_kernel_client(store)
    )

    assert result.status == "written"
    # One refused attempt, and no retry: the incumbent is this same record.
    assert len([c for c in store.calls if c[0] == "set_champion"]) == 1


def test_kernel_agent_write_refuses_to_publish_over_an_unreadable_record(tmp_path):
    """The published record is the base of every merge, and the write replaces
    the document wholesale: merging against a failed read would publish this
    run's columns alone and destroy the identity's accumulated kernels."""
    store = _FakeStore(champion=40.0, metric="kernel_gain_pct")
    client = _kernel_client(store)

    def _read(cid, dest):
        raise KBStoreError("GET best-record -> HTTP 500")

    client.read = _read  # type: ignore[method-assign]

    result = write_kernel_agent_kb(
        _kernel_state(tmp_path), "kernel:hyperloom-m:h", client=client
    )

    assert result.status == "error"
    assert "published_record_unreadable" in result.reason
    assert not [c for c in store.calls if c[0] in ("put_knowledge", "put_dir")]
    assert store.published_knowledge is None


def test_kernel_agent_write_displaces_one_ref_without_touching_its_namesake(tmp_path):
    """A record can name two files sharing a basename across directories. Moving
    the colliding one must not repoint the other, whose bytes would then be
    referenced by nothing, dropped from the upload, and replayed as a patch."""
    store = _FakeStore(champion=40.0, metric="kernel_gain_pct")
    published_dir = tmp_path / "published"
    patches = published_dir / "files" / "kernel" / "rewrite" / "patches"
    sources = published_dir / "files" / "kernel" / "rewrite" / "source"
    patches.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)
    (patches / "x.py").write_text("OLD-PATCH", encoding="utf-8")
    (sources / "x.py").write_text("OLD-SOURCE", encoding="utf-8")
    published = {
        "value": {
            "gemm": {"optimizations": []},
            "fusion": {"items": []},
            "rewrite": {
                "items": [
                    {
                        "kernel_name": "kOLD",
                        "e2e_gain_pct": 40.0,
                        "patch": "kernel/rewrite/patches/x.py",
                        "source_files": ["kernel/rewrite/source/x.py"],
                    }
                ]
            },
        }
    }
    client = _published_kernel_client(store, published, published_dir)
    uploaded = _capture_uploads(store)
    state = _kernel_state(tmp_path, gain=12.5)
    # This session's own rewrite collides on the patch basename only.
    session_patch = tmp_path / "x.py"
    session_patch.write_text("NEW-PATCH", encoding="utf-8")
    session_source = tmp_path / "target.py"
    session_source.write_text("print('new')", encoding="utf-8")
    state.optimization_stack.append(
        {
            "action": "integrate",
            "decision": "KEEP",
            "kernel_id": "kNEW",
            "kernel_name": "kNEW",
            "gain_pct": 3.0,
            "tput": 6700.0,
            "patch_path": str(session_patch),
            "target_file": str(session_source),
        }
    )

    result = write_kernel_agent_kb(state, "kernel:hyperloom-m:h", client=client)

    assert result.status == "written"
    inherited = next(
        item
        for item in store.published_knowledge["value"]["rewrite"]["items"]
        if item["kernel_name"] == "kOLD"
    )
    # The displaced patch moved; the same-named source kept its own ref.
    assert inherited["patch"] != "kernel/rewrite/patches/x.py"
    assert inherited["source_files"] == ["kernel/rewrite/source/x.py"]
    assert uploaded[inherited["patch"]] == b"OLD-PATCH"
    assert uploaded["kernel/rewrite/source/x.py"] == b"OLD-SOURCE"


def test_kernel_agent_write_takes_over_an_equally_scored_foreign_champion(tmp_path):
    """Records written before this scheme sit on per-run sessions. Adding a
    kernel that does not raise the best gain leaves the score equal, so without
    an equal-score takeover the accumulated document stays off-champion and no
    reader ever sees it. The store promotes whatever it is told, so a conflict
    is worth retrying at an unchanged score."""
    store = _FakeStore(champion=12.5, conflict=True, metric="kernel_gain_pct")
    store.champion_session = "old-run-session"

    result = write_kernel_agent_kb(
        _kernel_state(tmp_path), "kernel:hyperloom-m:h", client=_kernel_client(store)
    )

    assert result.status == "written"
    promotions = [c for c in store.calls if c[0] == "set_champion"]
    assert len(promotions) == 2  # the refused one, then the takeover
    assert promotions[-1][2] == KERNEL_AGENT_SESSION_ID
    assert store.champion_session == KERNEL_AGENT_SESSION_ID


def test_recipe_write_leaves_an_equally_scored_champion_alone(tmp_path):
    """The recipe record competes rather than accumulates: an equal throughput
    is not a reason to displace a concurrent winner and churn the identity."""
    store = _FakeStore(champion=100.0, conflict=True)
    store.champion_session = "other-session"
    client = RemoteRecipeClient(store)  # type: ignore[arg-type]
    # Gating sees the old champion; the conflict re-read sees a concurrent
    # winner that landed exactly this run's score.
    values = [100.0, 130.0]

    def _rollup(canonical_id):
        store.calls.append(("get_rollup", canonical_id))
        value = values.pop(0) if len(values) > 1 else values[0]
        return {
            "champion": {
                "session_id": "other-session",
                "value": value,
                "metric": "optimized_throughput",
            }
        }

    store.get_rollup = _rollup  # type: ignore[method-assign]

    result = write_final_remote_recipe(
        _state(tmp_path), "inference:m:h:f:mt:a:v:p", "session-1", client=client
    )

    assert result.status == "written"
    # One refused promotion, and no retry over the equal-scored incumbent.
    assert len([c for c in store.calls if c[0] == "set_champion"]) == 1
    assert store.champion_session == "other-session"
