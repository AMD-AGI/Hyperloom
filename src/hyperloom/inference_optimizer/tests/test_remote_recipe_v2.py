# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.knowledge.agent_kb import (
    KernelAgentKB,
    PatchKB,
)
from hyperloom.orchestrator.knowledge.remote_recipe import (
    CURRENT_KNOWLEDGE_SCHEMA_VERSION,
    RECORD_KIND_HYPERLOOM_RECIPE,
    HyperloomRemoteKB,
    KBStoreClient,
    KBStoreError,
    KnowledgeSections,
    RemoteRecipeClient,
    RemoteRecipeConfigurationError,
    RemoteWarmRecipeAdapter as _RemoteWarmRecipeAdapter,
    build_remote_knowledge,
    has_new_keep,
    knowledge_to_warm_recipe,
    read_remote_recipe as _read_remote_recipe,
    write_final_remote_recipe,
)
from hyperloom.orchestrator.knowledge.remote_recipe.values import _Files, merge_staged_sections
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
    RecipeScope,
    RemoteRecipeValidationError,
)
from hyperloom.orchestrator.knowledge.remote_recipe.sanitize import (
    sanitize_publish_env_mapping,
    sanitize_publish_server_args,
    sanitize_shared_knowledge,
)
from hyperloom.orchestrator.knowledge.remote_recipe.values import (
    _Files,
    build_publishable_recipe_config,
    has_replay_material,
)
from hyperloom.orchestrator.loop.writeback import WritebackCollaborator

_DOWNLOAD_BYTES = b"verified artifact"
_DOWNLOAD_SHA256 = hashlib.sha256(_DOWNLOAD_BYTES).hexdigest()
_SCOPE = RecipeScope("forge", 8, 64, 1024, 256)


def read_remote_recipe(*args, **kwargs):
    kwargs.setdefault("scope", _SCOPE)
    return _read_remote_recipe(*args, **kwargs)


class RemoteWarmRecipeAdapter(_RemoteWarmRecipeAdapter):
    def __init__(self, *args, scope=_SCOPE, **kwargs):
        super().__init__(*args, scope=scope, **kwargs)


@pytest.fixture(autouse=True)
def _exported_draft_dir(tmp_path, monkeypatch):
    """Mirror production, where the orchestrator always exports a draft dir."""
    draft = tmp_path / "kb-draft-env"
    draft.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("KB_DRAFT_DIR", str(draft))


def _build(state, files_dir, *, sections=None):
    """Build a bundle, defaulting to an empty draft for tests that stage nothing."""
    files_dir = Path(files_dir)
    if sections is None:
        sections = KnowledgeSections(files_dir.with_name(f"{files_dir.name}-draft"))
    return build_remote_knowledge(state, files_dir, sections=sections)


def _state(tmp_path: Path) -> SimpleNamespace:
    authored_patch = tmp_path / "authored.patch"
    authored_patch.write_text("authored", encoding="utf-8")
    upstream_patch = tmp_path / "upstream.patch"
    upstream_patch.write_text("upstream", encoding="utf-8")
    tuned = tmp_path / "tuned.csv"
    tuned.write_text("M,N,K\n1,2,3\n", encoding="utf-8")
    fusion = tmp_path / "fusion.patch"
    fusion.write_text(
        "diff --git a/source.cu b/source.cu\n"
        "--- a/source.cu\n"
        "+++ b/source.cu\n"
        "@@ -1 +1 @@\n"
        "-// source\n"
        "+// optimized\n",
        encoding="utf-8",
    )
    rewrite = tmp_path / "rewrite.cu"
    rewrite.write_text(
        "diff --git a/source.cu b/source.cu\n"
        "--- a/source.cu\n"
        "+++ b/source.cu\n"
        "@@ -1 +1 @@\n"
        "-// source\n"
        "+// rewrite optimized\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.cu"
    source.write_text("// source", encoding="utf-8")
    return SimpleNamespace(
        session_id="session-1",
        recipe_kb_session_id="session-1",
        kernel_optimizer="native",
        tp=8,
        conc=64,
        isl=1024,
        osl=256,
        baseline_tput=100.0,
        current_best={
            "tput": 130.0,
            "extra_server_args": "--page-size 32",
            "extra_envs": {"FINAL": "1"},
        },
        cumulative_gain_validated=30.0,
        optimization_stack=[
            {
                "action": "explore",
                "source_phase": "EXPLORE",
                "candidate_extra_server_args": "--page-size 32",
                "candidate_extra_envs": {"VLLM_EXPLORE_TEST": "1"},
                "recipe_delta": {
                    "extra_server_args": "--page-size 32",
                    "extra_envs": {"VLLM_EXPLORE_TEST": "1"},
                    "remove_args": [],
                    "unset_envs": [],
                    "args_mode": "append",
                },
                "extra_server_args": "--page-size 32",
                "extra_envs": {"VLLM_EXPLORE_TEST": "1"},
                "patch_path": str(authored_patch),
                "tput": 110.0,
            },
            {
                "action": "integrate_patch",
                "source_phase": "FRAMEWORK_AGENT",
                "candidate_extra_server_args": "--enable-foo",
                "candidate_extra_envs": {"VLLM_FRAMEWORK_TEST": "1"},
                "recipe_delta": {
                    "extra_server_args": "--enable-foo",
                    "extra_envs": {"VLLM_FRAMEWORK_TEST": "1"},
                    "remove_args": [],
                    "unset_envs": [],
                    "args_mode": "append",
                },
                "extra_server_args": "--page-size 32 --enable-foo",
                "extra_envs": {"VLLM_FRAMEWORK_TEST": "1"},
                "patch_path": str(upstream_patch),
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
        gemm_tuning_attempts=[{"status": "failed", "decision": "REVERT", "error": "must not be persisted"}],
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
        last_action_failures=[{"action": "explore", "reason": "OOM"}],
        gaps=[{"description": "attention remains bound"}],
        warm_start_lessons=[{"statement": "use page size 32"}],
        warm_start_pitfalls=[{"description": "page size 1 regresses"}],
    )


def test_build_remote_knowledge_publishes_config_and_kernel_from_state(tmp_path: Path) -> None:
    bundle = _build(_state(tmp_path), tmp_path / "files")

    assert bundle.knowledge["optimized_throughput"] == 130.0
    assert bundle.knowledge["knowledge_schema_version"] == CURRENT_KNOWLEDGE_SCHEMA_VERSION
    assert bundle.knowledge["record_kind"] == RECORD_KIND_HYPERLOOM_RECIPE
    assert bundle.knowledge["validated_e2e_gain"] == 30.0
    value = bundle.knowledge["value"]
    assert sorted(value) == ["config", "kernel", "patch"]
    assert value["config"] == {
        "extra_server_args": "--page-size 32 --enable-foo",
        "extra_envs": {
            "VLLM_EXPLORE_TEST": "1",
            "VLLM_FRAMEWORK_TEST": "1",
        },
    }
    # Overlays reach the record only through the patch column's own staging, so
    # a stack entry naming a local patch file does not publish one by itself.
    assert value["patch"] == {}
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
    assert (tmp_path / "files" / rewrite["experience_document"]).is_file()
    serialized = json.dumps(bundle.knowledge)
    assert "object_id" not in serialized
    assert "bucket" not in serialized
    assert '"files": [' not in serialized


def test_publishable_config_excludes_runtime_and_enablement_bases() -> None:
    state = SimpleNamespace(
        current_best={
            "effective_extra_server_args": "--hld-only --baseline-only",
            "extra_envs": {"SGLANG_ENABLEMENT_ONLY": "1"},
        },
        warm_replay_outcome={},
        optimization_stack=[
            {
                "action": "integrate_patch",
                "baseline_enablement": True,
                "attribution_eligible": False,
                "recipe_delta": {
                    "extra_server_args": "--enablement-only",
                    "extra_envs": {"SGLANG_ENABLEMENT_ONLY": "1"},
                },
            },
            {
                "action": "explore",
                "variant_name": "optimized",
                "candidate_extra_server_args": "--page-size 64",
                "candidate_extra_envs": {"VLLM_OPTIMIZED": "1"},
                "recipe_delta": {
                    "extra_server_args": "--page-size 64",
                    "extra_envs": {"VLLM_OPTIMIZED": "1"},
                    "args_mode": "append",
                },
            },
        ],
    )

    assert build_publishable_recipe_config(state) == {
        "extra_server_args": "--page-size 64",
        "extra_envs": {"VLLM_OPTIMIZED": "1"},
    }


def test_publishable_config_flattens_replayed_base_and_new_delta() -> None:
    state = SimpleNamespace(
        warm_replay_outcome={"status": "reproduced"},
        optimization_stack=[
            {
                "action": "replay_warm_recipe",
                "recipe_delta": {
                    "extra_server_args": "--page-size 32",
                    "extra_envs": {"VLLM_PRIOR": "1"},
                    "args_mode": "replace",
                },
            },
            {
                "action": "explore",
                "variant_name": "new-keep",
                "candidate_extra_server_args": "--enable-foo",
                "candidate_extra_envs": {"VLLM_NEW": "1"},
                "recipe_delta": {
                    "extra_server_args": "--enable-foo",
                    "extra_envs": {"VLLM_NEW": "1"},
                    "args_mode": "append",
                },
            },
        ],
    )

    assert build_publishable_recipe_config(state) == {
        "extra_server_args": "--page-size 32 --enable-foo",
        "extra_envs": {"VLLM_PRIOR": "1", "VLLM_NEW": "1"},
    }


def test_publishable_config_replace_keeps_accumulated_recipe_stack() -> None:
    state = SimpleNamespace(
        warm_replay_outcome={"status": "reproduced"},
        optimization_stack=[
            {
                "action": "replay_warm_recipe",
                "recipe_delta": {
                    "extra_server_args": "--page-size 32",
                    "extra_envs": {"VLLM_PRIOR": "1"},
                    "args_mode": "replace",
                },
            },
            {
                "action": "explore",
                "recipe_delta": {
                    "extra_server_args": "--enable-foo",
                    "extra_envs": {"VLLM_NEW": "1"},
                    "args_mode": "replace",
                },
            },
        ],
    )

    assert build_publishable_recipe_config(state) == {
        "extra_server_args": "--page-size 32 --enable-foo",
        "extra_envs": {"VLLM_PRIOR": "1", "VLLM_NEW": "1"},
    }


def test_publishable_config_applies_remove_and_unset_without_dropping_base() -> None:
    state = SimpleNamespace(
        warm_replay_outcome={},
        optimization_stack=[
            {
                "action": "explore",
                "recipe_delta": {
                    "extra_server_args": "--keep 1 --drop 2",
                    "extra_envs": {"VLLM_KEEP": "1", "VLLM_DROP": "1"},
                    "args_mode": "append",
                },
            },
            {
                "action": "explore",
                "recipe_delta": {
                    "extra_server_args": "--new 3",
                    "extra_envs": {"VLLM_NEW": "1"},
                    "remove_args": ["--drop"],
                    "unset_envs": ["VLLM_DROP"],
                    "args_mode": "append",
                },
            },
        ],
    )

    assert build_publishable_recipe_config(state) == {
        "extra_server_args": "--keep 1 --new 3",
        "extra_envs": {"VLLM_KEEP": "1", "VLLM_NEW": "1"},
    }


def test_publishable_config_rejects_unstructured_framework_args_env() -> None:
    state = SimpleNamespace(
        warm_replay_outcome={},
        optimization_stack=[
            {
                "action": "explore",
                "variant_name": "bad-delta",
                "recipe_delta": {
                    "extra_server_args": "",
                    "extra_envs": {"EXTRA_SGLANG_ARGS": "--page-size 64"},
                },
            }
        ],
    )

    with pytest.raises(
        RemoteRecipeValidationError,
        match="extra_server_args, not envs",
    ):
        build_publishable_recipe_config(state)


def test_geak_recipe_keeps_kernel_partition_empty(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.kernel_optimizer = "geak"

    bundle = _build(state, tmp_path / "files-geak")

    assert bundle.knowledge["provenance"]["kernel_optimizer"] == "geak"
    assert bundle.knowledge["workload_shape"]["tp"] == 8
    assert bundle.knowledge["value"]["kernel"] == {
        "gemm": {"optimizations": []},
        "fusion": {"items": []},
        "rewrite": {"items": []},
    }
    assert not any(item.path.startswith("kernel/") for item in bundle.artifacts)


def test_fusion_writer_accepts_multi_file_patch(tmp_path: Path) -> None:
    state = _state(tmp_path)
    patch = Path(state.last_fusion["patch"])
    patch.write_text(
        "diff --git a/source.cu b/source.cu\n"
        "--- a/source.cu\n"
        "+++ b/source.cu\n"
        "@@ -1 +1 @@\n"
        "-// source\n"
        "+// optimized\n"
        "diff --git a/source_fused_ops.cu b/source_fused_ops.cu\n"
        "--- /dev/null\n"
        "+++ b/source_fused_ops.cu\n"
        "@@ -0,0 +1 @@\n"
        "+// new fused ops\n",
        encoding="utf-8",
    )

    bundle = _build(state, tmp_path / "files-multi-fusion")

    fusion = bundle.knowledge["value"]["kernel"]["fusion"]["items"][0]
    assert fusion["patch"].startswith("kernel/fusion/patches/")


def test_kernel_items_record_the_checkout_they_were_applied_into(tmp_path: Path) -> None:
    """Replay places a kernel patch only into its recorded root, so it must be published.

    The root is an absolute host path, which survives publication solely
    because it sits under the sanitizer's host-origin exemption.
    """
    bundle = _build(_state(tmp_path), tmp_path / "files-kernel-roots")

    kernel = bundle.knowledge["value"]["kernel"]
    roots = [item["host_origin"]["apply_root"] for column in ("fusion", "rewrite") for item in kernel[column]["items"]]

    assert roots, "fusion and rewrite each publish an item"
    assert all(root == str(tmp_path) for root in roots)


def test_kernel_fusion_that_cannot_name_its_checkout_is_dropped(tmp_path: Path) -> None:
    """An item that cannot name its checkout degrades to a drop, not an abort.

    Publishing it rootless would poison the combined replay, and raising would
    take config, patch, and the still-rooted kernels down with it. So the fusion
    item is dropped while the rest of the Recipe still publishes -- and because a
    successful build passes the section mismatch guard, the staged fusion patch
    is proven to leave no orphan behind.
    """
    state = _state(tmp_path)
    state.last_fusion.pop("kernel_repo", None)
    state.last_fusion["source_file"] = "source.cu"
    state.last_fusion["target_file"] = "source.cu"
    for row in state.optimization_stack:
        if str(row.get("action") or "").lower() == "fusion":
            row["target_file"] = "source.cu"

    bundle = _build(state, tmp_path / "files-fusion-unrooted")
    kernel = bundle.knowledge["value"]["kernel"]
    assert kernel["fusion"]["items"] == []
    assert "kernel/fusion/patches" not in json.dumps(kernel["fusion"])
    # The rest of the session is unharmed: the rooted rewrite still publishes.
    assert len(kernel["rewrite"]["items"]) == 1


def test_rewrite_writer_accepts_multi_file_patch(tmp_path: Path) -> None:
    state = _state(tmp_path)
    patch = Path(state.optimization_stack[4]["patch_path"])
    patch.write_text(
        "diff --git a/source.cu b/source.cu\n"
        "--- a/source.cu\n"
        "+++ b/source.cu\n"
        "@@ -1 +1 @@\n"
        "-// source\n"
        "+// optimized\n"
        "diff --git a/source_helpers.cu b/source_helpers.cu\n"
        "--- /dev/null\n"
        "+++ b/source_helpers.cu\n"
        "@@ -0,0 +1 @@\n"
        "+// helper\n",
        encoding="utf-8",
    )

    bundle = _build(state, tmp_path / "files-multi-rewrite")

    rewrite = bundle.knowledge["value"]["kernel"]["rewrite"]["items"][0]
    assert rewrite["patch"].startswith("kernel/rewrite/patches/")


def test_remote_recipe_projects_workload_shape_for_donor_gating(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    state.conc = 64
    state.isl = 1024
    state.osl = 256
    bundle = _build(state, tmp_path / "files-shape")

    row = knowledge_to_warm_recipe(
        {
            "canonical_id": "inference:m:h:f:mt:a:v:p",
            "session_id": "session-1",
            "schema_version": 2,
            "knowledge": bundle.knowledge,
            "view": {"replayable": True},
        }
    )

    assert bundle.knowledge["workload_shape"] == {
        "tp": 8,
        "conc": 64,
        "isl": 1024,
        "osl": 256,
    }
    assert {key: row[key] for key in ("tp", "conc", "isl", "osl")} == {
        "tp": 8,
        "conc": 64,
        "isl": 1024,
        "osl": 256,
    }


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
                "config": {
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
                        "note": ("failed at /home/operator/session/log.txt with TOKEN=secret"),
                    }
                },
            }
        }
    )

    config = sanitized["value"]["config"]
    assert config == {
        "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
        "extra_server_args": "--page-size 32",
    }
    rewrite = sanitized["value"]["kernel"]["rewrite"]
    assert "workspace" not in rewrite
    assert "api_token" not in rewrite
    assert "[LOCAL_PATH]" in rewrite["note"]
    assert "secret" not in rewrite["note"]


def test_host_origin_is_the_one_subtree_that_keeps_absolute_paths() -> None:
    """A KEEP that cannot name its checkout cannot be replayed on another layout."""
    sanitized = sanitize_shared_knowledge(
        {
            "value": {
                "patch": {
                    "patches": ["patch/overlays/000000/00-fix.patch"],
                    "provenance": [
                        {
                            "stack_index": 0,
                            "host_origin": {
                                "framework_root": "/sglang",
                                "snapshot": "/session/optimization_stack/src/spec-1",
                                "manifest": "/session/optimization_stack/src/spec-1/manifest.json",
                                "patches": ["/session/optimization_stack/src/spec-1/realized.patch"],
                                "HF_TOKEN": "secret",
                            },
                        }
                    ],
                },
                "kernel": {"gemm": {"workspace": "/workspace/session"}},
            }
        }
    )

    origin = sanitized["value"]["patch"]["provenance"][0]["host_origin"]
    assert origin["framework_root"] == "/sglang"
    assert origin["snapshot"] == "/session/optimization_stack/src/spec-1"
    assert origin["patches"] == ["/session/optimization_stack/src/spec-1/realized.patch"]
    # The exemption is about absolute paths only.
    assert "HF_TOKEN" not in origin
    # Everywhere else a host path is still a leak.
    assert sanitized["value"]["kernel"]["gemm"] == {}


def test_empty_phase_sections_do_not_copy_cumulative_current_best(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.optimization_stack = [row for row in state.optimization_stack if row.get("source_phase") == "EXPLORE"]
    bundle = _build(state, tmp_path / "files-config-only")
    value = bundle.knowledge["value"]
    assert value["config"]["extra_server_args"] == "--page-size 32"
    assert value["patch"] == {}


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

    bundle = _build(state, tmp_path / "files-geak")
    kernel = bundle.knowledge["value"]["kernel"]
    assert set(kernel) == {"gemm", "fusion", "rewrite"}
    assert all(not artifact.path.startswith("kernel/geak/") for artifact in bundle.artifacts)


def test_micro_keep_without_integrate_stack_is_not_written(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.optimization_stack = [row for row in state.optimization_stack if row.get("action") != "integrate"]
    bundle = _build(state, tmp_path / "files-no-integrate")
    assert bundle.knowledge["value"]["kernel"]["rewrite"]["items"] == []


@pytest.mark.parametrize(
    ("warm_replay_outcome", "adopted"),
    [
        (
            {
                "status": "reproduced",
                "kernel": {"status": "kept", "kept": 1},
            },
            True,
        ),
        (
            {
                "status": "failed",
                "kernel": {"status": "reverted", "kept": 0},
            },
            False,
        ),
    ],
)
def test_prior_kernel_adoption_requires_successful_kernel_replay(
    tmp_path: Path,
    warm_replay_outcome: dict,
    adopted: bool,
) -> None:
    prior_root = tmp_path / "prior"
    prior_ref = "kernel/gemm/artifacts/tuned.csv"
    prior_file = prior_root / "files" / prior_ref
    prior_file.parent.mkdir(parents=True)
    prior_file.write_text("prior\n", encoding="utf-8")
    unreplayed_ref = "kernel/gemm/artifacts/unreplayed.csv"
    (prior_root / "files" / unreplayed_ref).write_text("unreplayed\n", encoding="utf-8")
    (prior_root / "recipe.json").write_text(
        json.dumps(
            {
                "value": {
                    "kernel": {
                        "gemm": {
                            "optimizations": [
                                {"id": "prior-gemm", "tuned_file": prior_ref},
                                {
                                    "id": "unreplayed-gemm",
                                    "tuned_file": unreplayed_ref,
                                },
                            ]
                        },
                        "fusion": {"items": []},
                        "rewrite": {"items": []},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    state = _state(tmp_path)
    state.optimization_stack = [row for row in state.optimization_stack if row.get("source_phase") != "KERNEL_AGENT"]
    state.last_gemm_tuning = {}
    state.last_fusion = {}
    state.last_fusion_integrate = {}
    state.kernel_opt_task_attempts = {}
    state.warm_replay_outcome = warm_replay_outcome
    state.warm_kernel_kb_plan = [
        {
            "column": "gemm",
            "decision": "KEEP",
            "recipe_row": {"id": "prior-gemm", "tuned_file": prior_ref},
        }
    ]
    sections = KnowledgeSections(
        tmp_path / "draft",
        warm_start_dir=prior_root,
    )

    bundle = _build(
        state,
        tmp_path / "files-prior-kernel",
        sections=sections,
    )

    optimizations = bundle.knowledge["value"]["kernel"]["gemm"]["optimizations"]
    if adopted:
        assert optimizations == [{"id": "prior-gemm", "tuned_file": prior_ref}]
        assert {artifact.path for artifact in bundle.artifacts} == {prior_ref}
    else:
        assert optimizations == []
        assert bundle.artifacts == []


def test_prior_kernel_artifact_conflict_is_renamed_and_remapped(
    tmp_path: Path,
) -> None:
    prior_root = tmp_path / "prior-conflict"
    prior_ref = "kernel/gemm/artifacts/tuned.csv"
    prior_file = prior_root / "files" / prior_ref
    prior_file.parent.mkdir(parents=True)
    prior_file.write_text("prior\n", encoding="utf-8")
    (prior_root / "recipe.json").write_text(
        json.dumps(
            {
                "value": {
                    "kernel": {
                        "gemm": {"optimizations": [{"id": "prior-gemm", "tuned_file": prior_ref}]},
                        "fusion": {"items": []},
                        "rewrite": {"items": []},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    state = _state(tmp_path)
    state.warm_replay_outcome = {
        "status": "reproduced",
        "kernel": {"status": "kept", "kept": 1},
    }
    state.warm_kernel_kb_plan = [
        {
            "column": "gemm",
            "decision": "KEEP",
            "recipe_row": {"id": "prior-gemm", "tuned_file": prior_ref},
        }
    ]
    sections = KnowledgeSections(
        tmp_path / "draft-conflict",
        warm_start_dir=prior_root,
    )

    bundle = _build(
        state,
        tmp_path / "files-conflict",
        sections=sections,
    )

    optimizations = bundle.knowledge["value"]["kernel"]["gemm"]["optimizations"]
    refs = [row["tuned_file"] for row in optimizations]
    assert prior_ref in refs
    assert len(refs) == 2
    assert len(set(refs)) == 2
    assert set(refs).issubset({artifact.path for artifact in bundle.artifacts})


def test_micro_keep_with_e2e_revert_but_no_integrate_stack_is_not_written(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    state.optimization_stack = [row for row in state.optimization_stack if row.get("action") != "integrate"]
    state.kernel_integrate_attempts = {
        "rmsnorm": {
            "kernel_id": "rmsnorm",
            "integration_id": "integration-rmsnorm-1",
            "last_decision": "REVERT",
            "best_gain_pct": -2.0,
        }
    }
    bundle = _build(state, tmp_path / "files-revert")
    assert bundle.knowledge["value"]["kernel"]["rewrite"]["items"] == []


def test_integrate_stack_is_authoritative_for_rewrite_patch(tmp_path: Path) -> None:
    state = _state(tmp_path)
    unintegrated_patch = tmp_path / "micro-only.cu"
    unintegrated_patch.write_text("// micro only", encoding="utf-8")
    attempt = state.kernel_opt_task_attempts["rmsnorm"]
    attempt["last_artifact_path"] = str(unintegrated_patch)
    bundle = _build(state, tmp_path / "files-integrated")
    rewrite = bundle.knowledge["value"]["kernel"]["rewrite"]["items"][0]
    assert rewrite["patch"].endswith("/rewrite.cu")
    assert "micro-only.cu" not in json.dumps(rewrite)
    assert rewrite["speedup"] == 1.5


def test_integrated_rewrite_missing_artifact_fails_closed(tmp_path: Path) -> None:
    state = _state(tmp_path)
    integrate = next(row for row in state.optimization_stack if row.get("action") == "integrate")
    integrate["patch_path"] = str(tmp_path / "missing-rewrite.cu")
    state.kernel_opt_task_attempts["rmsnorm"]["last_artifact_path"] = ""
    with pytest.raises(RemoteRecipeValidationError, match="kernel/rewrite"):
        _build(state, tmp_path / "files-missing-rewrite")


def test_accepted_gemm_missing_tuned_file_fails_closed(tmp_path: Path) -> None:
    state = _state(tmp_path)
    gemm = next(row for row in state.optimization_stack if row.get("action") == "gemm_tuning")
    gemm["tuned_file"] = str(tmp_path / "missing-tuned.csv")
    state.last_gemm_tuning["tuned_file"] = gemm["tuned_file"]
    with pytest.raises(RemoteRecipeValidationError, match="kernel/gemm"):
        _build(state, tmp_path / "files-missing-gemm")


def test_accepted_fusion_missing_patch_or_target_fails_closed(tmp_path: Path) -> None:
    state = _state(tmp_path)
    fusion = next(row for row in state.optimization_stack if row.get("action") == "fusion")
    fusion["patch_path"] = str(tmp_path / "missing-fusion.patch")
    state.last_fusion["patch"] = fusion["patch_path"]
    with pytest.raises(RemoteRecipeValidationError, match="kernel/fusion"):
        _build(state, tmp_path / "files-missing-fusion")


def test_bundle_rejects_path_mismatch_and_prefix(tmp_path: Path) -> None:
    source = tmp_path / "a"
    source.write_text("x", encoding="utf-8")
    bundle = KnowledgeBundle({"value": {}}, [Artifact("patch/a", source)])
    with pytest.raises(RemoteRecipeValidationError, match="absent from knowledge"):
        bundle.validate()
    bad = KnowledgeBundle({"value": {}}, [Artifact("files/patch/a", source)])
    with pytest.raises(RemoteRecipeValidationError, match="files/ prefix"):
        bad.validate()
    missing = KnowledgeBundle({"value": {"patch": {"patches": ["patch/missing.patch"]}}})
    with pytest.raises(RemoteRecipeValidationError, match="missing artifacts"):
        missing.validate()


def test_mixed_slash_free_text_is_not_treated_as_an_artifact_ref(
    tmp_path: Path,
) -> None:
    source = tmp_path / "accepted.patch"
    source.write_text("diff", encoding="utf-8")
    files = _Files(tmp_path / "bundle-files")
    ref = files.add(source, category="patch", kind="patches")
    knowledge = {
        "value": {
            "patch": {
                "patches": [ref],
                "note": "see notes at a/b\\c",
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

        def get_hyperloom_recipe_view(self, _canonical_id, *, scope=None):
            return self.envelope

    identity = "inference:m:h:f:mt:a:v:p"
    miss = tmp_path / "miss"
    miss.mkdir()
    (miss / "stale").write_text("old", encoding="utf-8")
    stale_generation = tmp_path / ".miss.generation-abandoned"
    stale_generation.mkdir()
    (stale_generation / "partial").write_text("old", encoding="utf-8")
    assert (
        RemoteRecipeClient(_ReadStore()).read(  # type: ignore[arg-type]
            identity,
            miss,
            scope=_SCOPE,
        )
        is None
    )
    assert not miss.exists()
    assert not stale_generation.exists()


def test_artifact_rejects_symlink_and_oversized_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("x", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(RemoteRecipeValidationError, match="symlink"):
        Artifact("patch/link", link).validate()
    oversized = tmp_path / "oversized"
    with oversized.open("wb") as handle:
        handle.truncate(MAX_FILE_BYTES + 1)
    with pytest.raises(RemoteRecipeValidationError, match="limit"):
        Artifact("patch/oversized", oversized).validate()


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
        kernel_optimizer="native",
        tp=8,
        conc=64,
        isl=1024,
        osl=256,
    )
    read_result = {"canonical_id": "inference:m:h:f:mt:a:v:p"}
    write_result = SimpleNamespace(status="written", session_id=expected)
    calls: list[tuple] = []

    def _read(identity, destination, *, scope, client):
        calls.append(("read", identity, destination, scope, client))
        return read_result

    def _write(state_arg, identity, session_id, *, client):
        calls.append(("write", state_arg, identity, session_id, client))
        return write_result

    monkeypatch.setattr(remote_recipe, "read_remote_recipe", _read)
    monkeypatch.setattr(remote_recipe, "write_final_remote_recipe", _write)

    identity = "inference:m:h:f:mt:a:v:p"
    destination = tmp_path / "download"
    actual_read_result = facade.read(identity, destination, _SCOPE)
    actual_write_result = facade.write(identity, state)
    assert actual_read_result is read_result
    assert actual_write_result is write_result
    assert calls == [
        ("read", identity, destination, _SCOPE, client),
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
        classmethod(lambda cls: (_ for _ in ()).throw(AssertionError("degraded CLOSE constructed HyperloomRemoteKB"))),
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
        classmethod(lambda cls: (_ for _ in ()).throw(AssertionError("local CLOSE constructed HyperloomRemoteKB"))),
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
        json.loads(line) for line in recipe_snapshot_audit_jsonl(tmp_path).read_text(encoding="utf-8").splitlines()
    ]
    assert audit_rows[-1]["status"] == "written"
    assert audit_rows[-1]["generator"] == "close"
    assert audit_rows[-1]["result"]["canonical_id"] == ("inference:m:h:f:mt:a:v:p")


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
        classmethod(lambda cls: (_ for _ in ()).throw(OSError("transport down"))),
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

    row = json.loads(recipe_snapshot_audit_jsonl(tmp_path).read_text(encoding="utf-8"))
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
        self.conflict = conflict
        self.metric = metric
        self.calls: list[tuple] = []
        self.published_knowledge: dict | None = None
        self.uploaded_paths: set[str] = set()
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
            "scope": _SCOPE.as_dict(),
            "view": {
                "source": "current",
                "replayable": True,
                "replay_disabled_reason": None,
            },
            "artifacts": {
                "file_count": 1,
                "files": [
                    {
                        "path": "kernel/rewrite/verified.bin",
                        "size": len(_DOWNLOAD_BYTES),
                        "sha256": _DOWNLOAD_SHA256,
                    }
                ],
            },
            "knowledge": {
                "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
                "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
                "optimized_throughput": 125.0,
                "validated_e2e_gain": 25.0,
                "workload_shape": {
                    "tp": 8,
                    "conc": 64,
                    "isl": 1024,
                    "osl": 256,
                },
                "provenance": {"kernel_optimizer": "forge"},
                "value": {
                    "kernel": {
                        "gemm": {},
                        "fusion": {},
                        "rewrite": {},
                    },
                    "config": {
                        "extra_server_args": "--page-size 32",
                        "extra_envs": {"A": "1"},
                    },
                    "patch": {},
                },
                "lessons": [{"statement": "x"}],
            },
        }

    def get_rollup(self, canonical_id, *, scope=None):
        self.calls.append(("get_rollup", canonical_id, scope))
        champion = {"session_id": "champion-session", "value": self.champion}
        if self.metric is not None:
            champion["metric"] = self.metric
        return {"champion": champion}

    def get_hyperloom_recipe_view(self, canonical_id, *, scope=None):
        self.calls.append(("get_hyperloom_recipe_view", canonical_id, scope))
        return self.envelope

    def put_dir(self, canonical_id, session_id, files_dir):
        self.calls.append(("put_dir", canonical_id, session_id, Path(files_dir)))
        root = Path(files_dir)
        self.uploaded_paths = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        return {
            path.relative_to(root).as_posix(): f"kb://{path.relative_to(root).as_posix()}"
            for path in root.rglob("*")
            if path.is_file()
        }

    def put_knowledge(
        self,
        canonical_id,
        knowledge,
        *,
        session_id="",
        mode="merge",
        scope=None,
    ):
        self.calls.append(("put_knowledge", canonical_id, session_id, mode, scope))
        self.published_knowledge = json.loads(json.dumps(knowledge))

    def set_champion(self, canonical_id, session_id, *, metric, value, scope=None):
        self.calls.append(("set_champion", canonical_id, session_id, metric, value, scope))
        if self.conflict:
            self.conflict = False
            self.champion = value - 1
            raise KBStoreError("POST champion -> HTTP 409: write_conflict")

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


def test_view_rejects_a_different_recipe_scope() -> None:
    store = _FakeStore()
    store.envelope["scope"] = {**_SCOPE.as_dict(), "conc": 128}

    with pytest.raises(RemoteRecipeValidationError, match="requested scope"):
        RemoteRecipeClient(store).get_view(  # type: ignore[arg-type]
            "inference:m:h:f:mt:a:v:p",
            _SCOPE,
        )


def test_view_accepts_scope_dimensions_echoed_as_strings() -> None:
    store = _FakeStore()
    store.envelope["scope"] = {
        key: str(value) if key != "kernel_optimizer" else value for key, value in _SCOPE.as_dict().items()
    }
    store.envelope["knowledge"]["workload_shape"] = {
        key: str(value) for key, value in store.envelope["knowledge"]["workload_shape"].items()
    }

    assert RemoteRecipeClient(store).get_view(  # type: ignore[arg-type]
        "inference:m:h:f:mt:a:v:p",
        _SCOPE,
    )


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
    assert store.calls[2][-2] == "replace"
    assert store.calls[3][-3:-1] == ("optimized_throughput", 130.0)
    assert store.calls[0][-1] == _SCOPE.as_dict()
    assert store.calls[2][-1] == _SCOPE.as_dict()
    assert store.calls[3][-1] == _SCOPE.as_dict()
    assert len([call for call in store.calls if call[0] == "put_knowledge"]) == 1
    assert all(not str(call[1]).startswith("kernel:") for call in store.calls if len(call) > 1)
    assert store.published_knowledge is not None
    assert "kernel" in store.published_knowledge["value"]
    assert result.status == "written"


def test_write_boundary_sanitizes_directly_constructed_bundle(tmp_path: Path) -> None:
    store = _FakeStore()
    bundle = KnowledgeBundle(
        {
            "optimized_throughput": 130.0,
            "value": {
                "config": {
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
        scope=_SCOPE,
        optimized_throughput=130.0,
        files_dir=tmp_path,
    )

    assert result.status == "written"
    assert store.published_knowledge is not None
    published_config = store.published_knowledge["value"]["config"]
    assert published_config == {
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


def test_empty_replay_material_skips_even_when_throughput_beats_champion(
    tmp_path: Path,
) -> None:
    store = _FakeStore(champion=100.0)
    bundle = KnowledgeBundle(
        {
            "knowledge_schema_version": 1,
            "record_kind": "hyperloom_recipe",
            "value": {
                "config": {"extra_server_args": "", "extra_envs": {}},
                "patch": {"patches": []},
                "kernel": {"gemm": {}, "fusion": {}, "rewrite": {}},
            },
        }
    )
    result = RemoteRecipeClient(store).write_if_better(  # type: ignore[arg-type]
        "inference:m:h:f:mt:a:v:p",
        "session-1",
        bundle,
        scope=_SCOPE,
        optimized_throughput=200.0,
        files_dir=tmp_path,
    )
    assert result.status == "skipped"
    assert result.reason == "empty_replay_material"
    assert store.calls == []


def test_provenance_alone_is_not_replay_material() -> None:
    """Provenance describes the overlays; without one there is nothing to replay."""
    assert not has_replay_material(
        {
            "value": {
                "config": {},
                "patch": {"provenance": [{"stack_index": 0, "complete": True}]},
                "kernel": {},
            }
        }
    )
    assert has_replay_material(
        {
            "value": {
                "config": {},
                "patch": {"patches": ["patch/overlays/000000/00-fix.patch"]},
                "kernel": {},
            }
        }
    )


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
        def get_rollup(self, canonical_id, *, scope=None):
            self.calls.append(("get_rollup", canonical_id, scope))
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
        def get_rollup(self, canonical_id, *, scope=None):
            self.calls.append(("get_rollup", canonical_id, scope))
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
        "get_hyperloom_recipe_view",
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
    assert saved["view"] == {
        "source": "current",
        "replayable": True,
        "replay_disabled_reason": None,
    }
    assert not (tmp_path / "bundle" / "values.json").exists()
    assert not (tmp_path / "bundle" / "manifest.json").exists()
    assert (tmp_path / "bundle" / "files").is_dir()
    assert not (tmp_path / "bundle" / "files" / "old.patch").exists()
    assert (tmp_path / "bundle" / "files" / "kernel" / "rewrite" / "verified.bin").read_bytes() == _DOWNLOAD_BYTES
    assert {path.name for path in destination.iterdir()} == {"recipe.json", "files"}


def test_history_only_view_verifies_empty_listing_without_download(
    tmp_path: Path,
) -> None:
    store = _FakeStore()
    store.envelope["view"] = {
        "source": "legacy_gbrain",
        "replayable": False,
        "replay_disabled_reason": "legacy_history_only",
    }
    store.envelope["artifacts"] = {
        "file_count": 0,
        "files": [],
    }
    store.files_listing = {"files": []}
    store.envelope["knowledge"]["what_worked"] = [{"description": "old win"}]

    row = read_remote_recipe(
        "inference:m:h:f:mt:a:v:p",
        tmp_path / "history-only",
        client=RemoteRecipeClient(store),  # type: ignore[arg-type]
    )

    assert row["view"]["replayable"] is False
    assert row["what_worked"] == [{"description": "old win"}]
    assert [call[0] for call in store.calls] == [
        "get_hyperloom_recipe_view",
        "list_session_files",
    ]
    assert (tmp_path / "history-only" / "files").is_dir()


def test_kernel_reads_same_downloaded_inference_recipe_without_second_get(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = _FakeStore()
    store.envelope["knowledge"]["value"]["kernel"] = {
        "rewrite": {
            "items": [
                {
                    "kernel_name": "verified",
                    "patch": "kernel/rewrite/verified.bin",
                }
            ]
        }
    }
    destination = tmp_path / "shared-recipe"
    document = read_remote_recipe(
        "inference:m:h:f:mt:a:v:p",
        destination,
        client=RemoteRecipeClient(store),  # type: ignore[arg-type]
    )
    assert document is not None
    monkeypatch.setenv("KB_DRAFT_DIR", str(tmp_path / "draft"))
    monkeypatch.setenv("KB_WARM_START_DIR", str(destination))

    kernel = KernelAgentKB.open()

    rewrite = kernel.read_rewrite()["items"][0]
    assert rewrite["kernel_name"] == "verified"
    assert rewrite["patch"] == "kernel/rewrite/verified.bin"
    assert kernel.prior_file("kernel/rewrite/verified.bin") is not None
    assert [call[0] for call in store.calls].count("get_hyperloom_recipe_view") == 1
    assert not (tmp_path / "runtime" / "kernel_agent_kb").exists()


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


def test_view_manifest_mismatch_deactivates_without_download(
    tmp_path: Path,
) -> None:
    store = _FakeStore()
    store.files_listing["files"][0]["sha256"] = "0" * 64
    destination = tmp_path / "concurrent-mismatch"
    destination.mkdir()
    (destination / "recipe.json").write_text("stale", encoding="utf-8")

    with pytest.raises(
        RemoteRecipeValidationError,
        match="does not match the selected",
    ):
        read_remote_recipe(
            "inference:m:h:f:mt:a:v:p",
            destination,
            client=RemoteRecipeClient(store),  # type: ignore[arg-type]
        )

    assert not destination.exists()
    assert not any(call[0] == "download_session" for call in store.calls)


def test_read_rejects_non_json_knowledge_and_deactivates_destination(
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

    assert not destination.exists()
    assert [call[0] for call in store.calls] == ["get_hyperloom_recipe_view"]


def test_read_replaces_existing_files_symlink_without_following(tmp_path: Path) -> None:
    store = _FakeStore(champion=125.0)
    destination = tmp_path / "bundle-link"
    target = tmp_path / "outside"
    target.mkdir()
    destination.mkdir()
    (destination / "files").symlink_to(target, target_is_directory=True)
    read_remote_recipe(
        "inference:m:h:f:mt:a:v:p",
        destination,
        client=RemoteRecipeClient(store),  # type: ignore[arg-type]
    )
    assert (destination / "files").is_dir()
    assert not (destination / "files").is_symlink()
    assert list(target.iterdir()) == []


def test_read_replaces_destination_symlink_without_following(tmp_path: Path) -> None:
    store = _FakeStore(champion=125.0)
    target = tmp_path / "destination-target"
    target.mkdir()
    destination = tmp_path / "destination-link"
    destination.symlink_to(target, target_is_directory=True)
    read_remote_recipe(
        "inference:m:h:f:mt:a:v:p",
        destination,
        client=RemoteRecipeClient(store),  # type: ignore[arg-type]
    )
    assert destination.is_dir()
    assert not destination.is_symlink()
    assert list(target.iterdir()) == []


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
        "get_hyperloom_recipe_view",
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


def test_a_staged_file_is_published_under_its_own_section(tmp_path: Path) -> None:
    patch = tmp_path / "authored.patch"
    patch.write_text("authored", encoding="utf-8")
    sections = _sections(tmp_path)
    refs = PatchKB(sections).stage_patches(
        [patch],
        stack_index=2,
    )
    bundle = _build(_state(tmp_path), tmp_path / "files", sections=sections)
    published = {artifact.path for artifact in bundle.artifacts}
    assert refs[0] in published
    assert (tmp_path / "files" / refs[0]).is_file()


def test_merge_staged_sections_unions_and_dedups_prior_refs(tmp_path: Path) -> None:
    """Replay refs already on the section must union with newly staged refs, without duplicates."""
    patch_a = tmp_path / "authored_a.patch"
    patch_b = tmp_path / "authored_b.patch"
    patch_a.write_text("authored_a", encoding="utf-8")
    patch_b.write_text("authored_b", encoding="utf-8")
    sections = _sections(tmp_path)
    staged_refs = FrameworkAgentKB(sections).stage_patches([patch_a, patch_b], stack_index=2)
    staged_ref_a, staged_ref_b = staged_refs[0], staged_refs[1]
    prior_patch = "framework/overlays/000000/00-replayed.patch"
    prior_artifact = "framework/artifacts/prior.bin"
    # before = [prior, staged_ref_a]; after = [staged_ref_a, staged_ref_b].
    # Union must keep prior, dedup the overlap, and append staged_ref_b.
    # `if not before and after:` at values.py (discard after when before is
    # non-empty) silently drops staged_ref_b — this shape turns that red.
    after_patches = list(sections.staged("framework").knowledge.get("patches") or [])
    assert after_patches == [staged_ref_a, staged_ref_b]
    value = {
        "framework": {
            "patches": [prior_patch, staged_ref_a],
            "artifacts": [prior_artifact],
        }
    }
    merged = merge_staged_sections(value, sections, _Files(tmp_path / "files"))
    assert merged == ["framework"]
    assert value["framework"]["patches"] == [prior_patch, staged_ref_a, staged_ref_b]
    assert value["framework"]["artifacts"] == [prior_artifact]


def test_non_overlay_owner_patch_ref_fails_before_publish(
    tmp_path: Path,
) -> None:
    patch = tmp_path / "invalid-owner-ref.patch"
    patch.write_text("invalid", encoding="utf-8")
    sections = _sections(tmp_path)
    sections.write(
        "patch",
        {"patches": ["patch/patches/invalid-owner-ref.patch"]},
        files=[patch],
        kind="patches",
    )

    with pytest.raises(
        RemoteRecipeValidationError,
        match="invalid overlay ref",
    ):
        _build(
            _state(tmp_path),
            tmp_path / "files-invalid-owner-ref",
            sections=sections,
        )


def test_missing_workspace_does_not_scan_current_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "unrelated.diff").write_text("unrelated", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    sources, missing = WritebackCollaborator._keep_patch_sources({}, None)

    assert sources == []
    assert missing == []


def test_orphaned_required_staged_file_fails_close(tmp_path: Path) -> None:
    patch = tmp_path / "orphan.patch"
    patch.write_text("orphan", encoding="utf-8")
    sections = _sections(tmp_path)
    sections.write("patch", {"note": "no ref"}, files=[patch], kind="patches")
    state = _state(tmp_path)
    state.optimization_stack[0]["kb_required_owner"] = "FRAMEWORK_AGENT"

    with pytest.raises(
        RemoteRecipeValidationError,
        match="staged section 'patch' file mismatch",
    ):
        _build(
            state,
            tmp_path / "files-orphan",
            sections=sections,
        )


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
        ("sha", "does not match the selected"),
        ("size", "does not match the selected"),
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
    destination.mkdir()
    (destination / "recipe.json").write_text("stale", encoding="utf-8")
    with pytest.raises(RemoteRecipeValidationError, match=match):
        read_remote_recipe(
            "inference:m:h:f:mt:a:v:p",
            destination,
            client=RemoteRecipeClient(store),  # type: ignore[arg-type]
        )
    assert not destination.exists()


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
            {"files": [{"path": f"f-{index}", "sha256": _DOWNLOAD_SHA256} for index in range(513)]},
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
    assert not destination.exists()
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
def test_bad_envelope_deactivates_destination(
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
    assert not destination.exists()
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
    bundle = _build(state, tmp_path / "finite-knowledge")
    assert bundle.knowledge["optimized_throughput"] == 0.0
    assert bundle.knowledge["validated_e2e_gain"] == 0.0


def _current_knowledge(*, timeline: list[str] | None = None) -> dict:
    return {
        "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
        "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
        "optimized_throughput": 10.0,
        "validated_e2e_gain": 2.0,
        "value": {
            "config": {
                "extra_server_args": "--page-size 32",
                "extra_envs": {"CURRENT": "1"},
            },
            "patch": {"patches": list(timeline or [])},
            "kernel": {"gemm": {}, "fusion": {}, "rewrite": {}},
        },
    }


def test_current_contract_projection_happy_path() -> None:
    row = knowledge_to_warm_recipe(
        {
            "schema_version": 2,
            "canonical_id": "inference:m:h:f:mt:a:v:p",
            "session_id": "s",
            "knowledge": _current_knowledge(),
        }
    )
    assert "best_config" not in row
    assert "patch_timeline" not in row
    assert row["record_kind"] == RECORD_KIND_HYPERLOOM_RECIPE


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("knowledge_schema_version", None, "knowledge_schema_version"),
        ("knowledge_schema_version", 2, "knowledge_schema_version"),
        ("knowledge_schema_version", True, "knowledge_schema_version"),
        ("knowledge_schema_version", "1", "knowledge_schema_version"),
        ("record_kind", None, "record_kind"),
        ("record_kind", "other", "record_kind"),
    ],
)
def test_current_contract_rejects_missing_or_wrong_identity_fields(
    field: str,
    value: object,
    match: str,
) -> None:
    knowledge = _current_knowledge()
    if value is None:
        knowledge.pop(field)
    else:
        knowledge[field] = value
    with pytest.raises(RemoteRecipeValidationError, match=match):
        knowledge_to_warm_recipe(knowledge)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("config", None, "value.config"),
        ("patch", None, "value.patch"),
        ("patch", {"patches": "notalist"}, "flat string list"),
        ("patch", {"patches": [{"patch": "x"}]}, "flat string list"),
        ("kernel", None, "value.kernel"),
    ],
)
def test_current_contract_rejects_missing_required_value_fields(
    field: str,
    value: object,
    match: str,
) -> None:
    knowledge = _current_knowledge()
    if value is None:
        knowledge["value"].pop(field)
    else:
        knowledge["value"][field] = value
    with pytest.raises(RemoteRecipeValidationError, match=match):
        knowledge_to_warm_recipe(knowledge)


def test_current_warm_adapter_keeps_replay_payload_out_of_t0(tmp_path: Path) -> None:
    ref = "patch/overlays/000002/00-pr-7.patch"
    second_ref = "patch/overlays/000003/00-followup.patch"

    class _Remote:
        def read(self, identity: str, destination: Path, scope: RecipeScope):
            for item, content in (
                (ref, "diff --git a/a b/a\n"),
                (second_ref, "diff --git a/b b/b\n"),
            ):
                patch = destination / "files" / item
                patch.parent.mkdir(parents=True, exist_ok=True)
                patch.write_text(content, encoding="utf-8")
            document = {
                "schema_version": 2,
                "canonical_id": identity,
                "session_id": "champion",
                **_current_knowledge(timeline=[ref, second_ref]),
                "view": {
                    "source": "current",
                    "replayable": True,
                    "replay_disabled_reason": None,
                },
            }
            (destination / "recipe.json").write_text(
                json.dumps(document),
                encoding="utf-8",
            )
            return document

    adapter = RemoteWarmRecipeAdapter(  # type: ignore[arg-type]
        _Remote(),
        tmp_path / "remote-recipe",
    )
    row = adapter.get_authoritative_recipe(canonical_id="inference:m:h:f:mt:a:v:p")

    assert "patch_timeline" not in row
    assert "best_config" not in row
    assert "prs_tested" not in row
    assert "required_patch_timeline" not in row
    assert row["replayable"] is True
    assert row["view_source"] == "current"


def test_remote_adapter_pages_history_then_materializes_l2_donor(
    tmp_path: Path,
) -> None:
    from hyperloom.orchestrator.knowledge.recipe_kb_t0 import (
        _cascade_warm_start_search,
    )

    exact = "inference:target:mi300x:sglang:qwen:qwenarch:1.0:fp8"
    unproven = "inference:unproven:mi300x:sglang:qwen:qwenarch:1.0:fp8"
    donor = "inference:donor:mi300x:sglang:qwen:qwenarch:1.0:fp8"
    patch_ref = "patch/overlays/000001/00-donor.patch"

    class _Remote:
        def __init__(self):
            self.search_calls = []
            self.view_calls = []
            self.materialized = []

        def search_identities(self, **kwargs):
            self.search_calls.append(dict(kwargs))
            if kwargs["offset"] == 0:
                return {
                    "items": [
                        {
                            "canonical_id": unproven,
                            "dimensions": {
                                "architectures": "qwenarch",
                                "model_type": "qwen",
                            },
                        }
                    ],
                    "total": 2,
                    "next_offset": 1,
                }
            return {
                "items": [
                    {
                        "canonical_id": donor,
                        "dimensions": {
                            "architectures": "qwenarch",
                            "model_type": "qwen",
                        },
                    }
                ],
                "total": 2,
                "next_offset": None,
            }

        def _document(self, identity: str):
            replayable = identity in {unproven, donor}
            source = "current" if replayable else "legacy_gbrain"
            timeline = [patch_ref] if identity == donor else []
            knowledge = _current_knowledge(timeline=timeline)
            knowledge["validated_e2e_gain"] = 7.0 if identity == donor else 0.0 if replayable else 41.0
            knowledge["what_worked"] = [{"description": f"history:{identity}"}]
            knowledge["value"]["config"] = {
                "extra_server_args": "--donor" if replayable else "",
                "extra_envs": {},
            }
            knowledge["value"]["patch"] = {"patches": timeline}
            return {
                "schema_version": 2,
                "canonical_id": identity,
                "session_id": f"session-{identity.split(':')[1]}",
                **knowledge,
                "view": {
                    "source": source,
                    "replayable": replayable,
                    "replay_disabled_reason": (None if replayable else "legacy_history_only"),
                },
            }

        def get_view(self, identity: str, scope: RecipeScope):
            self.view_calls.append(identity)
            return self._document(identity)

        def _write(self, identity: str, destination: Path, document: dict):
            destination.mkdir(parents=True, exist_ok=True)
            files = destination / "files"
            files.mkdir(parents=True, exist_ok=True)
            if identity == donor:
                patch = files / patch_ref
                patch.parent.mkdir(parents=True, exist_ok=True)
                patch.write_text("donor patch", encoding="utf-8")
            (destination / "recipe.json").write_text(
                json.dumps(document),
                encoding="utf-8",
            )
            return document

        def read(self, identity: str, destination: Path, scope: RecipeScope):
            return self._write(
                identity,
                destination,
                self._document(identity),
            )

        def materialize_view(
            self,
            identity: str,
            destination: Path,
            envelope: dict,
            scope: RecipeScope,
        ):
            self.materialized.append(identity)
            return self._write(identity, destination, envelope)

    remote = _Remote()
    main = tmp_path / "remote-recipe"
    adapter = RemoteWarmRecipeAdapter(remote, main)  # type: ignore[arg-type]

    row, tier, confidence = _cascade_warm_start_search(
        adapter,  # type: ignore[arg-type]
        cid=exact,
        hw="mi300x",
        framework="sglang",
        model_type_val="qwen",
        architectures_val=["QwenArch"],
        arch_slug="qwenarch",
        fw_version="1.0",
        precision="fp8",
        warm_prefer=None,
    )

    assert row["canonical_id"] == donor
    assert row["replayable"] is True
    assert row["what_worked"][0] == {"description": f"history:{donor}"}
    assert row["validated_gain_pct"] == 7.0
    assert row["sessions"][0]["gain_pct"] == 7.0
    assert row["exact_history"]["what_worked"][0] == {"description": f"history:{exact}"}
    assert row["exact_history"]["sessions"][0]["gain_pct"] == 41.0
    assert row["exact_history"]["validated_gain_pct"] == 41.0
    assert tier == "same_arch_class"
    assert confidence == 0.95
    assert [call["offset"] for call in remote.search_calls] == [0, 1]
    assert remote.view_calls == [unproven, donor]
    assert remote.materialized == [donor]
    assert remote.search_calls[0]["match"] == {
        "hardware": "mi300x",
        "framework_name": "sglang",
        "model_type": "qwen",
        "architectures": "qwenarch",
        "framework_version": "1.0",
        "precision": "fp8",
    }
    selected = json.loads((main / "recipe.json").read_text(encoding="utf-8"))
    assert selected["canonical_id"] == donor
    assert (main / "files" / patch_ref).read_text(encoding="utf-8") == "donor patch"
    patch_kb = PatchKB(KnowledgeSections(tmp_path / "draft", warm_start_dir=main))
    assert patch_kb.read_patches() == [patch_ref]
    candidates = main.parent / f".{main.name}-candidates"
    assert not candidates.exists()


def test_remote_adapter_forwards_hardware_in(tmp_path: Path) -> None:
    class _Remote:
        def __init__(self) -> None:
            self.kwargs = {}

        def search_identities(self, **kwargs):
            self.kwargs = dict(kwargs)
            return {"items": [], "total": 0, "next_offset": None}

    remote = _Remote()
    adapter = RemoteWarmRecipeAdapter(  # type: ignore[arg-type]
        remote,
        tmp_path / "unused",
    )

    assert (
        adapter.search(
            label_match={"framework": "sglang"},
            hardware_in=["mi300x", "mi325x"],
        )
        == []
    )
    assert remote.kwargs["hardware_in"] == ["mi300x", "mi325x"]
    assert remote.kwargs["match"] == {"framework_name": "sglang"}


def test_remote_adapter_stops_on_empty_page_with_next_offset(
    tmp_path: Path,
) -> None:
    class _Remote:
        def __init__(self) -> None:
            self.calls = 0

        def search_identities(self, **_kwargs):
            self.calls += 1
            return {"items": [], "next_offset": self.calls * 100}

    remote = _Remote()
    adapter = RemoteWarmRecipeAdapter(  # type: ignore[arg-type]
        remote,
        tmp_path / "empty-page",
    )

    assert adapter.search(label_match={}, limit=100) == []
    assert remote.calls == 1


def test_remote_adapter_caps_metadata_scan_without_downloading(
    tmp_path: Path,
) -> None:
    identities = [f"inference:model-{index}:mi300x:sglang:qwen:qwenarch:1.0:bf16" for index in range(6)]

    class _Remote:
        def __init__(self) -> None:
            self.view_calls: list[str] = []

        def search_identities(self, **_kwargs):
            return {
                "items": [{"canonical_id": identity, "dimensions": {}} for identity in identities],
                "total": len(identities),
                "next_offset": None,
            }

        def get_view(self, identity: str, scope: RecipeScope):
            self.view_calls.append(identity)
            return {
                "schema_version": 2,
                "canonical_id": identity,
                "session_id": f"session-{identity}",
                **_current_knowledge(),
                "view": {
                    "source": "current",
                    "replayable": True,
                    "replay_disabled_reason": None,
                },
            }

        def read(self, *_args, **_kwargs):
            raise AssertionError("candidate metadata scan downloaded a bundle")

    remote = _Remote()
    destination = tmp_path / "warm"
    stale_candidates = tmp_path / ".warm-candidates"
    stale_candidates.mkdir()
    (stale_candidates / "stale").write_text("old", encoding="utf-8")
    adapter = RemoteWarmRecipeAdapter(remote, destination)  # type: ignore[arg-type]
    adapter.search_candidate_cap = 3

    rows = adapter.search(label_match={"framework": "sglang"}, limit=100)

    assert len(rows) == 3
    assert remote.view_calls == identities[:3]
    assert not stale_candidates.exists()
    assert not destination.exists()


def test_remote_adapter_detects_kernel_only_replay_material(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    knowledge = _current_knowledge()
    knowledge["value"]["config"] = {}
    knowledge["value"]["patch"] = {}
    knowledge["value"]["kernel"]["rewrite"] = {"items": [{"kernel_name": "fused"}]}
    (candidate / "recipe.json").write_text(
        json.dumps(knowledge),
        encoding="utf-8",
    )

    assert RemoteWarmRecipeAdapter._candidate_has_replay_material(candidate)

    knowledge["value"]["kernel"]["rewrite"] = {}
    (candidate / "recipe.json").write_text(
        json.dumps(knowledge),
        encoding="utf-8",
    )
    assert not RemoteWarmRecipeAdapter._candidate_has_replay_material(candidate)


def test_patch_column_returns_the_recorded_replay_order(tmp_path: Path) -> None:
    timeline = [
        "patch/overlays/000002/00-pr-7.patch",
        "patch/overlays/000003/00-followup.patch",
    ]
    warm = tmp_path / "warm"
    warm.mkdir()
    (warm / "recipe.json").write_text(
        json.dumps(_current_knowledge(timeline=timeline)),
        encoding="utf-8",
    )
    kb = PatchKB(KnowledgeSections(tmp_path / "draft", warm_start_dir=warm))

    assert kb.read_patches() == timeline


@pytest.mark.parametrize(
    "ref",
    ["/absolute.patch", "../escape.patch", "framework/../../escape.patch"],
)
def test_current_warm_adapter_rejects_unsafe_timeline_refs(
    tmp_path: Path,
    ref: str,
) -> None:
    class _Remote:
        def read(self, identity: str, destination: Path, scope: RecipeScope):
            return {
                "canonical_id": identity,
                **_current_knowledge(timeline=[ref]),
            }

    adapter = RemoteWarmRecipeAdapter(  # type: ignore[arg-type]
        _Remote(),
        tmp_path / "unsafe",
    )

    with pytest.raises(RemoteRecipeValidationError):
        adapter.get_authoritative_recipe(canonical_id="inference:test")


def test_owner_sdk_rejects_symlink_timeline_ref(
    tmp_path: Path,
) -> None:
    ref = "patch/overlays/000000/00-link.patch"

    warm = tmp_path / "symlink"
    outside = tmp_path / "outside.patch"
    outside.write_text("secret", encoding="utf-8")
    link = warm / "files" / ref
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    sections = KnowledgeSections(tmp_path / "draft", warm_start_dir=warm)

    assert PatchKB(sections).prior_file(ref) is None


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


def test_obsolete_remote_contract_exports_are_removed() -> None:
    from hyperloom.orchestrator.knowledge import remote_recipe

    for name in (
        "convert_v1_recipe_to_knowledge",
        "envelope_to_v1_recipe",
        "read_remote_champion",
    ):
        assert not hasattr(remote_recipe, name)
    assert not hasattr(PatchKB, "write_snapshot")
    # patch_roots could never survive publication, so the surface is gone.
    assert not hasattr(PatchKB, "read_patch_roots")
    assert not hasattr(RemoteRecipeClient, "read_champion")


def test_vendored_sdk_exposes_view_and_identity_search() -> None:
    assert callable(KBStoreClient.get_hyperloom_recipe_view)
    assert callable(KBStoreClient.search_identities)
    view_source = inspect.getsource(RemoteRecipeClient.get_view)
    assert "get_hyperloom_recipe_view" in view_source
    assert "get_best_record" not in view_source


def test_vendored_sdk_matches_upstream_git_blob() -> None:
    path = Path(kb_store_client.__file__)
    content = path.read_bytes()
    digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    assert digest == "3402092b4cac1e85e9ad9baae77b8b0020259158"


def test_vendored_sdk_uses_new_view_and_search_routes() -> None:
    client = KBStoreClient.__new__(KBStoreClient)
    calls = []

    def _request(method, path, payload=None):
        calls.append((method, path, payload))
        return {"items": [], "total": 0, "next_offset": None}

    client._request = _request  # type: ignore[method-assign]
    scope = _SCOPE.as_dict()
    assert client.get_hyperloom_recipe_view("inference:m:h", scope=scope) == {
        "items": [],
        "total": 0,
        "next_offset": None,
    }
    client.search_identities(
        scheme="inference",
        match={"framework_name": "sglang"},
        hardware_in=["mi300x"],
        offset=10,
        limit=5,
    )
    client.get_rollup("inference:m:h", scope=scope)
    client.put_knowledge(
        "inference:m:h",
        {"value": {}},
        session_id="session-1",
        mode="replace",
        scope=scope,
    )
    client.set_champion(
        "inference:m:h",
        "session-1",
        metric="optimized_throughput",
        value=10.0,
        scope=scope,
    )

    assert calls == [
        (
            "GET",
            "/v1/kb/inference:m:h/views/hyperloom-recipe?kernel_optimizer=forge&tp=8&conc=64&isl=1024&osl=256",
            None,
        ),
        (
            "POST",
            "/v1/kb/search",
            {
                "scheme": "inference",
                "match": {"framework_name": "sglang"},
                "offset": 10,
                "limit": 5,
                "hardware_in": ["mi300x"],
            },
        ),
        (
            "GET",
            "/v1/kb/inference:m:h/sessions?kernel_optimizer=forge&tp=8&conc=64&isl=1024&osl=256",
            None,
        ),
        (
            "POST",
            "/v1/kb/inference:m:h",
            {
                "knowledge": {"value": {}},
                "mode": "replace",
                "session_id": "session-1",
                "scope": scope,
            },
        ),
        (
            "POST",
            "/v1/kb/inference:m:h/champion",
            {
                "session_id": "session-1",
                "metric": "optimized_throughput",
                "value": 10.0,
                "scope": scope,
            },
        ),
    ]
