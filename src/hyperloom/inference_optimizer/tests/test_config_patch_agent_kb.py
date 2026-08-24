# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.knowledge.agent_kb import (
    ExploreAgentKB,
    FrameworkAgentKB,
    RecipeReplayKB,
)
from hyperloom.orchestrator.knowledge.remote_recipe._vendor.kb_store_client import (
    KnowledgeSections,
)
from hyperloom.orchestrator.knowledge.remote_recipe.values import (
    CURRENT_KNOWLEDGE_SCHEMA_VERSION,
    RECORD_KIND_HYPERLOOM_RECIPE,
    RemoteRecipeValidationError,
    _Files,
    build_remote_knowledge,
    knowledge_to_warm_recipe,
)
from hyperloom.orchestrator.phases.prelude import _merge_current_recipe_configs


def _state(stack: list[dict]) -> SimpleNamespace:
    normalized_stack = []
    for raw in stack:
        row = dict(raw)
        if (
            str(row.get("action") or "").lower() == "replay_warm_recipe"
            and "recipe_delta" not in row
        ):
            row["recipe_delta"] = {
                "extra_server_args": str(
                    row.get("candidate_extra_server_args")
                    or row.get("extra_server_args")
                    or ""
                ),
                "extra_envs": dict(row.get("extra_envs") or {}),
                "remove_args": [],
                "unset_envs": [],
                "args_mode": "replace",
            }
        normalized_stack.append(row)
    return SimpleNamespace(
        optimization_stack=normalized_stack,
        current_best={"tput": 130.0},
        cumulative_gain_validated=30.0,
        gain_per_stack_entry=[],
        session_id="s1",
        recipe_kb_session_id="s1",
        warm_replay_outcome={},
        kernel_optimizer="native",
        tp=8,
        conc=64,
        isl=1024,
        osl=256,
    )


def test_no_draft_is_a_noop(monkeypatch) -> None:
    monkeypatch.delenv("KB_DRAFT_DIR", raising=False)
    monkeypatch.delenv("KB_WARM_START_DIR", raising=False)

    kb = ExploreAgentKB.open()

    assert kb.active is False
    assert kb.read() == {}
    assert kb.read_config() == {"extra_server_args": "", "extra_envs": {}}
    assert kb.stage_patches([], stack_index=0) == []


def test_recipe_replay_reads_single_final_config(tmp_path: Path) -> None:
    warm = tmp_path / "warm"
    warm.mkdir()
    (warm / "recipe.json").write_text(
        json.dumps(
            {
                "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
                "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
                "value": {
                    "config": {
                        "extra_server_args": "--page-size 32",
                        "extra_envs": {"VLLM_TEST": "1"},
                    },
                    "patch_timeline": [],
                },
            }
        ),
        encoding="utf-8",
    )
    replay = RecipeReplayKB(
        KnowledgeSections(tmp_path / "draft", warm_start_dir=warm)
    )

    assert replay.read_config() == {
        "extra_server_args": "--page-size 32",
        "extra_envs": {"VLLM_TEST": "1"},
    }


def test_read_and_section_isolation(tmp_path: Path) -> None:
    warm = tmp_path / "warm"
    warm.mkdir()
    (warm / "recipe.json").write_text(
        json.dumps(
            {
                "value": {
                    "explore": {
                        "extra_server_args": "--prior",
                        "extra_envs": {"VLLM_PRIOR": "1"},
                        "patches": ["explore/overlays/000000/00-prior.patch"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    sections = KnowledgeSections(tmp_path / "draft", warm_start_dir=warm)
    explore = ExploreAgentKB(sections)
    framework = FrameworkAgentKB(sections)

    assert explore.read_config() == {
        "extra_server_args": "--prior",
        "extra_envs": {"VLLM_PRIOR": "1"},
    }
    assert explore.read_patches() == [
        "explore/overlays/000000/00-prior.patch"
    ]
    assert framework.read() == {}
    assert framework.read_patches() == []


def test_ordered_patch_refs_are_deterministic_and_idempotent(tmp_path: Path) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    kb = FrameworkAgentKB(sections)
    first = tmp_path / "fix one.diff"
    second = tmp_path / "tune.patch"
    first.write_bytes(b"diff --git a/a b/a\n")
    second.write_bytes(b"diff --git a/b b/b\n")

    refs = kb.stage_patches([first, second], stack_index=7)
    again = kb.stage_patches([first, second], stack_index=7)

    assert refs == again == [
        "framework/overlays/000007/00-fix-one.patch",
        "framework/overlays/000007/01-tune.patch",
    ]
    staged = sections.staged("framework")
    assert staged.knowledge["patches"] == refs
    assert [path.relative_to(sections.files_dir).as_posix() for path in staged.files] == refs
    assert (sections.files_dir / refs[0]).read_bytes() == first.read_bytes()


def test_same_ref_cannot_be_replaced_with_different_bytes(tmp_path: Path) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    kb = ExploreAgentKB(sections)
    first = tmp_path / "a" / "same.patch"
    second = tmp_path / "b" / "same.patch"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    assert kb.stage_patches([first], stack_index=2)
    assert kb.stage_patches([second], stack_index=2) == []
    ref = "explore/overlays/000002/00-same.patch"
    assert (sections.files_dir / ref).read_bytes() == b"first"


def test_multi_patch_staging_is_atomic_on_unreadable_member(
    tmp_path: Path,
) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    kb = FrameworkAgentKB(sections)
    prior = tmp_path / "prior.patch"
    first = tmp_path / "first.patch"
    missing = tmp_path / "missing.patch"
    prior.write_bytes(b"prior")
    first.write_bytes(b"first")
    assert kb.stage_patches([prior], stack_index=2)
    section_file = sections.root / "sections" / "framework.json"
    before_section = section_file.read_bytes()
    before_files = sorted(
        path.relative_to(sections.files_dir)
        for path in sections.files_dir.rglob("*")
        if path.is_file()
    )

    assert kb.stage_patches([first, missing], stack_index=3) == []
    assert section_file.read_bytes() == before_section
    assert sorted(
        path.relative_to(sections.files_dir)
        for path in sections.files_dir.rglob("*")
        if path.is_file()
    ) == before_files


def test_patch_staging_rejects_more_than_100_members_atomically(
    tmp_path: Path,
    caplog,
) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    kb = ExploreAgentKB(sections)
    patches: list[Path] = []
    for index in range(101):
        patch = tmp_path / f"{index}.patch"
        patch.write_bytes(str(index).encode())
        patches.append(patch)

    with caplog.at_level("WARNING"):
        assert kb.stage_patches(patches, stack_index=0) == []
    assert "maximum is 100" in caplog.text
    assert sections.staged("explore") is None
    assert not any(
        path.is_file() for path in sections.files_dir.rglob("*")
    )


def test_close_timeline_orders_patches_across_owners(tmp_path: Path) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    explore = ExploreAgentKB(sections)
    framework = FrameworkAgentKB(sections)
    p0 = tmp_path / "framework.diff"
    p1 = tmp_path / "explore.diff"
    p0.write_bytes(b"framework")
    p1.write_bytes(b"explore")
    framework.stage_patches([p0], stack_index=0)
    explore.stage_patches([p1], stack_index=1)

    bundle = build_remote_knowledge(
        _state(
            [
                {"action": "framework", "source_phase": "FRAMEWORK_AGENT"},
                {"action": "integrate_patch", "source_phase": "EXPLORE"},
            ]
        ),
        tmp_path / "files",
        sections=sections,
    )

    assert (
        bundle.knowledge["knowledge_schema_version"]
        == CURRENT_KNOWLEDGE_SCHEMA_VERSION
    )
    assert bundle.knowledge["record_kind"] == RECORD_KIND_HYPERLOOM_RECIPE
    timeline = bundle.knowledge["value"]["patch_timeline"]
    assert timeline == [
        "framework/overlays/000000/00-framework.patch",
        "explore/overlays/000001/00-explore.patch",
    ]
    artifact_paths = {artifact.path for artifact in bundle.artifacts}
    assert set(timeline) <= artifact_paths
    bundle.validate()


def test_current_contract_persists_one_final_config(
    tmp_path: Path,
) -> None:
    state = _state(
        [
            {
                "action": "explore",
                "source_phase": "EXPLORE",
                "recipe_delta": {
                    "extra_server_args": "--explore-old",
                    "extra_envs": {"VLLM_OWNER": "explore"},
                    "args_mode": "append",
                },
                "extra_server_args": "--explore-old",
                "extra_envs": {"VLLM_OWNER": "explore"},
            },
            {
                "action": "framework",
                "source_phase": "FRAMEWORK_AGENT",
                "recipe_delta": {
                    "extra_server_args": "--framework-final",
                    "extra_envs": {"VLLM_OWNER": "framework"},
                    "args_mode": "replace",
                },
                "extra_server_args": "--framework-final",
                "extra_envs": {"VLLM_OWNER": "framework"},
            },
        ]
    )
    state.current_best = {
        "tput": 150.0,
        "extra_server_args": "--framework-final",
        "extra_envs": {"VLLM_OWNER": "framework"},
    }
    bundle = build_remote_knowledge(
        state,
        tmp_path / "files",
        sections=KnowledgeSections(tmp_path / "draft"),
    )

    assert "replay_config" not in bundle.knowledge["value"]
    assert bundle.knowledge["value"]["config"] == {
        "extra_server_args": "--framework-final",
        "extra_envs": {"VLLM_OWNER": "framework"},
    }
    assert set(bundle.knowledge["value"]["explore"]) == {"patches", "artifacts"}
    assert set(bundle.knowledge["value"]["framework"]) == {"patches", "artifacts"}
    row = knowledge_to_warm_recipe(
        {
            "canonical_id": "inference:test",
            "knowledge": bundle.knowledge,
        }
    )
    assert "best_config" not in row
    assert "patch_timeline" not in row


def test_current_owner_config_merge_dedupes_identical_values() -> None:
    args, envs = _merge_current_recipe_configs(
        {
            "extra_server_args": "--page-size 32 --shared",
            "extra_envs": {"SHARED": "1", "EXPLORE": "1"},
        },
        {
            "extra_server_args": "--shared --framework",
            "extra_envs": {"SHARED": "1", "FRAMEWORK": "1"},
        },
        {
            "extra_server_args": "--shared --kernel",
            "extra_envs": {"SHARED": "1", "KERNEL": "1"},
        },
    )
    assert args == "--page-size 32 --shared --framework --kernel"
    assert envs == {
        "SHARED": "1",
        "EXPLORE": "1",
        "FRAMEWORK": "1",
        "KERNEL": "1",
    }


@pytest.mark.parametrize(
    ("explore", "framework", "match"),
    [
        (
            {"extra_server_args": "--page-size 32"},
            {"extra_server_args": "--page-size 64"},
            "--page-size",
        ),
        (
            {"extra_envs": {"SHARED": "1"}},
            {"extra_envs": {"SHARED": "2"}},
            "SHARED",
        ),
    ],
)
def test_current_owner_config_merge_fails_closed_on_conflict(
    explore: dict,
    framework: dict,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _merge_current_recipe_configs(explore, framework)


@pytest.mark.parametrize(
    "kernel",
    [
        {"extra_server_args": "--page-size 64"},
        {"extra_envs": {"SHARED": "kernel"}},
    ],
)
def test_current_kernel_config_merge_fails_closed_on_conflict(
    kernel: dict,
) -> None:
    with pytest.raises(ValueError, match="conflict"):
        _merge_current_recipe_configs(
            {
                "extra_server_args": "--page-size 32",
                "extra_envs": {"SHARED": "recipe"},
            },
            {},
            kernel,
        )


def test_close_drops_permanently_missing_owner_section(tmp_path: Path) -> None:
    state = _state([])
    state.kb_stage_outbox = [
        {
            "id": "FRAMEWORK_AGENT:0",
            "owner": "FRAMEWORK_AGENT",
            "stack_index": 0,
            "missing_patch_sources": ["missing.patch"],
        }
    ]

    bundle = build_remote_knowledge(state, tmp_path / "files")

    assert bundle.knowledge["provenance"]["dropped_staged_sections"] == [
        "FRAMEWORK_AGENT:0"
    ]


def test_close_fails_with_transient_incomplete_required_section(
    tmp_path: Path,
) -> None:
    state = _state([])
    state.kb_stage_outbox = [{"id": "FRAMEWORK_AGENT:0"}]

    with pytest.raises(
        RemoteRecipeValidationError,
        match="required section staging is incomplete",
    ):
        build_remote_knowledge(state, tmp_path / "files")


@pytest.mark.parametrize("section", ["explore", "framework"])
def test_close_fails_for_required_staged_section_with_missing_file(
    tmp_path: Path,
    section: str,
) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    section_file = sections.root / "sections" / f"{section}.json"
    section_file.parent.mkdir(parents=True)
    ref = f"{section}/overlays/000000/00-missing.patch"
    section_file.write_text(
        json.dumps(
            {
                "knowledge": {"patches": [ref]},
                "files": [ref],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RemoteRecipeValidationError,
        match=f"staged section '{section}' file mismatch",
    ):
        build_remote_knowledge(
            _state(
                [
                    {
                        "action": "framework",
                        "kb_required_owner": (
                            "EXPLORE"
                            if section == "explore"
                            else "FRAMEWORK_AGENT"
                        ),
                    }
                ]
            ),
            tmp_path / "files",
            sections=sections,
        )


def test_close_fails_for_empty_required_staged_section(
    tmp_path: Path,
) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    section_file = sections.root / "sections" / "explore.json"
    section_file.parent.mkdir(parents=True)
    section_file.write_text(
        json.dumps({"knowledge": {}, "files": []}),
        encoding="utf-8",
    )

    with pytest.raises(
        RemoteRecipeValidationError,
        match="required staged section 'explore' is empty",
    ):
        build_remote_knowledge(
            _state(
                [
                    {
                        "action": "explore",
                        "kb_required_owner": "EXPLORE",
                    }
                ]
            ),
            tmp_path / "files",
            sections=sections,
        )


@pytest.mark.parametrize(
    ("owner", "expected"),
    [
        ("EXPLORE", "explore"),
        ("FRAMEWORK_AGENT", "framework"),
    ],
)
def test_close_fails_when_required_owner_section_is_missing(
    tmp_path: Path,
    owner: str,
    expected: str,
) -> None:
    state = _state(
        [{"action": "integrate_patch", "kb_required_owner": owner}]
    )

    with pytest.raises(
        RemoteRecipeValidationError,
        match=f"'{expected}'",
    ):
        build_remote_knowledge(
            state,
            tmp_path / "files",
            sections=KnowledgeSections(tmp_path / "draft"),
        )


def test_close_adopts_only_successfully_replayed_prior_overlays(
    tmp_path: Path,
) -> None:
    old_ref = "framework/overlays/000004/00-upstream.patch"
    warm = tmp_path / "warm"
    old_patch = warm / "files" / old_ref
    old_patch.parent.mkdir(parents=True)
    old_patch.write_bytes(b"prior replayed bytes")
    (warm / "recipe.json").write_text(
        json.dumps(
            {
                "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
                "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
                "value": {
                    "kernel": {},
                    "explore": {
                        "extra_server_args": "--prior-config",
                        "extra_envs": {"VLLM_PRIOR": "1"},
                        "patches": [],
                    },
                    "framework": {"patches": [old_ref]},
                    "patch_timeline": [old_ref],
                },
            }
        ),
        encoding="utf-8",
    )
    sections = KnowledgeSections(tmp_path / "draft", warm_start_dir=warm)
    state = _state(
        [
            {
                "action": "replay_warm_recipe",
                "source_phase": "PRELUDE",
                "extra_server_args": "--prior-config",
                "extra_envs": {"VLLM_PRIOR": "1"},
            }
        ]
    )
    state.warm_replay_outcome = {
        "status": "reproduced",
        "replayed_patch_refs": [old_ref],
    }
    state.current_best.update(
        {
            "extra_server_args": "--prior-config",
            "extra_envs": {"VLLM_PRIOR": "1"},
        }
    )

    bundle = build_remote_knowledge(
        state,
        tmp_path / "files",
        sections=sections,
    )

    adopted = "framework/overlays/000000/00-upstream.patch"
    assert bundle.knowledge["value"]["config"] == {
        "extra_server_args": "--prior-config",
        "extra_envs": {"VLLM_PRIOR": "1"},
    }
    assert bundle.knowledge["value"]["framework"]["patches"] == [adopted]
    assert bundle.knowledge["value"]["patch_timeline"][0] == adopted
    assert (tmp_path / "files" / adopted).read_bytes() == old_patch.read_bytes()


@pytest.mark.parametrize("failure_mode", ["missing", "symlink", "unreadable"])
def test_close_fails_when_replayed_prior_overlay_cannot_be_read(
    tmp_path: Path,
    monkeypatch,
    failure_mode: str,
) -> None:
    old_ref = "framework/overlays/000004/00-upstream.patch"
    warm = tmp_path / f"warm-{failure_mode}"
    source = warm / "files" / old_ref
    source.parent.mkdir(parents=True)
    if failure_mode == "symlink":
        outside = tmp_path / "outside.patch"
        outside.write_bytes(b"outside")
        source.symlink_to(outside)
    elif failure_mode == "unreadable":
        source.write_bytes(b"bytes")
        real_read_bytes = Path.read_bytes

        def _read_bytes(path: Path):
            if path == source:
                raise OSError("unreadable")
            return real_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", _read_bytes)
    (warm / "recipe.json").write_text(
        json.dumps(
            {
                "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
                "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
                "value": {
                    "kernel": {},
                    "explore": {},
                    "framework": {"patches": [old_ref]},
                    "patch_timeline": [old_ref],
                },
            }
        ),
        encoding="utf-8",
    )
    state = _state(
        [{"action": "replay_warm_recipe", "source_phase": "PRELUDE"}]
    )
    state.warm_replay_outcome = {
        "status": "reproduced",
        "replayed_patch_refs": [old_ref],
    }

    with pytest.raises(RemoteRecipeValidationError):
        build_remote_knowledge(
            state,
            tmp_path / f"files-{failure_mode}",
            sections=KnowledgeSections(
                tmp_path / f"draft-{failure_mode}",
                warm_start_dir=warm,
            ),
        )


def test_close_fails_when_replayed_prior_overlay_adoption_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    old_ref = "framework/overlays/000004/00-upstream.patch"
    warm = tmp_path / "warm-adopt-failure"
    source = warm / "files" / old_ref
    source.parent.mkdir(parents=True)
    source.write_bytes(b"bytes")
    (warm / "recipe.json").write_text(
        json.dumps(
            {
                "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
                "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
                "value": {
                    "kernel": {},
                    "explore": {},
                    "framework": {"patches": [old_ref]},
                    "patch_timeline": [old_ref],
                },
            }
        ),
        encoding="utf-8",
    )
    state = _state(
        [{"action": "replay_warm_recipe", "source_phase": "PRELUDE"}]
    )
    state.warm_replay_outcome = {
        "status": "reproduced",
        "replayed_patch_refs": [old_ref],
    }
    monkeypatch.setattr(
        _Files,
        "adopt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("copy failed")),
    )

    with pytest.raises(
        RemoteRecipeValidationError,
        match="cannot adopt",
    ):
        build_remote_knowledge(
            state,
            tmp_path / "files-adopt-failure",
            sections=KnowledgeSections(
                tmp_path / "draft-adopt-failure",
                warm_start_dir=warm,
            ),
        )


def test_close_adopts_aggregate_timeline_over_100_members(
    tmp_path: Path,
) -> None:
    warm = tmp_path / "warm-many"
    refs = [
        f"framework/overlays/000004/{index:02d}-p{index}.patch"
        for index in range(101)
    ]
    for index, ref in enumerate(refs):
        patch = warm / "files" / ref
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_bytes(f"patch-{index}".encode())
    (warm / "recipe.json").write_text(
        json.dumps(
            {
                "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
                "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
                "value": {
                    "kernel": {},
                    "explore": {},
                    "framework": {"patches": refs},
                    "patch_timeline": refs,
                },
            }
        ),
        encoding="utf-8",
    )
    state = _state(
        [{"action": "replay_warm_recipe", "source_phase": "PRELUDE"}]
    )
    state.warm_replay_outcome = {
        "status": "reproduced",
        "replayed_patch_refs": refs,
    }

    bundle = build_remote_knowledge(
        state,
        tmp_path / "many-files",
        sections=KnowledgeSections(
            tmp_path / "many-draft",
            warm_start_dir=warm,
        ),
    )

    timeline = bundle.knowledge["value"]["patch_timeline"]
    assert len(timeline) == 101
    assert timeline[-1].startswith(
        "framework/overlays/000000/100-"
    )
    bundle.validate()


def test_close_drops_already_present_prior_overlay(tmp_path: Path) -> None:
    old_ref = "framework/overlays/000004/00-upstream.patch"
    warm = tmp_path / "warm"
    old_patch = warm / "files" / old_ref
    old_patch.parent.mkdir(parents=True)
    old_patch.write_bytes(b"already in base")
    (warm / "recipe.json").write_text(
        json.dumps(
            {
                "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
                "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
                "value": {
                    "kernel": {},
                    "explore": {"patches": []},
                    "framework": {"patches": [old_ref]},
                    "patch_timeline": [old_ref],
                },
            }
        ),
        encoding="utf-8",
    )
    sections = KnowledgeSections(tmp_path / "draft", warm_start_dir=warm)
    state = _state(
        [{"action": "replay_warm_recipe", "source_phase": "PRELUDE"}]
    )
    state.warm_replay_outcome = {
        "status": "reproduced",
        # already_present is intentionally absent from replayed_patch_refs.
        "replayed_patch_refs": [],
    }

    bundle = build_remote_knowledge(
        state,
        tmp_path / "files",
        sections=sections,
    )

    assert bundle.knowledge["value"]["patch_timeline"] == []
    assert bundle.knowledge["value"]["framework"]["patches"] == []
    assert not (tmp_path / "files" / old_ref).exists()
