# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The three published columns: what each facade stages, reads, and refuses."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.knowledge.agent_kb import (
    ConfigKB,
    KernelAgentKB,
    PatchKB,
)
from hyperloom.orchestrator.knowledge.remote_recipe._vendor.kb_store_client import (
    KnowledgeSections,
)
from hyperloom.orchestrator.knowledge.remote_recipe.values import (
    CURRENT_KNOWLEDGE_SCHEMA_VERSION,
    RECORD_KIND_HYPERLOOM_RECIPE,
    RemoteRecipeValidationError,
    build_remote_knowledge,
    knowledge_to_warm_recipe,
)
from hyperloom.orchestrator.phases.prelude import _merge_named_current_recipe_configs


def _state(stack: list[dict]) -> SimpleNamespace:
    normalized_stack = []
    for raw in stack:
        row = dict(raw)
        if str(row.get("action") or "").lower() == "replay_warm_recipe" and "recipe_delta" not in row:
            row["recipe_delta"] = {
                "extra_server_args": str(row.get("candidate_extra_server_args") or row.get("extra_server_args") or ""),
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


# -- inactive draft ---------------------------------------------------------


@pytest.mark.parametrize("facade", [ConfigKB, PatchKB, KernelAgentKB])
def test_no_draft_leaves_every_facade_inactive(monkeypatch, facade) -> None:
    monkeypatch.delenv("KB_DRAFT_DIR", raising=False)
    monkeypatch.delenv("KB_WARM_START_DIR", raising=False)

    kb = facade.open()

    assert kb.active is False
    assert kb.prior() == {}


def test_no_draft_makes_staging_a_noop(monkeypatch) -> None:
    monkeypatch.delenv("KB_DRAFT_DIR", raising=False)
    monkeypatch.delenv("KB_WARM_START_DIR", raising=False)

    staged_refs = PatchKB.open().stage_patches([], stack_index=0)
    config_staged = ConfigKB.open().stage({"extra_server_args": "--x"})
    assert staged_refs == []
    assert config_staged is False


# -- config column ----------------------------------------------------------


def test_config_reads_the_single_final_value(tmp_path: Path) -> None:
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
                        "extra_envs": {"SGLANG_USE_AITER": "1"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = ConfigKB(KnowledgeSections(tmp_path / "draft", warm_start_dir=warm)).read()

    assert config == {
        "extra_server_args": "--page-size 32",
        "extra_envs": {"SGLANG_USE_AITER": "1"},
    }


def test_config_read_is_a_cold_start_without_the_column(tmp_path: Path) -> None:
    warm = tmp_path / "warm"
    warm.mkdir()
    (warm / "recipe.json").write_text(json.dumps({"value": {"patch": {"patches": []}}}), encoding="utf-8")

    config = ConfigKB(KnowledgeSections(tmp_path / "draft", warm_start_dir=warm)).read()

    assert config == {"extra_server_args": "", "extra_envs": {}}


def test_config_stage_replaces_rather_than_accumulates(tmp_path: Path) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    config_kb = ConfigKB(sections)

    assert config_kb.stage({"extra_server_args": "--first", "extra_envs": {"SGLANG_USE_AITER": "1"}})
    assert config_kb.stage({"extra_server_args": "--second", "extra_envs": {}})

    assert sections.staged("config").knowledge == {
        "extra_server_args": "--second",
        "extra_envs": {},
    }


@pytest.mark.parametrize(
    "kernel",
    [
        {"extra_server_args": "--page-size 64"},
        {"extra_envs": {"SHARED": "kernel"}},
    ],
)
def test_config_and_kernel_merge_fails_closed_on_conflict(kernel: dict) -> None:
    with pytest.raises(ValueError, match="conflict"):
        _merge_named_current_recipe_configs(
            [
                ("config", {"extra_server_args": "--page-size 32", "extra_envs": {"SHARED": "recipe"}}),
                ("kernel", kernel),
            ]
        )


# -- patch column: overlays -------------------------------------------------


def test_overlay_refs_are_deterministic_and_idempotent(tmp_path: Path) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    kb = PatchKB(sections)
    first = tmp_path / "fix one.diff"
    second = tmp_path / "tune.patch"
    first.write_bytes(b"diff --git a/a b/a\n")
    second.write_bytes(b"diff --git a/b b/b\n")

    refs = kb.stage_patches([first, second], stack_index=7)
    again = kb.stage_patches([first, second], stack_index=7)

    assert (
        refs
        == again
        == [
            "patch/overlays/000007/00-fix-one.patch",
            "patch/overlays/000007/01-tune.patch",
        ]
    )
    staged = sections.staged("patch")
    assert staged.knowledge["patches"] == refs
    assert [path.relative_to(sections.files_dir).as_posix() for path in staged.files] == refs
    assert (sections.files_dir / refs[0]).read_bytes() == first.read_bytes()


def test_recorded_overlay_order_is_the_replay_order(tmp_path: Path) -> None:
    """Zero-padded indices make the recorded order the apply order."""
    sections = KnowledgeSections(tmp_path / "draft")
    kb = PatchKB(sections)
    late = tmp_path / "late.patch"
    early = tmp_path / "early.patch"
    late.write_bytes(b"late")
    early.write_bytes(b"early")

    kb.stage_patches([late], stack_index=11)
    kb.stage_patches([early], stack_index=2)

    assert kb._staged()[0]["patches"] == [
        "patch/overlays/000002/00-early.patch",
        "patch/overlays/000011/00-late.patch",
    ]


def test_same_ref_cannot_be_replaced_with_different_bytes(tmp_path: Path) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    kb = PatchKB(sections)
    first = tmp_path / "a" / "same.patch"
    second = tmp_path / "b" / "same.patch"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    assert kb.stage_patches([first], stack_index=2)
    assert kb.stage_patches([second], stack_index=2) == []
    assert (sections.files_dir / "patch/overlays/000002/00-same.patch").read_bytes() == b"first"


def test_multi_patch_staging_is_atomic_on_unreadable_member(tmp_path: Path) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    kb = PatchKB(sections)
    prior = tmp_path / "prior.patch"
    first = tmp_path / "first.patch"
    missing = tmp_path / "missing.patch"
    prior.write_bytes(b"prior")
    first.write_bytes(b"first")
    assert kb.stage_patches([prior], stack_index=2)
    section_file = sections.root / "sections" / "patch.json"
    before_section = section_file.read_bytes()
    before_files = sorted(
        path.relative_to(sections.files_dir) for path in sections.files_dir.rglob("*") if path.is_file()
    )

    assert kb.stage_patches([first, missing], stack_index=3) == []

    assert section_file.read_bytes() == before_section
    assert (
        sorted(path.relative_to(sections.files_dir) for path in sections.files_dir.rglob("*") if path.is_file())
        == before_files
    )


def test_overlay_staging_rejects_more_than_100_members_atomically(tmp_path: Path, caplog) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    kb = PatchKB(sections)
    patches: list[Path] = []
    for index in range(101):
        patch = tmp_path / f"{index}.patch"
        patch.write_bytes(str(index).encode())
        patches.append(patch)

    with caplog.at_level("WARNING"):
        assert kb.stage_patches(patches, stack_index=0) == []

    assert "maximum is 100" in caplog.text
    assert sections.staged("patch") is None
    assert not any(path.is_file() for path in sections.files_dir.rglob("*"))


# -- patch column: overlay provenance ---------------------------------------


def test_provenance_records_how_the_overlay_was_captured(tmp_path: Path) -> None:
    sections = KnowledgeSections(tmp_path / "draft")

    assert PatchKB(sections).stage_provenance(
        stack_index=3,
        base_sha="abc123",
        complete=False,
        artifacts_outside_root=2,
        realized=True,
    )

    assert sections.staged("patch").knowledge["provenance"] == [
        {
            "stack_index": 3,
            "base_sha": "abc123",
            "complete": False,
            "artifacts_outside_root": 2,
            "realized": True,
        }
    ]


def test_provenance_carries_no_files(tmp_path: Path) -> None:
    """Provenance is metadata; nothing about it is uploadable."""
    sections = KnowledgeSections(tmp_path / "draft")

    PatchKB(sections).stage_provenance(stack_index=0, base_sha="abc")

    assert sections.staged("patch").files == []


def test_restaging_provenance_replaces_its_row(tmp_path: Path) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    kb = PatchKB(sections)

    assert kb.stage_provenance(stack_index=1, complete=True)
    assert kb.stage_provenance(stack_index=1, complete=False)

    rows = sections.staged("patch").knowledge["provenance"]
    assert len(rows) == 1
    assert rows[0]["complete"] is False


def test_provenance_is_ordered_by_stack_index(tmp_path: Path) -> None:
    warm = tmp_path / "warm"
    warm.mkdir()
    (warm / "recipe.json").write_text(
        json.dumps(
            {
                "value": {
                    "patch": {
                        "provenance": [
                            {"stack_index": 5, "complete": True},
                            {"stack_index": 1, "complete": True},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    rows = PatchKB(KnowledgeSections(tmp_path / "draft", warm_start_dir=warm)).read_provenance()

    assert [row["stack_index"] for row in rows] == [1, 5]


def test_prior_file_refuses_refs_outside_the_column(tmp_path: Path) -> None:
    warm = tmp_path / "warm"
    ref = "patch/overlays/000000/00-prior.patch"
    (warm / "files" / ref).parent.mkdir(parents=True)
    (warm / "files" / ref).write_bytes(b"prior")
    (warm / "recipe.json").write_text(json.dumps({"value": {"patch": {"patches": [ref]}}}), encoding="utf-8")
    kb = PatchKB(KnowledgeSections(tmp_path / "draft", warm_start_dir=warm))

    assert kb.read_patches() == [ref]
    assert kb.prior_file(ref) is not None
    assert kb.prior_file("kernel/gemm/artifacts/other.json") is None
    assert kb.prior_file("../escape.patch") is None


# -- kernel column ----------------------------------------------------------


def test_kernel_column_is_published_empty_for_geak(tmp_path: Path) -> None:
    sections = KnowledgeSections(tmp_path / "draft")

    assert KernelAgentKB(sections).stage_from_state(_state([]), kernel_optimizer="geak")

    assert sections.staged("kernel").knowledge == {
        "gemm": {"optimizations": []},
        "fusion": {"items": []},
        "rewrite": {"items": []},
    }


def test_kernel_read_strips_host_local_source_metadata(tmp_path: Path) -> None:
    warm = tmp_path / "warm"
    warm.mkdir()
    (warm / "recipe.json").write_text(
        json.dumps(
            {
                "value": {
                    "kernel": {
                        "gemm": {
                            "optimizations": [
                                {
                                    "kernel_name": "moe",
                                    "tuned_file": "kernel/gemm/artifacts/moe.json",
                                    "source_file": "/workspace/local.py",
                                    "target_files": ["/workspace/other.py"],
                                }
                            ]
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    gemm = KernelAgentKB(KnowledgeSections(tmp_path / "draft", warm_start_dir=warm)).read_gemm()

    assert gemm == {
        "optimizations": [
            {
                "kernel_name": "moe",
                "tuned_file": "kernel/gemm/artifacts/moe.json",
            }
        ]
    }


# -- CLOSE assembly ---------------------------------------------------------


def test_close_publishes_exactly_three_columns(tmp_path: Path) -> None:
    state = _state(
        [
            {
                "action": "explore",
                "source_phase": "FRAMEWORK_AGENT",
                "recipe_delta": {
                    "extra_server_args": "--first",
                    "extra_envs": {"SGLANG_USE_AITER": "0"},
                    "args_mode": "append",
                },
            },
            {
                "action": "framework",
                "source_phase": "FRAMEWORK_AGENT",
                "recipe_delta": {
                    "extra_server_args": "--final",
                    "extra_envs": {"SGLANG_USE_AITER": "1"},
                    "args_mode": "replace",
                },
            },
        ]
    )

    bundle = build_remote_knowledge(
        state,
        tmp_path / "files",
        sections=KnowledgeSections(tmp_path / "draft"),
    )

    value = bundle.knowledge["value"]
    assert sorted(value) == ["config", "kernel", "patch"]
    assert "patch_timeline" not in value
    assert value["config"] == {
        "extra_server_args": "--first --final",
        "extra_envs": {"SGLANG_USE_AITER": "1"},
    }
    # A session that harvested no overlay leaves the column unproduced rather
    # than publishing an empty shell, so the record distinguishes the two.
    assert value["patch"] == {}
    assert bundle.knowledge["provenance"]["staged_sections"] == ["config", "kernel"]
    row = knowledge_to_warm_recipe({"canonical_id": "inference:test", "knowledge": bundle.knowledge})
    assert "best_config" not in row
    assert "patch_timeline" not in row
    bundle.validate()


def test_close_carries_overlays_and_provenance_into_the_patch_column(tmp_path: Path) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    patch = tmp_path / "fix.patch"
    patch.write_bytes(b"diff")
    kb = PatchKB(sections)
    refs = kb.stage_patches([patch], stack_index=0)
    kb.stage_provenance(stack_index=0, base_sha="abc123", complete=True, realized=True)

    bundle = build_remote_knowledge(
        _state([{"action": "integrate_patch", "source_phase": "FRAMEWORK_AGENT", "kb_required_owner": "PATCH"}]),
        tmp_path / "files",
        sections=sections,
    )

    column = bundle.knowledge["value"]["patch"]
    assert column["patches"] == refs
    assert [row["realized"] for row in column["provenance"]] == [True]
    # Only the overlay is uploadable; provenance adds no artifacts.
    assert {artifact.path for artifact in bundle.artifacts} == {refs[0]}
    bundle.validate()


def test_close_requires_a_staged_draft(tmp_path: Path) -> None:
    with pytest.raises(RemoteRecipeValidationError, match="staged draft"):
        build_remote_knowledge(_state([]), tmp_path / "files", sections=None)


def test_close_drops_permanently_missing_patch_staging(tmp_path: Path) -> None:
    state = _state([])
    state.kb_stage_outbox = [
        {
            "id": "FRAMEWORK_AGENT:0",
            "owner": "FRAMEWORK_AGENT",
            "stack_index": 0,
            "missing_patch_sources": ["missing.patch"],
        }
    ]

    bundle = build_remote_knowledge(
        state,
        tmp_path / "files",
        sections=KnowledgeSections(tmp_path / "draft"),
    )

    assert bundle.knowledge["provenance"]["dropped_staged_sections"] == ["FRAMEWORK_AGENT:0"]


def test_close_fails_with_transient_incomplete_patch_staging(tmp_path: Path) -> None:
    state = _state([])
    state.kb_stage_outbox = [{"id": "FRAMEWORK_AGENT:0"}]

    with pytest.raises(RemoteRecipeValidationError, match="required section staging is incomplete"):
        build_remote_knowledge(
            state,
            tmp_path / "files",
            sections=KnowledgeSections(tmp_path / "draft"),
        )


def test_close_fails_for_staged_column_with_a_missing_file(tmp_path: Path) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    section_file = sections.root / "sections" / "patch.json"
    section_file.parent.mkdir(parents=True)
    ref = "patch/overlays/000000/00-missing.patch"
    section_file.write_text(
        json.dumps({"knowledge": {"patches": [ref]}, "files": [ref]}),
        encoding="utf-8",
    )

    with pytest.raises(RemoteRecipeValidationError, match="staged section 'patch' file mismatch"):
        build_remote_knowledge(
            _state([{"action": "framework", "kb_required_owner": "PATCH"}]),
            tmp_path / "files",
            sections=sections,
        )


def test_close_fails_for_an_empty_required_column(tmp_path: Path) -> None:
    sections = KnowledgeSections(tmp_path / "draft")
    section_file = sections.root / "sections" / "patch.json"
    section_file.parent.mkdir(parents=True)
    section_file.write_text(json.dumps({"knowledge": {}, "files": []}), encoding="utf-8")

    with pytest.raises(RemoteRecipeValidationError, match="required staged section 'patch' is empty"):
        build_remote_knowledge(
            _state([{"action": "explore", "kb_required_owner": "PATCH"}]),
            tmp_path / "files",
            sections=sections,
        )


@pytest.mark.parametrize("owner", ["PATCH", "EXPLORE", "FRAMEWORK_AGENT"])
def test_close_fails_when_a_demanded_patch_column_never_landed(tmp_path: Path, owner: str) -> None:
    """Any recorded owner spelling demands the one patch column."""
    state = _state([{"action": "integrate_patch", "kb_required_owner": owner}])

    with pytest.raises(RemoteRecipeValidationError, match="'patch'"):
        build_remote_knowledge(
            state,
            tmp_path / "files",
            sections=KnowledgeSections(tmp_path / "draft"),
        )


def test_close_adopts_only_successfully_replayed_prior_overlays(tmp_path: Path) -> None:
    old_ref = "patch/overlays/000004/00-upstream.patch"
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
                    "config": {"extra_server_args": "--prior-config", "extra_envs": {}},
                    "kernel": {},
                    "patch": {"patches": [old_ref]},
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
            }
        ]
    )
    state.warm_replay_outcome = {"status": "reproduced", "replayed_patch_refs": [old_ref]}

    bundle = build_remote_knowledge(state, tmp_path / "files", sections=sections)

    adopted = "patch/overlays/000000/00-upstream.patch"
    assert bundle.knowledge["value"]["patch"]["patches"] == [adopted]
    assert {artifact.path for artifact in bundle.artifacts} == {adopted}
    bundle.validate()


def test_close_carries_prior_apply_root_onto_adopted_overlay(tmp_path: Path) -> None:
    """An adopted overlay keeps the checkout its prior record named.

    Dropping the prior apply_root would leave the re-homed overlay rootless, and
    the next generation's fail-closed replay would then skip the whole Recipe --
    so the root travels with the bytes onto the new ref.
    """
    old_ref = "patch/overlays/000004/00-upstream.patch"
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
                    "config": {"extra_server_args": "--prior-config", "extra_envs": {}},
                    "kernel": {},
                    "patch": {
                        "patches": [old_ref],
                        "provenance": [
                            {
                                "stack_index": 4,
                                "host_origin": {"apply_roots": {old_ref: "/sgl-workspace/sglang"}},
                            }
                        ],
                    },
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
            }
        ]
    )
    state.warm_replay_outcome = {"status": "reproduced", "replayed_patch_refs": [old_ref]}

    bundle = build_remote_knowledge(state, tmp_path / "files", sections=sections)

    adopted = "patch/overlays/000000/00-upstream.patch"
    patch = bundle.knowledge["value"]["patch"]
    assert patch["patches"] == [adopted]
    roots = {
        ref: root
        for row in patch.get("provenance", [])
        for ref, root in row.get("host_origin", {}).get("apply_roots", {}).items()
    }
    assert roots == {adopted: "/sgl-workspace/sglang"}
    bundle.validate()


def test_close_leaves_adopted_overlay_rootless_when_prior_had_no_root(tmp_path: Path) -> None:
    """Nothing is invented: an ancestor that already lost its root stays rootless."""
    old_ref = "patch/overlays/000004/00-upstream.patch"
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
                    "config": {"extra_server_args": "--prior-config", "extra_envs": {}},
                    "kernel": {},
                    "patch": {"patches": [old_ref]},
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
            }
        ]
    )
    state.warm_replay_outcome = {"status": "reproduced", "replayed_patch_refs": [old_ref]}

    bundle = build_remote_knowledge(state, tmp_path / "files", sections=sections)

    adopted = "patch/overlays/000000/00-upstream.patch"
    patch = bundle.knowledge["value"]["patch"]
    assert patch["patches"] == [adopted]
    roots = {
        ref: root
        for row in patch.get("provenance", [])
        for ref, root in row.get("host_origin", {}).get("apply_roots", {}).items()
    }
    assert roots == {}
    bundle.validate()


def test_close_fails_when_a_replayed_prior_overlay_is_absent_from_prior_knowledge(tmp_path: Path) -> None:
    warm = tmp_path / "warm"
    (warm / "files").mkdir(parents=True)
    (warm / "recipe.json").write_text(
        json.dumps({"value": {"config": {}, "kernel": {}, "patch": {"patches": []}}}),
        encoding="utf-8",
    )
    state = _state([{"action": "replay_warm_recipe", "source_phase": "PRELUDE"}])
    state.warm_replay_outcome = {
        "status": "reproduced",
        "replayed_patch_refs": ["patch/overlays/000004/00-gone.patch"],
    }

    with pytest.raises(RemoteRecipeValidationError, match="absent from prior knowledge"):
        build_remote_knowledge(
            state,
            tmp_path / "files",
            sections=KnowledgeSections(tmp_path / "draft", warm_start_dir=warm),
        )


def test_provenance_records_a_checkout_per_overlay_ref(tmp_path: Path) -> None:
    """Overlays cut from different trees each keep their own checkout.

    One answer for the set would place an overlay against a tree it was never
    measured on, so the ref is what the root hangs off.
    """
    sections = KnowledgeSections(tmp_path / "draft")

    assert PatchKB(sections).stage_provenance(
        stack_index=0,
        host_origin={
            "apply_roots": {
                "patch/overlays/000000/00-sglang.patch": "/sglang",
                "patch/overlays/000000/01-tuned-csv.patch": "/workspace/tuning",
            },
            "snapshot": "/session/optimization_stack/src/spec-1",
            "manifest": "/session/optimization_stack/src/spec-1/manifest.json",
            "sources": ["/session/optimization_stack/src/spec-1/realized.patch"],
        },
    )

    assert sections.staged("patch").knowledge["provenance"][0]["host_origin"] == {
        "apply_roots": {
            "patch/overlays/000000/00-sglang.patch": "/sglang",
            "patch/overlays/000000/01-tuned-csv.patch": "/workspace/tuning",
        },
        "snapshot": "/session/optimization_stack/src/spec-1",
        "manifest": "/session/optimization_stack/src/spec-1/manifest.json",
        "sources": ["/session/optimization_stack/src/spec-1/realized.patch"],
    }


def test_provenance_keeps_only_absolute_origins(tmp_path: Path) -> None:
    """A relative value would be an artifact ref, which ``patches`` already carries."""
    sections = KnowledgeSections(tmp_path / "draft")

    PatchKB(sections).stage_provenance(
        stack_index=0,
        host_origin={
            "apply_roots": {"patch/overlays/000000/00-fix.patch": "relative/tree"},
            "snapshot": "relative/dir",
            "sources": ["patch/overlays/000000/00-fix.patch", "/abs/realized.patch"],
        },
    )

    assert sections.staged("patch").knowledge["provenance"][0]["host_origin"] == {
        "sources": ["/abs/realized.patch"],
    }


def test_provenance_omits_host_origin_when_nothing_absolute_is_known(tmp_path: Path) -> None:
    sections = KnowledgeSections(tmp_path / "draft")

    PatchKB(sections).stage_provenance(stack_index=0, host_origin={"apply_roots": {}})

    assert "host_origin" not in sections.staged("patch").knowledge["provenance"][0]
