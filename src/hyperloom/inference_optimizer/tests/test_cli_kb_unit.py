# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for local and KB Store Recipe bootstrap paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest


from hyperloom.inference_optimizer.cli import kb as cli_kb


def _args(**over):
    base = dict(
        degraded_kb=False,
        pr_monitor_enabled=True,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_attach_recipe_audit_hook_appends_jsonl(tmp_path) -> None:
    import json

    from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStore, RecipeKB
    from hyperloom.inference_optimizer.session.session_paths import recipe_snapshot_audit_jsonl

    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"))
    cli_kb._attach_recipe_audit_hook(kb, tmp_path)
    assert callable(kb.audit_hook)
    kb.audit_hook({"method": "get_recipe", "resolution": "local", "hit": False})
    path = recipe_snapshot_audit_jsonl(tmp_path)
    assert path.exists()
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["method"] == "get_recipe"
    assert "ts" in row


def test_bootstrap_recipe_kb_degraded_returns_none(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(tmp_path / "kb"))
    kb = cli_kb._bootstrap_recipe_kb(
        _args(degraded_kb=True),
        session_dir=tmp_path,
        manifest={"model_name": "m"},
        resume=False,
    )
    assert kb is None
    assert "DISABLED (--degraded-kb)" in capsys.readouterr().out


def test_bootstrap_recipe_kb_success(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(tmp_path / "kb"))
    calls = []
    monkeypatch.setattr(cli_kb, "run_t0_anchor", lambda *a, **k: calls.append(k))
    kb = cli_kb._bootstrap_recipe_kb(
        _args(),
        session_dir=tmp_path,
        manifest={"model_name": "m"},
        resume=False,
    )
    assert kb is not None
    assert calls


def test_bootstrap_recipe_kb_t0_failure_continues(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(tmp_path / "kb"))

    from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStoreError

    def _boom(*a, **k):
        raise LocalRecipeStoreError("t0 down")

    monkeypatch.setattr(cli_kb, "run_t0_anchor", _boom)
    args = _args()
    kb = cli_kb._bootstrap_recipe_kb(
        args,
        session_dir=tmp_path,
        manifest={"model_path": "/models/Qwen", "stack_fingerprint": {"rocm": "6.2"}, "image": "img@sha"},
        resume=False,
    )
    assert kb is not None
    assert args.kb_degraded_reason == "t0_runtime_fail"


def test_bootstrap_recipe_kb_remote_stores_metadata_not_replay_payload(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from hyperloom.orchestrator.knowledge import remote_recipe

    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.setenv("KB_STORE_URL", "https://kb.test")
    monkeypatch.setenv("KB_STORE_TOKEN", "token")
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState.load_or_init(tmp_path)
    state.kernel_optimizer = "native"
    state.tp = 8
    state.conc = 64
    state.isl = 8192
    state.osl = 1024
    state.save(tmp_path)
    reads: list[tuple[str, Path, object]] = []

    class _Remote:
        def read(self, identity: str, destination: Path, scope):
            reads.append((identity, destination, scope))
            return {
                "schema_version": 2,
                "knowledge_schema_version": (remote_recipe.CURRENT_KNOWLEDGE_SCHEMA_VERSION),
                "record_kind": remote_recipe.RECORD_KIND_HYPERLOOM_RECIPE,
                "canonical_id": identity,
                "session_id": "champion-session",
                "optimized_throughput": 120.0,
                "validated_e2e_gain": 20.0,
                "value": {
                    "kernel": {},
                    "patch": {"patches": []},
                    "config": {
                        "extra_server_args": "--page-size 32",
                        "extra_envs": {"SGLANG_USE_AITER": "1"},
                    },
                },
            }

        def search_identities(self, *, scheme, match, hardware_in=None, offset=0, limit=50):
            return {"items": [], "next_offset": None}

    monkeypatch.setattr(
        remote_recipe.RemoteRecipeClient,
        "from_env",
        classmethod(lambda cls: _Remote()),
    )
    assert (
        cli_kb._bootstrap_recipe_kb(
            _args(),
            session_dir=tmp_path,
            manifest={
                "model_name": "m",
                "stack_fingerprint": {"rocm": "6.2"},
                "image": "img@sha",
            },
            resume=False,
        )
        is None
    )
    persisted = SharedState.load_or_init(tmp_path)
    assert persisted.stack_fingerprint_meta["rocm"] == "6.2"
    assert persisted.stack_fingerprint_meta["image_digest"] == "img@sha"
    assert "recommended_replay" not in persisted.warm_start_context
    assert persisted.warm_start_context["match"]["source"] == "kb-store"
    assert "best_config" not in persisted.warm_start_recipe["recipe"]
    assert "patch_timeline" not in persisted.warm_start_recipe["recipe"]
    assert len(reads) == 1
    assert reads[0][1] == tmp_path / "runtime" / "remote_recipe"
    assert reads[0][2].as_dict() == {
        "kernel_optimizer": "forge",
        "tp": 8,
        "conc": 64,
        "isl": 8192,
        "osl": 1024,
    }
    assert "current Recipe warm replay" in capsys.readouterr().out


def test_bootstrap_knowledge_plane_validates_remote_without_recipe_dispatcher(monkeypatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.delenv("KB_STORE_URL", raising=False)
    monkeypatch.delenv("KB_STORE_TOKEN", raising=False)
    with pytest.raises(ValueError, match="KB_STORE_URL"):
        cli_kb._bootstrap_knowledge_plane(
            _args(pr_monitor_enabled=False),
            recipe_kb_client=None,
        )


def test_bootstrap_knowledge_plane_degraded_bypasses_remote_validation(monkeypatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.delenv("KB_STORE_URL", raising=False)
    monkeypatch.delenv("KB_STORE_TOKEN", raising=False)

    plane = cli_kb._bootstrap_knowledge_plane(
        _args(degraded_kb=True, pr_monitor_enabled=False),
        recipe_kb_client=None,
    )

    assert plane.kb_disabled is True
    assert plane.recipe_kb is None


def test_bootstrap_knowledge_plane_enabled() -> None:
    plane = cli_kb._bootstrap_knowledge_plane(_args(pr_monitor_enabled=True))
    assert plane is not None


def test_bootstrap_knowledge_plane_disabled() -> None:
    plane = cli_kb._bootstrap_knowledge_plane(_args(pr_monitor_enabled=False, pr_degraded_reason="explicit_flag"))
    assert plane is not None


def test_attach_recipe_audit_hook_write_error_is_swallowed(tmp_path, monkeypatch) -> None:
    """A write failure inside the hook is swallowed, not raised."""
    from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStore, RecipeKB
    from hyperloom.inference_optimizer.session import session_paths as sp

    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"))

    class _BadPath:
        parent = None

        def mkdir(self, *_a, **_k):
            raise OSError("boom")

    bad = _BadPath()
    bad.parent = bad  # type: ignore[assignment]
    monkeypatch.setattr(sp, "recipe_snapshot_audit_jsonl", lambda _sd: bad)
    cli_kb._attach_recipe_audit_hook(kb, tmp_path)
    kb.audit_hook({"method": "search"})
