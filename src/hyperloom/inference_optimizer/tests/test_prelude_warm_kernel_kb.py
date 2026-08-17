# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE preparation of the inference Recipe's kernel section.

PRELUDE reads nested ``value.kernel.gemm/fusion/rewrite`` through
:class:`KernelAgentKB` from the same already-downloaded inference Recipe used by
Explore and Framework. The Recipe replay task grades the combined set once.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.knowledge.agent_kb import KernelAgentKB
from hyperloom.orchestrator.knowledge.remote_recipe._vendor.kb_store_client import (
    KnowledgeSections,
)
from hyperloom.orchestrator.phases.prelude import PreludePhase


class _StubPrelude:
    """Minimal carrier binding the prelude warm-kernel methods to a stub state."""

    _collect_warm_kernel_plan = PreludePhase._collect_warm_kernel_plan
    _parse_diff_target = staticmethod(PreludePhase._parse_diff_target)
    _resolve_kernel_target_path = PreludePhase._resolve_kernel_target_path
    _warm_kernel_extra_envs = staticmethod(PreludePhase._warm_kernel_extra_envs)
    _revert_warm_kernel_patches = staticmethod(PreludePhase._revert_warm_kernel_patches)
    _snapshot_warm_kernel_target = PreludePhase._snapshot_warm_kernel_target
    _set_warm_kernel_outcome = PreludePhase._set_warm_kernel_outcome
    _prepare_warm_kernel_kb = PreludePhase._prepare_warm_kernel_kb

    def _warm_kernel_gate_reason(self) -> str:
        """Stubbed gate: these tests exercise the replay itself."""
        return self.gate_reason

    def _resolve_kernel_target_paths(self, entry: dict) -> list[str]:
        """Keep legacy single-target fixtures focused on preparation behavior."""
        target = self._resolve_kernel_target_path(entry)
        return [target] if target else []

    def __init__(self, session_dir: Path, reader: object | None = None) -> None:
        self.session_dir = session_dir
        self.shared_state = SimpleNamespace(
            warm_kernel_kb_attempted=False,
            warm_kernel_kb_plan=[],
            warm_replay_outcome={},
            save=lambda *_a, **_k: None,
        )
        self._reader = reader
        self.gate_reason = ""
        self.applied: list[str] = []
        self.booked: list[dict] = []
        self.validations: list[dict] = []
        self.reverted: list[dict] = []

    def _open_warm_kernel_section(self) -> object | None:
        """Return the preloaded inference Recipe section facade."""
        return self._reader

    async def _record_warm_kernel_keep(self, result, pending, envs, args, applied) -> None:
        self.booked.append(
            {"result": result, "pending": pending, "envs": envs, "args": args}
        )

    def _apply_warm_kernel_patch(self, entry: dict, target: str) -> dict:
        self.applied.append(target)
        return {"status": "ok", "manifest_path": f"{target}.manifest"}


def _kernel_record(tmp_path: Path, value: dict, files: dict[str, str]) -> Path:
    """Write one downloaded inference Recipe with nested kernel content."""
    record = tmp_path / "remote_recipe"
    (record / "files").mkdir(parents=True, exist_ok=True)
    (record / "recipe.json").write_text(
        json.dumps({"value": {"kernel": value}}),
        encoding="utf-8",
    )
    for rel, text in files.items():
        target = record / "files" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return record


def _kernel_reader(record: Path) -> KernelAgentKB:
    return KernelAgentKB(
        KnowledgeSections(record / "draft", warm_start_dir=record)
    )


def _rewrite_item(target: Path, name: str = "k1") -> dict:
    return {
        "kernel_name": name,
        "patch": f"kernel/rewrite/{name}.py",
        "source_files": [f"kernel/rewrite/{name}.py"],
        "target_path": str(target),
    }


def _grading(stub: _StubPrelude, decision: str, gain: float = 5.0):
    async def _validate(extra_envs: dict, extra_server_args: str) -> dict:
        stub.validations.append(
            {"extra_envs": extra_envs, "extra_server_args": extra_server_args}
        )
        return {"status": "ok", "decision": decision, "gain_pct": gain}

    return _validate


def test_parse_diff_target_reads_plus_header(tmp_path: Path) -> None:
    patch = tmp_path / "k.diff"
    patch.write_text(
        "diff --git a/pkg/foo.py b/pkg/foo.py\n--- a/pkg/foo.py\n+++ b/pkg/foo.py\n@@\n-x\n+y\n",
        encoding="utf-8",
    )
    assert PreludePhase._parse_diff_target(str(patch)) == "pkg/foo.py"


def test_resolve_target_from_diff_header_against_roots(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "site-packages"
    live = root / "pkg" / "foo.py"
    live.parent.mkdir(parents=True, exist_ok=True)
    live.write_text("old", encoding="utf-8")
    patch = tmp_path / "k.diff"
    patch.write_text("+++ b/pkg/foo.py\n", encoding="utf-8")

    import hyperloom.orchestrator.framework.paths as paths

    monkeypatch.setattr(paths, "resolve_patch_target_roots", lambda: (str(root) + "/",))
    stub = _StubPrelude(tmp_path)

    resolved = stub._resolve_kernel_target_path({"patch_path": str(patch)})
    assert resolved == str(live)


def test_warm_kernel_envs_point_the_recorded_var_at_the_local_file() -> None:
    entry = {
        "column": "gemm",
        "source_paths": ["/local/dl/tunableop_results.csv"],
        "meta": {
            "recommended_env": {"HL_TUNABLEOP_MODE": "candidate"},
            "e2e_results": {"kept": [{"env_var": "PYTORCH_TUNABLEOP_FILENAME"}]},
        },
    }

    envs = PreludePhase._warm_kernel_extra_envs(entry)

    assert envs["PYTORCH_TUNABLEOP_FILENAME"] == "/local/dl/tunableop_results.csv"
    assert envs["HL_TUNABLEOP_MODE"] == "candidate"


def test_warm_kernel_envs_empty_without_a_recorded_env_var() -> None:
    entry = {"column": "gemm", "source_paths": ["/local/dl/t.csv"], "meta": {}}

    assert PreludePhase._warm_kernel_extra_envs(entry) == {}


def test_warm_kernel_apply_prefers_deploy_patch_over_source_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Authoring source context must never overwrite the deploy patch target."""
    captured: dict = {}
    materialized: dict = {}

    def _apply(payload, *, session_dir, kernel_id):
        captured.update(payload)
        captured["session_dir"] = session_dir
        captured["kernel_id_arg"] = kernel_id
        return {"status": "ok"}

    def _materialize(*, patch_path, repo_root, snapshot_dir):
        materialized.update(
            {
                "patch_path": Path(patch_path),
                "repo_root": Path(repo_root),
                "snapshot_dir": Path(snapshot_dir),
            }
        )
        return str(snapshot_dir)

    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._maybe_apply_kernel_patch",
        _apply,
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers.materialize_unified_patch_snapshot",
        _materialize,
    )
    session_dir = tmp_path / "session"
    framework_root = tmp_path / "framework"
    relative_target = Path("vllm/v1/attention/ops/prefix_prefill.py")
    target = framework_root / relative_target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("original\n", encoding="utf-8")
    deploy_patch = session_dir / "deploy.patch"
    deploy_patch.parent.mkdir(parents=True, exist_ok=True)
    deploy_patch.write_text(
        "\n".join(
            [
                f"diff --git a/{relative_target} b/{relative_target}",
                f"--- a/{relative_target}",
                f"+++ b/{relative_target}",
                "@@ -1 +1 @@",
                "-original",
                "+patched",
                "",
            ]
        ),
        encoding="utf-8",
    )
    source_snapshot = tmp_path / "attention.py"

    PreludePhase._apply_warm_kernel_patch(
        SimpleNamespace(
            session_dir=session_dir,
            _parse_diff_target=PreludePhase._parse_diff_target,
        ),
        {
            "patch_path": str(deploy_patch),
            "source_paths": [str(source_snapshot)],
            "meta": {"kernel_name": "k008"},
        },
        str(target),
    )

    assert captured["patch_path"] == str(deploy_patch)
    assert captured["patch_path"] != str(source_snapshot)
    assert captured["target_file"] == str(target)
    assert captured["snapshot_dir"] == str(materialized["snapshot_dir"])
    assert captured["kernel_repo"] == str(framework_root)
    assert materialized["patch_path"] == deploy_patch
    assert materialized["repo_root"] == framework_root


def test_warm_kernel_apply_keeps_source_only_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict = {}

    def _apply(payload, **_kwargs):
        captured.update(payload)
        return {"status": "ok"}

    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._maybe_apply_kernel_patch",
        _apply,
    )
    source_snapshot = tmp_path / "replacement.py"

    PreludePhase._apply_warm_kernel_patch(
        SimpleNamespace(session_dir=tmp_path),
        {"source_paths": [str(source_snapshot)], "meta": {}},
        str(tmp_path / "target.py"),
    )

    assert captured["patch_path"] == str(source_snapshot)


def test_multi_file_manifest_and_target_snapshot_both_roll_back(
    tmp_path: Path,
) -> None:
    from hyperloom.orchestrator.kernel.request_handlers import (
        _maybe_apply_kernel_patch,
    )

    live = tmp_path / "framework"
    materialized = tmp_path / "materialized"
    for root, prefix in ((live, "old"), (materialized, "new")):
        (root / "pkg").mkdir(parents=True)
        (root / "pkg/a.py").write_text(f"{prefix}-a\n", encoding="utf-8")
        (root / "pkg/b.py").write_text(f"{prefix}-b\n", encoding="utf-8")
    patch = tmp_path / "multi.patch"
    patch.write_text(
        "\n".join(
            [
                "diff --git a/pkg/a.py b/pkg/a.py",
                "--- a/pkg/a.py",
                "+++ b/pkg/a.py",
                "@@ -1 +1 @@",
                "-old-a",
                "+new-a",
                "diff --git a/pkg/b.py b/pkg/b.py",
                "--- a/pkg/b.py",
                "+++ b/pkg/b.py",
                "@@ -1 +1 @@",
                "-old-b",
                "+new-b",
                "",
            ]
        ),
        encoding="utf-8",
    )
    session_dir = tmp_path / "session"
    target = live / "pkg/a.py"
    snapshot = PreludePhase._snapshot_warm_kernel_target(
        SimpleNamespace(session_dir=session_dir),
        str(target),
        0,
    )
    applied = _maybe_apply_kernel_patch(
        {
            "patch_path": str(patch),
            "target_file": str(target),
            "snapshot_dir": str(materialized),
            "kernel_repo": str(live),
            "allow_unknown_target": True,
        },
        session_dir=session_dir,
        kernel_id="multi",
    )

    assert applied["status"] == "ok"
    assert (live / "pkg/a.py").read_text(encoding="utf-8") == "new-a\n"
    assert (live / "pkg/b.py").read_text(encoding="utf-8") == "new-b\n"

    rollback = PreludePhase._revert_warm_kernel_patches(
        [applied],
        [snapshot],
    )

    assert rollback == {"ok": True, "errors": []}
    assert (live / "pkg/a.py").read_text(encoding="utf-8") == "old-a\n"
    assert (live / "pkg/b.py").read_text(encoding="utf-8") == "old-b\n"


@pytest.mark.asyncio
async def test_inactive_without_record(tmp_path: Path) -> None:
    # No inference Recipe section is available (cold start / KB not configured).
    stub = _StubPrelude(tmp_path, reader=None)

    outcome = await stub._prepare_warm_kernel_kb()

    assert outcome == {"status": "skipped", "reason": "no_kernel_section"}


@pytest.mark.asyncio
async def test_one_shot_save_failure_stops_before_kernel_apply(
    tmp_path: Path,
) -> None:
    target = tmp_path / "serving/kernel.py"
    target.parent.mkdir(parents=True)
    target.write_text("old", encoding="utf-8")
    record = _kernel_record(
        tmp_path,
        {"rewrite": {"items": [_rewrite_item(target)]}},
        {"kernel/rewrite/k1.py": "new"},
    )
    stub = _StubPrelude(tmp_path, reader=_kernel_reader(record))

    def _fail_save(*_args, **_kwargs):
        raise OSError("state unavailable")

    stub.shared_state.save = _fail_save

    outcome = await stub._prepare_warm_kernel_kb()

    assert outcome["status"] == "error"
    assert outcome["reason"] == (
        "kernel_attempt_state_persist_failed:OSError"
    )
    assert stub.applied == []
    assert target.read_text(encoding="utf-8") == "old"


@pytest.mark.asyncio
async def test_prepared_state_save_failure_rolls_back_kernel_set(
    tmp_path: Path,
) -> None:
    target = tmp_path / "serving/kernel.py"
    target.parent.mkdir(parents=True)
    target.write_text("old", encoding="utf-8")
    record = _kernel_record(
        tmp_path,
        {"rewrite": {"items": [_rewrite_item(target)]}},
        {"kernel/rewrite/k1.py": "new"},
    )
    stub = _StubPrelude(tmp_path, reader=_kernel_reader(record))
    save_calls = 0
    rollbacks: list[tuple[list[dict], list[dict]]] = []

    def _save(*_args, **_kwargs):
        nonlocal save_calls
        save_calls += 1
        if save_calls >= 4:
            raise OSError("state unavailable")

    stub.shared_state.save = _save
    stub._revert_warm_kernel_patches = (  # type: ignore[method-assign]
        lambda applied, snapshots=None: (
            rollbacks.append((list(applied), list(snapshots or [])))
            or {"ok": True, "errors": []}
        )
    )

    outcome = await stub._prepare_warm_kernel_kb()

    assert outcome["status"] == "error"
    assert outcome["reason"] == (
        "kernel_prepared_state_persist_failed:OSError"
    )
    assert len(rollbacks) == 1
    assert len(rollbacks[0][0]) == 1
    assert len(rollbacks[0][1]) == 1
    assert outcome["pending"] == []
    assert outcome["applied"] == []
    assert stub.shared_state.warm_replay_pending == {}


@pytest.mark.asyncio
async def test_unresolvable_target_defers(tmp_path: Path) -> None:
    record = _kernel_record(
        tmp_path,
        {
            "rewrite": {
                "items": [
                    {
                        "kernel_name": "k1",
                        "patch": "kernel/rewrite/k.diff",
                        "source_files": ["kernel/rewrite/k.py"],
                    }
                ]
            },
            "gemm": {"optimizations": [{"tuned_file": "kernel/gemm/t.json"}]},
        },
        {
            # No diff header -> target cannot be resolved.
            "kernel/rewrite/k.diff": "no header here",
            "kernel/rewrite/k.py": "print(1)",
            "kernel/gemm/t.json": "{}",
        },
    )
    stub = _StubPrelude(tmp_path, reader=_kernel_reader(record))
    stub._validate_warm_kernel_set = _grading(stub, "KEEP")  # type: ignore[method-assign]

    outcome = await stub._prepare_warm_kernel_kb()

    assert outcome["status"] == "loaded"
    assert outcome["pending"] == []
    assert outcome["deferred"] == outcome["total"]
    # Nothing was staged, so nothing was measured.
    assert stub.validations == []


@pytest.mark.asyncio
async def test_set_is_staged_without_a_separate_rebaseline(tmp_path: Path) -> None:
    # Three champions are prepared for the combined Recipe replay task.
    targets = []
    items = []
    for name in ("k1", "k2"):
        target = tmp_path / "serving" / f"{name}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("old", encoding="utf-8")
        targets.append(str(target))
        items.append(_rewrite_item(target, name))
    record = _kernel_record(
        tmp_path,
        {
            "rewrite": {"items": items},
            "gemm": {
                "optimizations": [
                    {
                        "tuned_file": "kernel/gemm/t.csv",
                        "e2e_results": {"kept": [{"env_var": "AITER_CONFIG"}]},
                    }
                ]
            },
        },
        {
            "kernel/rewrite/k1.py": "print('k1')",
            "kernel/rewrite/k2.py": "print('k2')",
            "kernel/gemm/t.csv": "M,N,K\n",
        },
    )
    stub = _StubPrelude(tmp_path, reader=_kernel_reader(record))
    stub._validate_warm_kernel_set = _grading(stub, "KEEP", gain=7.5)  # type: ignore[method-assign]

    outcome = await stub._prepare_warm_kernel_kb()

    assert sorted(stub.applied) == sorted(targets)
    assert stub.validations == []
    assert stub.booked == []
    assert outcome["status"] == "prepared"
    assert len(outcome["pending"]) == 3
    assert outcome["extra_envs"]["AITER_CONFIG"].endswith("kernel/gemm/t.csv")


@pytest.mark.asyncio
async def test_prepared_set_waits_for_combined_verdict(tmp_path: Path) -> None:
    target = tmp_path / "serving" / "kernel.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old", encoding="utf-8")
    record = _kernel_record(
        tmp_path,
        {"rewrite": {"items": [_rewrite_item(target)]}},
        {"kernel/rewrite/k1.py": "print('new')"},
    )
    stub = _StubPrelude(tmp_path, reader=_kernel_reader(record))
    reverted: list[list[dict]] = []
    stub._revert_warm_kernel_patches = (  # type: ignore[method-assign]
        lambda applied, snapshots=None: {
            "ok": not bool(reverted.append(applied)),
            "errors": [],
        }
    )

    outcome = await stub._prepare_warm_kernel_kb()

    assert outcome["status"] == "prepared"
    assert len(outcome["pending"]) == 1
    assert stub.validations == []
    assert reverted == []


@pytest.mark.asyncio
async def test_kernel_snapshot_is_durable_before_first_mutation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "serving" / "kernel.py"
    target.parent.mkdir(parents=True)
    target.write_text("old")
    record = _kernel_record(
        tmp_path,
        {"rewrite": {"items": [_rewrite_item(target)]}},
        {"kernel/rewrite/k1.py": "new"},
    )
    stub = _StubPrelude(tmp_path, reader=_kernel_reader(record))

    def _crash(_entry: dict, live_target: str) -> dict:
        pending = stub.shared_state.warm_replay_pending
        assert pending["kernel_snapshots"][0]["target"] == live_target
        Path(live_target).write_text("partially mutated")
        raise SystemExit("crash window")

    stub._apply_warm_kernel_patch = _crash  # type: ignore[method-assign]
    with pytest.raises(SystemExit, match="crash window"):
        await stub._prepare_warm_kernel_kb()

    restored = PreludePhase._restore_warm_kernel_snapshots(
        stub.shared_state.warm_replay_pending["kernel_snapshots"]
    )
    assert restored["ok"] is True
    assert target.read_text() == "old"


@pytest.mark.asyncio
async def test_symlink_kernel_target_is_rejected_and_prior_mutation_rolls_back(
    tmp_path: Path,
) -> None:
    first = tmp_path / "serving" / "first.py"
    real = tmp_path / "serving" / "real.py"
    link = tmp_path / "serving" / "link.py"
    first.parent.mkdir(parents=True)
    first.write_text("first-old")
    real.write_text("real-old")
    link.symlink_to(real)
    record = _kernel_record(
        tmp_path,
        {
            "rewrite": {
                "items": [
                    _rewrite_item(first, "k1"),
                    _rewrite_item(link, "k2"),
                ]
            }
        },
        {
            "kernel/rewrite/k1.py": "first-new",
            "kernel/rewrite/k2.py": "link-new",
        },
    )
    stub = _StubPrelude(tmp_path, reader=_kernel_reader(record))

    def _apply(_entry: dict, target: str) -> dict:
        Path(target).write_text("mutated")
        stub.applied.append(target)
        return {"status": "ok", "target_file": target}

    stub._apply_warm_kernel_patch = _apply  # type: ignore[method-assign]

    outcome = await stub._prepare_warm_kernel_kb()

    assert outcome["status"] == "error"
    assert outcome["rollback"]["ok"] is True
    assert stub.applied == [str(first)]
    assert first.read_text() == "first-old"
    assert link.is_symlink()
    assert real.read_text() == "real-old"
    assert stub.shared_state.warm_replay_pending == {}


@pytest.mark.asyncio
async def test_empty_kernel_plan_clears_stale_warm_pending(
    tmp_path: Path,
) -> None:
    record = _kernel_record(tmp_path, {"rewrite": {"items": []}}, {})
    stub = _StubPrelude(tmp_path, reader=_kernel_reader(record))
    stub.shared_state.warm_replay_pending = {"status": "preparing_kernel"}

    outcome = await stub._prepare_warm_kernel_kb()

    assert outcome["status"] == "empty"
    assert stub.shared_state.warm_replay_pending == {}


@pytest.mark.asyncio
async def test_loaded_zero_mutation_plan_clears_warm_pending(
    tmp_path: Path,
) -> None:
    missing_target = tmp_path / "missing" / "kernel.py"
    record = _kernel_record(
        tmp_path,
        {"rewrite": {"items": [_rewrite_item(missing_target)]}},
        {"kernel/rewrite/k1.py": "replacement"},
    )
    stub = _StubPrelude(tmp_path, reader=_kernel_reader(record))
    stub.shared_state.warm_replay_pending = {"status": "old"}

    outcome = await stub._prepare_warm_kernel_kb()

    assert outcome["status"] == "loaded"
    assert outcome["pending"] == []
    assert outcome["applied"] == []
    assert stub.shared_state.warm_replay_pending == {}


@pytest.mark.asyncio
async def test_fusion_env_switches_reach_the_measurement(tmp_path: Path) -> None:
    # A fusion patch only takes effect with its recorded env switches; applying
    # the file alone re-measures the unfused path and reverts a good champion.
    target = tmp_path / "serving" / "model.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old", encoding="utf-8")
    record = _kernel_record(
        tmp_path,
        {
            "fusion": {
                "items": [
                    {
                        "kernel_name": "f1",
                        "patch": "kernel/fusion/f.py",
                        "source_file": "kernel/fusion/f.py",
                        "target_path": str(target),
                        "extra_envs": {"SGLANG_USE_AITER": "1"},
                        "env_flags": {"ZAYA_FUSED_HYBRID_RESIDUAL": "1"},
                    }
                ]
            }
        },
        {"kernel/fusion/f.py": "print('fused')"},
    )
    stub = _StubPrelude(tmp_path, reader=_kernel_reader(record))
    stub._validate_warm_kernel_set = _grading(stub, "KEEP", gain=4.0)  # type: ignore[method-assign]

    outcome = await stub._prepare_warm_kernel_kb()

    assert outcome["status"] == "prepared"
    assert outcome["extra_envs"] == {
        "SGLANG_USE_AITER": "1",
        "ZAYA_FUSED_HYBRID_RESIDUAL": "1",
    }


@pytest.mark.asyncio
async def test_gemm_without_env_var_is_deferred(tmp_path: Path) -> None:
    # Nothing names the env var that should carry the tuned table, so there is
    # no safe way to re-apply it: defer rather than guess.
    record = _kernel_record(
        tmp_path,
        {"gemm": {"optimizations": [{"tuned_file": "kernel/gemm/t.csv"}]}},
        {"kernel/gemm/t.csv": "M,N,K\n16,512,7168\n"},
    )
    stub = _StubPrelude(tmp_path, reader=_kernel_reader(record))
    stub._validate_warm_kernel_set = _grading(stub, "KEEP")  # type: ignore[method-assign]

    outcome = await stub._prepare_warm_kernel_kb()

    assert outcome["status"] == "loaded"
    assert outcome["deferred"] == 1 and outcome["pending"] == []
    assert stub.validations == []


@pytest.mark.asyncio
async def test_one_shot_guard(tmp_path: Path) -> None:
    stub = _StubPrelude(
        tmp_path,
        reader=_kernel_reader(_kernel_record(tmp_path, {}, {})),
    )
    stub.shared_state.warm_kernel_kb_attempted = True

    outcome = await stub._prepare_warm_kernel_kb()

    assert outcome == {"status": "skipped", "reason": "already_attempted"}
