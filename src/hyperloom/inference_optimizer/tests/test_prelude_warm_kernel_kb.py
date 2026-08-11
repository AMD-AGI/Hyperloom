# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE read-apply of the independent kernel-agent KB record (validated).

PRELUDE reads the standalone ``kernel:`` KB Store record (flat
``value.gemm``/``value.fusion``/``value.rewrite`` + a ``files/`` tree) through
:class:`KernelRecordReader`, resolves each champion patch's target in the live
source tree, then applies it via ``integrate_handler`` which re-baselines and
KEEPs only on a win. GEMM is parameter-shaped and deferred to the kernel phase.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.knowledge.agent_kb import KernelRecordReader
from hyperloom.orchestrator.phases.prelude import PreludePhase


class _StubPrelude:
    """Minimal carrier binding the prelude warm-kernel methods to a stub state."""

    _collect_warm_kernel_plan = PreludePhase._collect_warm_kernel_plan
    _parse_diff_target = staticmethod(PreludePhase._parse_diff_target)
    _resolve_kernel_target_path = PreludePhase._resolve_kernel_target_path
    _integrate_warm_kernel = PreludePhase._integrate_warm_kernel
    _warm_kernel_extra_envs = staticmethod(PreludePhase._warm_kernel_extra_envs)
    _integrate_warm_gemm = PreludePhase._integrate_warm_gemm
    _maybe_apply_warm_kernel_kb = PreludePhase._maybe_apply_warm_kernel_kb

    def __init__(self, session_dir: Path, reader: object | None = None) -> None:
        self.session_dir = session_dir
        self.shared_state = SimpleNamespace(
            warm_kernel_kb_attempted=False,
            warm_kernel_kb_plan=[],
            warm_kernel_kb_outcome={},
            save=lambda *_a, **_k: None,
        )
        self._reader = reader

    def _open_warm_kernel_record(self) -> object | None:
        """Stubbed record open: return a preloaded reader (or None = cold)."""
        return self._reader


def _kernel_record(tmp_path: Path, value: dict, files: dict[str, str]) -> Path:
    """Write an independent kernel-agent record dir (flat value + files/)."""
    record = tmp_path / "kernel_agent_kb"
    (record / "files").mkdir(parents=True, exist_ok=True)
    (record / "recipe.json").write_text(json.dumps({"value": value}), encoding="utf-8")
    for rel, text in files.items():
        target = record / "files" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return record


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


@pytest.mark.asyncio
async def test_inactive_without_record(tmp_path: Path) -> None:
    # No kernel: record downloaded (cold start / remote KB not configured).
    stub = _StubPrelude(tmp_path, reader=None)

    outcome = await stub._maybe_apply_warm_kernel_kb()

    assert outcome == {"status": "skipped", "reason": "kb_inactive"}


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
    stub = _StubPrelude(tmp_path, reader=KernelRecordReader(record))

    outcome = await stub._maybe_apply_warm_kernel_kb()

    assert outcome["status"] == "loaded"
    assert outcome["kept"] == 0 and outcome["reverted"] == 0
    assert outcome["deferred"] == outcome["total"]
    assert set(outcome["columns"]) == {"gemm", "rewrite"}


@pytest.mark.asyncio
async def test_resolved_target_is_validated_and_kept(tmp_path: Path) -> None:
    target = tmp_path / "serving" / "kernel.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old", encoding="utf-8")
    record = _kernel_record(
        tmp_path,
        {
            "rewrite": {
                "items": [
                    {
                        "kernel_name": "k1",
                        "patch": "kernel/rewrite/k.py",
                        "source_files": ["kernel/rewrite/k.py"],
                        "target_path": str(target),
                    }
                ]
            }
        },
        {"kernel/rewrite/k.py": "print('new')"},
    )
    stub = _StubPrelude(tmp_path, reader=KernelRecordReader(record))

    calls: list[tuple[dict, str]] = []

    async def _fake_integrate(entry: dict, resolved: str) -> dict:
        calls.append((entry, resolved))
        return {"status": "ok", "decision": "KEEP", "gain_pct": 5.0}

    stub._integrate_warm_kernel = _fake_integrate  # type: ignore[method-assign]

    outcome = await stub._maybe_apply_warm_kernel_kb()

    assert outcome["status"] == "kept"
    assert outcome["kept"] == 1
    assert calls and calls[0][1] == str(target)


@pytest.mark.asyncio
async def test_revert_decision_is_counted(tmp_path: Path) -> None:
    target = tmp_path / "serving" / "kernel.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old", encoding="utf-8")
    record = _kernel_record(
        tmp_path,
        {
            "rewrite": {
                "items": [
                    {
                        "kernel_name": "k1",
                        "source_files": ["kernel/rewrite/k.py"],
                        "target_path": str(target),
                    }
                ]
            }
        },
        {"kernel/rewrite/k.py": "print('new')"},
    )
    stub = _StubPrelude(tmp_path, reader=KernelRecordReader(record))

    async def _fake_integrate(entry: dict, resolved: str) -> dict:
        return {"status": "ok", "decision": "REVERT", "gain_pct": -2.0}

    stub._integrate_warm_kernel = _fake_integrate  # type: ignore[method-assign]

    outcome = await stub._maybe_apply_warm_kernel_kb()

    assert outcome["status"] == "reverted"
    assert outcome["reverted"] == 1 and outcome["kept"] == 0


def test_warm_gemm_envs_point_the_recorded_var_at_the_local_tuned_file() -> None:
    entry = {
        "column": "gemm",
        "source_paths": ["/local/dl/tunableop_results.csv"],
        "meta": {
            "recommended_env": {"HL_TUNABLEOP_MODE": "candidate"},
            "e2e_results": {
                "kept": [
                    {
                        "env_var": "PYTORCH_TUNABLEOP_FILENAME",
                        "envs": {"HL_TUNABLEOP_MODE": "candidate"},
                    }
                ]
            },
        },
    }

    envs = PreludePhase._warm_kernel_extra_envs(entry)

    assert envs["PYTORCH_TUNABLEOP_FILENAME"] == "/local/dl/tunableop_results.csv"
    assert envs["HL_TUNABLEOP_MODE"] == "candidate"


def test_warm_gemm_envs_empty_without_a_recorded_env_var() -> None:
    entry = {"column": "gemm", "source_paths": ["/local/dl/t.csv"], "meta": {}}

    assert PreludePhase._warm_kernel_extra_envs(entry) == {}


@pytest.mark.asyncio
async def test_gemm_record_is_applied_through_its_env_bundle(tmp_path: Path) -> None:
    # A GEMM champion is parameter-shaped: it is applied by re-pointing its
    # recorded env var at the downloaded tuned table, then validated.
    record = _kernel_record(
        tmp_path,
        {
            "gemm": {
                "optimizations": [
                    {
                        "variant_name": "forge_vllm_dense_tunableop",
                        "tuned_file": "kernel/gemm/t.csv",
                        "e2e_gain_pct": 10.1,
                        "recommended_env": {"HL_TUNABLEOP_MODE": "candidate"},
                        "e2e_results": {
                            "kept": [{"env_var": "PYTORCH_TUNABLEOP_FILENAME"}]
                        },
                    }
                ]
            }
        },
        {"kernel/gemm/t.csv": "M,N,K\n16,512,7168\n"},
    )
    stub = _StubPrelude(tmp_path, reader=KernelRecordReader(record))

    applied: list[dict] = []

    async def _fake_integrate_gemm(entry: dict) -> dict:
        applied.append(entry)
        return {"status": "ok", "decision": "KEEP", "gain_pct": 10.1}

    stub._integrate_warm_gemm = _fake_integrate_gemm  # type: ignore[method-assign]

    outcome = await stub._maybe_apply_warm_kernel_kb()

    assert outcome["status"] == "kept"
    assert outcome["kept"] == 1 and outcome["deferred"] == 0
    envs = applied[0]["extra_envs"]
    assert envs["PYTORCH_TUNABLEOP_FILENAME"].endswith("kernel/gemm/t.csv")
    assert envs["HL_TUNABLEOP_MODE"] == "candidate"


@pytest.mark.asyncio
async def test_fusion_env_switches_reach_the_integrate_payload(tmp_path: Path) -> None:
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
    stub = _StubPrelude(tmp_path, reader=KernelRecordReader(record))

    seen: list[dict] = []

    async def _fake_integrate(entry: dict, resolved: str) -> dict:
        seen.append(entry)
        return {"status": "ok", "decision": "KEEP", "gain_pct": 4.0}

    stub._integrate_warm_kernel = _fake_integrate  # type: ignore[method-assign]

    outcome = await stub._maybe_apply_warm_kernel_kb()

    assert outcome["kept"] == 1
    assert seen[0]["extra_envs"] == {
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
    stub = _StubPrelude(tmp_path, reader=KernelRecordReader(record))

    outcome = await stub._maybe_apply_warm_kernel_kb()

    assert outcome["status"] == "loaded"
    assert outcome["deferred"] == 1 and outcome["kept"] == 0


@pytest.mark.asyncio
async def test_one_shot_guard(tmp_path: Path) -> None:
    stub = _StubPrelude(tmp_path, reader=KernelRecordReader(_kernel_record(tmp_path, {}, {})))
    stub.shared_state.warm_kernel_kb_attempted = True

    outcome = await stub._maybe_apply_warm_kernel_kb()

    assert outcome == {"status": "skipped", "reason": "already_attempted"}
