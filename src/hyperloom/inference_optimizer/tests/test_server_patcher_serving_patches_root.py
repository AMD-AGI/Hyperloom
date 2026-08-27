# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolution of KernelForge's ``serving_patches`` tree.

The SGLang fp8 block-scale CK patch is entirely fail-soft: every miss returns
``None`` and the run continues with ``SGLANG_FP8_BLOCKSCALE_CK_MAX_M`` quietly
no-opping on an unpatched tree. That makes a resolver regression invisible in a
green run -- the patch simply stops being applied and the speedup disappears --
so the resolution order is asserted directly here.

Before KernelForge was vendored into Hyperloom this read ``$FORGE_PATH`` and
nothing else, so an unset env var meant "no patches". The packaged tree is now
the normal answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._server_patcher import _resolve_serving_patches_root


def _fake_checkout(root: Path) -> Path:
    tree = root / "serving_patches"
    (tree / "sglang").mkdir(parents=True)
    return tree


def test_packaged_tree_is_used_when_nothing_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stock install must resolve without any environment at all."""
    monkeypatch.delenv("FORGE_PATH", raising=False)

    resolved = _resolve_serving_patches_root(None)

    assert resolved is not None, "the packaged serving_patches tree did not resolve"
    assert resolved.is_dir()
    assert resolved.name == "serving_patches"
    # It is the packaged copy, not something left over on the machine.
    from kernelforge.resources import packaged_data_root

    assert resolved.parent == packaged_data_root()


def test_forge_path_checkout_wins_over_the_packaged_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A dev override is the whole reason $FORGE_PATH survives."""
    tree = _fake_checkout(tmp_path / "KernelForge")
    monkeypatch.setenv("FORGE_PATH", str(tmp_path / "KernelForge"))

    assert _resolve_serving_patches_root(None) == tree


def test_explicit_root_wins_over_the_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tree = _fake_checkout(tmp_path / "explicit")
    monkeypatch.setenv("FORGE_PATH", str(_fake_checkout(tmp_path / "env").parent))

    assert _resolve_serving_patches_root(tmp_path / "explicit") == tree


def test_a_checkout_without_the_tree_falls_through_to_the_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A $FORGE_PATH that predates serving_patches must not disable the patch.

    The override is a preference, not a veto: pointing at an older checkout
    used to leave the resolver with nothing, which silently dropped the patch.
    """
    empty = tmp_path / "old-checkout"
    empty.mkdir()
    monkeypatch.setenv("FORGE_PATH", str(empty))

    resolved = _resolve_serving_patches_root(None)

    assert resolved is not None
    assert resolved.is_dir()


def test_missing_kernelforge_is_fail_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hyperloom must stay usable on a host without the forge extra installed."""
    monkeypatch.delenv("FORGE_PATH", raising=False)
    import builtins

    real_import = builtins.__import__

    def no_kernelforge(name, *args, **kwargs):
        if name == "kernelforge.resources" or name.split(".")[0] == "kernelforge":
            raise ImportError("kernelforge is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_kernelforge)

    assert _resolve_serving_patches_root(None) is None
