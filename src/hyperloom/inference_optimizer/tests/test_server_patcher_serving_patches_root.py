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
the normal answer, and the only override left is ``$KERNELFORGE_PROJECT_ROOT``,
which substitutes the whole data tree rather than a repository checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._server_patcher import _resolve_serving_patches_root


def _fake_tree(root: Path) -> Path:
    tree = root / "serving_patches"
    (tree / "sglang").mkdir(parents=True)
    return tree


@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own data-tree override must not decide these assertions."""
    monkeypatch.delenv("KERNELFORGE_PROJECT_ROOT", raising=False)


def test_packaged_tree_is_used_when_nothing_is_configured() -> None:
    """The stock install must resolve without any environment at all."""
    resolved = _resolve_serving_patches_root(None)

    assert resolved is not None, "the packaged serving_patches tree did not resolve"
    assert resolved.is_dir()
    assert resolved.name == "serving_patches"
    # It is the packaged copy, not something left over on the machine.
    from kernelforge.resources import packaged_data_root

    assert resolved.parent == packaged_data_root()


def test_explicit_root_wins_over_the_packaged_tree(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The caller-supplied root is the highest-precedence override.

    Patching SGLang from anywhere other than the shipped tree is worth a line
    in the log: every failure on this path is silent, so an override that wins
    should not be something you discover by reading a diff months later.
    """
    tree = _fake_tree(tmp_path / "explicit")

    with caplog.at_level("WARNING"):
        assert _resolve_serving_patches_root(tmp_path / "explicit") == tree

    assert any("not the one packaged with kernelforge" in record.message for record in caplog.records)


def test_an_explicit_root_without_the_tree_falls_through_to_the_package(tmp_path: Path) -> None:
    """The override is a preference, not a veto.

    Pointing at a root that carries no ``serving_patches`` used to leave the
    resolver with nothing, which silently dropped the patch entirely.
    """
    empty = tmp_path / "no-patches-here"
    empty.mkdir()

    resolved = _resolve_serving_patches_root(empty)

    assert resolved is not None
    assert resolved.is_dir()


def test_project_root_override_wins_and_is_logged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``$KERNELFORGE_PROJECT_ROOT`` is the surviving env-var override.

    It is how an air-gapped operator drops in a newer sglang patch ahead of an
    image rebuild, so it must beat the packaged copy -- and say that it did.
    """
    project_root = tmp_path / "kernelforge-project"
    tree = _fake_tree(project_root)
    monkeypatch.setenv("KERNELFORGE_PROJECT_ROOT", str(project_root))

    with caplog.at_level("WARNING"):
        assert _resolve_serving_patches_root(None) == tree

    assert any("KERNELFORGE_PROJECT_ROOT override" in record.message for record in caplog.records)


def test_a_project_root_without_the_tree_falls_through_to_the_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A data-tree substitution that carries no patches must not disable them."""
    project_root = tmp_path / "partial-project"
    project_root.mkdir()
    monkeypatch.setenv("KERNELFORGE_PROJECT_ROOT", str(project_root))

    resolved = _resolve_serving_patches_root(None)

    assert resolved is not None
    assert resolved.is_dir()
    from kernelforge.resources import packaged_data_root

    assert resolved.parent == packaged_data_root()


def test_missing_kernelforge_is_fail_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hyperloom must stay usable on a host without the forge extra installed."""
    import builtins

    real_import = builtins.__import__

    def no_kernelforge(name, *args, **kwargs):
        if name.split(".")[0] == "kernelforge":
            raise ImportError("kernelforge is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_kernelforge)

    assert _resolve_serving_patches_root(None) is None
