# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SGLang plan discovery must fail-soft, not crash, on a namespace-package install.

A PEP 420 namespace-package ``sglang`` install (``sglang.__file__ is None``,
resolved only via ``sglang.__path__``) used to make ``_discover_sglang_plan`` /
``_discover_sglang_ck_plan`` raise ``TypeError`` from ``Path(sglang.__file__)``
instead of returning ``None`` like every other fail-soft condition in this
module. That breaks the documented contract of
``ensure_sglang_patched_for_tracelens`` / ``ensure_sglang_patched_for_ck_blockscale``
("any failure returns False") and crashes the profile run that calls them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from hyperloom.orchestrator.actions.executors import _server_patcher as sp


def _fake_sglang_namespace_package(tmp_path: Path, *, version: str = "0.5.11") -> ModuleType:
    """Build a fake ``sglang`` module that mimics a namespace-package install.

    ``__file__`` is ``None`` (the actual PEP 420 shape); only ``__path__``
    resolves to the package directory.
    """
    pkg_dir = tmp_path / "site-packages" / "sglang"
    (pkg_dir / "srt").mkdir(parents=True)
    fake = ModuleType("sglang")
    fake.__file__ = None  # type: ignore[assignment]
    fake.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
    fake.__version__ = version
    return fake


def _tracelens_root_with_sglang_patches(tmp_path: Path, *, version_subdir: str = "sglang_0_5_11") -> Path:
    tracelens_root = tmp_path / "tracelens"
    patches_dir = tracelens_root / "examples" / "custom_workflows" / "inference_analysis" / "sglang_roofline_patches" / version_subdir
    patches_dir.mkdir(parents=True)
    (patches_dir / "0001-annotations.patch").write_text(
        "--- a/python/sglang/srt/managers/scheduler.py\n"
        "+++ b/python/sglang/srt/managers/scheduler.py\n",
        encoding="utf-8",
    )
    return tracelens_root


def test_sglang_package_dir_falls_back_to_path_when_file_is_none(tmp_path: Path) -> None:
    fake = _fake_sglang_namespace_package(tmp_path)

    resolved = sp._sglang_package_dir(fake)

    assert resolved == Path(fake.__path__[0]).resolve()


def test_sglang_package_dir_is_none_without_file_or_path() -> None:
    fake = ModuleType("sglang")
    fake.__file__ = None  # type: ignore[assignment]
    fake.__path__ = []  # type: ignore[attr-defined]

    assert sp._sglang_package_dir(fake) is None


def test_discover_sglang_plan_does_not_crash_on_namespace_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduces the TypeError this test guards against (pre-fix: crash)."""
    fake = _fake_sglang_namespace_package(tmp_path)
    monkeypatch.setitem(sys.modules, "sglang", fake)
    tracelens_root = _tracelens_root_with_sglang_patches(tmp_path)

    # Must not raise TypeError("... not NoneType ...") from Path(sglang.__file__).
    plan = sp._discover_sglang_plan(tracelens_root)

    assert plan is not None
    assert plan.framework == "sglang"
    assert plan.version == "0.5.11"
    assert plan.apply_root == Path(fake.__path__[0]).resolve()


def test_discover_sglang_ck_plan_does_not_crash_on_namespace_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_sglang_namespace_package(tmp_path)
    monkeypatch.setitem(sys.modules, "sglang", fake)
    # The CK plan requires the patch TARGET file to already exist (the patch
    # edits it in place rather than creating it).
    pkg_dir = Path(fake.__path__[0])
    fp8_utils = pkg_dir / "srt" / "layers" / "quantization" / "fp8_utils.py"
    fp8_utils.parent.mkdir(parents=True)
    fp8_utils.write_text("", encoding="utf-8")
    serving_patches_root = tmp_path / "kernelforge" / "serving_patches"
    ck_patches_dir = serving_patches_root / "sglang" / "sglang_0_5_11"
    ck_patches_dir.mkdir(parents=True)
    (ck_patches_dir / "0001-ck-blockscale.patch").write_text(
        "--- a/python/sglang/srt/layers/quantization/fp8_utils.py\n"
        "+++ b/python/sglang/srt/layers/quantization/fp8_utils.py\n",
        encoding="utf-8",
    )

    plan = sp._discover_sglang_ck_plan(serving_patches_root.parent)

    assert plan is not None
    assert plan.framework == "sglang-ck"
    assert plan.apply_root == Path(fake.__path__[0]).resolve()


def test_discover_sglang_plan_still_works_for_a_regular_file_backed_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard: the common (non-namespace) install path is unchanged."""
    pkg_dir = tmp_path / "site-packages" / "sglang"
    (pkg_dir / "srt").mkdir(parents=True)
    init_file = pkg_dir / "__init__.py"
    init_file.write_text("", encoding="utf-8")
    fake = ModuleType("sglang")
    fake.__file__ = str(init_file)  # type: ignore[assignment]
    fake.__version__ = "0.5.11"
    monkeypatch.setitem(sys.modules, "sglang", fake)
    tracelens_root = _tracelens_root_with_sglang_patches(tmp_path)

    plan = sp._discover_sglang_plan(tracelens_root)

    assert plan is not None
    # Wheel layout: apply_root is the sglang package dir itself, strip=3.
    assert plan.apply_root == pkg_dir.resolve()
    assert plan.apply_strip == 3
