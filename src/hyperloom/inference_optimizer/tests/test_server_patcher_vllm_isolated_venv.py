# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""vLLM TraceLens patch discovery for isolated venv installs."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors import _server_patcher as sp


def _make_tracelens_root(tmp_path: Path, version: str = "0.21.0") -> Path:
    tracelens_root = tmp_path / "TraceLens"
    patches = tracelens_root / "examples" / "custom_workflows" / "inference_analysis" / "vllm_patches"
    patches.mkdir(parents=True)
    (patches / f"config_vllm_v{version}.patch").write_text("dummy\n", encoding="utf-8")
    return tracelens_root


def _make_isolated_vllm(
    tmp_path: Path,
    *,
    version: str = "0.21.0",
    with_profiler: bool = True,
) -> Path:
    venv_root = tmp_path / "vllm-venv"
    site = venv_root / "lib" / "python3.12" / "site-packages"
    vllm_pkg = site / "vllm"
    (vllm_pkg / "config").mkdir(parents=True)
    (vllm_pkg / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    (site / f"vllm-{version}.dist-info").mkdir()
    if with_profiler:
        (vllm_pkg / "config" / "profiler.py").write_text("# profiler\n", encoding="utf-8")
    return venv_root


def test_isolated_venv_discovers_plan(tmp_path: Path, monkeypatch):
    tracelens_root = _make_tracelens_root(tmp_path)
    venv_root = _make_isolated_vllm(tmp_path)
    monkeypatch.setenv("VLLM_VENV_ROOT", str(venv_root))
    monkeypatch.delenv("VLLM_PYTHON", raising=False)

    plan = sp._discover_vllm_plan(tracelens_root)

    assert plan is not None
    assert plan.version == "0.21.0"
    assert plan.apply_root == venv_root / "lib" / "python3.12" / "site-packages"
    assert plan.sentinel_file == plan.apply_root / "vllm" / "config" / "profiler.py"


def test_isolated_venv_missing_returns_none(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    monkeypatch.delenv("VLLM_PYTHON", raising=False)

    assert sp._discover_vllm_plan(_make_tracelens_root(tmp_path)) is None


def test_isolated_venv_no_profiler_sentinel_returns_none(tmp_path: Path, monkeypatch):
    tracelens_root = _make_tracelens_root(tmp_path)
    venv_root = _make_isolated_vllm(tmp_path, with_profiler=False)
    monkeypatch.setenv("VLLM_VENV_ROOT", str(venv_root))
    monkeypatch.delenv("VLLM_PYTHON", raising=False)

    assert sp._discover_vllm_plan(tracelens_root) is None
