# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where the measured server imports each package from, and a patch written into a copy it does not import."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.kernel import campaign_baseline
from hyperloom.orchestrator.kernel.campaign_baseline import served_import_roots
from hyperloom.orchestrator.kernel.served_imports import round_import_roots, unserved_writes
from inference_suite.contracts import RunLayout
from inference_suite.tests.test_compose import _launch


def _round(run_dir: Path, roots: dict[str, str]) -> Path:
    launch = _launch()
    launch = replace(launch, framework=replace(launch.framework, import_roots=roots))
    RunLayout(run_dir).root.mkdir(parents=True)
    RunLayout(run_dir).launch.write_text(json.dumps(launch.to_document()), encoding="utf-8")
    return run_dir


def _package(root: Path, *files: str) -> Path:
    """A package at ``root`` holding ``files``, each subdirectory a package too."""
    for name in ("__init__.py", *files):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".py" and path.parent != root:
            (path.parent / "__init__.py").write_text("", encoding="utf-8")
        path.write_text("", encoding="utf-8")
    return root.resolve()


def test_a_round_records_where_its_server_imported_each_package_from(tmp_path: Path) -> None:
    roots = {"vllm": "/venv/vllm", "aiter": "/venv/aiter"}

    assert round_import_roots(_round(tmp_path / "run", roots)) == roots
    assert round_import_roots(tmp_path) == {}
    assert round_import_roots(None) == {}


def test_the_trees_sealed_are_where_the_measured_server_imports_from(tmp_path: Path) -> None:
    run_dir = _round(tmp_path / "run", {"vllm": "/venv/vllm", "aiter": "/venv/aiter"})
    state = SimpleNamespace(current_best_measurement={"benchmark_workspace": str(run_dir)}, framework="vllm")

    assert served_import_roots(state) == ("/venv/vllm", "/venv/aiter")


def test_a_server_no_round_asked_is_taken_to_import_this_hosts_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(campaign_baseline, "head_import_roots", lambda framework: (f"/head/{framework}",))
    run_dir = _round(tmp_path / "run", {})
    state = SimpleNamespace(current_best_measurement={"benchmark_workspace": str(run_dir)}, framework="vllm")

    assert served_import_roots(state) == ("/head/vllm",)


def test_only_a_python_file_in_a_copy_of_a_probed_package_the_server_does_not_import_is_unserved(
    tmp_path: Path,
) -> None:
    served = _package(tmp_path / "venv" / "aiter", "ops/gemm.py")
    copy = _package(tmp_path / "src" / "aiter", "ops/gemm.py", "csrc/gemm.cu")
    unprobed = _package(tmp_path / "venv" / "triton", "language/core.py")

    written = [
        served / "ops" / "gemm.py",
        copy / "ops" / "gemm.py",
        copy / "csrc" / "gemm.cu",
        unprobed / "language" / "core.py",
    ]

    assert unserved_writes([str(path) for path in written], {"aiter": str(served)}) == [str(copy / "ops" / "gemm.py")]
