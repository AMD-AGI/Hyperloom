"""apply_kernel_patch known-target roots (dist-packages / dynamic discovery)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from inference_optimizer.orchestrator import framework_paths as fp

_APPLY_TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "kernel-agent"
    / "tools"
    / "apply_kernel_patch.py"
)


@pytest.fixture(scope="module")
def apply_tool() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_apply_kernel_patch_roots_test", _APPLY_TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_known_target_roots_includes_dist_packages_vllm(
    apply_tool, monkeypatch,
) -> None:
    monkeypatch.setattr(fp, "_discover_installed_framework_roots", lambda: (
        "/usr/local/lib/python3.12/dist-packages/vllm/",
    ))
    monkeypatch.delenv("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", raising=False)
    apply_tool._CACHED_KNOWN_TARGET_ROOTS = None
    roots = apply_tool.known_target_roots()
    assert "/usr/local/lib/python3.12/dist-packages/vllm/" in roots


def test_detect_strategy_accepts_dist_packages_vllm_py(
    apply_tool, monkeypatch,
) -> None:
    target = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/parameter.py",
    )
    monkeypatch.setattr(
        apply_tool,
        "known_target_roots",
        lambda: ("/usr/local/lib/python3.12/dist-packages/vllm/",),
    )
    strat = apply_tool._detect_strategy(target, allow_unknown_target=False)
    assert strat["compiled"] is False
