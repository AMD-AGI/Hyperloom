# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Multi-node detection is env-only for the kernel-agent patch tool.

Regression guard for the hardening that gates multi-node fan-out solely on the
trusted in-process ``$INFERENCE_OPTIMIZER_NODES`` signal, so a co-tenant cannot
force multi-node behavior by planting a world-writable multi-node state file.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_APPLY_TOOL_PATH = Path(__file__).resolve().parent.parent / "tools" / "apply_kernel_patch.py"


@pytest.fixture(scope="module")
def akp() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("_akp_multinode_under_test", _APPLY_TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_is_multi_node_true_when_env_ge_2(akp, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    assert akp._is_multi_node() is True


def test_is_multi_node_false_when_unset_or_single(akp, monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    assert akp._is_multi_node() is False
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")
    assert akp._is_multi_node() is False


def test_is_multi_node_false_on_non_numeric(akp, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "not-an-int")
    assert akp._is_multi_node() is False


def test_is_multi_node_ignores_planted_state_file(akp, monkeypatch, tmp_path):
    planted = tmp_path / "multi_node_state.json"
    planted.write_text('{"nodes": 8}', encoding="utf-8")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(planted))
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    assert akp._is_multi_node() is False
