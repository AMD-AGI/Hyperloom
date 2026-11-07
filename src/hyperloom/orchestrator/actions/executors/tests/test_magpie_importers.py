# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The point, measurement and session modules every round reads reach no module the Magpie removal deletes."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

_EXECUTORS = "hyperloom.orchestrator.actions.executors"
_MAGPIE_ONLY = frozenset(
    {
        f"{_EXECUTORS}._inferencex_patcher",
        f"{_EXECUTORS}._magpie_patcher",
        f"{_EXECUTORS}._recipe_script",
        f"{_EXECUTORS}.benchmark_backend",
        f"{_EXECUTORS}.magpie_agentx",
        f"{_EXECUTORS}.magpie_confined",
        "hyperloom.inference_optimizer.inferencex_checkout",
    }
)
#: Native AgentX points still materialize through Magpie's launch contract until inference_suite translates them.
_STILL_ON_MAGPIE = {f"{_EXECUTORS}._workload_envs": {f"{_EXECUTORS}.magpie_agentx"}}


def _imported(module: str) -> set[str]:
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin is not None
    package = module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(Path(spec.origin).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


@pytest.mark.parametrize(
    "module",
    [
        f"{_EXECUTORS}.benchmark_config",
        f"{_EXECUTORS}.benchmark_result",
        f"{_EXECUTORS}._workload_envs",
        "hyperloom.orchestrator.phases.prelude",
        "hyperloom.orchestrator.loop._resume_consistency",
    ],
)
def test_the_module_imports_nothing_the_magpie_removal_deletes(module: str) -> None:
    assert _imported(module) & _MAGPIE_ONLY == _STILL_ON_MAGPIE.get(module, set())
