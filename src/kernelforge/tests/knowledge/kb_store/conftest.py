# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A workspace holding one Triton kernel, and a local-KB configuration over it."""

from __future__ import annotations

import pytest

from kernelforge.config import Config, KnowledgeConfig
from kernelforge.tests.knowledge.kb_store._flat_experience import KERNEL_SOURCE


@pytest.fixture()
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "kernel.py").write_text(KERNEL_SOURCE, encoding="utf-8")
    return root


@pytest.fixture()
def config(tmp_path, workspace):
    knowledge = KnowledgeConfig.from_env({}, mode="local", local_root=tmp_path / "knowledge")
    return Config.from_env(
        workspace=str(workspace),
        gpu_target="gfx942",
        gpu_type="mi300x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )
