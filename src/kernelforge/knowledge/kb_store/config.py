# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The experience-store configuration a caller's runtime config carries."""

from __future__ import annotations

from typing import Any

from kernelforge.config import KnowledgeConfig


def knowledge_config_from_runtime(config: Any) -> KnowledgeConfig:
    """Extract a validated ``KnowledgeConfig`` from the runtime config."""
    if isinstance(config, KnowledgeConfig):
        return config
    parsed = getattr(config, "knowledge_config", None)
    if isinstance(parsed, KnowledgeConfig):
        return parsed
    return KnowledgeConfig.from_env()
