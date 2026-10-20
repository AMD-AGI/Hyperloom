# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Python SDKs a launch's agent runtimes import, and when each is needed."""

from __future__ import annotations

import os
from collections.abc import Mapping

from hyperloom.common.llm_config import AGENT_BACKEND_DEEPSEEK, AGENT_BACKEND_ENV, ORCHESTRATION_BACKEND_ENV

#: Both agent runtimes ship by default: Hyperloom routes every LLM interaction
#: through one of them, and a deployment may be Anthropic-only, OpenAI-only, or both.
_DEFAULT_RUNTIMES: tuple[tuple[str, str], ...] = (
    ("claude_agent_sdk", "claude-agent-sdk>=0.2.110"),
    ("openai_codex", "openai-codex>=0.144"),
    ("openai", "openai>=1.50"),
    ("httpx", "httpx>=0.27"),
)

#: Every published DeepSeek Harness SDK release is a pre-release, so the floor names one: pip
#: admits a pre-release only when the specifier itself does.
DEEPSEEK_HARNESS_SDK_SPEC = "deepseek-harness-sdk>=0.1.5rc1"


def harness_runtime_requirements(env: Mapping[str, str] | None = None) -> tuple[tuple[str, str], ...]:
    """``(module, pip spec)`` pairs this launch needs installed for the DeepSeek Harness.

    Empty unless the launch names the harness, for orchestration or for the
    other roles: no role selects it on a credential shape.
    """
    source = env if env is not None else os.environ
    named = {(source.get(name) or "").strip().lower() for name in (ORCHESTRATION_BACKEND_ENV, AGENT_BACKEND_ENV)}
    if AGENT_BACKEND_DEEPSEEK not in named:
        return ()
    return (("deepseek_harness", DEEPSEEK_HARNESS_SDK_SPEC),)


def agent_runtime_requirements(env: Mapping[str, str] | None = None) -> tuple[tuple[str, str], ...]:
    """``(module, pip spec)`` pairs of every Python SDK this launch's agent runtimes import."""
    return (*_DEFAULT_RUNTIMES, *harness_runtime_requirements(env))
