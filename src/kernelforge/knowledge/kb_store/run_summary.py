# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one LLM call that summarizes a run for its KB record, and the answer it is held to."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from pydantic import Field

from kernelforge.agent_backends import AgentRunResult, AgentRunSpec, AgentToolPolicy, create_registered_backend
from kernelforge.agent_backends.runtime import agent_runtime
from kernelforge.agent_backends.structured_output import AgentAnswer, structured_answer

log = logging.getLogger(__name__)

_LLM_TIMEOUT_SEC = 150

# Bound the LLM inputs so a huge trajectory / source file can't blow the prompt.
MAX_DIGEST_CHARS = 8000
MAX_SOURCE_CHARS = 6000

#: The operator categories a run's summary files it under.
SUMMARY_CATEGORIES = ("gemm", "attention", "moe", "communication", "others")

_SUMMARY_SYSTEM = (
    "You analyze the trajectory of an autonomous GPU-kernel optimization run and "
    "extract a compact, structured summary. You never write code - you only "
    "report what the winning change did and classify the operator."
)


class RunSummary(AgentAnswer):
    """What one run's winning change did, and the operator it optimized."""

    category: Literal[SUMMARY_CATEGORIES] = Field(description="The operator's category.")
    strategy: str = Field(description="One sentence: the direction of the winning change.")
    recipe: str = Field(description="Concrete, reproducible steps of the winning change.")
    lessons: str = Field(description="Distilled: what worked, what failed, and the pitfalls for this operator.")


async def _query_llm(config, workspace: str, prompt: str, usage=None) -> AgentRunResult:
    """Run one no-edit query through the globally configured agent backend."""
    backend = create_registered_backend(agent_runtime(config))
    return await backend.run(
        AgentRunSpec(
            role="experience summary",
            system_prompt=_SUMMARY_SYSTEM,
            user_prompt=prompt,
            cwd=workspace,
            writable=False,
            timeout_sec=_LLM_TIMEOUT_SEC,
            tool_policy=AgentToolPolicy(
                read=False,
                search=False,
                write=False,
                shell=False,
                max_turns=1,
            ),
            protected_globs=["*"],
            output_schema=RunSummary,
        ),
        usage=usage,
    )


def summarize_run(config, workspace: str, prompt: str, usage=None) -> dict[str, Any]:
    """Summarize a run with one LLM call on ``prompt``; an answer in no usable shape raises ``StructuredOutputError``."""
    result = asyncio.run(
        asyncio.wait_for(
            _query_llm(config, workspace, prompt, usage=usage),
            timeout=_LLM_TIMEOUT_SEC,
        )
    )
    summary = structured_answer(result, RunSummary).model_dump()
    log.info("experience summarize: category=%s strategy=%r", summary["category"], summary["strategy"][:60])
    return summary


def summary_prompt(op: str, digest: str, kernel_source: str) -> str:
    """Build the user prompt asking for a run's summary."""
    digest = (digest or "")[:MAX_DIGEST_CHARS]
    kernel_source = (kernel_source or "")[:MAX_SOURCE_CHARS]
    return f"""\
Operator under optimization: {op}

## Kernel source (target of the run, possibly truncated)
{kernel_source or "(unavailable)"}

## Optimization trajectory (attempts, diffs, per-iteration lessons)
{digest or "(no trajectory recorded)"}

## Your task
Classify the operator and summarize the WINNING change. Leave a field empty
when the trajectory does not show it; never invent.
"""


def normalize_summary(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce a summary the caller supplied into the fields we store, with safe fallbacks."""

    def _str(key: str) -> str:
        val = raw.get(key)
        return val.strip() if isinstance(val, str) and val.strip() else ""

    category = _str("category").lower()
    if category not in SUMMARY_CATEGORIES:
        category = "others"
    return {
        "category": category,
        "strategy": _str("strategy"),
        "recipe": _str("recipe"),
        "lessons": _str("lessons"),
    }
