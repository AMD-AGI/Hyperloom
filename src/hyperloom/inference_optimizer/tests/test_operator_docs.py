# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The operator docs name the flags and the tool-cache variable the code accepts, so a rename fails here."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.orchestrator.actions.executors._suite_config import TOOL_CACHE_ENV

DOCS = Path(__file__).resolve().parents[4] / "docs"
_FLAG = re.compile(r"`(--[a-z][a-z-]*)")


def _optimize_flags() -> set[str]:
    commands = next(action for action in _build_parser()._actions if isinstance(action, argparse._SubParsersAction))
    return {flag for action in commands.choices["optimize"]._actions for flag in action.option_strings}


def test_every_flag_the_agentx_workflow_names_is_an_optimize_flag() -> None:
    section = (DOCS / "how-to" / "optimize.md").read_text(encoding="utf-8").split("## Optimize an AgentX workload")[1]
    named = set(_FLAG.findall(section))
    assert "--no-agentx-accuracy" in named
    assert named <= _optimize_flags()


@pytest.mark.parametrize(
    "doc", ["reference/environment-variables.md", "reference/authentication.md", "install/install.md"]
)
def test_the_tool_cache_variable_is_documented_under_its_code_name(doc: str) -> None:
    assert f"`{TOOL_CACHE_ENV}`" in (DOCS / doc).read_text(encoding="utf-8")
