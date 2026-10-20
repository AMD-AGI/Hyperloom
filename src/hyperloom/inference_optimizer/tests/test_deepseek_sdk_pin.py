# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The DeepSeek Harness packages: a pin that installs, and installed whenever a launch names the harness."""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement

try:  # tomllib is stdlib from 3.11; the ``ci`` extra pins tomli for 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]

from hyperloom.common import agent_runtime_requirements, llm_config

# The SDK has only ever been published as pre-releases.
_PUBLISHED_RELEASE = "0.1.5rc1"

_PYPROJECT = Path(__file__).resolve().parents[4] / "pyproject.toml"


def _requirement(spec: str) -> Requirement:
    requirement = Requirement(spec)
    assert requirement.name == "deepseek-harness-sdk"
    return requirement


def test_the_shared_pin_admits_the_published_release() -> None:
    """A floor above every published version is an extra that can never install."""
    assert _requirement(agent_runtime_requirements.DEEPSEEK_HARNESS_SDK_SPEC).specifier.contains(_PUBLISHED_RELEASE)


def test_the_deepseek_extra_uses_the_shared_pin() -> None:
    """pip reads the extra and preflight reads the constants; both must install the same thing."""
    extras = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert extras["deepseek"] == [
        agent_runtime_requirements.DEEPSEEK_HARNESS_SDK_SPEC,
        agent_runtime_requirements.DEEPSEEK_SESSION_LOG_DECODER_SPEC,
    ]


@pytest.mark.parametrize(
    ("env", "needed"),
    [
        ({}, False),
        ({llm_config.AGENT_BACKEND_ENV: "codex"}, False),
        ({llm_config.ORCHESTRATION_BACKEND_ENV: "deepseek"}, True),
        ({llm_config.AGENT_BACKEND_ENV: "deepseek"}, True),
        # Specialists, Forge and TraceLens run on the harness even when orchestration does not.
        ({llm_config.ORCHESTRATION_BACKEND_ENV: "claude", llm_config.AGENT_BACKEND_ENV: "deepseek"}, True),
    ],
)
def test_a_launch_that_names_the_harness_for_any_role_installs_it(env: dict[str, str], needed: bool) -> None:
    modules = [module for module, _spec in agent_runtime_requirements.harness_runtime_requirements(env)]
    assert modules == (["deepseek_harness", "zstandard"] if needed else [])
