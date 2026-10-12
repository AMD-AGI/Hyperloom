# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The DeepSeek Harness SDK pin must admit a release that is actually published."""

from __future__ import annotations

from pathlib import Path

from packaging.requirements import Requirement

try:  # tomllib is stdlib from 3.11; the ``ci`` extra pins tomli for 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]

from hyperloom.common import llm_config

# The SDK has only ever been published as pre-releases.
_PUBLISHED_RELEASE = "0.1.5rc1"

_PYPROJECT = Path(__file__).resolve().parents[4] / "pyproject.toml"


def _requirement(spec: str) -> Requirement:
    requirement = Requirement(spec)
    assert requirement.name == "deepseek-harness-sdk"
    return requirement


def test_the_shared_pin_admits_the_published_release() -> None:
    """A floor above every published version is an extra that can never install."""
    assert _requirement(llm_config.DEEPSEEK_HARNESS_SDK_SPEC).specifier.contains(_PUBLISHED_RELEASE)


def test_the_deepseek_extra_uses_the_shared_pin() -> None:
    """pip reads the extra and preflight reads the constant; both must install the same thing."""
    extras = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert extras["deepseek"] == [
        llm_config.DEEPSEEK_HARNESS_SDK_SPEC,
        llm_config.DEEPSEEK_SESSION_LOG_DECODER_SPEC,
    ]
