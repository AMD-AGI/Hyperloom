# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The commits of the tools a base is measured with, the one place each is pinned.

A pin moves only by a pull request that changes it here, never with the head of a tool's ``main``: two sessions
that resolve their bases a day apart still measure them with the same client. Staging copies each tool at its pin
and a round refuses to run a tool staged at any other commit. lm-eval is pinned by release, the form PyPI serves it in,
and the AgentX client's interpreter by the URL and sha256 of its build.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class PinnedArchive:
    """A file staging downloads by URL and accepts only with this sha256; ``version`` is what it unpacks to."""

    url: str
    sha256: str
    version: str


#: The InferenceX checkout whose harness drives load against every base.
INFERENCEX_CLIENT_COMMIT = "41cc713af82bc69204f572e03c2dd8f60485e6ac"
#: The name InferenceX is staged, and recorded in a launch, under.
INFERENCEX_TOOL = "inferencex"
#: Where staging fetches InferenceX at its pin when it is given no checkout.
INFERENCEX_REPO_URL = "https://github.com/SemiAnalysisAI/InferenceX.git"
#: The lm-eval release the eval scores with, installed into the interpreter that runs rounds. The ``runner`` extra in
#: ``pyproject.toml`` pins the same release for the sealed runner's lock.
LM_EVAL_VERSION = "0.4.13"
#: The distribution lm-eval installs as, read back in the interpreter the eval process runs under.
LM_EVAL_DISTRIBUTION = "lm_eval"
#: What staging installs: lm-eval with the ``api`` extra its chat-completions model imports.
LM_EVAL_REQUIREMENT = f"{LM_EVAL_DISTRIBUTION}[api]=={LM_EVAL_VERSION}"
#: The sha256 of the one wheel PyPI serves for that release; ``requirements/runner*.lock`` list it for the sealed runner.
LM_EVAL_WHEEL_SHA256 = "5daaa1973bf874005f64f28d3834b875f6886f0d6475878e6a6c821994a5286a"
#: The relocatable CPython the AgentX client runs under: a python-build-standalone ``install_only`` build for Linux
#: x86_64, unpacked as is, so a staging can be built in one directory and renamed into another.
AGENTX_PYTHON_BUILD = PinnedArchive(
    url=(
        "https://github.com/astral-sh/python-build-standalone/releases/download/20261021/"
        "cpython-3.12.15%2B20261021-x86_64-unknown-linux-gnu-install_only.tar.gz"
    ),
    sha256="5ff9f0602ce964a529ed1e8555d3abab6341926d724402ba9a15e9e877191124",
    version="3.12.15",
)

#: Every pinned tool's commit, by the name it is staged under.
PINNED_TOOLS: Mapping[str, str] = MappingProxyType({INFERENCEX_TOOL: INFERENCEX_CLIENT_COMMIT})

__all__ = [
    "AGENTX_PYTHON_BUILD",
    "INFERENCEX_CLIENT_COMMIT",
    "INFERENCEX_REPO_URL",
    "INFERENCEX_TOOL",
    "LM_EVAL_DISTRIBUTION",
    "LM_EVAL_REQUIREMENT",
    "LM_EVAL_VERSION",
    "LM_EVAL_WHEEL_SHA256",
    "PINNED_TOOLS",
    "PinnedArchive",
]
