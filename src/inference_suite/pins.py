# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The commits of the tools a base is measured with, the one place each is pinned.

A pin moves only by a pull request that changes it here, never with the head of a tool's ``main``: two sessions
that resolve their bases a day apart still measure them with the same client. Staging copies each tool at its pin
and a round refuses to run a tool staged at any other commit. lm-eval is pinned by release, the form PyPI serves it in.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

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

#: Every pinned tool's commit, by the name it is staged under.
PINNED_TOOLS: Mapping[str, str] = MappingProxyType({INFERENCEX_TOOL: INFERENCEX_CLIENT_COMMIT})

__all__ = [
    "INFERENCEX_CLIENT_COMMIT",
    "INFERENCEX_REPO_URL",
    "INFERENCEX_TOOL",
    "LM_EVAL_DISTRIBUTION",
    "LM_EVAL_REQUIREMENT",
    "LM_EVAL_VERSION",
    "PINNED_TOOLS",
]
