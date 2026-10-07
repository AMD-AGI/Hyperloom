# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which frameworks the MLPerf agentic client will drive.

The client speaks OpenAI chat-completions to ``$PORT`` and delegates the server to Magpie's
``{framework}_{gpu}.sh``, so the gate is on framework *kind*, not on a name list.
"""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer import framework_registry
from hyperloom.inference_optimizer.agentx.preflight import (
    AgentXPreflightError,
    _check_mlperf_framework,
)

_SERVING = [n for n in framework_registry.names() if not framework_registry.is_scriptable(n)]
_SCRIPTABLE = [n for n in framework_registry.names() if framework_registry.is_scriptable(n)]


@pytest.mark.parametrize("framework", _SERVING)
def test_every_serving_framework_is_accepted(framework):
    """atom is the one this gate was widened for; sglang and vllm must not regress."""
    _check_mlperf_framework({"FRAMEWORK": framework})


def test_atom_is_among_the_accepted_serving_frameworks():
    """Named explicitly: the parametrised case would still pass if atom left the registry."""
    assert "atom" in _SERVING
    _check_mlperf_framework({"FRAMEWORK": "atom"})


@pytest.mark.parametrize("framework", _SCRIPTABLE)
def test_a_scriptable_framework_is_refused(framework):
    """No HTTP endpoint to drive."""
    with pytest.raises(AgentXPreflightError, match="scriptable"):
        _check_mlperf_framework({"FRAMEWORK": framework})


def test_an_unregistered_framework_is_refused():
    with pytest.raises(AgentXPreflightError, match="not a registered framework"):
        _check_mlperf_framework({"FRAMEWORK": "tensorrt-llm"})


def test_an_absent_framework_defers():
    """Magpie resolves the server script; an unset FRAMEWORK is not this gate's to refuse."""
    _check_mlperf_framework({})
    _check_mlperf_framework({"FRAMEWORK": ""})


@pytest.mark.parametrize("raw", ["ATOM", "  atom  ", "Atom"])
def test_the_framework_is_matched_case_and_space_insensitively(raw):
    _check_mlperf_framework({"FRAMEWORK": raw})
