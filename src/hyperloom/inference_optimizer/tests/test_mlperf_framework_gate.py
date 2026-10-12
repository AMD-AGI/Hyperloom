# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which frameworks the MLPerf agentic client will drive.

The client speaks OpenAI chat-completions to ``$PORT`` and delegates the server to Magpie's
``{framework}_{gpu}.sh``, so the gate is on framework *kind*, not on a name list.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

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


# --- AGENTX_KEEP_ALIVE_ENV names a variable for the client to export, so it is a shell indirection ---

_CLIENT = Path(__file__).resolve().parents[1] / "assets" / "agentx" / "mlperf_agentic_client.sh"

# The block the client runs, lifted verbatim so the test exercises the shipped text rather than a paraphrase.
_KEEPALIVE_BLOCK = r"""
log() { printf '%s\n' "$*" >&2; }
_KEEPALIVE_S=900
__SNIPPET__
echo "SURVIVED"
"""


def _keepalive_snippet() -> str:
    text = _CLIENT.read_text(encoding="utf-8")
    start = text.index('  if [ -n "${AGENTX_KEEP_ALIVE_ENV:-}" ]; then')
    end = text.index("\n  fi\n", start) + len("\n  fi\n")
    return textwrap.dedent(text[start:end])


def _run(env_name, preset=None):
    env = dict(os.environ, AGENTX_KEEP_ALIVE_ENV=env_name)
    if preset is not None:
        env["ATOM_HTTP_KEEP_ALIVE"] = preset
    return subprocess.run(
        ["bash", "-c", _KEEPALIVE_BLOCK.replace("__SNIPPET__", _keepalive_snippet())],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_the_keep_alive_name_is_blocked_from_variant_envs():
    """An LLM-proposed variant env must not be able to choose which variable the client exports."""
    from hyperloom.common.env_safety import is_allowed_variant_env_key

    assert not is_allowed_variant_env_key("AGENTX_KEEP_ALIVE_ENV")


def test_a_plain_name_is_exported_with_the_default():
    r = _run("ATOM_HTTP_KEEP_ALIVE")
    assert "SURVIVED" in r.stdout
    assert "ATOM_HTTP_KEEP_ALIVE=900" in r.stderr


def test_a_preset_value_is_kept():
    r = _run("ATOM_HTTP_KEEP_ALIVE", preset="120")
    assert "ATOM_HTTP_KEEP_ALIVE=120" in r.stderr


@pytest.mark.parametrize(
    "hostile",
    [
        'X}"; touch /tmp/hyperloom_keepalive_pwned; echo "${Y',  # closes the expansion, then runs a command
        "X; touch /tmp/hyperloom_keepalive_pwned",
        "X$(touch /tmp/hyperloom_keepalive_pwned)",
        "X`touch /tmp/hyperloom_keepalive_pwned`",
        "1STARTS_WITH_DIGIT",
        "has-a-dash",
        "has space",
    ],
)
def test_a_name_that_is_not_an_identifier_is_refused_and_runs_nothing(hostile, tmp_path):
    marker = Path("/tmp/hyperloom_keepalive_pwned")
    marker.unlink(missing_ok=True)
    r = _run(hostile)
    assert r.returncode == 2, r.stdout
    assert "SURVIVED" not in r.stdout
    assert "is not a variable name" in r.stderr
    assert not marker.exists(), "the rejected name still executed a command"
