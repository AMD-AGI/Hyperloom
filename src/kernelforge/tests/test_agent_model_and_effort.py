# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every agent session takes its model and effort from the runtime.

Two properties are asserted here, and they are the same property seen from two
sides: what a session runs as is a deployment decision, not a call-site one.

* No call site writes ``reasoning_effort=`` on a run spec. One that did would
  be ignored under :meth:`AgentRunSpec.resolved` -- the runtime outranks it --
  and a dead argument that reads like a live one is worse than none, because
  the next reader believes it.
* The context window is applied where the runtime is built and nowhere else,
  and it is empty unless a deployment names one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from kernelforge.agent_backends.base import AgentRunSpec, AgentRuntimeConfig
from kernelforge.agent_backends.model_context import with_context_window
from kernelforge.agent_backends.registry import resolve_agent_runtime

# The vendored tests live inside the package, so the source root is one level
# up from this directory -- not ``parents[1] / "src"`` as it is upstream.
_SRC = Path(__file__).resolve().parents[1]


def _python_sources() -> list[Path]:
    """Every shipped module, excluding the tests and the backends themselves."""
    return [
        path
        for path in sorted(_SRC.rglob("*.py"))
        if "tests" not in path.parts and path.parent.name != "agent_backends"
    ]


def test_no_call_site_pins_a_reasoning_effort() -> None:
    """No shipped module passes ``reasoning_effort=`` to anything."""
    offenders: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "reasoning_effort":
                        offenders.append(f"{path.relative_to(_SRC)}:{keyword.value.lineno}")
            elif isinstance(node, ast.arguments):
                for argument in [*node.args, *node.kwonlyargs, *node.posonlyargs]:
                    if argument.arg == "reasoning_effort":
                        offenders.append(f"{path.relative_to(_SRC)}:{argument.lineno}")
    # ``config``/``cli`` carry the operator's value to the runtime, which is the
    # one direction that is allowed; they are matched by name, not by position,
    # so a new module cannot inherit the exemption by accident.
    allowed = {"config.py", "cli.py", "orchestrator/agent.py", "orchestrator/supervisor.py"}
    unexpected = [entry for entry in offenders if entry.rsplit(":", 1)[0] not in allowed]
    assert not unexpected, "call sites pinning a reasoning effort: " + ", ".join(unexpected)


def test_runtime_effort_outranks_the_spec() -> None:
    """A spec's own effort loses to the runtime's."""
    runtime = AgentRuntimeConfig(provider="claude", model="claude-opus-5", reasoning_effort="medium")
    spec = AgentRunSpec(system_prompt="", user_prompt="", cwd="/tmp", reasoning_effort="max")
    assert spec.resolved(runtime).reasoning_effort == "medium"


def test_spec_effort_survives_a_runtime_that_names_none() -> None:
    """The spec is the fallback, not the loser, when the runtime is silent."""
    runtime = AgentRuntimeConfig(provider="claude", model="claude-opus-5", reasoning_effort="")
    spec = AgentRunSpec(system_prompt="", user_prompt="", cwd="/tmp", reasoning_effort="max")
    assert spec.resolved(runtime).reasoning_effort == "max"


def test_no_context_window_by_default() -> None:
    """An unconfigured deployment gets the plain model id.

    This is the case that matters: the AMD gateway publishes no windowed ids,
    and a session asking for one is rejected outright with "Invalid model
    name", so an unconditional suffix would fail every run.
    """
    runtime = resolve_agent_runtime("claude", model="claude-opus-5")
    assert runtime.model == "claude-opus-5"
    spec = AgentRunSpec(system_prompt="", user_prompt="", cwd="/tmp")
    assert spec.resolved(runtime).model == "claude-opus-5"


def test_configured_context_window_reaches_every_session() -> None:
    """A named window lands on the model id once, for Claude only."""
    runtime = resolve_agent_runtime("claude", model="claude-opus-5", context_window="1m")
    assert runtime.model == "claude-opus-5[1m]"
    spec = AgentRunSpec(system_prompt="", user_prompt="", cwd="/tmp")
    assert spec.resolved(runtime).model == "claude-opus-5[1m]"

    codex = resolve_agent_runtime("codex", model="gpt-5.6", context_window="1m")
    assert codex.model == "gpt-5.6"


@pytest.mark.parametrize(
    ("model", "window", "expected"),
    [
        ("claude-opus-5", "", "claude-opus-5"),
        ("claude-opus-5", "1m", "claude-opus-5[1m]"),
        ("claude-opus-5[1m]", "1m", "claude-opus-5[1m]"),
        ("claude-opus-5[1m]", "200k", "claude-opus-5[1m]"),
        ("  claude-opus-5  ", " 1m ", "claude-opus-5[1m]"),
        ("gpt-5.6", "1m", "gpt-5.6"),
        ("", "1m", ""),
    ],
)
def test_with_context_window(model: str, window: str, expected: str) -> None:
    """The rewrite is Claude-only, idempotent, and a no-op without a window."""
    assert with_context_window(model, window) == expected
