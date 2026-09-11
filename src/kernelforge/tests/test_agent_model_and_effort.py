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
* The one exception is ``max_reasoning_effort``, which can only lower and is
  spent on a single structurally-non-reasoning call; a second one has to be
  argued for by editing the test that names it.
"""

from __future__ import annotations

import ast
from pathlib import Path


from kernelforge.agent_backends.base import AgentRunSpec, AgentRuntimeConfig
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
    allowed = {
        "config.py",
        "cli.py",
        "orchestrator/agent.py",
        "orchestrator/supervisor.py",
        "fusion/command.py",
        "gemm_tune/tier3/generate.py",
    }
    unexpected = [entry for entry in offenders if entry.rsplit(":", 1)[0] not in allowed]
    assert not unexpected, "call sites pinning a reasoning effort: " + ", ".join(unexpected)


def test_every_runtime_is_built_where_the_switches_are_read() -> None:
    """No module builds an agent runtime outside the list that reads the env.

    Deleting a ``reasoning_effort="high"`` literal makes a call site *look*
    like it defers to the runtime while it still builds its own, which is a
    silently pinned session rather than a converged one. The property that
    actually holds the convergence together is this one: the set of modules
    that construct a runtime is closed, and every member has been checked to
    read ``CLAUDE_MODEL``/``CODEX_MODEL`` and the effort variables. A new
    construction site has to be added here, which is where that check happens.
    """
    builders = {"resolve_agent_runtime", "AgentRuntimeConfig"}
    found: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in builders:
                found.append(str(path.relative_to(_SRC)))
    allowed = {
        "config.py",
        "fusion/command.py",
        "gemm_tune/tier3/generate.py",
        "orchestrator/agent.py",
        "orchestrator/supervisor.py",
    }
    unexpected = sorted(set(found) - allowed)
    assert not unexpected, "runtimes built outside the switch-reading modules: " + ", ".join(unexpected)


def test_runtime_effort_outranks_the_spec() -> None:
    """A spec's own effort loses to the runtime's."""
    runtime = AgentRuntimeConfig(provider="claude", model="claude-opus-5", reasoning_effort="medium")
    spec = AgentRunSpec(system_prompt="", user_prompt="", cwd="/tmp", reasoning_effort="xhigh")
    assert spec.resolved(runtime).reasoning_effort == "medium"


def test_spec_effort_survives_a_runtime_that_names_none() -> None:
    """The spec is the fallback, not the loser, when the runtime is silent."""
    runtime = AgentRuntimeConfig(provider="claude", model="claude-opus-5", reasoning_effort="")
    spec = AgentRunSpec(system_prompt="", user_prompt="", cwd="/tmp", reasoning_effort="xhigh")
    assert spec.resolved(runtime).reasoning_effort == "xhigh"


def test_no_window_suffix_ever_reaches_the_model_id() -> None:
    """Forge names no context window: the id goes out exactly as configured.

    Upstream KernelForge appends ``[1m]`` on the strength of a gateway that
    serves it. The gateway Hyperloom deploys against validates the suffix and
    answers "400 Invalid model name" to every bracketed id, so an appended
    window is not a smaller session but a failed one -- and ``[200k]`` is not
    the spelling of the default window either, the bare id already is. Forge
    has no use for the number on its own: it runs no compaction and no token
    budget, so the suffix was the only thing a window could have driven. The
    plumbing is gone, and this test fails if a re-port brings it back.
    """
    runtime = resolve_agent_runtime("claude", model="claude-opus-5")
    assert runtime.model == "claude-opus-5"
    spec = AgentRunSpec(system_prompt="", user_prompt="", cwd="/tmp")
    assert spec.resolved(runtime).model == "claude-opus-5"
    assert not hasattr(runtime, "context_window")


def test_an_effort_ceiling_only_ever_lowers() -> None:
    """A capped session runs at the cap, never above and never below it."""
    spec = AgentRunSpec(
        system_prompt="",
        user_prompt="",
        cwd="/tmp",
        max_reasoning_effort="low",
    )
    for asked in ("medium", "high", "xhigh", "max"):
        runtime = AgentRuntimeConfig(provider="claude", model="m", reasoning_effort=asked)
        assert spec.resolved(runtime).reasoning_effort == "low"
    # An operator already at the cap keeps their own value: the cap is a
    # ceiling on this call's cost, not a floor under it.
    runtime = AgentRuntimeConfig(provider="claude", model="m", reasoning_effort="low")
    assert spec.resolved(runtime).reasoning_effort == "low"


def test_an_unranked_effort_is_left_alone() -> None:
    """A name outside the ladder reaches the provider unchanged, to be rejected there."""
    spec = AgentRunSpec(system_prompt="", user_prompt="", cwd="/tmp", max_reasoning_effort="low")
    runtime = AgentRuntimeConfig(provider="claude", model="m", reasoning_effort="turbo")
    assert spec.resolved(runtime).reasoning_effort == "turbo"


def test_ordinary_sessions_carry_no_ceiling() -> None:
    """Without a cap the deployment's effort is what runs."""
    spec = AgentRunSpec(system_prompt="", user_prompt="", cwd="/tmp")
    assert spec.max_reasoning_effort == ""
    runtime = AgentRuntimeConfig(provider="claude", model="m", reasoning_effort="high")
    assert spec.resolved(runtime).reasoning_effort == "high"


#: The two places a call site is allowed to cap its own effort, and why.
#:
#: ``plan_critic`` -- the width repair is a parse, not reasoning work, and it is
#: the round's only conditional call; the ceiling is what keeps it a small
#: fraction of the review it follows.
#:
#: ``orchestration`` -- the round partition divides ground the analyses already
#: name, which is a reading task rather than the deepest reasoning the round
#: does, while the plans behind it keep the maximum. Named here as one file
#: rather than one call because reaching the spec from the call site is three
#: keyword pass-throughs (``_run`` -> ``_run_result`` -> ``_run_spec``); the
#: cap is still written in exactly one place, ``ROUND_PARTITION_EFFORT_CEILING``.
_CAPPED_CALL_SITES = {"orchestrator/plan_critic.py", "orchestrator/orchestration.py"}


def test_effort_ceilings_are_confined_to_the_two_argued_call_sites() -> None:
    """A ceiling is an exception; new ones have to be argued for here."""
    offenders: set[str] = set()
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "max_reasoning_effort":
                        offenders.add(str(path.relative_to(_SRC)))
    assert offenders == _CAPPED_CALL_SITES, sorted(offenders)


def test_every_runtime_names_a_model_it_resolved_for_that_provider() -> None:
    """A runtime is built with a model read for the provider being built.

    Two ways to get this wrong, and the tree had one of each: pass no ``model``
    at all and the registry default silently wins over ``CODEX_MODEL``; pass
    the already-resolved ``config.agent_model`` across a provider switch and a
    Claude id reaches the OpenAI-protocol gateway. Both are invisible to the
    closed-set test above, which only says *where* runtimes are built.
    """
    offenders: list[str] = []
    for path in _python_sources():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id != "resolve_agent_runtime":
                continue
            model = next((kw.value for kw in node.keywords if kw.arg == "model"), None)
            where = f"{path.relative_to(_SRC)}:{node.lineno}"
            if model is None:
                offenders.append(f"{where} (no model=)")
                continue
            text = ast.get_source_segment(source, model) or ""
            # ``config.agent_model`` was resolved for whichever provider the
            # config settled on, which is not the one being built here.
            if "config.agent_model" in text:
                offenders.append(f"{where} (carries {text} across a provider switch)")
            elif "resolve_agent_model" not in text and "resolve_agent_model" not in source:
                offenders.append(f"{where} ({text} is not read per-provider)")
    # ``config.py`` holds the resolved value on the dataclass and reads the pair
    # itself once the provider is settled, so it is the one legitimate hand-off.
    offenders = [entry for entry in offenders if not entry.startswith("config.py:")]
    assert not offenders, "runtimes built without a model resolved for their provider: " + ", ".join(offenders)
