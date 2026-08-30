"""A turn that must write files may not downgrade the configured sandbox.

Observed in a real OpenAI-only forge-loop run, three prep attempts in a row:

    bwrap: Failed to make / slave: Permission denied

    Unable to prepare the driver because every filesystem tool failed during
    sandbox initialization. This blocked both reading the required invocation
    specification and writing driver.py; the placeholder remains unchanged.

The deployment had already resolved ``bypass`` -- the operator's statement that
this process is isolated externally and that no OS-level sandbox is to be built,
which is the only workable answer on a host with no bubblewrap. Each driver
authoring site then pinned the mode to ``workspace-write`` to make sure the turn
could write, and in doing so demanded the very confinement the operator had
opted out of. The turn kept its write permission and lost every file tool.

These tests pin the distinction the pinning lost: raising a read-only policy to a
writable one is required, lowering an already-writable one is not.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentRuntimeConfig,
    with_writable_sandbox,
)
from kernelforge.loop import task_preparer
from kernelforge.rewrite_by_flydsl import (
    flydsl_rewrite_driver_preparation as driver_preparation,
)


def _runtime(sandbox_mode: str) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        provider="codex",
        model="gpt-test",
        sandbox_mode=sandbox_mode,
    )


class _RecordingBackend:
    """Stand in for a provider and remember the runtime it was built with."""

    capabilities = AgentCapabilities(requires_workspace_cwd=False)

    def __init__(self, runtime: AgentRuntimeConfig) -> None:
        self.runtime = runtime

    async def run(self, spec, usage=None):
        return SimpleNamespace(text="done")


def _capture_runtime(monkeypatch, module) -> list[AgentRuntimeConfig]:
    """Record every runtime ``module`` hands to the backend factory."""
    seen: list[AgentRuntimeConfig] = []

    def factory(runtime: AgentRuntimeConfig) -> _RecordingBackend:
        seen.append(runtime)
        return _RecordingBackend(runtime)

    monkeypatch.setattr(module, "create_registered_backend", factory)
    return seen


@pytest.mark.parametrize(
    "sandbox_mode",
    ["bypass", "workspace-write"],
)
def test_with_writable_sandbox_keeps_a_mode_that_already_permits_writes(
    sandbox_mode: str,
) -> None:
    """Leave a writable policy exactly as the operator configured it."""
    runtime = _runtime(sandbox_mode)

    assert with_writable_sandbox(runtime).sandbox_mode == sandbox_mode


def test_with_writable_sandbox_raises_a_read_only_mode() -> None:
    """Grant the least permissive writable policy when writes are forbidden."""
    assert with_writable_sandbox(_runtime("read-only")).sandbox_mode == "workspace-write"


def test_with_writable_sandbox_reads_an_unnormalized_mode() -> None:
    """Recognize a read-only policy however the operator spelled it."""
    assert with_writable_sandbox(_runtime(" Read-Only ")).sandbox_mode == "workspace-write"


def test_with_writable_sandbox_preserves_the_rest_of_the_runtime() -> None:
    """Change the sandbox policy alone, never the provider or the model."""
    runtime = _runtime("read-only")

    raised = with_writable_sandbox(runtime)

    assert (raised.provider, raised.model) == (runtime.provider, runtime.model)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("bypass", "bypass"),
        ("workspace-write", "workspace-write"),
        ("read-only", "workspace-write"),
    ],
)
def test_prepare_agent_never_downgrades_the_configured_sandbox(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configured: str,
    expected: str,
) -> None:
    """Run the prep turn under the configured policy, raised only if read-only."""
    seen = _capture_runtime(monkeypatch, task_preparer)
    config = SimpleNamespace(agent_runtime=lambda: _runtime(configured))

    asyncio.run(
        task_preparer._run_prepare_agent(
            config=config,
            workspace=tmp_path,
            system_prompt="Author the driver.",
            prompt="Write driver.py.",
            timeout_sec=5,
        )
    )

    assert [runtime.sandbox_mode for runtime in seen] == [expected]


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("bypass", "bypass"),
        ("workspace-write", "workspace-write"),
        ("read-only", "workspace-write"),
    ],
)
def test_flydsl_driver_preparation_never_downgrades_the_configured_sandbox(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configured: str,
    expected: str,
) -> None:
    """Apply the same policy to the FlyDSL rewrite driver authoring turn."""
    seen = _capture_runtime(monkeypatch, driver_preparation)
    config = SimpleNamespace(
        agent_runtime=lambda: _runtime(configured),
        max_turns=10,
    )

    asyncio.run(
        driver_preparation._run_agent(
            config=config,
            stage=tmp_path,
            stage_driver=tmp_path / "driver.py",
            evidence_paths=set(),
            prompt="Write driver.py.",
            timeout_sec=5,
            progress_log=[],
        )
    )

    assert [runtime.sandbox_mode for runtime in seen] == [expected]
