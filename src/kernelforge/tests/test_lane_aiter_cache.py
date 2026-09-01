# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every Implementer lane compiles into its own AITER build cache.

aiter's ``get_module`` imports a JIT module by name out of ``AITER_JIT_DIR`` and
never checks the ``.so`` against the source it was built from. Two lanes sharing
one cache therefore load each other's binaries: a lane validates and benchmarks
code it did not write, and reports the number as its own.

The lanes run concurrently inside one Forge process, so the routing cannot be a
process-wide ``os.environ`` write -- one lane's write would be the other lane's
too. These tests pin what environment each spawned session actually receives.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.agent_backends.base import (
    AgentRunSpec,
    AgentRuntimeConfig,
    AgentToolPolicy,
)
from kernelforge.agent_backends.claude import ClaudeBackend
from kernelforge.agent_backends.codex import CodexBackend
from kernelforge import cli
from kernelforge.config import Config
from kernelforge.loop import aiter_cache
from kernelforge.loop.fanout import SERIALIZED_DRIVER_NAME
from kernelforge.llm.git import git


@pytest.fixture(autouse=True)
def _isolate_aiter_env():
    """Keep the campaign cache variables this file sets out of later tests.

    ``configure_aiter_cache_isolation`` writes ``os.environ`` directly, and
    monkeypatch cannot roll that back: it only restores keys it recorded, and
    ``delenv`` on an absent key records nothing.
    """
    keys = (
        "AITER_ROOT_DIR",
        "AITER_JIT_DIR",
        "AITER_REBUILD",
        "FORGE_AITER_CACHE_ROOT",
        "FORGE_AITER_CACHE_OWNER_PID",
    )
    saved = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _FakeOptions:
    """Stand-in for ClaudeAgentOptions that records what it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _result_message():
    return SimpleNamespace(
        content=[SimpleNamespace(text="done")],
        total_cost_usd=0.0,
        subtype="success",
        num_turns=1,
        session_id="s",
    )


def _recording_claude_backend(spawned: list[dict[str, str]]) -> ClaudeBackend:
    """A Claude backend whose SDK records the environment a session would get.

    claude-agent-sdk builds the CLI subprocess environment as the inherited
    process environment with ``ClaudeAgentOptions.env`` applied over it, so the
    recorded mapping is what that session's build and benchmark commands read.
    """
    backend = ClaudeBackend.__new__(ClaudeBackend)
    backend.runtime = AgentRuntimeConfig(provider="claude", model="fake-model")
    backend.fallback_reason = ""

    async def fake_query(prompt, options):
        spawned.append({**os.environ, **options.kwargs.get("env", {})})
        yield _result_message()

    backend._query = fake_query
    backend._options_type = _FakeOptions
    return backend


def _implementer_through(backend):
    """Stand in for make_agent_fn, running one real backend session per call."""

    def make_agent(**_kwargs):
        async def agent(kernel_path: str, plan: str) -> str:
            await backend.run(
                AgentRunSpec(
                    system_prompt="implementer",
                    user_prompt=plan,
                    cwd=str(Path(kernel_path).parent),
                    tool_policy=AgentToolPolicy(read=True, write=True, shell=True, max_turns=8),
                )
            )
            return plan

        return agent

    return make_agent


def _campaign(tmp_path: Path) -> tuple[Config, Path]:
    """A campaign workspace with a driver and one declared source file."""
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "forge_driver.py").write_text("pass\n")
    (workspace / "src" / "kernel.py").write_text("VALUE = 0\n")
    return Config(workspace=str(workspace)), workspace


def _lane_factory(config: Config, workspace: Path, make_agent):
    return cli._make_lane_agent_factory(
        make_agent=make_agent,
        config=config,
        workspace_dir=str(workspace),
        driver=str(workspace / "forge_driver.py"),
        source_files=[str(workspace / "src" / "kernel.py")],
        session_kwargs={},
    )


def _lane_dir(tmp_path: Path, lane_id: str) -> Path:
    """A lane copy as the fan-out leaves it: its own repository."""
    lane_dir = tmp_path / "lanes" / lane_id
    (lane_dir / "src").mkdir(parents=True)
    (lane_dir / "src" / "kernel.py").write_text("VALUE = 0\n")
    git("init", "--quiet", cwd=lane_dir)
    git("config", "user.email", "lane@test", cwd=lane_dir)
    git("config", "user.name", "lane", cwd=lane_dir)
    git("add", "-A", cwd=lane_dir)
    git("commit", "-m", "lane baseline", cwd=lane_dir)
    return lane_dir


async def _run_lanes(tmp_path: Path, factory, lane_ids: tuple[str, ...]) -> None:
    """Run one session per lane concurrently, as the round's fan-out does."""

    async def lane(lane_id: str) -> None:
        lane_dir = _lane_dir(tmp_path, lane_id)
        session = factory(str(lane_dir), str(lane_dir / SERIALIZED_DRIVER_NAME))
        # Hand control back so both lanes are inside their session before either
        # spawns: a routing that writes the process environment would give the
        # first lane whatever the second lane wrote last.
        await asyncio.sleep(0)
        await session(str(lane_dir / "src" / "kernel.py"), f"plan {lane_id}")

    await asyncio.gather(*(lane(lane_id) for lane_id in lane_ids))


async def test_two_concurrent_lanes_are_spawned_with_different_aiter_caches(
    tmp_path,
):
    """Two lanes must not be able to load each other's compiled modules."""
    config, workspace = _campaign(tmp_path)
    campaign = aiter_cache.configure_aiter_cache_isolation(tmp_path / "experiments")
    spawned: list[dict[str, str]] = []

    await _run_lanes(
        tmp_path,
        _lane_factory(config, workspace, _implementer_through(_recording_claude_backend(spawned))),
        ("1", "2"),
    )

    assert len(spawned) == 2
    for key in ("AITER_ROOT_DIR", "AITER_JIT_DIR", "FORGE_AITER_CACHE_ROOT"):
        assert len({env[key] for env in spawned}) == 2, key
    assert str(campaign.aiter_root_dir) not in {env["AITER_ROOT_DIR"] for env in spawned}
    assert str(campaign.aiter_jit_dir) not in {env["AITER_JIT_DIR"] for env in spawned}
    assert str(campaign.cache_root) not in {env["FORGE_AITER_CACHE_ROOT"] for env in spawned}


async def test_a_lane_cache_is_created_beside_the_lane_copy(tmp_path):
    """The cache goes where the round's fan-out removes it, but not into the lane.

    Inside the lane copy it would be the lane's own worktree, and every compiled
    artifact would become an untracked file a backend can reject the session for.
    """
    config, workspace = _campaign(tmp_path)
    aiter_cache.configure_aiter_cache_isolation(tmp_path / "experiments")
    spawned: list[dict[str, str]] = []

    await _run_lanes(
        tmp_path,
        _lane_factory(config, workspace, _implementer_through(_recording_claude_backend(spawned))),
        ("1",),
    )

    lane_dir = (tmp_path / "lanes" / "1").resolve()
    cache_root = Path(spawned[0]["FORGE_AITER_CACHE_ROOT"])
    assert cache_root.parent == lane_dir.parent
    assert not cache_root.is_relative_to(lane_dir)
    assert Path(spawned[0]["AITER_ROOT_DIR"]).is_dir()
    assert Path(spawned[0]["AITER_JIT_DIR"]).is_dir()


async def test_running_lanes_leaves_the_campaign_cache_selected(tmp_path):
    """A lane routes its own subprocess, never the Forge process it runs in.

    Everything else in the process -- the canonical correctness run, the
    benchmark, the lock cleanup -- reads these variables, so a lane that wrote
    them would move the campaign's own measurements into the lane's cache.
    """
    config, workspace = _campaign(tmp_path)
    campaign = aiter_cache.configure_aiter_cache_isolation(tmp_path / "experiments")

    await _run_lanes(
        tmp_path,
        _lane_factory(config, workspace, _implementer_through(_recording_claude_backend([]))),
        ("1", "2"),
    )

    assert os.environ["AITER_ROOT_DIR"] == str(campaign.aiter_root_dir)
    assert os.environ["AITER_JIT_DIR"] == str(campaign.aiter_jit_dir)
    assert os.environ["FORGE_AITER_CACHE_ROOT"] == str(campaign.cache_root)


async def test_a_lane_denied_its_own_cache_is_refused_rather_than_shared(tmp_path):
    """A lane that cannot get a private cache must not fall back to the shared one.

    Falling back is the whole defect: the lane would compile into the campaign
    cache beside its siblings and trust whatever module came back.
    """
    config, workspace = _campaign(tmp_path)
    aiter_cache.configure_aiter_cache_isolation(tmp_path / "experiments")
    lane_dir = _lane_dir(tmp_path, "1")
    lane_dir.with_name(lane_dir.name + cli._LANE_AITER_CACHE_SUFFIX).write_text("not a directory\n")

    factory = _lane_factory(config, workspace, _implementer_through(_recording_claude_backend([])))

    with pytest.raises(OSError):
        factory(str(lane_dir), str(lane_dir / SERIALIZED_DRIVER_NAME))


async def test_a_session_outside_a_lane_keeps_the_process_environment(tmp_path):
    """Pin the single-lane path: an ordinary session carries no overlay at all."""
    del tmp_path
    captured: dict = {}
    backend = ClaudeBackend.__new__(ClaudeBackend)
    backend.runtime = AgentRuntimeConfig(provider="claude", model="fake-model")
    backend.fallback_reason = ""

    async def fake_query(prompt, options):
        captured["options"] = options
        yield _result_message()

    backend._query = fake_query
    backend._options_type = _FakeOptions

    await backend.run(
        AgentRunSpec(
            system_prompt="implementer",
            user_prompt="tune it",
            cwd=str(Path.cwd()),
            tool_policy=AgentToolPolicy(read=True, write=True, shell=True),
        )
    )

    assert "env" not in captured["options"].kwargs


async def test_claude_hands_the_session_environment_to_its_cli(tmp_path):
    """The overlay reaches the provider option the SDK spawns the CLI with."""
    workspace = _lane_dir(tmp_path, "claude-env")
    captured: dict = {}
    backend = ClaudeBackend.__new__(ClaudeBackend)
    backend.runtime = AgentRuntimeConfig(provider="claude", model="fake-model")
    backend.fallback_reason = ""

    async def fake_query(prompt, options):
        captured["options"] = options
        yield _result_message()

    backend._query = fake_query
    backend._options_type = _FakeOptions

    await backend.run(
        AgentRunSpec(
            system_prompt="implementer",
            user_prompt="tune it",
            cwd=str(workspace),
            env={"AITER_JIT_DIR": "/lane/1/jit"},
        )
    )

    assert captured["options"].kwargs["env"] == {"AITER_JIT_DIR": "/lane/1/jit"}


def test_codex_hands_the_session_environment_to_its_app_server(tmp_path):
    """Codex spawns its app server with the overlay applied over the child env."""
    backend = CodexBackend(
        runtime=AgentRuntimeConfig(
            provider="codex",
            model="gpt-test",
            options={"home": str(tmp_path / "codex-home")},
        ),
        gateway={
            "base_url": "https://gateway.example.invalid/v1",
            "key_env": "FAKE_CODEX_API_KEY",
            "headers": {"user": "test-user"},
        },
    )
    sdk = SimpleNamespace(CodexConfig=lambda **kwargs: SimpleNamespace(**kwargs))
    spec = AgentRunSpec(
        system_prompt="implementer",
        user_prompt="tune it",
        cwd=str(tmp_path),
        env={"AITER_JIT_DIR": "/lane/1/jit"},
    )

    config = backend._sdk_config(
        sdk=sdk,
        spec=spec,
        child_env={"PATH": "/usr/bin", "AITER_JIT_DIR": "/campaign/jit"},
    )

    assert config.env["AITER_JIT_DIR"] == "/lane/1/jit"
    assert config.env["PATH"] == "/usr/bin"
