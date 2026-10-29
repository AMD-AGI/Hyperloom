# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""forge-fuse charges its discovery and authoring calls to the run's usage ledger, the way forge-loop charges its own."""

from __future__ import annotations

import subprocess
from pathlib import Path

from kernelforge.agent_backends.base import AgentCapabilities, AgentRunResult, AgentRuntimeConfig
from kernelforge.agent_backends.usage import UsageAccumulator
from kernelforge.agent_backends.usage_ledger import UsageLedgerFile, read_usage_ledger
from kernelforge.workflows.forge_fusion.author import AuthorAgent, AuthorWorkspace, run_author
from kernelforge.workflows.forge_fusion.discover import DiscoverySession, registered_agent_llm_fn


class _PricedBackend:
    """Reports one priced call per run into the usage it is handed, as the registered backends do."""

    name = "claude"
    capabilities = AgentCapabilities(sandbox=True, requires_workspace_cwd=True)
    runtime = AgentRuntimeConfig(provider="claude", model="claude-test")

    def __init__(self, text: str) -> None:
        self.text = text

    async def run(self, spec, usage=None):
        if usage is not None:
            usage.add_usage({"input_tokens": 100, "output_tokens": 10}, total_cost_usd=0.25, role=spec.role)
        return AgentRunResult(text=self.text, end_reason="agent_stopped")


def _ledger_usage(out: Path) -> UsageAccumulator:
    return UsageAccumulator(on_update=UsageLedgerFile(out).publish)


def test_discovery_calls_reach_the_ledger(tmp_path: Path) -> None:
    out = tmp_path / "out"
    llm_fn = registered_agent_llm_fn(
        _PricedBackend('[{"pattern": "residual_add_rmsnorm"}]'),
        DiscoverySession(workdir=str(tmp_path), usage=_ledger_usage(out)),
    )

    llm_fn("propose fusions")

    ledger = read_usage_ledger(out)
    assert ledger["calls"] == 1
    assert ledger["total_cost_usd"] == 0.25


def test_authoring_calls_reach_the_ledger(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "target.py"
    target.write_text("TARGET = 'baseline'\n", encoding="utf-8")
    for args in (
        ["init", "-q"],
        ["add", "target.py"],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base"],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    out = tmp_path / "out"

    rc = run_author(
        "author the fusion",
        AuthorWorkspace(workdir=str(repo), log_path=str(tmp_path / "author.log"), target_files=[str(target)]),
        AuthorAgent(backend=_PricedBackend("AUTHORING_RESULT: ok"), usage=_ledger_usage(out)),
    )

    assert rc == 0
    ledger = read_usage_ledger(out)
    assert ledger["calls"] == 1
    assert ledger["total_cost_usd"] == 0.25
