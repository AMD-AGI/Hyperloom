# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An autonomous branch's runtime records its agent's manifest; the notebook keeps it with every credential dropped,
and the archived search tree and the breakdown carry it."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook import Role
from hyperloom_multibranch.contracts.notebook.interfaces import WriterNotAllowed
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import BranchDriver, LineageKind
from hyperloom_multibranch.notebook.knowledge.breakdown import build_breakdown
from hyperloom_multibranch.notebook.local.agent_manifest import (
    REDACTED,
    is_credential,
    redacted_command,
    sanitized_manifest,
    without_userinfo,
)
from hyperloom_multibranch.notebook.rsi.archive import tree_document, tree_from_document, tree_from_snapshot
from hyperloom_multibranch.notebook.tests.factories import FakeClock
from hyperloom_multibranch.notebook.tests.sessions import METRIC, Writers, scripted_session

SOURCE = "https://github.com/org/agent.git"
MANIFEST: dict[str, object] = {
    "source": "https://bot:ghp_secret@github.com/org/agent.git",
    "commit": "abc123",
    "install": ["pip install -e .", "huggingface-cli login --token hf_abc"],
    "config_file": "campaign.yaml",
    "config": {"model": "/models/Qwen3-8B", "tp": 8, "max_tokens": 512, "server": {"api_key": "sk-1", "port": 30000}},
    "launch": "HF_TOKEN=hf_abc OPENAI_API_KEY='sk-2' python run.py --tokenizer /models/tok --api-key sk-3",
    "resume": "",
    "keep": {"path": "keeps.json", "field": "count", "where": {}},
    "config_source": None,
    "summary_field": "",
    "needs": ["gpu"],
    "ledger": None,
}
AS_GIVEN = ("commit", "config_file", "keep", "config_source", "needs", "ledger")


def test_a_name_holds_a_credential_when_one_of_its_words_says_so() -> None:
    assert all(is_credential(name) for name in ("HF_TOKEN", "apiKey", "OPENAI_API_KEY", "db_password", "secret"))
    assert not any(is_credential(name) for name in ("max_tokens", "tokenizer", "keep", "kv_cache_dtype", "monkey"))


def test_the_manifest_keeps_the_campaign_and_drops_every_credential() -> None:
    kept = sanitized_manifest(MANIFEST)

    assert kept["source"] == SOURCE
    assert kept["config"] == {
        "model": "/models/Qwen3-8B",
        "tp": 8,
        "max_tokens": 512,
        "server": {"api_key": REDACTED, "port": 30000},
    }
    assert kept["install"] == ["pip install -e .", f"huggingface-cli login --token {REDACTED}"]
    assert kept["launch"] == (
        f"HF_TOKEN={REDACTED} OPENAI_API_KEY={REDACTED} python run.py --tokenizer /models/tok --api-key {REDACTED}"
    )
    assert {name: kept[name] for name in AS_GIVEN} == {name: MANIFEST[name] for name in AS_GIVEN}
    assert redacted_command("--api-key=sk-4 --port 1") == f"--api-key={REDACTED} --port 1"
    assert without_userinfo("git@github.com:org/agent.git") == "git@github.com:org/agent.git"


async def test_an_autonomous_branchs_runtime_records_its_agent_and_the_tree_and_the_breakdown_carry_it(
    tmp_path: Path,
) -> None:
    root, clock = tmp_path / "notebook", FakeClock()
    observer = await scripted_session(root, clock)
    w = Writers(root, clock)
    opened = LineageEvent(LineageKind.OPEN, "a1", from_commit="c0", reason="agent", driver=BranchDriver.AUTONOMOUS)
    await w.as_role(Role.SCHEDULER).record_lineage(opened)
    runtime = w.as_role(Role.SPECIALIST_RUNTIME, "a1")

    await runtime.record_agent_manifest("a1", MANIFEST)
    snapshot = await observer.snapshot()
    tree = tree_from_snapshot(snapshot, METRIC)

    record = snapshot.agent_manifests["a1"]
    assert record.manifest == sanitized_manifest(MANIFEST) and record.at > 0
    (agent,) = (branch for branch in tree.branches if branch.name == "a1")
    assert agent.agent == sanitized_manifest(MANIFEST)
    assert all(branch.agent is None for branch in tree.branches if branch.name != "a1")
    assert tree_from_document(tree_document(tree)) == tree
    assert build_breakdown(snapshot)["agent_manifests"]["a1"]["manifest"]["source"] == SOURCE
    with pytest.raises(WriterNotAllowed):
        await runtime.record_agent_manifest("s1", MANIFEST)
    with pytest.raises(WriterNotAllowed):
        await w.as_role(Role.SCHEDULER).record_agent_manifest("a1", MANIFEST)
