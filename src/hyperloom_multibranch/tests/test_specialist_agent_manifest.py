# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from typing import Any

import pytest

from hyperloom_multibranch.specialist.agent_manifest import (
    AgentManifest,
    ConfigSource,
    KeepSignal,
    LedgerSource,
    ManifestError,
    SnapshotSource,
)
from hyperloom_multibranch.contracts.blackboard.vocab import LearningKind

LEDGER = {
    "path": "sessions/b01/ledger.jsonl",
    "id": "id",
    "kind": "type",
    "text": "claim",
    "kinds": {"FACT": "insight", "DEAD_END": "pitfall"},
}


def _document(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "source": "github.com/acme/agent",
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "install": ["pip install -e ."],
        "config_file": "campaign.json",
        "config": {"model": "/models/qwen", "framework": "sglang"},
        "launch": "python3 -m agent run --config campaign.json",
        "resume": "python3 -m agent run --config campaign.json --resume",
        "keep": {"path": "run/session.json", "field": "lessons", "where": {"decision": "KEEP"}},
        "config_source": {"path": "agent/config.json", "server_args": "server_flags", "envs": "server_env"},
        "summary_field": "lessons.-1.done",
        "needs": ["anthropic"],
        "ledger": LEDGER,
    }
    document.update(overrides)
    return document


def test_a_manifest_reads_back_what_it_writes() -> None:
    manifest = AgentManifest.read(_document())

    assert manifest.keep == KeepSignal("run/session.json", "lessons", {"decision": "KEEP"})
    assert manifest.config_source == ConfigSource("agent/config.json", "server_flags", "server_env")
    kinds = {"FACT": LearningKind.INSIGHT, "DEAD_END": LearningKind.PITFALL}
    assert manifest.ledger == LedgerSource("sessions/b01/ledger.jsonl", "id", "type", "claim", kinds)
    assert AgentManifest.read(json.loads(json.dumps(manifest.to_document()))) == manifest


def test_an_agent_that_keeps_no_ledger_names_none() -> None:
    assert AgentManifest.read({key: value for key, value in _document().items() if key != "ledger"}).ledger is None
    assert AgentManifest.read(_document(ledger=None)).ledger is None


def test_an_agent_with_no_keep_signal_is_read_as_collected_once() -> None:
    assert AgentManifest.read(_document(keep=None)).keep is None


def test_a_null_optional_field_is_empty_not_the_text_none() -> None:
    manifest = AgentManifest.read(_document(resume=None, summary_field=None, keep={"path": "s.json", "field": None}))

    assert (manifest.resume, manifest.summary_field, manifest.keep.counter) == ("", "", "")
    with pytest.raises(ManifestError, match="resume"):
        AgentManifest.read(_document(resume=3))


@pytest.mark.parametrize(
    ("overrides", "names"),
    [
        ({"launch": ""}, "launch"),
        ({"ledger": {**LEDGER, "kinds": {"FACT": "truth"}}}, "ledger.kinds"),
        ({"ledger": {**LEDGER, "path": "../ledger.jsonl"}}, "ledger.path"),
        ({"ledger": {**LEDGER, "text": ""}}, "ledger.text"),
        ({"install": "pip install ."}, "install"),
        ({"config_file": "/etc/agent.json"}, "config_file"),
        ({"keep": {"path": "../outside/session.json"}}, "keep.path"),
        ({"config_source": {"path": "c.json", "server_args": "", "envs": "env"}}, "config_source.server_args"),
        ({"surprise": 1}, "surprise"),
        ({"keep": {"path": "s.json", "fields": "lessons"}}, "fields"),
        ({"result_dir": "results"}, "result_dir"),
    ],
)
def test_a_manifest_that_breaks_the_schema_is_refused_naming_the_field(overrides: dict[str, Any], names: str) -> None:
    with pytest.raises(ManifestError, match=names):
        AgentManifest.read(_document(**overrides))


def test_a_keep_counter_counts_the_matching_items_of_its_field() -> None:
    keep = KeepSignal("s.json", "lessons", {"decision": "KEEP"})
    state = {"lessons": [{"decision": "KEEP"}, {"decision": "REVERT"}, {"decision": "KEEP"}]}

    assert keep.count(state) == 2
    assert KeepSignal("s.json", "keeps").count({"keeps": 5}) == 5


SNAPSHOTS = {"session": "sessions/b01"}


def test_an_agent_that_writes_a_snapshot_per_keep_names_its_session_and_its_outbox_follows_its_config() -> None:
    manifest = AgentManifest.read(_document(keep=None, config_source=None, snapshots=SNAPSHOTS))
    renamed = AgentManifest.read(
        _document(keep=None, config_source=None, snapshots=SNAPSHOTS, config={"component": {"outbox": "snaps"}})
    )

    assert manifest.snapshots == SnapshotSource("sessions/b01")
    assert (manifest.snapshots_outbox, manifest.snapshots_exit) == (
        "sessions/b01/keeps", "sessions/b01/component_exit.json",
    )  # fmt: skip
    assert renamed.snapshots_outbox == "sessions/b01/snaps"
    assert AgentManifest.read(json.loads(json.dumps(manifest.to_document()))) == manifest
    assert AgentManifest.read(_document()).snapshots is None


@pytest.mark.parametrize(
    ("overrides", "names"),
    [
        ({"keep": None, "config_source": None, "snapshots": {"session": "/runs/b01"}}, "snapshots.session"),
        ({"config_source": None, "snapshots": SNAPSHOTS}, "keep"),
        ({"keep": None, "snapshots": SNAPSHOTS}, "config_source"),
        ({"keep": None, "config_source": None, "snapshots": {"session": "s", "outbox": "keeps"}}, "outbox"),
    ],
)
def test_a_snapshot_source_that_breaks_the_schema_is_refused_naming_the_field(
    overrides: dict[str, Any], names: str
) -> None:
    with pytest.raises(ManifestError, match=names):
        AgentManifest.read(_document(**overrides))
