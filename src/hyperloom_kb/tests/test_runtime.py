from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom_kb import (
    PACKAGED_DECLARATION,
    NoOpExperienceKB,
    ReadStatus,
    RemoteClient,
    RemoteClientError,
    RemoteExperienceKB,
    experience_kb_from_env,
    load_declaration,
)


def test_no_configuration_returns_true_noop() -> None:
    kb = experience_kb_from_env({})
    assert isinstance(kb, NoOpExperienceKB)
    assert kb.enabled is False
    assert kb.degraded is False

    session = kb.begin(any_host_argument="is ignored")
    assert session.record is None
    assert session.decide(change={"knob": "x"}, rationale={"reasoning": "ignored"}) is None
    assert session.complete(outcome={"decision": "failed"}, reflection={"text": "ignored"}) is None
    assert session.publish() is None
    assert kb.read("Select the next action.", {}).status is ReadStatus.DISABLED


def test_a_service_url_and_token_build_the_remote_collector(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    flushed: list[RemoteClient] = []
    monkeypatch.setattr(RemoteClient, "flush_spool", lambda self: flushed.append(self) or ())
    env = {"HYPERLOOM_KB_URL": "http://127.0.0.1:8787", "HYPERLOOM_KB_TOKEN": "token"}

    kb = experience_kb_from_env(env)
    placed = experience_kb_from_env(env, spool_root=tmp_path / "spool")

    assert isinstance(kb, RemoteExperienceKB)
    assert kb.schema_ref == load_declaration(PACKAGED_DECLARATION).schema_ref
    assert kb.client.config.spool_root == Path("~/.cache/hyperloom/kb-spool").expanduser()
    assert isinstance(placed, RemoteExperienceKB)
    assert placed.client.config.spool_root == tmp_path / "spool"
    assert flushed == [kb.client, placed.client]


def test_a_service_url_without_a_token_fails_fast() -> None:
    with pytest.raises(RemoteClientError, match="HYPERLOOM_KB_TOKEN"):
        experience_kb_from_env({"HYPERLOOM_KB_URL": "http://127.0.0.1:8787"})
