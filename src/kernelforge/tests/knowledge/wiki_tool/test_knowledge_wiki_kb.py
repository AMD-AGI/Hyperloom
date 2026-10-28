"""The Experience KB settings a knowledge base checkout's scripts get, so its search reads the KB beside the wiki."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom_kb.collect import load_mapping
from kernelforge.config import Config
from kernelforge.knowledge.kb_store.identity.run import campaign_identity
from kernelforge.knowledge.wiki_tool.techniques import TechniqueVocabulary
from kernelforge.knowledge.wiki_tool import KnowledgeWikiTool
from kernelforge.knowledge.wiki_tool import kb
from kernelforge.knowledge.wiki_tool.tool import SERVER_KEY
from kernelforge.tests.knowledge.wiki_tool.test_knowledge_wiki_tool import make_wiki

URL = "http://127.0.0.1:20001"
TOKEN = "kb-secret-token"
CAMPAIGN = {
    "operator_name": "_softmax_kernel",
    "kernel_backend": "triton",
    "gpu_type": "mi300x",
    "framework": "unknown",
    "kernel_path": "softmax_kernel.py",
    "driver_sha256": "ab" * 32,
}


@pytest.fixture
def config(tmp_path, monkeypatch):
    for name in (kb.URL_ENV, kb.TOKEN_ENV, kb.TOKEN_FILE_ENV, kb.SCHEMA_ENV, kb.CONTEXT_ENV, kb.CACHE_ENV):
        monkeypatch.delenv(name, raising=False)
    config = Config(knowledge_wiki_root=str(make_wiki(tmp_path / "wiki")), knowledge_wiki_round=True)
    config.experiments_dir = tmp_path / "experiments"
    return config


def _server_env(config) -> dict[str, str]:
    return KnowledgeWikiTool.from_config(config).mcp_servers("analysis")[SERVER_KEY].env


def test_without_a_kb_url_and_token_the_scripts_get_no_kb_settings(config, monkeypatch):
    monkeypatch.setenv(kb.URL_ENV, URL)

    assert not [name for name in _server_env(config) if name.startswith("HYPERLOOM_KB_")]


def test_the_scripts_get_the_users_kb_with_its_token_in_a_private_file_and_kernelforges_schema(config, monkeypatch):
    monkeypatch.setenv(kb.URL_ENV, URL)
    monkeypatch.setenv(kb.TOKEN_ENV, TOKEN)
    kb.write_kb_context(config.experiments_dir, CAMPAIGN)

    env = _server_env(config)

    assert env[kb.URL_ENV] == URL
    assert kb.TOKEN_ENV not in env and TOKEN not in env.values()
    token_file = Path(env[kb.TOKEN_FILE_ENV])
    assert token_file.read_text() == TOKEN
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    assert env[kb.SCHEMA_ENV] == load_mapping(kb.KERNELFORGE_MAPPING).declaration.schema_ref
    assert json.loads(Path(env[kb.CONTEXT_ENV]).read_text()) == {"identity": campaign_identity(CAMPAIGN)}
    assert campaign_identity(CAMPAIGN)["kernel"] == "softmax"


def test_the_user_may_name_another_schema_a_cache_and_a_token_file(config, monkeypatch, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN)
    monkeypatch.setenv(kb.URL_ENV, URL)
    monkeypatch.setenv(kb.TOKEN_FILE_ENV, str(token_file))
    monkeypatch.setenv(kb.SCHEMA_ENV, "schema:sha256:" + "c" * 64)
    monkeypatch.setenv(kb.CACHE_ENV, str(tmp_path / "cache"))

    env = _server_env(config)

    assert env[kb.TOKEN_FILE_ENV] == str(token_file)
    assert env[kb.SCHEMA_ENV] == "schema:sha256:" + "c" * 64
    assert env[kb.CACHE_ENV] == str(tmp_path / "cache")
    assert kb.CONTEXT_ENV not in env


def test_the_campaign_start_names_its_kernel_and_where_the_kb_search_reads(config, monkeypatch, capsys):
    from kernelforge.workflows.forge_loop.bootstrap.run import _prepare_knowledge_base

    monkeypatch.setenv(kb.URL_ENV, URL)
    monkeypatch.setenv(kb.TOKEN_ENV, TOKEN)

    _prepare_knowledge_base(config, SimpleNamespace(to_dict=lambda: CAMPAIGN))

    out = capsys.readouterr().out
    assert f"Knowledge base KB search: {URL} ({kb.kernelforge_schema_ref()})" in out
    assert TOKEN not in out
    assert (config.experiments_dir / kb.CONTEXT_NAME).is_file()


def test_the_implementer_declares_in_the_knowledge_bases_vocabulary_unless_given_another(config, tmp_path):
    data = Path(config.knowledge_wiki_root) / "data"
    data.mkdir()
    (data / "tags.yaml").write_text("techniques:\n  - split-k\n  - kernel-fusion\n")
    own = tmp_path / "own.yaml"
    own.write_text("techniques:\n  - stream-k\n")

    assert TechniqueVocabulary.from_config(config).terms == ("split-k", "kernel-fusion")
    config.technique_vocabulary = str(own)
    assert TechniqueVocabulary.from_config(config).terms == ("stream-k",)
