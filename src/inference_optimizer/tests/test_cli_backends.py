"""CLI wiring tests for ``--backend claude`` + ``--auto-install``.

We don't actually launch the SDK or the conductor here; we only verify
that the CLI argument plumbing reaches the right code paths.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from inference_optimizer import cli as cli_mod
from inference_optimizer.bootstrap.errors import MissingDependency


# ---------------------------------------------------------------------------
def _mk_args(**overrides):
    parser = cli_mod._build_argparser()
    args_list = [
        "--model", "Qwen3-8B",
        "--max-hours", "0.001",
        "--backend", "claude",
    ]
    for k, v in overrides.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                args_list.append(flag)
        else:
            args_list.extend([flag, str(v)])
    return parser.parse_args(args_list)


def test_cli_parses_backend_claude_flag():
    args = _mk_args()
    assert args.backend == "claude"
    assert args.auto_install is None


def test_cli_parses_auto_install_flag():
    args = _mk_args(auto_install=True)
    assert args.auto_install is True


def test_cli_resolve_auto_install_from_env(monkeypatch):
    """Post-v0.8: default flipped from False to True. Env var still wins
    over the default but loses to an explicit CLI flag."""
    args = _mk_args()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_AUTO_INSTALL", "1")
    assert cli_mod._resolve_auto_install(args) is True

    monkeypatch.setenv("INFERENCE_OPTIMIZER_AUTO_INSTALL", "yes")
    assert cli_mod._resolve_auto_install(args) is True

    monkeypatch.setenv("INFERENCE_OPTIMIZER_AUTO_INSTALL", "false")
    assert cli_mod._resolve_auto_install(args) is False

    # Default with no env / no flag is now True (not False).
    monkeypatch.delenv("INFERENCE_OPTIMIZER_AUTO_INSTALL", raising=False)
    assert cli_mod._resolve_auto_install(args) is True


def test_cli_resolve_auto_install_no_install_flag_wins(monkeypatch):
    """``--no-auto-install`` must override both the True default and a
    truthy env value."""
    parser = cli_mod._build_argparser()
    args = parser.parse_args([
        "--model", "x", "--max-hours", "0.001",
        "--backend", "claude", "--no-auto-install",
    ])
    assert args.auto_install is False
    monkeypatch.setenv("INFERENCE_OPTIMIZER_AUTO_INSTALL", "1")
    assert cli_mod._resolve_auto_install(args) is False


def test_cli_resolve_auto_install_flag_overrides_env(monkeypatch):
    args = _mk_args(auto_install=True)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_AUTO_INSTALL", "no")
    assert cli_mod._resolve_auto_install(args) is True


def test_cli_bootstrap_for_backend_skips_for_mock(monkeypatch):
    """``--backend mock`` must not call ensure_claude_cli."""
    parser = cli_mod._build_argparser()
    args = parser.parse_args(["--model", "x", "--max-hours", "0.001"])

    called: list[bool] = []

    def fake_ensure(**kwargs):
        called.append(True)
        return None

    monkeypatch.setattr(cli_mod, "ensure_claude_cli", fake_ensure)
    import logging
    log = logging.getLogger("test")
    out = cli_mod._bootstrap_for_backend(args, log)
    assert out is None
    assert called == []


def test_cli_bootstrap_for_backend_calls_ensure(monkeypatch):
    """``--backend claude`` must invoke ensure_claude_cli."""
    args = _mk_args(auto_install=True)

    seen = {}

    class FakeReport:
        def summary(self):
            return "ok"

        class probe_after:  # noqa: N801 — match real attribute access
            node_path = "/tmp/node"
            claude_path = "/tmp/claude"

    def fake_ensure(**kwargs):
        seen.update(kwargs)
        return FakeReport()

    monkeypatch.setattr(cli_mod, "ensure_claude_cli", fake_ensure)
    import logging
    log = logging.getLogger("test")
    out = cli_mod._bootstrap_for_backend(args, log)
    assert out is not None
    assert seen.get("auto_install") is True


def test_cli_bootstrap_surfaces_missing_dependency(monkeypatch, capsys):
    """When deps can't be installed (auto_install effective + bootstrap
    still fails — e.g. offline), print clean instructions and exit 2."""
    parser = cli_mod._build_argparser()
    # Explicit --no-auto-install so the bootstrap path raises immediately
    # rather than trying to download (the test's fake_ensure raises
    # regardless, but this keeps the intent obvious).
    args = parser.parse_args([
        "--model", "Qwen3-8B", "--max-hours", "0.001",
        "--backend", "claude", "--no-auto-install",
    ])

    def fake_ensure(**kwargs):
        raise MissingDependency(
            "node missing\nclaude CLI missing",
            missing=("node", "claude"),
        )

    monkeypatch.setattr(cli_mod, "ensure_claude_cli", fake_ensure)
    import logging
    log = logging.getLogger("test")
    with pytest.raises(SystemExit) as exc:
        cli_mod._bootstrap_for_backend(args, log)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "node missing" in err
    assert "--auto-install" in err


def test_cli_build_backend_mock_returns_mockbackend():
    parser = cli_mod._build_argparser()
    args = parser.parse_args([
        "--model", "x", "--max-hours", "0.001", "--backend", "mock",
    ])
    backend = cli_mod._build_backend(args)
    from inference_optimizer.orchestrator.backends import MockBackend
    assert isinstance(backend, MockBackend)


def test_cli_build_backend_claude_constructs_with_model(monkeypatch):
    args = _mk_args(claude_model="claude-opus-4-7")

    # Stub the SDK import so ClaudeBackend construction succeeds.
    from inference_optimizer.orchestrator.backends import claude as claude_mod

    def fake_import_sdk():
        def fake_query(*, prompt, options):  # pragma: no cover
            yield None

        class FakeOpts:
            def __init__(self, **kw):
                self.kw = kw

        class _FakeSdk:
            __name__ = "fake_sdk"

        return fake_query, FakeOpts, lambda m: "", _FakeSdk()

    monkeypatch.setattr(claude_mod, "_import_sdk", fake_import_sdk)
    backend = cli_mod._build_backend(args)
    from inference_optimizer.orchestrator.backends.claude import ClaudeBackend
    assert isinstance(backend, ClaudeBackend)
    assert backend.model == "claude-opus-4-7"
    # Fake SDK has no `tool` / `create_sdk_mcp_server`; the backend should
    # silently degrade to JSON-in-text mode rather than raising.
    assert backend.has_emit_intent_tool is False


def test_cli_parses_backend_codex_flag():
    parser = cli_mod._build_argparser()
    args = parser.parse_args([
        "--model", "x", "--max-hours", "0.001",
        "--backend", "codex",
        "--codex-model", "gpt-5.4",
        "--codex-base-url", "https://example.invalid/v1",
    ])
    assert args.backend == "codex"
    assert args.codex_model == "gpt-5.4"
    assert args.codex_base_url == "https://example.invalid/v1"


def test_cli_bootstrap_skipped_for_codex_backend(monkeypatch):
    """``--backend codex`` doesn't need Node / claude CLI."""
    parser = cli_mod._build_argparser()
    args = parser.parse_args([
        "--model", "x", "--max-hours", "0.001", "--backend", "codex",
    ])
    called: list[bool] = []
    monkeypatch.setattr(cli_mod, "ensure_claude_cli",
                        lambda **_kw: called.append(True))
    import logging
    out = cli_mod._bootstrap_for_backend(args, logging.getLogger("t"))
    assert out is None
    assert called == []


def test_cli_build_backend_codex_constructs_with_model(monkeypatch):
    parser = cli_mod._build_argparser()
    args = parser.parse_args([
        "--model", "x", "--max-hours", "0.001",
        "--backend", "codex",
        "--codex-model", "gpt-5.4",
    ])

    # Stub the SDK import so CodexBackend construction succeeds without
    # the openai package being importable.
    from inference_optimizer.orchestrator.backends import codex as codex_mod

    class _FakeSdk:
        AsyncOpenAI = object  # never invoked because tests inject sdk_call

    monkeypatch.setattr(codex_mod, "_import_openai_sdk", lambda: _FakeSdk)
    backend = cli_mod._build_backend(args)
    from inference_optimizer.orchestrator.backends.codex import CodexBackend
    assert isinstance(backend, CodexBackend)
    assert backend.model == "gpt-5.4"
    assert backend.base_url is None


def test_cli_build_backend_codex_uses_env_model_when_flag_missing(
    monkeypatch,
):
    parser = cli_mod._build_argparser()
    args = parser.parse_args([
        "--model", "x", "--max-hours", "0.001", "--backend", "codex",
    ])
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    from inference_optimizer.orchestrator.backends import codex as codex_mod
    monkeypatch.setattr(codex_mod, "_import_openai_sdk",
                        lambda: type("S", (), {"AsyncOpenAI": object}))
    backend = cli_mod._build_backend(args)
    assert backend.model == "gpt-5.4-mini"
