# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the startup-robustness preflight + launch-info wire format (cli.py)."""

from __future__ import annotations

import argparse
import json
import os
import time

import pytest

from hyperloom_common.llm import llm_config
from hyperloom.inference_optimizer import cli
from hyperloom.inference_optimizer.cli import credentials as cli_credentials
from hyperloom.inference_optimizer.cli import model_catalog
from hyperloom.inference_optimizer import gpu_types

_OAUTH_ENV = "_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN"))
_OPENAI_KEY = "_".join(("OPENAI", "API", "KEY"))
_ANTHROPIC_KEY = "_".join(("ANTHROPIC", "API", "KEY"))
_ANTHROPIC_TOKEN = "_".join(("ANTHROPIC", "AUTH", "TOKEN"))


# _validate_credentials
@pytest.fixture
def clean_creds_env(monkeypatch):
    for var in (
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "_".join(("OPENAI", "API", "KEY")),
        "_".join(("ANTHROPIC", "API", "KEY")),
        "_".join(("ANTHROPIC", "AUTH", "TOKEN")),
        "_".join(("DEEPSEEK", "API", "KEY")),
        "DEEPSEEK_BASE_URL",
        _OAUTH_ENV,
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    "env",
    [
        pytest.param({_OPENAI_KEY: "fake-token", "OPENAI_BASE_URL": "https://gateway.example/v1"}, id="single-gateway"),
        pytest.param(
            {"ANTHROPIC_BASE_URL": "https://api.anthropic.com", _ANTHROPIC_KEY: "anthropic-fake-token"},
            id="anthropic-only-entrypoint",
        ),
        pytest.param({_ANTHROPIC_TOKEN: "anthropic-fake-token"}, id="anthropic-auth-token-only"),
        pytest.param({_ANTHROPIC_KEY: "anthropic-fake-token"}, id="official-anthropic-key-only"),
        pytest.param({_OPENAI_KEY: "openai-fake-token"}, id="official-openai-key-only"),
        pytest.param({_OAUTH_ENV: "sk-ant-oat01-fake"}, id="oauth-token-only"),
        pytest.param(
            {_OAUTH_ENV: "sk-ant-oat01-fake", _OPENAI_KEY: "ak-openai", "OPENAI_BASE_URL": "https://gw.example.com/v1"},
            id="oauth-alongside-configured-openai-side",
        ),
        pytest.param(
            {
                "OPENAI_BASE_URL": "https://gw.example.com/v1",
                "ANTHROPIC_BASE_URL": "https://gw.example.com",
                _OPENAI_KEY: "ak-gw",
                _ANTHROPIC_KEY: "ak-gw",
            },
            id="one-gateway-configured-on-both-sides",
        ),
        pytest.param(
            {
                "OPENAI_BASE_URL": "https://api.openai.com/v1",
                _OPENAI_KEY: "sk-openai",
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
                _ANTHROPIC_KEY: "sk-ant",
            },
            id="dual-entry-with-distinct-keys",
        ),
    ],
)
def test_validate_credentials_accepts_each_legal_shape(clean_creds_env, env):
    """Every configured side has its own key, plus its own base URL or the official one that key implies."""
    for name, value in env.items():
        clean_creds_env.setenv(name, value)
    cli_credentials._validate_credentials()


@pytest.mark.parametrize(
    "env",
    [
        pytest.param(
            {"ANTHROPIC_BASE_URL": "https://api.anthropic.com", _OPENAI_KEY: "openai-fake-token"},
            id="openai-key-with-anthropic-url",
        ),
        pytest.param(
            {_OAUTH_ENV: "sk-ant-oat01-fake", "OPENAI_BASE_URL": "https://gw.example.com/v1"},
            id="oauth-with-bare-openai-base-url",
        ),
        pytest.param(
            {"ANTHROPIC_BASE_URL": "https://gw.example.com/anthropic", _ANTHROPIC_KEY: "gw-key", _OPENAI_KEY: "gw-key"},
            id="gateway-anthropic-url-with-bare-openai-key",
        ),
        pytest.param(
            {"OPENAI_BASE_URL": "https://gw.example.com/v1", _ANTHROPIC_KEY: "sk-ant-real"},
            id="openai-gateway-with-foreign-anthropic-key",
        ),
        pytest.param(
            {"ANTHROPIC_BASE_URL": "https://gw.example.com/anthropic", _OPENAI_KEY: "sk-openai-real"},
            id="anthropic-gateway-with-foreign-openai-key",
        ),
        pytest.param(
            {"OPENAI_BASE_URL": "https://gw.example.com/v1", _OPENAI_KEY: "ak-gw", _ANTHROPIC_KEY: "sk-ant-real"},
            id="gateway-key-plus-foreign-anthropic-key",
        ),
        pytest.param(
            {"OPENAI_BASE_URL": "https://gw.example.com/v1", _OPENAI_KEY: "ak-gw", _ANTHROPIC_KEY: "ak-gw"},
            id="mirrored-key-without-its-own-base-url",
        ),
    ],
)
def test_validate_credentials_rejects_a_cross_provider_pairing(clean_creds_env, capsys, env):
    """Each case leaves one side to borrow the other's key or base URL, which Hyperloom never does."""
    for name, value in env.items():
        clean_creds_env.setenv(name, value)
    with pytest.raises(SystemExit) as exc_info:
        cli_credentials._validate_credentials()
    assert exc_info.value.code == 2
    assert "Conflicting LLM credentials" in capsys.readouterr().err


@pytest.mark.parametrize(
    "anthropic_env",
    [
        pytest.param(("_".join(("ANTHROPIC", "API", "KEY")), "sk-ant-api"), id="api-key"),
        pytest.param((_OAUTH_ENV, "sk-ant-oat01-fake"), id="oauth-token"),
        pytest.param(("_".join(("DEEPSEEK", "API", "KEY")), "sk-deepseek"), id="deepseek-key"),
    ],
)
def test_validate_credentials_accepts_implied_endpoints_on_both_sides(clean_creds_env, anthropic_env):
    """A key that implies its own official endpoint never borrows the other side's, so it pairs with a bare OPENAI_API_KEY."""
    clean_creds_env.setenv(*anthropic_env)
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "ak-openai")
    cli_credentials._validate_credentials()


def test_validate_credentials_warns_when_api_key_shadows_oauth(clean_creds_env, capsys):
    """The Claude CLI prefers the API key, so the subscription would go unused."""
    clean_creds_env.setenv(_OAUTH_ENV, "sk-ant-oat01-fake")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "sk-ant-api")
    cli_credentials._validate_credentials()
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert _OAUTH_ENV in err
    assert "_".join(("ANTHROPIC", "API", "KEY")) in err


def test_validate_credentials_warns_when_oauth_widens_an_openai_only_deploy(clean_creds_env, capsys):
    """A stray token silently moves orchestration off the configured gateway."""
    clean_creds_env.setenv(_OAUTH_ENV, "sk-ant-oat01-fake")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "ak-openai")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    cli_credentials._validate_credentials()
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "OPENAI_BASE_URL" in err


def test_validate_credentials_does_not_warn_when_anthropic_side_was_intended(clean_creds_env, capsys):
    """An explicit Anthropic key means the dual-sided shape was deliberate."""
    clean_creds_env.setenv(_OAUTH_ENV, "sk-ant-oat01-fake")
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "sk-ant-api")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "ak-openai")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    cli_credentials._validate_credentials()
    assert "orchestration runs on the Claude subscription" not in capsys.readouterr().err


def test_validate_credentials_oauth_only_is_not_warned_about(clean_creds_env, capsys):
    clean_creds_env.setenv(_OAUTH_ENV, "sk-ant-oat01-fake")
    cli_credentials._validate_credentials()
    assert "WARNING" not in capsys.readouterr().err


def test_validate_credentials_exits_2_when_no_key(clean_creds_env, capsys):
    """A base URL without any key is rejected."""
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    with pytest.raises(SystemExit) as exc_info:
        cli_credentials._validate_credentials()
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "usable endpoint/key pair" in err


def test_validate_credentials_names_the_missing_pair(clean_creds_env, capsys):
    with pytest.raises(SystemExit):
        cli_credentials._validate_credentials()
    err = capsys.readouterr().err
    missing_line = err.split("Missing required credential(s):")[1].split("\n")[0]
    assert "usable endpoint/key pair" in missing_line
    assert "OPENAI_API_KEY" in missing_line
    assert "CLAUDE_CODE_OAUTH_TOKEN" in missing_line


# _resolve_llm_endpoints
@pytest.mark.parametrize(
    ("env", "anthropic_url", "openai_url"),
    [
        pytest.param(
            {_OAUTH_ENV: "sk-ant-oat01-fake"},
            "https://api.anthropic.com",
            "",
            id="oauth-only-implies-official-anthropic",
        ),
        pytest.param(
            {"OPENAI_BASE_URL": "https://gateway.example/v1"},
            "",
            "https://gateway.example/v1",
            id="openai-only-leaves-anthropic-unset",
        ),
        pytest.param(
            {"ANTHROPIC_BASE_URL": "https://gateway.example/anthropic"},
            "https://gateway.example/anthropic",
            "",
            id="anthropic-only-leaves-openai-unset",
        ),
        pytest.param(
            {_ANTHROPIC_KEY: "anthropic-fake-token"}, "https://api.anthropic.com", "", id="official-anthropic-key-only"
        ),
        pytest.param(
            {_OPENAI_KEY: "openai-fake-token"}, "", "https://api.openai.com/v1", id="official-openai-key-only"
        ),
        pytest.param(
            {
                _OPENAI_KEY: "ak-gw",
                _ANTHROPIC_KEY: "ak-gw",
                "OPENAI_BASE_URL": "https://gw.example.com/api/v1/llm-proxy/v1",
                "ANTHROPIC_BASE_URL": "https://gw.example.com/api/v1/llm-proxy",
            },
            "https://gw.example.com/api/v1/llm-proxy",
            "https://gw.example.com/api/v1/llm-proxy/v1",
            id="one-gateway-under-both-names",
        ),
        pytest.param(
            {
                "ANTHROPIC_BASE_URL": "https://deepseek.example/anthropic",
                "OPENAI_BASE_URL": "https://deepseek.example/v1",
            },
            "https://deepseek.example/anthropic",
            "https://deepseek.example/v1",
            id="dual-protocol-gateway-keeps-both-sides",
        ),
        pytest.param(
            {"ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic", _ANTHROPIC_KEY: "deepseek-fake-token"},
            "https://api.deepseek.com/anthropic",
            "",
            id="deepseek-anthropic-only-leaves-openai-unset",
        ),
        pytest.param(
            {_ANTHROPIC_KEY: "anthropic-fake-token", _OPENAI_KEY: "openai-fake-token"},
            "https://api.anthropic.com",
            "https://api.openai.com/v1",
            id="both-official-keys-no-urls",
        ),
        pytest.param(
            {"ANTHROPIC_BASE_URL": "https://api.anthropic.com", "OPENAI_BASE_URL": "https://api.openai.com/v1"},
            "https://api.anthropic.com",
            "https://api.openai.com/v1",
            id="both-kept-distinct",
        ),
        pytest.param({}, "", "", id="neither-set-returns-empty"),
    ],
)
def test_resolve_llm_endpoints_reads_each_side_from_its_own_env(clean_creds_env, env, anthropic_url, openai_url):
    """Each side resolves to its own base URL, else to the official one its own key implies, else to nothing."""
    for name, value in env.items():
        clean_creds_env.setenv(name, value)
    assert cli_credentials._resolve_llm_endpoints() == (anthropic_url, openai_url)


def test_openai_key_only_makes_claude_follow_codex_before_preflight(clean_creds_env):
    """Key-only official OpenAI must select Codex orchestration before preflight writes OPENAI_BASE_URL."""
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "openai-fake-token")
    assert llm_config.is_openai_only() is True


def test_anthropic_only_makes_codex_follow_claude(clean_creds_env):
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-fake-token")
    assert llm_config.is_anthropic_only() is True


def test_critic_agent_without_a_runtime_root_refuses_to_start(monkeypatch, capsys):
    """No degraded critic: ``--critic-agent`` with no reachable runtime exits 2 before any backend is built."""
    monkeypatch.setattr(cli, "_resolve_agent_root", lambda: None)
    monkeypatch.delenv("CRITIC_KB_CLIENT_MODE", raising=False)
    args = argparse.Namespace(
        no_kernel=False,
        no_framework_agent=False,
        critic_backend="agent",
        critic_protocol="auto",
    )
    with pytest.raises(SystemExit) as exc_info:
        cli._resolve_agent_lineup(args)
    assert exc_info.value.code == 2
    assert "critic-agent runtime not found" in capsys.readouterr().err


def test_build_backends_keeps_anthropic_critic_when_codex_follows_claude(
    clean_creds_env,
    monkeypatch,
    tmp_path,
):
    """Stale OpenAI env after preflight must not force critic-agent onto Codex."""
    from hyperloom.inference_optimizer.cli import backends as cli_backends

    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-fake-token")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://api.anthropic.com")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "stale-openai-token")

    class _FakeClaude:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeCodex:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeCriticAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(cli_backends, "ClaudeBackend", _FakeClaude)
    monkeypatch.setattr(cli_backends, "CodexBackend", _FakeCodex)
    monkeypatch.setattr(cli_backends, "CriticAgentBackend", _FakeCriticAgent)

    built = cli_backends._build_backends(
        cli_backends.ModelRouting(
            claude_model="claude-opus-4-8", codex_model="stale-codex-model", codex_follows_claude=True
        ),
        cli_backends.CriticSetup(choice="agent", agent_root=tmp_path),
        session_dir=tmp_path,
    )

    critic = built["critic"]
    assert isinstance(critic, _FakeCriticAgent)
    assert critic.kwargs["protocol"] == "anthropic"
    assert critic.kwargs["claude_model"] == "claude-opus-4-8"


# _validate_and_resolve_claude_model catalog candidates
def _record_probed_urls(monkeypatch, catalog: set[str]) -> list[str]:
    probed: list[str] = []

    def fake_probe(*, base_url: str, api_key: str):
        probed.append(base_url)
        return catalog if api_key else None

    monkeypatch.setattr(model_catalog, "_probe_llm_catalog", fake_probe)
    return probed


def test_catalog_probe_skips_keyless_anthropic_side_and_uses_the_gateway(clean_creds_env):
    """A stray subscription token must not cost the gateway its verification."""
    clean_creds_env.setenv(_OAUTH_ENV, "sk-ant-oat01-fake")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "ak-openai")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    probed = _record_probed_urls(clean_creds_env, {"claude-opus-5"})
    args = argparse.Namespace(claude_model="claude-opus-5")

    model_catalog._validate_and_resolve_claude_model(
        args, ("https://api.anthropic.com", "https://gw.example.com/v1"), claude_follows_codex=False
    )

    assert probed == ["https://gw.example.com/v1"]


def test_catalog_probe_keeps_anthropic_side_when_it_has_its_own_key(clean_creds_env):
    """A keyed Anthropic side still owns the Claude catalog, gateway untouched."""
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "sk-ant-api")
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "ak-openai")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    probed = _record_probed_urls(clean_creds_env, {"claude-opus-5"})
    args = argparse.Namespace(claude_model="claude-opus-5")

    model_catalog._validate_and_resolve_claude_model(
        args, ("https://api.anthropic.com", "https://gw.example.com/v1"), claude_follows_codex=False
    )

    assert probed == ["https://api.anthropic.com"]


# _resolve_gpu_type
@pytest.mark.parametrize(
    ("user_specified", "probed", "gpu"),
    [
        pytest.param("", "mi355x", "mi355x", id="probe-only"),
        pytest.param("mi300x", "", "mi300x", id="user-only"),
        pytest.param("mi355x", "mi355x", "mi355x", id="agreement-silent"),
        pytest.param("", "", "", id="no-inputs-returns-empty"),
    ],
)
def test_resolve_gpu_type_without_a_disagreement_is_silent(user_specified, probed, gpu):
    """The probe wins when it ran; a failed probe (CPU sandbox) leaves the user's --gpu-type as-is."""
    assert gpu_types._resolve_gpu_type(user_specified=user_specified, probed=probed) == (gpu, [])


def test_resolve_gpu_type_disagreement_probe_always_wins():
    """On disagreement the probe wins unconditionally and warns loudly."""
    gpu, warns = gpu_types._resolve_gpu_type(
        user_specified="mi300x",
        probed="mi355x",
    )
    assert gpu == "mi355x"
    assert len(warns) == 1
    assert "mi300x" in warns[0]
    assert "mi355x" in warns[0]


# _emit_launch_info
def test_emit_launch_info_prints_kv_sentinel(tmp_path, capsys):
    session_dir = tmp_path / "model" / "20260101T000000Z"
    session_dir.mkdir(parents=True)
    cli._emit_launch_info(
        cli.LaunchInfo(
            pid=12345,
            session_dir=session_dir,
            session_id="sess-xyz",
            run_log="/tmp/run.log",
            gpu_type="mi355x",
            framework="sglang",
            model="/models/qwen3",
        ),
        launch_info_file=None,
    )
    out = capsys.readouterr().out
    assert "HYPERLOOM_LAUNCH " in out
    line = [ln for ln in out.splitlines() if ln.startswith("HYPERLOOM_LAUNCH")][0]
    body = line[len("HYPERLOOM_LAUNCH ") :]
    parsed = dict(token.split("=", 1) for token in body.split(" "))
    assert parsed["pid"] == "12345"
    assert parsed["session_dir"] == str(session_dir)
    assert parsed["session_id"] == "sess-xyz"
    assert parsed["gpu_type"] == "mi355x"
    assert parsed["framework"] == "sglang"
    assert parsed["model"] == "/models/qwen3"


def test_emit_launch_info_writes_json_file(tmp_path, capsys):
    session_dir = tmp_path / "sess"
    session_dir.mkdir()
    out_file = tmp_path / "subdir" / "launch.json"
    cli._emit_launch_info(
        cli.LaunchInfo(
            pid=7777,
            session_dir=session_dir,
            session_id="sid",
            run_log="",
            gpu_type="mi300x",
            framework="vllm",
            model="m",
        ),
        launch_info_file=str(out_file),
    )
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["pid"] == 7777
    assert data["session_dir"] == str(session_dir)
    assert data["session_id"] == "sid"
    assert data["framework"] == "vllm"
    assert data["manifest"] == str(session_dir / "manifest.json")
    out = capsys.readouterr().out
    assert "Launch info file" in out
    assert str(out_file) in out


def test_launch_info_is_emitted_on_resume_too():
    """``--resume-from`` reuses the launch template, so its ``--launch-info-file`` must be honoured.

    The health check reads pid and session_dir from that file; a resume that skips it leaves the launcher unable
    to identify the optimizer it just started.
    """
    from .conftest import cli_call_sequence

    for path in ("_resume_session", "_seed_fresh_session"):
        assert "_emit_launch_info" in cli_call_sequence(path), f"{path} does not emit launch info"


# clean_stale_aiter_locks
def _make_aiter_tree(root):
    """Build a minimal aiter jit/build/ layout with mixed lock ages."""
    stale_mtime = time.time() - 30 * 60

    (root / "module_moe" / "build").mkdir(parents=True)
    (root / "module_other" / "build").mkdir(parents=True)

    top_stale = root / "lock_module_moe_stale"
    top_fresh = root / "lock_module_moe_fresh"
    inner_stale_lock = root / "module_moe" / "build" / "lock"
    inner_stale_ninja = root / "module_moe" / "build" / ".ninja_lock"
    inner_non_lock = root / "module_moe" / "build" / "compile_commands.json"
    inner_fresh = root / "module_other" / "build" / "lock"
    bare_so = root / "some_random.so"

    for p, content in (
        (top_stale, "stale"),
        (top_fresh, "fresh"),
        (inner_stale_lock, "stale"),
        (inner_stale_ninja, "stale"),
        (inner_non_lock, "{}"),
        (inner_fresh, "fresh"),
        (bare_so, "fake binary"),
    ):
        p.write_text(content)

    for p in (top_stale, inner_stale_lock, inner_stale_ninja):
        os.utime(p, (stale_mtime, stale_mtime))

    return {
        "stale_top": top_stale,
        "fresh_top": top_fresh,
        "stale_inner_lock": inner_stale_lock,
        "stale_inner_ninja": inner_stale_ninja,
        "non_lock": inner_non_lock,
        "fresh_inner": inner_fresh,
        "bare_so": bare_so,
    }


def test_clean_stale_aiter_locks_deletes_stale_keeps_fresh(tmp_path):
    layout = _make_aiter_tree(tmp_path)
    stats = cli.clean_stale_aiter_locks(
        aiter_jit_dir=tmp_path,
        stale_minutes=5,
    )
    assert stats["deleted"] == 3
    assert stats["skipped_fresh"] == 2
    assert stats["errors"] == 0
    assert not layout["stale_top"].exists()
    assert not layout["stale_inner_lock"].exists()
    assert not layout["stale_inner_ninja"].exists()
    assert layout["fresh_top"].exists()
    assert layout["fresh_inner"].exists()
    assert layout["non_lock"].exists()
    assert layout["bare_so"].exists()


def test_clean_stale_aiter_locks_handles_missing_dir():
    """When aiter cannot be located, return empty stats — never raise."""
    stats = cli.clean_stale_aiter_locks(
        aiter_jit_dir=type("X", (), {"is_dir": lambda self: False})(),
    )
    assert stats["scanned"] == 0
    assert stats["deleted"] == 0


def test_clean_stale_aiter_locks_respects_stale_minutes(tmp_path):
    """Bumping the threshold up keeps moderately-old locks alive."""
    (tmp_path / "lock_module_x").write_text("x")
    moderately_old = time.time() - 4 * 60
    os.utime(tmp_path / "lock_module_x", (moderately_old, moderately_old))
    stats = cli.clean_stale_aiter_locks(
        aiter_jit_dir=tmp_path,
        stale_minutes=10,
    )
    assert stats["deleted"] == 0
    assert stats["skipped_fresh"] == 1
    assert (tmp_path / "lock_module_x").exists()


def test_clean_stale_aiter_locks_auto_discovers_via_env_override(
    tmp_path,
    monkeypatch,
):
    """``$INFERENCE_OPTIMIZER_AITER_JIT_DIR`` resolves when no explicit dir is passed."""
    (tmp_path / "build").mkdir()
    stale_lock = tmp_path / "build" / "lock_module_z"
    stale_lock.write_text("x")
    stale_mtime = time.time() - 30 * 60
    os.utime(stale_lock, (stale_mtime, stale_mtime))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_AITER_JIT_DIR", str(tmp_path))
    stats = cli.clean_stale_aiter_locks(stale_minutes=5)
    assert stats["dir"] in {str(tmp_path), str(tmp_path / "build")}
    assert stats["deleted"] == 1
    assert not stale_lock.exists()
