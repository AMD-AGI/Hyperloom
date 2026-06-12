# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for direct-gateway auth setup in ``_preflight``.

Pins the post-auth-proxy contract: legacy ``127.0.0.1:4002`` URLs and stale key
aliases are rewritten to be consistent with ``OPENAI_BASE_URL`` / ``SAFE_API_KEY``.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from typing import Any

import pytest

from inference_optimizer import cli


# _preflight() override semantics
@pytest.fixture
def stub_install_steps(monkeypatch, tmp_path):
    """Stub out heavyweight install steps so _preflight() is fast."""
    monkeypatch.setattr(cli, "_load_dotenv_fallback", lambda: None)
    # N24: stub the kernel-agent env fallback (it hard-fails when missing);
    # its fail-loud behaviour is covered by test_n24_kernel_agent_env_hardfail.
    monkeypatch.setattr(cli, "_load_kernel_agent_env_fallback", lambda: None)

    # InferenceX setup is orthogonal to the auth block under test. On a CI
    # runner the auto-detect finds no checkout and the clone path is a real
    # ``git fetch`` against GitHub (no network / no writable runtime dir),
    # so _preflight() would hit ``sys.exit(2)`` before reaching the auth
    # logic. Point INFERENCEX_PATH at a writable dir so detection short-
    # circuits, and stub the clone as a belt-and-braces fallback.
    inferencex_dir = tmp_path / "InferenceX"
    (inferencex_dir / "benchmarks").mkdir(parents=True)
    (inferencex_dir / "benchmarks" / "benchmark_lib.sh").write_text(
        "# stub", encoding="utf-8"
    )
    monkeypatch.setenv("INFERENCEX_PATH", str(inferencex_dir))
    monkeypatch.setattr(
        cli, "_clone_inferencex", lambda dest: str(inferencex_dir)
    )

    def _fake_which(name: str):
        return f"/usr/bin/{name}"

    monkeypatch.setattr(cli.shutil, "which", _fake_which)

    class _FakeCompleted:
        def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(cmd, *args, **kwargs):
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    return None


@pytest.fixture
def clean_url_env(monkeypatch):
    """Strip the URL env vars so each test starts from a known state."""
    for var in (
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
        "OOB_BASE_URL",
        "GEAK_BASE_URL",
        "LLM_API_BASE",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "OOB_API_KEY",
        "GEAK_API_KEY",
        "LLM_API_KEY",
        "AMD_LLM_API_KEY",
        "SAFE_API_KEY",
        "INFERENCEX_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_derive_anthropic_base_url_strips_openai_v1_suffix():
    assert (
        cli._derive_anthropic_base_url(
            "https://gateway.example/api/v1/llm-proxy/v1/"
        )
        == "https://gateway.example/api/v1/llm-proxy"
    )


def test_preflight_rewrites_legacy_proxy_url_and_auth_aliases(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SAFE_API_KEY", "new-safe-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://gateway.example/api/v1/llm-proxy/v1",
    )
    monkeypatch.setenv(
        "ANTHROPIC_BASE_URL",
        "http://127.0.0.1:4002/api/v1/llm-proxy",
    )
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "OOB_API_KEY",
        "GEAK_API_KEY",
        "LLM_API_KEY",
        "AMD_LLM_API_KEY",
    ):
        monkeypatch.setenv(name, "old-key")
    for name in ("OOB_BASE_URL", "GEAK_BASE_URL", "LLM_API_BASE"):
        monkeypatch.setenv(name, "http://127.0.0.1:4002/api/v1/llm-proxy/v1")

    config_dir = tmp_path / ".claude"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        '{"primaryApiKey":"old-key","customApiUrl":"http://127.0.0.1:4002/v1"}',
        encoding="utf-8",
    )

    resolved = cli._preflight()

    assert resolved == (
        "https://gateway.example/api/v1/llm-proxy",
        "https://gateway.example/api/v1/llm-proxy/v1",
    )
    assert cli.os.environ["ANTHROPIC_BASE_URL"] == resolved[0]
    assert cli.os.environ["OPENAI_BASE_URL"] == resolved[1]
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "OOB_API_KEY",
        "GEAK_API_KEY",
        "LLM_API_KEY",
        "AMD_LLM_API_KEY",
    ):
        assert cli.os.environ[name] == "new-safe-key"
    for name in ("OOB_BASE_URL", "GEAK_BASE_URL", "LLM_API_BASE"):
        assert cli.os.environ[name] == resolved[1]

    config_text = (config_dir / "config.json").read_text(encoding="utf-8")
    assert "127.0.0.1:4002" not in config_text
    assert '"primaryApiKey": "new-safe-key"' in config_text
    assert '"customApiUrl": "https://gateway.example/api/v1/llm-proxy"' in config_text


def test_preflight_preserves_operator_geak_tunnel_url(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    """#521: an operator-pinned GEAK/OOB tunnel URL survives preflight.

    GEAK runs in a separate network namespace that cannot reach the gateway
    directly; the operator points GEAK_BASE_URL at the host-local reverse
    tunnel. Preflight must NOT clobber it back to the direct gateway URL, while
    still defaulting the unset LLM_API_BASE to the gateway.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SAFE_API_KEY", "safe-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://gateway.example/api/v1/llm-proxy/v1",
    )
    tunnel = "https://127.0.0.1:18444/api/v1/llm-proxy/v1"
    monkeypatch.setenv("GEAK_BASE_URL", tunnel)
    monkeypatch.setenv("OOB_BASE_URL", tunnel)
    # LLM_API_BASE left unset → should default to the gateway.

    resolved = cli._preflight()

    gateway = resolved[1]
    # Operator tunnel preserved.
    assert cli.os.environ["GEAK_BASE_URL"] == tunnel
    assert cli.os.environ["OOB_BASE_URL"] == tunnel
    # Unset alias still defaults to the gateway.
    assert cli.os.environ["LLM_API_BASE"] == gateway


def test_preflight_rewrites_stale_proxy_even_when_operator_set(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    """A leftover 127.0.0.1:4002 value is force-rewritten, not preserved."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SAFE_API_KEY", "safe-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://gateway.example/api/v1/llm-proxy/v1",
    )
    monkeypatch.setenv(
        "GEAK_BASE_URL", "http://127.0.0.1:4002/api/v1/llm-proxy/v1",
    )

    resolved = cli._preflight()

    assert cli.os.environ["GEAK_BASE_URL"] == resolved[1]
    assert "127.0.0.1:4002" not in cli.os.environ["GEAK_BASE_URL"]


def test_is_stale_proxy_url_matches_legacy_only():
    assert cli._is_stale_proxy_url("http://127.0.0.1:4002/api/v1/llm-proxy/v1")
    assert not cli._is_stale_proxy_url("https://127.0.0.1:18444/api/v1/llm-proxy/v1")
    assert not cli._is_stale_proxy_url("https://gateway.example/v1")
    assert not cli._is_stale_proxy_url("")
    assert not cli._is_stale_proxy_url(None)


# _ensure_python_sdks
class _RecordingRun:
    """Test double for subprocess.run that records calls and replays a script."""

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls: list[list[str]] = []

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(list(cmd))
        if not self.script:
            raise AssertionError(f"unexpected subprocess call: {cmd}")
        item = self.script.pop(0)
        if callable(item):
            return item(cmd, *args, **kwargs)
        return item


class _Completed:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_ensure_python_sdks_skips_when_all_present(monkeypatch, capsys):
    """All three import-checks return rc=0 → no pip install fires."""
    runner = _RecordingRun([
        _Completed(returncode=0),
        _Completed(returncode=0),
        _Completed(returncode=0),
    ])
    monkeypatch.setattr(cli.subprocess, "run", runner)

    cli._ensure_python_sdks("/opt/venv/bin/python", [])

    assert len(runner.calls) == 3
    for call in runner.calls:
        assert call[0] == "/opt/venv/bin/python"
        assert call[1] == "-c"
        assert call[2].startswith("import ")
    captured = capsys.readouterr().out
    assert "claude_agent_sdk OK" in captured
    assert "openai OK" in captured
    assert "httpx OK" in captured


def test_ensure_python_sdks_installs_missing_claude_agent_sdk(monkeypatch, capsys):
    """When `import claude_agent_sdk` fails, pip install runs with the SAME interpreter."""
    runner = _RecordingRun([
        _Completed(returncode=1),
        _Completed(returncode=0),
        _Completed(returncode=0),
        _Completed(returncode=0),
    ])
    monkeypatch.setattr(cli.subprocess, "run", runner)

    cli._ensure_python_sdks("/opt/venv/bin/python", ["--break-system-packages"])

    assert len(runner.calls) == 4
    install_call = runner.calls[1]
    assert install_call[0] == "/opt/venv/bin/python"
    assert install_call[1:4] == ["-m", "pip", "install"]
    assert "--break-system-packages" in install_call
    assert any(arg.startswith("claude-agent-sdk") for arg in install_call)
    captured = capsys.readouterr().out
    assert "installing claude-agent-sdk" in captured
    assert "installed claude-agent-sdk" in captured


# _unset_hip_visible_devices
def test_unset_hip_visible_devices_pops_when_rocr_present(monkeypatch, capsys):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3")

    cli._unset_hip_visible_devices()

    import os as _os
    assert "HIP_VISIBLE_DEVICES" not in _os.environ
    assert _os.environ["ROCR_VISIBLE_DEVICES"] == "0,1,2,3"
    assert "WARNING" in capsys.readouterr().out


def test_unset_hip_visible_devices_keeps_hip_when_rocr_unset(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)

    cli._unset_hip_visible_devices()

    import os as _os
    assert _os.environ["HIP_VISIBLE_DEVICES"] == "0,1,2,3"


# _validate_and_resolve_claude_model — hard gate
def _make_args(**overrides) -> argparse.Namespace:
    """Build a minimal Namespace; translates legacy ``critic_mock`` into ``critic_backend``."""
    base = dict(
        claude_model="claude-opus-4-7",
        codex_model="gpt-5.4",
        critic_backend="mock",
        kernel_codex=True,
        no_kernel=False,
    )
    if "critic_mock" in overrides:
        base["critic_backend"] = "mock" if overrides.pop("critic_mock") else "agent"
    base.update(overrides)
    return argparse.Namespace(**base)


# _resolve_robustness_choice — default backend
def test_resolve_robustness_choice_defaults_to_agent():
    args = _make_args(robustness_backend=None)

    assert cli.DEFAULT_ROBUSTNESS_BACKEND == "agent"
    assert cli._resolve_robustness_choice(args) == "agent"


def test_resolve_robustness_choice_explicit_mock_wins():
    args = _make_args(robustness_backend="mock")

    assert cli._resolve_robustness_choice(args) == "mock"


def test_resolve_robustness_choice_env_override_still_works(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND", "mock")
    reloaded_cli = importlib.reload(cli)
    try:
        args = _make_args(robustness_backend=None)
        assert reloaded_cli.DEFAULT_ROBUSTNESS_BACKEND == "mock"
        assert reloaded_cli._resolve_robustness_choice(args) == "mock"
    finally:
        monkeypatch.delenv(
            "INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND",
            raising=False,
        )
        importlib.reload(cli)


def test_validate_claude_model_rejects_unsupported_arg(monkeypatch, capsys):
    """`--claude-model claude-opus-4-5` aborts before the catalog probe."""
    probe_calls: list[str] = []

    def _no_probe(**kwargs):
        probe_calls.append(kwargs.get("base_url", ""))
        raise AssertionError("probe should not be reached on static-gate fail")

    monkeypatch.setattr(cli, "_probe_llm_catalog", _no_probe)

    args = _make_args(claude_model="claude-opus-4-5")
    with pytest.raises(SystemExit) as exc_info:
        cli._validate_and_resolve_claude_model(args, None)
    assert exc_info.value.code == 2
    assert probe_calls == []
    err = capsys.readouterr().err
    assert "claude-opus-4-5" in err
    assert "claude-opus-4-7" in err
    assert "claude-opus-4-6" in err


def test_validate_claude_model_custom_allowed_when_optout_set(monkeypatch, capsys):
    """#340: opt-out lets a non-AMD orchestration model pass when in catalog."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", "1")
    monkeypatch.setattr(
        cli, "_probe_llm_catalog",
        lambda **kw: {"my-org/custom-claude", "gpt-5.4"},
    )
    args = _make_args(claude_model="my-org/custom-claude")
    catalog = cli._validate_and_resolve_claude_model(args, None)

    assert args.claude_model == "my-org/custom-claude"
    assert "my-org/custom-claude" in catalog
    out = capsys.readouterr().out
    assert "confirmed in gateway catalog" in out


def test_validate_claude_model_custom_optout_no_amd_fallback(monkeypatch, capsys):
    """#340: under opt-out a catalog miss errors (no silent opus-4-6 fallback)."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", "1")
    monkeypatch.setattr(
        cli, "_probe_llm_catalog",
        # 4-6 present, but custom id absent → must NOT fall back to 4-6.
        lambda **kw: {"claude-opus-4-6", "gpt-5.4"},
    )
    args = _make_args(claude_model="my-org/custom-claude")
    with pytest.raises(SystemExit) as exc_info:
        cli._validate_and_resolve_claude_model(args, None)
    assert exc_info.value.code == 2
    assert args.claude_model == "my-org/custom-claude"
    err = capsys.readouterr().err
    assert "not present in gateway catalog" in err


def test_validate_claude_model_custom_optout_rejects_empty(monkeypatch, capsys):
    """#340: opt-out with an empty model id aborts before the probe."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", "1")

    def _no_probe(**kwargs):
        raise AssertionError("probe should not run on empty-model abort")

    monkeypatch.setattr(cli, "_probe_llm_catalog", _no_probe)
    args = _make_args(claude_model="")
    with pytest.raises(SystemExit) as exc_info:
        cli._validate_and_resolve_claude_model(args, None)
    assert exc_info.value.code == 2
    assert "is empty" in capsys.readouterr().err


def test_validate_claude_model_optout_off_still_hard_gates(monkeypatch, capsys):
    """Default (opt-out unset): custom model still rejected by static gate."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", raising=False)

    def _no_probe(**kwargs):
        raise AssertionError("probe should not run on static-gate fail")

    monkeypatch.setattr(cli, "_probe_llm_catalog", _no_probe)
    args = _make_args(claude_model="my-org/custom-claude")
    with pytest.raises(SystemExit) as exc_info:
        cli._validate_and_resolve_claude_model(args, None)
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL" in err


def test_validate_claude_model_4_7_in_catalog_keeps_choice(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "_probe_llm_catalog",
        lambda **kw: {"claude-opus-4-7", "claude-opus-4-6", "gpt-5.4"},
    )
    args = _make_args(claude_model="claude-opus-4-7")
    catalog = cli._validate_and_resolve_claude_model(args, None)

    assert args.claude_model == "claude-opus-4-7"
    assert "claude-opus-4-7" in catalog
    assert "confirmed in gateway catalog" in capsys.readouterr().out


def test_validate_claude_model_4_7_missing_falls_back_to_4_6(monkeypatch, capsys):
    """Catalog has 4-6 only → arg rewritten with a WARN."""
    monkeypatch.setattr(
        cli, "_probe_llm_catalog",
        lambda **kw: {"claude-opus-4-6", "claude-opus-4-5", "gpt-5.4"},
    )
    args = _make_args(claude_model="claude-opus-4-7")
    cli._validate_and_resolve_claude_model(args, None)

    assert args.claude_model == "claude-opus-4-6"
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "falling back" in out
    assert "claude-opus-4-6" in out


def test_validate_claude_model_neither_in_catalog_aborts(monkeypatch, capsys):
    """Catalog missing both 4-7 and 4-6 → sys.exit(2)."""
    monkeypatch.setattr(
        cli, "_probe_llm_catalog",
        lambda **kw: {"claude-opus-4-5", "claude-haiku-4-5-20251001", "gpt-5.4"},
    )
    args = _make_args(claude_model="claude-opus-4-7")

    with pytest.raises(SystemExit) as exc_info:
        cli._validate_and_resolve_claude_model(args, None)
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "claude-opus-4-7" in err
    assert "claude-opus-4-6" in err
    assert "claude-opus-4-5" in err


def test_validate_claude_model_aborts_when_catalog_unreachable(monkeypatch, capsys):
    """Catalog probe returned None → sys.exit(2)."""
    monkeypatch.setattr(cli, "_probe_llm_catalog", lambda **kw: None)
    args = _make_args(claude_model="claude-opus-4-7")

    with pytest.raises(SystemExit) as exc_info:
        cli._validate_and_resolve_claude_model(args, None)
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "gateway catalog unreachable" in err
    assert "Refusing to start" in err


# _probe_llm_catalog — retry behaviour
class _FakeResp:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


def test_probe_llm_catalog_retries_on_transient_error_then_succeeds(monkeypatch):
    """First two attempts raise, third returns 200 → set returned + 2 sleeps."""
    sleeps: list[float] = []

    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    attempt: list[int] = [0]

    def _flaky_get(url, **kwargs):
        attempt[0] += 1
        if attempt[0] <= 2:
            raise RuntimeError(f"transient {attempt[0]}")
        return _FakeResp(200, {"data": [{"id": "claude-opus-4-7"}]})

    fake_httpx = type("FakeHttpx", (), {"get": staticmethod(_flaky_get)})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    ids = cli._probe_llm_catalog(
        base_url="https://gateway/v1", api_key="sk-test",
    )
    assert ids == {"claude-opus-4-7"}
    assert attempt[0] == 3
    assert sleeps[:2] == list(cli._CATALOG_RETRY_DELAYS_SEC[:2])


def test_probe_llm_catalog_returns_none_when_all_attempts_fail(monkeypatch, capsys):
    """All 4 attempts fail → returns None."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    def _always_500(url, **kwargs):
        return _FakeResp(500, None)

    fake_httpx = type("FakeHttpx", (), {"get": staticmethod(_always_500)})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    ids = cli._probe_llm_catalog(
        base_url="https://gateway/v1", api_key="sk-test",
    )
    assert ids is None
    out = capsys.readouterr().out
    expected_attempts = 1 + len(cli._CATALOG_RETRY_DELAYS_SEC)
    assert out.count("catalog probe attempt") == expected_attempts
    assert "exhausted" in out


def test_probe_llm_catalog_returns_none_for_empty_base_url(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert cli._probe_llm_catalog(base_url="", api_key="sk-test") is None


# _smoke_test_codex_model — WARN-only
def test_smoke_test_codex_model_warns_when_missing(monkeypatch, capsys):
    args = _make_args(codex_model="gpt-99.9")
    catalog = {"claude-opus-4-7", "gpt-5.4", "gpt-4.1"}
    cli._smoke_test_codex_model(args, catalog)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "gpt-99.9" in out
    assert "gpt-5.4" in out


def test_smoke_test_codex_model_skipped_when_unused(monkeypatch, capsys):
    """--critic-mock + --kernel-claude → no probe / no warn."""
    args = _make_args(
        codex_model="gpt-totally-fake",
        critic_mock=True,
        kernel_codex=False,
    )
    catalog = {"claude-opus-4-7"}
    cli._smoke_test_codex_model(args, catalog)
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "gpt-totally-fake" not in out


def test_smoke_test_codex_model_skipped_when_no_kernel(monkeypatch, capsys):
    """--no-kernel hides kernel_codex; critic-mock avoids Codex entirely."""
    args = _make_args(
        codex_model="gpt-totally-fake",
        critic_mock=True,
        kernel_codex=True,
        no_kernel=True,
    )
    catalog = {"claude-opus-4-7"}
    cli._smoke_test_codex_model(args, catalog)
    out = capsys.readouterr().out
    assert "WARNING" not in out


def test_smoke_test_codex_model_confirms_when_present(capsys):
    args = _make_args(codex_model="gpt-5.4", critic_mock=False)
    catalog = {"claude-opus-4-7", "gpt-5.4"}
    cli._smoke_test_codex_model(args, catalog)
    out = capsys.readouterr().out
    assert "confirmed in gateway catalog" in out
    assert "WARNING" not in out


def test_smoke_test_codex_model_no_op_on_probe_failure(capsys):
    """If catalog_ids is None we silently skip; no spurious WARNs."""
    args = _make_args(codex_model="gpt-5.4", critic_mock=False)
    cli._smoke_test_codex_model(args, None)
    assert capsys.readouterr().out == ""


# Merged from test_v08_ir3_preflight.py

"""KB_design_continue §3.3 / IR-3 — preflight soft-degrade tests."""


import argparse
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from inference_optimizer import cli as cli_module


def _ns(**overrides) -> argparse.Namespace:
    defaults: dict = {
        "degraded_kb": False,
        "degraded_pr": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_marker(
    marker_path: Path, *,
    kb_reachable: bool, pr_reachable: bool,
    kb_skipped: bool = False, pr_skipped: bool = False,
    kb_failure_reason: str | None = None,
    pr_failure_reason: str | None = None,
) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "kb_reachable": kb_reachable,
        "pr_reachable": pr_reachable,
        "kb_skipped":   kb_skipped,
        "pr_skipped":   pr_skipped,
    }
    if kb_failure_reason is not None:
        payload["kb_failure_reason"] = kb_failure_reason
    if pr_failure_reason is not None:
        payload["pr_failure_reason"] = pr_failure_reason
    marker_path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_run_writes_marker(marker_path: Path, **marker_kwargs):
    """Return a ``subprocess.run`` stub that writes ``marker_path`` and returns an appropriate rc."""
    def _runner(cmd, env=None, check=False, timeout=None):
        _write_marker(marker_path, **marker_kwargs)
        kb_skipped = marker_kwargs.get("kb_skipped", False)
        pr_skipped = marker_kwargs.get("pr_skipped", False)
        kb_ok = marker_kwargs.get("kb_reachable", False)
        pr_ok = marker_kwargs.get("pr_reachable", False)
        rc = 0
        if not kb_skipped and not kb_ok:
            rc = 1
        if not pr_skipped and not pr_ok:
            rc = 1
        return subprocess.CompletedProcess(cmd, rc)
    return _runner


@pytest.fixture
def marker_path(tmp_path, monkeypatch) -> Path:
    user_data = tmp_path / "user_data"
    monkeypatch.setenv("USER_DATA_PATH", str(user_data))
    return user_data / "runtime" / "cortex" / ".kb_preflight.json"


# 1. KB ok + PR ok → both reachable, reasons None.
def test_ir3_kb_ok_pr_ok(marker_path):
    args = _ns()
    with patch.object(
        cli_module.subprocess, "run",
        side_effect=_fake_run_writes_marker(marker_path, kb_reachable=True, pr_reachable=True),
    ):
        cli_module._run_ir3_preflight(args)
    assert args.cortex_enabled is True
    assert args.pr_monitor_enabled is True
    assert args.kb_degraded_reason is None
    assert args.pr_degraded_reason is None


# 2. KB 5xx + no flag → soft degrade ir3_auto; cli does not abort.
def test_ir3_kb_5xx_auto_degrade(marker_path):
    args = _ns()
    with patch.object(
        cli_module.subprocess, "run",
        side_effect=_fake_run_writes_marker(
            marker_path, kb_reachable=False, pr_reachable=True,
            kb_failure_reason="500",
        ),
    ):
        cli_module._run_ir3_preflight(args)
    assert args.cortex_enabled is False
    assert args.pr_monitor_enabled is True
    assert args.kb_degraded_reason == "ir3_auto"
    assert args.pr_degraded_reason is None


# 3. KB 5xx + --degraded-kb → script gets SKIP_KB_PROBE=1, reason=explicit.
def test_ir3_kb_explicit_flag(marker_path):
    args = _ns(degraded_kb=True)
    seen_env: dict = {}

    def _runner(cmd, env=None, check=False, timeout=None):
        seen_env.update(env or {})
        _write_marker(marker_path, kb_reachable=False, pr_reachable=True, kb_skipped=True)
        return subprocess.CompletedProcess(cmd, 0)

    with patch.object(cli_module.subprocess, "run", side_effect=_runner):
        cli_module._run_ir3_preflight(args)
    assert seen_env.get("SKIP_KB_PROBE") == "1"
    assert "SKIP_PR_PROBE" not in seen_env
    assert args.cortex_enabled is False
    assert args.kb_degraded_reason == "explicit_flag"
    assert args.pr_monitor_enabled is True
    assert args.pr_degraded_reason is None


# 4. KB 401 + non-empty token → kb reachable (auth path).
def test_ir3_kb_401_with_token(marker_path, monkeypatch):
    monkeypatch.setenv("KB_SERVICE_TOKEN", "tok-abc")
    args = _ns()
    with patch.object(
        cli_module.subprocess, "run",
        side_effect=_fake_run_writes_marker(marker_path, kb_reachable=True, pr_reachable=True),
    ):
        cli_module._run_ir3_preflight(args)
    assert args.cortex_enabled is True
    assert args.kb_degraded_reason is None


# 5. KB 401 + empty token → soft degrade ir3_auto, marker missing_token.
def test_ir3_kb_401_missing_token(marker_path, monkeypatch):
    monkeypatch.delenv("KB_SERVICE_TOKEN", raising=False)
    args = _ns()
    with patch.object(
        cli_module.subprocess, "run",
        side_effect=_fake_run_writes_marker(
            marker_path, kb_reachable=False, pr_reachable=True,
            kb_failure_reason="missing_token",
        ),
    ):
        cli_module._run_ir3_preflight(args)
    assert args.cortex_enabled is False
    assert args.kb_degraded_reason == "ir3_auto"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["kb_failure_reason"] == "missing_token"


# 6. PR timeout + --degraded-pr → reason=explicit_flag, KB stays ok.
def test_ir3_pr_explicit_flag_kb_ok(marker_path):
    args = _ns(degraded_pr=True)
    seen_env: dict = {}

    def _runner(cmd, env=None, check=False, timeout=None):
        seen_env.update(env or {})
        _write_marker(marker_path, kb_reachable=True, pr_reachable=False, pr_skipped=True)
        return subprocess.CompletedProcess(cmd, 0)

    with patch.object(cli_module.subprocess, "run", side_effect=_runner):
        cli_module._run_ir3_preflight(args)
    assert seen_env.get("SKIP_PR_PROBE") == "1"
    assert "SKIP_KB_PROBE" not in seen_env
    assert args.cortex_enabled is True
    assert args.kb_degraded_reason is None
    assert args.pr_monitor_enabled is False
    assert args.pr_degraded_reason == "explicit_flag"


# 7. Both flags → preflight_kb.sh NOT invoked.
def test_ir3_both_flags_short_circuit(marker_path):
    args = _ns(degraded_kb=True, degraded_pr=True)
    with patch.object(cli_module.subprocess, "run") as run_mock:
        cli_module._run_ir3_preflight(args)
        run_mock.assert_not_called()
    assert args.cortex_enabled is False
    assert args.pr_monitor_enabled is False
    assert args.kb_degraded_reason == "explicit_flag"
    assert args.pr_degraded_reason == "explicit_flag"


# 8. No cortex KB URL → KB probe skipped, stays local-only without soft-degrading.
def test_ir3_no_kb_url_skips_probe_local_only(marker_path, monkeypatch):
    monkeypatch.delenv("CORTEX_KB_URL", raising=False)
    args = _ns()
    seen_env: dict = {}

    def _runner(cmd, env=None, check=False, timeout=None):
        seen_env.update(env or {})
        _write_marker(
            marker_path, kb_reachable=False, pr_reachable=True, kb_skipped=True,
        )
        return subprocess.CompletedProcess(cmd, 0)

    with patch.object(cli_module.subprocess, "run", side_effect=_runner):
        cli_module._run_ir3_preflight(args)
    assert "CORTEX_KB_URL" not in seen_env
    assert args.cortex_enabled is True
    assert args.kb_degraded_reason is None


# 9. Explicit --cortex-kb-url → injected into the probe environment.
def test_ir3_explicit_kb_url_injected_into_probe_env(marker_path, monkeypatch):
    monkeypatch.delenv("CORTEX_KB_URL", raising=False)
    args = _ns(cortex_kb_url="http://my-kb.example")
    seen_env: dict = {}

    def _runner(cmd, env=None, check=False, timeout=None):
        seen_env.update(env or {})
        _write_marker(marker_path, kb_reachable=True, pr_reachable=True)
        return subprocess.CompletedProcess(cmd, 0)

    with patch.object(cli_module.subprocess, "run", side_effect=_runner):
        cli_module._run_ir3_preflight(args)
    assert seen_env.get("CORTEX_KB_URL") == "http://my-kb.example"
    assert args.cortex_enabled is True
    assert args.kb_degraded_reason is None


# Bonus: CLI flag plumbing
def test_cli_parser_exposes_degraded_flags():
    parser = cli_module._build_parser()
    args = parser.parse_args(["optimize", "--model", "/x", "--degraded-kb"])
    assert args.degraded_kb is True
    assert args.degraded_pr is False
    args = parser.parse_args(["optimize", "--model", "/x", "--degraded-pr"])
    assert args.degraded_pr is True
    assert args.degraded_kb is False
    args = parser.parse_args(["optimize", "--model", "/x"])
    assert args.degraded_kb is False
    assert args.degraded_pr is False
