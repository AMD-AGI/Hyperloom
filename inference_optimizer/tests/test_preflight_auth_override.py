"""Regression tests for the auth-proxy URL force-override in ``_preflight``.

The skill's failure mode that motivated these tests:

* User has ``ANTHROPIC_BASE_URL`` already set in env (shell rc, ``.env``,
  k8s secret, container env) pointing at the upstream gateway.
* Old ``_preflight()`` used ``os.environ.setdefault`` for ``ANTHROPIC_BASE_URL``
  → the externally-preset URL was preserved → Claude CLI bypassed the auth-proxy
  on ``127.0.0.1:4002`` → x-api-key reached the gateway → 401 → SDK hung at
  "Waiting for first result before closing stdin".

These tests pin the new contract: when the auth-proxy is alive,
``ANTHROPIC_BASE_URL`` and ``OPENAI_BASE_URL`` are force-overridden to the
proxy URL regardless of preset value, and the two vars are kept consistent
on both the success and the fallback paths.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from typing import Any

import pytest

from inference_optimizer import cli


# ---------------------------------------------------------------------------
# pure helper: _derive_proxy_urls
# ---------------------------------------------------------------------------
def test_derive_proxy_urls_strips_trailing_v1_for_anthropic_only():
    upstream = "https://oci-slc.primus-safe.amd.com/api/v1/llm-proxy/v1"
    a, o = cli._derive_proxy_urls(upstream, 4002)
    assert a == "http://127.0.0.1:4002/api/v1/llm-proxy"
    assert o == "http://127.0.0.1:4002/api/v1/llm-proxy/v1"


def test_derive_proxy_urls_no_v1_suffix_keeps_path():
    upstream = "https://gateway.example/llm/proxy"
    a, o = cli._derive_proxy_urls(upstream, 4002)
    # No trailing /v1, so anthropic and openai paths are identical.
    assert a == "http://127.0.0.1:4002/llm/proxy"
    assert o == "http://127.0.0.1:4002/llm/proxy"


def test_derive_proxy_urls_honours_custom_port():
    a, o = cli._derive_proxy_urls(
        "https://x/api/v1/llm-proxy/v1", proxy_port=4099
    )
    assert a == "http://127.0.0.1:4099/api/v1/llm-proxy"
    assert o == "http://127.0.0.1:4099/api/v1/llm-proxy/v1"


# ---------------------------------------------------------------------------
# _preflight() override semantics
# ---------------------------------------------------------------------------
@pytest.fixture
def stub_install_steps(monkeypatch):
    """Stub out the heavyweight install steps so _preflight() is fast.

    We only care about the auth-proxy override block here; the ray/Magpie/
    InferenceX install paths are exercised elsewhere.
    """
    monkeypatch.setattr(cli, "_load_dotenv_fallback", lambda: None)

    def _fake_which(name: str):
        return f"/usr/bin/{name}"  # pretend ray + python3 are present

    monkeypatch.setattr(cli.shutil, "which", _fake_which)

    class _FakeCompleted:
        def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(cmd, *args, **kwargs):
        # Magpie import check; pretend it's already installed.
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


# ---------------------------------------------------------------------------
def test_override_replaces_external_anthropic_base_url(
    stub_install_steps, clean_url_env, monkeypatch
):
    """External ANTHROPIC_BASE_URL → upstream MUST be replaced by proxy URL."""
    upstream_openai = "https://gateway.example/api/v1/llm-proxy/v1"
    upstream_anthropic = "https://gateway.example/api/v1/llm-proxy"  # different host on purpose
    proxy_anthropic = "http://127.0.0.1:4002/api/v1/llm-proxy"
    proxy_openai = "http://127.0.0.1:4002/api/v1/llm-proxy/v1"

    monkeypatch.setenv("SAFE_API_KEY", "ak-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", upstream_openai)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", upstream_anthropic)

    def _fake_proxy(safe_key, base_url):
        # _preflight passes OPENAI_BASE_URL as base_url; the helper would
        # normally derive proxy URLs from it. Return what the live helper
        # returns on success.
        return (proxy_anthropic, proxy_openai)

    monkeypatch.setattr(cli, "_ensure_auth_proxy_and_claude_config", _fake_proxy)

    cli._preflight()

    import os as _os
    assert _os.environ["ANTHROPIC_BASE_URL"] == proxy_anthropic
    assert _os.environ["OPENAI_BASE_URL"] == proxy_openai
    # OOB / GEAK / LLM_API_BASE keep upstream — they speak Bearer natively.
    assert _os.environ["OOB_BASE_URL"] == upstream_openai
    assert _os.environ["GEAK_BASE_URL"] == upstream_openai
    assert _os.environ["LLM_API_BASE"] == upstream_openai


def test_override_when_anthropic_was_unset(
    stub_install_steps, clean_url_env, monkeypatch
):
    """When ANTHROPIC_BASE_URL was never set, override should still apply."""
    upstream = "https://gateway.example/api/v1/llm-proxy/v1"
    proxy_anthropic = "http://127.0.0.1:4002/api/v1/llm-proxy"
    proxy_openai = "http://127.0.0.1:4002/api/v1/llm-proxy/v1"

    monkeypatch.setenv("SAFE_API_KEY", "ak-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", upstream)
    # ANTHROPIC_BASE_URL deliberately unset.
    monkeypatch.setattr(
        cli,
        "_ensure_auth_proxy_and_claude_config",
        lambda *a, **kw: (proxy_anthropic, proxy_openai),
    )

    cli._preflight()

    import os as _os
    assert _os.environ["ANTHROPIC_BASE_URL"] == proxy_anthropic
    assert _os.environ["OPENAI_BASE_URL"] == proxy_openai


def test_consistency_invariant_on_success(
    stub_install_steps, clean_url_env, monkeypatch
):
    """ANTHROPIC_BASE_URL and OPENAI_BASE_URL must always agree on host:port."""
    monkeypatch.setenv("SAFE_API_KEY", "ak-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway/api/v1/llm-proxy/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://different-host/api")
    monkeypatch.setattr(
        cli,
        "_ensure_auth_proxy_and_claude_config",
        lambda *a, **kw: (
            "http://127.0.0.1:4002/api/v1/llm-proxy",
            "http://127.0.0.1:4002/api/v1/llm-proxy/v1",
        ),
    )

    cli._preflight()

    import os as _os
    from urllib.parse import urlparse

    a = urlparse(_os.environ["ANTHROPIC_BASE_URL"])
    o = urlparse(_os.environ["OPENAI_BASE_URL"])
    assert (a.scheme, a.hostname, a.port) == (o.scheme, o.hostname, o.port)


def test_proxy_failure_falls_back_to_orig(
    stub_install_steps, clean_url_env, monkeypatch
):
    """When auth-proxy returns None, originals are restored (consistency kept)."""
    orig_anthropic = "https://core42.primus-safe.amd.com/api/v1/llm-proxy"
    orig_openai = "https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1"

    monkeypatch.setenv("SAFE_API_KEY", "ak-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", orig_openai)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", orig_anthropic)
    monkeypatch.setattr(
        cli, "_ensure_auth_proxy_and_claude_config", lambda *a, **kw: None
    )

    cli._preflight()

    import os as _os
    # Originals restored — nothing was force-overridden because proxy is down.
    assert _os.environ["ANTHROPIC_BASE_URL"] == orig_anthropic
    assert _os.environ["OPENAI_BASE_URL"] == orig_openai


def test_proxy_failure_with_unset_anthropic_keeps_unset(
    stub_install_steps, clean_url_env, monkeypatch
):
    """If the user had no ANTHROPIC_BASE_URL preset and proxy is down, we
    must NOT leak ``OPENAI_BASE_URL`` into ``ANTHROPIC_BASE_URL`` — the
    consistency invariant says "both proxy or both orig" and orig here
    means "unset"."""
    orig_openai = "https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1"

    monkeypatch.setenv("SAFE_API_KEY", "ak-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", orig_openai)
    # ANTHROPIC_BASE_URL deliberately unset.
    monkeypatch.setattr(
        cli, "_ensure_auth_proxy_and_claude_config", lambda *a, **kw: None
    )

    cli._preflight()

    import os as _os
    assert "ANTHROPIC_BASE_URL" not in _os.environ
    assert _os.environ["OPENAI_BASE_URL"] == orig_openai


# ---------------------------------------------------------------------------
# _proxy_alive — TCP probe
# ---------------------------------------------------------------------------
def test_proxy_alive_returns_false_on_refused(monkeypatch):
    import socket

    class _RefusingSocket:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a, **kw):
            return False

        def settimeout(self, _t):
            pass

        def connect(self, _addr):
            raise ConnectionRefusedError()

    monkeypatch.setattr(socket, "socket", _RefusingSocket)
    assert cli._proxy_alive(4002) is False


def test_proxy_alive_returns_true_on_connect(monkeypatch):
    import socket

    class _ConnectingSocket:
        def __init__(self, *a, **kw):
            self.connected = False

        def __enter__(self):
            return self

        def __exit__(self, *a, **kw):
            return False

        def settimeout(self, _t):
            pass

        def connect(self, _addr):
            self.connected = True

    monkeypatch.setattr(socket, "socket", _ConnectingSocket)
    assert cli._proxy_alive(4002) is True


# ---------------------------------------------------------------------------
# _ensure_python_sdks
# ---------------------------------------------------------------------------
class _RecordingRun:
    """Test double for subprocess.run that records calls and replays a script."""

    def __init__(self, script: list[Any]):
        # script: list of either CompletedProcess-like objects or callables
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
        _Completed(returncode=0),  # import claude_agent_sdk
        _Completed(returncode=0),  # import openai
        _Completed(returncode=0),  # import httpx
    ])
    monkeypatch.setattr(cli.subprocess, "run", runner)

    cli._ensure_python_sdks("/opt/venv/bin/python", [])

    assert len(runner.calls) == 3
    for call in runner.calls:
        assert call[0] == "/opt/venv/bin/python"
        assert call[1] == "-c"
        # Only the import-probe shape: ["python", "-c", "import X"]
        assert call[2].startswith("import ")
    captured = capsys.readouterr().out
    assert "claude_agent_sdk OK" in captured
    assert "openai OK" in captured
    assert "httpx OK" in captured


def test_ensure_python_sdks_installs_missing_claude_agent_sdk(monkeypatch, capsys):
    """When `import claude_agent_sdk` fails, pip install is invoked.

    Pins the contract that the SAME interpreter (sys.executable in the
    real preflight; here we pass /opt/venv/bin/python) is used for both
    the probe and the install — cross-interpreter installs are the bug."""
    runner = _RecordingRun([
        _Completed(returncode=1),  # import claude_agent_sdk fails
        _Completed(returncode=0),  # pip install claude-agent-sdk
        _Completed(returncode=0),  # import openai
        _Completed(returncode=0),  # import httpx
    ])
    monkeypatch.setattr(cli.subprocess, "run", runner)

    cli._ensure_python_sdks("/opt/venv/bin/python", ["--break-system-packages"])

    # Probe + install + 2 more probes
    assert len(runner.calls) == 4
    install_call = runner.calls[1]
    assert install_call[0] == "/opt/venv/bin/python"
    assert install_call[1:4] == ["-m", "pip", "install"]
    assert "--break-system-packages" in install_call
    assert any(arg.startswith("claude-agent-sdk") for arg in install_call)
    captured = capsys.readouterr().out
    assert "installing claude-agent-sdk" in captured
    assert "installed claude-agent-sdk" in captured


# ---------------------------------------------------------------------------
# _ensure_oob_proxy_source
# ---------------------------------------------------------------------------
def test_ensure_oob_proxy_source_no_op_when_present(monkeypatch, tmp_path, capsys):
    """auth_proxy.py already at HYPERLOOM_ROOT/OOB/oob_cli/ → returns True silently."""
    hyperloom_root = tmp_path / "hyperloom"
    target_dir = hyperloom_root / "OOB" / "oob_cli"
    target_dir.mkdir(parents=True)
    (target_dir / "auth_proxy.py").write_text("# stub\n", encoding="utf-8")

    monkeypatch.setenv("HYPERLOOM_ROOT", str(hyperloom_root))

    # No fallback candidates should ever be touched
    monkeypatch.setattr(
        cli, "_OOB_SRC_CANDIDATES",
        ("/nonexistent/path/A", "/nonexistent/path/B"),
    )
    monkeypatch.delenv("OOB_SRC", raising=False)

    assert cli._ensure_oob_proxy_source() is True
    captured = capsys.readouterr().out
    # Must not log a bootstrap message — file was already there.
    assert "bootstrapped auth_proxy.py" not in captured
    assert "WARNING" not in captured


def test_ensure_oob_proxy_source_bootstraps_from_first_candidate(
    monkeypatch, tmp_path, capsys,
):
    """Missing target → copy from first existing candidate."""
    hyperloom_root = tmp_path / "hyperloom"  # empty; target absent

    src_a = tmp_path / "src_a"
    src_a.mkdir()
    (src_a / "auth_proxy.py").write_text("# v1\n", encoding="utf-8")
    (src_a / "other.py").write_text("# helper\n", encoding="utf-8")

    src_b = tmp_path / "src_b"
    src_b.mkdir()
    (src_b / "auth_proxy.py").write_text("# v2\n", encoding="utf-8")

    monkeypatch.setenv("HYPERLOOM_ROOT", str(hyperloom_root))
    monkeypatch.delenv("OOB_SRC", raising=False)
    monkeypatch.setattr(
        cli, "_OOB_SRC_CANDIDATES", (str(src_a), str(src_b)),
    )

    assert cli._ensure_oob_proxy_source() is True
    target = hyperloom_root / "OOB" / "oob_cli" / "auth_proxy.py"
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == "# v1\n"  # first candidate wins
    # Sibling file from the source dir should also be copied (copytree).
    assert (hyperloom_root / "OOB" / "oob_cli" / "other.py").is_file()
    assert "bootstrapped auth_proxy.py" in capsys.readouterr().out


def test_ensure_oob_proxy_source_env_override_wins(
    monkeypatch, tmp_path, capsys,
):
    """$OOB_SRC takes precedence over the hard-coded candidate list."""
    hyperloom_root = tmp_path / "hyperloom"

    env_src = tmp_path / "env_src"
    env_src.mkdir()
    (env_src / "auth_proxy.py").write_text("# from-env\n", encoding="utf-8")

    other_src = tmp_path / "other"
    other_src.mkdir()
    (other_src / "auth_proxy.py").write_text("# from-default\n", encoding="utf-8")

    monkeypatch.setenv("HYPERLOOM_ROOT", str(hyperloom_root))
    monkeypatch.setenv("OOB_SRC", str(env_src))
    monkeypatch.setattr(cli, "_OOB_SRC_CANDIDATES", (str(other_src),))

    assert cli._ensure_oob_proxy_source() is True
    target = hyperloom_root / "OOB" / "oob_cli" / "auth_proxy.py"
    assert target.read_text(encoding="utf-8") == "# from-env\n"


def test_ensure_oob_proxy_source_warns_when_no_source(
    monkeypatch, tmp_path, capsys,
):
    """No source anywhere → returns False + WARNING line."""
    hyperloom_root = tmp_path / "hyperloom"
    monkeypatch.setenv("HYPERLOOM_ROOT", str(hyperloom_root))
    monkeypatch.delenv("OOB_SRC", raising=False)
    monkeypatch.setattr(
        cli, "_OOB_SRC_CANDIDATES",
        (str(tmp_path / "missing_a"), str(tmp_path / "missing_b")),
    )

    assert cli._ensure_oob_proxy_source() is False
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "auth_proxy.py source not located" in out


# ---------------------------------------------------------------------------
# _unset_hip_visible_devices
# ---------------------------------------------------------------------------
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
    # Still present — we only unset HIP when ROCR is also there.
    assert _os.environ["HIP_VISIBLE_DEVICES"] == "0,1,2,3"


# ---------------------------------------------------------------------------
# _validate_and_resolve_claude_model — hard gate
# ---------------------------------------------------------------------------
def _make_args(**overrides) -> argparse.Namespace:
    """Build a minimal Namespace for _validate_and_resolve_claude_model.

    Tests historically passed ``critic_mock=True/False`` here; that flag was
    folded into ``critic_backend`` (one of ``mock`` / ``agent`` /
    ``codex_bare``) when CriticAgentBackend landed. We translate
    ``critic_mock`` into the new attribute for back-compat.
    """
    base = dict(
        claude_model="claude-opus-4-7",
        codex_model="gpt-5.4",
        critic_backend="mock",
        kernel_codex=True,
        no_kernel=False,
    )
    if "critic_mock" in overrides:
        base["critic_backend"] = "mock" if overrides.pop("critic_mock") else "codex_bare"
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# _resolve_robustness_choice — default backend
# ---------------------------------------------------------------------------
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
    """`--claude-model claude-opus-4-5` aborts BEFORE catalog probe."""
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
    """Catalog has 4-6 only → arg rewritten + WARN."""
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
    # The catalog enumeration in the error should list what WAS available.
    assert "claude-opus-4-5" in err


def test_validate_claude_model_aborts_when_catalog_unreachable(monkeypatch, capsys):
    """Catalog probe returned None (gateway unreachable) → sys.exit(2)."""
    monkeypatch.setattr(cli, "_probe_llm_catalog", lambda **kw: None)
    args = _make_args(claude_model="claude-opus-4-7")

    with pytest.raises(SystemExit) as exc_info:
        cli._validate_and_resolve_claude_model(args, None)
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "gateway catalog unreachable" in err
    assert "Refusing to start" in err


def test_validate_claude_model_uses_proxy_url_when_available(monkeypatch):
    """When auth-proxy is alive, probe routes through 127.0.0.1:4002."""
    seen_base_urls: list[str] = []

    def _capture_probe(**kw):
        seen_base_urls.append(kw["base_url"])
        return {"claude-opus-4-7"}

    monkeypatch.setattr(cli, "_probe_llm_catalog", _capture_probe)
    proxy_urls = (
        "http://127.0.0.1:4002/api/v1/llm-proxy",       # anthropic
        "http://127.0.0.1:4002/api/v1/llm-proxy/v1",    # openai (this one probed)
    )
    args = _make_args(claude_model="claude-opus-4-7")
    cli._validate_and_resolve_claude_model(args, proxy_urls)
    assert seen_base_urls == [proxy_urls[1]]


# ---------------------------------------------------------------------------
# _probe_llm_catalog — retry behaviour
# ---------------------------------------------------------------------------
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
    # 3 attempts total: initial (0s sleep skipped) + 2 retry sleeps.
    # _CATALOG_RETRY_DELAYS_SEC[:2] should have been waited.
    assert sleeps[:2] == list(cli._CATALOG_RETRY_DELAYS_SEC[:2])


def test_probe_llm_catalog_returns_none_when_all_attempts_fail(monkeypatch, capsys):
    """All 4 attempts fail → returns None (caller decides to abort)."""
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
    # Each attempt should log; total = 1 initial + len(_CATALOG_RETRY_DELAYS_SEC).
    expected_attempts = 1 + len(cli._CATALOG_RETRY_DELAYS_SEC)
    assert out.count("catalog probe attempt") == expected_attempts
    assert "exhausted" in out


def test_probe_llm_catalog_returns_none_for_empty_base_url(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert cli._probe_llm_catalog(base_url="", api_key="sk-test") is None


# ---------------------------------------------------------------------------
# _smoke_test_codex_model — WARN-only
# ---------------------------------------------------------------------------
def test_smoke_test_codex_model_warns_when_missing(monkeypatch, capsys):
    args = _make_args(codex_model="gpt-99.9")  # not in catalog
    catalog = {"claude-opus-4-7", "gpt-5.4", "gpt-4.1"}
    cli._smoke_test_codex_model(args, catalog)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "gpt-99.9" in out
    # The catalog snippet should list available gpt models.
    assert "gpt-5.4" in out


def test_smoke_test_codex_model_skipped_when_unused(monkeypatch, capsys):
    """--critic-mock + --kernel-claude (kernel_codex=False) → no probe / no warn."""
    args = _make_args(
        codex_model="gpt-totally-fake",
        critic_mock=True,
        kernel_codex=False,
    )
    catalog = {"claude-opus-4-7"}  # no gpt
    cli._smoke_test_codex_model(args, catalog)
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "gpt-totally-fake" not in out


def test_smoke_test_codex_model_skipped_when_no_kernel(monkeypatch, capsys):
    """--no-kernel hides kernel_codex; only --critic-real keeps codex live."""
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
    """If catalog_ids is None (probe failed earlier — Claude already aborted),
    we silently skip; no spurious WARNs."""
    args = _make_args(codex_model="gpt-5.4", critic_mock=False)
    cli._smoke_test_codex_model(args, None)
    assert capsys.readouterr().out == ""


# ============================================================================
# Merged from test_v08_ir3_preflight.py
# ============================================================================

"""KB_design_continue §3.3 / IR-3 — preflight soft-degrade tests.

7 cases (per the implementation plan):
1. KB ok + PR ok → both reachable, reasons ``None``.
2. KB 5xx + no flag → ``cortex_enabled=False``, ``kb_degraded_reason="ir3_auto"``;
   **cli does not abort** (soft degrade).
3. KB 5xx + ``--degraded-kb`` → script invoked with ``SKIP_KB_PROBE=1``,
   ``kb_degraded_reason="explicit_flag"``.
4. KB 401 + non-empty ``KB_SERVICE_TOKEN`` → kb reachable.
5. KB 401 + empty token → soft degrade ``ir3_auto`` +
   marker ``kb_failure_reason="missing_token"``.
6. PR timeout + ``--degraded-pr`` → ``pr_degraded_reason="explicit_flag"``,
   KB still ok.
7. Both flags → ``preflight_kb.sh`` is **not** invoked at all
   (``subprocess.run.assert_not_called()``).
"""


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
    """Return a ``subprocess.run`` stub that writes ``marker_path``
    with the given content + returns rc=0 / rc=1 as appropriate."""
    def _runner(cmd, env=None, check=False, timeout=None):
        _write_marker(marker_path, **marker_kwargs)
        # Compute rc: 1 if any not-skipped branch is unreachable, else 0.
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


# ---------------------------------------------------------------------------
# 1. KB ok + PR ok → both reachable, reasons None.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2. KB 5xx + no flag → soft degrade ir3_auto; cli does not abort.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 3. KB 5xx + --degraded-kb → script gets SKIP_KB_PROBE=1, reason=explicit.
# ---------------------------------------------------------------------------
def test_ir3_kb_explicit_flag(marker_path):
    args = _ns(degraded_kb=True)
    seen_env: dict = {}

    def _runner(cmd, env=None, check=False, timeout=None):
        seen_env.update(env or {})
        # Write a marker as if the script ran with kb skipped.
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


# ---------------------------------------------------------------------------
# 4. KB 401 + non-empty token → kb reachable (auth path).
# ---------------------------------------------------------------------------
def test_ir3_kb_401_with_token(marker_path, monkeypatch):
    monkeypatch.setenv("KB_SERVICE_TOKEN", "tok-abc")
    args = _ns()
    # The actual probe semantics live in preflight_kb.sh; here we
    # simulate "401 with token → kb_reachable=true" by writing the
    # marker as reachable.
    with patch.object(
        cli_module.subprocess, "run",
        side_effect=_fake_run_writes_marker(marker_path, kb_reachable=True, pr_reachable=True),
    ):
        cli_module._run_ir3_preflight(args)
    assert args.cortex_enabled is True
    assert args.kb_degraded_reason is None


# ---------------------------------------------------------------------------
# 5. KB 401 + empty token → soft degrade ir3_auto, marker missing_token.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 6. PR timeout + --degraded-pr → reason=explicit_flag, KB stays ok.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 7. Both flags → preflight_kb.sh NOT invoked.
# ---------------------------------------------------------------------------
def test_ir3_both_flags_short_circuit(marker_path):
    args = _ns(degraded_kb=True, degraded_pr=True)
    with patch.object(cli_module.subprocess, "run") as run_mock:
        cli_module._run_ir3_preflight(args)
        run_mock.assert_not_called()
    assert args.cortex_enabled is False
    assert args.pr_monitor_enabled is False
    assert args.kb_degraded_reason == "explicit_flag"
    assert args.pr_degraded_reason == "explicit_flag"


# ---------------------------------------------------------------------------
# Bonus: CLI flag plumbing
# ---------------------------------------------------------------------------
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
