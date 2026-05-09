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

import subprocess
from typing import Any

import pytest

from inference_optimizer import cli


# ---------------------------------------------------------------------------
# pure helper: _derive_proxy_urls
# ---------------------------------------------------------------------------
def test_derive_proxy_urls_strips_trailing_v1_for_anthropic_only():
    upstream = "https://oci-slc.example-internal-host.invalid/api/v1/llm-proxy/v1"
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
    orig_anthropic = "https://core42.example-internal-host.invalid/api/v1/llm-proxy"
    orig_openai = "https://core42.example-internal-host.invalid/api/v1/llm-proxy/v1"

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
    orig_openai = "https://core42.example-internal-host.invalid/api/v1/llm-proxy/v1"

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
