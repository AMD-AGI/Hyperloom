"""N24 — ``_load_kernel_agent_env_fallback`` hard-fails on bad state.

Background: pre-N24 the fallback printed a WARN and let _preflight()
continue when $USER_DATA_PATH/runtime/kernel-agent.env.sh was missing
or empty. The May 2026 Qwen1.5-7B 10h silent stall traced back to that
WARN: USER_DATA_PATH was pointed at a per-session subdir so runtime/
didn't exist there, the file was silently skipped,
HYPERLOOM_KERNEL_AGENT_ROOT stayed unset, and 5 consecutive
trace_analyze sub-steps failed quietly — the LLM heartbeat'd for 7.5h
without ever advancing. N24 makes the fallback abort the process with
sys.exit(2) and a clear actionable message so the operator notices
within the first 30 seconds, not 10 hours in.

These tests pin the contract:

* HYPERLOOM_KERNEL_AGENT_ROOT already set -> noop (no exit, no print).
* env file present + valid -> sources vars, no exit.
* USER_DATA_PATH unset (and KERNEL_AGENT_ENV unset, root unset)
  -> sys.exit(2) with a USER_DATA_PATH-mentioning message.
* env file missing -> sys.exit(2) mentioning install.sh + the path.
* env file present but doesn't define HYPERLOOM_KERNEL_AGENT_ROOT
  -> sys.exit(2) with malformed/stale message.
* Sourced file populates os.environ on success.
"""

from __future__ import annotations

import pytest

from inference_optimizer import cli


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Strip the three relevant env vars so each test sees the bare
    state the fallback was designed for."""
    for var in (
        "HYPERLOOM_KERNEL_AGENT_ROOT",
        "KERNEL_AGENT_ENV",
        "USER_DATA_PATH",
    ):
        monkeypatch.delenv(var, raising=False)


def test_noop_when_root_already_set(monkeypatch, capsys):
    """If HYPERLOOM_KERNEL_AGENT_ROOT is set (operator pre-sourced /
    sandbox injected), the fallback returns immediately."""
    monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", "/opt/kernel-agent")
    cli._load_kernel_agent_env_fallback()
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err == ""


def test_aborts_when_no_user_data_path(monkeypatch, capsys):
    """No HYPERLOOM_KERNEL_AGENT_ROOT, no KERNEL_AGENT_ENV, no
    USER_DATA_PATH -> we have nothing to resolve, so abort loudly
    instead of WARN-and-continue."""
    with pytest.raises(SystemExit) as excinfo:
        cli._load_kernel_agent_env_fallback()
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "USER_DATA_PATH" in err
    assert "install.sh" in err


def test_aborts_when_env_file_missing(tmp_path, monkeypatch, capsys):
    """USER_DATA_PATH set but runtime/kernel-agent.env.sh doesn't
    exist (most common N17 misuse: USER_DATA_PATH pointed at a
    per-session subdir) -> sys.exit(2) with an actionable message."""
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    with pytest.raises(SystemExit) as excinfo:
        cli._load_kernel_agent_env_fallback()
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "kernel-agent.env.sh" in err
    assert "install.sh" in err
    assert str(tmp_path) in err


def test_aborts_when_env_file_does_not_define_root(tmp_path, monkeypatch, capsys):
    """File exists but the magic var isn't defined -> stale / malformed
    file, abort loudly (the file sourcing must actually achieve its
    only purpose)."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "kernel-agent.env.sh").write_text(
        "# stale file\nexport SOMETHING_ELSE=1\n", encoding="utf-8",
    )
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    with pytest.raises(SystemExit) as excinfo:
        cli._load_kernel_agent_env_fallback()
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "HYPERLOOM_KERNEL_AGENT_ROOT" in err
    assert "stale" in err or "malformed" in err


def test_sources_vars_on_success(tmp_path, monkeypatch, capsys):
    """Happy path: a well-formed env file sets HYPERLOOM_KERNEL_AGENT_ROOT
    + extras into os.environ and prints a single-line provenance log."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "kernel-agent.env.sh").write_text(
        "# valid env file\n"
        "export HYPERLOOM_KERNEL_AGENT_ROOT=/opt/kernel-agent\n"
        "export KERNEL_AGENT_LOG_LEVEL=INFO\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    cli._load_kernel_agent_env_fallback()
    import os as _os
    assert _os.environ["HYPERLOOM_KERNEL_AGENT_ROOT"] == "/opt/kernel-agent"
    assert _os.environ["KERNEL_AGENT_LOG_LEVEL"] == "INFO"
    out = capsys.readouterr().out
    assert "loaded" in out
    assert "kernel-agent" in out


def test_env_wins_over_file(tmp_path, monkeypatch, capsys):
    """If a var is already set in os.environ, the file does NOT
    overwrite it (env-wins, documented contract)."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "kernel-agent.env.sh").write_text(
        "export HYPERLOOM_KERNEL_AGENT_ROOT=/from/file\n"
        "export KERNEL_AGENT_LOG_LEVEL=INFO\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("KERNEL_AGENT_LOG_LEVEL", "DEBUG")  # pre-set
    cli._load_kernel_agent_env_fallback()
    import os as _os
    # HYPERLOOM_KERNEL_AGENT_ROOT was unset -> file value wins.
    assert _os.environ["HYPERLOOM_KERNEL_AGENT_ROOT"] == "/from/file"
    # KERNEL_AGENT_LOG_LEVEL was pre-set -> env wins.
    assert _os.environ["KERNEL_AGENT_LOG_LEVEL"] == "DEBUG"


def test_explicit_kernel_agent_env_overrides_user_data_path(
    tmp_path, monkeypatch,
):
    """$KERNEL_AGENT_ENV (when set) wins over deriving the path from
    USER_DATA_PATH/runtime/, so operators can point at a custom file
    location without having to symlink."""
    custom = tmp_path / "custom-loc.sh"
    custom.write_text(
        "export HYPERLOOM_KERNEL_AGENT_ROOT=/from/custom\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KERNEL_AGENT_ENV", str(custom))
    monkeypatch.setenv("USER_DATA_PATH", "/nonexistent/should-not-be-used")
    cli._load_kernel_agent_env_fallback()
    import os as _os
    assert _os.environ["HYPERLOOM_KERNEL_AGENT_ROOT"] == "/from/custom"
