"""Regression tests for ``_check_tracelens_cli`` hard-gate.

Pins SKILL Step 2 step 8.5 contract:

* Missing ``TraceLens_generate_perf_report_pytorch_inference`` on PATH ->
  ``sys.exit(2)`` with an actionable error pointing at ``install.sh``.
* Present on PATH -> returns silently (no exit, no print).

Why this gate exists: brain-generated ``launch.sh`` / ``relaunch.sh``
wrappers that source only ``runtime/kernel-agent.env.sh`` and skip
``install.sh`` look healthy until kernel-agent shells out to
``tracelens_analysis.py``, ~1h into the session. Without this gate the
robustness agent's J3 ``tracelens_cli_missing`` HIGH signal at tick ~6 was
the first surfacing. Moving discovery to launch (mirroring
``_gate_claude_model`` step 10) turns a delayed silent strike into an
upfront fail-fast.
"""

from __future__ import annotations

import pytest

from inference_optimizer import cli


def test_check_tracelens_cli_passes_when_cli_present(monkeypatch, capsys):
    """All required CLI names resolve via ``shutil.which`` -> silent return."""
    def fake_which(name: str) -> str | None:
        if name in cli._TRACELENS_REQUIRED_CLIS:
            return f"/opt/venv/bin/{name}"
        return None

    monkeypatch.setattr(cli.shutil, "which", fake_which)
    cli._check_tracelens_cli()  # should not raise
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_check_tracelens_cli_exits_2_when_cli_missing(monkeypatch, capsys):
    """Missing CLI -> sys.exit(2) with install.sh remediation hint."""
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)

    with pytest.raises(SystemExit) as excinfo:
        cli._check_tracelens_cli()
    assert excinfo.value.code == 2

    err = capsys.readouterr().err
    assert "TraceLens CLI(s) not on PATH" in err
    assert "TraceLens_generate_perf_report_pytorch_inference" in err
    assert "install.sh" in err
    assert "kernel-agent.env.sh" in err
    assert "Refusing to start" in err


def test_check_tracelens_cli_error_uses_user_data_path_env(
    monkeypatch, capsys,
):
    """Remediation hint should respect ``$USER_DATA_PATH`` override."""
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    monkeypatch.setenv("USER_DATA_PATH", "/custom/session/root")

    with pytest.raises(SystemExit):
        cli._check_tracelens_cli()
    err = capsys.readouterr().err
    assert "/custom/session/root/runtime/kernel-agent.env.sh" in err


def test_check_tracelens_cli_error_falls_back_to_default_when_env_unset(
    monkeypatch, capsys,
):
    """Without ``$USER_DATA_PATH`` the hint cites the default session dir."""
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    monkeypatch.delenv("USER_DATA_PATH", raising=False)

    with pytest.raises(SystemExit):
        cli._check_tracelens_cli()
    err = capsys.readouterr().err
    assert "/workspace/hyperloom/runtime/kernel-agent.env.sh" in err
