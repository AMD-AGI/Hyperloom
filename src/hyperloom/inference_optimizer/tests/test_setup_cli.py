# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from pathlib import Path

from hyperloom.inference_optimizer import setup


class _Completed:
    returncode = 7


def test_setup_cli_forwards_flags_and_workspace_env(tmp_path: Path, monkeypatch):
    installer = tmp_path / "install_baremetal.sh"
    installer.write_text("#!/bin/sh\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def _fake_run(cmd, *, env):
        seen["cmd"] = cmd
        seen["env"] = env
        return _Completed()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, "_INSTALL_BAREMETAL_SH", installer)
    monkeypatch.setattr(setup, "_PACKAGE_SKILL", tmp_path / "SKILL.md")
    monkeypatch.setattr(setup.subprocess, "run", _fake_run)

    rc = setup.main(["--check-only", "--dry-run", "--", "--install-framework", "none"])

    assert rc == 7
    assert seen["cmd"] == [
        "bash",
        str(installer),
        "--check-only",
        "--dry-run",
        "--install-framework",
        "none",
    ]
    env = seen["env"]
    assert env["REPO_ROOT"] == str(tmp_path)
    assert env["HYPERLOOM_ENV_FILE"] == str(tmp_path / ".env")
    assert env["HYPERLOOM_SKILL_PATH"] == str(tmp_path / "SKILL.md")


def test_setup_cli_reports_missing_installer(tmp_path: Path, monkeypatch, capsys):
    missing = tmp_path / "missing.sh"
    monkeypatch.setattr(setup, "_INSTALL_BAREMETAL_SH", missing)

    rc = setup.main([])

    assert rc == 1
    assert str(missing) in capsys.readouterr().err
