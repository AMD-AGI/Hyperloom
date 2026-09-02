# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hermes executable resolution must be shared by every transport."""

from pathlib import Path

from hyperloom.common import hermes_runtime


def test_resolve_hermes_executable_honors_configured_absolute_path(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "custom-hermes"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("HYPERLOOM_HERMES_BIN", str(binary))

    assert hermes_runtime.resolve_hermes_executable() == str(binary.resolve())


def test_resolve_hermes_executable_rejects_nonexecutable_override(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "custom-hermes"
    binary.write_text("not executable", encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_HERMES_BIN", str(binary))

    assert hermes_runtime.resolve_hermes_executable() == ""


def test_external_sandbox_requires_explicit_truthy_value(monkeypatch) -> None:
    monkeypatch.setattr(hermes_runtime, "running_in_container", lambda _env=None: True)
    monkeypatch.delenv(hermes_runtime.HERMES_EXTERNAL_SANDBOX_ENV, raising=False)
    assert not hermes_runtime.hermes_external_sandbox_enabled()
    monkeypatch.setenv(hermes_runtime.HERMES_EXTERNAL_SANDBOX_ENV, "1")
    assert hermes_runtime.hermes_external_sandbox_enabled()


def test_external_sandbox_declaration_fails_outside_container(monkeypatch) -> None:
    monkeypatch.setenv(hermes_runtime.HERMES_EXTERNAL_SANDBOX_ENV, "1")
    monkeypatch.setattr(hermes_runtime, "running_in_container", lambda _env=None: False)
    assert not hermes_runtime.hermes_external_sandbox_enabled()
