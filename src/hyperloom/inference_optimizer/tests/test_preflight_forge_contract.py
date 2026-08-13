# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Launcher preflight for the forge-loop argv contract.

The image that runs a session is the only place that knows which KernelForge is
installed, and a retired option there costs every forge attempt rc=2 -- hours
into the run, visible only in per-attempt logs. The preflight turns that into a
launch-time failure that names the option.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "preflight_optimizer.py"
)
_SPEC = importlib.util.spec_from_file_location("preflight_optimizer", _MODULE_PATH)
assert _SPEC and _SPEC.loader
preflight = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(preflight)


def _declared_options() -> set[str]:
    import sys

    backends = (
        Path(__file__).resolve().parents[3]
        / "hyperloom"
        / "agents"
        / "kernel"
        / "tools"
        / "backends"
    )
    sys.path.insert(0, str(backends))
    import forge_submit

    return set(forge_submit.FORGE_LOOP_OPTIONS)


def test_a_forge_loop_missing_one_option_fails_the_launch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    declared = _declared_options()
    retired = "--gpu-type"
    assert retired in declared, "pick an option the launcher actually passes"
    monkeypatch.setattr(
        preflight, "_forge_loop_options", lambda _path: declared - {retired}
    )

    assert preflight._print_forge_loop_contract() is False
    assert retired in capsys.readouterr().err


def test_a_matching_forge_loop_passes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A newer KernelForge may accept more than we pass; only shortfalls matter.
    accepted = _declared_options() | {"--some-new-option"}
    monkeypatch.setattr(preflight, "_forge_loop_options", lambda _path: accepted)

    assert preflight._print_forge_loop_contract() is True
    assert "forge_loop_contract_ok" in capsys.readouterr().out


def test_an_absent_kernelforge_does_not_fail_the_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run without KernelForge has no forge backend to break."""
    monkeypatch.setattr(preflight, "_forge_loop_options", lambda _path: None)

    assert preflight._print_forge_loop_contract() is True
