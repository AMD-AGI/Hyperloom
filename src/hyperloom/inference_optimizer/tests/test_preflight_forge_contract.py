# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launch-time gate on the forge-loop argv contract.

Hyperloom resolves KernelForge through ``$FORGE_PATH``, so which build a session
will dispatch is only knowable at launch. A retired option costs every forge
attempt with rc=2 while the session still finishes reporting success, so the
preflight refuses to start instead.
"""

from __future__ import annotations

import pytest

from hyperloom.agents.kernel.tools.backends import forge_submit
from hyperloom.inference_optimizer.cli import preflight


def test_a_forge_loop_missing_one_option_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    retired = "--gpu-type"
    assert retired in forge_submit.FORGE_LOOP_OPTIONS
    monkeypatch.setattr(forge_submit, "forge_loop_contract_gaps", lambda: [retired])

    with pytest.raises(SystemExit) as exit_info:
        preflight._check_forge_loop_contract()

    assert exit_info.value.code == 2
    assert retired in capsys.readouterr().err


def test_a_matching_forge_loop_starts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(forge_submit, "forge_loop_contract_gaps", lambda: [])

    preflight._check_forge_loop_contract()

    assert "forge-loop contract OK" in capsys.readouterr().out


def test_an_uninspectable_kernelforge_does_not_block_the_launch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A run that cannot import KernelForge has no forge backend to break."""
    monkeypatch.setattr(forge_submit, "forge_loop_contract_gaps", lambda: None)

    preflight._check_forge_loop_contract()

    assert "unchecked" in capsys.readouterr().out


def test_the_gate_reads_the_kernelforge_a_dispatch_would_use(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """`$FORGE_PATH` decides, so the gate must resolve it the same way.

    Reading an installed package while a dispatch would use $FORGE_PATH would
    check a build that never runs.
    """
    resolved: list[str] = []
    monkeypatch.setattr(
        forge_submit,
        "_ensure_forge_on_path",
        lambda: resolved.append("resolved") or "",
    )
    monkeypatch.setenv("FORGE_PATH", str(tmp_path))

    forge_submit.forge_loop_contract_gaps()

    assert resolved == ["resolved"]
