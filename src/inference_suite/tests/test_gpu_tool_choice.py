# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of which SMI tool reads a host's cards.

``amd-smi`` is the platform's own and wins wherever it is installed, so an AMD machine reads exactly what it read
before this exists. A host without it is not an AMD host and is read through ``nvidia-smi``. With neither tool the
name still comes back as the platform's own, so the refusal reads the way it always has instead of claiming a check
that cannot exist.
"""

from __future__ import annotations

from inference_suite.runtime import gpus


def test_the_platforms_own_tool_wins_where_it_is_installed(monkeypatch) -> None:
    monkeypatch.setattr(
        gpus.shutil, "which", lambda name: f"/usr/bin/{name}" if name in {"amd-smi", "nvidia-smi"} else None
    )
    assert gpus._smi_tool() == "amd-smi"


def test_a_host_without_the_platforms_tool_is_read_through_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr(gpus.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    assert gpus._smi_tool() == "nvidia-smi"


def test_a_host_with_neither_tool_still_names_the_platform_own(monkeypatch) -> None:
    monkeypatch.setattr(gpus.shutil, "which", lambda name: None)
    assert gpus._smi_tool() == "amd-smi"
