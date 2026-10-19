# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU product identity must survive architecture-sharing and missing SMI tools."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer import gpu_types


@pytest.mark.parametrize("board", ["mi300a", "mi300x", "mi308x", "mi325x", "mi355x"])
def test_smi_product_name_resolves_to_the_actual_board(monkeypatch, board):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=f"GPU[0] : AMD Instinct {board.upper()}")
    )
    assert gpu_types._autodetect_gpu_type() == board


@pytest.mark.parametrize("smi_available", [True, False])
@pytest.mark.parametrize("board", ["mi300a", "mi300x", "mi308x", "mi325x", "mi355x"])
def test_torch_product_name_takes_precedence_over_shared_architecture(monkeypatch, smi_available, board):
    def smi(*args, **kwargs):
        if not smi_available:
            raise FileNotFoundError("rocm-smi")
        return SimpleNamespace(stdout="GPU[0] : Card series: N/A")

    arch, _cus = gpu_types.amd_gpu_dispatch_identity(board)
    props = SimpleNamespace(name=f"AMD Instinct {board.upper()}", gcnArchName=f"{arch}:sramecc+:xnack-")
    monkeypatch.setattr(subprocess, "run", smi)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda _: props))
    )
    assert gpu_types._autodetect_gpu_type() == board


def test_mi300a_custom_runner_keeps_its_own_identity():
    assert gpu_types._gpu_runner_type("mi300a") == "mi300a"


def test_mi300a_session_preserves_product_identity(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.cli import bootstrap
    from hyperloom.inference_optimizer.cli.parser import _build_parser
    from hyperloom.inference_optimizer.session.manifest import build_manifest

    monkeypatch.setenv("FRAMEWORK", "custom")
    monkeypatch.setenv("TARGET_GPU_TYPE", "mi300a")
    monkeypatch.setenv("GPU_TYPE", "mi300a")
    args = _build_parser().parse_args(
        ["optimize", "--model", "/models/custom", "--framework", "custom", "--gpu-type", "mi300a"]
    )
    manifest = build_manifest(tmp_path, args=args, session_id="mi300a-session")
    state = bootstrap._seed_shared_state(tmp_path, args, session_id="mi300a-session")
    assert manifest["gpu_type"] == "mi300a"
    assert state.gpu_type == "mi300a"
    assert gpu_types.amd_gpu_dispatch_identity() == ("gfx942", 228)


@pytest.mark.parametrize("key", ["TARGET_GPU_TYPE", "GPU_TYPE"])
def test_mi300a_environment_hint_resolves_dispatch(monkeypatch, key):
    monkeypatch.delenv("TARGET_GPU_TYPE", raising=False)
    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.setenv(key, "MI300A")
    assert gpu_types.amd_gpu_dispatch_identity() == ("gfx942", 228)
