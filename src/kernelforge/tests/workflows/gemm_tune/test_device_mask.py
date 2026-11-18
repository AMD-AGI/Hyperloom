# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A tuner subprocess sees one HIP/CUDA device mask, whichever of the two it inherited."""

from __future__ import annotations

import sys
from pathlib import Path

from kernelforge.workflows.gemm_tune.utils import run_subprocess

# What the generated vllm_dense_tunableop and vllm_moe_triton scripts do before they import torch.
_TUNER_PREAMBLE = (
    "import os; "
    "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0'); "
    "os.environ.setdefault('HIP_VISIBLE_DEVICES', '0'); "
    "print(os.environ['HIP_VISIBLE_DEVICES'], os.environ['CUDA_VISIBLE_DEVICES'])"
)


def _masks_seen(tmp_path: Path) -> str:
    rc, out, _err = run_subprocess([sys.executable, "-c", _TUNER_PREAMBLE], cwd=tmp_path, timeout_s=60)
    assert rc == 0
    return out.strip()


def test_an_inherited_hip_mask_reaches_cuda_too(tmp_path: Path, monkeypatch) -> None:
    """The aborted tuner: preflight had re-indexed HIP to 0,1 within ROCR, and the script defaulted CUDA to 0."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert _masks_seen(tmp_path) == "0,1 0,1"


def test_an_inherited_cuda_mask_reaches_hip_too(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    assert _masks_seen(tmp_path) == "2 2"


def test_hip_wins_when_the_two_masks_disagree(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert _masks_seen(tmp_path) == "1 1"


def test_without_an_inherited_mask_the_script_pins_its_own_gpu(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert _masks_seen(tmp_path) == "0 0"
