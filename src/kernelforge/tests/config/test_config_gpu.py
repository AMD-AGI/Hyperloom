# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Config never assumes a GPU the run did not name."""

from __future__ import annotations

from kernelforge.config import Config


def test_an_unresolved_gpu_target_names_no_chip(monkeypatch):
    # Regression: every caller that did not resolve the run's arch was handed gfx942.
    monkeypatch.delenv("GPU_TARGET", raising=False)

    assert Config().gpu_target == ""
    assert Config.from_env().gpu_target == ""


def test_the_gpu_target_environment_is_honoured(monkeypatch):
    monkeypatch.setenv("GPU_TARGET", "gfx950")

    assert Config.from_env().gpu_target == "gfx950"
