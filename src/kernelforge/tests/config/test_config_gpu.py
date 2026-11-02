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


def test_the_gpu_type_follows_the_gpu_target(monkeypatch):
    # Regression: the SKU defaulted to mi355x whatever the arch, filing gfx942 runs under MI355X.
    monkeypatch.setenv("GPU_TARGET", "gfx942")

    assert Config.from_env().gpu_type == "mi300x"
    assert Config.from_env(gpu_type="MI325X").gpu_type == "mi325x"


def test_no_gpu_target_names_no_gpu_type(monkeypatch):
    monkeypatch.delenv("GPU_TARGET", raising=False)

    assert Config().gpu_type == ""
    assert Config.from_env().gpu_type == ""
