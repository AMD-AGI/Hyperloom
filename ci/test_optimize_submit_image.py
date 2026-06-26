#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for CI optimize image selection."""

from __future__ import annotations

import optimize_submit as opt


def test_mimo_uses_current_sglang_profilerfix_image():
    image = opt.detect_image("sglang", "XiaomiMiMo/MiMo-V2-7B")

    assert image == opt._default_sglang_image()
    assert "v0.5.12-rocm720-mi30x-profilerfix" in image
    assert "v0.5.11" not in image
