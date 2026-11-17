# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The speculative-decoding reference names the golden layer inference_suite enforces."""

from __future__ import annotations

from inference_suite import golden

from hyperloom.inference_optimizer.session.paths import asset_prompt_references_dir


def test_the_speculative_decoding_reference_points_at_the_golden_layer_inference_suite_enforces():
    doc = (asset_prompt_references_dir() / "speculative_decoding.md").read_text(encoding="utf-8")
    assert "## A base with a golden layer owns acceptance" in doc
    assert "inference_suite.golden" in doc
    for name in golden.__all__:
        if name.startswith("GOLDEN_OWNED_"):
            assert f"`{name}`" in doc
