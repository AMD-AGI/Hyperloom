# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom.inspector.model_names import normalize


@pytest.mark.parametrize(
    "spelling",
    [
        "claude-opus-4-6",
        "Claude-Opus-4.6",
        "anthropic/claude-opus-4-6",
        "us.anthropic.claude-opus-4-6-v1:0",
        "claude-opus-4-6@20260101",
        "claude-opus-4-6-20260101",
        "claude-opus-4-6[1m]",
    ],
)
def test_model_spellings_normalize_alike(spelling):
    assert normalize(spelling) == "claude-opus-4-6"


@pytest.mark.parametrize(
    "name",
    ["x-20260101-latest", "x[1m][2m]", "x-v1:0-latest", "us.anthropic.claude-opus-4-6-v1:0", "deepseek-v3", "a.b_c d"],
)
def test_a_normalized_name_normalizes_to_itself(name):
    assert normalize(normalize(name)) == normalize(name)


def test_a_bare_version_stays_part_of_the_name():
    assert normalize("deepseek-v3") == "deepseek-v3" != normalize("deepseek-v4")  # not both "deepseek"
    assert normalize("jina-embeddings-v3") == "jina-embeddings-v3"
    assert normalize("anthropic.claude-opus-4-6-v2:0") == "claude-opus-4-6"  # Bedrock's version suffix
