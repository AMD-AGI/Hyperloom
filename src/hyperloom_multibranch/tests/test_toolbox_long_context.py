# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The context a rendered point's launch serves, read as the runner reads it, or ``None`` when it cannot be read."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.framework_registry import FRAMEWORKS
from hyperloom_multibranch.toolbox.long_context import launch_served_context


def _point(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "point.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_context_flag_of_the_launch_is_what_it_serves(tmp_path: Path) -> None:
    args = FRAMEWORKS["sglang"].extra_args_env
    text = f"benchmark:\n  framework: sglang\n  envs:\n    {args}: --context-length 8192\n"

    assert launch_served_context(_point(tmp_path, text)) == 8192


@pytest.mark.parametrize("text", ["", "- a\n", "other: {}\n", "benchmark:\n  envs: [TP]\n", "benchmark: [\n"])
def test_a_point_that_cannot_be_read_serves_none(tmp_path: Path, text: str) -> None:
    assert launch_served_context(_point(tmp_path, text)) is None


def test_a_missing_point_serves_none(tmp_path: Path) -> None:
    assert launch_served_context(tmp_path / "absent.yaml") is None
