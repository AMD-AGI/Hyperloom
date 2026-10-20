# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exit codes of the session-report CLI when its input cannot be read."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.tools import dump_session_report


def _unusable_input(tmp_path: Path, case: str) -> Path:
    path = tmp_path / "session_breakdown.json"
    if case == "directory":
        path.mkdir()
    else:
        path.write_text("{bad", encoding="utf-8")
    return path


@pytest.mark.parametrize("case", ["directory", "invalid_json"])
def test_unreadable_breakdown_exits_with_2(tmp_path: Path, case: str) -> None:
    path = _unusable_input(tmp_path, case)

    assert dump_session_report.main(["--input", str(path), "--no-llm"]) == 2


def test_unexpected_read_failure_surfaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "session_breakdown.json"
    path.write_text("{}", encoding="utf-8")

    def _boom(*_a, **_k):
        raise RuntimeError("read broke")

    monkeypatch.setattr(Path, "read_text", _boom)

    with pytest.raises(RuntimeError, match="read broke"):
        dump_session_report.main(["--input", str(path), "--no-llm"])
