# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from kernelforge.kernel_rewrite_controller import (
    ControllerLayout,
    HandoffContractError,
    read_handoff,
)
from kernelforge.kernel_rewrite_controller.contracts import (
    SERVING_CONTEXT_FILENAME,
    TRACE_EVIDENCE_FILENAME,
    WORKLOAD_FILENAME,
)
from kernelforge.kernel_rewrite_controller.paths import (
    safe_operator_id,
    safe_relative_path,
)


def test_read_handoff_returns_all_markdown_documents(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    documents = {
        WORKLOAD_FILENAME: "# Workload\n",
        SERVING_CONTEXT_FILENAME: "# Serving Context\n",
        TRACE_EVIDENCE_FILENAME: "# Trace Evidence\n",
    }
    for name, text in documents.items():
        (handoff / name).write_text(text, encoding="utf-8")

    bundle = read_handoff(handoff)

    assert bundle.root == handoff.resolve()
    assert bundle.workload == documents[WORKLOAD_FILENAME]
    assert bundle.serving_context == documents[SERVING_CONTEXT_FILENAME]
    assert bundle.trace_evidence == documents[TRACE_EVIDENCE_FILENAME]


@pytest.mark.parametrize(
    "missing",
    [WORKLOAD_FILENAME, SERVING_CONTEXT_FILENAME, TRACE_EVIDENCE_FILENAME],
)
def test_read_handoff_rejects_a_missing_required_document(tmp_path: Path, missing: str) -> None:
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    for name in (WORKLOAD_FILENAME, SERVING_CONTEXT_FILENAME, TRACE_EVIDENCE_FILENAME):
        if name != missing:
            (handoff / name).write_text("", encoding="utf-8")

    with pytest.raises(HandoffContractError, match=missing):
        read_handoff(handoff)


def test_controller_layout_anchors_all_paths_under_output_dir(
    tmp_path: Path,
    operator_id: str,
) -> None:
    layout = ControllerLayout(tmp_path / "output")

    assert layout.tasks_root == (tmp_path / "output" / "controller" / "tasks").resolve()
    assert layout.workspace_dir(operator_id).parent == layout.workspaces_root
    assert layout.patch_dir(operator_id).parent == layout.patches_root


@pytest.mark.parametrize("value", ["", ".", "..", "/absolute", "../escape", "a\\b", "a\x00b"])
def test_safe_relative_path_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError):
        safe_relative_path(value, field_name="kernel_path")


@pytest.mark.parametrize("value", ["", ".", "..", "a/b", "a\\b", "a\x00b"])
def test_safe_operator_id_rejects_unsafe_directory_names(value: str) -> None:
    with pytest.raises(ValueError):
        safe_operator_id(value)
