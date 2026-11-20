# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelforge.workflows.rewrite_controller import (
    ControllerLayout,
    HandoffContractError,
    read_handoff,
)
from kernelforge.workflows.rewrite_controller.contracts import (
    REPOSITORIES_FILENAME,
    SERVING_CONTEXT_FILENAME,
    TRACE_EVIDENCE_FILENAME,
    WORKLOAD_FILENAME,
)
from kernelforge.workflows.rewrite_controller.paths import (
    operator_directory_name,
    safe_operator_id,
    safe_relative_path,
)

_DOCUMENTS = (WORKLOAD_FILENAME, SERVING_CONTEXT_FILENAME, TRACE_EVIDENCE_FILENAME)


def _repositories(*roots: object) -> str:
    return json.dumps({"repositories": [str(root) for root in roots]})


def _handoff(tmp_path: Path, repositories: str) -> Path:
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    for name in _DOCUMENTS:
        (handoff / name).write_text("", encoding="utf-8")
    (handoff / REPOSITORIES_FILENAME).write_text(repositories, encoding="utf-8")
    return handoff


def test_read_handoff_returns_all_markdown_documents_and_the_declared_repositories(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    repository = tmp_path / "framework"
    documents = {
        WORKLOAD_FILENAME: "# Workload\n",
        SERVING_CONTEXT_FILENAME: "# Serving Context\n",
        TRACE_EVIDENCE_FILENAME: "# Trace Evidence\n",
        REPOSITORIES_FILENAME: _repositories(repository),
    }
    for name, text in documents.items():
        (handoff / name).write_text(text, encoding="utf-8")

    bundle = read_handoff(handoff)

    assert bundle.root == handoff.resolve()
    assert bundle.workload == documents[WORKLOAD_FILENAME]
    assert bundle.serving_context == documents[SERVING_CONTEXT_FILENAME]
    assert bundle.trace_evidence == documents[TRACE_EVIDENCE_FILENAME]
    assert bundle.repositories == (repository.resolve(),)


@pytest.mark.parametrize("missing", [*_DOCUMENTS, REPOSITORIES_FILENAME])
def test_read_handoff_rejects_a_missing_required_document(tmp_path: Path, missing: str) -> None:
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    for name in _DOCUMENTS:
        if name != missing:
            (handoff / name).write_text("", encoding="utf-8")
    if missing != REPOSITORIES_FILENAME:
        (handoff / REPOSITORIES_FILENAME).write_text(_repositories(tmp_path), encoding="utf-8")

    with pytest.raises(HandoffContractError, match=missing):
        read_handoff(handoff)


@pytest.mark.parametrize("entries", [[], ["relative/repo"], [7]])
def test_repositories_must_list_at_least_one_absolute_path(tmp_path: Path, entries: list) -> None:
    with pytest.raises(HandoffContractError, match=REPOSITORIES_FILENAME):
        read_handoff(_handoff(tmp_path, json.dumps({"repositories": entries})))


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


def test_operator_directory_name_encodes_canonical_separators(operator_id: str) -> None:
    directory = operator_directory_name(operator_id)

    assert ":" not in directory
    assert directory.startswith("kernel%3Aforge-loop%3A")
    assert len(directory.encode("utf-8")) <= 240


def test_operator_directory_name_is_injective_over_percent_signs() -> None:
    """Two identities must never share one directory."""
    encoded_colon = operator_directory_name("kernel:a:b:c:d:e:f")
    literal_escape = operator_directory_name("kernel:a%3Ab:c:d:e:f")

    assert encoded_colon != literal_escape


def test_operator_directory_name_hashes_values_above_portable_limit() -> None:
    operator_id = "kernel:" + ":".join(["segment" * 60] * 6)

    directory = operator_directory_name(operator_id)

    assert directory.startswith("operator-")
    assert len(directory.encode("utf-8")) <= 240
