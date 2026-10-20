"""Reading a declared source file for operation-definition lookup."""

from __future__ import annotations

import pytest

from kernelforge.knowledge.kb_store.identity import source_resolution


@pytest.mark.parametrize(
    "path_of",
    [
        lambda tmp_path: str(tmp_path / "missing.py"),
        lambda tmp_path: str(tmp_path),
    ],
    ids=["missing", "directory"],
)
def test_an_unreadable_source_reads_as_empty(tmp_path, path_of):
    assert source_resolution._read_text_safe(path_of(tmp_path)) == ""


def test_a_path_that_cannot_be_opened_at_all_is_not_an_unreadable_source():
    with pytest.raises(ValueError, match="embedded null byte"):
        source_resolution._read_text_safe("kernel\0.py")
