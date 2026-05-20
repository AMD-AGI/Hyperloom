"""grep_scanner fallback tests (P2 PR-E)."""

from __future__ import annotations

from pathlib import Path

import pytest

from framework_agent.agent.grep_scanner import scan_module_via_grep


_FIXTURES = Path(__file__).parent / "fixtures"


def test_grep_scanner_extracts_argparse_flags():
    """Same fixture, but go through the grep path directly."""
    path = _FIXTURES / "mini_sglang/python/sglang/srt/server_args.py"
    flags = scan_module_via_grep(path, framework="sglang", source_root=_FIXTURES / "mini_sglang")
    cli_flags = [f for f in flags if f.surface == "cli"]
    names = {f.flag_name for f in cli_flags}
    assert "--max-running-requests" in names
    assert "--cuda-graph-max-bs" in names
    # via marker downgraded for KB confidence purposes.
    assert all(f.via == "argparse_grep" for f in cli_flags)


def test_grep_scanner_extracts_dataclass_fields():
    """Dataclass fields under @dataclass class header."""
    path = _FIXTURES / "mini_sglang/python/sglang/srt/server_args.py"
    flags = scan_module_via_grep(path, framework="sglang", source_root=_FIXTURES / "mini_sglang")
    dc_flags = [f for f in flags if f.surface == "config"]
    names = {f.flag_name for f in dc_flags}
    # ServerArgs is the @dataclass; max_num_seqs is one of its fields.
    assert "max_num_seqs" in names or "model_path" in names
    assert all(f.via == "dataclass_grep" for f in dc_flags)


def test_grep_scanner_handles_broken_syntax_gracefully():
    """broken.py with malformed syntax -- grep_scanner still walks
    line-by-line and surfaces nothing rather than crashing."""
    path = _FIXTURES / "mini_sglang/python/sglang/srt/configs/broken.py"
    flags = scan_module_via_grep(path, framework="sglang", source_root=_FIXTURES / "mini_sglang")
    # broken.py has no add_argument calls and no class header, so 0 hits
    # is the expected output.
    assert flags == []


def test_grep_scanner_nonexistent_file_returns_empty():
    flags = scan_module_via_grep(Path("/does/not/exist.py"))
    assert flags == []


def test_grep_scanner_pydantic_basemodel_fields():
    """Fields under class X(BaseModel): hit the dataclass_grep branch
    (we don't distinguish pydantic-grep vs dataclass-grep)."""
    path = _FIXTURES / "mini_sglang/python/sglang/srt/configs/scheduler.py"
    flags = scan_module_via_grep(path, framework="sglang", source_root=_FIXTURES / "mini_sglang")
    config_flags = [f for f in flags if f.surface == "config"]
    names = {f.flag_name for f in config_flags}
    assert "max_num_batched_tokens" in names or "enable_dp_attention" in names
