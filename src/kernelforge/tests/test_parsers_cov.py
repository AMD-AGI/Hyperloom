"""Coverage tests for rocprofv3 CSV parser and compiler-output parser.

Pure logic — CSV fixtures via tmp_path, no GPU / no rocprofv3.
"""

from __future__ import annotations


from kernelforge.mcp_server.parsers.compiler_output import (
    RegisterInfo,
    parse_compiler_errors,
    parse_compiler_warnings,
    parse_register_info,
)


# ─── RegisterInfo ───


def test_register_info_occupancy_and_summary():
    info = RegisterInfo(vgpr=200, agpr=64, sgpr=100, lds_bytes=40960, spill_bytes=0)
    analysis = info.occupancy_analysis
    assert "occupancy≥2" in analysis
    assert "AGPR=64" in analysis
    assert "SGPR=100" in analysis
    assert "dual-occupancy OK" in analysis
    assert not info.has_spill
    assert "Analysis:" in info.summary()


def test_register_info_high_pressure_and_spill():
    info = RegisterInfo(vgpr=300, lds_bytes=90 * 1024, spill_bytes=128)
    analysis = info.occupancy_analysis
    assert "occupancy=1 ONLY" in analysis
    assert "single-occupancy" in analysis
    assert "SPILL" in analysis
    assert info.has_spill


def test_register_info_unknown():
    assert RegisterInfo().occupancy_analysis == "unknown"


# ─── parse_register_info fallbacks ───


def test_parse_register_info_primary_patterns():
    text = """
    .vgpr_count: 240
    .agpr_count: 128
    .sgpr_count: 102
    .lds_size: 65536
    ScratchSize: 16
    Occupancy: 2
    """
    info = parse_register_info(text)
    assert (info.vgpr, info.agpr, info.sgpr) == (240, 128, 102)
    assert info.lds_bytes == 65536
    assert info.spill_bytes == 16
    assert info.occupancy == 2


def test_parse_register_info_alternative_patterns():
    text = """
    NumVgprs: 96
    NumSgprs: 48
    LDSByteSize: 8192
    .scratch_memory_size: 64
    """
    info = parse_register_info(text)
    assert info.vgpr == 96
    assert info.sgpr == 48
    assert info.lds_bytes == 8192
    assert info.spill_bytes == 64


# ─── errors / warnings ───


def test_parse_compiler_warnings():
    text = "a.cpp:1: warning: unused var\nb.cpp:2: error: boom\n"
    warnings = parse_compiler_warnings(text)
    assert len(warnings) == 1
    assert "unused var" in warnings[0]


def test_parse_compiler_errors_empty():
    assert parse_compiler_errors("all good\n") == []
