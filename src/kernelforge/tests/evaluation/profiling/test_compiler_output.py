"""Tests for compiler-output register parsing."""

from __future__ import annotations


from kernelforge.evaluation.profiling.compiler_output import (
    RegisterInfo,
    parse_register_info,
)
from kernelforge.utils.gpu_arch import arch_profile


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


def test_register_info_reads_the_occupancy_steps_of_its_target():
    rdna3 = arch_profile("gfx1100").occupancy

    full = RegisterInfo(vgpr=96, lds_bytes=64 * 1024, occupancy_rule=rdna3).occupancy_analysis
    assert "VGPR=96 (≤96: 16 waves/SIMD" in full
    assert "LDS=64.0KB (≤64KB: fits the 64 KiB work-group cap" in full

    pressured = RegisterInfo(vgpr=128, lds_bytes=65 * 1024, occupancy_rule=rdna3).occupancy_analysis
    assert "VGPR=128 (>96: fewer than 16 waves/SIMD" in pressured
    assert "will not launch" in pressured
    assert "occupancy=1 ONLY" not in pressured


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
