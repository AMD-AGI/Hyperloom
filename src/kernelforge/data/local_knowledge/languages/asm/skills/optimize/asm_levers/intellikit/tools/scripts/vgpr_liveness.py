#!/usr/bin/env python3
"""
VGPR Liveness Analyzer for AMD gfx950 (CDNA4) assembly kernels.

Parses AMDGCN assembly (.s files), computes per-VGPR liveness intervals,
identifies dead windows, and suggests register remappings to improve occupancy.

Usage:
    python3 vgpr_liveness.py kernel.s [--target-vgprs N] [--json] [--verbose]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Occupancy helpers
# ---------------------------------------------------------------------------

def vgpr_alloc(n: int) -> int:
    """VGPRs are allocated in groups of 4."""
    return ((n + 3) // 4) * 4


def max_waves(next_free_vgpr: int) -> int:
    """Compute max waves/SIMD for gfx950 (512 unified VGPRs)."""
    if next_free_vgpr == 0:
        return 8  # degenerate
    return 512 // vgpr_alloc(next_free_vgpr)


def next_occupancy_boundary(current_vgprs: int) -> Optional[int]:
    """
    Find the largest next_free_vgpr N that would give one more wave than current.
    Returns None if already at max occupancy (8 waves).

    We need: floor(512 / ceil(N/4)*4) >= target_waves
    So: ceil(N/4)*4 <= floor(512 / target_waves)
    The max allocation block is floor(512 / target_waves), rounded down to multiple of 4.
    The max N is that allocation block size.
    """
    cur_waves = max_waves(current_vgprs)
    if cur_waves >= 8:
        return None
    target_waves = cur_waves + 1
    # Max allocation size for target_waves
    max_alloc = (512 // target_waves)
    # Round down to multiple of 4
    max_alloc = (max_alloc // 4) * 4
    if max_alloc == 0:
        return None
    # max N = max_alloc (since vgpr_alloc(max_alloc) = max_alloc when it's a multiple of 4)
    return max_alloc


# ---------------------------------------------------------------------------
# Assembly parser
# ---------------------------------------------------------------------------

# Match VGPR references: v0, v83, v[2:17], v[66:67]
RE_VREG_SINGLE = re.compile(r'\bv(\d+)\b')
RE_VREG_RANGE = re.compile(r'\bv\[(\d+):(\d+)\]')

# Match labels
RE_LABEL = re.compile(r'^(\.[A-Za-z_]\w*|BB\d+_\d+):')

# Match branch instructions
RE_BRANCH = re.compile(r'\bs_(branch|cbranch_\w+)\b')

# Match comment-only lines
RE_COMMENT = re.compile(r'^\s*(;|//|$)')

# Match directive lines
RE_DIRECTIVE = re.compile(r'^\s*\.')

# Match llvm-objdump disassembly trailing encoding: // 000000002300: BE860003
RE_OBJDUMP_SUFFIX = re.compile(r'\s*//\s*[0-9A-Fa-f]+:\s+[0-9A-Fa-f\s]+$')

# Match llvm-objdump section headers and symbol lines
RE_OBJDUMP_HEADER = re.compile(r'^Disassembly of section|^[0-9a-f]+ <.*>:|file format')

# Match .amdhsa_next_free_vgpr
RE_NEXT_FREE_VGPR = re.compile(r'\.amdhsa_next_free_vgpr\s+(\d+)')
RE_ACCUM_OFFSET = re.compile(r'\.amdhsa_accum_offset\s+(\d+)')

# Store instructions (data source is NOT the first operand)
STORE_MNEMONICS = {
    'buffer_store_dword', 'buffer_store_dwordx2', 'buffer_store_dwordx4',
    'buffer_store_short', 'buffer_store_byte', 'buffer_store_dwordx3',
    'buffer_store_short_d16_hi', 'buffer_store_byte_d16_hi',
    'global_store_dword', 'global_store_dwordx2', 'global_store_dwordx4',
    'global_store_short', 'global_store_byte', 'global_store_dwordx3',
    'flat_store_dword', 'flat_store_dwordx2', 'flat_store_dwordx4',
    'flat_store_short', 'flat_store_byte', 'flat_store_dwordx3',
}

DS_WRITE_MNEMONICS = {
    'ds_write_b32', 'ds_write_b64', 'ds_write_b128',
    'ds_write2_b32', 'ds_write2_b64',
    'ds_write_b8', 'ds_write_b16',
}

# DS read instructions write to the first (dest) operand
DS_READ_MNEMONICS = {
    'ds_read_b32', 'ds_read_b64', 'ds_read_b128',
    'ds_read2_b32', 'ds_read2_b64',
    'ds_read_b8', 'ds_read_b16',
}

# LDS with special semantics
DS_SPECIAL_MNEMONICS = {
    'ds_bpermute_b32',  # dest=first, src addr=second, src data=third
}

# MFMA instructions: last operand is accumulator (both read and written)
RE_MFMA = re.compile(r'^v_mfma_')

# Instructions that write to SGPR/VCC, not VGPR (first operand is not dest VGPR)
SGPR_DEST_PREFIXES = ('v_cmp_', 'v_cmpx_')

# Buffer load with LDS destination (writes to LDS, not VGPR)
RE_BUFFER_LOAD_LDS = re.compile(r'buffer_load_\w+.*\blds\b')


@dataclass
class VGPRAccess:
    """A single read or write to a VGPR on a specific line."""
    line: int
    is_def: bool  # True = write/define, False = read/use


@dataclass
class LiveInterval:
    """A contiguous interval [first_def, last_use] for a VGPR."""
    first_def: int
    last_use: int

    def overlaps(self, other: 'LiveInterval') -> bool:
        return self.first_def <= other.last_use and other.first_def <= self.last_use


@dataclass
class StructureInfo:
    """Basic loop/branch structure info."""
    labels: dict = field(default_factory=dict)        # label -> line number
    branches: list = field(default_factory=list)       # (line, target_label_or_offset)
    back_edges: list = field(default_factory=list)     # (branch_line, target_line) for loops
    loop_ranges: list = field(default_factory=list)    # (start_line, end_line)


def parse_vgpr_refs(operand_str: str) -> list[int]:
    """Extract all VGPR numbers referenced in an operand string."""
    vgprs = []
    # First find ranges v[N:M]
    for m in RE_VREG_RANGE.finditer(operand_str):
        lo, hi = int(m.group(1)), int(m.group(2))
        vgprs.extend(range(lo, hi + 1))
    # Remove range matches from string to avoid double-counting
    cleaned = RE_VREG_RANGE.sub('', operand_str)
    # Then find singles vN
    for m in RE_VREG_SINGLE.finditer(cleaned):
        vgprs.append(int(m.group(1)))
    return vgprs


def split_operands(operand_str: str) -> list[str]:
    """
    Split an instruction's operand string by commas, respecting brackets.
    E.g. "v[2:17], v[66:67], v[2:17]" -> ["v[2:17]", "v[66:67]", "v[2:17]"]
    """
    parts = []
    depth = 0
    current = []
    for ch in operand_str:
        if ch in '([':
            depth += 1
            current.append(ch)
        elif ch in ')]':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())
    return parts


def classify_instruction(mnemonic: str, operands: list[str], full_line: str):
    """
    Classify which operands are defs (written) and which are uses (read).
    Returns (def_vgprs: list[int], use_vgprs: list[int]).
    """
    def_vgprs = []
    use_vgprs = []

    if not operands:
        return def_vgprs, use_vgprs

    # buffer_load with LDS destination - no VGPR dest, all operands are uses
    if RE_BUFFER_LOAD_LDS.search(full_line):
        for op in operands:
            use_vgprs.extend(parse_vgpr_refs(op))
        return def_vgprs, use_vgprs

    # MFMA instructions: v_mfma_* dst, srcA, srcB, accum
    # dst and accum are the same register (written), srcA and srcB are read
    # accum is also read (it's an accumulate)
    if RE_MFMA.match(mnemonic):
        if len(operands) >= 4:
            # First operand = destination (written)
            def_vgprs.extend(parse_vgpr_refs(operands[0]))
            # Source operands (read)
            use_vgprs.extend(parse_vgpr_refs(operands[1]))
            use_vgprs.extend(parse_vgpr_refs(operands[2]))
            # Accumulator (last operand) - read AND written
            accum_vgprs = parse_vgpr_refs(operands[3])
            use_vgprs.extend(accum_vgprs)
            # If accum is different from dst, also mark accum as def
            # (but typically they're the same for in-place MFMA)
            # The def is already captured via operands[0]
        return def_vgprs, use_vgprs

    # v_cmp_* / v_cmpx_* write to SGPR/VCC, all VGPR operands are reads
    if any(mnemonic.startswith(p) for p in SGPR_DEST_PREFIXES):
        for op in operands:
            use_vgprs.extend(parse_vgpr_refs(op))
        return def_vgprs, use_vgprs

    # Store instructions: first VGPR operand is data (read), rest are address (read)
    if mnemonic in STORE_MNEMONICS:
        for op in operands:
            use_vgprs.extend(parse_vgpr_refs(op))
        return def_vgprs, use_vgprs

    # DS write instructions: all VGPR operands are reads (addr + data)
    if mnemonic in DS_WRITE_MNEMONICS:
        for op in operands:
            use_vgprs.extend(parse_vgpr_refs(op))
        return def_vgprs, use_vgprs

    # DS read instructions: first operand is dest (written), rest are addr (read)
    if mnemonic in DS_READ_MNEMONICS:
        def_vgprs.extend(parse_vgpr_refs(operands[0]))
        for op in operands[1:]:
            use_vgprs.extend(parse_vgpr_refs(op))
        return def_vgprs, use_vgprs

    # ds_bpermute_b32: dest, addr, src_data
    if mnemonic in DS_SPECIAL_MNEMONICS:
        if len(operands) >= 1:
            def_vgprs.extend(parse_vgpr_refs(operands[0]))
        for op in operands[1:]:
            use_vgprs.extend(parse_vgpr_refs(op))
        return def_vgprs, use_vgprs

    # Load instructions: first operand is dest (written), rest are address (read)
    if any(mnemonic.startswith(p) for p in (
        'buffer_load_', 'global_load_', 'flat_load_',
    )):
        def_vgprs.extend(parse_vgpr_refs(operands[0]))
        for op in operands[1:]:
            use_vgprs.extend(parse_vgpr_refs(op))
        return def_vgprs, use_vgprs

    # v_readfirstlane_b32 writes to SGPR, reads VGPR
    if mnemonic == 'v_readfirstlane_b32':
        # First operand is SGPR dest, second is VGPR source
        for op in operands[1:]:
            use_vgprs.extend(parse_vgpr_refs(op))
        return def_vgprs, use_vgprs

    # v_readlane_b32 writes to SGPR, reads VGPR
    if mnemonic == 'v_readlane_b32':
        for op in operands[1:]:
            use_vgprs.extend(parse_vgpr_refs(op))
        return def_vgprs, use_vgprs

    # v_writelane_b32 dst_vgpr, src_sgpr, lane_sgpr - writes to VGPR
    if mnemonic == 'v_writelane_b32':
        def_vgprs.extend(parse_vgpr_refs(operands[0]))
        for op in operands[1:]:
            use_vgprs.extend(parse_vgpr_refs(op))
        return def_vgprs, use_vgprs

    # v_mbcnt_lo_u32_b32 / v_mbcnt_hi_u32_b32: dest is VGPR, sources may include VGPR
    # Generic VALU: first operand is dest (written), rest are sources (read)
    # This covers v_mov_b32, v_add_u32, v_and_b32, v_or_b32, v_mul_f32, etc.
    # Also covers v_mad_u64_u32, v_fma_f32, v_cndmask_b32, etc.
    # Also covers v_cvt_*, v_exp_f32, v_rcp_f32, v_log_f32, etc.
    # Also covers v_pk_mul_f32 (packed) - first operand is dest range

    # Special case: v_mad_u64_u32 vDST, sCC, vSRC0, vSRC1, vSRC2
    # dst is first, sCC is second (SGPR), rest are sources
    if mnemonic in ('v_mad_u64_u32', 'v_mad_i64_i32'):
        def_vgprs.extend(parse_vgpr_refs(operands[0]))
        # operands[1] is SGPR carry-out, skip
        for op in operands[2:]:
            use_vgprs.extend(parse_vgpr_refs(op))
        return def_vgprs, use_vgprs

    # v_div_scale_f32 dst, sCC, src0, src1, src2
    if mnemonic == 'v_div_scale_f32':
        def_vgprs.extend(parse_vgpr_refs(operands[0]))
        # operands[1] is SGPR pair, skip for VGPR tracking
        for op in operands[2:]:
            use_vgprs.extend(parse_vgpr_refs(op))
        return def_vgprs, use_vgprs

    # Default VALU pattern: first operand = dest, rest = sources
    # Only if instruction starts with 'v_' (VALU/VOP)
    if mnemonic.startswith('v_'):
        def_vgprs.extend(parse_vgpr_refs(operands[0]))
        for op in operands[1:]:
            use_vgprs.extend(parse_vgpr_refs(op))
        return def_vgprs, use_vgprs

    # SGPR instructions (s_*) - only track VGPR reads if any VGPR appears
    # (rare but possible, e.g. s_* doesn't typically use VGPRs)
    # Just collect all VGPR refs as uses (conservative)
    for op in operands:
        use_vgprs.extend(parse_vgpr_refs(op))

    return def_vgprs, use_vgprs


def parse_assembly(lines: list[str], verbose: bool = False):
    """
    Parse assembly lines and return:
    - vgpr_accesses: dict[int, list[VGPRAccess]] per VGPR number
    - declared_vgpr_count: from .amdhsa_next_free_vgpr if present
    - structure: StructureInfo
    - instruction_count: number of instruction lines parsed
    """
    vgpr_accesses: dict[int, list[VGPRAccess]] = defaultdict(list)
    structure = StructureInfo()
    declared_vgpr_count = None
    accum_offset = None
    instruction_count = 0
    in_code_section = False

    for line_num_0, raw_line in enumerate(lines):
        line_num = line_num_0 + 1  # 1-indexed
        line = raw_line.rstrip()

        # Track .amdhsa_next_free_vgpr
        m = RE_NEXT_FREE_VGPR.search(line)
        if m:
            declared_vgpr_count = int(m.group(1))
            continue

        m = RE_ACCUM_OFFSET.search(line)
        if m:
            accum_offset = int(m.group(1))
            continue

        # Skip llvm-objdump headers (file format, section, symbol lines)
        if RE_OBJDUMP_HEADER.search(line):
            # "Disassembly of section .text:" means we're entering code
            if 'section .text' in line:
                in_code_section = True
            continue

        # Skip comments
        if RE_COMMENT.match(line):
            continue

        # Track .text section
        if line.strip() == '.text':
            in_code_section = True
            continue

        # Stop at .rodata or metadata
        if line.strip() in ('.rodata', '.amdgpu_metadata'):
            in_code_section = False
            continue

        # Skip directives (but track labels)
        label_m = RE_LABEL.match(line.strip())
        if label_m:
            structure.labels[label_m.group(1)] = line_num
            continue

        if RE_DIRECTIVE.match(line) and not label_m:
            continue

        if not in_code_section:
            continue

        # Parse instruction
        stripped = line.strip()
        if not stripped or stripped.startswith(';') or stripped.startswith('//'):
            continue

        # Strip llvm-objdump encoding suffix: // 000000002300: BE860003
        stripped = RE_OBJDUMP_SUFFIX.sub('', stripped).rstrip()
        if not stripped:
            continue

        # Extract mnemonic and operands
        # Handle instructions like: v_mfma_f32_32x32x16_bf16 v[2:17], v[66:67], v[68:69], v[2:17]
        parts = stripped.split(None, 1)
        if not parts:
            continue

        mnemonic = parts[0]

        # Skip non-instruction lines (labels already handled)
        if mnemonic.endswith(':'):
            # For llvm-objdump, symbol labels look like "symbol_name:"
            # but they may also be hex-address labels from the disassembler
            continue

        # Skip pure scalar instructions that can't reference VGPRs
        # (optimization: still parse them to be safe)

        operand_str = parts[1] if len(parts) > 1 else ''

        # Remove inline comments (but not the objdump suffix which was already stripped)
        comment_idx = operand_str.find('//')
        if comment_idx >= 0:
            operand_str = operand_str[:comment_idx]
        comment_idx = operand_str.find(';')
        if comment_idx >= 0:
            operand_str = operand_str[:comment_idx]

        operands = split_operands(operand_str)
        instruction_count += 1

        # Track branches for structure analysis
        if RE_BRANCH.search(mnemonic):
            target = operands[-1].strip() if operands else ''
            structure.branches.append((line_num, target))

        # Classify def/use
        def_vgprs, use_vgprs = classify_instruction(mnemonic, operands, stripped)

        for v in def_vgprs:
            vgpr_accesses[v].append(VGPRAccess(line=line_num, is_def=True))

        for v in use_vgprs:
            vgpr_accesses[v].append(VGPRAccess(line=line_num, is_def=False))

        if verbose and (def_vgprs or use_vgprs):
            def_str = ', '.join(f'v{v}' for v in sorted(set(def_vgprs))) if def_vgprs else '-'
            use_str = ', '.join(f'v{v}' for v in sorted(set(use_vgprs))) if use_vgprs else '-'
            print(f"  L{line_num:4d}: {mnemonic:40s}  DEF={def_str:30s}  USE={use_str}")

    # Detect back-edges (branch to earlier line = likely loop)
    for br_line, target in structure.branches:
        # Target could be a label or a numeric offset
        if target in structure.labels:
            target_line = structure.labels[target]
            if target_line < br_line:
                structure.back_edges.append((br_line, target_line))
                structure.loop_ranges.append((target_line, br_line))

    return vgpr_accesses, declared_vgpr_count, accum_offset, structure, instruction_count


# ---------------------------------------------------------------------------
# Liveness analysis
# ---------------------------------------------------------------------------

def compute_liveness(
    vgpr_accesses: dict[int, list[VGPRAccess]],
    structure: StructureInfo,
    first_code_line: int,
    last_code_line: int,
) -> dict[int, list[LiveInterval]]:
    """
    Compute live intervals for each VGPR.

    Strategy: conservative. For each VGPR, find contiguous def->use intervals.
    If a VGPR is used inside any loop, extend its liveness to cover the entire
    loop range (since it could be live across iterations).
    """
    liveness: dict[int, list[LiveInterval]] = {}

    for vgpr, accesses in sorted(vgpr_accesses.items()):
        if not accesses:
            continue

        # Sort by line number
        sorted_accesses = sorted(accesses, key=lambda a: a.line)

        # Build intervals: each def starts a new interval, ended by the last
        # use before the next def (or end of function)
        intervals = []
        current_def = None
        current_last_use = None

        for acc in sorted_accesses:
            if acc.is_def:
                # If we had an open interval, close it
                if current_def is not None:
                    last = current_last_use if current_last_use is not None else current_def
                    intervals.append(LiveInterval(current_def, last))
                current_def = acc.line
                current_last_use = acc.line  # def also counts as a use (value is created)
            else:
                # Use
                if current_def is None:
                    # Use before any def in this function - the register is
                    # live-in (e.g., function argument or uninitialized read).
                    # Start interval from the first code line.
                    current_def = first_code_line
                current_last_use = acc.line

        # Close the last interval
        if current_def is not None:
            last = current_last_use if current_last_use is not None else current_def
            intervals.append(LiveInterval(current_def, last))

        # Merge overlapping/adjacent intervals
        intervals.sort(key=lambda iv: iv.first_def)
        merged = []
        for iv in intervals:
            if merged and iv.first_def <= merged[-1].last_use + 1:
                merged[-1].last_use = max(merged[-1].last_use, iv.last_use)
            else:
                merged.append(LiveInterval(iv.first_def, iv.last_use))

        # Conservative loop extension: if any access falls within a loop,
        # extend the interval to cover the full loop range
        if structure.loop_ranges:
            extended = []
            for iv in merged:
                new_start = iv.first_def
                new_end = iv.last_use
                for loop_start, loop_end in structure.loop_ranges:
                    # If the interval overlaps with the loop at all
                    if iv.first_def <= loop_end and iv.last_use >= loop_start:
                        new_start = min(new_start, loop_start)
                        new_end = max(new_end, loop_end)
                extended.append(LiveInterval(new_start, new_end))

            # Re-merge after extension
            extended.sort(key=lambda iv: iv.first_def)
            merged = []
            for iv in extended:
                if merged and iv.first_def <= merged[-1].last_use + 1:
                    merged[-1].last_use = max(merged[-1].last_use, iv.last_use)
                else:
                    merged.append(LiveInterval(iv.first_def, iv.last_use))

        liveness[vgpr] = merged

    return liveness


def find_dead_windows(
    liveness: dict[int, list[LiveInterval]],
    first_code_line: int,
    last_code_line: int,
) -> dict[int, list[LiveInterval]]:
    """
    For each VGPR, find windows where it is NOT live (dead windows).
    Returns dict[vgpr] -> list of LiveInterval representing dead windows.
    """
    dead_windows: dict[int, list[LiveInterval]] = {}

    for vgpr, intervals in sorted(liveness.items()):
        windows = []
        # Before first interval
        if intervals and intervals[0].first_def > first_code_line:
            windows.append(LiveInterval(first_code_line, intervals[0].first_def - 1))
        # Between intervals
        for i in range(len(intervals) - 1):
            gap_start = intervals[i].last_use + 1
            gap_end = intervals[i + 1].first_def - 1
            if gap_start <= gap_end:
                windows.append(LiveInterval(gap_start, gap_end))
        # After last interval
        if intervals and intervals[-1].last_use < last_code_line:
            windows.append(LiveInterval(intervals[-1].last_use + 1, last_code_line))

        if windows:
            dead_windows[vgpr] = windows

    return dead_windows


# ---------------------------------------------------------------------------
# Remapping suggestions
# ---------------------------------------------------------------------------

@dataclass
class RemapSuggestion:
    """Suggestion to remap a high VGPR to a lower one during a dead window."""
    high_vgpr: int
    low_vgpr: int
    high_interval: LiveInterval  # when the high VGPR is live
    low_dead_window: LiveInterval  # when the low VGPR is dead
    overlap: LiveInterval  # the actual usable overlap window


def find_remapping_suggestions(
    liveness: dict[int, list[LiveInterval]],
    dead_windows: dict[int, list[LiveInterval]],
    target_vgprs: int,
    current_max_vgpr: int,
) -> list[RemapSuggestion]:
    """
    Find VGPRs above target_vgprs that can be remapped to lower VGPRs
    during their dead windows.

    A high VGPR H can be remapped to a low VGPR L if:
    - H >= target_vgprs (it's above the threshold)
    - L < target_vgprs (it's below the threshold)
    - L has a dead window that fully contains H's live interval
    """
    suggestions = []

    # High VGPRs: those at or above target that we want to eliminate
    high_vgprs = sorted([v for v in liveness if v >= target_vgprs], reverse=True)

    # Track which low VGPRs + dead windows have already been claimed
    claimed: set[tuple[int, int, int]] = set()  # (low_vgpr, window_start, window_end)

    for h in high_vgprs:
        h_intervals = liveness[h]
        best_suggestion = None

        for h_iv in h_intervals:
            # Find a low VGPR with a dead window that fully contains this interval
            for l_vgpr in sorted(dead_windows.keys()):
                if l_vgpr >= target_vgprs:
                    continue  # only remap to below-threshold VGPRs

                for dw in dead_windows[l_vgpr]:
                    claim_key = (l_vgpr, dw.first_def, dw.last_use)
                    if claim_key in claimed:
                        continue

                    # Check if dead window fully contains the high VGPR's live interval
                    if dw.first_def <= h_iv.first_def and dw.last_use >= h_iv.last_use:
                        overlap = LiveInterval(
                            max(dw.first_def, h_iv.first_def),
                            min(dw.last_use, h_iv.last_use)
                        )
                        suggestion = RemapSuggestion(
                            high_vgpr=h,
                            low_vgpr=l_vgpr,
                            high_interval=h_iv,
                            low_dead_window=dw,
                            overlap=overlap,
                        )
                        if best_suggestion is None or l_vgpr < best_suggestion.low_vgpr:
                            best_suggestion = suggestion
                        break  # Take first suitable dead window for this low VGPR

            if best_suggestion:
                break  # Found a mapping for this high VGPR

        if best_suggestion:
            claimed.add((
                best_suggestion.low_vgpr,
                best_suggestion.low_dead_window.first_def,
                best_suggestion.low_dead_window.last_use,
            ))
            suggestions.append(best_suggestion)

    return suggestions


# ---------------------------------------------------------------------------
# Compute peak liveness (VGPRs simultaneously live at each line)
# ---------------------------------------------------------------------------

def compute_peak_liveness(
    liveness: dict[int, list[LiveInterval]],
    first_code_line: int,
    last_code_line: int,
) -> tuple[int, int]:
    """
    Compute the peak number of simultaneously live VGPRs and the line where it occurs.
    Uses an event-based sweep (efficient for large line ranges).
    """
    events = []  # (line, +1 or -1)
    for vgpr, intervals in liveness.items():
        for iv in intervals:
            events.append((iv.first_def, 1))
            events.append((iv.last_use + 1, -1))

    events.sort()

    peak = 0
    peak_line = first_code_line
    current = 0
    for line, delta in events:
        current += delta
        if current > peak:
            peak = current
            peak_line = line

    return peak, peak_line


# ---------------------------------------------------------------------------
# Main analysis and reporting
# ---------------------------------------------------------------------------

def analyze_kernel(
    filepath: str,
    target_vgprs: Optional[int] = None,
    output_json: bool = False,
    verbose: bool = False,
):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse
    vgpr_accesses, declared_count, accum_offset, structure, instr_count = parse_assembly(
        lines, verbose=verbose
    )

    if not vgpr_accesses:
        print("ERROR: No VGPR accesses found. Is this a valid AMDGCN assembly file?",
              file=sys.stderr)
        return 1

    # Determine code range
    all_lines = []
    for accs in vgpr_accesses.values():
        for a in accs:
            all_lines.append(a.line)
    first_code_line = min(all_lines)
    last_code_line = max(all_lines)

    # Observed max VGPR
    observed_max = max(vgpr_accesses.keys()) + 1  # next_free = max + 1

    # Use declared count if available, otherwise observed
    current_vgpr_count = declared_count if declared_count else observed_max

    # Compute liveness
    liveness = compute_liveness(vgpr_accesses, structure, first_code_line, last_code_line)

    # Peak liveness
    peak_live, peak_line = compute_peak_liveness(liveness, first_code_line, last_code_line)

    # Determine target
    if target_vgprs is None:
        boundary = next_occupancy_boundary(current_vgpr_count)
        if boundary is None:
            target_vgprs = current_vgpr_count  # already at max occupancy
        else:
            target_vgprs = boundary

    # Dead windows
    dead_windows = find_dead_windows(liveness, first_code_line, last_code_line)

    # Remapping suggestions
    suggestions = find_remapping_suggestions(
        liveness, dead_windows, target_vgprs, current_vgpr_count
    )

    # Compute new VGPR count after remapping
    remapped_out = set(s.high_vgpr for s in suggestions)
    remaining_vgprs = set(vgpr_accesses.keys()) - remapped_out
    new_max = max(remaining_vgprs) + 1 if remaining_vgprs else 0

    # Current occupancy
    cur_waves = max_waves(current_vgpr_count)
    new_waves = max_waves(new_max)

    # Identify VGPRs above target
    above_target = sorted([v for v in liveness if v >= target_vgprs])
    # Those above target that we could NOT remap
    unremapped = sorted(set(above_target) - remapped_out)

    # VGPRs in target..max range with their intervals
    high_vgpr_details = {}
    for v in above_target:
        intervals = liveness[v]
        high_vgpr_details[v] = {
            'intervals': [(iv.first_def, iv.last_use) for iv in intervals],
            'remapped': v in remapped_out,
        }

    # Build report
    report = {
        'file': filepath,
        'instruction_count': instr_count,
        'register_summary': {
            'declared_vgpr_count': declared_count,
            'observed_max_vgpr': observed_max,
            'effective_vgpr_count': current_vgpr_count,
            'allocated_vgprs': vgpr_alloc(current_vgpr_count),
            'current_waves_per_simd': cur_waves,
            'accum_offset': accum_offset,
            'peak_simultaneously_live': peak_live,
            'peak_live_at_line': peak_line,
        },
        'occupancy_analysis': {
            'current_occupancy': f'{cur_waves} waves/SIMD',
            'target_vgpr_count': target_vgprs,
            'vgprs_to_eliminate': len(above_target),
            'vgprs_above_target': above_target,
        },
        'high_vgpr_details': high_vgpr_details,
        'remapping_suggestions': [
            {
                'high_vgpr': f'v{s.high_vgpr}',
                'low_vgpr': f'v{s.low_vgpr}',
                'high_live_window': f'L{s.high_interval.first_def}-L{s.high_interval.last_use}',
                'low_dead_window': f'L{s.low_dead_window.first_def}-L{s.low_dead_window.last_use}',
                'rationale': (
                    f'v{s.high_vgpr} is only live L{s.high_interval.first_def}-'
                    f'L{s.high_interval.last_use}, '
                    f'v{s.low_vgpr} is dead L{s.low_dead_window.first_def}-'
                    f'L{s.low_dead_window.last_use}'
                ),
            }
            for s in suggestions
        ],
        'unremappable_vgprs': [
            {
                'vgpr': f'v{v}',
                'intervals': [(iv.first_def, iv.last_use) for iv in liveness[v]],
                'reason': 'No dead window in lower VGPR fully contains this live interval',
            }
            for v in unremapped
        ],
        'occupancy_impact': {
            'original_vgpr_count': current_vgpr_count,
            'original_waves': cur_waves,
            'new_vgpr_count': new_max,
            'new_allocated': vgpr_alloc(new_max),
            'new_waves': new_waves,
            'occupancy_improved': new_waves > cur_waves,
            'remappings_applied': len(suggestions),
        },
    }

    if output_json:
        print(json.dumps(report, indent=2))
    else:
        print_text_report(report, liveness, dead_windows, above_target, target_vgprs)

    return 0


def print_text_report(
    report: dict,
    liveness: dict[int, list[LiveInterval]],
    dead_windows: dict[int, list[LiveInterval]],
    above_target: list[int],
    target_vgprs: int,
):
    """Print a human-readable text report."""
    rs = report['register_summary']
    oa = report['occupancy_analysis']
    oi = report['occupancy_impact']

    print("=" * 72)
    print("VGPR LIVENESS ANALYSIS")
    print("=" * 72)
    print(f"File:                    {report['file']}")
    print(f"Instructions parsed:     {report['instruction_count']}")
    print()

    print("--- Register Usage Summary ---")
    if rs['declared_vgpr_count'] is not None:
        print(f"Declared (next_free_vgpr):  {rs['declared_vgpr_count']}")
    print(f"Observed max VGPR:          v{rs['observed_max_vgpr'] - 1} "
          f"(next_free = {rs['observed_max_vgpr']})")
    print(f"Effective VGPR count:       {rs['effective_vgpr_count']}")
    print(f"Allocated VGPRs (x4):       {rs['allocated_vgprs']}")
    print(f"Current occupancy:          {rs['current_waves_per_simd']} waves/SIMD")
    if rs['accum_offset'] is not None:
        print(f"Accumulator offset:         {rs['accum_offset']}")
    print(f"Peak simultaneously live:   {rs['peak_simultaneously_live']} VGPRs "
          f"(at line {rs['peak_live_at_line']})")
    print()

    print("--- Occupancy Analysis ---")
    print(f"Target VGPR count:    {oa['target_vgpr_count']} "
          f"(would give {max_waves(oa['target_vgpr_count'])} waves/SIMD)")
    print(f"VGPRs above target:   {oa['vgprs_to_eliminate']} "
          f"({', '.join(f'v{v}' for v in oa['vgprs_above_target'])})")
    print()

    if above_target:
        print("--- High VGPR Details ---")
        for v in above_target:
            details = report['high_vgpr_details'][v]
            ivs = details['intervals']
            status = "REMAPPED" if details['remapped'] else "unremapped"
            iv_strs = [f'L{s}-L{e}' for s, e in ivs]
            print(f"  v{v:3d}  [{status:>10s}]  live: {', '.join(iv_strs)}")
        print()

    if report['remapping_suggestions']:
        print("--- Remapping Suggestions ---")
        for s in report['remapping_suggestions']:
            print(f"  {s['high_vgpr']:>5s} -> {s['low_vgpr']:<5s}  "
                  f"(high live {s['high_live_window']}, "
                  f"low dead {s['low_dead_window']})")
        print()

    if report['unremappable_vgprs']:
        print("--- Unremappable VGPRs ---")
        for u in report['unremappable_vgprs']:
            ivs = [f'L{s}-L{e}' for s, e in u['intervals']]
            print(f"  {u['vgpr']}: live {', '.join(ivs)}")
            print(f"         {u['reason']}")
        print()

    print("--- Occupancy Impact ---")
    print(f"Original:  {oi['original_vgpr_count']} VGPRs "
          f"(alloc {vgpr_alloc(oi['original_vgpr_count'])}) -> "
          f"{oi['original_waves']} waves/SIMD")
    print(f"After:     {oi['new_vgpr_count']} VGPRs "
          f"(alloc {oi['new_allocated']}) -> "
          f"{oi['new_waves']} waves/SIMD")
    if oi['occupancy_improved']:
        delta = oi['new_waves'] - oi['original_waves']
        pct = delta / oi['original_waves'] * 100
        print(f"IMPROVEMENT: +{delta} wave(s)/SIMD ({pct:.0f}% more occupancy)")
    else:
        deficit = oi['original_vgpr_count'] - oi['new_vgpr_count']
        needed = oi['original_vgpr_count'] - report['occupancy_analysis']['target_vgpr_count']
        print(f"Reduced by {deficit} VGPRs, but need {needed} total to reach next tier.")

    print(f"Remappings applied:  {oi['remappings_applied']}")
    print("=" * 72)

    # Advice
    if oi['occupancy_improved']:
        print()
        print("HOW TO APPLY:")
        print("  1. For each remapping above, replace the high VGPR with the low VGPR")
        print("     in all instructions within the high VGPR's live window.")
        print("  2. If the low VGPR held a loop-invariant value that's needed later,")
        print("     add a 'trampoline re-init' block to restore it after the remapped window.")
        print("  3. Update .amdhsa_next_free_vgpr and .amdhsa_accum_offset accordingly.")
        print("  4. Re-assemble and verify correctness on hardware.")


def main():
    parser = argparse.ArgumentParser(
        description='VGPR Liveness Analyzer for gfx950 (CDNA4) assembly kernels'
    )
    parser.add_argument('input', help='Path to .s assembly file')
    parser.add_argument('--target-vgprs', type=int, default=None,
                        help='Target VGPR count (default: next occupancy tier)')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')
    parser.add_argument('--verbose', action='store_true',
                        help='Show per-instruction def/use details')
    args = parser.parse_args()

    sys.exit(analyze_kernel(args.input, args.target_vgprs, args.json, args.verbose))


if __name__ == '__main__':
    main()
