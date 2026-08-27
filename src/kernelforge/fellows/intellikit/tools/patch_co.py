#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Patch a .co code object with new machine code from a .s file.

Takes the original .co (with correct ELF metadata) and a modified .s file,
assembles the .s into machine code, and replaces the .text section in the .co.

Usage:
    python3 patch_co.py original.co modified.s output.co [--container-cmd CMD]

This avoids needing to emit correct AMDGPU metadata directives — we just
reuse the reference .co's ELF headers and note section.
"""

import struct
import subprocess
import sys
import os
import tempfile


def read_elf_sections(data):
    """Parse ELF64 section headers."""
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]
    e_shnum = struct.unpack_from("<H", data, 0x3C)[0]
    e_shstrndx = struct.unpack_from("<H", data, 0x3E)[0]

    sections = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh = struct.unpack_from("<IIQQQQIIQQ", data, off)
        sections.append(
            {
                "name_idx": sh[0],
                "type": sh[1],
                "flags": sh[2],
                "addr": sh[3],
                "offset": sh[4],
                "size": sh[5],
                "link": sh[6],
                "info": sh[7],
                "addralign": sh[8],
                "entsize": sh[9],
            }
        )

    # resolve names
    if e_shstrndx < len(sections):
        strtab = sections[e_shstrndx]
        strtab_data = data[strtab["offset"] : strtab["offset"] + strtab["size"]]
        for s in sections:
            end = strtab_data.find(b"\x00", s["name_idx"])
            s["name"] = strtab_data[s["name_idx"] : end].decode("ascii", errors="replace")

    return sections


def find_text_section(data):
    """Find .text section offset and size."""
    sections = read_elf_sections(data)
    for s in sections:
        if s.get("name") == ".text":
            return s["offset"], s["size"]
    raise ValueError("No .text section found")


def assemble_to_bin(s_path, llvm_mc_path="llvm-mc", mcpu="gfx950"):
    """Assemble .s to raw machine code bytes."""
    with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as tmp:
        obj_path = tmp.name

    try:
        result = subprocess.run(
            [llvm_mc_path, "-triple=amdgcn-amd-amdhsa", f"-mcpu={mcpu}", "-filetype=obj", "-o", obj_path, s_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Assembly failed:\n{result.stderr}")

        with open(obj_path, "rb") as f:
            obj_data = f.read()

        text_off, text_size = find_text_section(obj_data)
        return obj_data[text_off : text_off + text_size]
    finally:
        os.unlink(obj_path)


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} original.co modified.s output.co")
        sys.exit(1)

    orig_co = sys.argv[1]
    mod_s = sys.argv[2]
    out_co = sys.argv[3]

    llvm_mc = "llvm-mc"
    mcpu = "gfx950"
    for arg_i, arg in enumerate(sys.argv):
        if arg == "--llvm-mc" and arg_i + 1 < len(sys.argv):
            llvm_mc = sys.argv[arg_i + 1]
        if arg == "--mcpu" and arg_i + 1 < len(sys.argv):
            mcpu = sys.argv[arg_i + 1]

    with open(orig_co, "rb") as f:
        co_data = bytearray(f.read())

    text_off, text_size = find_text_section(bytes(co_data))
    print(f"Original .text: offset=0x{text_off:x}, size={text_size} bytes")

    new_code = assemble_to_bin(mod_s, llvm_mc, mcpu)
    print(f"New code: {len(new_code)} bytes")

    if len(new_code) != text_size:
        print(f"WARNING: size mismatch! Original={text_size}, new={len(new_code)}")
        if len(new_code) > text_size:
            print("ERROR: new code is larger than original — cannot patch in place")
            sys.exit(1)
        # pad with NOPs (0xBF800000) to match size
        nop = b"\x00\x00\x80\xbf"
        while len(new_code) < text_size:
            new_code += nop
        print(f"Padded to {len(new_code)} bytes with NOPs")

    co_data[text_off : text_off + text_size] = new_code

    with open(out_co, "wb") as f:
        f.write(co_data)

    print(f"Wrote {out_co} ({len(co_data)} bytes)")


if __name__ == "__main__":
    main()
