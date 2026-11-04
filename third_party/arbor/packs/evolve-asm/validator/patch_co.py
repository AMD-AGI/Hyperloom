#!/usr/bin/env python3
"""ASM round-trip tool: assemble a modified .s and splice it into a reference .co.

Reuses the reference .co ELF metadata so the agent never has to emit correct
AMDGPU metadata directives — only the .text section changes.

Usage:
    python3 patch_co.py ref.co modified.s out.co [--mcpu gfx950] [--llvm-mc PATH]
"""
import os, struct, subprocess, sys, tempfile


def _read_sections(data: bytes) -> list[dict]:
    e_shoff     = struct.unpack_from('<Q', data, 0x28)[0]
    e_shentsize = struct.unpack_from('<H', data, 0x3A)[0]
    e_shnum     = struct.unpack_from('<H', data, 0x3C)[0]
    e_shstrndx  = struct.unpack_from('<H', data, 0x3E)[0]
    sections = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh = struct.unpack_from('<IIQQQQIIQQ', data, off)
        sections.append({'name_idx': sh[0], 'type': sh[1], 'flags': sh[2],
                         'addr': sh[3], 'offset': sh[4], 'size': sh[5]})
    if e_shstrndx < len(sections):
        strtab = sections[e_shstrndx]
        strtab_data = data[strtab['offset']:strtab['offset'] + strtab['size']]
        for s in sections:
            end = strtab_data.find(b'\x00', s['name_idx'])
            s['name'] = strtab_data[s['name_idx']:end].decode('ascii', errors='replace')
    return sections


def _find_text(data: bytes) -> tuple[int, int]:
    for s in _read_sections(data):
        if s.get('name') == '.text':
            return s['offset'], s['size']
    raise ValueError("No .text section in .co")


def _assemble(s_path: str, llvm_mc: str, mcpu: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as tmp:
        obj = tmp.name
    try:
        r = subprocess.run([llvm_mc, '-triple=amdgcn-amd-amdhsa', f'-mcpu={mcpu}',
                            '-filetype=obj', '-o', obj, s_path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Assembly failed:\n{r.stderr.strip()}")
        with open(obj, 'rb') as f:
            obj_data = f.read()
        off, sz = _find_text(obj_data)
        return obj_data[off:off + sz]
    finally:
        try: os.unlink(obj)
        except OSError: pass


def patch_co(ref_co: str, modified_s: str, out_co: str,
             mcpu: str = 'gfx950', llvm_mc: str = 'llvm-mc') -> int:
    with open(ref_co, 'rb') as f:
        co = bytearray(f.read())
    text_off, text_sz = _find_text(bytes(co))
    print(f"ref .text: offset=0x{text_off:x} size={text_sz}")

    new_code = bytearray(_assemble(modified_s, llvm_mc, mcpu))
    print(f"new code: {len(new_code)} bytes")

    if len(new_code) > text_sz:
        print(f"ERROR: new code ({len(new_code)}B) > original ({text_sz}B)", file=sys.stderr)
        return 1
    if len(new_code) < text_sz:
        nop = b'\x00\x00\x80\xBF'  # s_nop
        new_code += nop * ((text_sz - len(new_code)) // 4)
        print(f"padded to {len(new_code)} bytes with s_nop")

    co[text_off:text_off + text_sz] = new_code
    with open(out_co, 'wb') as f:
        f.write(co)
    print(f"wrote {out_co} ({len(co)} bytes)")
    return 0


def main() -> None:
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} ref.co modified.s out.co [--mcpu gfx950] [--llvm-mc PATH]")
        sys.exit(1)
    ref_co, modified_s, out_co = sys.argv[1], sys.argv[2], sys.argv[3]
    mcpu = 'gfx950'; llvm_mc = 'llvm-mc'
    i = 4
    while i < len(sys.argv):
        if sys.argv[i] == '--mcpu' and i+1 < len(sys.argv): mcpu = sys.argv[i+1]; i += 2
        elif sys.argv[i] == '--llvm-mc' and i+1 < len(sys.argv): llvm_mc = sys.argv[i+1]; i += 2
        else: i += 1
    sys.exit(patch_co(ref_co, modified_s, out_co, mcpu=mcpu, llvm_mc=llvm_mc))

if __name__ == '__main__':
    main()
