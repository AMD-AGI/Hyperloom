#!/usr/bin/env python3
"""
Patch SGLang to enable FP8-quantized custom allreduce on ROCm/Aiter.

The Aiter CustomAllreduce already supports `open_fp8_quant=True` which quantizes
allreduce messages to FP8 before communication, halving the data volume.
But SGLang's dispatch code never passes this flag (defaults to False).

This patch modifies the _all_reduce_out_place method to pass open_fp8_quant=True
when using the custom allreduce ("ca") path. This is controlled by the env var
SGLANG_CA_FP8_QUANT=1 (default: 0, safe to enable/disable).

Usage:
    python3 patch_fp8_allreduce.py apply    # Apply the patch
    python3 patch_fp8_allreduce.py revert   # Revert to original
"""

import sys
import shutil
import os

TARGET = "/sgl-workspace/sglang/python/sglang/srt/distributed/parallel_state.py"
BACKUP = TARGET + ".fp8ar_bak"

ORIGINAL = '            out = ca_comm.custom_all_reduce(input_)'
PATCHED = '            _fp8_ar = os.environ.get("SGLANG_CA_FP8_QUANT", "0") == "1"\n            out = ca_comm.custom_all_reduce(input_, open_fp8_quant=_fp8_ar)'

IMPORT_CHECK = "import os  # fp8_ar_patch"
IMPORT_ANCHOR = "from typing import TYPE_CHECKING"


def apply_patch():
    content = open(TARGET).read()
    if PATCHED in content:
        print("Patch already applied.")
        return
    if ORIGINAL not in content:
        print(f"ERROR: Could not find target string in {TARGET}")
        sys.exit(1)

    shutil.copy2(TARGET, BACKUP)
    print(f"Backup saved to {BACKUP}")

    content = content.replace(ORIGINAL, PATCHED)
    if IMPORT_CHECK not in content:
        content = content.replace(IMPORT_ANCHOR, f"{IMPORT_ANCHOR}\n{IMPORT_CHECK}", 1)

    open(TARGET, 'w').write(content)
    print(f"Patch applied. Set SGLANG_CA_FP8_QUANT=1 to enable FP8 allreduce quantization.")


def revert_patch():
    if not os.path.exists(BACKUP):
        print("No backup found. Nothing to revert.")
        return
    shutil.copy2(BACKUP, TARGET)
    os.remove(BACKUP)
    print(f"Reverted to original.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("apply", "revert"):
        print("Usage: python3 patch_fp8_allreduce.py [apply|revert]")
        sys.exit(1)
    if sys.argv[1] == "apply":
        apply_patch()
    else:
        revert_patch()
