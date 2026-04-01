#!/usr/bin/env python3
"""Patch Inductor standalone kernel files with GEAK-optimized function body.

CRITICAL: Only replaces the kernel function body (@triton.jit def ... onwards).
Preserves the original @triton_heuristics decorator and inductor_meta — these
contain launcher configuration (num_load, grid_type, etc.) that Triton's
CachingAutotuner depends on. Replacing them causes:
  TypeError: launcher() got multiple values for argument 'stream'

Usage:
    python patch_inductor.py patch \\
        --kernel-name triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0 \\
        --geak-file /path/to/geak_optimized_kernel.py \\
        --target-file /tmp/torchinductor_root/.../triton_red_fused_....py

    python patch_inductor.py revert --target-file <standalone_file.py>
"""

import argparse
import glob
import os
import re
import shutil
import sys


def is_standalone_kernel(content: str) -> bool:
    return ("@triton_heuristics" in content and
            "async_compile" not in content and
            "def call(" not in content)


def extract_function_body(source: str, func_name: str) -> str | None:
    """Extract everything from '@triton.jit' + 'def func_name(...)' to end of function."""
    pattern = rf'(@triton\.jit\s*\ndef\s+{re.escape(func_name)}\s*\([^)]*\).*)'
    match = re.search(pattern, source, re.DOTALL)
    if match:
        return match.group(1)
    return None


def replace_function_body(original: str, func_name: str, new_body: str) -> str:
    """Replace function body in original file, keeping decorator/meta intact."""
    pattern = rf'(@triton\.jit\s*\ndef\s+{re.escape(func_name)}\s*\([^)]*\).*)'
    match = re.search(pattern, original, re.DOTALL)
    if not match:
        return original
    return original[:match.start()] + new_body


def _clear_binary_caches(cache_dir: str) -> None:
    """Remove compiled .so/.json and triton caches so Triton recompiles the patched kernel."""
    cleared = 0
    for pat in [f"{cache_dir}/**/*.so", f"{cache_dir}/**/*.json"]:
        for f in glob.glob(pat, recursive=True):
            os.remove(f)
            cleared += 1
    for triton_cache in [
        os.path.expanduser("~/.triton/cache"),
        os.path.expanduser("~/.cache/triton"),
        os.environ.get("TRITON_CACHE_DIR", ""),
    ]:
        if triton_cache and os.path.exists(triton_cache):
            shutil.rmtree(triton_cache)
            print(f"  Cleared triton cache: {triton_cache}")
    print(f"\nCleared {cleared} binary cache files + triton caches")


def _find_cache_root(file_path: str) -> str | None:
    """Walk up from file_path to find the torchinductor cache root."""
    p = os.path.dirname(os.path.abspath(file_path))
    while p != "/":
        if "torchinductor" in os.path.basename(p):
            return p
        p = os.path.dirname(p)
    return None


def patch_single_file(
    kernel_name: str,
    geak_source: str,
    target_file: str,
    dry_run: bool = False,
) -> bool:
    """Patch exactly one standalone file. Returns True on success."""
    new_body = extract_function_body(geak_source, kernel_name)
    if not new_body:
        print(f"ERROR: Could not find function '{kernel_name}' in GEAK output")
        return False

    if not os.path.isfile(target_file):
        print(f"ERROR: Target file does not exist: {target_file}")
        return False

    content = open(target_file).read()
    if not is_standalone_kernel(content):
        print(f"ERROR: {target_file} is not a standalone kernel file "
              "(missing @triton_heuristics or contains async_compile/def call)")
        return False

    if kernel_name not in content:
        print(f"ERROR: kernel '{kernel_name}' not found in {target_file}")
        return False

    new_content = replace_function_body(content, kernel_name, new_body)
    if new_content == content:
        print(f"WARNING: replacement produced identical content — nothing changed")
        return False

    if dry_run:
        print(f"DRY-RUN would patch: {target_file}")
        return True

    shutil.copy2(target_file, target_file + ".bak")
    open(target_file, "w").write(new_content)
    print(f"PATCHED: {target_file}")

    cache_dir = _find_cache_root(target_file)
    if cache_dir:
        _clear_binary_caches(cache_dir)
    return True


def revert_file(target_file: str) -> bool:
    """Revert a single file from its .bak backup."""
    bak = target_file + ".bak"
    if not os.path.isfile(bak):
        print(f"No backup found: {bak}")
        return False
    shutil.copy2(bak, target_file)
    os.remove(bak)
    print(f"Reverted: {target_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Patch Inductor kernels with GEAK output")
    sub = parser.add_subparsers(dest="command")

    p_patch = sub.add_parser("patch", help="Apply GEAK patch")
    p_patch.add_argument("--kernel-name", required=True)
    p_patch.add_argument("--geak-file", required=True, help="GEAK optimized kernel file")
    p_patch.add_argument("--target-file", required=True, help="Exact standalone file to patch")
    p_patch.add_argument("--dry-run", action="store_true")

    p_revert = sub.add_parser("revert", help="Revert patches")
    p_revert.add_argument("--target-file", required=True, help="Revert a single file")

    args = parser.parse_args()
    if args.command == "patch":
        geak_source = open(args.geak_file).read()
        ok = patch_single_file(args.kernel_name, geak_source, args.target_file, args.dry_run)
        sys.exit(0 if ok else 1)
    elif args.command == "revert":
        ok = revert_file(args.target_file)
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
