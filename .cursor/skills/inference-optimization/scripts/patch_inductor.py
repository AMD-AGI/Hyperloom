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


def extract_function_signature(source: str, func_name: str) -> list[str] | None:
    """Extract parameter names from 'def func_name(param1, param2, ...)'."""
    pattern = rf'def\s+{re.escape(func_name)}\s*\(([^)]*)\)'
    match = re.search(pattern, source)
    if match:
        params = [p.strip().split(':')[0].split('=')[0].strip()
                  for p in match.group(1).split(',')]
        return [p for p in params if p]
    return None


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


def _find_best_config(target_file: str) -> str | None:
    """Find the .best_config file in the same directory as target_file."""
    target_dir = os.path.dirname(os.path.abspath(target_file))
    configs = glob.glob(os.path.join(target_dir, "*.best_config"))
    return configs[0] if configs else None


def _update_best_config(target_file: str, config_updates: dict, dry_run: bool = False) -> bool:
    """Update .best_config file in the same directory as target_file.

    config_updates: dict of keys to update, e.g. {"XBLOCK": 4, "R0_BLOCK": 2048, "num_warps": 4}
    Only specified keys are updated; other keys are preserved.
    """
    import json

    config_path = _find_best_config(target_file)
    if not config_path:
        print(f"WARNING: No .best_config found in {os.path.dirname(target_file)}")
        return False

    with open(config_path) as f:
        cfg = json.load(f)

    before = {k: cfg.get(k) for k in config_updates}

    if dry_run:
        print(f"DRY-RUN would update {config_path}: {before} -> {config_updates}")
        return True

    shutil.copy2(config_path, config_path + ".bak")
    cfg.update(config_updates)
    with open(config_path, "w") as f:
        json.dump(cfg, f)

    print(f"UPDATED .best_config: {config_path}")
    for k, v in config_updates.items():
        print(f"  {k}: {before[k]} -> {v}")
    return True


def patch_single_file(
    kernel_name: str,
    geak_source: str,
    target_file: str,
    best_config: dict | None = None,
    dry_run: bool = False,
) -> bool:
    """Patch exactly one standalone file. Returns True on success.

    If best_config is provided, also updates the .best_config file in the same
    directory with the given tiling parameters (e.g. XBLOCK, R0_BLOCK, num_warps).
    """
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

    geak_sig = extract_function_signature(geak_source, kernel_name)
    target_sig = extract_function_signature(content, kernel_name)
    if geak_sig and target_sig and geak_sig != target_sig:
        print(f"ERROR: Signature mismatch — GEAK has {len(geak_sig)} params "
              f"({', '.join(geak_sig[:4])}...) but target has {len(target_sig)} "
              f"({', '.join(target_sig[:4])}...). "
              f"Same kernel name, different shape variant. Skipping.")
        return False

    new_content = replace_function_body(content, kernel_name, new_body)
    if new_content == content:
        print(f"WARNING: replacement produced identical content — nothing changed")
        return False

    if dry_run:
        print(f"DRY-RUN would patch: {target_file}")
        if best_config:
            _update_best_config(target_file, best_config, dry_run=True)
        return True

    shutil.copy2(target_file, target_file + ".bak")
    open(target_file, "w").write(new_content)
    print(f"PATCHED: {target_file}")

    if best_config:
        _update_best_config(target_file, best_config)

    cache_dir = _find_cache_root(target_file)
    if cache_dir:
        _clear_binary_caches(cache_dir)
    return True


def revert_file(target_file: str) -> bool:
    """Revert a single file and its .best_config from .bak backups."""
    reverted_any = False

    bak = target_file + ".bak"
    if os.path.isfile(bak):
        shutil.copy2(bak, target_file)
        os.remove(bak)
        print(f"Reverted: {target_file}")
        reverted_any = True
    else:
        print(f"No backup found: {bak}")

    config_path = _find_best_config(target_file)
    if config_path:
        config_bak = config_path + ".bak"
        if os.path.isfile(config_bak):
            shutil.copy2(config_bak, config_path)
            os.remove(config_bak)
            print(f"Reverted: {config_path}")
            reverted_any = True

    return reverted_any


def main():
    parser = argparse.ArgumentParser(description="Patch Inductor kernels with GEAK output")
    sub = parser.add_subparsers(dest="command")

    p_patch = sub.add_parser("patch", help="Apply GEAK patch")
    p_patch.add_argument("--kernel-name", required=True)
    p_patch.add_argument("--geak-file", required=True, help="GEAK optimized kernel file")
    p_patch.add_argument("--target-file", required=True, help="Exact standalone file to patch")
    p_patch.add_argument("--best-config", default=None,
                         help='JSON dict of .best_config overrides, e.g. \'{"XBLOCK":4,"num_warps":4}\'')
    p_patch.add_argument("--dry-run", action="store_true")

    p_revert = sub.add_parser("revert", help="Revert patches (kernel .py + .best_config)")
    p_revert.add_argument("--target-file", required=True, help="Revert a single file")

    args = parser.parse_args()
    if args.command == "patch":
        import json as _json
        geak_source = open(args.geak_file).read()
        bc = _json.loads(args.best_config) if args.best_config else None
        ok = patch_single_file(args.kernel_name, geak_source, args.target_file, bc, args.dry_run)
        sys.exit(0 if ok else 1)
    elif args.command == "revert":
        ok = revert_file(args.target_file)
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
