###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Opt-in LIVE integration test for the v2 source-resolver stack.

Runs the three real modules against the *actually installed* framework tree
inside a real vLLM ROCm container (e.g. ``vllm/vllm-openai-rocm:v0.26.0``) --
no fakes, no GPU needed (this only scans source):

1. ``source_env``            -> discover_frameworks() dictionaries + metadata,
                                and prove the discovered csrc holds real kernels.
2. ``kernel_source_index``   -> build + cache the kernel index; the cache is
                                written to a repo-local dir you can inspect.
3. ``source_resolver``    -> resolve many real kernels to their file/line.
4. full cache                -> generate the complete kernel cache (JSON, plus a
                                human-readable dump) into the repo-local dir.

The persistent cache lands under ``results/resolver_benchmarks_vllm/ksi_cache/``
(inside the mounted repo, so it shows up on the host for manual verification).

Gated by ``HYPERLOOM_LIVE_STACK=1`` so it is skipped in ordinary (no-container)
runs. Run it directly inside a serving-framework ROCm container::

    HYPERLOOM_LIVE_STACK=1 HYPERLOOM_DISCOVER_ONLY=aiter,vllm \
    HYPERLOOM_EXPECT_FRAMEWORKS=aiter,vllm \
    PYTHONPATH=src python3 src/hyperloom/agents/kernel/tests/test_live_source_stack.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from hyperloom.agents.kernel.tools import (
    kernel_source_index,
    source_env,
    source_resolver,
)

# A kernel known to ship in aiter (macro/launch-bounds template) -- the headline
# case the attribute-aware scanner recovers. Sampling falls back to other indexed
# kernels if a given image renamed it, so the test stays robust across versions.
_PREFERRED_KERNEL = "paged_attention_ll4mi_reduce_kernel"

# How many kernels each section must exercise (coverage knob).
_SAMPLE_N = 10

# Repo root (…/Hyperloom) resolved from this file's location, and a persistent,
# human-inspectable cache directory inside the mounted repo.
REPO_ROOT = Path(__file__).resolve().parents[5]
LOCAL_CACHE_DIR = REPO_ROOT / "results" / "resolver_benchmarks_vllm" / "ksi_cache"


def _require_live() -> None:
    if os.environ.get("HYPERLOOM_LIVE_STACK", "").strip() != "1":
        raise SystemExit(
            "SKIP: set HYPERLOOM_LIVE_STACK=1 and run inside a serving-framework "
            "ROCm container (see this module's docstring for the exact command)."
        )


def _expected_frameworks() -> set[str]:
    """Frameworks that MUST be discovered, from ``$HYPERLOOM_EXPECT_FRAMEWORKS``.

    Defaults to ``aiter,vllm`` (vLLM images). For SGLang images set
    ``HYPERLOOM_EXPECT_FRAMEWORKS=aiter,sglang``.
    """
    raw = os.environ.get("HYPERLOOM_EXPECT_FRAMEWORKS", "aiter,vllm")
    return {n.strip().lower() for n in raw.split(",") if n.strip()}


def _use_local_cache() -> None:
    """Point the index cache at the repo-local dir so artifacts persist on host."""
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HYPERLOOM_KSI_CACHE_DIR"] = str(LOCAL_CACHE_DIR)


# --- 1) source_env: dictionaries + metadata + real kernels ------------------
def _kernels_in_csrc(fw: dict[str, source_env.FrameworkRoot], limit: int):
    """Yield up to ``limit`` distinct ``(name, file, line)`` from discovered csrc."""
    seen: set[str] = set()
    for name in sorted(fw):
        for path in kernel_source_index._native_files(fw[name].csrc_roots):
            for kname, line in kernel_source_index._scan_file(path):
                if kname in seen:
                    continue
                seen.add(kname)
                yield kname, path, line
                if len(seen) >= limit:
                    return


def check_source_env() -> dict[str, source_env.FrameworkRoot]:
    """discover_frameworks() yields metadata AND the discovered csrc holds kernels."""
    fw = source_env.discover_frameworks()

    print("  discover_frameworks() ->")
    for name, fr in sorted(fw.items()):
        print(f"    {name}: version={fr.version!r} root={fr.root}")
        for cr in fr.csrc_roots:
            print(f"        csrc_root: {cr}  (exists={Path(cr).is_dir()})")

    assert fw, "no frameworks discovered (is this running inside a serving image?)"
    assert isinstance(fw, dict)

    # Each expected framework (aiter,vllm for vLLM; aiter,sglang for SGLang) must
    # be discovered, be a real dir, and report a plausible version. Any reported
    # csrc root must actually exist on disk.
    expected = _expected_frameworks()
    for name in sorted(expected):
        assert name in fw, f"{name} not discovered: {sorted(fw)}"
        fr = fw[name]
        assert isinstance(fr, source_env.FrameworkRoot)
        assert fr.root.is_dir(), fr.root
        assert re.match(r"^\d+\.\d+", fr.version), f"odd {name} version: {fr.version!r}"
        assert all(Path(cr).is_dir() for cr in fr.csrc_roots), fr.csrc_roots

    # At least one discovered framework must ship native csrc (aiter in both the
    # vLLM and SGLang images; SGLang also ships sgl-kernel/csrc).
    assert any(fr.csrc_roots for fr in fw.values()), "no native csrc discovered"

    # Metadata helpers: fingerprint is a stable 16-hex key; tag names every
    # expected framework.
    fp1 = source_env.fingerprint(fw)
    fp2 = source_env.fingerprint(fw)
    assert fp1 == fp2 and re.fullmatch(r"[0-9a-f]{16}", fp1), fp1
    tag = source_env.version_tag(fw)
    for name in expected:
        assert name in tag, tag
    print(f"  fingerprint={fp1}  version_tag={tag}")

    # Prove discovery pointed at real source: the csrc trees must hold >= 10
    # genuine kernel definitions (file exists, positive line), which we print.
    kernels = list(_kernels_in_csrc(fw, _SAMPLE_N))
    assert len(kernels) >= _SAMPLE_N, f"only {len(kernels)} kernels found in csrc"
    print(f"  {len(kernels)} kernel definitions found in discovered csrc:")
    for kname, path, line in kernels:
        assert path.is_file() and line > 0, (kname, path, line)
        print(f"      {kname}  @ {path.name}:{line}")
    return fw


# --- 2) kernel_source_index: build + persistent local cache -----------------
def check_index(fw: dict[str, source_env.FrameworkRoot]) -> kernel_source_index.SourceIndex:
    """build_index() populates the index; the cache is written to a local dir."""
    idx = kernel_source_index.build_index(fw)
    print(f"  build_index -> {idx.symbol_count} symbols / {idx.file_count} files (build_ms={idx.build_ms})")
    assert idx.symbol_count > 0 and idx.file_count > 0
    assert idx.fingerprint == source_env.fingerprint(fw)

    # At least 10 index records must point at real files with positive lines.
    checked = 0
    for name, recs in idx.symbol_index.items():
        rec = recs[0]
        assert Path(str(rec["file"])).is_file(), (name, rec)
        assert int(rec["line"]) > 0, (name, rec)
        checked += 1
        if checked >= _SAMPLE_N:
            break
    assert checked >= _SAMPLE_N, f"only {checked} indexed records verifiable"
    print(f"  verified {checked} index records point at real file:line")

    # Cache is written to the repo-local dir at a fingerprint-keyed path we can
    # inspect. Force a clean miss, confirm the file appears, then confirm a hit.
    _use_local_cache()
    cache_path = kernel_source_index._cache_path(idx.fingerprint)
    if cache_path.exists():
        cache_path.unlink()

    built = kernel_source_index.load_or_build(fw)  # miss -> builds + saves
    assert built.build_ms >= 0.0
    assert cache_path.exists(), f"cache not written to {cache_path}"
    assert cache_path.parent == LOCAL_CACHE_DIR, cache_path
    size = cache_path.stat().st_size
    print(f"  cache written: {cache_path} ({size} bytes)")

    reloaded = kernel_source_index.load_or_build(fw)  # hit -> build_ms == 0
    assert reloaded.build_ms == 0.0, "expected a cache hit"
    assert reloaded.symbol_count == built.symbol_count
    print("  cache round-trip OK (miss builds to local dir, second load is a hit)")
    return built


# --- 3) source_resolver: resolve many real kernels -----------------------
def _sample_indexed_kernels(idx: kernel_source_index.SourceIndex, n: int) -> list[str]:
    """Pick ``n`` kernel names whose first record is on disk (preferred first)."""
    picks: list[str] = []
    if idx.lookup(_PREFERRED_KERNEL):
        picks.append(_PREFERRED_KERNEL)
    for name, recs in idx.symbol_index.items():
        if len(picks) >= n:
            break
        if name in picks:
            continue
        if recs and Path(str(recs[0]["file"])).is_file():
            picks.append(name)
    assert len(picks) >= n, f"only {len(picks)} on-disk kernels to sample"
    return picks[:n]


def _serving_framework() -> str:
    """Serving-framework hint for resolve() (expected set minus aiter)."""
    serving = sorted(_expected_frameworks() - {"aiter"})
    return serving[0] if serving else "vllm"


def check_resolver(idx: kernel_source_index.SourceIndex) -> None:
    """resolve() finds each sampled kernel's file/line and honestly misses junk."""
    names = _sample_indexed_kernels(idx, _SAMPLE_N)
    framework = _serving_framework()
    print(f"  resolving {len(names)} kernels via source_resolver (framework={framework}):")
    for sym in names:
        res = source_resolver.resolve("live::op_not_in_hints", framework=framework, device_kernel_name=sym, index=idx)
        assert res.method == "symbol_index", (sym, res)
        assert res.symbol == sym and res.patchable is True, (sym, res)
        assert res.source_file and Path(res.source_file).is_file(), (sym, res)
        assert res.line and res.line > 0, (sym, res)

        # Cross-check with the indexer: the file really defines that kernel (ties
        # all three modules together -- discovered, indexed, and re-parsed).
        text = Path(res.source_file).read_text(encoding="utf-8", errors="ignore")
        defined = {n for n, _pos in kernel_source_index._iter_global_defs(text)}
        assert sym in defined, sym
        print(f"      {sym}  ->  {Path(res.source_file).name}:{res.line}")

    # Honest miss: a symbol that does not exist resolves to unresolved (no guess).
    miss = source_resolver.resolve(
        "live::missing",
        framework=framework,
        device_kernel_name="definitely_not_a_kernel_zzz",
        index=idx,
    )
    assert miss.method == "unresolved" and miss.source_file == "", miss

    # Legacy-tuple shape stays a drop-in for the bypass resolver.
    last = source_resolver.resolve("live::op", framework=framework, device_kernel_name=names[0], index=idx)
    src, method = last.as_legacy_tuple()
    assert src == last.source_file and method == "symbol_index"
    print("  resolver hits + honest-miss + legacy-tuple OK")


# --- 4) full cache: generate the complete kernel cache locally --------------
def check_full_cache(fw: dict[str, source_env.FrameworkRoot]) -> None:
    """Generate the full kernel cache (as the module does) into the local dir."""
    _use_local_cache()
    idx = kernel_source_index.load_or_build(fw)
    cache_path = kernel_source_index._cache_path(idx.fingerprint)
    assert cache_path.exists(), cache_path

    # Machine cache is a compact one-liner; also emit a sorted, indented dump
    # (kernel -> ["file:line", ...]) for easy manual verification on the host.
    readable = LOCAL_CACHE_DIR / f"ksi_{idx.fingerprint}_readable.json"
    mapping = {name: [f"{r['file']}:{r['line']}" for r in recs] for name, recs in sorted(idx.symbol_index.items())}
    payload = {
        "fingerprint": idx.fingerprint,
        "version_tag": idx.version_tag,
        "symbol_count": idx.symbol_count,
        "file_count": idx.file_count,
        "kernels": mapping,
    }
    readable.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    assert len(mapping) == idx.symbol_count, (len(mapping), idx.symbol_count)
    parsed = json.loads(readable.read_text(encoding="utf-8"))
    assert parsed["symbol_count"] == idx.symbol_count

    print(f"  full cache ({idx.symbol_count} kernels / {idx.file_count} files):")
    print(f"      machine cache : {cache_path} ({cache_path.stat().st_size} bytes)")
    print(f"      readable dump : {readable} ({readable.stat().st_size} bytes)")


# --- 5) real Triton/TileLang .py kernels: discovery + editable gate ---------
def check_py_kernels(fw: dict[str, source_env.FrameworkRoot]) -> None:
    """Find real ``@triton.jit`` kernels in the installed tree; the editable gate
    must accept them (their source comes from the trace ``kernel_file``)."""
    found: list[Path] = []
    for name in sorted(fw):  # aiter sorts first: its ops/ hold triton kernels
        for dirpath, _dirs, names in os.walk(fw[name].root):
            for nm in names:
                if not nm.endswith(".py"):
                    continue
                p = Path(dirpath) / nm
                try:
                    if "@triton.jit" in p.read_text(encoding="utf-8", errors="ignore"):
                        found.append(p)
                except OSError:
                    continue
                if len(found) >= _SAMPLE_N:
                    break
            if len(found) >= _SAMPLE_N:
                break
        if len(found) >= _SAMPLE_N:
            break

    assert found, "no real @triton.jit .py kernels found in the installed tree"
    print(f"  {len(found)} real @triton.jit kernel files found:")
    # Every real, repo-resident triton .py must be classified editable. (The
    # editable Triton source is provided upstream by the trace ``kernel_file``;
    # the finder itself resolves only native device kernels by symbol.)
    for p in found:
        assert source_resolver.is_editable_source(str(p)), p
        print(f"      {p.name}: editable (repo-resident @triton.jit)")


@pytest.mark.skipif(
    os.environ.get("HYPERLOOM_LIVE_STACK", "").strip() != "1",
    reason="LIVE stack test: set HYPERLOOM_LIVE_STACK=1 inside a serving-framework ROCm container.",
)
def test_live_source_stack() -> None:
    """Opt-in LIVE end-to-end check of the resolver stack (collected + skippable).

    Skipped by default so ordinary (no-container) CI collects it as one skipped
    case rather than the file contributing zero tests. Inside a real framework
    container it exercises discovery -> index -> resolve -> cache -> Triton gate.
    """
    fw = check_source_env()
    idx = check_index(fw)
    check_resolver(idx)
    check_full_cache(fw)
    check_py_kernels(fw)


def _run_all() -> int:
    _require_live()

    print("[source_env (dictionaries + metadata + kernels)]")
    fw = check_source_env()
    print("PASS  source_env\n")

    print("[kernel_source_index (build + local cache)]")
    idx = check_index(fw)
    print("PASS  kernel_source_index\n")

    print("[source_resolver (resolve real kernels)]")
    check_resolver(idx)
    print("PASS  source_resolver\n")

    print("[full cache (generate local artifacts)]")
    check_full_cache(fw)
    print("PASS  full_cache\n")

    print("[real Triton .py kernels (discovery + editable gate)]")
    check_py_kernels(fw)
    print("PASS  py_kernels\n")

    print(f"5/5 sections passed  (cache dir: {LOCAL_CACHE_DIR})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
