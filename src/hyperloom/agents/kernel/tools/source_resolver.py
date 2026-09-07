###############################################################################
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Version-robust op -> editable-source resolver (the \"active finder\")."""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import shutil
import subprocess  # nosec B404 - invokes c++filt with a fixed, non-shell argv.
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:  # package import (TraceLens route / tests)
    from . import kernel_source_index, source_env
    from ._bypass_source_resolver import is_editable_source
except ImportError:  # flat top-level import (bypass route puts tools/ on sys.path)
    import kernel_source_index  # type: ignore[no-redef]
    import source_env  # type: ignore[no-redef]
    from _bypass_source_resolver import is_editable_source  # type: ignore[no-redef]

__all__ = [
    "ResolveResult",
    "resolve",
    "resolve_source",
    "base_symbol",
    "latency_report",
    "reset_latency",
]

# CK (Composable Kernel) template instantiations have no single editable ``__global__`` source, so they are gated
# non-patchable from the symbol alone (no JSON).
_CK_DEMANGLED_RE = re.compile(r"(?:^|[^A-Za-z0-9_])ck(?:_tile)?::")
# Mangled (Itanium) fallback for when ``c++filt`` is absent: the ``ck`` / ``ck_tile`` namespace is length-prefixed
# (e.g. ``...2ck15kernel...`` / ``...7ck_tileI...``), so classification does not depend on binutils.
_CK_MANGLED_RE = re.compile(r"\d(?:ck_tile|ck)(?=[0-9IE])")

# A plain C/C++ identifier (used by the fallback demangler).
_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


@dataclass
class ResolveResult:
    """Outcome of one resolve, including the measured latency."""

    source_file: str
    line: int | None
    symbol: str
    patchable: bool
    method: str
    elapsed_ms: float
    reason: str = ""

    def as_legacy_tuple(self) -> tuple[str, str]:
        """Legacy ``(source_file, method)`` shape for drop-in compatibility."""
        if self.source_file or self.method == "non_patchable":
            return (self.source_file, self.method)
        return ("", "unresolved")


# Latency instrumentation
@dataclass
class _LatencyBucket:
    version_tag: str
    samples: list[float] = field(default_factory=list)
    index_build_ms: float = 0.0


_LATENCY: dict[str, _LatencyBucket] = {}
# Cap retained samples per version so a long-lived session cannot grow the list unboundedly; percentiles over the most
# recent window are what the report needs.
_LATENCY_MAX_SAMPLES = 50_000


def _record_latency(version_tag: str, elapsed_ms: float) -> None:
    bucket = _LATENCY.setdefault(version_tag, _LatencyBucket(version_tag=version_tag))
    if len(bucket.samples) >= _LATENCY_MAX_SAMPLES:
        del bucket.samples[0]
    bucket.samples.append(elapsed_ms)


def reset_latency() -> None:
    """Clear all recorded latency samples (useful for a fresh benchmark)."""
    _LATENCY.clear()


def latency_report() -> dict[str, Any]:
    """Summarize resolve latency per detected framework version."""
    out: dict[str, Any] = {}
    for tag, bucket in _LATENCY.items():
        s = sorted(bucket.samples)
        n = len(s)
        if n == 0:
            out[tag] = {"count": 0, "index_build_ms": round(bucket.index_build_ms, 2)}
            continue

        def _pct(p: float) -> float:
            idx = min(n - 1, int(round(p * (n - 1))))
            return round(s[idx], 3)

        out[tag] = {
            "count": n,
            "avg_ms": round(sum(s) / n, 3),
            "p50_ms": _pct(0.50),
            "p95_ms": _pct(0.95),
            "max_ms": round(s[-1], 3),
            "index_build_ms": round(bucket.index_build_ms, 2),
        }
    return out


# Symbol normalization
@functools.lru_cache(maxsize=8192)
def _cxxfilt_base(mangled: str) -> str:
    """Demangle via ``c++filt`` when available (``\"\"`` on failure)."""
    if not shutil.which("c++filt"):
        return ""
    try:
        proc = subprocess.run(  # nosec B603 B607 - fixed argv, no shell.
            ["c++filt", mangled],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("c++filt demangle failed for %r: %s", mangled, exc)
        return ""


def _base_from_demangled(name: str) -> str:
    """Extract the base kernel identifier from a demangled/plain symbol."""
    # Keep only the head before params/templates, drop namespaces, then take the last token (drops any leading return
    # type/qualifiers: "void ns::foo" -> "foo").
    head = re.split(r"[(<]", name.strip(), maxsplit=1)[0].split("::")[-1]
    tokens = head.split()
    return tokens[-1] if tokens else ""


def _base_from_mangled(mangled: str) -> str:
    """Fallback: parse Itanium length-prefixed identifiers from a mangled name."""
    names: list[str] = []
    i, n = 0, len(mangled)
    while i < n:
        if mangled[i].isdigit():
            j = i
            while j < n and mangled[j].isdigit():
                j += 1
            length = int(mangled[i:j])
            ident = mangled[j : j + length]
            if _IDENT_RE.match(ident):
                names.append(ident)
            i = j + length
        else:
            i += 1
    if not names:
        return ""
    # Prefer an identifier that looks like a kernel; else the last one.
    for nm in reversed(names):
        if "kernel" in nm.lower():
            return nm
    return names[-1]


@functools.lru_cache(maxsize=8192)
def base_symbol(device_kernel_name: str) -> str:
    """Reduce any device kernel symbol to its stable base name."""
    raw = (device_kernel_name or "").strip()
    if not raw:
        return ""
    if raw.startswith("_Z"):
        demangled = _cxxfilt_base(raw)
        if demangled and demangled != raw:
            return _base_from_demangled(demangled)
        return _base_from_mangled(raw)
    return _base_from_demangled(raw)


# Non-patchable detection (symbol-derived, no external metadata)
def _non_patchable_kind(device_kernel_name: str) -> str:
    """Return a non-patchable kind label from the symbol alone (``\"\"`` if none)."""
    raw = (device_kernel_name or "").strip()
    if not raw:
        return ""
    if raw.startswith("_Z"):
        demangled = _cxxfilt_base(raw)
        if demangled:
            return "aiter_ck" if _CK_DEMANGLED_RE.search(demangled.lower()) else ""
        # c++filt absent: classify from the mangled namespace prefix instead.
        return "aiter_ck" if _CK_MANGLED_RE.search(raw) else ""
    return "aiter_ck" if _CK_DEMANGLED_RE.search(raw.lower()) else ""


# Resolution
def _rank_records(records: list[dict[str, object]], framework: str) -> list[dict[str, object]]:
    """Rank candidate definition records: framework hint > arch tag > path len."""
    arch = os.environ.get("HYPERLOOM_TARGET_ARCH", "").strip().lower()
    fw = (framework or "").lower()

    def score(rec: dict[str, object]) -> tuple[int, int, int]:
        path = str(rec.get("file", "")).lower()
        fw_match = 1 if fw and rec.get("framework") == fw else 0
        arch_match = 1 if arch and arch in path else 0
        # Prefer shorter paths (canonical location over vendored copies).
        return (fw_match, arch_match, -len(path))

    return sorted(records, key=score, reverse=True)


def _verify_symbol(path: str, base: str) -> bool:
    """Confirm ``base`` actually appears in ``path`` (guards stale index)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        log.debug("active-finder: cannot read %r for symbol verify: %s", path, exc)
        return False
    return base in text


def resolve(
    op_name: str,
    *,
    framework: str = "",
    device_kernel_name: str = "",
    index: kernel_source_index.SourceIndex | None = None,
) -> ResolveResult:
    """Resolve a kernel to its editable source in the installed tree (timed)."""
    started = time.perf_counter()
    idx = index if index is not None else kernel_source_index.load_or_build()

    def finish(res: ResolveResult) -> ResolveResult:
        res.elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        _record_latency(idx.version_tag, res.elapsed_ms)
        return res

    # Cheap gate first (symbol-derived): CK template instantiations have no single editable source, so bail with a
    # clear reason.
    nonp_kind = _non_patchable_kind(device_kernel_name)
    if nonp_kind:
        return finish(
            ResolveResult(
                source_file="",
                line=None,
                symbol="",
                patchable=False,
                method="non_patchable",
                elapsed_ms=0.0,
                reason=nonp_kind,
            )
        )

    # Symbol-first (authoritative): the trace's device kernel name.
    base = base_symbol(device_kernel_name) if device_kernel_name else ""
    if base:
        records = _rank_records(idx.lookup(base), framework)
        # A bare base name that maps to >1 definition is disambiguated only by the ranking heuristic
        # (framework/arch/path), so leave a trace for anyone auditing a suspicious rewrite.
        if len(records) > 1:
            log.debug(
                "active-finder: %d candidate records for base %r (framework=%r); ranking picked by heuristic",
                len(records),
                base,
                framework,
            )
        for rec in records:
            path = str(rec.get("file", ""))
            if not is_editable_source(path):
                continue
            if not _verify_symbol(path, base):
                continue
            line_val = rec.get("line")
            line_no = line_val if isinstance(line_val, int) and line_val > 0 else None
            return finish(
                ResolveResult(
                    source_file=path,
                    line=line_no,
                    symbol=base,
                    patchable=True,
                    method="symbol_index",
                    elapsed_ms=0.0,
                )
            )
        log.debug("active-finder: no editable/verified source for base %r", base)

    return finish(
        ResolveResult(
            source_file="",
            line=None,
            symbol="",
            patchable=False,
            method="unresolved",
            elapsed_ms=0.0,
            reason="no live match",
        )
    )


def resolve_source(
    op_name: str,
    *,
    framework: str = "",
    device_kernel_name: str = "",
) -> tuple[str, str]:
    """Resolve to the legacy ``(source_file, method)`` shape."""
    return resolve(op_name, framework=framework, device_kernel_name=device_kernel_name).as_legacy_tuple()


# Latency benchmark CLI
def _sample_candidates(index: kernel_source_index.SourceIndex, top_k: int) -> list[dict[str, str]]:
    """Build sample candidates from the live index's base kernel symbols."""
    out: list[dict[str, str]] = []
    for sym in index.symbol_index:
        out.append({"op_name": "", "device_kernel_name": sym})
        if top_k and len(out) >= top_k:
            break
    return out


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - standalone CLI driver
    """CLI: ``--bench`` times the finder over sample kernel candidates."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the v2 source finder latency.")
    parser.add_argument("--bench", action="store_true", help="Run the latency benchmark.")
    parser.add_argument("--top-k", type=int, default=15, help="Number of candidates to resolve.")
    parser.add_argument("--framework", default="vllm", help="Framework hint (vllm/sglang).")
    parser.add_argument("--candidates", default="", help="Optional JSON file: [{op_name, device_kernel_name}].")
    args = parser.parse_args(argv)

    reset_latency()
    fw = source_env.discover_frameworks()
    if not fw:
        print("No frameworks (vllm/sglang/aiter) discovered; cannot benchmark.")
        return 1

    t0 = time.perf_counter()
    index = kernel_source_index.build_index(fw)
    index_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    _LATENCY.setdefault(index.version_tag, _LatencyBucket(version_tag=index.version_tag)).index_build_ms = index_ms

    if args.candidates:
        try:
            cands = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"Failed to read candidates: {exc}")
            return 1
    else:
        cands = _sample_candidates(index, args.top_k)

    resolved = 0
    for c in cands:
        res = resolve(
            c.get("op_name", ""),
            framework=args.framework,
            device_kernel_name=c.get("device_kernel_name", ""),
            index=index,
        )
        if res.source_file:
            resolved += 1

    report = latency_report()
    print(f"Version: {index.version_tag}")
    print(f"Index build: {index_ms} ms ({index.symbol_count} symbols / {index.file_count} files)")
    print(f"Candidates: {len(cands)} | resolved: {resolved}")
    for tag, stats in report.items():
        print(f"Latency[{tag}]: {json.dumps(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
