# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Machine-local gate between a KernelForge artifact and the serving engine.

Run as a subprocess against a free GPU *before* any framework source is
patched::

    python -m hyperloom.forge_kernels.preflight \\
        --pack-dir <installed pack> --out <installed pack>/preflight.json

For every candidate shape in the manifest it builds the kernel, launches it,
scores the result against the framework reference, and micro-benchmarks it both
eagerly and under a CUDA/HIP graph. Only shapes that build, clear the SNR gate,
and are not slower than the reference land in ``verified`` — which is the
allowlist :mod:`hyperloom.forge_kernels._dispatch` consults at runtime.

Why this exists as its own step rather than trusting the manifest: a kernel that
is correct and fast in KernelForge's own harness can still be unusable in a
given serving image. Both failure modes are real and both are silent without a
gate:

- **FlyDSL API drift.** The generated kernel calls names the installed FlyDSL
  no longer exports. :mod:`._compat` repairs the ones we know about; anything
  left surfaces here as a build failure instead of a crashed server.
- **Shape coverage.** A builder picks an internal strategy from ``(M, N,
  dtype)``, and the strategy it picks for an out-of-contract shape may not be
  launchable at all (e.g. a fallback path that exceeds the AMDGPU 256-thread
  default workgroup cap because it never declares ``known_block_size``).

Running in a subprocess is deliberate: FlyDSL JIT failures can abort the
process, and the orchestrator must survive that.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

#: Default correctness floor. f32 row-wise softmax lands near 140 dB and bf16
#: near 95 dB, so this only ever rejects real breakage.
DEFAULT_MIN_SNR_DB = 30.0

#: A pack has to be at least this fast (graph-mode) relative to the framework
#: reference to be worth patching the server for.
DEFAULT_MIN_SPEEDUP = 1.0

_TORCH_DTYPES = {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}


def _reference(x: Any) -> Any:
    import torch

    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def _snr_db(reference: Any, got: Any) -> float:
    import math

    reference = reference.float()
    got = got.float()
    noise = (reference - got).pow(2).mean().item()
    signal = reference.pow(2).mean().item()
    if noise <= 0.0:
        return float("inf")
    if signal <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(signal / noise)


#: Launches captured into a single graph. Graph *replay* costs ~9 us on
#: MI300-class hardware, which is more than a small softmax takes, so a 1-node
#: graph measures the replay overhead and reports every fast kernel as exactly
#: 1.00x. Capturing a run of launches amortizes that away.
_GRAPH_INNER_ITERS = 20


def _time_graph_ms(make_call: Any, iters: int = 50) -> float:
    """Per-launch cost of ``make_call`` under CUDA/HIP graph replay.

    ``make_call`` is a factory rather than a plain callable so the FlyDSL
    stream handle is created *inside* the capture: a handle bound beforehand
    enqueues onto the pre-capture stream and records an empty graph, which
    silently times nothing at all.
    """
    import torch

    warm = make_call()
    for _ in range(10):
        warm()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call = make_call()
        for _ in range(_GRAPH_INNER_ITERS):
            call()
    torch.cuda.synchronize()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (iters * _GRAPH_INNER_ITERS)


def _time_eager_ms(call: Any, iters: int = 100) -> float:
    """Per-call cost with launches enqueued back to back (no per-call sync)."""
    import torch

    for _ in range(20):
        call()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run(pack_dir: Path, *, min_snr_db: float, min_speedup: float) -> dict[str, Any]:
    """Build, score and benchmark every candidate shape; return the report."""
    import torch

    from ._compat import install as install_compat
    from ._packs import Pack

    manifest = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "pack": manifest.get("name") or pack_dir.name,
        "op": manifest.get("op"),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_properties(0).gcnArchName if torch.cuda.is_available() else "",
        "torch": torch.__version__,
        "min_snr_db": min_snr_db,
        "min_speedup": min_speedup,
        "verified": [],
        "rejected": [],
        "ok": False,
    }

    if not torch.cuda.is_available():
        report["reason"] = "no GPU visible to the preflight process"
        return report

    try:
        report["compat_aliases"] = install_compat()
        import flydsl

        report["flydsl"] = getattr(flydsl, "__version__", "")
    except Exception as e:  # noqa: BLE001 - report, do not raise
        report["reason"] = f"flydsl unusable: {type(e).__name__}: {e}"
        return report

    pack = Pack(
        name=str(report["pack"]),
        root=pack_dir,
        op=str(manifest.get("op") or ""),
        builder=str(manifest.get("builder") or ""),
        manifest=manifest,
        preflight={},
        verified=frozenset(),
    )
    try:
        pack.load_module()
    except Exception as e:  # noqa: BLE001
        report["reason"] = f"kernel module did not import: {type(e).__name__}: {e}"
        return report

    probes = [
        _probe_one(pack, shape, min_snr_db=min_snr_db, min_speedup=min_speedup)
        for shape in manifest.get("probe_shapes") or ()
    ]

    # The runtime allowlist is keyed on (N, dtype) but M is free, so a single
    # failing M has to disqualify the whole family: otherwise a shape that only
    # works at the probed M would be dispatched at every other M in production.
    poisoned = {(e["N"], e["dtype"]) for e in probes if not e["_ok"]}
    for entry in probes:
        passed = entry.pop("_ok")
        if passed and (entry["N"], entry["dtype"]) in poisoned:
            entry["reason"] = "another M for this (N, dtype) failed; family disqualified"
            passed = False
        report["verified" if passed else "rejected"].append(entry)

    report["ok"] = bool(report["verified"])
    if not report["ok"]:
        report["reason"] = "every candidate shape was rejected"
    return report


def _probe_one(pack: Any, shape: dict[str, Any], *, min_snr_db: float, min_speedup: float) -> dict[str, Any]:
    import torch

    m, n, tag = int(shape["M"]), int(shape["N"]), str(shape.get("dtype", "f32"))
    entry: dict[str, Any] = {"M": m, "N": n, "dtype": tag, "_ok": False}

    torch_dtype = getattr(torch, _TORCH_DTYPES.get(tag, ""), None)
    if torch_dtype is None:
        entry["reason"] = f"unsupported dtype tag {tag!r}"
        return entry

    try:
        builder = getattr(pack.load_module(), pack.builder)
        launcher = builder(m, n, tag)
    except Exception as e:  # noqa: BLE001
        entry["reason"] = f"build failed: {type(e).__name__}: {e}"
        return entry

    try:
        import flydsl.expr as fx

        x = torch.randn(m, n, device="cuda", dtype=torch_dtype)
        out = torch.empty_like(x)

        def call() -> None:
            launcher(x, out, m, stream=fx.Stream(torch.cuda.current_stream().cuda_stream))

        call()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        entry["reason"] = f"launch failed: {type(e).__name__}: {e}"
        return entry

    entry["snr_db"] = round(_snr_db(_reference(x), out), 2)
    if entry["snr_db"] < min_snr_db:
        entry["reason"] = f"SNR {entry['snr_db']:.1f} dB below the {min_snr_db:.1f} dB gate"
        return entry

    try:
        entry["eager_us"] = round(_time_eager_ms(call) * 1000, 2)
        entry["ref_eager_us"] = round(_time_eager_ms(lambda: torch.softmax(x, dim=-1)) * 1000, 2)
        entry["graph_us"] = round(_time_graph_ms(lambda: call) * 1000, 2)
        entry["ref_graph_us"] = round(_time_graph_ms(lambda: lambda: torch.softmax(x, dim=-1)) * 1000, 2)
    except Exception as e:  # noqa: BLE001 - correctness already passed; note and keep
        entry["reason"] = f"benchmark failed: {type(e).__name__}: {e}"
        return entry

    entry["graph_speedup"] = round(entry["ref_graph_us"] / entry["graph_us"], 3) if entry["graph_us"] else 0.0
    entry["eager_speedup"] = round(entry["ref_eager_us"] / entry["eager_us"], 3) if entry["eager_us"] else 0.0
    if entry["graph_speedup"] < min_speedup:
        entry["reason"] = f"graph-mode speedup {entry['graph_speedup']:.2f}x below the {min_speedup:.2f}x gate"
        return entry

    entry["_ok"] = True
    return entry


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Always writes a report; exit code 0 iff any shape passed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min-snr-db", type=float, default=DEFAULT_MIN_SNR_DB)
    parser.add_argument("--min-speedup", type=float, default=DEFAULT_MIN_SPEEDUP)
    args = parser.parse_args(argv)

    pack_dir: Path = args.pack_dir
    out: Path = args.out or (pack_dir / "preflight.json")
    try:
        report = run(pack_dir, min_snr_db=args.min_snr_db, min_speedup=args.min_speedup)
    except Exception as e:  # noqa: BLE001 - a crashed probe is still a verdict
        report = {
            "schema_version": 1,
            "pack": pack_dir.name,
            "ok": False,
            "reason": f"preflight crashed: {type(e).__name__}: {e}",
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
