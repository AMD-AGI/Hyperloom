#!/usr/bin/env python3
"""kbench: a reference op against a candidate at the served shapes (kernel-plan stages S1-S3).

Correctness first: per-dtype tolerance (bf16/fp16/fp32), SNR in dB (fp8) or exact bits
(integers, packed fp4, e8m0 scales), plus negative controls that must fail the same check.
Then GPU time per call inside a captured HIP graph, with a capture guard and an A B B A
block order for drift, and eager time for comparison. Writes a JSON report.

The module contract and a worked example are in SKILL.md, section 5.
"""
import argparse
import importlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time

EXAMPLE = """example:
  python3 kbench.py bench_norm.py --ref ref --cand cand \\
      --case decode-8=8x7168:bf16,7168:bf16 --case decode-64=64x7168:bf16,7168:bf16 \\
      --out /tmp/kbench-norm.json
exit status: 0 correct (and timed); 1 a check failed, a negative control did not fire,
or graph capture was invalid; 2 usage error."""

# Per-dtype defaults: (rtol, atol) for allclose; dB for SNR. Override with --rtol/--atol/--snr-db.
TOL = {"float32": (1e-4, 1e-5), "float16": (1e-3, 1e-3), "bfloat16": (1e-2, 1e-2)}  # bf16, fp16: AITER's tests
SNR_DB = {"float8_e4m3fn": 25.0, "float8_e4m3fnuz": 25.0, "float8_e5m2": 20.0, "float8_e5m2fnuz": 20.0}
DTYPES = {"bf16": "bfloat16", "fp16": "float16", "f16": "float16", "fp32": "float32", "f32": "float32",
          "fp8": "float8_e4m3fn", "e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2",
          "e4m3fnuz": "float8_e4m3fnuz", "e5m2fnuz": "float8_e5m2fnuz",
          "fp4x2": "float4_e2m1fn_x2", "e8m0": "float8_e8m0fnu",
          "i8": "int8", "u8": "uint8", "i16": "int16", "i32": "int32", "i64": "int64", "bool": "bool"}
torch = None  # imported in main(), so --help works without a GPU or torch


def parse_args(argv):
    p = argparse.ArgumentParser(prog="kbench.py", description=__doc__.split("\n\n")[0], epilog=EXAMPLE,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("module", help="Python file or importable module that defines the callables")
    p.add_argument("--ref", default="ref", help="reference: NAME in MODULE, or FILE.py:NAME / pkg.mod:NAME")
    p.add_argument("--cand", default="cand", help="candidate, same forms (default: cand)")
    p.add_argument("--inputs", default="make_inputs",
                   help="optional builder in MODULE: make_inputs(case, gen, device) -> list of tensors")
    p.add_argument("--spec", help='cases as JSON or @file.json: {"cases": [{"name": "d8", "inputs": '
                                  '[{"shape": [8, 7168], "dtype": "bf16"}], "kwargs": {}}]}')
    p.add_argument("--case", action="append", default=[], help="shorthand, repeatable: NAME=8x7168:bf16,7168:bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric", choices=["auto", "allclose", "snr", "exact"],
                   help="default auto: allclose for bf16/fp16/fp32, snr for fp8, exact bits otherwise")
    p.add_argument("--rtol", type=float)
    p.add_argument("--atol", type=float)
    p.add_argument("--snr-db", type=float, help="SNR threshold; required with --metric snr on non-fp8 outputs")
    p.add_argument("--max-mismatch", type=float, default=0.0, help="fraction of elements allowed out of tolerance")
    p.add_argument("--warmup", type=int, default=10, help="eager calls before capture (JIT compiles here)")
    p.add_argument("--replays", type=int, default=50, help="timed graph replays per block (N)")
    p.add_argument("--calls-per-graph", type=int, default=10, help="calls captured in one graph (K)")
    p.add_argument("--rotate", type=int, default=1, help="input copies cycled in the graph, to read past caches")
    p.add_argument("--ref-only", action="store_true", help="S1: time the reference alone")
    p.add_argument("--time-anyway", action="store_true", help="time even when correctness fails")
    p.add_argument("--no-time", action="store_true", help="correctness only")
    p.add_argument("--device", default="cuda", help="cuda (ROCm too) or cpu (correctness only)")
    p.add_argument("--out", default="kbench.json")
    return p.parse_args(argv)


def die(msg):
    print(f"kbench: {msg}", file=sys.stderr)
    sys.exit(2)


def load_module(spec):
    if spec.endswith(".py") or "/" in spec:
        path = os.path.abspath(spec)
        sys.path.insert(0, os.path.dirname(path))  # let it import its neighbours
        s = importlib.util.spec_from_file_location("kbench_" + os.path.basename(path)[:-3], path)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
        return mod
    return importlib.import_module(spec)


def resolve(mod, name):
    if ":" in name:
        where, name = name.rsplit(":", 1)
        mod = load_module(where)
    return getattr(mod, name)


def load_cases(args):
    spec = {}
    if args.spec:
        text = open(args.spec[1:]).read() if args.spec.startswith("@") else args.spec
        spec = json.loads(text)
    cases = list(spec.get("cases", []))
    for s in args.case:
        name, _, body = s.partition("=")
        ins = []
        for item in filter(None, body.split(",")):
            shape, _, dt = item.partition(":")
            ins.append({"shape": [int(d) for d in shape.split("x") if d], "dtype": dt or "bf16"})
        cases.append({"name": name, "inputs": ins})
    if not cases:
        die("no cases; give --spec or --case")
    return cases, spec


def tdtype(name):
    dt = getattr(torch, DTYPES.get(name, name), None)
    if dt is None and name in ("fp4x2", "e8m0"):
        dt = torch.uint8  # older torch: packed bytes
    if dt is None:
        die(f"unknown dtype {name!r}")
    return dt


def make_tensor(d, gen, dev):
    """init: randn (default, times "scale"), rand, zeros, ones or arange. Integers, packed fp4 and
    e8m0 scales get random values in [low, high); e8m0 defaults to exponents near 2^0."""
    shape, name, init = d["shape"], d.get("dtype", "bf16"), d.get("init", "randn")
    dt, packed = tdtype(name), name in ("e8m0", "fp4x2")
    if init == "arange":
        x = torch.arange(math.prod(shape)).reshape(shape)
    elif packed or not dt.is_floating_point:
        lo, hi = {"e8m0": (124, 131), "fp4x2": (0, 256), "uint8": (0, 256), "int8": (-128, 128),
                  "bool": (0, 2)}.get(name if packed else str(dt)[6:], (0, 100))
        x = torch.randint(int(d.get("low", lo)), int(d.get("high", hi)), shape, generator=gen)
    elif init in ("zeros", "ones"):
        x = getattr(torch, init)(shape)
    else:
        x = (torch.rand if init == "rand" else torch.randn)(shape, generator=gen) * float(d.get("scale", 1.0))
        if "float8" in str(dt):
            x = x.clamp(-torch.finfo(dt).max, torch.finfo(dt).max)
    return (x.to(torch.uint8).view(dt) if packed else x.to(dt)).to(dev)


def as_list(out):
    if out is None or isinstance(out, torch.Tensor):
        return [] if out is None else [out]
    return [t for t in out if isinstance(t, torch.Tensor)]


def ints(t):
    """Same-size integer view, for bit comparisons and sentinels."""
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def metric_for(t, args, case, spec):
    """CLI flag, then the case, then the spec's top level, then the dtype's default."""
    name = str(t.dtype).replace("torch.", "")

    def pick(k, default):
        return next((v for v in (getattr(args, k), case.get(k), spec.get(k)) if v is not None), default)
    m = pick("metric", "auto")
    if m == "auto":
        m = "snr" if name in SNR_DB else "allclose" if name in TOL else "exact"
    rtol, atol = TOL.get(name, (0.0, 0.0))
    snr = pick("snr_db", SNR_DB.get(name))
    if m == "snr" and snr is None:
        die(f"--metric snr on a {name} output needs --snr-db (the plan's §4 precision)")
    return {"metric": m, "rtol": pick("rtol", rtol), "atol": pick("atol", atol), "snr_db": snr,
            "max_mismatch": args.max_mismatch}


def compare(r, c, m):
    res = {"dtype": str(r.dtype).replace("torch.", ""), "shape": list(r.shape), "metric": m["metric"]}
    if r.shape != c.shape or r.dtype != c.dtype:
        return dict(res, passed=False, why=f"candidate is {list(c.shape)} {c.dtype}")
    n = max(1, r.numel())
    res["bitwise_equal"] = bool(torch.equal(ints(r), ints(c)))
    if m["metric"] == "exact":
        bad = int((ints(r) != ints(c)).sum())
        return dict(res, mismatches=bad, passed=bad <= m["max_mismatch"] * n)
    rf, cf = r.float(), c.float()
    nonfin = ~torch.isfinite(rf) | ~torch.isfinite(cf)
    nf_bad = int((nonfin & ~(rf.isnan() & cf.isnan()) & (rf != cf)).sum())  # inf, nan must match
    ok = ~nonfin
    err = torch.where(ok, cf, 0) - torch.where(ok, rf, 0)
    sig, noise = float(torch.where(ok, rf, 0).pow(2).sum()), float(err.pow(2).sum())
    snr = math.inf if noise == 0 else (10 * math.log10(sig / noise) if sig > 0 else -math.inf)
    res.update(snr_db=round(snr, 2) if math.isfinite(snr) else str(snr),
               max_abs_err=float(err.abs().max()) if err.numel() else 0.0, nonfinite_mismatches=nf_bad)
    if m["metric"] == "snr":
        return dict(res, threshold_db=m["snr_db"], passed=nf_bad == 0 and snr >= m["snr_db"])
    bad = int((err.abs() > m["atol"] + m["rtol"] * rf.abs()).sum())
    return dict(res, rtol=m["rtol"], atol=m["atol"], mismatches=bad,
                passed=nf_bad == 0 and bad <= m["max_mismatch"] * n)


def negative_controls(r, c, m):
    """Perturbed copies of the candidate's output. Each must FAIL compare(), or the check is vacuous."""
    shifted = ints(c).reshape(-1).roll(1).reshape(c.shape).view(c.dtype)  # same values, wrong places
    if m["metric"] == "exact":
        noisy = ints(c).clone().reshape(-1)
        noisy[: max(1, noisy.numel() // 100)] ^= 1
        noisy = noisy.reshape(c.shape).view(c.dtype)
    elif m["metric"] == "snr":  # noise 6 dB under the threshold
        rf = r.float()
        sigma = math.sqrt(float(rf.pow(2).mean()) / 10 ** ((m["snr_db"] - 6) / 10))
        g = torch.Generator().manual_seed(0)
        noisy = (rf + sigma * torch.randn(rf.shape, generator=g).to(rf.device)).to(c.dtype)
    else:  # ten times the tolerance, every element (with a floor, for zero tolerances)
        rf = r.float()
        noisy = (c.float() + 10 * (m["atol"] + m["rtol"] * rf.abs()) + 1e-3 * (rf.abs() + 1)).to(c.dtype)
    out = {"shift": not compare(r, shifted, m)["passed"], "noise": not compare(r, noisy, m)["passed"]}
    if not out["shift"]:
        out["why"] = "a shifted output passes: output near-constant, inputs degenerate, or tolerance too wide"
    return out


def stats(us):
    s = sorted(us)
    return {"n": len(s), "min": round(s[0], 3), "median": round(statistics.median(s), 3),
            "p90": round(s[int(0.9 * (len(s) - 1))], 3), "worst": round(s[-1], 3)}


def capture(fn, sets, kwargs, calls):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(calls):
            outs = as_list(fn(*sets[i % len(sets)], **kwargs))
    return g, outs


def guard(fn, args, pristine, kwargs, truth, metrics):
    """A kernel launched off the capture stream leaves an empty graph that replays in no time.
    Fill the graph's outputs with a sentinel, replay once, and require the right answer."""
    g, outs = capture(fn, [args], kwargs, 1)
    for o in outs:
        v = ints(o)
        v.fill_(255 if v.dtype == torch.uint8 else -1)
    for a, p in zip(args, pristine):  # after the sentinel: an in-place kernel's outputs are its inputs
        a.copy_(p)
    g.replay()
    torch.cuda.synchronize()
    if len(outs) != len(truth):
        return False
    return all(compare(t, o, m)["passed"] for t, o, m in zip(truth, outs, metrics))


def replay_block(g, n, calls):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(n + 1)]
    ev[0].record()
    for i in range(n):  # back to back, as in serving; one sync at the end
        g.replay()
        ev[i + 1].record()
    torch.cuda.synchronize()
    return [ev[i].elapsed_time(ev[i + 1]) * 1000.0 / calls for i in range(n)]


def eager_us(fn, args, kwargs, n):
    out = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn(*args, **kwargs)
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1000.0)
    return out


def time_case(fns, inputs, kwargs, truth, metrics, args):
    """fns: {"ref": f, "cand": f}. Blocks run ref, cand, cand, ref so drift shows on both sides."""
    calls = max(args.calls_per_graph, args.rotate)
    res, graphs = {"calls_per_graph": calls, "rotate": args.rotate}, {}
    for side, fn in fns.items():
        sets = [[a.clone() for a in inputs] for _ in range(args.rotate)]
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):  # JIT, autotune and workspace allocation happen here
            for i in range(max(1, args.warmup)):
                fn(*sets[i % len(sets)], **kwargs)
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()
        try:
            if not guard(fn, [a.clone() for a in inputs], inputs, kwargs, truth[side], metrics):
                raise RuntimeError("replay did not recompute the output: kernel not captured "
                                   "(launched on another stream?)")
            graphs[side] = capture(fn, sets, kwargs, calls)[0]
        except Exception as e:  # report and fail; never time an invalid graph
            res[side] = {"graph_valid": False, "why": f"{type(e).__name__}: {e}"[:400]}
        res.setdefault(side, {"graph_valid": True, "blocks": []})
        raw = eager_us(fn, sets[0], kwargs, args.replays)
        res[side].update(eager_us=stats(raw), eager_samples_us=[round(t, 3) for t in raw])
    order = [s for s in ("ref", "cand", "cand", "ref") if s in graphs] if len(graphs) == 2 else list(graphs) * 2
    for side in order:
        res[side]["blocks"].append(replay_block(graphs[side], args.replays, calls))
    for side in graphs:
        b1, b2 = res[side].pop("blocks")
        res[side].update(graph_us=stats(b1 + b2), graph_samples_us=[round(t, 3) for t in b1 + b2])
        res[side]["drift_pct"] = round(100 * (statistics.median(b2) / statistics.median(b1) - 1), 2)
    if "ref" in graphs and "cand" in graphs:
        res["speedup_median"] = round(res["ref"]["graph_us"]["median"] / res["cand"]["graph_us"]["median"], 3)
    res["valid"] = all(res[s]["graph_valid"] for s in fns)
    return res


def run_case(case, spec, args, ref, cand, builder):
    dev = torch.device(args.device)
    gen = torch.Generator().manual_seed(args.seed)
    kwargs = case.get("kwargs", {})
    if builder is None and "inputs" not in case:
        die(f"case {case.get('name')!r} has no inputs and the module has no make_inputs")
    inputs = builder(case, gen, dev) if builder else [make_tensor(d, gen, dev) for d in case["inputs"]]
    rec = {"name": case.get("name"), "inputs": [[list(t.shape), str(t.dtype)[6:]] for t in inputs]}
    footprint = sum(t.numel() * t.element_size() for t in inputs) / 2 ** 20
    rec["input_mib"] = round(footprint, 2)
    truth = {"ref": [o.clone() for o in as_list(ref(*[a.clone() for a in inputs], **kwargs))]}
    if not truth["ref"]:
        die("the reference returned no tensor; an in-place kernel returns the tensor it wrote")
    metrics = [metric_for(t, args, case, spec) for t in truth["ref"]]
    ok = True
    if cand is not None:
        got = [o.clone() for o in as_list(cand(*[a.clone() for a in inputs], **kwargs))]
        if len(got) != len(truth["ref"]):
            outs = [{"passed": False, "why": f"{len(got)} outputs, reference has {len(truth['ref'])}"}]
        else:
            outs = [dict(compare(r, c, m), negative_control=negative_controls(r, c, m))
                    for r, c, m in zip(truth["ref"], got, metrics)]
        ok = all(o["passed"] and o.get("negative_control", {}).get("shift") and o["negative_control"]["noise"]
                 for o in outs)
        rec["correct"] = {"passed": bool(ok), "outputs": outs}
        truth["cand"] = truth["ref"]
    if args.no_time or dev.type != "cuda":
        rec["timing"] = "skipped" + ("" if args.no_time else " (no GPU device)")
    elif ok or args.time_anyway:
        fns = {"ref": ref} if cand is None else {"ref": ref, "cand": cand}
        rec["timing"] = time_case(fns, inputs, kwargs, truth, metrics, args)
        if footprint * args.rotate < 256:
            rec["timing"]["note"] = ("inputs fit in the 256 MiB Infinity Cache: replays read warm; "
                                     "use --rotate if the served path reads them cold")
        ok = ok and rec["timing"]["valid"]
    else:
        rec["timing"] = "skipped: correctness failed (gate correctness before timing)"
    return rec, ok


def summary(rec):
    line = f"case {rec['name']}:"
    if "correct" in rec:
        line += " correct " + ("PASS" if rec["correct"]["passed"] else "FAIL")
        for i, o in enumerate(rec["correct"]["outputs"]):
            nc = o.get("negative_control", {})
            snr = f" snr={o['snr_db']}dB" if "snr_db" in o else ""
            line += (f"\n  out{i} {o.get('dtype')} {o.get('metric')} pass={o['passed']} "
                     f"mismatches={o.get('mismatches', '-')}{snr} "
                     f"neg(shift={nc.get('shift')}, noise={nc.get('noise')}) {o.get('why', nc.get('why', ''))}")
    t = rec["timing"]
    if isinstance(t, str):
        return line + "\n  timing " + t
    for side in ("ref", "cand"):
        if side in t:
            s = t[side]
            g = s.get("graph_us")
            line += (f"\n  {side}: graph median {g['median']} us (min {g['min']}, worst {g['worst']}, "
                     f"drift {s['drift_pct']}%)" if g else f"\n  {side}: graph INVALID {s.get('why')}")
            line += f" | eager median {s['eager_us']['median']} us"
    if "speedup_median" in t:
        line += f"\n  ref/cand (graph median) {t['speedup_median']}x"
    return line + (f"\n  note: {t['note']}" if "note" in t else "")


def main(argv=None):
    global torch
    args = parse_args(sys.argv[1:] if argv is None else argv)
    import torch as _torch
    torch = _torch
    cases, spec = load_cases(args)
    mod = load_module(args.module)
    ref = resolve(mod, args.ref)
    cand = None if args.ref_only else resolve(mod, args.cand)
    builder = getattr(mod, args.inputs, None)
    report = {"tool": "kbench", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv,
              "seed": args.seed, "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
              "env": {k: v for k, v in os.environ.items()
                      if k.startswith(("AITER_", "FLYDSL_", "TRITON_", "HIP_", "ROCR_", "VLLM_ROCM_"))}}
    if args.device.startswith("cuda") and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.device(args.device))
        report["device"] = {"name": p.name, "arch": getattr(p, "gcnArchName", None), "cus": p.multi_processor_count}
    report["cases"], all_ok = [], True
    for case in cases:
        rec, ok = run_case(case, spec, args, ref, cand, builder)
        report["cases"].append(rec)
        all_ok = all_ok and ok
        print(summary(rec), flush=True)
    report["passed"] = all_ok
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"{'PASS' if all_ok else 'FAIL'}; report: {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
