"""`python -m ruppell ...`

  run     --out DIR [--build DIR] [--witness] -- COMMAND...
          Run COMMAND with the runtime loaded, writing records to DIR.
          --witness runs it without the runtime, in witness mode, for an
          independent collector wrapped around it (see `witness`).
  export  RUN_DIR [-o trace.json] [--no-flows] [--steps A:B]
          Write a Chrome trace (a window of steps, for serving-size runs) and
          print the validation block, per process, with the cross-rank merge
          and the schema check. --format torch [--process N]: one process in
          the PyTorch profiler's vocabulary, ruppell scopes as the ops that
          launched each kernel, for tools such as AMD TraceLens.
  census  RUN_DIR [--phase NAME=START:END ...]
          Per step and process: GPU time by op and scope, busy time, period,
          fields, client phase (census.jsonl, census_scopes.npz).
  witness RUN_DIR WITNESS_DIR [--tolerance 0.03] [--spread WITNESS_DIR2]
          Compare RUN_DIR with rocprofv3's kernel trace in WITNESS_DIR, or on
          CUDA with an nsys export of a separate run (WITNESS_DIR2: a second
          one, whose difference from the first bounds the time check).
          --counters [--also DIR ...] [--same-process]: hardware counters
          (gray-parrot) against rocprofv3 --pmc's counter_collection.csv, by
          step and position in separate runs, or by dispatch id in one process
          (ruppell/gray_parrot/witness.py).
  report counters RUN_DIR [--json FILE] [--top N]
          Hardware counters (gray-parrot) per kernel name and grid: each
          counter's mean, minimum, maximum and number of samples.
  tracelens RUN_DIR [-o DIR] [--steps A:B] [--op-names path|op]
          Every rank in the PyTorch profiler's vocabulary, one file each
          (rank<N>.json), for TraceLens (see ruppell/export/torch_profiler.py).
          Ops are named by module path with layers folded (path, the default)
          or by the scope's op (op); aten names with the CPU op tree.
          --kind K [--max-steps N]: only N traced steps of kind K, evenly
          spaced (files TraceLens reads in minutes, not hours).
  birdeye SRC [--format auto|ruppell|nsys|rocprof|kineto] [--steps A:B]
          [--phase NAME=START:END ...] [--rank N] [--out DIR] [--short-us 10]
          [--top 30] [--comm ruppell|tracelens]
          A TraceLens-style report (REPORT.md and one CSV per table: GPU
          timeline, kernels, ops, categories, short kernels, collectives,
          steps, GEMM) from a ruppell run, its Chrome trace, a PyTorch
          profiler trace, an nsys SQLite export or a rocprofv3 kernel trace
          (docs/birdeye.md). OUT defaults to RUN/birdeye or FILE_birdeye.
  birdeye-compare OUT_DIR SRC_OR_DIR... [--tags a,b,...]
          Several birdeye reports (or sources, reported first) side by side,
          the first as the baseline: compare_<table>.csv and COMPARE.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def _sha256(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _commit() -> Optional[str]:
    try:
        c = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        c = ""
    if not c and os.path.exists(os.path.join(REPO, ".commit")):
        # a tree copied to a node without .git carries its commit in .commit,
        # and .dirty lists what differed from it
        with open(os.path.join(REPO, ".commit")) as f:
            c = f.read().strip()
        dirty = os.path.join(REPO, ".dirty")
        if c and os.path.exists(dirty) and os.path.getsize(dirty):
            c += "+dirty"
    return c or None


def _libs(build: str):
    tool = os.path.join(build, "libruppell_tool.so")
    marker = os.path.join(build, "libruppell_marker.so")
    for p in (tool, marker):
        if not os.path.exists(p):
            sys.exit(f"ruppell: {p} not found; build first (runtime/build.sh)")
    return tool, marker


def _sdk_roctx(tool: str) -> Optional[str]:
    """The profiler SDK's ROCTX library, beside the SDK the ROCm tool is linked against
    (ROCM_PATH or /opt/rocm otherwise)."""
    dirs = []
    try:
        out = subprocess.run(["ldd", tool], capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            if "librocprofiler-sdk.so" in line and "=>" in line:
                dirs.append(os.path.dirname(line.split("=>")[1].split("(")[0].strip()))
    except (OSError, subprocess.SubprocessError):
        pass
    dirs += [os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib"), "/opt/rocm/lib"]
    for d in dirs:
        p = os.path.join(d, "librocprofiler-sdk-roctx.so")
        if os.path.exists(p):
            return os.path.realpath(p)
    return None


def _platform(build: str) -> str:
    """rocm or cuda, as runtime/build.sh wrote it (builds before it: rocm)."""
    path = os.path.join(build, "PLATFORM")
    if not os.path.exists(path):
        return "rocm"
    with open(path) as f:
        return f.read().strip() or "rocm"


def cmd_run(a) -> int:
    tool, marker = _libs(os.path.abspath(a.build))
    platform = _platform(os.path.abspath(a.build))
    cmd: List[str] = a.command[1:] if a.command and a.command[0] == "--" else a.command
    if not cmd:
        sys.exit("ruppell run: no command given")
    env = dict(os.environ)
    env["RUPPELL_MARKER_LIB"] = marker
    env["PYTHONPATH"] = os.path.join(REPO, "python") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    info = {
        "command": cmd,
        "commit": _commit(),
        "tool_library": tool,
        "tool_library_sha256": _sha256(tool),
        "marker_library_sha256": _sha256(marker),
        "mode": "witness" if a.witness else "active",
        "platform": platform,
    }
    if a.witness:
        env["RUPPELL_MODE"] = "witness"
        env.pop("RUPPELL_TOOL_LIB", None)
    elif platform == "cuda":
        env["RUPPELL_TOOL_LIB"] = tool
        env["RUPPELL_OUT"] = out
        # The driver loads one injection library, and CUPTI takes one
        # subscriber (docs/feasibility-cuda.md): refuse rather than displace
        # another tool.
        prior = env.get("CUDA_INJECTION64_PATH")
        if prior and os.path.abspath(prior) != tool:
            sys.exit(f"ruppell run: CUDA_INJECTION64_PATH is already set ({prior}); one CUPTI tool per process")
        env["CUDA_INJECTION64_PATH"] = tool
        info["other_tools"] = []
        # NVTX ranges (RUPPELL_NVTX=1, roadmap-v2 M7): NVTX loads one injection library, at
        # the process's first NVTX call; the tool hands it on to CUPTI. Another tool's
        # injection (nsys) is left in place, and the run records no ranges (meta.json says so).
        if env.get("RUPPELL_NVTX", "0") not in ("", "0"):
            nvtx = env.get("NVTX_INJECTION64_PATH")
            if nvtx and os.path.abspath(nvtx) != tool:
                print(f"ruppell run: NVTX_INJECTION64_PATH is already set ({nvtx}); NVTX ranges will not be recorded",
                      file=sys.stderr)
            else:
                env["NVTX_INJECTION64_PATH"] = tool
                info["nvtx_injection"] = tool
    else:
        env["RUPPELL_TOOL_LIB"] = tool
        env["RUPPELL_OUT"] = out
        # Join, never replace, a tool already configured (for example rocprofv3
        # wrapping this command as the in-process witness).
        prior = [p for p in env.get("ROCP_TOOL_LIBRARIES", "").split(":") if p and p != tool]
        env["ROCP_TOOL_LIBRARIES"] = ":".join(prior + [tool])
        info["other_tools"] = prior
        if prior:
            # Another tool (rocprofv3) preloads the SDK so it initializes in a
            # library constructor; preload this tool as well so it is present
            # when the SDK looks for tools.
            pre = [p for p in env.get("LD_PRELOAD", "").split(":") if p and p != tool]
            env["LD_PRELOAD"] = ":".join([tool] + pre)
        if env.get("RUPPELL_NVTX", "0") not in ("", "0") and env.get("RUPPELL_ROCTX_PRELOAD") != "0":
            # ROCTX ranges (roadmap-v2 M7): PyTorch calls ROCTX through roctracer's libroctx64,
            # which has its own implementation and never reaches the profiler SDK (MI350,
            # 2026-10-20: 0 ranges). The SDK's ROCTX library, preloaded, takes those calls, as
            # rocprofv3 arranges it (every range arrived under it, and with this preload alone).
            roctx = _sdk_roctx(tool)
            pre = [p for p in env.get("LD_PRELOAD", "").split(":") if p]
            if roctx and not any(os.path.basename(p).startswith("librocprofiler-sdk-roctx") for p in pre):
                env["LD_PRELOAD"] = ":".join(pre + [roctx])
                info["roctx_preload"] = roctx
            elif not roctx:  # (rocprofv3, when it is the other tool, preloads it itself)
                print("ruppell run: librocprofiler-sdk-roctx.so not found; ROCTX ranges from libroctx64 "
                      "(PyTorch) will not be recorded", file=sys.stderr)
    # NCCL's or RCCL's own view of its calls (roadmap-v2 M8): ruppell's profiler plugin, which the
    # library loads at its first communicator. On by default since 2026-10-21, when its cost came
    # to about the library's own event handling's (runner ruppell-accuracy/results/m8, step 7);
    # RUPPELL_NCCL=0 turns it off. Another plugin is left in place.
    nccl = env.get("RUPPELL_NCCL", "")
    if not a.witness and nccl != "0":
        plug = os.path.join(os.path.dirname(tool), "libruppell_nccl_profiler.so")
        prior_plug = env.get("NCCL_PROFILER_PLUGIN")
        if prior_plug and os.path.abspath(prior_plug) != plug:
            print(f"ruppell run: NCCL_PROFILER_PLUGIN is already set ({prior_plug}); NCCL's calls will not be recorded",
                  file=sys.stderr)
        elif not os.path.exists(plug):
            if nccl:  # asked for; a build from before the plugin stays quiet by default
                print(f"ruppell run: {plug} not found (build with runtime/build.sh); NCCL's calls will not be recorded",
                      file=sys.stderr)
        else:
            env["NCCL_PROFILER_PLUGIN"] = plug
            info["nccl_profiler_plugin"] = plug
    # Hardware counters (gray-parrot; off unless RUPPELL_COUNTERS or RUPPELL_DEVICE_COUNTERS is
    # set): the sets resolved per architecture into the groups the tool reads (sets/<arch>.yaml;
    # the CUDA tool reads RUPPELL_COUNTER_GROUPS_SM_<CC> and counts per step).
    if not a.witness and (env.get("RUPPELL_COUNTERS") or env.get("RUPPELL_DEVICE_COUNTERS")):
        from ruppell import gray_parrot

        try:
            groups = gray_parrot.run_env(env)
        except ValueError as e:
            sys.exit(f"ruppell run: {e}")
        env.update(groups)
        info["counters"] = {k: env[k] for k in sorted(env) if k.startswith(("RUPPELL_COUNTER", "RUPPELL_DEVICE_"))}
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump(info, f, indent=2)
    # Forward SIGINT and SIGTERM to the command and wait for it: a server
    # shutting down must be let finish (its workers flush their records), not
    # killed with this wrapper.
    proc = subprocess.Popen(cmd, env=env)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, _f: proc.send_signal(s))
    return proc.wait()


def _steps_arg(s: Optional[str]):
    if not s:
        return None
    lo, _, hi = s.partition(":")
    return (int(lo), int(hi))


def cmd_export(a) -> int:
    if a.format == "torch":
        from ruppell.export.torch_profiler import export as export_torch

        out = a.o or os.path.join(a.run_dir, f"torch_trace_p{a.process}.json")
        print(json.dumps(export_torch(a.run_dir, out, process=a.process, steps=_steps_arg(a.steps)) | {"trace": out}, indent=2))
        return 0
    from ruppell.export.chrome import export

    out = a.o or os.path.join(a.run_dir, "trace.json")
    rep = export(a.run_dir, out, flows=not a.no_flows, steps=_steps_arg(a.steps))
    with open(os.path.join(a.run_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    summary = {
        "trace": out,
        "window": rep["window"],
        "merge": rep["merge"],
        "schema": {"ok": rep["schema"].get("ok"), "processes_checked": rep["schema"].get("processes_checked")},
        "processes": [
            {k: p[k] for k in ("process", "label", "dispatches", "attributed_share", "attributed_by", "graph_summary",
                               "steps", "dropped_records")}
            | {"unattributed": p["unattributed"][:5]}
            for p in rep["processes"]
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_tracelens(a) -> int:
    from ruppell.export.torch_profiler import write_ranks

    out = a.o or os.path.join(a.run_dir, "tracelens")
    for p in write_ranks(a.run_dir, out, steps=_steps_arg(a.steps), op_names=a.op_names,
                         kind=a.kind, max_steps=a.max_steps):
        print(p)
    return 0


def cmd_census(a) -> int:
    from ruppell.census import compute

    phases = {}
    for spec in a.phase or []:
        name, _, rng = spec.partition("=")
        lo, _, hi = rng.partition(":")
        phases[name] = (float(lo), float(hi))
    summary = compute(a.run_dir, phases=phases or None)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_witness(a) -> int:
    if a.counters:
        from ruppell.gray_parrot.witness import compare as compare_counters

        res = compare_counters(a.run_dir, [a.witness_dir] + (a.also or []), same_process=a.same_process)
        with open(os.path.join(a.run_dir, "witness_counters.json"), "w") as f:
            json.dump(res, f, indent=2)
        for pr in res["processes"]:
            print(json.dumps({k: v for k, v in pr.items() if k != "by_kernel_and_counter"}))
            for e in pr["by_kernel_and_counter"]:
                if (e["deterministic"] and e["equal"] != e["compared"]) or (not e["deterministic"] and not e["median_within_spread"]):
                    print("MISMATCH", pr["process"], json.dumps(e)[:400])
        print(json.dumps({"passes": res["passes"], "all_deterministic_equal": res["all_deterministic_equal"],
                          "all_others_medians_within_spread": res["all_others_medians_within_spread"]}))
        return 0 if res["passes"] else 1
    from ruppell.validate.witness import compare

    res = compare(a.run_dir, a.witness_dir, a.tolerance, a.spread)
    with open(os.path.join(a.run_dir, "witness.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "processes"}, indent=2))
    for pr in res["processes"]:
        print(json.dumps({k: v for k, v in pr.items() if k != "rows"}))
        for r in pr["rows"]:
            if not (r["counts_equal"] and r["time_within_tolerance"]):
                print("MISMATCH", pr["process"], json.dumps(r)[:400])
    copies_ok = res.get("copies_all_matched") is not False and res.get("copies_equal") is not False
    ok = res["all_counts_equal"] and res["all_times_within_tolerance"] and res.get("all_steps_identical", True)
    return 0 if ok and copies_ok else 1


def cmd_report(a) -> int:
    from ruppell.gray_parrot.report import markdown, markdown_steps, per_kernel, per_step

    from ruppell.gray_parrot import pm

    # CUDA's device samples (RUPPELL_DEVICE_COUNTERS) are stored raw and evaluated here, once
    for e in pm.evaluate(a.run_dir):
        if "error" in e:
            print(f"device samples of {e['dir']} not evaluated: {e['error']}", file=sys.stderr)
    dev = pm.activity(a.run_dir)
    rep = per_kernel(a.run_dir)
    steps = per_step(a.run_dir)   # per counter step: CUDA's only scope, and ROCm's step readings
    if a.json:
        with open(a.json, "w") as f:
            json.dump({**rep, "per_step": steps["processes"], "device": dev["processes"]}, f, indent=2)
    if not rep["processes"] and not any(p["table"] for p in steps["processes"]) and not dev["processes"]:
        print(f"{a.run_dir}: no counter records (run with RUPPELL_COUNTERS or RUPPELL_DEVICE_COUNTERS set)")
        return 1
    if dev["processes"]:
        print(pm.markdown(dev))
    if rep["processes"]:
        print(markdown(rep, a.top))
    if any(p["table"] for p in steps["processes"]):
        print(markdown_steps(steps, a.top))
    return 0


def _phases(specs) -> Optional[dict]:
    phases = {}
    for spec in specs or []:
        name, _, rng = spec.partition("=")
        lo, _, hi = rng.partition(":")
        phases[name] = (float(lo), float(hi))
    return phases or None


def cmd_birdeye(a) -> int:
    from ruppell.birdeye import run

    res = run(a.src, out=a.out, fmt=a.format, steps=_steps_arg(a.steps), phases=_phases(a.phase), rank=a.rank,
              short_us=a.short_us, top=a.top, comm=a.comm)
    print(json.dumps({"out": res["out"], "tables": res["tables"], "skipped": res["skipped"],
                      "rank": res["summary"]["rank"], "kernels": res["summary"]["kernels"]}, indent=2))
    print(os.path.join(res["out"], "REPORT.md"))
    return 0


def cmd_birdeye_compare(a) -> int:
    from ruppell.birdeye.compare import compare

    tags = [t for t in a.tags.split(",")] if a.tags else None
    res = compare(a.out_dir, a.inputs, tags, comm=a.comm, short_us=a.short_us)
    print(json.dumps(res, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ruppell", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--build", default=os.environ.get("RUPPELL_BUILD", os.path.join(REPO, "runtime", "build")))
    r.add_argument("--witness", action="store_true")
    r.add_argument("command", nargs=argparse.REMAINDER)
    e = sub.add_parser("export")
    e.add_argument("run_dir")
    e.add_argument("-o")
    e.add_argument("--no-flows", action="store_true")
    e.add_argument("--steps", help="export a window of steps A:B (engine steps when the run has them)")
    e.add_argument("--format", choices=("chrome", "torch"), default="chrome",
                   help="torch: one process in the PyTorch profiler's vocabulary, scopes as ops (for TraceLens)")
    e.add_argument("--process", type=int, default=0, help="--format torch: which process that ran kernels (index)")
    t = sub.add_parser("tracelens")
    t.add_argument("run_dir")
    t.add_argument("-o")
    t.add_argument("--steps", help="a window of steps A:B (engine steps when the run has them)")
    t.add_argument("--op-names", choices=("path", "op"), default="path")
    t.add_argument("--kind", help="only traced engine steps of this kind (decode, prefill, mixed)")
    t.add_argument("--max-steps", type=int, default=0, help="with --kind: at most this many, evenly spaced")
    c = sub.add_parser("census")
    c.add_argument("run_dir")
    c.add_argument("--phase", action="append", help="NAME=START:END in realtime seconds (repeatable)")
    w = sub.add_parser("witness")
    w.add_argument("run_dir")
    w.add_argument("witness_dir")
    w.add_argument("--tolerance", type=float, default=0.03)
    w.add_argument("--spread", help="a second witness run (nsys): times may differ by its own run-to-run spread")
    w.add_argument("--counters", action="store_true", help="hardware counters against rocprofv3 --pmc (gray-parrot)")
    w.add_argument("--also", action="append", help="with --counters: another rocprofv3 run, for the spread (repeatable)")
    w.add_argument("--same-process", action="store_true", help="with --counters: rocprofv3 counted in ruppell's process")
    rp = sub.add_parser("report", help="reports on a run's records")
    rp.add_argument("what", choices=("counters",))
    rp.add_argument("run_dir")
    rp.add_argument("--json", help="write the whole report as JSON here")
    rp.add_argument("--top", type=int, default=0, help="rows per process (0: all)")
    be = sub.add_parser("birdeye", help="a TraceLens-style report from a run or a trace")
    be.add_argument("src")
    be.add_argument("--format", choices=("auto", "ruppell", "nsys", "rocprof", "kineto"), default="auto")
    be.add_argument("--steps", help="steps A:B (step ids)")
    be.add_argument("--phase", action="append", help="NAME=START:END: realtime s for ruppell runs, else s from "
                                                     "the source's first activity (repeatable)")
    be.add_argument("--rank", type=int, help="the rank of the per-rank tables (default: the lowest)")
    be.add_argument("--out", help="output directory (default RUN/birdeye, FILE_birdeye)")
    be.add_argument("--short-us", type=float, default=10.0, help="short-kernel threshold, µs")
    be.add_argument("--top", type=int, default=30, help="rows per table in REPORT.md")
    be.add_argument("--comm", choices=("ruppell", "tracelens"), default="ruppell",
                    help="collective classifier: ruppell's (limits.COLLECTIVE) or TraceLens's names")
    bc = sub.add_parser("birdeye-compare", help="several birdeye reports side by side")
    bc.add_argument("out_dir")
    bc.add_argument("inputs", nargs="+")
    bc.add_argument("--tags", help="comma-separated tags, one per input (default: their base names)")
    bc.add_argument("--comm", choices=("ruppell", "tracelens"), default="ruppell")
    bc.add_argument("--short-us", type=float, default=10.0)
    a = p.parse_args(argv)
    return {"run": cmd_run, "export": cmd_export, "census": cmd_census, "witness": cmd_witness, "report": cmd_report,
            "tracelens": cmd_tracelens, "birdeye": cmd_birdeye, "birdeye-compare": cmd_birdeye_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
