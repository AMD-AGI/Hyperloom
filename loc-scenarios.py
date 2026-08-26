#!/usr/bin/env python3
"""Feature-level ledger for src/hyperloom prod code, and core-only scenarios.

Groups prod modules into features at the granularity a product decision is
actually made at (one agent, one subsystem area), then prices two scenarios:
  A. keep every feature, delete only what is provably redundant;
  B. keep the loop README advertises as the product, drop the rest.

For scenario B it also counts the import edges that cross the keep/drop line,
since those are the edges that would have to be cut or stubbed.

Output: loc-scenarios.json
"""
from __future__ import annotations

import ast
import collections
import glob
import json
import os

from radon.raw import analyze

ROOT = "src/hyperloom"
RENDERERS = "src/hyperloom/inference_optimizer/breakdown/reporters/_renderers/*.py"


def physical_lines(src: str) -> int:
    """Same convention as loc-census.py: a file with no trailing newline still
    ends on a line."""
    if not src:
        return 0
    return src.count("\n") if src.endswith("\n") else src.count("\n") + 1


def is_test(path: str) -> bool:
    parts = path.split(os.sep)
    base = parts[-1]
    return "tests" in parts or base.startswith("test_") or base == "conftest.py"


def feature_of(path: str) -> str:
    """Group a prod file into a decision-sized feature bucket."""
    rel = os.path.relpath(path, ROOT).split(os.sep)
    if rel[0] == "agents":
        return f"agents/{rel[1]}" if len(rel) > 1 else "agents"
    if rel[0] in ("orchestrator", "inference_optimizer"):
        return f"{rel[0]}/{rel[1]}" if len(rel) > 2 else f"{rel[0]}/_root"
    return rel[0]


def feature_of_module(mod: str) -> str:
    parts = mod.split(".")  # hyperloom.<a>.<b>...
    if len(parts) < 2:
        return "_root"
    if parts[1] == "agents":
        return f"agents/{parts[2]}" if len(parts) > 2 else "agents"
    if parts[1] in ("orchestrator", "inference_optimizer"):
        return f"{parts[1]}/{parts[2]}" if len(parts) > 3 else f"{parts[1]}/_root"
    return parts[1]


def main() -> None:
    feats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    mod_feat: dict[str, str] = {}
    graph: dict[str, set[str]] = {}
    known: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            if is_test(path):
                continue
            src = open(path, encoding="utf-8", errors="replace").read()
            f = feature_of(path)
            try:
                r = analyze(src)
                sloc, lloc = r.sloc, r.lloc
            except (SyntaxError, ValueError):
                sloc = lloc = 0
            feats[f]["files"] += 1
            feats[f]["physical"] += physical_lines(src)
            feats[f]["sloc"] += sloc
            feats[f]["lloc"] += lloc

            rel = os.path.relpath(path, os.path.dirname(ROOT))
            mod = rel[:-3].replace(os.sep, ".")
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            mod_feat[mod] = f
            known.add(mod)
            try:
                tree = ast.parse(src)
            except SyntaxError:
                graph[mod] = set()
                continue
            pkg = mod.rsplit(".", 1)[0] if "." in mod else mod
            targets: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    targets |= {a.name for a in node.names if a.name.startswith("hyperloom")}
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        base = pkg.split(".")
                        strip = node.level - 1
                        base = base[: len(base) - strip] if strip else base
                        t = ".".join(base + ([node.module] if node.module else []))
                    elif node.module and node.module.startswith("hyperloom"):
                        t = node.module
                    else:
                        continue
                    targets.add(t)
            graph[mod] = targets

    def resolve(t: str) -> str | None:
        if t in known:
            return t
        parent = t.rsplit(".", 1)[0]
        return parent if parent in known else None

    # Feature-to-feature import edges.
    edges: collections.Counter = collections.Counter()
    for mod, targets in graph.items():
        src_f = mod_feat[mod]
        for t in targets:
            r = resolve(t)
            if not r:
                continue
            dst_f = mod_feat[r]
            if dst_f != src_f:
                edges[(src_f, dst_f)] += 1

    total = collections.Counter()
    for c in feats.values():
        total.update(c)

    # Tier A core: every feature README calls the product itself — the Arbor
    # loop (including its specialist/knowledge/role/bus infrastructure), the
    # TraceLens+GEAK kernel path, the session report, and the CLI.
    core = {
        "common",
        "orchestrator/_root",
        "orchestrator/loop",
        "orchestrator/phases",
        "orchestrator/actions",
        "orchestrator/state",
        "orchestrator/measurement",
        "orchestrator/kernel",
        "orchestrator/knowledge",
        "orchestrator/specialists",
        "orchestrator/roles",
        "orchestrator/prompts",
        "orchestrator/policy",
        "orchestrator/scoring",
        "orchestrator/bus",
        "orchestrator/trace",
        "inference_optimizer/_root",
        "inference_optimizer/cli",
        "inference_optimizer/breakdown",
        "inference_optimizer/session",
        "inference_optimizer/protocol",
        "agents/kernel",
    }
    core = {f for f in core if f in feats}
    dropped = set(feats) - core

    def sums(names) -> dict:
        out = collections.Counter()
        for n in names:
            out.update(feats[n])
        return dict(out)

    crossing = {
        f"{a} -> {b}": c for (a, b), c in edges.items() if a in core and b in dropped
    }

    # Tier B: optional surface that sits *inside* a kept feature. One workload
    # class (inference serving), one kernel backend (GEAK), one kernel language
    # route, and only the report sections the session summary needs.
    tier_b = {
        "diffusion workload path": [
            "src/hyperloom/agents/kernel/tools/diffusion_flops.py",
            "src/hyperloom/agents/kernel/tools/diffusion_roofline.py",
            "src/hyperloom/agents/kernel/tools/_denoise_steps.py",
        ],
        "collective codegen": [
            "src/hyperloom/agents/kernel/tools/collective_driver_generator.py",
            "src/hyperloom/agents/kernel/tools/forge_collective.py",
            "src/hyperloom/agents/kernel/tools/_collective_names.py",
            "src/hyperloom/agents/kernel/tools/_nccl_summary_candidates.py",
        ],
        "Forge backend (GEAK kept)": [
            "src/hyperloom/agents/kernel/tools/backends/forge_submit.py",
            "src/hyperloom/agents/kernel/tools/forge_fusion.py",
            "src/hyperloom/agents/kernel/tools/forge_gemm_tuning.py",
        ],
        "FlyDSL rewrite route": [
            "src/hyperloom/agents/kernel/tools/backends/_flydsl_rewrite.py",
        ],
        "non-core report sections": [
            p
            for p in sorted(glob.glob(RENDERERS))
            if os.path.basename(p)
            not in {
                "__init__.py",
                "_invocation.py",
                "final.py",
                "session.py",
                "workload.py",
                "optimizations.py",
                "kernel_lifecycle.py",
                "roofline.py",
            }
        ],
    }
    tier_b_detail = {}
    tier_b_total = collections.Counter()
    for name, paths in tier_b.items():
        c = collections.Counter()
        for p in paths:
            if not os.path.exists(p):
                continue
            src = open(p, encoding="utf-8", errors="replace").read()
            try:
                r = analyze(src)
                sloc, lloc = r.sloc, r.lloc
            except (SyntaxError, ValueError):
                sloc = lloc = 0
            c["files"] += 1
            c["physical"] += physical_lines(src)
            c["sloc"] += sloc
            c["lloc"] += lloc
        tier_b_detail[name] = dict(c)
        tier_b_total.update(c)

    result = {
        "features": {k: dict(v) for k, v in sorted(feats.items(), key=lambda kv: -kv[1]["lloc"])},
        "total_prod": dict(total),
        "scenario_core": {
            "keep": sorted(core),
            "drop": sorted(dropped),
            "kept": sums(core),
            "dropped": sums(dropped),
            "edges_to_cut": dict(sorted(crossing.items(), key=lambda kv: -kv[1])),
        },
        "tier_b_trim": {"detail": tier_b_detail, "total": dict(tier_b_total)},
        "feature_edges": {f"{a} -> {b}": c for (a, b), c in edges.most_common()},
    }
    json.dump(result, open("loc-scenarios.json", "w"), indent=1)

    print(f"{'feature':34s}{'files':>6s}{'physical':>10s}{'sloc':>9s}{'lloc':>9s}")
    for k, v in result["features"].items():
        mark = "KEEP" if k in core else "drop"
        print(f"{k:34s}{v['files']:6d}{v['physical']:10d}{v['sloc']:9d}{v['lloc']:9d}  {mark}")
    t = result["total_prod"]
    print(f"{'TOTAL prod':34s}{t['files']:6d}{t['physical']:10d}{t['sloc']:9d}{t['lloc']:9d}")
    sc = result["scenario_core"]
    k, d = sc["kept"], sc["dropped"]
    print(f"\ncore keep : {k['files']:4d} files  {k['physical']:7d} physical  {k['sloc']:7d} sloc  {k['lloc']:7d} lloc")
    print(f"core drop : {d['files']:4d} files  {d['physical']:7d} physical  {d['sloc']:7d} sloc  {d['lloc']:7d} lloc")
    print(f"kept share: {k['lloc'] / t['lloc']:.1%} of prod lloc")
    tb = result["tier_b_trim"]
    print("\ntier B — optional surface inside the kept core:")
    for name, v in tb["detail"].items():
        print(f"   {name:30s} {v.get('files', 0):3d} files  {v.get('physical', 0):6d} physical  {v.get('lloc', 0):6d} lloc")
    t2 = tb["total"]
    print(f"   {'TOTAL':30s} {t2.get('files', 0):3d} files  {t2.get('physical', 0):6d} physical  {t2.get('lloc', 0):6d} lloc")
    print(
        f"\ncore after tier B: {k['lloc'] - t2.get('lloc', 0)} lloc "
        f"({(k['lloc'] - t2.get('lloc', 0)) / t['lloc']:.1%} of prod lloc)"
    )
    print(f"\nedges that would have to be cut (core -> dropped): {len(sc['edges_to_cut'])} feature pairs")
    for e, c in list(sc["edges_to_cut"].items())[:12]:
        print(f"   {e}: {c}")


if __name__ == "__main__":
    main()
