#!/usr/bin/env python3
"""Classify the removable share of src/hyperloom without changing behaviour.

Three independent levers, each measured rather than asserted:
  1. unreachable prod modules, split by who still references them;
  2. duplicated function bodies (token-identical) inside prod and inside tests;
  3. the comment / docstring / blank share, reported as structure, not savings.

Reads loc-census.json. Output: loc-redundancy.json.
"""
from __future__ import annotations

import ast
import collections
import hashlib
import io
import json
import os
import subprocess
import tokenize

ROOT = "src/hyperloom"
CENSUS = json.load(open("loc-census.json"))


def is_test(path: str) -> bool:
    parts = path.split(os.sep)
    base = parts[-1]
    return "tests" in parts or base.startswith("test_") or base == "conftest.py"


def all_py() -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        out += [os.path.join(dirpath, f) for f in sorted(filenames) if f.endswith(".py")]
    return out


def module_path(mod: str) -> str:
    rel = mod.replace(".", os.sep)
    for cand in (f"src/{rel}.py", f"src/{rel}/__init__.py"):
        if os.path.exists(cand):
            return cand
    return ""


def line_counts(path: str) -> tuple[int, int, int]:
    src = open(path, encoding="utf-8", errors="replace").read()
    physical = src.count("\n")
    from radon.raw import analyze

    try:
        r = analyze(src)
        return physical, r.sloc, r.lloc
    except (SyntaxError, ValueError):
        return physical, 0, 0


def classify_unreached() -> dict:
    """For each unreachable prod module, find who still mentions it."""
    unreached = sorted(
        set(m for m in _prod_modules()) - set(_reachable())
    )
    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for mod in unreached:
        path = module_path(mod)
        if not path:
            continue
        leaf = mod.rsplit(".", 1)[-1]
        # Who mentions this module name anywhere in the repo, excluding itself?
        hits = subprocess.run(
            ["rg", "-l", "--no-messages", rf"\b{leaf}\b", "src", "scripts", "docs", "examples", "ci"],
            capture_output=True,
            text=True,
        ).stdout.split()
        hits = [h for h in hits if os.path.abspath(h) != os.path.abspath(path)]
        prod_hits = [h for h in hits if h.endswith(".py") and not is_test(h) and h.startswith("src/")]
        test_hits = [h for h in hits if h.endswith(".py") and is_test(h)]
        doc_hits = [h for h in hits if not h.endswith(".py") or not h.startswith("src/")]

        if prod_hits:
            bucket = "referenced_by_prod"  # dynamic use the import graph missed
        elif test_hits:
            bucket = "test_only"  # code kept alive only by its own tests
        elif doc_hits:
            bucket = "doc_or_ops_only"
        else:
            bucket = "no_reference_anywhere"
        phys, sloc, lloc = line_counts(path)
        buckets[bucket].append(
            {
                "module": mod,
                "path": path,
                "physical": phys,
                "sloc": sloc,
                "lloc": lloc,
                "prod_refs": len(prod_hits),
                "test_refs": len(test_hits),
                "other_refs": len(doc_hits),
            }
        )

    summary = {}
    for name, rows in buckets.items():
        summary[name] = {
            "modules": len(rows),
            "physical": sum(r["physical"] for r in rows),
            "sloc": sum(r["sloc"] for r in rows),
            "lloc": sum(r["lloc"] for r in rows),
            "top": sorted(rows, key=lambda r: -r["lloc"])[:15],
        }
    return summary


def _prod_modules() -> list[str]:
    mods = []
    for p in all_py():
        if is_test(p):
            continue
        rel = os.path.relpath(p, os.path.dirname(ROOT))
        mod = rel[:-3].replace(os.sep, ".")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        mods.append(mod)
    return mods


_REACH_CACHE: list[str] | None = None


def _reachable() -> list[str]:
    global _REACH_CACHE
    if _REACH_CACHE is None:
        sc = CENSUS["reachability"]["scenarios"]["unreached_from_any_entry"]
        unreached = set(sc["modules_list"])
        _REACH_CACHE = [m for m in _prod_modules() if m not in unreached]
    return _REACH_CACHE


def func_bodies(path: str) -> list[tuple[str, str, int, int]]:
    """(name, token-hash, physical span, lloc) for every function in a file."""
    src = open(path, encoding="utf-8", errors="replace").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.split("\n")
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start, end = node.lineno, node.end_lineno or node.lineno
        seg = "\n".join(lines[start - 1 : end])
        # Normalise: drop comments, docstrings, spacing, and the function's own name.
        toks = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(seg.lstrip()).readline):
                if tok.type in (
                    tokenize.COMMENT,
                    tokenize.NL,
                    tokenize.NEWLINE,
                    tokenize.INDENT,
                    tokenize.DEDENT,
                    tokenize.ENCODING,
                    tokenize.ENDMARKER,
                ):
                    continue
                toks.append(tok.string)
        except (tokenize.TokenError, IndentationError):
            continue
        if len(toks) < 25:  # ignore trivial getters; they are not the bloat
            continue
        body = toks[toks.index("(") if "(" in toks else 0 :]
        h = hashlib.sha256("\x00".join(body).encode()).hexdigest()
        stmts = sum(1 for n in ast.walk(node) if isinstance(n, ast.stmt))
        out.append((node.name, h, end - start + 1, stmts))
    return out


def duplicate_functions(paths: list[str]) -> dict:
    groups: dict[str, list[tuple[str, str, int, int]]] = collections.defaultdict(list)
    total_funcs = 0
    for p in paths:
        for name, h, span, stmts in func_bodies(p):
            total_funcs += 1
            groups[h].append((p, name, span, stmts))
    dups = {h: v for h, v in groups.items() if len(v) > 1}
    redundant_copies = sum(len(v) - 1 for v in dups.values())
    redundant_physical = sum(v[0][2] * (len(v) - 1) for v in dups.values())
    redundant_stmts = sum(v[0][3] * (len(v) - 1) for v in dups.values())
    top = sorted(dups.values(), key=lambda v: -(v[0][2] * (len(v) - 1)))[:15]
    return {
        "functions_scanned": total_funcs,
        "duplicate_groups": len(dups),
        "redundant_copies": redundant_copies,
        "redundant_physical": redundant_physical,
        "redundant_stmts": redundant_stmts,
        "top": [
            {
                "name": v[0][1],
                "copies": len(v),
                "physical_each": v[0][2],
                "locations": [f"{p}:{n}" for p, n, _, _ in v[:6]],
            }
            for v in top
        ],
    }


def main() -> None:
    paths = all_py()
    prod = [p for p in paths if not is_test(p)]
    test = [p for p in paths if is_test(p)]

    result = {
        "unreachable": classify_unreached(),
        "duplicate_functions": {
            "prod": duplicate_functions(prod),
            "test": duplicate_functions(test),
        },
    }
    json.dump(result, open("loc-redundancy.json", "w"), indent=1)

    print("=== unreachable prod modules, by who still references them ===")
    for name, r in sorted(result["unreachable"].items(), key=lambda kv: -kv[1]["lloc"]):
        print(f"{name:24s} modules {r['modules']:4d}  physical {r['physical']:7d}  sloc {r['sloc']:7d}  lloc {r['lloc']:7d}")
        for row in r["top"][:5]:
            print(f"      {row['lloc']:6d} lloc  {row['module']}")
    print()
    for scope in ("prod", "test"):
        d = result["duplicate_functions"][scope]
        print(
            f"=== duplicate function bodies ({scope}) === scanned {d['functions_scanned']}, "
            f"{d['duplicate_groups']} groups, {d['redundant_copies']} redundant copies, "
            f"{d['redundant_physical']} physical lines, {d['redundant_stmts']} stmts"
        )
        for row in d["top"][:8]:
            print(f"      {row['copies']}x {row['physical_each']:4d} lines  {row['name']}  e.g. {row['locations'][0]}")
        print()


if __name__ == "__main__":
    main()
