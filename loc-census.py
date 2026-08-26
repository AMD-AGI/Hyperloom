#!/usr/bin/env python3
"""LOC census for src/hyperloom.

Emits four line-count definitions per file (physical / SLOC / AST statements /
radon LLOC), a prod-vs-test split, duplicate detection, and a module-level
import graph used to derive reachability from the real CLI entry points.

Output: loc-census.json (consumed by make_loc_slide.py and loc-report.md).
"""
from __future__ import annotations

import ast
import collections
import hashlib
import io
import json
import os
import sys
import tokenize

from radon.raw import analyze as radon_analyze

ROOT = "src/hyperloom"

# pyproject [project.scripts] plus the module paths the orchestrator spawns
# directly (`python -m hyperloom.agents.*`) and the __main__-guarded CLIs.
ENTRY_POINTS = {
    "optimizer_cli": ["hyperloom.inference_optimizer.cli"],
    "bootstrap": ["hyperloom.inference_optimizer.setup"],
    "framework_agent": ["hyperloom.agents.framework.runtime.cli"],
    "robustness_agent": ["hyperloom.agents.robustness.main"],
    "quantization_agent": ["hyperloom.agents.quantization.cli"],
    "critic_agent": ["hyperloom.agents.critic.runtime.cli"],
    "multi_node": ["hyperloom.inference_optimizer.multi_node.cli"],
}


def is_test(path: str) -> bool:
    parts = path.split(os.sep)
    base = parts[-1]
    return (
        "tests" in parts
        or base.startswith("test_")
        or base == "conftest.py"
        or base.endswith("_test.py")
    )


def measure(src: str) -> dict:
    """physical / blank / comment-only / docstring-only / sloc / ast statements."""
    lines = src.split("\n")
    if not src:
        physical = 0
    else:
        physical = len(lines) - 1 if src.endswith("\n") else len(lines)

    comment_lines: set[int] = set()
    string_lines: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                comment_lines.update(range(tok.start[0], tok.end[0] + 1))
            elif tok.type == tokenize.STRING:
                string_lines.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass

    # Docstring lines: string literals that are standalone expression statements.
    doc_lines: set[int] = set()
    stmts = 0
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.stmt):
                is_doc = (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                )
                if is_doc:
                    doc_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
                else:
                    stmts += 1
    except SyntaxError:
        stmts = -1

    blank = 0
    comment_only = 0
    doc_only = 0
    code = 0
    for i, text in enumerate(lines[:physical], start=1):
        if not text.strip():
            blank += 1
        elif i in doc_lines:
            doc_only += 1
        elif i in comment_lines and i not in string_lines:
            # A line holding only a comment (no code token before it).
            stripped = text.strip()
            if stripped.startswith("#"):
                comment_only += 1
            else:
                code += 1
        else:
            code += 1
    try:
        r = radon_analyze(src)
        lloc, radon_sloc = r.lloc, r.sloc
    except (SyntaxError, ValueError):
        lloc = radon_sloc = 0

    return {
        "physical": physical,
        "blank": blank,
        "comment": comment_only,
        "docstring": doc_only,
        "sloc": code,
        "stmts": stmts,
        "lloc": lloc,
        "radon_sloc": radon_sloc,
    }


def has_main_guard(src: str) -> bool:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        t = node.test
        if (
            isinstance(t, ast.Compare)
            and isinstance(t.left, ast.Name)
            and t.left.id == "__name__"
            and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in t.comparators)
        ):
            return True
    return False


def string_module_refs(src: str) -> set[str]:
    """hyperloom.* module paths that appear inside string literals.

    These are the modules spawned as ``python -m ...`` or resolved through
    importlib, which the import graph cannot see.
    """
    refs: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return refs
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for tok in node.value.replace(",", " ").split():
                if tok.startswith("hyperloom.") and all(
                    part.isidentifier() for part in tok.split(".")
                ):
                    refs.add(tok)
    return refs


def normalized_hash(src: str) -> str:
    """Hash of code tokens only: ignores comments, docstrings, blank lines, spacing."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENCODING,
            ):
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return "unparsed:" + hashlib.sha256(src.encode()).hexdigest()
    return hashlib.sha256("\x00".join(out).encode()).hexdigest()


def module_name(path: str) -> str:
    rel = os.path.relpath(path, os.path.dirname(ROOT))
    mod = rel[:-3].replace(os.sep, ".")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    return mod


def imports_of(src: str, mod: str) -> set[str]:
    """Absolute hyperloom.* module targets imported by this module."""
    found: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return found
    pkg = mod.rsplit(".", 1)[0] if "." in mod else mod
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("hyperloom"):
                    found.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                # level 1 = current package; each extra level strips one more.
                strip = node.level - 1
                base = base[: len(base) - strip] if strip else base
                target = ".".join(base + ([node.module] if node.module else []))
            elif node.module and node.module.startswith("hyperloom"):
                target = node.module
            else:
                continue
            found.add(target)
            for a in node.names:
                found.add(f"{target}.{a.name}")
    return found


def main() -> None:
    files: dict[str, dict] = {}
    by_hash: dict[str, list[str]] = collections.defaultdict(list)
    by_norm: dict[str, list[str]] = collections.defaultdict(list)
    graph: dict[str, set[str]] = {}
    mod_to_path: dict[str, str] = {}

    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            raw = open(path, "rb").read()
            src = raw.decode("utf-8", errors="replace")
            m = measure(src)
            m["test"] = is_test(path)
            m["module"] = module_name(path)
            m["main_guard"] = has_main_guard(src)
            m["string_refs"] = sorted(string_module_refs(src))
            files[path] = m
            by_hash[hashlib.sha256(raw).hexdigest()].append(path)
            by_norm[normalized_hash(src)].append(path)
            mod_to_path[m["module"]] = path
            graph[m["module"]] = imports_of(src, m["module"])

    def agg(paths) -> dict:
        keys = ("physical", "sloc", "blank", "comment", "docstring", "lloc", "radon_sloc")
        out = {"files": 0, "stmts": 0, **{k: 0 for k in keys}}
        for p in paths:
            m = files[p]
            out["files"] += 1
            for k in keys:
                out[k] += m[k]
            out["stmts"] += max(m["stmts"], 0)
        return out

    prod = [p for p, m in files.items() if not m["test"]]
    test = [p for p, m in files.items() if m["test"]]

    # Subsystem = first two path components under src/hyperloom, or the top one.
    def subsystem(path: str) -> str:
        rel = os.path.relpath(path, ROOT)
        parts = rel.split(os.sep)
        return parts[0] if len(parts) == 1 else parts[0]

    subs: dict[str, dict] = {}
    for name in sorted({subsystem(p) for p in files}):
        sel = [p for p in files if subsystem(p) == name]
        subs[name] = {
            "all": agg(sel),
            "prod": agg([p for p in sel if not files[p]["test"]]),
            "test": agg([p for p in sel if files[p]["test"]]),
        }

    # Duplicates: exact bytes, and code-token-identical (ignoring comments/docstrings).
    def dup_report(groups: dict[str, list[str]]) -> dict:
        dups = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
        redundant_files = 0
        redundant_physical = 0
        redundant_sloc = 0
        detail = []
        for k, v in dups.items():
            keep, drop = v[0], v[1:]
            redundant_files += len(drop)
            redundant_physical += sum(files[p]["physical"] for p in drop)
            redundant_sloc += sum(files[p]["sloc"] for p in drop)
            detail.append(
                {
                    "keep": keep,
                    "drop": drop,
                    "sloc_each": files[keep]["sloc"],
                    "physical_each": files[keep]["physical"],
                    "test": files[keep]["test"],
                }
            )
        detail.sort(key=lambda d: -d["physical_each"] * len(d["drop"]))
        return {
            "groups": len(dups),
            "redundant_files": redundant_files,
            "redundant_physical": redundant_physical,
            "redundant_sloc": redundant_sloc,
            "detail": detail,
        }

    exact = dup_report(by_hash)
    normalized = dup_report(by_norm)

    # Reachability from the console entry point over prod modules only.
    prod_mods = {files[p]["module"] for p in prod}

    def resolve(target: str) -> str | None:
        if target in mod_to_path:
            return target
        parent = target.rsplit(".", 1)[0]
        return parent if parent in mod_to_path else None

    def closure(seeds: list[str]) -> set[str]:
        seen: set[str] = set()
        stack = [s for s in seeds if s in mod_to_path]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for t in graph.get(cur, ()):  # noqa: SIM118
                r = resolve(t)
                if r and r not in seen and r in prod_mods:
                    stack.append(r)
        return seen

    scenarios: dict[str, dict] = {}
    for name, seeds in ENTRY_POINTS.items():
        mods = closure(seeds)
        scenarios[name] = {
            "seeds": seeds,
            "modules": len(mods),
            "metrics": agg([mod_to_path[m] for m in mods]),
        }

    # Every module runnable in its own right, plus every module named in a
    # string literal (spawned via `python -m` or importlib).
    main_mods = sorted(files[p]["module"] for p in prod if files[p]["main_guard"])
    referenced: set[str] = set()
    for p in prod:
        for r in files[p]["string_refs"]:
            hit = resolve(r)
            if hit and hit in prod_mods:
                referenced.add(hit)

    console_seeds = sorted(s for v in ENTRY_POINTS.values() for s in v)
    every_seed = sorted(set(console_seeds) | set(main_mods) | referenced)
    all_reachable = closure(every_seed)
    scenarios["all_entry_points"] = {
        "seeds": every_seed,
        "seed_counts": {
            "console_scripts": len(console_seeds),
            "main_guarded": len(main_mods),
            "string_referenced": len(referenced),
        },
        "modules": len(all_reachable),
        "metrics": agg([mod_to_path[m] for m in all_reachable]),
    }
    unreached = sorted(prod_mods - all_reachable)
    scenarios["unreached_from_any_entry"] = {
        "seeds": [],
        "modules": len(unreached),
        "metrics": agg([mod_to_path[m] for m in unreached]),
        "modules_list": unreached,
    }

    # Anything imported by nobody (excluding package __init__ and __main__).
    imported: set[str] = set()
    for mod, targets in graph.items():
        for t in targets:
            r = resolve(t)
            if r:
                imported.add(r)
    orphans = sorted(
        m
        for m in prod_mods
        if m not in imported
        and not m.endswith("__init__")
        and not m.endswith("__main__")
        and m not in {s for seeds in ENTRY_POINTS.values() for s in seeds}
    )

    result = {
        "root": ROOT,
        "entry_points": ENTRY_POINTS,
        "totals": {"all": agg(files), "prod": agg(prod), "test": agg(test)},
        "subsystems": subs,
        "duplicates": {"exact": exact, "normalized": normalized},
        "reachability": {
            "prod_modules": len(prod_mods),
            "scenarios": scenarios,
            "orphan_modules": len(orphans),
            "orphans": agg([mod_to_path[m] for m in orphans]),
            "orphan_list": orphans,
        },
    }
    json.dump(result, open("loc-census.json", "w"), indent=1)

    t = result["totals"]
    hdr = f"{'':18s} {'files':>7s} {'physical':>10s} {'SLOC':>10s} {'radon sloc':>11s} {'LLOC':>10s} {'ast stmts':>10s}"
    print(hdr)
    for k in ("all", "prod", "test"):
        r = t[k]
        print(
            f"{k:18s} {r['files']:7d} {r['physical']:10d} {r['sloc']:10d} "
            f"{r['radon_sloc']:11d} {r['lloc']:10d} {r['stmts']:10d}"
        )
    print()
    for name, r in subs.items():
        a, p, s = r["all"], r["prod"], r["test"]
        print(
            f"{name:20s} files {a['files']:4d}  physical {a['physical']:7d}  sloc {a['sloc']:7d}  "
            f"lloc {a['lloc']:7d}   (prod lloc {p['lloc']:6d} / test lloc {s['lloc']:6d})"
        )
    print()
    print(f"{'scenario':28s} {'mods':>5s} {'files':>6s} {'physical':>9s} {'SLOC':>8s} {'LLOC':>8s}")
    for name, sc in scenarios.items():
        m = sc["metrics"]
        print(
            f"{name:28s} {sc['modules']:5d} {m['files']:6d} {m['physical']:9d} {m['sloc']:8d} {m['lloc']:8d}"
        )
    print()
    print(
        f"exact dup: {exact['groups']} groups, {exact['redundant_files']} redundant files, "
        f"{exact['redundant_physical']} physical / {exact['redundant_sloc']} sloc"
    )
    print(
        f"token-identical dup: {normalized['groups']} groups, {normalized['redundant_files']} redundant files, "
        f"{normalized['redundant_physical']} physical / {normalized['redundant_sloc']} sloc"
    )
    rc = result["reachability"]
    print(f"prod modules: {rc['prod_modules']}")
    print(
        f"orphan prod modules (imported by nobody): {rc['orphan_modules']}, "
        f"{rc['orphans']['sloc']} sloc / {rc['orphans']['lloc']} lloc"
    )


if __name__ == "__main__":
    sys.exit(main())
