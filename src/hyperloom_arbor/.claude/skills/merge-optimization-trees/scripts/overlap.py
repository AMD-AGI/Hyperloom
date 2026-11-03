#!/usr/bin/env python3
"""Compare two patches to classify a merge candidate as NEW, ALT or DUP.

Reports, per file touched by both sides: the changed-function sets, the
new-symbol sets, and whether they intersect.

    python overlap.py base.patch candidate.patch --strip-prefix vllm/

Reading the result:

    no shared files                          -> NEW  (presumed additive)
    shared files, no shared functions        -> NEW  (co-located; textual
                                                conflicts without semantic)
    shared functions, no shared new symbols  -> ALT  (two implementations of
                                                one idea; needs a knob each)
    shared new symbols                       -> DUP  (confirm by diffing the
                                                symbol bodies)

Stackability is a separate, read-only check:

    git apply --check -p1 candidate.patch

No third-party dependencies.
"""

import argparse
import fnmatch
import json
import re
import sys

DIFF_RE = re.compile(r"^diff --git a/(?P<a>\S+) b/(?P<b>\S+)")
HUNK_RE = re.compile(r"^@@ .*@@\s*(?P<ctx>.*)$")
ENCLOSING_RE = re.compile(r"(?:def|class)\s+(\w+)")
ADDED_DEF_RE = re.compile(r"^\+\s*(?:async\s+def|def|class)\s+(\w+)")
SKIP_SUFFIXES = (".rej", ".orig")


def parse_patch(text, strip_prefix=None, exclude=()):
    """Map path -> {add, del, functions, new_symbols}.

    `functions` comes from hunk-header context, i.e. the enclosing definition
    git reports for each hunk. `new_symbols` are definitions the patch adds.
    """
    files = {}
    cur = None
    for line in text.splitlines():
        m = DIFF_RE.match(line)
        if m:
            path = m.group("b")
            if strip_prefix and path.startswith(strip_prefix):
                path = path[len(strip_prefix):]
            dropped = path.endswith(SKIP_SUFFIXES) or any(
                fnmatch.fnmatch(path, pat) for pat in exclude
            )
            cur = None if dropped else files.setdefault(
                path, {"add": 0, "del": 0, "functions": set(), "new_symbols": set()}
            )
            continue
        if cur is None:
            continue
        m = HUNK_RE.match(line)
        if m:
            fn = ENCLOSING_RE.search(m.group("ctx"))
            if fn:
                cur["functions"].add(fn.group(1))
            continue
        if line.startswith("+") and not line.startswith("+++"):
            cur["add"] += 1
            d = ADDED_DEF_RE.match(line)
            if d:
                cur["new_symbols"].add(d.group(1))
        elif line.startswith("-") and not line.startswith("---"):
            cur["del"] += 1
    return files


def classify(shared_functions, shared_symbols, shared_files):
    if shared_symbols:
        return "DUP"
    if shared_functions:
        return "ALT"
    return "NEW"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_patch")
    ap.add_argument("candidate_patch")
    ap.add_argument("--strip-prefix", default=None,
                    help="path prefix to drop from both sides, e.g. vllm/")
    ap.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                    help="ignore paths matching this glob; repeatable. Use for "
                         "scaffolding and instrumentation, which is dropped "
                         "rather than merged (e.g. --exclude 'arbor/*')")
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="also write the full result as JSON")
    args = ap.parse_args(argv)

    try:
        base = parse_patch(open(args.base_patch, errors="replace").read(),
                           args.strip_prefix, args.exclude)
        cand = parse_patch(open(args.candidate_patch, errors="replace").read(),
                           args.strip_prefix, args.exclude)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not base or not cand:
        empty = "base" if not base else "candidate"
        print(f"error: no file diffs parsed from the {empty} patch — "
              f"check it is a git-format diff", file=sys.stderr)
        return 2

    sb, sc = set(base), set(cand)
    both, only_b, only_c = sorted(sb & sc), sorted(sb - sc), sorted(sc - sb)

    def totals(d):
        return sum(f["add"] for f in d.values()), sum(f["del"] for f in d.values())

    ba, bd = totals(base)
    ca, cd = totals(cand)
    print(f"base      : {len(sb):3d} files  +{ba}/-{bd}")
    print(f"candidate : {len(sc):3d} files  +{ca}/-{cd}")

    all_shared_fn, all_shared_sym = set(), set()
    per_file = {}
    print(f"\n--- touched by both ({len(both)}) ---")
    for path in both:
        b, c = base[path], cand[path]
        fns = b["functions"] & c["functions"]
        syms = b["new_symbols"] & c["new_symbols"]
        all_shared_fn |= fns
        all_shared_sym |= syms
        per_file[path] = {
            "base": {"add": b["add"], "del": b["del"]},
            "candidate": {"add": c["add"], "del": c["del"]},
            "shared_functions": sorted(fns),
            "shared_new_symbols": sorted(syms),
            "class": classify(fns, syms, True),
        }
        print(f"  {path}")
        print(f"      base +{b['add']}/-{b['del']}   "
              f"candidate +{c['add']}/-{c['del']}   [{per_file[path]['class']}]")
        print(f"      shared functions  : {', '.join(sorted(fns)) or 'NONE'}")
        if syms:
            print(f"      shared new symbols: {', '.join(sorted(syms))}")
    if not both:
        print("  none")

    print(f"\n--- base only ({len(only_b)}) ---")
    for p in only_b:
        print(f"  {p:<66} +{base[p]['add']}/-{base[p]['del']}")
    print(f"\n--- candidate only ({len(only_c)}) ---")
    for p in only_c:
        print(f"  {p:<66} +{cand[p]['add']}/-{cand[p]['del']}")

    verdict = classify(all_shared_fn, all_shared_sym, both)
    print(f"\n=== overall: {verdict} ===")
    print(f"  shared files       : {len(both)}")
    print(f"  shared functions   : {len(all_shared_fn)}")
    print(f"  shared new symbols : {len(all_shared_sym)}"
          f"{' -> confirm by diffing the symbol bodies' if all_shared_sym else ''}")
    if verdict == "ALT":
        print("  -> give each implementation its own knob and measure both")
    elif verdict == "NEW" and both:
        print("  -> additive, but expect textual conflicts in the shared files")

    if args.json:
        payload = {
            "verdict": verdict,
            "shared_files": both,
            "base_only": only_b,
            "candidate_only": only_c,
            "shared_functions": sorted(all_shared_fn),
            "shared_new_symbols": sorted(all_shared_sym),
            "per_file": per_file,
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
