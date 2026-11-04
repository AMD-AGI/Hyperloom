#!/usr/bin/env python3
"""Check a dossier, or one of its update rounds, before calling it done.

    python3 -m arbor.dossier.check <dir>                  # the dossier: README.md, parts 01-11
    python3 -m arbor.dossier.check <dir> --round u3       # round u3: 03, 06, 07, 10 and 11 .u3 files
    python3 -m arbor.dossier.check <dir> --round u3 --no-index   # a round staged outside the dossier

The dossier skill's ``check-dossier.py`` runs this module from the skill's place, for packs and callers that name it
(the dossier plugin's skills call the module).

<dir> is the dossier's directory (docs/dossier/<model>-<workload>-<gpu>/), or
for a round staged elsewhere (a campaign's <session>/reports/) that directory
with --no-index.

A file passes when it exists and holds real content: at least --min-words
words, and --min-numbers numbers, that are not headings, template labels,
<placeholders>, table rules or empty cells. A part whose status line says
"missing" fails, whatever its length. Kernel plans (NN.uK-plan-*.md) are
checked the same way when present. README.md must list the round. A round's
part 11 must end in a `## Map` block whose `DIRECTIONS:` line passes
arbor.directions (the line a campaign reads); a whole dossier's part 11 is
checked for it too, but not failed, since older dossiers predate it. A
round's kernel plan with an `ITEMS:` block (its `## Items` section) must pass
arbor.stages; a plan without one is noted, not failed, for the same reason.

Exit 0 when every required file passes, 1 otherwise. This checks that the
work was written down, not that it is right.
"""
import argparse
import glob
import json
import os
import re
import sys

from arbor.dossier.parts import BASE_PARTS, OPTIONAL_PARTS, ROUND_PARTS  # noqa: E402,F401 - the parts, in one place

PLACEHOLDER = re.compile(r"<[^<>]{3,300}>")  # may span lines, as the template's do
LABEL = re.compile(r"\*\*(Question|Status|Sources|Done when|Round \w+)\.?\*\*\.?", re.I)
NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?\s?(?:%|µs|us|ms|s|GiB|MiB|GB|TB/s|x|×)?")
WORD = re.compile(r"[A-Za-z][A-Za-z'-]+|\d+(?:\.\d+)?")


def real_content(text: str) -> tuple[int, int]:
    """(words, numbers) outside headings, labels, placeholders and table rules."""
    words = numbers = 0
    text = PLACEHOLDER.sub(" ", text)
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("```"):
            continue
        if re.fullmatch(r"\|?[\s:|-]*\|?", s):  # a table rule, or an empty row
            continue
        s = LABEL.sub(" ", s)
        s = re.sub(r"\|", " ", s)
        words += len(WORD.findall(s))
        numbers += len(NUMBER.findall(s))
    return words, numbers


def status_of(text: str) -> str:
    m = re.search(r"\*\*Status\.?\*\*\.?\s*([A-Za-z ]+?)[,.(\n]", text)
    return m.group(1).strip().lower() if m else ""


def check_file(path: str, min_words: int, min_numbers: int) -> dict:
    if not path or not os.path.isfile(path):
        return {"file": path, "ok": False, "why": "missing"}
    text = open(path, encoding="utf-8", errors="replace").read()
    words, numbers = real_content(text)
    status = status_of(text)
    why = []
    if words < min_words:
        why.append(f"{words} words of real content < {min_words}")
    if numbers < min_numbers:
        why.append(f"{numbers} numbers < {min_numbers}")
    if status.startswith("missing"):
        why.append("its status says missing")
    return {"file": path, "ok": not why, "words": words, "numbers": numbers,
            "status": status, "why": "; ".join(why)}


def find(dirpath: str, prefix: str, name: str) -> str:
    """The part's file: <prefix>-<name>.md if present, else the first other match."""
    named = os.path.join(dirpath, f"{prefix}-{name.replace(' ', '-')}.md")
    if os.path.isfile(named):
        return named
    hits = [h for h in sorted(glob.glob(os.path.join(dirpath, f"{prefix}-*.md")))
            if "-plan-" not in os.path.basename(h)]
    return hits[0] if hits else ""


def _arbor(name: str):
    """arbor.<name>, or None when it is not importable."""
    try:
        return __import__(f"arbor.{name}", fromlist=[name])
    except ImportError:
        return None


def check_items(path: str) -> dict:
    """A kernel plan's `ITEMS:` block, as a lane reads it (arbor/stages.py)."""
    row = {"part": "plan items (ITEMS:)", "file": path, "ok": True}
    stages = _arbor("stages")
    if stages is None:
        row["why"] = "arbor.stages is not importable here: not checked"
        return row
    text = open(path, encoding="utf-8", errors="replace").read()
    if not stages._blocks(text):
        row["why"] = "no ITEMS block (not required of a plan written before stage gates)"
        return row
    items, errs = stages.parse_items(text)
    if errs:
        row.update(ok=False, why="; ".join(errs[:3]))
    else:
        row["why"] = f"{len(items)} items, stages {' '.join(sorted({it['stage'] for it in items}))}"
    return row


def check_map(path: str, required: bool) -> dict:
    """Part 11's `## Map` line, as an Arbor campaign reads it (arbor/directions.py)."""
    row = {"part": "map (DIRECTIONS:)", "file": path or "11*.md", "ok": not required}
    if not path or not os.path.isfile(path):
        row["why"] = "no part 11"
        return row
    directions = _arbor("directions")
    if directions is None:
        row.update(ok=True, why="arbor.directions is not importable here: not checked")
        return row
    p = directions.parse_agent_lines(open(path, encoding="utf-8", errors="replace").read())
    if p.new_map is None:
        row["why"] = "; ".join(p.errors[:3]) or "no DIRECTIONS: line (the part's ## Map block)"
        if not required:
            row["why"] += " (not required of a dossier written before map rounds)"
    else:
        n_open = sum(1 for e in p.new_map if e["status"] == "open")
        row.update(ok=True, why=f"{len(p.new_map)} directions, {n_open} open")
    return row


RULES_HEADING = re.compile(r"^#{2,4}\s+The benchmark's rules\b", re.I | re.M)
TABLE_ROW = re.compile(r"^[ \t]*\|(?![ \t]*:?-{3})(.+)\|[ \t]*$", re.M)


def check_rules(path: str) -> dict:
    """Part 4's `## The benchmark's rules`: a table with a header and at least one rule under it."""
    row = {"part": "04 rules (the benchmark's)", "file": path or "04*.md", "ok": False}
    if not path or not os.path.isfile(path):
        row["why"] = "no part 4"
        return row
    text = open(path, encoding="utf-8", errors="replace").read()
    heading = RULES_HEADING.search(text)
    if not heading:
        row["why"] = "no `## The benchmark's rules` heading"
        return row
    section = re.split(r"^#{1,4}\s", text[heading.end():], maxsplit=1, flags=re.M)[0]
    rules = len(TABLE_ROW.findall(section)) - 1  # less the header row
    row.update(ok=rules >= 1, why=f"{rules} rules" if rules >= 1 else "the rules table lists no rule")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--round", help="an update round, e.g. u3")
    ap.add_argument("--no-index", action="store_true",
                    help="do not require README.md (a round staged outside the dossier)")
    ap.add_argument("--min-words", type=int, default=300)
    ap.add_argument("--min-numbers", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    rows = []
    if a.round:
        k = a.round if a.round.startswith("u") else f"u{a.round}"
        for nn, name in ROUND_PARTS.items():
            path = find(a.dir, f"{nn}.{k}", name)
            r = check_file(path, a.min_words, a.min_numbers)
            r["part"] = f"{nn}.{k} {name}"
            r["file"] = r["file"] or f"{nn}.{k}-*.md"
            rows.append(r)
        for path in sorted(glob.glob(os.path.join(a.dir, f"11.{k}-plan-*.md"))):
            r = check_file(path, a.min_words, a.min_numbers)
            r["part"] = f"11.{k} kernel plan"
            rows.append(r)
            rows.append(check_items(path))
        rows.append(check_map(find(a.dir, f"11.{k}", ROUND_PARTS["11"]), required=True))
        if not a.no_index:
            idx = os.path.join(a.dir, "README.md")
            listed = os.path.isfile(idx) and re.search(
                rf"\b{re.escape(k)}\b", open(idx, encoding="utf-8", errors="replace").read())
            rows.append({"part": "index", "file": idx, "ok": bool(listed),
                         "why": f"lists round {k}" if listed else f"README.md does not list round {k}"})
    else:
        for nn, name in {**BASE_PARTS, **OPTIONAL_PARTS}.items():
            path = find(a.dir, nn, name)
            if not path and nn in OPTIONAL_PARTS:
                continue
            r = check_file(path, a.min_words, a.min_numbers)
            r["part"] = f"{nn} {name}"
            r["file"] = r["file"] or f"{nn}-*.md"
            rows.append(r)
        r = check_file(os.path.join(a.dir, "README.md"), 100, 0)
        r["part"] = "index"
        rows.append(r)
        rows.append(check_rules(find(a.dir, "04", BASE_PARTS["04"])))
        rows.append(check_map(find(a.dir, "11", BASE_PARTS["11"]), required=False))

    ok = all(r["ok"] for r in rows)
    if a.json:
        print(json.dumps({"ok": ok, "dir": a.dir, "round": a.round, "files": rows}, indent=1))
    else:
        for r in rows:
            mark = "ok  " if r["ok"] else "FAIL"
            detail = r.get("why") if not r["ok"] or "words" not in r else None
            detail = detail or (f"{r.get('words', '')} words, {r.get('numbers', '')} numbers"
                                      + (f", status {r['status']}" if r.get("status") else ""))
            print(f"{mark}  {r['part']:<22} {os.path.basename(r['file']):<34} {detail}")
        print("PASS" if ok else "FAIL: write or finish the files marked FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
