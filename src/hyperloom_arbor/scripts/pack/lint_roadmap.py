"""Lint a pack's roadmap.yaml: unique ids, known sections, graded evidence, known gates.

    python3 scripts/pack/lint_roadmap.py packs/<pack>/roadmap.yaml

`problems()` is the check itself, so the review gate (arbor/review.py) runs the same one
before a human approves a campaign.
"""
import collections
import sys

import yaml


def problems(d) -> list[str]:
    """What is wrong with the roadmap ``d``; [] when nothing is."""
    if not isinstance(d, dict) or not isinstance(d.get("items"), list) or "sections" not in d:
        return ["not a roadmap: it needs `sections` and a list of `items`"]
    found = []
    ids = [i.get("id") for i in d["items"]]
    if len(ids) != len(set(ids)):
        found.append("duplicate ids")
    for it in d["items"]:
        if it.get("section") not in d["sections"]:
            found.append(f"{it.get('id')}: unknown section")
        for e in it.get("evidence", []):
            if e.get("grade") == "measured" and not e.get("instrument"):
                found.append(f"{it.get('id')}: measured without instrument")
            if e.get("grade") == "reported" and not e.get("system"):
                found.append(f"{it.get('id')}: reported without naming its system")
        for dep in it.get("gates", []):
            if dep not in ids:
                found.append(f"{it.get('id')}: gates unknown item {dep}")
    return found


def main(path: str) -> int:
    d = yaml.safe_load(open(path))
    found = problems(d)
    for p in found:
        print(f"ERROR: {p}")
    if found:
        return 1
    print(f"{len(d['items'])} items, sections {list(d['sections'])}")
    g = collections.Counter(e["grade"] for it in d["items"] for e in it.get("evidence", []))
    print("evidence grades:", dict(g))
    for it in d["items"]:
        grades = {e["grade"] for e in it.get("evidence", [])}
        flag = "" if grades & {"measured", "inferred"} else "   <-- LEAD ONLY"
        print(f"  {it['id']:3s} [{','.join(sorted(grades)):25s}] {it['title'][:50]}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
