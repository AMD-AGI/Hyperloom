#!/usr/bin/env python3
"""Score a parallel round's dossier drafts: the parts check-dossier.py requires, and the map.

    python3 score_dossier.py <workspace> [--round uK] [--json OUT]

The round's parts are reports/03, 06, 07, 10 and 11 .uK (the dossier skill's
update round). Each earns 10 points when it passes check-dossier.py's content
check (300 words and 10 numbers of real content), and a share of 9 below that;
part 11's `## Map` line earns 10 more when arbor.directions reads it. A round
kernel plan whose ITEMS block does not parse costs 5. The score is a floor on
form, floored at 1; the critic and the map's checks read for substance.

The round comes from --round, else the workspace's .git/PAR_ROUND, which
scripts/parallel/dossier_round.sh writes. check-dossier.py is the skill's own,
from the checkout this pack sits in. This is the parallel-dossier pack's
protected bench (validator/).
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

CHECK = Path(__file__).resolve().parents[3] / ".claude" / "skills" / "dossier" / "check-dossier.py"


def _checker():
    spec = importlib.util.spec_from_file_location("check_dossier", CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score(ws: Path, k: str) -> dict:
    cd = _checker()
    reports = str(ws / "reports")
    parts, total = {}, 0
    for nn, name in cd.ROUND_PARTS.items():
        row = cd.check_file(cd.find(reports, f"{nn}.{k}", name), 300, 10)
        pts = 10 if row["ok"] else int(9 * min(1.0, min(row.get("words", 0) / 300, row.get("numbers", 0) / 10)))
        parts[f"{nn}.{k}"] = {"points": pts, "why": row.get("why", "")}
        total += pts
    m = cd.check_map(cd.find(reports, f"11.{k}", cd.ROUND_PARTS["11"]), required=True)
    parts["map"] = {"points": 10 if m["ok"] else 0, "why": m.get("why", "")}
    total += parts["map"]["points"]
    for plan in sorted(glob.glob(os.path.join(reports, f"11.{k}-plan-*.md"))):
        row = cd.check_items(plan)
        if not row["ok"]:
            parts[os.path.basename(plan)] = {"points": -5, "why": row.get("why", "")}
            total -= 5
    return {"dossier_score": max(1, total), "round": k, "parts": parts}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("workspace")
    ap.add_argument("--round")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    ws = Path(a.workspace)
    k = a.round or ((ws / ".git" / "PAR_ROUND").read_text().strip() if (ws / ".git" / "PAR_ROUND").exists() else "")
    if not k:
        print("no round: give --round uK, or run in a workspace dossier_round.sh built", file=sys.stderr)
        return 2
    k = k if k.startswith("u") else f"u{k}"
    out = score(ws, k)
    for part, row in out["parts"].items():
        print(f"{row['points']:>3}  {part}  {row['why']}")
    print(f"dossier_score {out['dossier_score']}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
