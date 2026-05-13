"""Pick the next 60 models for Friday's batch.

Source of truth for "already done": luochen dashboard /api/tasks (covers
every record we've ever published, including today's 4 new). We subtract
that from the top600 candidates list, preserving the candidates' download-rank
order, and take the first 60.
"""
import json
import sys
from urllib.request import urlopen

LUOCHEN_TASKS = "http://core42.example-internal-host.invalid/hyperloom-results/api/tasks?limit=500&sort_by=updated_at&order=desc"
TOP600 = "ci/candidates/top600_2026-05-12.json"


def norm(s):
    return (s or "").lower().strip()


def main():
    # 1. luochen done list
    with urlopen(LUOCHEN_TASKS, timeout=30) as r:
        d = json.load(r)
    rows = d.get("tasks") or d.get("items") or d.get("results") or []
    done_models = set()
    for row in rows:
        m = norm(row.get("model"))
        if not m:
            # Some manual records have only display_name — best-effort map
            dn = norm(row.get("display_name"))
            if dn:
                done_models.add(dn)
            continue
        done_models.add(m)
    print(f"luochen has {len(rows)} task rows, {len(done_models)} unique models")

    # 2. top600 candidates (preserve order)
    with open(TOP600, encoding="utf-8") as f:
        cands = json.load(f).get("candidates", [])
    print(f"top600 candidates pool: {len(cands)} models")

    # 3. Subtract
    pending = []
    skipped = 0
    for c in cands:
        repo = c["repo_id"]
        if norm(repo) in done_models:
            skipped += 1
            continue
        pending.append(c)
    print(f"after skip: {len(pending)} pending ({skipped} already done)\n")

    # 4. First 60
    sel = pending[:60]
    print("Picked 60 models in candidate order:\n")
    for i, c in enumerate(sel, 1):
        print(f"  {i:3d}. {c['repo_id']:55s} "
              f"{c.get('arch',''):25s} {c.get('precision',''):6s} "
              f"params={c.get('params_b','?'):>6} GB={c.get('weight_gb','?')}")

    # 5. Two splits: 40 hyperloom + 20 sandbox
    HYPERLOOM = 40
    SANDBOX = 20
    hyperloom_ids = [c["repo_id"] for c in sel[:HYPERLOOM]]
    sandbox_ids = [c["repo_id"] for c in sel[HYPERLOOM:HYPERLOOM+SANDBOX]]
    print(f"\n=== Dispatch A ({len(hyperloom_ids)} models -> core42-hyperloom) ===")
    print("models='" + " ".join(hyperloom_ids) + "'")
    print(f"\n=== Dispatch B ({len(sandbox_ids)} models -> core42-sandbox) ===")
    print("models='" + " ".join(sandbox_ids) + "'")


if __name__ == "__main__":
    sys.exit(main() or 0)
