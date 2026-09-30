"""Write the census target lists from the Pulse index: every session name, and the recent ones.

Pulse's last_days filter keys on the last observation, so re-observed old sessions come back too;
"recent" means started on or after PULSE_RECENT_SINCE.
"""

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("PULSE_ROUND_DIR", "/wekafs/csl/Hyperloom-Sessions/meta_rsi/pulse15d"))
RECENT = os.environ.get("PULSE_RECENT_SINCE", "2026-09-15")


def main() -> None:
    names, seen, recent = [], set(), set()
    rows = [json.loads(line) for line in open(ROOT / "00_enum_global/index_rows.jsonl")]
    for r in sorted(rows, key=lambda r: r.get("session_started_at") or "", reverse=True):
        name = r.get("session_id")
        if name and name not in seen:
            seen.add(name)
            names.append(name)
            if (r.get("session_started_at") or "")[:10] >= RECENT:
                recent.add(name)
    (ROOT / "targets_all.txt").write_text("\n".join(names) + "\n")
    (ROOT / "targets_recent.txt").write_text("\n".join(n for n in names if n in recent) + "\n")
    print(f"{len(names)} names, {len(recent)} started on or after {RECENT}")


if __name__ == "__main__":
    main()
