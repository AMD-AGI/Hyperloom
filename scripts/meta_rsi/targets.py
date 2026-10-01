# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write the census target lists from the Pulse index: every session name, and the recent ones.

Pulse's last_days filter keys on the last observation, so re-observed old sessions come back too;
"recent" means started on or after PULSE_RECENT_SINCE.
"""

from __future__ import annotations

import json

from round_env import recent_since, round_dir


def main() -> None:
    root, since = round_dir(), recent_since()
    names, seen, recent = [], set(), set()
    rows = [json.loads(line) for line in open(root / "00_enum_global/index_rows.jsonl")]
    for r in sorted(rows, key=lambda r: r.get("session_started_at") or "", reverse=True):
        name = r.get("session_id")
        if name and name not in seen:
            seen.add(name)
            names.append(name)
            if (r.get("session_started_at") or "")[:10] >= since:
                recent.add(name)
    (root / "targets_all.txt").write_text("\n".join(names) + "\n")
    (root / "targets_recent.txt").write_text("\n".join(n for n in names if n in recent) + "\n")
    print(f"{len(names)} names, {len(recent)} started on or after {since}")


if __name__ == "__main__":
    main()
