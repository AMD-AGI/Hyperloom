"""An AITER tuned table a change serves must have AITER's own columns.

A change can point an ``AITER_CONFIG_*`` variable at a tuned CSV of its own (a retuned MoE or GEMM table). AITER reads
it by the header of its shipped table for that variable, and a host may refuse it outright. Under MB, Arbor kept
several changes whose ``AITER_CONFIG_FMOE`` table lacked the ``_tag`` column, and MB refused to import every one of
them. So the harness checks the header against the installed AITER's before any GPU time, and refuses there, where the
agent can still fix it.

The shipped tables, per variable, are KernelForge's ``config_files_and_merge.md``. Columns AITER fills itself when a
table lacks them (``gfx`` from ``cu_num``; ``xbf16``, ``run_1stage``, ``ksplit`` as 0) are not required.
"""

from __future__ import annotations

import json

from arbor.execute.workspace import run_in

#: Each override variable, and the tuned table AITER ships for it under ``<aiter>/configs/``.
SHIPPED = {
    "AITER_CONFIG_GEMM_BF16": "bf16_tuned_gemm.csv",
    "AITER_CONFIG_BF16_BATCHED_GEMM": "bf16_tuned_batched_gemm.csv",
    "AITER_CONFIG_GEMM_A8W8": "a8w8_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE": "a8w8_bpreshuffle_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "a8w8_blockscale_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
    "AITER_CONFIG_A8W8_BATCHED_GEMM": "a8w8_tuned_batched_gemm.csv",
    "AITER_CONFIG_GEMM_A4W4": "a4w4_blockscale_tuned_gemm.csv",
    "AITER_CONFIG_FMOE": "tuned_fmoe.csv",
    "AITER_CONFIG_GROUPED_FMOE": "tuned_grouped_fmoe.csv",
}
#: Columns AITER's merge fills when a table lacks them (``update_config_files``).
FILLED = frozenset({"gfx", "xbf16", "run_1stage", "ksplit"})

# Run where the server runs: the shipped headers, and each table's header, as JSON.
_PROBE = r"""
import json, os, sys
asked = json.loads(sys.argv[1])
try:
    import aiter
    roots = [os.path.join(os.path.dirname(aiter.__file__), "configs"),
             os.path.join(os.path.dirname(os.path.dirname(aiter.__file__)), "aiter", "configs")]
except Exception as exc:
    print(json.dumps({"error": f"aiter is not importable here: {exc}"})); sys.exit(0)
def header(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return [c.strip() for c in f.readline().strip().split(",")]
    except OSError:
        return None
out = {"shipped": {}, "tables": {}}
for var, (shipped, paths) in asked.items():
    out["shipped"][var] = next((h for h in (header(os.path.join(r, shipped)) for r in roots) if h), None)
    out["tables"][var] = {p: header(p) for p in paths}
print(json.dumps(out))
"""


def problems(container: str, server_env: dict, workdir: str | None = None) -> list[str]:
    """What is wrong with the tuned tables ``server_env`` points AITER at, read where the server runs: each table
    that lacks a column of AITER's shipped table for its variable, or cannot be read. ``[]`` when nothing is wrong,
    or nothing can be checked (no override, or no AITER to compare with)."""
    asked = {var: (SHIPPED[var], [p for p in str(value).split(":") if p.strip()])
             for var, value in (server_env or {}).items() if var in SHIPPED and str(value).strip()}
    if not asked:
        return []
    try:
        r = run_in(container, ["python3", "-c", _PROBE, json.dumps(asked)], workdir=workdir, capture_output=True, text=True,
                   timeout=120)
        found = json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception:
        return []
    if "error" in found:
        return []
    return compare(found.get("shipped") or {}, found.get("tables") or {})


def compare(shipped: dict, tables: dict) -> list[str]:
    """The problems, from the shipped headers and each table's header (None when a file did not read)."""
    out = []
    for var, by_path in tables.items():
        want = shipped.get(var)
        for path, have in by_path.items():
            if have is None:
                out.append(f"{var}: {path} cannot be read")
                continue
            if not want:
                continue
            missing = [c for c in want if c and c not in have and c not in FILLED]
            if missing:
                out.append(f"{var}: {path} lacks AITER's columns {', '.join(missing)} "
                           f"(the shipped {SHIPPED[var]} has them)")
    return out
