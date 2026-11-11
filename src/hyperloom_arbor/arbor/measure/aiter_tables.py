"""An AITER tuned table a change serves must have AITER's own columns.

A change can point an ``AITER_CONFIG_*`` variable at a tuned CSV of its own (a retuned MoE or GEMM table). AITER reads
it by the header of its shipped table for that variable, and a host may refuse it outright. A table
without one of AITER's columns (``_tag`` in the fused-MoE table) is refused on import. So the harness checks the
header against the installed AITER's before any GPU time, and refuses there, where the agent can still fix it.

The shipped tables, per variable, are KernelForge's ``config_files_and_merge.md``. Columns AITER fills itself when a
table lacks them (``gfx`` from ``cu_num``; ``xbf16``, ``run_1stage``, ``ksplit`` as 0) are not required.

Only the tables a change brings are checked. AITER's own per-model tables (``configs/model_configs/``) lack ``_tag``
too, and a change that keeps them in its ``AITER_CONFIG_*`` list, beside its own, was refused for them. A table is
AITER's own when it lies in AITER's package and is unchanged: its digest is the one in the installed wheel's
``RECORD``, or, for a git checkout of AITER, it is tracked and unchanged since the change's base. Anything else is the
change's, and checked.

The base of a git checkout is the campaign's baseline of that tree (``<session>/trees.json``, arbor/execute/trees.py),
not its HEAD: the harness commits AITER's tree with every iteration, so a table the iteration edited is tracked and
clean at HEAD by the time it is checked, and was skipped. A tree without a baseline (no editable package, or under a
host) is compared with its HEAD, which no iteration commits.
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
import base64, hashlib, json, os, subprocess, sys
asked = json.loads(sys.argv[1])
bases = {os.path.realpath(k): v for k, v in (json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}).items()}
try:
    import aiter
    pkg = os.path.realpath(os.path.dirname(aiter.__file__))
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
record = {}
try:
    from importlib import metadata
    for dist in metadata.distributions():
        files = list(dist.files or [])
        if not any(str(f).replace(os.sep, "/") == "aiter/__init__.py" for f in files):
            continue
        for f in files:
            if f.hash is not None and f.hash.mode == "sha256":
                record[os.path.realpath(str(f.locate()))] = f.hash.value
        break
except Exception:
    pass
def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, timeout=30)
def stock(path):
    real = os.path.realpath(path)
    if not real.startswith(pkg + os.sep):
        return False
    try:
        if real in record:
            with open(real, "rb") as f:
                digest = base64.urlsafe_b64encode(hashlib.sha256(f.read()).digest()).rstrip(b"=").decode()
            return digest == record[real]
        top = git("-C", os.path.dirname(real), "rev-parse", "--show-toplevel")
        if top.returncode != 0:
            return False
        top = top.stdout.strip()
        base = bases.get(os.path.realpath(top)) or "HEAD"
        rel = os.path.relpath(real, os.path.realpath(top)).replace(os.sep, "/")
        return (git("-C", top, "cat-file", "-e", f"{base}:{rel}").returncode == 0
                and git("-C", top, "diff", "--quiet", base, "--", real).returncode == 0)
    except Exception:
        return False
out = {"shipped": {}, "tables": {}, "stock": []}
for var, (shipped, paths) in asked.items():
    out["shipped"][var] = next((h for h in (header(os.path.join(r, shipped)) for r in roots) if h), None)
    out["tables"][var] = {p: header(p) for p in paths}
    out["stock"] += [p for p in paths if stock(p)]
print(json.dumps(out))
"""


def problems(container: str, server_env: dict, workdir: str | None = None, bases: dict | None = None) -> list[str]:
    """What is wrong with the tuned tables ``server_env`` points AITER at, read where the server runs: each table
    that lacks a column of AITER's shipped table for its variable, or cannot be read. ``[]`` when nothing is wrong,
    or nothing can be checked (no override, or no AITER to compare with). ``bases``: each git tree's baseline commit
    (``trees.baselines``), what its tables are compared with."""
    asked = {var: (SHIPPED[var], [p for p in str(value).split(":") if p.strip()])
             for var, value in (server_env or {}).items() if var in SHIPPED and str(value).strip()}
    if not asked:
        return []
    try:
        r = run_in(container, ["python3", "-c", _PROBE, json.dumps(asked), json.dumps(bases or {})], workdir=workdir,
                   capture_output=True, text=True, timeout=120)
        found = json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception:
        return []
    if "error" in found:
        return []
    return compare(found.get("shipped") or {}, found.get("tables") or {}, stock=found.get("stock") or ())


def compare(shipped: dict, tables: dict, stock=()) -> list[str]:
    """The problems, from the shipped headers and each table's header (None when a file did not read). ``stock``:
    the paths that are AITER's own, unchanged, which the change did not bring and which are not checked."""
    out = []
    stock = set(stock)
    for var, by_path in tables.items():
        want = shipped.get(var)
        for path, have in by_path.items():
            if path in stock:
                continue
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
