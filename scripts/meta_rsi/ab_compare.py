#!/usr/bin/env python3
"""Side-by-side comparison of sessions that ran the same workload with different harness sets.

    python scripts/meta_rsi/ab_compare.py A=SESSION_DIR B=SESSION_DIR [...] [--out compare.md]

The first arm is the reference; every other column also shows its change against it.
Scores come from each session's journal (the measurement path), never from this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from . import _paths
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _paths  # noqa: F401

from hyperloom.inference_optimizer.trace.meta_rsi_ledger import LEDGER_FILENAME, build_ledger


def load_ledger(session_dir: Path) -> dict[str, Any]:
    """Rebuild from the traces so a ledger written mid-run is never compared stale."""
    ledger = build_ledger(session_dir)
    saved = session_dir / "reports" / LEDGER_FILENAME
    if not ledger["tokens"]["total"]["calls"] and saved.exists():
        return json.loads(saved.read_text())
    return ledger


def _delta(value: float | None, ref: float | None) -> str:
    if value is None or ref in (None, 0):
        return ""
    return f" ({100 * (value - ref) / ref:+.0f}%)"


def _fmt_m(v: float | None) -> str:
    return "-" if v is None else f"{v / 1e6:.2f}M"


def compare(arms: list[tuple[str, dict[str, Any]]]) -> str:
    ref = arms[0][1]
    lines = []
    add = lines.append
    add("| metric | " + " | ".join(label for label, _ in arms) + " |")
    add("|---|" + "---:|" * len(arms))

    def row(name: str, get, fmt=_fmt_m, with_delta: bool = True) -> None:
        cells = []
        rv = get(ref)
        for _, led in arms:
            v = get(led)
            cell = fmt(v)
            if with_delta and led is not ref:
                cell += _delta(v, rv)
            cells.append(cell)
        add(f"| {name} | " + " | ".join(cells) + " |")

    def fp(led: dict[str, Any], *path: str) -> Any:
        cur: Any = led.get("harness_fingerprint") or {}
        for p in path:
            cur = cur.get(p) if isinstance(cur, dict) else None
        return cur

    row("session", lambda l: Path(l["session_dir"]).name, fmt=str, with_delta=False)
    row(
        "code revision",
        lambda l: f"{fp(l, 'code', 'revision') or l.get('code_revision')}{'+dirty' if fp(l, 'code', 'dirty') else ''}",
        fmt=str,
        with_delta=False,
    )
    row("prompt text digest", lambda l: fp(l, "text", "digest") or "-", fmt=str, with_delta=False)
    row("stop reason", lambda l: l.get("stop_reason") or "-", fmt=str, with_delta=False)
    row("LLM span (min)", lambda l: l.get("llm_span_min"), fmt=lambda v: "-" if v is None else f"{v:.0f}")
    row("LLM calls", lambda l: l["tokens"]["total"]["calls"], fmt=lambda v: str(v))
    row("billed tokens", lambda l: l["tokens"]["total"]["billed"])
    row("cost-weighted tokens", lambda l: l["tokens"]["total"]["weighted"])
    comps = sorted({c for _, l in arms for c in l["tokens"]["by_component"]})
    for c in comps:
        row(f"  weighted: {c}", lambda l, c=c: (l["tokens"]["by_component"].get(c) or {}).get("weighted"))
    models = sorted({m for _, l in arms for m in l["tokens"]["by_model"]})
    for m in models:
        row(f"  billed on {m}", lambda l, m=m: (l["tokens"]["by_model"].get(m) or {}).get("billed"))
    row(
        "baseline throughput",
        lambda l: l["outcome"].get("baseline_throughput"),
        fmt=lambda v: "-" if v is None else f"{v:.1f}",
        with_delta=False,
    )
    row(
        "final throughput",
        lambda l: l["outcome"].get("final_throughput"),
        fmt=lambda v: "-" if v is None else f"{v:.1f}",
    )
    row(
        "total gain %",
        lambda l: l["outcome"].get("total_gain_pct"),
        fmt=lambda v: "-" if v is None else f"{v:.1f}",
        with_delta=False,
    )
    row(
        "config KEEP / REVERT",
        lambda l: f"{l['outcome'].get('config_keep', 0)} / {l['outcome'].get('config_revert', 0)}",
        fmt=str,
        with_delta=False,
    )
    row("weighted tokens per config KEEP", lambda l: l["efficiency"].get("weighted_per_config_keep"))
    row("billed tokens per gain %", lambda l: l["efficiency"].get("billed_per_gain_pct"))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("arms", nargs="+", help="LABEL=SESSION_DIR, reference arm first")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    arms = []
    for spec in args.arms:
        label, _, path = spec.partition("=")
        if not path:
            ap.error(f"expected LABEL=SESSION_DIR, got {spec!r}")
        arms.append((label, load_ledger(Path(path))))
    text = compare(arms)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
