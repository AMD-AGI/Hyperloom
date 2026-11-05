# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which kernels a session recompiled over and over, read off the JIT cache it filled.

A triton kernel is compiled once per distinct set of specialising constants, and every first compile stalls the
whole engine for a few hundred milliseconds. A constant that tracks a runtime quantity rather than a tuning choice
turns that into one compile per value: a GLM-5.3 AgentX run compiled 238 variants of one aiter bmm/rope kernel
because its grid constant moved with the prefill token count, and 59 of 227 variants in that run compiled after the
server reported ready, 23 of them inside the scored window. TTFT p90 measured 0.97 s cold against 0.55 s warm, and
two real optimizations were rejected twice on cold numbers before the cause was found.

Many variants is not itself the finding -- a tile-tuned GEMM legitimately carries dozens. What separates the two is
where the variants differ: a tuning table spreads across several constants, while a runtime quantity leaks into one
and takes a new value per call shape.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Variants below this are the ordinary cost of a tuned kernel and are not worth a reader's attention.
MIN_VARIANTS: int = 16

#: Share of a kernel's variants one constant must account for before the spread is called a leak rather than a
#: tuning table. A table spreads across several constants, so no single one reaches this.
LEAK_VALUE_SHARE: float = 0.5

#: Kernels named in a report, worst first.
TOP_N: int = 10


def _variant_name(variant_dir: Path) -> tuple[str, str]:
    """``(base kernel, specialising suffix)`` for one compiled variant, or ``("", "")`` when unreadable."""
    for meta in sorted(variant_dir.glob("*.json")):
        if meta.name.startswith("__grp__"):
            continue
        try:
            name = str((json.loads(meta.read_text(encoding="utf-8")) or {}).get("name") or "")
        except (OSError, ValueError) as exc:
            log.debug("jit_variants: %s unreadable: %s", meta, exc)
            continue
        base = meta.stem
        if name.startswith(base):
            return base, name[len(base) :].strip("_")
        return base, ""
    return "", ""


def _varying_positions(suffixes: list[str]) -> list[dict[str, Any]]:
    """Underscore-separated positions whose value differs across variants, widest spread first.

    Every variant of one kernel carries the same constants in the same order, so only their values move: a
    positional comparison finds them without having to split names like ``BLOCK_SIZE_N_256`` into key and value.
    Skipped when the suffixes do not share a shape, which means they are not one kernel's constants after all.
    """
    tokens = [s.split("_") for s in suffixes if s]
    if len(tokens) < 2 or len({len(t) for t in tokens}) != 1:
        return []
    out: list[dict[str, Any]] = []
    for index in range(len(tokens[0])):
        values = {t[index] for t in tokens}
        if len(values) < 2:
            continue
        label = "_".join(tokens[0][max(0, index - 2) : index]) or "?"
        out.append({"after": label, "distinct_values": len(values), "examples": sorted(values)[:4]})
    out.sort(key=lambda row: row["distinct_values"], reverse=True)
    return out


def variant_counts(cache_root: str | Path | None) -> list[dict[str, Any]]:
    """Per base kernel, how many variants a session compiled and where they differ, worst first.

    A cache this process cannot stat counts as a cache with nothing to report.
    The callers render a hand-off document and a gap set from several censuses
    at once, and a session whose cache a privileged earlier run left unreadable
    would otherwise lose the launch, graph and front-end findings along with
    this one.
    """
    root = Path(str(cache_root or "")).expanduser()
    if not str(cache_root or "").strip():
        return []
    try:
        if not root.is_dir():
            return []
    except OSError:
        return []
    by_kernel: dict[str, list[str]] = {}
    for meta in root.glob("**/*.json"):
        if meta.name.startswith("__grp__") or not meta.parent.is_dir():
            continue
        base, suffix = _variant_name(meta.parent)
        if base:
            by_kernel.setdefault(base, []).append(suffix)
    rows: list[dict[str, Any]] = []
    for base, suffixes in by_kernel.items():
        count = len(suffixes)
        if count < MIN_VARIANTS:
            continue
        positions = _varying_positions(sorted(set(suffixes)))
        widest = positions[0]["distinct_values"] if positions else 0
        rows.append(
            {
                "kernel": base,
                "variants": count,
                "varying": positions[:3],
                # One constant carrying most of the spread is a runtime quantity reaching a constexpr, not a table.
                "single_constant_leak": widest >= count * LEAK_VALUE_SHARE,
            }
        )
    rows.sort(key=lambda row: row["variants"], reverse=True)
    return rows[:TOP_N]


def render_jit_variants(rows: list[dict[str, Any]] | None) -> list[str]:
    """Render the recompilation census as report lines, or nothing when no kernel crossed the floor."""
    if not rows:
        return []
    lines = [
        "| kernel | variants | widest-spread constant | reading |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        widest = (row.get("varying") or [{}])[0]
        after = str(widest.get("after") or "?")
        distinct = widest.get("distinct_values")
        spread = f"after `…_{after}_`: {distinct} values" if distinct else "spread across several"
        reading = (
            "one constant tracks a runtime quantity; each call shape pays a compile"
            if row.get("single_constant_leak")
            else "spread across constants, consistent with a tuning table"
        )
        lines.append(f"| `{row['kernel']}` | {row['variants']} | {spread} | {reading} |")
    return lines


__all__ = [
    "LEAK_VALUE_SHARE",
    "MIN_VARIANTS",
    "TOP_N",
    "render_jit_variants",
    "variant_counts",
]
