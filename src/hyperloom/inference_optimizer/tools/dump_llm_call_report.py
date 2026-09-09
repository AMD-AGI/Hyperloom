#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render the per-LLM-call report for a session as a phase -> task tree.

Reads the two ledgers written during a run -- ``llm_calls.jsonl`` (one row per
agentic turn) and its ``llm_calls_detail.jsonl`` sidecar (one row per real API
call) -- plus every out-of-process ``ext/`` shard, including the GEAK spend the
harvester recovers. Rows are placed in a tree by ``phase`` then by the segments
of ``task_path``, and every node reports the roll-up of the calls beneath it:
call counts, ISL/OSL, tokens (input / thinking / output / cache), wall-clock
(total / thinking / output), USD (total / thinking / output) and tool calls,
each against its share of its parent.

Usage::

    python -m hyperloom.inference_optimizer.tools.dump_llm_call_report \\
        --session-dir /shared/hyperloom-sessions/<user>/<sid>

Writes ``llm_call_report.md`` and ``llm_call_report.json`` to
``$USER_DATA_PATH/reports/<session-id>/`` unless ``--output-dir`` says
otherwise.

Two properties matter more than the totals themselves:

* A turn with no detail rows is still counted, once, from the turn row. A turn
  that has detail rows is counted **only** from them, so the same spend is
  never added twice.
* An unpriced call is left out of the USD total rather than counted as free,
  and the coverage section says how many such calls there were. A partial total
  presented as a total is worse than no total.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

log = logging.getLogger("dump_llm_call_report")

#: Separator used inside the ledgers' flat ``task_path`` string.
PATH_SEPARATOR = "/"

#: ``cost_source`` value meaning the call could not be priced at all.
COST_UNAVAILABLE = "unavailable"

#: ``timing_source`` value meaning the thinking/output split was estimated from
#: the token ratio rather than measured off stream events.
TIMING_APPORTIONED = "apportioned"

#: Node label for rows that carry no ``phase``.
UNPHASED = "(no phase)"

#: Node label for rows that carry no ``task_path``.
UNPATHED = "(no task path)"

_TOKEN_FIELDS = (
    ("input_tokens", "tokens_in"),
    ("output_tokens", "tokens_out"),
    ("reasoning_output_tokens", "tokens_thinking"),
    ("cache_creation_input_tokens", "tokens_cache_write"),
    ("cache_read_input_tokens", "tokens_cache_read"),
)

_TIME_FIELDS = (
    ("latency_ms", "ms_total"),
    ("thinking_ms", "ms_thinking"),
    ("output_ms", "ms_output"),
)

_COST_FIELDS = (
    ("cost_usd", "usd_total"),
    ("cost_input_usd", "usd_input"),
    ("cost_output_usd", "usd_output"),
    ("cost_thinking_usd", "usd_thinking"),
    ("cost_cache_usd", "usd_cache"),
)


def _num(value: Any) -> float:
    """Read a numeric ledger field, treating a missing one as zero.

    Args:
        value: Ledger field value, possibly ``None`` or of a foreign type.

    Returns:
        The value as a float, or ``0.0``.
    """
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


@dataclass
class Totals:
    """Roll-up of every API call beneath one node of the report tree."""

    calls: int = 0
    turns: int = 0
    tool_calls: int = 0
    isl: int = 0
    osl: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_thinking: int = 0
    tokens_cache_write: int = 0
    tokens_cache_read: int = 0
    ms_total: float = 0.0
    ms_thinking: float = 0.0
    ms_output: float = 0.0
    usd_total: float = 0.0
    usd_input: float = 0.0
    usd_output: float = 0.0
    usd_thinking: float = 0.0
    usd_cache: float = 0.0
    calls_priced: int = 0
    calls_timed: int = 0
    calls_apportioned: int = 0
    errors: int = 0
    models: dict[str, int] = field(default_factory=dict)
    cost_sources: dict[str, int] = field(default_factory=dict)

    def add_call(self, row: dict[str, Any]) -> None:
        """Fold one API call (or an undetailed turn) into this roll-up.

        Args:
            row: A ledger row from either the detail sidecar or the turn ledger.
        """
        self.calls += 1
        for src, dst in _TOKEN_FIELDS:
            setattr(self, dst, getattr(self, dst) + int(_num(row.get(src))))
        isl = row.get("isl")
        osl = row.get("osl")
        # Turn rows carry no ISL/OSL: they are derived here on the same rule
        # the detail writer uses, so a mixed tree stays comparable.
        self.isl += (
            int(_num(isl))
            if isl is not None
            else (
                int(_num(row.get("input_tokens")))
                + int(_num(row.get("cache_read_input_tokens")))
                + int(_num(row.get("cache_creation_input_tokens")))
            )
        )
        self.osl += (
            int(_num(osl))
            if osl is not None
            else (int(_num(row.get("output_tokens"))) + int(_num(row.get("reasoning_output_tokens"))))
        )
        for src, dst in _TIME_FIELDS:
            setattr(self, dst, getattr(self, dst) + _num(row.get(src)))
        if row.get("latency_ms") is not None:
            self.calls_timed += 1
        if str(row.get("timing_source") or "").strip().lower() == TIMING_APPORTIONED:
            self.calls_apportioned += 1
        source = str(row.get("cost_source") or "").strip().lower()
        if source and source != COST_UNAVAILABLE:
            for src, dst in _COST_FIELDS:
                setattr(self, dst, getattr(self, dst) + _num(row.get(src)))
            self.calls_priced += 1
            self.cost_sources[source] = self.cost_sources.get(source, 0) + 1
        tools = row.get("tool_calls")
        if isinstance(tools, list):
            self.tool_calls += len(tools)
        else:
            self.tool_calls += int(_num(row.get("tool_call_count")))
        if str(row.get("status") or "ok").strip().lower() not in ("", "ok"):
            self.errors += 1
        model = str(row.get("model") or "").strip()
        if model:
            self.models[model] = self.models.get(model, 0) + 1

    def merge(self, other: "Totals") -> None:
        """Add a child's roll-up into this one.

        Args:
            other: The child roll-up to absorb.
        """
        for name in (
            "calls",
            "turns",
            "tool_calls",
            "isl",
            "osl",
            "tokens_in",
            "tokens_out",
            "tokens_thinking",
            "tokens_cache_write",
            "tokens_cache_read",
            "ms_total",
            "ms_thinking",
            "ms_output",
            "usd_total",
            "usd_input",
            "usd_output",
            "usd_thinking",
            "usd_cache",
            "calls_priced",
            "calls_timed",
            "calls_apportioned",
            "errors",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for src, dst in ((other.models, self.models), (other.cost_sources, self.cost_sources)):
            for k, v in src.items():
                dst[k] = dst.get(k, 0) + v

    def as_dict(self) -> dict[str, Any]:
        """Project to a JSON-safe dict with the derived averages included.

        Returns:
            The roll-up, with USD rounded to cents-of-a-cent and per-call means.
        """
        out: dict[str, Any] = {
            "calls": self.calls,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "isl": self.isl,
            "osl": self.osl,
            "tokens_in": self.tokens_in,
            "tokens_thinking": self.tokens_thinking,
            "tokens_out": self.tokens_out,
            "tokens_cache_write": self.tokens_cache_write,
            "tokens_cache_read": self.tokens_cache_read,
            "tokens_total": self.isl + self.osl,
            "ms_total": round(self.ms_total, 1),
            "ms_thinking": round(self.ms_thinking, 1),
            "ms_output": round(self.ms_output, 1),
            "usd_total": round(self.usd_total, 6),
            "usd_input": round(self.usd_input, 6),
            "usd_output": round(self.usd_output, 6),
            "usd_thinking": round(self.usd_thinking, 6),
            "usd_cache": round(self.usd_cache, 6),
            "calls_priced": self.calls_priced,
            "calls_timed": self.calls_timed,
            "calls_apportioned": self.calls_apportioned,
            "errors": self.errors,
            "cost_sources": dict(sorted(self.cost_sources.items())),
            "models": dict(sorted(self.models.items(), key=lambda kv: (-kv[1], kv[0]))),
        }
        if self.calls:
            out["mean_isl"] = round(self.isl / self.calls, 1)
            out["mean_osl"] = round(self.osl / self.calls, 1)
        if self.calls_timed:
            out["mean_ms"] = round(self.ms_total / self.calls_timed, 1)
        return out


@dataclass
class Node:
    """One node of the phase -> task -> subtask tree."""

    name: str
    children: dict[str, "Node"] = field(default_factory=dict)
    own: Totals = field(default_factory=Totals)
    total: Totals = field(default_factory=Totals)

    def child(self, name: str) -> "Node":
        """Return (creating if needed) the child node called *name*.

        Args:
            name: Path segment.

        Returns:
            The child node.
        """
        node = self.children.get(name)
        if node is None:
            node = Node(name=name)
            self.children[name] = node
        return node

    def roll_up(self) -> Totals:
        """Recompute ``total`` from ``own`` plus every descendant.

        Returns:
            This node's inclusive roll-up.
        """
        total = Totals()
        total.merge(self.own)
        for kid in self.children.values():
            total.merge(kid.roll_up())
        self.total = total
        return total

    def as_dict(self) -> dict[str, Any]:
        """Project the subtree to JSON, heaviest child first.

        Returns:
            A nested dict of ``{name, totals, own, children}``.
        """
        kids = sorted(self.children.values(), key=lambda n: (-n.total.usd_total, -n.total.calls, n.name))
        return {
            "name": self.name,
            "totals": self.total.as_dict(),
            "own": self.own.as_dict() if self.own.calls else None,
            "children": [k.as_dict() for k in kids],
        }


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each parseable object from a JSONL ledger.

    A truncated final line is normal for a run that was killed, so a bad line
    is skipped rather than aborting the whole report.

    Args:
        path: Ledger file; a missing file yields nothing.

    Yields:
        One row dict per readable line.
    """
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def load_ledgers(session_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load every turn row and every detail row for a session.

    Both the in-process ledgers and the ``ext/`` shards written by
    out-of-process children (GEAK above all) are read; the ``.detail.jsonl``
    shards are the per-API-call side of the same pair.

    Args:
        session_dir: The session root directory.

    Returns:
        ``(turn_rows, detail_rows)``.
    """
    from hyperloom.inference_optimizer.session.session_paths import (
        llm_call_detail_path,
        llm_calls_path,
        trace_ext_dir,
    )

    turns = list(read_jsonl(llm_calls_path(session_dir)))
    details = list(read_jsonl(llm_call_detail_path(session_dir)))
    ext = trace_ext_dir(session_dir)
    if ext.is_dir():
        for shard in sorted(ext.glob("*.jsonl")):
            if shard.name.endswith(".detail.jsonl"):
                details.extend(read_jsonl(shard))
            else:
                turns.extend(read_jsonl(shard))
    return turns, details


def _segments(row: dict[str, Any]) -> list[str]:
    """Return the tree path a row belongs under, phase first.

    Args:
        row: A turn or detail row.

    Returns:
        Path segments, always at least one deep.
    """
    phase = str(row.get("phase") or "").strip() or UNPHASED
    raw = str(row.get("task_path") or "").strip()
    parts = [p for p in raw.split(PATH_SEPARATOR) if p] if raw else []
    return [phase] + (parts or [UNPATHED])


def build_tree(turns: Iterable[dict[str, Any]], details: Iterable[dict[str, Any]]) -> tuple[Node, dict[str, Any]]:
    """Place every call in the phase -> task tree and roll the totals up.

    A turn that expanded into detail rows contributes only those rows; a turn
    with none is counted once from the turn row itself, so nothing is double
    counted and nothing is dropped.

    Args:
        turns: Turn-level rows.
        details: Per-API-call rows.

    Returns:
        ``(root, coverage)`` where *coverage* records what the totals do not
        cover: undetailed turns, unpriced and untimed calls, apportioned splits.
    """
    turns = list(turns)
    details = list(details)
    detailed_ids = {str(r.get("call_id")) for r in details if r.get("call_id")}

    root = Node(name="session")
    coverage: dict[str, Any] = {
        "turns_total": len(turns),
        "turns_with_detail": 0,
        "turns_without_detail": 0,
        "detail_rows": len(details),
        "detail_rows_orphaned": 0,
    }

    def place(row: dict[str, Any]) -> Node:
        node = root
        for seg in _segments(row):
            node = node.child(seg)
        return node

    turn_ids = {str(r.get("call_id")) for r in turns if r.get("call_id")}
    for row in details:
        place(row).own.add_call(row)
        if str(row.get("call_id") or "") not in turn_ids:
            coverage["detail_rows_orphaned"] += 1
    for row in turns:
        node = place(row)
        node.own.turns += 1
        if str(row.get("call_id") or "") in detailed_ids:
            coverage["turns_with_detail"] += 1
        else:
            coverage["turns_without_detail"] += 1
            node.own.add_call(row)

    total = root.roll_up()
    coverage["calls_counted"] = total.calls
    coverage["calls_unpriced"] = total.calls - total.calls_priced
    coverage["calls_untimed"] = total.calls - total.calls_timed
    coverage["calls_timing_apportioned"] = total.calls_apportioned
    coverage["cost_sources"] = dict(sorted(total.cost_sources.items()))
    return root, coverage


def _pct(part: float, whole: float) -> str:
    """Format *part* as a percentage of *whole*.

    Args:
        part: Numerator.
        whole: Denominator; zero yields a dash.

    Returns:
        A short percentage string, or ``"-"``.
    """
    return f"{100.0 * part / whole:.1f}%" if whole else "-"


def _share(node: Totals, parent: Totals | None) -> str:
    """Format a node's share of its parent, by cost where cost is known.

    An unpriced run would show a dash in every row, which says nothing about
    where the work went, so the share falls back to call count there.

    Args:
        node: The node's roll-up.
        parent: The parent's roll-up; ``None`` at the root.

    Returns:
        A percentage string.
    """
    if parent is None:
        return "100.0%"
    if parent.usd_total:
        return _pct(node.usd_total, parent.usd_total)
    return _pct(node.calls, parent.calls)


def _render_rows(node: Node, depth: int, parent: Totals | None, out: list[str]) -> None:
    """Append one markdown table row per node, depth-first.

    Args:
        node: Subtree root to render.
        depth: Indentation depth (0 = session).
        parent: The parent's roll-up, for the share column.
        out: Line accumulator (mutated).
    """
    t = node.total
    indent = "&nbsp;" * 4 * depth
    label = f"{indent}{'└ ' if depth else ''}`{node.name}`"
    out.append(
        "| "
        + " | ".join(
            [
                label,
                str(t.calls),
                _share(t, parent),
                f"{t.isl:,}",
                f"{t.osl:,}",
                f"{t.tokens_in:,}",
                f"{t.tokens_thinking:,}",
                f"{t.tokens_out:,}",
                f"{t.ms_total / 1000.0:,.1f}",
                f"{t.ms_thinking / 1000.0:,.1f}",
                f"{t.ms_output / 1000.0:,.1f}",
                f"{t.usd_total:,.4f}",
                f"{t.usd_thinking:,.4f}",
                f"{t.usd_output:,.4f}",
                str(t.tool_calls),
            ]
        )
        + " |"
    )
    for kid in sorted(node.children.values(), key=lambda n: (-n.total.usd_total, -n.total.calls, n.name)):
        _render_rows(kid, depth + 1, t, out)


def _has_node(node: Node, name: str) -> bool:
    """Report whether any node in the subtree carries *name*.

    Args:
        node: Subtree root.
        name: Segment to look for.

    Returns:
        ``True`` when the segment appears anywhere beneath (or at) *node*.
    """
    if node.name == name:
        return True
    return any(_has_node(kid, name) for kid in node.children.values())


def render_markdown(session_id: str, root: Node, coverage: dict[str, Any]) -> str:
    """Render the whole report as markdown.

    Args:
        session_id: Session identifier, for the heading.
        root: The rolled-up tree.
        coverage: The coverage dict from :func:`build_tree`.

    Returns:
        The markdown document.
    """
    t = root.total
    lines = [
        f"# LLM call report — `{session_id}`",
        "",
        "## Session totals",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| API calls | {t.calls:,} |",
        f"| Agentic turns | {t.turns:,} |",
        f"| Tool calls | {t.tool_calls:,} |",
        f"| ISL (input sequence length) | {t.isl:,} |",
        f"| OSL (output sequence length) | {t.osl:,} |",
        f"| Tokens — input | {t.tokens_in:,} |",
        f"| Tokens — thinking | {t.tokens_thinking:,} |",
        f"| Tokens — output (visible) | {t.tokens_out:,} |",
        f"| Tokens — cache write / read | {t.tokens_cache_write:,} / {t.tokens_cache_read:,} |",
        f"| Time — total | {t.ms_total / 1000.0:,.1f} s |",
        f"| Time — thinking | {t.ms_thinking / 1000.0:,.1f} s |",
        f"| Time — output | {t.ms_output / 1000.0:,.1f} s |",
        f"| Cost — total | ${t.usd_total:,.4f} |",
        f"| Cost — input / cache | ${t.usd_input:,.4f} / ${t.usd_cache:,.4f} |",
        f"| Cost — thinking | ${t.usd_thinking:,.4f} |",
        f"| Cost — output | ${t.usd_output:,.4f} |",
        f"| Failed calls | {t.errors:,} |",
        "",
        "Thinking tokens sit beside the visible output tokens, not inside them: "
        "`OSL` is output + thinking, and the cost total is input + output + "
        "thinking + cache.",
        "",
        "## Phase → task → subtask tree",
        "",
        "| Node | Calls | $ share | ISL | OSL | Tok in | Tok think | Tok out "
        "| Time s | Think s | Out s | $ total | $ think | $ out | Tools |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    _render_rows(root, 0, None, lines)

    lines += ["", "## Models", "", "| Model | Calls |", "| --- | ---: |"]
    for model, count in sorted(t.models.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{model}` | {count:,} |")
    if not t.models:
        lines.append("| _(no model recorded)_ | 0 |")

    lines += ["", "## Coverage", "", "What the numbers above do and do not cover.", ""]
    unpriced = int(coverage.get("calls_unpriced", 0))
    untimed = int(coverage.get("calls_untimed", 0))
    apportioned = int(coverage.get("calls_timing_apportioned", 0))
    lines += [
        f"- **Priced:** {t.calls_priced:,} of {t.calls:,} calls carry a cost. "
        + (
            f"**{unpriced:,} do not and are excluded from every USD figure** — the totals are a floor, not the bill."
            if unpriced
            else "Every call is priced."
        ),
        f"- **Cost sources:** {coverage.get('cost_sources') or '(none)'} — a provider figure is the charge itself; "
        "a derived one is the shipped rate card applied to the token counts.",
        f"- **Timed:** {t.calls_timed:,} of {t.calls:,} calls carry a latency."
        + (f" {untimed:,} do not." if untimed else ""),
        f"- **Thinking/output split:** {apportioned:,} call(s) had it apportioned from the token ratio rather than "
        "measured off stream events; read those two columns as estimates.",
        f"- **Turns:** {coverage.get('turns_with_detail', 0):,} of {coverage.get('turns_total', 0):,} expanded into "
        f"per-call detail rows; the other {coverage.get('turns_without_detail', 0):,} are each counted once from the "
        "turn row alone, so their per-call breakdown is the turn's aggregate.",
    ]
    orphans = int(coverage.get("detail_rows_orphaned", 0))
    if orphans:
        lines.append(
            f"- **Orphans:** {orphans:,} detail row(s) join to no turn row — usually a child shard whose turn row "
            "was written by a process that did not finish."
        )
    if not _has_node(root, "geak"):
        lines.append(
            "- **GEAK:** no `geak` subtree is present. If this run invoked the kernel agent, its spend "
            "(historically the majority of a session's bill) was not harvested."
        )
    return "\n".join(lines) + "\n"


def _resolve_output_dir(session_dir: Path, explicit: Path | None) -> Path:
    """Pick where the report is written.

    Args:
        session_dir: The session root directory.
        explicit: ``--output-dir``, when given.

    Returns:
        The directory to write into.
    """
    if explicit is not None:
        return explicit
    base = os.environ.get("USER_DATA_PATH", "").strip()
    if base:
        return Path(base) / "reports" / session_dir.name
    return session_dir / "reports"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument vector; defaults to ``sys.argv``.

    Returns:
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description="Render a Hyperloom per-LLM-call report.")
    p.add_argument("--session-dir", "-s", required=True, type=Path, help="Session root directory")
    p.add_argument("--output-dir", "-o", type=Path, default=None, help="Where to write the report")
    p.add_argument("--max-depth", type=int, default=0, help="Trim the tree to this depth (0 = no limit)")
    return p.parse_args(argv)


def _collapse(node: Node) -> None:
    """Fold every descendant's own calls into *node*, then drop the descendants.

    The point of trimming is a shorter table, not a smaller total, so the
    dropped levels' spend has to survive as the surviving node's own.

    Args:
        node: The node to collapse into.
    """
    for kid in node.children.values():
        _collapse(kid)
        node.own.merge(kid.own)
    node.children.clear()


def _trim(node: Node, depth: int, limit: int) -> None:
    """Drop children below *limit*, keeping their totals folded into the parent.

    Args:
        node: Subtree root.
        depth: Current depth.
        limit: Maximum depth to keep; ``0`` disables trimming.
    """
    if limit and depth >= limit:
        _collapse(node)
        return
    for kid in node.children.values():
        _trim(kid, depth + 1, limit)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv``.

    Returns:
        ``0`` on success, ``2`` when the session directory has no ledger.
    """
    logging.basicConfig(
        level=os.environ.get("HYPERLOOM_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    session_dir = args.session_dir
    if not session_dir.is_dir():
        log.error("session dir not found: %s", session_dir)
        return 2

    turns, details = load_ledgers(session_dir)
    if not turns and not details:
        log.error("no LLM ledger rows under %s/reports/trace", session_dir)
        return 2

    root, coverage = build_tree(turns, details)
    if args.max_depth:
        _trim(root, 0, args.max_depth)
        root.roll_up()

    out_dir = _resolve_output_dir(session_dir, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "llm_call_report.md"
    json_path = out_dir / "llm_call_report.json"
    md_path.write_text(render_markdown(session_dir.name, root, coverage))
    json_path.write_text(
        json.dumps(
            {"session_id": session_dir.name, "coverage": coverage, "tree": root.as_dict()},
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )
    log.info(
        "wrote %s and %s (%d calls, %d turns, $%.4f)",
        md_path,
        json_path,
        root.total.calls,
        root.total.turns,
        root.total.usd_total,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
