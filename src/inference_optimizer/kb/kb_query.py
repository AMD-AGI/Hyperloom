"""kb_query — top-k recall over kb/entries.jsonl + kb/insights.jsonl.

Usage::

    python -m inference_optimizer.kb.kb_query "<query>" \\
            --kb-dir <session>/kb --top-k 5 --compact

Output (stdout): markdown bullet list of (category, model, lesson, gain, ts)
tuples; format depends on ``--compact`` / ``--json``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Iterable


__all__ = ["main", "score_record", "tokenize"]


# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


# ---------------------------------------------------------------------------
def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    except OSError:
        return []
    return out


def score_record(query_tokens: list[str], record: dict) -> float:
    """BM25-ish overlap score; deliberately simple, no embeddings.

    score = sum_t  tf(t, doc) × (1 + log(1 + gain))   for each query
                                                 token t
    where ``tf`` is the boolean indicator (token present at least once).
    """
    if not query_tokens:
        return 0.0
    doc = " ".join(
        str(record.get(k, ""))
        for k in ("category", "model", "model_family", "action", "lesson")
    )
    doc += " " + " ".join(record.get("tags", []) or [])
    doc_tokens = set(tokenize(doc))
    overlap = sum(1 for t in query_tokens if t in doc_tokens)
    if overlap == 0:
        return 0.0
    gain = float(record.get("gain", 0.0))
    boost = 1.0 + math.log1p(max(0.0, gain))
    status = str(record.get("status", "")).lower()
    if status in ("keep", "keeper", "succeeded"):
        boost *= 1.25
    elif status in ("revert", "fail", "reverted"):
        boost *= 0.75
    # recency bonus (1.0 → 1.25 for last 7 days)
    ts = float(record.get("ts", 0.0))
    if ts > 0:
        age_days = max(0.0, (time.time() - ts) / 86400.0)
        recency = max(1.0, 1.25 - 0.05 * age_days)
        boost *= recency
    return overlap * boost


# ---------------------------------------------------------------------------
def _format_compact(records: list[dict]) -> str:
    lines = []
    for rec in records:
        lines.append(
            f"- ({rec.get('category', '?')}, {rec.get('model', '?')}, "
            f"gain={rec.get('gain', 0.0):.2f}, status={rec.get('status', '?')}) "
            f"{rec.get('lesson', '')}"
        )
    return "\n".join(lines) or "(no matches)"


def _format_full(records: list[dict]) -> str:
    parts = []
    for rec in records:
        parts.append(
            "- **{cat}** | model=`{mdl}` ({fam}) action=`{act}` "
            "gain=**{gain}%** status=`{stat}` ts={ts}\n"
            "  {lesson}".format(
                cat=rec.get("category", "?"),
                mdl=rec.get("model", "?"),
                fam=rec.get("model_family", "?"),
                act=rec.get("action", "?"),
                gain=rec.get("gain", 0.0),
                stat=rec.get("status", "?"),
                ts=rec.get("ts", 0),
                lesson=rec.get("lesson", "").replace("\n", " "),
            )
        )
    return "\n".join(parts) or "(no matches)"


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kb_query")
    parser.add_argument("query", help="free-text query")
    parser.add_argument("--kb-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)

    qtokens = tokenize(args.query)
    if not qtokens:
        print("(empty query)")
        return 0

    entries = list(_iter_jsonl(args.kb_dir / "entries.jsonl"))
    insights = list(_iter_jsonl(args.kb_dir / "insights.jsonl"))
    pool = entries + insights

    scored = [
        (score_record(qtokens, rec), rec) for rec in pool
    ]
    scored = [(s, r) for (s, r) in scored if s > 0.0]
    scored.sort(key=lambda x: x[0], reverse=True)

    top = [r for _, r in scored[: max(0, int(args.top_k))]]

    if args.as_json:
        print(json.dumps(top, default=str))
    elif args.compact:
        print(_format_compact(top))
    else:
        print(_format_full(top))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
