"""kb_ingest — append a single lesson to ``kb/entries.jsonl``.

Usage::

    python -m inference_optimizer.kb.kb_ingest \\
        --kb-dir <session>/kb \\
        --category model_class_lesson \\
        --model gpt-oss-20b \\
        --action backends \\
        --lesson "vllm beat sglang by 9% on dense models" \\
        --tags '["dense","backend_choice"]' \\
        --gain 9.0 \\
        --status keep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..orchestrator.kb import KBEntry, _model_family


__all__ = ["main"]


def _parse_tags(value: str | None) -> list[str]:
    if not value:
        return []
    value = value.strip()
    if value.startswith("["):
        try:
            data = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--tags was not valid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise SystemExit("--tags JSON must decode to a list")
        return [str(x) for x in data]
    return [t.strip() for t in value.split(",") if t.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kb_ingest")
    parser.add_argument("--kb-dir", required=True, type=Path)
    parser.add_argument("--category", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--lesson", required=True)
    parser.add_argument("--tags", default="")
    parser.add_argument("--gain", type=float, default=0.0)
    parser.add_argument(
        "--status",
        default="observation",
        choices=("keep", "revert", "fail", "observation"),
    )
    parser.add_argument("--user-id", default="default")
    parser.add_argument("--ts", type=float, default=None)
    args = parser.parse_args(argv)

    args.kb_dir.mkdir(parents=True, exist_ok=True)
    rec = KBEntry(
        category=args.category,
        user_id=args.user_id,
        model=args.model,
        model_family=_model_family(args.model),
        action=args.action,
        lesson=args.lesson,
        tags=_parse_tags(args.tags),
        gain=float(args.gain),
        status=args.status,
        ts=float(args.ts) if args.ts is not None else time.time(),
    )
    out = args.kb_dir / "entries.jsonl"
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec.to_dict(), default=str) + "\n")
    print(f"appended 1 record to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
