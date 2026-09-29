#!/usr/bin/env python3
"""Mine Hyperloom sessions for token waste: where each call's prompt comes from, what repeats.

    python scripts/meta_rsi/analyze.py SESSION_DIR [SESSION_DIR ...] --out DIR

Per session it reports, from the traces the session already wrote:

* orchestration prompt anatomy: size of each ``=== section ===`` block, its share of the
  prompt, and how much of it is byte-identical to the previous call in the same phase;
* history overhead: context tokens the provider read minus the current prompt and the
  phase system prompt, i.e. what the multi-turn conversation carried along;
* repetition: consecutive orchestration replies that restate the previous one, phase-skip
  escalations, and the policy-denial count the prompt itself reports;
* critic and specialist prompt anatomy (markdown / numbered ``## N.`` sections);
* specialist task profiles: turns, context growth per turn, tokens, and the journal
  verdict of the task, to price work that ended in REVERT;
* forge calls.

Read-only on the session directories; results go to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

try:
    from . import _paths
except ImportError:  # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _paths  # noqa: F401

from hyperloom.inference_optimizer.trace.meta_rsi_ledger import (
    billed_tokens,
    build_ledger,
    call_tokens,
    component_of,
    context_tokens,
    iter_jsonl,
)

CHARS_PER_TOKEN = 4.0

_ORCH_HEAD = re.compile(r"(?m)^(=== .+? ===|[A-Z][A-Za-z ]{2,40}:)[ \t]*$")
_MD_HEAD = re.compile(r"(?m)^(#{1,3} [^\n]+)$")
_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")
_ESCALATE = re.compile(r"escalate_strategy_change|skip_to_(?:sweep|kernel|close|framework)")
_RESEND = re.compile(r"\bre-?(?:sent|send|emit(?:ted)?|issu(?:e|ed)|submit(?:ted)?)\b", re.I)
_DENIAL_TOTAL = re.compile(r"Recent policy denials \(newest last, total=(\d+)\)")


def _tok(chars: float) -> float:
    return chars / CHARS_PER_TOKEN


def normalize_heading(head: str) -> str:
    return _DIGITS.sub("#", head.strip())


def split_sections(text: str, pattern: re.Pattern[str]) -> "OrderedDict[str, str]":
    """Split a prompt at heading lines; repeated headings get a ``#k`` suffix."""
    marks = [(m.start(), m.group(1)) for m in pattern.finditer(text)]
    out: OrderedDict[str, str] = OrderedDict()
    if not marks:
        out["<body>"] = text
        return out
    if marks[0][0] > 0:
        out["<preamble>"] = text[: marks[0][0]]
    for i, (off, head) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        key = normalize_heading(head)
        k, n = key, 1
        while k in out:
            n += 1
            k = f"{key}#{n}"
        out[k] = text[off:end]
    return out


def _base_key(key: str) -> str:
    return re.sub(r"#\d+$", "", key)


def anatomy(prompts: list[str], pattern: re.Pattern[str]) -> dict[str, Any]:
    """Section sizes and unchanged-vs-previous share over an ordered prompt sequence."""
    total: Counter[str] = Counter()
    same: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    prev: OrderedDict[str, str] | None = None
    chars = 0
    for p in prompts:
        secs = split_sections(p, pattern)
        chars += len(p)
        for k, v in secs.items():
            b = _base_key(k)
            total[b] += len(v)
            seen[b] += 1
            if prev is not None and prev.get(k) == v:
                same[b] += len(v)
        prev = secs
    n = len(prompts) or 1
    rows = []
    for key, size in total.most_common():
        rows.append(
            {
                "section": key,
                "avg_tok": round(_tok(size / n)),
                "share": round(size / chars, 4) if chars else 0.0,
                "unchanged_vs_prev": round(same[key] / size, 4) if size else 0.0,
                "present_in": seen[key],
            }
        )
    return {"calls": len(prompts), "avg_tok": round(_tok(chars / n)), "sections": rows}


def _norm_reply(text: str) -> str:
    return _WS.sub(" ", _DIGITS.sub("#", (text or "").lower())).strip()[:160]


def _system_prompt_tokens(sdir: Path, role: str) -> dict[str, float]:
    out: dict[str, float] = {}
    base = sdir / "agents" / role
    for f in base.glob("system_prompt*.snapshot.md"):
        name = f.name[len("system_prompt") : -len(".snapshot.md")].strip(".") or "default"
        try:
            out[name] = _tok(len(f.read_text()))
        except OSError:
            continue
    return out


def analyze_session(session_dir: str | Path) -> dict[str, Any]:
    sdir = Path(session_dir)
    trace = sdir / "reports" / "trace"
    calls = list(iter_jsonl(trace / "llm_calls.jsonl"))
    ledger = build_ledger(sdir, calls)
    by_call_id = {c.get("call_id"): c for c in calls if c.get("call_id")}

    orch_prompts: dict[str, list[str]] = defaultdict(list)
    orch_rows: list[dict[str, Any]] = []
    critic_prompts: list[str] = []
    specialist_prompts: list[str] = []
    for rec in iter_jsonl(trace / "conversations.jsonl"):
        comp = component_of(rec)
        prompt = rec.get("prompt") or ""
        if comp == "orchestration":
            phase = str(rec.get("phase") or "none")
            orch_prompts[phase].append(prompt)
            orch_rows.append(
                {
                    "phase": phase,
                    "prompt_tok": _tok(len(prompt)),
                    "reply": rec.get("response") or "",
                    "call": by_call_id.get(rec.get("call_id")),
                    "denials": max((int(x) for x in _DENIAL_TOTAL.findall(prompt)), default=0),
                }
            )
        elif comp == "critic":
            critic_prompts.append(prompt)
        elif comp == "specialist":
            specialist_prompts.append(prompt)

    # history overhead: context read beyond this turn's prompt and the phase system prompt
    sys_tok = _system_prompt_tokens(sdir, "orchestration")
    hist_share: dict[str, list[float]] = defaultdict(list)
    hist_tok: dict[str, list[float]] = defaultdict(list)
    for r in orch_rows:
        c = r["call"]
        if not c:
            continue
        ctx = context_tokens(c)
        if ctx <= 0:
            continue
        sp = sys_tok.get(r["phase"], sys_tok.get("default", 0.0))
        h = max(0.0, ctx - r["prompt_tok"] - sp)
        hist_share[r["phase"]].append(h / ctx)
        hist_tok[r["phase"]].append(h)

    # repetition of orchestration replies within a phase
    rep: dict[str, dict[str, Any]] = {}
    for phase in orch_prompts:
        rows = [r for r in orch_rows if r["phase"] == phase]
        dup = resend = escal = 0
        prev = None
        for r in rows:
            norm = _norm_reply(r["reply"])
            if prev is not None and norm[:80] == prev[:80]:
                dup += 1
            if _RESEND.search(r["reply"] or ""):
                resend += 1
            if _ESCALATE.search(r["reply"] or ""):
                escal += 1
            prev = norm
        ctx = [context_tokens(r["call"]) for r in rows if r["call"]]
        wasted = sum(context_tokens(r["call"]) for r in rows if r["call"] and _RESEND.search(r["reply"] or ""))
        rep[phase] = {
            "calls": len(rows),
            "consecutive_repeat": dup,
            "resend_replies": resend,
            "escalations": escal,
            "max_denials_reported": max((r["denials"] for r in rows), default=0),
            "ctx_tok_in_resend_replies": wasted,
            "ctx_p50": int(statistics.median(ctx)) if ctx else 0,
        }

    # specialist task profiles from per-turn call records
    journal = ledger["outcome"]
    verdict_by_task: dict[str, str] = {}
    try:
        jrows = json.loads((sdir / "reports" / "optimization_journal.json").read_text()).get("entries") or []
    except (OSError, ValueError, AttributeError):
        jrows = []
    for e in jrows:
        if isinstance(e, dict) and e.get("task_id"):
            verdict_by_task[str(e["task_id"])] = f"{e.get('outcome')}:{e.get('kind')}"
    tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in calls:
        if component_of(c) == "specialist" and c.get("task_id"):
            tasks[str(c["task_id"])].append(c)
    task_rows = []
    for tid, rows in tasks.items():
        rows.sort(key=lambda r: (_turn(r), str(r.get("ts") or "")))
        ctx = [context_tokens(r) for r in rows]
        billed = sum(billed_tokens(call_tokens(r)) for r in rows)
        task_rows.append(
            {
                "task_id": tid,
                "phase": rows[0].get("phase"),
                "turns": len(rows),
                "ctx_first": ctx[0],
                "ctx_last": ctx[-1],
                "ctx_growth_per_turn": round((ctx[-1] - ctx[0]) / max(1, len(rows) - 1)),
                "billed": billed,
                "verdict": verdict_by_task.get(tid, "none"),
            }
        )
    spend_by_verdict: Counter[str] = Counter()
    for t in task_rows:
        spend_by_verdict[t["verdict"].split(":")[0]] += t["billed"]
    turns = [t["turns"] for t in task_rows]
    growth = [t["ctx_growth_per_turn"] for t in task_rows if t["turns"] > 1]

    forge = [c for c in calls if component_of(c) == "forge"]
    return {
        "session_dir": str(sdir),
        "ledger": ledger,
        "orchestration": {
            "system_prompt_tok": {k: round(v) for k, v in sys_tok.items()},
            "anatomy_by_phase": {ph: anatomy(ps, _ORCH_HEAD) for ph, ps in orch_prompts.items()},
            "history_share_p50": {ph: round(statistics.median(v), 3) for ph, v in hist_share.items() if v},
            "history_tok_p50": {ph: round(statistics.median(v)) for ph, v in hist_tok.items() if v},
            "repetition": rep,
        },
        "critic": {
            "system_prompt_tok": {k: round(v) for k, v in _system_prompt_tokens(sdir, "critic").items()},
            "anatomy": anatomy(critic_prompts, _MD_HEAD),
        },
        "specialist": {
            "initial_prompt_anatomy": anatomy(specialist_prompts, _MD_HEAD),
            "tasks": len(task_rows),
            "turns_p50": statistics.median(turns) if turns else 0,
            "turns_max": max(turns) if turns else 0,
            "ctx_growth_per_turn_p50": statistics.median(growth) if growth else 0,
            "billed_by_verdict": dict(spend_by_verdict),
            "top_tasks": sorted(task_rows, key=lambda t: -t["billed"])[:8],
        },
        "forge": {
            "calls": len(forge),
            "ctx": sorted((context_tokens(c) for c in forge), reverse=True)[:5],
            "billed": sum(billed_tokens(call_tokens(c)) for c in forge),
        },
        "journal": {k: journal.get(k) for k in ("total_gain_pct", "config_keep", "config_revert", "keep", "revert")},
    }


def _turn(row: dict[str, Any]) -> int:
    try:
        return int(row.get("turn") or 0)
    except (TypeError, ValueError):
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sessions", nargs="+", help="session directories")
    ap.add_argument("--out", required=True, help="directory for per-session JSON results")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in args.sessions:
        sdir = Path(s)
        try:
            res = analyze_session(sdir)
        except Exception as exc:  # noqa: BLE001 -- one bad session must not stop the batch
            print(f"skip {sdir}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        (out / f"{sdir.name}.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
        tot = res["ledger"]["tokens"]["total"]
        print(
            f"{sdir.name}: calls={tot['calls']} billed={tot['billed'] / 1e6:.1f}M weighted={tot['weighted'] / 1e6:.1f}M"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
