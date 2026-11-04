"""render.py — render one Arbor trajectory as a single readable file.

``SESSION_LOG.md`` is the answer to "where do I see the whole run": every user
prompt, assistant response, tool call with its arguments, tool output, mark,
and gate, in order, in one file. Claude Code's own transcript already holds
most of this, but it is per *Claude* session rather than per *Arbor* session,
it lives under ``~/.claude/`` rather than with the run's artifacts, and it is
raw JSONL. This closes all three gaps: ``trajectory.jsonl`` has already
stitched the restarts together, and this renders it.

Two widths, because the same run has two audiences:

  * ``--full``    — everything, for reading one iteration closely.
  * ``--compact`` — head/tail of large tool outputs, for skimming a long run.

Large outputs go inside ``<details>`` blocks so a multi-day session stays
openable in an editor or on GitHub.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import read_jsonl  # noqa: E402

#: Output longer than this is folded into a <details> block. Roughly a screen
#: and a half — beyond that, inlining it buries the next step.
FOLD_THRESHOLD = 800

#: Per-entry clip in compact mode.
COMPACT_LIMIT = 600

_KIND_LABEL = {
    "user_prompt": "USER",
    "assistant_text": "ASSISTANT",
    "thinking": "THINKING",
    "tool_call": "TOOL",
    "mark": "MARK",
    "gate": "GATE",
}


def _fence(text: str, lang: str = "") -> str:
    """Fence text, escaping any fence inside it so the block cannot break out."""
    safe = text.replace("```", "``​`")
    return f"```{lang}\n{safe}\n```"


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} more chars]"


def _fold(summary: str, body: str, lang: str = "") -> str:
    return (
        f"<details><summary>{summary}</summary>\n\n{_fence(body, lang)}\n\n</details>"
    )


def build_agent_index(steps: list[dict[str, Any]]) -> dict[str, int]:
    """Map each Agent dispatch's tool_use_id to a small stable ordinal.

    With 16 dispatches and 1,101 subagent steps in one run, "this line came
    from a subagent" is not enough to read by — you need to know *which*.
    """
    index: dict[str, int] = {}
    for step in steps:
        if step.get("tool_name") == "Agent" and step.get("tool_use_id"):
            index.setdefault(step["tool_use_id"], len(index) + 1)
    return index


def _actor_tag(step: dict[str, Any], agent_index: dict[str, int]) -> str:
    """The leading, greppable actor marker for one step.

    Deliberately at the front of the line and fixed-shape. The previous form
    put a `subagent` badge in the middle, after a variable-length tool name and
    before an optional duration — invisible when skimming and impossible to
    grep reliably, which is how 1,101 subagent lines read as if they were the
    orchestrator's.
    """
    if step.get("actor") != "subagent":
        return "MAIN "
    parent = step.get("parent_agent_id")
    n = agent_index.get(parent)
    if n:
        return f"SUB#{n}"
    # No Agent window covers this step. It is hook-only — no transcript line —
    # which is normally proof of a subagent, but with no open dispatch it could
    # equally be a background continuation or an orchestrator call whose
    # transcript entry is missing. Marked uncertain rather than asserted.
    return "SUB?  "


def _render_step(step: dict[str, Any], compact: bool,
                 agent_index: dict[str, int] | None = None) -> str:
    kind = step.get("kind", "?")
    label = _KIND_LABEL.get(kind, kind.upper())
    ts = (step.get("ts") or "")[11:19] or "--:--:--"
    agent_index = agent_index or {}
    tag = _actor_tag(step, agent_index)
    badge = ""  # actor now leads the line; see _actor_tag
    ts = f"{ts} {tag}"
    out: list[str] = []

    if kind == "thinking":
        # Rendered even though empty: an omitted block would read as "the model
        # did not reason here", when in fact the model withheld the text while
        # still billing the tokens.
        if step.get("text_available"):
            body = step.get("text", "")
            body = _shorten(body, COMPACT_LIMIT) if compact else body
            out.append(f"**`{ts}` THINKING**{badge}\n\n{_fence(body)}")
        else:
            siglen = step.get("signature_len", 0)
            model = step.get("model") or "this model"
            out.append(
                f"**`{ts}` THINKING**{badge} — "
                f"*withheld by `{model}` ({siglen} char signature only; "
                f"token count is still recorded)*"
            )
        return "\n".join(out)

    if kind in ("user_prompt", "assistant_text"):
        body = step.get("text", "")
        body = _shorten(body, COMPACT_LIMIT) if compact else body
        out.append(f"**`{ts}` {label}**{badge}\n\n{_fence(body)}")
        return "\n".join(out)

    if kind == "tool_call":
        name = step.get("tool_name", "?")
        dur = step.get("duration_ms")
        dur_s = f" · {dur} ms" if dur is not None else ""
        err = " · **ERROR**" if step.get("tool_error") else ""
        out.append(f"**`{ts}` TOOL · {name}**{badge}{dur_s}{err}")

        args = json.dumps(step.get("tool_input") or {}, indent=2, default=str)
        if len(args) > FOLD_THRESHOLD and compact:
            out.append(_fold(f"arguments ({len(args)} chars)", args, "json"))
        else:
            out.append(_fence(args, "json"))

        output = step.get("tool_output") or ""
        clipped = step.get("tool_output_clipped_bytes") or 0
        if output:
            note = f" (+{clipped} bytes clipped at capture)" if clipped else ""
            if len(output) > FOLD_THRESHOLD:
                shown = _shorten(output, COMPACT_LIMIT) if compact else output
                out.append(_fold(f"output ({len(output)} chars){note}", shown))
            else:
                out.append(_fence(output))
        return "\n".join(out)

    if kind in ("mark", "gate"):
        mark = step.get("mark") or {}
        mk = mark.get("kind", "?")
        if mk == "gate":
            decision = mark.get("decision", "?")
            icon = "✅" if decision == "KEEP" else "↩️"
            delta = mark.get("delta_pct")
            delta_s = f"{delta:+.1f}%" if isinstance(delta, (int, float)) else "?"
            tput = mark.get("tput")
            tput_s = f"{tput:.1f} tok/s" if isinstance(tput, (int, float)) else "?"
            out.append(
                f"**`{ts}` {icon} GATE — {decision}** · {delta_s} · {tput_s} · "
                f"correctness={mark.get('correctness', '?')}"
            )
            if mark.get("change"):
                out.append(f"> {mark['change']}")
        elif mk == "branch":
            out.append(
                f"**`{ts}` ▶ BRANCH** `{mark.get('branch_id', '?')}` "
                f"· tier=`{mark.get('tier', '?')}`"
            )
            if mark.get("hypothesis"):
                out.append(f"> {mark['hypothesis']}")
        elif mk == "escalation":
            out.append(
                f"**`{ts}` ⚠ ESCALATION** — null_streak={mark.get('null_streak', '?')}"
            )
        else:
            out.append(f"**`{ts}` MARK · {mk}** — {json.dumps(mark, default=str)[:300]}")
        return "\n".join(out)

    return f"**`{ts}` {label}**{badge}"


def render(session_dir: Path, compact: bool = False) -> Path:
    """Write ``reports/trace/SESSION_LOG.md`` for one session."""
    trace_dir = session_dir / "reports" / "trace"
    steps = list(read_jsonl(session_dir / "trajectory.jsonl"))
    agent_index = build_agent_index(steps)
    try:
        meta = json.loads((trace_dir / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    lines: list[str] = []
    lines.append(f"# Arbor session log — `{session_dir.name}`\n")

    cfg = meta.get("config") or {}
    rows = [
        ("Model", cfg.get("model_name")),
        ("Hardware", cfg.get("gpu_type")),
        ("Framework", cfg.get("framework")),
        ("Container", cfg.get("docker_container")),
        ("TP", cfg.get("tp")),
        ("Goal", cfg.get("goal")),
        ("Arbor SHA", meta.get("arbor_git_sha")),
        ("Started", meta.get("started")),
        ("Steps", len(steps)),
    ]
    lines.append("| | |")
    lines.append("|---|---|")
    for k, v in rows:
        if v not in (None, ""):
            lines.append(f"| **{k}** | {v} |")
    lines.append("")

    sessions = meta.get("claude_sessions") or []
    if len(sessions) > 1:
        # Worth stating plainly: this file is the only place the legs of a
        # /loop run appear as one timeline.
        lines.append(
            f"> Stitched from **{len(sessions)}** Claude sessions "
            f"(a `/loop` run reopens a transcript on every restart).\n"
        )

    if meta.get("thinking_capture") == "withheld-by-model":
        lines.append(
            "> **Note:** this model withholds reasoning text (it returns a "
            "signature and bills the tokens, but emits no text on any capture "
            "route). `THINKING` entries below show a placeholder; reasoning "
            "*volume* is still reported in `EVAL_REPORT.md`. Models measured to "
            "return the text: `claude-sonnet-4.6`, `claude-haiku-4.5`.\n"
        )

    iterations = sorted({s["iteration"] for s in steps if s.get("iteration") is not None})
    if iterations:
        toc = " · ".join(f"[iter {i}](#iteration-{i})" for i in iterations)
        lines.append(f"**Iterations:** {toc}\n")

    if any(st.get("actor") == "subagent" for st in steps):
        lines.append(
            "**Reading this log:** every entry is prefixed `MAIN` "
            "(the orchestrator) or `SUB#n` (the n-th dispatched specialist). "
            "`SUB#n` steps sit under the `TOOL · Agent` call that spawned them. "
            "`SUB?` means no dispatch window covers the step, so its actor is inferred, "
            "not established. Grep `MAIN \u0060` or `SUB#4\u0060` to isolate one actor.\n"
        )
    lines.append("---\n")

    current_iter = "__unset__"
    for step in steps:
        it = step.get("iteration")
        if it != current_iter:
            current_iter = it
            if it is None:
                lines.append("\n## Setup\n")
            else:
                outcome = step.get("iteration_outcome") or {}
                dec = outcome.get("decision")
                suffix = f" — {dec}" if dec else ""
                lines.append(f"\n## Iteration {it}{suffix}\n")
        lines.append(_render_step(step, compact, agent_index))
        lines.append("")

    out_path = trace_dir / "SESSION_LOG.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--compact", action="store_true",
                        help="Clip large tool outputs (default: full fidelity)")
    args = parser.parse_args(argv)

    path = render(Path(args.session_dir).resolve(), compact=args.compact)
    size = path.stat().st_size
    print(f"[arbor-trace] wrote {path} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
