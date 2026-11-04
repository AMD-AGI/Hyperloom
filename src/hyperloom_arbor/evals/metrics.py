"""metrics.py — turn one trajectory into eval numbers.

Everything here is computed from ``trajectory.jsonl`` alone, so a session can
be scored long after it ran, on a machine with no GPU and no container.

Four families, answering four different questions:

  * **Outcome** — did it work? The number the campaign already reports.
  * **Efficiency** — what did it cost to get there? Gain per GPU-hour and per
    million tokens, which is what actually separates two harnesses that reach
    the same throughput.
  * **Protocol adherence** — did the agent follow its own instructions? This
    is the one that makes a prompt edit measurable instead of hopeful. Two
    commits in this repo (`d0d0cfd`, `82053f4`) rewrote prompt text to force
    Agent-tool dispatch with no way to check whether behavior changed;
    ``agent_dispatch_rate`` and ``subagent_patch_applied_rate`` are that check.
  * **Pathology** — how did it waste time? Repeated changes, stalls, crash
    loops.

Metrics degrade rather than fail. A session with no marks still yields
outcome and efficiency numbers from artifacts and tool arguments; only the
intent-dependent adherence checks go ``None``, and a ``None`` is reported as
unknown rather than as zero, because scoring an unmeasured run as a failing
one is worse than admitting the gap.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import read_jsonl  # noqa: E402

#: Server launches and kills, from hyperloom-optimize.md's gate step. Each
#: restart costs 4–5 minutes on a large model, so the count is a real term in
#: the efficiency budget rather than trivia.
_SERVER_START_RE = re.compile(r"vllm\.entrypoints|sglang\.launch_server|api_server")
_SERVER_KILL_RE = re.compile(r"pkill\s+-f")

#: Env vars actually injected into the server, i.e. `docker exec -e KEY=VALUE`.
#: Deliberately not a bare `KEY=` match: the agent sets shell locals like
#: SESSION_DIR in the same command, and counting those produced pure noise.
_ENV_INJECT_RE = re.compile(r"-e\s+([A-Z][A-Z0-9_]{3,})=([^\s\"']+)")
_FLAG_RE = re.compile(r"(--[a-z][a-z0-9-]{3,})")

#: Flags every launch carries regardless of what is being tested. Reporting
#: ``--model`` or ``--port`` as a "retried change" is noise dressed as a finding,
#: and the earlier "present in every iteration" filter did not catch them: they
#: appeared in 4 of 5 iterations (two runs never reached a launch), so they
#: survived the intersection and were reported anyway.
_STRUCTURAL_FLAGS = frozenset({
    "--model", "--port", "--host", "--tensor-parallel-size", "--trust-remote-code",
    "--served-model-name", "--download-dir", "--max-model-len", "--quantization",
    "--max-num-seqs", "--showmemuse", "--enforce-eager",
})


#: The tier ladder from hyperloom-optimize.md. Config is a warmup the prompt
#: says to leave by iteration 3.
CONFIG_TIER_DEADLINE = 3

#: Escalation is mandatory at this null streak in a non-config tier.
ESCALATION_THRESHOLD = 3


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _span_seconds(steps: list[dict[str, Any]]) -> float | None:
    times = [t for t in (_parse_ts(s.get("ts")) for s in steps) if t]
    if len(times) < 2:
        return None
    return (max(times) - min(times)).total_seconds()


def _pct(num: float, den: float) -> float | None:
    return round(100.0 * num / den, 2) if den else None


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


def outcome_metrics(steps: list[dict[str, Any]], session_json: dict[str, Any],
                    artifacts: dict[str, Any]) -> dict[str, Any]:
    """Did it work."""
    gates = [s for s in steps if s.get("kind") == "gate"]
    decisions = [(s.get("mark") or {}).get("decision") for s in gates]
    keeps = [d for d in decisions if d == "KEEP"]
    reverts = [d for d in decisions if d == "REVERT"]

    # baseline is a plain float in state.py; fixtures may write it as a dict — handle both
    _baseline_raw = session_json.get("baseline")
    if isinstance(_baseline_raw, dict):
        baseline = _baseline_raw.get("tput") if _baseline_raw.get("tput") is not None else _baseline_raw.get("metric_value")
    else:
        baseline = _baseline_raw  # float or None

    # state.py serializes best_metric; fall back to old best_tput key for legacy sessions
    best = session_json.get("best_metric")
    if best is None:
        best = session_json.get("best_tput")

    # Fall back to the artifacts when session.json is absent: bench_output.json
    # is written by bench.py regardless of what the agent recorded.
    tputs = [a.get("output_throughput") for a in artifacts.values()
             if isinstance(a.get("output_throughput"), (int, float))]
    if best is None and tputs:
        best = max(tputs)
    if baseline is None and tputs:
        baseline = tputs[0]

    gain = None
    if isinstance(baseline, (int, float)) and isinstance(best, (int, float)) and baseline:
        gain = round(100.0 * (best - baseline) / baseline, 2)

    correctness_fails = sum(
        1 for s in gates if (s.get("mark") or {}).get("correctness") == "fail"
    )

    # A gate is only evidence if a bench artifact on disk corroborates the
    # number the agent reported. Without this, the whole report is the agent
    # grading its own homework.
    corroborated = unverified = 0
    below_noise = []
    for g in gates:
        mark = g.get("mark") or {}
        it = g.get("iteration")
        art = artifacts.get(it) if it is not None else None
        claimed = mark.get("tput")
        if not art or not art.get("measurements"):
            unverified += 1
        else:
            values = [m["output_throughput"] for m in art["measurements"]]
            mean = sum(values) / len(values)
            if isinstance(claimed, (int, float)) and (
                any(abs(claimed - v) < 0.5 for v in values) or abs(claimed - mean) < 1.0
            ):
                corroborated += 1
            else:
                unverified += 1
        # An effect smaller than the run-to-run spread of the same config is
        # not a measurement, whatever the KEEP threshold says.
        spread = (art or {}).get("bench_spread_pct")
        delta = mark.get("delta_pct")
        if (spread is not None and isinstance(delta, (int, float))
                and abs(delta) < spread):
            # Carry the individual runs. The spread alone can mislead: in the
            # observed case it was large only because a hot-cache run the agent
            # had itself discarded as invalid was among the artifacts, while the
            # two cold runs spread far less. A reader needs the values to judge
            # which comparison is the honest one.
            below_noise.append({"iteration": it, "delta_pct": delta,
                                "noise_spread_pct": spread,
                                "decision": mark.get("decision"),
                                "runs": [(m["file"], round(m["output_throughput"], 2))
                                         for m in (art or {}).get("measurements", [])]})

    return {
        "baseline_tput": baseline,
        "best_tput": best,
        "total_gain_pct": gain,
        "gates": len(gates),
        "keeps": len(keeps),
        "reverts": len(reverts),
        "keep_rate_pct": _pct(len(keeps), len(gates)),
        "correctness_failures": correctness_fails,
        "gates_corroborated_by_artifact": corroborated,
        "gates_unverified": unverified,
        "gates_within_noise_floor": below_noise or None,
        "iterations_observed": len({s["iteration"] for s in steps
                                    if s.get("iteration") is not None}),
    }


def _billed_usages(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Yield each message's ``usage`` exactly once.

    A single assistant message becomes several trajectory steps — a thinking
    block, some text, one per tool_use — and every step carries that message's
    whole ``usage`` dict. Summing per step therefore counts the same tokens
    once per block: measured at 2.60x for output tokens and 3.02x for thinking
    on the 3-hour validation run, which made every cost-per-percent number in
    the report roughly two and a half times too expensive.

    Steps predating ``message_id`` (or reconstructed from hooks alone, which
    have no usage at all) fall back to identity so nothing is silently dropped.
    """
    seen: set[Any] = set()
    out: list[dict[str, Any]] = []
    for s in steps:
        u = s.get("usage") or {}
        if not u:
            continue
        mid = s.get("message_id")
        if mid is None:
            out.append(u)          # cannot dedupe it; counting once is the best guess
            continue
        if mid in seen:
            continue
        seen.add(mid)
        out.append(u)
    return out


def efficiency_metrics(steps: list[dict[str, Any]],
                       price_in: float | None, price_out: float | None) -> dict[str, Any]:
    """What it cost."""
    tool_calls = [s for s in steps if s.get("kind") == "tool_call"]

    tok_in = tok_out = tok_cache_read = tok_cache_write = tok_think = 0
    for u in _billed_usages(steps):
        tok_in += u.get("input_tokens") or 0
        tok_out += u.get("output_tokens") or 0
        tok_cache_read += u.get("cache_read_input_tokens") or 0
        tok_cache_write += u.get("cache_creation_input_tokens") or 0
        # Reasoning volume survives even when reasoning text does not: models
        # that withhold the text still bill and report the tokens. This is the
        # only reasoning-effort signal available on an Opus 5 run.
        tok_think += (u.get("output_tokens_details") or {}).get("thinking_tokens") or 0

    thinking_steps = [s for s in steps if s.get("kind") == "thinking"]
    thinking_text_chars = sum(len(s.get("text") or "") for s in thinking_steps)

    wall = _span_seconds(steps)
    # Split by actor: subagents run concurrently with the orchestrator and
    # with each other, so summed tool time legitimately exceeds wall clock
    # (it did: 11,987 s of tool time inside 11,923 s of wall clock). Reporting
    # one total invites reading that as impossible rather than as parallelism.
    tool_ms = sum(s.get("duration_ms") or 0 for s in tool_calls)
    tool_ms_orch = sum(s.get("duration_ms") or 0 for s in tool_calls
                       if s.get("actor") != "subagent")
    tool_ms_sub = tool_ms - tool_ms_orch

    # Time inside iterations whose gate reverted — the directly wasted budget.
    by_iter: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for s in steps:
        by_iter[s.get("iteration")].append(s)
    reverted_seconds = 0.0
    for it, group in by_iter.items():
        if it is None:
            continue
        outcome = next((g.get("iteration_outcome") for g in group
                        if g.get("iteration_outcome")), None)
        if outcome and outcome.get("decision") == "REVERT":
            reverted_seconds += _span_seconds(group) or 0.0

    # Time to the first KEEP: the metric that separates a harness that finds a
    # win early from one that finds the same win after burning a night.
    t0 = _parse_ts(steps[0].get("ts")) if steps else None
    first_keep_s = None
    for s in steps:
        if s.get("kind") == "gate" and (s.get("mark") or {}).get("decision") == "KEEP":
            tk = _parse_ts(s.get("ts"))
            if t0 and tk:
                first_keep_s = (tk - t0).total_seconds()
            break

    server_starts = sum(
        1 for s in tool_calls
        if _SERVER_START_RE.search(json.dumps(s.get("tool_input") or {}, default=str))
        and not _SERVER_KILL_RE.search(json.dumps(s.get("tool_input") or {}, default=str))
    )

    cost = None
    if price_in is not None and price_out is not None:
        cost = round((tok_in / 1e6) * price_in + (tok_out / 1e6) * price_out, 4)

    return {
        "wall_clock_seconds": round(wall, 1) if wall else None,
        "tool_calls": len(tool_calls),
        "tool_time_seconds_total": round(tool_ms / 1000.0, 1),
        "tool_time_seconds_orchestrator": round(tool_ms_orch / 1000.0, 1),
        "tool_time_seconds_subagents_concurrent": round(tool_ms_sub / 1000.0, 1),
        "input_tokens": tok_in,
        "output_tokens": tok_out,
        "thinking_tokens": tok_think,
        "thinking_share_of_output_pct": _pct(tok_think, tok_out) if tok_out else None,
        "thinking_blocks": len(thinking_steps),
        "thinking_text_chars": thinking_text_chars,
        # 0 chars against a non-zero token count means the model withheld the
        # text — see docs/archive/evals-README.md "Reasoning text". Not a collector fault.
        "thinking_text_captured": bool(thinking_text_chars) if tok_think else None,
        "cache_read_tokens": tok_cache_read,
        "cache_write_tokens": tok_cache_write,
        "total_tokens": tok_in + tok_out,
        "estimated_cost_usd": cost,
        "seconds_in_reverted_iterations": round(reverted_seconds, 1),
        "reverted_time_share_pct": _pct(reverted_seconds, wall) if wall else None,
        "seconds_to_first_keep": round(first_keep_s, 1) if first_keep_s else None,
        "server_starts": server_starts,
    }


def adherence_metrics(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Did the agent follow hyperloom-optimize.md.

    Each check names the instruction it tests, so a failing number points at a
    line of prompt rather than at a vague sense that the run went badly.
    """
    by_iter: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for s in steps:
        by_iter[s.get("iteration")].append(s)
    real_iters = {k: v for k, v in by_iter.items() if k is not None}

    # "You MUST use the Agent tool to spawn a specialist subagent" (line 165).
    iters_with_agent = sum(
        1 for group in real_iters.values()
        if any(s.get("tool_name") == "Agent" for s in group)
    )
    agent_dispatches = sum(1 for s in steps if s.get("tool_name") == "Agent")

    # "Config is a warmup, not a strategy" — leave it by iteration 3 (line 70).
    config_iters = [it for it, group in real_iters.items()
                    if any(s.get("tier") == "config" for s in group)]
    config_exit = (max(config_iters) if config_iters else None)

    # Research protocol is mandatory before a framework/kernel change (304-319).
    research_ok = research_total = 0
    for it, group in real_iters.items():
        tier = next((s.get("tier") for s in group if s.get("tier")), None)
        if tier not in ("framework", "kernel", "feature"):
            continue
        research_total += 1
        searches = sum(1 for s in group
                       if s.get("tool_name") in ("WebSearch", "WebFetch"))
        if searches >= 1:
            research_ok += 1

    # "One change at a time. Gate it before touching anything else." (line 254)
    multi_gate_iters = sum(
        1 for group in real_iters.values()
        if sum(1 for s in group if s.get("kind") == "gate") > 1
    )

    escalations = sum(1 for s in steps
                      if (s.get("mark") or {}).get("kind") == "escalation")

    # "These two scripts are never modified by Claude" (README.md). Edits to the
    # measuring instrument deserve a first-class check rather than a reading of
    # the log, because they silently change what every throughput number means.
    # Observed for real: on a cluster whose vLLM had moved benchmark_serving.py
    # behind the CLI, the agent patched bench.sh to keep the benchmark runnable
    # — a defensible fix, and still something a reviewer must be told about.
    harness_edits = []
    for s in steps:
        if s.get("tool_name") not in ("Edit", "Write", "NotebookEdit"):
            continue
        target = str((s.get("tool_input") or {}).get("file_path", ""))
        if target.endswith(("bench.sh", "correctness.sh")):
            harness_edits.append({"ts": s.get("ts"), "file": target,
                                  "iteration": s.get("iteration")})
    # An edit before the first gate applies to baseline and every iteration
    # alike, so within-run comparisons still hold; one after does not.
    first_gate_ts = next((s.get("ts") for s in steps if s.get("kind") == "gate"), None)
    edits_after_baseline = sum(
        1 for e in harness_edits
        if first_gate_ts and (e["ts"] or "") > first_gate_ts
    )

    # Subagents that actually applied a patch, vs. ones that only reported.
    # This is exactly what commit 82053f4 tried to enforce by prompt.
    sub_steps = [s for s in steps if s.get("actor") == "subagent"]
    sub_edit_tools = sum(
        1 for s in sub_steps
        if s.get("tool_name") in ("Edit", "Write", "NotebookEdit")
        or (s.get("tool_name") == "Bash"
            and re.search(r"\bpatch\b|git apply",
                          json.dumps(s.get("tool_input") or {}, default=str)))
    )

    return {
        "agent_dispatches": agent_dispatches,
        "iterations_with_agent_dispatch": iters_with_agent,
        "agent_dispatch_rate_pct": _pct(iters_with_agent, len(real_iters)),
        "config_tier_last_iteration": config_exit,
        "config_tier_exit_ok": (config_exit <= CONFIG_TIER_DEADLINE
                                if config_exit is not None else None),
        "research_before_code_iterations": research_total,
        "research_protocol_followed": research_ok,
        "research_protocol_rate_pct": _pct(research_ok, research_total),
        "iterations_with_multiple_gates": multi_gate_iters,
        "escalations_fired": escalations,
        "subagent_steps": len(sub_steps),
        "subagent_patch_actions": sub_edit_tools,
        "benchmark_harness_edits": len(harness_edits),
        "benchmark_harness_edits_after_first_gate": edits_after_baseline,
        "benchmark_harness_edit_detail": harness_edits or None,
    }


def branch_metrics(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-branch rollup — the unit Arbor's DFS actually searches in.

    ``branch_id`` was carried all the way into the trajectory and then scored
    nowhere, even though branches are the search structure: without this you
    cannot ask which hypothesis was worth its GPU time.

    Each row also carries the branch's ``hypothesis`` next to the ``change``
    its gate actually tested. Those two diverging is the stale-mark problem —
    the mark is written when a branch is picked and never revised, so an agent
    that pivots mid-iteration leaves every later step labelled with the plan it
    abandoned. This is surfaced for a human to read rather than auto-flagged:
    deciding whether a change "matches" a hypothesis is a judgement, and a
    keyword heuristic here would produce exactly the confident-sounding noise
    the other pathology checks were rewritten to remove.
    """
    by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        bid = step.get("branch_id")
        if bid:
            by_branch[bid].append(step)

    rows = []
    for bid, group in by_branch.items():
        gate = next((g for g in group if g.get("kind") == "gate"), None)
        mark = (gate or {}).get("mark") or {}
        branch_mark = next(
            (g.get("mark") for g in group
             if (g.get("mark") or {}).get("kind") == "branch"), None
        ) or {}
        billed = _billed_usages(group)
        tok = sum(u.get("output_tokens") or 0 for u in billed)
        think = sum((u.get("output_tokens_details") or {}).get("thinking_tokens") or 0
                    for u in billed)
        rows.append({
            "branch_id": bid,
            "tier": next((s.get("tier") for s in group if s.get("tier")), None),
            "iteration": next((s.get("iteration") for s in group
                               if s.get("iteration") is not None), None),
            "steps": len(group),
            "subagent_steps": sum(1 for s in group if s.get("actor") == "subagent"),
            "wall_seconds": round(_span_seconds(group) or 0.0, 1),
            "output_tokens": tok,
            "thinking_tokens": think,
            "decision": mark.get("decision"),
            "delta_pct": mark.get("delta_pct"),
            "hypothesis": (branch_mark.get("hypothesis") or "")[:400],
            "gated_change": (mark.get("change") or "")[:400],
        })
    rows.sort(key=lambda r: -(r["wall_seconds"] or 0))
    return rows


def pathology_metrics(steps: list[dict[str, Any]], session_dir: Path,
                      baseline_flags: frozenset[str] = frozenset()) -> dict[str, Any]:
    """How it wasted time."""
    tool_calls = [s for s in steps if s.get("kind") == "tool_call"]

    # A retried change is only waste if it was already tried and REVERTED.
    #
    # An earlier version counted any env var or flag seen in more than one
    # server launch, which reported `--model×2` and `--port×2` — the constants
    # present in every launch — as repeated changes, and `SESSION_DIR×3`, a
    # shell variable that never reaches the server. Noise that looks like a
    # finding is worse than no finding, so this now measures the real thing:
    # settings carried by an iteration that reverted, and tried again later.
    per_iter_settings: dict[Any, set[str]] = defaultdict(set)
    reverted_iters: set[Any] = set()
    for s in tool_calls:
        it = s.get("iteration")
        if it is None:
            continue
        outcome = s.get("iteration_outcome") or {}
        if outcome.get("decision") == "REVERT":
            reverted_iters.add(it)
        blob = json.dumps(s.get("tool_input") or {}, default=str)
        if not _SERVER_START_RE.search(blob):
            continue
        # Only `-e KEY=VALUE`: an injected server env var, not a shell local.
        for key, val in _ENV_INJECT_RE.findall(blob):
            per_iter_settings[it].add(f"{key}={val}")
        for flag in set(_FLAG_RE.findall(blob)):
            # Baseline configuration is not a change and can never be a retry.
            if flag in _STRUCTURAL_FLAGS or flag in baseline_flags:
                continue
            per_iter_settings[it].add(flag)

    # Baseline settings appear in every iteration; they are configuration, not
    # a retried change. Only settings absent from at least one iteration can be
    # a deliberate change at all.
    all_iters = set(per_iter_settings)
    universal = set.intersection(*per_iter_settings.values()) if per_iter_settings else set()

    retried: dict[str, int] = {}
    if reverted_iters and len(all_iters) > 1:
        for setting in set().union(*per_iter_settings.values()) - universal:
            carriers = {it for it, s in per_iter_settings.items() if setting in s}
            if carriers & reverted_iters and len(carriers) > 1:
                retried[setting] = len(carriers)

    # Distinguish "measured, found nothing" from "could not measure".
    measurable = bool(reverted_iters) and len(all_iters) > 1

    # Identical tool call issued repeatedly — a loop the agent did not notice.
    digests = Counter(s.get("tool_input_digest") for s in tool_calls
                      if s.get("tool_input_digest"))
    duplicate_calls = sum(v - 1 for v in digests.values() if v > 1)

    errored = sum(1 for s in tool_calls if s.get("tool_error"))

    return {
        "duplicate_tool_calls": duplicate_calls,
        "duplicate_tool_call_rate_pct": _pct(duplicate_calls, len(tool_calls)),
        "errored_tool_calls": errored,
        "errored_tool_call_rate_pct": _pct(errored, len(tool_calls)),
        "retried_after_revert": retried if measurable else None,
        "retried_after_revert_count": len(retried) if measurable else None,
        "aborted": (session_dir / "ABORTED.md").exists(),
    }


def tool_histogram(steps: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(s.get("tool_name") for s in steps
                        if s.get("kind") == "tool_call" and s.get("tool_name")).most_common())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def compute(session_dir: Path, price_in: float | None = None,
            price_out: float | None = None) -> dict[str, Any]:
    """Compute every metric family for one session."""
    steps = list(read_jsonl(session_dir / "trajectory.jsonl"))
    trace_dir = session_dir / "reports" / "trace"

    try:
        summary = json.loads((trace_dir / "collect_summary.json").read_text(encoding="utf-8"))
    except Exception:
        summary = {}
    try:
        meta = json.loads((trace_dir / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    # The config's own server_flags are the baseline: present in every launch
    # by construction, so never a change under test.
    baseline_flags = frozenset(
        re.findall(r"(--[a-z][a-z0-9-]{3,})",
                   (meta.get("config") or {}).get("server_flags", "") or "")
    )

    session_json = summary.get("session_json") or {}
    artifacts = {int(k): v for k, v in (summary.get("artifacts") or {}).items()}

    # Two different "models" are in play and conflating them is easy: the one
    # being optimized (Qwen3-32B) and the one doing the optimizing (Opus 5).
    # Derived from the transcript rather than $ANTHROPIC_MODEL, which is unset
    # on this deployment — the default is picked per-model-tier instead.
    driving = Counter(s["model"] for s in steps if s.get("model"))

    return {
        "session": session_dir.name,
        "session_dir": str(session_dir),
        "model": (meta.get("config") or {}).get("model_name"),
        "orchestrator_model": driving.most_common(1)[0][0] if driving else None,
        "hardware": (meta.get("config") or {}).get("gpu_type"),
        "framework": (meta.get("config") or {}).get("framework"),
        "arbor_git_sha": meta.get("arbor_git_sha"),
        "steps": len(steps),
        "outcome": outcome_metrics(steps, session_json, artifacts),
        "efficiency": efficiency_metrics(steps, price_in, price_out),
        "adherence": adherence_metrics(steps),
        "pathology": pathology_metrics(steps, session_dir, baseline_flags),
        "branches": branch_metrics(steps),
        "tools": tool_histogram(steps),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--price-in", type=float, default=None,
                        help="USD per 1M input tokens. Omitted by default — no "
                             "prices are hardcoded, so cost is reported only "
                             "when you supply the rates you are actually billed.")
    parser.add_argument("--price-out", type=float, default=None,
                        help="USD per 1M output tokens.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON")
    args = parser.parse_args(argv)

    result = compute(Path(args.session_dir).resolve(), args.price_in, args.price_out)
    out = json.dumps(result, indent=2, default=str)
    (Path(args.session_dir) / "reports" / "trace" / "metrics.json").write_text(
        out, encoding="utf-8"
    )
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
