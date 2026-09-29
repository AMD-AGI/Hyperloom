# Meta-RSI round 1: where Hyperloom's tokens go (session evidence)

Scope: every Hyperloom session since 2026-08-25 that left LLM traces on this host's Weka
mount (61 sessions: 57 under `/wekafs/csl`, 3 under `/wekafs/zgong`, 1 under
`/wekafs/users/4f2656cd…`), plus the 1,309 Claude Code transcripts of the agents those
sessions spawned (`~/.claude/projects`, 65 sessions). Numbers are reproducible with:

```bash
python scripts/meta_rsi/analyze.py <session dirs> --out RESULTS       # per-session prompt anatomy, repetition
python scripts/meta_rsi/summarize.py RESULTS                          # cross-session tables
python scripts/meta_rsi/transcripts.py --out TX                       # agent transcripts: turns, tools, results
python scripts/meta_rsi/ab_compare.py A=<session> B=<session>         # side-by-side ledger
```

"Weighted" tokens price each class against an uncached input token (cache read x0.1,
cache write x1.25, output x5), so a cache-read-heavy call is not overstated.

## 1. Totals

| component | calls | billed | weighted | weighted share |
|---|---:|---:|---:|---:|
| specialist | 10,383 | 786.7M | 186.5M | 59% |
| orchestration | 902 | 214.0M | 95.8M | 30% |
| forge | 9 | 108.3M | 31.2M | 10% |
| critic | 147 | 1.2M | 1.5M | 0.5% |

Largest component x phase: specialist@FRAMEWORK_AGENT 38%, orchestration@KERNEL_AGENT 22%,
specialist@PRELUDE 19%, forge 10%.

Agent transcripts (per model request): specialists run 17 turns (p50) and the first turn
already reads 35,079 tokens while their initial user prompt is 2,551 tokens; orchestration
ticks run 4 requests (p50) with a 58,727-token first request for a 7,799-token prompt.

## 2. Findings, with the change each one drives

### F1. Built-in tool schemas nobody calls are re-read on every turn (code)

Claude Code 2.1.197 ships 26 built-in tools. Measured first-turn context for a trivial
prompt (`/tmp`, 15-token system prompt; `claude --print --output-format json`):

| configuration | first-turn context |
|---|---:|
| specialist today (`--disallowedTools KillShell,SlashCommand`) | 32,541 |
| `--tools` = the tools specialists actually call (Bash, Read, Write, Edit, Glob, Grep, WebSearch, WebFetch, Task) | 6,309 |
| `--tools` Bash, Read, Write, Edit, Glob, Grep | 4,236 |
| no built-in tools | 156 |

Per-tool schema cost (first-turn tokens over the no-tool base): Workflow 8,018, DesignSync
3,603, Monitor 3,174, Skill 2,654, CronCreate 1,823, TaskUpdate 1,696, ScheduleWakeup 1,695,
Task 1,670, EnterWorktree 1,599, Bash 1,326, TaskCreate 1,284, ExitWorktree 1,199, … Across
449 specialist transcripts the tools actually called were Bash (12,045), Read (360),
WebSearch (228), Write (222), Edit (191), WebFetch (109), Agent/Task (91).

The orchestration backend passes only `allowed_tools`, which is a permission list: every
built-in stays loaded. Measured with the SDK and the real KERNEL_AGENT system prompt:
39,775 tokens today vs 15,548 with `tools=["Read", "Bash"]` (-24,227 per model request).

KernelForge already made this change upstream (#1443, "`allowed_tools` is only a
permission list", -27,068 tokens measured); specialists and orchestration did not.

* Change: `--tools` for specialists (`orchestrator/specialists/subprocess_.py`), SDK `tools=`
  for orchestration (`orchestrator/roles/claude.py`). MCP tools are unaffected.
* Estimate on this corpus: specialists 9,857 turns x 26.2k = 258M billed / ~39M weighted
  (21% of specialist weighted); orchestration 1,330 requests x 24.2k = 32M billed.

### F2. The specialist prompt tells the agent to spend turns reading the clock (text)

`specialist_prompt_builder.py` section `## 2a. EXECUTION BUDGET` says: "Self-throttle:
check elapsed wall-clock with Bash (`date -u +%s` vs the start above)". 41% of specialist
turns return under ~300 tokens and edit nothing; those status-check turns read 48% of all
specialist context (392M tokens), because each one re-reads the whole context:

| status check | turns | context read |
|---|---:|---:|
| clock check (`date`) | 1,203 | 97.8M |
| `sleep` then check | 466 | 67.1M |
| `tail` a log | 521 | 59.9M |
| `grep` | 673 | 54.7M |
| `cat` / JSON result | 526 | 47.7M |

523 of 647 `sleep` calls wait 60 s or less, although the Bash tool accepts timeouts of
15-30 minutes (specialists already pass `timeout` up to 30 min).

* Change: state the absolute deadline and forbid stand-alone clock checks; tell the agent to
  wait with one blocking command instead of sleep-and-check loops.

### F3. KERNEL-phase orchestration prompts re-send the same 27k tokens every tick (code + text)

Orchestration prompt anatomy in KERNEL_AGENT (644 calls, 7 sessions, 34.2k tokens avg):

| section | avg tokens | share | unchanged vs previous tick |
|---|---:|---:|---:|
| `Findings:` (research hints) | 17,749 | 52% | 98% |
| `=== Specialist findings ===` (every round ever) | 5,420 | 16% | 99% |
| `Residual questions:` | 4,469 | 13% | 100% |
| `=== Shared session state ===` | 4,296 | 13% | 0% |

The block comes from `_specialist_findings_block()` in `orchestrator/loop/conversation.py`,
which renders every research hint and every specialist round with no bound
(`research_hints.json` was 97 KB in the zgong session). The prompt also opens with the
sections that change every tick (phase clock, mission progress, time budget, shared state),
so the unchanged 27k tokens behind them never hit the prompt cache: in that session's 631
KERNEL calls the median request wrote 70,004 tokens to cache and read 151,293, and cache
writes were 80% of the phase's weighted cost (54.5M of 68.4M).

* Change: bound the block (newest rounds and hints under a character budget) and add a
  `get_specialist_findings` context tool for the rest; order the tick prompt stable-first,
  volatile-last so the stable part is served from cache.

### F4. A deferred integrate turns KERNEL into a 2.7-hour loop (code + text)

zgong `20260924T162544Z-d7064552` (Qwen3.5-122B-A10B-FP8, vLLM, TP2): 341 of 631 KERNEL
orchestration replies re-send the same `integrate` for the pending KEEP
`llm_input_residual_rmsnorm`, and 266 carry `escalate_strategy_change{skip_to_sweep}`, which
the phase machine keeps refusing because `kernel_work_pending()` is true while any KEEP is
pending. Those replies read 76.7M context tokens. The same workload by the same user one week
earlier (`20260917T080829Z-8d4e588d`) spent 67.1M billed tokens for +194.0%; this run spent
204.7M for +77.6% (orchestration weighted 1.77M -> 69.38M).

* Change: when the reply re-emits an integrate that is still deferred and the state digest is
  unchanged, skip the LLM tick and let the coordinator retry; after N deferrals of the same
  KEEP, release it through the existing `_clear_pending_integrate` path with a recorded reason.
  `orchestration.md` states that a deferred integrate must not be re-sent.

### F5. Already fixed on main (no change)

Forge: 257 transcripts in this corpus predate #1443 (the old tree `d8cfe2c1a` does not
contain it); main already narrows Forge tools and defers its knowledge maps. Forge megaprompts
(8-20M-token sessions) are a pre-#1443 artefact.

### F6. Small or not actionable

Critic: 0.5% of weighted tokens. `CLAUDE.md` -> `AGENTS.md` (2.7k tokens) exists on main, but
the SDK loads no setting sources unless asked, and the specialist worktrees (framework trees)
carry none. Exact repeats of an identical tool call are rare (14 in 449 specialist runs).

## 3. The model lever

Which roles can run on an open model is decided by replaying recorded calls
(`reports/trace/conversations.jsonl` has the full prompt and reply of every orchestration and
critic call) against GLM-5.3-Flash and scoring agreement with the recorded Opus decision; see
the report for the result.
