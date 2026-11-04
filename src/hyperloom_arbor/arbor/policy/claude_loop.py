"""claude_loop — Harness-owned lifecycle with agent-driven decisions.

The harness owns all mechanics: server start/stop, bench, git commit/revert.
The agent only decides what to change and edits arbor/config.json.
The critic independently verifies the result.

Flow:
  1. Harness: escalation check, render program (knowledge + ledger + history)
  2. Agent: reads program, edits, emits CLASS + LEDGER lines, says DONE
  3. Harness: commits. A read-only PROBE stops here — no server run.
  4. Harness: stops server, starts server, waits healthy, runs bench
  5. Harness: persists ledger records, then the GATE decides mechanically
     using the declared claim class → KEEP / PROBE / REVERT
  6. Critic: audits diff and class honesty; may only downgrade to REVERT
  7. Harness: applies decision
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from arbor import component, frameworks
from arbor.cli.iterate import _find_arbor_root
from arbor.measure.integrity import snapshot_digests, verify_unchanged
from arbor.measure.intents import ALL_NAMES, fill as _fill_intents, resolve
from arbor.pack import agent_skills, load_campaign, render_program
from arbor.state import CampaignState


# ── Agent prompt ────────────────────────────────────────────────────────────

def _build_agent_prompt(program: str, campaign, state, iteration: int) -> str:
    config = campaign.config
    container = config.get("docker_container", "")
    workspace = config.get("workspace_root", "")

    prompt_file = Path(__file__).parent / "claude_loop_prompt.md"
    instructions = _fill_intents(prompt_file.read_text()) if prompt_file.exists() else ""

    # Component mode: a dossier pass says so, then this turn's time, then the host's
    # profile and what it has already merged (its notes), each read every iteration, then
    # the host's rules, last, so they are what the agent reads after the pack's knowledge
    # (which may say `docker exec` or `pkill`).
    rules = ""
    if component.is_component(config):
        blocks = (component.dossier_pass_block(config), component.turn_block(config),
                  component.profile_block(config), component.notes_block(config))
        rules = "".join(f"\n{b}" for b in blocks if b) + f"\n{_component_rules(config)}"
    return (
        f"{program}\n\n"
        f"---\n\n"
        f"## Iteration {iteration} — tier: {state.tier}\n\n"
        f"{instructions}\n\n"
        f"Container: `{container}`\n"
        f"Workspace: `{workspace}`\n"
        f"{rules}"
    )


# ── Critic prompt ───────────────────────────────────────────────────────────

def _build_critic_prompt(
    state: CampaignState, iteration: int,
    git_diff: str, bench_output: dict,
    correctness: dict, integrity_ok: bool, integrity_reason: str,
    threshold: float = 1.01,
    claim_class: str = "B",
    change: str = "",
    gate_decision: str = "",
    gate_reason: str = "",
    constraints: list[str] | None = None,
    agent_output: str = "",
    out_of_tree_block: str = "",
    arm_block: str = "",
    bundle: dict | None = None,
    accuracy: dict | None = None,
) -> str:
    obj = state.metric_name or "output_throughput"
    direction = state.metric_direction or "max"
    best = state.best_metric

    critic_template = Path(__file__).parent / "critic_prompt.md"
    critic_instructions = (
        _fill_intents(critic_template.read_text()) if critic_template.exists() else ""
    )

    # The pack's rules decide what each class is allowed to claim. Without them
    # the critic cannot tell a foundation step from a failed optimization, and
    # Check 2 degenerates into a coin flip.
    constraints_block = ""
    if constraints:
        rules = "\n".join(f"- {c}" for c in constraints)
        constraints_block = (
            f"### Pack rules — these define what each class may claim\n\n{rules}\n\n"
        )

    # The agent's own reasoning is the evidence for a class-A or PROBE claim.
    # Tail it: the declaration and its justification land at the end.
    agent_block = ""
    if agent_output:
        tail = agent_output.strip()[-4000:]
        agent_block = f"### Agent's report\n\n```\n{tail}\n```\n\n"

    # A declared bundle is one claim (arbor/measure/bundles.py): the diff should
    # carry its members and nothing that is not the bundle.
    bundle_line = ""
    if bundle and bundle.get("enabled", True):
        bundle_line = (f"Bundle: `{bundle['id']}` — {', '.join(bundle.get('members', []))} "
                       f"({bundle.get('reason', '')})\n")
    elif bundle:
        bundle_line = f"Bundle: `{bundle['id']}` declared, but bundles are not enabled in this pack\n"

    accuracy_line = ""
    if accuracy:
        accuracy_line = (f"Accuracy, full GSM8K (for {', '.join(accuracy.get('by') or [])}): "
                         f"{accuracy.get('score')} against {accuracy.get('golden')} "
                         f"− {accuracy.get('tolerance')} ({accuracy.get('source')})\n")

    return (
        f"{critic_instructions}\n\n"
        f"---\n\n"
        f"## Evidence for Iteration {iteration}\n\n"
        f"### Gate decision (already applied — you may only downgrade)\n"
        f"**{gate_decision}** — {gate_reason}\n\n"
        f"### Claim\n"
        f"Declared intent: **{resolve(claim_class).name}**\n"
        f"Change: {change}\n"
        f"{bundle_line}\n"
        f"{constraints_block}"
        f"### Objective\n"
        f"Metric: `{obj}` (direction: {direction})\n"
        f"Current best: {best}\n"
        f"Threshold for a class-B claim: {threshold}x\n\n"
        f"### Validator Integrity\n"
        f"{'PASS' if integrity_ok else 'FAIL'}: {integrity_reason}\n\n"
        f"### Correctness\n"
        f"Passed: {correctness.get('passed', False)}\n"
        f"Response: {correctness.get('response', correctness.get('error', 'unknown'))}\n"
        f"{accuracy_line}\n"
        f"{arm_block}"
        f"{agent_block}"
        f"{out_of_tree_block}"
        f"### Git Diff\n"
        f"```diff\n{git_diff}\n```\n\n"
        f"### Bench Output\n"
        f"```json\n{json.dumps(bench_output, indent=2)}\n```\n\n"
        f"Now decide: UPHOLD or REVERT."
    )


# ── SDK wrappers ────────────────────────────────────────────────────────────

def _log_block(prefix: str, block) -> None:
    if hasattr(block, "text") and block.text:
        print(f"[{prefix}] {block.text[:600]}", file=sys.stderr, flush=True)
    elif hasattr(block, "name"):
        inp = getattr(block, "input", {})
        detail = inp.get("command") or inp.get("path") or str(inp)[:120]
        print(f"[{prefix}] TOOL: {block.name} — {detail[:200]}", file=sys.stderr, flush=True)
    elif hasattr(block, "content") and block.content:
        # Tool result
        content = block.content if isinstance(block.content, str) else str(block.content)
        print(f"[{prefix}] RESULT: {content[:400]}", file=sys.stderr, flush=True)


def _fix_api_env() -> None:
    """Ensure ANTHROPIC_BASE_URL points to a live gateway port.

    The shell may have a stale port (e.g. 9102 from a dead proxy).
    Probe known ports and fix os.environ before the SDK inherits it.
    """
    import os, urllib.request
    current = os.environ.get("ANTHROPIC_BASE_URL", "")
    if current:
        try:
            req = urllib.request.Request(current + "/v1/messages",
                                        method="POST",
                                        headers={"Content-Type": "application/json"},
                                        data=b'{}')
            urllib.request.urlopen(req, timeout=2)
            return
        except urllib.error.HTTPError:
            return
        except Exception:
            pass
    for port in (9101, 9100, 9102):
        try:
            req = urllib.request.Request(f"http://localhost:{port}/v1/messages",
                                        method="POST",
                                        headers={"Content-Type": "application/json"},
                                        data=b'{}')
            urllib.request.urlopen(req, timeout=2)
        except urllib.error.HTTPError:
            os.environ["ANTHROPIC_BASE_URL"] = f"http://localhost:{port}"
            print(f"[arbor] Fixed ANTHROPIC_BASE_URL → localhost:{port}",
                  file=sys.stderr, flush=True)
            return
        except Exception:
            continue


_fix_api_env()


def _report_agent_tmp_health() -> None:
    """Log the state of the directory the agent CLI needs, BEFORE it runs.

    The CLI creates ``/tmp/claude-<uid>`` at startup and dies instantly if it
    cannot. Three campaigns have aborted on exactly that, and every time the
    evidence had to be reconstructed after the fact from a node that had
    already been released. One cheap line per iteration turns that into a
    record.

    WHAT THE ERRNO MEANS HERE, because it is the diagnosis. On this cluster
    Slurm runs ``JobContainerType=job_container/tmpfs``, so /tmp is a bind
    mount from a PER-JOB path (``/slurm/<jobid>/.<jobid>/_tmp``, confirmed by
    findmnt). If that bind SOURCE is removed while the mount
    stays, every lookup under /tmp returns ENOENT rather than ENOSPC or
    EACCES -- which matches the observed failure exactly, and matches the
    campaign's own report that its baseline artifacts under /tmp had been
    reaped. So: ENOENT here points at the mount, a full filesystem would say
    ENOSPC, and permissions would say EACCES. Record which one it actually is.

    This never raises. A health probe that can break the loop it is protecting
    is worse than no probe.
    """
    try:
        probe = Path(tempfile.gettempdir()) / f"claude-{os.getuid()}"
        listable = os.path.isdir("/tmp") and bool(os.listdir("/tmp") is not None)
        st = os.statvfs("/tmp")
        free_inodes = st.f_favail
        free_gb = st.f_bavail * st.f_frsize / 1e9
        marker = probe.parent / f".arbor-tmp-probe-{os.getpid()}"
        marker.mkdir(parents=True, exist_ok=True)
        marker.rmdir()
        print(f"[arbor] /tmp OK (listable={listable} free={free_gb:.0f}GB "
              f"inodes={free_inodes} agent-dir-exists={probe.exists()})",
              file=sys.stderr, flush=True)
    except OSError as e:
        print(f"[arbor] /tmp UNHEALTHY: {e.__class__.__name__} errno="
              f"{getattr(e, 'errno', '?')} {e} — the agent will almost "
              f"certainly fail to start. ENOENT here means the per-job bind "
              f"mount lost its source, not that the disk is full.",
              file=sys.stderr, flush=True)
    except Exception as e:  # never break the loop from a health probe
        print(f"[arbor] /tmp probe failed to run: {e}", file=sys.stderr, flush=True)


class AgentLaunchError(RuntimeError):
    """The agent process failed, carrying whatever it actually said on stderr.

    WHY THIS EXISTS. When the CLI dies at startup the SDK raises an exception
    whose whole text is

        Command failed with exit code 1 (exit code: 1)
        Error output: Check stderr output for details

    -- a message that names a stream nobody was reading. Three campaigns have
    now died this way (v3 on 2026-09-19, v5, and the cuda-graph campaign),
    costing ~3 h of exclusive MI355X time on the first occasion alone, and each
    time the diagnosis had to be reconstructed from a single line that happened
    to leak into the driver's own stdout. The real cause was one `mkdir` errno
    that was being thrown away.

    A failure that is going to abort the campaign should at minimum say what it
    was. The stderr callback below captures it; this exception carries it.
    """


def _skill_options(options_cls, skills: list[str] | None,
                   allowed_tools: list[str]) -> tuple[dict, list[str]]:
    """SDK options for a pack's skills (Pack.skills), and the tools to allow.

    None leaves the options as they were before packs listed skills. A list
    loads the project's settings, so `.claude/skills/` and with it
    `.claude/settings.json` (its trajectory hooks run in the agent too), and
    lets the agent invoke only the listed skills. An SDK without the `skills`
    option gets the Skill tool instead, which cannot narrow the list; one
    without `setting_sources` cannot load skills at all. Both are logged.
    """
    if skills is None:
        return {}, allowed_tools
    params = inspect.signature(options_cls).parameters
    if "setting_sources" not in params:
        print(f"[arbor] this claude_agent_sdk cannot load project skills; "
              f"the pack's skills {skills} are unavailable", file=sys.stderr, flush=True)
        return {}, allowed_tools
    opts = {"setting_sources": ["project"]}
    if "skills" in params:
        opts["skills"] = list(skills)   # the SDK allows the Skill tool itself
    elif skills:
        print(f"[arbor] this claude_agent_sdk has no `skills` option; allowing the "
              f"Skill tool, so every project skill is invocable, not only {skills}",
              file=sys.stderr, flush=True)
        allowed_tools = [*allowed_tools, "Skill"]
    return opts, allowed_tools


# An agent that cannot reach its model has done nothing, so it is started again before the
# iteration is booked not measured: under Hyperloom on Cluster C (2026-10-29) a 20-minute gateway
# outage made three starts in a row fail with ERR_PROXY_TUNNEL, and the not-measured streak
# ended a 9-hour run. Connection-level and server-side (5xx, overloaded) errors only: a 4xx
# (a refused key, a model the CLI does not support) does not heal by waiting.
CONNECT_ERRORS = ("couldn't connect", "could not connect", "err_proxy", "econnrefused",
                  "econnreset", "etimedout", "enotfound", "eai_again", "socket hang up",
                  "connection error", "connect timeout", "network error", "fetch failed",
                  "api error: 500", "api error: 502", "api error: 503", "api error: 504",
                  "api error: 529", "overloaded")
CONNECT_WAIT_ENV = "ARBOR_AGENT_CONNECT_WAIT_S"
CONNECT_WAIT_S = 900.0          # in all, per turn; a component phase's cap still bounds it
CONNECT_BACKOFF_S = (30.0, 60.0, 120.0, 240.0, 300.0)
# The model name on the assistant message the CLI writes for a failed API call.
SYNTHETIC_MODEL = "<synthetic>"

# The last agent start's connection failure, or None once a turn has reached its model. When the
# not-measured streak ends the run, it says whether the model could be reached (_write_aborted).
_unreachable: str | None = None


def _connect_failure(error: BaseException) -> bool:
    """True when ``error`` says the agent could not reach its model, before it did anything."""
    if getattr(error, "arbor_started", False):
        return False
    text = str(error).lower()
    return any(m in text for m in CONNECT_ERRORS)


def _connect_wait_budget() -> float:
    try:
        return max(0.0, float(os.environ.get(CONNECT_WAIT_ENV, CONNECT_WAIT_S)))
    except ValueError:
        return CONNECT_WAIT_S


async def _run_claude(prompt: str, cwd: str, allowed_tools: list[str],
                      label: str = "agent", skills: list[str] | None = None,
                      **_kw) -> str:
    """One agent turn. A start that cannot reach the model is retried with backoff, up to
    ``ARBOR_AGENT_CONNECT_WAIT_S`` (900 s) in all, before the error reaches the caller."""
    global _unreachable
    budget, waited, attempt = _connect_wait_budget(), 0.0, 0
    while True:
        attempt += 1
        try:
            out = await _run_claude_once(prompt, cwd, allowed_tools, label, skills)
            _unreachable = None
            return out
        except Exception as e:
            # Set before the wait: a component phase's cap can cancel the wait itself.
            connect = _connect_failure(e)
            _unreachable = str(e)[:200] if connect else None
            if not connect or waited >= budget:
                raise
            wait = min(CONNECT_BACKOFF_S[min(attempt, len(CONNECT_BACKOFF_S)) - 1], budget - waited)
            print(f"[arbor] {label}: the agent could not reach its model ({str(e)[:200]}); "
                  f"starting it again in {wait:.0f} s (attempt {attempt + 1}, "
                  f"{waited / 60:.1f} of {budget / 60:.0f} min waited)", file=sys.stderr, flush=True)
            await asyncio.sleep(wait)
            waited += wait


async def _run_claude_once(prompt: str, cwd: str, allowed_tools: list[str],
                           label: str, skills: list[str] | None) -> str:
    if os.environ.get("ARBOR_AGENT_BACKEND", "claude").lower() in {
        "openai", "gpt", "openai-compatible",
    }:
        from arbor.policy.openai_backend import run
        return await asyncio.to_thread(run, prompt, cwd, allowed_tools, label)

    from claude_agent_sdk import ClaudeAgentOptions, query

    # Bounded on purpose: this is a crash trail, not a log. A CLI that fails at
    # startup emits a handful of lines; one that is merely chatty must not be
    # able to grow the driver's memory across a 24 h campaign.
    stderr_lines: list[str] = []

    def _capture_stderr(line: str) -> None:
        line = (line or "").rstrip()
        if not line:
            return
        if len(stderr_lines) < 200:
            stderr_lines.append(line)

    skill_opts, allowed_tools = _skill_options(ClaudeAgentOptions, skills, allowed_tools)
    options = ClaudeAgentOptions(
        cwd=cwd,
        allowed_tools=allowed_tools,
        permission_mode="acceptEdits",
        stderr=_capture_stderr,
        **skill_opts,
    )

    text_parts: list[str] = []
    sdk_errors: list[str] = []   # the CLI's own error results, e.g. "API Error: ..."

    try:
        async for message in query(prompt=prompt, options=options):
            if skills and getattr(message, "subtype", None) == "init":
                # The SDK's own account of what loaded, checked once per run.
                # Its list is every skill the CLI discovered (the project's
                # and the CLI's bundled ones); the `skills` option hides the
                # rest from the model and the Skill tool refuses them, so the
                # agent can use only the listed ones that were found.
                loaded = (getattr(message, "data", None) or {}).get("skills") or []
                names = {s if isinstance(s, str) else (s.get("name") if isinstance(s, dict) else None)
                         for s in loaded} - {None}
                enabled = [s for s in skills if s in names]
                missing = [s for s in skills if s not in names]
                print(f"[arbor] {label} skills: {enabled} (of {len(names)} discovered)"
                      + (f"; MISSING {missing}" if missing else ""),
                      file=sys.stderr, flush=True)
            if getattr(message, "is_error", False):
                err_msg = getattr(message, "result", str(message))
                sdk_errors.append(str(err_msg)[:300])
                print(f"[arbor] SDK error: {err_msg}", file=sys.stderr, flush=True)
            if hasattr(message, "content"):
                # The CLI reports a failed API call as an assistant message of its own, from
                # the model "<synthetic>" ("API Error: Couldn't connect through your proxy"):
                # not the agent's words, so neither its output nor a sign that the turn started.
                # Counted as text, it made every gateway outage look like a turn that did work,
                # which is never retried.
                synthetic = getattr(message, "model", None) == SYNTHETIC_MODEL
                for block in message.content:
                    if hasattr(block, "text") and synthetic:
                        sdk_errors.append(block.text[:300])
                    elif hasattr(block, "text"):
                        text_parts.append(block.text)
                    _log_block(label, block)
    except Exception as e:
        # Re-raise with the stderr attached rather than swallowing it. The
        # caller records this string in session.json and in ABORTED.md, so
        # whatever lands here is what a human reads days later. `arbor_started`
        # says the turn produced text: _run_claude never starts such a turn again.
        started = bool(text_parts)
        if not stderr_lines and not sdk_errors:
            e.arbor_started = started
            raise
        trail = " | ".join(stderr_lines[-12:])
        for line in stderr_lines[-12:]:
            print(f"[arbor] agent stderr: {line}", file=sys.stderr, flush=True)
        sdk = f" :: sdk: {' | '.join(sdk_errors[-3:])}" if sdk_errors else ""
        err = AgentLaunchError(f"{e} :: stderr: {trail}{sdk}")
        err.arbor_started = started
        raise err from e

    if not text_parts and sdk_errors:
        # A turn with no word of the agent's that ended on the CLI's error alone is a start
        # that failed, whether or not the CLI then exited non-zero.
        err = AgentLaunchError(f"the agent's turn ended in an error :: sdk: {' | '.join(sdk_errors[-3:])}")
        err.arbor_started = False
        raise err

    # NEWLINE, not "". Separate text blocks are separate chunks and a block does
    # not reliably end in one. Gluing them ran the last line of one block into
    # the first line of the next, so a `CLASS: PROBE` that started its own block
    # landed MID-LINE -- invisible to the anchored parsers below. v2 iteration 1
    # declared CLASS: PROBE, was read as the default B, missed the read-only
    # probe path, and was recorded as "REVERT: no changes".
    return "\n".join(text_parts)


def _run_phase(coro, phase: str, config: dict):
    """``asyncio.run(coro)``: one agent turn (``phase`` is agent, server, bench or critic).

    Standalone: exactly that, with no timeout. Component mode: bounded by
    component.phase_seconds, the phase's cap or the time left before the deadline,
    whichever is less; past it the turn is cancelled and PhaseTimeout raised, which
    the caller books as NOT MEASURED. MB saw one unbounded agent turn hang for 511
    minutes, and a session iterating an hour after its deadline.
    """
    timeout = component.phase_seconds(config, phase)
    if timeout is None:
        return asyncio.run(coro)
    component.phase(phase)
    if timeout <= 0:
        coro.close()
        raise component.PhaseTimeout(phase, 0, f"{phase} phase not started: no time is left "
                                               "before the deadline (component mode)")
    t0 = time.monotonic()
    try:
        return asyncio.run(asyncio.wait_for(coro, timeout))
    except TimeoutError:   # asyncio.TimeoutError is TimeoutError from Python 3.11
        if time.monotonic() - t0 < timeout * 0.99:
            raise          # the agent's own, not the phase's
        raise component.PhaseTimeout(phase, timeout) from None


def _write_aborted(session_dir_p: Path, msg: str) -> None:
    """ABORTED.md, which ends the run: a broken harness, or, when the last agent start could
    not reach its model, component.UNREACHABLE_HEAD, on which a component run exits 3 so the
    host resumes it once the model answers again (a gateway outage, Cluster C 2026-10-29)."""
    if _unreachable:
        text = (f"{component.UNREACHABLE_HEAD}\n\nThe model could not be reached: "
                f"{_unreachable}\n{msg}\n")
    else:
        text = f"# Harness failure\n\n{msg}\n"
    (session_dir_p / "ABORTED.md").write_text(text)


def _not_measured(state, session_dir_p: Path, iteration: int, change: str, reason: str,
                  claim_class: str = "B", findings: list | None = None) -> dict:
    """Book the iteration NOT MEASURED, as the gate's INVALID path does: tree preserved,
    counters unchanged, ABORTED.md after too many in a row."""
    print(f"[arbor] NOT MEASURED — {reason}. Tree preserved, counters unchanged.",
          file=sys.stderr, flush=True)
    state.record_invalid(iteration=iteration, change=change, tier=state.tier, reason=reason,
                         claim_class=claim_class, findings=findings or [])
    state.save(session_dir_p / "session.json")
    if state.harness_broken():
        msg = (f"{state.invalid_streak} consecutive not-measured iterations "
               f"— the harness is broken, not the campaign. Last: {reason}")
        print(f"[arbor] ABORT: {msg}", file=sys.stderr, flush=True)
        _write_aborted(session_dir_p, msg)
    return {"iteration": iteration, "decision": "INVALID", "metric": None, "change": change,
            "claim_class": claim_class, "findings": findings or [], "reason": reason}


def _parse_critic_decision(output: str) -> tuple[str, str]:
    """Parse the critic's verdict on the gate's decision.

    The critic can only UPHOLD or downgrade to REVERT — it never produces a
    KEEP, because the gate owns the metric rule. A critic that says nothing
    parseable is treated as UPHOLD: the gate already applied every mechanical
    rule, so a silent critic must not be able to discard a legitimate result.
    Older prompts emitted ``KEEP``; accept it as a synonym for UPHOLD.
    """
    decision = "UPHOLD"
    reason = "critic produced no objection"
    for line in output.strip().splitlines():
        line = line.strip()
        m = re.match(r'DECISION:\s*(UPHOLD|KEEP|REVERT|RECHECK)', line, re.IGNORECASE)
        if m:
            raw = m.group(1).upper()
            decision = "UPHOLD" if raw in ("UPHOLD", "KEEP") else raw
        m = re.match(r'REASON:\s*(.*)', line, re.IGNORECASE)
        if m:
            reason = m.group(1).strip()
    return decision, reason


_MARKDOWN_DECORATION = "*_`>#- \t"


def _last_verdict(output: str, *tokens: str) -> tuple[str, str] | None:
    """The last line of ``output`` that starts with one of ``tokens``: (token, what follows it).

    A verdict is a line of its own. `_run_claude` returns every text block of the turn
    joined, so a token mentioned in passing ("once it is healthy I'll report SERVER_READY
    or SERVER_FAILED") is not one: a substring match on that interim status reverted a
    correct, much faster kernel under Hyperloom.
    Markdown around the line (backticks, bold, a quote or list marker) is ignored, and the
    token must end there (``METRICS`` is not ``METRIC``).
    """
    found = None
    for raw in (output or "").splitlines():
        line = raw.strip().lstrip(">-*` \t")
        for token in tokens:
            if line.upper().startswith(token):
                rest = line[len(token):]
                if rest and rest[0] not in ": \t*`":
                    continue
                found = (token, rest.lstrip(":*` \t").rstrip("*` \t"))
                break
    return found


def _server_failure(server_output: str) -> str | None:
    """Why the server agent gave up (its last verdict is ``SERVER_FAILED``), or None."""
    verdict = _last_verdict(server_output, "SERVER_READY", "SERVER_FAILED")
    if verdict is None or verdict[0] != "SERVER_FAILED":
        return None
    return verdict[1][:500] or "the server agent reported SERVER_FAILED without a reason"


def _bench_failure(bench_output: str) -> str | None:
    """Why the bench agent gave up (its last verdict is ``BENCH_FAILED``, not a later
    ``METRIC:``), or None."""
    verdict = _last_verdict(bench_output, "METRIC", "BENCH_FAILED")
    if verdict is None or verdict[0] != "BENCH_FAILED":
        return None
    return verdict[1][:300] or "the bench agent reported BENCH_FAILED without a reason"


def _extract_change(output: str) -> str | None:
    """The agent's DONE line: the iteration's `change` in session.json.

    Agents decorate it the way they decorate INTENT (`**DONE: ...**`,
    `- DONE: ...`, `**DONE:** ...`), and _extract_class already reads through
    that. This did not: a decorated line parsed as nothing, and the iteration
    was recorded as "iteration N". Two campaigns' first iterations lost their
    change text that way.
    """
    for line in reversed(output.strip().splitlines()):
        line = line.strip().strip(_MARKDOWN_DECORATION)
        if line.lower().startswith("done"):
            return line[4:].lstrip(":" + _MARKDOWN_DECORATION).rstrip(_MARKDOWN_DECORATION) or None
    return None


def _extract_bundle(output: str) -> str | None:
    """The agent's `BUNDLE: <id>` line, a catalogued bundle (arbor/measure/bundles.py).

    Read through the same decoration as DONE (`**BUNDLE:** \\`id\\``); the colon
    is required, so prose that says "bundle the two knobs" is not a declaration.
    """
    for line in reversed(output.strip().splitlines()):
        line = line.strip().strip(_MARKDOWN_DECORATION)
        m = re.match(r"BUNDLE\s*[*_]*\s*:[\s*_`]*([A-Za-z0-9][\w.-]*)", line, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    return None


def _settings_pair(container: str, workspace: str, config: dict) -> tuple[dict, dict]:
    """The arm's settings before and after its commit (knobs.settings): arbor/config.json
    at HEAD~1 and HEAD over the campaign config. A file absent at HEAD~1 (re-seeded
    after a revert) reads as the campaign's own."""
    from arbor import knobs
    from arbor.execute.workspace import config_at
    return (knobs.settings(config, config_at(container, workspace, "HEAD~1")),
            knobs.settings(config, config_at(container, workspace, "HEAD")))


def _check_switches(state, config: dict, container: str, workspace: str, session_dir: Path,
                    iter_dir: Path, iteration: int, agent_output: str, git_diff: str,
                    pair: tuple[dict, dict] | None) -> str:
    """S4 in the harness, at a KEEP (arbor/measure/switches.py): "" when the KEEP stands, else
    why it is a REVERT. The record goes to iter_N/switches/, the row, and, while the KEEP
    stands, ``<session>/switches.json``."""
    from arbor import knobs
    from arbor.execute.bench import _live_env, run_correctness
    from arbor.execute.workspace import run_in
    from arbor.measure import accuracy as _accuracy
    from arbor.measure import switches as _switches

    settings = pair[1] if pair else knobs.settings(config, None)
    launch_env = {k: str(v) for k, v in (config.get("server_env") or {}).items()}

    def existed(name: str) -> bool:
        r = run_in(container, ["git", "grep", "-q", "-F", "-e", name, "HEAD~1", "--"],
                   workdir=workspace, capture_output=True, text=True)
        return r.returncode != 1   # only "not found" proves it new

    def jit_files(cfg: dict) -> set[str]:
        roots = _switches.jit_roots({**_live_env(cfg), **found})
        r = run_in(container, ["bash", "-c", _switches.jit_command(roots)],
                   capture_output=True, text=True, timeout=300)
        return {line for line in r.stdout.splitlines() if line.strip()}

    found, refusals = _switches.landed(agent_output, git_diff, settings, launch_env, existed)
    if not found and not refusals:
        return ""
    refusals += _switches.default_on(found, settings, launch_env, _switches.code_defaults(git_diff))
    if refusals:
        rec = {"switches": found, "verdict": "fail", "reasons": refusals}
        _switches.save(iter_dir, rec)
    else:
        print(f"[arbor] S4: serving {', '.join(f'{k}={v}' for k, v in sorted(found.items()))} "
              f"for the accuracy check, the JIT caches and the dispatch log", file=sys.stderr, flush=True)
        rec = _switches.check(
            config, iter_dir, found,
            gsm8k=_accuracy.gsm8k_workload(config, state.policy)[0],
            tolerance=state.policy.accuracy_tolerance,
            start=lambda cfg, env: _start_server(cfg, extra_env=env), healthy=_wait_healthy,
            stop=lambda cfg: _stop_server(cfg, container, cfg.get("gpu_ids")),
            run=run_correctness, jit_files=jit_files,
            read_log=lambda cfg: _server_log_tail(cfg, container, 200_000))
    state.iteration_notes["switches"] = _switches.brief(rec)
    print(f"[arbor] S4: {rec['verdict']} — {'; '.join(rec.get('reasons') or []) or 'every check passed'}",
          file=sys.stderr, flush=True)
    if rec["verdict"] == "fail":
        return "; ".join(rec["reasons"])
    try:
        _switches.remember(session_dir, iteration, rec)
    except OSError as e:   # bookkeeping: the verdict stands
        print(f"[arbor] WARNING: switches.json not written: {e}", file=sys.stderr, flush=True)
    return ""


def _persist_findings(campaign, agent_output: str, iteration: int,
                      scope: dict | None = None) -> list[str]:
    """Write the agent's LEDGER records and return their ids.

    Stamps each record with the iteration and, when it declares ``config_keys``,
    a fingerprint of the config it was measured under — that is what lets a
    later iteration notice the fact has gone stale. A DEAD_END or FACT also
    gets the iteration's image and operating point (arbor/measure/scope.py),
    unless the agent gave its own.
    """
    from arbor import ledger as _ledger

    findings = _ledger.parse_agent_blocks(agent_output)
    if not findings or not campaign.session_root:
        return []
    lpath = _ledger.ledger_path(campaign.session_root)
    ids: list[str] = []
    for f in findings:
        f.iteration = iteration
        if f.config_keys and not f.config_fingerprint:
            f.config_fingerprint = _ledger.fingerprint(campaign.config, f.config_keys)
        if scope and f.scope is None and f.type in ("DEAD_END", "FACT"):
            f.scope = scope
        _ledger.append(lpath, f)
        ids.append(f.id)
    print(f"[arbor] Ledger: +{len(ids)} record(s) — {', '.join(ids)}",
          file=sys.stderr, flush=True)
    return ids


def _synthesize_dead_end(campaign, iteration: int, change: str, reason: str,
                         gate, claim_class: str, scope: dict | None = None,
                         bundle: dict | None = None) -> list[str]:
    """Write the DEAD_END a rejected iteration owes but did not record.

    A REVERT that emits no `LEDGER:` line teaches the campaign nothing: the
    change is undone, the history row scrolls out of the window after ten more
    iterations, and the next agent is free to try the same thing. The cuda-graph
    campaign is the argument for this — iterations 38, 41 and 44 were reverted,
    and the discovery that all three were NULL (the bar was the maximum of its
    own cohort, unreachable by 6-8 of 10 identical runs) was possible only
    because that agent happened to write ledger entries voluntarily. Nothing in
    the harness required it.

    This record is deliberately worse than one the agent writes. It knows the
    number and the gate's reason; it does not know what the agent now believes
    about why. `revisit_if` says so, so a later iteration reads it as a lead to
    re-open rather than a settled question -- which matters precisely because a
    reverted iteration may have been refused by the instrument rather than by
    its own merits.

    A reverted bundle (arbor/measure/bundles.py) is one dead end, named for the
    bundle: its members were tried together, and none of them alone.
    """
    from arbor import ledger as _ledger

    if not campaign.session_root:
        return []
    if bundle:
        change = f"bundle {bundle['id']} ({', '.join(bundle.get('members', []))}): {change}"

    def _pct(v) -> str:
        return "not measured" if v is None else f"{v:+.2f}%"

    # A REVERT reached WITHOUT a measurement — a correctness failure, or a
    # critic override on an iteration the gate never scored — leaves delta_pct
    # at its 0.0 default. Printing "+0.00%" there would read as a measured
    # null, which is the single most misleading thing this record could say: a
    # later iteration would take it as evidence the change is inert.
    if not gate.correctness_passed:
        what = "accuracy" if reason.startswith("accuracy") else "correctness"
        axes = f"no measurement — {what} failed before the metric was scored"
    else:
        axes = f"{campaign.pack.objective.get('metric', 'primary')} {_pct(gate.delta_pct)}"
        sec_key = campaign.pack.objective.get("measurement", {}).get("secondary_metric")
        if sec_key:
            axes += f", {sec_key} {_pct(gate.secondary_delta_pct)}"

    f = _ledger.Finding(
        id=f"iter-{iteration}-reverted",
        type="DEAD_END",
        claim=f"[{claim_class}] {change or '(no DONE line)'} — REVERTED: {reason}",
        evidence=f"iter_{iteration}/ ({axes})",
        revisit_if=(
            "the agent recorded no reason, so this says only that the change "
            "did not survive the gate on this run -- re-open if the "
            "measurement frame is later shown to have moved, or if the "
            "mechanism is re-derived from a different direction"
        ),
        iteration=iteration,
        scope=scope,
    )
    _ledger.append(_ledger.ledger_path(campaign.session_root), f)
    print(f"[arbor] Ledger: synthesized DEAD_END {f.id} — a REVERT with no "
          f"agent record ({axes})", file=sys.stderr, flush=True)
    return [f.id]


def _finalize_iteration(
    *, state, campaign, iteration: int,
    container: str, workspace: str,
    iter_dir, session_dir_p, arbor_root: str,
    bench_output: dict, correctness: dict,
    change: str, claim_class: str, agent_output: str,
    iter_start_ts: float = 0.0,
    arm: dict | None = None,
    scope: dict | None = None,
) -> dict:
    """Integrity, gate, critic, record. Shared by the full and read-only paths.

    Order matters: the GATE decides first, mechanically, using the claim class.
    The critic runs second and may only downgrade. Previously the critic was the
    only decision-maker while its own prompt claimed a mechanical gate had
    already run — so no component applied the threshold at all.

    ``arm`` is the live server's comparison with the arm's claim
    (arbor/measure/armcheck.py); None where no server ran or the pack turned
    the check off. ``scope`` is the iteration's image and operating point
    (arbor/measure/scope.py), stamped on the dead ends it records.
    """
    # save_patch is used further down to archive a KEEP/PROBE's diff. It was
    # never imported, so the call raised NameError -- and the `except` around it
    # reinterprets ANY failure as "downgrade to REVERT". Iteration 2 therefore
    # recorded REVERT even though the gate returned PROBE and the critic upheld
    # it. Same shape as the snapshot_digests NameError that corrupted iteration
    # 1: an undefined name inside a try whose handler turns a crash into a
    # verdict.
    from arbor.execute.workspace import git, save_landed_patch, save_patch, save_scratch_patch
    from arbor.measure import armcheck, scope as _scope
    from arbor.measure.gate import run_gate

    # The operating point the bench says it ran at, over the configured one.
    if scope is not None:
        scope = _scope.refine(scope, bench_output)
        _scope.write(scope, iter_dir)
        state.iteration_notes["scope"] = scope

    # ── Integrity ───────────────────────────────────────────────────────────
    integrity_ok = True
    integrity_reason = "ok"
    if state.validator_digests:
        current = snapshot_digests(container, list(state.validator_digests.keys()))
        integrity = verify_unchanged(state.validator_digests, current)
        integrity_ok = integrity.ok
        integrity_reason = integrity.reason

    # ── Git diff ────────────────────────────────────────────────────────────
    try:
        git_diff = git(container, workspace, "diff", "HEAD~1", "HEAD")
    except RuntimeError:
        git_diff = "(could not read diff)"

    # ── Out-of-tree artifacts (evaluators, knowledge notes) ─────────────────
    # Written to the pack dir on shared storage, so invisible to git_diff.
    out_of_tree: list = []
    out_of_tree_block = ""
    if iter_start_ts:
        try:
            pack_dir = campaign.pack.pack_dir
            out_of_tree = _out_of_tree_artifacts(pack_dir, iter_start_ts)
            if out_of_tree:
                out_of_tree_block = _render_out_of_tree_block(out_of_tree, pack_dir)
                n = _archive_out_of_tree(out_of_tree, pack_dir, iter_dir)
                print(
                    f"[arbor] Out-of-tree artifacts: {len(out_of_tree)} file(s) "
                    f"in the pack dir, {n} archived to iter_{iteration}/artifacts/",
                    file=sys.stderr, flush=True,
                )
        except Exception as e:  # bookkeeping must never decide a verdict
            print(f"[arbor] out-of-tree artifact scan failed: {e}",
                  file=sys.stderr, flush=True)

    # The arm's settings before and after its commit, for the checks that name
    # settings: a bundle's members, the knobs that change numerics, the switches
    # a KEEP lands. Not on a read-only probe: nothing was served.
    served = bench_output.get("status") != "skipped"
    pair, pair_error = None, ""
    if served:
        try:
            pair = _settings_pair(container, workspace, campaign.config)
        except Exception as e:  # noqa: BLE001
            pair_error = str(e)
            print(f"[arbor] could not read the arm's settings: {e}", file=sys.stderr, flush=True)

    # ── A declared bundle (arbor/measure/bundles.py) ────────────────────────
    # One claim when the pack allows bundles and the arm is the bundle it names;
    # an ordinary change otherwise, and the row says why. Its dead ends are
    # scoped to the bundle.
    bundle = None
    bundle_id = _extract_bundle(agent_output)
    if bundle_id and served:
        if state.policy.bundles:
            from arbor.measure import bundles as _bundles
            if pair is None:   # an unproven bundle is not the bundle
                bundle = {"id": bundle_id, "members": [], "ok": False,
                          "reason": f"bundle {bundle_id}: could not be checked ({pair_error})"}
            else:
                bundle = _bundles.check(bundle_id, *pair, arm)
            scope = {**(scope or {}), "bundle": bundle_id}
        else:
            bundle = {"id": bundle_id, "enabled": False,
                      "reason": "bundles not enabled in this pack; gated as an ordinary change"}
        state.iteration_notes["bundle"] = {k: bundle[k] for k in ("id", "members", "enabled")
                                           if k in bundle}
        print(f"[arbor] Bundle {bundle_id}: {bundle['reason']}", file=sys.stderr, flush=True)
    claimed_bundle = bundle if state.policy.bundles else None

    # ── Ledger: persist what the agent learned, whatever the verdict ────────
    # Done before the gate on purpose. A REJECT that leaves behind a measuring
    # instrument and a number nobody had before is a successful iteration, and
    # the pack says to record it that way — so the record must not be
    # conditional on the change surviving.
    finding_ids = _persist_findings(campaign, agent_output, iteration, scope)

    # ── The arm, as the live server shows it ────────────────────────────────
    # On the row whatever the mode, so the history and the critic see it; the
    # gate refuses on it only when the pack enforces it.
    arm_mode = state.policy.arm_check
    if arm:
        state.iteration_notes["arm_check"] = armcheck.brief(arm)

    def decide(acc: dict | None):
        return run_gate(
            campaign.pack.objective, state.best_metric,
            bench_output, correctness, integrity_ok,
            claim_class=claim_class,
            secondary_best=state.best_secondary,
            arm=arm if arm_mode == "enforce" else None,
            bundle=claimed_bundle,
            accuracy=acc,
        )

    # ── Accuracy (arbor/measure/accuracy.py) ────────────────────────────────
    # The full eval set, on a GSM8K workload, when the change moved a numerics
    # knob, turned on a switch a KEEP landed off, or the agent said NUMERICS, and
    # in component mode at every KEEP; only where the gate would judge the arm on
    # it: not after a failed correctness check, a moved validator, or a refused
    # arm or bundle.
    from arbor.measure import accuracy as _accuracy
    from arbor.measure import switches as _switches
    accuracy = None
    if served and _accuracy.gsm8k_workload(campaign.config, state.policy)[0]:
        from arbor import knobs as _knobs
        on = _switches.turned_on(session_dir_p, *pair) if pair else []
        by = _accuracy.trigger(_knobs.changes(*pair) if pair else {}, agent_output, on)
        refused = (not integrity_ok or not correctness.get("passed")
                   or (arm_mode == "enforce" and (arm or {}).get("status") == "mismatch")
                   or (claimed_bundle is not None and not claimed_bundle.get("ok")))
        # Component mode: the gate decides without the eval first, so only an arm it
        # would keep pays for the full set.
        if (not by and not refused and _accuracy.every_keep(campaign.config, state.policy)
                and decide(None).decision == "KEEP"):
            by = [_accuracy.EVERY_KEEP]
        if by and not refused:
            print(f"[arbor] Accuracy gate: full GSM8K for {', '.join(by)}", file=sys.stderr, flush=True)
            try:
                accuracy = _accuracy.measure(campaign.config, iter_dir, by,
                                             state.policy.accuracy_tolerance)
            except Exception as e:  # noqa: BLE001 - an eval that did not run measured nothing
                accuracy = {"triggered": True, "by": by, "score": None,
                            "error": f"accuracy eval failed: {e}"}
            state.iteration_notes["accuracy"] = _accuracy.brief(accuracy)
            print(f"[arbor] Accuracy: {accuracy.get('score')} against {accuracy.get('golden')} "
                  f"({accuracy.get('source')})", file=sys.stderr, flush=True)

    # ── Gate decides ────────────────────────────────────────────────────────
    gate = decide(accuracy)
    decision, reason = gate.decision, gate.reason
    print(f"[arbor] Gate: {decision} ({claim_class}) — {reason}", file=sys.stderr, flush=True)

    # ── Not measured ────────────────────────────────────────────────────────
    # Short-circuit before the critic. There is nothing for a critic to judge:
    # the harness failed, so no evidence about the change exists. The tree
    # stands, the counters do not move, and the next iteration retries.
    if decision == "INVALID":
        print(f"[arbor] NOT MEASURED — tree preserved, counters unchanged, "
              f"iteration will be retried", file=sys.stderr, flush=True)
        state.record_invalid(
            iteration=iteration, change=change, tier=state.tier,
            reason=reason, claim_class=claim_class, findings=finding_ids,
        )
        state.save(session_dir_p / "session.json")
        if state.harness_broken():
            msg = (f"{state.invalid_streak} consecutive not-measured iterations "
                   f"— the harness is broken, not the campaign. Last: {reason}")
            print(f"[arbor] ABORT: {msg}", file=sys.stderr, flush=True)
            _write_aborted(session_dir_p, msg)
        return {"iteration": iteration, "decision": "INVALID", "metric": None,
                "change": change, "claim_class": claim_class,
                "findings": finding_ids, "reason": reason}

    # A PROBE whose whole point was to close an uncertainty but that recorded
    # nothing has produced no deliverable. The critic would catch this, but the
    # check is mechanical so it should not cost a model call.
    if decision == "PROBE" and not finding_ids:
        decision = "REVERT"
        reason = "probe closed no ledger entry — nothing was learned"
        print(f"[arbor] Gate override: {reason}", file=sys.stderr, flush=True)

    # ── Critic may downgrade ────────────────────────────────────────────────
    if decision != "REVERT":
        print(f"[arbor] Running critic...", file=sys.stderr, flush=True)
        critic_prompt = _build_critic_prompt(
            state, iteration, git_diff, bench_output,
            correctness, integrity_ok, integrity_reason,
            threshold=campaign.pack.objective.get("measurement", {}).get("threshold_rel", 1.01),
            claim_class=claim_class, change=change,
            gate_decision=decision, gate_reason=reason,
            constraints=campaign.pack.constraints,
            agent_output=agent_output,
            out_of_tree_block=out_of_tree_block,
            arm_block=armcheck.critic_block(arm, arm_mode),
            bundle=bundle, accuracy=accuracy,
        )
        try:
            critic_output = _run_phase(
                _run_claude(critic_prompt, arbor_root,
                            allowed_tools=["Read", "Bash"],
                            label="critic"),
                "critic", campaign.config,
            )
            verdict, critic_reason = _parse_critic_decision(critic_output)
        except Exception as e:
            # A critic crash must not discard a result the gate already
            # accepted on measured evidence. Uphold and say so. A critic that ran
            # out of time (component mode's PhaseTimeout) is the same: the gate
            # measured the arm, and the host gates what Arbor keeps again.
            verdict, critic_reason = "UPHOLD", f"critic unavailable ({e}) — gate verdict stands"
            print(f"[arbor] Critic error: {e}", file=sys.stderr, flush=True)

        print(f"[arbor] Critic: {verdict} — {critic_reason}", file=sys.stderr, flush=True)
        if verdict == "REVERT":
            decision = "REVERT"
            reason = f"critic override: {critic_reason}"

    # ── S4: the switches a KEEP lands (arbor/measure/switches.py) ───────────
    # Off by default, served on with the accuracy check, no JIT while serving, a
    # dispatch line from every rank. Before the short-prompt check, whose second
    # launch serves the tree as kept again. A document workload serves nothing.
    if (decision == "KEEP" and served
            and (getattr(campaign.pack, "workload", None) or {}).get("kind") != "document"):
        refusal = _check_switches(state, campaign.config, container, workspace, session_dir_p,
                                  iter_dir, iteration, agent_output, git_diff, pair)
        if refusal:
            decision = "REVERT"
            reason = f"S4: {refusal}"

    # ── Component mode: the short-prompt check at a KEEP ────────────────────
    # A second process of the same tree, probed and compared with the first: a
    # disagreement above the session's floor is the defect GSM8K did not see
    # (arbor/measure/short_prompts.py). One that cannot finish refuses
    # nothing; the host re-gates every KEEP.
    if decision == "KEEP" and component.is_component(campaign.config):
        from arbor.measure import short_prompts as _sp
        try:
            sp = _sp.check_keep(
                campaign.config, session_dir_p, iter_dir,
                relaunch=lambda: _relaunch_own_server(campaign.config),
                probe_seconds=lambda: component.phase_seconds(campaign.config, "short_prompts"))
        except Exception as e:
            sp = {"verdict": "not_run", "reason": f"the check failed: {e}"}
        # The second process's server log, before the next launch overwrites it.
        log = component.server_paths(component.session_root(campaign.config))[1]
        if (iter_dir / "short_prompts_2.json").is_file() and log.is_file():
            try:
                import shutil
                shutil.copyfile(log, iter_dir / "server_2.log")
            except OSError as e:
                print(f"[arbor] WARNING: second launch's server log not kept: {e}",
                      file=sys.stderr, flush=True)
        state.iteration_notes["short_prompts"] = _sp.brief(sp)
        print(f"[arbor] Short prompts at KEEP: {sp['verdict']} — {sp.get('reason', '')}",
              file=sys.stderr, flush=True)
        if sp["verdict"] == "defect":
            decision = "REVERT"
            reason = f"short prompts: {sp['reason']}"

    # ── A REVERT owes a record ──────────────────────────────────────────────
    # Placed after the critic so it captures the FINAL reason, including a
    # critic override. Bookkeeping, so a failure here must never change the
    # verdict -- that mistake has been made three times in this function.
    if decision == "REVERT" and not finding_ids:
        try:
            finding_ids = _synthesize_dead_end(
                campaign, iteration, change, reason, gate, claim_class, scope,
                bundle=claimed_bundle)
        except Exception as e:
            print(f"[arbor] DEAD_END synthesis failed: {e}",
                  file=sys.stderr, flush=True)

    # ── Apply ───────────────────────────────────────────────────────────────
    metric_key = campaign.pack.objective.get("metric", "output_throughput")
    metric_value = bench_output.get(metric_key, 0.0)
    if not isinstance(metric_value, (int, float)):
        metric_value = 0.0

    if decision in ("KEEP", "PROBE"):
        # Archiving the diff is BOOKKEEPING, not judgement. It must never
        # overturn a verdict the gate reached and the critic upheld -- and it has
        # now done so three times:
        #   iter 1  NameError: snapshot_digests   -> spurious REVERT
        #   iter 2  NameError: save_patch          -> spurious REVERT
        #   iter 3  save_patch: "not a git repository" (the server-start step ran
        #           `docker rm -f` + `docker run`, recreating the container and
        #           destroying the repo mid-iteration) -> spurious REVERT
        # Each time the pattern was identical: an infrastructure failure inside a
        # try whose handler rewrites it as a scientific conclusion. Record the
        # failure, keep the verdict.
        try:
            save_patch(container, workspace, str(iter_dir / "patch.diff"))
        except Exception as e:
            print(f"[arbor] WARNING: save_patch failed: {e}", file=sys.stderr)
            print(f"[arbor] keeping {decision} — patch archival does not affect the verdict",
                  file=sys.stderr)
            reason = f"{reason} [patch not archived: {e}]"
        # The agent's scratch (arbor/eval/ probes and evaluators) beside landed.patch,
        # which no longer carries it: both verdicts keep the commit that holds it.
        try:
            save_scratch_patch(container, workspace,
                               str(session_dir_p / "scratch.patch"), campaign.config)
        except Exception as e:
            print(f"[arbor] WARNING: scratch.patch not written: {e}", file=sys.stderr)

    if decision == "KEEP":
        # Mirror the CUMULATIVE landed tree to shared storage. The repo is
        # node-local and dies with the container; without this, every KEEP dies
        # with it. Same rule as above — bookkeeping never overturns a verdict.
        try:
            save_landed_patch(container, workspace,
                              str(session_dir_p / "landed.patch"), campaign.config)
        except Exception as e:
            print(f"[arbor] WARNING: landed.patch not written: {e} — the landed "
                  f"tree is NOT recoverable if this container dies",
                  file=sys.stderr)
        # The other editable packages (AITER) beside it: landed.patch is the workspace
        # only, and a KEEP's AITER edit the agent never wrote into landed_aiter.py died
        # with its container (arbor/execute/packages.py). Bookkeeping too.
        try:
            from arbor.execute import packages as _packages
            kept = _packages.capture(container, campaign.config, session_dir_p)
            if kept:
                print("[arbor] landed packages: " + ", ".join(
                    f"{n} {len(v['files'])} file(s)" for n, v in kept.items()), flush=True)
        except Exception as e:
            print(f"[arbor] WARNING: landed packages not captured: {e} — an edit outside "
                  f"the workspace is NOT recoverable if this container dies", file=sys.stderr)
        # What the served tree links to outside itself (c0's AITER JIT modules) and how it
        # is rebuilt: recorded, never stored (arbor/execute/outside.py). Bookkeeping too.
        try:
            from arbor.execute import outside as _outside
            doc = _outside.record(container, workspace, campaign.config,
                                  session_dir_p / _outside.RECORD)
            if doc["counts"]["links"]:
                print(f"[arbor] outside the tree: {doc['counts']}", flush=True)
        except Exception as e:
            print(f"[arbor] WARNING: outside.json not written: {e}", file=sys.stderr)

    if decision == "KEEP":
        if component.is_component(campaign.config):
            # Component mode: the KEEP's snapshot (docs/component-mode.md §4), written
            # BEFORE record_keep, so it is whole before the KEEP count the host polls in
            # session.json moves. Bookkeeping, like landed.patch: a failure is logged
            # and the verdict stands; the next snapshot's `previous` shows the gap.
            try:
                component.snapshot_keep(
                    campaign.config, session_dir_p, container=container, workspace=workspace,
                    number=component.keep_count(state.lessons) + 1, iteration=iteration,
                    metric=metric_value, change=change, intent=resolve(claim_class).name)
            except Exception as e:
                print(f"[arbor] WARNING: KEEP snapshot not written: {e} — the host reads this "
                      f"KEEP from the live workspace", file=sys.stderr, flush=True)
        # Did this KEEP actually move the metric, or did it ship on evidence?
        # The gate distinguishes these; only a real advance resets the
        # escalation counter.
        # The gate already decided whether this cleared the BOOKING bar; do not
        # re-derive it here against a different threshold. `booked` is what
        # resets the escalation counter — a change kept for clearing the noise
        # band, or kept as a frontier trade, is retained but is not a claimed
        # advance.
        state.record_keep(
            iteration=iteration, metric=metric_value,
            change=change, tier=state.tier,
            claim_class=claim_class, findings=finding_ids,
            advanced_metric=gate.booked,
            secondary=gate.secondary_value,
            frontier=gate.frontier,
        )
    elif decision == "PROBE":
        # The tree is NOT reverted. A probe's deliverable — a new evaluator, an
        # instrument, a ledger entry — is exactly what we want to keep.
        state.record_probe(
            iteration=iteration, change=change, tier=state.tier,
            findings=finding_ids, reason=reason, metric=metric_value,
        )
    else:
        state.record_revert(
            iteration=iteration, metric=metric_value,
            change=change, tier=state.tier, reason=reason,
            claim_class=claim_class, findings=finding_ids,
            # Only the GATE's own reading counts toward the suspect streak. A
            # critic override is a judgement about the diff, not a measurement
            # of the second axis, so it must not accumulate evidence against
            # the incumbent.
            secondary_decided=(gate.secondary_decided
                               and gate.verdict == "REVERT"),
        )
        state.save(session_dir_p / "session.json")
        if gate.revert_tree:
            _safe_revert(container, workspace)

    state.save(session_dir_p / "session.json")

    result = {
        "iteration": iteration,
        "decision": decision,
        "metric": metric_value,
        "change": change,
        "claim_class": claim_class,
        "findings": finding_ids,
        "critic_reason": reason,
    }
    print(json.dumps(result), flush=True)
    return result


def _extract_class(output: str) -> str:
    """Read the agent's declared intent, in the legacy `claim_class` spelling.

    WIN (`B`) is the safe default because it is the strictest bar — an agent
    that fails to declare gets held to the metric threshold rather than
    slipping through the evidence-gated path.

    Both keywords (`INTENT:` and the older `CLASS:`) and both vocabularies
    (`WIN`/`FIX`/`FIX-CORRECTNESS`/`LEARN` and `B`/`A`/`A-CORRECTNESS`/`PROBE`)
    are accepted, because a campaign resumed mid-flight carries agent habits
    from before the rename and a declaration that fails to parse is charged the
    strictest bar. The alternation comes from ``intents.ALL_NAMES``, sorted
    longest-first — regex alternation is first-match-wins, so without that
    ordering `FIX-CORRECTNESS` would match as `FIX` and a correctness fix would
    silently lose its catastrophe cap.

    Two passes. The anchored pass is authoritative. The fallback finds a
    declaration that is decorated or run into surrounding text -- `**INTENT:
    LEARN**`, `- CLASS: A`, or a block boundary that glued it mid-line. Falling
    back because of formatting silently applies the STRICTEST bar to work that
    declared an evidence-gated intent, which is the worst direction for the
    error to go: it cost v2 iteration 1 its probe path and recorded a
    knowledge-producing iteration as a REVERT.
    """
    lines = list(reversed(output.strip().splitlines()))
    _names = "|".join(re.escape(n) for n in ALL_NAMES)
    pattern = rf'(?:INTENT|CLASS):\s*\(?({_names})\)?(?![\w-])'
    for line in lines:
        m = re.match(pattern, line.strip(), re.IGNORECASE)
        if m:
            return _as_claim_class(m.group(1))
    for line in lines:
        m = re.search(pattern, line, re.IGNORECASE)
        if m:
            return _as_claim_class(m.group(1))
    return _as_claim_class(None)


def _as_claim_class(name: str | None) -> str:
    """Normalise either vocabulary to the legacy string, via the registry.

    Returning the alias rather than the new name is deliberate and permanent:
    `claim_class` is what lands in session.json, and
    check_review_consistency.py reads those strings out of four archived
    campaigns. The rename is a change of what the agent WRITES, not of what the
    campaign RECORDS.
    """
    return resolve(name).aliases[0]


# Paths a read-only PROBE may touch. Anything here is incapable of changing
# serving behaviour, so the 30-minute server run buys nothing and is skipped.
# `arbor/config.json` is written by the HARNESS on every iteration, not by the
# agent. Counting it as a non-inert path forced every probe through a full
# ~45-minute server+bench cycle, making a read-only probe cost exactly as much
# as a real change — part of why the dsv41flash campaign stopped probing.
READONLY_PROBE_PREFIXES = ("eval/", "knowledge/", "ledger.jsonl", "arbor/")


OUT_OF_TREE_ARTIFACT_DIRS = ("eval", "knowledge")
MAX_OUT_OF_TREE_ARTIFACTS = 25


def _out_of_tree_artifacts(pack_dir, since_ts: float) -> list[tuple[Path, int]]:
    """Files written under the pack's eval/ and knowledge/ dirs this iteration.

    These live on SHARED STORAGE, outside the container's git workspace, so they
    never appear in the diff the critic reviews.

    Job 38943 iteration 2 is why this exists. The agent found a genuine ROCm
    feature gap, wrote a 22 KB grouped ``tl.dot_scaled`` MXFP8 BMM evaluator,
    swept it against the BF16 einsum on real weights, and recorded seven ledger
    findings. The gate returned PROBE. The critic then reverted it with "the
    diff does not implement the claim" -- correctly, on the evidence it was
    shown, because the only thing in the workspace diff was arbor/config.json.
    A probe whose entire deliverable is an instrument is structurally invisible
    to the critic without this, and the pack was rewritten to ENCOURAGE probes.
    """
    found: list[tuple[Path, int]] = []
    for sub in OUT_OF_TREE_ARTIFACT_DIRS:
        d = Path(pack_dir) / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            try:
                if not p.is_file():
                    continue
                st = p.stat()
            except OSError:
                continue
            if st.st_mtime >= since_ts:
                found.append((p, st.st_size))
    return found[:MAX_OUT_OF_TREE_ARTIFACTS]


def _render_out_of_tree_block(artifacts, pack_dir) -> str:
    """Critic-facing summary of out-of-tree artifacts, with a head of each file."""
    if not artifacts:
        return ""
    lines = []
    for p, size in artifacts:
        try:
            rel = p.relative_to(Path(pack_dir))
        except ValueError:
            rel = p
        lines.append(f"- `{rel}` ({size} bytes)")
        if p.suffix in (".py", ".json", ".md", ".txt"):
            try:
                head = p.read_text(errors="replace")[:1200]
            except OSError:
                continue
            lines.append(f"\n```\n{head}\n```\n")
    return (
        "### Out-of-tree artifacts written this iteration\n\n"
        "These files are part of the deliverable. They live in the pack "
        "directory on shared storage, NOT in the git workspace, so they are "
        "absent from the diff below BY DESIGN — evaluators and knowledge notes "
        "belong here. Do not treat their absence from the diff as evidence the "
        "work was not done.\n\n"
        + "\n".join(lines) + "\n\n"
    )


def _archive_out_of_tree(artifacts, pack_dir, iter_dir) -> int:
    """Copy this iteration's out-of-tree artifacts into iter_N/artifacts/.

    The pack dir is mutable and shared; the per-iteration copy is what makes an
    instrument auditable after the fact. Never raises -- this is bookkeeping.
    """
    n = 0
    dest_root = Path(iter_dir) / "artifacts"
    for p, _ in artifacts:
        try:
            rel = p.relative_to(Path(pack_dir))
        except ValueError:
            rel = Path(p.name)
        dest = dest_root / rel
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
            n += 1
        except OSError:
            continue
    return n


def _probe_is_readonly(paths: list[str]) -> bool:
    """True when every changed path is inert with respect to the served model."""
    if not paths:
        return False
    return all(
        any(p.startswith(pre) or f"/{pre}" in p for pre in READONLY_PROBE_PREFIXES)
        for p in paths
    )


# ── Server lifecycle (harness-owned) ────────────────────────────────────────

def _server_launch_cmd(config: dict, server_flags: str | None = None,
                       server_env: dict | None = None) -> str:
    """Build the framework-appropriate server launch command.

    This used to hardcode `python3 -m sglang.launch_server`, so a campaign with
    `"framework": "vllm"` — like dsv41flash — failed at server start on every
    iteration. The framework is declared in the config; honour it. The command
    itself is that framework's adapter's (arbor/frameworks/).

    `server_flags` defaults to the LIVE flags (workspace arbor/config.json over
    the static config), not to the static config alone. Reading the static
    config here is what silently discarded every flag-valued change an agent
    made — see arbor.execute.bench._live_flags for the full account. Callers
    that already resolved the flags pass them in to avoid a second container
    read.

    `server_env` is the same for the env, and the command carries it as
    `env K=V ...` in front of the server. It used to carry none: the server
    sub-agent launched with `docker exec <c> bash -c '<cmd>'`, so the server ran
    on the container's own env while the baseline was launched with
    `-e server_env` (scripts/model-m-campaign/baseline_model_m.py), and an agent's env
    change never reached it (arbor/measure/armcheck.py records what does).

    In component mode (arbor/component.py) the command never carries a GPU mask
    (the host sets ROCR_/HIP_/CUDA_VISIBLE_DEVICES; MB found Arbor's servers on
    cards outside its lease), starts the server in a process group of its own,
    and writes its PID and log to ``<session>/server/``: what the harness stops is
    that group and nothing else.
    """
    from arbor.execute.bench import _live_env, _live_flags
    if server_flags is None:
        server_flags = _live_flags(config)
    if server_env is None:
        server_env = _live_env(config)
    if component.is_component(config):
        pid_file, log = component.server_paths(component.session_root(config))
        return frameworks.get(config).launch_cmd(config, component.strip_masks(server_env),
                                                 server_flags, log_path=str(log),
                                                 pid_file=str(pid_file))
    return frameworks.get(config).launch_cmd(config, server_env, server_flags)


def _start_server(config: dict, extra_env: dict | None = None) -> int:
    """Start the inference server inside the container. Returns the server PID.

    In component mode on this host instead, with no GPU selection of its own (the
    host's masks hold), and the KV trace in the session rather than /tmp.
    ``extra_env`` is set over the arm's env: the switches a KEEP lands, served on
    (arbor/measure/switches.py).
    """
    container = config.get("docker_container", "")
    port = config.get("port", 8888)
    comp = component.is_component(config)

    from arbor.execute.bench import _live_env, _live_flags

    # Read flags the same way env is read: workspace arbor/config.json wins.
    # Reading the static config here is what silently discarded every
    # flag-valued change an agent made — see _live_flags for the full account.
    server_flags = _live_flags(config)
    env = {**_live_env(config), **(extra_env or {})}

    # The GPU selection (ROCR_VISIBLE_DEVICES alone, never HIP's too), then the
    # env the flags call for (HiCache's KV trace): arbor/frameworks/base.py says
    # why each is what it is.
    fw = frameworks.get(config)
    if not comp:
        env = fw.device_env(env, ",".join(map(str, component.gpu_list(config.get("gpu_ids")) or [])))
    env.update(fw.extra_env(server_flags))
    if comp:
        server_dir = component.server_paths(component.session_root(config))[0].parent
        server_dir.mkdir(parents=True, exist_ok=True)
        if "SGLANG_KV_TRACE_PATH" in env:
            env["SGLANG_KV_TRACE_PATH"] = str(server_dir / "kv_trace.jsonl")

    # The command carries the env (`env K=V ...`), as the server sub-agent's does.
    launch_cmd = _server_launch_cmd(config, server_flags, env)

    r = subprocess.run(
        ["bash", "-c", launch_cmd] if comp else ["docker", "exec", container, "bash", "-c", launch_cmd],
        capture_output=True, text=True, timeout=30,
    )
    pid_str = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "0"
    try:
        return int(pid_str)
    except ValueError:
        return 0


def _collect_kv_trace(container: str, iter_dir: Path, arbor_root: str,
                      local: Path | None = None) -> None:
    """Copy KV trace from container after bench, run analyzer, save results.

    ``local``: component mode's trace, a file in the session on this host (no docker).
    """
    trace_path = iter_dir / "kv_trace.jsonl"
    try:
        if local is not None:
            if not Path(local).is_file():
                return
            import shutil
            shutil.copyfile(local, trace_path)
        else:
            r = subprocess.run(
                ["docker", "cp", f"{container}:/tmp/kv_trace.jsonl", str(trace_path)],
                capture_output=True, timeout=30,
            )
            if r.returncode != 0:
                return
        if not trace_path.exists() or trace_path.stat().st_size == 0:
            return

        print(f"[arbor] KV trace: {trace_path.stat().st_size / 1024:.0f} KB", file=sys.stderr, flush=True)

        analyzer = Path(arbor_root) / "packs" / "agentx" / "tools" / "kv_trace_analyze.py"
        if analyzer.exists():
            analysis_path = iter_dir / "kv_trace_analysis.json"
            r = subprocess.run(
                [sys.executable, str(analyzer), str(trace_path), "--json", str(analysis_path)],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0 and analysis_path.exists():
                print(f"[arbor] KV trace analysis saved to {analysis_path}", file=sys.stderr, flush=True)
            if r.stdout.strip():
                summary_path = iter_dir / "kv_trace_summary.txt"
                summary_path.write_text(r.stdout)
    except Exception as e:
        print(f"[arbor] KV trace collection: {e}", file=sys.stderr, flush=True)


def _relaunch_own_server(config: dict) -> bool:
    """Component mode: stop the server Arbor runs, boot the same tree again, wait for health.

    The short-prompt check's second process (arbor/measure/short_prompts.py), within the
    server phase's cap and the deadline. False when the old one would not stop or the new
    one does not come up in that time.
    """
    from arbor.execute import preflight
    pid_file = component.server_paths(component.session_root(config))[0]
    if preflight.stop_own_server(pid_file).get("status") == "running":
        return False
    cap = component.phase_seconds(config, "server")
    if cap is not None and cap <= 0:
        return False
    component.phase("server")
    if not _start_server(config):
        return False
    return _wait_healthy(config, timeout=int(cap if cap is not None else 600))


def _wait_healthy(config: dict, timeout: int = 600) -> bool:
    port = config.get("port", 8888)
    health = frameworks.get(config).health_path
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ["curl", "-sf", f"http://localhost:{port}{health}"],
                capture_output=True, timeout=5,
            )
            if r.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, OSError):
            pass
        time.sleep(10)
    return False


def _arm_check(state, config: dict, iter_dir, prior: dict | None = None) -> dict | None:
    """Read the live server against the arm's claim (arbor/measure/armcheck.py).

    Taken once the server is healthy and again before the gate, because the
    bench sub-agent may restart the server; the later read wins unless it
    could not read the server at all. Bookkeeping: it never raises.
    """
    from arbor.measure import armcheck
    mode = state.policy.arm_check
    if mode not in armcheck.MODES:
        print(f"[arbor] policy arm_check {mode!r} is not one of {armcheck.MODES}; "
              f"using 'record'", file=sys.stderr, flush=True)
        state.policy.arm_check = mode = "record"
    if mode == "off":
        return None
    try:
        if component.is_component(config):
            # The server this harness launched, by its PID file, on this host.
            pid_file, log = component.server_paths(component.session_root(config))
            arm = armcheck.check(config, iter_dir, pid_file=pid_file, server_log=log)
        else:
            arm = armcheck.check(config, iter_dir)
    except Exception as e:  # noqa: BLE001
        arm = {"status": "unknown", "reason": f"arm check failed: {e}"}
    if arm["status"] == "unknown" and prior and prior.get("status") != "unknown":
        return prior
    print(f"[arbor] Arm check ({mode}): {arm['status']} — {arm.get('reason', '')}"[:600],
          file=sys.stderr, flush=True)
    return arm


# The server agent's steps that do not depend on where the server runs: the same
# text in the container's prompt and in component mode's.
_NEVER_WEAKEN = (
    "3. **Never weaken a performance-affecting flag to get the server up.**\n"
    "   `--max-num-batched-tokens`, `--max-num-seqs`, `--max-model-len`,\n"
    "   `--gpu-memory-utilization` and the speculative config define WHAT IS BEING\n"
    "   MEASURED. The baseline was taken with the configured values; a server that\n"
    "   only starts with smaller ones produces a number that cannot be compared to\n"
    "   it, and reporting that number as the iteration's result is worse than\n"
    "   reporting no number at all. If the configured values crash, say\n"
    "   `SERVER_FAILED: <reason>` and stop. Reducing them to reach a green health\n"
    "   check is never the right answer.\n\n"
)


def _server_ready_steps(config: dict, port) -> str:
    return (
        f"4. Once `/health` returns OK, send 3 real inference requests to confirm the server handles actual load:\n"
        f"   ```bash\n"
        f"   for i in 1 2 3; do\n"
        f"     curl -sf http://localhost:{port}/v1/chat/completions \\\n"
        f"       -H 'Content-Type: application/json' \\\n"
        f"       -d '{{\"model\":\"{config.get('model_name','')}\",\"messages\":[{{\"role\":\"user\",\"content\":\"say hi\"}}],\"max_tokens\":10}}' \\\n"
        f"       && echo OK || echo FAIL\n"
        f"   done\n"
        f"   ```\n"
        f"   If any request fails or the server crashes, investigate the server log, fix the root cause, and restart.\n\n"
        f"5. When health passes AND inference requests succeed, say:\n"
        f"   ```\n"
        f"   SERVER_READY\n"
        f"   ```\n"
        f"   If you cannot get it stable after 3 restart attempts, say:\n"
        f"   ```\n"
        f"   SERVER_FAILED: <reason>\n"
        f"   ```\n"
    )


def _component_kill(pid_file) -> str:
    """The one way an agent stops the server in component mode: its process group."""
    return f"kill -TERM -- -$(cat {shlex.quote(str(pid_file))})"


def _component_rules(config: dict) -> str:
    """What every agent is told in component mode (docs/component-mode.md).

    Each rule is one of MB's sandbox findings: Arbor stopped MB's servers with a
    pattern pkill, ran on cards outside its lease, and loaded kernel tables from
    paths outside the workspace, which the host never merges.
    """
    session = component.session_root(config)
    pid_file, _ = component.server_paths(session)
    host = component.block(config).get("host") or "a host"
    return (
        f"## HARD CONSTRAINTS — Arbor runs inside {host}\n"
        f"The host owns this node: its GPUs, the other servers on it, and the clock. These "
        f"rules override anything the pack's knowledge or a skill says.\n"
        f"- Stop the server only with `{_component_kill(pid_file)}`: its own process group, by "
        f"the PID its launch wrote. Never `pkill`, `killall` or `pgrep` by pattern, and never "
        f"`kill -9`: the other servers on this node are the host's, and a server killed with "
        f"SIGKILL leaks its GPU memory.\n"
        f"- Never run `docker`: there is no container. The server and the bench run on this host.\n"
        f"- Never set `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`, in "
        f"`arbor/config.json` or on a command line: the host sets them for its GPU lease, and "
        f"the harness drops them from the launch.\n"
        f"- Use only port {config.get('port', 8888)}.\n"
        f"- A kernel-table variable (`AITER_CONFIG_*`) whose value is a `:`-separated list was "
        f"given by the host: its default-table copies first, then the start's tuned rows, last. "
        f"Add to it, never replace it. Keep every part in order and append your own rows file, "
        f"holding only shapes the given parts do not (aiter refuses a shape listed twice). To "
        f"retune a shape the start's rows hold, put your own copy in place of the start's rows "
        f"file: those rows less the shapes, plus your new rows. Never drop or edit the "
        f"default-table parts: without the start's rows the first A/B measures a regression.\n"
        f"- Keep every file you write in the workspace (`{config.get('workspace_root', '')}`) or "
        f"the session (`{session}`), never in /tmp or anywhere else: a kernel table or tuning "
        f"file outside the workspace is not there when the host merges the change.\n"
    )


def _component_server_prompt(config: dict, workspace: str, port) -> str:
    """The server agent's prompt in component mode: the launch on this host, its log and
    PID file in the session, and the component rules in place of the container's."""
    pid_file, log = component.server_paths(component.session_root(config))
    return (
        f"The harness has committed your change. Now start the server and confirm it is healthy.\n\n"
        f"{_component_rules(config)}\n"
        f"Workspace: `{workspace}`\n"
        f"Port: {port}\n"
        f"Server log: `{log}`\n"
        f"Server PID file: `{pid_file}`\n\n"
        f"## What to do\n\n"
        f"1. Start the server with this command. It carries the live `server_env` (as `env K=V ...`,\n"
        f"   without GPU masks) and `server_flags` from `{workspace}/arbor/config.json`, starts the\n"
        f"   server in a process group of its own, and writes its PID to the PID file:\n"
        f"   ```bash\n"
        f"   {_server_launch_cmd(config)}\n"
        f"   ```\n"
        f"   If you change `arbor/config.json` to get the server up, rebuild the command the\n"
        f"   same way: every `server_env` entry as `env K=V` in front of the server, every\n"
        f"   flag after it, inside the same `setsid bash -c 'echo $$ > <PID file>; exec ...'`.\n"
        f"   A server started any other way is not the arm being measured, the harness reads\n"
        f"   the live process's env and flags by that PID, and only that PID can be stopped.\n\n"
        f"2. Monitor startup — repeat every 60 seconds until healthy or crashed:\n"
        f"   ```bash\n"
        f"   sleep 60 && \\\n"
        f"     curl -sf http://localhost:{port}/health && echo HEALTHY ; \\\n"
        f"     tail -30 {shlex.quote(str(log))}\n"
        f"   ```\n"
        f"   **Never use a loop that runs many iterations in one bash call — always one sleep+check per tool call so you can read the output and react.**\n\n"
        + _NEVER_WEAKEN +
        f"4. If the log shows an error (invalid flag, missing module, OOM, crash), **fix it immediately**:\n"
        f"   - Invalid flag → edit `{workspace}/arbor/config.json` to remove or replace it\n"
        f"   - Missing module → remove the flag that requires it; check valid options with: `{frameworks.get(config).help_cmd()} 2>&1 | grep <flag>`\n"
        f"   - Stop a crashed server with `{_component_kill(pid_file)}`, wait until\n"
        f"     `kill -0 -- -$(cat {shlex.quote(str(pid_file))})` fails, then start it again with fixed flags\n\n"
        + _server_ready_steps(config, port)
    )


def _stop_server(config: dict, container: str, gpu_ids, **kw) -> None:
    """Stop the serving server: before a launch, after a failed one, or when VRAM is held.

    Standalone: preflight.stop_server, every framework's pattern swept in the
    container and a wait for the GPUs to empty. Component mode: the server this
    harness launched and only it, by the process group in its PID file
    (preflight.stop_own_server). Inside MB, the sweep's pattern pkill stopped MB's
    own servers, and its VRAM wait counted the host's memory as ours.
    """
    from arbor.execute import preflight
    if component.is_component(config):
        preflight.stop_own_server(component.server_paths(component.session_root(config))[0])
        return
    preflight.stop_server(container, gpu_ids=gpu_ids, **kw)


def _server_log_tail(config: dict, container: str, lines: int = 20) -> str:
    """The server log's last lines: the container's /tmp/server.log, or in component
    mode the session's log on this host."""
    if component.is_component(config):
        log = component.server_paths(component.session_root(config))[1]
        try:
            return "\n".join(log.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""
    r = subprocess.run(
        ["docker", "exec", container, "tail", f"-{lines}", "/tmp/server.log"],
        capture_output=True, text=True,
    )
    return r.stdout


def _safe_revert(container: str, workspace: str) -> None:
    from arbor.execute.workspace import git, revert as _revert
    try:
        git(container, workspace, "checkout", "--", ".")
        git(container, workspace, "clean", "-fd")
    except RuntimeError:
        pass
    try:
        _revert(container, workspace)
    except RuntimeError as e:
        print(f"[arbor] revert failed: {e}", file=sys.stderr, flush=True)


def _mean_script(bench_output_path: str, metric_key: str, secondary_key: str | None = None) -> str:
    """The fixed block that books the MEAN of the replays' ``<BENCH_OUTPUT>.run*`` files.

    Run as given, never by the agent's own arithmetic: asking a model to average
    numbers it just read is an avoidable way to get a wrong bar.
    """
    keys = [metric_key, "p90_interactivity", "p50_interactivity"]
    if secondary_key and secondary_key not in keys:
        keys.append(secondary_key)
    keys_src = ",".join(f"'{k}'" for k in keys)
    return (
        f"   python3 - <<'AGG'\n"
        f"import json, glob, statistics as st\n"
        f"runs = []\n"
        f"for f in sorted(glob.glob('{bench_output_path}.run*')):\n"
        f"    try: runs.append(json.load(open(f)))\n"
        f"    except Exception: pass\n"
        f"ok = [r for r in runs if r.get('status') == 'ok']\n"
        f"if not ok:\n"
        f"    json.dump({{'status':'failed','error':'no successful replay'}},\n"
        f"              open('{bench_output_path}','w'))\n"
        f"else:\n"
        f"    agg = dict(ok[-1])\n"
        f"    for key in ({keys_src}):\n"
        f"        vals = [r[key] for r in ok if isinstance(r.get(key), (int, float))]\n"
        f"        if vals:\n"
        f"            agg[key] = round(st.mean(vals), 5)\n"
        f"            agg[key + '_runs'] = vals\n"
        f"            agg[key + '_spread_pct'] = round(100*(max(vals)-min(vals))/st.mean(vals), 3) if len(vals) > 1 else 0.0\n"
        f"    agg['replays'] = len(ok)\n"
        f"    agg['aggregation'] = 'mean'\n"
        f"    json.dump(agg, open('{bench_output_path}','w'), indent=1)\n"
        f"    print('[bench] mean over', len(ok), 'replays:',\n"
        f"          {{k: agg[k] for k in agg if k.endswith('_runs') or k=='{metric_key}'}})\n"
        f"AGG\n"
    )


# How a pack's second metric is read off a host's client that reports latencies rather than the
# metric itself; the serving pack's validator/bench.sh computes them the same way.
SECONDARY_FROM_LATENCY = {
    "p90_interactivity": "1000 / the p90 time per output token (TPOT) in ms",
    "p50_interactivity": "1000 / the median TPOT in ms",
}


def _host_bench_steps(command: str, port, iter_dir, bench_output_path: str,
                      metric_key: str, repeats: int, secondary_key: str | None = None) -> str:
    """The bench agent's step 1 in component mode when the host gives its bench command.

    The host's client, with ``{port}`` and ``{out}`` filled and nothing else changed,
    for every measurement: the candidate, a baseline, a re-run. ``{out}`` is a
    directory per replay under ``iter_N/host_bench/``, made here, empty. The agent
    reads the metric from what the client wrote into BENCH_OUTPUT (one JSON file per
    replay); with replays, the fixed block books the mean, as with the pack's bench.
    The pack's second metric (``secondary_key``) is recorded beside it when the client
    gives it: without it the gate is one-axis, blind to a gain bought with the other
    axis (`secondary: None` on every KEEP).
    """
    base = Path(iter_dir) / "host_bench"
    outs = [base] if repeats <= 1 else [base / f"run{i}" for i in range(1, repeats + 1)]
    for d in outs:
        d.mkdir(parents=True, exist_ok=True)
    if secondary_key:
        record = ("python3 -c 'import json, sys; d = {\"status\": \"ok\", \"" + metric_key
                  + "\": float(sys.argv[1]), \"source\": \"bench_command\"}; d.update({\""
                  + secondary_key + "\": float(sys.argv[3])} if len(sys.argv) > 3 else {}); "
                  "json.dump(d, open(sys.argv[2], \"w\"))'")
        sec_arg = f" <{secondary_key}>"
        how = SECONDARY_FROM_LATENCY.get(secondary_key)
        second = (f"   Read `{secondary_key}` too, the pack's second metric: the value the client\n"
                  f"   reports for it" + (f", or {how} when it reports that instead" if how else "")
                  + f". Pass it as the\n   last value. If the client gives neither, leave it out: the gate then judges\n"
                  f"   `{metric_key}` alone.\n")
    else:
        record = ("python3 -c 'import json, sys; json.dump({\"status\": \"ok\", \"" + metric_key
                  + "\": float(sys.argv[1]), \"source\": \"bench_command\"}, open(sys.argv[2], \"w\"))'")
        sec_arg, second = "", ""
    rule = (f"   Use it, exactly as given here, for every measurement this iteration: the\n"
            f"   candidate, and any baseline or re-measurement you take. Change nothing in it (no\n"
            f"   flag, no prompt set, no rate or concurrency) and do not run the pack's bench\n"
            f"   script: the host ranks what you keep with this client, and a number from any\n"
            f"   other client is not comparable to its gate.\n")
    runs = "".join(f"   {component.fill_bench(command, port, d)}\n" for d in outs)
    if repeats <= 1:
        return (
            f"1. Run the host's bench command:\n"
            f"   ```bash\n{runs}   ```\n"
            + rule +
            f"   It writes its results under `{outs[0]}`. Read `{metric_key}` from them (the value\n"
            f"   the host's client reports for it) and record it, with this command as it is:\n"
            f"   ```bash\n"
            f"   {record} <value> {bench_output_path}{sec_arg}\n"
            f"   ```\n"
            + second + "\n"
        )
    return (
        f"1. Run the host's bench command {repeats} times, one replay per line, and aggregate:\n"
        f"   ```bash\n{runs}   ```\n"
        + rule +
        f"   Each replay writes its results under its own directory (`{outs[0]}`, ...). After\n"
        f"   replay i, read its `{metric_key}` (the value the host's client reports for it) and\n"
        f"   record it, with this command as it is:\n"
        f"   ```bash\n"
        f"   {record} <value> {bench_output_path}.run<i>{sec_arg}\n"
        f"   ```\n"
        + second +
        f"   Then aggregate. Run this EXACTLY:\n"
        f"   ```bash\n"
        + _mean_script(bench_output_path, metric_key, secondary_key) +
        f"   ```\n"
        f"   REPORT THE MEAN that block wrote, not any individual replay.\n\n"
    )


# ── Main iteration ──────────────────────────────────────────────────────────

def run_iteration(
    config_path: str,
    arbor_root: str,
    session_dir: str,
    iteration: int,
) -> dict:
    """One iteration: agent edits → harness commits/restarts/benches → critic gates."""
    from arbor.cli.iterate import cmd_begin
    from arbor.execute.workspace import (
        changed_files, commit, has_changes, save_patch,
    )
    from arbor.execute.bench import run_bench, run_correctness
    from arbor.execute.watch import gpu_vram_used
    from arbor.measure.integrity import snapshot_digests, verify_unchanged
    import argparse

    session_dir_p = Path(session_dir)
    state = CampaignState.load(session_dir_p / "session.json")
    campaign = load_campaign(config_path, arbor_root)
    config = campaign.config
    container = config.get("docker_container", "")
    workspace = config.get("workspace_root", "")
    port = config.get("port", 8888)

    iter_dir = session_dir_p / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Anchors the out-of-tree artifact scan: anything the agent writes into the
    # pack's eval/ or knowledge/ dirs from here on belongs to this iteration.
    iter_start_ts = time.time()

    # arbor/config.json can be deleted by a revert (see ensure_workspace_config).
    # Without it the server runs on defaults and the measurement is silently
    # non-comparable, so re-seed BEFORE the agent or the bench can observe it.
    if container and workspace:
        try:
            from arbor.execute.workspace import (
                ensure_git_excludes, ensure_workspace_config,
            )
            if ensure_git_excludes(container, workspace):
                print("[arbor] added __pycache__/*.pyc to .git/info/exclude",
                      file=sys.stderr, flush=True)
            if ensure_workspace_config(
                container, workspace,
                component.seed_env(config), config.get("server_flags", ""),
            ):
                print("[arbor] arbor/config.json was missing — re-seeded from "
                      "the campaign config", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[arbor] could not verify arbor/config.json: {e}",
                  file=sys.stderr, flush=True)
        # A module link whose target this container lacks is rebuilt by its JIT over the
        # link; git must never commit that binary (arbor/execute/outside.py).
        try:
            from arbor.execute import outside as _outside
            marked = _outside.prepare(container, workspace, config)
            if marked:
                print(f"[arbor] outside the tree: {len(marked)} module link(s) left to their "
                      f"JIT to rebuild; git skips them", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[arbor] WARNING: module links not prepared: {e} — a rebuilt module can "
                  f"enter the next commit", file=sys.stderr, flush=True)

    # The image and operating point this iteration measures at, for its row and its
    # dead ends (arbor/measure/scope.py). Read once the workspace config is in place.
    from arbor.measure import scope as _scope
    scope = _scope.capture(config, iter_dir)
    print(f"[arbor] Scope: {_scope.describe(scope) or 'unknown'}", file=sys.stderr, flush=True)

    # ── Step 0: Zombie cleanup ──────────────────────────────────────────────
    gpu_ids = component.gpu_list(config.get("gpu_ids"))
    try:
        if component.is_component(config):
            # No VRAM reading and no stop for memory: the cards and what holds memory on
            # them are the host's. The server this harness launched last iteration, if
            # it still runs, is ours: stopped by its PID file.
            _stop_server(config, container, gpu_ids)
        elif gpu_ids:
            vram = gpu_vram_used(gpu_ids)
            if any(gb >= 10 for gb in vram.values()):
                print(f"[arbor] VRAM held on our GPUs {vram} — stopping server", file=sys.stderr, flush=True)
                _stop_server(config, container, gpu_ids, graceful_wait=15, vram_check_timeout=60)
            else:
                print(f"[arbor] GPUs clean {vram}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[arbor] pre-iteration cleanup: {e}", file=sys.stderr, flush=True)

    # ── Step 1: Escalation + render program ─────────────────────────────────
    import io as _io
    begin_ns = argparse.Namespace(
        config=config_path, session_dir=session_dir,
        iter=iteration, branch_id=None, tier=state.tier, hypothesis=None,
    )
    _saved_stdout = sys.stdout
    sys.stdout = _io.StringIO()
    try:
        cmd_begin(begin_ns)
    finally:
        sys.stdout = _saved_stdout

    pre_begin_state = state
    state = CampaignState.load(session_dir_p / "session.json")
    state.iteration_started = iter_start_ts   # the row this iteration records carries its wall time
    state.iteration_notes = {"scope": scope}  # and only this iteration's notes
    # The map stopped the campaign in cmd_begin (arbor/directions.py): no agent
    # runs, and the driver's STOP check ends the loop.
    if (session_dir_p / "STOP").exists():
        print("[arbor] STOP written before the agent ran -- not running iteration "
              f"{iteration}", file=sys.stderr, flush=True)
        return {"iteration": iteration, "decision": "STOPPED", "metric": None,
                "change": (session_dir_p / "STOP").read_text(errors="replace").strip()[:200]}
    # The map ended its round in cmd_begin (arbor/rounds.py): no agent runs, and the
    # driver runs the round-end A/B before the next iteration. Component mode has no
    # round ends (the host gates and merges), so one left pending is not waited on.
    from arbor import rounds as _rounds
    if _rounds.pending(session_dir_p) and not component.is_component(config):
        print(f"[arbor] a round end is pending -- not running iteration {iteration}", file=sys.stderr, flush=True)
        return {"iteration": iteration, "decision": "ROUND_END", "metric": None,
                "change": (_rounds.pending(session_dir_p) or {}).get("detail", "")}
    from arbor import directions as _directions
    map_doc = _directions.current(campaign, state)
    applied_escalation = state.tier if state.tier != pre_begin_state.tier else None
    program = render_program(campaign, state, arbor_root, escalation=applied_escalation,
                             scope=scope)

    # ── Step 2: Agent decides + edits ───────────────────────────────────────
    _report_agent_tmp_health()
    print(f"[arbor] Running agent...", file=sys.stderr, flush=True)
    prompt = _build_agent_prompt(program, campaign, state, iteration)
    try:
        agent_output = _run_phase(
            _run_claude(prompt, arbor_root,
                        allowed_tools=["Bash", "Read", "Edit"],
                        label="agent", skills=agent_skills(campaign.pack)),
            "agent", config,
        )
        change = _extract_change(agent_output) or f"iteration {iteration}"
        claim_class = _extract_class(agent_output)
    except component.PhaseTimeout as e:
        # Component mode: the turn ran out of time mid-edit. What it left uncommitted is
        # no candidate, and the host reads arbor/config.json from the live workspace, so
        # it goes; the committed tree stays as it is.
        from arbor.execute.workspace import git as _git
        for args in (("checkout", "--", "."), ("clean", "-fd")):
            try:
                _git(container, workspace, *args)
            except RuntimeError:
                pass
        return _not_measured(state, session_dir_p, iteration, f"iteration {iteration} ({e})",
                             str(e), claim_class="PROBE")
    except Exception as e:
        # AN AGENT THAT NEVER RAN IS NOT EVIDENCE ABOUT THE CANDIDATE.
        #
        # This used to fall through as claim_class B with an empty diff, so the
        # iteration restarted the server, ran a full bench against unchanged
        # code, and recorded "REVERT: no changes" — indistinguishable in
        # session.json from "the agent wrote something that regressed". It then
        # cooled down and did it again. On 2026-09-19 the v3 campaign's node
        # lost /tmp, the agent died in milliseconds with
        #     ENOENT: no such file or directory, mkdir '/tmp/claude-<pid>'
        # and the loop produced 33 consecutive REVERTs — 24 of them inside 13
        # minutes — burning ~3 h of exclusive MI355X time and manufacturing a
        # null_streak of 24 that reads exactly like diminishing returns.
        #
        # This is the same category as a bench that never ran, which
        # record_invalid() already handles correctly: the tree is preserved,
        # null_streak and the tier counters do NOT move, and only
        # invalid_streak advances — so MAX_CONSECUTIVE_INVALID stops the loop
        # and writes ABORTED.md instead of consuming the allocation.
        reason = f"agent did not run: {e}"
        print(f"[arbor] Agent error: {e}", file=sys.stderr, flush=True)
        print(f"[arbor] NOT MEASURED — the agent never ran, so there is no "
              f"evidence about any change. Tree preserved, counters unchanged.",
              file=sys.stderr, flush=True)
        state.record_invalid(
            iteration=iteration, change=f"iteration {iteration} ({reason})",
            tier=state.tier, reason=reason, claim_class="PROBE", findings=[],
        )
        state.save(session_dir_p / "session.json")
        if state.harness_broken():
            msg = (f"{state.invalid_streak} consecutive not-measured iterations "
                   f"— the harness is broken, not the campaign. Last: {reason}")
            print(f"[arbor] ABORT: {msg}", file=sys.stderr, flush=True)
            _write_aborted(session_dir_p, msg)
        return {"iteration": iteration, "decision": "INVALID", "metric": None,
                "change": f"iteration {iteration} ({reason})",
                "claim_class": "PROBE", "findings": [], "reason": reason}

    # The agent's map lines (DIRECTIONS / DIRECTION), applied before any return
    # path: a dossier iteration changes no workspace file and returns early.
    if map_doc is not None:
        try:
            for ev in _directions.apply_output(session_dir_p, agent_output, iteration, campaign, state,
                                               scope=scope):
                what = ev.get("note") or ev.get("status") or ev.get("error")
                print(f"[arbor] map: {'OK' if ev.get('ok') else 'REJECTED'} "
                      f"{ev.get('id', '')} {what}".replace("  ", " "), file=sys.stderr, flush=True)
        except Exception as e:  # noqa: BLE001 - the map must not cost the iteration
            print(f"[arbor] map: could not apply the agent's map lines: {e}", file=sys.stderr, flush=True)

    # A probe budget the agent talked past is not a probe. Demote it to B so it
    # is held to the throughput bar rather than skipping the server run. Not
    # while the map is written or updated: that round is LEARNs by design.
    if (claim_class == "PROBE" and state.probes_exhausted()
            and not _directions.waives_learn_cap(map_doc, campaign, state)):
        print(
            f"[arbor] PROBE declared but budget exhausted "
            f"(probe_streak={state.probe_streak}/{state.learn_budget()}) "
            f"— demoting to class B",
            file=sys.stderr, flush=True,
        )
        claim_class = "B"

    print(f"[arbor] Claim class: {claim_class} — {change}", file=sys.stderr, flush=True)

    # ── Step 3: Harness commits ─────────────────────────────────────────────
    if not has_changes(container, workspace):
        # A pure-scan PROBE legitimately changes no files — "I read the upstream
        # branches and there is nothing new" is a result the pack explicitly
        # asks for. It still has to produce a ledger record to count.
        scan_ids = _persist_findings(campaign, agent_output, iteration, scope)
        if claim_class == "PROBE" and scan_ids:
            state.record_probe(
                iteration=iteration, change=change, tier=state.tier,
                findings=scan_ids, reason="scan probe — no workspace change",
            )
            state.save(session_dir_p / "session.json")
            print(f"[arbor] PROBE: scan recorded {scan_ids}", file=sys.stderr, flush=True)
            return {"iteration": iteration, "decision": "PROBE", "metric": 0.0,
                    "change": change, "claim_class": claim_class, "findings": scan_ids}
        state.record_revert(
            iteration=iteration, metric=0.0,
            change=change, tier=state.tier,
            reason="agent made no changes to workspace",
            claim_class=claim_class, findings=scan_ids,
        )
        state.save(session_dir_p / "session.json")
        print(f"[arbor] REVERT: no changes", file=sys.stderr, flush=True)
        return {"iteration": iteration, "decision": "REVERT", "metric": 0.0, "change": change}

    sha = commit(container, workspace, f"iter-{iteration}: {change}")
    print(f"[arbor] Committed: {sha[:12]} — {change}", file=sys.stderr, flush=True)

    # ── Step 3b: Read-only probe short-circuit ──────────────────────────────
    # A probe confined to eval/ and the ledger cannot change what the server
    # does, so the 30-minute server run would measure nothing. This is the
    # pack's own rule made mechanical: search in eval/, confirm on the server.
    #
    # Read off the registry rather than compared to a literal: WHICH intents may
    # skip the server is a per-intent rule (`server: skip_if_readonly`), so it
    # belongs on the row with the rest of them. The two `claim_class == "PROBE"`
    # tests above are budget bookkeeping, not a gate rule, and stay as they are.
    if resolve(claim_class).skips_server_when_readonly:
        from arbor.execute.workspace import git as _git
        try:
            changed = [
                p.strip() for p in
                _git(container, workspace, "diff", "--name-only", "HEAD~1", "HEAD").splitlines()
                if p.strip()
            ]
        except RuntimeError:
            changed = []
        if _probe_is_readonly(changed):
            print(
                f"[arbor] Read-only probe ({len(changed)} file(s) under "
                f"{'/, '.join(READONLY_PROBE_PREFIXES)}) — skipping server run",
                file=sys.stderr, flush=True,
            )
            return _finalize_iteration(
                state=state, campaign=campaign, iteration=iteration,
                container=container, workspace=workspace,
                iter_dir=iter_dir, session_dir_p=session_dir_p, arbor_root=arbor_root,
                bench_output={"status": "skipped", "reason": "read-only probe"},
                correctness={"passed": True, "response": "skipped — read-only probe"},
                change=change, claim_class=claim_class, agent_output=agent_output,
                iter_start_ts=iter_start_ts, scope=scope,
            )
        print(
            f"[arbor] Probe touches non-inert paths {changed[:5]} — full lifecycle required",
            file=sys.stderr, flush=True,
        )

    # ── Step 3c: A document has no server: the harness runs the validators ──
    # A `document` workload (a plan, written before an optimization round)
    # produces text, not a serving tree. Its
    # bench scores the committed draft and its correctness script checks the
    # draft's hard requirements. Both are the pack's protected validators, run
    # here by the harness with no agent between the draft and the number --
    # the server and bench agents of steps 4 and 5 have nothing to do.
    if campaign.pack.workload.get("kind") == "document":
        print("[arbor] Document workload — the harness runs the pack's bench and "
              "correctness on the committed draft; no server", file=sys.stderr, flush=True)
        doc_env = {"WORKSPACE_ROOT": workspace, "ITERATION": str(iteration),
                   "SESSION_DIR": str(session_dir_p)}
        bench_output = run_bench(config, str(iter_dir), extra_env=doc_env)
        print(f"[arbor] Bench result: {json.dumps(bench_output)[:300]}", file=sys.stderr, flush=True)
        correctness = run_correctness(config, str(iter_dir), extra_env=doc_env)
        print(f"[arbor] Correctness: {correctness.get('passed', False)}", file=sys.stderr, flush=True)
        return _finalize_iteration(
            state=state, campaign=campaign, iteration=iteration,
            container=container, workspace=workspace,
            iter_dir=iter_dir, session_dir_p=session_dir_p, arbor_root=arbor_root,
            bench_output=bench_output, correctness=correctness,
            change=change, claim_class=claim_class, agent_output=agent_output,
            iter_start_ts=iter_start_ts, scope=scope,
        )

    # ── Step 4: Agent starts server, watches log, fixes errors, confirms healthy ─
    print(f"[arbor] Stopping old server...", file=sys.stderr, flush=True)
    _stop_server(config, container, gpu_ids, graceful_wait=15, vram_check_timeout=120)
    if component.is_component(config):
        # The launch writes the server's PID and log here (arbor/component.py).
        component.server_paths(component.session_root(config))[0].parent.mkdir(
            parents=True, exist_ok=True)

    # Component mode: on this host, with no container, the PID file and the rules.
    server_prompt = _component_server_prompt(config, workspace, port) if component.is_component(config) else (
        f"The harness has committed your change. Now start the server and confirm it is healthy.\n\n"
        f"## HARD CONSTRAINT — never recreate the container\n"
        f"Do NOT run `docker rm`, `docker run`, `docker create`, or any launch script "
        f"for `{container}`. Restart the SERVER PROCESS inside the existing container "
        f"only. The container filesystem holds the git repository at `{workspace}` "
        f"AND the source change you are about to measure. Recreating the container "
        f"resets both to the image, so the bench would silently measure UNMODIFIED "
        f"code and report it as your change's result. This has already happened once "
        f"and cost an iteration. If the container is genuinely gone or unhealthy, stop "
        f"and say SERVER_FAILED: container unusable — do not rebuild it yourself.\n\n"
        f"Container: `{container}`\n"
        f"Port: {port}\n"
        f"Server log: `/tmp/server.log`\n\n"
        f"## What to do\n\n"
        f"1. Start the server. The command carries the live `server_env` (as `env K=V ...`)\n"
        f"   and `server_flags` from `{workspace}/arbor/config.json`, the way the baseline\n"
        f"   was launched:\n"
        f"   ```bash\n"
        f"   docker exec {container} bash -c {shlex.quote(_server_launch_cmd(config))}\n"
        f"   ```\n"
        f"   If you change `arbor/config.json` to get the server up, rebuild the command the\n"
        f"   same way: every `server_env` entry as `env K=V` in front of the server, every\n"
        f"   flag after it. A server started without them is not the arm being measured, and\n"
        f"   the harness reads the live process's env and flags to check.\n\n"
        f"2. Monitor startup — repeat every 60 seconds until healthy or crashed:\n"
        f"   ```bash\n"
        f"   sleep 60 && \\\n"
        f"     curl -sf http://localhost:{port}/health && echo HEALTHY ; \\\n"
        f"     docker exec {container} tail -30 /tmp/server.log\n"
        f"   ```\n"
        f"   **Never use a loop that runs many iterations in one bash call — always one sleep+check per tool call so you can read the output and react.**\n\n"
        + _NEVER_WEAKEN +
        f"4. If the log shows an error (invalid flag, missing module, OOM, crash), **fix it immediately**:\n"
        f"   - Invalid flag → edit `{workspace}/arbor/config.json` to remove or replace it\n"
        f"   - Missing module → remove the flag that requires it; check valid options with: `docker exec {container} {frameworks.get(config).help_cmd()} 2>&1 | grep <flag>`\n"
        f"   - Kill crashed server, restart with fixed flags\n\n"
        + _server_ready_steps(config, port)
    )

    print(f"[arbor] Agent starting server...", file=sys.stderr, flush=True)
    try:
        server_output = _run_phase(
            _run_claude(server_prompt, arbor_root,
                        allowed_tools=["Bash", "Read", "Edit"],
                        label="server"),
            "server", config,
        )
    except component.PhaseTimeout as e:
        # Component mode: a server that did not come up in time is infrastructure, not
        # evidence about the change.
        _stop_server(config, container, gpu_ids)
        return _not_measured(state, session_dir_p, iteration, change, str(e), claim_class)
    except Exception as e:
        server_output = ""
        print(f"[arbor] Server agent error: {e}", file=sys.stderr, flush=True)

    reason = _server_failure(server_output)   # a verdict line, not a mention (#23)
    if reason is not None:
        print(f"[arbor] Server agent gave up: {reason}", file=sys.stderr, flush=True)
        state.record_revert(
            iteration=iteration, metric=0.0,
            change=change, tier=state.tier,
            reason=f"server failed: {reason}",
        )
        state.save(session_dir_p / "session.json")
        _safe_revert(container, workspace)
        _stop_server(config, container, gpu_ids)
        return {"iteration": iteration, "decision": "REVERT", "metric": 0.0, "change": change}

    # Final health check
    healthy = _wait_healthy(config, timeout=120)
    if not healthy:
        tail = _server_log_tail(config, container, 20)
        reason = f"server not healthy after agent startup. Log: {tail.strip()[-500:]}"
        print(f"[arbor] {reason}", file=sys.stderr, flush=True)
        state.record_revert(
            iteration=iteration, metric=0.0,
            change=change, tier=state.tier,
            reason=reason,
        )
        state.save(session_dir_p / "session.json")
        _safe_revert(container, workspace)
        _stop_server(config, container, gpu_ids)
        return {"iteration": iteration, "decision": "REVERT", "metric": 0.0, "change": change}

    print(f"[arbor] Server healthy.", file=sys.stderr, flush=True)
    arm = _arm_check(state, config, iter_dir)

    # ── Step 5: Agent runs bench, watches it, handles crashes ──────────────
    bench_script = config.get("bench_script", "")
    # BENCH_OUTPUT is consumed INSIDE the container, so it must be a CONTAINER
    # path. `iter_dir` is a host path under session_root, and session_root is
    # bind-mounted at `session_mount` (default /workspace/sessions) -- handing
    # the host path to `docker exec` makes the container silently create that
    # directory in its own writable layer and write there. The host sees
    # nothing, the driver falls back to parsing the agent's METRIC: line, and
    # everything except the primary metric is lost: no agg json (so
    # duration_seconds cannot be checked, which is the campaign's first
    # integrity rule) and no secondary metric (so the two-axis gate is inert).
    #
    # Run 2 iteration 1 measured a gain on p90_interactivity, the second axis,
    # larger than its throughput gain -- and the gate never saw it. Run 1 lost iterations 8, 10 and 15 the
    # same way.
    comp = component.is_component(config)
    session_mount = config.get("session_mount", "/workspace/sessions")
    if comp:
        # Component mode: the bench runs on this host, where the host path is the path.
        bench_output_path = str(iter_dir / "bench_output.json")
    else:
        try:
            rel = Path(iter_dir).relative_to(Path(campaign.session_root).parent)
            bench_output_path = f"{session_mount}/{rel}/bench_output.json"
        except ValueError:
            # iter_dir is not under the mounted tree; fall back to the host path and
            # say so, rather than silently writing into the container's own layer.
            bench_output_path = str(iter_dir / "bench_output.json")
            print(f"[arbor] WARNING: {iter_dir} is not under {campaign.session_root} — "
                  f"BENCH_OUTPUT falls back to a host path the container may not share",
                  file=sys.stderr)
    metric_key = campaign.pack.objective.get("metric", "output_throughput")

    bench_env = {
        "PORT": str(port),
        "MODEL_PATH": config.get("model_path", ""),
        "MODEL_NAME": config.get("model_name", ""),
        "CONCURRENCY": str(config.get("concurrency", 14)),
        "NUM_PROMPTS": str(config.get("num_prompts", 640)),
        "BENCH_OUTPUT": bench_output_path,
        "BENCH_SERVING": config.get("bench_serving_script", ""),
    }
    server_env = config.get("server_env", {})
    for k, v in (component.strip_masks(server_env) if comp else server_env).items():
        bench_env[k] = str(v)

    # Every key, one -e per line. This used to be `list(bench_env.items())[:8]`,
    # which silently dropped everything after the 8th entry -- and since
    # server_env is merged in AFTER the 7 base keys, the cut fell immediately
    # after ROCR_VISIBLE_DEVICES. DURATION never reached bench_agentx.sh, which
    # then fell back to its own default of 3600s: every iteration ran a 60-minute
    # replay instead of the configured 600s, and was measured under different
    # conditions than the baseline. A truncated env is never the right answer --
    # if a command is too long, wrap it, do not silently shorten what it says.
    bench_env_flags = " \\\n     ".join(f'-e {k}="{v}"' for k, v in bench_env.items())
    # The bench's runner and where it runs: `docker exec -e K="v" ... <container> bash`,
    # or in component mode `env K=v ... bash` on this host (no docker there).
    runner, target, out_flag = "docker exec", f"{container} bash", f'-e BENCH_OUTPUT="{bench_output_path}"'
    if comp:
        bench_env_flags = " \\\n     ".join(f"{k}={shlex.quote(v)}" for k, v in bench_env.items())
        runner, target, out_flag = "env", "bash", f"BENCH_OUTPUT={shlex.quote(bench_output_path)}"

    # REPEATED MEASUREMENT. `bench_repeats` > 1 runs the replay N times and
    # books the MEAN, because a single replay is not a reliable estimate of this
    # workload: re-measuring a byte-identical v3 tree on its own node, same code
    # path, same DURATION, differed by more than the gate, and v3's own
    # post-iteration-6 measurements spanned about as much. Arbor books best_metric as a
    # running MAX, so with noise that large the bar lands on the luckiest draw
    # and nothing afterwards can clear it.
    #
    # The aggregation is done by a fixed python block, NOT by the sub-agent:
    # asking a model to average numbers it just read is an avoidable way to get
    # a wrong bar. Individual runs are preserved under `runs` so the spread is
    # auditable and L0-AUDIT can widen or tighten the bands from real data.
    # Read from the pack's measurement block FIRST, because that block is
    # reloaded every iteration (driver.py) while `config` is loaded once. So a
    # campaign can start at 2 replays while the noise floor is still being
    # established and drop to 1 once L0-AUDIT has characterised the spread --
    # without restarting the job, which would cost the allocation. The config
    # key stays supported as the static fallback.
    _meas = campaign.pack.objective.get("measurement", {})
    bench_repeats = int(
        _meas.get("bench_repeats", config.get("bench_repeats", 1)) or 1)
    if bench_repeats > 1:
        run_env_flags = bench_env_flags.replace(
            out_flag,
            out_flag.split("BENCH_OUTPUT=")[0] + 'BENCH_OUTPUT="${OUT}"')
        run_block = (
            f"1. Run the bench {bench_repeats} times and aggregate. Run this EXACTLY:\n"
            f"   ```bash\n"
            f"   for i in $(seq 1 {bench_repeats}); do\n"
            f"     OUT={bench_output_path}.run$i\n"
            f"     echo \"[bench] replay $i/{bench_repeats} -> $OUT\"\n"
            f"     {runner} \\\n     " + run_env_flags +
            f" \\\n     {target} {bench_script}\n"
            f"   done\n"
            + _mean_script(bench_output_path, metric_key) +
            f"   ```\n"
            f"   REPORT THE MEAN that block wrote, not any individual replay.\n"
            f"   If one replay fails but others succeed, the block aggregates the\n"
            f"   successful ones and says how many -- state that in your summary.\n\n"
        )
    else:
        run_block = (
            f"1. Run the bench:\n"
            f"   ```bash\n"
            f"   {runner} \\\n     "
            + bench_env_flags
            + f" \\\n     {target} {bench_script}\n"
            f"   ```\n"
            f"   The bench writes results to `{bench_output_path}` on the host.\n\n"
        )

    # Where the server's log is read and how it is restarted: in the container, or in
    # component mode on this host, by its PID file, with the component rules up front.
    head = f"Container: `{container}`\n"
    watch = f"docker exec {container} tail -5 /tmp/server.log"
    restart = "   - Restart the server: kill old process, start fresh, wait healthy\n"
    if comp:
        pid_file, server_log = component.server_paths(component.session_root(config))
        head = f"{_component_rules(config)}\nServer log: `{server_log}`\n"
        watch = f"tail -5 {shlex.quote(str(server_log))}"
        restart = (f"   - Restart the server: stop it with `{_component_kill(pid_file)}`, wait until its\n"
                   f"     group is gone, start it again with this launch, then wait until healthy:\n"
                   f"     `{_server_launch_cmd(config)}`\n")
    # Component mode with the host's bench command (component.bench_command): every
    # measurement uses the host's own client, so Arbor ranks a candidate the way the
    # host's gate does (MB: another client sends other prompts, which route the MoE's
    # tokens otherwise). Without one, the pack's bench, as standalone runs it.
    bench_line = f"Bench script: `{bench_script}`\n"
    host_bench = component.bench_command(config) if comp else None
    if host_bench:
        bench_line = "Bench: the host's own command (step 1), not the pack's bench script\n"
        run_block = _host_bench_steps(
            host_bench, port, iter_dir, bench_output_path, metric_key, bench_repeats,
            campaign.pack.objective.get("measurement", {}).get("secondary_metric"))
    bench_prompt = (
        f"Run the bench and report the metric. The server is healthy on port {port}.\n\n"
        + head + bench_line +
        f"Output file: `{bench_output_path}`\n"
        f"Metric to report: `{metric_key}`\n"
        + (f"Replays this iteration: {bench_repeats} (book the MEAN)\n" if bench_repeats > 1 else "")
        + f"\n## What to do\n\n"
        + run_block
        + f"2. Watch progress — every 60s read bench logs:\n"
        f"   ```bash\n"
        f"   sleep 60 && {watch}\n"
        f"   ```\n\n"
        f"3. If the server crashes during the bench (ServerDisconnectedError, SIGQUIT, OOM in log):\n"
        f"   - Fix the root cause (reduce a memory-heavy flag, adjust config)\n"
        + restart +
        f"   - Re-run the bench\n\n"
        f"4. When done, read the output and report the metric:\n"
        f"   ```bash\n"
        f"   cat {bench_output_path}\n"
        f"   ```\n"
        f"   Then say:\n"
        f"   ```\n"
        f"   METRIC: <value>  (e.g. METRIC: 245.3)\n"
        f"   ```\n"
        f"   If the bench failed after all retries, say:\n"
        f"   ```\n"
        f"   BENCH_FAILED: <reason>\n"
        f"   ```\n"
    )

    print(f"[arbor] Agent running bench...", file=sys.stderr, flush=True)
    try:
        bench_agent_output = _run_phase(
            _run_claude(bench_prompt, arbor_root,
                        allowed_tools=["Bash", "Read", "Edit"],
                        label="bench"),
            "bench", config,
        )
    except component.PhaseTimeout as e:
        # Component mode: a bench cut off by its cap or the deadline measured nothing
        # that can be booked, even if a replay finished.
        return _not_measured(state, session_dir_p, iteration, change, str(e), claim_class)
    except Exception as e:
        bench_agent_output = ""
        print(f"[arbor] Bench agent error: {e}", file=sys.stderr, flush=True)

    # Extract reported metric
    reported_metric = None
    for line in bench_agent_output.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("METRIC:"):
            try:
                reported_metric = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass

    # Read bench_output.json from disk (bench script writes it)
    bench_output_file = iter_dir / "bench_output.json"
    if bench_output_file.exists():
        try:
            bench_output = json.loads(bench_output_file.read_text())
            bench_output.setdefault("status", "ok")
        except Exception:
            bench_output = {"status": "failed", "error": "invalid JSON in bench_output.json"}
    elif _bench_failure(bench_agent_output) is not None:   # a verdict line, after any METRIC (#23)
        bench_output = {"status": "failed", "error": _bench_failure(bench_agent_output)}
    elif reported_metric is not None:
        bench_output = {"status": "ok", metric_key: reported_metric, "source": "agent_reported"}
    else:
        bench_output = {"status": "failed", "error": "bench_output.json not found and no metric reported"}

    print(f"[arbor] Bench result: {json.dumps({k: v for k, v in bench_output.items() if k not in ('agg_json_path',)})[:300]}", file=sys.stderr, flush=True)

    # ── Step 5b: Collect KV trace if present ────────────────────────────────
    if component.is_component(config):
        _collect_kv_trace(container, iter_dir, arbor_root, local=component.server_paths(
            component.session_root(config))[0].parent / "kv_trace.jsonl")
    else:
        _collect_kv_trace(container, iter_dir, arbor_root)

    # ── Step 6: Harness runs correctness ────────────────────────────────────
    component.phase("correctness")   # the heartbeat's, in component mode
    # Component mode bounds it like every other phase: the pack's bench_timeout alone
    # (2400 s for Model M) let it run past the host's deadline. Cut off, it books nothing.
    cap = component.phase_seconds(config, "correctness")
    if cap is not None and cap <= 0:
        return _not_measured(state, session_dir_p, iteration, change,
                             "correctness phase not started: no time is left before the "
                             "deadline", claim_class)
    correctness = run_correctness(config, str(iter_dir), cap=cap)
    if correctness.get("capped"):
        return _not_measured(state, session_dir_p, iteration, change,
                             f"correctness phase timed out after {correctness['capped']} s "
                             f"(its cap or the deadline)", claim_class)
    print(f"[arbor] Correctness: {correctness.get('passed', False)}", file=sys.stderr, flush=True)
    # Component mode: the short-prompt probe on this server, recorded only; a KEEP
    # compares it with a second process of the tree (arbor/measure/short_prompts.py).
    sp_cap = component.phase_seconds(config, "short_prompts")
    if correctness.get("passed") and sp_cap is not None and sp_cap > 0:
        from arbor.measure import short_prompts as _sp
        component.phase("short_prompts")
        doc = _sp.probe_iteration(config, session_dir_p, iter_dir, sp_cap)
        print(f"[arbor] Short prompts: {doc.get('answered', 0)} of {doc.get('prompts', 64)} answered"
              + (f" ({doc['error']})" if doc.get("error") else ""), file=sys.stderr, flush=True)
    arm = _arm_check(state, config, iter_dir, prior=arm)

    return _finalize_iteration(
        state=state, campaign=campaign, iteration=iteration,
        container=container, workspace=workspace,
        iter_dir=iter_dir, session_dir_p=session_dir_p, arbor_root=arbor_root,
        bench_output=bench_output, correctness=correctness,
        change=change, claim_class=claim_class, agent_output=agent_output,
        iter_start_ts=iter_start_ts, arm=arm, scope=scope,
    )
