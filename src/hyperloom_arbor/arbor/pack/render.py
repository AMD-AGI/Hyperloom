"""Render the agent's instruction document from knowledge + state."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from arbor.measure.intents import from_legacy, is_history_row
from arbor.pack.loader import agent_skills

# How many recent non-KEEP iterations to show in full. KEEPs are always shown —
# they describe the tree the agent is actually looking at.
HISTORY_WINDOW = 10

# What the agent is told about a skill the pack lists, where the skill's own
# description is not the whole rule. A pack that lists "profiler" profiles
# with it, ahead of any profiling procedure its knowledge describes; a pack
# that does not keeps its own (operator, 2026-10-06).
_SKILL_RULES = {
    "profiler": (
        "every profile this campaign takes, to find where time goes or to check "
        "a mechanism, follows this skill: ruppell for traces, rocprofv3 for "
        "counters, TraceLens for reading. It takes precedence over any profiling "
        "procedure in the knowledge above (the PyTorch profiler, captures under "
        "`--enforce-eager`). The validator and the evaluators are unchanged."
    ),
    "dossier": (
        "read this campaign's dossier before choosing what to work on. The skill "
        "looks it up first and uses it; it refreshes only stale parts, and writes "
        "a dossier only when none exists."
    ),
    # Operator, 2026-10-20: the balanced pack's three reference skills.
    "hardware": (
        "price every floor, roofline and kernel plan against what this device "
        "measurably achieves (its table, with sources), not the datasheet; its "
        "reference pages hold the ISA, LDS and MFMA limits a kernel must respect."
    ),
    "serving-playbook": (
        "what 1,258 InferenceX changes learned. Read big-directions.md before "
        "ranking the dossier's directions, amd-knobs.md before a knob search, "
        "numerics-knobs.md before a change that may alter outputs, and "
        "image-manifest.md when a change shows no effect or a feature fails. "
        "A gain it reports elsewhere is a lead, never this node's number."
    ),
    "kernels": (
        "before writing or porting a kernel: AITER's operators and tuned tables "
        "first; then which tool (Triton, FlyDSL, CK or HIP) and why; "
        "probe-image.sh for what the container ships; kbench.py to prove a "
        "kernel correct and fast at the served shapes, stage by stage."
    ),
}


def _lesson_label(lesson: dict) -> str:
    """`INTENT → VERDICT`, or the annotation's own word.

    Used by BOTH the per-iteration line and the header tally, because a
    document that says "26 PROBE" above rows labelled "LEARN → KEEP" makes the
    reader reconcile two vocabularies — the exact cost the split was meant to
    remove. Annotation rows (BASELINE / ESCALATE / NOTE) are exempt: no gate
    produced them and no intent was declared, so re-deriving a pair would print
    every hand-written NOTE as a plain KEEP.
    """
    d = lesson.get("decision", "?")
    if is_history_row(d):
        return d
    intent, verdict = from_legacy(d, lesson.get("claim_class", ""))
    return f"{intent.name} → {verdict}"


def _lesson_line(lesson: dict) -> str:
    d = lesson.get("decision", "?")
    c = lesson.get("change", "?")
    m = lesson.get("metric", "?")
    if lesson.get("decision") == "INVALID":
        # There is no number: the iteration never produced a measurement, and
        # printing "None" reads like a zero.
        m = "NOT MEASURED — this change is still in the tree, unjudged"
    t = lesson.get("tier", "?")
    r = lesson.get("reason", "")
    # `intent → verdict`, unconditionally. The old form annotated the class only
    # when it "added information" (not B, and not already implied by the
    # decision) — a conditional that existed because one word did both jobs, so
    # the decision sometimes named the class and sometimes did not. Now that
    # they are two axes, always printing both is shorter to read and never
    # ambiguous. Annotation rows are exempt: no gate produced them and no intent
    # was declared, so they print their stored word verbatim.
    line = f"- iter {lesson.get('iteration', '?')}: **{_lesson_label(lesson)}** [{t}] {c} → {m}"
    if r:
        line += f" ({r})"
    found = lesson.get("findings") or []
    if found:
        line += f" [ledger: {', '.join(found)}]"
    # The live server lacked part of what the arm claimed (arbor/measure/armcheck.py).
    arm = lesson.get("arm_check") or {}
    if arm.get("ok") is False:
        line += f" [arm differed: {'; '.join(arm.get('missing', [])[:3])}]"
    acc = lesson.get("accuracy") or {}
    if acc:
        line += f" [accuracy {acc.get('score')} vs {acc.get('golden')}]"
    bundle = lesson.get("bundle") or {}
    if bundle.get("enabled") is False:
        line += f" [BUNDLE {bundle.get('id')} not taken: bundles not enabled in this pack]"
    elif bundle:
        line += f" [bundle {bundle.get('id')}]"
    return line


def _render_history(lessons: list[dict]) -> str:
    """Full KEEPs plus a recent window of everything else.

    A REVERT from thirty iterations ago is already represented in the ledger as
    a DEAD_END if it taught anything; repeating its one-line summary forever
    costs prompt budget and teaches nothing the ledger does not.
    """
    # INVALID is pinned alongside KEEP, not windowed. An INVALID iteration was
    # never measured and its change is STILL IN THE TREE — eliding it after ten
    # more iterations would leave the agent editing on top of code it has no
    # record of.
    #
    # Deliberately still a literal, and NOT the rule "verdict != REVERT" that
    # the intent split makes tempting. Those are not the same set: a LEARN that
    # passed has verdict KEEP, so that rule would pin every probe — 25 of them
    # in the cuda-graph campaign, almost all of whose artifacts live in `eval/`
    # rather than in the tree. It would also pin ESCALATE. Which rows are worth
    # permanent prompt budget is a judgement about cost, not something derivable
    # from the verdict, so it stays written down.
    pinned = ("KEEP", "BASELINE", "INVALID", "MAP_ROUND", "ROUND_AB")
    keeps = [l for l in lessons if l.get("decision") in pinned]
    others = [l for l in lessons if l.get("decision") not in pinned]
    recent_others = others[-HISTORY_WINDOW:]
    elided = len(others) - len(recent_others)

    shown = sorted(
        keeps + recent_others,
        key=lambda l: (l.get("iteration", 0), l.get("ts", 0)),
    )

    counts: dict[str, int] = {}
    for l in lessons:
        label = _lesson_label(l)
        counts[label] = counts.get(label, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))

    lines = [f"## History ({len(lessons)} iterations — {summary})\n"]
    if elided > 0:
        lines.append(
            f"_{elided} older non-KEEP iteration(s) elided. Durable conclusions "
            f"from them belong in the ledger above._\n"
        )
    lines.extend(_lesson_line(l) for l in shown)
    return "\n".join(lines)


_ANNOTATION_DECISIONS = {"BASELINE", "ESCALATE", "NOTE", "REBASELINE", "MAP_ROUND", "ROUND_END", "ROUND_AB"}

# How many items of each kind the NOTICE block lists. Long enough to be a real
# menu, short enough not to crowd out the ledger it is pointing back at.
_BARREN_LIST_CAP = 6


def _drought_rows(state) -> list[dict]:
    """The iterations inside the current drought, oldest first."""
    n = getattr(state, "barren_streak", 0)
    if n <= 0:
        return []
    real = [l for l in (state.lessons or [])
            if l.get("decision") not in _ANNOTATION_DECISIONS]
    return real[-n:]


def _render_barren(campaign, state) -> str:
    """The BARREN NOTICE / BARREN ACT blocks. See docs/barren-and-learn-design.md.

    Shaped after the `secondary_bar_suspect` block above, which is the house
    pattern for this class of problem: state the statistical inference, make
    the condition visible, name the CHEAP experiment that settles it, and never
    auto-correct. A harness that silently adjusts its own bars is how a
    campaign talks itself into a number no tree produces.
    """
    if not hasattr(state, "barren_stage"):
        return ""
    stage = state.barren_stage()
    if stage == "clear":
        return ""

    n = state.barren_streak
    k = max(0, (state.iteration or 0) - n)
    rows = _drought_rows(state)

    if stage == "act":
        # WHAT THIS MUST NOT DO: name a component. The same leak-check that
        # governs a pack's constraint text governs this block — it has to be
        # true of every campaign, and a campaign that is told where to look has
        # not discovered anything.
        # NOT a claim about what the campaign has or has not measured. The
        # design sketched two variants ("nothing has measured it" / "the last
        # measurement was iteration X"), and rendering the first against the 2b
        # campaign — which holds ~20 denominator and census ledger entries —
        # produced a flat falsehood. A block the agent can catch lying is a
        # block it discounts. Asking instead is always true, and after N landed
        # changes STALENESS is the live question anyway.
        prov = ("Name the measurement that answer rests on, and when it was "
                "last taken against the CURRENT tree.")
        return (
            f"## THE FRAME IS SUSPECT — {n} iterations, no advance\n\n"
            f"{n} consecutive iterations produced no booked advance on either "
            f"axis. Across that many UNRELATED changes, competently executed "
            f"and measured, that is better explained by this campaign "
            f"optimizing components that are not the cost than by {n} "
            f"independent failures.\n\n"
            f"This is the same inference the harness already makes about a "
            f"suspect secondary bar, applied one level up. When a long run of "
            f"unrelated changes all read null, the thing they have in common "
            f"is not the changes. It is the DENOMINATOR.\n\n"
            f"Every one of those {n} iterations assumed a particular answer to "
            f"\"where does the time go\". {prov}\n\n"
            f"The cheap experiment that settles it: **price the WHOLE RUN.** "
            f"Not another component evaluator — an instrument that accounts "
            f"for wall-clock end to end, and that is capable of telling you "
            f"the component you have been tuning is a small share of it. Build "
            f"it to BRACKET rather than to agree: two independent estimators "
            f"that should disagree with each other if your model of the run is "
            f"wrong. Build it so it CAN come back and say no.\n\n"
            f"Declare `INTENT: LEARN`. State the decision rule BEFORE you see "
            f"the number: which split would send you somewhere else entirely, "
            f"and where would you go."
        )

    # ── NOTICE ──────────────────────────────────────────────────────────────
    mix = Counter(l.get("intent") or l.get("claim_class") or "?" for l in rows)
    mix_s = ", ".join(f"{v}x {kk}" for kk, v in
                      sorted(mix.items(), key=lambda kv: -kv[1])) or "unrecorded"
    closed = sum(len(l.get("findings") or []) for l in rows)

    parts = [
        f"## No booked advance in {n} iterations\n",
        f"`{state.metric_name or 'the primary metric'}` has sat at "
        f"{state.best_metric} and `{state.secondary_name or 'the second axis'}` "
        f"at {state.best_secondary} since iteration {k}. Those {n} iterations "
        f"were {mix_s} — they were not idle, and they closed {closed} ledger "
        f"findings between them, but none of them cleared a bar.\n",
        "Read that as information, not as a verdict: the cheap moves adjacent "
        "to the current tree are spent, and the next advance is probably not "
        "adjacent to the last one.\n",
    ]

    # (1) Open findings older than the drought. A finding re-recorded during
    # the drought supersedes its own id and carries the newer iteration, so
    # "older than k" is a serviceable proxy for "nothing here has cited it".
    uncited = []
    if campaign.session_root:
        from arbor import ledger as _ledger
        try:
            _all = _ledger.current(
                _ledger.load(_ledger.ledger_path(campaign.session_root)))
            uncited = [f for f in _ledger.open_probes(_all)
                       if (f.iteration or 0) < k]
        except Exception:
            uncited = []
    if uncited:
        parts.append(
            f"**{len(uncited)} open findings nothing in this drought has "
            f"cited:**\n" + "\n".join(
                f"- `{f.id}` — {str(f.claim)[:150]}"
                for f in uncited[:_BARREN_LIST_CAP]) + "\n")

    # (2) Re-land candidates. This is the highest-yield move in the corpus and
    # nothing currently asks for it: the 2c campaign re-landed previously
    # REVERTed work at iterations 15, 17, 19 and 24 — unprompted — and cleared
    # a bar every time.
    # Only intents that BOOK. A `LEARN -> REVERT` is a probe that closed no
    # finding, not a change sitting on the shelf — listing those as re-land
    # candidates (the first draft did) points the agent at nothing.
    reverts = [l for l in (state.lessons or [])
               if l.get("verdict") == "REVERT"
               and (l.get("iteration") or 0) < k
               and (l.get("intent") or "WIN") != "LEARN"]
    if reverts:
        parts.append(
            f"**{len(reverts)} changes REVERTed before iteration {k}.** A "
            f"change that lost against the incumbent of {k} is a different "
            f"experiment against this one:\n" + "\n".join(
                f"- iter {l.get('iteration')}: {str(l.get('change'))[:130]}"
                for l in reverts[-_BARREN_LIST_CAP:]) + "\n")

    from arbor.state import MAX_CONSECUTIVE_PROBES
    parts.append(
        f"**Your LEARN budget is {state.learn_budget()} consecutive this "
        f"iteration, not {MAX_CONSECUTIVE_PROBES}.** A LEARN touching only "
        f"`eval/` and the ledger skips the server run entirely — minutes "
        f"instead of ~45. While this drought lasts, knowledge is the cheapest "
        f"thing on the menu.\n")

    # THE CLOSING PARAGRAPH IS NOT A HEDGE. A forced-diversity rule is the
    # obvious move here and it is wrong: the 2d campaign's entire lead came
    # from going DEEPER on one seam it had already touched (iterations 22, 24,
    # 28 and 32 are all the same one). Breadth is the right prior when you do
    # not know where the cost is; it is the wrong rule once you do.
    parts.append(
        "You are NOT required to change direction. Going deeper on a seam "
        "that is already paying is frequently the right move. But say which "
        "it is: name the one finding or the one seam the next advance comes "
        "from, and name what would tell you it is the wrong one.")

    return "\n".join(parts)


def _skill_description(arbor_root: Path, name: str) -> str:
    try:
        text = (arbor_root / ".claude" / "skills" / name / "SKILL.md").read_text()
    except OSError:
        return ""
    m = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _render_skills(skills: list[str], arbor_root: Path) -> str:
    lines = [
        "## Skills\n",
        "Use these through the Skill tool, or read `.claude/skills/<name>/SKILL.md` "
        "in the Arbor tree:\n",
    ]
    for name in skills:
        rule = _SKILL_RULES.get(name) or _skill_description(arbor_root, name)
        lines.append(f"- `{name}`: {rule or 'no description'}")
    return "\n".join(lines)


def _render_bundles(campaign) -> str:
    """The catalog's bundles for this framework, when the pack allows claiming one."""
    from arbor import knobs
    from arbor.measure.bundles import members
    rows = knobs.bundles((campaign.config.get("framework") or "").lower() or None)
    if not rows:
        return ""
    lines = [
        "## Bundles: one claim\n",
        "A bundle is a documented set of settings that pays only together (the serving-playbook "
        "skill, `amd-knobs.md` §0 rule 6). Apply every member in `arbor/config.json` in one "
        "iteration and write `BUNDLE: <id>` beside `DONE:`; the gate takes the set as one change. "
        "It is NOT_MEASURED unless every member is set as listed and the live server shows each "
        "one. A reverted bundle is one dead end, for the bundle, not for its members.\n",
    ]
    lines += [f"- `{b['id']}`: {', '.join(members(b))}. {b.get('when', '')}".rstrip() for b in rows]
    return "\n".join(lines)


def _render_accuracy_gate(policy, config: dict | None = None) -> str:
    """What calls for the accuracy gate, on a GSM8K workload (arbor/measure/accuracy.py)."""
    from arbor import component
    every = (" In component mode every KEEP is held to it, whatever it changed: the host imports "
             "what you keep." if component.is_component(config or {}) else "")
    return (
        "## Accuracy gate\n\n"
        "A change that moves a knob marked NUM in the serving playbook (`knobs.json`: quantized "
        "collectives, KV and compute dtypes, MoE activation formats, ...) is held to the full "
        "GSM8K set (1319 questions) against the baseline's full-set score minus "
        f"{policy.accuracy_tolerance}; below it is a REVERT. So is a code change you mark "
        "`NUMERICS: yes` beside `DONE:`: write it for any change that is not bit-exact (a kernel, "
        "a fused path, a precision), and so is a change that turns on a switch an earlier KEEP "
        f"landed off.{every}"
    )


def _render_switches() -> str:
    """S4's rules, which the harness checks at a KEEP (arbor/measure/switches.py)."""
    return (
        "## A switch that defaults off (a kernel plan's S4)\n\n"
        "Code that lands behind a new environment switch is checked by the harness when the gate "
        "would KEEP it. Declare each switch on its own line beside `DONE:`, with the value that "
        "turns it on: `SWITCH: VLLM_MY_KERNEL=1`. The harness then:\n"
        "- refuses the KEEP when the arm or the campaign's launch sets the switch on: it lands "
        "off, and a later iteration turns it on (S5, held to the accuracy gate);\n"
        "- serves the tree again with every switch on and runs the accuracy check there: the "
        "full GSM8K on a GSM8K workload, the pack's correctness check on any other;\n"
        "- refuses it when a file appears in a JIT cache (Triton, AITER, torch extensions, "
        "inductor, vLLM, FlyDSL) once that server is healthy: compile and warm the kernel before "
        "the server reports healthy;\n"
        "- refuses it unless that server's log has a line naming the switch from every rank: log "
        "one, with the switch's name, where the kernel dispatches.\n\n"
        "A new variable your change reads that the tree did not name, that the arm leaves unset "
        "and that you did not declare, refuses the KEEP: the harness cannot know how to turn it "
        "on."
    )


def render_program(campaign, state, arbor_root: str | Path,
                   escalation: str | None = None, scope: dict | None = None) -> str:
    """Merge harness knowledge + pack knowledge + campaign state into one document.

    ``escalation`` is the tier the harness has already applied this iteration.
    Pass it explicitly so the rendered block matches what actually happened —
    calling check_escalation() here would re-evaluate against already-reset counters.

    ``scope`` is this iteration's image and operating point (arbor/measure/scope.py),
    against which the ledger flags dead ends recorded elsewhere.
    """
    arbor_root = Path(arbor_root)
    sections = []

    sections.append(f"# {campaign.pack.name} — {campaign.goal}\n")

    # The regime, stated once at the top. 2b's 31-iteration drought was
    # invisible for as long as it was because every counter anyone looked at
    # faithfully reported `null_streak=2`.
    if hasattr(state, "diagnose") and state.lessons:
        sections.append(f"_{state.diagnose()}_")

    # Pack description — the pack author's own framing of the campaign.
    #
    # Rendered directly under the title and ahead of every knowledge file,
    # because it is the one place the pack states what this campaign IS
    # before it starts stating rules.
    #
    # It was parsed into Pack.description from the beginning but never
    # rendered until 2026-09-20. Packs had been written against the
    # assumption that it reached the agent, so agentx-dsv41-5's header --
    # "at CONCURRENCY 4", the C4 baseline and target, and the warning that
    # the inherited roadmap's numbers are C32 LEADS -- was invisible to the
    # agent it was written for, and the C4/C32 distinction survived only in
    # the config goal and knowledge/operating-point.md.
    if campaign.pack.description:
        sections.append(campaign.pack.description.strip())

    # Harness-level knowledge
    harness_kb = arbor_root / "arbor" / "knowledge"
    if harness_kb.is_dir():
        for f in sorted(harness_kb.glob("*.md")):
            sections.append(f.read_text().strip())

    # Pack-level knowledge
    if campaign.pack.knowledge_dir.is_dir():
        for f in sorted(campaign.pack.knowledge_dir.glob("*.md")):
            sections.append(f.read_text().strip())

    # Objective
    obj = campaign.pack.objective
    unit = obj.get('unit', '')
    unit_s = f" {unit}" if unit else ""
    sections.append(
        f"## Objective\n\n"
        f"- Kind: {obj['kind']}\n"
        f"- Metric: {obj['metric']} ({obj['direction']}){unit_s}\n"
        f"- Baseline: {state.baseline}\n"
        f"- Best: {state.best_metric}\n"
        f"- Target: {campaign.config.get('target', 'not set')}\n"
    )

    # Editable environment — derived entirely from config, no hardcoded paths.
    # This is the agent's authoritative list of what it may touch. Anything
    # outside these paths is read-only. The harness enforces this at gate time.
    container = campaign.config.get("docker_container", "")
    workspace = campaign.config.get("workspace_root", "")
    pkgs = campaign.config.get("editable_packages", [])
    from arbor import component
    from arbor.execute.workspace import HOST
    if container == HOST and workspace and component.is_component(campaign.config):
        # Component mode (arbor/component.py): a serving workspace on this host, under
        # a host that owns the node; the agent prompt ends with the host's rules.
        lines = [
            "## Editable environment\n",
            f"Workspace on this host (git-tracked by the harness): `{workspace}`",
            f"  - Config file (config-tier changes): `{workspace}/arbor/config.json`",
        ]
        for pkg in pkgs:
            rebuild = f" — rebuild: `{pkg['rebuild']}`" if pkg.get("rebuild") else ""
            lines.append(f"  - {pkg['name']}: `{pkg['path']}`{rebuild}")
        lines.append("\n**Only edit files at those paths.** There is no container: the server "
                     "runs on this host, on the GPUs the host leases to this campaign, beside "
                     "the host's own. Everything else — the Arbor harness, the pack's validators, "
                     "the host's files — is read-only.")
        sections.append("\n".join(lines))
    elif container == HOST and workspace:
        # A document campaign: the workspace is a directory on this host.
        sections.append("\n".join([
            "## Editable environment\n",
            f"Workspace on this host (git-tracked by the harness): `{workspace}`\n",
            "**Only edit files in that directory.** There is no container and no server. "
            "Everything else — the Arbor harness, the pack's validators, other campaigns' "
            "files — is read-only.",
        ]))
    elif container and workspace:
        lines = [
            "## Editable environment\n",
            f"Container: `{container}`\n",
            "**Only edit files inside the container at the paths listed below.**",
            "Everything else — other users' home dirs, system packages, the Arbor harness — is read-only.\n",
            f"Primary workspace (git-tracked by harness): `{workspace}`",
            f"  - Config file (config-tier changes): `{workspace}/arbor/config.json`",
        ]
        for pkg in pkgs:
            rebuild = f" — rebuild: `{pkg['rebuild']}`" if pkg.get("rebuild") else ""
            lines.append(f"  - {pkg['name']}: `{pkg['path']}`{rebuild}")
        sections.append("\n".join(lines))

    # Directives — campaign-level focus areas from config
    directives = campaign.config.get("directives", [])
    if directives:
        sections.append("## Directives\n")
        for d in directives:
            sections.append(f"- {d}")

    # Constraints — pack-level rules that always apply
    if campaign.pack.constraints:
        sections.append("## Constraints\n")
        for c in campaign.pack.constraints:
            sections.append(f"- {c}")

    # Skills — after the knowledge and constraints, because a listed skill
    # can take precedence over a procedure they describe (_SKILL_RULES).
    skills = agent_skills(campaign.pack)
    if skills:
        sections.append(_render_skills(skills, arbor_root))

    # Bundles, where the pack's policy allows claiming one (arbor/measure/bundles.py).
    if getattr(getattr(state, "policy", None), "bundles", False):
        _bundles = _render_bundles(campaign)
        if _bundles:
            sections.append(_bundles)
    policy = getattr(state, "policy", None)
    config = getattr(campaign, "config", None) or {}
    if getattr(policy, "accuracy_gate", False):
        from arbor.measure.accuracy import gsm8k_workload
        if gsm8k_workload(config, policy)[0]:
            sections.append(_render_accuracy_gate(policy, config))
    sections.append(_render_switches())

    # Escalation — show the block only when the harness has already applied it this
    # iteration. The caller passes the tier explicitly so this function stays pure.
    if escalation:
        sections.append(
            f"## ESCALATION — MANDATORY\n\n"
            f"**The harness has forced escalation to tier `{escalation}`.** "
            f"This is not a suggestion. You MUST NOT propose changes at a lower tier. "
            f"The previous tier is exhausted.\n\n"
            f"Tier `{escalation}` means: go deeper. Trace the critical path, "
            f"read the source, find the real bottleneck, write real code.\n"
        )

    # The map (arbor/directions.py): the directions this round has explored, or
    # the instruction to write or update the map.
    from arbor import directions as _directions
    _map = _directions.current(campaign, state)
    if _map is not None:
        _map_block = _directions.render(_map, state, campaign)
        if _map_block:
            sections.append(_map_block)
    _map_round_open = _directions.suppresses_barren(_map, campaign, state)

    # Search space
    space = campaign.space
    if space:
        sections.append(
            f"\n## Search space\n\n"
            f"- Levels: {space.get('levels', [])}\n"
            f"- Current tier: **{state.tier}**\n"
            f"- Knobs: {space.get('knobs', [])}\n"
        )

    # Knowledge ledger — the agent-owned counterpart to knowledge/*.md.
    # Rendered before history on purpose: what is known and what is still open
    # matters more to the next decision than the order things were tried in.
    if campaign.session_root:
        from arbor import ledger as _ledger
        findings = _ledger.load(_ledger.ledger_path(campaign.session_root))
        if findings:
            rendered = _ledger.render(findings, campaign.config, scope=scope)
            if rendered:
                sections.append(rendered)

    # History — distilled, not dumped. Every KEEP is kept because those are the
    # changes actually sitting in the tree; everything else is windowed to the
    # recent past. Rendering all N forever crowds out the ledger above it.
    if state.lessons:
        sections.append(_render_history(state.lessons))

    # Probe budget — stated as an OPPORTUNITY, not only as a cap. Previously
    # this section rendered solely when probe_streak was non-zero, so an agent
    # that had not probed recently never saw that probing was available at all;
    # combined with a rising null_streak the only visible signal was pressure
    # to ship. Always render the budget, and when the loop is stuck at the
    # terminal tier say plainly that buying knowledge is the required move.
    # The secondary incumbent is suspect — say so before anything else, because
    # every verdict below it is being read through the same instrument.
    if getattr(state, "secondary_bar_suspect", None) and state.secondary_bar_suspect():
        _sec_key = (campaign.pack.objective.get("measurement", {})
                    .get("secondary_metric", "the second axis"))
        sections.append(
            f"## THE `{_sec_key}` BAR IS SUSPECT — measure it before trusting it\n\n"
            f"{state.secondary_only_streak} REVERTs in a row were decided by "
            f"`{_sec_key}` ALONE: the primary metric stayed inside its noise "
            f"band every time and only the second axis objected. Across "
            f"UNRELATED changes that is better explained by a high incumbent "
            f"than by that many real secondary-only regressions — "
            f"`best_secondary` is a high-water mark over single measurements, "
            f"so it keeps whichever draw happened to be high and drifts upward "
            f"with every ratchet.\n\n"
            f"The cheap experiment that settles it: **bench the CURRENT tree "
            f"UNCHANGED** and compare against `{_sec_key}` = "
            f"{getattr(state, 'best_secondary', None)}. A null re-bench that "
            f"comes back below the incumbent means the bar, not the change, is "
            f"what has been doing the rejecting — and the last "
            f"{state.secondary_only_streak} verdicts need re-reading. Declare "
            f"it `INTENT: LEARN`, and state the decision rule BEFORE you see "
            f"the number."
        )

    # The drought blocks. They render ABOVE the LEARN budget because the budget
    # they announce is the one the drought just widened. Not while the map is
    # written or updated: that round is LEARNs by design, and a drought notice
    # would push it to ship instead.
    _barren = None if _map_round_open else _render_barren(campaign, state)
    if _barren:
        sections.append(_barren)

    from arbor.state import MAX_CONSECUTIVE_PROBES
    _streak = getattr(state, "probe_streak", 0)
    _budget = (state.learn_budget() if hasattr(state, "learn_budget")
               else MAX_CONSECUTIVE_PROBES)
    remaining = _budget - _streak
    _levels = getattr(campaign.pack, "levels", None) or None
    _mandated = hasattr(state, "probe_mandated") and state.probe_mandated(_levels)
    if _map_round_open:
        sections.append(_directions.waiver_block(_map))
    elif remaining <= 0:
        sections.append(
            "## LEARN budget — EXHAUSTED\n\n"
            f"The last {_streak} iterations were LEARNs. "
            "This iteration must carry a real change (`WIN` or `FIX`), not another."
        )
    elif _mandated:
        sections.append(
            "## LEARN REQUIRED this iteration\n\n"
            f"{state.null_streak} iterations in a row have produced no advance, "
            f"and tier `{state.tier}` is the last one — there is nothing left to "
            "escalate into. That is the signal that the campaign does not know "
            "what to do next, which makes another blind change the worst "
            "available move.\n\n"
            "Declare `INTENT: LEARN` and close one of the open UNKNOWN or "
            "ASSUMPTION entries above. A LEARN that closes a finding resets the "
            "null streak — it counts as progress, because it is."
        )
    else:
        sections.append(
            f"## LEARN budget\n\n{remaining} consecutive LEARN(s) available. "
            "A LEARN that closes an open UNKNOWN or ASSUMPTION counts as "
            "progress and resets the null streak; use one whenever the next "
            "change would otherwise rest on an unverified assumption."
        )

    # KV trace analysis from the most recent iteration (if available)
    if campaign.session_root and state.iteration > 0:
        import json as _json
        kv_analysis = campaign.session_root / f"iter_{state.iteration}" / "kv_trace_analysis.json"
        if kv_analysis.exists():
            try:
                kv = _json.loads(kv_analysis.read_text())
                kv_lines = [
                    "## KV Cache Analysis (most recent iteration)\n",
                    f"- Duration: {kv.get('duration_s', 0):.0f}s",
                    f"- Evictions: {kv.get('num_evictions', 0):,}",
                    f"- GPU→CPU transfers: {kv.get('num_backups', 0):,} ({kv.get('total_backup_gb', 0):.1f} GB, BW p50={kv.get('backup_bw_p50_gbps', 0):.1f} GB/s)",
                    f"- CPU→GPU transfers: {kv.get('num_loadbacks', 0):,} ({kv.get('total_loadback_gb', 0):.1f} GB, BW p50={kv.get('loadback_bw_p50_gbps', 0):.1f} GB/s)",
                    "",
                    "Use `kv_trace_simulator.py` on this trace to test different HiCache configs without re-running the bench.",
                ]
                sections.append("\n".join(kv_lines))
            except Exception:
                pass

    return "\n\n".join(sections) + "\n"
