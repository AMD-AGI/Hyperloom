# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist sub-agent prompt assembler.

Returns ``(system_prompt, user_prompt)``: the system prompt carries the
immutable contract (identity / output protocol / iron rules) so the
backend can cache it across specialists; the user prompt carries per-task
context (hardware / gap / KB / recipe / PR / source hint). Most sections
render a ``(none)`` placeholder when empty; the execution-budget and
PD-disaggregation sections are omitted entirely. Not pure: the PD section
reads multi-node state (env + on-disk ``multi_node_state.json``) and the
enablement section builds the enablement mandate, which probes framework
source roots on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import first_text
from hyperloom.common.prompt_safety import defang_prompt_structure
from hyperloom.inference_optimizer.framework_registry import python_package
from .agentx_context import corpus_lines, grading_lines, launch_ownership_lines
from .proposal_schemas import (
    EXPLORE_VARIANT_CONTRACT,
    EXPLORE_VARIANT_EXAMPLE,
    UPSTREAM_CANDIDATE_CONTRACT,
    UPSTREAM_CANDIDATE_EXAMPLE,
)
from .specialist_focus import _DOMAIN_FOCUS_TEMPLATES, _is_agentx
from ..measurement.runtime_findings import CORRECTNESS_FIX_MAX_DROP_PCT, CORRECTNESS_FIX_RULES

from ..specialists.domains import domain_for_tag
from ..specialists.profile import MODE_PATCH
from ..specialists.prompt_inputs import SpecialistPromptInputs


_NONE_PLACEHOLDER = "(none)"
# GPU share below which re-dispatching an op to another backend is not worth the
# equivalence work; keeps per-op glue out of the substitution directive.
_VENDOR_SUBSTITUTION_MIN_GPU_PCT = 5.0


# Per-task-kind brief appended by _section_mandate; only the per-dispatch dynamic
# context (PR lead, critic feedback, apply errors, residual questions) lives in
# ``inp.notes``. Boilerplate that is the same for every dispatch of a given kind
# lives here so it is maintained in one place.
_TASK_KIND_BRIEFS: dict[str, str] = {
    "framework_authoring": (
        "Read the upstream PR as inspiration, then deliver the best win for this"
        " model / hardware / workload — go beyond the diff where the live source justifies it."
        " Deliverable: a unified-diff patch in your worktree (``patches_written``)"
        " OR a config/env ``proposal_set`` entry when the win is a flag toggle."
        " The Coordinator applies and benches it; you do not benchmark."
    ),
    "framework_local_explore": (
        "No upstream PR was found. Author the best throughput win directly from"
        " the live source + profiling evidence. Read the source trees in Section 7"
        " and the roofline (Section 4a) to locate the hot path."
        " You MAY use WebSearch / WebFetch to compare the local checkout against"
        " the latest upstream code and port a newer optimisation when behind."
        " Deliverable: a unified-diff patch (``patches_written``)"
        " OR a config/env ``proposal_set`` entry. You do not benchmark."
    ),
    "explore_apply_retry": (
        "A previous patch failed to apply against the live source tree."
        " Study the apply errors in the notes, produce a corrected patch."
    ),
}


# Forbids global process cleanup that could kill the optimizer's serving /
# benchmark process. Shared by bash-enabled specialist and leaf prompts.
BASH_KILL_SAFETY_PREAMBLE = (
    "Do NOT run global process cleanup. Never run `ps aux | grep ... | xargs "
    "kill`, `pgrep -f ... | xargs kill`, or `killall` — these can kill the "
    "optimizer's serving / benchmark process. Only manage processes you "
    "started yourself, by their own PID."
)


# Section 1 — Identity & autonomy
def _authors_patches(inp: SpecialistPromptInputs) -> bool:
    """Whether this dispatch may author patches; research mode gets no worktree."""
    return inp.mode == MODE_PATCH


#: Throughput domains whose focus block offers flags first, so the coda has
#: something to raise the sights from. ``enablement_specialist`` is absent
#: because its ladder book says climb only as far as runnability needs;
#: ``framework_rewrite_specialist`` because it authors source rewrites and
#: nothing else, and already carries its own "Where the wins are" taxonomy;
#: ``speculative_decoding_specialist`` because its focus names no flags at all.
_THROUGHPUT_AUTHORING_DOMAINS = frozenset(
    {
        "serving_specialist",
        "kernel_switch_specialist",
        "comm_specialist",
        "compiler_specialist",
        "system_specialist",
    }
)


def _source_patch_altitude_block(inp: SpecialistPromptInputs) -> list[str]:
    """Where the durable wins are, for a throughput dispatch that may author code."""
    if not (_authors_patches(inp) and inp.domain.key in _THROUGHPUT_AUTHORING_DOMAINS):
        return []
    return [
        "",
        "### Altitude — where the durable wins are",
        "",
        "The flags above are cheap to validate and their ceiling is fixed by",
        "whoever wired them up; a source change moves that ceiling and keeps",
        "paying in later rounds. When the cheap levers do not close the gap, stop",
        "permuting them and read the source for:",
        "- **An un-bridged capability switch** — a `*_supported()` / `*_enabled()`",
        "  / `is_*()` predicate that takes this model / GPU / precision off a",
        "  faster path. Bridging it creates a lever nobody had.",
        "- **A faster path already present** — reached never, or with arguments",
        "  that disqualify it.",
        "- **A real bug** — a wrong bound, a stale cache key, a fallback that",
        "  fires when it should not. Fixing it is a KEEP like any other.",
        "- **Component usage this architecture argues against** — the order,",
        "  granularity or fusion of the pieces, not their settings.",
        "Deliver a unified diff in your worktree `patches/` dir, small (target ≤5",
        "files) and grounded: a patch you cannot point at a profile line or a",
        "source predicate is budget spent for nothing.",
    ]


def _domain_focus_blocks(inp: SpecialistPromptInputs) -> list[str]:
    """Render the focus block of the active domain, then of each extra tag on a multi-tag dispatch, closing on the altitude coda that spans them."""
    body: list[str] = []
    rendered: set[str] = set()
    for key in (inp.domain.key, *(getattr(domain_for_tag(t), "key", "") for t in inp.extra_focus_tags)):
        focus = _DOMAIN_FOCUS_TEMPLATES.get(key) if key and key not in rendered else None
        lines = focus(inp) if focus is not None else []
        if not lines:
            continue
        body.extend(["", f"### Domain focus — {key}", "", *lines])
        rendered.add(key)
    body.extend(_source_patch_altitude_block(inp))
    return body


def _section_identity(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 1 (identity & autonomy) of the specialist prompt.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the identity section.
    """
    if _authors_patches(inp):
        capability_line = (
            "probe the host via Bash, **author source patches into your isolated"
            " worktree**, and use as many of your ``max_turns`` LLM turns as you need"
        )
        deliverable_line = "(Section 8) carrying ``proposal_set`` + ``patches_written``. The hard"
    else:
        capability_line = "probe the host via Bash, and use as many of your ``max_turns`` LLM turns as you need"
        deliverable_line = "(Section 8) carrying ``proposal_set``. The hard"
    if inp.allocated_gpu_ids:
        leaf_examples = "bench N candidates of one lever at once, or read several subsystems"
        leaf_devices = "inherit your VISIBLE_DEVICES (so they share your GPU and cannot oversubscribe), and "
    else:
        leaf_examples = "read several subsystems at once"
        leaf_devices = ""
    fanout_line = (
        f"Fan-out: to parallelize independent single-shot sub-tasks (e.g. {leaf_examples}), "
        'you MAY ``Task(subagent_type="hyperloom-leaf")``. Leaves are single-turn, '
        f"{leaf_devices}cannot fan out further. Use leaves for breadth; do multi-round "
        "depth (e.g. coordinate-descent autotune) yourself."
    )
    body: list[str] = [
        "## 1. IDENTITY & AUTONOMY",
        "",
        f"You are a fully autonomous **{inp.domain.key}** dispatched by the",
        f"Hyperloom Coordinator. Layer: {inp.domain.layer}.",
        f"KB anchor: {inp.domain.kb_anchor}.",
        "",
        f"Description: {inp.domain.description or '(generic)'}",
        "",
        "You operate **autonomously** inside your domain — no per-step approval",
        "is needed. You have full authority to read any code on this host — the",
        "trees in Section 7 are where to start, not where to stop — search any",
        "public GitHub repo or NVIDIA PR,",
        capability_line,
        "to be thorough. Be creative. Investigate deeply. One-turn shortcuts",
        "are discouraged when a real bottleneck is on the table — but stop once",
        "rounds stop yielding new findings; the wall clock is not the only stop",
        "signal. Quality over quantity: **2 proposals is the norm, 4 the hard",
        "cap**. One real beats two padded; an empty ``proposal_set`` with a",
        "clear ``summary`` beats one padded.",
        "",
        "Division of labour: the Coordinator owns the serving GPU, runs the E2E",
        "benchmark, and decides KEEP/REVERT — you do not have to validate final",
        "throughput yourself. Your single deliverable is ONE final ``specialist_done``",
        deliverable_line,
        "capability boundary is fixed by Section 9 Iron Rules; everything inside",
        "it is yours.",
        "",
        fanout_line,
    ]
    body.extend(_domain_focus_blocks(inp))
    if inp.scope == "domains":
        body.extend(_cross_domain_block(inp))
    elif inp.scope == "freeform":
        body.extend(_freeform_block(inp))
    if inp.allocated_gpu_ids:
        body.extend(_gpu_autonomy_block(inp))
    body.extend(_cpu_selfcheck_block(inp))
    if inp.auto_retry_reason.strip():
        body.extend(_auto_retry_note_block(inp))
    return body


def _auto_retry_note_block(inp: SpecialistPromptInputs) -> list[str]:
    """Heads-up block when this dispatch is a bounded auto-retry of a prior
    transient (timeout / crash / stale-heartbeat) attempt. Advisory only —
    the mandate is unchanged; the note nudges the specialist to scope its work
    so it finishes within budget this time.

    Args:
        inp: The specialist prompt inputs (reads ``auto_retry_reason``).

    Returns:
        The rendered auto-retry notice lines.
    """
    reason = inp.auto_retry_reason.strip()
    return [
        "",
        "### Auto-retry notice",
        "",
        "Your previous attempt on this task did NOT finish cleanly — the "
        f"Coordinator is re-dispatching you. Reason: ``{reason}``.",
        "This is a transient infrastructure failure (a timeout, crash, or "
        + "silent hang), not a rejection of the approach. Scope your "
        + "investigation so you reach a single ``specialist_done`` within "
        + "``max_turns`` this time: prefer fewer, higher-confidence probes, "
        + "emit heartbeats, and avoid long-running shell that risks the same "
        + "timeout.",
    ]


def _gpu_autonomy_block(inp: SpecialistPromptInputs) -> list[str]:
    """On-GPU autonomy block appended for GPU specialists (those with a card
    allocation). Frames the broad capabilities the specialist has on its own
    leased cards and surfaces the *optional* ``rebench`` helper — none of it is
    a mandate; the Coordinator's ``integrate_patch`` E2E gate stays the single
    authoritative measure of truth.

    Args:
        inp: The specialist prompt inputs.

    Returns:
        The rendered on-GPU autonomy lines.
    """
    cards = ", ".join(str(g) for g in inp.allocated_gpu_ids)
    return [
        "",
        "### On-GPU autonomy (your leased cards)",
        "",
        f"You exclusively own GPU card(s) [{cards}] for this task. On those "
        + "cards you are free to do whatever converges on a benched win:",
        "- For kernel/config autotune, search the installed framework/source "
        + "first for maintained benchmark/tuning entrypoints, config lookup "
        + "paths, and nearby config families; prefer those.",
        "- If the built-in path is missing or incomplete, write a small "
        + "source-derived harness around the framework primitive/config override "
        + "API. Use warmups, true-default/current/candidate baselines, "
        + "median/min-of-reps, and an accuracy guard.",
        "- Write and run arbitrary scripts — autotune harnesses, "
        + "microbenchmarks, profilers (rocprof / torch.profiler / your own "
        + "breakdown).",
        "- Start / restart a real server on your own cards and benchmark it " + "however you see fit.",
        "- Profile freely to get a fresh trace after a change — don't rely only "
        + "on the static roofline snapshot you were handed.",
        "- Tune the framework's config-file levers (e.g. MoE/GEMM/attention "
        + "Triton config JSONs) — a missing/untuned config is often the single "
        + "biggest lever.",
        "- Self-check accuracy (advisory ``max_abs_err`` / gsm8k) when you want "
        + "to — the Coordinator gate stays authoritative, so this is guidance, "
        + "not a requirement.",
        "",
        "Optional helper: a ``rebench`` convenience reuses the real Magpie "
        + "serving + benchmark path on your leased cards, so you can get numbers "
        + "directly comparable to the ``integrate_patch`` gate in one call:",
        "    python -m hyperloom.orchestrator.specialists.rebench \\",
        "        --config <magpie.yaml> --output ./scratch/rebench " + "[--extra-args '<server args>']",
        "  It prints a JSON result with ``output_throughput``. It is OPTIONAL "
        + "— you may instead write your own bench/autotune script. Throughput "
        + "does NOT have to come from rebench.",
    ]


def _cpu_selfcheck_block(inp: SpecialistPromptInputs) -> list[str]:
    """Optional ``selfcheck`` helper for a patch specialist with a worktree and no GPU."""
    if inp.allocated_gpu_ids or inp.mode != MODE_PATCH or not inp.worktree_package_dir:
        return []
    package = python_package(inp.framework)
    if package is None:
        return []
    return [
        "",
        "Optional helper: ``selfcheck`` installs your worktree's package into a private venv,",
        "imports it, byte-compiles the files you changed and runs any pytest targets you pass:",
        "    python -m hyperloom.orchestrator.specialists.selfcheck \\",
        f"        --worktree {inp.workspace_path} --package-dir {inp.worktree_package_dir} "
        f"--package {package} [--pytest <target>]",
        "  It prints a JSON result. A framework with compiled extensions (e.g. vLLM) rebuilds",
        "  them on install, which can take well over your wall budget.",
    ]


def _freeform_block(inp: SpecialistPromptInputs) -> list[str]:
    """Free-form mandate appended when ``scope == 'freeform'``. The
    specialist is NOT bound
    to the domain catalogue — the Orchestration ``task_description`` is the
    whole mandate. The single deliverable is still ONE ``specialist_done``.

    Args:
        inp: The specialist prompt inputs (reads ``task_description``).

    Returns:
        The rendered free-form mandate lines.
    """
    desc = (inp.task_description or "").strip() or "(no task description provided)"
    if _authors_patches(inp):
        reach = "upstream PRs, host probing, source patches"
        deliverable = "``proposal_set`` + ``patches_written``"
    else:
        reach = "upstream PRs, host probing"
        deliverable = "``proposal_set``"
    return [
        "",
        "### Free-form mandate (scope = freeform)",
        "",
        "You are dispatched as a **free-form** specialist: you are NOT bound to "
        + "the domain catalogue above. The Orchestration mandate below is your "
        + f"whole task — investigate it wherever it leads (framework internals, {reach}).",
        "",
        "Mandate from Orchestration:",
        "",
        f"> {desc}",
        "",
        "Set ``scope='freeform'`` on each proposal. Your single deliverable is "
        + f"still ONE ``specialist_done`` carrying {deliverable}. Never "
        + "self-report numeric speedups — the Coordinator measures gain.",
    ]


def _cross_domain_block(inp: SpecialistPromptInputs) -> list[str]:
    """Cross-domain mandate appended when ``scope == 'domains'``. The
    single deliverable is still
    ONE ``specialist_done``; the difference is the patch may span every domain
    in scope and the Critic will hold it to the cross-domain rules.

    Args:
        inp: The specialist prompt inputs (reads ``extra_focus_tags`` /
            ``domain``).

    Returns:
        The rendered cross-domain mandate lines.
    """
    tags = ", ".join(inp.extra_focus_tags) if inp.extra_focus_tags else inp.domain.key
    return [
        "",
        "### Cross-domain mandate (scope = domains)",
        "",
        f"You are dispatched as a **cross-domain** specialist over: {tags}.",
        "You may author a single coherent patch that spans these domains "
        + "together when (and only when) the change must happen jointly — a "
        + "combination no single-domain specialist could surface from within "
        + "its own boundary.",
        "",
        "In your ``specialist_done`` you MUST justify the combination:",
        "- give an independent rationale for the change **within each domain** " + "in scope;",
        "- name the **coupling points** (why these changes must land together) "
        + "and at least one **side effect** of the combination;",
        "- show this is genuine cross-domain synthesis, not a concatenation of "
        + "two independent single-domain edits (that is an explore grid combo, "
        + "not a cross-domain patch).",
        "Set ``scope='domains'`` on the proposal so the Critic attaches the "
        + "cross-domain review rules. Never self-report numeric speedups — the "
        + "Coordinator measures gain.",
    ]


# Section 0 — Mandate (deliverable contract + run status)
def _section_mandate(inp: SpecialistPromptInputs) -> list[str]:
    """Render §0: deliverable contract, run status, and task-kind brief."""
    # Deliverable line based on scope × mode.
    if inp.scope == "freeform":
        anchor = inp.task_description.split("\n")[0].strip()[:120]
        deliverable = "freeform investigation — see task description"
    else:
        anchor = inp.gap_canonical_id
        if not _authors_patches(inp):
            deliverable = "findings and up to 6 ranked config variants (read-only; no patch)"
        elif inp.scope == "domains":
            deliverable = "a coupled patch spanning multiple domains + up to 6 ranked config variants"
        else:
            deliverable = "a source patch and/or up to 6 ranked config variants addressing the gap below"

    rows: list[str] = [
        "## 0. MANDATE",
        "",
        f"- deliverable: {deliverable}",
    ]
    if anchor:
        rows.append(f"- anchor: `{anchor}`")
    rows.extend(_mandate_run_status(inp))

    rows.extend(
        [
            "",
            "Judged by: the Coordinator benches your proposals end-to-end against",
            "the sealed baseline and decides KEEP/REVERT; the accuracy gate runs",
            "alongside. You are not asked to prove the number.",
        ]
    )

    brief = _TASK_KIND_BRIEFS.get(inp.task_kind.strip(), "")
    if brief:
        rows.extend(["", brief])

    if inp.pr_lead:
        title = defang_prompt_structure(first_text(inp.pr_lead.get("title")))
        url = first_text(inp.pr_lead.get("url"))
        diff_url = first_text(inp.pr_lead.get("diff_url"))
        rows.append("")
        if title:
            rows.append(f"PR lead: {title}")
        if url:
            rows.append(f"PR url: {url}")
        if diff_url:
            rows.append(f"Diff: {diff_url} (fetch with WebFetch)")

    if inp.prior_attempts:
        rows.extend(
            [
                "",
                "Already tried this session — avoid the same or equivalent change:",
            ]
        )
        for att in inp.prior_attempts[:20]:
            ref = first_text(att.get("ref"))
            if not ref:
                continue
            status = first_text(att.get("status"), "?")
            gain = att.get("gain_pct")
            gain_str = f" gain={float(gain):+.2f}%" if isinstance(gain, (int, float)) else ""
            why = first_text(att.get("why"))
            why_str = f" — {why}" if why else ""
            rows.append(f"  - {ref} [{status}]{gain_str}{why_str}")

    return rows


def _mandate_run_status(inp: SpecialistPromptInputs) -> list[str]:
    """Render §0's read-only run status; empty until a baseline or a KEEP threshold exists."""
    from hyperloom.inference_optimizer.framework_registry import format_primary_metric

    if not (inp.baseline_tput > 0 or inp.keep_threshold_pct > 0):
        return []
    rows = ["", "Run status (read-only context; do NOT re-state these as your own measurements):"]
    if inp.baseline_tput > 0:
        rows.append(f"- baseline: {format_primary_metric(inp.framework, inp.baseline_tput)}")
    if inp.current_tput > 0:
        rows.append(f"- current best: {format_primary_metric(inp.framework, inp.current_tput)}")
    if inp.current_launch_args:
        rows.append(f"- current best server args (complete launch, not a delta): {inp.current_launch_args}")
    if inp.cumulative_gain_validated != 0:
        rows.append(f"- validated cumulative gain: {inp.cumulative_gain_validated:+.2f}%")
    if inp.keep_threshold_pct > 0:
        rows.append(f"- KEEP threshold this cycle: {inp.keep_threshold_pct:.2f}%")
    if inp.applied_stack:
        stack_items = ", ".join(
            f"{e.get('variant_name', '?')} ({e.get('gain_pct', 0):+.2f}%)" for e in inp.applied_stack[:6]
        )
        rows.append(f"- applied stack: {stack_items}")
    return rows


# Section 2 — Hardware context
def _section_hardware(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 2 (hardware + workload context) of the prompt.

    Emits GPU type, TP, HBM, peak TFLOPs, and any populated workload
    fields (precision, concurrency, ISL/OSL, max_model_len, arch notes).

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the hardware-context section.
    """
    rows: list[str] = ["## 2. HARDWARE CONTEXT", ""]
    if inp.gpu_type:
        rows.append(f"- gpu_type: {inp.gpu_type}")
    else:
        rows.append(f"- gpu_type: {_NONE_PLACEHOLDER}")
    if inp.allocated_gpu_ids:
        rows.append("- allocated specialist GPU ids: " + ", ".join(str(g) for g in inp.allocated_gpu_ids))
    if inp.tp > 0:
        rows.append(f"- TP: {inp.tp}")
    else:
        rows.append(f"- TP: {_NONE_PLACEHOLDER}")
    # Workload context.
    workload_rows: list[str] = []
    if inp.precision:
        workload_rows.append(f"- precision: {inp.precision}")
    if inp.conc > 0:
        workload_rows.append(f"- concurrency: {inp.conc}")
    if _is_agentx(inp):
        # The corpus fixes the request shape, so ISL/OSL carry no information.
        workload_rows += corpus_lines(inp.agentx_corpus_shape)
        workload_rows += grading_lines(inp.agentx_grading, inp.agentx_backend)
        workload_rows += launch_ownership_lines(inp.agentx_backend)
    else:
        if inp.isl > 0:
            workload_rows.append(f"- ISL (input seq len): {inp.isl}")
        if inp.osl > 0:
            workload_rows.append(f"- OSL (output seq len): {inp.osl}")
    if inp.max_model_len > 0:
        workload_rows.append(f"- max_model_len: {inp.max_model_len}")
    if workload_rows:
        rows.append("")
        rows.append("Workload:")
        rows.extend(workload_rows)
    if inp.arch_notes:
        rows.append("")
        rows.append(f"Model architecture (advisory): {inp.arch_notes}")
    if inp.target_gap_notes:
        rows.append("")
        rows.append(inp.target_gap_notes)
    return rows


# Section 2a — Execution budget (wall-clock)
def _section_execution_budget(inp: SpecialistPromptInputs) -> list[str]:
    """Render the wall-clock budget block so the specialist can self-throttle.

    Renders the time left on the dispatch deadline (seconds + minutes) and the
    dispatch start timestamp. Returns ``[]`` when no budget was supplied, so the
    section is omitted entirely rather than emitting a placeholder.

    Args:
        inp: The specialist prompt inputs (reads ``wall_budget_sec`` /
            ``started_at_iso``).

    Returns:
        The rendered execution-budget section lines, or ``[]`` when no budget
        is set.
    """
    if inp.wall_budget_sec <= 0:
        return []
    mins = inp.wall_budget_sec / 60.0
    rows = [
        "## 2a. EXECUTION BUDGET (wall-clock)",
        "",
        f"- Hard wall-clock budget for this entire dispatch: **{inp.wall_budget_sec:.0f}s (~{mins:.0f} min)**.",
    ]
    if inp.started_at_iso:
        rows.append(f"- Dispatch started at: {inp.started_at_iso} (UTC).")
    rows.extend(
        [
            "- The Coordinator hard-kills your subprocess when this budget is "
            + "exhausted — turns are NOT the stop signal. Scope your work to "
            + "reach a deliverable conclusion inside the budget.",
            "- Self-throttle: check elapsed wall-clock with Bash "
            + "(``date -u +%s`` vs the start above), keep your "
            + "``specialist_done.partial.json`` checkpoint current, and write "
            + "the final ``specialist_done.json`` before the budget runs out so "
            + "your best work is never lost to a kill.",
        ]
    )
    return rows


# Section 3 — Gap statement
def _section_gap(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 3 (gap statement) of the specialist prompt.

    Emits the canonical gap id, layer, symptom, and most-recent evidence
    JSON, or a ``(none)`` placeholder when no gap is set.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the gap-statement section.
    """
    rows = ["## 3. GAP STATEMENT", ""]
    if not inp.gap_canonical_id:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    rows.append(f"- gap_canonical_id: `{inp.gap_canonical_id}`")
    if inp.gap_layer:
        rows.append(f"- layer: {inp.gap_layer}")
    if inp.gap_symptom:
        rows.append(f"- symptom: {inp.gap_symptom}")
    if inp.gap_evidence:
        rows.append("")
        rows.append("Most recent evidence:")
        rows.append("```json")
        rows.append(json.dumps(inp.gap_evidence, sort_keys=True, indent=2))
        rows.append("```")
    return rows


# Section 4 — optional KB context
def _is_cold_start(inp: SpecialistPromptInputs) -> bool:
    """Return True when every prior KB/PR/research source is empty, so the
    cold-start directive is injected in place of the KB block.

    Args:
        inp: The specialist prompt inputs.

    Returns:
        True when every prior KB/PR/research source is empty.
    """
    return (
        not inp.experience_kb_block
        and not inp.warm_start_recipe
        and not inp.warm_start_lessons
        and not inp.warm_start_pitfalls
        and not inp.research_hints
    )


# Cold-start directive: fall back to the Section 1 defaults, or exit empty with a rationale.
_COLD_START_DIRECTIVE = (
    "**COLD-START MODE — no priors available.**",
    "",
    "All prior sources for this gap are empty:",
    "",
    "- KB context: ``(none)`` — no RecipeKB warm-start facts or "
    + "research hints were available for this "
    + "(model, hardware, domain) tuple.",
    "- Warm-start recipe: ``(none)`` (Section 5).",
    "- Use ``mcp__pr_monitor__*`` tools (Section 6) to query PRs on demand.",
    "",
    "**Directive — a coin-flip proposal is worse than none.** "
    + "Treat the *Winning techniques* + *Pitfalls* bullets in "
    + "Section 1 as your fallback prior and take the **1–2 "
    + "most conservative, well-attested defaults** that fit "
    + "the hardware (Section 2) and the gap symptom "
    + "(Section 3); flag each ``provenance: "
    + "domain_focus_default`` and call it an unvalidated "
    + "fallback in the proposal's ``reason``. If none clears "
    + "that bar, emit ``proposal_set=[]`` and cite in ``summary`` "
    + "which you considered and why each was rejected — a "
    + "bare empty exit with no rationale reads as a tool "
    + "failure. Do NOT add a ``confidence`` field: "
    + "self-reported confidence / gain fields are stripped "
    + "before review. Record in ``residual_questions`` what "
    + "RecipeKB, research, or ``mcp__pr_monitor__*`` query a "
    + "future round should pursue.",
)


def _section_kb_context(inp: SpecialistPromptInputs) -> list[str]:
    """Build the advisory KB-context section of the specialist prompt.

    Renders the research hints, the cold-start directive, or the none placeholder.

    Args:
        inp: Assembled prompt inputs for the current dispatch.

    Returns:
        Prompt lines for the KB-context section.
    """
    rows = ["## 4. KB CONTEXT (optional, advisory)", ""]
    if inp.research_hints:
        rows.extend(
            [
                "Structured KB context is empty for this (model, hardware, domain), but "
                + "the research scout collected source-backed priors this "
                + "session. Treat these as your advisory prior (co-equal with "
                + "RecipeKB priors; the Critic still gates the final answer):",
                "",
                inp.research_hints,
                "",
                "Anchor proposals on these hints where they fit the gap " + "(Section 3) and hardware (Section 2).",
            ]
        )
        return rows
    if _is_cold_start(inp):
        rows.extend(_COLD_START_DIRECTIVE)
    else:
        rows.extend(
            [
                _NONE_PLACEHOLDER,
                "",
                "(No structured KB context supplied. Use Sections 1, 3, 5, "
                + "and 6 plus source inspection; record missing RecipeKB / "
                + "research / PR questions in ``residual_questions`` so a "
                + "future round can warm richer advisory context.)",
            ]
        )
    return rows


def _section_experience_kb(inp: SpecialistPromptInputs) -> list[str]:
    """Render the Experience KB section; omitted when this dispatch rendered no Experience.

    Args:
        inp: Assembled prompt inputs for the current dispatch.

    Returns:
        Prompt lines carrying the Experience service block, or ``[]``.
    """
    if not inp.experience_kb_block:
        return []
    return [
        "## 4b. EXPERIENCE KB (measured outcomes from earlier sessions)",
        "",
        "Compare each Experience's identity and baseline configuration with Sections 2 and 3 "
        + "before relying on it. When one shaped a proposal, cite it in that proposal's "
        + "``experience_citations``, or for a patch you wrote in the payload's top-level "
        + "``experience_citations``: ``{id, stance, claim}`` with ``stance`` one of ``adopt`` "
        + "(you did its change), ``adapt`` (you did it modified), ``avoid`` (you left it out "
        + "because of its outcome), or ``contrast`` (you chose a different change designed "
        + "against it), and ``claim`` one sentence on why. Only ids shown below are kept.",
        "",
        inp.experience_kb_block,
    ]


def _vendor_substitution_candidates(hot_kernels: Any) -> list[dict[str, Any]]:
    """Select hot ``aten::`` ops worth re-dispatching to a different backend.

    An ``aten::`` name means the op still runs through PyTorch's own dispatch, so
    an alternative backend is available to the call site. A vendor kernel that is
    already in the trace under its own name has nothing left to swap.

    Args:
        hot_kernels: The ``hot_kernels_top15`` rows from the roofline evidence.

    Returns:
        Qualifying rows ordered by descending GPU share.
    """
    if not isinstance(hot_kernels, list):
        return []
    rows: list[dict[str, Any]] = []
    for entry in hot_kernels:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        if not name.startswith("aten::"):
            continue
        gpu_pct = entry.get("gpu_pct")
        if not isinstance(gpu_pct, (int, float)) or float(gpu_pct) < _VENDOR_SUBSTITUTION_MIN_GPU_PCT:
            continue
        rows.append(entry)
    rows.sort(key=lambda row: float(row.get("gpu_pct") or 0.0), reverse=True)
    return rows


def _vendor_substitution_directive(hot_kernels: Any) -> list[str]:
    """Render the backend-substitution directive for hot ATen ops (empty when none qualify)."""
    candidates = _vendor_substitution_candidates(hot_kernels)
    if not candidates:
        return []
    rows = [
        "**Backend substitution — these ops still dispatch through PyTorch:**",
        "",
        "| name | gpu_pct | category | call site |",
        "|---|---:|---|---|",
    ]
    for entry in candidates:
        gpu_pct = float(entry.get("gpu_pct") or 0.0)
        rows.append(
            f"| {entry.get('name')} | {gpu_pct:.2f}% | "
            f"{entry.get('kernel_category') or ''} | {entry.get('source_file') or ''} |"
        )
    rows.extend(
        [
            "",
            "You may not rewrite the body of a kernel PyTorch owns. You MAY change "
            "which kernel the **call site** dispatches to, and when the call site is "
            "in the tree you are optimizing that is an ordinary source rewrite — "
            "same class as any other switch you author, behind its own default-off "
            "environment switch, with the original path kept as the fallback.",
            "",
            "Section 7 lists the accelerator libraries installed in this container. "
            "Read them to find an entry point matching the op's semantics and "
            "tensor layout, and verify equivalence on the real shapes before "
            "proposing it. An op holding a large share of device time is worth more "
            "than anything you can win from the glue around it.",
        ]
    )
    return rows


def _section_runtime_findings(inp: SpecialistPromptInputs) -> list[str]:
    """Render the scanned runtime findings; omitted when none were injected."""
    if not inp.runtime_findings:
        return []
    return [
        "## 4c. RUNTIME FINDINGS (current best server.log, already scanned)",
        "",
        "Do not re-read the server log for these. A disabled or falling-back hot "
        "path is restored first; do not optimize the fallback implementation.",
        "",
        "```text",
        inp.runtime_findings,
        "```",
    ]


_ANALYSIS_MD_READ_GUIDE = (
    "Use the `Read` tool on this path for the full TraceLens report (~10-20 KB). All section headings are stable: "
    "`## Executive Summary` / `## Top Operations` / `## Compute Kernel Optimizations` / "
    "`## Kernel Fusion Opportunities` / `## System-Level Optimizations` / `## Recommendations`."
)


def _section_roofline_evidence(inp: SpecialistPromptInputs) -> list[str]:
    """Render the ROOFLINE EVIDENCE section from ``inp.roofline_evidence``;
    empty evidence renders a heading + ``(none)`` placeholder.

    Args:
        inp: The specialist prompt inputs (reads ``roofline_evidence``).

    Returns:
        The rendered roofline-evidence section lines.
    """
    rows = ["## 4a. ROOFLINE EVIDENCE", ""]
    ev = inp.roofline_evidence
    if not ev:
        rows.append(
            "(none — no fresh roofline snapshot has been recorded yet. "
            "The Coordinator auto-enqueues `roofline` at the end of "
            "PRELUDE and again after every 10% watermark crossing; if "
            "you are seeing this, the snapshot is still in-flight.)"
        )
        return rows

    snap_id = ev.get("roofline_snapshot_id")
    if snap_id is not None:
        rows.append(f"**TraceLens snapshot #{snap_id}**")
        rows.append("")

    summary = ev.get("executive_summary")
    if summary:
        rows.append("**Executive Summary:**")
        for label, key in (
            ("Compute %", "compute_pct"),
            ("Idle %", "idle_pct"),
            ("Exposed Comm %", "comm_pct"),
            ("Exposed memcpy %", "memcpy_pct"),
            ("Top bottleneck", "top_bottleneck"),
        ):
            val = summary.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, (int, float)):
                rows.append(f"- {label}: {float(val):.1f}%")
            else:
                rows.append(f"- {label}: {val}")
        rows.append("")

    hot = ev.get("hot_kernels_top15")
    if hot:
        rows.append("**Top hot kernels (kernel_id | name | gpu_pct | bottleneck | source_file):**")
        rows.append("")
        rows.append("| kernel_id | name | gpu_pct | bottleneck | source_file |")
        rows.append("|---|---|---:|---|---|")
        for k in hot:
            kid = str(k.get("kernel_id") or "")
            name = str(k.get("name") or "")
            gpu_pct = k.get("gpu_pct")
            gpu_pct_str = f"{float(gpu_pct):.2f}%" if isinstance(gpu_pct, (int, float)) else "—"
            bottleneck = str(k.get("bottleneck") or "")
            src = str(k.get("source_file") or "")
            rows.append(f"| `{kid}` | {name} | {gpu_pct_str} | {bottleneck} | {src} |")
        rows.append("")

    directive = _vendor_substitution_directive(hot)
    if directive:
        rows.extend(directive)
        rows.append("")

    analysis_path = ev.get("analysis_md_path")
    if analysis_path:
        is_md = Path(analysis_path).suffix == ".md"
        guide = _ANALYSIS_MD_READ_GUIDE if is_md else "Use the `Read` tool on this path for the full TraceLens report."
        rows += [f"**Full {'analysis.md' if is_md else 'TraceLens report'} path:** `{analysis_path}`", "", guide]
    return rows


# Section 5 — Recipe summary
def _section_recipe(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 5 (warm-start recipe summary) of the prompt.

    Dumps the T0 warm-start recipe as JSON, or a ``(none)`` placeholder
    when no warm-start recipe was supplied.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the recipe-summary section.
    """
    rows = ["## 5. WARM-START RECIPE SUMMARY", ""]
    if not inp.warm_start_recipe:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    rows.append("**find-recipe result:**")
    rows.append("```json")
    rows.append(json.dumps(inp.warm_start_recipe, sort_keys=True, separators=(",", ":")))
    rows.append("```")
    return rows


# Section 5b — Related lessons (positive priors from prior KEEPs)
def _section_lessons(inp: SpecialistPromptInputs) -> list[str]:
    """Render the recipe row's ``lessons`` from prior KEEPs, compactly
    (statement + measured_impact).

    Rows are the flat shape ``Recipe.to_dict`` writes —
    ``{statement, measured_impact, ...}`` — normalised by ``recipe_kb_t0``
    before they land on ``warm_start_lessons``.

    Args:
        inp: The specialist prompt inputs (reads ``warm_start_lessons``).

    Returns:
        The rendered related-lessons section lines.
    """
    rows = ["## 5b. RELATED LESSONS (prior KEEPs on this model+hw)", ""]
    for point in inp.warm_start_lessons:
        line = _kb_point_line(inp, point, text_key="statement", count_label="validated")
        if not line:
            continue
        rows.append(line)
        impact_str = _render_measured_impact(point.get("measured_impact")) if isinstance(point, dict) else ""
        if impact_str:
            rows.append(f"    impact: {impact_str}")
    if len(rows) == 2:  # only the header + blank line: no lessons, or all filtered out
        rows.append(_NONE_PLACEHOLDER)
    return rows


def _kb_point_line(
    inp: SpecialistPromptInputs,
    point: Any,
    *,
    text_key: str,
    count_label: str,
    lead_meta_key: str = "",
) -> str:
    """Render one RecipeKB lesson / pitfall row as a prompt bullet.

    Args:
        inp: The specialist prompt inputs (reads the framework version).
        point: The row: a ``{text_key, confidence, validated_count, ...}``
            dict, or a bare string.
        text_key: The dict key holding the row's text.
        count_label: The label ``validated_count`` renders under.
        lead_meta_key: A dict key rendered first in the metadata, if any.

    Returns:
        The bullet line, or "" when the row carries no text.
    """
    # External warm-start data may arrive as a plain string rather than a
    # dict "point"; render the bare text to tolerate shape drift.
    if isinstance(point, str):
        text = point.strip()
        return f"- **{defang_prompt_structure(text)}**" if text else ""
    if not isinstance(point, dict):
        return ""
    text = str(point.get(text_key) or "").strip()
    if not text:
        return ""
    meta_bits: list[str] = []
    lead = str(point.get(lead_meta_key) or "").strip() if lead_meta_key else ""
    if lead:
        meta_bits.append(f"{lead_meta_key}={lead}")
    conf = point.get("confidence")
    if isinstance(conf, (int, float)) and conf > 0:
        meta_bits.append(f"conf={float(conf):.2f}")
    # validated_count is the strongest cross-session signal; fall back to source_session_id.
    vc = point.get("validated_count")
    if isinstance(vc, int) and vc > 1:
        meta_bits.append(f"{count_label}={vc}")
    recent_ids = point.get("source_session_ids")
    if isinstance(recent_ids, list) and recent_ids:
        meta_bits.append(f"recent={recent_ids[-1]}")
    else:
        src_sid = str(point.get("source_session_id") or "").strip()
        if src_sid:
            meta_bits.append(f"src={src_sid}")
    meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
    # Version-mismatch annotation; the LLM gets the final call.
    return f"- **{defang_prompt_structure(text)}**{meta}{_format_version_note(inp, point)}"


def _format_version_note(
    inp: SpecialistPromptInputs,
    row: dict[str, Any],
) -> str:
    """Render a ``[from sglang@X.Y, you're on A.B]`` annotation when
    the row's framework_version differs; empty when either side is
    unknown or they match.

    Args:
        inp: The specialist prompt inputs (reads ``framework`` /
            ``framework_version``).
        row: The lesson or pitfall row (reads ``framework_version``).

    Returns:
        The version-mismatch annotation, or "" when unknown or matching.
    """
    lesson_fv = str(row.get("framework_version") or "").strip()
    current_fv = (inp.framework_version or "").strip()
    if not lesson_fv or not current_fv:
        return ""
    if lesson_fv == current_fv:
        return ""
    framework_label = (inp.framework or "framework").strip() or "framework"
    return f" [from {framework_label}@{lesson_fv}, you're on {current_fv}]"


def _render_measured_impact(raw: Any) -> str:
    """Back-compat renderer for a row's ``measured_impact`` (dict, legacy
    string, or other).

    Args:
        raw: The ``measured_impact`` value (dict, string, None, or other).

    Returns:
        A compact human-readable impact string ("" when ``raw`` is None).
    """
    if isinstance(raw, dict):
        parts: list[str] = []
        gain = raw.get("gain_pct")
        if isinstance(gain, (int, float)):
            parts.append(f"+{float(gain):.2f}%")
        tput = raw.get("throughput_after")
        if isinstance(tput, (int, float)):
            parts.append(f"tput={float(tput):.1f}")
        depth = raw.get("stack_depth_at_apply")
        if isinstance(depth, int):
            parts.append(f"depth={depth}")
        when = str(raw.get("measured_at") or "").strip()
        if when:
            parts.append(when[:10])  # keep yyyy-mm-dd for compactness
        return ", ".join(parts)
    if isinstance(raw, str):
        return raw.strip()
    if raw is None:
        return ""
    return str(raw).strip()


# Section 5c — Known pitfalls (anti-priors from prior REVERTs)
def _section_pitfalls(inp: SpecialistPromptInputs) -> list[str]:
    """Render the recipe row's ``pitfalls`` from prior REVERTs (description +
    severity); framed as forbidden paths, not suggestions.

    Rows are the flat ``{description, severity, ...}`` shape ``Recipe.to_dict``
    writes, normalised by ``recipe_kb_t0``.

    Args:
        inp: The specialist prompt inputs (reads ``warm_start_pitfalls``).

    Returns:
        The rendered known-pitfalls section lines.
    """
    rows = ["## 5c. KNOWN PITFALLS (do NOT repeat — prior REVERTs)", ""]
    for point in inp.warm_start_pitfalls:
        line = _kb_point_line(inp, point, text_key="description", count_label="observed", lead_meta_key="severity")
        if line:
            rows.append(line)
    if len(rows) == 2:  # only the header + blank line: no pitfalls, or all filtered out
        rows.append(_NONE_PLACEHOLDER)
    return rows


# Section 6 — PR query capability
def _section_pr_feed(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 6 (PR query capability) of the specialist prompt.

    When the PR Monitor MCP is available, describes the self-serve query tools
    and lists the global repo allowlist the specialist may query. When
    unavailable, outputs a placeholder.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the PR-query capability section.
    """
    from hyperloom.orchestrator.policy.gate import PR_MONITOR_TOOL_NAMES
    from hyperloom.orchestrator.specialists.domains import PR_QUERY_REPOS

    rows = ["## 6. PR MONITOR", ""]
    if not inp.pr_monitor_available:
        rows.append("(unavailable: pr_monitor disabled)")
        return rows
    _prefix = "mcp__pr_monitor__"
    tool_shorts = sorted(t[len(_prefix) :] for t in PR_MONITOR_TOOL_NAMES if t.startswith(_prefix))
    rows += [
        "Use ``mcp__pr_monitor__*`` tools to query PRs on demand:",
        " / ".join(f"``{t}``" for t in tool_shorts),
        "",
        "Repos you may query:",
    ]
    for repo in PR_QUERY_REPOS:
        rows.append(f"- {repo}")
    return rows


# Section 7 — Local source navigation hint
def _source_root_row(root: str, *, worktree_base: str) -> str:
    """Render one source root, annotated with what a patch against it can do.

    Args:
        root (str): The source root to render.
        worktree_base (str): The tree the specialist's worktree stands for,
            when there is one.

    Returns:
        str: A markdown list row for the root.
    """
    base = (worktree_base or "").rstrip("/")
    if base and Path(root.rstrip("/")).is_relative_to(base):
        if (Path(base) / ".git").exists():
            return f"- {root} — git checkout; your worktree was cut from it"
        return f"- {root} — installed package; your worktree holds a git snapshot of it, so edit its files there"
    if (Path(root) / ".git").is_dir():
        return f"- {root} — git checkout"
    return f"- {root} — installed package, no git tree: re-author upstream diffs against it, never apply them"


def _resolve_focus_dir(hint: str, session_tree: str) -> str:
    """Return ``hint`` as an absolute path under the session tree.

    Checklist directories are repo-relative (``vllm/model_executor/...``), and
    what that is relative to depends on the tree's shape: a pip-installed tree is
    the package directory itself (``.../dist-packages/vllm/``) and joins onto its
    parent, while a checkout (``/sgl-workspace/vllm/``) holds the package one
    level down and joins directly. The tree decides, not the hint -- matching on
    the hint's leading segment reads a checkout as a package tree whenever the
    two share a name, which is every entry in :data:`_DEFAULT_SOURCE_ROOTS`.

    Args:
        hint (str): The focus directory, repo-relative or absolute.
        session_tree (str): The tree this session is optimising, or ``""``.

    Returns:
        str: The hint unchanged when absolute or without a tree, else the
            resolved absolute form.
    """
    if not session_tree or Path(hint).is_absolute():
        return hint
    tree = Path(session_tree.rstrip("/"))
    base = tree.parent if (tree / "__init__.py").is_file() else tree
    joined = base / hint.lstrip("/")
    return f"{joined}/" if hint.endswith("/") else str(joined)


def _section_source_hint(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 7 (local source navigation hint) of the prompt.

    Leads with the tree this session optimises: the root list cannot express it,
    since its order records only how roots were discovered.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the source-hint section.
    """
    rows = ["## 7. LOCAL SOURCE NAVIGATION HINT", ""]
    session_tree = (inp.session_framework_tree or "").strip()
    others = tuple(r for r in inp.framework_source_roots if r.rstrip("/") != session_tree.rstrip("/"))
    if not session_tree and not others and not inp.source_hint_directories:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    if session_tree:
        rows.append("The tree this session is optimising — start here:")
        rows.append(_source_root_row(session_tree, worktree_base=inp.worktree_base))
        rows.append("")
    if others:
        rows.append("Other source trees on this host:" if session_tree else "Source trees on this host:")
        for root in others:
            rows.append(_source_root_row(root, worktree_base=inp.worktree_base))
    if inp.source_hint_directories:
        rows.append("")
        rows.append("Where the evidence points — read these first, then widen:")
        for hint in inp.source_hint_directories:
            rows.append(f"- {_resolve_focus_dir(hint, session_tree)}")
    rows.append("")
    rows.append(
        "These are starting points, not a boundary — read anything on the host "
        "that answers the question. Patches still go through ``integrate_patch`` "
        "(Section 9), whichever tree they name."
    )
    rows.append("")
    rows.append(
        "Use ``WebSearch`` to look up the latest upstream version of the local "
        "repo and compare the implementation you intend to modify against what "
        "is there now. Use ``WebFetch`` to read the relevant file or PR "
        "directly — before authoring a patch, confirm whether the upstream "
        "repo already contains the fix or optimization you are about to write."
    )
    return rows


# Section 8 — Output protocol
def _section_output_protocol(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 8 (output protocol) of the specialist prompt."""
    workspace = inp.workspace_path or "<workspace>"
    authors_patches = _authors_patches(inp)
    if inp.domain.key == "candidate_discovery_specialist":
        proposal_example, proposal_contract = UPSTREAM_CANDIDATE_EXAMPLE, UPSTREAM_CANDIDATE_CONTRACT
    else:
        proposal_example, proposal_contract = EXPLORE_VARIANT_EXAMPLE, EXPLORE_VARIANT_CONTRACT

    exit_lines = [
        "**Exit — file write:** write the ``specialist_done`` payload (schema below) to",
        f"``{workspace}/specialist_done.json`` as your **absolute last action**.",
        "The dispatcher polls for that file as the exit signal; stop after writing.",
    ]

    if authors_patches:
        patch_fields = [
            "- ``deliverable`` declares what the round produced and is",
            "  REQUIRED whenever you changed anything: ``{tree_id, targets,",
            "  patches, artifacts, envs, server_args, setup_commands}``.",
            "  ``targets`` lists every file you edited, relative to your",
            "  worktree — the harvest is scoped to it, so a file you changed",
            "  and did not declare is not shipped. Do NOT put hashes in it;",
            "  the harness computes them where your work was validated.",
            "- ``patches_written`` lists paths (relative to your",
            "  workspace or worktree) of any unified-diff patch files you",
            "  authored this round. Empty list = no patches; downstream",
            "  ``integrate_patch`` action skips when empty.",
            "- ``artifacts_written`` lists any non-diff tuned artifacts to install",
            "  (e.g. an autotuned config JSON) as objects ``{source, target, kind,",
            "  description}``: ``source`` is a path inside your worktree, ``target``",
            "  is the install path — PREFER a framework-relative path; an absolute",
            "  path is accepted only if it resolves inside an allowlisted framework",
            "  root. ``integrate_patch`` backs up the target, installs the artifact,",
            "  runs the same E2E gate, and restores the backup on REVERT. A non-diff",
            "  tuned artifact is a FULL result — keep ``proposal_set`` non-empty or",
            "  list the artifact in ``artifacts_written``.",
            "- ``resolves_finding`` (optional string ``rule_id:subject``): set it only",
            f"  when the patch fixes a ``{'`` / ``'.join(sorted(CORRECTNESS_FIX_RULES))}`` entry from section 4c,",
            "  copied verbatim. The gate then verifies the entry is gone from the new log and",
            f"  accuracy passes, and may KEEP a drop of up to {CORRECTNESS_FIX_MAX_DROP_PCT:.0f}% on the graded axis",
            "  (throughput, or p50 interactivity with the other interactivity guards still held)",
            "  within the latency budget.",
        ]
        no_output = "  AND no ``patches_written``/``artifacts_written``; in that case"
    else:
        patch_fields = []
        no_output = "  and no findings; in that case"
    no_proposal_rule = [
        "- An empty ``proposal_set`` is legitimate ONLY when you have no actionable proposals",
        no_output,
        "  and you must put the reason in ``summary``.",
    ]

    return [
        "## 8. OUTPUT PROTOCOL",
        "",
        *exit_lines,
        "",
        "**Messages from the Orchestrator (check this as you work):** read",
        f"``{workspace}/inbox.json`` whenever you finish a step. It is a JSON",
        "list of ``{from, ts, body}`` entries, absent until the Orchestrator",
        "sends one. It is how the Orchestrator answers a question you raised",
        "or redirects you mid-run — if it tells you the mandate changed,",
        "follow it rather than finishing the original plan. Never write to",
        "this file.",
        "",
        "**Incremental checkpoint (do this throughout the run):** every time",
        "you reach a new finding or finish a candidate, rewrite your",
        "best-so-far payload to",
        f"``{workspace}/specialist_done.partial.json`` (write to",
        f"``{workspace}/specialist_done.partial.json.tmp`` first, then rename",
        "over the partial so a reader never sees a half-written file). This",
        "partial uses the **same payload schema** as the final file but does",
        "**NOT** end the run — keep working. There is a wall-clock budget; if",
        "you are stopped before finishing, whatever is in the partial is",
        "preserved as your result, so keep it current. The Orchestrator also",
        "reads each rewrite while you are still running — it is how you report",
        "direction and raise ``residual_questions`` early enough to get an",
        "answer back through ``inbox.json``. Write the final",
        "``specialist_done.json`` (which ends the run) only once, as your",
        "absolute last action.",
        "",
        "Payload schema (identical for both channels):",
        "",
        "```json",
        json.dumps(
            {
                "intent_type": "specialist_done",
                "payload": {
                    "gap_canonical_id": inp.gap_canonical_id or "<echo from dispatch>",
                    "domain": inp.domain.key,
                    "proposal_set": [proposal_example],
                    **({"patches_written": [], "experience_citations": []} if authors_patches else {}),
                    "summary": "≤ 500 char overview of what you tried this round",
                    "confidence": 0.6,
                    "new_findings": [],
                    "residual_questions": [],
                },
            },
            sort_keys=True,
            indent=2,
        ),
        "```",
        "",
        "Field contract:",
        "",
        *proposal_contract,
        (
            "- ``proposal_set``: **2 entries is the norm, 4 the hard cap.** You "
            "are a curator, not a brainstormer: rank by expected gain x "
            "confidence, drop anything contradicting ``pr_evidence``, and stop "
            "at 2. A 3rd or 4th must beat the median "
            "of the first two. Padding is a failure, not thoroughness: each "
            "weak entry costs a Critic reject and a slot on the serial "
            "benchmark queue. One real proposal is a better round than two "
            "padded ones, and an empty ``proposal_set`` with a clear ``summary`` "
            "is better than one."
        ),
        (
            "- The Critic reviews each surviving variant against the KB "
            + "before benchmarking, so a marginal-quality proposal costs you "
            + "a reject (and a pitfall fact that will warn future sessions "
            + "off the same dead-end)."
        ),
        *patch_fields,
        *no_proposal_rule,
        "- ``new_findings`` is a list of learned items. Research scouts must",
        "  emit source-backed ``{what, source, expected_impact, accuracy_risk,",
        "  domain_tags[]}`` records.",
        "- ``residual_questions`` carries to the next specialist round.",
        "",
        "**Heartbeat (Channel B only):** When running in subprocess mode,",
        f"write ``{workspace}/heartbeat.json`` periodically (≤5 min apart)",
        "via Bash so the dispatcher knows you are still alive. Format:",
        '``{"ts": "<iso8601>", "status": "running", "note": "<short>"}``.',
        "Going silent past 5 minutes kills your subprocess.",
        "",
        (f"Hard cap: at most **{inp.max_turns}** LLM turns. Emit specialist_done before the cap."),
    ]


# Section 9 — Iron rules
def _section_iron_rules(inp: SpecialistPromptInputs) -> list[str]:
    """Render Section 9 (iron rules) of the specialist prompt."""
    workspace = inp.workspace_path or "<runs/specialist/<task_id>/>"
    authors_patches = _authors_patches(inp)
    if inp.allocated_gpu_ids:
        cards = ", ".join(str(g) for g in inp.allocated_gpu_ids)
        gpu_rule = [
            f"1. You EXCLUSIVELY own GPU card(s) [{cards}] for this task. On",
            "   those cards do whatever you want: edit code, build, start/stop",
            "   your own servers, profile, autotune, install tuned artifacts,",
            "   and run real benchmark loops. The ONE thing you must NOT do:",
            "   touch the production serving process or its cards — co-residing",
            "   on them would corrupt both your measurement and production.",
            "   Manage only processes YOU started, by their own PID/PGID.",
        ]
    else:
        last = "   try and optionally author patches." if authors_patches else "   try."
        gpu_rule = [
            "1. You have no GPU allocation for this task, so do not run GPU",
            "   benchmarks or start servers. The ONE hard boundary that always",
            "   holds: never touch the production serving process / its cards /",
            "   port 8888. The Coordinator runs benchmarks; you propose what to",
            last,
        ]
    if authors_patches:
        integration_rule = [
            "2. **You MAY** produce changes for integration, but stage them ONLY",
            f"   inside your own worktree at ``{workspace}/``. Two output kinds:",
            "   - Unified-diff patches: ``git diff > patches/NNN_<slug>.patch``",
            "     from the worktree; list paths in ``patches_written``.",
            "   - Tuned non-diff artifacts (e.g. an autotuned config JSON): write",
            "     under the worktree and list in ``artifacts_written`` as",
            "     ``{source, target, kind, description}``.",
            "   **NEVER** ``git apply`` / ``git commit`` against the shared source",
            "   trees in Section 7 directly — ``integrate_patch`` is the single",
            "   integration point, whichever tree the patch names.",
        ]
    else:
        integration_rule = [
            "2. **Read-only dispatch:** you have no worktree and MUST NOT author",
            "   patches or edit the source trees. Report what you found",
            "   through ``specialist_done``; a patch-capable specialist authors any",
            "   source change you recommend.",
        ]
    return [
        "## 9. IRON RULES (Inv-5.1 / Inv-5.3)",
        "",
        *gpu_rule,
        *integration_rule,
        "3. Only ``specialist_done``, ``send_message``, and ``alert`` are",
        "   accepted intents; all others are dropped.",
        "4. You **MUST** finish within ``max_turns`` LLM turns and end with",
        "   exactly one ``specialist_done`` exit signal. Silence past the cap",
        "   synthesizes an empty done.",
        f"5. Use ``{workspace}/`` for ALL writes. It and ``SESSION_DIR`` are the",
        "   only directories the dispatcher hands you to write; the source trees",
        "   are yours to read.",
        "6. On tool error or no useful action left, emit",
        "   ``specialist_done{proposal_set=[], summary='<why>'}``.",
        f"7. {BASH_KILL_SAFETY_PREAMBLE}",
    ]


# Section 1b — Enablement playbook (per-task; enablement specialist only)
def _section_enablement_playbook(inp: SpecialistPromptInputs) -> list[str]:
    """Render the per-task enablement mandate + ladder book into the user prompt.

    Renders the mandate's ``task_description`` (which embeds the ladder book)
    from the verdict the dispatch was decided on, carried verbatim in
    ``enablement_failure_signature``. Kept in the user prompt so the cached
    system prompt stays task-independent.

    The dispatch's own evidence — source lines near the offending site (plus the
    checkpoint weight inventory on a weight-init failure) and the ranked bridging
    refs — is folded in from ``enablement_*`` inputs. Without them the mandate
    renders its generic skeleton, so the agent is told to find a bridge while the
    candidates already discovered for it are withheld.

    Args:
        inp: Assembled prompt inputs for the current dispatch.

    Returns:
        list[str]: The enablement-playbook section lines.
    """
    from hyperloom.common.failure_signature import EnablementRequest, FailureSignature
    from hyperloom.orchestrator.enablement.mandate import build_mandate

    model = str((inp.gap_evidence or {}).get("model") or "").strip()
    req = EnablementRequest(
        framework=(inp.framework or "").strip().lower(),
        model=model or "(target model)",
        repo_url="",
        gpu_type=(inp.gpu_type or "").strip().lower(),
    )
    mandate = build_mandate(
        req,
        FailureSignature.from_dict(inp.enablement_failure_signature),
        candidate_refs=inp.enablement_candidate_refs,
        source_context=inp.enablement_source_context,
    )
    rows = ["## 1b. ENABLEMENT PLAYBOOK", ""]
    rows.extend(mandate.task_description.splitlines())
    acc_cfg = inp.enablement_accepted_config or {}
    acc_envs = {str(k): str(v) for k, v in (acc_cfg.get("extra_envs") or {}).items()}
    acc_args = str(acc_cfg.get("extra_server_args") or "").strip()
    if acc_envs or acc_args:
        rows.append("")
        rows.append("### Config already in effect")
        rows.append(
            "Prior rounds accepted these and your bench launches with them. Re-propose "
            "one only to override or remove it."
        )
        rows.extend(f"- `export {k}={v}`" for k, v in acc_envs.items())
        if acc_args:
            rows.append(f"- server args: `{acc_args}`")
    return rows


# Section 1a — PD-disaggregation
def _section_pd_disaggregation(inp: SpecialistPromptInputs) -> list[str]:
    """§1a — PD-disaggregation context (omitted unless pd_mode==disaggregated).

    Surfaces the prefill/decode split so the specialist targets each role's
    distinct bottleneck (prefill: compute / TTFT; decode: memory-bandwidth /
    TPOT) and the KV-transfer path, instead of treating the server as one pool.
    Reads the multi-node state directly; returns ``[]`` on the single-node /
    aggregated paths so the section is dropped.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs (unused; the
            PD topology is read from multi-node state).

    Returns:
        list[str]: The PD-disaggregation section lines, or ``[]`` when not
        disaggregated.
    """
    from hyperloom.orchestrator.actions.executors._multi_node_env import (
        pd_topology_from_state,
    )

    pd = pd_topology_from_state()
    if not pd:
        return []
    tb = pd.get("transfer_backend") or "the KV transfer backend"
    return [
        "## 1a. PD-DISAGGREGATION (prefill/decode separated)",
        "",
        "This deployment runs **prefill/decode disaggregation**: prefill and "
        "decode execute on SEPARATE GPU nodes, exchanging KV cache via "
        f"`{tb}`. Optimize each role for its OWN bottleneck — do NOT treat the "
        "server as a single pool:",
        "",
        f"- **Prefill** ({pd.get('prefill_nodes')} node(s), tp={pd.get('prefill_tp')}, "
        f"ep={pd.get('prefill_ep')}): compute-bound; drives **TTFT**. Levers: "
        "chunked-prefill size, attention backend for long ISL, MoE dispatch, "
        "prefill token/batch budget.",
        f"- **Decode** ({pd.get('decode_nodes')} node(s), tp={pd.get('decode_tp')}, "
        f"ep={pd.get('decode_ep')}): memory-bandwidth-bound; drives **TPOT/ITL**. "
        "Levers: KV-cache / mem-fraction, max-running-requests, dp-attention, "
        "decode MoE a2a backend.",
        f"- **KV transfer** (`{tb}`): watch bootstrap / transfer stalls; RDMA/IB "
        "device selection affects decode start latency.",
        "- **Balance**: tune the prefill:decode node/TP ratio to the workload shape "
        + "— a saturated role caps end-to-end throughput.",
        "",
        "Per-role GPU telemetry is in the benchmark report's "
        "`gpu_monitor_by_role` (prefill vs decode util / power / VRAM); use it to "
        "confirm which role is the bottleneck BEFORE proposing changes.",
    ]


# Top-level assembler
def build_specialist_prompts(inp: SpecialistPromptInputs) -> tuple[str, str]:
    """Assemble the full specialist prompt from its section builders.

    The system prompt carries the immutable contract (identity, output
    protocol, iron rules) and the user prompt carries the per-task context
    (hardware, optional PD-disaggregation, execution budget, gap, KB,
    roofline, recipe, lessons, pitfalls, PR feed, source hint and
    orchestration notes). The split lets the LLM backend cache the system
    prompt across specialists.

    Args:
        inp (SpecialistPromptInputs): The assembled prompt inputs.

    Returns:
        tuple[str, str]: The ``(system_prompt, user_prompt)`` pair.
    """

    system_sections = [
        _section_identity(inp),
        _section_output_protocol(inp),
        _section_iron_rules(inp),
    ]
    if inp.domain.key == "enablement_specialist":
        # Pre-baseline enablement: the perf context (roofline / recipe / lessons /
        # pitfalls / KB subgraph) is noise when the server cannot boot or the
        # baseline fails its accuracy eval. Carry only the failure, the tiered
        # playbook, and the tools to discover + navigate a fix.
        user_sections = [
            _section_mandate(inp),
            _section_hardware(inp),
            _section_pd_disaggregation(inp),  # § 1a (omitted unless disaggregated)
            _section_execution_budget(inp),  # omitted when no budget
            _section_gap(inp),
            _section_enablement_playbook(inp),  # § 1b (mandate + ladder book)
            _section_pr_feed(inp),
            _section_source_hint(inp),
        ]
    elif inp.domain.key == "custom_specialist":
        # The operator focus is the mandate; perf context is left to the operator text.
        # Experience KB stays: the warmup records its block as shown to every FRAMEWORK_AGENT specialist.
        user_sections = [
            _section_mandate(inp),
            _section_hardware(inp),
            _section_pd_disaggregation(inp),
            _section_execution_budget(inp),
            _section_gap(inp),
            _section_experience_kb(inp),
        ]
        if _authors_patches(inp):
            user_sections.append(_section_source_hint(inp))
    else:
        user_sections = [
            _section_mandate(inp),
            _section_hardware(inp),
            _section_pd_disaggregation(inp),  # § 1a (PD-disaggregation; omitted unless disaggregated)
            _section_execution_budget(inp),  # omitted when no budget
            _section_gap(inp),
            _section_kb_context(inp),
            _section_roofline_evidence(inp),
            _section_experience_kb(inp),
            _section_runtime_findings(inp),
            _section_recipe(inp),
            _section_lessons(inp),
            _section_pitfalls(inp),
            _section_pr_feed(inp),
            _section_source_hint(inp),
        ]
    if inp.notes:
        user_sections.append(
            [
                "## 10. NOTES FROM ORCHESTRATION",
                "",
                defang_prompt_structure(inp.notes),
            ]
        )

    def _flatten(sections: list[list[str]]) -> str:
        """Join section line-lists into a single newline-separated string.

        Inserts one blank line between non-empty sections and appends a
        trailing newline.

        Args:
            sections (list[list[str]]): The per-section lists of lines.

        Returns:
            str: The flattened prompt text.
        """
        out: list[str] = []
        for sec in sections:
            if not sec:  # skip omitted sections (e.g. no execution budget)
                continue
            if out:
                out.append("")
            out.extend(sec)
        return "\n".join(out) + "\n"

    return _flatten(system_sections), _flatten(user_sections)


__all__ = [
    "build_specialist_prompts",
]
