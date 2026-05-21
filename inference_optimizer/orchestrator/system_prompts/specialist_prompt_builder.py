"""Specialist sub-agent prompt assembler — v0.8 M5 (KB_design §3.5 §6).

The Coordinator hands the SpecialistRunner a typed input bundle and
this module returns the fully-assembled 9-section prompt. The 9 sections
are fixed; each is independently nullable (renders as ``(none)``
placeholder) so the prompt structure stays stable regardless of which
KB / PR / source-tree slots happen to be populated for the current
specialist + gap.

Output is a tuple ``(system_prompt, user_prompt)`` where:

* ``system_prompt`` carries sections 1 (identity), 8 (output protocol),
  9 (iron rules) — the immutable specialist contract.
* ``user_prompt`` carries sections 2 (hardware), 3 (gap), 4 (KB), 5
  (recipe), 6 (PR), 7 (source hint) — the per-task context.

This split lets the LLM backend cache the system prompt across multiple
specialists in the same session (identity / iron rules don't change).

Pure function: no IO besides reading the assembled inputs, no env
access, no logging side effects. The output is snapshotted to
``runs/specialist/<task_id>/prompt.md`` by SpecialistRunner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ..specialist_domains import (
    DEFAULT_SPECIALIST_MAX_TURNS,
    SpecialistDomain,
    get_domain,
)


_NONE_PLACEHOLDER = "(none)"


# ---------------------------------------------------------------------------
# PR-A6 (Arbor-into-Hyperloom) — per-domain focus templates
#
# Each entry produces the body that the prompt builder injects into
# Section 1 under "### Domain focus — <key>". The shape mirrors
# Arbor's ``agent expertise`` table (launcher/orchestrator.md):
# - "What you read" (which sub-trees of the KB / which framework
#   directories to grep first).
# - "Winning techniques" (concrete patterns the specialist should
#   sanity-check against the gap before proposing).
# - "Pitfalls" (anti-patterns that historically reverted on this
#   domain — sourced from KB_design lessons + Arbor's lessons table).
#
# When a domain key is missing from this map, ``_section_identity``
# falls back to the generic body (the v0.8 M5 default).
# ---------------------------------------------------------------------------


def _focus_framework_specialist(inp: SpecialistPromptInputs) -> list[str]:
    return [
        "You target **vLLM / SGLang scheduler / cuda_graph / kv_cache** code.",
        "",
        "**What to read first**",
        "- `vllm/v1/engine/` and `vllm/v1/worker/` (scheduler, model_runner).",
        "- `sglang/python/sglang/srt/scheduler/` and `sglang/python/sglang/srt/managers/`.",
        "- KB anchor `framework.*` (cuda_graph / batching / chunked_prefill / kv_cache).",
        "",
        "**Winning techniques to consider**",
        "- `--enable-chunked-prefill` + matched `--max-num-batched-tokens`.",
        "- `--enforce-eager=false` + cuda graph capture for stable batch sizes.",
        "- `--kv-cache-dtype fp8_e4m3` when the gap is HBM-bound (gate accuracy!).",
        "- `--max-num-seqs` tuning at concurrency boundaries.",
        "",
        "**Pitfalls (historical REVERTs)**",
        "- Raising `--max-num-seqs` past 512 on MI300X → OOM on 671B MoE models.",
        "- `cuda_graph` + dynamic batch sizes → silent recapture cost > savings.",
        "- Chunked prefill without `--max-num-batched-tokens` → tail latency",
        "  regressions invisible to throughput-only benches.",
    ]


def _focus_kernel_specialist(inp: SpecialistPromptInputs) -> list[str]:
    return [
        "You target **aiter / SGLang kernels / triton** code (attention,",
        "MoE, GEMM, fused-attention paths).",
        "",
        "**What to read first**",
        "- `aiter/csrc/` and `aiter/aiter/ops/` (CK / hipBLASLt wrappers).",
        "- `sglang/python/sglang/srt/layers/attention/` (backend selection).",
        "- KB anchor `kernel.*` (CDNA3 tiling / MoE / attention / GEMM).",
        "",
        "**Winning techniques to consider**",
        "- Switch attention backend (`ROCM_AITER_MLA` ↔ `TRITON_MLA` ↔",
        "  `ROCM_AITER_TRITON_MLA`) at the workload's prefill/decode mix.",
        "- `VLLM_ROCM_USE_AITER=1` umbrella + per-op overrides for MoE / RMSNorm.",
        "- Tile-size tuning for `M < 256` GEMMs (hipBLASLt vs Triton).",
        "- Fused-attention enable flags for prefill chunks.",
        "",
        "**Pitfalls**",
        "- Forcing AITER MLA on workloads with short OSL — kernel selection",
        "  cost dominates the saving.",
        "- Mixing `--attention-backend` with `--enforce-eager=true` invalidates",
        "  cuda graphs silently.",
        "- Trying triton fp4 paths on CDNA3 without `AMDGCN_USE_BUFFER_OPS=1`.",
    ]


def _focus_comm_specialist(inp: SpecialistPromptInputs) -> list[str]:
    return [
        "You target **RCCL / NCCL / QuickReduce / AllReduce** code and tuning.",
        "",
        "**What to read first**",
        "- `vllm/distributed/`, `vllm/distributed/parallel_state.py`.",
        "- `aiter/csrc/quick_reduce/` and RCCL plugin paths.",
        "- KB anchor `communication.*` (allreduce / QuickReduce / topology).",
        "",
        "**Winning techniques to consider**",
        "- `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` when message size > 1MiB.",
        "- `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` tuning per topology.",
        "- TP allreduce vs PP collective trade-offs at high concurrency.",
        "",
        "**Pitfalls**",
        "- INT4 QuickReduce at TP=2 — overhead dominates the bandwidth savings.",
        "- Tuning NCCL env vars without confirming `rocm-smi --showtopo` shows",
        "  the expected XGMI / PCIe topology.",
    ]


def _focus_compiler_specialist(inp: SpecialistPromptInputs) -> list[str]:
    return [
        "You target **torch.compile / inductor / triton / AMDGCN** codegen",
        "and register-pressure tuning.",
        "",
        "**What to read first**",
        "- `triton/python/triton/runtime/` and `triton/lib/Conversion/`.",
        "- `torch/_inductor/codegen/triton.py` and `torch/_inductor/scheduler.py`.",
        "- KB anchor `compiler.*` (inductor / triton / AMDGCN).",
        "",
        "**Winning techniques to consider**",
        "- `--compilation-config '{\"level\": 3, ...}'` with surgical level=2",
        "  fallback for kernels that don't quantise cleanly.",
        "- `torch._inductor.config.triton.unique_kernel_names` + per-kernel",
        "  autotune cache pinning.",
        "- VGPR-budget tuning via `num_warps` / `num_stages` in @triton.autotune.",
        "",
        "**Pitfalls**",
        "- Raising level=3 globally — some kernels recompile on every batch",
        "  size, wiping the gain.",
        "- VGPR > 256 spills to scratch on CDNA3; profile occupancy first.",
    ]


def _focus_system_specialist(inp: SpecialistPromptInputs) -> list[str]:
    return [
        "You target **KFD driver / ROCm runtime / memory / dispatch overhead**.",
        "",
        "**What to read first**",
        "- `/sys/class/kfd/kfd/` (read-only probes via Bash).",
        "- `rocm-smi --showmeminfo VRAM` / `rocm-smi --showtopo`.",
        "- KB anchor `systems.*` (KFD / dispatch / memory).",
        "",
        "**Winning techniques to consider**",
        "- `HSA_ENABLE_SDMA=0` when host↔device dispatch dominates.",
        "- `HIP_HIDDEN_FREE_MEM` to expose hidden VRAM headroom for large MoE.",
        "- `numactl --cpunodebind` pinning at high concurrency.",
        "",
        "**Pitfalls**",
        "- `HSA_ENABLE_SDMA=0` on small-message decode workloads → latency up.",
        "- Disabling `--gpu-memory-utilization` headroom past 0.95 → OOM on",
        "  prefill chunks for long-context workloads.",
    ]


def _focus_pr_intel_specialist(inp: SpecialistPromptInputs) -> list[str]:
    return [
        "You are a **cross-repo PR researcher**. Your role is NOT to propose",
        "configuration knobs — it is to surface PRs / commits / issues from",
        "(ROCm/aiter, sgl-project/sglang, ROCm/vllm, triton-lang/triton,",
        "ROCm/rccl) that other specialists should follow up on.",
        "",
        "**What to do**",
        "- Use ``mcp__pr_monitor__*`` + ``WebSearch`` to find recent PRs",
        "  related to the gap.",
        "- For each PR, extract: (repo, number, title, summary, files",
        "  touched, NVIDIA equivalent if any).",
        "- Surface as ``proposal_set`` entries where ``provenance`` = research",
        "  and ``pr_evidence`` is non-empty. Do NOT propose source patches",
        "  yourself — that's the kernel / framework specialist's job once",
        "  they read your PR list.",
        "",
        "**Pitfalls**",
        "- Citing a PR without verifying its target framework matches the",
        "  current install.",
        "- Spending more than one round; PR intel is best dispatched once",
        "  per gap and used as input to other specialists.",
    ]


_DOMAIN_FOCUS_TEMPLATES: dict[
    str, "Callable[[SpecialistPromptInputs], list[str]]"
] = {
    "framework_specialist": _focus_framework_specialist,
    "kernel_specialist":    _focus_kernel_specialist,
    "comm_specialist":      _focus_comm_specialist,
    "compiler_specialist":  _focus_compiler_specialist,
    "system_specialist":    _focus_system_specialist,
    "pr_intel_specialist":  _focus_pr_intel_specialist,
}


@dataclass(frozen=True)
class SpecialistPromptInputs:
    """Typed inputs the Coordinator hands to the prompt builder."""

    # Identity
    task_id: str
    domain: SpecialistDomain
    max_turns: int = DEFAULT_SPECIALIST_MAX_TURNS

    # Hardware context (§3.5 §6 part 2)
    gpu_type: str = ""
    tp: int = 1
    hbm_gb: float = 0.0
    peak_tflops: float = 0.0
    arch_notes: str = ""

    # Gap statement (§3.5 §6 part 3)
    gap_canonical_id: str = ""
    gap_symptom: str = ""
    gap_layer: str = ""
    gap_evidence: dict[str, Any] = field(default_factory=dict)

    # Cortex KB sub-graph (§3.5 §6 part 4)
    kb_subgraph: dict[str, Any] = field(default_factory=dict)

    # Recipe summary from T0 ``find-recipe`` (§3.5 §6 part 5)
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)

    # PR feed (§3.5 §6 part 6)
    pr_feed: list[dict[str, Any]] = field(default_factory=list)
    pr_monitor_available: bool = True

    # Local source navigation hint (§3.5 §6 part 7)
    framework_source_roots: tuple[str, ...] = ()
    source_hint_directories: tuple[str, ...] = ()

    # Workspace path (for transcript / heartbeat instructions)
    workspace_path: str = ""

    # Free-form notes from Orchestration (e.g. previous-round resid_qs)
    notes: str = ""


# ---------------------------------------------------------------------------
# Section 1 — Identity & autonomy
# ---------------------------------------------------------------------------
def _section_identity(inp: SpecialistPromptInputs) -> list[str]:
    body: list[str] = [
        "## 1. IDENTITY & AUTONOMY",
        "",
        f"You are a **{inp.domain.key}** specialist sub-agent (KB_design §3.5).",
        f"Layer: {inp.domain.layer}.",
        f"KB anchor: {inp.domain.kb_anchor}.",
        "",
        f"Description: {inp.domain.description or '(generic)'}",
        "",
        "Your single mission is to **produce proposals + evidence**, and",
        "optionally author source patches into your isolated worktree.",
        "The Hyperloom Coordinator owns all KEEP/REVERT decisions, server",
        "benchmarks, accuracy gates, and KB writes; you participate by",
        "emitting ONE final ``specialist_done`` (intent OR done-file per",
        "Section 8) carrying your proposal_set and ``patches_written``.",
    ]
    # PR-A6 (Arbor-into-Hyperloom): per-domain expertise + focus
    # paragraph. Each domain template emphasises the surface area the
    # specialist should reason about + the typical winning techniques
    # (lifted from Arbor's orchestrator.md "agent expertise" table).
    focus = _DOMAIN_FOCUS_TEMPLATES.get(inp.domain.key)
    if focus is not None:
        body.append("")
        body.append(f"### Domain focus — {inp.domain.key}")
        body.append("")
        body.extend(focus(inp))
    return body


# ---------------------------------------------------------------------------
# Section 2 — Hardware context
# ---------------------------------------------------------------------------
def _section_hardware(inp: SpecialistPromptInputs) -> list[str]:
    rows: list[str] = ["## 2. HARDWARE CONTEXT", ""]
    if inp.gpu_type:
        rows.append(f"- gpu_type: {inp.gpu_type}")
    else:
        rows.append(f"- gpu_type: {_NONE_PLACEHOLDER}")
    rows.append(f"- TP: {inp.tp}")
    if inp.hbm_gb > 0:
        rows.append(f"- HBM per GPU: {inp.hbm_gb:.1f} GB")
    if inp.peak_tflops > 0:
        rows.append(f"- Peak TFLOPs (declared): {inp.peak_tflops:.1f}")
    if inp.arch_notes:
        rows.append("")
        rows.append(f"Architecture notes: {inp.arch_notes}")
    return rows


# ---------------------------------------------------------------------------
# Section 3 — Gap statement
# ---------------------------------------------------------------------------
def _section_gap(inp: SpecialistPromptInputs) -> list[str]:
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


# ---------------------------------------------------------------------------
# Section 4 — Cortex KB sub-graph
# ---------------------------------------------------------------------------
def _section_kb_subgraph(inp: SpecialistPromptInputs) -> list[str]:
    rows = ["## 4. CORTEX KB SUB-GRAPH", ""]
    if not inp.kb_subgraph:
        rows.extend([
            _NONE_PLACEHOLDER,
            "",
            "(No KB sub-graph supplied; this is the M5 default when "
            "Cortex T1 traverse hasn't been wired up for the specialist's "
            "anchor yet. Use mcp__cortex_kb__traverse tool calls to "
            "navigate the KB if needed.)",
        ])
        return rows
    rows.append("```json")
    rows.append(json.dumps(inp.kb_subgraph, sort_keys=True, indent=2))
    rows.append("```")
    return rows


# ---------------------------------------------------------------------------
# Section 5 — Recipe summary
# ---------------------------------------------------------------------------
def _section_recipe(inp: SpecialistPromptInputs) -> list[str]:
    rows = ["## 5. WARM-START RECIPE SUMMARY", ""]
    has_recipe = bool(inp.warm_start_recipe)
    has_pitfalls = bool(inp.warm_start_pitfalls)
    if not has_recipe and not has_pitfalls:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    if has_recipe:
        rows.append("**find-recipe result:**")
        rows.append("```json")
        rows.append(json.dumps(inp.warm_start_recipe, sort_keys=True, indent=2))
        rows.append("```")
    if has_pitfalls:
        rows.append("")
        rows.append("**Known pitfalls (do NOT repeat):**")
        for p in inp.warm_start_pitfalls:
            rows.append(f"- {json.dumps(p, sort_keys=True)}")
    return rows


# ---------------------------------------------------------------------------
# Section 6 — PR feed
# ---------------------------------------------------------------------------
def _section_pr_feed(inp: SpecialistPromptInputs) -> list[str]:
    rows = ["## 6. PR FEED", ""]
    if not inp.pr_monitor_available:
        rows.append("(empty: pr_monitor unavailable)")
        return rows
    if not inp.pr_feed:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    for pr in inp.pr_feed:
        title = str(pr.get("title") or "").strip()
        url = str(pr.get("url") or "").strip()
        labels = pr.get("labels") or []
        labels_text = (
            " " + " ".join(f"[{l}]" for l in labels)
            if isinstance(labels, list) and labels else ""
        )
        rows.append(f"- {title} — <{url}>{labels_text}")
    return rows


# ---------------------------------------------------------------------------
# Section 7 — Local source navigation hint
# ---------------------------------------------------------------------------
def _section_source_hint(inp: SpecialistPromptInputs) -> list[str]:
    rows = ["## 7. LOCAL SOURCE NAVIGATION HINT", ""]
    if not inp.framework_source_roots and not inp.source_hint_directories:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    if inp.framework_source_roots:
        rows.append("Framework source roots (read-only):")
        for p in inp.framework_source_roots:
            rows.append(f"- {p}")
    if inp.source_hint_directories:
        rows.append("")
        rows.append("Focus directories for this domain:")
        for p in inp.source_hint_directories:
            rows.append(f"- {p}")
    rows.append("")
    rows.append(
        "These trees are read-only. Use Read / Grep / Glob to navigate. "
        "Do NOT attempt Edit / Write / git apply (PolicyGate R4)."
    )
    return rows


# ---------------------------------------------------------------------------
# Section 8 — Output protocol
# ---------------------------------------------------------------------------
def _section_output_protocol(inp: SpecialistPromptInputs) -> list[str]:
    workspace = inp.workspace_path or "<workspace>"
    return [
        "## 8. OUTPUT PROTOCOL",
        "",
        "Your run terminates by producing **exactly one** specialist_done",
        "record. The Hyperloom runner accepts either of two equivalent",
        "exit channels — use whichever your tool surface supports:",
        "",
        "**Channel A — ``emit_intent`` tool (in-process / SDK runtime):**",
        "Call the ``emit_intent`` tool exactly once with an intent of type",
        "``specialist_done`` and the payload schema below.",
        "",
        "**Channel B — file write (subprocess / production runtime,",
        "PR-A2 Arbor-into-Hyperloom):** When ``emit_intent`` is not in",
        "your tool list, write the same payload to",
        f"``{workspace}/specialist_done.json`` via the ``Write`` tool as",
        "your **absolute last action**. The Hyperloom dispatcher polls",
        "for that file and treats its appearance as the run's exit",
        "signal. After writing it, stop — do not call any further tools.",
        "",
        "Payload schema (identical for both channels):",
        "",
        "```json",
        json.dumps({
            "intent_type": "specialist_done",
            "payload": {
                "gap_canonical_id": inp.gap_canonical_id or "<echo from dispatch>",
                "domain": inp.domain.key,
                "proposal_set": [
                    {
                        "name": "<unique-in-round>",
                        "extra_args": "--example-flag value",
                        "extra_envs": {"EXAMPLE_ENV": "1"},
                        "reason": "why this might help the gap",
                        "kb_evidence": [],
                        "pr_evidence": [],
                        "source_evidence": []
                    }
                ],
                "patches_written": [],
                "empty": False,
                "summary": "≤ 500 char overview of what you tried this round",
                "confidence": 0.6,
                "new_findings": [],
                "residual_questions": []
            },
        }, sort_keys=True, indent=2),
        "```",
        "",
        "Field contract (KB_design §3.5 §7 + PR-A2 Arbor extensions):",
        "",
        "- ``proposal_set`` items reuse the §3.4 explore variant schema.",
        "- ``patches_written`` (PR-A2) lists paths (relative to your",
        "  workspace or worktree) of any unified-diff patch files you",
        "  authored this round. Empty list = no patches; downstream",
        "  ``integrate_patch`` action skips when empty.",
        "- ``empty=true`` is legitimate when you have no actionable proposals;",
        "  in that case ``proposal_set=[]`` and you must put the reason in",
        "  ``summary``.",
        "- ``new_findings`` becomes HYPOTHESIZED KB edges at T4 commit even",
        "  when not KEEP'd this round — surface anything you learned.",
        "- ``residual_questions`` carries to the next specialist round.",
        "",
        "**Heartbeat (Channel B only):** When running in subprocess mode,",
        f"write ``{workspace}/heartbeat.json`` periodically (≤5 min apart)",
        "via Bash so the dispatcher knows you are still alive. Format:",
        "``{\"ts\": \"<iso8601>\", \"status\": \"running\", \"note\": \"<short>\"}``.",
        "Going silent past 5 minutes kills your subprocess.",
        "",
        (
            f"Hard cap: at most **{inp.max_turns}** LLM turns. Silence past "
            "the cap = stale (robustness will synthesize an empty done)."
        ),
    ]


# ---------------------------------------------------------------------------
# Section 9 — Iron rules
# ---------------------------------------------------------------------------
def _section_iron_rules(inp: SpecialistPromptInputs) -> list[str]:
    workspace = inp.workspace_path or "<runs/specialist/<task_id>/>"
    return [
        "## 9. IRON RULES (Inv-5.1 / Inv-5.2 / Inv-5.3)",
        "",
        "1. **NEVER** touch the serving GPU (no Magpie / no benchmark / no",
        "   server restart / no vllm or sglang process control). The",
        "   Coordinator runs benchmarks; you only propose what to try and",
        "   optionally author patches.",
        "2. **You MAY** write source patches, but ONLY into your own",
        f"   worktree at ``{workspace}/`` (a git checkout branched off",
        "   the framework HEAD just for this task). Concretely:",
        "   - Edit files inside the worktree.",
        "   - ``git diff > patches/NNN_<slug>.patch`` from inside the",
        "     worktree to produce a unified-diff patch file.",
        "   - List patch paths in ``patches_written`` in your",
        "     ``specialist_done`` payload (relative to the worktree).",
        "   You **MUST NEVER** ``git apply``, ``git commit``, restart a",
        "   server, or otherwise mutate the main ``framework_source_roots``",
        "   directly — the orchestrator's ``integrate_patch`` action is",
        "   the single integration point that applies your patches with",
        "   the throughput + accuracy gate. (PR-A2, Arbor-into-Hyperloom:",
        "   Inv-5.1 updated.)",
        "3. **NEVER** call ``cortex-kb`` write endpoints (propose-point /",
        "   propose-edge / hypothesize / ingest-attempt / verify / commit)",
        "   directly. The Coordinator owns KB writes (PolicyGate R4); your",
        "   only KB read paths are ``mcp__cortex_kb__traverse`` /",
        "   ``find_recipe`` / ``query``.",
        "4. **NEVER** emit any intent other than ``specialist_done``,",
        "   ``send_message`` (heartbeat), or ``alert``. Any other intent",
        "   type triggers PolicyGate R3 ``specialist_done_source``.",
        "5. You **MUST** finish within ``max_turns`` LLM turns and end with",
        "   a single ``specialist_done`` exit signal (intent OR",
        "   ``specialist_done.json`` file write per Section 8). Sub-agent",
        "   silence past the cap is treated as stale (an empty",
        "   ``specialist_done`` is synthesized for you so the EXPLORE",
        "   round still progresses).",
        f"6. Use ``{workspace}/`` for ALL writes (patches, transcript notes,",
        "   heartbeat). Do not write anywhere else in the filesystem; the",
        "   dispatcher only exposes this directory + read-only access to",
        "   ``framework_source_roots`` and SESSION_DIR.",
        "7. If you hit a tool error or run out of useful actions, emit",
        "   ``specialist_done{empty=true, summary='<why>'}`` rather than",
        "   stalling.",
    ]


# ---------------------------------------------------------------------------
# Top-level assembler
# ---------------------------------------------------------------------------
def build_specialist_prompts(inp: SpecialistPromptInputs) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for one specialist task."""

    system_sections = [
        _section_identity(inp),
        _section_output_protocol(inp),
        _section_iron_rules(inp),
    ]
    user_sections = [
        _section_hardware(inp),
        _section_gap(inp),
        _section_kb_subgraph(inp),
        _section_recipe(inp),
        _section_pr_feed(inp),
        _section_source_hint(inp),
    ]
    if inp.notes:
        user_sections.append([
            "## 10. NOTES FROM ORCHESTRATION",
            "",
            inp.notes,
        ])

    def _flatten(sections: list[list[str]]) -> str:
        out: list[str] = []
        for sec in sections:
            if out:
                out.append("")
            out.extend(sec)
        return "\n".join(out) + "\n"

    return _flatten(system_sections), _flatten(user_sections)


def build_specialist_prompts_for_domain(
    *,
    task_id: str,
    domain_key: str,
    **kwargs: Any,
) -> tuple[str, str]:
    """Helper that resolves ``domain_key`` to a SpecialistDomain first."""
    domain = get_domain(domain_key)
    if domain is None:
        raise ValueError(
            f"unknown specialist domain={domain_key!r}; see specialist_domains"
        )
    inp = SpecialistPromptInputs(task_id=task_id, domain=domain, **kwargs)
    return build_specialist_prompts(inp)


__all__ = [
    "SpecialistPromptInputs",
    "build_specialist_prompts",
    "build_specialist_prompts_for_domain",
]
