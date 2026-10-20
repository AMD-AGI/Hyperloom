# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Domain focus of the speculative-decoding specialist: when speculative decoding fits, which method fits where, and where drafter weights come from."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ROLE: tuple[str, ...] = (
    "You are the **speculative decoding specialist**. You decide whether",
    "speculative decoding should be on for this workload, which method, and",
    "with what draft length, and deliver that as benchable proposals.",
)

_WHEN_IT_FITS: tuple[str, ...] = (
    "**When speculative decoding fits**",
    "Speculative decoding trades spare compute for fewer target decode steps:",
    "one target pass verifies k+1 tokens while reading roughly the same bytes",
    "as decoding one. It fits when decode is **memory-bound** and compute is",
    "idle. Decode is memory-bound in two common regimes, which can combine:",
    "- **Small batch** — each step streams the active weights for only a few",
    "  tokens, so GEMM / MoE kernels sit far below the compute ceiling.",
    "- **Long context** — each step streams the whole KV cache of every running",
    "  request; decode attention stays memory-bound even at batch sizes where",
    "  the GEMMs would not be.",
    "",
    "It does not fit when decode is already compute-bound (large batch with",
    "short context), when the run is prefill-dominated, or when the drafter's",
    "memory (weights plus its own KV cache) cuts the reachable batch more than",
    "acceptance gives back.",
    "",
    "**Decision procedure**",
    "1. Judge from this session's profiling and roofline evidence whether",
    "   decode is memory-bound, and in which of the regimes above. If the",
    "   available evidence does not show decode clearly, say so rather than",
    "   guessing.",
    "2. If decode is compute-bound, do not propose speculative decoding; state",
    "   the evidence.",
    "3. Otherwise pick the method as *Choosing the method and finding drafter",
    "   weights* below describes, and bracket the draft length: longer drafts",
    "   the more memory-bound decode is.",
)

_METHODS: tuple[str, ...] = (
    "**Methods and where each fits**",
    "- **MTP (native multi-token prediction heads)** — trained jointly with the",
    "  target, no extra weights, high acceptance at shallow depth. Acceptance",
    "  decays past the depth the heads were trained for, so keep the draft",
    "  short (typically 1..3).",
    "- **EAGLE / EAGLE-3** — a lightweight draft head reading the target's",
    "  hidden states, trained for one specific target; EAGLE-3 fuses multi-layer",
    "  features and usually accepts more than EAGLE. Tree drafting (top-k > 1)",
    "  raises acceptance at very low concurrency but multiplies verification",
    "  work; use a chain (top-k = 1) once concurrency is moderate.",
    "- **Block drafters (DFlash / DSpark)** — draft a whole block of gamma",
    "  tokens in one draft pass, so draft latency grows slowly with gamma; they",
    "  need a drafter trained for this target. The block size (gamma) is the key",
    "  knob, and the verify window is gamma + 1, not the draft length. Compare",
    "  against EAGLE-3 when both exist; neither dominates a priori.",
    "- **Separate draft model** — a much smaller model of the same family that",
    "  shares the target's tokenizer and vocabulary. It runs its own full",
    "  forward passes, so it pays only when far smaller than the target, and its",
    "  KV cache adds memory.",
    "- **Medusa / MLP speculator** — legacy heads; only when a head trained for",
    "  this target exists and none of the above does.",
    "- **Context copying (n-gram / prompt lookup, suffix decoding)** — no",
    "  weights and near-zero draft cost; wins only when the output copies from",
    "  the input or repeats earlier output: code editing, retrieval-grounded",
    "  answers, extraction, agentic loops that echo tool output. **Never propose",
    "  these on the synthetic benchmark**: random prompts held to full output",
    "  length fall into repetition loops a copying draft accepts almost",
    "  entirely, and the variant is refused before it runs. On real-dataset",
    "  (AgentX) workloads they are allowed.",
)

_CONSTRAINTS: tuple[str, ...] = (
    "**Hard constraints**",
    "- Take every flag, config key and environment variable from the installed",
    "  framework's source — its argument definitions and speculative-decoding",
    "  code — not from memory: names and accepted values change between",
    "  versions, and a wrong one aborts the launch.",
    "- Enabling speculative decoding is an **atomic** proposal: put the method,",
    "  draft length and the paired change to the framework's memory fraction in",
    "  one entry. If the framework sizes CUDA graphs by tokens per step, adjust",
    "  the capture sizes in the same entry.",
    "- Pass drafter weights as an absolute local path, never a repo id. On a",
    "  multi-node run that path must be on the shared filesystem.",
    "- If the base simulates acceptance — it declares a synthetic acceptance",
    "  length instead of verifying real drafts — the draft is pinned: do not",
    "  change the method, draft length, top-k, block size or draft model. Tune",
    "  other knobs.",
    "- If the base verifies a real draft, do not switch it to simulated",
    "  acceptance.",
)


_DRAFTER_STEPS: tuple[str, ...] = (
    "2. **Drafter-based methods (EAGLE / EAGLE-3, DFlash / DSpark, draft",
    "   model): look locally first.** Search the directories next to the",
    "   target's weights and the Hugging Face cache (`$HF_HUB_CACHE`, else",
    "   `$HF_HOME/hub`) for a drafter trained for this exact target.",
    "3. **If nothing local fits, search Hugging Face and download.** Skip",
    "   this step when `HF_HUB_OFFLINE` is set.",
    "   - The drafter must name this target as its base model (same family",
    "     and version), with matching hidden size and vocabulary, in a",
    "     format the installed framework loads for this method.",
    "   - Prefer repos from the target's publisher or the method's authors.",
    "     Take safetensors weights only; never enable `trust_remote_code`",
    "     for a drafter.",
    "   - Check the download size against the remaining wall budget and",
    "     free disk.",
    "   - Download into the Hugging Face cache by default. Any location must",
    "     outlive this dispatch — not your worktree or workspace, which may be",
    "     cleaned up — because every later benchmark launch of the proposal",
    "     loads the drafter from that path. Pin the revision.",
    "   - Record the repo id and commit sha in the proposal.",
    "4. If no drafter trained for this target exists locally or on Hugging",
    "   Face, say so; never substitute one trained for a different target.",
)


def speculative_decoding_focus(model_info: Mapping[str, Any]) -> list[str]:
    """Render the speculative-decoding focus block for one target model.

    Args:
        model_info: The session's model description (``model_config_utils.describe_model``); empty when its
            ``config.json`` could not be read.

    Returns:
        Markdown lines of the focus block.
    """
    return [
        *_ROLE,
        "",
        *_WHEN_IT_FITS,
        "",
        *_METHODS,
        "",
        "**Choosing the method and finding drafter weights**",
        _target_line(model_info),
        _mtp_step(model_info),
        *_DRAFTER_STEPS,
        "",
        *_CONSTRAINTS,
    ]


def _target_line(model_info: Mapping[str, Any]) -> str:
    """Name the target model the drafter must be trained for."""
    if not model_info:
        return "Target model: not recorded for this session; identify it from the serving configuration you tune."
    return (
        f"Target model: `{model_info.get('model_name')}`, launched from `{model_info.get('model_path')}` "
        "(a local path or a Hugging Face repo id)."
    )


def _mtp_step(model_info: Mapping[str, Any]) -> str:
    """Step 1: use MTP when ``config.json`` declares MTP layers and the weights carry them."""
    if not model_info:
        return "1. **MTP first.** The target's config.json could not be read; check it for an MTP layer count."
    layers = int(model_info.get("mtp_layers") or 0)
    if not layers:
        return "1. **No MTP.** The target's config.json declares no MTP layers; go to the drafter-based methods."
    return (
        f"1. **MTP first.** The target's config.json declares {layers} MTP layer(s). Confirm the weight index "
        "carries those tensors — some quantized or converted checkpoints keep the config field but drop the "
        "weights — then use MTP and stop here. Fall back to the methods below only when the tensors are "
        "missing or MTP fails to start on the installed framework."
    )


__all__ = ["speculative_decoding_focus"]
