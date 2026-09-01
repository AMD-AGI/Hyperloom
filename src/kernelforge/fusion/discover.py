# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 2 (LLM-autonomous discovery): find fusible op chains from trace + source.

Unlike :func:`locate.build_recipes` (pattern-library — capped to known templates),
this asks the LLM to READ the launch-bound decode profile (the trace's hot kernels)
plus the model source and PROPOSE fusible op chains itself. It therefore surfaces
fusions no template encodes — e.g. ZAYA's eager CCA QK chain, whose kernels are
generic ``elementwise``/``cast``/``mul`` and are thus invisible to category-based
patterns (``rmsnorm``+``rope`` share was only ~0.02, below the pattern threshold).

Discovery is no longer trace-only. The primary evidence is still the MEASURED
trace and the REAL source, but a bounded retrieval step over ``local_knowledge``
may additionally surface names of existing ROCm operators. Its limits matter:

* Ranking uses whole-word overlap between observed kernel/category names and the
  document text -- never substring or prefix matching, which would let ``add``
  match ``padding`` and recommend an unrelated operator.
* Retrieval only proposes names. It confirms nothing about shape, dtype, cache
  layout, or numerics; every hint carries a ``score`` and the author must verify
  the operator and record parity before keeping it.
* A retrieved name is not an answer key. The operator still has to correspond to
  a chain that this trace shows running back-to-back and that this source
  actually contains.

The Agent call sits behind an injectable ``llm_fn`` (text prompt -> text), so the
prompt assembly and JSON parsing are unit-testable without a live provider.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import inspect
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from kernelforge.llm import (
    normalize_anthropic_base_url,
    resolve_anthropic_gateway,
    resolve_openai_gateway,
)
from kernelforge.agent_backends.base import AgentRunSpec, AgentToolPolicy, watchdog_timeout_sec
from kernelforge.resources import resource_path

from .diagnose import LAUNCH_BOUND_CATEGORIES, categories_in_text, categorize_kernel_name
from .llm_failure import (
    API_ERROR,
    DEFAULT_ATTEMPTS,
    DEFAULT_BASE_DELAY_SEC,
    DEFAULT_DEADLINE_SEC,
    DEFAULT_MAX_DELAY_SEC,
    NOT_CONFIGURED,
    RETRYABLE_KINDS,
    LlmUnavailableError,
    classify_llm_error,
    env_setting,
    is_agent_safety_error,
    retry_delay,
)
from .locate import (
    _read_source,
    _unclaimable_note,
    covered_by_vllm_compile_pass,
    out_of_scope_terms,
    rank_recipes,
    vllm_compile_pass_state,
)
from .models import Diagnosis, Recipe
from .vllm_passes import PassState, resolve_target_runtime

log = logging.getLogger("forge_fusion")

LlmFn = Callable[[str], str]  # prompt -> raw model text (expected to contain JSON)

# Each proposed fusion costs discovery tokens plus one authoring subprocess and
# validation pass, and that cost is paid before the E2E gate can reject it. Keep
# the default modest; raise it deliberately via ``FORGE_MAX_FUSIONS``.
_DEFAULT_MAX_FUSIONS = 4

# Discovery is handed read and search tools, and the first tool call ends the
# turn. A budget of one therefore guarantees a turn_cap on any session that uses
# the tools it was given, which is what discovery is for. Retries do not help:
# each one opens another single-turn session. Read-only exploration is cheap
# enough to allow a handful of turns; raise it via ``FORGE_FUSION_DISCOVERY_TURNS``.
#
# A handful turned out not to be enough on a large model: on DeepSeek-V4-Flash a
# budget of 12 hit the cap on both attempts it was given, while 60 completed and
# proposed four fusions on each of three runs. The cap is a ceiling and not a
# budget -- a session that finishes in eight turns costs eight turns whatever the
# ceiling is -- so it is set where exploring a large model tree fits under it.
DEFAULT_DISCOVERY_TURNS = 60

# A reasoning model spends this budget on thinking before it writes anything, so
# a small cap does not truncate the answer -- it removes it. Measured against the
# gateway with claude-opus-5 on a real discovery prompt: at 2400 every one of five
# attempts came back with an empty completion; at 16000 the response was 5102
# characters of closed JSON carrying four proposals. Override with
# ``FORGE_FUSION_LLM_MAX_TOKENS``.
DEFAULT_LLM_MAX_TOKENS = 16000


def _resolve_max_fusions(value: Optional[int] = None) -> int:
    if value is not None:
        return max(1, int(value))
    raw = os.environ.get("FORGE_MAX_FUSIONS", "").strip()
    if raw:
        with contextlib.suppress(ValueError):
            return max(1, int(raw))
    return _DEFAULT_MAX_FUSIONS


def hot_kernels_from_trace(
    trace_path: str | Path, *, top_n: int = 15, launch_bound_only: bool = True
) -> list[dict[str, Any]]:
    """Top GPU kernels from a kineto trace by total-duration share.

    Args:
        trace_path: Path to the ``*.trace.json[.gz]``.
        top_n: How many kernels to return.
        launch_bound_only: When True, drop compute-bound categories
            (gemm/attention/conv/moe) so the list is the fusible launch-bound tail.

    Returns:
        ``[{"name", "category", "share", "count", "avg_us"}, ...]`` (share of total
        GPU-kernel time), ordered by descending share.
    """
    p = Path(trace_path)
    try:
        opener = gzip.open if (p.suffix == ".gz" or p.name.endswith(".json.gz")) else open
        with opener(p, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    events = data.get("traceEvents") if isinstance(data, dict) else None
    if not isinstance(events, list):
        return []

    agg: dict[str, list[float]] = {}
    total = 0.0
    for ev in events:
        if not isinstance(ev, dict) or ev.get("cat") != "kernel":
            continue
        try:
            dur = float(ev.get("dur"))
        except (TypeError, ValueError):
            continue
        if dur <= 0:
            continue
        name = str(ev.get("name") or "")
        d = agg.setdefault(name, [0.0, 0])
        d[0] += dur
        d[1] += 1
        total += dur
    if total <= 0:
        return []

    rows: list[dict[str, Any]] = []
    compute = {"gemm", "attention", "conv", "moe"}
    for name, (dur, count) in agg.items():
        cat = categorize_kernel_name(name)
        if launch_bound_only and cat in compute:
            continue
        rows.append(
            {
                "name": name,
                "category": cat,
                "share": dur / total,
                "count": count,
                "avg_us": dur / count,
            }
        )
    rows.sort(key=lambda r: r["share"], reverse=True)
    return rows[:top_n]


def _load_trace_events(trace_path: str | Path) -> list[dict[str, Any]]:
    path = Path(trace_path)
    try:
        opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".json.gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return []
    events = payload.get("traceEvents") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def kernel_names_from_trace(trace_path: str | Path, *, top_n: int = 40) -> list[str]:
    """Distinct kernel names ranked by total duration, compute kernels included.

    :func:`hot_kernels_from_trace` deliberately drops GEMM and attention because
    they are not fusion targets themselves. Operator retrieval still needs them:
    an epilogue operator is identified by the GEMM it attaches to, so excluding
    compute names would make a gated-GEMM card unreachable.
    """
    totals: dict[str, float] = defaultdict(float)
    for event in _load_trace_events(trace_path):
        if event.get("cat") != "kernel":
            continue
        try:
            duration = float(event.get("dur"))
        except (TypeError, ValueError):
            continue
        if duration <= 0:
            continue
        name = str(event.get("name") or "")
        if name:
            totals[name] += duration
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:top_n]]


def ordered_fusion_boundaries_from_trace(
    trace_path: str | Path,
    *,
    top_n: int = 16,
    max_chain_len: int = 8,
    min_repeats: int = 2,
) -> list[dict[str, Any]]:
    """Recover repeated compute-to-compute fusion boundaries from stream order.

    Unlike the launch-bound hot table, this deliberately retains GEMM and
    attention endpoints. That exposes epilogue/prologue opportunities such as a
    GEMM followed by one activation, while also preserving longer post-processing
    runs that end in a cache write before attention.
    """
    streams: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    total_kernel_us = 0.0
    for event in _load_trace_events(trace_path):
        if event.get("cat") != "kernel":
            continue
        try:
            timestamp = float(event.get("ts"))
            duration = float(event.get("dur"))
        except (TypeError, ValueError):
            continue
        if duration <= 0:
            continue
        name = str(event.get("name") or "")
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        stream_key = (
            args.get("device", event.get("pid", 0)),
            args.get("stream", event.get("tid", 0)),
        )
        streams[stream_key].append(
            {
                "name": name,
                "category": categorize_kernel_name(name),
                "ts": timestamp,
                "dur": duration,
            }
        )
        total_kernel_us += duration

    compute_categories = {"gemm", "attention", "conv", "moe"}

    def normalized_name(name: str) -> str:
        value = re.sub(r"0x[0-9a-f]+", "0x*", name.lower())
        value = re.sub(r"\b\d+\b", "N", value)
        return re.sub(r"\s+", " ", value).strip()[:160]

    aggregated: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}

    def record(segment: list[dict[str, Any]]) -> None:
        if len(segment) < 2 or len(segment) > max_chain_len:
            return
        categories = [str(item["category"]) for item in segment]
        if not any(category in LAUNCH_BOUND_CATEGORIES for category in categories):
            return
        key = tuple((str(item["category"]), normalized_name(str(item["name"]))) for item in segment)
        interior_count = max(0, len(segment) - 2)
        if len(segment) == 3 and categories[0] == "gemm" and categories[1] in LAUNCH_BOUND_CATEGORIES:
            boundary_kind = "epilogue"
        elif categories[0] in compute_categories:
            boundary_kind = "compute_boundary"
        else:
            boundary_kind = "vertical"
        # The trailing kernel is the NEXT compute anchor. It is kept as adjacency
        # evidence but is not part of what can be fused: a native prologue
        # operator fuses norm/RoPE/cache-write, never the attention kernel it
        # feeds. The leading anchor is likewise the producer, except that an
        # ``epilogue`` boundary may absorb it (see boundary_kind).
        terminal_compute = categories[-1] if categories[-1] in compute_categories else ""
        fusable_categories = categories[1:-1] if terminal_compute else categories[1:]
        row = aggregated.setdefault(
            key,
            {
                "signature": " -> ".join(categories),
                "categories": categories,
                "fusable_categories": fusable_categories,
                "terminal_compute": terminal_compute,
                "kernels": [str(item["name"])[:200] for item in segment],
                "count": 0,
                "total_us": 0.0,
                "boundary_kind": boundary_kind,
                "launches_removed_upper_bound": max(1, interior_count),
            },
        )
        row["count"] += 1
        row["total_us"] += sum(float(item["dur"]) for item in segment)

    for stream_events in streams.values():
        ordered = sorted(stream_events, key=lambda item: item["ts"])
        start_index: Optional[int] = None
        for index, event in enumerate(ordered):
            if event["category"] not in compute_categories:
                continue
            if start_index is not None:
                record(ordered[start_index : index + 1])
            start_index = index
        if start_index is not None:
            record(ordered[start_index:])

    rows: list[dict[str, Any]] = []
    for row in aggregated.values():
        if int(row["count"]) < min_repeats:
            continue
        row["avg_chain_us"] = row["total_us"] / row["count"]
        # Ranking heuristic only, NOT a true fraction of GPU time: a kernel that
        # sits between two compute anchors belongs to two overlapping segments,
        # so its duration is counted once per segment and shares can sum above 1.
        row["share_heuristic"] = row["total_us"] / total_kernel_us if total_kernel_us > 0 else 0.0
        rows.append(row)
    rows.sort(
        key=lambda row: (
            int(row["count"]) * int(row["launches_removed_upper_bound"]),
            float(row["total_us"]),
        ),
        reverse=True,
    )
    return rows[:top_n]


def _semantic_terms(boundary: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for value in boundary.get("categories", []):
        terms.update(re.findall(r"[a-z0-9]+", str(value).lower()))
    for value in boundary.get("kernels", []):
        terms.update(re.findall(r"[a-z0-9]+", str(value).lower()))

    aliases = {
        "activation": {"act", "gelu", "relu", "silu", "swiglu"},
        "copy": {"cache", "copy", "store", "write"},
        "elementwise": {"copy", "elementwise", "materialize"},
        "gemm": {"gate", "gemm", "linear", "matmul", "projection", "up"},
        "rmsnorm": {"norm", "rms", "rmsnorm"},
        "rope": {"rope", "rotary"},
        "attention": {"attention", "attn"},
    }
    for term in tuple(terms):
        if term in aliases:
            terms.update(aliases[term])
        if "cache" in term:
            terms.update({"cache", "store", "write"})
        if term in {"act", "activation"}:
            terms.update(aliases["activation"])
    return {term for term in terms if len(term) >= 3}


def _default_knowledge_root() -> Path:
    configured = os.environ.get("FORGE_LOCAL_KNOWLEDGE", "").strip()
    if configured:
        return Path(configured)
    return resource_path("local_knowledge", missing_ok=True)


def _tokens(text: str) -> set[str]:
    """Whole-word tokens of ``text``.

    Retrieval matches on these sets rather than on substrings: ``add`` is a
    substring of ``padding`` and ``norm`` is a prefix of ``normalization``, and
    neither implies the document describes the observed operation. A false recall
    is more harmful than a miss, because the author is then instructed to
    integrate an unrelated operator.
    """
    return set(re.findall(r"[a-z0-9]+", text.lower()))


_OPERATOR_MARKERS = (
    "fused",
    "gemm",
    "rope",
    "cache",
    "silu",
    "swiglu",
    "rmsnorm",
    "attention",
)
_OPERATOR_PATTERN = re.compile(r"`([A-Za-z_][A-Za-z0-9_.:]*)`")
_DECLARED_OPERATOR_PATTERN = re.compile(r"^operator:\s*([A-Za-z_][A-Za-z0-9_.:]*)\s*$", re.MULTILINE)

# Parsed knowledge documents, keyed by (path, mtime_ns, size) so an unchanged
# knowledge base is not re-read and re-parsed on every discovery run.
_KNOWLEDGE_CACHE: dict[tuple[str, int, int], list[dict[str, Any]]] = {}


def _operator_terms(operator: str) -> set[str]:
    terms = _tokens(operator)
    if any(term.startswith("gate") for term in terms):
        terms.update({"gate", "activation"})
    if "kv" in terms:
        terms.update({"cache", "store"})
    if "qk" in terms:
        terms.update({"norm", "rope"})
    return {term for term in terms if len(term) >= 3}


def _parse_knowledge_document(path: Path) -> list[dict[str, Any]]:
    """Extract every candidate operator mention from one knowledge document."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []
    declared_match = _DECLARED_OPERATOR_PATTERN.search(text)
    declared_operator = declared_match.group(1) if declared_match else ""
    entries: list[dict[str, Any]] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph_tokens = _tokens(paragraph)
        for operator in _OPERATOR_PATTERN.findall(paragraph):
            lowered_operator = operator.lower()
            if not any(marker in lowered_operator for marker in _OPERATOR_MARKERS):
                continue
            if "_" not in operator and "." not in operator:
                continue
            entries.append(
                {
                    "operator": operator,
                    "operator_terms": _operator_terms(operator),
                    "paragraph_tokens": paragraph_tokens,
                    "paragraph_length": len(paragraph),
                    "evidence": re.sub(r"\s+", " ", paragraph).strip()[:300],
                    "is_declared": operator == declared_operator,
                }
            )
    return entries


def _knowledge_entries(root: Path) -> list[tuple[Path, list[dict[str, Any]]]]:
    documents: list[tuple[Path, list[dict[str, Any]]]] = []
    for path in root.rglob("*.md"):
        try:
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size)
        except OSError:
            continue
        entries = _KNOWLEDGE_CACHE.get(key)
        if entries is None:
            entries = _parse_knowledge_document(path)
            _KNOWLEDGE_CACHE[key] = entries
        documents.append((path, entries))
    return documents


def existing_operator_hints_from_knowledge(
    knowledge_root: str | Path | None,
    boundaries: list[dict[str, Any]],
    *,
    limit: int = 12,
    fallback_categories: Optional[list[str]] = None,
    fallback_kernel_names: Optional[list[str]] = None,
    min_score_ratio: float = 0.25,
) -> list[dict[str, Any]]:
    """Retrieve existing ROCm operator names using observed runtime semantics.

    This is evidence retrieval, not model-name matching: documents rank by
    whole-word overlap with the observed operation categories and kernel names,
    and every hint carries its ``score`` so the author can tell a strong match
    from a marginal one.

    ``fallback_categories`` / ``fallback_kernel_names`` (typically the diagnosis
    categories and hot-kernel names) are always folded in as an extra evidence
    source. Ordered boundaries require ``min_repeats`` occurrences to exist at
    all, so a short trace can leave them empty while the hot-kernel table still
    proves a launch-bound chain; without this, retrieval would silently go dark.
    """
    root = Path(knowledge_root) if knowledge_root else _default_knowledge_root()
    if not root.is_dir():
        return []

    evidence_sources = list(boundaries)
    if fallback_categories or fallback_kernel_names:
        evidence_sources.append(
            {
                "categories": list(fallback_categories or []),
                "kernels": list(fallback_kernel_names or []),
            }
        )
    if not evidence_sources:
        return []

    boundary_terms = [_semantic_terms(source) for source in evidence_sources]
    boundary_terms = [terms for terms in boundary_terms if terms]
    if not boundary_terms:
        return []

    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for path, entries in _knowledge_entries(root):
        for entry in entries:
            op_terms: set[str] = entry["operator_terms"]
            paragraph_tokens: set[str] = entry["paragraph_tokens"]
            score = float("-inf")
            for terms in boundary_terms:
                evidence_overlap = len(terms & paragraph_tokens)
                operator_overlap = len(op_terms & terms)
                candidate_score = operator_overlap * 100.0 + evidence_overlap * 10.0 - entry["paragraph_length"] / 500.0
                if entry["is_declared"]:
                    candidate_score += 500.0
                score = max(score, candidate_score)
            if score < 20.0:
                continue
            try:
                relative_path = path.relative_to(root).as_posix()
            except ValueError:
                relative_path = str(path)
            row = {
                "operator": entry["operator"],
                "path": relative_path,
                "evidence": entry["evidence"],
                "score": round(score, 1),
            }
            previous = best.get(entry["operator"])
            if previous is None or score > previous[0]:
                best[entry["operator"]] = (score, row)

    ranked = sorted(
        best.values(),
        key=lambda item: (item[0], item[1]["operator"]),
        reverse=True,
    )
    if not ranked:
        return []
    # Pre-trim marginal matches relative to the best one, so a long tail of weak
    # hints cannot pad the prompt and inflate downstream authoring attempts.
    cutoff = ranked[0][0] * min_score_ratio
    return [row for score, row in ranked[:limit] if score >= cutoff]


# Terms a proposal may declare in ``ops``. Two consumers read the result, which
# is why one list covers both: the op-category vocabulary that forms the KB
# identity, plus the finer terms the compile-pass table keys on (``mla``,
# ``quant``, ``qk_norm`` ...) which no category can express.
#
# Declaring beats inferring because both consumers used to keyword-match the
# model's prose, and prose varies per run. Measured against one unchanged trace,
# a proposal that merely mentioned writing to the KV cache picked up a ``copy``
# category it did not fuse, and a reworded proposal stopped matching the
# compile-pass keywords -- which changed which candidate ranked first and thus
# which key the run looked up.
# What a fused kernel COMPUTES. This is the fusion's identity, so every term has
# to answer one question -- "does the kernel carry out this operation?" -- and
# has to be recognised by ``categories_in_text``, since that is what turns the
# declaration into the category set the KB keys on.
#
# ``cast`` and ``moe`` are absent because they fail that second requirement: the
# category rules match kernel-name spellings (``_cast``, ``fused_moe``), so a
# bare declaration of either produces no category and would be silently inert.
# Named activations (silu/gelu/swiglu) are absent too -- ``activation`` covers
# them for the compile-pass table, and naming one in a prompt hands the model a
# specific fusion it was not asked to look for.
FUSION_OP_VOCAB: frozenset[str] = frozenset(
    {
        "activation",
        "add",
        "conv",
        "copy",
        "gemm",
        "layernorm",
        "mul",
        "reduce",
        "rmsnorm",
        "rope",
        "sample",
    }
)

# HOW the kernel is built, not what it computes: precision, architecture variant,
# and where in the model it sits. Separated from the ops on purpose.
#
# Two reasons. These terms cannot be judged by the ops question -- a kernel does
# not "perform fp8" or "perform mla" -- so mixing them into one list left the
# model applying a rule that fit only half the entries. And they should not move
# the key: a run that reads the same fusion as fp8 rather than quantized, or is
# unsure whether the chain counts as attention, must still look up where the
# previous run stored it. Measured over 20 runs, ``attention`` was the one term
# that flipped, and it contributes nothing to identity -- nearly every decode
# fusion sits next to attention or the MLP.
FUSION_TRAIT_VOCAB: frozenset[str] = frozenset(
    {
        "attention",
        "concat",
        "dual",
        "fp8",
        "k_norm",
        "kvcache",
        "mla",
        "q_norm",
        "qk_norm",
        "quant",
    }
)

_OP_VOCAB_FOR_PROMPT = ", ".join(sorted(FUSION_OP_VOCAB))
_TRAIT_VOCAB_FOR_PROMPT = ", ".join(sorted(FUSION_TRAIT_VOCAB))


def _declared_terms(item: Any, field: str, vocab: frozenset[str]) -> list[str]:
    """A proposal's declaration for ``field``, normalized; ``[]`` when unusable.

    Unknown entries are dropped rather than trusted: an invented term would
    otherwise invent an identity segment, and two runs inventing different ones
    would split a single fusion across two keys.
    """
    raw = item.get(field) if isinstance(item, dict) else None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    return sorted({token for entry in raw if (token := str(entry).strip().lower().replace("-", "_")) in vocab})


def declared_ops(item: Any) -> list[str]:
    """The ops a proposal claims its kernel computes."""
    return _declared_terms(item, "ops", FUSION_OP_VOCAB)


def declared_traits(item: Any) -> list[str]:
    """The build traits a proposal declares (precision, variant, placement)."""
    return _declared_terms(item, "traits", FUSION_TRAIT_VOCAB)


def build_discovery_prompt(
    *,
    model_type: str,
    framework: str,
    source_text: str,
    diagnosis: Diagnosis,
    hot_kernels: list[dict[str, Any]],
    shapes: dict[str, Any],
    max_fusions: int = _DEFAULT_MAX_FUSIONS,
    ordered_boundaries: Optional[list[dict[str, Any]]] = None,
    existing_operator_hints: Optional[list[dict[str, str]]] = None,
) -> str:
    """Assemble the discovery prompt from runtime, source, and operator evidence.

    No model-specific answer is encoded. Existing operator names are included only
    when semantic retrieval ties their documented operation chain to an observed
    repeated runtime boundary.
    """
    lb = ", ".join(sorted(LAUNCH_BOUND_CATEGORIES))
    hot_lines = "\n".join(
        f"  - {k['category']:11s} {k['share'] * 100:5.1f}%  (n={k['count']}, avg={k['avg_us']:.1f}us)  {k['name'][:90]}"
        for k in hot_kernels
    )

    def boundary_avg_us(boundary: dict[str, Any]) -> float:
        if "avg_chain_us" in boundary:
            return float(boundary["avg_chain_us"])
        return float(boundary.get("total_us", 0.0)) / max(1, int(boundary["count"]))

    def boundary_line(boundary: dict[str, Any]) -> str:
        line = (
            "  - "
            f"{boundary['signature']} "
            f"(repeats={boundary['count']}, "
            f"avg-chain={boundary_avg_us(boundary):.1f}us, "
            f"kind={boundary['boundary_kind']}, "
            f"removable-launches<={boundary['launches_removed_upper_bound']})"
        )
        # The trailing compute kernel proves adjacency but is not fusable, so the
        # fusable span is spelled out to keep the proposed chain from swallowing
        # the attention (or other compute) kernel it feeds.
        fusable = boundary.get("fusable_categories")
        terminal = str(boundary.get("terminal_compute") or "")
        if fusable:
            line += f"\n    fusable-span={' -> '.join(fusable)}"
        if terminal:
            line += (
                f"\n    NOTE: fuse the prologue only; do NOT include the terminal {terminal} kernel in the fused chain."
            )
        line += f"\n    kernels: {' | '.join(boundary.get('kernels', []))[:500]}"
        return line

    boundary_lines = "\n".join(boundary_line(boundary) for boundary in (ordered_boundaries or []))

    def operator_line(hint: dict[str, Any]) -> str:
        score = hint.get("score")
        score_text = f" score={float(score):.1f}" if score is not None else ""
        return f"  - `{hint['operator']}` ({hint['path']}){score_text}: {hint['evidence']}"

    operator_lines = "\n".join(operator_line(hint) for hint in (existing_operator_hints or []))
    dom = ", ".join(diagnosis.dominant_categories)
    return f"""You are analyzing the DECODE path of a {framework} model (`model_type={model_type}`)
on AMD MI325X (gfx942), bf16, to find SOURCE-LEVEL KERNEL FUSIONS. Analyze only; do
not edit anything. Return your answer as JSON (schema below).

## The lever (general, not model-specific)
The decode path is launch/dispatch bound: launch_bound_share={diagnosis.launch_bound_share:.2f}
of GPU-busy time is in many tiny fp32 ops ({lb}) rather than the heavy GEMM/attention.
Each tiny op is a separate kernel launch + HBM round-trip. The win is to FUSE a
CONTIGUOUS chain of these tiny ops (as they appear in the decode forward, between two
heavy ops) into ONE kernel — fewer launches, fewer round-trips.

## Measured launch-bound hot kernels (top, this trace)
{hot_lines}
Dominant launch-bound categories: {dom}
Representative decode shapes: {shapes}

## Ordered fusion boundaries (same stream, compute endpoints retained)
{boundary_lines or "  - none"}

## Existing ROCm operator evidence
Retrieved by whole-word overlap between the observed kernel/category names and the
documented operation chain; a higher score means stronger evidence. The score is a
retrieval hint, NOT a correctness claim: you must still confirm from the source and
the card that shape, dtype, and cache layout actually match.
{operator_lines or "  - none found"}

## Your task
Read the model source below and identify up to {max_fusions} CONTIGUOUS op chains in
the DECODE forward that are worth fusing. Include GEMM/attention prologue or epilogue
boundaries when the ordered trace proves adjacency. Treat an existing ROCm operator
that covers a larger boundary as an `integration` candidate and benchmark/wire it
before proposing a new kernel. Judge from the SOURCE and ordered trace what actually
runs back-to-back on the decode path.

Constraints for each proposed fusion:
- Must be a real contiguous chain in this source (name the exact functions/methods).
- SCOPE — the single hardest constraint, and the one that wastes a whole run when
  it is broken. The fusion is delivered by REPLACING one call site in the source
  file printed below, so the entire chain must live inside THAT file, and every
  tensor your kernel takes as input must already be a local name at that call
  site. Do not fuse across a boundary: not into a method defined in another
  module (an imported `XMLP`, an imported norm class), and not into work the
  framework performs below the call (in vLLM the KV-cache write happens inside
  the attention backend, so `key_cache` / `value_cache` / `slot_mapping` are NOT
  reachable from a model `forward` and a fusion folding them in cannot be wired).
  Before proposing, name the exact call site you would replace and check that
  every input is in scope there. A chain that fails this test is worth zero
  end-to-end even when its microbenchmark is 30x.
- One patch, one file. If two different modules each hold a fusible chain,
  propose them as two SEPARATE entries, each self-contained in its own file --
  never one entry spanning both.
- ROCm-native: it will be authored as a Triton kernel; do NOT propose reusing a
  framework CUDA-only fused op.
- Existing AITER/CK/HIP/Triton operators listed above are allowed and preferred when
  their semantics, dtype, shape, and cache layout match.
- The correctness reference must be the REAL eager op imported from this source
  (say which symbol to import), never a re-derivation.

## Output — a single JSON array (and nothing after it). Be TERSE to fit the
## response budget: keep ``fusion_math`` <= 2 sentences and ``rationale`` <= 1
## sentence. Each element:
{{"name": "<short_id>", "env_flag": "<{model_type.upper()}_FUSED_...>",
  "op_chain": "<the eager methods/ops fused, e.g. A + B>",
  "ops": [<every op YOUR fused kernel computes itself, chosen ONLY from:
          {_OP_VOCAB_FOR_PROMPT}.
          This list IS the fusion's identity: two runs proposing the same fusion
          must produce the same list, so decide by one test rather than by
          impression. For each candidate ask: does my kernel carry out that
          computation? If the op runs in the surrounding module, or you only
          read its result, or you only hand your result to it, then it is NOT
          yours -- leave it out. Where the kernel sits is irrelevant; only what
          it computes counts. List every op that passes the test, and nothing
          else.>],
  "traits": [<how the kernel is built, chosen ONLY from:
          {_TRAIT_VOCAB_FOR_PROMPT}.
          Precision, architecture variant, and which part of the model this sits
          in. These describe the kernel rather than name an operation it
          performs, so they do NOT belong in "ops". Omit the field when none
          apply.>],
  "source_anchors": ["<symbol/line to grep>", "..."],
  "fusion_math": "<what the fused kernel computes, precisely>",
  "eager_reference": "<which real symbol(s) to import + call for the parity ref>",
  "candidate_kind": "<integration|new_fusion|replacement>",
  "existing_operator": "<operator name when candidate_kind=integration, else empty>",
  "priority": <0.0-1.0 by expected launch-bound time saved>,
  "rationale": "<why this chain, tied to the hot kernels above>"}}

## Model source (`{model_type}` in {framework})
```python
{source_text}
```
"""


def _salvage_objects(text: str) -> list[dict[str, Any]]:
    """Recover every complete top-level ``{...}`` object from (possibly truncated)
    text, ignoring braces inside strings. Used when the enclosing JSON array is
    unclosed because the model response was cut off at ``max_tokens`` — the
    complete objects before the cut are still usable proposals.
    """
    out: list[dict[str, Any]] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    with contextlib.suppress(json.JSONDecodeError, ValueError):
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            out.append(obj)
    return out


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Pull JSON fusion proposals out of model text.

    Tries, in order: a fenced ```json [...]``` block, then any balanced top-level
    ``[...]`` span, then (fallback for a response truncated at ``max_tokens``) the
    set of complete ``{...}`` objects.
    """
    if not text:
        return []
    fences = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    candidates = list(fences)
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start : i + 1])
    for cand in reversed(candidates):
        try:
            parsed = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
            return parsed
    # Fallback: salvage complete objects from a truncated/unclosed array, then
    # the object the cut left half-written -- with three quarters of responses
    # arriving truncated, that last object is often the only one there is.
    salvaged = _salvage_objects(text)
    repaired = _repair_truncated_object(text)
    if repaired is not None and repaired not in salvaged:
        salvaged.append(repaired)
    if not salvaged:
        log.warning(
            "discovery: no JSON proposals parsed from %d chars of model text; "
            "this is a parse failure, NOT a no_opportunity result",
            len(text),
        )
    return salvaged


# A repaired object has to carry enough of the fusion description to act on.
# A name and an env flag alone would only send the author stage looking for
# something the model never got round to describing.
_REPAIRED_REQUIRED_ANY = ("op_chain", "fusion_math")


def _repair_truncated_object(text: str) -> dict[str, Any] | None:
    """Recover the proposal that a cut-off response left half-written.

    Rewinds the trailing unclosed object to its last complete ``"key": value``
    boundary and closes it there. Returns ``None`` unless the result still
    describes a fusion, so a response cut inside the very first field is
    dropped rather than turned into an empty proposal.
    """
    depth = 0
    start = -1
    in_str = False
    esc = False
    boundaries: list[int] = []
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
                boundaries = []
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    start = -1
                    boundaries = []
        elif ch == "," and depth == 1:
            boundaries.append(i)
    if depth <= 0 or start < 0:
        return None

    attempts = [text[start:] + ('"' if in_str else "") + "}" * depth]
    attempts.extend(text[start:mark] + "}" for mark in reversed(boundaries))
    for attempt in attempts:
        try:
            obj = json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict) or not obj:
            continue
        if not str(obj.get("name") or "").strip():
            continue
        if any(str(obj.get(key) or "").strip() for key in _REPAIRED_REQUIRED_ANY):
            return obj
    return None


def _norm_env_flag(flag: str, model_type: str) -> str:
    """Normalize/model-prefix a proposed env flag (e.g. FUSED_QK -> ZAYA_FUSED_QK)."""
    f = re.sub(r"\s+", "_", (flag or "FUSED").strip()).upper()
    prefix = f"{model_type.upper()}_" if model_type else ""
    if prefix and not f.startswith(prefix):
        f = prefix + f
    return f


def parse_discovered_recipes(
    text: str,
    *,
    model_type: str,
    framework: str,
    source_file: str,
    shapes: dict[str, Any],
    category_shares: dict[str, float] | None = None,
    pass_probe: Optional[Callable[[str], PassState]] = None,
    framework_root: str = "",
) -> list[Recipe]:
    """Convert the LLM's JSON proposals into ranked :class:`Recipe` objects.

    ``category_shares`` are the op-category shares the trace actually measured.
    They are used to drop a proposal whose ops were never observed at all, which
    is the signature of an LLM inventing a fusion the workload does not perform.
    They deliberately do NOT filter the categories that identify the fusion: a
    fusion is the same fusion whatever a given trace happened to sample, and
    letting run-time sampling into the identity would split one fusion across
    several pages. This mirrors the pattern route, where the trace decides
    whether a pattern TRIGGERS while its identity stays the fixed pattern id.

    A proposal vLLM implements as a compile pass is dropped only when that pass is
    ENABLED; when it exists, is off and is flippable it becomes a ``compile_pass``
    recipe, and when it is absent / undecidable / pinned off by the optimization
    level the proposal stays authoring work with ``compile_pass_note`` recording why.
    """
    runtime = resolve_target_runtime(framework, framework_root=framework_root)
    # The same file the prompt embedded, re-read so the scope gate below judges a
    # proposal against exactly the source the model was shown.
    source_text = _read_source(source_file)
    out: list[Recipe] = []
    for i, item in enumerate(_extract_json_array(text)):
        name = str(item.get("name") or f"discovered_{i + 1}").strip()
        try:
            priority = float(item.get("priority"))
        except (TypeError, ValueError):
            priority = max(0.1, 1.0 - 0.1 * i)  # preserve LLM order when absent
        anchors = item.get("source_anchors") or []
        if isinstance(anchors, str):
            anchors = [anchors]
        op_chain = str(item.get("op_chain") or "")
        fusion_math = str(item.get("fusion_math") or op_chain or "")
        # The fusion-DEFINING fields (name / op-chain / math) -- NOT the free-prose
        # rationale or grep anchors, which can mention an op in passing and would
        # attach a category the fusion does not actually involve.
        defining_text = " ".join([name, op_chain, fusion_math])
        # What this fusion IS, as opposed to how this run described it. Both the
        # category set below and the compile-pass gate further down read this one
        # string, so it decides the key -- which is why a declaration from a fixed
        # vocabulary is preferred over the prose. The prose remains the fallback
        # for a model that ignores the field, at the cost of that run's identity
        # depending on its wording.
        declared = declared_ops(item)
        traits = declared_traits(item)
        identity_text = " ".join(declared) if declared else defining_text
        # The gate keys on precision and variant words (quant, fp8, mla, kvcache)
        # that no op category expresses, so it needs more than ``declared``.
        # Where that comes from depends on whether the model supplied ``traits``:
        #
        # * It did -- use the declarations alone. They say precisely which
        #   variant this is, and adding prose can only introduce words the model
        #   did not mean. A wording that happens to mention the KV cache matched
        #   ``fuse_rope_kvcache`` while its terser twin matched ``qk_norm_rope``,
        #   and since a claimed pass rewrites the pattern id, that split the key.
        # * It did not -- fall back to the prose. ``traits`` is the optional
        #   field, so this is the common case, and without the fallback the gate
        #   loses every keyword it matches on: the run then hand-writes a kernel
        #   vLLM already ships, under a different key than a run that declared.
        #
        # Either way ``identity_text`` above is untouched, so the key's category
        # segment still comes from the declaration alone.
        gate_text = " ".join([*declared, *traits]) if traits else " ".join([*declared, defining_text])
        # Recover the op categories: from the declaration when there is one, else
        # from the prose via the fixed, model-agnostic vocabulary. The KB keys on
        # this, and ``op_chain`` is not kept on the Recipe, so it has to happen
        # here while the field is still in scope.
        matched_categories = categories_in_text(identity_text)
        # Hallucination gate FIRST: a proposal whose ops the trace never measured
        # has nothing to remove, and that is true whether we would author it or
        # claim a framework pass for it.
        if category_shares and matched_categories:
            if not any(float(category_shares.get(c, 0.0)) > 0.0 for c in matched_categories):
                log.info(
                    "discovery: dropping %s (proposed ops %s absent from the trace)",
                    name,
                    ",".join(matched_categories),
                )
                continue
        # SCOPE gate: a fusion is wired by replacing ONE call site in the file the
        # model was shown, so a proposal claiming ops that file never performs is
        # unwireable no matter how good the kernel is. Dropping it here costs one
        # JSON object; keeping it costs a full authoring campaign that ends in an
        # orphan module (see ``locate.out_of_scope_terms``).
        outside = out_of_scope_terms(source_text, [*declared, *traits])
        if outside:
            log.info(
                "discovery: dropping %s (%s not performed in %s -- the fusion crosses "
                "a module boundary and has no wireable call site there)",
                name,
                ",".join(outside),
                Path(source_file).name or source_file,
            )
            continue
        # Compile-pass gate: never author a chain vLLM fuses at compile time.
        # Matched keyword-only, because the gate's own vocabulary differs from the
        # op-category vocabulary derived above. Reads ``identity_text`` for the
        # same reason that does: claiming a pass rewrites the pattern id, so a
        # match that flips on a rewording would move the key with it.
        pass_name = covered_by_vllm_compile_pass(
            matched_categories=[],
            text=gate_text,
            framework=framework,
        )
        pass_note = ""
        if pass_name:
            state = vllm_compile_pass_state(pass_name, probe=pass_probe, runtime=runtime)
            if state is not None and state.enabled is True:
                log.info(
                    "discovery: dropping %s, vLLM compile pass %s already fuses it",
                    name,
                    state.flag or pass_name,
                )
                continue  # the framework really does fuse this: authoring is a no-op
            if state is not None and state.claimable:
                # Present but switched off and flippable: claim the native pass.
                out.append(
                    Recipe(
                        pattern_id=f"compile_pass:{state.flag}",
                        description=(
                            f"vLLM implements this chain as compile pass `{state.flag}`, but "
                            f"it is DISABLED in this install: enable the native pass instead "
                            f"of authoring a kernel ({str(item.get('rationale') or op_chain or name)[:200]})"
                        ),
                        env_flag="",
                        source_file=state.config_file,
                        source_hints=[state.flag],
                        fusion_math=fusion_math,
                        eager_reference_hint="",
                        shapes=shapes,
                        matched_categories=matched_categories,
                        trigger_share=priority,
                        rocm_native=True,
                        source_confirmed=True,
                        already_satisfied=False,
                        candidate_kind="compile_pass",
                        compile_pass_flag=state.flag,
                    )
                )
                continue
            # Absent / undecidable / pinned off: the framework is NOT fusing this
            # for us, so keep the proposal as authoring work and record why.
            if state is not None:
                pass_note = _unclaimable_note(state)
                log.info("compile pass not claimed for %s: %s", name, pass_note)
        existing_operator = str(item.get("existing_operator") or "").strip()
        candidate_kind = str(item.get("candidate_kind") or "").strip().lower()
        if candidate_kind not in {"integration", "new_fusion", "replacement"}:
            candidate_kind = "integration" if existing_operator else "new_fusion"
        # ``integration`` is only meaningful with a named operator: the authoring
        # prompt injects its "benchmark the existing operator first" block only
        # when both fields are set, so an operator-less integration would claim
        # the kind while silently skipping the constraint.
        if candidate_kind == "integration" and not existing_operator:
            candidate_kind = "new_fusion"
        out.append(
            Recipe(
                pattern_id=f"llm:{name}",
                description=str(item.get("rationale") or op_chain or name)[:300],
                env_flag=_norm_env_flag(str(item.get("env_flag") or "FUSED"), model_type),
                source_file=source_file,
                source_hints=[str(a) for a in anchors],
                fusion_math=fusion_math,
                eager_reference_hint=str(item.get("eager_reference") or ""),
                shapes=shapes,
                matched_categories=matched_categories,
                trigger_share=priority,
                rocm_native=True,
                source_confirmed=True,  # the LLM read the real source to propose it
                already_satisfied=False,
                candidate_kind=candidate_kind,
                existing_operator=existing_operator,
                compile_pass_note=pass_note,
            )
        )
    out.sort(key=lambda r: r.trigger_share, reverse=True)
    return rank_recipes(out)


def complete_with_retry(
    client: Any,
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay_sec: float = DEFAULT_BASE_DELAY_SEC,
    max_delay_sec: float = DEFAULT_MAX_DELAY_SEC,
    deadline_sec: float | None = None,
    sleep: Callable[[float], Any] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> str:
    """Ask the gateway once, retrying only the failures a retry can fix.

    Retryable means :data:`~kernelforge.fusion.llm_failure.RETRYABLE_KINDS`: a generic
    API error or a timeout. A timeout says the request did not come back THIS
    time, which is the transient degradation this chain exists for; credentials
    and an over-long prompt fail the same way forever and stop on the first one.

    ``max_tokens`` stays fixed across attempts. The previous hedge shrank it on
    every retry, but the gateway's 400s carry no reason and recur at any cap
    (measured success rates were indistinguishable from 512 to 16384 tokens), so
    shrinking only truncated the answer we were trying to get; the one failure a
    smaller request does fix — an over-long prompt — classifies as
    ``context_length`` and is not retried at all.

    ``deadline_sec`` bounds the wall clock, because the attempt count does not:
    each attempt can sit on the client's read timeout, so a retried timeout is
    the one kind that could otherwise hold discovery for over an hour.

    An empty completion counts as a failure, not as an answer: discovery's
    prompt requires a JSON array, so a model that genuinely found nothing
    replies ``[]``. Treating "" as "no fusions" is the same conflation this
    module exists to prevent.
    """
    import time as _time

    pause = sleep or _time.sleep
    clock = monotonic or _time.monotonic
    budget = float(
        deadline_sec
        if deadline_sec is not None
        else env_setting("FORGE_LLM_RETRY_DEADLINE_SEC", DEFAULT_DEADLINE_SEC, cast=float)
    )
    started_at = clock()
    last_error = ""
    last_kind = API_ERROR
    for attempt in range(1, attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text
            last_error = "gateway returned an empty completion"
            last_kind = API_ERROR
        except Exception as exc:  # noqa: BLE001 — classified immediately below
            kind = classify_llm_error(exc)
            last_error = f"{type(exc).__name__}: {str(exc)[:240]}"
            last_kind = kind
            if kind not in RETRYABLE_KINDS:
                raise LlmUnavailableError(
                    f"discovery LLM call failed ({kind}): {last_error}",
                    kind=kind,
                    attempts=attempt,
                ) from exc
        log.warning("discovery llm_fn attempt %d/%d failed: %s", attempt, attempts, last_error)
        if attempt >= attempts:
            break
        if budget > 0 and (clock() - started_at) >= budget:
            raise LlmUnavailableError(
                f"discovery LLM unreachable after {attempt} attempt(s) and "
                f"{clock() - started_at:.0f}s (deadline {budget:.0f}s): {last_error}",
                kind=last_kind,
                attempts=attempt,
            )
        pause(retry_delay(attempt, base_sec=base_delay_sec, max_sec=max_delay_sec))
    raise LlmUnavailableError(
        f"discovery LLM unreachable after {attempts} attempts: {last_error}",
        kind=last_kind,
        attempts=attempts,
    )


class DiscoverySafetyError(RuntimeError):
    """Report any source mutation made by a discovery-only Agent session."""


_DISCOVERY_SYSTEM_PROMPT = """\
You are the read-only discovery stage of KernelForge forge-fuse.
Analyze the evidence in the user prompt and return only the requested final text.
Do not edit, create, delete, or rename files. Do not run commands that modify the
workspace. An OS-level full-access preset is valid only when an explicit external
sandbox is authoritative; it does not grant logical write or shell permission.
"""


def _protected_file_snapshot(
    protected_files: list[str],
) -> dict[Path, tuple[bool, bytes, int]]:
    """Capture exact bytes so discovery can detect and undo source mutations."""
    snapshot: dict[Path, tuple[bool, bytes, int]] = {}
    for value in protected_files:
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        try:
            exists = path.is_file()
            snapshot[path] = (
                exists,
                path.read_bytes() if exists else b"",
                path.stat().st_mode & 0o777 if exists else 0,
            )
        except OSError as exc:
            raise DiscoverySafetyError(f"cannot snapshot protected discovery source {path}: {exc}") from exc
    return snapshot


def _restore_changed_protected_files(
    snapshot: dict[Path, tuple[bool, bytes, int]],
) -> list[str]:
    """Restore changed protected files and return their paths."""
    changed: list[str] = []
    for path, (existed, content, mode) in snapshot.items():
        try:
            exists = path.is_file()
            differs = exists != existed
            if exists and existed:
                differs = path.read_bytes() != content or (path.stat().st_mode & 0o777) != mode
            if not differs:
                continue
            changed.append(str(path))
            if existed:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                path.chmod(mode)
            elif path.exists() or path.is_symlink():
                path.unlink()
        except OSError as exc:
            raise DiscoverySafetyError(f"discovery modified protected source {path} and restore failed: {exc}") from exc
    return changed


def _run_agent_discovery_once(
    backend: Any,
    spec: AgentRunSpec,
    *,
    timeout_s: int,
    protected_files: list[str],
) -> Any:
    """Run one SDK turn and enforce the provider-neutral source invariant."""
    snapshot = _protected_file_snapshot(protected_files)

    async def _run() -> Any:
        return await asyncio.wait_for(
            backend.run(spec),
            timeout=watchdog_timeout_sec(max(1, int(timeout_s))),
        )

    try:
        result = asyncio.run(_run())
    except BaseException as exc:
        changed = _restore_changed_protected_files(snapshot)
        if changed:
            raise DiscoverySafetyError("discovery agent modified protected source: " + ", ".join(changed)) from exc
        raise
    changed = _restore_changed_protected_files(snapshot)
    if changed:
        raise DiscoverySafetyError("discovery agent modified protected source: " + ", ".join(changed))
    return result


def registered_agent_llm_fn(
    backend: Any,
    *,
    model: str = "",
    timeout_s: int = 900,
    log_path: str = "",
    workdir: str = ".",
    protected_files: Optional[list[str]] = None,
    attempts: Optional[int] = None,
    base_delay_sec: Optional[float] = None,
    max_delay_sec: Optional[float] = None,
    deadline_sec: Optional[float] = None,
    sleep: Optional[Callable[[float], Any]] = None,
    monotonic: Optional[Callable[[], float]] = None,
) -> LlmFn:
    """Adapt one registered Agent backend into discovery's text interface.

    The source is already embedded in the prompt, so the session gets read/search
    tools but no write or shell tools. ``allow_dirty_baseline`` lets the turn start
    from a worktree the caller already left dirty, without also claiming the
    ``read_only_resume`` contract: this is not a resume, and asserting it would opt
    the session out of the workspace guard's read-only fast path and so demand that
    ``cwd`` be a git worktree -- which a pip-installed framework never is. A
    backend's explicit external-sandbox bypass remains an OS-isolation choice,
    independent from this logical write policy. No provider fallback occurs here;
    the caller owns runtime resolution.
    """
    import time as _time

    selected_model = model.strip() or str(getattr(getattr(backend, "runtime", None), "model", "")).strip()
    resolved_attempts = (
        int(attempts)
        if attempts is not None
        else int(
            env_setting(
                "FORGE_FUSION_LLM_ATTEMPTS",
                DEFAULT_ATTEMPTS,
                cast=int,
            )
        )
    )
    resolved_turns = max(
        1,
        int(env_setting("FORGE_FUSION_DISCOVERY_TURNS", DEFAULT_DISCOVERY_TURNS, cast=int)),
    )
    resolved_base_delay = (
        float(base_delay_sec)
        if base_delay_sec is not None
        else float(
            env_setting(
                "FORGE_FUSION_LLM_RETRY_BASE_SEC",
                DEFAULT_BASE_DELAY_SEC,
                cast=float,
            )
        )
    )
    resolved_max_delay = (
        float(max_delay_sec)
        if max_delay_sec is not None
        else float(
            env_setting(
                "FORGE_FUSION_LLM_RETRY_MAX_SEC",
                DEFAULT_MAX_DELAY_SEC,
                cast=float,
            )
        )
    )
    resolved_deadline = (
        float(deadline_sec)
        if deadline_sec is not None
        else float(
            env_setting(
                "FORGE_LLM_RETRY_DEADLINE_SEC",
                DEFAULT_DEADLINE_SEC,
                cast=float,
            )
        )
    )
    pause = sleep or _time.sleep
    clock = monotonic or _time.monotonic
    protected = list(protected_files or [])

    def _record_transcript(progress: list[str], text: str) -> None:
        """Persist what the session did, whatever the outcome.

        Written on failure too: the end reason alone cannot tell a session that
        ran out of turns apart from one the gateway dropped, and without the
        transcript a discovery that fails every attempt leaves nothing to
        diagnose from.
        """
        if not log_path:
            return
        with contextlib.suppress(OSError):
            Path(log_path).write_text("\n".join([*progress, text]).strip() + "\n", encoding="utf-8")

    def _fn(prompt: str) -> str:
        started_at = clock()
        last_error = ""
        last_kind = API_ERROR
        progress: list[str] = []
        for attempt in range(1, max(1, resolved_attempts) + 1):
            spec = AgentRunSpec(
                system_prompt=_DISCOVERY_SYSTEM_PROMPT,
                user_prompt=prompt,
                cwd=workdir,
                model=selected_model,
                writable=False,
                timeout_sec=max(1, int(timeout_s)),
                reasoning_effort="high",
                tool_policy=AgentToolPolicy(
                    read=True,
                    search=True,
                    write=False,
                    shell=False,
                    max_turns=resolved_turns,
                ),
                protected_globs=["*"],
                # Not read_only_resume: discovery only needs the "tolerate a dirty
                # worktree" half of that flag, and claiming the resume contract
                # disqualifies this session from the guard's read-only fast path
                # (workspace_guard.is_read_only_session), forcing a git-worktree
                # requirement on a cwd that is routinely a pip install root.
                allow_dirty_baseline=True,
                progress_log=progress,
            )
            try:
                result = _run_agent_discovery_once(
                    backend,
                    spec,
                    timeout_s=timeout_s,
                    protected_files=protected,
                )
                text = str(getattr(result, "text", "") or "").strip()
                end_reason = str(getattr(result, "end_reason", "agent_stopped") or "agent_stopped")
                cut_short = end_reason in {"turn_cap", "timeout"}
                # A cut-short session still answered if it got its proposals out
                # first, and discovery spends turns by design -- it is handed
                # read and search tools precisely so it explores. Discarding
                # parseable proposals because the ceiling was brushed throws away
                # the work and retries into the same ceiling.
                usable = text and (not cut_short or _extract_json_array(text))
                if usable and end_reason != "sdk_error":
                    if cut_short:
                        log.warning(
                            "discovery Agent ended with %s but its proposals parsed; using them",
                            end_reason,
                        )
                    _record_transcript(progress, text)
                    return text
                last_error = (
                    f"{backend.name} returned no final text" if not text else f"{backend.name} ended with {end_reason}"
                )
                last_kind = API_ERROR
            except DiscoverySafetyError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                if is_agent_safety_error(exc):
                    raise DiscoverySafetyError("discovery Agent safety violation: " + str(exc)) from exc
                last_kind = classify_llm_error(exc)
                last_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                if last_kind not in RETRYABLE_KINDS:
                    raise LlmUnavailableError(
                        f"discovery Agent call failed ({last_kind}): {last_error}",
                        kind=last_kind,
                        attempts=attempt,
                    ) from exc
            log.warning(
                "discovery Agent attempt %d/%d failed: %s",
                attempt,
                max(1, resolved_attempts),
                last_error,
            )
            if attempt >= max(1, resolved_attempts):
                break
            elapsed = clock() - started_at
            if resolved_deadline > 0 and elapsed >= resolved_deadline:
                _record_transcript(progress, last_error)
                raise LlmUnavailableError(
                    f"discovery Agent produced no usable answer in {attempt} "
                    f"attempt(s) and {elapsed:.0f}s "
                    f"(deadline {resolved_deadline:.0f}s): {last_error}",
                    kind=last_kind,
                    attempts=attempt,
                )
            pause(
                retry_delay(
                    attempt,
                    base_sec=resolved_base_delay,
                    max_sec=resolved_max_delay,
                )
            )
        _record_transcript(progress, last_error)
        raise LlmUnavailableError(
            f"discovery Agent produced no usable answer in {max(1, resolved_attempts)} attempt(s): {last_error}",
            kind=last_kind,
            attempts=max(1, resolved_attempts),
        )

    return _fn


@dataclass(frozen=True)
class _CompletionMessage:
    """The one field :func:`complete_with_retry` reads off a completion."""

    content: str


@dataclass(frozen=True)
class _CompletionChoice:
    message: _CompletionMessage


@dataclass(frozen=True)
class _Completion:
    """A reply in the chat-completions shape, whatever produced it."""

    choices: list[_CompletionChoice]

    @classmethod
    def of(cls, text: str) -> _Completion:
        """Wrap plain text so every provider path returns the same shape."""
        return cls(choices=[_CompletionChoice(_CompletionMessage(text))])


def _chat_shaped_client(completions: Any) -> Any:
    """Wrap a ``.create()`` in the ``client.chat.completions`` attribute path.

    :func:`complete_with_retry` navigates that path, so each provider adapter is
    reached the same way rather than the retry chain learning about any of them.
    """
    chat = type("_Chat", (), {"completions": completions})()
    return type("_Client", (), {"chat": chat})()


def _anthropic_text(message: Any) -> str:
    """Concatenate the text blocks of a Messages reply, ignoring the rest.

    A thinking-enabled deployment puts a ``thinking`` block first, so reading
    ``content[0]`` would drop the answer and look like an empty completion.
    """
    blocks = getattr(message, "content", None)
    if not isinstance(blocks, list):
        return ""
    return "".join(str(getattr(b, "text", "") or "") for b in blocks if getattr(b, "type", "") == "text")


class _AnthropicChatCompletions:
    """The Messages API behind the chat-completions call shape.

    Discovery's retry chain, failure classification and deadline all live in
    :func:`complete_with_retry`, which speaks to a client. Adapting the protocol
    here keeps both provider lines on that one chain instead of growing a
    second, subtly different one.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def create(self, *, model: str, temperature: float, max_tokens: int, messages: list[dict[str, str]]) -> Any:
        # APIStatusError carries status_code, which classify_llm_error reads
        # before it falls back to scanning the message, so a 401/403/413 stops
        # on the first attempt instead of consuming the retry budget.
        #
        # ``temperature`` is still a Messages API field, but anthropic 1.x
        # dropped it from create()'s typed signature, and that signature has no
        # **kwargs -- passing it named is a TypeError. classify_llm_error reads
        # that as a transient fault, so it burned the whole retry budget on a
        # call that could never succeed. Send it in the body when the installed
        # SDK will not name it.
        payload: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if _anthropic_create_names_temperature(self._client):
            payload["temperature"] = temperature
        else:
            payload["extra_body"] = {"temperature": temperature}
        reply = self._client.messages.create(**payload)
        return _Completion.of(_anthropic_text(reply))


def _anthropic_create_names_temperature(client: Any) -> bool:
    """Whether this SDK's ``messages.create`` takes ``temperature`` by name.

    Defaults to True for anything unintrospectable -- a stub or a ``**kwargs``
    passthrough is happier with the named form, and the caller only needs the
    negative answer to be right.
    """
    try:
        params = inspect.signature(client.messages.create).parameters
    except (AttributeError, TypeError, ValueError):  # pragma: no cover - exotic stubs
        return True
    return "temperature" in params or any(pm.kind is inspect.Parameter.VAR_KEYWORD for pm in params.values())


def _anthropic_client(*, timeout_s: int, verify: bool) -> Any | None:
    """A Messages-protocol client for the Anthropic line, or ``None`` if unset.

    Requires both halves: unlike the Claude CLI, which can run on a Max login
    with neither, this is a direct API call with nowhere to get a default
    endpoint or credential from.

    The credential travels in the header its own kind requires --
    ``ANTHROPIC_API_KEY`` as ``x-api-key``, ``ANTHROPIC_AUTH_TOKEN`` as a bearer
    token -- which the SDK derives from which argument it is passed as. A
    gateway wanting something else again (APIM's subscription key) adds it
    through ``ANTHROPIC_CUSTOM_HEADERS``.
    """
    gateway = resolve_anthropic_gateway()
    key = os.environ.get(gateway.key_env, "").strip() if gateway.key_env else ""
    if not gateway.has_endpoint or not key:
        return None

    # DefaultHttpxClient, not httpx.Client: the SDK validates http_client
    # against the httpx flavour it was built on, and anthropic 1.x moved to
    # httpx2. Handing it the wrong one is a TypeError at construction, which
    # surfaces as "llm setup failed" on every discovery call.
    from anthropic import Anthropic, DefaultHttpxClient

    credential = {"auth_token": key} if gateway.key_env == "ANTHROPIC_AUTH_TOKEN" else {"api_key": key}
    sdk = Anthropic(
        base_url=normalize_anthropic_base_url(gateway.base_url),
        default_headers=gateway.headers or None,
        http_client=DefaultHttpxClient(verify=verify, timeout=timeout_s),
        # Discovery owns the retry policy: complete_with_retry classifies each
        # failure and enforces a wall-clock deadline, and a second silent layer
        # underneath it would multiply the attempts and blow through that bound.
        max_retries=0,
        **credential,
    )
    return _chat_shaped_client(_AnthropicChatCompletions(sdk))


def default_llm_fn(
    *,
    model: str = "claude-opus-4-7",
    timeout_s: int = 900,
    log_path: str = "",
    max_tokens: Optional[int] = None,
    gpu: str = "",  # gpu unused (text call); kept for call-site compat
) -> LlmFn:
    """Legacy bare-completion adapter retained for direct API compatibility.

    The forge-fuse CLI does not use this path: it constructs one registered
    Agent backend and injects :func:`registered_agent_llm_fn`. Existing callers
    that import this helper continue to get the historical OpenAI-compatible
    completion behavior.

    Discovery only READS (the source and retrieved operator evidence are embedded
    in the prompt) and RETURNS JSON, so a single chat completion suffices.
    Endpoint, credential and headers come from the OpenAI line via
    :func:`~kernelforge.llm.resolve_openai_gateway`, and ``ANTHROPIC_SKIP_TLS_VERIFY``
    / ``NODE_TLS_REJECT_UNAUTHORIZED`` are honored for the gateway's
    self-signed cert.

    Raises :class:`~kernelforge.fusion.llm_failure.LlmUnavailableError` when the model
    was never reached — an unconfigured gateway, an unusable client, or a
    gateway that kept failing. It must never return ``""`` for those, because
    the caller cannot tell that apart from the model proposing nothing, and the
    run would publish ``no_opportunity`` for a model it never analyzed.

    Retry budget is tunable without a redeploy via ``FORGE_FUSION_LLM_ATTEMPTS``,
    ``FORGE_FUSION_LLM_RETRY_BASE_SEC`` and ``FORGE_FUSION_LLM_RETRY_MAX_SEC``.
    """

    resolved_max_tokens = (
        int(max_tokens)
        if max_tokens is not None
        else int(env_setting("FORGE_FUSION_LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS, cast=int))
    )

    def _fn(prompt: str) -> str:
        skip_tls = (
            os.environ.get("ANTHROPIC_SKIP_TLS_VERIFY", "").strip().lower() in ("1", "true", "yes")
            or os.environ.get("NODE_TLS_REJECT_UNAUTHORIZED", "").strip() == "0"
        )
        gateway = resolve_openai_gateway()
        key = os.environ.get(gateway.key_env, "").strip() if gateway.key_env else ""
        try:
            if gateway.is_complete() and key:
                # APIM gateways (e.g. AMD) enforce an Ocp-Apim-Subscription-Key
                # header the OpenAI SDK never sends from api_key; without
                # default_headers the gateway 401s "missing subscription key".
                # These come from the resolved provider, so the other side's
                # headers can never leak onto this endpoint.
                # DefaultHttpxClient for the same reason as the Anthropic leg
                # above: the SDK type-checks http_client against its own httpx.
                from openai import DefaultHttpxClient, OpenAI

                client_kwargs: dict[str, Any] = {
                    "base_url": gateway.base_url,
                    "api_key": key,
                    "http_client": DefaultHttpxClient(verify=not skip_tls, timeout=timeout_s),
                }
                if gateway.headers:
                    client_kwargs["default_headers"] = gateway.headers
                client: Any = OpenAI(**client_kwargs)
            else:
                client = _anthropic_client(timeout_s=timeout_s, verify=not skip_tls)
        except Exception as exc:  # noqa: BLE001 — client construction failure.
            raise LlmUnavailableError(
                f"discovery llm_fn setup failed: {type(exc).__name__}: {str(exc)[:240]}",
                kind=classify_llm_error(exc),
            ) from exc
        if client is None:
            raise LlmUnavailableError(
                "discovery llm_fn: no LLM gateway configured (needs either "
                "OPENAI_BASE_URL + OPENAI_API_KEY for the OpenAI-compatible "
                "protocol, or ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY for the "
                "Anthropic Messages protocol)",
                kind=NOT_CONFIGURED,
            )

        out = complete_with_retry(
            client,
            prompt,
            model=model,
            max_tokens=resolved_max_tokens,
            attempts=int(env_setting("FORGE_FUSION_LLM_ATTEMPTS", DEFAULT_ATTEMPTS, cast=int)),
            base_delay_sec=float(env_setting("FORGE_FUSION_LLM_RETRY_BASE_SEC", DEFAULT_BASE_DELAY_SEC, cast=float)),
            max_delay_sec=float(env_setting("FORGE_FUSION_LLM_RETRY_MAX_SEC", DEFAULT_MAX_DELAY_SEC, cast=float)),
        )
        if log_path:
            with contextlib.suppress(OSError):
                Path(log_path).write_text(out, encoding="utf-8")
        return out

    return _fn


def discover_recipes(
    diagnosis: Diagnosis,
    *,
    model_type: str,
    framework: str,
    source_file: str,
    shapes: dict[str, Any],
    trace_path: str,
    llm_fn: Optional[LlmFn] = None,
    max_fusions: Optional[int] = None,
    top_kernels: int = 15,
    knowledge_root: str | Path | None = None,
    pass_probe: Optional[Callable[[str], PassState]] = None,
    framework_root: str = "",
) -> list[Recipe]:
    """LLM-autonomous discovery: propose fusible chains from the trace + source.

    Returns an empty list when the diagnosis is not a candidate, the source cannot
    be read, or the LLM proposes nothing parseable. The CLI injects
    :func:`registered_agent_llm_fn`; the legacy default remains only for direct
    callers that omit ``llm_fn``.

    An empty list means discovery looked and found nothing. When it could not
    look at all, ``llm_fn`` raises
    :class:`~kernelforge.fusion.llm_failure.LlmUnavailableError` and that propagates:
    the caller has to record an unreachable model as such, not as a verdict.
    """
    if not diagnosis.is_candidate:
        return []
    try:
        source_text = Path(source_file).read_text(encoding="utf-8") if source_file else ""
    except OSError:
        source_text = ""
    if not source_text:
        log.warning("discovery: model source unreadable (%s); cannot self-discover", source_file)
        return []
    hot = hot_kernels_from_trace(trace_path, top_n=top_kernels)
    ordered_boundaries = ordered_fusion_boundaries_from_trace(trace_path)
    # Hot kernels and the diagnosis categories are folded in as a second evidence
    # source: ordered boundaries need repeats to exist, so a short trace would
    # otherwise leave retrieval with nothing to match against.
    existing_operator_hints = existing_operator_hints_from_knowledge(
        knowledge_root,
        ordered_boundaries,
        fallback_categories=list(diagnosis.dominant_categories),
        fallback_kernel_names=kernel_names_from_trace(trace_path),
    )
    prompt = build_discovery_prompt(
        model_type=model_type,
        framework=framework,
        source_text=source_text,
        diagnosis=diagnosis,
        hot_kernels=hot,
        shapes=shapes,
        max_fusions=_resolve_max_fusions(max_fusions),
        ordered_boundaries=ordered_boundaries,
        existing_operator_hints=existing_operator_hints,
    )
    fn = llm_fn or default_llm_fn()
    raw = fn(prompt)
    recipes = parse_discovered_recipes(
        raw,
        model_type=model_type,
        framework=framework,
        source_file=source_file,
        shapes=shapes,
        category_shares=diagnosis.category_shares,
        pass_probe=pass_probe,
        framework_root=framework_root,
    )
    log.info(
        "discovery proposed %d fusion(s): %s",
        len(recipes),
        ", ".join(r.pattern_id for r in recipes),
    )
    return recipes
