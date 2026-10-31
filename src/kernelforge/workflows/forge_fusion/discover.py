# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 2 (LLM-autonomous discovery): find fusible op chains from trace + source."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from hyperloom_common.coerce import to_float
from kernelforge.agent_backends.base import AgentRunSpec, AgentToolPolicy, watchdog_timeout_sec
from kernelforge.agent_backends.session_resume import is_api_failure
from kernelforge.config.resources import resource_path

from .diagnose import LAUNCH_BOUND_CATEGORIES, _load_trace_events, categories_in_text, categorize_kernel_name
from .discovery_prompt import DISCOVERY_PROMPT_TEMPLATE
from .llm_failure import (
    API_ERROR,
    DEFAULT_ATTEMPTS,
    DEFAULT_BASE_DELAY_SEC,
    DEFAULT_DEADLINE_SEC,
    DEFAULT_MAX_DELAY_SEC,
    RETRYABLE_KINDS,
    LlmUnavailableError,
    classify_llm_error,
    env_setting,
    is_agent_safety_error,
    retry_delay,
)
from .locate import (
    FrameworkInstall,
    _compile_pass_recipe,
    _read_source,
    _unclaimable_note,
    covered_by_vllm_compile_pass,
    out_of_scope_terms,
    rank_recipes,
    vllm_compile_pass_state,
)
from .models import Diagnosis, Recipe
from .vllm_passes import resolve_target_runtime

log = logging.getLogger("forge_fusion")

LlmFn = Callable[[str], str]  # prompt -> raw model text (expected to contain JSON)


@dataclass(frozen=True)
class DiscoveryTarget:
    """What discovery proposes fusions for: the model, the framework install, the source file, shapes and scope."""

    model_type: str
    install: FrameworkInstall
    source_file: str
    shapes: dict[str, Any]
    repo_scope: bool = False
    repo_root: str = ""


@dataclass(frozen=True)
class DiscoveryEvidence:
    """The runtime evidence a discovery prompt cites: diagnosis, hot kernels, fusion boundaries, known operators."""

    diagnosis: Diagnosis
    hot_kernels: list[dict[str, Any]]
    ordered_boundaries: Optional[list[dict[str, Any]]] = None
    existing_operator_hints: Optional[list[dict[str, str]]] = None


@dataclass(frozen=True)
class DiscoveryPrompting:
    """How many fusions discovery asks for, and the operator-knowledge corpus its hints come from."""

    max_fusions: Optional[int] = None
    knowledge_root: str | Path | None = None


@dataclass(frozen=True)
class DiscoverySession:
    """One read-only discovery session: model, timeout, transcript, working dir, guarded files, usage accumulator."""

    model: str = ""
    timeout_s: int = 900
    log_path: str = ""
    workdir: str = "."
    protected_files: Sequence[str] = ()
    usage: Any = None


@dataclass(frozen=True)
class DiscoveryRetry:
    """Overrides for how many attempts a discovery call makes and how it backs off between them."""

    attempts: Optional[int] = None
    base_delay_sec: Optional[float] = None
    max_delay_sec: Optional[float] = None
    sleep: Optional[Callable[[float], Any]] = None


# Each proposed fusion costs discovery tokens plus one authoring subprocess and validation pass, and that cost is paid
# before the E2E gate can reject it.
_DEFAULT_MAX_FUSIONS = 4
# Hot kernels quoted in the discovery prompt.
_TOP_KERNELS = 15
# Operator hints offered, and the share of the best score a hint must reach.
_OPERATOR_HINT_LIMIT = 12
_OPERATOR_HINT_MIN_SCORE_RATIO = 0.25

# Discovery is handed read and search tools, and the first tool call ends the turn.
DEFAULT_DISCOVERY_TURNS = 60

# Heavy kernels a fusible chain runs between; everything else is launch-bound tail.
COMPUTE_CATEGORIES: frozenset[str] = frozenset({"gemm", "attention", "conv", "moe"})


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
    """Top GPU kernels from a kineto trace by total-duration share."""
    agg: dict[str, list[float]] = {}
    total = 0.0
    for ev in _load_trace_events(trace_path):
        if ev.get("cat") != "kernel":
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
    for name, (dur, count) in agg.items():
        cat = categorize_kernel_name(name)
        if launch_bound_only and cat in COMPUTE_CATEGORIES:
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


def kernel_names_from_trace(trace_path: str | Path, *, top_n: int = 40) -> list[str]:
    """Distinct kernel names ranked by total duration, compute kernels included."""
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


def stream_ordered_kernels(
    trace_path: str | Path,
) -> tuple[dict[tuple[Any, Any], list[dict[str, Any]]], float]:
    """Kernel events bucketed by GPU stream, each bucket ordered by timestamp.

    Returns the buckets and the summed kernel duration. Execution order only holds
    within one stream, so every adjacency question has to be asked per bucket.
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
    for bucket in streams.values():
        bucket.sort(key=lambda item: item["ts"])
    return streams, total_kernel_us


def ordered_fusion_boundaries_from_trace(
    trace_path: str | Path,
    *,
    top_n: int = 16,
    max_chain_len: int = 8,
    min_repeats: int = 2,
) -> list[dict[str, Any]]:
    """Recover repeated compute-to-compute fusion boundaries from stream order."""
    streams, total_kernel_us = stream_ordered_kernels(trace_path)

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
        elif categories[0] in COMPUTE_CATEGORIES:
            boundary_kind = "compute_boundary"
        else:
            boundary_kind = "vertical"
        # The trailing kernel is the NEXT compute anchor.
        terminal_compute = categories[-1] if categories[-1] in COMPUTE_CATEGORIES else ""
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

    for ordered in streams.values():
        start_index: Optional[int] = None
        for index, event in enumerate(ordered):
            if event["category"] not in COMPUTE_CATEGORIES:
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
        # Ranking heuristic only, NOT a true fraction of GPU time: a kernel that sits between two compute anchors
        # belongs to two overlapping segments, so its duration is counted once per segment and shares can sum above 1.
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
    """Whole-word tokens of ``text``."""
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

# Parsed knowledge documents, keyed by (path, mtime_ns, size) so an unchanged knowledge base is not re-read and
# re-parsed on every discovery run.
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
    fallback_categories: Optional[list[str]] = None,
    fallback_kernel_names: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Retrieve existing ROCm operator names using observed runtime semantics."""
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
    boundary_terms = [_semantic_terms(source) for source in evidence_sources]
    if not any(boundary_terms):
        return []

    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for path, entries in _knowledge_entries(root):
        for entry in entries:
            overlap = max(
                len(entry["operator_terms"] & terms) * 100.0 + len(terms & entry["paragraph_tokens"]) * 10.0
                for terms in boundary_terms
            )
            score = overlap - entry["paragraph_length"] / 500.0
            if entry["is_declared"]:
                score += 500.0
            if score < 20.0:
                continue
            row = {
                "operator": entry["operator"],
                "path": path.relative_to(root).as_posix(),
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
    # Pre-trim marginal matches relative to the best one, so a long tail of weak hints cannot pad the prompt and
    # inflate downstream authoring attempts.
    cutoff = ranked[0][0] * _OPERATOR_HINT_MIN_SCORE_RATIO
    return [row for score, row in ranked[:_OPERATOR_HINT_LIMIT] if score >= cutoff]


# Terms a proposal may declare in ``ops``.
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

# HOW the kernel is built, not what it computes: precision, architecture variant, and where in the model it sits.
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
    """A proposal's declaration for ``field``, normalized; ``[]`` when unusable."""
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


_SCOPE_SINGLE = """- SCOPE — the single hardest constraint, and the one that wastes a whole run when
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
  never one entry spanning both."""

_SCOPE_REPO = """- SCOPE — the WHOLE repository is in scope and editable, and a fusion may span as
  many files as it needs. Name the file holding the call site you would replace in
  "source_file", and every OTHER existing file that has to change in
  "additional_files". There is no single-file rule here: do not shrink a chain to
  fit one file.
- FIND THE CODE, do not guess it. Nothing is embedded in this prompt; you have read
  and search tools and a repository. The entry point named below is where to START,
  not the answer: a model file routinely reaches the anchored work through one
  opaque call (`attn_backend.forward_xxx(...)`, `self.indexer(...)`, a dispatch
  through an env gate) whose operands are not local names there. Grep for the ops
  in the anchor evidence, follow the imports and calls until you reach the code that
  actually issues those kernels, and propose the fusion THERE.
- Every path you return must be a file that EXISTS in this repository, spelled as it
  is on disk (repo-relative or absolute). A proposal whose "source_file" does not
  resolve is discarded rather than retargeted, so open each file before naming it.
- Name the exact call site you would replace, and account for every tensor your
  kernel takes as input: each one must be a local name at that call site, or be
  produced in one of the files you listed in "additional_files". Work the framework
  performs below your call is reachable ONLY if you list the file performing it.
- One entry is ONE fusion, delivered as one patch, however many files it touches.
  Two genuinely independent chains are two entries."""

_FUSION_CONSTRAINTS_TAIL = """- ROCm-native: it will be authored as a Triton kernel; do NOT propose reusing a
  framework CUDA-only fused op.
- Existing AITER/CK/HIP/Triton operators listed above are allowed and preferred when
  their semantics, dtype, shape, and cache layout match.
- Do NOT change what a library GEMM dispatches to. A tuned GEMM (flydsl / aiter /
  hipBLASLt / CK) selects its kernel from the (dtype, layout, shape) of the call,
  so asking that same call for a different output dtype (`out_dtype=`, `otype=`),
  a transposed operand, or an `out=` buffer of another type drops it off the tuned
  table onto a different, UNTUNED solution. The displaced kernel is typically
  several times the size of the elementwise op being fused, so the substitution
  costs far more than the fusion saves even though the fused call site alone
  benchmarks faster. When the op you are fusing sits between a library GEMM and
  its consumer, leave the GEMM call byte-identical and fuse into the PROLOGUE of
  the CONSUMER instead -- consumers are frequently templated on their input dtype
  already and upcast on load, which removes the same launch for free. A GEMM
  epilogue is in scope only when the GEMM is one YOU author.
- The correctness reference must be the REAL eager op imported from this source
  (say which symbol to import), never a re-derivation."""


def fusion_constraints(repo_scope: bool = False) -> str:
    """The rules that decide whether a proposal can be wired at all.

    Scope is the only rule that changes with how much source the run offered: with
    one file the model must stay inside it, and with the whole repository it must
    go looking for the file that owns the chain.
    """
    scope = _SCOPE_REPO if repo_scope else _SCOPE_SINGLE
    chain_rule = (
        "- Must be a real contiguous chain in the repository (name the exact files, functions/methods)."
        if repo_scope
        else "- Must be a real contiguous chain in this source (name the exact functions/methods)."
    )
    return f"""Constraints for each proposed fusion:
{chain_rule}
{scope}
{_FUSION_CONSTRAINTS_TAIL}"""


def render_source_files(source_files: Sequence[str], *, model_type: str, framework: str) -> str:
    """Embed each in-scope file under its own path heading.

    The path is the label the model answers with in ``source_file``, so it is
    printed exactly as it will have to be matched back.
    """
    blocks = [f"### {path}\n```python\n{text}\n```" for path in source_files if (text := _read_source(path))]
    if not blocks:
        return ""
    return f"## Model source (`{model_type}` in {framework})\n" + "\n\n".join(blocks)


def render_repo_scope_brief(
    repo_root: str,
    entry_files: Sequence[str],
    *,
    model_type: str,
    framework: str,
) -> str:
    """Hand discovery the repository instead of one file's embedded text.

    Repo scope exists because the file that owns the fusible chain cannot be
    derived from the model config: it is reached through a call, not through a
    naming convention. Embedding a guessed shortlist would only move that guess
    earlier in the pipeline, so the agent is given the tree and its own read and
    search tools and is expected to locate the code itself.
    """
    lines = [
        f"## The framework repository (`{model_type}` in {framework})",
        f"Repository root: {repo_root or '(the working directory of this session)'}",
        "",
        "No source is embedded in this prompt. Open whatever you need with your read",
        "and search tools; every file in this repository is in scope and editable.",
    ]
    known = [str(path) for path in entry_files if path]
    if known:
        lines += [
            "",
            "Entry point(s) resolved for this model -- where to START, not the answer:",
            *(f"  {path}" for path in known),
        ]
    return "\n".join(lines)


def _output_schema_block(model_type: str, *, repo_scope: bool = False) -> str:
    """The JSON contract every discovery prompt asks the model to answer in."""
    source_field = (
        (
            '  "source_file": "<exact on-disk path of the file holding the call site '
            'you would replace>",\n'
            '  "additional_files": [<exact on-disk path of every OTHER existing file '
            "this fusion must also edit; [] when the call-site file is enough. Do NOT "
            "list the new fused-kernel module here -- it does not exist yet>],\n"
        )
        if repo_scope
        else ""
    )
    return f"""## Output — a single JSON array (and nothing after it). Be TERSE to fit the
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
{source_field}  "fusion_math": "<what the fused kernel computes, precisely>",
  "eager_reference": "<which real symbol(s) to import + call for the parity ref>",
  "candidate_kind": "<integration|new_fusion|replacement>",
  "existing_operator": "<operator name when candidate_kind=integration, else empty>",
  "priority": <0.0-1.0 by expected launch-bound time saved>,
  "rationale": "<why this chain, tied to the hot kernels above>"}}"""


def build_discovery_prompt(
    target: DiscoveryTarget,
    source_text: str,
    evidence: DiscoveryEvidence,
    max_fusions: int = _DEFAULT_MAX_FUSIONS,
) -> str:
    """Assemble the discovery prompt from runtime, source, and operator evidence.

    Repo scope embeds no source at all and points the agent at the repository
    instead, naming the target's source file as the entry point to start from.
    """
    model_type, framework, shapes, repo_scope = (
        target.model_type,
        target.install.framework,
        target.shapes,
        target.repo_scope,
    )
    diagnosis, hot_kernels = evidence.diagnosis, evidence.hot_kernels
    if repo_scope:
        source_block = render_repo_scope_brief(
            target.repo_root,
            [target.source_file] if target.source_file else [],
            model_type=model_type,
            framework=framework,
        )
    else:
        source_block = f"## Model source (`{model_type}` in {framework})\n```python\n{source_text}\n```"
    read_verb = (
        "Explore the repository described below and identify"
        if repo_scope
        else "Read the model source below and identify"
    )
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
        # The trailing compute kernel proves adjacency but is not fusable, so the fusable span is spelled out to keep
        # the proposed chain from swallowing the attention (or other compute) kernel it feeds.
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

    boundary_lines = "\n".join(boundary_line(boundary) for boundary in (evidence.ordered_boundaries or []))

    def operator_line(hint: dict[str, Any]) -> str:
        score = hint.get("score")
        score_text = f" score={float(score):.1f}" if score is not None else ""
        return f"  - `{hint['operator']}` ({hint['path']}){score_text}: {hint['evidence']}"

    operator_lines = "\n".join(operator_line(hint) for hint in (evidence.existing_operator_hints or []))
    dom = ", ".join(diagnosis.dominant_categories)
    return DISCOVERY_PROMPT_TEMPLATE.format(
        framework=framework,
        model_type=model_type,
        diagnosis=diagnosis,
        lb=lb,
        hot_lines=hot_lines,
        dom=dom,
        shapes=shapes,
        boundary_lines=boundary_lines or "  - none",
        operator_lines=operator_lines or "  - none found",
        read_verb=read_verb,
        max_fusions=max_fusions,
        fusion_constraints=fusion_constraints(repo_scope=repo_scope),
        output_schema_block=_output_schema_block(model_type, repo_scope=repo_scope),
        source_block=source_block,
    )


def _salvage_objects(text: str) -> list[dict[str, Any]]:
    """Recover every complete top-level ``{...}`` object from (possibly truncated) text, ignoring braces inside strings.

    The object a cut left half-written comes last, once repaired, unless it repeats one already recovered.
    """
    out: list[dict[str, Any]] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    commas: list[int] = []
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
                commas = []
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    obj = json.loads(text[start : i + 1])
                    if isinstance(obj, dict):
                        out.append(obj)
        elif ch == "," and depth == 1:
            commas.append(i - start)
    repaired = _repair_truncated_object(text[start:], depth, in_str, commas) if depth else None
    if repaired is not None and repaired not in out:
        out.append(repaired)
    return out


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Pull JSON fusion proposals out of model text."""
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
    # Fallback: salvage complete objects from a truncated/unclosed array, then the object the cut left half-written --
    # with three quarters of responses arriving truncated, that last object is often the only one there is.
    salvaged = _salvage_objects(text)
    if not salvaged:
        log.warning(
            "discovery: no JSON proposals parsed from %d chars of model text; "
            "this is a parse failure, NOT a no_opportunity result",
            len(text),
        )
    return salvaged


# A repaired object has to carry enough of the fusion description to act on.
_REPAIRED_REQUIRED_ANY = ("op_chain", "fusion_math")


def _repair_truncated_object(fragment: str, depth: int, in_str: bool, commas: list[int]) -> dict[str, Any] | None:
    """Recover the proposal that a cut-off response left half-written.

    ``fragment`` runs from the object's opening brace to the cut, with ``depth`` braces and maybe a string still
    open; ``commas`` are the offsets of its top-level separators, where a field before the cut ends complete.
    """
    attempts = [fragment + ('"' if in_str else "") + "}" * depth]
    attempts.extend(fragment[:mark] + "}" for mark in reversed(commas))
    for attempt in attempts:
        try:
            obj = json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict) or not str(obj.get("name") or "").strip():
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


def _field_text(item: dict[str, Any], key: str, default: str = "") -> str:
    """A proposal field as text, ``default`` when the model left it empty or null."""
    return str(item.get(key) or default)


def _trace_lacks(categories: list[str], category_shares: dict[str, float] | None) -> bool:
    """Whether the trace measured category shares but none of ``categories``."""
    if not category_shares or not categories:
        return False
    return {category for category, share in category_shares.items() if float(share) > 0.0}.isdisjoint(categories)


def _as_sequence(value: Any) -> list[Any]:
    """A proposal field that may arrive as one string or a list of them."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def resolve_repo_file(value: Any, repo_root: str) -> str:
    """Resolve a repo-scope proposal's claimed path to a real file, or ``""``.

    Repo scope removes the offered-file list that
    :func:`resolve_proposed_source_file` matches against, so existing on disk and
    living inside the repository is the whole check -- and it is a HARD one.
    Falling back to the model file the way single-file discovery does would
    reintroduce the silent retarget that repo scope exists to remove: the caller
    would get a recipe pointing at a file the model never proposed.
    """
    claimed = str(value or "").strip()
    if not claimed:
        return ""
    root: Optional[Path] = None
    if repo_root:
        with contextlib.suppress(OSError):
            root = Path(repo_root).expanduser().resolve()
    candidates = [Path(claimed).expanduser()]
    if root is not None and not candidates[0].is_absolute():
        candidates.insert(0, root / candidates[0])
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        if root is not None and not resolved.is_relative_to(root):
            log.warning("discovery: ignoring %s, which is outside the framework repo %s", resolved, root)
            continue
        return str(resolved)
    return ""


def resolve_proposed_source_file(value: Any, in_scope_files: Sequence[str], default: str) -> str:
    """Map a proposal's claimed call-site file onto one the run actually offered.

    The model is asked to copy a path back verbatim, and mostly does; a trailing
    component still identifies the file unambiguously when it does not. Anything
    that matches nothing falls back to the primary file rather than inventing a
    target the loop cannot track.
    """
    claimed = str(value or "").strip()
    if not claimed or not in_scope_files:
        return default
    for path in in_scope_files:
        if claimed == path:
            return path
    claimed_parts = Path(claimed).parts
    for path in in_scope_files:
        parts = Path(path).parts
        if claimed_parts and parts[-len(claimed_parts) :] == claimed_parts:
            return path
    for path in in_scope_files:
        if Path(path).name == Path(claimed).name:
            return path
    log.warning(
        "discovery: proposed source_file %r matches no in-scope file; using %s",
        claimed[:120],
        Path(default).name or default,
    )
    return default


def _proposal_edit_files(item: dict[str, Any], target: DiscoveryTarget, name: str) -> list[str]:
    """The files a proposal wires itself into, its call site first; ``[]`` drops the proposal.

    Single-file discovery maps the claimed file onto the one source file the prompt embedded. Repo scope embedded
    nothing, so each proposal names its own files, and one whose call site does not resolve is dropped rather than
    retargeted onto the target's source file.
    """
    if not target.repo_scope:
        scope_files = [target.source_file] if target.source_file else []
        return [resolve_proposed_source_file(item.get("source_file"), scope_files, target.source_file)]
    call_site = resolve_repo_file(item.get("source_file"), target.repo_root)
    if not call_site:
        log.warning(
            "discovery: dropping %s -- its source_file %r is not a file in %s",
            name,
            str(item.get("source_file") or "")[:200],
            target.repo_root or "the framework repo",
        )
        return []
    files = [call_site]
    for value in _as_sequence(item.get("additional_files")):
        resolved = resolve_repo_file(value, target.repo_root)
        if not resolved:
            log.warning(
                "discovery: %s listed an additional file that does not resolve (%r); ignoring it",
                name,
                str(value)[:200],
            )
        elif resolved not in files:
            files.append(resolved)
    return files


def _scope_gate_drops(name: str, files: list[str], terms: list[str], explicit_target: bool) -> bool:
    """Whether the scope gate drops a proposal declaring ``terms`` that none of its edit ``files`` perform.

    A fusion is wired by replacing a call site in the files the proposal claims, so one claiming ops those files do
    not perform is unwireable no matter how good the kernel is. The files are the ones the model was shown or named,
    never a file the chain was not claimed to be in. The gate recognizes nine coarse terms, so it can reject a
    wireable chain; an operator-named target is kept with a warning, because discarding what was explicitly asked
    for hides that.
    """
    outside = out_of_scope_terms("\n".join(_read_source(path) for path in files), terms)
    if outside:
        log.log(
            logging.WARNING if explicit_target else logging.INFO,
            "discovery: %s %s (%s not performed in %s -- the fusion crosses "
            "a module boundary and has no wireable call site there)",
            "keeping operator-named" if explicit_target else "dropping",
            name,
            ",".join(outside),
            _file_names(files),
        )
    return bool(outside) and not explicit_target


def _file_names(paths: Sequence[str]) -> str:
    """``paths`` by file name, for a log line."""
    return ", ".join(Path(p).name or p for p in paths)


def parse_discovered_recipes(
    text: str,
    target: DiscoveryTarget,
    *,
    category_shares: dict[str, float] | None = None,
    explicit_target: bool = False,
) -> list[Recipe]:
    """Convert the LLM's JSON proposals into ranked :class:`Recipe` objects.

    ``explicit_target``: an operator named the target, so the scope gate keeps, with a warning, what it would drop.
    """
    runtime = resolve_target_runtime(target.install.framework, framework_root=target.install.framework_root)
    out: list[Recipe] = []
    for i, item in enumerate(_extract_json_array(text)):
        name = _field_text(item, "name", f"discovered_{i + 1}").strip()
        op_chain = _field_text(item, "op_chain")
        fusion_math = _field_text(item, "fusion_math", op_chain)
        # The fusion-DEFINING fields (name / op-chain / math) -- NOT the free-prose rationale or grep anchors, which
        # can mention an op in passing and would attach a category the fusion does not actually involve.
        defining_text = " ".join([name, op_chain, fusion_math])
        # What this fusion IS, as opposed to how this run described it.
        declared, traits = declared_ops(item), declared_traits(item)
        # The op categories: from the declaration when there is one, else from the prose.
        categories = categories_in_text(" ".join(declared or [defining_text]))
        # Hallucination gate FIRST: a proposal whose ops the trace never measured has nothing to remove, and that is
        # true whether we would author it or claim a framework pass for it.
        if _trace_lacks(categories, category_shares):
            log.info("discovery: dropping %s (proposed ops %s absent from the trace)", name, ",".join(categories))
            continue
        edit_files = _proposal_edit_files(item, target, name)
        if not edit_files or _scope_gate_drops(name, edit_files, [*declared, *traits], explicit_target):
            continue
        # Compile-pass gate: never author a chain vLLM fuses at compile time. It keys on precision and variant words
        # (quant, fp8, mla, kvcache) that no op category expresses, so it needs more than ``declared``.
        pass_name = covered_by_vllm_compile_pass(
            matched_categories=[],
            text=" ".join([*declared, *(traits or [defining_text])]),
            framework=target.install.framework,
        )
        state = vllm_compile_pass_state(pass_name, probe=target.install.pass_probe, runtime=runtime)
        if state is not None and state.enabled is True:
            log.info("discovery: dropping %s, vLLM compile pass %s already fuses it", name, state.flag or pass_name)
            continue  # the framework really does fuse this: authoring is a no-op
        # Absent / undecidable / pinned off: the framework is NOT fusing this for us, so keep the proposal as
        # authoring work and record why.
        pass_note = _unclaimable_note(state)
        if pass_note:
            log.info("compile pass not claimed for %s: %s", name, pass_note)
        existing_operator = _field_text(item, "existing_operator").strip()
        candidate_kind = _field_text(item, "candidate_kind").strip().lower()
        # ``integration`` is only meaningful with a named operator: the authoring prompt injects its "benchmark the
        # existing operator first" block only when both fields are set.
        if candidate_kind not in {"new_fusion", "replacement"}:
            candidate_kind = "integration" if existing_operator else "new_fusion"
        recipe = Recipe(
            pattern_id=f"llm:{name}",
            description=_field_text(item, "rationale", op_chain or name)[:300],
            env_flag=_norm_env_flag(_field_text(item, "env_flag", "FUSED"), target.model_type),
            source_file=edit_files[0],
            extra_files=edit_files[1:],
            source_hints=[str(a) for a in _as_sequence(item.get("source_anchors") or [])],
            fusion_math=fusion_math,
            eager_reference_hint=_field_text(item, "eager_reference"),
            shapes=target.shapes,
            matched_categories=categories,
            trigger_share=to_float(item.get("priority"), max(0.1, 1.0 - 0.1 * i)),  # preserve LLM order when absent
            source_confirmed=True,  # the LLM read the real source to propose it
            candidate_kind=candidate_kind,
            existing_operator=existing_operator,
            compile_pass_note=pass_note,
        )
        if state is not None and state.claimable:
            recipe = _compile_pass_recipe(recipe, state, "chain", recipe.description[:200])
        elif edit_files != [target.source_file]:
            log.info("discovery: %s wires into %s", name, _file_names(edit_files))
        out.append(recipe)
    return rank_recipes(sorted(out, key=lambda r: r.trigger_share, reverse=True))


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
    usage: Any = None,
) -> Any:
    """Run one SDK turn and enforce the provider-neutral source invariant."""
    snapshot = _protected_file_snapshot(protected_files)

    async def _run() -> Any:
        return await asyncio.wait_for(
            backend.run(spec, usage=usage),
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


def _override_or_env(override: Any, name: str, default: float, cast: Callable[[Any], Any]) -> Any:
    """An explicit override, else the environment setting ``name`` with its default."""
    return cast(override) if override is not None else cast(env_setting(name, default, cast=cast))


def _discovery_run_spec(
    prompt: str, session: DiscoverySession, model: str, max_turns: int, progress: list[str]
) -> AgentRunSpec:
    """One read-only discovery turn in the session's working directory, logging into ``progress``."""
    return AgentRunSpec(
        system_prompt=_DISCOVERY_SYSTEM_PROMPT,
        user_prompt=prompt,
        role="fusion discovery",
        cwd=session.workdir,
        model=model,
        writable=False,
        timeout_sec=max(1, int(session.timeout_s)),
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=False,
            shell=False,
            max_turns=max_turns,
        ),
        protected_globs=["*"],
        # Not read_only_resume: discovery only needs the "tolerate a dirty worktree" half of that flag, and
        # claiming the resume contract disqualifies this session from the guard's read-only fast path
        # (workspace_guard.is_read_only_session), forcing a git-worktree requirement on a cwd that is
        # routinely a pip install root.
        allow_dirty_baseline=True,
        progress_log=progress,
    )


def registered_agent_llm_fn(
    backend: Any,
    session: DiscoverySession = DiscoverySession(),
    retry: DiscoveryRetry = DiscoveryRetry(),
) -> LlmFn:
    """Adapt one registered Agent backend into discovery's text interface."""
    import time as _time

    timeout_s, log_path = session.timeout_s, session.log_path
    selected_model = session.model.strip() or str(getattr(getattr(backend, "runtime", None), "model", "")).strip()
    resolved_attempts = _override_or_env(retry.attempts, "FORGE_FUSION_LLM_ATTEMPTS", DEFAULT_ATTEMPTS, int)
    resolved_turns = max(1, int(env_setting("FORGE_FUSION_DISCOVERY_TURNS", DEFAULT_DISCOVERY_TURNS, cast=int)))
    resolved_base_delay = _override_or_env(
        retry.base_delay_sec, "FORGE_FUSION_LLM_RETRY_BASE_SEC", DEFAULT_BASE_DELAY_SEC, float
    )
    resolved_max_delay = _override_or_env(
        retry.max_delay_sec, "FORGE_FUSION_LLM_RETRY_MAX_SEC", DEFAULT_MAX_DELAY_SEC, float
    )
    resolved_deadline = float(env_setting("FORGE_LLM_RETRY_DEADLINE_SEC", DEFAULT_DEADLINE_SEC, cast=float))
    pause = retry.sleep or _time.sleep
    clock = _time.monotonic
    protected = list(session.protected_files or [])

    def _record_transcript(progress: list[str], text: str) -> None:
        """Persist what the session did, whatever the outcome."""
        if not log_path:
            return
        with contextlib.suppress(OSError):
            Path(log_path).write_text("\n".join([*progress, text]).strip() + "\n", encoding="utf-8")

    def _back_off(attempt: int, started_at: float, progress: list[str], last_error: str, last_kind: str) -> None:
        """Sleep before the next attempt, or give up once the retry deadline has passed."""
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

    def _fn(prompt: str) -> str:
        started_at = clock()
        last_error = ""
        last_kind = API_ERROR
        progress: list[str] = []
        for attempt in range(1, max(1, resolved_attempts) + 1):
            spec = _discovery_run_spec(prompt, session, selected_model, resolved_turns, progress)
            try:
                result = _run_agent_discovery_once(
                    backend,
                    spec,
                    timeout_s=timeout_s,
                    protected_files=protected,
                    usage=session.usage,
                )
                text = str(getattr(result, "text", "") or "").strip()
                end_reason = result.end_reason
                cut_short = end_reason in {"turn_cap", "timeout"}
                # A cut-short session still answered if it got its proposals out first, and discovery spends turns by
                # design -- it is handed read and search tools precisely so it explores.
                usable = text and (not cut_short or _extract_json_array(text))
                if usable and not is_api_failure(result):
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
            except Exception as exc:
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
            _back_off(attempt, started_at, progress, last_error, last_kind)
        _record_transcript(progress, last_error)
        raise LlmUnavailableError(
            f"discovery Agent produced no usable answer in {max(1, resolved_attempts)} attempt(s): {last_error}",
            kind=last_kind,
            attempts=max(1, resolved_attempts),
        )

    return _fn


def discover_recipes(
    diagnosis: Diagnosis,
    target: DiscoveryTarget,
    trace_path: str,
    llm_fn: LlmFn,
    prompting: DiscoveryPrompting = DiscoveryPrompting(),
) -> list[Recipe]:
    """LLM-autonomous discovery: propose fusible chains from the trace + source."""
    source_file, repo_scope = target.source_file, target.repo_scope
    if not diagnosis.is_candidate:
        return []
    try:
        source_text = Path(source_file).read_text(encoding="utf-8") if source_file else ""
    except OSError:
        source_text = ""
    if not source_text and not repo_scope:
        log.warning("discovery: model source unreadable (%s); cannot self-discover", source_file)
        return []
    hot = hot_kernels_from_trace(trace_path, top_n=_TOP_KERNELS)
    ordered_boundaries = ordered_fusion_boundaries_from_trace(trace_path)
    # Hot kernels and the diagnosis categories are folded in as a second evidence source: ordered boundaries need
    # repeats to exist, so a short trace would otherwise leave retrieval with nothing to match against.
    existing_operator_hints = existing_operator_hints_from_knowledge(
        prompting.knowledge_root,
        ordered_boundaries,
        fallback_categories=list(diagnosis.dominant_categories),
        fallback_kernel_names=kernel_names_from_trace(trace_path),
    )
    prompt = build_discovery_prompt(
        target,
        source_text,
        DiscoveryEvidence(
            diagnosis=diagnosis,
            hot_kernels=hot,
            ordered_boundaries=ordered_boundaries,
            existing_operator_hints=existing_operator_hints,
        ),
        _resolve_max_fusions(prompting.max_fusions),
    )
    raw = llm_fn(prompt)
    recipes = parse_discovered_recipes(raw, target, category_shares=diagnosis.category_shares)
    log.info(
        "discovery proposed %d fusion(s): %s",
        len(recipes),
        ", ".join(r.pattern_id for r in recipes),
    )
    return recipes
