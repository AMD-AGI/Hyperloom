# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The text of the decode-path discovery prompt, with a named slot for every piece of evidence it cites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from hyperloom_common.coerce import to_float

from .author import _arch_phrase
from .diagnose import LAUNCH_BOUND_CATEGORIES
from .locate import FrameworkInstall, _read_source
from .models import Diagnosis


@dataclass(frozen=True)
class DiscoveryTarget:
    """What discovery proposes fusions for: the model, the framework install, the source file, shapes and scope."""

    model_type: str
    install: FrameworkInstall
    source_file: str
    shapes: dict[str, Any]
    repo_scope: bool = False
    repo_root: str = ""
    gpu_arch: str = ""


@dataclass(frozen=True)
class DiscoveryEvidence:
    """The evidence a discovery prompt cites: diagnosis, hot kernels, fusion boundaries, known operators, profiler hints."""

    diagnosis: Diagnosis
    hot_kernels: list[dict[str, Any]]
    ordered_boundaries: Optional[list[dict[str, Any]]] = None
    existing_operator_hints: Optional[list[dict[str, str]]] = None
    fusion_hints: Sequence[dict[str, Any]] = ()


# Each proposed fusion costs discovery tokens plus one authoring subprocess and validation pass, and that cost is paid
# before the E2E gate can reject it.
_DEFAULT_MAX_FUSIONS = 4

DISCOVERY_PROMPT_TEMPLATE = """You are analyzing the DECODE path of a {framework} model (`model_type={model_type}`)
on {arch_phrase}, bf16, to find SOURCE-LEVEL KERNEL FUSIONS. Analyze only; do
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
{boundary_lines}

## Existing ROCm operator evidence
Retrieved by whole-word overlap between the observed kernel/category names and the
documented operation chain; a higher score means stronger evidence. The score is a
retrieval hint, NOT a correctness claim: you must still confirm from the source and
the card that shape, dtype, and cache layout actually match.
{operator_lines}

{fusion_hints_block}## Your task
{read_verb} up to {max_fusions} CONTIGUOUS op chains in
the DECODE forward that are worth fusing. Include GEMM/attention prologue or epilogue
boundaries when the ordered trace proves adjacency. Treat an existing ROCm operator
that covers a larger boundary as an `integration` candidate and benchmark/wire it
before proposing a new kernel. Judge from the SOURCE and ordered trace what actually
runs back-to-back on the decode path.

{fusion_constraints}

{output_schema_block}

{source_block}
"""


def _boundary_avg_us(boundary: dict[str, Any]) -> float:
    if "avg_chain_us" in boundary:
        return float(boundary["avg_chain_us"])
    return float(boundary.get("total_us", 0.0)) / max(1, int(boundary["count"]))


def _boundary_line(boundary: dict[str, Any]) -> str:
    line = (
        "  - "
        f"{boundary['signature']} "
        f"(repeats={boundary['count']}, "
        f"avg-chain={_boundary_avg_us(boundary):.1f}us, "
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
        line += f"\n    NOTE: fuse the prologue only; do NOT include the terminal {terminal} kernel in the fused chain."
    line += f"\n    kernels: {' | '.join(boundary.get('kernels', []))[:500]}"
    return line


def _operator_line(hint: dict[str, Any]) -> str:
    score = hint.get("score")
    score_text = f" score={float(score):.1f}" if score is not None else ""
    return f"  - `{hint['operator']}` ({hint['path']}){score_text}: {hint['evidence']}"


def render_boundary_lines(boundaries: Sequence[dict[str, Any]] | None) -> str:
    """The ordered fusion boundaries of the trace, one entry per same-stream chain."""
    return "\n".join(_boundary_line(boundary) for boundary in boundaries or [])


def render_operator_lines(hints: Sequence[dict[str, Any]] | None) -> str:
    """The existing operators retrieval matched against the trace, with their recall scores."""
    return "\n".join(_operator_line(hint) for hint in hints or [])


def render_fusion_hints(hints: Sequence[dict[str, Any]]) -> str:
    """The advisory profiler-hints section of the discovery prompt; empty when no hints were given."""
    if not hints:
        return ""
    lines = [
        "## External profiler hints (advisory)",
        "Another profiler (e.g. TraceLens) flagged these unfused kernel runs, largest projected saving first. Long",
        "kernel names may be truncated with `...`. They are leads, not answers: propose a chain only where the",
        "trace evidence above and the source confirm it.",
    ]
    for hint in hints:
        pct = to_float(hint.get("projected_e2e_pct"))
        saving = f", projected E2E saving {pct:.1f}%" if pct is not None else ""
        lines.append(f"  - {hint.get('kind') or 'unclassified'}: {hint.get('operation') or ''}{saving}")
        members = [m for m in hint.get("members") or [] if isinstance(m, dict)]
        kernels = " | ".join(" + ".join(map(str, m.get("kernel_names") or [])) for m in members)
        launchers = sorted({str(m["kernel_launcher_path"]) for m in members if m.get("kernel_launcher_path")})
        if kernels:
            lines.append(f"    kernels: {kernels[:500]}")
        if launchers:
            lines.append(f"    launchers: {'; '.join(launchers)}")
    return "\n".join(lines) + "\n\n"


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
    model_type, framework = target.model_type, target.install.framework
    shapes, repo_scope = target.shapes, target.repo_scope
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

    dom = ", ".join(diagnosis.dominant_categories)
    return DISCOVERY_PROMPT_TEMPLATE.format(
        fusion_hints_block=render_fusion_hints(evidence.fusion_hints),
        framework=framework,
        model_type=model_type,
        arch_phrase=_arch_phrase(target.gpu_arch),
        diagnosis=diagnosis,
        lb=lb,
        hot_lines=hot_lines,
        dom=dom,
        shapes=shapes,
        boundary_lines=render_boundary_lines(evidence.ordered_boundaries) or "  - none",
        operator_lines=render_operator_lines(evidence.existing_operator_hints) or "  - none found",
        read_verb=read_verb,
        max_fusions=max_fusions,
        fusion_constraints=fusion_constraints(repo_scope=repo_scope),
        output_schema_block=_output_schema_block(model_type, repo_scope=repo_scope),
        source_block=source_block,
    )
