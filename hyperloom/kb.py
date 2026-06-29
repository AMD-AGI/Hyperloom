"""Knowledge base — domain-aware KB for agent prompts.

Organizes optimization knowledge by domain (kernel, framework, compiler, etc.)
and selects relevant content based on the current task context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

KB_ROOT = Path(__file__).parent / "kb"

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "kernel": [
        "gemm", "moe", "mixture of experts", "expert", "experts", "attention",
        "flash", "triton", "ck", "hip", "kernel", "fused", "warp", "block",
        "grouped gemm", "decode", "launch-bound", "launch bound",
    ],
    "communication": ["nccl", "allreduce", "allgather", "p2p", "collective", "rccl", "bandwidth"],
    "compiler": ["compile", "inductor", "torch.compile", "graph", "fusion", "codegen"],
    "framework": ["vllm", "sglang", "tensorrt", "onnx", "framework", "serving", "batch"],
    "fusion": [
        "fused", "fusion", "combine", "merge", "operator",
        "shared expert", "shared-expert", "shared_expert", "shared experts",
        "fold", "grouped gemm", "moe",
    ],
    "systems": ["memory", "scheduling", "prefetch", "pipeline", "overlap", "async"],
}

# Stopwords excluded from the token-overlap fallback (so generic words like
# "model"/"speed" don't spuriously match every KB file).
_FALLBACK_STOPWORDS = frozenset({
    "model", "models", "speed", "faster", "improve", "optimize", "optimise",
    "optimization", "performance", "inference", "throughput", "serving",
    "reduce", "increase", "better", "using", "with", "this", "that", "from",
    "should", "could", "would", "which", "while", "their", "there",
})


@dataclass
class KBFile:
    """A knowledge base document."""

    domain: str
    filename: str
    path: str
    content: str = ""


def list_domains() -> list[str]:
    """List available KB domains."""
    if not KB_ROOT.exists():
        return []
    return [d.name for d in KB_ROOT.iterdir() if d.is_dir()]


def get_domain_files(domain: str) -> list[KBFile]:
    """Get all KB files for a domain."""
    domain_dir = KB_ROOT / domain
    if not domain_dir.exists():
        return []
    files = []
    for f in sorted(domain_dir.glob("*.md")):
        files.append(KBFile(domain=domain, filename=f.name, path=str(f)))
    return files


def _fallback_domains(context_lower: str) -> list[str]:
    """Token-overlap fallback when no keyword domain matched.

    Scans each domain's markdown for significant words from the task context
    (length >= 4, not a stopword) and ranks domains by how many distinct context
    tokens appear. This recovers cases the keyword list misses — e.g. a task that
    only names the model ("minimax m3") still finds the KB file that mentions it.
    """
    import re

    tokens = {
        w for w in re.findall(r"[a-z0-9][a-z0-9_-]{3,}", context_lower)
        if w not in _FALLBACK_STOPWORDS
    }
    if not tokens:
        return []
    scored: list[tuple[str, int]] = []
    for domain in list_domains():
        domain_text_parts: list[str] = []
        for f in get_domain_files(domain):
            try:
                domain_text_parts.append(Path(f.path).read_text(errors="replace").lower())
            except OSError:
                continue
        domain_text = "\n".join(domain_text_parts)
        score = sum(1 for tok in tokens if tok in domain_text)
        if score > 0:
            scored.append((domain, score))
    scored.sort(key=lambda x: -x[1])
    return [d for d, _ in scored]


def select_kb(
    context: str,
    max_files: int = 5,
    domains: list[str] | None = None,
) -> list[KBFile]:
    """Select relevant KB files based on task context.

    Resolution order:
      1. explicit ``domains`` (caller-provided), else
      2. keyword match against ``DOMAIN_KEYWORDS``, else
      3. token-overlap fallback scan over KB content (:func:`_fallback_domains`).
    """
    context_lower = context.lower()

    # Keyword/content-derived domains — always computed so they can supplement
    # (or stand in for) caller-provided hints. A caller may pass a `domains`
    # value that doesn't map to any real KB directory (e.g. "moe", "rocm");
    # without this fallback that would silently starve the agent of KB context.
    scored_domains: list[tuple[str, int]] = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in context_lower)
        if score > 0:
            scored_domains.append((domain, score))
    scored_domains.sort(key=lambda x: -x[1])
    keyword_domains = [d for d, _ in scored_domains]
    if not keyword_domains:
        keyword_domains = _fallback_domains(context_lower)

    if domains:
        # Honor caller hints first, then append keyword-derived domains so a
        # bogus/unknown domain hint never suppresses an otherwise-relevant file.
        ordered_domains = list(domains)
        for d in keyword_domains:
            if d not in ordered_domains:
                ordered_domains.append(d)
    else:
        ordered_domains = keyword_domains

    selected: list[KBFile] = []
    seen: set[str] = set()
    for domain in ordered_domains:
        files = get_domain_files(domain)
        for f in files[:2]:
            if len(selected) >= max_files:
                break
            if f.path in seen:
                continue
            seen.add(f.path)
            selected.append(f)

    return selected


def load_kb_content(files: list[KBFile]) -> str:
    """Load and concatenate KB file contents."""
    parts = []
    for f in files:
        path = Path(f.path)
        if path.exists():
            content = path.read_text(errors="replace")
            parts.append(f"## [{f.domain}] {f.filename}\n\n{content}\n")
            f.content = content
    return "\n---\n".join(parts)


def contribute_to_kb(domain: str, filename: str, content: str) -> None:
    """Add a new finding to the KB."""
    domain_dir = KB_ROOT / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    (domain_dir / filename).write_text(content)
    log.info("Contributed to KB: %s/%s", domain, filename)
