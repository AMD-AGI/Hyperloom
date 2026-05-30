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
    "kernel": ["gemm", "attention", "flash", "triton", "ck", "hip", "kernel", "fused", "warp", "block"],
    "communication": ["nccl", "allreduce", "allgather", "p2p", "collective", "rccl", "bandwidth"],
    "compiler": ["compile", "inductor", "torch.compile", "graph", "fusion", "codegen"],
    "framework": ["vllm", "sglang", "tensorrt", "onnx", "framework", "serving", "batch"],
    "fusion": ["fused", "fusion", "combine", "merge", "operator"],
    "systems": ["memory", "scheduling", "prefetch", "pipeline", "overlap", "async"],
}


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


def select_kb(context: str, max_files: int = 5) -> list[KBFile]:
    """Select relevant KB files based on task context keywords."""
    context_lower = context.lower()
    scored_domains: list[tuple[str, int]] = []

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in context_lower)
        if score > 0:
            scored_domains.append((domain, score))

    scored_domains.sort(key=lambda x: -x[1])

    selected: list[KBFile] = []
    for domain, _ in scored_domains:
        files = get_domain_files(domain)
        for f in files[:2]:
            if len(selected) >= max_files:
                break
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
