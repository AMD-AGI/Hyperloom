"""Knowledge-base selector + contributor for framework-agent.

Derived from ``TBO/src/arbor/kb.py`` (the 4-file domain schema) with two
framework-agent-specific changes:

* ``KB_ROOT`` is resolved at call time via ``_resolve_kb_root()`` instead of
  being a hardcoded module-level path - the framework-agent KB lives under
  ``${FRAMEWORK_AGENT_KB_DIR}`` (env), with a sane fallback for tests.
* A new :func:`synthesize_findings` distils a list of :class:`Finding`
  records into a markdown blob suitable for ``contribute_to_kb``. The
  default path is pure-Python (zero deps); ``with_llm=True`` lazy-imports
  ``claude_agent_sdk`` (same pattern as kernel-agent's
  ``tracelens_skill_runner``) and only raises on real failure.

Domain matching keywords + the per-domain priority order
(``empirical_kb.md`` -> ``shared_pitfalls.md`` -> rest) follow Arbor 1:1.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import Finding


DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "kernel": ["kernel", "gemm", "moe", "attention", "fmoe", "ck", "triton"],
    "communication": [
        "allreduce", "nccl", "rccl", "quickreduce", "communication", "collective",
    ],
    "compiler": ["compiler", "inductor", "codegen"],
    "framework": ["vllm", "sglang", "framework", "scheduler", "cuda_graph", "cudagraph"],
    "fusion": ["fusion", "fused", "overlap"],
    "systems": ["system", "hip", "rocm", "driver", "launch", "dispatch"],
    "pr_intelligence": ["pr", "github", "upstream", "patch"],
    "recipes": ["recipe", "warm_start", "best_config", "prior_session"],
}

_PRIORITY_FILES = ["empirical_kb.md", "shared_pitfalls.md"]

_DEFAULT_MODEL = "claude-opus-4-7"


@dataclass
class KBFile:
    """A single resolved KB markdown file with its content already loaded."""

    path: Path
    domain: str
    content: str


def _resolve_kb_root() -> Path:
    """Resolve the active KB root, preferring the env override.

    Resolution order:

    1. ``FRAMEWORK_AGENT_KB_DIR`` env var if set (set by ``install.sh``);
    2. ``${FRAMEWORK_AGENT_ROOT}/kb`` if ``FRAMEWORK_AGENT_ROOT`` is set;
    3. ``${repo}/framework-agent/kb`` derived from this file's location.

    Resolved each call so tests can monkeypatch the environment.
    """
    explicit = os.environ.get("FRAMEWORK_AGENT_KB_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    root = os.environ.get("FRAMEWORK_AGENT_ROOT", "").strip()
    if root:
        return Path(root).expanduser() / "kb"
    # ``__file__`` is .../framework-agent/src/framework_agent/kb.py;
    # parents[2] is .../framework-agent/.
    return Path(__file__).resolve().parents[2] / "kb"


def list_domains() -> list[str]:
    """List domain directories under the active KB root (sorted)."""
    root = _resolve_kb_root()
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def get_domain_files(domain: str) -> list[Path]:
    """List all files in a given domain directory (sorted)."""
    root = _resolve_kb_root()
    domain_dir = root / domain
    if not domain_dir.is_dir():
        return []
    return sorted(domain_dir.iterdir())


def _prioritized_files(domain: str) -> list[Path]:
    """Return domain files with priority entries (empirical / pitfalls) first."""
    root = _resolve_kb_root()
    domain_dir = root / domain
    if not domain_dir.is_dir():
        return []
    priority: list[Path] = []
    rest: list[Path] = []
    for p in sorted(domain_dir.iterdir()):
        if not p.is_file():
            continue
        (priority if p.name in _PRIORITY_FILES else rest).append(p)
    priority.sort(key=lambda p: _PRIORITY_FILES.index(p.name))
    return priority + rest


def _match_domains(task_description: str) -> list[str]:
    """Match domains by keyword (case-insensitive) against the task text."""
    lower = task_description.lower()
    matched: set[str] = set()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            matched.add(domain)
    return sorted(matched)


def _load_file(path: Path, domain: str) -> KBFile | None:
    """Best-effort read of a single KB file; OSErrors swallowed."""
    try:
        content = path.read_text()
    except OSError:
        return None
    return KBFile(path=path, domain=domain, content=content)


def select_kb(
    task_description: str,
    domains: list[str] | None = None,
) -> list[KBFile]:
    """Return prioritised KB files for the given task / domain list.

    If ``domains`` is None, derive them from ``task_description`` via
    keyword match; if keywords don't hit anything, fall back to a
    full-text scan across all domains' priority files.
    """
    if domains is None:
        domains = _match_domains(task_description)
        if not domains:
            lower = task_description.lower()
            for domain in list_domains():
                for path in _prioritized_files(domain):
                    try:
                        if lower in path.read_text().lower():
                            domains.append(domain)
                            break
                    except OSError:
                        continue
    results: list[KBFile] = []
    seen: set[Path] = set()
    for domain in domains:
        for path in _prioritized_files(domain):
            if path in seen:
                continue
            seen.add(path)
            kb_file = _load_file(path, domain)
            if kb_file is not None:
                results.append(kb_file)
    return results


def load_kb_content(paths: list[Path]) -> str:
    """Concatenate the contents of multiple KB files with blank-line separation."""
    parts: list[str] = []
    for p in paths:
        try:
            parts.append(p.read_text())
        except OSError:
            continue
    return "\n\n".join(parts)


def contribute_to_kb(
    domain: str,
    finding: str,
    source: str,
    session_id: str,
) -> Path:
    """Append a single finding to ``${KB}/<domain>/empirical_kb.md``.

    Creates the domain directory and the file if missing. Returns the
    file path so callers can log/audit the location.
    """
    root = _resolve_kb_root()
    domain_dir = root / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    target = domain_dir / "empirical_kb.md"
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    entry = (
        f"\n\n---\n"
        f"**[{timestamp}]** source=`{source}` session=`{session_id}`\n\n"
        f"{finding}\n"
    )
    with target.open("a") as f:
        f.write(entry)
    return target


def _render_finding_markdown(finding: Finding) -> str:
    """Render a single Finding into a markdown subsection."""
    lines = [f"### {finding.title or 'untitled finding'}"]
    if finding.source:
        lines.append(f"- source: `{finding.source}`")
    if finding.session_id:
        lines.append(f"- session: `{finding.session_id}`")
    if finding.candidate_ref:
        lines.append(f"- candidate: `{finding.candidate_ref}`")
    if finding.metrics:
        lines.append("- metrics:")
        for key in sorted(finding.metrics):
            lines.append(f"  - `{key}` = {finding.metrics[key]}")
    if finding.body:
        lines.append("")
        lines.append(finding.body.rstrip())
    return "\n".join(lines)


def _synthesize_pure_python(domain: str, findings: list[Finding]) -> str:
    """Pure-Python distillation: emit a stable markdown digest of findings.

    No LLM in the loop, so the output is deterministic and safe to use
    in test assertions. Layout:
        ## Synthesised findings - <domain>
        <one ### section per finding>
        ## Aggregate metrics (when metric keys repeat)
    """
    if not findings:
        return f"## Synthesised findings - {domain}\n\n_no findings_\n"
    lines: list[str] = [f"## Synthesised findings - {domain}", ""]
    for f in findings:
        lines.append(_render_finding_markdown(f))
        lines.append("")
    # Aggregate per-metric counts so a reader can spot repeat signals.
    counts: dict[str, int] = {}
    for f in findings:
        for k in f.metrics:
            counts[k] = counts.get(k, 0) + 1
    repeated = {k: v for k, v in counts.items() if v > 1}
    if repeated:
        lines.append("## Aggregate metrics")
        for key in sorted(repeated):
            lines.append(f"- `{key}` reported by {repeated[key]} candidates")
        lines.append("")
    return "\n".join(lines)


def _build_llm_prompt(domain: str, findings: list[Finding]) -> str:
    """Build the prompt fed to claude_agent_sdk when ``with_llm=True``."""
    raw = _synthesize_pure_python(domain, findings)
    return (
        "You are a curator for the framework-agent knowledge base. "
        f"The findings below are raw observations under domain '{domain}'. "
        "Summarise them into a single markdown section suitable for appending "
        "to empirical_kb.md. Keep concrete numbers verbatim. Do not invent "
        "data. Output markdown only, no preamble.\n\n"
        "Raw findings:\n\n"
        f"{raw}\n"
    )


def _synthesize_via_llm(
    domain: str,
    findings: list[Finding],
    *,
    model: str,
) -> str:
    """Distil findings via claude_agent_sdk; raises if the SDK is missing.

    Mirrors kernel-agent's lazy-import pattern: ImportError surfaces
    a RuntimeError that tells the operator how to install the SDK.
    Network / SDK errors are *not* caught - callers see the original
    exception so misconfiguration is loud.
    """
    try:
        import claude_agent_sdk as sdk  # type: ignore  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via test stub
        raise RuntimeError(
            "claude_agent_sdk not installed; run framework-agent install "
            "with the [claude] extra or reuse kernel-agent's install.sh"
        ) from exc
    if not (hasattr(sdk, "query") and hasattr(sdk, "ClaudeAgentOptions")):
        raise RuntimeError(
            "claude_agent_sdk missing required attributes (query / ClaudeAgentOptions)"
        )

    prompt = _build_llm_prompt(domain, findings)
    options = sdk.ClaudeAgentOptions(model=model, system_prompt="")
    chunks: list[str] = []
    # claude_agent_sdk.query returns an async generator. We block on it
    # synchronously via asyncio.run() to stay friendly to the existing
    # explore --execute call site (no event loop running).
    import asyncio

    async def _drive() -> None:
        """Async driver that drains the SDK's message stream."""
        async for message in sdk.query(prompt=prompt, options=options):
            for text in _iter_message_text(message):
                if text:
                    chunks.append(text)

    asyncio.run(_drive())
    return "".join(chunks).strip() or _synthesize_pure_python(domain, findings)


def _iter_message_text(message) -> Iterable[str]:
    """Best-effort extraction of text from a claude_agent_sdk message.

    The SDK has changed message shape across versions; we accept any
    of: a plain string, ``.text``, or a ``.content`` list of blocks
    each with ``.text``. Anything else yields nothing.
    """
    if isinstance(message, str):
        yield message
        return
    text = getattr(message, "text", None)
    if isinstance(text, str):
        yield text
    content = getattr(message, "content", None)
    if isinstance(content, list):
        for block in content:
            block_text = getattr(block, "text", None)
            if isinstance(block_text, str):
                yield block_text


def synthesize_findings(
    domain: str,
    findings: list[Finding],
    *,
    with_llm: bool = False,
    model: str = _DEFAULT_MODEL,
) -> str:
    """Distil ``findings`` into a markdown blob ready for ``contribute_to_kb``.

    Default path is pure-Python (no SDK, no network). Set
    ``with_llm=True`` to route through claude_agent_sdk; the SDK is
    lazy-imported on first use and raises RuntimeError with a clear
    install hint when missing. Operators looking for full determinism
    should keep the default.
    """
    if not with_llm:
        return _synthesize_pure_python(domain, findings)
    return _synthesize_via_llm(domain, findings, model=model)


def search_kb(query: str, *, domains: list[str] | None = None) -> list[KBFile]:
    """Substring search across all (or selected) domains; case-insensitive.

    Returns a deduplicated list of KBFile records whose ``content``
    contains ``query`` (case-insensitive). Order follows
    :func:`_prioritized_files` within each domain.
    """
    needle = query.lower()
    domains = domains or list_domains()
    hits: list[KBFile] = []
    seen: set[Path] = set()
    for domain in domains:
        for path in _prioritized_files(domain):
            if path in seen:
                continue
            seen.add(path)
            kb_file = _load_file(path, domain)
            if kb_file is None:
                continue
            if needle in kb_file.content.lower():
                hits.append(kb_file)
    return hits


__all__ = [
    "KBFile",
    "DOMAIN_KEYWORDS",
    "list_domains",
    "get_domain_files",
    "select_kb",
    "load_kb_content",
    "contribute_to_kb",
    "synthesize_findings",
    "search_kb",
]
