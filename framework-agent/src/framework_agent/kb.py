# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Knowledge-base selector + contributor for framework-agent.

``KB_ROOT`` is resolved at call time via ``_resolve_kb_root()`` (under
``${FRAMEWORK_AGENT_KB_DIR}``, with a test fallback) so tests can monkeypatch
the environment. :func:`synthesize_findings` distils :class:`Finding` records
into a markdown blob for ``contribute_to_kb``; the default path is pure-Python
(zero deps), ``with_llm=True`` lazy-imports ``claude_agent_sdk``. Per-domain
priority order is ``empirical_kb.md`` -> ``shared_pitfalls.md`` -> rest.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import Finding


# Per-framework KB partition root: ``<KB_ROOT>/framework_optimization/
# <framework>/`` keeps framework-specific findings out of the cross-framework
# ``framework`` domain bag. Auto-created lazily by
# :func:`contribute_to_kb_for_framework` on first finding.
_FRAMEWORK_OPTIMIZATION_ROOT: str = "framework_optimization"


def path_for_framework(framework: str) -> Path:
    """Resolve the KB sub-partition path for a per-framework finding bag.

    Returns ``<KB_ROOT>/framework_optimization/<framework_lower>/``; the name
    is lowercased and stripped (``"  Atom  "`` and ``"ATOM"`` resolve to
    ``"atom"``). The partition dir may not exist until
    ``contribute_to_kb_for_framework`` creates it.

    Args:
        framework: Framework name; empty / whitespace-only resolves to the
            ``framework_optimization`` root (treated as "not selected").

    Returns:
        The KB partition path for the framework.
    """
    fw = (framework or "").strip().lower()
    root = _resolve_kb_root()
    if not fw:
        return root / _FRAMEWORK_OPTIMIZATION_ROOT
    return root / _FRAMEWORK_OPTIMIZATION_ROOT / fw


def contribute_to_kb_for_framework(
    framework: str,
    finding: str,
    source: str,
    session_id: str,
) -> Path:
    """Append a finding to the per-framework KB partition.

    Mirrors :func:`contribute_to_kb` but writes under
    :func:`path_for_framework`, for findings tied to a specific framework
    rather than a cross-framework domain. Partition dir is created lazily on
    first write.

    Args:
        framework: Framework whose partition to write to.
        finding: The finding body (Markdown).
        source: Provenance string recorded in the entry header.
        session_id: Session identifier recorded in the entry header.

    Returns:
        Path to the ``empirical_kb.md`` file that was appended to.
    """
    fw_dir = path_for_framework(framework)
    fw_dir.mkdir(parents=True, exist_ok=True)
    target = fw_dir / "empirical_kb.md"
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


DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "kernel": ["kernel", "gemm", "moe", "attention", "fmoe", "ck", "triton"],
    "communication": [
        "allreduce", "nccl", "rccl", "quickreduce", "communication", "collective",
    ],
    "compiler": ["compiler", "inductor", "codegen"],
    "framework": ["vllm", "sglang", "atom", "framework", "scheduler", "cuda_graph", "cudagraph"],
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
    """Resolve the active KB root each call (so tests can monkeypatch env).

    Order: (1) ``FRAMEWORK_AGENT_KB_DIR``; (2) ``${FRAMEWORK_AGENT_ROOT}/kb``;
    (3) ``${repo}/framework-agent/kb`` derived from this file's location.

    Returns:
        The resolved KB root path.
    """
    explicit = os.environ.get("FRAMEWORK_AGENT_KB_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    root = os.environ.get("FRAMEWORK_AGENT_ROOT", "").strip()
    if root:
        return Path(root).expanduser() / "kb"
    # parents[2] of .../framework_agent/kb.py is .../framework-agent/.
    return Path(__file__).resolve().parents[2] / "kb"


def list_domains() -> list[str]:
    """List domain directories under the active KB root (sorted).

    Returns:
        list[str]: Sorted domain directory names, or an empty list when the KB
            root does not exist.
    """
    root = _resolve_kb_root()
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def get_domain_files(domain: str) -> list[Path]:
    """List all files in a given domain directory (sorted).

    Args:
        domain (str): Domain directory name under the KB root.

    Returns:
        list[Path]: Sorted entries in the domain directory, or an empty list
            when the directory does not exist.
    """
    root = _resolve_kb_root()
    domain_dir = root / domain
    if not domain_dir.is_dir():
        return []
    return sorted(domain_dir.iterdir())


def _prioritized_files(domain: str) -> list[Path]:
    """Return domain files with priority entries (empirical / pitfalls) first.

    Args:
        domain (str): Domain directory name under the KB root.

    Returns:
        list[Path]: Files with :data:`_PRIORITY_FILES` ordered first, followed
            by the remaining files; empty when the directory does not exist.
    """
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
    """Match domains by keyword (case-insensitive) against the task text.

    Args:
        task_description (str): Free-text task description to match.

    Returns:
        list[str]: Sorted domain names whose keywords appear in the text.
    """
    lower = task_description.lower()
    matched: set[str] = set()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            matched.add(domain)
    return sorted(matched)


def _load_file(path: Path, domain: str) -> KBFile | None:
    """Best-effort read of a single KB file; OSErrors swallowed.

    Args:
        path (Path): File to read.
        domain (str): Domain the file belongs to, recorded on the record.

    Returns:
        KBFile | None: The loaded record, or ``None`` if the file cannot be
            read.
    """
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

    Args:
        task_description (str): Task text used to derive domains when
            ``domains`` is None.
        domains (list[str] | None): Explicit domain list, or ``None`` to
            auto-derive.

    Returns:
        list[KBFile]: Deduplicated, prioritised KB files for the resolved
            domains.
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
    """Concatenate the contents of multiple KB files with blank-line separation.

    Args:
        paths (list[Path]): Files to read and concatenate; unreadable files are
            skipped.

    Returns:
        str: The concatenated file contents joined by blank lines.
    """
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

    Creates the domain directory and the file if missing.

    Args:
        domain (str): Domain directory name under the KB root.
        finding (str): Markdown finding body to append.
        source (str): Source tag recorded in the entry header.
        session_id (str): Session identifier recorded in the entry header.

    Returns:
        Path: The ``empirical_kb.md`` file the finding was appended to.
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
    """Render a single Finding into a markdown subsection.

    Args:
        finding (Finding): The finding to render.

    Returns:
        str: A markdown subsection with the title, optional metadata, metrics,
            and body.
    """
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
    """Pure-Python distillation: deterministic markdown digest of findings.

    Layout: ``## Synthesised findings - <domain>``, one ``###`` section per
    finding, then ``## Aggregate metrics`` when metric keys repeat.

    Args:
        domain: Domain label for the digest header.
        findings: Findings to render.

    Returns:
        The deterministic Markdown digest.
    """
    if not findings:
        return f"## Synthesised findings - {domain}\n\n_no findings_\n"
    lines: list[str] = [f"## Synthesised findings - {domain}", ""]
    for f in findings:
        lines.append(_render_finding_markdown(f))
        lines.append("")
    # Per-metric counts surface repeat signals across candidates.
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
    """Build the prompt fed to claude_agent_sdk when ``with_llm=True``.

    Args:
        domain (str): Domain label included in the prompt.
        findings (list[Finding]): Raw findings rendered into the prompt body.

    Returns:
        str: The full curator prompt string.
    """
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
    """Distil findings via claude_agent_sdk.

    Network / SDK errors are not caught so misconfiguration is loud.

    Args:
        domain: Domain label for the synthesis.
        findings: Findings to summarize.
        model: SDK model identifier to use.

    Returns:
        The LLM-synthesized Markdown, falling back to the pure-Python
        digest when the SDK returns nothing.

    Raises:
        RuntimeError: If ``claude_agent_sdk`` is missing or lacks required
            attributes.
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
    # sdk.query is an async generator; block via asyncio.run() since the
    # explore --execute call site has no event loop running.
    import asyncio

    async def _drive() -> None:
        """Stream the SDK query and accumulate text into ``chunks``.

        Iterates the async generator returned by ``sdk.query`` and appends every
        non-empty text fragment from each message to the enclosing ``chunks``
        list.
        """
        async for message in sdk.query(prompt=prompt, options=options):
            for text in _iter_message_text(message):
                if text:
                    chunks.append(text)

    asyncio.run(_drive())
    return "".join(chunks).strip() or _synthesize_pure_python(domain, findings)


def _iter_message_text(message) -> Iterable[str]:
    """Best-effort text extraction from a claude_agent_sdk message.

    Accepts a plain string, ``.text``, or a ``.content`` list of blocks each
    with ``.text`` (SDK message shape varies across versions).

    Args:
        message: An SDK message object or string.

    Yields:
        Each non-empty text fragment found on the message.
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
    """Distil ``findings`` into a markdown blob for ``contribute_to_kb``.

    Default is pure-Python (deterministic, no SDK/network). ``with_llm=True``
    routes through a lazy-imported claude_agent_sdk.

    Args:
        domain: Domain label for the synthesis.
        findings: Findings to distill.
        with_llm: Whether to route through the LLM synthesizer.
        model: SDK model identifier (used only when ``with_llm`` is True).

    Returns:
        The synthesized Markdown blob.
    """
    if not with_llm:
        return _synthesize_pure_python(domain, findings)
    return _synthesize_via_llm(domain, findings, model=model)


def search_kb(query: str, *, domains: list[str] | None = None) -> list[KBFile]:
    """Case-insensitive substring search across all (or selected) domains.

    Args:
        query: Substring to search for (matched case-insensitively).
        domains: Domains to search; defaults to all known domains.

    Returns:
        Deduplicated :class:`KBFile` records whose ``content`` contains the
        query, ordered per :func:`_prioritized_files` within each domain.
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
    "contribute_to_kb_for_framework",
    "path_for_framework",
    "synthesize_findings",
    "search_kb",
]
