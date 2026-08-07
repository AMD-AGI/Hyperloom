# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knowledge-base selector + contributor for framework-agent.

The KB splits in two. :func:`packaged_kb_root` is read-only seed data shipped
in the wheel; :func:`mutable_kb_root` is the per-deployment partition this
session reads and writes, and is the single owner of that path for both this
module and the orchestrator's ``kb_writeback``. Both resolve at call time so
tests can monkeypatch the environment. :func:`synthesize_findings` distils
:class:`Finding` records into a markdown blob for ``contribute_to_kb``; the
default path is pure-Python (zero deps), ``with_llm=True`` lazy-imports
``claude_agent_sdk``. Per-domain priority order is ``empirical_kb.md`` ->
``shared_pitfalls.md`` -> rest.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import Finding

_log = logging.getLogger(__name__)


# Per-framework KB partition root under ``<KB_ROOT>/framework_optimization/``.
_FRAMEWORK_OPTIMIZATION_ROOT: str = "framework_optimization"

#: The only supported override for the mutable KB root; both this module and
#: ``kb_writeback`` honour it. It reaches the process through the
#: ``INFERENCE_OPTIMIZER_`` prefix rule in the ``common/env_safety`` dotenv
#: allowlist, which is a prefix rule rather than an entry for this name.
KB_ROOT_ENV: str = "INFERENCE_OPTIMIZER_FA_KB_PATH"

#: Workspace subdirectory holding this KB. Deliberately not ``kb``: that is the
#: legacy recipe root (``inference_optimizer.cli.kb._legacy_recipe_root``, still
#: read by the one-time recipe migration), and ``list_domains`` reports every
#: directory under this root as a framework domain, so sharing it would surface
#: recipe trees as framework domains. The current recipe root is
#: ``<workspace>/knowledge`` and never collided.
_MUTABLE_KB_DIRNAME: str = "framework-kb"

#: Where the writer put this partition before it was given its own directory.
#: Same value as ``inference_optimizer.cli.kb._legacy_recipe_root``'s leaf, which
#: this package cannot import; the guard test asserts they still agree.
_LEGACY_WORKSPACE_KB_DIRNAME: str = "kb"

#: Workspace root when ``USER_DATA_PATH`` is unset. Mirrors
#: ``session.paths.DEFAULT_SESSION_DIR``, which this package cannot import:
#: the ``fa`` CLI runs standalone and must not depend on inference_optimizer.
_DEFAULT_WORKSPACE_ROOT: str = "/workspace/hyperloom"

#: Withdrawn override. Only the reader honoured it, so setting it split the KB
#: in two. ``FRAMEWORK_AGENT_ROOT`` is deliberately absent: it means "where
#: this skill is installed", is used for other purposes, and never reached the
#: reader anyway because no installer exports it.
_REMOVED_KB_ROOT_ENV: str = "FRAMEWORK_AGENT_KB_DIR"


#: Marker written into the migrated partition so the copy happens exactly once
#: and its provenance stays inspectable.
_MIGRATION_MARKER: str = ".migrated-from-legacy-kb.json"


class KBConfigurationError(RuntimeError):
    """The environment asks for a KB layout this build no longer supports."""


def prepare_kb_environment() -> None:
    """Start-up sequence for the framework KB: validate, then migrate.

    Both entry points that can reach this KB — the inference_optimizer preflight
    and the standalone ``fa`` CLI — call this one function, so a future start-up
    step is added in one place instead of being remembered in two.

    Raises:
        KBConfigurationError: If the environment asks for an unsupported layout.
    """
    check_kb_configuration()
    migrate_legacy_partition_once()


def check_kb_configuration() -> None:
    """Reject a KB environment that would silently split reads from writes.

    Called once at start-up rather than from the resolver. Callers of the read
    path treat their KB lookups as advisory and swallow failures so an
    unreadable ledger cannot block dispatch; raising from the resolver would
    therefore turn a misconfiguration into a silently disabled accuracy gate,
    which is the failure mode this whole area is trying to remove.

    Raises:
        KBConfigurationError: If the withdrawn reader-only override is set.
    """
    if os.environ.get(_REMOVED_KB_ROOT_ENV, "").strip():
        raise KBConfigurationError(
            f"{_REMOVED_KB_ROOT_ENV} is no longer honoured because it redirected only the "
            f"reader, leaving writes behind in the previous location. "
            f"Set {KB_ROOT_ENV} instead; it moves both halves of the KB together."
        )


def migrate_legacy_partition_once() -> Path | None:
    """Carry the framework partition over from the legacy ``<workspace>/kb`` root.

    Until this KB was given its own directory, the writer put the ledger under
    ``<workspace>/kb/framework_optimization``. The writer was working, so every
    deployment that ever ran a FRAMEWORK phase has real data there — leaving it
    behind silently empties the dedup ledger and re-proposes PRs that already
    lost an accuracy gate.

    Only runs when the destination has no framework data yet, so it can never
    overwrite a live partition, and stages the copy under a sibling directory so
    an interrupted run does not leave a half-populated destination that the next
    start-up would mistake for a live one. The source is left in place.

    Skipped entirely when :data:`KB_ROOT_ENV` is set: the operator named a
    location, and the legacy default was never theirs.

    Returns:
        The destination partition when data was migrated, else ``None``.
    """
    if os.environ.get(KB_ROOT_ENV, "").strip():
        return None

    workspace = Path(os.environ.get("USER_DATA_PATH", "").strip() or _DEFAULT_WORKSPACE_ROOT).expanduser()
    source = workspace / _LEGACY_WORKSPACE_KB_DIRNAME / _FRAMEWORK_OPTIMIZATION_ROOT
    destination = mutable_kb_root() / _FRAMEWORK_OPTIMIZATION_ROOT

    if not source.is_dir() or not any(source.iterdir()):
        return None
    if destination.exists() and any(destination.iterdir()):
        return None

    staging = destination.with_name(f"{destination.name}.migrating")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        shutil.copytree(source, staging)
        (staging / _MIGRATION_MARKER).write_text(
            json.dumps({"version": 1, "source": str(source)}, sort_keys=True),
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _log.warning(
        "FRAMEWORK KB: migrated the legacy partition %s -> %s. The source is left in place; "
        "remove it once the new location looks right.",
        source,
        destination,
    )
    return destination


def path_for_framework(framework: str) -> Path:
    """Resolve the KB sub-partition path for a per-framework finding bag.

    Returns ``<KB_ROOT>/framework_optimization/<framework_lower>/``; the name
    is lowercased and stripped (``"  Atom  "`` and ``"ATOM"`` resolve to
    ``"atom"``). The partition dir may not exist yet.

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


DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "kernel_agent": ["kernel_agent", "gemm", "moe", "attention", "fmoe", "ck", "triton"],
    "communication": [
        "allreduce",
        "nccl",
        "rccl",
        "quickreduce",
        "communication",
        "collective",
    ],
    "compiler": ["compiler", "inductor", "codegen"],
    "framework": ["vllm", "sglang", "atom", "framework", "scheduler", "cuda_graph", "cudagraph"],
    "fusion": ["fusion", "fused", "overlap"],
    "systems": ["system", "hip", "rocm", "driver", "launch", "dispatch"],
    "pr_intelligence": ["pr", "github", "upstream", "patch"],
    "recipes": ["recipe", "warm_start", "best_config", "prior_session"],
}

_PRIORITY_FILES = ["empirical_kb.md", "shared_pitfalls.md"]

_DEFAULT_MODEL = "claude-opus-5"


@dataclass
class KBFile:
    """A single resolved KB markdown file with its content already loaded."""

    path: Path
    domain: str
    content: str


def packaged_kb_root() -> Path:
    """Root of the read-only KB shipped inside the wheel.

    Holds seed data seeded at build time, currently just the cross-framework
    module map. Nothing writes here: an installed package may sit on a
    read-only filesystem, and an upgrade would overwrite whatever was added.

    Returns:
        The packaged KB root path.
    """
    return Path(__file__).resolve().parent / "kb"


def mutable_kb_root() -> Path:
    """Root of the KB partition this session reads and writes.

    The single owner of that path. Both the ``fa`` reader and the
    orchestrator's ``kb_writeback`` resolve through here, so a deployment that
    moves the KB moves both halves at once; resolving it independently on each
    side is what left written lessons unreadable by the next session.

    Resolved per call rather than at import, because the environment is not
    fully settled when this module is first imported.

    Deliberately not ``<workspace>/kb``: that is the legacy recipe root, and
    this reader enumerates whatever directories sit under its own root, so
    sharing one would present recipe trees as framework domains.

    Returns:
        ``$INFERENCE_OPTIMIZER_FA_KB_PATH`` when set, else
        ``<workspace>/framework-kb`` where the workspace is ``$USER_DATA_PATH``
        or the pod-local default.
    """
    override = os.environ.get(KB_ROOT_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    workspace = os.environ.get("USER_DATA_PATH", "").strip() or _DEFAULT_WORKSPACE_ROOT
    return Path(workspace).expanduser() / _MUTABLE_KB_DIRNAME


def _resolve_kb_root() -> Path:
    """Resolve the active KB root each call (so tests can monkeypatch env).

    Never raises: read paths swallow their own failures by design, so an
    exception here would be absorbed rather than surfaced. The withdrawn
    override is rejected by :func:`check_kb_configuration` at start-up.

    Returns:
        The resolved KB root path.
    """
    return mutable_kb_root()


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
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    entry = f"\n\n---\n**[{timestamp}]** source=`{source}` session=`{session_id}`\n\n{finding}\n"
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
        raise RuntimeError("claude_agent_sdk missing required attributes (query / ClaudeAgentOptions)")

    prompt = _build_llm_prompt(domain, findings)
    options = sdk.ClaudeAgentOptions(model=model, system_prompt="")
    chunks: list[str] = []
    # sdk.query is an async generator; block via asyncio.run().
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


def read_pr_ledger(kb_root: Path | None = None) -> list[dict]:
    """Read the framework PR outcome ledger from ``lessons.jsonl``.

    The reader is intentionally tolerant: missing files and malformed JSONL
    rows return/skip rather than raising, preserving cold-start behavior.

    Args:
        kb_root: Optional KB root override. Defaults to :func:`_resolve_kb_root`.

    Returns:
        Parsed JSON objects from
        ``<kb_root>/framework_optimization/lessons.jsonl``.
    """
    root = kb_root or _resolve_kb_root()
    path = root / _FRAMEWORK_OPTIMIZATION_ROOT / "lessons.jsonl"
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


__all__ = [
    "KBFile",
    "DOMAIN_KEYWORDS",
    "list_domains",
    "get_domain_files",
    "select_kb",
    "contribute_to_kb",
    "path_for_framework",
    "synthesize_findings",
    "search_kb",
    "read_pr_ledger",
]
