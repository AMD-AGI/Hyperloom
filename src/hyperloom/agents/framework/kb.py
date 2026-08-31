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
import uuid
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
_POD_LOCAL_WORKSPACE: str = "/workspace"


def _default_workspace_root() -> str:
    """Container images ship a writable ``/workspace``; bare metal off root has
    neither it nor permission to create it, so fall back to the caller's dir."""
    probe = _POD_LOCAL_WORKSPACE
    while not os.path.exists(probe) and probe != os.path.dirname(probe):
        probe = os.path.dirname(probe)
    if os.access(probe, os.W_OK):
        return _DEFAULT_WORKSPACE_ROOT
    return os.path.join(os.getcwd(), "session")


#: Withdrawn override. Only the reader honoured it, so setting it split the KB
#: in two. ``FRAMEWORK_AGENT_ROOT`` is deliberately absent: it means "where
#: this skill is installed", is used for other purposes, and never reached the
#: reader anyway because no installer exports it.
_REMOVED_KB_ROOT_ENV: str = "FRAMEWORK_AGENT_KB_DIR"


def prepare_kb_environment() -> None:
    """Start-up sequence for the framework KB: report the environment, then migrate.

    Both entry points that can reach this KB — the inference_optimizer preflight
    and the standalone ``fa`` CLI — call this one function, so a future start-up
    step is added in one place instead of being remembered in two.

    **Cannot stop a session.** Nothing this KB does at start-up is worth refusing
    to run over: the phase treats an unreadable or empty ledger as a cold start,
    and a session that disabled the phase never reads it at all. The guarantee is
    enforced here rather than left to each step, so a step added later inherits
    it; ``test_start_up_never_raises`` holds the line. Problems are announced at
    warning level, which is where an operator can act on them.
    """
    try:
        check_kb_configuration()
        migrate_legacy_partition_once()
    except Exception:  # noqa: BLE001 — start-up for an advisory KB may not fail a run
        _log.warning("FRAMEWORK KB: start-up preparation failed; continuing without it", exc_info=True)


def check_kb_configuration() -> None:
    """Report an environment naming a KB variable this build no longer reads.

    Announced, not rejected. The withdrawn override was dangerous because only
    the reader honoured it, so setting it split the KB in two without saying so.
    Now that the reader and ``kb_writeback`` both resolve through
    :func:`mutable_kb_root`, it is inert: the KB lands in the same correct place
    whether or not it is exported. Refusing to start would guard nothing and
    would strand a deployment still carrying it in a file someone forgot about.

    Names the resolved root as well as the replacement, so an operator who did
    mean to move the KB can see where it actually went.
    """
    if not os.environ.get(_REMOVED_KB_ROOT_ENV, "").strip():
        return
    _log.warning(
        "FRAMEWORK KB: %s is set but no longer read. It only ever redirected the reader, which is "
        "how reads and writes came to point at different places; it is now ignored and this KB "
        "resolves to %s. Use %s instead — that one moves both halves together.",
        _REMOVED_KB_ROOT_ENV,
        mutable_kb_root(),
        KB_ROOT_ENV,
    )


def migrate_legacy_partition_once() -> Path | None:
    """Carry the framework partition over from the legacy ``<workspace>/kb`` root.

    Until this KB was given its own directory, the writer put the ledger under
    ``<workspace>/kb/framework_optimization``. The writer was working, so every
    deployment that ever ran a FRAMEWORK phase has real data there — leaving it
    behind silently empties the dedup ledger and re-proposes PRs that already
    lost an accuracy gate.

    Never raises. A missing ledger is a cold start, which the phase handles, so
    this is a convenience and must not be able to stop a session — least of all
    a ``--no-framework-agent`` one that will never read this KB. A full disk or
    one unreadable file therefore costs a warning, not the run.

    Only runs when the destination has no framework data yet, so it can never
    overwrite a live partition. Skipped entirely when :data:`KB_ROOT_ENV` is
    set: the operator named a location, and the legacy default was never theirs.
    The source is left in place.

    Returns:
        The destination partition when data was migrated, else ``None``.
    """
    if os.environ.get(KB_ROOT_ENV, "").strip():
        return None

    workspace = Path(os.environ.get("USER_DATA_PATH", "").strip() or _default_workspace_root()).expanduser()
    source = workspace / _LEGACY_WORKSPACE_KB_DIRNAME / _FRAMEWORK_OPTIMIZATION_ROOT
    destination = framework_optimization_root()

    try:
        if not source.is_dir() or not any(source.iterdir()):
            return None
        if destination.exists() and any(destination.iterdir()):
            return None
        _copy_partition_atomically(source, destination)
    except Exception:  # noqa: BLE001 — a convenience copy may not stop the run
        _log.warning(
            "FRAMEWORK KB: could not carry the legacy partition over from %s; continuing with "
            "whatever is at %s. The FRAMEWORK phase treats a missing ledger as a cold start, so "
            "it may re-propose PRs it has already tried.",
            source,
            destination,
            exc_info=True,
        )
        return None

    _log.warning(
        "FRAMEWORK KB: migrated the legacy partition %s -> %s. The source is left in place; "
        "remove it once the new location looks right.",
        source,
        destination,
    )
    return destination


def _copy_partition_atomically(source: Path, destination: Path) -> None:
    """Copy ``source`` onto a not-yet-existing ``destination`` in one visible step.

    Staged and renamed so an interrupted copy cannot leave a half-populated
    destination that the next start-up would read as a live partition.

    The staging directory is unique per process and sits beside the KB root
    rather than inside it. Two starts sharing a ``USER_DATA_PATH`` (the
    orchestrator and the ``fa`` CLI, say) would otherwise stage onto the same
    path and delete each other's work; and ``list_domains`` reports every
    directory under the root as a framework domain, so one left behind by a
    crash would surface as a domain.

    The rename is what serialises concurrent migrations: whoever arrives second
    finds a non-empty destination and fails, which the caller downgrades.

    Args:
        source: The populated legacy partition.
        destination: The path to create; must not already hold data.
    """
    root = destination.parent
    staging = root.with_name(f"{root.name}.migrating-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        # symlinks=True: copy links as links. Following them would pull the
        # content of whatever they point at — possibly outside the workspace —
        # into a directory the KB reader serves as its own.
        shutil.copytree(source, staging, symlinks=True)
        root.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


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
    workspace = os.environ.get("USER_DATA_PATH", "").strip() or _default_workspace_root()
    return Path(workspace).expanduser() / _MUTABLE_KB_DIRNAME


def framework_optimization_root() -> Path:
    """The partition holding the lessons ledger, for reader and writer alike.

    ``kb_writeback`` resolves the ledger directory through here rather than
    re-spelling the leaf, so the two halves cannot come to disagree about the
    name the way they once disagreed about the root.

    Returns:
        ``<KB_ROOT>/framework_optimization``.
    """
    return mutable_kb_root() / _FRAMEWORK_OPTIMIZATION_ROOT


def _resolve_kb_root() -> Path:
    """Resolve the active KB root each call (so tests can monkeypatch env).

    Never raises: read paths swallow their own failures by design, so an
    exception here would be absorbed rather than surfaced. The withdrawn
    override is reported by :func:`check_kb_configuration` at start-up and
    otherwise ignored.

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

    from hyperloom.common.llm_attribution import sdk_env_overlay

    prompt = _build_llm_prompt(domain, findings)
    option_kwargs: dict[str, object] = {"model": model, "system_prompt": ""}
    overlay = sdk_env_overlay(component="framework", operation="synthesize_kb")
    if overlay:
        option_kwargs["env"] = overlay
    options = sdk.ClaudeAgentOptions(**option_kwargs)
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
    """Yield the non-empty text fragments of a claude_agent_sdk message."""
    from hyperloom.common.claude_oneshot import message_text

    yield from (fragment for fragment in message_text(message) if fragment)


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
