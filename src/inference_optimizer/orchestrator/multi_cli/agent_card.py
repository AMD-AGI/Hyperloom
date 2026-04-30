"""AgentCard — declarative description of one CLI agent.

In ``--mode multi-cli`` every agent role lives at
``src/inference_optimizer/agents/<name>/`` with at least:

    agent_card.yaml         — capabilities + launch hints (this file)
    system_prompt.md        — what the CLI sees on every restart
    scripts/                — (optional) agent-private helper scripts

The launcher discovers all cards on startup and spawns one CLI process
per ``status: enabled`` card. The Router uses the same cards to scope
which inbox/outbox files exist, which capabilities each agent advertises
and which backend to dispatch a message to.

Schema (subset enforced in v0)
------------------------------

    name:               str             # must equal the directory name
    role:               str             # one of {executor, critic, triage, kernel}
                                        # (used for AgentRole lookup + PolicyGate)
    backend:            "claude" | "codex" | "mock"
    capabilities:       list[str]       # informational; for plug-and-play discovery
    allowed_modes:      list[str]       # ExecutionMode values
    enabled:            bool            (default true)
    system_prompt:      str | null      # filename relative to the card dir;
                                        # default: ``system_prompt.md``
    restart_policy:
        max_restarts:     int            (default 50)
        backoff_seconds:  int            (default 15)
        continue_flag:    bool           (default true; passes ``--continue``
                                          on restart for Claude)
    inbox_filename:     str             (default ``inbox.jsonl``)
    outbox_filename:    str             (default ``outbox.jsonl``)
    extra:              dict             # free-form per-card extras
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..execution_mode import ExecutionMode


# ---------------------------------------------------------------------------
class AgentCardError(RuntimeError):
    """Raised when an agent_card.yaml fails to load or validate."""


_VALID_ROLES: frozenset[str] = frozenset(
    # v0.4 MVP roster (standalone_agent_design §13.1):
    # executor / critic / triage / kernel — all Claude-backed.
    {"executor", "critic", "triage", "kernel"}
)
_VALID_BACKENDS: frozenset[str] = frozenset({"claude", "codex", "mock", "mock-cli"})

DEFAULT_INBOX = "inbox.jsonl"
DEFAULT_OUTBOX = "outbox.jsonl"
DEFAULT_PROMPT = "system_prompt.md"
DEFAULT_MAX_RESTARTS = 50
DEFAULT_BACKOFF_S = 15


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RestartPolicy:
    max_restarts: int = DEFAULT_MAX_RESTARTS
    backoff_seconds: int = DEFAULT_BACKOFF_S
    continue_flag: bool = True


@dataclass(frozen=True)
class AgentCard:
    """Loaded ``agent_card.yaml`` plus path bookkeeping."""

    name: str
    role: str
    backend: str
    card_path: Path
    card_dir: Path
    capabilities: tuple[str, ...] = ()
    allowed_modes: tuple[ExecutionMode, ...] = ()
    enabled: bool = True
    system_prompt: str = DEFAULT_PROMPT
    inbox_filename: str = DEFAULT_INBOX
    outbox_filename: str = DEFAULT_OUTBOX
    restart_policy: RestartPolicy = field(default_factory=RestartPolicy)
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def system_prompt_path(self) -> Path:
        """Absolute path to this agent's system prompt file."""
        return self.card_dir / self.system_prompt

    def applies_to(self, mode: ExecutionMode) -> bool:
        """Return True if this agent should be active in ``mode``."""
        if not self.allowed_modes:
            return True
        return mode in self.allowed_modes


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_agent_card(card_path: Path) -> AgentCard:
    """Load and validate one ``agent_card.yaml`` file."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - tested via dep
        raise AgentCardError(
            "PyYAML is required to load agent_card.yaml; pip install PyYAML"
        ) from exc

    p = Path(card_path)
    if not p.is_file():
        raise AgentCardError(f"agent_card not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AgentCardError(f"failed to parse {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise AgentCardError(f"{p}: top-level YAML must be a mapping")

    return _build_card(raw, card_path=p)


def discover_agent_cards(agents_root: Path) -> dict[str, AgentCard]:
    """Scan ``agents_root/<name>/agent_card.yaml`` and return ``{name: card}``.

    Disabled cards (``enabled: false``) are still returned so callers can
    surface them in diagnostics; filter via :meth:`AgentCard.enabled` at
    the call site.

    Raises :class:`AgentCardError` on duplicate name, malformed file, or
    schema mismatch — discovery should fail fast at startup.
    """
    root = Path(agents_root)
    if not root.is_dir():
        return {}
    found: dict[str, AgentCard] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith((".", "_")):
            continue
        cardfile = child / "agent_card.yaml"
        if not cardfile.is_file():
            continue
        card = load_agent_card(cardfile)
        if card.name in found:
            raise AgentCardError(
                f"duplicate agent name {card.name!r} (file={cardfile})"
            )
        found[card.name] = card
    return found


# ---------------------------------------------------------------------------
def _build_card(raw: dict[str, Any], *, card_path: Path) -> AgentCard:
    name = str(raw.get("name", "")).strip()
    expected = card_path.parent.name
    if not name:
        raise AgentCardError(f"{card_path}: 'name' is required")
    if name != expected:
        raise AgentCardError(
            f"{card_path}: name {name!r} does not match directory {expected!r}"
        )

    role = str(raw.get("role", "")).strip()
    if role not in _VALID_ROLES:
        raise AgentCardError(
            f"{card_path}: role {role!r} must be one of {sorted(_VALID_ROLES)}"
        )

    backend = str(raw.get("backend", "")).strip()
    if backend not in _VALID_BACKENDS:
        raise AgentCardError(
            f"{card_path}: backend {backend!r} must be one of {sorted(_VALID_BACKENDS)}"
        )

    enabled = bool(raw.get("enabled", True))
    capabilities = _to_str_tuple(raw.get("capabilities", ()), card_path, "capabilities")

    modes_raw = raw.get("allowed_modes", ())
    modes = tuple(_resolve_mode(m, card_path) for m in (modes_raw or ()))

    rp_raw = raw.get("restart_policy", {}) or {}
    if not isinstance(rp_raw, dict):
        raise AgentCardError(f"{card_path}: restart_policy must be a mapping")
    restart_policy = RestartPolicy(
        max_restarts=int(rp_raw.get("max_restarts", DEFAULT_MAX_RESTARTS)),
        backoff_seconds=int(rp_raw.get("backoff_seconds", DEFAULT_BACKOFF_S)),
        continue_flag=bool(rp_raw.get("continue_flag", True)),
    )

    extra_raw = raw.get("extra", {}) or {}
    if not isinstance(extra_raw, dict):
        raise AgentCardError(f"{card_path}: extra must be a mapping")

    return AgentCard(
        name=name,
        role=role,
        backend=backend,
        card_path=card_path,
        card_dir=card_path.parent,
        capabilities=capabilities,
        allowed_modes=modes,
        enabled=enabled,
        system_prompt=str(raw.get("system_prompt") or DEFAULT_PROMPT),
        inbox_filename=str(raw.get("inbox_filename") or DEFAULT_INBOX),
        outbox_filename=str(raw.get("outbox_filename") or DEFAULT_OUTBOX),
        restart_policy=restart_policy,
        extra=dict(extra_raw),
    )


def _resolve_mode(value: Any, card_path: Path) -> ExecutionMode:
    try:
        return ExecutionMode(str(value))
    except ValueError as exc:
        raise AgentCardError(
            f"{card_path}: unknown allowed_modes entry {value!r}"
        ) from exc


def _to_str_tuple(
    value: Any, card_path: Path, field_name: str
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    raise AgentCardError(
        f"{card_path}: {field_name} must be a list, got {type(value).__name__}"
    )


__all__ = [
    "AgentCard",
    "AgentCardError",
    "DEFAULT_BACKOFF_S",
    "DEFAULT_INBOX",
    "DEFAULT_MAX_RESTARTS",
    "DEFAULT_OUTBOX",
    "DEFAULT_PROMPT",
    "RestartPolicy",
    "discover_agent_cards",
    "load_agent_card",
]
