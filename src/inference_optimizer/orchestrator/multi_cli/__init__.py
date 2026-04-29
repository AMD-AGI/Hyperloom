"""Multi-CLI agent runtime — A2A v0.

In ``--mode multi-cli`` each agent role runs as an independent CLI process
(``claude --print --continue`` for Claude roles, ``codex`` with explicit
conversation log for Codex roles), spawned via the marathon-style
``while ... claude --print --continue`` restart-loop launcher.

Agent processes communicate through JSONL **inbox/outbox** files under
``$SESSION_DIR/agents/<name>/``. The :class:`MultiCLIRouter` watches every
agent's outbox, runs each emitted intent through the existing
:class:`PolicyGate`, applies the side-effect via the Conductor's normal
``_handle_intent`` pipeline, and mirrors all resulting bus events back
into the recipients' inbox files.

This way:

* Phase 1 (this module) ships a *passive mirror* — the Router can be wired
  into the existing single-process Conductor to also write JSONL alongside
  SQLite events, with zero behavioural change for the in-process reactors.
* Phase 2 swaps in the launcher + replaces the in-process reactor for one
  role at a time, validating cross-process PolicyGate enforcement.
* Phase 3 migrates every role and turns ``--mode multi-cli`` into the
  default.

Public surface
--------------

- :class:`Envelope` — single canonical JSONL envelope (works for both
  inbox messages and outbox intents via the ``kind`` discriminator).
- :class:`AgentCard` — schema + auto-discovery for each agent's
  ``agent_card.yaml``.
- :class:`AgentInbox` / :class:`AgentOutbox` — append-only JSONL files
  with monotonic ``seq`` + filesystem cursor.
- :class:`MultiCLIRouter` — the JSONL ↔ SQLite bridge (Phase 1: passive
  mirror; Phase 2: also dispatches outbox intents through PolicyGate).

References
----------

- Plan: ``.cursor/plans/multi-cli-agents-a2a_*.plan.md``
- DESIGN ADR-2 / ADR-12: persistent agents + A2A
- DESIGN ADR-4: long-process memory is a mirage — we use restart-loop
  + ``--continue`` instead, with file-based memory in the session dir.
- Marathon: ``marathon/launcher/run.sh`` lines 419-477 (the
  ``write_pane_script`` blueprint we generalise here).

Stability
---------

* :class:`Envelope` JSONL schema, :class:`AgentCard` YAML schema, and
  the inbox/outbox / cursor file naming convention are **stable v0**;
  we will only add fields, never remove or rename, until v1 ships.
* :class:`MultiCLIRouter` Python signature is stable; internal cursor
  formats may evolve as we move from polling to inotify.
* :class:`MultiCLILauncher` shell template is **best-effort** — operators
  who deploy via tmux/systemd/k8s should treat the generated script as
  a starting point and copy it into their own supervisor.
* :class:`CodexConversationLog` budget defaults are tunable per-deployment;
  the JSONL turn shape (``role/content/ts/attempt``) is stable.

Phase 4 (post-Phase-3): we keep ``--transport single-proc`` as the
default and supported transport. Multi-CLI is opt-in via
``--transport multi-cli`` / ``--transport hybrid --cli-agents ...``.
JSONL stays the A2A wire format; we have no plan to upgrade to Google
A2A spec until external-agent integration becomes a concrete need.
"""

from __future__ import annotations

from .agent_card import (
    AgentCard,
    AgentCardError,
    RestartPolicy,
    discover_agent_cards,
    load_agent_card,
)
from .codex_continuity import (
    CodexConversationLog,
    CodexPromptComposer,
    ConversationTurn,
    DEFAULT_CHAR_BUDGET,
    DEFAULT_CONVERSATION_FILENAME,
    naive_summariser,
    update_after_restart,
)
from .envelope import (
    Envelope,
    EnvelopeError,
    EnvelopeKind,
    write_envelope,
    read_envelopes,
    iter_new_envelopes,
)
from .launcher import (
    LauncherError,
    MultiCLILauncher,
    StagedAgent,
)
from .router import (
    MultiCLIRouter,
    RouterError,
    agent_inbox_path,
    agent_outbox_path,
    agent_session_dir,
)


__all__ = [
    "AgentCard",
    "AgentCardError",
    "CodexConversationLog",
    "CodexPromptComposer",
    "ConversationTurn",
    "DEFAULT_CHAR_BUDGET",
    "DEFAULT_CONVERSATION_FILENAME",
    "Envelope",
    "EnvelopeError",
    "EnvelopeKind",
    "LauncherError",
    "MultiCLILauncher",
    "MultiCLIRouter",
    "RestartPolicy",
    "RouterError",
    "StagedAgent",
    "agent_inbox_path",
    "agent_outbox_path",
    "agent_session_dir",
    "discover_agent_cards",
    "iter_new_envelopes",
    "load_agent_card",
    "naive_summariser",
    "read_envelopes",
    "update_after_restart",
    "write_envelope",
]
