"""Backend abstraction for agent reactors.

Real backends call Claude / Codex SDKs; ``MockBackend`` returns scripted
:class:`Intent` lists so we can run end-to-end smoke tests without API keys.
See DESIGN §5.1 / §10.5.
"""
from .base import Backend, BackendCall, BackendError
from .mock import MockBackend, ScriptStep

# ClaudeBackend is imported lazily because it pulls in the optional
# ``claude-agent-sdk`` dependency at construction time. Callers do
# ``from inference_optimizer.orchestrator.backends import ClaudeBackend``
# and trigger a clear ``BackendError`` if the SDK isn't installed.
try:  # pragma: no cover — exercised by tests via direct import
    from .claude import ClaudeBackend
except ImportError as _exc:  # pragma: no cover
    ClaudeBackend = None  # type: ignore[assignment]
    _claude_import_error = _exc
else:
    _claude_import_error = None

__all__ = [
    "Backend",
    "BackendCall",
    "BackendError",
    "ClaudeBackend",
    "MockBackend",
    "ScriptStep",
]
