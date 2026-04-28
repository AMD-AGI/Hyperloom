"""Backend abstraction for agent reactors.

Real backends call Claude / Codex SDKs; ``MockBackend`` returns scripted
:class:`Intent` lists so we can run end-to-end smoke tests without API keys.
See DESIGN §5.1 / §10.5.
"""
from .base import Backend, BackendCall, BackendError
from .mock import MockBackend, ScriptStep

# ClaudeBackend / CodexBackend are imported lazily because they pull in
# optional SDK dependencies (``claude-agent-sdk`` / ``openai``) at
# construction time. Callers do
# ``from inference_optimizer.orchestrator.backends import ClaudeBackend``
# and trigger a clear ``BackendError`` only when the dependency is missing
# *and* they actually instantiate the backend.
try:  # pragma: no cover — exercised by tests via direct import
    from .claude import ClaudeBackend
except ImportError as _exc:  # pragma: no cover
    ClaudeBackend = None  # type: ignore[assignment]
    _claude_import_error = _exc
else:
    _claude_import_error = None

try:  # pragma: no cover — exercised by tests via direct import
    from .codex import CodexBackend
except ImportError as _exc:  # pragma: no cover
    CodexBackend = None  # type: ignore[assignment]
    _codex_import_error = _exc
else:
    _codex_import_error = None

__all__ = [
    "Backend",
    "BackendCall",
    "BackendError",
    "ClaudeBackend",
    "CodexBackend",
    "MockBackend",
    "ScriptStep",
]
