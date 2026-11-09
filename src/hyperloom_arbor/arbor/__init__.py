"""Arbor.

``~/.arbor/settings.env`` (or the file ``ARBOR_SETTINGS`` names): a site's own values for Arbor's ``ARBOR_*``
settings, one ``KEY=VALUE`` per line (``#`` comments, an ``export`` prefix and quotes allowed), read when Arbor is
imported, for each key the environment does not set already. The engine carries no site's values: the collectors' share
root, a site's own projects and the like live there, or in the environment. Keys other than ``ARBOR_*`` are ignored.

``ARBOR_SITE``: a directory of packages Arbor itself imports (``claude-agent-sdk`` and what it needs), put on this
process's ``sys.path`` only. A host whose environment cannot install into
the interpreter's site-packages (a read-only venv in a sandbox) installs there with ``pip install --target`` and names
it here. It never goes on ``PYTHONPATH``: a server the agent starts inherits Arbor's environment, and the target's
pydantic, starlette, uvicorn and anyio would shadow the image's. A process that imports no Arbor module never reads it.
"""

import os as _os
import sys as _sys


def _read_settings() -> None:
    path = _os.environ.get("ARBOR_SETTINGS") or _os.path.join(_os.path.expanduser("~"), ".arbor", "settings.env")
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        key = key[len("export "):].strip() if key.startswith("export ") else key
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key.startswith("ARBOR_") and key not in _os.environ:
            _os.environ[key] = value


_read_settings()

_site = _os.environ.get("ARBOR_SITE", "").strip()
if _site and _os.path.isdir(_site) and _site not in _sys.path:
    # After the script's directory or the working directory (sys.path[0]), so Arbor's own checkout stays first,
    # and before the interpreter's site-packages, which the target's packages were resolved against.
    _sys.path.insert(1, _site)
del _os, _sys, _site, _read_settings
