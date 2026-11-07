"""Arbor.

``ARBOR_SITE``: a directory of packages Arbor itself imports (``claude-agent-sdk`` and what it needs), put on this
process's ``sys.path`` only. A host whose environment cannot install into
the interpreter's site-packages (a read-only venv in a sandbox) installs there with ``pip install --target`` and names
it here. It never goes on ``PYTHONPATH``: a server the agent starts inherits Arbor's environment, and the target's
pydantic, starlette, uvicorn and anyio would shadow the image's. A process that imports no Arbor module never reads it.
"""

import os as _os
import sys as _sys

_site = _os.environ.get("ARBOR_SITE", "").strip()
if _site and _os.path.isdir(_site) and _site not in _sys.path:
    # After the script's directory or the working directory (sys.path[0]), so Arbor's own checkout stays first,
    # and before the interpreter's site-packages, which the target's packages were resolved against.
    _sys.path.insert(1, _site)
del _os, _sys, _site
