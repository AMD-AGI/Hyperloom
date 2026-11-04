"""Keep a tuner's spawned workers able to time their kernels.

aiter's tuners time each candidate with torch's profiler in workers started with ``spawn``. Once the tuner's own
process has profiled (its compare report does), rocprofiler-sdk exports ``ROCPROFILER_REGISTER_LIBRARY`` into its
environment, every worker spawned after inherits it, and a worker's profiler then records no GPU event: each candidate
reads 0 us. A ``sitecustomize`` first on the tuner's ``PYTHONPATH`` drops the variable as each interpreter starts,
before torch loads, and then runs the ``sitecustomize`` it shadows, if there is one.
"""

from __future__ import annotations

import functools
import os
import tempfile
from pathlib import Path

INHERITED_PROFILER_VARS = ("ROCPROFILER_REGISTER_LIBRARY",)

_SITECUSTOMIZE = f"""\
import importlib.machinery as _machinery
import importlib.util as _util
import os as _os
import sys as _sys

for _name in {INHERITED_PROFILER_VARS!r}:
    _os.environ.pop(_name, None)

_here = _os.path.dirname(_os.path.abspath(__file__))
_rest = [p for p in _sys.path if _os.path.abspath(p or ".") != _here]
_spec = _machinery.PathFinder.find_spec("sitecustomize", _rest)
if _spec is not None and _spec.loader is not None:
    _module = _util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
"""


@functools.cache
def _site_dir() -> Path:
    directory = Path(tempfile.mkdtemp(prefix="gemm-tune-site-"))
    (directory / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    return directory


def without_inherited_profiler(env: dict[str, str]) -> dict[str, str]:
    """``env`` for a tuner whose spawned workers must not inherit a profiler registration."""

    for name in INHERITED_PROFILER_VARS:
        env.pop(name, None)
    site = str(_site_dir())
    paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join([site, *(p for p in paths if p != site)])
    return env
