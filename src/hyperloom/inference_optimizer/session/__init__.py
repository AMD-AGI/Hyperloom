# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Session paths, manifest, and single-optimizer lock.

tree-reform.MD §5.3/§2.4 (P2.4): merges the four former
``inference_optimizer`` top-level modules that jointly define "where a
session lives on disk" into one subpackage:

* ``paths.py`` — workspace/session-directory resolution + runtime-asset
  paths (``workspace_root``, ``session_dir``, ``make_session_dir``, ...).
* ``session_paths.py`` — per-session artifact path helpers (everything
  *inside* a session directory: ``manifest_path``, ``runs_dir``,
  ``reports_dir``, ``cortex_*``, ...).
* ``lock.py`` (formerly ``session_lock.py``) — the single-optimizer
  advisory ``flock`` guard (issue #592).
* ``manifest.py`` — the ``manifest.json`` writer/reader (schema v3).

This ``__init__.py`` intentionally does **not** re-export the submodules'
public symbols: callers import the fully-qualified submodule path (e.g.
``from hyperloom.inference_optimizer.session import paths`` or
``from hyperloom.inference_optimizer.session.paths import workspace_root``)
so the four former flat modules stay individually addressable rather than
collapsing into one ambiguous namespace.

Naming-collision fixups made during the merge (tree-reform.MD §2.4):

* ``paths.kernel_agent_runs_root`` (the kernel-agent tool output *root*,
  ``<sd>/kernel-agent``) was renamed to :func:`paths.kernel_agent_root` —
  it collided in name (but not meaning) with
  :func:`session_paths.kernel_agent_runs_root`, which returns the
  *``runs/`` subdirectory one level below* that root. The ``session_paths``
  name was kept as-is since it already describes the directory it returns.
* ``paths.optimizer_runs_dir`` was a byte-for-byte duplicate of
  :func:`session_paths.optimizer_runs_dir`; the ``paths.py`` copy was
  deleted and :mod:`session_paths` is now the single implementation.
"""
