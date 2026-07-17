# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session paths, manifest, and single-optimizer lock.

Subpackage defining "where a session lives on disk":

* ``paths.py`` — workspace/session-directory resolution + runtime-asset
  paths (``workspace_root``, ``session_dir``, ``make_session_dir``, ...).
* ``session_paths.py`` — per-session artifact path helpers (everything
  *inside* a session directory: ``manifest_path``, ``runs_dir``,
  ``reports_dir``, ``cortex_*``, ...).
* ``lock.py`` — the single-optimizer advisory ``flock`` guard.
* ``manifest.py`` — the ``manifest.json`` writer/reader (schema v3).

This ``__init__.py`` intentionally does **not** re-export the submodules'
public symbols: callers import the fully-qualified submodule path so the
submodules stay individually addressable rather than collapsing into one
ambiguous namespace.
"""
