"""Sphinx configuration for the Hyperloom documentation build.

This configuration wires up :mod:`sphinx.ext.autodoc` (plus
:mod:`sphinx.ext.napoleon` for the Google-style docstrings used across the
codebase) and renders the result with the Furo theme. The API reference pages
are generated automatically from the in-code docstrings via recursive
``autosummary`` stubs, so no per-module ``.rst`` files need to be maintained by
hand.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# conf.py lives in <repo>/docs; the importable source packages live at the repo
# root and under the per-agent ``src`` / package directories. Add them all to
# sys.path so autodoc can import the modules it documents.
DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
for _path in (
    REPO_ROOT,
    REPO_ROOT / "robustness-agent" / "src",
    REPO_ROOT / "critic-agent",
    REPO_ROOT / "framework-agent" / "src",
):
    if _path.is_dir():
        sys.path.insert(0, str(_path))

# -- Project information -----------------------------------------------------
project = "Hyperloom"
author = "AMD AGI"
copyright = f"{datetime.now():%Y}, {author}"
# Keep in sync with pyproject.toml [project].version.
release = "0.6.0"
version = "0.6"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",        # pull docstrings out of the source
    "sphinx.ext.autosummary",    # auto-generate per-module API stub pages
    "sphinx.ext.napoleon",       # understand Google- (and NumPy-) style docstrings
    "sphinx.ext.viewcode",       # add "[source]" links next to documented objects
    "sphinx.ext.intersphinx",    # cross-link to the Python/3rd-party docs
    "sphinx.ext.todo",
    "myst_parser",               # render the existing Markdown guides in docs/
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Treat both reStructuredText and Markdown as documentation sources so the
# hand-written guides already in docs/ are picked up alongside the .rst files.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate = True          # build stub pages from the autosummary directives
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"     # render type hints in the description, not the signature
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
autoclass_content = "class"           # use the class docstring (not __init__) for the class body

# Many runtime modules import heavy / hardware-bound third-party packages at
# import time. Mock them so the documentation build does not require a full
# runtime environment (GPUs, SDK credentials, etc.).
autodoc_mock_imports = [
    # Third-party runtime dependencies.
    "claude_agent_sdk",
    "openai",
    "httpx",
    "yaml",
    "markdownify",
    "cachetools",
    "respx",
    "numpy",
    "pandas",
    "matplotlib",
    "torch",
    "triton",
    "ray",
    # POSIX-only stdlib modules imported by the runtime. Mocking them lets the
    # docs build on non-POSIX hosts (e.g. Windows CI) without import errors.
    "fcntl",
    "termios",
    "grp",
    "pwd",
    "resource",
]

# -- napoleon (Google-style docstrings) --------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True
napoleon_preprocess_types = True

# -- MyST (Markdown) ---------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "linkify", "tasklist"]
myst_heading_anchors = 3
# The repo's Markdown guides are top-level docs; suppress noisy warnings about
# non-consecutive header levels in those hand-written files.
suppress_warnings = ["myst.header"]

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- todo --------------------------------------------------------------------
todo_include_todos = True

# -- HTML output (Furo) ------------------------------------------------------
html_theme = "furo"
html_title = f"{project} {version}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/AMD-AGI/Hyperloom",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
}

# Fail the build loudly in CI when a cross-reference cannot be resolved.
nitpicky = bool(os.environ.get("SPHINX_NITPICKY"))
