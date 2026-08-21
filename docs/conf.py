# Copyright Advanced Micro Devices, Inc. or its affiliates.
# SPDX-License-Identifier: MIT

# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add repo root and ``src`` for autodoc.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _path in (_repo_root, os.path.join(_repo_root, "src")):
    if os.path.isdir(_path):
        sys.path.insert(0, _path)

"""
html_theme is usually unchanged (rocm_docs_theme).
flavor defines the site header display, select the flavor for the corresponding portals
flavor options: rocm, rocm-docs-home, rocm-blogs, rocm-ds, instinct, ai-developer-hub, local, generic
"""

# Keep in sync with pyproject.toml [project].version.
version_number = "1.0.0b2"

html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "hyperloom",
    "header_title": f"Hyperloom {version_number}",
    "header_link": "https://rocm.docs.amd.com/projects/hyperloom/en/latest/",
    "version_list_link": False,
    "use_repository_button": True,
    "use_issues_button": True,
    "nav_secondary_items": {
        "GitHub": "https://github.com/AMD-AGI/Hyperloom",
        "Community": False,
        "Blogs": "https://rocm.blogs.amd.com/",
        "ROCm Developer Hub": "https://www.amd.com/en/developer/resources/rocm-hub.html",
        "Infinity Hub": "https://www.amd.com/en/developer/resources/infinity-hub.html",
        "Support": "https://github.com/AMD-AGI/Hyperloom/issues/new/choose",
    },
    "link_main_doc": False,
}

# ``use_repository_button`` makes sphinx-book-theme call get_repo_parts(context),
# which walks ["github", "bitbucket", "gitlab"] looking for a ``<provider>_url``
# key and returns None -- not a tuple -- when it finds none. Its caller unpacks
# that return value unconditionally, so a missing key is a build-time
# ``TypeError: cannot unpack non-iterable NoneType object`` rather than a
# skipped button.
#
# Nothing in the stack supplies the key on its own: pydata-sphinx-theme has the
# provider defaults, but only inside the "edit this page" path, and
# rocm-docs-core defaults ``use_edit_page_button`` off.
#
# Injected per page rather than declared as ``html_context``. Assigning that
# config directly is enough to fix the local build, but it suppresses
# rocm-docs-core's own defaults (it only sets them when the user has not) and it
# overwrites the context Read the Docs injects for its version switcher -- which
# turned a green RTD build red. Adding the keys at render time leaves both
# untouched, and ``setdefault`` means an explicit value elsewhere still wins.
_SOURCE_REPO_CONTEXT = {
    "github_url": "https://github.com",
    "github_user": "AMD-AGI",
    "github_repo": "Hyperloom",
    "github_version": "main",
    "doc_path": "docs",
}


def setup(app):
    """Supply the repository keys sphinx-book-theme unpacks without checking."""

    def _add_source_repo_context(_app, _pagename, _templatename, context, _doctree):
        for key, value in _SOURCE_REPO_CONTEXT.items():
            context.setdefault(key, value)

    # Ahead of sphinx-book-theme's own html-page-context handlers (default 500).
    app.connect("html-page-context", _add_source_repo_context, priority=100)


# Article info display
setting_all_article_info = True
all_article_info_os = ["linux"]
all_article_info_author = ""

# for PDF output on Read the Docs
project = "Hyperloom"
author = "Advanced Micro Devices, Inc."
copyright = "2025 Advanced Micro Devices, Inc."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"  # Defines Table of Content structure definition path

exclude_patterns = [
    "_build",
    "_templates",
]

suppress_warnings = [
    # Autosummary API pages contain toctrees that external-toc warns about.
    "etoc.toctree",
]

"""
Doxygen Settings
Ensure Doxyfile is located at docs/doxygen.
If the component does not need doxygen, delete this section for optimal build time
"""
# doxygen_root = "doxygen"
# doxysphinx_enabled = True
# doxygen_project = {
#    "name": "doxygen",
#    "path": "doxygen/xml",
# }

# Add more addtional package accordingly
extensions = [
    "rocm_docs",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
]

html_title = f"{project} {version_number} documentation"

external_projects_current_project = "Hyperloom"
