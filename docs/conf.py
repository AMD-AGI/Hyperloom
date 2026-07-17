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
version_number = "0.8.0"

html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "generic",
    "header_title": f"Hyperloom {version_number}",
    "header_link": False,
    "version_list_link": False,
    "nav_secondary_items": {
        "GitHub": "https://github.com/AMD-AGI/Hyperloom",
        "Community": False,
        "Blogs": "https://rocm.blogs.amd.com/",
        "ROCm Developer Hub": "https://www.amd.com/en/developer/resources/rocm-hub.html",
        "Instinct™ Docs": "https://instinct.docs.amd.com/",
        "Infinity Hub": "https://www.amd.com/en/developer/resources/infinity-hub.html",
        "Support": "https://github.com/AMD-AGI/Hyperloom/issues/new/choose",
    },
    "link_main_doc": False,
}


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
