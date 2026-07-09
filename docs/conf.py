# Copyright Advanced Micro Devices, Inc. or its affiliates.
# SPDX-License-Identifier: MIT

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
#  https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# robustness_agent and framework_agent use a src/ layout.
# inference_optimizer lives at the repo root and needs no extra path entry.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo_root, "robustness-agent", "src"))
sys.path.insert(0, os.path.join(_repo_root, "framework-agent", "src"))

"""
html_theme is usually unchanged (rocm_docs_theme).
flavor defines the site header display, select the flavor for the corresponding portals
flavor options: rocm, rocm-docs-home, rocm-blogs, rocm-ds, instinct, ai-developer-hub, local, generic
"""

# Dynamically extract component version
#with open("../anc/version.txt", encoding="utf-8") as f:
#    version_full = f.read().strip()
    # Only get the major and minor
version_number = "0.1.0"

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


# This section turns on/off article info
setting_all_article_info = True
all_article_info_os = ["linux"]
all_article_info_author = ""

# for PDF output on Read the Docs
project = "Hyperloom"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"  # Defines Table of Content structure definition path

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
