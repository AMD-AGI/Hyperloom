# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).parent.resolve()
ROOT_DIR = DOCS_DIR.parent


def copy_rtd_file(src_path: Path, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest_path)
    print(f"Copied {src_path} -> {dest_path}")


gh_changelog_path = ROOT_DIR / "CHANGELOG.md"
rtd_changelog_path = DOCS_DIR / "release" / "changelog.md"
copy_rtd_file(gh_changelog_path, rtd_changelog_path)

# Mark the consolidated changelog as orphan to prevent Sphinx from warning about missing toctree entries
with open(rtd_changelog_path, "r+", encoding="utf-8") as file:
    content = file.read()
    file.seek(0)
    file.write(":orphan:\n" + content)

latex_engine = "xelatex"
latex_elements = {
    "fontpkg": r"""
\usepackage{tgtermes}
\usepackage{tgheros}
\renewcommand\ttdefault{txtt}
"""
}

# Keep in sync with pyproject.toml [project].version.
version_number = "1.0.0a1"

external_projects_current_project = "rocm"
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "rocm",
    "link_main_doc": False,
    "repository_url": "https://github.com/AMD-AGI/Hyperloom",
    "use_repository_button": True,
    "use_issues_button": True,
}
html_title = f"Hyperloom {version_number}"

setting_all_article_info = False
all_article_info_os = ["linux"]
all_article_info_author = ""

external_toc_path = "./sphinx/_toc.yml"

extensions = [
    "rocm_docs",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
]



html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "rocm.docs.amd.com")
html_context = {}
if os.environ.get("READTHEDOCS", "") == "True":
    html_context["READTHEDOCS"] = True


numfig = False

exclude_patterns = [
    "_build",
    "_templates",
    "exclude/**",
    "**/include/**",
    "**/images/**",
]
