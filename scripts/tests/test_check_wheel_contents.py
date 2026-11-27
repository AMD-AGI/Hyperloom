# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Inspector page check in scripts/check_wheel_contents.py: self-contained, no inline scripts."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("check_wheel_contents", _ROOT / "scripts" / "check_wheel_contents.py")
check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check)

# The shape Vite writes: a module script and a stylesheet, both relative.
_VITE_PAGE = """<!doctype html>
<html lang="en">
  <head>
    <title>Hyperloom Inspector</title>
    <script type="module" crossorigin src="./assets/index-QPjJ1TBn.js"></script>
    <link rel="stylesheet" crossorigin href="./assets/index-DbQn1dPJ.css">
  </head>
  <body><div id="app" data-src="//not-a-load"></div></body>
</html>
"""


def _errors(tmp_path, page: str | None) -> list[str]:
    wheel = tmp_path / "w.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        if page is not None:
            zf.writestr(check._INSPECTOR_PAGE, page)
    with zipfile.ZipFile(wheel) as zf:
        return check._check_inspector_page_is_self_contained(zf, zf.namelist())


def test_a_vite_page_passes(tmp_path):
    assert _errors(tmp_path, _VITE_PAGE) == []


@pytest.mark.parametrize(
    "tag",
    [
        '<script type="module" src="https://cdn.example.com/x.js"></script>',
        "<link rel=stylesheet href=//cdn.example.com/x.css>",
        "<img SRC='http://example.com/x.png'>",
    ],
)
def test_a_resource_from_another_host_fails(tmp_path, tag):
    errors = _errors(tmp_path, _VITE_PAGE.replace("</head>", f"{tag}</head>"))
    assert len(errors) == 1
    assert "another host" in errors[0]


@pytest.mark.parametrize("tag", ["<script>boot()</script>", '<script type="module" data-src="x">boot()</script>'])
def test_an_inline_script_fails(tmp_path, tag):
    errors = _errors(tmp_path, _VITE_PAGE.replace("</body>", f"{tag}</body>"))
    assert len(errors) == 1
    assert "inline <script>" in errors[0]


def test_a_missing_page_is_left_to_the_required_files_check(tmp_path):
    assert _errors(tmp_path, None) == []
    assert check._INSPECTOR_PAGE in check._REQUIRED_FILES
