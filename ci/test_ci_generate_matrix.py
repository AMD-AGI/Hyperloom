# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ci/generate_matrix.py (ci-config.yaml → GHA matrix)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import generate_matrix as gm  # noqa: E402


def _write_config(tmp_path: Path, models: list[dict]) -> Path:
    import yaml

    p = tmp_path / "ci-config.yaml"
    p.write_text(yaml.safe_dump({"models": models}), encoding="utf-8")
    return p


def test_entry_key_prefers_explicit_key():
    assert gm._entry_key({"key": "k1", "inferenceX_key": "ifx"}) == "k1"


def test_entry_key_falls_back_to_inferencex_key():
    assert gm._entry_key({"inferenceX_key": "ifx"}) == "ifx"


def test_generate_matrix_all_models(tmp_path: Path):
    cfg = _write_config(
        tmp_path,
        [
            {"inferenceX_key": "a"},
            {"key": "b", "inferenceX_key": "ignored"},
        ],
    )
    matrix = gm.generate_matrix(str(cfg))
    assert matrix == {"include": [{"key": "a"}, {"key": "b"}]}


def test_generate_matrix_selected_subset(tmp_path: Path):
    cfg = _write_config(
        tmp_path,
        [
            {"inferenceX_key": "a"},
            {"inferenceX_key": "b"},
            {"inferenceX_key": "c"},
        ],
    )
    matrix = gm.generate_matrix(str(cfg), selected_models="a,c")
    assert matrix == {"include": [{"key": "a"}, {"key": "c"}]}


def test_generate_matrix_selected_with_whitespace(tmp_path: Path):
    cfg = _write_config(tmp_path, [{"inferenceX_key": "a"}, {"inferenceX_key": "b"}])
    matrix = gm.generate_matrix(str(cfg), selected_models="  a  ")
    assert matrix == {"include": [{"key": "a"}]}


def test_generate_matrix_empty_models(tmp_path: Path):
    cfg = _write_config(tmp_path, [])
    assert gm.generate_matrix(str(cfg)) == {"include": []}


def test_main_writes_github_output(tmp_path: Path, monkeypatch, capsys):
    _write_config(tmp_path, [{"inferenceX_key": "a"}])
    out_file = tmp_path / "gh_out.txt"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INPUT_MODELS", "")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))
    gm.main()
    written = out_file.read_text(encoding="utf-8")
    assert written.startswith("matrix=")
    payload = json.loads(written.split("matrix=", 1)[1])
    assert payload == {"include": [{"key": "a"}]}


def test_main_prints_when_no_github_output(tmp_path: Path, monkeypatch, capsys):
    _write_config(tmp_path, [{"inferenceX_key": "z"}])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INPUT_MODELS", "")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    gm.main()
    stdout = capsys.readouterr().out
    assert json.loads(stdout) == {"include": [{"key": "z"}]}
