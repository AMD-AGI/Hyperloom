"""Tests for ``scripts/patch_inductor`` — IMPL-CHECKLIST §9.9‒9.11."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.scripts import patch_inductor as pi
from inference_optimizer.scripts.patch_inductor import (
    PatchInductorError,
    main,
    validate_argv,
)


# ---------------------------------------------------------------------------
# validate_argv
# ---------------------------------------------------------------------------
def test_validate_requires_target_file():
    with pytest.raises(PatchInductorError, match="target-file"):
        validate_argv(["--best-config", "cfg.json"])


def test_validate_rejects_cache_dir():
    with pytest.raises(PatchInductorError, match="cache-dir"):
        validate_argv(["--target-file", "k.py", "--cache-dir", "/tmp"])


def test_validate_requires_best_config_when_block_size():
    with pytest.raises(PatchInductorError, match="best-config"):
        validate_argv([
            "--target-file", "k.py",
            "--tuning-keys", "block_size",
        ])


def test_validate_requires_best_config_when_num_warps():
    with pytest.raises(PatchInductorError, match="best-config"):
        validate_argv([
            "--target-file", "k.py",
            "--tuning-keys", "num_warps,fanout",
        ])


def test_validate_passes_with_full_args():
    validate_argv([
        "--target-file", "k.py",
        "--best-config", "cfg.json",
        "--tuning-keys", "block_size,num_warps",
    ])


def test_validate_passes_when_no_tuning_keys():
    validate_argv(["--target-file", "k.py"])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def test_main_returns_zero_on_valid_invocation(capsys, tmp_path: Path):
    target = tmp_path / "kernel.py"
    target.write_text("# pretend kernel\n", encoding="utf-8")
    rc = main([
        "--target-file", str(target),
        "--best-config", "cfg.json",
        "--tuning-keys", "block_size",
    ])
    out = capsys.readouterr().out.strip()
    manifest = json.loads(out)
    assert rc == 0
    assert manifest["target_file"] == str(target)
    assert manifest["tuning_keys"] == ["block_size"]


def test_main_returns_two_when_target_missing(capsys):
    rc = main(["--best-config", "cfg.json"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "target-file" in err


def test_main_dry_run_flag_in_manifest(capsys, tmp_path: Path):
    target = tmp_path / "x.py"
    target.write_text("x", encoding="utf-8")
    rc = main(["--target-file", str(target), "--dry-run"])
    out = capsys.readouterr().out.strip()
    manifest = json.loads(out)
    assert rc == 0
    assert manifest["dry_run"] is True
