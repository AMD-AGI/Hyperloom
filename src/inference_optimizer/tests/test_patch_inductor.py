"""Tests for ``scripts/patch_inductor`` — Plan A (IR-6 soft).

The post-Plan-A behaviour is:

* By default ``validate_argv`` returns the (possibly empty) list of
  violations and emits a WARNING line on stderr per violation. ``main``
  always returns 0 even when violations fire, so a kernel-opt loop
  isn't aborted by a borderline argv.
* When ``INFERENCE_OPTIMIZER_IR6_STRICT=1`` is set, the legacy hard
  contract is restored: ``validate_argv`` raises
  :class:`PatchInductorError`, ``main`` returns 2, and stderr carries
  the diagnostic.
"""
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
# validate_argv — soft default (Plan A)
# ---------------------------------------------------------------------------
def test_soft_validate_warns_on_missing_target_file(capsys, monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_IR6_STRICT", raising=False)
    errors = validate_argv(["--best-config", "cfg.json"])
    err = capsys.readouterr().err
    assert "target-file" in err
    # Soft mode never raises; errors list is empty (only populated in strict).
    assert errors == []


def test_soft_validate_warns_on_cache_dir(capsys, monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_IR6_STRICT", raising=False)
    validate_argv(["--target-file", "k.py", "--cache-dir", "/tmp"])
    err = capsys.readouterr().err
    assert "cache-dir" in err


def test_soft_validate_warns_on_missing_best_config_with_block_size(
    capsys, monkeypatch
):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_IR6_STRICT", raising=False)
    validate_argv(["--target-file", "k.py", "--tuning-keys", "block_size"])
    err = capsys.readouterr().err
    assert "best-config" in err


def test_soft_validate_no_warning_when_full_args(capsys, monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_IR6_STRICT", raising=False)
    errors = validate_argv([
        "--target-file", "k.py",
        "--best-config", "cfg.json",
        "--tuning-keys", "block_size,num_warps",
    ])
    err = capsys.readouterr().err
    assert errors == []
    assert "patch_inductor" not in err  # no warnings emitted


def test_soft_validate_no_warning_when_no_tuning_keys(capsys, monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_IR6_STRICT", raising=False)
    validate_argv(["--target-file", "k.py"])
    err = capsys.readouterr().err
    assert "patch_inductor" not in err


# ---------------------------------------------------------------------------
# validate_argv — strict mode (env override) preserves legacy behaviour
# ---------------------------------------------------------------------------
def test_strict_mode_raises_on_missing_target_file(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_IR6_STRICT", "1")
    with pytest.raises(PatchInductorError, match="target-file"):
        validate_argv(["--best-config", "cfg.json"])


def test_strict_mode_raises_on_cache_dir(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_IR6_STRICT", "1")
    with pytest.raises(PatchInductorError, match="cache-dir"):
        validate_argv(["--target-file", "k.py", "--cache-dir", "/tmp"])


def test_strict_mode_raises_on_missing_best_config(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_IR6_STRICT", "1")
    with pytest.raises(PatchInductorError, match="best-config"):
        validate_argv([
            "--target-file", "k.py",
            "--tuning-keys", "num_warps,fanout",
        ])


def test_strict_mode_recognises_truthy_values(monkeypatch):
    for v in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_IR6_STRICT", v)
        with pytest.raises(PatchInductorError):
            validate_argv(["--best-config", "cfg.json"])


# ---------------------------------------------------------------------------
# main — soft default returns 0 even on warnings
# ---------------------------------------------------------------------------
def test_main_returns_zero_on_valid_invocation(
    capsys, tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_IR6_STRICT", raising=False)
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
    assert manifest["ir6_strict"] is False


def test_main_returns_zero_with_warning_when_target_missing(
    capsys, monkeypatch
):
    """Plan A: missing --target-file → WARN + rc=0 (legacy: rc=2)."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_IR6_STRICT", raising=False)
    rc = main(["--best-config", "cfg.json"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "target-file" in captured.err
    manifest = json.loads(captured.out.strip())
    assert manifest["target_file"] is None


def test_main_strict_mode_returns_two_when_target_missing(capsys, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_IR6_STRICT", "1")
    rc = main(["--best-config", "cfg.json"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "target-file" in err


def test_main_dry_run_flag_in_manifest(capsys, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_IR6_STRICT", raising=False)
    target = tmp_path / "x.py"
    target.write_text("x", encoding="utf-8")
    rc = main(["--target-file", str(target), "--dry-run"])
    out = capsys.readouterr().out.strip()
    manifest = json.loads(out)
    assert rc == 0
    assert manifest["dry_run"] is True


def test_ir6_strict_enabled_helper(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_IR6_STRICT", raising=False)
    assert pi.ir6_strict_enabled() is False
    monkeypatch.setenv("INFERENCE_OPTIMIZER_IR6_STRICT", "1")
    assert pi.ir6_strict_enabled() is True
    monkeypatch.setenv("INFERENCE_OPTIMIZER_IR6_STRICT", "no")
    assert pi.ir6_strict_enabled() is False
