# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A run that produced nothing usable must leave the framework as it found it.

Observed live: a failed pass left a pip-installed vllm carrying code that never
passed validation, and it had to be restored by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from kernelforge.fusion import command as cli
from kernelforge.fusion.command import (
    _discard_failed_attempt,
    _live_file_restored,
    _needs_discard,
    _snapshot_fusion_source,
)
from kernelforge.fusion.command import main as cli_main
from kernelforge.fusion.loop import LoopResult
from kernelforge.fusion.models import ValidationResult

PRISTINE = "import torch\n\n\ndef forward(x):\n    return x\n"
AUTHORED = "import torch\nfrom .qwen3_fused import fused\n\n\ndef forward(x):\n    return fused(x)\n"


def _tree(tmp_path, *, with_framework_fused: bool = False, source_name: str = "qwen3.py"):
    """A framework-like install with the model source, plus an output dir."""
    root = tmp_path / "site-packages"
    source = root / "vllm" / "model_executor" / "models" / source_name
    source.parent.mkdir(parents=True)
    source.write_text(PRISTINE, encoding="utf-8")
    if with_framework_fused:
        # Ships with the framework: matches the fused-name marker but is not ours.
        (source.parent / "llama_fused_moe.py").write_text("SHIPPED = 1\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    return root, source, out


def test_the_source_is_restored(tmp_path):
    root, source, out = _tree(tmp_path)
    pristine = _snapshot_fusion_source(str(root), str(source), out)
    source.write_text(AUTHORED, encoding="utf-8")

    _discard_failed_attempt(str(root), str(source), out, pristine)
    assert source.read_text(encoding="utf-8") == PRISTINE


def test_a_missing_snapshot_does_not_swallow_the_body_exception(tmp_path):
    """``return`` in ``finally`` would eat this, which is why CodeQL flags it.

    Compile-pass uses this manager around live edits. If the snapshot itself
    failed (no file, unreadable), a later exception must still be the caller's
    to handle -- otherwise a failed smoke looks like success.
    """
    missing = tmp_path / "does-not-exist.py"
    with pytest.raises(RuntimeError, match="smoke failed"):
        with _live_file_restored(str(missing)):
            raise RuntimeError("smoke failed")


def test_the_attempt_is_preserved_before_being_discarded(tmp_path):
    """Throwing the work away outright would lose the only record of it."""
    root, source, out = _tree(tmp_path)
    pristine = _snapshot_fusion_source(str(root), str(source), out)
    source.write_text(AUTHORED, encoding="utf-8")

    _discard_failed_attempt(str(root), str(source), out, pristine)
    kept = out / ".failed" / "vllm" / "model_executor" / "models" / "qwen3.py"
    assert kept.is_file()
    assert kept.read_text(encoding="utf-8") == AUTHORED


def test_author_created_modules_are_removed(tmp_path):
    root, source, out = _tree(tmp_path)
    pristine = _snapshot_fusion_source(str(root), str(source), out)
    source.write_text(AUTHORED, encoding="utf-8")
    created = source.parent / "qwen3_fused.py"
    created.write_text("def fused(x):\n    return x\n", encoding="utf-8")

    _discard_failed_attempt(str(root), str(source), out, pristine)
    assert not created.exists()


def test_framework_modules_matching_the_marker_are_left_alone(tmp_path):
    """Absence from the snapshot is the test, not the file name."""
    root, source, out = _tree(tmp_path, with_framework_fused=True)
    pristine = _snapshot_fusion_source(str(root), str(source), out)
    shipped = source.parent / "llama_fused_moe.py"
    source.write_text(AUTHORED, encoding="utf-8")

    _discard_failed_attempt(str(root), str(source), out, pristine)
    assert shipped.is_file()
    assert shipped.read_text(encoding="utf-8") == "SHIPPED = 1\n"


def test_unrelated_files_are_never_touched(tmp_path):
    root, source, out = _tree(tmp_path)
    pristine = _snapshot_fusion_source(str(root), str(source), out)
    sibling = source.parent / "llama.py"
    sibling.write_text("LLAMA = 1\n", encoding="utf-8")
    source.write_text(AUTHORED, encoding="utf-8")

    _discard_failed_attempt(str(root), str(source), out, pristine)
    assert sibling.read_text(encoding="utf-8") == "LLAMA = 1\n"


def test_a_missing_snapshot_makes_it_a_no_op(tmp_path):
    """Without a pristine reference there is nothing safe to restore to."""
    root, source, out = _tree(tmp_path)
    source.write_text(AUTHORED, encoding="utf-8")

    _discard_failed_attempt(str(root), str(source), out, "")
    assert source.read_text(encoding="utf-8") == AUTHORED
    assert not (out / ".failed").exists()


def test_the_pristine_snapshot_still_defaults_to_its_own_directory(tmp_path):
    """The subdir parameter must not have moved the normal snapshot."""
    root, source, out = _tree(tmp_path)
    returned = _snapshot_fusion_source(str(root), str(source), out)
    assert Path(returned) == out / ".pristine"
    assert (out / ".pristine" / "vllm" / "model_executor" / "models" / "qwen3.py").is_file()


def test_a_framework_module_survives_a_failed_snapshot_copy(tmp_path, monkeypatch):
    """Copying a sibling may fail without failing the run, so it cannot be the judge.

    Snapshotting a sibling is deliberately non-fatal. If the rollback then treats
    "absent from the snapshot" as "author wrote it", one unlucky copy -- a
    permission error, a full disk -- is enough to delete a framework file out of
    site-packages.
    """
    root, source, out = _tree(tmp_path, with_framework_fused=True)
    shipped = source.parent / "llama_fused_moe.py"
    real_copy = cli.shutil.copy2

    def flaky_copy(src, dst, *args, **kwargs):
        if Path(src).name == shipped.name:
            raise OSError("disk full")
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(cli.shutil, "copy2", flaky_copy)
    pristine = _snapshot_fusion_source(str(root), str(source), out)
    assert pristine, "the main source must still be snapshotted"
    assert not (Path(pristine) / "vllm" / "model_executor" / "models" / shipped.name).exists()
    source.write_text(AUTHORED, encoding="utf-8")

    monkeypatch.setattr(cli.shutil, "copy2", real_copy)
    _discard_failed_attempt(str(root), str(source), out, pristine)
    assert shipped.is_file(), "a framework module must not be deleted over a copy failure"


def test_the_model_source_is_never_the_file_that_gets_deleted(tmp_path):
    """The source's own name can match the marker, and it was just restored.

    The inventory lists the source's SIBLINGS -- export needs to tell a new module
    from a shipped one, and the source is neither. "Absent from the inventory"
    therefore also describes the source itself, so a model file named like
    ``fused_moe.py`` would be restored from the snapshot and then deleted.
    """
    root, source, out = _tree(tmp_path, source_name="fused_moe.py")
    pristine = _snapshot_fusion_source(str(root), str(source), out)
    source.write_text(AUTHORED, encoding="utf-8")

    _discard_failed_attempt(str(root), str(source), out, pristine)
    assert source.is_file(), "the restored model source must not then be deleted"
    assert source.read_text(encoding="utf-8") == PRISTINE


def test_without_a_recorded_inventory_nothing_is_deleted(tmp_path):
    """An unknown starting state cannot justify deleting from site-packages."""
    root, source, out = _tree(tmp_path, with_framework_fused=True)
    pristine = _snapshot_fusion_source(str(root), str(source), out)
    (Path(pristine) / ".fused_siblings").unlink()
    source.write_text(AUTHORED, encoding="utf-8")
    created = source.parent / "qwen3_fused.py"
    created.write_text("def fused(x):\n    return x\n", encoding="utf-8")

    _discard_failed_attempt(str(root), str(source), out, pristine)
    assert source.read_text(encoding="utf-8") == PRISTINE, "the source is still restored"
    assert created.is_file(), "without an inventory, deleting is a guess"


def test_no_inventory_means_nothing_is_claimed_as_author_created(tmp_path):
    """Deleting from site-packages on a guess is not worth risking."""
    from kernelforge.fusion.command import _author_created_modules

    root = tmp_path / "site-packages"
    models = root / "pkg"
    models.mkdir(parents=True)
    source = models / "toylm.py"
    source.write_text("x\n", encoding="utf-8")
    (models / "toylm_fused.py").write_text("y\n", encoding="utf-8")

    assert _author_created_modules(str(source), str(tmp_path / "nope")) == []


# --- which outcomes leave the framework dirty ----------------------------- #
def test_a_rejected_attempt_needs_discarding():
    assert _needs_discard(False, None) is True


def test_an_accepted_attempt_with_a_patch_is_already_restored():
    assert _needs_discard(True, SimpleNamespace(patch="diff --git a/x b/x\n")) is False


def test_an_accepted_attempt_whose_export_came_back_empty_needs_discarding():
    """The run looks successful right up to there being nothing to show for it.

    This is the branch that used to fall through: not a failure, so the failed
    path was skipped, and no patch, so the restore was skipped too -- leaving the
    framework carrying edits that no artifact records.
    """
    assert _needs_discard(True, None) is True
    assert _needs_discard(True, SimpleNamespace(patch="")) is True


# --- the same thing through the CLI --------------------------------------- #
FRAMEWORK_SOURCE = """\
import torch
from sglang.srt.layers.layernorm import RMSNorm


class ToyLMDecoderLayer(torch.nn.Module):
    def forward(self, hidden_states, residual):
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        return hidden_states
"""


def _fake_framework(tmp_path):
    """An sglang-shaped tree under --framework-root, so nothing real is touched.

    ``--framework-root`` is honoured ahead of the installed package, and the
    ``__init__.py`` chain pins the package root inside tmp_path, which keeps the
    rollback away from whatever sglang this machine happens to have installed.
    """
    root = tmp_path / "framework"
    models = root / "sglang" / "srt" / "models"
    models.mkdir(parents=True)
    for pkg in (root / "sglang", root / "sglang" / "srt", models):
        (pkg / "__init__.py").write_text("", encoding="utf-8")
    source = models / "toylm.py"
    source.write_text(FRAMEWORK_SOURCE, encoding="utf-8")
    return root, source


def _model_dir(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "toylm", "hidden_size": 2048, "num_attention_heads": 16}),
        encoding="utf-8",
    )
    return model


def _trace(tmp_path):
    path = tmp_path / "decode.trace.json"
    events = [
        {"cat": "kernel", "name": "Cijk_gemm", "ts": 0, "dur": 40},
        {"cat": "kernel", "name": "add_rmsnorm_quant_kernel", "ts": 200, "dur": 12},
        {"cat": "kernel", "name": "vectorized_elementwise CUDAFunctor_add", "ts": 400, "dur": 10},
        {"cat": "kernel", "name": "vectorized_elementwise silu", "ts": 600, "dur": 8},
    ]
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return path


def test_a_kept_run_restores_the_framework(tmp_path, monkeypatch):
    """The ordinary success path: patch exported, tree put back.

    This is the case a refactor of the surrounding branches can silently drop --
    the restore hangs off the same condition as the export, so folding it under
    another branch leaves the author's edits sitting in the framework. On the
    multi-patch path each sibling is exported inside ``on_keep`` and the shared
    tree is restored to base by the autoloop's ``shadow.reset_to_base()`` (a
    git-level wipe of EVERY keeper's edits), not by the single-patch
    ``restore_exported_changes``. The invariant this guards -- the author's edits
    must not be left sitting in the framework -- is asserted on the file content.
    """
    root, source = _fake_framework(tmp_path)
    baseline = source.read_text(encoding="utf-8")

    def fake_run_fusion_loop(recipes, *, framework, campaign_fn, config, on_keep=None):
        source.write_text(
            FRAMEWORK_SOURCE.replace("import torch\n", "import torch\nFUSED = 1\n"),
            encoding="utf-8",
        )
        return LoopResult(
            kept=True,
            best=ValidationResult(
                correctness_passed=True,
                max_abs_err=0.001,
                rtol=0.02,
                kernel_speedup=1.42,
                eager_us=100.0,
                fused_us=70.4,
                kept=True,
                note="ok",
            ),
            best_recipe=recipes[0],
            history=[],
            experience_path=None,
            termination_reason="kept",
        )

    monkeypatch.setattr(cli, "run_fusion_loop", fake_run_fusion_loop)
    monkeypatch.setattr(cli, "_author_baseline_harness", lambda *a, **k: (True, ""))
    monkeypatch.setattr(cli, "serving_smoke", lambda *a, **k: (True, ""))

    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli_main,
        [
            "--trace",
            str(_trace(tmp_path)),
            "--model-path",
            str(_model_dir(tmp_path)),
            "--framework",
            "sglang",
            "--output-dir",
            str(out),
            "--framework-root",
            str(root),
            "--author",
        ],
    )
    assert result.exit_code == 0, result.output
    assert source.read_text(encoding="utf-8") == baseline, (
        "a KEPT run must put the framework back (multi-patch restores via shadow reset)"
    )
