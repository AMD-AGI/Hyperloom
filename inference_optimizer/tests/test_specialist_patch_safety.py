# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Patch-safety grounding tests — focus on the ``missing_target`` guard that
drops patches whose modify/delete targets are absent from the framework tree
(a hallucinated layout that can never apply)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from inference_optimizer.orchestrator.specialist_patch_safety import (
    GROUND_APPLIES,
    GROUND_MISSING_TARGET,
    GROUND_NOT_DIFF,
    GROUND_STALE,
    PatchSafetyReport,
    ground_patch_text,
    patch_file_targets,
    patch_targets_missing,
    vet_patches,
)


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t.local",
        GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t.local",
    )
    subprocess.run(["git", "init", "-b", "main", str(path)],
                   check=True, capture_output=True, env=env)
    pkg = path / "vllm" / "model_executor" / "models"
    pkg.mkdir(parents=True)
    (pkg / "qwen3.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."],
                   check=True, capture_output=True, env=env)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"],
                   check=True, capture_output=True, env=env)


_PATCH_EXISTING = """\
diff --git a/vllm/model_executor/models/qwen3.py b/vllm/model_executor/models/qwen3.py
--- a/vllm/model_executor/models/qwen3.py
+++ b/vllm/model_executor/models/qwen3.py
@@ -1,2 +1,2 @@
 def f():
-    return 1
+    return 2
"""

_PATCH_MISSING = """\
diff --git a/vllm/distributed/device_communicators/cuda_communicator.py b/vllm/distributed/device_communicators/cuda_communicator.py
--- a/vllm/distributed/device_communicators/cuda_communicator.py
+++ b/vllm/distributed/device_communicators/cuda_communicator.py
@@ -1,1 +1,1 @@
-OLD
+NEW
"""

_PATCH_NEW_FILE = """\
diff --git a/vllm/compilation/passes/fusion/new_pass.py b/vllm/compilation/passes/fusion/new_pass.py
new file mode 100644
--- /dev/null
+++ b/vllm/compilation/passes/fusion/new_pass.py
@@ -0,0 +1,1 @@
+# brand new file
"""


def test_patch_file_targets_parses_pairs():
    pairs = patch_file_targets(_PATCH_NEW_FILE)
    assert pairs == [("/dev/null", "b/vllm/compilation/passes/fusion/new_pass.py")]


def test_targets_missing_flags_absent_modify_target(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    assert patch_targets_missing(_PATCH_MISSING, repo) == [
        "a/vllm/distributed/device_communicators/cuda_communicator.py",
    ]


def test_targets_missing_passes_existing_file(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    assert patch_targets_missing(_PATCH_EXISTING, repo) == []


def test_targets_missing_exempts_new_file_creation(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    # ``--- /dev/null`` => pure creation, never a missing-target violation.
    assert patch_targets_missing(_PATCH_NEW_FILE, repo) == []


def test_ground_patch_text_missing_target_is_garbage(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    res = ground_patch_text(_PATCH_MISSING, base_checkout=repo)
    assert res.verdict == GROUND_MISSING_TARGET
    assert res.is_garbage
    assert "cuda_communicator.py" in res.detail


def test_ground_patch_text_applies_for_existing_target(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    res = ground_patch_text(_PATCH_EXISTING, base_checkout=repo)
    assert res.verdict == GROUND_APPLIES
    assert not res.is_garbage


def test_ground_patch_text_stale_kept_when_target_exists(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    # File exists but hunk context is wrong => stale (kept, not garbage):
    stale = _PATCH_EXISTING.replace("    return 1", "    return 999")
    res = ground_patch_text(stale, base_checkout=repo)
    assert res.verdict == GROUND_STALE
    assert not res.is_garbage


def test_vet_patches_drops_missing_target_keeps_stale(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    miss = tmp_path / "miss.patch"
    miss.write_text(_PATCH_MISSING, encoding="utf-8")
    ok = tmp_path / "ok.patch"
    ok.write_text(_PATCH_EXISTING, encoding="utf-8")
    kept, dropped, grounding = vet_patches(
        [str(miss), str(ok)], base_checkout=repo,
    )
    assert kept == [str(ok)]
    assert len(dropped) == 1
    assert dropped[0]["verdict"] == GROUND_MISSING_TARGET
    assert grounding[str(miss)] == GROUND_MISSING_TARGET

    report = PatchSafetyReport(kept_patches=kept, dropped=dropped, grounding=grounding)
    notes = " ".join(report.notes())
    assert "patch_safety_missing_target" in notes


def test_ground_patch_text_unchecked_without_base():
    # No base checkout => structural-only; never a missing-target false negative.
    res = ground_patch_text(_PATCH_MISSING, base_checkout=None)
    assert res.verdict != GROUND_MISSING_TARGET


def test_non_diff_still_not_diff(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    res = ground_patch_text("not a patch at all", base_checkout=repo)
    assert res.verdict == GROUND_NOT_DIFF
