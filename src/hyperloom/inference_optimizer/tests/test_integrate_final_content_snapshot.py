# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Integrate must apply from FINAL patch bytes, not the fusion pristine snapshot.

``snapshot_dir`` names two different things on the two sides of the nomination
wire: the fusion exporter records the pre-authoring *pristine* tree it diffed
against, while ``apply_kernel_patch`` reads the field as the *post-patch final*
contents it copies from. Handing the former to the latter loses a real KEEP --
the pristine tree is missing every module the fusion authored, so the apply
pre-flight refuses the whole patch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel.request_handlers import _final_content_snapshot

MODELS = "python/sglang/srt/models"
PRISTINE_SRC = "def forward(x):\n    return x\n"
FUSED_MODULE = f"{MODELS}/qwen3_fused_llm_ar_residual_rmsnorm_fp8quant_wiring.py"

# The shape forge-fusion emits for one sibling: the wiring edit plus the module
# that sibling authored.
PATCH = f"""\
diff --git a/{MODELS}/qwen3.py b/{MODELS}/qwen3.py
--- a/{MODELS}/qwen3.py
+++ b/{MODELS}/qwen3.py
@@ -1,2 +1,3 @@
+FUSED = 1
 def forward(x):
     return x
diff --git a/{FUSED_MODULE} b/{FUSED_MODULE}
new file mode 100644
--- /dev/null
+++ b/{FUSED_MODULE}
@@ -0,0 +1 @@
+def fused(): return 1
"""


@pytest.fixture
def repo(tmp_path):
    """A framework checkout at the pre-fusion base, as integrate finds it."""
    root = tmp_path / "sglang"
    (root / MODELS).mkdir(parents=True)
    (root / MODELS / "qwen3.py").write_text(PRISTINE_SRC, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=t@t.local", "-c", "user.name=t", "commit", "-qm", "base"],
        check=True,
        capture_output=True,
    )
    return root


@pytest.fixture
def patch_file(tmp_path):
    p = tmp_path / "fusion_llm_ar_residual_rmsnorm_fp8quant_wiring.patch"
    p.write_text(PATCH, encoding="utf-8")
    return p


@pytest.fixture
def pristine(tmp_path):
    """What the fusion exporter records: the tree BEFORE authoring.

    Two entries, exactly as the failing e2e produced -- the un-edited source and
    the fused-sibling inventory. The authored module is absent by construction.
    """
    root = tmp_path / ".pristine_llm_ar_residual_rmsnorm_fp8quant_wiring"
    (root / MODELS).mkdir(parents=True)
    (root / MODELS / "qwen3.py").write_text(PRISTINE_SRC, encoding="utf-8")
    (root / ".fused_siblings").write_text("", encoding="utf-8")
    return root


def _writes(snapshot: str) -> dict[str, str]:
    root = Path(snapshot)
    return {
        rel: (root / rel).read_text(encoding="utf-8")
        for rel in (f"{MODELS}/qwen3.py", FUSED_MODULE)
        if (root / rel).is_file()
    }


def test_pristine_snapshot_is_replaced_with_materialized_final_content(repo, patch_file, pristine):
    """Regression: the fusion sibling's pristine dir must not be used as final content."""
    # Precondition: this is what the pending record carries, and it cannot serve
    # the apply -- the authored module simply is not in it.
    assert not (pristine / FUSED_MODULE).exists()

    resolved = _final_content_snapshot(
        patch_path=str(patch_file),
        snapshot_dir=str(pristine),
        repo_root=str(repo),
    )

    assert resolved != str(pristine), "pristine snapshot was passed through as final content"
    writes = _writes(resolved)
    # Every path the patch writes now has its POST-patch bytes.
    assert set(writes) == {f"{MODELS}/qwen3.py", FUSED_MODULE}
    assert writes[f"{MODELS}/qwen3.py"].startswith("FUSED = 1")
    assert "def fused()" in writes[FUSED_MODULE]


def test_missing_snapshot_is_materialized(repo, patch_file):
    """No snapshot at all (the collective lane's shape) also materializes."""
    resolved = _final_content_snapshot(
        patch_path=str(patch_file),
        snapshot_dir=None,
        repo_root=str(repo),
    )
    assert resolved and set(_writes(resolved)) == {f"{MODELS}/qwen3.py", FUSED_MODULE}


def test_a_complete_snapshot_is_left_alone(repo, patch_file, tmp_path):
    """A dir that already holds final bytes for every write is used as-is.

    Re-materializing would be wasted work, and would discard a snapshot a
    producer built deliberately.
    """
    good = tmp_path / "already_final"
    (good / MODELS).mkdir(parents=True)
    (good / MODELS / "qwen3.py").write_text("FUSED = 1\n" + PRISTINE_SRC, encoding="utf-8")
    (good / FUSED_MODULE).write_text("def fused(): return 1\n", encoding="utf-8")

    assert _final_content_snapshot(
        patch_path=str(patch_file),
        snapshot_dir=str(good),
        repo_root=str(repo),
    ) == str(good)


def test_non_patch_artifact_is_untouched(repo, tmp_path):
    """Legacy full-source mode (``patch_path`` is a .py) must not be rerouted."""
    src = tmp_path / "optimized.py"
    src.write_text("x = 1\n", encoding="utf-8")
    assert (
        _final_content_snapshot(patch_path=str(src), snapshot_dir=None, repo_root=str(repo)) is None
    )


def test_unmaterializable_patch_falls_back_to_the_original(repo, tmp_path, pristine):
    """A patch that will not apply keeps the caller's value.

    Swallowing the error here would replace apply's precise complaint with a
    vaguer one from this helper.
    """
    bad = tmp_path / "conflict.patch"
    bad.write_text(
        f"diff --git a/{MODELS}/qwen3.py b/{MODELS}/qwen3.py\n"
        f"--- a/{MODELS}/qwen3.py\n"
        f"+++ b/{MODELS}/qwen3.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-this line is not in the base\n"
        "+replacement\n"
        " def forward(x):\n",
        encoding="utf-8",
    )
    assert _final_content_snapshot(
        patch_path=str(bad),
        snapshot_dir=str(pristine),
        repo_root=str(repo),
    ) == str(pristine)
