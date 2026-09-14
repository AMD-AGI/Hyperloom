# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A controller publication is git-applied before validation; integrate must not re-apply it.

``integrate_controller_patches`` runs ``git apply`` on the working tree and only
then calls the validator, which passes ``preapplied_git_patch``. The patch is a
unified diff, so it cannot serve as replacement source; the already-applied
worktree is handed to apply as its final-content snapshot instead, which keeps
the cache invalidation, multi-node fan-out and rebuild that the re-baseline
depends on.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.kernel import request_handlers as krh
from hyperloom.orchestrator.state.shared_state import SharedState

KERNEL_REL = "python/sglang/kernels/ops/attention/decode_attention.py"
BASE_SRC = "BLOCK_N = 16\n\n\ndef _fwd_grouped_kernel_stage1():\n    return BLOCK_N\n"
PATCHED_SRC = "BLOCK_N = 32\n\n\ndef _fwd_grouped_kernel_stage1():\n    return BLOCK_N\n"
PATCH = f"""\
diff --git a/{KERNEL_REL} b/{KERNEL_REL}
--- a/{KERNEL_REL}
+++ b/{KERNEL_REL}
@@ -1,4 +1,4 @@
-BLOCK_N = 16
+BLOCK_N = 32
\x20
\x20
 def _fwd_grouped_kernel_stage1():
"""


class ReachedRebaseline(Exception):
    """Raised in place of the re-baseline so the test stops at that seam."""


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    sd = make_session_dir()
    state = SharedState.load_or_init(sd)
    state.baseline_tput = 511.0
    state.save(sd)
    return sd


@pytest.fixture
def repo(tmp_path) -> Path:
    root = tmp_path / "sglang"
    (root / KERNEL_REL).parent.mkdir(parents=True)
    (root / KERNEL_REL).write_text(BASE_SRC, encoding="utf-8")
    for args in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.email=t@t.local", "-c", "user.name=t", "commit", "-qm", "base"],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root


@pytest.fixture
def patch_file(tmp_path) -> Path:
    p = tmp_path / "change.patch"
    p.write_text(PATCH, encoding="utf-8")
    return p


@pytest.fixture
def applied(repo, patch_file) -> Path:
    """The tree as the validator finds it: the diff is already in the worktree."""
    subprocess.run(
        ["git", "-C", str(repo), "apply", str(patch_file)],
        check=True,
        capture_output=True,
    )
    assert (repo / KERNEL_REL).read_text(encoding="utf-8") == PATCHED_SRC
    return repo


@pytest.fixture
def stop_at_rebaseline(monkeypatch):
    def boom(*_args, **_kwargs):
        raise ReachedRebaseline

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.baseline.BaselineExecutor",
        boom,
    )


def _controller_payload(applied: Path, patch_file: Path) -> dict:
    """Exactly what ``_default_validator`` sends for a controller publication."""
    return {
        "kernel_id": "kernel:forge-loop:fwd_grouped_kernel_stage1:sglang:0.5.17:triton:mi355x",
        "patch_path": str(patch_file),
        "target_file": str(applied / KERNEL_REL),
        "repo": str(applied),
        "patch_write_paths": [KERNEL_REL],
    }


@pytest.fixture
def through_apply(monkeypatch):
    """Record what the real apply was handed and what it returned."""
    real = krh._maybe_apply_kernel_patch
    calls: list[tuple[dict, dict]] = []

    def record(payload, **kwargs):
        result = real(payload, **kwargs)
        calls.append((payload, result))
        return result

    monkeypatch.setattr(krh, "_maybe_apply_kernel_patch", record)
    return calls


async def test_preapplied_patch_reaches_the_rebaseline(session_dir, applied, patch_file, stop_at_rebaseline):
    """The KEEP must be measured, not refused at apply."""
    with pytest.raises(ReachedRebaseline):
        await krh.integrate_handler(
            _controller_payload(applied, patch_file),
            session_dir=session_dir,
            preapplied_git_patch=True,
        )


async def test_preapplied_patch_still_goes_through_the_real_apply(
    session_dir, applied, patch_file, stop_at_rebaseline, through_apply
):
    """Bypassing apply also bypasses the invalidation, fan-out and rebuild it owns."""
    with pytest.raises(ReachedRebaseline):
        await krh.integrate_handler(
            _controller_payload(applied, patch_file),
            session_dir=session_dir,
            preapplied_git_patch=True,
        )

    assert len(through_apply) == 1
    _payload, result = through_apply[0]
    assert result["status"] == "ok", result
    # A real manifest is what drives revert, finalize and the rebuild check.
    assert Path(result["manifest_path"]).is_file()


async def test_the_apply_reads_the_worktree_bytes_not_the_diff(
    session_dir, applied, patch_file, stop_at_rebaseline, through_apply
):
    """The diff is a manifest of changed paths; routing it in as source corrupts the file."""
    with pytest.raises(ReachedRebaseline):
        await krh.integrate_handler(
            _controller_payload(applied, patch_file),
            session_dir=session_dir,
            preapplied_git_patch=True,
        )

    payload, _result = through_apply[0]
    snapshot = Path(payload["snapshot_dir"]) / KERNEL_REL
    assert snapshot.read_text(encoding="utf-8") == (applied / KERNEL_REL).read_text(encoding="utf-8")
    assert snapshot.read_text(encoding="utf-8").startswith("BLOCK_N = 32")


async def test_the_patched_file_survives_the_apply(session_dir, applied, patch_file, stop_at_rebaseline):
    """Landing the snapshot must leave the measured bytes exactly as published."""
    with pytest.raises(ReachedRebaseline):
        await krh.integrate_handler(
            _controller_payload(applied, patch_file),
            session_dir=session_dir,
            preapplied_git_patch=True,
        )

    assert (applied / KERNEL_REL).read_text(encoding="utf-8") == PATCHED_SRC


async def test_a_diff_without_the_flag_is_still_refused(session_dir, applied, patch_file, stop_at_rebaseline):
    """Scope guard: only the controller's pre-applied contract skips apply."""
    result = await krh.integrate_handler(_controller_payload(applied, patch_file), session_dir=session_dir)

    assert result["status"] == "failed"
    assert result["error_class"] == "apply_failed"


async def test_a_payload_cannot_claim_to_be_preapplied(session_dir, applied, patch_file, stop_at_rebaseline):
    """An agent's integrate params reach the payload verbatim, so the payload cannot carry this trust."""
    payload = {**_controller_payload(applied, patch_file), "_preapplied_git_patch": True}

    result = await krh.integrate_handler(payload, session_dir=session_dir)

    assert result["status"] == "failed"
    assert result["error_class"] == "apply_failed"
