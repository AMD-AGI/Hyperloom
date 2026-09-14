# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A controller publication is git-applied before validation; integrate must not re-apply it.

``integrate_controller_patches`` runs ``git apply`` on the working tree and only
then calls the validator, which passes ``preapplied_git_patch``. The patch itself
is a unified diff, so routing it back through the whole-file
``apply_kernel_patch`` contract fails and the measured KEEP is lost. Skipping the
apply still owes its aiter stale-binary guards, which the re-baseline reads off
the apply result.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.kernel import request_handlers as krh
from hyperloom.orchestrator.state.shared_state import SharedState

KERNEL_REL = "python/sglang/kernels/ops/attention/decode_attention.py"
BASE_SRC = "BLOCK_N = 16\n\n\ndef _fwd_grouped_kernel_stage1():\n    return BLOCK_N\n"
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
    assert (repo / KERNEL_REL).read_text(encoding="utf-8").startswith("BLOCK_N = 32")
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
        "patch_write_paths": [KERNEL_REL],
    }


async def test_preapplied_patch_reaches_the_rebaseline(session_dir, applied, patch_file, stop_at_rebaseline):
    """The KEEP must be measured, not refused at apply."""
    with pytest.raises(ReachedRebaseline):
        await krh.integrate_handler(
            _controller_payload(applied, patch_file),
            session_dir=session_dir,
            preapplied_git_patch=True,
        )


async def test_preapplied_patch_is_never_reapplied(session_dir, applied, patch_file, stop_at_rebaseline, monkeypatch):
    """The diff must not be routed back through the whole-file apply contract."""
    calls: list[dict] = []

    def record(payload, **kwargs):
        calls.append(payload)
        return {"status": "failed", "error": "should not be called"}

    monkeypatch.setattr(krh, "_maybe_apply_kernel_patch", record)

    with pytest.raises(ReachedRebaseline):
        await krh.integrate_handler(
            _controller_payload(applied, patch_file),
            session_dir=session_dir,
            preapplied_git_patch=True,
        )

    assert calls == []


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


@pytest.fixture
def stub_aiter_invalidation(monkeypatch):
    """Report the target as a cpp_itfs kernel, recording what was invalidated."""
    tool = krh._load_apply_tool()
    invalidated: list[str] = []

    def fake_jit(target_file, _backup_dir, **_kwargs):
        invalidated.append(f"jit:{target_file}")
        return {"status": "clean", "reason": "aiter jit/build/ does not exist"}

    def fake_cpp_itfs(target_file, _backup_dir, **_kwargs):
        invalidated.append(f"cpp_itfs:{target_file}")
        # Nothing cached yet: still a cpp_itfs kernel, so a rebuild is still owed.
        return {"status": "skipped", "is_cpp_itfs": True, "module_names": [], "invalidated_unix": 0.0}

    monkeypatch.setattr(tool, "_invalidate_aiter_jit_build", fake_jit)
    monkeypatch.setattr(tool, "_invalidate_aiter_cpp_itfs_cache", fake_cpp_itfs)
    return invalidated


async def test_preapplied_patch_invalidates_the_aiter_caches(
    session_dir, applied, patch_file, stop_at_rebaseline, stub_aiter_invalidation
):
    """Skipping the apply must not skip its stale-binary guards."""
    target = str(applied / KERNEL_REL)

    with pytest.raises(ReachedRebaseline):
        await krh.integrate_handler(
            _controller_payload(applied, patch_file),
            session_dir=session_dir,
            preapplied_git_patch=True,
        )

    assert stub_aiter_invalidation == [f"jit:{target}", f"cpp_itfs:{target}"]


async def test_preapplied_patch_forces_a_rebuild_for_a_cpp_itfs_target(
    session_dir, applied, patch_file, stub_aiter_invalidation, monkeypatch
):
    """An apply result carrying no backup record leaves AITER_REBUILD unset for the server."""
    monkeypatch.delenv("AITER_REBUILD", raising=False)
    seen: list[str | None] = []

    async def record_env(*_args, **_kwargs):
        seen.append(os.environ.get("AITER_REBUILD"))
        raise ReachedRebaseline

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.baseline.BaselineExecutor",
        lambda **_kwargs: SimpleNamespace(default_timeout_sec=60),
    )
    monkeypatch.setattr(krh, "_run_integrate_rebaseline_with_lock_retry", record_env)

    result = await krh.integrate_handler(
        _controller_payload(applied, patch_file),
        session_dir=session_dir,
        preapplied_git_patch=True,
    )

    assert seen == ["1"]
    assert result["error_class"] == "rebaseline_exception"


async def test_preapplied_patch_refuses_when_invalidation_fails(
    session_dir, applied, patch_file, stop_at_rebaseline, monkeypatch
):
    """Refuse to benchmark against an unknown/stale binary rather than measure it."""
    tool = krh._load_apply_tool()
    monkeypatch.setattr(
        tool,
        "_invalidate_aiter_jit_build",
        lambda *_args, **_kwargs: {"status": "failed", "error": "permission denied"},
    )

    result = await krh.integrate_handler(
        _controller_payload(applied, patch_file),
        session_dir=session_dir,
        preapplied_git_patch=True,
    )

    assert result["status"] == "failed"
    assert result["error_class"] == "apply_failed"
    assert result["decision"] == "REVERT"
    assert "permission denied" in result["error"]
