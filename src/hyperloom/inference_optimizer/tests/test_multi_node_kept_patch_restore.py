# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resuming a multi-node session puts its kept kernel patches back on the pods."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON
from hyperloom.orchestrator.actions.executors import _kernel_agent_tool as tool
from hyperloom.orchestrator.kernel import apply_kernel_patch as akp
from hyperloom.orchestrator.kernel import kept_patch_restore as restore
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.writeback import IntegrateRecoveryIncomplete

from .conftest import make_coordinator


@pytest.fixture
def coord(session_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Coordinator:
    """A resumed multi-node session with server control, whose patches fan out to stubbed pods."""
    for module in (akp, restore):
        monkeypatch.setattr(module, "_is_multi_node", lambda: True)
        monkeypatch.setattr(module, "_dispatch_multinode_apply", lambda apply: {"status": "ok", "per_node": []})
        monkeypatch.setattr(module, "_dispatch_multinode_finalize", lambda records_by_host: {"status": "ok"})
    monkeypatch.setattr(akp, "_dispatch_multinode_revert", lambda **_kw: {"status": "ok"})
    monkeypatch.setattr(restore, "external_has_server_control", lambda: True)
    coord = make_coordinator(session_dir)
    coord.writeback._resumed_from["is_resume"] = True
    return coord


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _apply(coord: Coordinator, kernel_id: str, **payload: str) -> dict:
    applied = tool._maybe_apply_kernel_patch(
        {**payload, "skip_rebuild": True}, session_dir=coord.session_dir, kernel_id=kernel_id
    )
    assert applied["status"] == "ok", applied
    return applied


def _keep(coord: Coordinator, kernel_id: str, **payload: str) -> None:
    kept = tool._maybe_finalize_kernel_patch(_apply(coord, kernel_id, **payload))
    assert kept["status"] == "ok", kept


def _keep_one_kernel_file(coord: Coordinator, tmp_path: Path) -> Path:
    target = _write(tmp_path / "repo" / "kernel.cpp", "int main() { return 0; }\n")
    kept = _write(tmp_path / "kept.cpp", "int main() { return 1; }\n")
    _keep(coord, "k001", patch_path=str(kept), target_file=str(target))
    return target.resolve()


def _record_sends(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, str]]:
    sent: list[tuple[Path, str]] = []

    def dispatch(apply: akp.PodPatchApply) -> dict:
        sent.append((apply.target_file, Path(apply.patch_path).read_text(encoding="utf-8")))
        return {"status": "ok", "per_node": [{"host": "pod-a", "backup_path": f"/b/{len(sent)}"}]}

    monkeypatch.setattr(restore, "_dispatch_multinode_apply", dispatch)
    return sent


@pytest.mark.asyncio
async def test_resume_restores_a_kept_kernel_patch_on_the_pods(
    coord: Coordinator, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _keep_one_kernel_file(coord, tmp_path)
    sent = _record_sends(monkeypatch)

    report = await coord.writeback._resume_consistency_pass()

    assert sent == [(target, "int main() { return 1; }\n")]
    assert {"kind": "restored_kept_kernel_files", "files": [str(target)]} in report["fixes"]


@pytest.mark.asyncio
async def test_each_kept_file_is_sent_once_with_its_current_content(
    coord: Coordinator, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = (tmp_path / "repo").resolve()
    kernel = _write(repo / "kernel.cpp", "int main() { return 0; }\n")
    helper = _write(repo / "helper.h", "#define K 0\n")
    other = _write(repo / "other.cpp", "int other() { return 0; }\n")
    _keep(
        coord,
        "k001",
        patch_path=str(_write(tmp_path / "k001.cpp", "int main() { return 1; }\n")),
        target_file=str(kernel),
    )
    snapshot = tmp_path / "snapshot"
    _write(snapshot / "kernel.cpp", "int main() { return 2; }\n")
    _write(snapshot / "helper.h", "#define K 2\n")
    diff = _write(
        tmp_path / "k002.patch",
        "diff --git a/kernel.cpp b/kernel.cpp\n--- a/kernel.cpp\n+++ b/kernel.cpp\n@@ -1 +1 @@\n"
        "-int main() { return 1; }\n+int main() { return 2; }\n"
        "diff --git a/helper.h b/helper.h\n--- a/helper.h\n+++ b/helper.h\n@@ -1 +1 @@\n"
        "-#define K 0\n+#define K 2\n",
    )
    _keep(
        coord, "k002", patch_path=str(diff), target_file=str(kernel), snapshot_dir=str(snapshot), kernel_repo=str(repo)
    )
    rejected = _apply(
        coord,
        "k003",
        patch_path=str(_write(tmp_path / "k003.cpp", "int other() { return 3; }\n")),
        target_file=str(other),
    )
    assert tool._maybe_revert_kernel_patch(rejected)["status"] == "ok"
    sent = _record_sends(monkeypatch)

    await coord.writeback._resume_consistency_pass()

    assert sorted(sent) == sorted([(kernel, "int main() { return 2; }\n"), (helper, "#define K 2\n")])


@pytest.mark.asyncio
async def test_a_kept_file_the_pods_refuse_halts_the_resume(
    coord: Coordinator, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _keep_one_kernel_file(coord, tmp_path)

    def refuse(apply: akp.PodPatchApply) -> dict:
        raise RuntimeError("multi-node apply-patch returned rc=1")

    monkeypatch.setattr(restore, "_dispatch_multinode_apply", refuse)

    with pytest.raises(IntegrateRecoveryIncomplete, match="cannot be restored on the multi-node pods") as halted:
        await coord.writeback._resume_consistency_pass()

    assert str(target) in str(halted.value)
    assert coord.shared_state.stop_reason == PATCH_RECOVERY_INCOMPLETE_STOP_REASON


@pytest.mark.asyncio
async def test_a_single_node_resume_sends_nothing(
    coord: Coordinator, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _keep_one_kernel_file(coord, tmp_path)
    sent = _record_sends(monkeypatch)
    monkeypatch.setattr(restore, "_is_multi_node", lambda: False)

    await coord.writeback._resume_consistency_pass()

    assert sent == []


@pytest.mark.asyncio
async def test_a_cluster_without_server_control_keeps_its_sources(
    coord: Coordinator, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _keep_one_kernel_file(coord, tmp_path)
    sent = _record_sends(monkeypatch)
    monkeypatch.setattr(restore, "external_has_server_control", lambda: False)

    await coord.writeback._resume_consistency_pass()

    assert sent == []
    assert "no server control" in caplog.text
