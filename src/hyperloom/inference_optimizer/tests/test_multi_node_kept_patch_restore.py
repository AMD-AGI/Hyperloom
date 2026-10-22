# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Adopting a multi-node cluster puts the session's kept kernel patches back on the pods."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli import multi_node as mn
from hyperloom.inference_optimizer.multi_node._internal import external_state as ext
from hyperloom.orchestrator.actions.executors import _kernel_agent_tool as tool
from hyperloom.orchestrator.kernel import apply_kernel_patch as akp
from hyperloom.orchestrator.kernel import kept_patch_restore as restore

_ADOPT_ARGS = argparse.Namespace(nodes=2, no_kernel=True)


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A session whose patches fan out to stubbed pods."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    monkeypatch.setattr(mn, "_session_dir_resolve", lambda: session_dir)
    monkeypatch.setattr(ext, "external_has_server_control", lambda: True)
    monkeypatch.setattr(akp, "_is_multi_node", lambda: True)
    for module in (akp, restore):
        monkeypatch.setattr(module, "_dispatch_multinode_apply", lambda apply: {"status": "ok", "per_node": []})
        monkeypatch.setattr(module, "_dispatch_multinode_finalize", lambda records_by_host: {"status": "ok"})
    monkeypatch.setattr(akp, "_dispatch_multinode_revert", lambda **_kw: {"status": "ok"})
    return session_dir


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _apply(session: Path, kernel_id: str, **payload: str) -> dict:
    applied = tool._maybe_apply_kernel_patch(
        {**payload, "skip_rebuild": True}, session_dir=session, kernel_id=kernel_id
    )
    assert applied["status"] == "ok", applied
    return applied


def _keep(session: Path, kernel_id: str, **payload: str) -> None:
    kept = tool._maybe_finalize_kernel_patch(_apply(session, kernel_id, **payload))
    assert kept["status"] == "ok", kept


def _record_sends(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, str]]:
    sent: list[tuple[Path, str]] = []

    def dispatch(apply: akp.PodPatchApply) -> dict:
        sent.append((apply.target_file, Path(apply.patch_path).read_text(encoding="utf-8")))
        return {"status": "ok", "per_node": [{"host": "pod-a", "backup_path": f"/b/{len(sent)}"}]}

    monkeypatch.setattr(restore, "_dispatch_multinode_apply", dispatch)
    return sent


def test_adopting_a_cluster_restores_a_kept_kernel_patch(session, tmp_path, monkeypatch):
    target = _write(tmp_path / "repo" / "kernel.cpp", "int main() { return 0; }\n")
    _keep(
        session,
        "k001",
        patch_path=str(_write(tmp_path / "kept.cpp", "int main() { return 1; }\n")),
        target_file=str(target),
    )
    sent = _record_sends(monkeypatch)

    mn._prepare_adopted_cluster(_ADOPT_ARGS, "infera", head_ip="")

    assert sent == [(target.resolve(), "int main() { return 1; }\n")]


def test_a_cluster_without_server_control_keeps_its_sources(session, tmp_path, monkeypatch, capsys):
    target = _write(tmp_path / "repo" / "kernel.cpp", "int main() { return 0; }\n")
    _keep(
        session,
        "k001",
        patch_path=str(_write(tmp_path / "kept.cpp", "int main() { return 1; }\n")),
        target_file=str(target),
    )
    sent = _record_sends(monkeypatch)
    monkeypatch.setattr(ext, "external_has_server_control", lambda: False)

    mn._prepare_adopted_cluster(_ADOPT_ARGS, "infera", head_ip="")

    assert sent == []
    assert "no server control" in capsys.readouterr().err


def test_each_kept_file_is_sent_once_with_its_current_content(session, tmp_path, monkeypatch):
    repo = (tmp_path / "repo").resolve()
    kernel = _write(repo / "kernel.cpp", "int main() { return 0; }\n")
    helper = _write(repo / "helper.h", "#define K 0\n")
    other = _write(repo / "other.cpp", "int other() { return 0; }\n")
    _keep(
        session,
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
        session,
        "k002",
        patch_path=str(diff),
        target_file=str(kernel),
        snapshot_dir=str(snapshot),
        kernel_repo=str(repo),
    )
    rejected = _apply(
        session,
        "k003",
        patch_path=str(_write(tmp_path / "k003.cpp", "int other() { return 3; }\n")),
        target_file=str(other),
    )
    assert tool._maybe_revert_kernel_patch(rejected)["status"] == "ok"
    sent = _record_sends(monkeypatch)

    mn._prepare_adopted_cluster(_ADOPT_ARGS, "infera", head_ip="")

    assert sorted(sent) == sorted([(kernel, "int main() { return 2; }\n"), (helper, "#define K 2\n")])


def test_a_kept_file_the_pods_refuse_stops_the_adoption(session, tmp_path, monkeypatch, capsys):
    target = _write(tmp_path / "repo" / "kernel.cpp", "int main() { return 0; }\n")
    _keep(
        session,
        "k001",
        patch_path=str(_write(tmp_path / "kept.cpp", "int main() { return 1; }\n")),
        target_file=str(target),
    )

    def refuse(apply: akp.PodPatchApply) -> dict:
        raise RuntimeError("multi-node apply-patch returned rc=1")

    monkeypatch.setattr(restore, "_dispatch_multinode_apply", refuse)

    with pytest.raises(SystemExit) as exited:
        mn._prepare_adopted_cluster(_ADOPT_ARGS, "infera", head_ip="")

    assert exited.value.code == 1
    assert str(target.resolve()) in capsys.readouterr().err
